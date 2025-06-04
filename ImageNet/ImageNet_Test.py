import os
import sys
import json

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torchvision.transforms import v2
from tqdm import tqdm

# Ensure the parent directory is in the path so we can import PatchViT
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.PatchViT7 import PatchViT


# -----------------------------------------------------------------------------
# Custom Map-style Dataset for ImageNet1k (non-streaming)
# -----------------------------------------------------------------------------
class ImageNetDataset(Dataset):
    def __init__(self, dataset, device, transform=None, max_size=224):
        """
        Args:
            dataset: a Hugging Face map-style dataset (streaming=False)
            transform: a torchvision.transforms pipeline to apply on the PIL image
        """
        self.dataset = dataset
        self.device = device
        self.transform = transform
        self.max_size = max_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image, label = sample['image'], sample['label']

        # Convert grayscale images to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        if type(image) == tuple:
            image = torch.stack(image, dim=0)

        return image, label


# -----------------------------------------------------------------------------
# One-Crop Evaluation Function (Top-1 Accuracy)
# -----------------------------------------------------------------------------
def evaluate_top1_onecrop(model, device, test_loader):
    model.eval()
    total_samples = 0
    correct_top1 = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="One-Crop Testing"):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            data = data.contiguous(memory_format=torch.channels_last)

            output = model(data)

            batch_size = data.size(0)
            total_samples += batch_size

            # Top-1 predictions
            pred = output.argmax(dim=-1)
            correct_top1 += pred.eq(target).sum().item()

    top1_accuracy = 100.0 * correct_top1 / total_samples if total_samples > 0 else 0.0
    print(f'One-Crop Test: Top-1 Accuracy: {correct_top1}/{total_samples} ({top1_accuracy:.2f}%)')
    return top1_accuracy


# -----------------------------------------------------------------------------
# Ten-Crop Evaluation Function (Top-1 Accuracy)
# -----------------------------------------------------------------------------
def evaluate_top1_tencrop(model, device, test_loader):
    model.eval()
    total_samples = 0
    correct_top1 = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Ten-Crop Testing"):
            # data shape: [B, 10, 3, H, W]
            bs, ncrops, c, h, w = data.size()
            data = data.view(bs * ncrops, c, h, w).to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            data = data.contiguous(memory_format=torch.channels_last)

            # Run all crops through the model
            outputs = model(data)  # shape: [B * 10, num_classes]

            # Reshape and average logits across crops
            outputs = outputs.view(bs, ncrops, -1)  # shape: [B, 10, num_classes]
            outputs_avg = outputs.mean(dim=1)       # shape: [B, num_classes]

            total_samples += bs
            pred = outputs_avg.argmax(dim=-1)       # shape: [B]
            correct_top1 += pred.eq(target).sum().item()

    top1_accuracy = 100.0 * correct_top1 / total_samples if total_samples > 0 else 0.0
    print(f'Ten-Crop Test: Top-1 Accuracy: {correct_top1}/{total_samples} ({top1_accuracy:.2f}%)')
    return top1_accuracy


# -----------------------------------------------------------------------------
# Main Testing Script
# -----------------------------------------------------------------------------
def main():
    # --------------------------------------------------------------------------------
    # User-editable paths and hyperparameters
    # --------------------------------------------------------------------------------
    # Path to the JSON config for PatchViT (same as training)
    config_path = os.path.join('model', 'configs', 'PViT7_ImageNet.json')

    # Path to the trained checkpoint (.tar or .pth) produced by the training script
    checkpoint_path = os.path.join('output', '2025-06-03', '16-01-PViT-ImageNet', 'ImageNet_PViT.tar')

    # Directory to cache/download ImageNet data
    hf_cache_dir = os.path.join('data', 'hf_cache')
    imagenet_data_dir = os.path.join('data', 'imagenet')

    # Batch size for testing
    batch_size = 64
    img_size = 256
    cpu_workers = 32

    # --------------------------------------------------------------------------------
    # Device configuration
    # --------------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # --------------------------------------------------------------------------------
    # Load the PatchViT configuration and initialize model
    # --------------------------------------------------------------------------------
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Could not find config file at: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    model = PatchViT(config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Initialized PViT for testing — Model size: {total_params/1e6:.2f} M parameters')

    # --------------------------------------------------------------------------------
    # Load checkpoint weights
    # --------------------------------------------------------------------------------
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Could not find checkpoint at: {checkpoint_path}")

    print(f'Loading checkpoint from: {checkpoint_path}')
    raw = torch.load(checkpoint_path, map_location=device)
    state_dict = raw.get('state_dict', raw)
    # Strip the "_orig_mod." prefix from any key that has it
    clean_state_dict = {
        (k[len("_orig_mod."): ] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(clean_state_dict)
    print("Checkpoint loaded successfully.")

    # --------------------------------------------------------------------------------
    # Define transforms
    # --------------------------------------------------------------------------------
    # One-shot (center-crop) transforms
    onecrop_transforms = v2.Compose([
        v2.Resize(img_size),                                  
        v2.CenterCrop(img_size),                              
        v2.ToImage(),                                         
        v2.ToDtype(torch.float32, scale=True),                
        v2.Normalize(mean=[0.485, 0.456, 0.406],               
                     std=[0.229, 0.224, 0.225]),              
    ])

    # Ten-crop transforms: produce a tuple of ten PIL images and then stack into a 5D tensor
    tencrop_transforms = v2.Compose([
        v2.Resize(int(img_size*1.1)),                                  
        v2.ToImage(),                                         
        v2.ToDtype(torch.float32, scale=True),                
        v2.Normalize(mean=[0.485, 0.456, 0.406],               
                     std=[0.229, 0.224, 0.225]),     
        v2.TenCrop(img_size),
    ])

    # --------------------------------------------------------------------------------
    # Load ImageNet1k "validation" split
    # --------------------------------------------------------------------------------
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.makedirs(imagenet_data_dir, exist_ok=True)

    test_dataset_raw = load_dataset(
        'ILSVRC/imagenet-1k',
        split='validation',  # test labels are all -1, so use "validation"
        trust_remote_code=True,
        streaming=False,
        data_dir=imagenet_data_dir,
        cache_dir=hf_cache_dir
    )
    print("Loaded 'validation' split of ImageNet1k.")

    # --------------------------------------------------------------------------------
    # Wrap raw dataset with our custom ImageNetDataset and transforms
    # --------------------------------------------------------------------------------
    test_dataset_onecrop = ImageNetDataset(
        dataset=test_dataset_raw,
        device=device,
        transform=onecrop_transforms,
        max_size=img_size
    )
    test_loader_onecrop = DataLoader(
        test_dataset_onecrop,
        batch_size=batch_size,
        num_workers=cpu_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # For ten-crop, we need to reload the dataset object, since transforms cannot be changed in place
    test_dataset_raw_tencrop = load_dataset(
        'ILSVRC/imagenet-1k',
        split='validation',
        trust_remote_code=True,
        streaming=False,
        data_dir=imagenet_data_dir,
        cache_dir=hf_cache_dir
    )
    test_dataset_tencrop = ImageNetDataset(
        dataset=test_dataset_raw_tencrop,
        device=device,
        transform=tencrop_transforms,
        max_size=img_size
    )
    test_loader_tencrop = DataLoader(
        test_dataset_tencrop,
        batch_size=batch_size,
        num_workers=cpu_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # --------------------------------------------------------------------------------
    # Run evaluation to compute both Top-1 accuracies
    # --------------------------------------------------------------------------------
    print("Starting one-shot (center-crop) evaluation...")
    top1_onecrop = evaluate_top1_onecrop(model, device, test_loader_onecrop)
    print(f'One-Shot Top-1 Accuracy: {top1_onecrop:.2f}%')

    print("Starting ten-crop evaluation...")
    top1_tencrop = evaluate_top1_tencrop(model, device, test_loader_tencrop)
    print(f'Ten-Crop Top-1 Accuracy: {top1_tencrop:.2f}%')

    # Optionally: save results to a JSON or print more details
    print(f'Finished testing.\nOne-Shot Top-1 Accuracy: {top1_onecrop:.2f}%\nTen-Crop Top-1 Accuracy: {top1_tencrop:.2f}%')


if __name__ == '__main__':
    main()
