import os
import sys
import json

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
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

        return image, label


# -----------------------------------------------------------------------------
# Evaluation Function (Top-1 Accuracy)
# -----------------------------------------------------------------------------
def evaluate_top1(model, device, test_loader, autocast):
    model.eval()
    total_samples = 0
    correct_top1 = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            data = data.contiguous(memory_format=torch.channels_last)

            if autocast:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model(data)
            else:
                output = model(data)

            batch_size = data.size(0)
            total_samples += batch_size

            # Top-1 predictions
            pred = output.argmax(dim=1, keepdim=True)
            correct_top1 += pred.eq(target.view_as(pred)).sum().item()

    top1_accuracy = 100.0 * correct_top1 / total_samples if total_samples > 0 else 0.0
    print(f'Test set: Top-1 Accuracy: {correct_top1}/{total_samples} ({top1_accuracy:.2f}%)')
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
    checkpoint_path = os.path.join('output', 'YOUR_DATE', 'YOUR_TIMESTAMP-PViT-ImageNet', 'ImageNet_PViT.tar')

    # Directory to cache/download ImageNet data
    hf_cache_dir = os.path.join('data', 'hf_cache')
    imagenet_data_dir = os.path.join('data', 'imagenet')

    # Batch size for testing
    batch_size = 512
    img_size = 224

    # Number of CPU workers for DataLoader
    cpu_workers = 32

    # Whether to use automatic mixed precision during inference
    autocast = True

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
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Checkpoint loaded successfully.")

    # --------------------------------------------------------------------------------
    # Define transforms (same as validation transforms used during training)
    # --------------------------------------------------------------------------------
    val_transforms = v2.Compose([
        v2.Resize(img_size),
        v2.CenterCrop(img_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    # --------------------------------------------------------------------------------
    # Load ImageNet1k "test" split
    # Note: If the "test" split is not available, fallback to the "validation" split.
    # --------------------------------------------------------------------------------
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.makedirs(imagenet_data_dir, exist_ok=True)

    try:
        test_dataset_raw = load_dataset(
            'ILSVRC/imagenet-1k',
            split='test',
            trust_remote_code=True,
            streaming=False,
            data_dir=imagenet_data_dir,
            cache_dir=hf_cache_dir
        )
        print("Loaded 'test' split of ImageNet1k.")
    except ValueError:
        # Fallback to using the validation split as the "test" set
        test_dataset_raw = load_dataset(
            'ILSVRC/imagenet-1k',
            split='validation',
            trust_remote_code=True,
            streaming=False,
            data_dir=imagenet_data_dir,
            cache_dir=hf_cache_dir
        )
        print("'test' split not found. Loaded 'validation' split as test set.")

    # --------------------------------------------------------------------------------
    # Wrap raw dataset with our custom ImageNetDataset and transforms
    # --------------------------------------------------------------------------------
    test_dataset = ImageNetDataset(
        dataset=test_dataset_raw,
        device=device,
        transform=val_transforms,
        max_size=img_size
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=cpu_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    # --------------------------------------------------------------------------------
    # Run evaluation to compute Top-1 accuracy
    # --------------------------------------------------------------------------------
    # Use the FlashAttention/CuDNN backends if you want to replicate the same attention kernels.
    sdpa_backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION]

    with sdpa_kernel(sdpa_backends):
        top1_acc = evaluate_top1(model, device, test_loader, autocast)

    # Optionally: save results to a JSON or print more details
    print(f'Finished testing. Top-1 Accuracy: {top1_acc:.2f}%')


if __name__ == '__main__':
    main()
