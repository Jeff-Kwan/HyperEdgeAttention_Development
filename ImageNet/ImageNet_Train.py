import os
import sys
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp

# Ensure the parent directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import HAT_Classifier


# -----------------------------------------------------------------------------
# Custom Map-style Dataset for ImageNet1k (non-streaming)
# -----------------------------------------------------------------------------
class ImageNetDataset(Dataset):
    def __init__(self, dataset, device, transform=None):
        """
        Args:
            dataset: a Hugging Face map-style dataset (streaming=False)
            transform: a torchvision.transforms pipeline to apply on the PIL image
        """
        self.dataset = dataset
        self.device = device
        self.transform = transform

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
# Training and Evaluation Functions
# -----------------------------------------------------------------------------
def train(model, device, train_loader, optimizer, criterion, epoch, autocast):
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
    for data, target in pbar:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        data = data.contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad(set_to_none=True)
        if autocast:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                output = model(data)
                loss = criterion(output, target)
        else:
            output = model(data)
            loss = criterion(output, target)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item(), norm=norm.item())

    return total_loss / len(train_loader)


def validate_model(model, device, val_loader, criterion, autocast):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            data = data.contiguous(memory_format=torch.channels_last)
            if autocast:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = model(data)
                    loss = criterion(output, target)
            else:
                output = model(data)
                loss = criterion(output, target)
            batch_size = data.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    accuracy = 100. * correct / total_samples if total_samples > 0 else 0.0
    print(f'Val set: Average loss: {val_loss:.4f} | Accuracy: {correct}/{total_samples} ({accuracy:.0f}%)')
    return val_loss, accuracy

# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------
def main():
    # Hyperparameters
    epochs = 100
    batch_size = 256
    learning_rate = 1e-3
    weight_decay = 1e-3
    enable_compile = True
    autocast = True
    matmul_precision = 'medium' if autocast else 'high'
    flash = True
    cpu_workers = min(max(1, mp.cpu_count() - 1), 64)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the configuration for HAT_Classifier (adjust path if needed)
    config_path = os.path.join('model', 'configs', 'HAT_Base.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize the model with configuration
    model = HAT_Classifier(config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Initialized HAT - Size: {total_params/1e6:.2f} M')

    # Create an output directory with timestamp
    now = datetime.now()
    timestamp = now.strftime("%H-%M")
    date_str = now.strftime("%Y-%m-%d")
    output_dir = os.path.join('output', date_str, f'{timestamp}-HAT-ImageNet')
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------------
    # Load the ImageNet1k dataset (non-streaming, map-style)
    # -----------------------------------------------------------------------------
    os.makedirs(os.path.join('data', 'hf_cache'), exist_ok=True)
    os.makedirs(os.path.join('data', 'imagenet'), exist_ok=True)
    train_dataset_raw = load_dataset('ILSVRC/imagenet-1k', split='train', 
        trust_remote_code=True, streaming=False, num_proc=cpu_workers,
        data_dir=os.path.join('data', 'imagenet'),
        cache_dir=os.path.join('data', 'hf_cache'))
    val_dataset_raw = load_dataset('ILSVRC/imagenet-1k', split='validation', 
        trust_remote_code=True, streaming=False, num_proc=cpu_workers,
        data_dir=os.path.join('data', 'imagenet'),
        cache_dir=os.path.join('data', 'hf_cache'))
    
    # Define Transform Pipelines for Training and Validation
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        # transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    
    # Wrap the datasets with our custom Map-style Dataset and proper transforms
    train_dataset = ImageNetDataset(train_dataset_raw, device, transform=train_transforms)
    val_dataset = ImageNetDataset(val_dataset_raw, device, transform=val_transforms)

    # Create DataLoaders (using pin_memory for faster transfers when using CUDA)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=cpu_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=cpu_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # Set up loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Enable compilation optimizations
    if enable_compile:
        print("Compiling model...")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision(matmul_precision)
        model = torch.compile(model, mode='max-autotune')
        if flash:
            assert autocast, "Flash Attention requires autocasting to bfloat16"
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(False)

    # To store metrics across epochs
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'model_size': total_params
    }

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch, autocast)
        val_loss, val_acc = validate_model(model, device, val_loader, criterion, autocast)
        scheduler.step()

        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['val_accuracy'].append(val_acc)

        # Save model checkpoint for this epoch
        ckpt_path = os.path.join(output_dir, f'ImageNet_HATClassifier.tar')
        torch.save(model.state_dict(), ckpt_path)

        # Save metrics to JSON
        with open(os.path.join(output_dir, 'ImageNet_metrics.json'), 'w') as f:
            json.dump(metrics, f)

        # Plot and save training and validation curves
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.plot(range(1, epoch + 1), metrics['train_loss'], label='Train Loss', color='tab:blue')
        ax1.plot(range(1, epoch + 1), metrics['val_loss'], label='Val Loss', color='tab:orange')
        ax1.tick_params(axis='y')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy')
        ax2.plot(range(1, epoch + 1), metrics['val_accuracy'], label='Accuracy', color='black')
        ax2.tick_params(axis='y')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')
        plt.title('Loss and Accuracy')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'ImageNet_losses.png'))
        plt.close(fig)

if __name__ == '__main__':
    main()
