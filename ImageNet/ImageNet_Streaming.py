import os
import sys
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset  # for streaming dataset
from torchvision import transforms  # using standard torchvision transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# Ensure the parent directory is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import HAT_Classifier

# -----------------------------------------------------------------------------
# Custom IterableDataset for Streaming ImageNet1k
# -----------------------------------------------------------------------------
class ImageNetStreamingDataset(IterableDataset):
    def __init__(self, dataset, transform=None):
        """
        Args:
            dataset: a streaming Hugging Face dataset (e.g., from load_dataset)
            transform: a torchvision.transforms pipeline to apply on the PIL image
        """
        self.dataset = dataset
        self.transform = transform

    def __iter__(self):
        # Yield the raw PIL image and label; transforms are applied later
        for sample in self.dataset:
            yield sample['image'], sample['label']

    def collate_fn(self, batch):
        images, labels = zip(*batch)
        processed_images = []
        for img in images:
            # Convert grayscale images to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
            processed_images.append(img)
        # Stack images and convert labels to tensor
        images = torch.stack(processed_images)
        labels = torch.tensor(labels, dtype=torch.long)
        return images, labels

# -----------------------------------------------------------------------------
# Define Transform Pipelines for Training and Validation
# -----------------------------------------------------------------------------
# For training: add common augmentations for ImageNet
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandAugment(),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# For validation: deterministic resize and center crop
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -----------------------------------------------------------------------------
# Training and Evaluation Functions
# -----------------------------------------------------------------------------
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    total_samples = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
    for data, target in pbar:
        # Data is already transformed and on CPU; move to device
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        batch_size = data.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        pbar.set_postfix(loss=loss.item())

    return total_loss / total_samples if total_samples > 0 else float('inf')

def validate_model(model, device, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
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
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 1e-3
    enable_compile = True
    cpu_workers = 4

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the configuration for HAT_Classifier (adjust the path if needed)
    config_path = os.path.join('model', 'configs', 'HAT_Base.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize the model with configuration
    model = HAT_Classifier(config).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Initialized HAT - Size: {total_params/1e6:.2f} M')

    # Enable compilation optimizations if desired
    if enable_compile:
        model = torch.compile(model)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    # Create an output directory with timestamp
    now = datetime.now()
    timestamp = now.strftime("%H-%M")
    date_str = now.strftime("%Y-%m-%d")
    output_dir = os.path.join('Output', date_str, f'{timestamp}-HAT-ImageNet')
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load the ImageNet1k dataset with streaming (Hugging Face datasets)
    # -------------------------------------------------------------------------
    train_dataset_raw = load_dataset('ILSVRC/imagenet-1k', split='train', trust_remote_code=True, streaming=True)
    val_dataset_raw = load_dataset('ILSVRC/imagenet-1k', split='validation', trust_remote_code=True, streaming=True)
    
    # Wrap the streaming datasets with our custom IterableDataset and proper transforms
    train_dataset = ImageNetStreamingDataset(train_dataset_raw, transform=train_transforms)
    val_dataset = ImageNetStreamingDataset(val_dataset_raw, transform=val_transforms)

    # Create DataLoaders (use pin_memory for faster host-to-device transfers when using CUDA)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=cpu_workers,
                              collate_fn=train_dataset.collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=cpu_workers,
                            collate_fn=val_dataset.collate_fn, pin_memory=True)

    # Set up loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # To store metrics across epochs
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'model_size': total_params
    }

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
        val_loss, val_acc = validate_model(model, device, val_loader, criterion)
        scheduler.step()

        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['val_accuracy'].append(val_acc)

        # Save model checkpoint for this epoch
        ckpt_path = os.path.join(output_dir, f'ImageNet_HATClassifer.tar')
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