import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim

import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import HAT_Classifier



def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        pbar.set_postfix(loss=loss.item())
        train_loss += loss.item()
    train_loss /= len(train_loader.dataset)
    return train_loss

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f} | Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)')
    return test_loss, accuracy

def main():
    os.makedirs('output', exist_ok=True)
    # Hyperparameters
    epochs = 100
    batch_size = 128
    learning_rate = 1e-3
    weight_decay = 1e-3

    enable_compile = True
    cpu_workers = 8
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # CIFAR10 dataset transformation: Convert to tensor and normalize
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=180, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-5, 5)),
        transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1, hue=0.01),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download and load the CIFAR10 training and test datasets
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            pin_memory=True, num_workers=cpu_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            pin_memory=True, num_workers=cpu_workers)

    # Initialize the model, loss function, and optimizer
    config = json.load(open('model/configs/HAT_CIFAR10.json'))
    model = HAT_Classifier(config).to(device)
    print(f'Initialized model with {sum(p.numel() for p in model.parameters())/1e3} K parameters')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Compilation Optimizations
    if enable_compile:
        model = torch.compile(model)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    # Training loop
    metrics = {
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': [],
        'model_size': sum(p.numel() for p in model.parameters())
    }
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, test_acc = test(model, device, test_loader, criterion)
        scheduler.step()
        metrics['train_loss'].append(train_loss)
        metrics['test_loss'].append(test_loss)
        metrics['test_accuracy'].append(test_acc)

        # Save the model
        torch.save(model.state_dict(), 'output/CIFAR10_HAT.tar')

        # Save metrics
        with open('output/CIFAR10_metrics.json', 'w') as f:
            json.dump(metrics, f)
        
        # Plot and save losses and accuracy
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.plot(range(epoch), metrics['train_loss'], label='Train Loss', color='tab:blue')
        ax1.plot(range(epoch), metrics['test_loss'], label='Test Loss', color='tab:orange')
        ax1.tick_params(axis='y')
        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy')
        ax2.plot(range(epoch), metrics['test_accuracy'], label='Accuracy', color='black')
        ax2.tick_params(axis='y')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')
        plt.title('Losses and Accuracy')
        plt.tight_layout()
        plt.savefig('output/CIFAR10_plot.png')
        plt.close(fig)

if __name__ == '__main__':
    main()