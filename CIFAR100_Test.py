import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.HAT2 import HAT_Classifier

def test_model(model, device, test_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.contiguous(memory_format=torch.channels_last)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            preds = output.argmax(dim=1)
            total_correct += (preds == target).sum().item()
            total_samples += data.size(0)
            exit()

    avg_loss = total_loss / total_samples
    accuracy = 100. * total_correct / total_samples
    print(f'Test Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

def main(date, hrmin):
    # Settings
    model_path = f'output/{date}/{hrmin}-HAT-CIFAR100/CIFAR100_HAT.tar'
    config_path = os.path.join('model', 'configs', 'HAT2_CIFAR100.json')
    batch_size = 128
    use_autocast = False  # If you used bfloat16 or float16 during training

    # Load device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config and model
    with open(config_path, 'r') as f:
        config = json.load(f)
    model = HAT_Classifier(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Loaded trained model.")

    # Define transform (same as validation)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    # Load test dataset
    test_dataset = datasets.CIFAR100(root='data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Set criterion
    criterion = nn.CrossEntropyLoss()

    # Run test
    test_model(model, device, test_loader, criterion)

if __name__ == '__main__':
    date = '2025-05-29'
    hrmin = '20-48'
    main(date, hrmin)
