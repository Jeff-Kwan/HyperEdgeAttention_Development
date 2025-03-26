import torch
import tqdm
import os
import json
import matplotlib.pyplot as plt
from time import time
import numpy as np

class Trainer():
    def __init__(self, model, optimizer, criterion, scheduler, train_params, output_dir, device, comments):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.train_params = train_params
        self.output_dir = output_dir
        self.device = device
        self.comments = comments

        self.train_losses = []
        self.val_losses = []
        self.val_metrics = {
            'accuracy': []
        }
        self.best_results = {}

        os.makedirs(output_dir, exist_ok=True)
        self.model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def train(self, train_loader, val_loader):
        self.model.to(self.device)
        epochs = self.train_params['epochs']
        aggregation = self.train_params['aggregation']

        self.start_time = time()
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            norm = torch.tensor(0.)
            p_bar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            self.optimizer.zero_grad()

            for i, (imgs, labels) in enumerate(p_bar):
                imgs = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                train_loss += loss.item()
                
                # Gradient accumulation
                if ((i+1) % aggregation == 0) or (i == len(train_loader)-1):
                    norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                p_bar.set_postfix({'Norm': norm.item(), 'Loss': loss.item()})

            self.scheduler.step()

            # Metrics
            val_loss, val_metrics = self.evaluate(val_loader)
            self.train_losses.append(train_loss / len(train_loader))
            self.val_losses.append(val_loss)
            self.val_metrics['accuracy'].append(val_metrics['accuracy'])
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {self.train_losses[-1]:.5f}, '
                  f'Val Loss: {self.val_losses[-1]:.5f}, Accuracy: {val_metrics["accuracy"]:.5f}')

            # Save results
            self.save_examples(val_loader)
            self.plot_results()
            self.save_checkpoint(epoch, val_metrics)

    def evaluate(self, data_loader):
        self.model.eval()
        loss_total = 0.0
        all_labels = []
        all_outputs = []
        with torch.inference_mode():
            for imgs, labels in data_loader:
                imgs = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                loss_total += loss.item()
                all_labels.append(labels.detach().cpu().numpy())
                all_outputs.append(outputs.detach().cpu().numpy())
        loss_total /= len(data_loader)
        metrics = self.calculate_metrics(all_labels, all_outputs)
        return loss_total, metrics

    def calculate_metrics(self, labels, outputs):
        # Concatenate all batches
        labels = np.concatenate(labels)
        outputs = np.concatenate(outputs, axis=0)
        preds = np.argmax(outputs, axis=1)
        accuracy = np.mean(preds == labels)
        return {'accuracy': accuracy}
    
    def save_checkpoint(self, epoch, val_metrics):
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'model.pth'))
        results = dict({
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        })
        with open(os.path.join(self.output_dir, 'metrics.json'), 'w') as f:
            json.dump(results, f, indent=4)

        # Save best model based on highest accuracy
        if self.val_metrics['accuracy'][-1] >= max(self.val_metrics['accuracy']):
            torch.save(self.model.state_dict(), os.path.join(self.output_dir, f'best_model.pth'))
            self.best_results = {
                'epoch (count from 0)': epoch,
                'train_loss': self.train_losses[-1],
                'val_loss': self.val_losses[-1],
                'val_metrics': val_metrics
            }

        elapsed_time = time() - self.start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        with open(os.path.join(self.output_dir, 'results.txt'), 'w') as f:
            f.write(f'Model size: {self.model_size/1e6} M\n')
            f.write(f'Training time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}\n\n')
            for comment in self.comments:
                f.write(f'{comment}\n')
            f.write(f'\nModel params: {json.dumps(self.model.model_params, indent=4)}\n')
            f.write(f'\nTrain params: {json.dumps(self.train_params, indent=4)}\n')
            f.write(f'\nBest validation results: {json.dumps(self.best_results, indent=4)}\n')
            f.write(f'\n~~~~~~ Test Results ~~~~~~\n')

    def plot_results(self):
        epochs = range(1, len(self.train_losses) + 1)
        
        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.plot(epochs, self.train_losses, label='Train Loss', color='tab:blue')
        ax1.plot(epochs, self.val_losses, label='Val Loss', color='tab:orange')
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Accuracy')
        ax2.plot(epochs, self.val_metrics['accuracy'], label='Accuracy', color='black')
        ax2.tick_params(axis='y')

        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.title('Losses and Accuracy')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'losses.png'))
        plt.close(fig)