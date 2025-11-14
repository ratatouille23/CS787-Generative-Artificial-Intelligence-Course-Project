import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import custom_dataset
from module import HAN
from torcheval.metrics import BinaryConfusionMatrix
import os
from pathlib import Path
import numpy as np
from visualizer import TrainingVisualizer

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class SimpleTrainer:

    def __init__(self, model, train_loader, dev_loader, optimizer, scheduler,
                 device, train_weights, dev_weights, save_dir='checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_weights = train_weights.to(device)
        self.dev_weights = dev_weights.to(device)
        self.best_loss = float('inf')
        self.best_epoch = 0

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.save_path = os.path.join(save_dir, 'best_model.pt')
        self.visualizer = TrainingVisualizer(save_dir='visualizations')

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (source, targets) in enumerate(self.train_loader):
            source = source.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(source)
            loss = F.cross_entropy(output, targets, weight=self.train_weights)

            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f'  Batch [{batch_idx + 1}/{len(self.train_loader)}], Loss: {loss.item():.4f}')

        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch} - Train Loss: {avg_loss:.4f}')
        return avg_loss

    def evaluate(self, data_loader, weights, mode='dev'):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        mat = BinaryConfusionMatrix()

        weights = weights.to(self.device)

        with torch.no_grad():
            for source, targets in data_loader:
                source = source.to(self.device)
                targets = targets.to(self.device)

                output = self.model(source)
                loss = F.cross_entropy(output, targets, weight=weights)
                pred = F.softmax(output, dim=1).argmax(dim=1)

                total_loss += loss.item()
                num_batches += 1
                mat.update(pred, targets)

        avg_loss = total_loss / num_batches
        metrics = self.calculate_metrics(mat)

        print(f'\n{mode.upper()} Results:')
        print(f'  Loss: {avg_loss:.4f}')
        print(f'  Accuracy: {metrics["acc"]:.4f}')
        print(f'  Precision: {metrics["pre"]:.4f}')
        print(f'  Recall: {metrics["rec"]:.4f}')
        print(f'  F1: {metrics["f1"]:.4f}')
        print(f'  MCC: {metrics["mcc"]:.4f}')
        print(
            f'  Confusion Matrix (TP, FN, FP, TN): {metrics["tp"]}, {metrics["fn"]}, {metrics["fp"]}, {metrics["tn"]}\n')

        return avg_loss, metrics

    def calculate_metrics(self, mat):
        mat_values = mat.compute()
        tp, fn, fp, tn = mat_values[0, 0].item(), mat_values[0, 1].item(), \
            mat_values[1, 0].item(), mat_values[1, 1].item()

        acc = (tp + tn) / (tp + fn + fp + tn) if (tp + fn + fp + tn) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        pre = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / (denom ** 0.5) if denom > 0 else 0

        return {
            'acc': acc, 'rec': rec, 'pre': pre, 'f1': f1, 'mcc': mcc,
            'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn
        }

    def save_checkpoint(self, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
        }, self.save_path)
        print(f'Model saved to {self.save_path}')

    def train(self, num_epochs):
        print(f'\nStarting training for {num_epochs} epochs...\n')

        for epoch in range(num_epochs):
            print(f'=== Epoch {epoch + 1}/{num_epochs} ===')
            train_loss = self.train_epoch(epoch)
            dev_loss, dev_metrics = self.evaluate(self.dev_loader, self.dev_weights, 'dev')
            current_lr = self.optimizer.param_groups[0]['lr']
            self.visualizer.update(
                epoch=epoch + 1,
                train_loss=train_loss,
                dev_metrics={
                    'loss': dev_loss,
                    'acc': dev_metrics['acc'],
                    'pre': dev_metrics['pre'],
                    'rec': dev_metrics['rec'],
                    'f1': dev_metrics['f1'],
                    'mcc': dev_metrics['mcc']
                },
                learning_rate=current_lr
            )
            if dev_loss < self.best_loss:
                self.best_loss = dev_loss
                self.best_epoch = epoch
                self.save_checkpoint(epoch)
                print(f'New best model! Dev loss: {dev_loss:.4f}')

            print('-' * 60)

        print(f'\nTraining completed!')
        print(f'Best epoch: {self.best_epoch}, Best dev loss: {self.best_loss:.4f}')

        return self.visualizer

def main():
    import config
    flags = config.args

    set_seed(flags.seed)

    device = torch.device(flags.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print('\nLoading datasets...')
    train_data = custom_dataset(flags.train_x_path, flags.train_y_path,
                                flags.days, flags.max_num_tweets_len,
                                flags.max_num_tokens_len)
    dev_data = custom_dataset(flags.dev_x_path, flags.dev_y_path,
                              flags.days, flags.max_num_tweets_len,
                              flags.max_num_tokens_len)
    test_data = custom_dataset(flags.test_x_path, flags.test_y_path,
                               flags.days, flags.max_num_tweets_len,
                               flags.max_num_tokens_len)

    print(f'Train samples: {len(train_data)}')
    print(f'Dev samples: {len(dev_data)}')
    print(f'Test samples: {len(test_data)}')

    train_loader = DataLoader(train_data, batch_size=flags.batch_size,
                              shuffle=True, num_workers=flags.num_workers)
    dev_loader = DataLoader(dev_data, batch_size=flags.batch_size,
                            shuffle=False, num_workers=flags.num_workers)
    test_loader = DataLoader(test_data, batch_size=flags.batch_size,
                             shuffle=False, num_workers=flags.num_workers)

    print('\nInitializing model...')
    model = HAN(flags)

    if flags.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=flags.learning_rate,
                                      weight_decay=flags.weight_decay)
    elif flags.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=flags.learning_rate,
                                    weight_decay=flags.weight_decay, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=flags.learning_rate,
                                     weight_decay=flags.weight_decay)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=flags.learning_rate,
        epochs=flags.train_epochs, steps_per_epoch=len(train_loader)
    )

    trainer = SimpleTrainer(
        model, train_loader, dev_loader, optimizer, scheduler,
        device, train_data.class_weights, dev_data.class_weights,
        save_dir=flags.save_dir
    )

    trainer.train(flags.train_epochs)

    print('\n=== Testing on best model ===')
    checkpoint = torch.load(trainer.save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_metrics = trainer.evaluate(test_loader, test_data.class_weights, 'test')

    print('\nDone!')


if __name__ == '__main__':
    main()