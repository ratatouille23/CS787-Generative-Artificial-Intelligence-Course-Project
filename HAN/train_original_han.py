import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
from torcheval.metrics import BinaryConfusionMatrix
from original_han import OriginalHAN, LightweightOriginalHAN
from visualizer import TrainingVisualizer
import time

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_gpu():
    print("\n" + "=" * 70)
    print("GPU CONFIGURATION CHECK")
    print("=" * 70)

    if torch.cuda.is_available():
        print(f"✓ CUDA is available!")
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
            print(f"  Memory reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
            print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("✗ CUDA is NOT available. Training will use CPU.")
        print("  Install CUDA-enabled PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118")



class OriginalHANDataset(Dataset):

    def __init__(self, x_path, y_path, days, max_sentences, max_words):
        self.x = np.loadtxt(x_path, delimiter=',', dtype=np.int32)
        self.y = np.loadtxt(y_path, delimiter=',', dtype=np.int64)

        self.x = self.x.reshape(-1, days, max_sentences, max_words)

        unique, counts = np.unique(self.y, return_counts=True)
        self.class_weights = len(self.y) / (len(unique) * counts)
        self.class_weights = torch.FloatTensor(self.class_weights)

        print(f"  Loaded {len(self.y)} samples")
        print(f"  Shape: {self.x.shape}")
        print(f"  Class distribution: {dict(zip(unique, counts))}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.LongTensor(self.x[idx]), torch.LongTensor([self.y[idx]])[0]


class Trainer:

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
        self.save_path = f'{save_dir}/best_model.pt'
        self.visualizer = TrainingVisualizer(save_dir='visualizations')

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        start_time = time.time()

        for batch_idx, (source, targets) in enumerate(self.train_loader):
            # Move to GPU - NON_BLOCKING for speed
            source = source.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Forward
            self.optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            output = self.model(source)
            loss = F.cross_entropy(output, targets, weight=self.train_weights)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                samples_per_sec = (batch_idx + 1) * self.train_loader.batch_size / elapsed

                print(f'  Batch [{batch_idx + 1}/{len(self.train_loader)}], '
                      f'Loss: {loss.item():.4f}, '
                      f'Speed: {samples_per_sec:.1f} samples/sec')

                # Print GPU utilization
                if torch.cuda.is_available() and (batch_idx + 1) % 50 == 0:
                    print(f'  GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / '
                          f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')

        avg_loss = total_loss / num_batches
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch} - Train Loss: {avg_loss:.4f}, Time: {epoch_time:.1f}s')
        current_lr = self.optimizer.param_groups[0]['lr']
        return avg_loss, current_lr

    def evaluate(self, data_loader, weights, mode='dev'):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        mat = BinaryConfusionMatrix()

        with torch.no_grad():
            for source, targets in data_loader:
                source = source.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                output = self.model(source)
                loss = F.cross_entropy(output, targets, weight=weights)
                pred = F.softmax(output, dim=1).argmax(dim=1)

                total_loss += loss.item()
                num_batches += 1
                mat.update(pred.cpu(), targets.cpu())  # Move to CPU for metrics

        avg_loss = total_loss / num_batches
        metrics = self.calculate_metrics(mat)

        print(f'\n{mode.upper()} Results:')
        print(f'  Loss: {avg_loss:.4f}')
        print(f'  Accuracy: {metrics["acc"]:.4f}')
        print(f'  Precision: {metrics["pre"]:.4f}')
        print(f'  Recall: {metrics["rec"]:.4f}')
        print(f'  F1: {metrics["f1"]:.4f}')
        print(f'  MCC: {metrics["mcc"]:.4f}\n')

        return avg_loss, metrics

    def calculate_metrics(self, mat):
        mat_values = mat.compute()
        tp = mat_values[0, 0].item()
        fn = mat_values[0, 1].item()
        fp = mat_values[1, 0].item()
        tn = mat_values[1, 1].item()

        acc = (tp + tn) / (tp + fn + fp + tn) if (tp + fn + fp + tn) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        pre = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / (denom ** 0.5) if denom > 0 else 0

        return {'acc': acc, 'rec': rec, 'pre': pre, 'f1': f1, 'mcc': mcc}

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

            train_loss, current_lr = self.train_epoch(epoch)
            dev_loss, dev_metrics = self.evaluate(self.dev_loader, self.dev_weights, 'dev')

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
                print(f'✓ New best model! Dev loss: {dev_loss:.4f}')

            print('-' * 60)

        print(f'\nTraining completed!')
        print(f'Best epoch: {self.best_epoch}, Best dev loss: {self.best_loss:.4f}')

        return self.visualizer


def main():
    import original_config as config
    flags = config.args

    check_gpu()
    set_seed(flags.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if device.type == 'cpu':
        print("   Install CUDA PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118\n")

    vocab_path = flags.dataset_save_dir + 'vocab.pkl'
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    word2idx = vocab_data['word2idx']
    vocab_size = len(word2idx)
    print(f'Vocabulary size: {vocab_size}')

    metadata_path = flags.dataset_save_dir + 'metadata.pkl'
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    print(f'Number of stocks: {metadata["num_stocks"]}')
    print(f'Stocks: {metadata["stocks"]}')

    train_data = OriginalHANDataset(
        flags.train_x_path, flags.train_y_path,
        flags.days, flags.max_num_tweets_len, flags.max_num_tokens_len
    )
    dev_data = OriginalHANDataset(
        flags.dev_x_path, flags.dev_y_path,
        flags.days, flags.max_num_tweets_len, flags.max_num_tokens_len
    )
    test_data = OriginalHANDataset(
        flags.test_x_path, flags.test_y_path,
        flags.days, flags.max_num_tweets_len, flags.max_num_tokens_len
    )

    train_loader = DataLoader(
        train_data,
        batch_size=flags.batch_size,
        shuffle=True,
        num_workers=flags.num_workers,
        pin_memory=flags.pin_memory if torch.cuda.is_available() else False,
        persistent_workers=True if flags.num_workers > 0 else False
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=flags.batch_size,
        shuffle=False,
        num_workers=flags.num_workers,
        pin_memory=flags.pin_memory if torch.cuda.is_available() else False,
        persistent_workers=True if flags.num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_data,
        batch_size=flags.batch_size,
        shuffle=False,
        num_workers=flags.num_workers,
        pin_memory=flags.pin_memory if torch.cuda.is_available() else False,
        persistent_workers=True if flags.num_workers > 0 else False
    )

    model = OriginalHAN(
        vocab_size=vocab_size,
        embedding_dim=flags.embedding_dim,
        hidden_dim=flags.hidden_dim,
        num_classes=flags.num_class,
        dropout=flags.dr
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {num_params:,}')

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=flags.learning_rate,
        weight_decay=flags.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=flags.learning_rate,
        epochs=flags.train_epochs,
        steps_per_epoch=len(train_loader)
    )

    trainer = Trainer(
        model, train_loader, dev_loader, optimizer, scheduler,
        device, train_data.class_weights, dev_data.class_weights,
        save_dir=flags.save_dir
    )

    print(f"\nTraining Configuration:")
    print(f"  Batch size: {flags.batch_size}")
    print(f"  Learning rate: {flags.learning_rate}")
    print(f"  Num workers: {flags.num_workers}")
    print(f"  Pin memory: {flags.pin_memory if torch.cuda.is_available() else False}")
    print()

    visualizer = trainer.train(flags.train_epochs)

    checkpoint = torch.load(trainer.save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss, test_metrics = trainer.evaluate(test_loader, test_data.class_weights, 'test')

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for source, targets in test_loader:
            source = source.to(device)
            output = model(source)
            pred = F.softmax(output, dim=1).argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(targets.numpy())

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_targets, all_preds)

    visualizer.generate_all_plots(
        confusion_matrix=cm,
        test_metrics={
            'loss': test_loss,
            'acc': test_metrics['acc'],
            'pre': test_metrics['pre'],
            'rec': test_metrics['rec'],
            'f1': test_metrics['f1'],
            'mcc': test_metrics['mcc']
        }
    )


if __name__ == '__main__':
    main()