import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
import json

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class TrainingVisualizer:
    def __init__(self, save_dir='visualizations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            'train_loss': [],
            'dev_loss': [],
            'train_acc': [],
            'dev_acc': [],
            'dev_precision': [],
            'dev_recall': [],
            'dev_f1': [],
            'dev_mcc': [],
            'learning_rates': [],
            'epochs': []
        }

    def update(self, epoch, train_loss, dev_metrics, learning_rate=None):
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['dev_loss'].append(dev_metrics.get('loss', 0))
        self.history['dev_acc'].append(dev_metrics.get('acc', 0))
        self.history['dev_precision'].append(dev_metrics.get('pre', 0))
        self.history['dev_recall'].append(dev_metrics.get('rec', 0))
        self.history['dev_f1'].append(dev_metrics.get('f1', 0))
        self.history['dev_mcc'].append(dev_metrics.get('mcc', 0))
        if learning_rate:
            self.history['learning_rates'].append(learning_rate)

    def plot_loss_curves(self, save=True, show=True):
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = self.history['epochs']
        ax.plot(epochs, self.history['train_loss'], 'b-o', label='Training Loss', linewidth=2, markersize=6)
        ax.plot(epochs, self.history['dev_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=6)

        best_idx = np.argmin(self.history['dev_loss'])
        best_epoch = self.history['epochs'][best_idx]
        best_loss = self.history['dev_loss'][best_idx]
        ax.plot(best_epoch, best_loss, 'g*', markersize=20, label=f'Best (Epoch {best_epoch})')

        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        if save:
            plt.savefig(self.save_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.save_dir / 'loss_curves.png'}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_accuracy_curves(self, save=True, show=True):
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = self.history['epochs']
        ax.plot(epochs, self.history['dev_acc'], 'g-o', label='Validation Accuracy', linewidth=2, markersize=6)

        best_idx = np.argmax(self.history['dev_acc'])
        best_epoch = self.history['epochs'][best_idx]
        best_acc = self.history['dev_acc'][best_idx]
        ax.plot(best_epoch, best_acc, 'r*', markersize=20, label=f'Best: {best_acc:.4f} (Epoch {best_epoch})')

        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (50%)')
        ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Good (60%)')

        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Validation Accuracy Over Time', fontsize=14, fontweight='bold')
        ax.set_ylim([0.45, max(self.history['dev_acc']) + 0.05])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        if save:
            plt.savefig(self.save_dir / 'accuracy_curves.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.save_dir / 'accuracy_curves.png'}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_all_metrics(self, save=True, show=True):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Comprehensive Training Metrics', fontsize=16, fontweight='bold')

        epochs = self.history['epochs']

        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.history['dev_loss'], 'r-s', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch', fontweight='bold')
        axes[0, 0].set_ylabel('Loss', fontweight='bold')
        axes[0, 0].set_title('Loss Curves', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, self.history['dev_acc'], 'g-o', label='Accuracy', linewidth=2)
        axes[0, 1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('Epoch', fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy', fontweight='bold')
        axes[0, 1].set_title('Validation Accuracy', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(epochs, self.history['dev_precision'], 'b-o', label='Precision', linewidth=2)
        axes[1, 0].plot(epochs, self.history['dev_recall'], 'r-s', label='Recall', linewidth=2)
        axes[1, 0].plot(epochs, self.history['dev_f1'], 'g-^', label='F1 Score', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontweight='bold')
        axes[1, 0].set_ylabel('Score', fontweight='bold')
        axes[1, 0].set_title('Precision, Recall, F1', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(epochs, self.history['dev_mcc'], 'm-o', label='MCC', linewidth=2)
        axes[1, 1].axhline(y=0.0, color='gray', linestyle='--', alpha=0.5, label='Random')
        axes[1, 1].set_xlabel('Epoch', fontweight='bold')
        axes[1, 1].set_ylabel('MCC', fontweight='bold')
        axes[1, 1].set_title('Matthews Correlation Coefficient', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.save_dir / 'all_metrics.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.save_dir / 'all_metrics.png'}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_confusion_matrix(self, confusion_matrix, labels=['Down', 'Up'],
                              normalize=False, save=True, show=True, title='Confusion Matrix'):
        fig, ax = plt.subplots(figsize=(8, 6))

        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title += ' (Normalized)'
        else:
            cm = confusion_matrix
            fmt = 'd'

        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', square=True,
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    cbar_kws={'label': 'Percentage' if normalize else 'Count'})

        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')

        tp, fn = confusion_matrix[0, 0], confusion_matrix[0, 1]
        fp, tn = confusion_matrix[1, 0], confusion_matrix[1, 1]

        accuracy = (tp + tn) / confusion_matrix.sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        textstr = f'Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(1.4, 0.5, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', bbox=props)

        if save:
            filename = title.lower().replace(' ', '_') + '.png'
            plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.save_dir / filename}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_learning_rate(self, save=True, show=True):
        if not self.history['learning_rates']:
            print("No learning rate history recorded")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = self.history['epochs']
        ax.plot(epochs, self.history['learning_rates'], 'b-o', linewidth=2, markersize=6)

        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        if save:
            plt.savefig(self.save_dir / 'learning_rate.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.save_dir / 'learning_rate.png'}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_metrics_comparison(self, save=True, show=True):
        fig, ax = plt.subplots(figsize=(10, 6))

        final_metrics = {
            'Accuracy': self.history['dev_acc'][-1],
            'Precision': self.history['dev_precision'][-1],
            'Recall': self.history['dev_recall'][-1],
            'F1 Score': self.history['dev_f1'][-1],
            'MCC': (self.history['dev_mcc'][-1] + 1) / 2  # Normalize MCC to 0-1 for viz
        }

        metrics = list(final_metrics.keys())
        values = list(final_metrics.values())
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

        bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Final Validation Metrics', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        if save:
            plt.savefig(self.save_dir / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.save_dir / 'metrics_comparison.png'}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_overfitting_analysis(self, save=True, show=True):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        epochs = self.history['epochs']

        loss_gap = np.array(self.history['dev_loss']) - np.array(self.history['train_loss'])
        ax1.plot(epochs, loss_gap, 'r-o', linewidth=2, markersize=6)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.fill_between(epochs, 0, loss_gap, where=(loss_gap > 0),
                         color='red', alpha=0.3, label='Overfitting')
        ax1.fill_between(epochs, 0, loss_gap, where=(loss_gap < 0),
                         color='green', alpha=0.3, label='Underfitting')
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Validation Loss - Training Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Overfitting Analysis (Loss Gap)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, self.history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
        ax2.plot(epochs, self.history['dev_loss'], 'r-s', label='Val Loss', linewidth=2)

        overfit_start = None
        for i in range(1, len(epochs)):
            if self.history['train_loss'][i] < self.history['train_loss'][i - 1] and \
                    self.history['dev_loss'][i] > self.history['dev_loss'][i - 1]:
                if overfit_start is None:
                    overfit_start = i

        if overfit_start:
            ax2.axvspan(epochs[overfit_start], epochs[-1], alpha=0.2, color='red',
                        label='Potential Overfitting')

        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax2.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.save_dir / 'overfitting_analysis.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.save_dir / 'overfitting_analysis.png'}")

        if show:
            plt.show()
        else:
            plt.close()

    def create_summary_report(self, test_metrics=None, save=True):
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        epochs = self.history['epochs']

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, self.history['train_loss'], 'b-o', label='Train', linewidth=2)
        ax1.plot(epochs, self.history['dev_loss'], 'r-s', label='Val', linewidth=2)
        ax1.set_title('Loss Curves', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs, self.history['dev_acc'], 'g-o', linewidth=2)
        ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax2.set_title('Validation Accuracy', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(epochs, self.history['dev_f1'], 'm-^', linewidth=2)
        ax3.set_title('F1 Score', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(epochs, self.history['dev_precision'], 'b-o', label='Precision', linewidth=2)
        ax4.plot(epochs, self.history['dev_recall'], 'r-s', label='Recall', linewidth=2)
        ax4.set_title('Precision & Recall', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(epochs, self.history['dev_mcc'], 'c-o', linewidth=2)
        ax5.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5)
        ax5.set_title('MCC', fontweight='bold')
        ax5.grid(True, alpha=0.3)

        ax6 = fig.add_subplot(gs[1, 2])
        final_metrics = {
            'Acc': self.history['dev_acc'][-1],
            'Prec': self.history['dev_precision'][-1],
            'Rec': self.history['dev_recall'][-1],
            'F1': self.history['dev_f1'][-1]
        }
        ax6.bar(final_metrics.keys(), final_metrics.values(),
                color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'], alpha=0.7)
        ax6.set_title('Final Metrics', fontweight='bold')
        ax6.set_ylim([0, 1])
        ax6.grid(True, axis='y', alpha=0.3)

        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')

        best_epoch = np.argmin(self.history['dev_loss'])
        summary_text = f"""
        TRAINING SUMMARY
        {'=' * 80}

        Best Epoch: {self.history['epochs'][best_epoch]}

        Validation Metrics at Best Epoch:
        • Loss:      {self.history['dev_loss'][best_epoch]:.4f}
        • Accuracy:  {self.history['dev_acc'][best_epoch]:.4f}
        • Precision: {self.history['dev_precision'][best_epoch]:.4f}
        • Recall:    {self.history['dev_recall'][best_epoch]:.4f}
        • F1 Score:  {self.history['dev_f1'][best_epoch]:.4f}
        • MCC:       {self.history['dev_mcc'][best_epoch]:.4f}

        Final Validation Metrics (Epoch {self.history['epochs'][-1]}):
        • Loss:      {self.history['dev_loss'][-1]:.4f}
        • Accuracy:  {self.history['dev_acc'][-1]:.4f}
        • Precision: {self.history['dev_precision'][-1]:.4f}
        • Recall:    {self.history['dev_recall'][-1]:.4f}
        • F1 Score:  {self.history['dev_f1'][-1]:.4f}
        • MCC:       {self.history['dev_mcc'][-1]:.4f}
        """

        if test_metrics:
            summary_text += f"""
        Test Set Metrics:
        • Loss:      {test_metrics.get('loss', 0):.4f}
        • Accuracy:  {test_metrics.get('acc', 0):.4f}
        • Precision: {test_metrics.get('pre', 0):.4f}
        • Recall:    {test_metrics.get('rec', 0):.4f}
        • F1 Score:  {test_metrics.get('f1', 0):.4f}
        • MCC:       {test_metrics.get('mcc', 0):.4f}
            """

        ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle('Comprehensive Training Report', fontsize=16, fontweight='bold')

        if save:
            plt.savefig(self.save_dir / 'training_report.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {self.save_dir / 'training_report.png'}")

        plt.show()

    def save_history(self, filename='training_history.json'):
        filepath = self.save_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"✓ Saved training history: {filepath}")

    def generate_all_plots(self, confusion_matrix=None, test_metrics=None):
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70 + "\n")

        print("1. Loss curves...")
        self.plot_loss_curves(show=False)

        print("2. Accuracy curves...")
        self.plot_accuracy_curves(show=False)

        print("3. All metrics...")
        self.plot_all_metrics(show=False)

        print("4. Metrics comparison...")
        self.plot_metrics_comparison(show=False)

        print("5. Overfitting analysis...")
        self.plot_overfitting_analysis(show=False)

        if self.history['learning_rates']:
            print("6. Learning rate schedule...")
            self.plot_learning_rate(show=False)

        if confusion_matrix is not None:
            print("7. Confusion matrix...")
            self.plot_confusion_matrix(confusion_matrix, show=False)

        print("8. Comprehensive report...")
        self.create_summary_report(test_metrics, save=True)

        print("9. Saving training history...")
        self.save_history()

        print("\n" + "=" * 70)
        print(f"ALL VISUALIZATIONS SAVED TO: {self.save_dir}")
        print("=" * 70 + "\n")