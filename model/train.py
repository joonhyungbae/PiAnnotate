"""
Training script for Fingering Correction Model (Transformer)
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

from .config import Config
from .model import FingeringCorrectionTransformer, CorrectionLoss, create_model
from .dataset import create_dataloaders, DatasetBuilder, save_samples, load_samples


class Trainer:
    """Trainer for fingering correction transformer"""
    
    def __init__(
        self,
        config: Config,
        model: Optional[nn.Module] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        device: Optional[str] = None
    ):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = None
        
        self.model = model.to(self.device) if model else None
        
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        self.current_epoch = 0
        self.global_step = 0
        
        self.writer = None
        self.history = defaultdict(list)
    
    def setup(self, samples_path: Optional[Path] = None):
        """Setup training components"""
        # Build or load data
        if self.train_loader is None:
            if samples_path and samples_path.exists():
                print(f"Loading samples from {samples_path}...")
                samples = load_samples(samples_path)
            else:
                print("Building samples from data...")
                builder = DatasetBuilder(self.config)
                samples = builder.build_samples()
                
                if samples_path:
                    save_samples(samples, samples_path)
            
            self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
                self.config, samples
            )
        
        # Create model
        if self.model is None:
            self.model = create_model(self.config).to(self.device)
        
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")
        
        # Optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.98)
        )
        
        # Learning rate scheduler with warmup
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            return 1.0
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Loss function
        self.loss_fn = CorrectionLoss(
            num_classes=self.config.num_classes,
            label_smoothing=self.config.label_smoothing
        )
        
        # TensorBoard
        if HAS_TENSORBOARD:
            log_dir = self.config.output_dir / "logs"
            self.writer = SummaryWriter(log_dir)
        
        # Save config
        self.config.save(self.config.output_dir / "config.json")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_metrics = defaultdict(list)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in pbar:
            features = batch['features'].to(self.device)
            targets = batch['targets'].to(self.device)
            original_classes = batch['original_classes'].to(self.device)
            note_mask = batch['note_mask'].to(self.device)
            seq_mask = batch['seq_mask'].to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            correction_logits, class_logits = self.model(
                features, note_mask, seq_mask, original_classes
            )
            
            # Loss
            loss, metrics = self.loss_fn(
                correction_logits, class_logits,
                original_classes, targets, note_mask
            )
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.gradient_clip
            )
            self.optimizer.step()
            self.scheduler.step()
            
            self.global_step += 1
            
            # Record
            for k, v in metrics.items():
                if k != 'num_samples':
                    epoch_metrics[k].append(v)
            
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'corr_acc': f"{metrics['correction_acc']:.4f}",
                'overall': f"{metrics['overall_acc']:.4f}"
            })
        
        return {k: np.mean(v) for k, v in epoch_metrics.items()}
    
    @torch.no_grad()
    def validate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Validate the model"""
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        
        all_metrics = defaultdict(list)
        
        # Per-class accuracy (for correction cases)
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        for batch in loader:
            features = batch['features'].to(self.device)
            targets = batch['targets'].to(self.device)
            original_classes = batch['original_classes'].to(self.device)
            note_mask = batch['note_mask'].to(self.device)
            seq_mask = batch['seq_mask'].to(self.device)
            
            correction_logits, class_logits = self.model(
                features, note_mask, seq_mask, original_classes
            )
            loss, metrics = self.loss_fn(
                correction_logits, class_logits,
                original_classes, targets, note_mask
            )
            
            for k, v in metrics.items():
                if k != 'num_samples':
                    all_metrics[k].append(v)
            
            # Per-class accuracy (for samples that need correction)
            needs_correction = (original_classes != targets) & note_mask
            class_preds = class_logits.argmax(dim=-1)
            
            for cls in range(1, self.config.num_classes):
                cls_mask = (targets == cls) & needs_correction
                if cls_mask.sum() > 0:
                    cls_correct = ((class_preds == targets) & cls_mask).sum().item()
                    class_correct[cls] += cls_correct
                    class_total[cls] += cls_mask.sum().item()
        
        result = {k: np.mean(v) for k, v in all_metrics.items()}
        
        # Per-class accuracy
        per_class_acc = {}
        for cls in range(1, self.config.num_classes):
            if class_total[cls] > 0:
                per_class_acc[cls] = class_correct[cls] / class_total[cls]
        result['per_class_accuracy'] = per_class_acc
        
        return result
    
    def train(self, num_epochs: Optional[int] = None):
        """Full training loop"""
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        has_val = self.val_loader is not None
        
        print(f"\nStarting training for {num_epochs} epochs...")
        if not has_val:
            print("  (No validation set - using test set for evaluation)")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # Train
            train_metrics = self.train_epoch()
            
            # Evaluate on val or test
            if has_val:
                eval_metrics = self.validate(self.val_loader)
                eval_name = "Val"
            else:
                eval_metrics = self.validate(self.test_loader)
                eval_name = "Test"
            
            # Log
            self._log_metrics(train_metrics, eval_metrics)
            
            # Check improvement (based on overall accuracy)
            current_acc = eval_metrics.get('overall_acc', 0)
            if current_acc > self.best_val_acc:
                self.best_val_acc = current_acc
                self.best_val_loss = eval_metrics['loss']
                self.epochs_without_improvement = 0
                self.save_checkpoint('best.pt')
                print(f"  New best! Overall Acc: {current_acc:.4f}")
            else:
                self.epochs_without_improvement += 1
            
            # Early stopping (also works based on test set)
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"\nEarly stopping after {epoch + 1} epochs (no improvement for {self.config.early_stopping_patience} epochs)")
                break
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch+1}.pt')
            
            # Print progress
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Corr Acc: {train_metrics.get('correction_acc', 0):.4f}, "
                  f"Overall: {train_metrics.get('overall_acc', 0):.4f}")
            print(f"  {eval_name:5s} - Loss: {eval_metrics['loss']:.4f}, "
                  f"Corr Acc: {eval_metrics.get('correction_acc', 0):.4f}, "
                  f"Overall: {eval_metrics.get('overall_acc', 0):.4f}")
        
        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed/60:.1f} minutes")
        print(f"Best {eval_name} Acc: {self.best_val_acc:.4f}")
        
        self.save_checkpoint('final.pt')
        
        if self.writer:
            self.writer.close()
    
    def _log_metrics(self, train_metrics: Dict, val_metrics: Dict):
        """Log metrics"""
        epoch = self.current_epoch
        
        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                if self.writer:
                    self.writer.add_scalar(f'train/{key}', value, epoch)
                self.history[f'train_{key}'].append(value)
        
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                if self.writer:
                    self.writer.add_scalar(f'val/{key}', value, epoch)
                self.history[f'val_{key}'].append(value)
        
        if self.writer:
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        path = self.config.output_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config': {
                'd_model': self.config.d_model,
                'nhead': self.config.nhead,
                'num_encoder_layers': self.config.num_encoder_layers,
                'num_decoder_layers': self.config.num_decoder_layers,
                'dim_feedforward': self.config.dim_feedforward,
                'dropout': self.config.dropout,
                'num_classes': self.config.num_classes,
                'max_notes_per_group': self.config.max_notes_per_group,
            }
        }
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        print(f"Loaded checkpoint from {path}")
        print(f"  Epoch: {self.current_epoch}, Best Acc: {self.best_val_acc:.4f}")


def main():
    """Main training script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train fingering transformer")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--samples-path', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='model/checkpoints')
    parser.add_argument('--max-seq-len', type=int, default=128)
    args = parser.parse_args()
    
    config = Config(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        dropout=args.dropout,
        output_dir=Path(args.output_dir)
    )
    
    trainer = Trainer(config)
    
    samples_path = Path(args.samples_path) if args.samples_path else config.output_dir / "samples_v2.pkl"
    trainer.setup(samples_path)
    trainer.train()


if __name__ == "__main__":
    main()
