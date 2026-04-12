#!/usr/bin/env python3
"""
Main script to run fingering correction transformer training

Usage:
    python model/run_training.py --epochs 100
    python model/run_training.py --epochs 50 --d-model 128 --nhead 4
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.config import Config
from model.dataset import DatasetBuilder, create_dataloaders, save_samples, load_samples
from model.train import Trainer
from model.utils import set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Train fingering correction transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model settings
    parser.add_argument('--d-model', type=int, default=256,
                        help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=4,
                        help='Number of encoder/decoder layers')
    parser.add_argument('--dim-feedforward', type=int, default=1024,
                        help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                        help='Warmup steps for learning rate')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing')
    
    # Data settings
    parser.add_argument('--max-seq-len', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--max-notes', type=int, default=8,
                        help='Maximum notes per chord')
    parser.add_argument('--onset-tolerance', type=int, default=3,
                        help='Frames tolerance for simultaneous notes')
    
    # Paths
    parser.add_argument('--output-dir', type=str, default='model/checkpoints',
                        help='Output directory for checkpoints')
    parser.add_argument('--samples-path', type=str, default=None,
                        help='Path to pre-built samples')
    parser.add_argument('--rebuild-samples', action='store_true',
                        help='Force rebuild samples')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Create config
    config = Config(
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        early_stopping_patience=args.patience,
        label_smoothing=args.label_smoothing,
        max_notes_per_group=args.max_notes,
        onset_tolerance=args.onset_tolerance,
        random_seed=args.seed,
        output_dir=Path(args.output_dir)
    )
    
    print("=" * 60)
    print("FINGERING CORRECTION TRANSFORMER TRAINING")
    print("=" * 60)
    print(f"\nModel Configuration:")
    print(f"  d_model: {args.d_model}")
    print(f"  nhead: {args.nhead}")
    print(f"  num_layers: {args.num_layers}")
    print(f"  dim_feedforward: {args.dim_feedforward}")
    print(f"\nTraining Configuration:")
    print(f"  epochs: {args.epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  learning_rate: {args.lr}")
    print(f"  max_seq_len: {args.max_seq_len}")
    print(f"\nOutput directory: {config.output_dir}")
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Samples path
    if args.samples_path:
        samples_path = Path(args.samples_path)
    else:
        samples_path = config.output_dir / "samples_v2.pkl"
    
    # Build or load samples
    if samples_path.exists() and not args.rebuild_samples:
        print(f"\nLoading cached samples from {samples_path}...")
        samples = load_samples(samples_path)
    else:
        print("\nBuilding note-level samples...")
        builder = DatasetBuilder(config)
        samples = builder.build_samples(max_seq_len=args.max_seq_len)
        save_samples(samples, samples_path)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config, samples, max_seq_len=args.max_seq_len
    )
    
    # Create trainer
    trainer = Trainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device
    )
    trainer.test_loader = test_loader
    
    trainer.setup()
    trainer.train()
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 60)
    
    best_checkpoint = config.output_dir / "best.pt"
    if best_checkpoint.exists():
        trainer.load_checkpoint(best_checkpoint)
    
    test_metrics = trainer.validate(test_loader)
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Overall Accuracy: {test_metrics['overall_acc']:.4f}")
    print(f"  Correction Accuracy: {test_metrics.get('correction_acc', 0):.4f}")
    
    print("\nPer-class accuracy (correction cases):")
    for cls, acc in sorted(test_metrics['per_class_accuracy'].items()):
        hand, finger = Config.class_to_hand_finger(cls)
        finger_name = config.number_to_finger.get(finger, str(finger))
        print(f"  {hand:5s} {finger_name:8s}: {acc:.4f}")
    
    # Save results
    import json
    results = {
        'loss': test_metrics['loss'],
        'overall_acc': test_metrics['overall_acc'],
        'correction_acc': test_metrics.get('correction_acc', 0),
        'per_class_accuracy': {str(k): v for k, v in test_metrics['per_class_accuracy'].items()}
    }
    with open(config.output_dir / "test_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTraining complete!")
    print(f"Best model saved to {best_checkpoint}")


if __name__ == "__main__":
    main()
