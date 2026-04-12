#!/usr/bin/env python3
"""
Held-out training wrapper for the diagnostic probe.

Excludes the 62 R2-checked pieces from the training pool, trains
the same architecture on the remaining 91 R1-only pieces, and
saves the resulting checkpoint to a separate directory so that
the original best.pt is untouched.

Usage (from repo root):
    CUDA_VISIBLE_DEVICES=1 python experiments/holdout_train.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch

from model.config import Config
from model.dataset import DatasetBuilder, create_dataloaders, save_samples
from model.train import Trainer
from model.utils import set_seed


def select_train_pieces() -> list[int]:
    """All R1-checked pieces that are NOT in R2 (the held-out set)."""
    sd = REPO / "data" / "metadata" / "annotation" / "status"
    train = []
    for p in sorted(sd.glob("*.json")):
        j = json.loads(p.read_text())
        pid = int(p.stem)
        if not j.get("review1"):
            continue
        if j.get("review2"):
            continue  # held out
        train.append(pid)
    return train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-rule-embed", action="store_true",
                    help="Zero out and freeze the original_class_embedding "
                         "(from-scratch baseline w/o rule-label input).")
    ap.add_argument("--output-suffix", type=str, default="",
                    help="Suffix appended to checkpoints_holdout dir name.")
    ap.add_argument("--train-pieces-file", type=str, default=None,
                    help="JSON file with {\"train\": [piece_ids]}; "
                         "overrides the default not-R2 selection.")
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=256)
    args = ap.parse_args()

    set_seed(args.seed)

    if args.train_pieces_file:
        spec = json.loads(Path(args.train_pieces_file).read_text())
        train_pieces = sorted(spec["train"])
        print(f"Held-out training: {len(train_pieces)} pieces "
              f"(custom from {args.train_pieces_file})")
    else:
        train_pieces = select_train_pieces()
        print(f"Held-out training: {len(train_pieces)} pieces "
              f"(R1-checked, NOT in R2)")
    print(f"seed={args.seed} no_rule_embed={args.no_rule_embed} "
          f"layers={args.num_layers} d_model={args.d_model}")

    suffix = args.output_suffix or f"_seed{args.seed}" + (
        "_norule" if args.no_rule_embed else "")
    output_dir = REPO / "model" / f"checkpoints_holdout{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        d_model=args.d_model,
        nhead=8 if args.d_model % 8 == 0 else 4,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        dim_feedforward=max(256, args.d_model * 2),
        dropout=0.1,
        num_epochs=200,                 # cap; early stopping will kick in
        batch_size=8,
        learning_rate=1e-4,
        weight_decay=1e-4,
        warmup_steps=1000,
        early_stopping_patience=20,
        label_smoothing=0.1,
        max_notes_per_group=8,
        onset_tolerance=3,
        random_seed=args.seed,
        output_dir=output_dir,
        project_root=REPO,
        dataset_path=REPO / "for_elise" / "dataset",
        fingering_path=REPO / "data" / "fingering",
        fingering_edited_path=REPO / "data" / "fingering_edited",
        meshes_path=REPO / "for_elise" / "piano_meshes",
    )

    # Save the train piece list so eval can verify
    (output_dir / "train_pieces.json").write_text(
        json.dumps({"train": train_pieces}, indent=2)
    )

    print("\nBuilding samples for held-out training set ...")
    builder = DatasetBuilder(config)
    samples = builder.build_samples(piece_ids=train_pieces, max_seq_len=128)
    save_samples(samples, output_dir / "samples_v2.pkl")

    print("\nCreating dataloaders ...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config, samples, max_seq_len=128
    )

    print("\nStarting training ...")
    trainer = Trainer(
        config=config, train_loader=train_loader,
        val_loader=val_loader, device=None
    )
    trainer.test_loader = test_loader
    trainer.setup()

    if args.no_rule_embed:
        # Zero out and freeze the rule-label embedding so the model
        # has no access to the rule fingering as input.
        with torch.no_grad():
            trainer.model.original_class_embedding.weight.zero_()
        trainer.model.original_class_embedding.weight.requires_grad = False
        print("--no-rule-embed: original_class_embedding frozen at zero")

    trainer.train()

    print("\nDone. Best checkpoint at:", output_dir / "best.pt")


if __name__ == "__main__":
    main()
