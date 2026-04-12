#!/usr/bin/env python3
"""
Run the held-out probe checkpoint on the 62 R2 pieces (held out
during training) and write the resulting AI fingerings to a
separate directory so that downstream evaluation can compare
them against the human-edited labels.

Usage (from repo root):
    CUDA_VISIBLE_DEVICES=1 python experiments/holdout_infer.py
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
from tqdm import tqdm

from model.config import Config
from model.batch_inference import process_piece, load_model

STATUS = REPO / "data" / "metadata" / "annotation" / "status"


def select_holdout_pieces(holdout_file: str | None = None) -> list[int]:
    """The 62 R2-checked pieces (held out during training), or
    a custom set from a JSON file with key 'holdout'."""
    if holdout_file:
        spec = json.loads(Path(holdout_file).read_text())
        return sorted(spec["holdout"])
    out = []
    for sp in sorted(STATUS.glob("*.json")):
        j = json.loads(sp.read_text())
        if j.get("review2"):
            out.append(int(sp.stem))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str,
                    default=str(REPO / "model" / "checkpoints_holdout" / "best.pt"))
    ap.add_argument("--output-dir", type=str,
                    default=str(REPO / "data" / "fingering_edited_ai_holdout"))
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--holdout-file", type=str, default=None,
                    help="JSON {\"holdout\": [piece_ids]}; default is "
                         "the 62 R2-checked pieces.")
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=256)
    args = ap.parse_args()
    CKPT = Path(args.checkpoint)
    OUT_DIR = Path(args.output_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    config = Config(
        d_model=args.d_model,
        nhead=8 if args.d_model % 8 == 0 else 4,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        dim_feedforward=max(256, args.d_model * 2),
        dropout=0.1,
        max_notes_per_group=8,
        onset_tolerance=3,
        random_seed=42,
        project_root=REPO,
        dataset_path=REPO / "for_elise" / "dataset",
        fingering_path=REPO / "data" / "fingering",
        fingering_edited_path=REPO / "data" / "fingering_edited",
        meshes_path=REPO / "for_elise" / "piano_meshes",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint {CKPT}")
    model = load_model(CKPT, config, device)

    pieces = select_holdout_pieces(args.holdout_file)
    print(f"Running held-out inference on {len(pieces)} pieces -> {OUT_DIR}")

    failed = []
    for pid in tqdm(pieces):
        try:
            fingering, _results, _missing = process_piece(
                pid, model, config, device,
                correction_threshold=args.threshold, handle_missing=True
            )
            with open(OUT_DIR / f"{pid:03d}.pkl", "wb") as f:
                pickle.dump(fingering, f)
        except Exception as e:
            failed.append({"piece": pid, "error": str(e)})

    if failed:
        print(f"\nFailed: {len(failed)}")
        for f in failed[:10]:
            print(" ", f)
    print("Done.")


if __name__ == "__main__":
    main()
