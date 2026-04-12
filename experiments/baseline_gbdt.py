#!/usr/bin/env python3
"""
Non-Transformer baseline: a per-note gradient-boosted classifier
on the same 77-d note features the Transformer probe sees.

Trains on the cached samples for the 91 non-R2 pieces (reusing
the held-out training cache from the main probe), then runs
inference on the 62 R2 held-out pieces with the same Eq. 2 gate
applied at the per-note level. Outputs are written in the same
fingering pkl format as holdout_infer.py so analyze.py and
per_piece_eval.py can score them unchanged.

Usage (from repo root):
    python experiments/baseline_gbdt.py --seed 0
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model.config import Config
from model.dataset import InferenceDatasetBuilder

STATUS = REPO / "data" / "metadata" / "annotation" / "status"
NUM_CLASSES = 11  # 0=missing, 1..10=hand+finger
GATE_TAU = 0.9
GATE_RATIO = 2.0


def load_train_samples(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Flatten cached training samples to per-note (X, y)."""
    cache = (REPO / f"model/checkpoints_holdout_norule_seed{seed}"
             / "samples_v2.pkl")
    print(f"Loading cached samples from {cache}")
    with open(cache, "rb") as f:
        samples = pickle.load(f)
    Xs, ys = [], []
    for s in samples:
        feats = s["features"]            # (seq, max_notes, 77)
        targets = s["targets"]           # (seq, max_notes)
        mask = s["note_mask"]            # (seq, max_notes)
        # vectorised flatten
        m = mask.astype(bool)
        Xs.append(feats[m])
        ys.append(targets[m])
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.int64)
    print(f"Train pool: {X.shape[0]:,} notes, {X.shape[1]} features")
    return X, y


def select_holdout_pieces() -> list[int]:
    out = []
    for sp in sorted(STATUS.glob("*.json")):
        j = json.loads(sp.read_text())
        if j.get("review2"):
            out.append(int(sp.stem))
    return out


def predict_piece(model: HistGradientBoostingClassifier,
                  config: Config, builder: InferenceDatasetBuilder,
                  piece_id: int):
    """
    Run GBDT inference on one piece. Returns the per-frame
    fingering structure used by analyze.py.
    """
    motion_path = config.dataset_path / f"{piece_id:03d}" / "motion.pkl"
    fingering_path = config.fingering_path / f"{piece_id:03d}.pkl"
    if not motion_path.exists() or not fingering_path.exists():
        raise FileNotFoundError(piece_id)
    with open(motion_path, "rb") as f:
        motion = pickle.load(f)
    with open(fingering_path, "rb") as f:
        original_fingering = pickle.load(f)

    samples = builder.build_samples_from_fingering(
        original_fingering=original_fingering,
        motion=motion,
        max_seq_len=128,
    )
    if not samples:
        return original_fingering

    finger_names = {1: "thumb", 2: "index", 3: "middle",
                    4: "ring", 5: "pinky"}

    n_frames = len(original_fingering)
    fingering_out = [list(f) if f else [] for f in original_fingering]

    # Build a map from (frame, key_idx) -> entry index for in-place edit
    for sample in samples:
        feats = sample["features"]            # (seq, max_notes, 77)
        mask = sample["note_mask"].astype(bool)
        orig = sample["original_classes"]     # (seq, max_notes)
        note_infos = sample["note_infos"]
        seq_len = feats.shape[0]
        max_notes = feats.shape[1]
        flat_X = feats[mask]
        if flat_X.size == 0:
            continue
        proba = model.predict_proba(flat_X)   # (N, n_classes)
        cls_to_idx = {c: i for i, c in enumerate(model.classes_)}
        # Walk per group/note in original order
        idx = 0
        for s in range(seq_len):
            for n in range(max_notes):
                if not mask[s, n]:
                    continue
                p = proba[idx]; idx += 1
                rule_class = int(orig[s, n])
                top1 = int(model.classes_[int(np.argmax(p))])
                top1_p = float(p[int(np.argmax(p))])
                rule_p = float(p[cls_to_idx[rule_class]]) \
                    if rule_class in cls_to_idx else 1e-6
                # Eq. 2 gate
                gate = (
                    (top1 != rule_class) and
                    (top1_p > GATE_TAU) and
                    (top1_p / max(rule_p, 1e-6) > GATE_RATIO)
                )
                # Missing (rule=0) → always use top1
                if rule_class == 0 and top1 > 0:
                    final_class = top1
                    was_corr = True
                elif gate and top1 > 0:
                    final_class = top1
                    was_corr = True
                else:
                    final_class = rule_class
                    was_corr = False
                if final_class == 0:
                    continue
                # Translate class to (hand, finger)
                hand, finger = Config.class_to_hand_finger(final_class)
                # Find the matching note onset frames and replace/insert
                info = note_infos[s][n]
                onset = info["onset_frame"]
                offset = info.get("offset_frame", onset)
                key_idx = info["key_idx"]
                onset = max(0, min(onset, n_frames - 1))
                offset = max(onset, min(offset, n_frames - 1))
                entry = {
                    "key_index": key_idx,
                    "hand": hand,
                    "finger": finger,
                    "finger_name": finger_names.get(finger, ""),
                    "was_corrected": was_corr,
                    "correction_prob": float(1.0 - rule_p),
                    "is_missing": rule_class == 0,
                }
                for fr in range(onset, offset + 1):
                    # remove any existing entry on the same key/hand
                    fingering_out[fr] = [
                        e for e in fingering_out[fr]
                        if not (e.get("key_index") == key_idx
                                and e.get("hand") == hand)
                    ]
                    fingering_out[fr].append(entry)
    return fingering_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0,
                    help="Reuse the cached samples from the corresponding "
                         "no-rule-embed seed run.")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=8)
    args = ap.parse_args()

    X, y = load_train_samples(args.seed)
    print(f"Training HistGradientBoostingClassifier "
          f"(max_iter={args.max_iter}, max_depth={args.max_depth})")
    t0 = time.time()
    model = HistGradientBoostingClassifier(
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        learning_rate=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=args.seed,
        verbose=0,
    )
    model.fit(X, y)
    print(f"Trained in {time.time() - t0:.1f}s. "
          f"Classes: {model.classes_.tolist()}")

    config = Config(
        d_model=64, nhead=4,
        num_encoder_layers=1, num_decoder_layers=1,
        dim_feedforward=256, dropout=0.1,
        max_notes_per_group=8, onset_tolerance=3,
        random_seed=args.seed,
        project_root=REPO,
        dataset_path=REPO / "for_elise" / "dataset",
        fingering_path=REPO / "data" / "fingering",
        fingering_edited_path=REPO / "data" / "fingering_edited",
        meshes_path=REPO / "for_elise" / "piano_meshes",
    )
    builder = InferenceDatasetBuilder(config)

    out_dir = REPO / f"data/fingering_edited_ai_holdout_gbdt_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pieces = select_holdout_pieces()
    print(f"GBDT inference on {len(pieces)} R2 pieces -> {out_dir}")
    failed = []
    for pid in tqdm(pieces):
        try:
            fingering = predict_piece(model, config, builder, pid)
            with open(out_dir / f"{pid:03d}.pkl", "wb") as f:
                pickle.dump(fingering, f)
        except Exception as e:
            failed.append({"piece": pid, "error": str(e)})
    if failed:
        print(f"\nFailed: {len(failed)}")
        for f in failed[:5]:
            print(" ", f)
    print("Done.")


if __name__ == "__main__":
    main()
