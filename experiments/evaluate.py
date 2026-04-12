#!/usr/bin/env python3
"""
Held-out evaluation of the Fingering Correction Transformer.

Computes, on a chosen split of pieces:
  - Rule-vs-edited per-note agreement (baseline)
  - AI-vs-edited per-note agreement (model)
  - Per-class accuracy (fingers 1..10)
  - Correction-head precision/recall: did the model flag the notes
    that the rule actually got wrong?
  - Optional: same numbers restricted to notes the rule mislabeled
    (the only notes the corrector can possibly improve)

Splits:
  --split r2     -> pieces with review2 completed (default)
  --split r3     -> pieces with review3 completed
  --split all    -> every R1-checked piece
  --split file:PATH -> newline-separated piece ids in PATH

Run from repo root:
    python experiments/evaluate.py --split r2 \
        > experiments/eval_r2.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
FING_DIR = DATA / "fingering"
EDIT_DIR = DATA / "fingering_edited"
AI_DIR = DATA / "fingering_edited_ai_v2"
STATUS_DIR = DATA / "metadata" / "annotation" / "status"


def load_pkl(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


def select_pieces(split: str) -> list[int]:
    ids = sorted(int(p.stem) for p in FING_DIR.glob("*.pkl") if p.stem.isdigit())
    if split == "all":
        return [i for i in ids if (STATUS_DIR / f"{i:03d}.json").exists()]
    if split.startswith("file:"):
        path = Path(split[5:])
        return sorted({int(x) for x in path.read_text().split() if x.strip()})
    field = {"r1": "review1", "r2": "review2", "r3": "review3"}[split]
    out = []
    for i in ids:
        sp = STATUS_DIR / f"{i:03d}.json"
        if not sp.exists():
            continue
        j = json.loads(sp.read_text())
        if j.get(field):
            out.append(i)
    return out


def index_frame(frame: list) -> dict:
    """Map (key_index, hand) -> entry dict, ignoring entries without key_index."""
    return {(e["key_index"], e["hand"]): e for e in (frame or []) if "key_index" in e}


def evaluate_piece(rule, edited, ai):
    """
    Returns dict of counters/totals for one piece.
    Aligns frames by index and notes by (key_index, hand).
    """
    n_frames = max(len(rule or []), len(edited or []), len(ai or []))
    stats = {
        "notes_total":   0,   # notes that exist in edited
        "rule_correct":  0,   # rule == edited
        "ai_correct":    0,   # ai   == edited
        # restricted to notes the rule got wrong
        "wrong_in_rule":     0,
        "ai_correct_on_wrong": 0,  # of those, AI fixed it
        # correction-head behaviour (uses ai entry's `was_corrected` flag)
        "tp_correction": 0,  # was_corrected & rule != edited
        "fp_correction": 0,  # was_corrected & rule == edited
        "fn_correction": 0,  # !was_corrected & rule != edited
        "tn_correction": 0,  # !was_corrected & rule == edited
        # per-class accuracy: indexed by ground-truth finger (1..10)
        "per_class_total":   Counter(),
        "per_class_ai_ok":   Counter(),
        "per_class_rule_ok": Counter(),
    }
    for i in range(n_frames):
        ef = edited[i] if i < len(edited or []) else None
        if not ef:
            continue
        rf = rule[i]   if i < len(rule or [])   else None
        af = ai[i]     if i < len(ai or [])     else None
        rmap = index_frame(rf)
        amap = index_frame(af)
        for e in ef:
            if "key_index" not in e:
                continue
            k = (e["key_index"], e["hand"])
            gt = e.get("finger")
            stats["notes_total"] += 1
            stats["per_class_total"][gt] += 1
            r_entry = rmap.get(k)
            a_entry = amap.get(k)
            r_pred = r_entry.get("finger") if r_entry else None
            a_pred = a_entry.get("finger") if a_entry else None
            rule_ok = (r_pred == gt)
            ai_ok = (a_pred == gt)
            if rule_ok:
                stats["rule_correct"] += 1
                stats["per_class_rule_ok"][gt] += 1
            else:
                stats["wrong_in_rule"] += 1
                if ai_ok:
                    stats["ai_correct_on_wrong"] += 1
            if ai_ok:
                stats["ai_correct"] += 1
                stats["per_class_ai_ok"][gt] += 1
            # correction-head signal lives on the AI entry
            if a_entry is not None and "was_corrected" in a_entry:
                was = bool(a_entry["was_corrected"])
                rule_wrong = not rule_ok
                if was and rule_wrong:
                    stats["tp_correction"] += 1
                elif was and not rule_wrong:
                    stats["fp_correction"] += 1
                elif (not was) and rule_wrong:
                    stats["fn_correction"] += 1
                else:
                    stats["tn_correction"] += 1
    return stats


def merge(into: dict, other: dict) -> None:
    for k, v in other.items():
        if isinstance(v, Counter):
            into.setdefault(k, Counter()).update(v)
        else:
            into[k] = into.get(k, 0) + v


def safe_div(a, b):
    return (a / b) if b else None


def summarise(agg: dict) -> dict:
    n = agg["notes_total"]
    wr = agg["wrong_in_rule"]
    tp, fp, fn, tn = (agg["tp_correction"], agg["fp_correction"],
                      agg["fn_correction"], agg["tn_correction"])
    return {
        "notes_total":          n,
        "rule_agreement":       safe_div(agg["rule_correct"], n),
        "ai_agreement":         safe_div(agg["ai_correct"], n),
        "absolute_improvement": (
            safe_div(agg["ai_correct"] - agg["rule_correct"], n)
        ),
        "relative_error_reduction": (
            safe_div(agg["ai_correct"] - agg["rule_correct"],
                     n - agg["rule_correct"])
        ),
        "wrong_in_rule":        wr,
        "ai_fix_rate_on_wrong": safe_div(agg["ai_correct_on_wrong"], wr),
        "correction_head": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": safe_div(tp, tp + fp),
            "recall":    safe_div(tp, tp + fn),
            "f1":        safe_div(2 * tp, 2 * tp + fp + fn),
        },
        "per_class_accuracy": {
            str(k): {
                "total": agg["per_class_total"][k],
                "rule":  safe_div(agg["per_class_rule_ok"][k],
                                  agg["per_class_total"][k]),
                "ai":    safe_div(agg["per_class_ai_ok"][k],
                                  agg["per_class_total"][k]),
            }
            for k in sorted(agg["per_class_total"])
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="r2",
                    help="r1 | r2 | r3 | all | file:PATH")
    args = ap.parse_args()

    pieces = select_pieces(args.split)
    if not pieces:
        print(json.dumps({"error": "no pieces in split", "split": args.split}))
        sys.exit(1)

    agg: dict = {}
    skipped = []
    for pid in pieces:
        try:
            rule = load_pkl(FING_DIR / f"{pid:03d}.pkl")
            edit = load_pkl(EDIT_DIR / f"{pid:03d}.pkl")
        except Exception as e:
            skipped.append({"piece": pid, "reason": f"load: {e}"})
            continue
        ap_path = AI_DIR / f"{pid:03d}.pkl"
        ai = load_pkl(ap_path) if ap_path.exists() else None
        merge(agg, evaluate_piece(rule, edit, ai))

    out = {
        "split": args.split,
        "pieces_evaluated": len(pieces) - len(skipped),
        "pieces_skipped": skipped,
        "summary": summarise(agg),
    }
    print(json.dumps(out, indent=2, default=int))


if __name__ == "__main__":
    main()
