#!/usr/bin/env python3
"""
Comprehensive analysis for the PiAnnotate paper.

Computes, on a chosen split:
  (A) R2 reversal decomposition by AI behavior on each note
  (B) Triage utility: review-load reduction and caught-error rate
  (C) Threshold sweep using correction_prob (no retraining)
  (D) Missing-fill vs overwrite breakdown
  (E) Stale check: AI output mtime vs review timestamps

Run from repo root:
    python experiments/analyze.py --split r2 \
        > experiments/analyze_r2.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
FING_DIR = DATA / "fingering"
EDIT_DIR = DATA / "fingering_edited"
AI_DIR = DATA / "fingering_edited_ai_v2"
STATUS_DIR = DATA / "metadata" / "annotation" / "status"

# CLI override of AI dir for held-out evaluation
import os
if os.environ.get("PIANNOTATE_AI_DIR"):
    AI_DIR = Path(os.environ["PIANNOTATE_AI_DIR"])


def load_pkl(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


def select_pieces(split: str) -> list[int]:
    ids = sorted(int(p.stem) for p in FING_DIR.glob("*.pkl") if p.stem.isdigit())
    if split == "all":
        return [i for i in ids if (STATUS_DIR / f"{i:03d}.json").exists()]
    if split.startswith("file:"):
        spec = json.loads(Path(split[5:]).read_text())
        # accept either {"holdout": [...]} or {"train": [...]} or list
        if isinstance(spec, list):
            return sorted(spec)
        for key in ("holdout", "pieces", "train"):
            if key in spec:
                return sorted(spec[key])
        raise ValueError(f"unknown split spec keys: {list(spec.keys())}")
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


def index_frame(frame):
    return {(e["key_index"], e["hand"]): e for e in (frame or []) if "key_index" in e}


def safe_div(a, b):
    return (a / b) if b else None


def analyze_piece(rule, edited, ai, thresholds):
    """
    Returns aggregated counters for one piece.

    Per-note categories (over notes that exist in `edited`):
        rule_correct, rule_wrong   -- vs edited
        ai_correct,   ai_wrong     -- vs edited (None if no AI entry)

    AI behavior on errors and non-errors:
        when rule_wrong:
            ai_fixed         (ai == edited)
            ai_kept_rule     (ai == rule, missed)
            ai_other         (ai != rule and ai != edited, wrong correction)
            ai_missing       (no AI entry)
        when rule_correct:
            ai_kept_correct  (ai == rule == edited)
            ai_broke         (ai != edited)  -- false correction
            ai_missing_ok    (no AI entry)

    Triage (using AI was_corrected flag as the trigger):
        flagged_total
        flagged_was_rule_wrong   (true positive)
        flagged_was_rule_right   (false positive)

    Missing-fill vs overwrite:
        rule_had_entry: notes where rule had a prediction for this key
        rule_no_entry:  notes where rule did not (model fills)
        per category, ai_correct counts.

    Threshold sweep (over correction_prob):
        for each tau, build hypothetical "if was_corrected at tau" and
        compare to gt-correction (rule != edited).
    """
    n_frames = max(len(rule or []), len(edited or []), len(ai or []))
    A = {
        "n_notes": 0,
        "rule_correct": 0, "rule_wrong": 0,
        "ai_correct": 0, "ai_wrong": 0, "ai_no_entry": 0,
        # decomposition
        "wrong_ai_fixed":    0,
        "wrong_ai_kept":     0,
        "wrong_ai_other":    0,
        "wrong_ai_missing":  0,
        "right_ai_kept":     0,
        "right_ai_broke":    0,
        "right_ai_missing":  0,
        # triage (model's was_corrected)
        "flagged":           0,
        "flagged_tp":        0,
        "flagged_fp":        0,
        "unflagged":         0,
        "unflagged_fn":      0,
        "unflagged_tn":      0,
        # missing-fill vs overwrite
        "rh_total": 0, "rh_ai_correct": 0,   # rule had entry
        "rn_total": 0, "rn_ai_correct": 0,   # rule had no entry (model fills)
        # threshold sweep accumulators (per tau: tp, fp, fn, tn)
        "sweep": {f"{t:.2f}": [0, 0, 0, 0] for t in thresholds},
    }

    for i in range(n_frames):
        ef = edited[i] if i < len(edited or []) else None
        if not ef:
            continue
        rf = rule[i] if i < len(rule or []) else None
        af = ai[i] if i < len(ai or []) else None
        rmap = index_frame(rf)
        amap = index_frame(af)
        for e in ef:
            if "key_index" not in e:
                continue
            k = (e["key_index"], e["hand"])
            gt = e.get("finger")
            r_entry = rmap.get(k)
            a_entry = amap.get(k)
            r_pred = r_entry.get("finger") if r_entry else None
            a_pred = a_entry.get("finger") if a_entry else None
            rule_ok = (r_pred == gt)
            ai_ok = (a_pred == gt) if a_entry is not None else None

            A["n_notes"] += 1
            if rule_ok:
                A["rule_correct"] += 1
            else:
                A["rule_wrong"] += 1

            if a_entry is None:
                A["ai_no_entry"] += 1
            elif ai_ok:
                A["ai_correct"] += 1
            else:
                A["ai_wrong"] += 1

            # Behavior decomposition
            if not rule_ok:
                if a_entry is None:
                    A["wrong_ai_missing"] += 1
                elif a_pred == gt:
                    A["wrong_ai_fixed"] += 1
                elif a_pred == r_pred:
                    A["wrong_ai_kept"] += 1
                else:
                    A["wrong_ai_other"] += 1
            else:
                if a_entry is None:
                    A["right_ai_missing"] += 1
                elif a_pred == gt:
                    A["right_ai_kept"] += 1
                else:
                    A["right_ai_broke"] += 1

            # Triage using was_corrected
            if a_entry is not None and "was_corrected" in a_entry:
                was = bool(a_entry["was_corrected"])
                if was:
                    A["flagged"] += 1
                    if not rule_ok:
                        A["flagged_tp"] += 1
                    else:
                        A["flagged_fp"] += 1
                else:
                    A["unflagged"] += 1
                    if not rule_ok:
                        A["unflagged_fn"] += 1
                    else:
                        A["unflagged_tn"] += 1

            # Missing-fill vs overwrite (does the rule have an entry for this key?)
            if r_entry is not None:
                A["rh_total"] += 1
                if a_entry is not None and a_pred == gt:
                    A["rh_ai_correct"] += 1
            else:
                A["rn_total"] += 1
                if a_entry is not None and a_pred == gt:
                    A["rn_ai_correct"] += 1

            # Threshold sweep (correction_prob >= tau ⇒ "would correct")
            if a_entry is not None and "correction_prob" in a_entry:
                p = float(a_entry["correction_prob"])
                gt_needs = (not rule_ok)
                for t in thresholds:
                    key = f"{t:.2f}"
                    pred_corr = (p >= t)
                    arr = A["sweep"][key]
                    if pred_corr and gt_needs:
                        arr[0] += 1
                    elif pred_corr and not gt_needs:
                        arr[1] += 1
                    elif (not pred_corr) and gt_needs:
                        arr[2] += 1
                    else:
                        arr[3] += 1
    return A


def merge(into, other):
    for k, v in other.items():
        if isinstance(v, dict):
            into.setdefault(k, {})
            for kk, vv in v.items():
                if isinstance(vv, list):
                    if kk not in into[k]:
                        into[k][kk] = list(vv)
                    else:
                        into[k][kk] = [a + b for a, b in zip(into[k][kk], vv)]
                else:
                    into[k][kk] = into[k].get(kk, 0) + vv
        else:
            into[k] = into.get(k, 0) + v


def stale_check(pieces):
    """For each piece, compare AI file mtime vs review2/review3 timestamps."""
    out = {"pieces": 0, "ai_predates_r2": 0, "ai_predates_r3": 0, "details": []}
    for pid in pieces:
        sp = STATUS_DIR / f"{pid:03d}.json"
        ap = AI_DIR / f"{pid:03d}.pkl"
        if not (sp.exists() and ap.exists()):
            continue
        out["pieces"] += 1
        j = json.loads(sp.read_text())
        ai_mtime = datetime.fromtimestamp(ap.stat().st_mtime)
        r2 = j.get("review2", {}) or {}
        r3 = j.get("review3", {}) or {}
        r2_t = r2.get("completed_at")
        r3_t = r3.get("completed_at")
        ai_pre_r2 = bool(r2_t and ai_mtime < datetime.fromisoformat(r2_t))
        ai_pre_r3 = bool(r3_t and ai_mtime < datetime.fromisoformat(r3_t))
        if ai_pre_r2:
            out["ai_predates_r2"] += 1
        if ai_pre_r3:
            out["ai_predates_r3"] += 1
    return out


def summarise(A, thresholds):
    n = A["n_notes"]
    rule_acc = safe_div(A["rule_correct"], n)
    ai_acc = safe_div(A["ai_correct"], n)
    wrong = A["rule_wrong"]
    right = A["rule_correct"]

    # R2 reversal decomposition
    decomp = {
        "on_rule_wrong_notes": {
            "total":   wrong,
            "fixed":   A["wrong_ai_fixed"],
            "kept_wrong": A["wrong_ai_kept"],
            "other_wrong": A["wrong_ai_other"],
            "no_ai":   A["wrong_ai_missing"],
            "fix_rate": safe_div(A["wrong_ai_fixed"], wrong),
        },
        "on_rule_right_notes": {
            "total":   right,
            "kept_right": A["right_ai_kept"],
            "broken":  A["right_ai_broke"],
            "no_ai":   A["right_ai_missing"],
            "break_rate": safe_div(A["right_ai_broke"], right),
        },
    }

    # Triage utility
    flagged = A["flagged"]
    unflagged = A["unflagged"]
    triage = {
        "flagged_notes": flagged,
        "unflagged_notes": unflagged,
        "review_load_reduction": safe_div(unflagged, flagged + unflagged),
        "flagged_precision": safe_div(A["flagged_tp"], flagged),  # of flagged, % truly wrong
        "rule_error_recall": safe_div(A["flagged_tp"],
                                      A["flagged_tp"] + A["unflagged_fn"]),
        "missed_errors": A["unflagged_fn"],
        "caught_errors": A["flagged_tp"],
    }

    # Missing-fill vs overwrite
    breakdown = {
        "rule_had_entry": {
            "notes": A["rh_total"],
            "ai_accuracy": safe_div(A["rh_ai_correct"], A["rh_total"]),
        },
        "rule_no_entry_filled_by_model": {
            "notes": A["rn_total"],
            "ai_accuracy": safe_div(A["rn_ai_correct"], A["rn_total"]),
        },
    }

    # Threshold sweep
    sweep = []
    for t in thresholds:
        tp, fp, fn, tn = A["sweep"][f"{t:.2f}"]
        sweep.append({
            "tau": t,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": safe_div(tp, tp + fp),
            "recall":    safe_div(tp, tp + fn),
            "f1":        safe_div(2 * tp, 2 * tp + fp + fn),
            "flagged_fraction": safe_div(tp + fp, tp + fp + fn + tn),
        })

    return {
        "n_notes": n,
        "rule_agreement": rule_acc,
        "ai_agreement":   ai_acc,
        "absolute_delta": (ai_acc - rule_acc) if (ai_acc and rule_acc) else None,
        "r2_reversal_decomposition": decomp,
        "triage": triage,
        "missing_fill_vs_overwrite": breakdown,
        "threshold_sweep": sweep,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="r2",
                    help="r1 | r2 | r3 | all")
    args = ap.parse_args()

    pieces = select_pieces(args.split)
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50,
                  0.60, 0.70, 0.80, 0.90, 0.95]

    agg = {}
    for pid in pieces:
        try:
            rule = load_pkl(FING_DIR / f"{pid:03d}.pkl")
            edit = load_pkl(EDIT_DIR / f"{pid:03d}.pkl")
        except Exception:
            continue
        ap_path = AI_DIR / f"{pid:03d}.pkl"
        ai = load_pkl(ap_path) if ap_path.exists() else None
        merge(agg, analyze_piece(rule, edit, ai, thresholds))

    out = {
        "split": args.split,
        "pieces_evaluated": len(pieces),
        "summary": summarise(agg, thresholds),
        "stale_check": stale_check(pieces),
    }
    print(json.dumps(out, indent=2, default=int))


if __name__ == "__main__":
    main()
