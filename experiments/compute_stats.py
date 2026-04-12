#!/usr/bin/env python3
"""
Aggregate dataset/HITL statistics for the KSMI paper.

Outputs a single JSON to stdout with:
  - piece counts at each pipeline stage
  - R1/R2/R3 completion counts
  - total notes (rule, edited, ai)
  - per-note "model correction rate" on R1-checked pieces
    (= fraction of notes where rule != edited)
  - AI acceptance proxy: fraction of AI-suggested notes that match
    the human-edited value on R1-checked pieces

Run from repo root:
    python experiments/compute_stats.py > experiments/stats.json
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

FING_DIR = DATA / "fingering"
EDIT_DIR = DATA / "fingering_edited"
AI_DIR = DATA / "fingering_edited_ai"
AI_V2_DIR = DATA / "fingering_edited_ai_v2"
STATUS_DIR = DATA / "metadata" / "annotation" / "status"


def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def piece_ids(d: Path) -> set[int]:
    if not d.exists():
        return set()
    return {int(p.stem) for p in d.glob("*.pkl") if p.stem.isdigit()}


def review_state(status_path: Path) -> dict:
    if not status_path.exists():
        return {"r1": False, "r2": False, "r3": False}
    j = json.loads(status_path.read_text())
    return {
        "r1": bool(j.get("review1")),
        "r2": bool(j.get("review2")),
        "r3": bool(j.get("review3")),
    }


def count_notes(frames) -> int:
    """frames is a list-of-lists; each inner list is fingering entries for that frame."""
    if frames is None:
        return 0
    n = 0
    for f in frames:
        if f:
            n += len(f)
    return n


def per_frame_match(rule, edited) -> tuple[int, int]:
    """
    Return (matched_notes, total_edited_notes).
    Compare element-wise per frame; if a frame is missing in rule, count as mismatch.
    """
    if not edited:
        return 0, 0
    matched = 0
    total = 0
    for i, ef in enumerate(edited):
        if not ef:
            continue
        rf = rule[i] if rule and i < len(rule) and rule[i] else []
        rmap = {(e.get("key_index"), e.get("hand")): e.get("finger")
                for e in rf if "key_index" in e}
        for e in ef:
            if "key_index" not in e:
                continue  # ambiguous-only entries: skip from comparison
            total += 1
            key = (e.get("key_index"), e.get("hand"))
            if rmap.get(key) == e.get("finger"):
                matched += 1
    return matched, total


def main() -> None:
    rule_ids = piece_ids(FING_DIR)
    edit_ids = piece_ids(EDIT_DIR)
    ai_ids = piece_ids(AI_DIR)
    ai_v2_ids = piece_ids(AI_V2_DIR)

    status_ids = sorted(rule_ids | edit_ids)

    r1, r2, r3 = 0, 0, 0
    r1_pieces = []
    for pid in status_ids:
        s = review_state(STATUS_DIR / f"{pid:03d}.json")
        if s["r1"]:
            r1 += 1
            r1_pieces.append(pid)
        if s["r2"]:
            r2 += 1
        if s["r3"]:
            r3 += 1

    # Note totals
    rule_notes = 0
    edit_notes = 0
    for pid in sorted(rule_ids):
        try:
            rule_notes += count_notes(load_pkl(FING_DIR / f"{pid:03d}.pkl"))
        except Exception as e:
            print(f"warn: rule {pid}: {e}", file=sys.stderr)
    for pid in sorted(edit_ids):
        try:
            edit_notes += count_notes(load_pkl(EDIT_DIR / f"{pid:03d}.pkl"))
        except Exception as e:
            print(f"warn: edit {pid}: {e}", file=sys.stderr)

    # Rule-vs-Edited agreement on R1-checked pieces
    # (= 1 - human_correction_rate)
    rule_edit_match = 0
    rule_edit_total = 0
    for pid in r1_pieces:
        try:
            rule = load_pkl(FING_DIR / f"{pid:03d}.pkl")
            edit = load_pkl(EDIT_DIR / f"{pid:03d}.pkl")
        except Exception:
            continue
        m, t = per_frame_match(rule, edit)
        rule_edit_match += m
        rule_edit_total += t

    # AI-vs-Edited agreement on R1-checked pieces (acceptance proxy)
    ai_edit_match = 0
    ai_edit_total = 0
    ai_dir_used = AI_V2_DIR if ai_v2_ids else AI_DIR
    for pid in r1_pieces:
        ap = ai_dir_used / f"{pid:03d}.pkl"
        if not ap.exists():
            continue
        try:
            ai = load_pkl(ap)
            edit = load_pkl(EDIT_DIR / f"{pid:03d}.pkl")
        except Exception:
            continue
        m, t = per_frame_match(ai, edit)
        ai_edit_match += m
        ai_edit_total += t

    out = {
        "pieces": {
            "rule": len(rule_ids),
            "edited": len(edit_ids),
            "ai_v1": len(ai_ids),
            "ai_v2": len(ai_v2_ids),
            "status_files": len(list(STATUS_DIR.glob("*.json"))),
        },
        "review_completion": {"r1": r1, "r2": r2, "r3": r3},
        "notes": {
            "rule_total": rule_notes,
            "edited_total": edit_notes,
        },
        "rule_vs_edited_on_r1": {
            "compared_notes": rule_edit_total,
            "matched": rule_edit_match,
            "agreement": (rule_edit_match / rule_edit_total) if rule_edit_total else None,
            "human_correction_rate": (1 - rule_edit_match / rule_edit_total) if rule_edit_total else None,
        },
        "ai_vs_edited_on_r1": {
            "ai_dir": str(ai_dir_used.relative_to(REPO)),
            "compared_notes": ai_edit_total,
            "matched": ai_edit_match,
            "agreement": (ai_edit_match / ai_edit_total) if ai_edit_total else None,
        },
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
