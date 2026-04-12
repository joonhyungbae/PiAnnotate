#!/usr/bin/env python3
"""
Per-piece per-note correctness counts for paired bootstrap.

For a given AI directory and a given split (default r2), output a
JSON with one record per piece:
  {piece_id, n_notes, rule_correct, probe_correct}

Both correctness counts are computed against the current human-edited
labels using the same matching as analyze.py (key by (key_index, hand)).

Usage from repo root:
    python experiments/per_piece_eval.py \
        --ai-dir data/fingering_edited_ai_holdout_seed0 \
        --split r2 \
        > experiments/seed_runs/per_piece_seed0.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
FING = DATA / "fingering"
EDIT = DATA / "fingering_edited"
STATUS = DATA / "metadata" / "annotation" / "status"


def load(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


def select_pieces(split: str) -> list[int]:
    ids = sorted(int(p.stem) for p in FING.glob("*.pkl") if p.stem.isdigit())
    if split == "all":
        return [i for i in ids if (STATUS / f"{i:03d}.json").exists()]
    if split.startswith("file:"):
        spec = json.loads(Path(split[5:]).read_text())
        if isinstance(spec, list):
            return sorted(spec)
        for key in ("holdout", "pieces", "train"):
            if key in spec:
                return sorted(spec[key])
        raise ValueError(f"unknown split spec keys: {list(spec.keys())}")
    field = {"r1": "review1", "r2": "review2", "r3": "review3"}[split]
    out = []
    for i in ids:
        sp = STATUS / f"{i:03d}.json"
        if not sp.exists():
            continue
        j = json.loads(sp.read_text())
        if j.get(field):
            out.append(i)
    return out


def index_frame(frame):
    return {(e["key_index"], e["hand"]): e for e in (frame or [])
            if "key_index" in e}


def evaluate_piece(rule, edited, ai) -> tuple[int, int, int]:
    n = max(len(rule or []), len(edited or []), len(ai or []))
    n_notes = rule_ok = probe_ok = 0
    for i in range(n):
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
            n_notes += 1
            r_entry = rmap.get(k)
            a_entry = amap.get(k)
            if r_entry and r_entry.get("finger") == gt:
                rule_ok += 1
            if a_entry and a_entry.get("finger") == gt:
                probe_ok += 1
    return n_notes, rule_ok, probe_ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai-dir", required=True, type=str)
    ap.add_argument("--split", default="r2")
    args = ap.parse_args()

    ai_dir = Path(args.ai_dir)
    if not ai_dir.is_absolute():
        ai_dir = REPO / ai_dir

    pieces = select_pieces(args.split)
    out = {"ai_dir": str(ai_dir), "split": args.split, "pieces": []}
    for pid in pieces:
        try:
            rule = load(FING / f"{pid:03d}.pkl")
            edit = load(EDIT / f"{pid:03d}.pkl")
        except Exception:
            continue
        ap_path = ai_dir / f"{pid:03d}.pkl"
        ai = load(ap_path) if ap_path.exists() else None
        n, r_ok, p_ok = evaluate_piece(rule, edit, ai)
        if n == 0:
            continue
        out["pieces"].append({
            "piece": pid,
            "n_notes": n,
            "rule_correct": r_ok,
            "probe_correct": p_ok,
        })
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
