#!/usr/bin/env python3
"""
Build a random 91/62 train/holdout split over the 153 R1-checked
pieces, distinct from the natural R2-vs-not-R2 split, and write
both halves to JSON files for holdout_train.py / holdout_infer.py.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATUS = REPO / "data" / "metadata" / "annotation" / "status"


def main() -> None:
    ids = []
    for sp in sorted(STATUS.glob("*.json")):
        j = json.loads(sp.read_text())
        if j.get("review1"):
            ids.append(int(sp.stem))
    rng = random.Random(9999)
    rng.shuffle(ids)
    holdout = sorted(ids[:62])
    train = sorted(ids[62:])
    print(f"train={len(train)} holdout={len(holdout)}")
    out_dir = REPO / "experiments" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "random_train.json").write_text(
        json.dumps({"train": train}, indent=2))
    (out_dir / "random_holdout.json").write_text(
        json.dumps({"holdout": holdout}, indent=2))
    print(f"wrote {out_dir}/random_train.json and random_holdout.json")


if __name__ == "__main__":
    main()
