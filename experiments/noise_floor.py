#!/usr/bin/env python3
"""
Noise-floor and multi-pass coverage statistics for the corpus.

We do not have R1-state snapshots of the edited labels, so a
classical inter-annotator agreement is not computable. Instead
we report two surrogates the reviewer can use to bound label
heterogeneity:

  1. Per-piece distribution of rule-vs-edited disagreement rates
     (median, IQR, min, max). A tight distribution means the
     8.18% corpus-level rule-error rate is representative; a
     wide one means it averages over very different pieces.

  2. Multi-pass coverage: how many pieces went through
     {R1 only, R1+R2, R1+R3, R1+R2+R3}. The expectation is that
     almost every piece has been touched at least twice, which
     places a soft floor on label quality.

Run from repo root:
    python experiments/noise_floor.py > experiments/noise_floor.json
"""

from __future__ import annotations

import json
import pickle
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FING = REPO / "data" / "fingering"
EDIT = REPO / "data" / "fingering_edited"
STATUS = REPO / "data" / "metadata" / "annotation" / "status"


def load(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


def per_piece_disagreement(pid: int) -> tuple[int, int] | None:
    rp = FING / f"{pid:03d}.pkl"
    ep = EDIT / f"{pid:03d}.pkl"
    if not (rp.exists() and ep.exists()):
        return None
    rule = load(rp)
    edit = load(ep)
    matched = total = 0
    for i, ef in enumerate(edit):
        if not ef:
            continue
        rf = rule[i] if i < len(rule) else None
        rmap = {(e.get("key_index"), e.get("hand")): e.get("finger")
                for e in (rf or []) if "key_index" in e}
        for e in ef:
            if "key_index" not in e:
                continue
            total += 1
            if rmap.get((e.get("key_index"), e.get("hand"))) == e.get("finger"):
                matched += 1
    return matched, total


def main() -> None:
    ids = sorted(int(p.stem) for p in FING.glob("*.pkl") if p.stem.isdigit())
    rates = []
    coverage = {"r1_only": 0, "r1_r2": 0, "r1_r3": 0, "r1_r2_r3": 0}
    for pid in ids:
        sp = STATUS / f"{pid:03d}.json"
        if not sp.exists():
            continue
        j = json.loads(sp.read_text())
        if not j.get("review1"):
            continue
        r2 = bool(j.get("review2"))
        r3 = bool(j.get("review3"))
        if r2 and r3:
            coverage["r1_r2_r3"] += 1
        elif r2:
            coverage["r1_r2"] += 1
        elif r3:
            coverage["r1_r3"] += 1
        else:
            coverage["r1_only"] += 1

        m = per_piece_disagreement(pid)
        if m is None or m[1] == 0:
            continue
        matched, total = m
        rates.append(1 - matched / total)

    rates.sort()
    n = len(rates)
    out = {
        "pieces_with_disagreement": n,
        "disagreement_rate_distribution": {
            "min":    rates[0],
            "p25":    rates[n // 4],
            "median": rates[n // 2],
            "p75":    rates[3 * n // 4],
            "max":    rates[-1],
            "mean":   statistics.mean(rates),
            "stdev":  statistics.stdev(rates) if n > 1 else 0.0,
        },
        "multi_pass_coverage": coverage,
        "pieces_with_at_least_two_passes":
            coverage["r1_r2"] + coverage["r1_r3"] + coverage["r1_r2_r3"],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
