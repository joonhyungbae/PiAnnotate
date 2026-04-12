#!/usr/bin/env python3
"""
Aggregate seed-variance held-out runs and compute mean ± std plus
a bootstrap CI on the absolute (probe - rule) margin.

Reads experiments/seed_runs/holdout_seed{0..4}.json and the
no-rule-embed baseline holdout_norule.json. Each file is the JSON
output of analyze.py --split r2 against a different AI directory.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED_DIR = ROOT / "seed_runs"


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def paired_cluster_bootstrap(per_piece: list[dict],
                              n_boot: int = 5000,
                              seed: int = 1234) -> tuple[float, float, float]:
    """
    Paired piece-level cluster bootstrap on the per-note margin
    (probe_correct - rule_correct).

    Resamples whole pieces with replacement (cluster-level), then
    computes the per-note margin on each resample as
        sum(probe_correct - rule_correct) / sum(n_notes)
    aggregated across the resampled pieces. This handles
    within-piece correlation and is paired (rule and probe see the
    same notes).

    Returns (point_estimate, lo_95, hi_95) on the margin in [0,1].
    """
    import random
    rng = random.Random(seed)
    n = len(per_piece)
    if n == 0:
        return 0.0, 0.0, 0.0
    diffs = [p["probe_correct"] - p["rule_correct"] for p in per_piece]
    counts = [p["n_notes"] for p in per_piece]
    point = sum(diffs) / sum(counts)

    samples = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        d = sum(diffs[i] for i in idxs)
        c = sum(counts[i] for i in idxs)
        samples.append(d / c if c > 0 else 0.0)
    samples.sort()
    lo = samples[int(0.025 * n_boot)]
    hi = samples[int(0.975 * n_boot)]
    return point, lo, hi


def load_per_piece(tag: str) -> list[dict] | None:
    p = SEED_DIR / f"per_piece{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("pieces", [])


def main() -> None:
    seeds = []
    per_piece_seeds = []
    for s in range(5):
        d = load(SEED_DIR / f"holdout_seed{s}.json")
        pp = load_per_piece(f"_seed{s}")
        if d:
            seeds.append((s, d))
            if pp is not None:
                per_piece_seeds.append((s, pp))

    norule = load(SEED_DIR / "holdout_norule.json")
    norule_seeds = []
    norule_per_piece_seeds = []
    for s in range(5):
        d = load(SEED_DIR / f"holdout_norule_seed{s}.json")
        pp = load_per_piece(f"_norule_seed{s}")
        if d:
            norule_seeds.append((s, d))
            if pp is not None:
                norule_per_piece_seeds.append((s, pp))

    random_seeds = []
    random_per_piece_seeds = []
    for s in range(5):
        d = load(SEED_DIR / f"holdout_random_seed{s}.json")
        pp = load_per_piece(f"_random_seed{s}")
        if d:
            random_seeds.append((s, d))
            if pp is not None:
                random_per_piece_seeds.append((s, pp))

    tiny_seeds = []
    tiny_per_piece_seeds = []
    for s in range(5):
        d = load(SEED_DIR / f"holdout_tiny_seed{s}.json")
        pp = load_per_piece(f"_tiny_seed{s}")
        if d:
            tiny_seeds.append((s, d))
            if pp is not None:
                tiny_per_piece_seeds.append((s, pp))

    print(f"Loaded {len(seeds)}/5 seeds; norule={'yes' if norule else 'no'}")
    print()

    rule_acc = None
    probe_accs = []
    fix_rates = []
    flag_precs = []
    flag_recs = []
    flag_rates = []
    break_rates = []
    n_notes = None
    rule_correct = None
    probe_correct_per_seed = []

    for s, d in seeds:
        sm = d["summary"]
        if rule_acc is None:
            rule_acc = sm["rule_agreement"]
            n_notes = sm["n_notes"]
            # rule_correct = round(rule_acc * n_notes)
        probe_accs.append(sm["ai_agreement"])
        rd = sm["r2_reversal_decomposition"]["on_rule_wrong_notes"]
        fix_rates.append(rd["fix_rate"])
        rr = sm["r2_reversal_decomposition"]["on_rule_right_notes"]
        break_rates.append(rr["break_rate"])
        t = sm["triage"]
        flag_precs.append(t["flagged_precision"])
        flag_recs.append(t["rule_error_recall"])
        flag_rates.append(1 - t["review_load_reduction"])
        probe_correct_per_seed.append(round(sm["ai_agreement"] * n_notes))

    def fmt(xs):
        if not xs:
            return "n/a"
        m = statistics.mean(xs)
        sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
        return f"{m*100:.2f} ± {sd*100:.2f}"

    print("=" * 60)
    print(f"R2 HELD-OUT  ({len(seeds)} seeds, {n_notes:,} notes)")
    print("=" * 60)
    print(f"  rule baseline   : {rule_acc*100:.2f}%")
    print(f"  probe agreement : {fmt(probe_accs)}%")
    margins = [p - rule_acc for p in probe_accs]
    print(f"  Δ vs rule       : "
          f"{statistics.mean(margins)*100:+.2f}pp "
          f"(min {min(margins)*100:+.2f}, max {max(margins)*100:+.2f})")
    print(f"  fix rate (rule-wrong) : {fmt(fix_rates)}%")
    print(f"  break rate (rule-right): {fmt(break_rates)}%")
    print(f"  triage flag rate: {fmt(flag_rates)}%")
    print(f"  triage P / R    : {fmt(flag_precs)} / {fmt(flag_recs)}%")

    # Paired piece-level cluster bootstrap on the margin, per seed
    if per_piece_seeds:
        print("  paired piece-level cluster bootstrap "
              "(95% CI on Δ, by seed):")
        seed_cis = []
        for s, pp in per_piece_seeds:
            point, lo, hi = paired_cluster_bootstrap(pp)
            print(f"    seed {s}: Δ={point*100:+.2f}pp "
                  f"[{lo*100:+.2f}, {hi*100:+.2f}]")
            seed_cis.append({"seed": s, "delta_pp": point * 100,
                             "lo_pp": lo * 100, "hi_pp": hi * 100})

        # Pool across seeds: compute the across-seed mean of point
        # estimates and a seed-level interval (mean ± 1.96·SE_seed).
        if len(seed_cis) > 1:
            pts = [s["delta_pp"] for s in seed_cis]
            m = statistics.mean(pts)
            sd = statistics.stdev(pts)
            se = sd / math.sqrt(len(pts))
            # Student t critical for n-1 df, two-sided 95%
            # n=5 -> df=4 -> t* = 2.776 (table value)
            t_crit = {2: 12.706, 3: 4.303, 4: 2.776,
                      5: 2.571, 6: 2.447}.get(len(pts) - 1, 1.96)
            print(f"    across seeds: Δ={m:+.2f}pp "
                  f"[{m-t_crit*se:+.2f}, {m+t_crit*se:+.2f}] "
                  f"(seed-level 95% Student t, df={len(pts)-1})")

    if norule_seeds:
        print()
        print("=" * 60)
        print(f"FROM-SCRATCH BASELINE  ({len(norule_seeds)} seeds, "
              f"no rule-label embedding)")
        print("=" * 60)
        nr_probe = []
        nr_fix = []
        nr_break = []
        nr_p = []
        nr_r = []
        for _, d in norule_seeds:
            sm = d["summary"]
            nr_probe.append(sm["ai_agreement"])
            rd = sm["r2_reversal_decomposition"]["on_rule_wrong_notes"]
            rr = sm["r2_reversal_decomposition"]["on_rule_right_notes"]
            nr_fix.append(rd["fix_rate"] or 0)
            nr_break.append(rr["break_rate"] or 0)
            nr_p.append(sm["triage"]["flagged_precision"])
            nr_r.append(sm["triage"]["rule_error_recall"])
        print(f"  rule baseline   : {rule_acc*100:.2f}%")
        print(f"  probe agreement : {fmt(nr_probe)}%")
        nr_margins = [p - rule_acc for p in nr_probe]
        print(f"  Δ vs rule       : "
              f"{statistics.mean(nr_margins)*100:+.2f}pp "
              f"(min {min(nr_margins)*100:+.2f}, "
              f"max {max(nr_margins)*100:+.2f})")
        print(f"  fix rate        : {fmt(nr_fix)}%")
        print(f"  break rate      : {fmt(nr_break)}%")
        print(f"  triage P / R    : {fmt(nr_p)} / {fmt(nr_r)}%")
        if norule_per_piece_seeds:
            print("  paired cluster bootstrap (no-rule-embed), by seed:")
            for s, pp in norule_per_piece_seeds:
                point, lo, hi = paired_cluster_bootstrap(pp)
                print(f"    seed {s}: Δ={point*100:+.2f}pp "
                      f"[{lo*100:+.2f}, {hi*100:+.2f}]")
            if len(norule_per_piece_seeds) > 1:
                pts = []
                for _, pp in norule_per_piece_seeds:
                    pt, _, _ = paired_cluster_bootstrap(pp)
                    pts.append(pt * 100)
                m = statistics.mean(pts)
                sd = statistics.stdev(pts)
                se = sd / math.sqrt(len(pts))
                t_crit = {2: 12.706, 3: 4.303, 4: 2.776,
                          5: 2.571, 6: 2.447}.get(len(pts) - 1, 1.96)
                print(f"    across seeds: Δ={m:+.2f}pp "
                      f"[{m-t_crit*se:+.2f}, {m+t_crit*se:+.2f}] "
                      f"(seed-level 95% Student t, df={len(pts)-1})")
    elif norule:
        # fallback: single-seed report
        sm = norule["summary"]
        print()
        print("=" * 60)
        print("FROM-SCRATCH BASELINE  (single seed, no rule-label embedding)")
        print("=" * 60)
        print(f"  rule baseline   : {sm['rule_agreement']*100:.2f}%")
        print(f"  probe agreement : {sm['ai_agreement']*100:.2f}%")
        print(f"  Δ vs rule       : "
              f"{(sm['ai_agreement']-sm['rule_agreement'])*100:+.2f}pp")

    def report_block(name: str, seeds_list, pp_list, rule_acc):
        if not seeds_list:
            return None
        print()
        print("=" * 60)
        print(f"{name}  ({len(seeds_list)} seeds)")
        print("=" * 60)
        probe_accs2 = []
        fix2 = []
        brk2 = []
        p2 = []
        r2 = []
        for _, d in seeds_list:
            sm = d["summary"]
            probe_accs2.append(sm["ai_agreement"])
            rd = sm["r2_reversal_decomposition"]["on_rule_wrong_notes"]
            rr = sm["r2_reversal_decomposition"]["on_rule_right_notes"]
            fix2.append(rd["fix_rate"] or 0)
            brk2.append(rr["break_rate"] or 0)
            p2.append(sm["triage"]["flagged_precision"] or 0)
            r2.append(sm["triage"]["rule_error_recall"] or 0)
        # rule baseline (could differ for random split)
        any_rule = seeds_list[0][1]["summary"]["rule_agreement"]
        print(f"  rule baseline   : {any_rule*100:.2f}%")
        print(f"  probe agreement : {fmt(probe_accs2)}%")
        margins2 = [p - any_rule for p in probe_accs2]
        print(f"  Δ vs rule       : "
              f"{statistics.mean(margins2)*100:+.2f}pp "
              f"(min {min(margins2)*100:+.2f}, "
              f"max {max(margins2)*100:+.2f})")
        print(f"  fix rate        : {fmt(fix2)}%")
        print(f"  break rate      : {fmt(brk2)}%")
        print(f"  triage P / R    : {fmt(p2)} / {fmt(r2)}%")
        if pp_list:
            for s, pp in pp_list:
                point, lo, hi = paired_cluster_bootstrap(pp)
                print(f"    seed {s}: Δ={point*100:+.2f}pp "
                      f"[{lo*100:+.2f}, {hi*100:+.2f}]")
            if len(pp_list) > 1:
                pts = []
                for _, pp in pp_list:
                    pt, _, _ = paired_cluster_bootstrap(pp)
                    pts.append(pt * 100)
                m = statistics.mean(pts)
                sd = statistics.stdev(pts)
                se = sd / math.sqrt(len(pts))
                t_crit = {2: 12.706, 3: 4.303, 4: 2.776,
                          5: 2.571, 6: 2.447}.get(len(pts) - 1, 1.96)
                print(f"    pooled (t df={len(pts)-1}): "
                      f"Δ={m:+.2f}pp [{m-t_crit*se:+.2f}, "
                      f"{m+t_crit*se:+.2f}]")

    report_block("RANDOM 91/62 SPLIT (no-rule-embed)",
                 random_seeds, random_per_piece_seeds, rule_acc)
    report_block("TINY BASELINE (1-layer, d=64, no-rule-embed)",
                 tiny_seeds, tiny_per_piece_seeds, rule_acc)

    # Save aggregate JSON
    out = {
        "n_seeds": len(seeds),
        "n_notes": n_notes,
        "rule_agreement": rule_acc,
        "probe_agreement_mean": (
            statistics.mean(probe_accs) if probe_accs else None),
        "probe_agreement_std": (
            statistics.stdev(probe_accs)
            if len(probe_accs) > 1 else 0.0),
        "margin_mean_pp": (
            statistics.mean(margins) * 100 if probe_accs else None),
        "margin_std_pp": (
            statistics.stdev(margins) * 100
            if len(probe_accs) > 1 else 0.0),
        "per_seed": [
            {"seed": s, "probe": d["summary"]["ai_agreement"]}
            for s, d in seeds
        ],
        "from_scratch": ({
            "probe": norule["summary"]["ai_agreement"],
            "margin_pp": (norule["summary"]["ai_agreement"]
                          - norule["summary"]["rule_agreement"]) * 100,
        } if norule else None),
    }
    (ROOT / "seed_aggregate.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
