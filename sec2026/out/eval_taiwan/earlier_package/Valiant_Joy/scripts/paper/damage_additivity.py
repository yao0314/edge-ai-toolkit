"""Experiment 12: is pruning damage additive across layers?

This replaces experiment 11, which died productively. The label-free probe ranks SINGLE-LAYER
damage well (Spearman +0.688 on classification, +0.703 on detection, both n=16), yet DP-knapsack
allocation built on exactly that signal loses to uniform pruning (-2.72 vs -2.40 pt at ratio 0.25).

If the signal is good and the solver is exact on its discretised problem, the failure has to be in
the OBJECTIVE. Every evaluate-each-layer-then-knapsack method minimises a SUM of per-layer damages,
which silently assumes damage composes additively. This measures whether it does.

    python scripts/paper/damage_additivity.py --config configs/experiments/classification_prune_alloc.yaml

Writes results/paper/additivity.csv (per-layer rows) and prints the plan-level comparison.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from core.baseline import get_baseline  # noqa: E402
from core.config import build_adapter, load_config, set_seed, setup_env  # noqa: E402
from core.prune import allocate_ratios, prune_model, recalibrate_bn  # noqa: E402


def measure(adapter, base, x, ignored, pcfg, ratio_per_layer, bn_batches):
    """Damage of applying exactly this ratio vector, with BN re-estimated. BN re-calibration must be
    identical in the single-layer and whole-plan arms or the stale-statistics effect (measured
    separately at 0.1000 -> 0.6522 for ratio 0.25) dominates and masks the additivity question."""
    m = prune_model(base, x, 0.0, ignored, importance=pcfg.get("importance", "l2"),
                    round_to=pcfg.get("round_to", 8), ratio_per_layer=dict(ratio_per_layer),
                    report={})
    recalibrate_bn(m, adapter.bn_calibration_loader(), adapter.device, bn_batches)
    return float(adapter.evaluate(m)["score"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/classification_prune_alloc.yaml")
    ap.add_argument("--target", type=float, default=0.25)
    ap.add_argument("--bn-batches", type=int, default=32)
    ap.add_argument("--out", default="results/paper/additivity.csv")
    a = ap.parse_args()

    setup_env()
    cfg = load_config(a.config)
    set_seed(cfg.get("seed", 0))
    adapter = build_adapter(cfg, None)
    base = get_baseline(adapter, None, model_cfg=cfg.get("model", {}))
    ev = adapter.evaluate(base)
    metric, base_score = ev["metric_name"], float(ev["score"])
    x, ignored = adapter.example_input(), adapter.ignored_layers(base)
    pcfg = dict(cfg.get("prune") or {})
    print(f"baseline {metric} {base_score:.4f}\n")

    rep = {}
    plan = allocate_ratios(adapter, base, a.target, pcfg, rep)
    plan = {k: v for k, v in plan.items() if v and v > 0}
    if not plan:
        print("the allocator produced an empty plan; nothing to test", file=sys.stderr)
        return 1
    print(f"DP plan: {len(plan)} layer(s) with a non-zero ratio "
          f"(probe took {rep.get('sensitivity_s')}s)\n")

    rows = []
    for i, (layer, r) in enumerate(sorted(plan.items()), 1):
        t0 = time.time()
        s = measure(adapter, base, x, ignored, pcfg, {layer: r}, a.bn_batches)
        rows.append({"plan": "dp", "layer": layer, "ratio": r, "alone_score": s,
                     "alone_damage_pt": (base_score - s) * 100, "seconds": round(time.time() - t0, 1)})
        print(f"  [{i:2d}/{len(plan)}] {layer:34s} r={r:.3f}  alone {s:.4f} "
              f"(-{rows[-1]['alone_damage_pt']:.2f} pt)  {rows[-1]['seconds']:.0f}s", flush=True)

    whole = measure(adapter, base, x, ignored, pcfg, plan, a.bn_batches)
    pred = sum(r["alone_damage_pt"] for r in rows)
    meas = (base_score - whole) * 100

    # Control: uniform is also a plan, over the same layers, at the target ratio. If additivity
    # fails equally for both, non-additivity cannot be what makes the ALLOCATOR specifically lose.
    uni = {k: a.target for k in plan}
    urows = []
    for layer in sorted(uni):
        s = measure(adapter, base, x, ignored, pcfg, {layer: a.target}, a.bn_batches)
        urows.append({"plan": "uniform", "layer": layer, "ratio": a.target, "alone_score": s,
                      "alone_damage_pt": (base_score - s) * 100, "seconds": None})
    uwhole = measure(adapter, base, x, ignored, pcfg, uni, a.bn_batches)
    upred = sum(r["alone_damage_pt"] for r in urows)
    umeas = (base_score - uwhole) * 100

    pd.DataFrame(rows + urows).to_csv(ROOT / a.out, index=False)

    print(f"\n{'plan':10s} {'sum of per-layer':>17s} {'measured together':>18s} {'gap':>9s} {'ratio':>7s}")
    print("-" * 66)
    for name, p, m in (("dp", pred, meas), ("uniform", upred, umeas)):
        print(f"{name:10s} {p:16.2f} pt {m:17.2f} pt {m - p:+8.2f} {m / p if p else float('nan'):7.2f}")
    print()
    print("gap > 0 (super-additive): layers interact to do MORE harm together than separately.")
    print("gap < 0 (sub-additive):   the network absorbs joint pruning better than the sum implies.")
    print("A knapsack minimising the SUM is optimising the wrong objective either way; the sign says")
    print("in which direction it is wrong, and the two rows say whether that is specific to the plan.")
    print(f"\nwrote {ROOT / a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
