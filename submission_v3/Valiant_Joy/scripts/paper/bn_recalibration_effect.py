"""How much does BatchNorm re-calibration matter after structured pruning?

Paper claim this reproduces: removing channels leaves every BatchNorm layer holding running
mean/var estimated over the OLD channel set, so the network normalises with constants that
describe tensors that no longer exist. Re-estimating them needs no labels and no gradients.

Reported as a single before/after pair. This script regenerates it and prints the numbers with
the configuration that produced them, so the figure in the paper has a reproducible source.

    python scripts/paper/bn_recalibration_effect.py --config configs/classification.yaml

Writes results/paper/bn_recalibration.csv.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from core.baseline import get_baseline  # noqa: E402
from core.config import build_adapter, load_config, set_seed, setup_env  # noqa: E402
from core.prune import prune_model, recalibrate_bn  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/classification.yaml")
    ap.add_argument("--ratios", default="0.25,0.50", help="comma-separated prune ratios to test")
    ap.add_argument("--bn-batches", type=int, default=32)
    ap.add_argument("--out", default="results/paper/bn_recalibration.csv")
    a = ap.parse_args()

    setup_env()
    cfg = load_config(a.config)
    set_seed(cfg.get("seed", 0))
    adapter = build_adapter(cfg, None)
    base = get_baseline(adapter, None, model_cfg=cfg.get("model", {}))

    ev = adapter.evaluate(base)
    metric, base_score = ev["metric_name"], float(ev["score"])
    print(f"baseline {metric} {base_score:.4f}\n")

    x = adapter.example_input()
    ignored = adapter.ignored_layers(base)
    pcfg = cfg.get("prune", {})
    rows = []
    for r in [float(v) for v in a.ratios.split(",")]:
        t0 = time.time()
        # Prune once, then evaluate the SAME pruned network twice: before and after re-estimating
        # BN statistics. Pruning twice would confound the comparison with pruning nondeterminism.
        pruned = prune_model(base, x, r, ignored,
                             importance=pcfg.get("importance", "l2"),
                             round_to=pcfg.get("round_to", 8), report={})
        before = float(adapter.evaluate(pruned)["score"])
        n = recalibrate_bn(pruned, adapter.bn_calibration_loader(), adapter.device, a.bn_batches)
        after = float(adapter.evaluate(pruned)["score"])
        params = sum(p.numel() for p in pruned.parameters())
        base_params = sum(p.numel() for p in base.parameters())
        rows.append({"config": Path(a.config).name, "metric": metric, "baseline": base_score,
                     "prune_ratio": r, "params_removed_pct": 100 * (1 - params / base_params),
                     "bn_batches": n, "before_recal": before, "after_recal": after,
                     "ratio_after_over_before": after / before if before > 0 else float("inf"),
                     "seconds": round(time.time() - t0, 1)})
        print(f"ratio {r:.2f}  params -{rows[-1]['params_removed_pct']:.1f}%  "
              f"{metric} before {before:.4f} -> after {after:.4f}  "
              f"({rows[-1]['ratio_after_over_before']:.2f}x), {rows[-1]['seconds']:.0f}s")

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
