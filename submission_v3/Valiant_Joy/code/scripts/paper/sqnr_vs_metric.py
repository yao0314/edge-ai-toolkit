"""Does a label-free output-SQNR probe predict the ACTUAL task-metric damage of pruning a layer?

Paper claim this reproduces: allocation schemes rank layers by a cheap proxy. If the proxy does
not track the metric being optimised, no allocation driven by it can work. On a detector we
measured a correlation of only +0.395 across six single-layer cuts — but six points is thin for a
correlation, so this script samples more layers, spread across the probe's own range.

Method, per layer:
  1. prune that layer ALONE at --ratio (every other layer untouched)
  2. re-estimate BatchNorm statistics (label-free), so the comparison is not confounded by the
     stale-statistics effect measured separately in bn_recalibration_effect.py
  3. evaluate the real task metric on the validation set
Then correlate the probe's score against the measured metric drop.

    python scripts/paper/sqnr_vs_metric.py --config configs/detection.yaml --n-layers 16

Writes results/paper/sqnr_vs_metric.csv.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.baseline import get_baseline  # noqa: E402
from core.config import build_adapter, load_config, set_seed, setup_env  # noqa: E402
from core.prune import layer_sensitivity, prune_model, recalibrate_bn  # noqa: E402


def spread(names_scores: dict, k: int) -> list:
    """k layers spread evenly across the probe's score range, so the correlation is not computed
    over a cluster. Taking the k best or k worst would bias it."""
    ranked = sorted(names_scores.items(), key=lambda kv: kv[1])
    if k >= len(ranked):
        return [n for n, _ in ranked]
    idx = np.linspace(0, len(ranked) - 1, k).round().astype(int)
    return [ranked[i][0] for i in sorted(set(idx))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/detection.yaml")
    ap.add_argument("--ratio", type=float, default=0.5, help="prune ratio applied to the single layer")
    ap.add_argument("--n-layers", type=int, default=16)
    ap.add_argument("--bn-batches", type=int, default=8)
    ap.add_argument("--out", default="results/paper/sqnr_vs_metric.csv")
    a = ap.parse_args()

    setup_env()
    cfg = load_config(a.config)
    set_seed(cfg.get("seed", 0))
    adapter = build_adapter(cfg, None)
    base = get_baseline(adapter, None, model_cfg=cfg.get("model", {}))

    ev = adapter.evaluate(base)
    metric, base_score = ev["metric_name"], float(ev["score"])
    print(f"baseline {metric} {base_score:.4f}")

    rep = {}
    sens = layer_sensitivity(adapter, base, {"probe_ratio": a.ratio, "probe_batches": 2,
                                             "round_to": cfg.get("prune", {}).get("round_to", 8)}, rep)
    if not sens:
        print("no layers probed", file=sys.stderr)
        return 1
    probe = {n: v["quality"] for n, v in sens.items()}
    print(f"probe: {len(probe)} layers in {rep.get('sensitivity_s')}s, "
          f"range {min(probe.values()):.2f}..{max(probe.values()):.2f} dB")

    picked = spread(probe, a.n_layers)
    print(f"evaluating {len(picked)} layers spread across that range\n")

    x = adapter.example_input()
    ignored = adapter.ignored_layers(base)
    rows = []
    for i, name in enumerate(picked, 1):
        t0 = time.time()
        cand = prune_model(base, x, 0.0, ignored,
                           importance=cfg.get("prune", {}).get("importance", "l2"),
                           round_to=cfg.get("prune", {}).get("round_to", 8),
                           ratio_per_layer={name: a.ratio}, report={})
        recalibrate_bn(cand, adapter.bn_calibration_loader(), adapter.device, a.bn_batches)
        score = float(adapter.evaluate(cand)["score"])
        rows.append({"layer": name, "probe_db": probe[name], "metric": metric,
                     "baseline": base_score, "after": score, "delta": score - base_score,
                     "param_frac": sens[name].get("param_frac"), "seconds": round(time.time() - t0, 1)})
        print(f"  [{i:2d}/{len(picked)}] {name:30s} probe {probe[name]:6.2f} dB  "
              f"{metric} {score:.4f} ({score - base_score:+.4f})  {rows[-1]['seconds']:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    p, d = df["probe_db"].values, df["delta"].values
    print(f"\nn = {len(df)}")
    if len(df) > 2 and p.std() > 0 and d.std() > 0:
        pear = float(np.corrcoef(p, d)[0, 1])
        from scipy.stats import spearmanr
        sp = spearmanr(p, d)
        print(f"  Pearson  r = {pear:+.3f}")
        print(f"  Spearman rho = {sp.statistic:+.3f}  (p = {sp.pvalue:.3f})")
        print("  +1 would mean the probe ranks layers correctly (high dB -> little damage);")
        print("  near 0 means an allocation driven by this proxy cannot work on this task.")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
