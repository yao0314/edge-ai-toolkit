"""Experiment 10: is "pruning earns nothing" a property of pruning, or of our recovery schedule?

Wang et al. (arXiv:2301.05219) revived 2017 L1-norm pruning to match modern methods purely by
improving the fine-tuning learning-rate schedule, and concluded that method A can beat B under one
comparison setup and lose under another. Our five structured strategies all ran 20 epochs with
warmup 0.05 and `recovery_lr_mult` left at its default, so the claim "no structured strategy fits a
2.0-point budget" is currently attributable to the recovery protocol rather than to pruning.

This sweeps the recovery schedule on the STRONGEST strategy (global pruning, -2.34 pt), because
showing that the best candidate cannot be rescued is the strong form of the claim; showing it for
the worst proves nothing.

No export, no quantization, no latency: only accuracy after recovery, which is the quantity the
budget is spent on.

    python scripts/paper/finetune_schedule_control.py --lr-mults 1,2,5,10,20 --epochs 20,40

Writes results/paper/finetune_schedule.csv.
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
from core.prune import prune_and_finetune  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/classification_prune_global.yaml")
    ap.add_argument("--ratio", type=float, default=0.25)
    ap.add_argument("--lr-mults", default="1,2,5,10,20")
    ap.add_argument("--epochs", default="20,40")
    ap.add_argument("--budget-pt", type=float, default=2.0,
                    help="the budget the paper claims no strategy fits; used only for reporting")
    ap.add_argument("--out", default="results/paper/finetune_schedule.csv")
    a = ap.parse_args()

    setup_env()
    cfg = load_config(a.config)
    set_seed(cfg.get("seed", 0))
    adapter = build_adapter(cfg, None)
    base = get_baseline(adapter, None, model_cfg=cfg.get("model", {}))
    ev = adapter.evaluate(base)
    metric, base_score = ev["metric_name"], float(ev["score"])
    print(f"baseline {metric} {base_score:.4f}   strategy: {Path(a.config).stem}   ratio {a.ratio}\n")

    pcfg = dict(cfg.get("prune") or {})
    rows = []
    for ep in [int(x) for x in a.epochs.split(",")]:
        for mult in [float(x) for x in a.lr_mults.split(",")]:
            set_seed(cfg.get("seed", 0))     # same init and same data order for every cell
            t0 = time.time()
            rep = {}
            model = prune_and_finetune(adapter, base, a.ratio, ep,
                                       {**pcfg, "recovery_lr_mult": mult}, report=rep)
            score = float(adapter.evaluate(model)["score"])
            drop = (score - base_score) * 100
            rows.append({"strategy": Path(a.config).stem, "ratio": a.ratio, "epochs": ep,
                         "recovery_lr_mult": mult, "recovery_lr": rep.get("recovery_lr"),
                         "metric": metric, "baseline": base_score, "after": score,
                         "drop_pt": drop, "fits_budget": bool(-drop <= a.budget_pt),
                         "seconds": round(time.time() - t0, 1)})
            print(f"  epochs {ep:3d}  lr_mult {mult:5.1f}  {metric} {score:.4f}  "
                  f"drop {drop:+.2f} pt  {'FITS' if -drop <= a.budget_pt else 'over'} "
                  f"{a.budget_pt} pt  ({rows[-1]['seconds']:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    best = df.loc[df["drop_pt"].idxmax()]
    print(f"\nbest cell: epochs {best['epochs']}, lr_mult {best['recovery_lr_mult']} "
          f"-> {best['drop_pt']:+.2f} pt")
    if df["fits_budget"].any():
        print(f"AT LEAST ONE SCHEDULE FITS THE {a.budget_pt}-POINT BUDGET. The headline claim must")
        print("change: pruning pays, but only under a recovery schedule the literature does not report.")
    else:
        print(f"No schedule fits the {a.budget_pt}-point budget. The claim is stable against the")
        print("recovery protocol, which is what the attack asked us to show.")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
