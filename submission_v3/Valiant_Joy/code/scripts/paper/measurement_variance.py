"""Experiment 9: how much of the accuracy number is noise?

Every claim in the paper compares variants by accuracy. "The best-compression variant changes in
6 of 8 settings as the budget moves" is worth nothing if the differences that drive those changes
sit inside the measurement's own spread. We have latency SDs from `measure.repeats`; we have no
accuracy SD at all.

Method: re-evaluate each variant K times on K independent random subsets of the validation set and
report the SD. Subset resampling (rather than bootstrapping per-sample correctness) is used because
it works for every metric shape we have — top-1, AUC and mAP alike — and mAP is not the mean of a
per-sample quantity, so a per-sample bootstrap would not be valid for it.

    python scripts/paper/measurement_variance.py --config configs/classification.yaml --repeats 8

Writes results/paper/variance_<config>.csv.
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
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from core.config import build_adapter, load_config, set_seed, setup_env  # noqa: E402


def subsample_val(adapter, frac: float, seed: int):
    """Replace the val loader with a random `frac` subset. Returns a restore callable.

    Resampling the EVALUATION SET is the point: it estimates how much the reported number depends
    on which samples happened to be in it, which is the uncertainty that matters when comparing two
    variants evaluated on the same set but claimed to differ by a fraction of a point.
    """
    data = adapter.load_data()
    val = data["val"]
    ds = val.dataset
    g = np.random.default_rng(seed)
    n = max(1, int(len(ds) * frac))
    idx = g.choice(len(ds), size=n, replace=False)
    sub = DataLoader(Subset(ds, sorted(idx.tolist())), batch_size=val.batch_size,
                     shuffle=False, collate_fn=getattr(val, "collate_fn", None),
                     num_workers=0)
    data["val"] = sub
    adapter._data = data if hasattr(adapter, "_data") else None

    def restore():
        data["val"] = val
    return restore, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/classification.yaml")
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--frac", type=float, default=0.8, help="fraction of the val set per draw")
    ap.add_argument("--models", default=None,
                    help="glob of ONNX files to evaluate; default: results/<run>/models/*.onnx")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    setup_env()
    cfg = load_config(a.config)
    set_seed(cfg.get("seed", 0))
    adapter = build_adapter(cfg, None)
    run = cfg.get("name") or Path(a.config).stem

    pattern = a.models or f"results/{run}/models/*.onnx"
    files = sorted(ROOT.glob(pattern))
    # Some sweeps vary INPUT RESOLUTION as well as precision and encode it as a variant prefix
    # (192_p00_int8, 320_p00_int8, 640_...). Feeding a 192-input model the adapter's 320 example
    # raises InvalidArgument, so keep only the models built for the adapter's own size.
    size = (cfg.get("adapter_args") or {}).get("img_size")
    if size is not None:
        keep = [f for f in files if not f.stem.split("_")[0].isdigit()
                or f.stem.startswith(f"{int(size)}_")]
        if len(keep) != len(files):
            print(f"  filtered {len(files) - len(keep)} model(s) built for another input size "
                  f"(adapter img_size={size})")
            files = keep
    if not files:
        print(f"no models matched {pattern}", file=sys.stderr)
        return 1
    print(f"{len(files)} model(s), {a.repeats} draws of {a.frac:.0%} of the val set each\n")

    rows = []
    for f in files:
        scores, t0 = [], time.time()
        for k in range(a.repeats):
            restore, n = subsample_val(adapter, a.frac, seed=1000 + k)
            try:
                scores.append(float(adapter.evaluate(str(f))["score"]))
            finally:
                restore()
        s = np.array(scores)
        rows.append({"run": run, "variant": f.stem, "n_draws": len(s), "frac": a.frac,
                     "val_subset_n": n, "mean": s.mean(), "sd": s.std(ddof=1),
                     "min": s.min(), "max": s.max(),
                     "sd_pt": s.std(ddof=1) * 100, "range_pt": (s.max() - s.min()) * 100,
                     "seconds": round(time.time() - t0, 1)})
        print(f"  {f.stem:24s} mean {s.mean():.4f}  sd {s.std(ddof=1) * 100:.3f} pt  "
              f"range {(s.max() - s.min()) * 100:.3f} pt  ({rows[-1]['seconds']:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / (a.out or f"results/paper/variance_{run}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nmedian accuracy SD: {df['sd_pt'].median():.3f} pt")
    print("Any variant-to-variant difference smaller than about 2x this is not separable.")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
