"""Follow-up to experiment 10: was the pruning gain real, or did we tune only one arm?

Experiment 10 found that raising `recovery_lr_mult` from 1 to 20 takes global pruning at ratio 0.25
from -2.08 pt to -0.93 pt, which brings it inside the 2.0-point budget and would give 5.56x
compression against 3.40x for quantize-only.

But `recovery_lr_mult` applies ONLY after pruning. The quantize-only arm never received an
equivalent amount of extra training. Comparing a tuned arm against an untuned one is precisely the
error Wang et al. (arXiv:2301.05219) describe, and it is the control that could take this finding
away again. So:

  arm 1  UNPRUNED, then the same epoch budget at the same learning rates
  arm 2  PRUNED at 0.25, then the same epoch budget at the same learning rates
  both   quantized to INT8 afterwards, because the deliverable is the quantized model and INT8 on
         top of a pruned network costs something that INT8 on a dense network may not

The comparison that matters is (arm 2 pruned + INT8) against (arm 1 unpruned + INT8), each at its
OWN best schedule. Anything else compares a tuned method with an untuned one.

    python scripts/paper/fair_schedule_comparison.py --lr-mults 1,5,20 --epochs 20

Writes results/paper/fair_schedule.csv.
"""
from __future__ import annotations

import argparse
import atexit
import copy
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from core.baseline import get_baseline  # noqa: E402
from core.config import build_adapter, load_config, set_seed, setup_env  # noqa: E402
from core.export import export_onnx  # noqa: E402
from core.prune import prune_and_finetune  # noqa: E402
from core.quantize import quantize_onnx  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments/classification_prune_global.yaml")
    ap.add_argument("--ratio", type=float, default=0.25)
    ap.add_argument("--lr-mults", default="1,5,20")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--budget-pt", type=float, default=2.0)
    ap.add_argument("--out", default="results/paper/fair_schedule.csv")
    a = ap.parse_args()

    setup_env()
    cfg = load_config(a.config)
    seed = cfg.get("seed", 0)
    set_seed(seed)
    adapter = build_adapter(cfg, None)
    base = get_baseline(adapter, None, model_cfg=cfg.get("model", {}))
    ev = adapter.evaluate(base)
    metric, base0 = ev["metric_name"], float(ev["score"])
    x = adapter.example_input()
    pcfg = dict(cfg.get("prune") or {})
    qcfg = dict(cfg.get("quantize") or {})
    # Static quantization needs real calibration batches; loaded once and shared by both arms so the
    # two are quantized identically. `calibrate_method: auto` is left alone here on purpose - each
    # arm is allowed its own best calibration, because the comparison is between DELIVERABLES, not
    # between quantizer settings.
    calib = adapter.load_data().get("calib")
    ex = cfg.get("export") or {}
    work = Path(tempfile.mkdtemp(prefix="fair_"))
    # mkdtemp does not clean up on its own: two runs of this script left 50 MB of ONNX
    # files in the system temp directory tonight. Registered rather than wrapped in a
    # context manager so an exception still leaves the files for post-mortem if the
    # process is killed, but a normal exit removes them.
    atexit.register(lambda: shutil.rmtree(work, ignore_errors=True))
    print(f"untrained-further baseline {metric} {base0:.4f}\n")

    def quantize_and_score(model, tag):
        """Accuracy and size of the QUANTIZED deliverable, which is what the budget is spent on."""
        f32 = export_onnx(model, x, work / f"{tag}.onnx", input_names=adapter.input_names(),
                          output_names=adapter.output_names(), opset=ex.get("opset", 17),
                          atol=ex.get("atol", 1e-3), rtol=ex.get("rtol", 1e-3))
        q = work / f"{tag}_int8.onnx"
        info = {}
        quantize_onnx(f32, q, calib, qcfg, info=info)
        return (float(adapter.evaluate(str(q))["score"]), q.stat().st_size,
                f32.stat().st_size, info.get("int8_sqnr_db"))

    rows = []
    # lr_mult 0 means NO extra training. It belongs in the unpruned arm's grid because the honest
    # reference is "the best quantize-only deliverable available", and doing nothing is one of the
    # options - the baseline may already be at convergence, in which case 20 more epochs can only
    # overfit it and would hand the pruned arm a win it did not earn.
    mults = [0.0] + [float(m) for m in a.lr_mults.split(",")]
    for mult in mults:
        for arm, ratio in (("unpruned", 0.0), ("pruned", a.ratio)):
            if mult == 0.0 and arm == "pruned":
                continue        # pruning with no recovery at all is not a candidate deliverable
            set_seed(seed)
            t0 = time.time()
            # ratio 0.0 with no allocation plan means prune_and_finetune does not prune and does not
            # fine-tune, so the unpruned arm is trained through adapter.finetune directly to give it
            # the SAME epoch budget and the same learning rate the pruned arm receives.
            if ratio > 0:
                m = prune_and_finetune(adapter, base, ratio, a.epochs,
                                       {**pcfg, "recovery_lr_mult": mult}, report={})
            elif mult == 0.0:
                m = base            # the untouched baseline, quantized: the do-nothing reference
            else:
                # exactly what prune_and_finetune does on the pruned arm: adapter finetune_lr
                # scaled by the same multiplier, same epoch count, same finetune entry point.
                lr = float(adapter.cfg.get("finetune_lr", 0.01)) * mult
                m = adapter.finetune(copy.deepcopy(base), a.epochs, lr)
            fp32_score = float(adapter.evaluate(m)["score"])
            q_score, q_bytes, f32_bytes, sqnr = quantize_and_score(m, f"{arm}_m{mult:g}")
            rows.append({"arm": arm, "ratio": ratio,
                         "epochs": 0 if mult == 0.0 else a.epochs, "lr_mult": mult,
                         "metric": metric, "baseline_untrained": base0,
                         "fp32_after": fp32_score, "int8_after": q_score,
                         "int8_bytes": q_bytes, "fp32_bytes": f32_bytes,
                         "int8_sqnr_db": sqnr, "seconds": round(time.time() - t0, 1)})
            print(f"  {arm:9s} lr_mult {mult:5.1f}  fp32 {fp32_score:.4f}  int8 {q_score:.4f}  "
                  f"{q_bytes / 2**20:.2f} MB  ({rows[-1]['seconds']:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # Each arm judged at its OWN best schedule, and the budget measured against the best UNPRUNED
    # accuracy achieved rather than the untrained starting point - otherwise the pruned arm is
    # credited with training the other arm never received.
    up = df[df["arm"] == "unpruned"]
    pr = df[df["arm"] == "pruned"]
    if up.empty or pr.empty:
        print("\nboth arms are needed for the comparison", file=sys.stderr)
        return 1
    bu, bp = up.loc[up["int8_after"].idxmax()], pr.loc[pr["int8_after"].idxmax()]
    ref = float(bu["int8_after"])
    print("\n" + "=" * 78)
    print(f"best unpruned + INT8 : {ref:.4f}  at lr_mult {bu['lr_mult']:g}  "
          f"{bu['int8_bytes'] / 2**20:.2f} MB")
    print(f"best pruned   + INT8 : {float(bp['int8_after']):.4f}  at lr_mult {bp['lr_mult']:g}  "
          f"{bp['int8_bytes'] / 2**20:.2f} MB")
    drop_vs_tuned = (float(bp["int8_after"]) - ref) * 100
    drop_vs_naive = (float(bp["int8_after"]) - base0) * 100
    print(f"\npruned drop vs the TUNED unpruned reference : {drop_vs_tuned:+.2f} pt"
          f"  -> {'FITS' if -drop_vs_tuned <= a.budget_pt else 'OVER'} the {a.budget_pt}-pt budget")
    print(f"pruned drop vs the untrained baseline      : {drop_vs_naive:+.2f} pt   (the flattering number)")
    print(f"compression of pruned over unpruned        : "
          f"{float(bu['int8_bytes']) / float(bp['int8_bytes']):.2f}x")
    if -drop_vs_tuned > a.budget_pt >= -drop_vs_naive:
        print("\nThe experiment-10 gain WAS an artifact of tuning only the pruned arm. Retract it.")
    elif -drop_vs_tuned <= a.budget_pt:
        print("\nThe gain SURVIVES a fairly-tuned reference. Pruning pays, and the schedule is the")
        print("reason the literature (and this project until today) missed it.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
