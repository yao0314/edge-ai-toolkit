"""Experiments 5, 6 and 7 as one factorial: is the INT8 slowdown a configuration artifact?

ONNX Runtime's own documentation names configuration traps that produce exactly the symptom this
paper reports, so a reviewer gets a free rejection unless they are ruled out:

  "S8S8 with QOperator will be slow on x86-64 CPUs and should be avoided in general"
  "it is not rare to get worse performance on old devices" (no VNNI / dot-product instructions)

And `core/quantize.py` carries an unsupported claim of its own:

  "activation_type": "uint8",  # U8S8: fast path on x86 (S8S8 QDQ was slower than FP32 there)

That parenthetical asserts a measurement nobody recorded. This either substantiates it or retracts
it.

The design is a FACTORIAL, not three separate sweeps, because ORT's documented trap is a specific
COMBINATION (QOperator x S8S8). Varying one axis at a time leaves the other as a confound:

    quant_format   {QDQ, QOperator}
    type pair      {U8S8, S8S8}
    reduce_range   {False, True}      -> primarily an ACCURACY control, not a speed one

    python scripts/paper/quant_config_factorial.py --config configs/classification.yaml

Writes results/paper/quant_factorial.csv.
"""
from __future__ import annotations

import argparse
import atexit
import itertools
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
from core.measure import runtime_metrics  # noqa: E402
from core.quantize import quantize_onnx  # noqa: E402


def _pin(label: str) -> dict:
    """Turn `calib_method_chosen` ("minmax" | "entropy" | "percentile 99.99") back into cfg overrides."""
    parts = label.split()
    if parts and parts[0] == "percentile":
        d = {"calibrate_method": "percentile"}
        if len(parts) > 1:
            d["calib_percentile"] = float(parts[1])
        return d
    return {"calibrate_method": parts[0] if parts else "minmax"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/classification.yaml")
    ap.add_argument("--threads", default="1,4")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    setup_env()
    cfg = load_config(a.config)
    set_seed(cfg.get("seed", 0))
    adapter = build_adapter(cfg, None)
    base = get_baseline(adapter, None, model_cfg=cfg.get("model", {}))
    run = cfg.get("name") or Path(a.config).stem
    x = adapter.example_input()
    # Static quantization needs real calibration batches. Loaded ONCE: load_data() re-reads the
    # dataset, and calling it per cell would make the 8 cells non-comparable on time as well as slow.
    calib = adapter.load_data().get("calib")

    ev = adapter.evaluate(base)
    metric, base_score = ev["metric_name"], float(ev["score"])
    threads = [int(t) for t in a.threads.split(",")]
    # Same timing settings as the headline measurements. A factorial run under a SHORTER timed
    # window than the claim it is defending is a weak defence: a reviewer would rightly ask whether
    # the configuration effect and the timing change were separable. Costs ~100s per config.
    mcfg = {"threads": threads, "warmup": 10, "runs": 100, "min_time_s": 2.0, "repeats": a.repeats}

    work = Path(tempfile.mkdtemp(prefix="qfact_"))
    # mkdtemp does not clean up on its own: two runs of this script left 50 MB of ONNX
    # files in the system temp directory tonight. Registered rather than wrapped in a
    # context manager so an exception still leaves the files for post-mortem if the
    # process is killed, but a normal exit removes them.
    atexit.register(lambda: shutil.rmtree(work, ignore_errors=True))
    fp32 = export_onnx(base, x, work / "fp32.onnx", input_names=adapter.input_names(),
                       output_names=adapter.output_names(),
                       opset=(cfg.get("export") or {}).get("opset", 17))
    fp32_m = runtime_metrics(fp32, x, mcfg)
    fp32_lat = {t: fp32_m.get(f"latency_p50_ms_t{t}") for t in threads}
    print(f"baseline {metric} {base_score:.4f}")
    print("fp32 latency: " + "  ".join(f"t{t}={fp32_lat[t]:.3f}ms" for t in threads) + "\n")

    qbase = dict(cfg.get("quantize") or {})
    # The shipped configs set `calibrate_method: auto`, which re-ranks four calibrators by SQNR on
    # every call. Left as-is, each of the 8 cells could land on a DIFFERENT calibrator, and
    # "QDQ vs QOperator" would then be confounded with "minmax vs percentile" - the factorial would
    # measure two things at once. So resolve `auto` ONCE, on the shipped default cell, and pin the
    # winner for every cell. The reference row therefore still matches the configuration the rest of
    # the paper was measured on, and the only things varying are the three factorial axes.
    if qbase.get("calibrate_method") == "auto":
        probe_info = {}
        quantize_onnx(fp32, work / "_calib_probe.onnx", calib, qbase, info=probe_info)
        chosen = probe_info.get("calib_method_chosen", "minmax")
        qbase.update(_pin(chosen))
        print(f"calibrate_method: auto resolved to {chosen!r}; pinned for all cells "
              f"(tried: {probe_info.get('calib_auto_tried')})")
    pct = qbase.get("calib_percentile") if qbase.get("calibrate_method") == "percentile" else ""
    print(f"pinned calibration: {qbase.get('calibrate_method')} {pct}".rstrip())

    rows = []
    for fmt, (act, wt), rr in itertools.product(
            ("qdq", "qoperator"), (("uint8", "int8"), ("int8", "int8")), (False, True)):
        tag = f"{fmt}_{act[0]}8{wt[0]}8_rr{int(rr)}".replace("uint8", "u").replace("int8", "i")
        t0 = time.time()
        out = work / f"{tag}.onnx"
        info = {}
        try:
            quantize_onnx(fp32, out, calib, {**qbase, "quant_format": fmt,
                                             "activation_type": act, "weight_type": wt,
                                             "reduce_range": rr}, info=info)
            score = float(adapter.evaluate(str(out))["score"])
            rm = runtime_metrics(out, x, mcfg)
        except Exception as e:
            print(f"  {tag:26s} FAILED: {type(e).__name__}: {str(e)[:90]}", flush=True)
            rows.append({"run": run, "quant_format": fmt, "activation_type": act,
                         "weight_type": wt, "reduce_range": rr, "error": str(e)[:200]})
            continue
        row = {"run": run, "tag": tag, "quant_format": fmt, "activation_type": act,
               "weight_type": wt, "reduce_range": rr, "metric": metric,
               "calibrate_method": qbase.get("calibrate_method"), "calib_percentile": pct,
               "baseline": base_score, "accuracy": score, "drop_pt": (score - base_score) * 100,
               "size_bytes": out.stat().st_size, "sqnr_db": info.get("int8_sqnr_db"),
               "seconds": round(time.time() - t0, 1)}
        for t in threads:
            lat = rm.get(f"latency_p50_ms_t{t}")
            row[f"latency_ms_t{t}"] = lat
            row[f"speedup_t{t}"] = fp32_lat[t] / lat if lat else None
            row[f"sd_ms_t{t}"] = rm.get(f"latency_p50_ms_sd_t{t}")
        rows.append(row)
        print(f"  {tag:26s} {metric} {score:.4f} ({row['drop_pt']:+.2f} pt)  " +
              "  ".join(f"t{t} {row[f'speedup_t{t}']:.2f}x" for t in threads if row.get(f"speedup_t{t}")),
              flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / (a.out or f"results/paper/quant_factorial_{run}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    ok = df[df.get("error").isna()] if "error" in df else df
    if len(ok):
        print("\nShipped default is qdq + U8S8 + reduce_range False. Its row is the reference;")
        print("any row that is FASTER than it means the default we measured the paper on is not")
        print("the best available configuration, which must be disclosed rather than hidden.")
        for t in threads:
            c = f"speedup_t{t}"
            if c in ok and ok[c].notna().any():
                b = ok.loc[ok[c].idxmax()]
                print(f"  t{t}: best is {b['tag']} at {b[c]:.2f}x "
                      f"({b['drop_pt']:+.2f} pt)")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
