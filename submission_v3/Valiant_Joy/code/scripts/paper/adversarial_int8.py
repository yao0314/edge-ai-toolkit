"""Adversarial tests of the two INT8 performance claims this paper rests on.

CLAIM A. Static QDQ INT8 is slower than fp32 on transformer graphs while dynamic INT8 is faster.
         Proposed mechanism: static quantization covers 26 op types (QLinearOpsRegistry) against
         dynamic's 7 (IntegerOpsRegistry), so on a heterogeneous graph it inserts int8/fp32
         boundaries everywhere - MobileViT static carries 416 QuantizeLinear nodes against 90
         compute ops, of which only 36% of dequantizes reach arithmetic.

CLAIM B. Quantization destroys parallel scaling: 6 variants are faster at 1 thread (1.04-1.68x)
         and slower at 4, with no counter-example.

Each claim has a decisive attack, and this script runs them. It deliberately calls ONNX Runtime's
quantization API DIRECTLY rather than going through core/quantize.py: an adversarial test of our
own finding must not inherit a possible bug in our own wrapper.

    python scripts/paper/adversarial_int8.py --model results/classification_mobilevit/models/p00_fp32.onnx

Writes results/paper/adversarial_<model>.csv.

--------------------------------------------------------------------------------------------------
ATTACK A1 - "it is the OP COVERAGE, not static-vs-dynamic"
  Run static QDQ restricted to exactly dynamic's op set ([MatMul]). Three outcomes, all
  informative:
    restricted-static is fast, like dynamic  -> the mechanism IS op coverage. Claim A stands but
                                                must be restated as a coverage effect, not a
                                                static/dynamic effect.
    restricted-static is still slow          -> coverage is not the mechanism. The cost is in Q/DQ
                                                node insertion itself, and the stated mechanism is
                                                WRONG even though the observation holds.
    restricted-static is slower than full    -> something else entirely; investigate before writing.

ATTACK A2 - "it is ORT's graph fusion, not the boundaries"
  fp32 may benefit from operator fusions that inserted Q/DQ nodes block. Measure both precisions
  with graph optimisation at ALL and at DISABLE_ALL. If the fp32 advantage disappears when fusion
  is off for both, the finding is about fusion opportunity, not about quantization.

ATTACK B1 - "small models just scale badly, regardless of precision"
  Compute scaling efficiency separately for each precision:
      eff(N) = latency(1) / (N * latency(N))
  If int8's efficiency is systematically below fp32's, claim B holds. If both degrade equally, the
  ratio crossing 1.0 is an artifact of int8 starting faster, not of worse scaling.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import pandas as pd  # noqa: E402
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType  # noqa: E402
from onnxruntime.quantization import quantize_dynamic, quantize_static  # noqa: E402

from core.cpu_topology import pick_cpus  # noqa: E402

OPT = {"all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
       "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL}


def spec(path: Path):
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    i = s.get_inputs()[0]
    return i.name, [1 if (isinstance(d, str) or d is None) else d for d in i.shape]


class Reader(CalibrationDataReader):
    """Fixed synthetic calibration data, identical for every variant, so calibration input is not a
    confound between the configurations being compared."""

    def __init__(self, name, shape, n=16, seed=0):
        g = np.random.default_rng(seed)
        self.data = [{name: g.random(shape).astype(np.float32)} for _ in range(n)]
        self.i = 0

    def get_next(self):
        if self.i >= len(self.data):
            return None
        self.i += 1
        return self.data[self.i - 1]


def latency(path: Path, threads: int, opt: str, runs: int, warmup: int) -> tuple[float, float]:
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = threads
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = OPT[opt]
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    name, shape = spec(path)
    x = np.random.rand(*shape).astype(np.float32)
    for _ in range(warmup):
        s.run(None, {name: x})
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        s.run(None, {name: x})
        t.append(time.perf_counter() - t0)
    t = np.array(t) * 1000
    return float(np.median(t)), float(t.std(ddof=1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="results/classification_mobilevit/models/p00_fp32.onnx",
                    help="an FP32 ONNX model to quantize several ways")
    ap.add_argument("--threads", default="1,2,4,6")
    ap.add_argument("--runs", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    src = ROOT / a.model if not Path(a.model).is_absolute() else Path(a.model)
    if not src.exists():
        print(f"no such model: {src}", file=sys.stderr)
        return 1
    threads = [int(t) for t in a.threads.split(",")]
    name, shape = spec(src)
    work = ROOT / "results" / "paper" / "_adv" / src.parent.parent.name
    work.mkdir(parents=True, exist_ok=True)
    print(f"{src.relative_to(ROOT)}   input {name} {shape}")
    print(f"pinned CPUs per thread count: " +
          ", ".join(f"{t}->{pick_cpus(t, 'auto')}" for t in threads) + "\n")

    # ---- build the variants -------------------------------------------------------------------
    variants = {"fp32": src}
    jobs = [
        ("static_full", dict(op_types_to_quantize=None)),
        ("static_matmul_only", dict(op_types_to_quantize=["MatMul"])),   # ATTACK A1
        ("static_conv_matmul", dict(op_types_to_quantize=["Conv", "MatMul"])),
    ]
    for tag, extra in jobs:
        out = work / f"{tag}.onnx"
        if not out.exists():
            try:
                quantize_static(str(src), str(out), Reader(name, shape),
                                quant_format=QuantFormat.QDQ, per_channel=True,
                                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                                **{k: v for k, v in extra.items() if v is not None})
            except Exception as e:
                print(f"  {tag}: quantization failed: {type(e).__name__}: {str(e)[:100]}")
                continue
        variants[tag] = out
    for tag, ops in (("dynamic_matmul", ["MatMul"]), ("dynamic_conv_matmul", ["Conv", "MatMul"])):
        out = work / f"{tag}.onnx"
        if not out.exists():
            try:
                quantize_dynamic(str(src), str(out), op_types_to_quantize=ops,
                                 per_channel=True, weight_type=QuantType.QInt8)
            except Exception as e:
                print(f"  {tag}: quantization failed: {type(e).__name__}: {str(e)[:100]}")
                continue
        variants[tag] = out

    # ---- graph statistics ---------------------------------------------------------------------
    stats = {}
    for tag, p in variants.items():
        g = onnx.load(str(p), load_external_data=False).graph
        ops = [n.op_type for n in g.node]
        stats[tag] = {"n_nodes": len(ops), "n_q": ops.count("QuantizeLinear"),
                      "n_dq": ops.count("DequantizeLinear"), "size_mb": p.stat().st_size / 2**20}

    # ---- measure -------------------------------------------------------------------------------
    rows = []
    for tag, p in variants.items():
        for opt in ("all", "none"):                                        # ATTACK A2
            for t in threads:
                med, sd = latency(p, t, opt, a.runs, a.warmup)
                rows.append({"model": src.parent.parent.name, "variant": tag, "opt_level": opt,
                             "threads": t, "latency_ms": med, "sd_ms": sd, **stats[tag]})
                print(f"  {tag:20s} opt={opt:4s} t={t:2d}  {med:8.3f} ms  (sd {sd:.3f})", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / (a.out or f"results/paper/adversarial_{src.parent.parent.name}.csv")
    df.to_csv(out, index=False)

    # ---- verdicts -------------------------------------------------------------------------------
    print("\n" + "=" * 78)
    for opt in ("all", "none"):
        d = df[df["opt_level"] == opt]
        base = {t: float(d[(d["variant"] == "fp32") & (d["threads"] == t)]["latency_ms"].iloc[0])
                for t in threads if len(d[(d["variant"] == "fp32") & (d["threads"] == t)])}
        print(f"\nspeedup vs fp32, graph optimisation = {opt}")
        print(f"{'variant':22s} {'Q/DQ':>7s} " + " ".join(f"{'t' + str(t):>8s}" for t in threads))
        for tag in variants:
            if tag == "fp32":
                continue
            cells = []
            for t in threads:
                r = d[(d["variant"] == tag) & (d["threads"] == t)]
                cells.append(f"{base[t] / float(r['latency_ms'].iloc[0]):8.2f}" if len(r) and t in base else "       -")
            print(f"{tag:22s} {stats[tag]['n_q'] + stats[tag]['n_dq']:7d} " + " ".join(cells))

    print("\nATTACK B1 - scaling efficiency  eff(N) = lat(1) / (N * lat(N)), higher is better")
    d = df[df["opt_level"] == "all"]
    print(f"{'variant':22s} " + " ".join(f"{'t' + str(t):>8s}" for t in threads))
    for tag in variants:
        g = d[d["variant"] == tag].set_index("threads")["latency_ms"]
        if 1 not in g.index:
            continue
        print(f"{tag:22s} " + " ".join(f"{g[1] / (t * g[t]):8.3f}" if t in g.index else "       -"
                                       for t in threads))
    print("\nIf int8 rows sit systematically BELOW the fp32 row, claim B holds. If every row")
    print("degrades together, the 4-thread crossing is int8 starting faster, not scaling worse.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
