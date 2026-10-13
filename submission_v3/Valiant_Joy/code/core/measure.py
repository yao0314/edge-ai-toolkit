"""Measure one model variant -> one row of metrics.

Latency / CPU / RAM are measured with ONNX Runtime (.onnx) or LiteRT (.tflite), CPU,
batch=1, inside an isolated subprocess (core/_probe.py). Numbers from a dev laptop are
RELATIVE: use them to compare variants, not as absolute UNO Q figures.

peak_rss_mb subtracts an "empty session" of the SAME runtime (column rss_baseline):
  empty_session : ONNX  - Identity model with the input shape
  empty_model   : TFLite - one-op TFLite model with the model's input shape/dtype (needs TensorFlow)
  runtime_only  : TFLite without TensorFlow - runtime imported, no model (interpreter overhead
                  then counts towards the model, so values are slightly higher)
It is the model's footprint *within its runtime*: compare variants of one runtime with it;
compare different runtimes (or match a whole-process official number) with peak_rss_total_mb.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

# torch is NOT imported at module level. Quantizing and measuring a supplied .onnx needs only
# onnxruntime + numpy + psutil, and on competition day the sandbox may not be able to install a
# 2-3 GB torch. Every torch use below is either lazy or guarded by "the caller passed a torch
# model". Type hints are strings (from __future__ import annotations), so they cost nothing.

from .cpu_topology import describe as describe_cpus
from .cpu_topology import pick_cpus
from .energy import measure_energy

log = logging.getLogger(__name__)
PROBE = Path(__file__).with_name("_probe.py")
MAKE_EMPTY_TFLITE = Path(__file__).with_name("_make_empty_tflite.py")
NAN = float("nan")

DEFAULT_MEASURE_CFG = {
    "threads": [1, 4],        # ORT intra-op threads; one latency/CPU/RSS set per value
    "primary_threads": None,  # which thread count fills the un-suffixed columns (default: first)
    "warmup": 10,
    "runs": 100,
    "spinning": True,         # ORT thread-pool spin-wait (raises CPU%, lowers latency)
    "opt_level": "all",
    # ORT's two memory strategies, both on by ITS default. The arena allocates a pool up front and
    # does not return it, so it looked like a free way to lower peak_rss_mb, which is a reported
    # number. Measured on a 2.6 MB INT8 CNN at 4 threads: peak RSS 62.0 -> 61.4 MB (1%) for 3% more
    # latency (p50 1.201 -> 1.242 ms). Not worth taking by default; exposed because a bigger model
    # or a dynamic-shape one may answer differently, and that has to be measured, not assumed.
    "cpu_mem_arena": True,
    "mem_pattern": True,
    "rss": True,
    "repeats": 1,             # probe runs per thread count: median is reported, spread as *_sd columns
    "sample_interval": 0.1,
    "min_time_s": 2.0,        # timed window lasts >= runs AND >= min_time_s (accurate CPU counters)
    "timeout": 900,
    "affinity": "auto",       # auto (fastest core class, 1 thread/physical core) | none | [cpu ids]
    # Warn when this much of the machine was busy with something else. Calibrated, not guessed: on an
    # idle host the floor here is 4-6%, and a large memory-bound model measured at 9.3% came out 13%
    # slower than the same model at 4.5% (27.0 vs 24.3 ms). Sensitivity depends on the model: a small
    # INT8 model moved 0.3% under the same 9% load, so treat this as "re-check", not "discard".
    "other_load_warn_pct": 8.0,   # RETAINED but NOT the alarm: it fires on 61% of measurements
                                  # while correlating -0.057 (p=0.39) with real core delivery.
    # The alarm that works, WHERE IT WORKS: cpu_ms_per_inf / (threads * latency), applied only at
    # thread counts at or below the physical core count. Loose by design - a memory-bound model at
    # 1 thread sits a little under 1.0 honestly (a large FP32 CNN measured 0.91).
    "core_delivery_warn": 0.90,
    "energy": {"provider": "none"},
}


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _as_float_numpy(x) -> np.ndarray:
    """torch tensor or numpy array -> float32 ndarray. The ONNX-only path has no torch, so the
    example input arrives as a plain array and must not be assumed to have .detach()."""
    if hasattr(x, "detach"):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def count_macs(model: nn.Module, example_input: torch.Tensor) -> int:
    """MACs via thop (ultralytics-thop). Counts multiply-accumulates of Conv/Linear/etc. at batch=1."""
    import torch
    from thop import profile

    m = copy.deepcopy(model).cpu().eval()
    x = example_input[:1].detach().cpu().float()
    with torch.no_grad():
        macs, _ = profile(m, inputs=(x,), verbose=False)
    return int(macs)


def count_macs_onnx(path: str | Path) -> int:
    """Fallback when no torch model exists: MACs of Conv/ConvTranspose/Gemm/MatMul from shapes."""
    import onnx
    from onnx import numpy_helper, shape_inference

    m = onnx.load(str(path))
    for vi in list(m.graph.input) + list(m.graph.output):
        d = vi.type.tensor_type.shape.dim
        if d and not d[0].HasField("dim_value"):
            d[0].dim_value = 1
    m = shape_inference.infer_shapes(m)
    shapes = {}
    for vi in list(m.graph.value_info) + list(m.graph.input) + list(m.graph.output):
        shapes[vi.name] = [x.dim_value for x in vi.type.tensor_type.shape.dim]
    for init in m.graph.initializer:
        shapes[init.name] = list(numpy_helper.to_array(init).shape)
    total = 0
    for n in m.graph.node:
        try:
            if n.op_type in ("Conv", "ConvTranspose"):
                w = shapes[n.input[1]]
                out = shapes[n.output[0]] if n.op_type == "Conv" else shapes[n.input[0]]
                total += int(np.prod(out)) * int(np.prod(w[1:])) * (1 if n.op_type == "Conv" else w[0])
            elif n.op_type in ("Gemm", "MatMul"):
                a, out = shapes[n.input[0]], shapes[n.output[0]]
                total += int(np.prod(out)) * int(a[-1])
        except (KeyError, IndexError):
            continue
    return total


def write_empty_model(shape, path: str | Path) -> Path:
    """Single Identity node with the model's input shape: the 'empty session' RSS baseline."""
    import onnx
    from onnx import TensorProto, helper

    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, list(shape))
    out = helper.make_tensor_value_info("output", TensorProto.FLOAT, list(shape))
    graph = helper.make_graph([helper.make_node("Identity", ["input"], ["output"])], "empty", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return Path(path)


def backend_of(path: str | Path) -> str:
    """Runtime for a model file, decided from its content (TFL3 magic) before its extension."""
    from .model_io import model_format

    return "tflite" if model_format(path) == "tflite" else "onnx"


def empty_tflite_for(model_path: str | Path) -> tuple[str, str]:
    """(baseline model path, rss_baseline label) for a TFLite model.

    Builds (once, cached in weights/_cache/) a one-op TFLite model with the same input
    shape and dtype, in a subprocess so TensorFlow never enters this process. Without
    TensorFlow: ('none', 'runtime_only').
    """
    from ai_edge_litert.interpreter import Interpreter

    from .config import ROOT

    d = Interpreter(model_path=str(model_path)).get_input_details()[0]
    shape = [int(s) for s in d["shape"]]
    dtype = np.dtype(d["dtype"]).name
    cache = ROOT / "weights" / "_cache" / f"empty_{'x'.join(map(str, shape))}_{dtype}.tflite"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        res = subprocess.run([sys.executable, str(MAKE_EMPTY_TFLITE), "--shape", ",".join(map(str, shape)),
                              "--dtype", dtype, "--out", str(cache)], capture_output=True, text=True)
        if res.returncode != 0 or not cache.exists() or cache.stat().st_size == 0:
            cache.unlink(missing_ok=True)
            if res.returncode != 3:
                log.warning("could not build empty TFLite baseline: %s", res.stderr[-500:])
            else:
                log.info("TensorFlow not installed: TFLite RSS baseline = runtime only")
            return "none", "runtime_only"
    return str(cache), "empty_model"


def _run_probe(model_path: str, input_npy: str, threads: int, cfg: dict, backend: str | None = None) -> dict:
    cmd = [sys.executable, str(PROBE), "--model", str(model_path), "--input", input_npy,
           "--threads", str(threads), "--warmup", str(cfg["warmup"]), "--runs", str(cfg["runs"]),
           "--opt-level", cfg["opt_level"], "--sample-interval", str(cfg["sample_interval"]),
           "--min-time", str(cfg["min_time_s"]), "--backend", backend or backend_of(model_path)]
    cpus = pick_cpus(threads, cfg.get("affinity", "auto"))
    if cpus:
        cmd += ["--affinity", ",".join(map(str, cpus))]
    if not cfg["spinning"]:
        cmd.append("--no-spinning")
    if not cfg.get("cpu_mem_arena", True):
        cmd.append("--no-cpu-mem-arena")
    if not cfg.get("mem_pattern", True):
        cmd.append("--no-mem-pattern")
    env = dict(os.environ, OMP_NUM_THREADS=str(threads))
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg["timeout"], env=env)
    if res.returncode != 0:
        raise RuntimeError(f"probe failed ({res.returncode}): {res.stderr[-2000:]}")
    return json.loads(res.stdout.strip().splitlines()[-1])


_EMPTY_CACHE: dict = {}


def runtime_metrics(model_path: str | Path, example_input: torch.Tensor, cfg: dict | None = None,
                    input_npy: str | None = None) -> dict:
    """Latency p50/p90, CPU utilisation and peak RSS (minus empty baseline) per thread count.

    model_path: ONNX (ONNX Runtime) or TFLite (LiteRT), detected from the file content.
    The empty baseline follows the runtime (see module docstring, column rss_baseline).
    """
    cfg = {**DEFAULT_MEASURE_CFG, **(cfg or {})}
    backend = backend_of(model_path)
    td = tempfile.TemporaryDirectory()
    if input_npy is None:
        input_npy = str(Path(td.name) / "x.npy")
        np.save(input_npy, _as_float_numpy(example_input[:1]))
    shape = tuple(example_input[:1].shape)
    if backend == "onnx":
        empty, baseline_kind = str(write_empty_model(shape, Path(td.name) / "empty.onnx")), "empty_session"
    else:
        empty, baseline_kind = empty_tflite_for(model_path)
    row = {"runtime": backend, "rss_baseline": baseline_kind}
    reps = max(int(cfg.get("repeats", 1)), 1)

    def probe(path, t):
        """Median over `repeats` independent probe processes (+ the spread, for repeatability)."""
        runs = [_run_probe(str(path), input_npy, t, cfg, backend) for _ in range(reps)]
        med = {k: float(np.median([r[k] for r in runs])) for k in runs[0] if isinstance(runs[0][k], (int, float))}
        med["_sd"] = {k: float(np.std([r[k] for r in runs])) for k in ("latency_p50_ms", "peak_rss_mb",
                                                                       "session_peak_rss_mb", "cpu_ms_per_inf")}
        return med

    try:
        for t in cfg["threads"]:
            r = probe(model_path, t)
            row[f"latency_p50_ms_t{t}"] = r["latency_p50_ms"]
            row[f"latency_p90_ms_t{t}"] = r["latency_p90_ms"]
            if reps > 1:
                row[f"latency_p50_ms_sd_t{t}"] = r["_sd"]["latency_p50_ms"]
                row[f"cpu_ms_per_inf_sd_t{t}"] = r["_sd"]["cpu_ms_per_inf"]
            row[f"cpu_util_pct_t{t}"] = r["cpu_util_pct"]
            row[f"cpu_util_pct_max_t{t}"] = r["cpu_util_pct_sampled_max"]
            row[f"cpu_ms_per_inf_t{t}"] = r["cpu_ms_per_inf"]
            row[f"n_runs_t{t}"] = r["n_runs"]
            row[f"other_load_pct_t{t}"] = r.get("other_load_pct", NAN)
            # CPU consumed per unit of wall time, relative to the width we asked for. AT OR BELOW
            # the physical core count this reads as "the fraction of requested cores that actually
            # did work" and is the contamination signal; ABOVE it the thread pool spin-waits and the
            # mean/median mismatch between cpu_ms_per_inf and latency_p50_ms pushes it over 1.0
            # (measured 2.43 at 8 threads on 6 P-cores), where it is a CPU-burn ratio instead.
            # Recorded at every thread count, interpreted only inside the valid range.
            # other_load_pct above is NOT a contamination signal at all: over 232 measurements it
            # fires on 141 while correlating with this quantity at Spearman -0.057, p=0.39. It is
            # kept only because removing it would change every existing CSV's schema.
            lat, cpu = r["latency_p50_ms"], r["cpu_ms_per_inf"]
            row[f"core_delivery_t{t}"] = (cpu / (t * lat)) if (lat and lat == lat) else NAN
            cpus = pick_cpus(t, cfg.get("affinity", "auto"))
            row[f"cpus_t{t}"] = ",".join(map(str, cpus)) if cpus else "unpinned"
            if cfg["rss"]:
                key = (backend, baseline_kind, empty if backend == "tflite" else shape, t, cfg["runs"],
                       cfg["spinning"], str(cfg.get("affinity")), reps)
                if key not in _EMPTY_CACHE:
                    _EMPTY_CACHE[key] = probe(empty, t)
                e = _EMPTY_CACHE[key]
                # primary: OS max RSS of the process (peak_wset / ru_maxrss) minus the empty session's.
                # Includes the model-load transient (graph optimisation holds extra weight copies).
                row[f"peak_rss_mb_t{t}"] = max(r["peak_rss_mb"] - e["peak_rss_mb"], 0.0)
                row[f"peak_rss_total_mb_t{t}"] = r["peak_rss_mb"]
                # secondary: RSS growth sampled every 2 ms after load (steady inference footprint)
                row[f"rss_steady_mb_t{t}"] = max(r["session_peak_rss_mb"] - e["session_peak_rss_mb"], 0.0)
                if reps > 1:  # repeatability: how much this RSS number moves between identical runs
                    row[f"peak_rss_mb_sd_t{t}"] = float(np.hypot(r["_sd"]["peak_rss_mb"], e["_sd"]["peak_rss_mb"]))
        p = primary_thread_count(cfg)
        # peak_rss_total_mb belongs here as much as peak_rss_mb does: it was measured per thread
        # count all along but never promoted, so anything reading the un-suffixed name silently got
        # the empty-session DELTA instead of the raw process peak - and raw process peak with no
        # subtraction is what the SEC 2026 organisers specified.
        for k in ("latency_p50_ms", "latency_p90_ms", "cpu_util_pct", "cpu_ms_per_inf",
                  "peak_rss_mb", "peak_rss_total_mb"):
            row[k] = row.get(f"{k}_t{p}", NAN)
        row["primary_threads"] = p
        # Say it out loud when the machine did not actually deliver the cores we asked for. The
        # previous version warned on other_load_pct, which does not measure that (see above).
        # Memory-bound models at 1 thread sit a little under 1.0 legitimately, so the threshold is
        # deliberately loose and the message says "check", not "discard".
        busiest = max((row.get(f"other_load_pct_t{t}", NAN) for t in cfg["threads"]), default=NAN)
        row["other_load_pct"] = busiest
        # ONLY interpretable as a delivery fraction at or below the physical core count. Above it,
        # spin-wait and the mean/median mismatch between cpu_ms_per_inf and latency_p50_ms push the
        # ratio above 1.0 - measured 2.43 at 8 threads on a 6 P-core machine. Gating on it there
        # would be the same unvalidated-instrument mistake this fix exists to correct.
        try:                       # physical cores in the fastest class; 0 if unknown
            from .cpu_topology import cpu_layout
            _lay = cpu_layout() or []
            _best = max((d["perf"] for d in _lay), default=None)
            phys = sum(1 for d in _lay if d["perf"] == _best) if _best is not None else 0
        except Exception:
            phys = 0
        delivered = [row.get(f"core_delivery_t{t}", NAN) for t in cfg["threads"]
                     if not phys or t <= phys]
        delivered = [d for d in delivered if d == d and d <= 1.5]
        row["core_delivery"] = min(delivered) if delivered else NAN
        floor = float(cfg.get("core_delivery_warn") or 0.0)
        if floor > 0 and delivered and min(delivered) < floor:
            log.warning("only %.0f%% of the requested cores delivered work (threshold %.0f%%): "
                        "either something else was using the machine, or this model is stalling on "
                        "memory. Check the *_sd spread before quoting these numbers.",
                        min(delivered) * 100, floor * 100)
    finally:
        td.cleanup()
    return row


def primary_thread_count(cfg: dict) -> int:
    """Which thread count fills the un-suffixed columns - and it must be one we actually measured.

    `primary_threads: 4` plus a profile that only measures `threads: [1]` (e.g. --quick) used to leave
    every un-suffixed column NaN, which silently emptied score_resource and both latency/RSS Pareto
    fronts. Fall back to the closest measured count instead, and say so: `primary_threads` in the CSV
    always records the count the reported numbers actually came from.
    """
    measured = list(cfg["threads"])
    want = cfg.get("primary_threads") or measured[0]
    if want in measured:
        return want
    got = max([t for t in measured if t <= want], default=min(measured))
    log.warning("primary_threads=%s was not measured (threads=%s); reporting the %s-thread numbers instead",
                want, measured, got)
    return got


def count_params_onnx(path: str | Path) -> int:
    """Weights of Conv/ConvTranspose/Gemm/MatMul (float or quantized initializers)."""
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(path))
    inits = {i.name: i for i in m.graph.initializer}
    # QDQ graphs: the weight reaches the op through DequantizeLinear
    dq = {n.output[0]: n.input[0] for n in m.graph.node if n.op_type == "DequantizeLinear"}
    total = 0
    for n in m.graph.node:
        if n.op_type in ("Conv", "ConvTranspose", "Gemm", "MatMul", "QLinearConv", "ConvInteger", "MatMulInteger"):
            for name in n.input[1:]:
                name = dq.get(name, name)
                if name in inits and len(inits[name].dims) >= 1:
                    total += int(np.prod(numpy_helper.to_array(inits[name]).shape))
    return total


# Names that almost always mean "lower is better". TaskAdapter.evaluate() is contracted to return a
# HIGHER-is-better score, and everything downstream assumes it: score.py eligibility rejects a
# variant whose value fell below the baseline, and pareto.py keeps points with y >= y[i]. Hand back
# a raw error and the whole ranking silently inverts - the best variant is marked ineligible and the
# worst wins. It is a wrong-answer failure, not a crash, so nothing else would catch it, and writing
# `return {"metric_name": "mae", "score": mae}` is the natural way to write an adapter in a hurry.
_LOWER_IS_BETTER_HINTS = ("mae", "mse", "rmse", "error", "err", "loss", "nll", "perplexity",
                          "ppl", "distance", "dist", "wer", "cer", "fid", "latency", "drop")
_warned_metrics: set = set()


def _warn_if_metric_looks_inverted(metric_name: str) -> bool:
    """True when the name suggests the adapter is returning a lower-is-better quantity."""
    name = str(metric_name or "").strip().lower()
    if not name or name in _warned_metrics:
        return False
    tokens = set(re.split(r"[^a-z0-9]+", name)) - {""}
    if not tokens & set(_LOWER_IS_BETTER_HINTS):
        return False
    _warned_metrics.add(name)
    log.warning(
        "metric_name %r looks like a LOWER-is-better quantity, but evaluate() is contracted to "
        "return a HIGHER-is-better score. If that is what it returns, the ranking is inverted: "
        "good variants get excluded by the accuracy constraint and the worst one wins. Return a "
        "negated or reciprocal score (e.g. -mae), or set score.groups.accuracy.higher_is_better "
        "false and mirror it in prune/pareto. If the name is a false alarm, ignore this.", metric_name)
    return True


def measure(adapter, model_path: str | Path, torch_model: nn.Module | None = None, variant: str = "",
            cfg: dict | None = None, extra: dict | None = None, example_input: torch.Tensor | None = None) -> dict:
    """Full metric row for one deployable model file (.onnx or .tflite).

    accuracy is computed on that file (what would be deployed). adapter=None is allowed:
    resources are then measured on `example_input` (random data of the right shape is fine)
    and accuracy stays empty. params/MACs come from the torch model when given (INT8 keeps
    the same count), else from the ONNX graph; TFLite files report neither.
    """
    cfg = {**DEFAULT_MEASURE_CFG, **(cfg or {})}
    model_path = Path(model_path)
    if example_input is None:
        if adapter is None:
            raise ValueError("measure() without an adapter needs example_input")
        example_input = adapter.example_input()
    x = example_input
    row: dict[str, Any] = {"variant": variant or model_path.stem}

    if adapter is not None:
        t0 = time.time()
        ev = adapter.evaluate(str(model_path))
        row["metric_name"] = ev["metric_name"]
        row["accuracy"] = float(ev["score"])
        _warn_if_metric_looks_inverted(row["metric_name"])
        row["eval_s"] = round(time.time() - t0, 2)
        # Any EXTRA keys an adapter returns are reported DETAIL, never the ranking metric. Only
        # `score` drives eligibility and the total score, so an adapter can surface a second view of
        # quality - a looser threshold, a per-object-size breakdown, an identity-consistency measure
        # - to inform a decision without silently changing which variant wins.
        for k, v in ev.items():
            if k in ("metric_name", "score"):
                continue
            if k in row:
                # never let a detail key overwrite a measurement: an adapter returning "accuracy",
                # "params" or "size_bytes" would silently replace the ranking metric or a measured
                # number with its own value
                log.warning("adapter.evaluate() returned %r, which is already a measured column; "
                            "reporting it as eval_%s instead", k, k)
                k = f"eval_{k}"
            row[k] = float(v) if isinstance(v, (int, float, bool)) else v
    else:
        row["metric_name"], row["accuracy"] = "", NAN

    files = [model_path] + sorted(model_path.parent.glob(model_path.name + ".data"))
    row["size_bytes"] = sum(f.stat().st_size for f in files)
    # Compressed size too: magnitude (unstructured) pruning leaves the tensor shapes alone, so its
    # entire benefit is invisible in size_bytes and only shows up once the file is compressed. Which
    # of the two "compression" means for scoring is unknown, so report both.
    row["size_bytes_gz"] = _gz_size(files)
    is_onnx_file = backend_of(model_path) == "onnx"
    row["params"], row["macs"] = NAN, NAN
    if torch_model is not None:
        row["params"] = count_params(torch_model)
        try:
            row["macs"] = count_macs(torch_model, x)
        except Exception as e:  # thop cannot trace some custom modules / TorchScript
            log.warning("thop failed (%s); counting MACs from ONNX", e)
            if is_onnx_file:
                row["macs"] = count_macs_onnx(model_path)
    elif is_onnx_file:
        row["params"] = count_params_onnx(model_path)
        row["macs"] = count_macs_onnx(model_path)

    with tempfile.TemporaryDirectory() as td:
        npy = str(Path(td) / "x.npy")
        np.save(npy, x[:1].detach().cpu().float().numpy())
        row.update(runtime_metrics(model_path, x, cfg, input_npy=npy))
        e, src = measure_energy(model_path, row["variant"], npy, cfg.get("energy"), extra)
    row["energy_j"] = e
    row["energy_source"] = src
    row["model_file"] = str(model_path)
    return row


def _gz_size(files: list[Path]) -> int:
    """Total gzip-compressed size, streamed (never holds a whole model in memory)."""
    import gzip
    import zlib

    total = 0
    for f in files:
        comp, n = zlib.compressobj(6, zlib.DEFLATED, 16 + zlib.MAX_WBITS), 0
        with open(f, "rb") as fh:
            while chunk := fh.read(1 << 20):
                n += len(comp.compress(chunk))
        total += n + len(comp.flush())
    del gzip
    return total


def host_peak_rss_mb(include_children: bool = False) -> float:
    """Lifetime peak RSS of the toolchain's own MAIN process, in MB.

    Every peak_rss_mb in the results is the *model's* inference memory, measured in an isolated
    subprocess. That is the number a judge cares about, but it is not what the toolkit needs to RUN:
    the trainer alone was observed at ~2.3 GB, an order of magnitude above the largest inference
    figure, plus ~180 MB per live dataloader worker. If the on-site sandbox is RAM-limited, that -
    not the optimised model - is the binding constraint, so record it.

    Only the main process, deliberately. `include_children` sums the children's LIFETIME peaks,
    which is not a real concurrent figure: the parent and each worker hit their own peaks at
    different moments, so the sum over-reports (it read 8.5 GB on a run whose observed total was
    ~3.2 GB). A true concurrent peak would need a sampler over the whole process tree; until then
    this column is the main process only, and worker cost has to be added per `num_workers`.
    """
    import psutil

    def _peak(pr):
        mi = pr.memory_info()
        wset = getattr(mi, "peak_wset", None)       # Windows only
        if wset:
            return float(wset)
        try:                                        # Linux/macOS: the kernel's high-water mark
            import resource
            r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            import sys
            return float(r if sys.platform == "darwin" else r * 1024)
        except Exception:
            return float(mi.rss)                    # last resort: current, and say so in the log

    proc = psutil.Process()
    peak = _peak(proc)
    if include_children:      # opt-in, and an over-estimate: see the note above
        for c in proc.children(recursive=True):
            try:
                peak += float(getattr(c.memory_info(), "peak_wset", 0) or c.memory_info().rss)
            except psutil.Error:      # a worker that exited between listing and reading
                pass
    return peak / 2**20


def _torch_env() -> dict:
    """torch details for the run record, or None when torch is absent - which is a legitimate
    configuration (quantize + measure a supplied .onnx), not an error."""
    try:
        import torch
    except ImportError:
        return {"version": None, "device": None, "cuda_device": None}
    cuda = torch.cuda.is_available()
    return {"version": torch.__version__, "device": "cuda" if cuda else "cpu",
            "cuda_device": torch.cuda.get_device_name(0) if cuda else None}


def env_info() -> dict:
    """Where the (relative) numbers were measured."""
    import platform

    import onnxruntime as ort
    import psutil

    return {
        "platform": platform.platform(), "machine": platform.machine(),
        "cpu": platform.processor(), "physical_cores": psutil.cpu_count(logical=False),
        "core_classes": describe_cpus(),
        "logical_cores": psutil.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "python": platform.python_version(), "torch": _torch_env()["version"], "onnxruntime": ort.__version__,
        # Inference is always measured on the CPU (isolated ORT subprocess), but PyTorch training and
        # fine-tuning default to CUDA when a GPU is present. Record it: on a CPU-only box the same run
        # produces the same metrics but takes far longer, and `--device cpu` reproduces it exactly.
        "torch_train_device": _torch_env()["device"],
        "cuda_device": _torch_env()["cuda_device"],
        "note": "latency/RAM/CPU measured on this machine's CPU: relative comparisons only",
    }


def format_table(rows: list[dict], cols: list[str] | None = None) -> str:
    cols = cols or ["variant", "metric_name", "accuracy", "size_bytes", "params", "macs",
                    "latency_p50_ms", "latency_p90_ms", "peak_rss_mb", "cpu_util_pct", "energy_j"]
    extra = sorted({k for r in rows for k in r if k.startswith("latency_p50_ms_t")})
    cols = cols + [c for c in extra if c not in cols]

    def fmt(v):
        if isinstance(v, float):
            if math.isnan(v):
                return "-"
            return f"{v:.4f}" if abs(v) < 10 else f"{v:,.1f}"
        if isinstance(v, int):
            return f"{v:,}"
        return str(v)

    table = [[fmt(r.get(c, "")) for c in cols] for r in rows]
    widths = [max(len(c), *(len(t[i]) for t in table)) for i, c in enumerate(cols)]
    line = " | ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "-|-".join("-" * w for w in widths)
    body = "\n".join(" | ".join(v.ljust(w) for v, w in zip(t, widths)) for t in table)
    return f"{line}\n{sep}\n{body}"
