"""INT8 quantization by wrapping ONNX Runtime's quantizer (no custom quantization algorithms here).

static : QDQ INT8, per-channel weights, activations calibrated on the adapter's "calib" loader.
dynamic: weights INT8, activations quantized at run time. With op_types_to_quantize [MatMul, Gemm]
         it is the robust choice for attention / SiLU-heavy models whose conv activations are
         channel-imbalanced (MobileViT: 0.9161 vs 0.668 for plain static INT8, FP32 0.9165).

Which nodes stay FP32 (union of):
  keep_fp32            node-name substrings / "re:" regexes. ONNX node names come from torch module
                       paths (e.g. "features.0" matches "/features/features.0/features.0.0/Conv").
                       Exclude CONTIGUOUS blocks (module prefixes): excluding a single op or op type
                       usually changes nothing, because its neighbours still quantize the shared tensors.
  keep_fp32_op_types   whole op types (same caveat).
  auto_keep_fp32       "rules" (default) | "search" | "none":
    R1 output tail     everything between the graph outputs and the last weighted op (Conv/Gemm/MatMul
                       with >= tail_min_params weights): post-processing math (box decode, concat of
                       quantities with different units, softmax) has ~no MACs, so FP32 costs ~nothing.
                       Fixes e.g. a detector output concat([boxes in pixels, scores in 0..1]) sharing
                       one INT8 scale (scale 1.53 -> every score < 0.77 became 0; mAP 0.407 -> 0.161).
    R2 concat ranges   WARNS about Concats whose calibrated input magnitudes differ by > concat_range_ratio
                       and that are still quantized (excluding a single Concat would not help in QDQ mode:
                       its inputs are quantized by their producers and its output by its consumers).
                       Fix such a case with a block in keep_fp32 or with auto_keep_fp32: search.
    search             R1+R2, then a label-free greedy search: blocks (module prefixes) ranked by how
                       much quantizing only that block hurts the output SQNR are kept FP32 until the
                       mean per-channel output SQNR reaches search_target_db (or search_max_blocks).

Calibration: ranges are computed once, batch by batch (bounded memory), cached, and reused for every
exclusion set. "entropy" uses 2048 histogram bins (ORT's quantize_static default of 128/128 bins leaves a
single threshold candidate, i.e. silently equals minmax). "percentile" uses calib_percentile.

Every run reports int8_sqnr_db: mean per-output-channel SQNR of the quantized vs FP32 model on
calibration data (label-free; < ~15 dB means the INT8 model is badly off) and int8_quantized_nodes,
which is 0 when nothing was actually quantized (then SQNR is meaninglessly perfect).

Two things SQNR cannot see. (1) Nothing quantized -> a perfect 60 dB, which is why int8_quantized_nodes
exists. (2) A QAT model (core/qat.py) is TRAINED to tolerate quantization error, so its outputs move a lot
while its decisions do not: MobileViT after QAT measured 6.7 dB yet 0.872 top-1, against 3.0 dB / 0.668 for
the same PTQ recipe without QAT. Read SQNR as "how far the outputs moved", not as accuracy, and always
confirm a QAT variant with the task metric.
"""
from __future__ import annotations

import logging
import re
import shutil
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from onnxruntime.quantization import CalibrationDataReader

log = logging.getLogger(__name__)

WEIGHTED_OPS = {"Conv", "ConvTranspose", "Gemm", "MatMul"}

DEFAULT_QUANT_CFG = {
    "method": "static",
    "per_channel": True,
    "activation_type": "uint8",  # U8S8: fast path on x86 (S8S8 QDQ was slower than FP32 there); fine on ARM
    "weight_type": "int8",
    "calibrate_method": "minmax",  # minmax | entropy | percentile | auto (measure, see _auto_calibrate)
    "calib_percentile": 99.99,
    # calibrate_method: auto - candidates tried in order, cheapest/most common first
    "calib_auto_methods": [{"calibrate_method": "minmax"},
                           {"calibrate_method": "percentile", "calib_percentile": 99.99},
                           {"calibrate_method": "percentile", "calib_percentile": 99.9},
                           {"calibrate_method": "entropy"}],
    "calib_auto_target_db": 15.0,  # stop at the first candidate this healthy (per-channel min SQNR);
                                   # same number as sqnr_warn_db - "not in the broken zone"
    # Break near-ties in min-SQNR on the mean. Default 0 = pure minimum-ranking. A 1.0 dB margin was
    # tried and reverted: it was calibrated on a stale measurement, and re-running on held-out data
    # showed the candidate the mean preferred scored WORSE than the one the minimum preferred
    # (0.2988 vs 0.3010 on the run's own metric). Raise it only with fresh evidence for that model.
    "calib_auto_tie_db": 0.0,
    "calib_batches": 16,
    "keep_fp32": [],
    "keep_fp32_op_types": [],
    "auto_keep_fp32": "rules",     # none | rules | search
    "tail_min_params": 1024,
    "concat_range_ratio": 32.0,
    "search_target_db": 20.0,
    "search_max_blocks": 6,
    "sqnr_batches": 4,
    "sqnr_warn_db": 15.0,         # below this the INT8 model is probably broken; 0 disables the warning
    "quant_format": "qdq",
    "preprocess": True,
    # Keep weights in a side .onnx.data file instead of inside the protobuf. Cuts the peak memory of
    # quantization (which was measured at ~26x the fp32 weight size) and is the only way past ONNX's
    # 2 GB protobuf limit. measure.py already sums the .data file into size_bytes.
    # ONNX's protobuf cannot exceed 2 GB, so a model above that MUST use this. It is not a memory
    # optimisation: measured on a 97 MB model it changed the peak by 2 MB and doubled the output
    # file (49.0 vs 24.6 MB). Treat it as a correctness escape hatch for very large models.
    "use_external_data_format": False,
    "ram_warn_factor": 28.0,      # measured peak / fp32 weight bytes; 0 disables the pre-flight check
    "reduce_range": False,
    "op_types_to_quantize": None,  # e.g. ["Conv", "Gemm"]: quantize only these op types
    "extra_options": None,         # passed through to ORT (e.g. {"ActivationSymmetric": True})
}


# ----------------------------------------------------------------------------- calibration data
def _batch_input(batch) -> np.ndarray:
    x = batch[0] if isinstance(batch, (list, tuple)) else batch
    x = x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    return np.ascontiguousarray(x, dtype=np.float32)


class LoaderCalibrationReader(CalibrationDataReader):
    """Feeds the first tensor of each calib batch to ORT's calibrator.

    Adapts to foreign ONNX models: NCHW/NHWC layout and a fixed batch size (split per sample).
    """

    def __init__(self, loader, input_name: str, max_batches: int, input_shape: list | None = None):
        self.loader, self.input_name, self.max_batches = loader, input_name, max_batches
        self.shape = input_shape or []
        self._it = None

    def get_next(self):
        if self._it is None:
            self._it = self._gen()
        return next(self._it, None)

    def _gen(self):
        from .runtime import adapt_layout, fixed_batch

        for i, batch in enumerate(self.loader):
            if i >= self.max_batches:
                return
            x = _batch_input(batch)
            if self.shape:
                x = adapt_layout(x, self.shape)
            fb = fixed_batch(self.shape)
            if fb and x.shape[0] != fb:
                for j in range(0, x.shape[0] - fb + 1, fb):
                    yield {self.input_name: x[j:j + fb]}
            else:
                yield {self.input_name: x}

    def rewind(self):
        self._it = None


class _OneShotReader(CalibrationDataReader):
    def __init__(self, feed):
        self.feed = feed

    def get_next(self):
        feed, self.feed = self.feed, None
        return feed


def _input_info(path):
    import onnx

    inp = onnx.load(str(path)).graph.input[0]
    return inp.name, [d.dim_value if d.HasField("dim_value") else None for d in inp.type.tensor_type.shape.dim]


def calibrate(model_path, loader, cfg: dict, cache_path) -> dict:
    """Compute activation ranges batch by batch, save them as an ORT calibration cache, return {tensor: (lo, hi)}."""
    from onnxruntime.quantization.calibrate import create_calibrator, save_tensors_data

    method = cfg["calibrate_method"]
    extra = {}
    if method == "entropy":
        extra = {"num_bins": 2048, "num_quantized_bins": 128}
    elif method == "percentile":
        extra = {"num_bins": 2048, "percentile": float(cfg["calib_percentile"])}
    # ORT's Calib* extra_options normally reach the calibrator through quantize_static; we calibrate
    # ourselves, so map them here too.
    for opt, key in (("CalibTensorRangeSymmetric", "symmetric"), ("CalibMovingAverage", "moving_average"),
                     ("CalibMovingAverageConstant", "averaging_constant"),
                     ("CalibMaxIntermediateOutputs", "max_intermediate_outputs"),
                     ("CalibPercentile", "percentile"), ("CalibNumBins", "num_bins")):
        if opt in (cfg.get("extra_options") or {}):
            extra[key] = cfg["extra_options"][opt]
    with tempfile.TemporaryDirectory(prefix="calib.") as td:
        cal = create_calibrator(Path(model_path), cfg["op_types_to_quantize"],
                                augmented_model_path=str(Path(td) / "augmented.onnx"),
                                calibrate_method=_METHODS()[method], extra_options=extra)
        name, shape = _input_info(model_path)
        reader = LoaderCalibrationReader(loader, name, cfg["calib_batches"], shape)
        n = 0
        while (feed := reader.get_next()) is not None:  # one batch at a time: bounded memory for histograms
            cal.collect_data(_OneShotReader(feed))
            n += 1
        if n == 0:
            raise ValueError("calibration loader yielded no data")
        data = cal.compute_data()
        del cal
    save_tensors_data(data, cache_path)
    return tensor_ranges(data)


def _METHODS():
    from onnxruntime.quantization import CalibrationMethod

    return {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
            "percentile": CalibrationMethod.Percentile}


def tensor_ranges(tensors_data) -> dict:
    out = {}
    for k, v in tensors_data.data.items():
        r = getattr(v, "range_value", None)
        if r is None and isinstance(v, tuple):
            r = v
        if r is not None:
            out[k] = (float(np.asarray(r[0]).min()), float(np.asarray(r[1]).max()))
    return out


# ----------------------------------------------------------------------------- graph rules
def match_nodes(onnx_path: str | Path, patterns: list[str], op_types: list[str] | None = None) -> list[str]:
    import onnx

    names = []
    for n in onnx.load(str(onnx_path)).graph.node:
        hit = any((re.search(p[3:], n.name) if p.startswith("re:") else p in n.name) for p in patterns)
        if hit or (op_types and n.op_type in op_types):
            names.append(n.name)
    return names


def _graph(path):
    import onnx

    g = onnx.load(str(path)).graph
    return g, {i.name: i for i in g.initializer}, {o: n for n in g.node for o in n.output}


def _weight_params(node, inits) -> int:
    if node.op_type not in WEIGHTED_OPS or len(node.input) < 2 or node.input[1] not in inits:
        return 0
    return int(np.prod(inits[node.input[1]].dims))


def output_tail(path, min_params: int = 1024) -> list[str]:
    """R1: nodes reachable backwards from the graph outputs before hitting a weighted op with >= min_params
    weights. Shape/Size inputs are not followed (metadata-only edges).

    Returns [] when the walk would cover every weighted op (a small model whose convs are all below
    min_params): excluding everything would leave an unquantized model that still looks like INT8.
    Unnamed nodes are skipped - ORT matches exclusions by name, and "" would match all of them.
    """
    g, inits, producer = _graph(path)
    seen, stack, tail = set(), [o.name for o in g.output], []
    unnamed = 0
    while stack:
        n = producer.get(stack.pop())
        if n is None or id(n) in seen:
            continue
        seen.add(id(n))
        if _weight_params(n, inits) >= min_params:
            continue
        if n.name:
            tail.append(n.name)
        else:
            unnamed += 1
        if n.op_type not in ("Shape", "Size"):
            stack.extend(i for i in n.input if i)
    if unnamed:
        log.warning("output tail: %d unnamed node(s) cannot be excluded by name; leaving them quantized", unnamed)
    weighted = [n for n in g.node if n.op_type in WEIGHTED_OPS and _weight_params(n, inits) > 0]
    if weighted and all(n.name in set(tail) for n in weighted):
        log.warning("output tail would cover every weighted op (tail_min_params=%d): rule disabled for this "
                    "model, nothing would be quantized", min_params)
        return []
    return tail


def concat_range_mismatch(path, ranges: dict, ratio: float = 32.0) -> list[str]:
    """R2: Concat nodes whose non-constant inputs' calibrated magnitudes differ by more than `ratio`."""
    g, inits, _ = _graph(path)
    consts = set(inits) | {n.output[0] for n in g.node if n.op_type == "Constant"}
    hits = []
    for n in g.node:
        if n.op_type != "Concat":
            continue
        mags = [max(abs(ranges[t][0]), abs(ranges[t][1])) for t in n.input if t not in consts and t in ranges]
        if len(mags) >= 2 and min(mags) > 0 and max(mags) / min(mags) > ratio:
            hits.append(n.name)
    return hits


def module_blocks(path, min_groups: int = 16, max_groups: int = 64, max_depth: int = 5) -> dict:
    """Group nodes by module-path prefix; the shallowest depth with >= min_groups groups holding a weighted op.

    Keeps the best non-empty grouping found: deeper levels return {} once the node names run out, and a
    model with few modules must still get its (coarser) blocks instead of no search at all.
    """
    g, _, _ = _graph(path)
    best: dict = {}
    for depth in range(1, max_depth + 1):
        grp = defaultdict(list)
        for n in g.node:
            parts = n.name.strip("/").split("/")
            if len(parts) > depth and n.name:
                grp["/" + "/".join(parts[:depth]) + "/"].append(n)
        grp = {k: [x.name for x in v] for k, v in grp.items() if any(x.op_type in WEIGHTED_OPS for x in v)}
        if not grp or len(grp) > max_groups:
            break
        best = grp
        if len(grp) >= min_groups:
            break
    return best


# ----------------------------------------------------------------------------- output SQNR (label-free)
class SqnrProbe:
    """Mean per-output-channel SQNR (dB) of a candidate model against the FP32 reference on calibration data.

    Per-channel (axis 1) so that a small-valued channel group (e.g. class scores next to pixel boxes) cannot
    hide behind a large one. Channels with ~zero reference energy are ignored; values clipped to [-20, 60].
    """

    def __init__(self, ref_path, loader, n_batches: int = 4):
        from .runtime import OnnxPredictor

        self.inputs = [_batch_input(b) for _, b in zip(range(n_batches), loader)]
        ref = OnnxPredictor(str(ref_path))
        self.ref = [self._as_list(ref(x)) for x in self.inputs]

    @staticmethod
    def _as_list(y):
        return y if isinstance(y, list) else [y]

    def score(self, path) -> dict:
        from .runtime import OnnxPredictor

        pred = OnnxPredictor(str(path))
        sig, err = defaultdict(float), defaultdict(float)
        for x, refs in zip(self.inputs, self.ref):
            for k, (r, q) in enumerate(zip(refs, self._as_list(pred(x)))):
                r, q = np.asarray(r, np.float64), np.asarray(q, np.float64)
                if r.ndim < 2:
                    r, q = r[:, None], q[:, None]
                axes = tuple(i for i in range(r.ndim) if i != 1)
                s, e = (r ** 2).sum(axes), ((r - q) ** 2).sum(axes)
                for c in range(len(s)):
                    sig[(k, c)] += float(s[c])
                    err[(k, c)] += float(e[c])
        if not sig:
            return {"sqnr_db": float("nan"), "sqnr_min_db": float("nan"), "sqnr_global_db": float("nan")}
        top = max(sig.values())
        per = [float(np.clip(10 * np.log10(sig[c] / max(err[c], 1e-30)), -20, 60))
               for c in sig if sig[c] > 1e-9 * top]
        if not per:  # reference output is identically zero on this data: nothing to compare
            return {"sqnr_db": float("nan"), "sqnr_min_db": float("nan"), "sqnr_global_db": float("nan")}
        glob = 10 * np.log10(sum(sig.values()) / max(sum(err.values()), 1e-30))
        return {"sqnr_db": float(np.mean(per)), "sqnr_min_db": float(np.min(per)),
                "sqnr_global_db": float(np.clip(glob, -20, 60))}


# ----------------------------------------------------------------------------- quantization
def _qtypes():
    from onnxruntime.quantization import QuantType

    return {"int8": QuantType.QInt8, "uint8": QuantType.QUInt8}


def _quantize_static_cached(src, out, cache, cfg, exclude) -> Path:
    from onnxruntime.quantization import QuantFormat, quantize_static

    qt = _qtypes()
    quantize_static(str(src), str(out), None,
                    quant_format=QuantFormat.QDQ if cfg["quant_format"] == "qdq" else QuantFormat.QOperator,
                    per_channel=cfg["per_channel"], activation_type=qt[cfg["activation_type"]],
                    weight_type=qt[cfg["weight_type"]], calibrate_method=_METHODS()[cfg["calibrate_method"]],
                    nodes_to_exclude=sorted(exclude), reduce_range=cfg["reduce_range"],
                    op_types_to_quantize=cfg["op_types_to_quantize"], extra_options=cfg["extra_options"],
                    use_external_data_format=bool(cfg.get("use_external_data_format", False)),
                    calibration_cache_path=str(cache))
    return Path(out)


def sensitivity_search(src, cache, cfg, probe: SqnrProbe, base: set, workdir) -> tuple[set, dict]:
    """Greedy block search (label-free): see module docstring."""
    t0 = time.time()
    g, _, _ = _graph(src)
    all_nodes = {n.name for n in g.node}
    blocks = module_blocks(src)
    steps = []

    def sqnr(exclude, tag):
        p = _quantize_static_cached(src, Path(workdir) / "search.onnx", cache, cfg, exclude)
        s = probe.score(p)["sqnr_db"]
        steps.append((tag, round(s, 2), len(exclude)))
        return s

    target = float(cfg["search_target_db"])
    cur = sqnr(base, "base")
    keep, kept_blocks = set(base), []
    if cur < target and blocks:
        # quantize ONLY block b (everything else FP32): low SQNR = the block's own INT8 error is large
        sens = sorted((sqnr(base | (all_nodes - set(nodes)), f"only {k}"), k) for k, nodes in blocks.items())
        for s_only, k in sens[: int(cfg["search_max_blocks"])]:
            cand = keep | set(blocks[k])
            s = sqnr(cand, f"+{k}")
            if s > cur:
                keep, cur = cand, s
                kept_blocks.append(k)   # only blocks actually kept in FP32, so the result is reproducible
            if cur >= target:
                break
    report = {"search_final_db": round(cur, 2), "search_blocks": len(blocks), "search_evals": len(steps),
              "search_s": round(time.time() - t0, 1), "search_kept": kept_blocks}
    log.info("sensitivity search: %s", report)
    return keep, report


def ram_preflight(fp32_path, cfg: dict) -> dict:
    """Warn BEFORE quantizing if this machine probably cannot hold it.

    Quantization peaks at a large multiple of the model's weight bytes - measured here 28x on a
    97 MB model (2.7 GB) and 26x on a 338 MB one (8.9 GB), while a 1.2 GB model drove a 16 GB
    machine to zero free memory. The multiplier lives inside ONNX Runtime's implementation, and the
    knobs that look like they should help do not: external data format changed the peak by 2 MB,
    skipping quant_pre_process changed nothing, and calibration batches 16 -> 2 changed nothing.

    So the useful thing is to predict it and say what to turn down, instead of dying half way
    through. Returns the estimate; set quantize.ram_warn_factor to 0 to skip.
    """
    import psutil

    factor = float(cfg.get("ram_warn_factor") or 0.0)
    fp32_path = Path(fp32_path)
    weights = fp32_path.stat().st_size + sum(
        f.stat().st_size for f in fp32_path.parent.glob(fp32_path.name + ".data"))
    est = weights * factor
    avail = psutil.virtual_memory().available
    info = {"model_bytes": weights, "est_peak_bytes": int(est), "available_bytes": int(avail)}
    if factor <= 0:
        return info
    if est > avail:
        log.warning("quantizing %s may need ~%.1f GB (%.0f MB of weights x %.0fx measured) but only "
                    "%.1f GB is free. If it is killed: drop adapter_args.batch_size / num_workers, "
                    "close other processes, or quantize on a bigger machine. A model over 2 GB also "
                    "needs quantize.use_external_data_format: true.",
                    fp32_path.name, est / 2**30, weights / 2**20, factor, avail / 2**30)
    elif est > 0.6 * avail:
        log.info("quantizing %s: ~%.1f GB estimated peak against %.1f GB free",
                 fp32_path.name, est / 2**30, avail / 2**30)
    return info


def _auto_calibrate(fp32_path, out_path, calib_loader, cfg: dict, info: dict) -> Path:
    """Pick the calibration method by measuring it, using the label-free SQNR probe.

    `minmax` puts the activation outliers inside the quantization range, which on an outlier-heavy
    model crushes every other channel toward zero; a percentile range clips them instead. Pruning
    makes outliers worse, so one model can need different calibrators before and after it. The swing
    is worth tens of accuracy points, it is not predictable from the architecture, and on site there
    are no labels to check against - so measure instead of guessing. (Measured before/after numbers:
    see "Calibration" in README.md.)

    Only ORT's own calibrators are used - this chooses between them, it does not implement one. The
    candidates are tried in order and the search stops at the first one healthier than
    `calib_auto_target_db`, so a well-behaved model pays for exactly one extra SQNR probe.
    """
    cands = list(cfg["calib_auto_methods"]) or [{"calibrate_method": "minmax"}]
    for over in cands:
        m = over.get("calibrate_method")
        if m not in ("minmax", "entropy", "percentile"):
            raise ValueError(f"calib_auto_methods entry {over!r} needs an explicit calibrate_method of "
                             "minmax | entropy | percentile")
    target = float(cfg["calib_auto_target_db"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if calib_loader is None or int(cfg["sqnr_batches"]) <= 0:
        # auto ranks candidates by the SQNR probe; without it there is nothing to rank on, so take the
        # first candidate rather than silently running all of them and keeping an arbitrary one
        first = {**cfg, **cands[0]}
        log.warning("calibrate_method: auto needs the SQNR probe (a calibration loader and "
                    "sqnr_batches > 0); falling back to %s", _method_label(cands[0]))
        quantize_onnx(fp32_path, out_path, calib_loader, first, info=info)
        info["calib_method_chosen"] = _method_label(cands[0])
        return out_path
    scored, tried = [], []
    with tempfile.TemporaryDirectory() as td:
        for i, over in enumerate(cands):
            # every candidate names a concrete method (validated above), so cfg's "auto" is never
            # inherited and this cannot recurse. Candidates are also spared the per-variant "looks
            # broken" warning - it is emitted once, for the winner.
            sub = {**cfg, **over, "sqnr_warn_db": 0}
            cand_info = {}
            cand_path = Path(td) / f"cand{i}.onnx"
            try:
                quantize_onnx(fp32_path, cand_path, calib_loader, sub, info=cand_info)
            except Exception as e:
                log.warning("calibrate_method auto: %s failed (%s)", _method_label(over), e)
                tried.append(f"{_method_label(over)}=failed({type(e).__name__})")
                continue
            lo = _finite(cand_info.get("int8_sqnr_min_db"))
            mean = _finite(cand_info.get("int8_sqnr_db"))
            scored.append((lo, mean, over, cand_info, cand_path))
            tried.append(f"{_method_label(over)}={lo:.1f}dB" if lo > float("-inf")
                         else f"{_method_label(over)}=no-probe")
            log.info("calibrate_method auto: %s -> %.1f dB (mean %.1f dB)", _method_label(over), lo, mean)
            if lo >= target:
                break
        if not scored:
            raise RuntimeError("calibrate_method: auto - every candidate failed; see the warnings above")
        # Rank on the per-channel MINIMUM: one destroyed channel group is the failure that matters and
        # it is invisible in the mean. But a fraction of a dB in that minimum is noise, so break near
        # ties on the mean - measured: on a detector the two best candidates sat 0.1 dB apart in the
        # minimum while their means differed by 0.5 dB, and the one the mean preferred was worth
        # +1.5 mAP. Selection happens here, inside the temp dir, so every candidate file is still
        # available to copy from.
        best_lo = max(s[0] for s in scored)
        pool = [s for s in scored if s[0] >= best_lo - float(cfg.get("calib_auto_tie_db") or 0.0)]
        db, mean, over, cand_info, cand_path = max(pool, key=lambda s: s[1])
        shutil.copyfile(cand_path, out_path)                    # keep the winner, not just its score
    info.update(cand_info)
    info["calib_method_chosen"] = _method_label(over)
    info["calib_auto_tried"] = tried      # a flat list of strings: it becomes one CSV cell
    if len(tried) > 1:
        log.info("calibrate_method auto: chose %s (%.1f dB) out of %d candidate(s)",
                 info["calib_method_chosen"], db, len(tried))
    warn_db = float(cfg.get("sqnr_warn_db") or 0.0)
    if warn_db > 0 and db < warn_db:      # the best available method is still not healthy: say so once
        info["int8_sqnr_warning"] = f"SQNR {db:.1f} dB < {warn_db:.0f} dB threshold (best of {len(tried)})"
        log.warning("%s: even the best calibration (%s) looks broken (%.1f dB < %.0f dB). Check accuracy "
                    "before shipping; consider auto_keep_fp32: search.",
                    out_path.name, info["calib_method_chosen"], db, warn_db)
    return out_path


def _finite(v) -> float:
    """A missing or NaN SQNR (no probe / all-zero reference) must sort last, not compare unpredictably."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("-inf")
    return f if f == f else float("-inf")


def _method_label(over: dict) -> str:
    m = over.get("calibrate_method", "minmax")
    return f"percentile {over['calib_percentile']:g}" if m == "percentile" and "calib_percentile" in over else m


def quantize_onnx(fp32_path: str | Path, out_path: str | Path, calib_loader=None, cfg: dict | None = None,
                  info: dict | None = None) -> Path:
    """Quantize an FP32 ONNX file. `info` (optional dict) receives: fp32_nodes, auto rule hits, int8_sqnr_db."""
    from onnxruntime.quantization import quantize_dynamic

    cfg = {**DEFAULT_QUANT_CFG, **(cfg or {})}
    if cfg["calibrate_method"] not in ("minmax", "entropy", "percentile", "auto"):
        raise ValueError(f"unknown calibrate_method {cfg['calibrate_method']!r}")
    info = info if info is not None else {}
    # once per user-facing call: _auto_calibrate recurses per candidate, and re-running the check
    # there both spams the log and overwrites the caller's available_bytes with a mid-run reading
    if not cfg.get("_ram_preflight_done"):
        info.update(ram_preflight(fp32_path, cfg))
    if cfg["calibrate_method"] == "auto" and cfg["method"] == "static":
        return _auto_calibrate(fp32_path, out_path, calib_loader, {**cfg, "_ram_preflight_done": True},
                               info)
    fp32_path, out_path = Path(fp32_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        src = fp32_path
        if cfg["preprocess"]:  # shape inference + graph optimisation recommended by ORT before quantizing
            try:
                from onnxruntime.quantization.shape_inference import quant_pre_process

                pre = Path(td) / "pre.onnx"
                quant_pre_process(str(fp32_path), str(pre), skip_symbolic_shape=True,
                                  save_as_external_data=bool(cfg.get("use_external_data_format", False)))
                src = pre
            except Exception as e:
                log.warning("quant_pre_process failed (%s); quantizing raw export", e)

        exclude = set(match_nodes(src, cfg["keep_fp32"], cfg["keep_fp32_op_types"]))
        auto = cfg["auto_keep_fp32"] or "none"
        if auto not in ("none", "rules", "search"):
            raise ValueError(f"unknown auto_keep_fp32 {auto!r}")
        if auto != "none":
            tail = output_tail(src, int(cfg["tail_min_params"]))
            exclude |= set(tail)
            info["fp32_tail_nodes"] = len(tail)

        if cfg["method"] == "dynamic":
            quantize_dynamic(str(src), str(out_path), per_channel=cfg["per_channel"],
                             use_external_data_format=bool(cfg.get("use_external_data_format", False)),
                             weight_type=_qtypes()[cfg["weight_type"]], nodes_to_exclude=sorted(exclude),
                             reduce_range=cfg["reduce_range"], op_types_to_quantize=cfg["op_types_to_quantize"],
                             extra_options=cfg["extra_options"])
        elif cfg["method"] == "static":
            if calib_loader is None:
                raise ValueError("static quantization needs a calibration loader")
            cache = Path(td) / "calib.json"
            ranges = calibrate(src, calib_loader, cfg, cache)
            if auto != "none":
                # R2 is a WARNING, not an exclusion: excluding a single Concat changes nothing in QDQ mode
                # (its inputs are quantized by their producers and its output by its consumers). A
                # mid-graph mismatch needs a block in keep_fp32, or auto_keep_fp32: search.
                mism = [n for n in concat_range_mismatch(src, ranges, float(cfg["concat_range_ratio"]))
                        if n not in exclude]
                info["concat_range_flags"] = len(mism)
                if mism:
                    log.warning("%d Concat node(s) mix inputs whose ranges differ by >%gx and are still "
                                "quantized: %s", len(mism), cfg["concat_range_ratio"], mism[:3])
            if auto == "search":
                probe = SqnrProbe(fp32_path, calib_loader, int(cfg["sqnr_batches"]))
                exclude, rep = sensitivity_search(src, cache, cfg, probe, exclude, td)
                info.update(rep)
            _quantize_static_cached(src, out_path, cache, cfg, exclude)
        else:
            raise ValueError(f"unknown quantization method {cfg['method']!r}")
    from .export import stamp_onnx

    stamp_onnx(out_path)   # ORT rewrites the graph: re-attach authorship to the quantized file
    info["fp32_nodes"] = len(exclude)
    import onnx

    info["int8_quantized_nodes"] = sum(1 for n in onnx.load(str(out_path)).graph.node
                                       if n.op_type in ("QuantizeLinear", "QLinearConv", "QLinearMatMul",
                                                        "ConvInteger", "MatMulInteger", "DynamicQuantizeLinear"))
    if info["int8_quantized_nodes"] == 0:
        log.warning("%s contains no quantized op: this variant is effectively FP32", out_path.name)
    if exclude:
        log.info("kept %d nodes in FP32", len(exclude))
    if calib_loader is not None and int(cfg["sqnr_batches"]) > 0:
        try:
            info.update({f"int8_{k}": v for k, v in
                         SqnrProbe(fp32_path, calib_loader, int(cfg["sqnr_batches"])).score(out_path).items()})
        except Exception as e:  # diagnostics must never fail a quantization
            log.warning("SQNR diagnostic failed: %s", e)
        # The number alone is easy to miss in a wide table, so say it out loud. Per-channel min, not the
        # mean: one destroyed channel group (e.g. box coords sharing a scale with 0-1 scores) is exactly
        # the failure this check exists to catch, and the mean hides it.
        warn_db = float(cfg.get("sqnr_warn_db") or 0.0)
        worst = min((info.get(k, float("inf")) for k in ("int8_sqnr_db", "int8_sqnr_min_db")), default=None)
        if warn_db > 0 and worst is not None and worst == worst and worst < warn_db:
            info["int8_sqnr_warning"] = (f"SQNR {worst:.1f} dB < {warn_db:.0f} dB threshold")
            log.warning("%s: quantization looks broken (worst per-channel SQNR %.1f dB < %.0f dB). "
                        "Check accuracy before shipping this variant; see quantize.output_tail / "
                        "sensitivity_search.", out_path.name, worst, warn_db)
    log.info("quantized %s -> %s (%.1f KB -> %.1f KB, sqnr %.1f dB)", fp32_path.name, out_path.name,
             fp32_path.stat().st_size / 1024, out_path.stat().st_size / 1024, info.get("int8_sqnr_db", float("nan")))
    return out_path

