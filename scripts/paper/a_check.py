"""Experiment A, section 2: the two frozen checkers and the flag-to-prediction step of section 2.3.

The specification is `results/paper/a_design_registered.md` (REGISTERED 2026-09-23T01:35+08:00).
This file implements that section and adds no rule of its own: where this script and the
registration disagree, the registration is right and this script is wrong.

  SC-B (section 2.1, PRIMARY)   reads one .onnx file and nothing else: pins the batch dimension of
                                the single input to 1, runs onnx shape inference, keeps every
                                Transpose with a fully static input shape that executes in 8 bits
                                (a DequantizeLinear -> Transpose -> QuantizeLinear sandwich, or an
                                input tensor that is already INT8/UINT8), and applies
                                predict(dims, perm, 1, 4, version) for 1.23.2 and 1.24.4.
  SC-A (section 2.2, SECONDARY) asks the same question about the graph ONNX Runtime actually runs:
                                ORT 1.24.4 dumps its optimized graph, a second 1.24.4 session runs
                                3 profiled iterations at 1 intra-op thread, the profile's
                                `args.input_type_shape` gives each executed Transpose its dtype and
                                shape, the dump gives its perm, the two are joined BY NODE NAME and
                                predict() is applied with the element size of the executed dtype.

The dispatch rule is NOT re-expressed here. `predict()`, `_single_axis()`, `_is_reshape()` and
`MLAS_BYTES` are imported from `scripts/paper/transpose_probe.py`, and even the reported
loops / M / N / moved-block width are parsed back out of the path string `predict()` returns, so
this script cannot drift from the rule the Experiment 27/27b predictions were made with.

NOTHING HERE IS TIMED. SC-A's profiling exists only to learn the executed dtype and shape; the
profile's durations are read by nothing in this file (registration section 2.2).

    # PRIMARY checker, file only
    .venv/Scripts/python.exe scripts/paper/a_check.py sc-b MODEL.onnx [MODEL.onnx ...] [--json OUT]

    # SECONDARY checker, executed graph (run from the project venv; it calls .envs/ort1244 itself)
    .venv/Scripts/python.exe scripts/paper/a_check.py sc-a MODEL.onnx ... --work-dir DIR [--json OUT]

    # both checkers plus section 2.3; screen_ms is READ from the runner's manifest, never measured
    .venv/Scripts/python.exe scripts/paper/a_check.py predict MODEL.onnx ... --work-dir DIR \
        --manifest results/expA/manifest.json [--json results/expA/predictions.json]

    # artefact 4: both checkers on the three local controls against the recorded ground truth
    .venv/Scripts/python.exe scripts/paper/a_check.py selftest --work-dir DIR \
        [--out results/expA/selftest.json]

`--work-dir` is always explicit and always outside the project: SC-A writes optimized-graph dumps,
profile JSONs and its child script there, and this script never deletes it.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import re
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import onnx
from onnx import TensorProto, helper, shape_inference

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transpose_probe import MLAS_BYTES, _is_reshape, _single_axis, predict  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "expA"
GROUND_TRUTH = ROOT / "results" / "transpose_probe" / "manifest.json"
REGISTRATION = ROOT / "results" / "paper" / "a_design_registered.md"
ORT1244_PYTHON = ROOT / ".envs" / "ort1244" / "Scripts" / "python.exe"

# ---- registered constants (sections 2.1, 2.2, 2.3) --------------------------------------------
OLD, NEW = "1.23.2", "1.24.4"
THREADS = 4                  # the thread count the prediction is made at (section 2.1 step 4)
ESIZE_8BIT = 1               # SC-B applies predict() with an 8-bit element by construction
DISPATCH_US = 0.52           # Exp 27b's reported per-dispatch cost, 0.51-0.53 us (section 2.3)
DISPATCH_MS = DISPATCH_US / 1000.0
FLOOR_REL = 0.07             # the study's between-session floor; the PREDICT-STEP threshold on rho
MAGNITUDE_BAND = (0.5, 2.0)  # REPORTED-ONLY band on measured_delta_ms / A_ms (section 2.3)
PROFILE_ITERATIONS = 3       # section 2.2 step 2
PROFILE_THREADS = 1          # section 2.2 step 2
SC_A_ORT = NEW               # section 2.2 step 1: the dump and the profile come from .envs/ort1244

# The three local files of section 1.3 / VA2. They are controls and are never in the matrix.
CONTROLS = {
    "mobilevit_xxs_64_int8 (positive control)":
        ROOT / "results/classification_mobilevit/models/p00_int8.onnx",
    "deit_tiny_int8 (negative control)":
        ROOT / "results/classification_deit/models/p00_int8.onnx",
    "mobilenetv2_int8 (negative control)":
        ROOT / "results/classification/models/p00_int8.onnx",
}
MOBILEVIT = "mobilevit_xxs_64_int8 (positive control)"
DEIT = "deit_tiny_int8 (negative control)"
MNV2 = "mobilenetv2_int8 (negative control)"

EIGHT_BIT_TYPES = (TensorProto.INT8, TensorProto.UINT8)
# Executed dtype name (as ORT's profile writes it) -> element size in bytes. SC-A only.
PROFILE_ELEM_BYTES = {
    "uint8": 1, "int8": 1, "bool": 1, "float16": 2, "bfloat16": 2, "uint16": 2, "int16": 2,
    "float": 4, "int32": 4, "uint32": 4, "int64": 8, "uint64": 8, "double": 8,
}


# ------------------------------------------------------------------------------------------------
# predict()'s own wording, parsed back: no arithmetic of the dispatch rule is repeated here
# ------------------------------------------------------------------------------------------------
_MLAS_RE = re.compile(r"^MlasTranspose<(\d+)-bit> (\w+) loops=(\d+) M=(\d+) N=(\d+)$")
_SCALAR_RE = re.compile(r"^scalar (\w+) (\d+)-bit loops=(\d+)$")


def _path_facts(path_string):
    """kind / moved_block_bytes / loops / M / N, read out of the string predict() returned.

    predict() names M and N only when it routes to MlasTranspose, so a node that stays scalar on
    both releases reports M = N = None. Nothing is recomputed here."""
    m = _MLAS_RE.match(path_string)
    if m:
        return {"kind": m.group(2), "moved_block_bytes": int(m.group(1)) // 8,
                "loops": int(m.group(3)), "M": int(m.group(4)), "N": int(m.group(5))}
    m = _SCALAR_RE.match(path_string)
    if m:
        return {"kind": m.group(1), "moved_block_bytes": int(m.group(2)) // 8,
                "loops": int(m.group(3)), "M": None, "N": None}
    return {"kind": None, "moved_block_bytes": None, "loops": None, "M": None, "N": None}


def node_record(name, dims, perm, esize, dtype, node_index=None):
    """One Transpose, scored by predict() on both releases. FLAGGED iff it gains dispatches."""
    dims, perm = list(dims), list(perm)
    p_old = predict(dims, perm, esize, THREADS, OLD)
    p_new = predict(dims, perm, esize, THREADS, NEW)
    facts = _path_facts(p_new["path"])
    if facts["loops"] is None:
        facts = _path_facts(p_old["path"])
    added = p_new["dispatches"] - p_old["dispatches"]
    sa = _single_axis(perm)
    return {
        "node_index": node_index, "name": name, "shape": dims, "perm": perm,
        "dtype": dtype, "elem_bytes": esize,
        "loops": facts["loops"], "M": facts["M"], "N": facts["N"],
        "moved_block_bytes": facts["moved_block_bytes"], "kind": facts["kind"],
        "reshape_copy": bool(_is_reshape(perm, dims)),
        "single_axis": list(sa) if sa is not None else None,
        f"path_{OLD}": p_old["path"], f"path_{NEW}": p_new["path"],
        f"dispatches_{OLD}": p_old["dispatches"], f"dispatches_{NEW}": p_new["dispatches"],
        "added_dispatches": added, "flagged": added > 0,
    }


def _totals(nodes):
    old = sum(n[f"dispatches_{OLD}"] for n in nodes)
    new = sum(n[f"dispatches_{NEW}"] for n in nodes)
    return {OLD: old, NEW: new}, new - old


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _attr(node, key, default=None):
    for a in node.attribute:
        if a.name == key:
            return helper.get_attribute_value(a)
    return default


# ------------------------------------------------------------------------------------------------
# SC-B, the file-only checker. PRIMARY. (registration section 2.1)
# ------------------------------------------------------------------------------------------------
def sc_b(model_path, batch=1):
    model = onnx.load(str(model_path))
    notes = []

    # 1. pin the batch dimension of the single input to 1; leave C,H,W as published
    init_names = {i.name for i in model.graph.initializer}
    real_inputs = [i for i in model.graph.input if i.name not in init_names]
    if len(real_inputs) != 1:
        notes.append(f"the file has {len(real_inputs)} non-initializer graph inputs, not 1 "
                     f"(admission rule A1); every one of them had its dim 0 pinned to {batch}")
    pinned = []
    for i in real_inputs:
        dim = i.type.tensor_type.shape.dim
        if len(dim) and not dim[0].HasField("dim_value"):
            was = dim[0].dim_param or "?"
            dim[0].ClearField("dim_param")
            dim[0].dim_value = batch
            pinned.append(f"{i.name}[0]={batch} (was '{was}')")
        for j, d in enumerate(dim):
            if j and not d.HasField("dim_value"):
                notes.append(f"{i.name}[{j}] is symbolic ('{d.dim_param}') and was NOT pinned; "
                             "every Transpose downstream of it stays non-static")

    # 2. shape inference
    inferred, inf_error = model, None
    try:
        inferred = shape_inference.infer_shapes(model, data_prop=True, strict_mode=False)
    except Exception as exc:                                   # noqa: BLE001 - report, never guess
        inf_error = f"{type(exc).__name__}: {exc}"
        notes.append(f"onnx shape inference failed ({inf_error}); shapes stay UNKNOWN")
    types = {}
    for vi in (list(inferred.graph.value_info) + list(inferred.graph.input)
               + list(inferred.graph.output)):
        tt = vi.type.tensor_type
        types[vi.name] = (tt.elem_type,
                          [d.dim_value if d.HasField("dim_value") else None for d in tt.shape.dim])
    for init in model.graph.initializer:
        types.setdefault(init.name, (init.data_type, list(init.dims)))

    producer, consumers = {}, collections.defaultdict(list)
    for n in model.graph.node:
        for o in n.output:
            producer[o] = n
        for i in n.input:
            consumers[i].append(n)
    graph_outputs = {o.name for o in model.graph.output}

    # 3. every Transpose with a fully static input shape that executes in 8 bits
    nodes, excluded, any_consumer_only, n_transposes = [], [], 0, 0
    for idx, n in enumerate(model.graph.node):
        if n.op_type != "Transpose":
            continue
        n_transposes += 1
        et, shape = types.get(n.input[0], (0, None))
        perm = _attr(n, "perm")
        perm = list(perm) if perm is not None else (
            list(reversed(range(len(shape)))) if shape is not None else None)
        static = shape is not None and perm is not None and len(shape) > 0 and None not in shape
        src = producer.get(n.input[0])
        cons = consumers.get(n.output[0], [])
        from_dq = src is not None and src.op_type == "DequantizeLinear"
        q_all = (bool(cons) and all(c.op_type == "QuantizeLinear" for c in cons)
                 and n.output[0] not in graph_outputs)
        q_any = any(c.op_type == "QuantizeLinear" for c in cons)
        sandwich = from_dq and q_all
        if from_dq and q_any and not q_all:
            any_consumer_only += 1
        already8 = et in EIGHT_BIT_TYPES
        if not (sandwich or already8):
            excluded.append({"node_index": idx, "name": n.name, "why": "not 8-bit-executing",
                             "elem_type": TensorProto.DataType.Name(et) if et else "UNKNOWN",
                             "producer": src.op_type if src is not None else None,
                             "consumers": sorted({c.op_type for c in cons})})
            continue
        if not static:
            excluded.append({"node_index": idx, "name": n.name,
                             "why": "input shape not fully static", "shape": shape, "perm": perm})
            continue
        # 4. the dispatch rule, at 8 bits, on both releases
        nodes.append(node_record(n.name, shape, perm, ESIZE_8BIT,
                                 TensorProto.DataType.Name(et) if et else "UNKNOWN", idx))

    census = collections.Counter(f"{n.domain or 'ai.onnx'}:{n.op_type}" for n in model.graph.node)
    opaque = {k: v for k, v in sorted(census.items()) if not k.startswith("ai.onnx:")}
    dispatches, added = _totals(nodes)
    flagged = [n for n in nodes if n["flagged"]]
    if any_consumer_only:
        notes.append(f"{any_consumer_only} Transpose(s) sit after a DequantizeLinear and feed a "
                     "QuantizeLinear among OTHER consumers; they are NOT counted as 8-bit-executing")
    if opaque:
        notes.append("the file carries operators outside the ai.onnx domain, which onnx shape "
                     f"inference cannot type: {opaque}")
    # 5. model-level outputs
    return {
        "checker": "SC-B", "role": "PRIMARY", "section": "2.1",
        "model_path": str(model_path).replace("\\", "/"), "sha256": sha256(model_path),
        "batch_pinned_to": batch, "pinned": pinned,
        "n_transposes_in_file": n_transposes,
        "n_8bit_executing": len(nodes),
        "n_excluded": len(excluded),
        "n_nonstatic_8bit": sum(1 for s in excluded if s["why"] == "input shape not fully static"),
        "n_flagged": len(flagged),
        "sig_b": bool(flagged),
        "dispatches": dispatches,
        "added_dispatches": added,
        "shape_inference_error": inf_error,
        "opaque_ops": opaque,
        "notes": notes,
        "flagged_nodes": [{"name": n["name"], "shape": n["shape"], "perm": n["perm"],
                           "loops": n["loops"], "M": n["M"], "N": n["N"],
                           "added_dispatches": n["added_dispatches"]} for n in flagged],
        "nodes": nodes,
        "excluded": excluded,
    }


# ------------------------------------------------------------------------------------------------
# SC-A, the executed-graph checker. SECONDARY. (registration section 2.2)
# ------------------------------------------------------------------------------------------------
SC_A_CHILD = textwrap.dedent('''
    """SC-A child, run by .envs/ort1244: dump the optimized graph, then 3 profiled iterations.

    Durations are never read; the profile is opened only for args.op_name / args.input_type_shape.
    """
    import json, os, sys
    import numpy as np
    import onnxruntime as ort

    src, dump, prof_dir, iters, threads = (sys.argv[1], sys.argv[2], sys.argv[3],
                                           int(sys.argv[4]), int(sys.argv[5]))
    free = {}
    for item in sys.argv[6:]:
        name, _, val = item.partition("=")
        free[name] = int(val)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.optimized_model_filepath = dump
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    for k, v in free.items():
        so.add_free_dimension_override_by_name(k, v)
    ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])

    so2 = ort.SessionOptions()
    so2.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so2.enable_profiling = True
    so2.profile_file_prefix = os.path.join(prof_dir, "prof")
    so2.intra_op_num_threads = threads
    so2.inter_op_num_threads = 1
    so2.log_severity_level = 3
    for k, v in free.items():
        so2.add_free_dimension_override_by_name(k, v)
    sess = ort.InferenceSession(src, so2, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    NP = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(double)": np.float64,
          "tensor(uint8)": np.uint8, "tensor(int8)": np.int8, "tensor(int32)": np.int32,
          "tensor(int64)": np.int64, "tensor(bool)": np.bool_}
    feed = {}
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) else free.get(d, 1) for d in i.shape]
        dt = NP.get(i.type)
        if dt is None:
            raise SystemExit("unsupported input type " + i.type)
        if dt in (np.float32, np.float16, np.float64):
            feed[i.name] = rng.standard_normal(shape).astype(dt)
        elif dt is np.bool_:
            feed[i.name] = np.zeros(shape, dtype=np.bool_)
        else:
            feed[i.name] = rng.integers(0, 128, size=shape, dtype=np.int64).astype(dt)
    for _ in range(iters):
        sess.run(None, feed)
    profile = sess.end_profiling()
    print(json.dumps({"onnxruntime": ort.__version__, "profile": profile, "dump": dump,
                      "inputs": [[i.name, i.type, [str(d) for d in i.shape]]
                                 for i in sess.get_inputs()],
                      "iterations": iters, "intra_op_num_threads": threads,
                      "free_dim_overrides": free}))
''')


def _dump_stem(model_path):
    """A work-dir name that stays unique when twenty projects all call the file p00_int8.onnx."""
    src = Path(model_path).resolve()
    tag = hashlib.sha1(str(src).lower().encode("utf-8")).hexdigest()[:8]
    parent = src.parent.parent.name or src.parent.name or "src"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in f"{parent}_{src.stem}")
    return f"{safe}_{tag}"


def _run_child(model_path, work_dir, free_dims=(), python=None, timeout=900):
    work = Path(work_dir)
    stem = _dump_stem(model_path)
    prof_dir = work / "profiles" / stem
    prof_dir.mkdir(parents=True, exist_ok=True)
    # Profiles are never deleted or globbed for: the child hands back the exact path
    # end_profiling() returned, so two runs of the same model can never be mixed up.
    child = work / "a_check_sc_a_child.py"
    child.write_text(SC_A_CHILD, encoding="utf-8")
    dump = work / f"opt_{SC_A_ORT}_{stem}.onnx"
    py = Path(python) if python else ORT1244_PYTHON
    if not py.exists():
        raise SystemExit(f"ERROR: no ONNX Runtime {SC_A_ORT} interpreter at {py}")
    cmd = [str(py), str(child), str(model_path), str(dump), str(prof_dir),
           str(PROFILE_ITERATIONS), str(PROFILE_THREADS)] + list(free_dims)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        return None, {"error": (r.stderr.strip() or r.stdout.strip())[-1500:],
                      "cmd": [str(c) for c in cmd]}
    return json.loads(r.stdout.strip().splitlines()[-1]), None


def _profile_transposes(profile_path):
    """{node name: (dtype, shape)} from the Node *_kernel_time events, plus what varied."""
    data = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    events = data["traceEvents"] if isinstance(data, dict) else data
    seen, variants, n_events = collections.OrderedDict(), collections.defaultdict(set), 0
    for e in events:
        if not isinstance(e, dict) or e.get("cat") != "Node":
            continue
        name = e.get("name", "")
        args = e.get("args") or {}
        if not name.endswith("_kernel_time") or args.get("op_name") != "Transpose":
            continue
        its = args.get("input_type_shape") or []
        if not its:
            continue
        first = its[0]
        dtype = next(iter(first))
        shape = [int(x) for x in first[dtype]]
        node = name[: -len("_kernel_time")]
        n_events += 1
        variants[node].add((dtype, tuple(shape)))
        seen.setdefault(node, (dtype, shape))
    return seen, {k: sorted(v) for k, v in variants.items() if len(v) > 1}, n_events


def sc_a(model_path, work_dir, free_dims=(), python=None):
    info, err = _run_child(model_path, work_dir, free_dims, python)
    if err is not None:
        return {"checker": "SC-A", "role": "SECONDARY", "section": "2.2",
                "model_path": str(model_path).replace("\\", "/"), "sha256": sha256(model_path),
                "ok": False, "sig_a": None, "added_dispatches": None, **err}
    executed, varying, n_events = _profile_transposes(info["profile"])
    opt = onnx.load(info["dump"])
    graph, duplicate_names = {}, []
    for idx, n in enumerate(opt.graph.node):
        if n.op_type == "Transpose":
            perm = _attr(n, "perm")
            if n.name in graph:          # the join is BY NODE NAME, so a repeat makes it ambiguous
                duplicate_names.append(n.name)
            graph[n.name] = (idx, list(perm) if perm is not None else None)

    nodes, unjoined, unknown_dtype = [], [], []
    for name, (dtype, shape) in executed.items():
        if name not in graph:
            unjoined.append({"name": name, "dtype": dtype, "shape": shape,
                             "why": "executed but no Transpose of that name in the optimized graph"})
            continue
        idx, perm = graph[name]
        if perm is None:
            perm = list(reversed(range(len(shape))))
        esize = PROFILE_ELEM_BYTES.get(dtype)
        if esize is None:
            unknown_dtype.append({"name": name, "dtype": dtype})
            continue
        nodes.append(node_record(name, shape, perm, esize, dtype, idx))
    nodes.sort(key=lambda r: r["node_index"])
    not_executed = [{"name": k, "node_index": v[0], "why": "in the optimized graph but no kernel "
                     "event in the profile"} for k, v in graph.items() if k not in executed]

    dispatches, added = _totals(nodes)
    flagged = [n for n in nodes if n["flagged"]]
    return {
        "checker": "SC-A", "role": "SECONDARY", "section": "2.2", "ok": True,
        "model_path": str(model_path).replace("\\", "/"), "sha256": sha256(model_path),
        "onnxruntime": info["onnxruntime"], "dump": info["dump"], "profile": info["profile"],
        "profile_iterations": info["iterations"],
        "intra_op_num_threads": info["intra_op_num_threads"],
        "free_dim_overrides": info["free_dim_overrides"], "inputs": info["inputs"],
        "n_transposes_optimized_graph": len(graph),
        "duplicate_transpose_names_in_graph": duplicate_names,
        "n_transposes_executed": len(executed),
        "n_scored": len(nodes),
        "n_unjoined": len(unjoined) + len(not_executed),
        "n_executed_not_in_graph": len(unjoined),
        "n_in_graph_not_executed": len(not_executed),
        "n_kernel_events": n_events,
        "executed_dtypes": sorted({n["dtype"] for n in nodes}),
        "dtype_or_shape_varied_between_iterations": varying,
        "unknown_dtypes": unknown_dtype,
        "n_flagged": len(flagged),
        "sig_a": bool(flagged),
        "dispatches": dispatches,
        "added_dispatches": added,
        "unjoined": unjoined + not_executed,
        "flagged_nodes": [{"name": n["name"], "node_index": n["node_index"], "shape": n["shape"],
                           "perm": n["perm"], "dtype": n["dtype"], "loops": n["loops"],
                           "M": n["M"], "N": n["N"],
                           "added_dispatches": n["added_dispatches"]} for n in flagged],
        "nodes": nodes,
    }


# ------------------------------------------------------------------------------------------------
# From flag to prediction (registration section 2.3)
# ------------------------------------------------------------------------------------------------
def prediction(sig_b, added_dispatches, screen_ms, added_dispatches_a=None):
    """The registered rule. screen_ms comes from the runner's manifest; nothing is measured here."""
    a_ms = added_dispatches * DISPATCH_MS
    if not screen_ms or screen_ms <= 0:
        return {"screen_ms": screen_ms, "sig_b": bool(sig_b),
                "added_dispatches": added_dispatches, "dispatch_us": DISPATCH_US,
                "A_ms": a_ms, "rho": None, "floor_rel": FLOOR_REL,
                "predicted_class": None, "sig_small": None,
                "error": "no usable screen_ms: rho and the predicted class are undefined"}
    rho = a_ms / screen_ms
    step = bool(sig_b) and rho >= FLOOR_REL
    out = {
        "screen_ms": screen_ms,
        "sig_b": bool(sig_b),
        "added_dispatches": added_dispatches,
        "dispatch_us": DISPATCH_US,
        "A_ms": a_ms,
        "rho": rho,
        "floor_rel": FLOOR_REL,
        "predicted_class": "PREDICT-STEP" if step else "PREDICT-SAFE",
        "sig_small": bool(sig_b) and rho < FLOOR_REL,
        "magnitude_band_reported_only": {"low": MAGNITUDE_BAND[0], "high": MAGNITUDE_BAND[1],
                                         "on": "measured_delta_ms / A_ms"},
    }
    if added_dispatches_a is not None:
        a_ms_a = added_dispatches_a * DISPATCH_MS
        out["secondary_SC_A"] = {"added_dispatches_a": added_dispatches_a, "A_ms_a": a_ms_a,
                                 "rho_a": a_ms_a / screen_ms}
    return out


def magnitude_ratio(measured_delta_ms, a_ms):
    """REPORTED-ONLY (section 2.3): it can never change a verdict."""
    if not a_ms:
        return {"ratio": None, "in_band": None}
    ratio = measured_delta_ms / a_ms
    return {"ratio": ratio, "in_band": MAGNITUDE_BAND[0] <= ratio <= MAGNITUDE_BAND[1]}


def _abs_key(p):
    """The model path as one absolute, lower-case, forward-slash key. Relative paths in a manifest
    are read relative to the project root, which is how every other artefact of Experiment A
    spells them."""
    q = Path(str(p).replace("\\", "/"))
    if not q.is_absolute():
        q = ROOT / q
    try:
        q = q.resolve()
    except OSError:
        pass
    return str(q).replace("\\", "/").lower()


def load_screen_ms(manifest_path):
    """Screen_ms from the runner's manifest (artefact 2), indexed by every plausible key.

    Three local control files are all called `p00_int8.onnx`, and the third-party pool may well
    repeat a file name too, so a key that two entries claim with DIFFERENT values is recorded as
    ambiguous and never used: the lookup falls through to the sha256 or the full path instead of
    silently taking one of them."""
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        entries = data.get("models") or data.get("measured") or data.get("entries")
        if entries is None:
            entries = [dict(v, **{"id": k}) for k, v in data.items() if isinstance(v, dict)]
    else:
        entries = data
    if isinstance(entries, dict):
        entries = [dict(v, **{"id": k}) for k, v in entries.items() if isinstance(v, dict)]
    index, ambiguous = {}, set()

    def put(key, value):
        key = str(key).lower()
        if key in index and index[key] != value:
            ambiguous.add(key)
        index[key] = value

    for e in entries:
        if not isinstance(e, dict) or "screen_ms" not in e:
            continue
        val = e["screen_ms"]
        for field in ("id", "variant", "stem", "name", "model", "sha256"):
            v = e.get(field)
            if isinstance(v, str) and v:
                put(v, val)
        for field in ("path", "file", "model_path"):
            v = e.get(field)
            if isinstance(v, str) and v:
                put(_abs_key(v), val)
                put(str(v).replace("\\", "/"), val)
                put(Path(str(v).replace("\\", "/")).name, val)
                put(Path(str(v).replace("\\", "/")).stem, val)
    for key in ambiguous:
        index.pop(key, None)
    return {"index": index, "ambiguous": sorted(ambiguous)}


def lookup_screen_ms(index, model_path, model_sha256, label=None):
    """(screen_ms, the key it was found under). Most specific key first; an ambiguous key is gone
    from the index already, so a name collision returns (None, None) instead of a wrong number."""
    amb = set(index.get("ambiguous", ())) if isinstance(index, dict) else set()
    table = index.get("index", index) if isinstance(index, dict) else index
    p = str(model_path).replace("\\", "/")
    keys = [model_sha256, _abs_key(model_path), p, Path(p).name, Path(p).stem]
    if label:
        keys.insert(1, label)
    for key in keys:
        k = str(key).lower()
        if k in table and k not in amb:
            return table[k], k
    return None, None


# ------------------------------------------------------------------------------------------------
# Self-test (artefact 4): both checkers on the three local controls, against the recorded truth
# ------------------------------------------------------------------------------------------------
def _strip_volatile(report):
    """Two runs are compared on everything else the checker reports. Only the profile's file name
    is dropped: ORT timestamps it, and it is a file name, not a decision."""
    r = copy.deepcopy(report)
    r.pop("profile", None)
    return r


def _check(checks, cid, what, source, computed, expected):
    ok = computed == expected
    checks.append({"id": cid, "what": what, "source": source, "computed": computed,
                   "expected": expected, "result": "PASS" if ok else "FAIL"})
    return ok


def selftest(work_dir, out_path=None, python=None):
    missing = [str(p) for p in CONTROLS.values() if not p.exists()]
    if missing:
        raise SystemExit("ERROR: missing local control file(s): " + ", ".join(missing))
    gt = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    want_all = gt["variants"]["tp_u8_all"]["transpose_paths_4T"]
    want_w2 = gt["variants"]["tp_u8_w2"]["transpose_paths_4T"]
    want_disp = gt["variants"]["tp_u8_all"]["predicted_dispatches"]

    b, a, b2, a2 = {}, {}, {}, {}
    for label, path in CONTROLS.items():
        print(f"running SC-B and SC-A twice on {label} ...")
        b[label], b2[label] = sc_b(path), sc_b(path)
        a[label], a2[label] = (sc_a(path, work_dir, python=python),
                               sc_a(path, work_dir, python=python))
        if not a[label].get("ok") or not a2[label].get("ok"):
            raise SystemExit(f"ERROR: SC-A failed on {label}: "
                             f"{a[label].get('error') or a2[label].get('error')}")

    checks, ok = [], True

    # ---- VA2, first half: the checkers are deterministic ---------------------------------------
    for label in CONTROLS:
        ok &= _check(checks, f"D-SC-B|{label}", "SC-B run twice gives identical output",
                     "registration VA2",
                     _strip_volatile(b[label]) == _strip_volatile(b2[label]), True)
        ok &= _check(checks, f"D-SC-A|{label}", "SC-A run twice gives identical output",
                     "registration VA2",
                     _strip_volatile(a[label]) == _strip_volatile(a2[label]), True)

    # ---- SC-B against the registration's VERIFIED table (section 2.1) --------------------------
    src21 = "a_design_registered.md section 2.1 (VERIFIED table)"
    for label, n8, nflag, added in ((MOBILEVIT, 39, 4, 832), (DEIT, 37, 0, 0), (MNV2, 0, 0, 0)):
        s = b[label]
        ok &= _check(checks, f"B1|{label}", "SC-B: 8-bit-executing Transposes in the file", src21,
                     s["n_8bit_executing"], n8)
        ok &= _check(checks, f"B2|{label}", "SC-B: flagged nodes", src21, s["n_flagged"], nflag)
        ok &= _check(checks, f"B3|{label}", "SC-B: added dispatches at 4 threads", src21,
                     s["added_dispatches"], added)
        ok &= _check(checks, f"B4|{label}", "SC-B: sig_b", src21, s["sig_b"], nflag > 0)
        ok &= _check(checks, f"B5|{label}", "SC-B: 8-bit Transposes dropped for a non-static shape",
                     "section 2.1 step 3", s["n_nonstatic_8bit"], 0)

    want_names_21 = [
        ["/stages/stages.2/stages.2.1/Transpose", [256, 2, 4, 2], [0, 2, 1, 3]],
        ["/stages/stages.2/stages.2.1/Transpose_3", [256, 4, 2, 2], [0, 2, 1, 3]],
        ["/stages/stages.3/stages.3.1/Transpose", [160, 2, 2, 2], [0, 2, 1, 3]],
        ["/stages/stages.3/stages.3.1/Transpose_3", [160, 2, 2, 2], [0, 2, 1, 3]],
    ]
    ok &= _check(checks, "B6|flagged node names", "SC-B: the four flagged nodes, name/shape/perm",
                 src21 + " (the four names) and VA2",
                 [[n["name"], n["shape"], n["perm"]] for n in b[MOBILEVIT]["flagged_nodes"]],
                 want_names_21)

    # ---- SC-A against results/transpose_probe/manifest.json and section 2.2 --------------------
    src22 = "a_design_registered.md section 2.2 (VERIFIED)"
    srcgt = "results/transpose_probe/manifest.json (Exp 27/27b ground truth)"
    for label, nex in ((MOBILEVIT, 56), (DEIT, 51), (MNV2, 2)):
        s = a[label]
        ok &= _check(checks, f"A1|{label}", "SC-A: executed Transposes", src22,
                     s["n_transposes_executed"], nex)
        ok &= _check(checks, f"A2|{label}", "SC-A: unjoined nodes, either direction", src22,
                     s["n_unjoined"], 0)
        ok &= _check(checks, f"A2b|{label}", "SC-A: no two Transposes share a name, so the join "
                     "by node name is unambiguous", "section 2.2 step 3",
                     s["duplicate_transpose_names_in_graph"], [])
        ok &= _check(checks, f"A3|{label}", "SC-A: executed dtypes", src22 + " (\"all uint8\")",
                     s["executed_dtypes"], ["uint8"])
        ok &= _check(checks, f"A4|{label}", "SC-A: dtype and shape stable over the 3 iterations",
                     "section 2.2 step 2", s["dtype_or_shape_varied_between_iterations"], {})
    for label, old, new in ((MOBILEVIT, 32, 864), (DEIT, 15, 15), (MNV2, 1, 1)):
        s = a[label]
        src = srcgt if label == MOBILEVIT else src22
        ok &= _check(checks, f"A5|{label}", f"SC-A: graph dispatch total at 4 threads on {OLD}",
                     src, s["dispatches"][OLD], old)
        ok &= _check(checks, f"A6|{label}", f"SC-A: graph dispatch total at 4 threads on {NEW}",
                     src, s["dispatches"][NEW], new)
        ok &= _check(checks, f"A7|{label}", "SC-A: added dispatches", src,
                     s["added_dispatches"], new - old)
    ok &= _check(checks, "A8|mobilevit dispatches vs manifest",
                 "SC-A: MobileViT dispatch totals equal the manifest's registered prediction", srcgt,
                 {OLD: a[MOBILEVIT]["dispatches"][OLD], NEW: a[MOBILEVIT]["dispatches"][NEW]},
                 {OLD: want_disp[OLD][str(THREADS)], NEW: want_disp[NEW][str(THREADS)]})
    ok &= _check(checks, "A9|mobilevit 56 rows vs manifest",
                 "SC-A: all 56 rows agree on node index, shape, perm and the path on both releases",
                 srcgt,
                 [[n["node_index"], n["shape"], n["perm"], n[f"path_{OLD}"], n[f"path_{NEW}"]]
                  for n in a[MOBILEVIT]["nodes"]],
                 [[w["idx"], list(w["shape"]), list(w["perm"]), w[OLD], w[NEW]] for w in want_all])
    ok &= _check(checks, "A10|mobilevit flagged vs manifest tp_u8_w2",
                 "SC-A: the flagged set is exactly the four recorded 2-byte-block nodes", srcgt,
                 [[n["node_index"], n["shape"], n["perm"]] for n in a[MOBILEVIT]["flagged_nodes"]],
                 [[w["idx"], list(w["shape"]), list(w["perm"])] for w in want_w2])
    ok &= _check(checks, "A11|mobilevit flagged names",
                 "SC-A: the four flagged nodes carry the names of section 2.1", src21,
                 [n["name"] for n in a[MOBILEVIT]["flagged_nodes"]], [w[0] for w in want_names_21])
    for label in (DEIT, MNV2):
        ok &= _check(checks, f"A12|{label}", "SC-A: flagged nodes", src22, a[label]["n_flagged"], 0)

    # ---- the two checkers against each other (section 2.2: SC-B's blind spot) ------------------
    disagreement = [{"model": label, "sig_b": b[label]["sig_b"], "sig_a": a[label]["sig_a"],
                     "added_b": b[label]["added_dispatches"],
                     "added_a": a[label]["added_dispatches"],
                     "transposes_in_file": b[label]["n_transposes_in_file"],
                     "transposes_executed": a[label]["n_transposes_executed"],
                     "agree_on_the_flag": b[label]["sig_b"] == a[label]["sig_a"]}
                    for label in CONTROLS]
    ok &= _check(checks, "X1|third control blind spot",
                 "the third control has no Transpose in the file but ORT executes two",
                 "a_design_registered.md section 2.2, last paragraph",
                 [b[MNV2]["n_transposes_in_file"], a[MNV2]["n_transposes_executed"]], [0, 2])

    # ---- section 2.3 arithmetic, on synthetic inputs (no model number is invented) -------------
    src23 = "a_design_registered.md section 2.3, applied to synthetic inputs"
    p1 = prediction(True, 832, 2.0)
    ok &= _check(checks, "P1|A_ms", "A_ms = added_dispatches * 0.00052 ms, at 832 dispatches", src23,
                 round(p1["A_ms"], 9), round(832 * 0.00052, 9))
    ok &= _check(checks, "P2|rho", "rho = A_ms / screen_ms, at a synthetic screen_ms of 2.0 ms",
                 src23, round(p1["rho"], 9), round(832 * 0.00052 / 2.0, 9))
    ok &= _check(checks, "P3|class", "sig_b and rho >= 0.07 -> PREDICT-STEP", src23,
                 [p1["predicted_class"], p1["sig_small"]], ["PREDICT-STEP", False])
    p2 = prediction(True, 100, 10.0)
    ok &= _check(checks, "P4|sig-small", "sig_b and rho < 0.07 -> PREDICT-SAFE and SIG-SMALL",
                 src23, [p2["predicted_class"], p2["sig_small"]], ["PREDICT-SAFE", True])
    p3 = prediction(False, 0, 10.0)
    ok &= _check(checks, "P5|cleared", "no flag -> PREDICT-SAFE and not SIG-SMALL", src23,
                 [p3["predicted_class"], p3["sig_small"]], ["PREDICT-SAFE", False])
    ok &= _check(checks, "P6|magnitude band",
                 "the REPORTED-ONLY band 0.5 <= measured_delta_ms / A_ms <= 2.0", src23,
                 [magnitude_ratio(0.4, 0.43264)["in_band"],
                  magnitude_ratio(0.1, 0.43264)["in_band"],
                  magnitude_ratio(1.0, 0.43264)["in_band"]], [True, False, False])

    report = {
        "experiment": "A",
        "artefact": "4 (checker self-test against ground truth)",
        "implements": "a_design_registered.md sections 2.1, 2.2, 2.3",
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "result": "PASS" if ok else "FAIL",
        "n_checks": len(checks),
        "n_pass": sum(1 for c in checks if c["result"] == "PASS"),
        "n_fail": sum(1 for c in checks if c["result"] == "FAIL"),
        "checks": checks,
        "provenance": {
            "a_check.py": {"path": "scripts/paper/a_check.py",
                           "sha256": sha256(Path(__file__).resolve())},
            "transpose_probe.py": {"path": "scripts/paper/transpose_probe.py",
                                   "sha256": sha256(ROOT / "scripts/paper/transpose_probe.py"),
                                   "imported": ["predict", "_single_axis", "_is_reshape",
                                                "MLAS_BYTES"]},
            "ground_truth": {"path": "results/transpose_probe/manifest.json",
                             "sha256": sha256(GROUND_TRUTH)},
            "registration": {"path": "results/paper/a_design_registered.md",
                             "sha256": sha256(REGISTRATION)},
            "onnx": onnx.__version__,
            "project_python": sys.executable.replace("\\", "/"),
            "sc_a_python": str(python or ORT1244_PYTHON).replace("\\", "/"),
            "sc_a_onnxruntime": a[MOBILEVIT].get("onnxruntime"),
            "work_dir": str(work_dir).replace("\\", "/"),
        },
        "constants": {"threads": THREADS, "versions": [OLD, NEW],
                      "MLAS_BYTES": {k: sorted(v) for k, v in MLAS_BYTES.items()},
                      "dispatch_us": DISPATCH_US, "floor_rel": FLOOR_REL,
                      "magnitude_band": list(MAGNITUDE_BAND),
                      "profile_iterations": PROFILE_ITERATIONS,
                      "profile_intra_op_threads": PROFILE_THREADS},
        "sc_a_vs_sc_b": disagreement,
        "models": {label: {
            "path": str(path).replace("\\", "/"), "sha256": sha256(path),
            "SC-B": {k: v for k, v in b[label].items() if k not in ("nodes", "excluded")},
            "SC-A": {k: v for k, v in a[label].items() if k != "nodes"},
            "SC-B_nodes": b[label]["nodes"],
            "SC-A_nodes": a[label]["nodes"],
        } for label, path in CONTROLS.items()},
    }
    out = Path(out_path) if out_path else OUT_DIR / "selftest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print()
    for c in checks:
        print(f"[{c['result']}] {c['id']}: {c['what']}")
        if c["result"] == "FAIL":
            print(f"        computed: {c['computed']}")
            print(f"        expected: {c['expected']}")
            print(f"        source:   {c['source']}")
    print(f"\nSELF-TEST: {report['result']}  ({report['n_pass']}/{report['n_checks']} checks)")
    print(f"wrote {out}")
    return report


# ------------------------------------------------------------------------------------------------
def _print_summary(rows):
    print()
    print(f"{'model':<40} {'fileT':>5} {'8bit':>5} {'flag':>4} {'added':>6} | "
          f"{'execT':>5} {'flagA':>5} {'addedA':>6} | {'rho':>8}  class")
    for r in rows:
        b, a, p = r.get("SC-B") or {}, r.get("SC-A") or {}, r.get("prediction") or {}
        rho = p.get("rho")
        print(f"{r['model'][:40]:<40} {b.get('n_transposes_in_file', '-'):>5} "
              f"{b.get('n_8bit_executing', '-'):>5} {b.get('n_flagged', '-'):>4} "
              f"{b.get('added_dispatches', '-'):>6} | {a.get('n_transposes_executed', '-'):>5} "
              f"{a.get('n_flagged', '-'):>5} {a.get('added_dispatches', '-'):>6} | "
              f"{('-' if rho is None else format(rho, '.5f')):>8}  "
              f"{p.get('predicted_class') or '-'}"
              + ("  SIG-SMALL" if p.get("sig_small") else ""))


def _label(path):
    """A display label that survives twenty files all called p00_int8.onnx."""
    p = Path(path)
    parent = p.parent.parent.name or p.parent.name
    return f"{parent}/{p.stem}" if parent else p.stem


def _cmd_models(args, do_b, do_a):
    index = load_screen_ms(args.manifest) if getattr(args, "manifest", None) else None
    rows, missing_screen = [], []
    for m in args.models:
        path = Path(m)
        if not path.exists():
            raise SystemExit(f"ERROR: no such model file: {path}")
        row = {"model": _label(path), "path": str(path).replace("\\", "/"), "sha256": sha256(path)}
        if do_b:
            row["SC-B"] = sc_b(path, batch=args.batch)
        if do_a:
            row["SC-A"] = sc_a(path, args.work_dir, free_dims=args.free_dim or (),
                               python=args.sc_a_python)
        if index is not None and do_b:
            screen, key = lookup_screen_ms(index, path, row["sha256"])
            if screen is None:
                missing_screen.append(str(path))
            row["prediction"] = prediction(row["SC-B"]["sig_b"], row["SC-B"]["added_dispatches"],
                                           screen, (row.get("SC-A") or {}).get("added_dispatches"))
            row["prediction"]["screen_ms_key"] = key
        rows.append(row)
    _print_summary(rows)
    out = {"experiment": "A", "implements": "a_design_registered.md sections 2.1, 2.2, 2.3",
           "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "versions": [OLD, NEW], "threads": THREADS,
           "a_check_sha256": sha256(Path(__file__).resolve()),
           "transpose_probe_sha256": sha256(ROOT / "scripts/paper/transpose_probe.py"),
           "dispatch_us": DISPATCH_US, "floor_rel": FLOOR_REL,
           "manifest": (str(args.manifest).replace("\\", "/")
                        if getattr(args, "manifest", None) else None),
           "manifest_ambiguous_keys": (index or {}).get("ambiguous", []) if index else [],
           "models_without_screen_ms": missing_screen, "models": rows}
    if getattr(args, "json", None):
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\nwrote {p}")
    if missing_screen:
        print("\nERROR: no screen_ms in the manifest for: " + ", ".join(missing_screen))
        return 2
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p, need_work_dir):
        p.add_argument("models", nargs="+")
        p.add_argument("--batch", type=int, default=1,
                       help="value the single input's batch dimension is pinned to (default 1)")
        p.add_argument("--json", default=None, help="write the full report here")
        if need_work_dir:
            p.add_argument("--work-dir", required=True,
                           help="scratch directory for SC-A's dumps, profiles and child script "
                                "(outside the project; never deleted by this script)")
            p.add_argument("--free-dim", action="append", metavar="NAME=VALUE", default=[],
                           help="pin a symbolic input dimension by name before ORT optimizes "
                                "(repeatable); default: none, as SC-A was verified")
            p.add_argument("--sc-a-python", default=None,
                           help=f"interpreter with ONNX Runtime {SC_A_ORT} (default .envs/ort1244)")

    b = sub.add_parser("sc-b", help="PRIMARY: the file-only checker (section 2.1)")
    add_common(b, need_work_dir=False)

    a = sub.add_parser("sc-a", help="SECONDARY: the executed-graph checker (section 2.2)")
    add_common(a, need_work_dir=True)

    p = sub.add_parser("predict", help="both checkers plus the section 2.3 prediction")
    add_common(p, need_work_dir=True)
    p.add_argument("--manifest", required=True,
                   help="the runner's manifest (artefact 2); screen_ms is READ from it")

    s = sub.add_parser("selftest", help="artefact 4: the three local controls vs the ground truth")
    s.add_argument("--work-dir", required=True)
    s.add_argument("--out", default=None, help="default results/expA/selftest.json")
    s.add_argument("--sc-a-python", default=None)

    args = ap.parse_args(argv)
    if args.cmd == "sc-b":
        return _cmd_models(args, do_b=True, do_a=False)
    if args.cmd == "sc-a":
        return _cmd_models(args, do_b=False, do_a=True)
    if args.cmd == "predict":
        return _cmd_models(args, do_b=True, do_a=True)
    rep = selftest(args.work_dir, args.out, args.sc_a_python)
    return 0 if rep["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
