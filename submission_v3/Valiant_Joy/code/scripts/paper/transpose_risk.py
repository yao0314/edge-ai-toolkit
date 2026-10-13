"""transpose_risk.py - static check: will this quantized ONNX file lose speed on ONNX Runtime 1.24+?

WHAT IT PREDICTS
----------------
Between ONNX Runtime v1.23.2 and v1.24.x, commit 42869ca (PR #25680, "FP16 inference performance
improvement on CPU") added one line to `onnxruntime/core/framework/transpose_helper.cc`:

    template <> struct has_mlas_transpose<uint16_t> : std::true_type {};

From 1.24, a single-axis Transpose whose *moved block* is exactly 2 bytes wide stops running a
scalar copy loop and calls `MlasTranspose<uint16_t>` once per outer loop instead. `MlasTranspose`
goes through `MlasExecuteThreaded`, which posts to the intra-op thread pool whenever its iteration
count (min(DegreeOfParallelism, M)) is greater than 1. On a small tensor the dispatch costs more
than the copy, so a file full of small 2-byte-block Transposes gets SLOWER at >1 thread and is
unchanged at 1 thread. 8-bit tensors reach this path whenever the moved block is 2 ELEMENTS.

This script reads a quantized ONNX file, finds every 8-bit Transpose in the EXECUTED graph,
computes each one's moved block, applies one flag rule (see FLAG_RULE / is_flagged below) and
prints a per-model prediction: "steps" or "does not step" on 1.24.4 relative to 1.23.2 at
%(threads)d threads. It is a static check: nothing is timed, nothing is downloaded, no network.

WHY THE EXECUTED GRAPH, AND WHY THE DUMP MUST COME FROM 1.23.2
--------------------------------------------------------------
ONNX Runtime deletes or fuses most of a QDQ file before inference (~90%% of the Q/DQ nodes go), and
its transpose optimizer moves, merges and re-creates Transposes. Counting Transposes in the file as
shipped gives the wrong answer. So the graph is dumped through
`SessionOptions.optimized_model_filepath` first, with ORT %(dumpver)s: that is the LAST release
before the change, so its executed graph is the "before" graph the prediction is about. (1.24.4
produces a node-for-node identical graph for the reference file - Experiment 23a - but the tool
does not assume that for unseen models.)

TWO-STEP RUN (the environments here are split on purpose)
---------------------------------------------------------
The ORT %(dumpver)s environment has onnxruntime but NOT the `onnx` package; the project venv has
`onnx` but a newer onnxruntime. So dumping and analysis run under different interpreters:

  step 1, DUMP (needs onnxruntime %(dumpver)s only; this file imports onnx lazily):
      <ort1232>\\Scripts\\python.exe scripts/paper/transpose_risk.py dump \\
          --work-dir WORKDIR model_a.onnx model_b.onnx
  step 2, ANALYSE (needs the `onnx` package; onnxruntime is never imported):
      .venv\\Scripts\\python.exe scripts/paper/transpose_risk.py analyse \\
          --csv OUT_PREFIX WORKDIR/opt_%(dumpver)s_model_a.onnx ...
  validation against the recorded ground truth (runs step 1 for you via --dump-python):
      .venv\\Scripts\\python.exe scripts/paper/transpose_risk.py validate --work-dir WORKDIR
  optional guard for an unseen model - dump it with 1.24.4 as well (--allow-ort-version) and check
  that the executed graph did not change, so a measured difference can only be the dispatch change:
      .venv\\Scripts\\python.exe scripts/paper/transpose_risk.py compare OLD_DUMP NEW_DUMP

`dump` writes a JSON sidecar beside every dump recording which onnxruntime made it, so `analyse`
can state the provenance instead of assuming it.

DEFINITION: THE MOVED BLOCK
---------------------------
See `moved_block_bytes` in `transpose_plan()`. Short version: ORT handles a permutation that moves
ONE axis (`IsTransposeMovingSingleAxis`) with `TransposeSingleAxis{Outwards,Inwards}`, which walk
the tensor as loops x (M x N) blocks and copy, as one unit, the run of memory formed by all input
axes strictly to the RIGHT of the rightmost of the two positions the moved axis occupies (its
source position and its destination position). Those axes keep the same relative order and the same
strides in input and output, so that run is contiguous in both; its size in bytes is what
`has_mlas_transpose<T>` is specialised on. moved_block_bytes = element_size * prod(that run).

STATUS OF THE NUMBERS THIS TOOL USES
------------------------------------
The code path model (reshape-copy / single-axis MLAS / single-axis scalar / untyped serial, and the
per-release MLAS block-size set) is re-implemented from the v1.24.4 sources and is the same model
that `scripts/paper/transpose_probe.py` used for the pre-registered Experiment 27/27b predictions.
`validate` re-derives it from a fresh dump and diffs it against the recorded manifest, so the two
implementations cannot drift apart silently.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "results" / "transpose_probe" / "manifest.json"
DEFAULT_DUMP_PYTHON = ROOT / ".envs" / "ort1232" / "Scripts" / "python.exe"

# ------------------------------------------------------------------------------------------------
# THE RULE. Every decision in this file goes through FLAG_RULE / is_flagged / model_prediction.
# Nothing else may re-express it.
# ------------------------------------------------------------------------------------------------
DUMP_ORT_VERSION = "1.23.2"          # release whose executed graph is the "before" graph
FLAG_RULE = {
    # SCOPE CHOICE, not a property of the mechanism: the switch is on the BLOCK width, so a
    # float16 tensor whose moved block is one element also changes path (Exp 27 measured that arm).
    # This checker is aimed at quantized files, so it only flags 8-bit element types.
    "element_bits": 8,               # uint8 / int8 activations (the quantized case under test)
    "moved_block_bytes": 2,          # the exact width has_mlas_transpose<uint16_t> switched on
    "min_M": 2,                      # MlasExecuteThreaded only posts when iterations > 1
    "threads": 4,                    # the thread count the prediction is made at
    "new_in": "1.24",
    "pull_request": "microsoft/onnxruntime#25680",
}
# Block widths (bytes) for which has_mlas_transpose<T> is true, per release. 1.23.2: uint8_t and
# uint32_t. 1.24+: uint16_t as well (PR #25680). 8-byte blocks are NOT on the MLAS path on either.
MLAS_BLOCK_BYTES = {"1.23.2": frozenset({1, 4}), "1.24.4": frozenset({1, 2, 4})}
# Model-level rule. Each flagged node adds `loops` thread-pool dispatches per inference. Experiment
# 27b measured 0.51-0.53 us per dispatch on this machine (results/transpose_probe/verdict_followup.json),
# so 100 dispatches is about 0.05 ms, just above the project's registered 0.034 ms absolute floor.
# The summary CSV also carries the raw dispatch counts, so a frozen list can be re-scored at any
# other threshold without re-running the tool.
MODEL_STEP_MIN_ADDED_DISPATCHES = 100

ELEM_BYTES = {  # onnx TensorProto elem_type -> bytes. Sub-byte types are deliberately absent.
    1: 4, 2: 1, 3: 1, 4: 2, 5: 2, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8, 12: 4, 13: 8, 16: 2,
}


def is_flagged(plan: dict) -> bool:
    """THE flag rule, expressed once. `plan` comes from transpose_plan()."""
    return (plan.get("elem_bits") == FLAG_RULE["element_bits"]
            and plan.get("moved_block_bytes") == FLAG_RULE["moved_block_bytes"]
            and plan.get("path") == "mlas_single_axis_new"
            and (plan.get("M") or 0) >= FLAG_RULE["min_M"])


def model_prediction(added_dispatches: int) -> str:
    return "steps" if added_dispatches >= MODEL_STEP_MIN_ADDED_DISPATCHES else "does not step"


# ------------------------------------------------------------------------------------------------
# ORT's CPU Transpose dispatch, re-implemented from the v1.24.4 sources
# (core/providers/cpu/tensor/transpose.cc: TransposeImpl, IsTransposeReshape,
#  IsTransposeMovingSingleAxis, TransposeSingleAxisOutwards/Inwards;
#  core/framework/transpose_helper.cc: has_mlas_transpose;
#  core/mlas/lib/transpose.cpp: MlasTranspose -> MlasExecuteThreaded).
# ------------------------------------------------------------------------------------------------
def is_reshape_perm(perm, dims) -> bool:
    """IsTransposeReshape: only size-1 axes move, so the copy is a plain memcpy."""
    last = 0
    for p in perm:
        if dims[p] == 1:
            continue
        if p < last:
            return False
        last = p
    return True


def single_axis_move(perm):
    """IsTransposeMovingSingleAxis -> ("out", from, to) | ("in", from, to) | None.

    "out": the axis at input position `from` moves LEFT to output position `to` (to < from).
    "in":  the axis at input position `from` moves RIGHT to output position `to` (from < to).
    """
    n = len(perm)

    def outwards(cur, moved_from):
        expected = cur - 1
        while cur < n:
            if perm[cur] != expected:
                return False
            if cur == moved_from:
                expected += 1
            expected += 1
            cur += 1
        return True

    def inwards(cur):
        started, expected, moved_to = cur, cur + 1, None
        while cur < n:
            if perm[cur] != expected:
                if started != perm[cur]:
                    return None
                moved_to = cur
            else:
                expected += 1
            cur += 1
        return moved_to

    for i, axis in enumerate(perm):
        if axis != i:
            if outwards(i + 1, axis):
                return "out", axis, i
            t = inwards(i)
            return ("in", i, t) if t is not None else None
    return None


def transpose_plan(dims, perm, elem_bytes):
    """How ORT executes this one Transpose, and how wide its moved block is.

    Returns a dict with `path` in {reshape_copy, untyped_serial, scalar_single_axis,
    mlas_single_axis_both, mlas_single_axis_new} plus moved_block_bytes / loops / M / N, and the
    per-release thread-pool dispatch counts at FLAG_RULE["threads"].

    moved_block_bytes: ORT's single-axis transpose copies, as one unit, the run of memory spanned by
    all INPUT axes strictly to the right of max(from, to) - the rightmost of the moved axis's source
    and destination positions. Every axis beyond that point keeps both its order and its stride in
    input and output (the permutation only relocates one axis, and it lands at or left of that
    point), so that run is contiguous on both sides and is what the kernel templates on:
        moved_block_bytes = elem_bytes * prod(dims[max(from, to) + 1:])
    For "out" that is dims[from+1:], for "in" it is dims[to+1:] - the two branches ORT writes out
    separately in TransposeSingleAxisOutwards / TransposeSingleAxisInwards.
    """
    plan = {"elem_bytes": elem_bytes, "elem_bits": (elem_bytes * 8) if elem_bytes else None,
            "moved_block_bytes": None, "loops": None, "M": None, "N": None, "kind": None}
    if is_reshape_perm(perm, dims):
        plan["path"] = "reshape_copy"
        return _with_dispatches(plan)
    sa = single_axis_move(perm)
    if sa is None:
        plan["path"] = "untyped_serial"
        return _with_dispatches(plan)
    kind, frm, to = sa
    total = math.prod(dims)
    if kind == "out":
        loops, block = math.prod(dims[:to]), math.prod(dims[frm + 1:])
        per_loop = total // loops // block
        m, n = per_loop // dims[frm], dims[frm]
    else:
        loops, block = math.prod(dims[:frm]), math.prod(dims[to + 1:])
        per_loop = total // loops // block
        m, n = dims[frm], per_loop // dims[frm]
    nbytes = block * elem_bytes
    plan.update({"kind": kind, "moved_block_bytes": nbytes, "loops": loops, "M": m, "N": n})
    on_old = nbytes in MLAS_BLOCK_BYTES[DUMP_ORT_VERSION]
    on_new = nbytes in MLAS_BLOCK_BYTES["1.24.4"]
    plan["path"] = ("mlas_single_axis_both" if on_old and on_new
                    else "mlas_single_axis_new" if on_new
                    else "scalar_single_axis")
    return _with_dispatches(plan)


def _with_dispatches(plan):
    """Thread-pool dispatches per inference at FLAG_RULE["threads"], per release.

    MlasTranspose posts one MlasExecuteThreaded per outer loop; MlasExecuteThreaded only posts when
    its iteration count min(DegreeOfParallelism, M) > 1, and with 1 thread there is no pool at all.
    """
    threads = FLAG_RULE["threads"]
    for ver in (DUMP_ORT_VERSION, "1.24.4"):
        on_mlas = plan["path"].startswith("mlas") and (
            plan["moved_block_bytes"] in MLAS_BLOCK_BYTES[ver])
        plan[f"dispatches_{ver}"] = (plan["loops"] if on_mlas and threads > 1
                                     and (plan["M"] or 0) > 1 else 0)
    plan["added_dispatches"] = plan["dispatches_1.24.4"] - plan[f"dispatches_{DUMP_ORT_VERSION}"]
    return plan


def path_string(plan, version):
    """The human-readable path label, in the same wording the Experiment 27 manifest recorded."""
    if plan["path"] == "reshape_copy":
        return "reshape-copy"
    if plan["path"] == "untyped_serial":
        return "untyped (serial)"
    nbytes = plan["moved_block_bytes"]
    if nbytes in MLAS_BLOCK_BYTES[version]:
        return (f"MlasTranspose<{8 * nbytes}-bit> {plan['kind']} "
                f"loops={plan['loops']} M={plan['M']} N={plan['N']}")
    return f"scalar {plan['kind']} {8 * nbytes}-bit loops={plan['loops']}"


# ------------------------------------------------------------------------------------------------
# Reading the executed graph (needs the `onnx` package; never imports onnxruntime)
# ------------------------------------------------------------------------------------------------
# Standard-op stand-ins for the contrib ops an ORT-optimized graph contains, so that onnx shape
# inference can run through it. Shapes and dtypes only: these are NOT semantic equivalents.
# The QLinear*/nchwc set is taken from scripts/paper/transpose_probe.py, where it was checked
# against the 1.23.2 and 1.24.4 dumps of the reference file (`--check-dump`).
PROXY_KNOWN = {
    "QLinearAdd", "QLinearMul", "QLinearSigmoid", "QLinearSoftmax", "QuickGelu", "Gelu",
    "FastGelu", "BiasGelu", "QLinearConcat", "QLinearGlobalAveragePool", "QGemm", "QLinearConv",
    "FusedConv", "FusedGemm", "MatMulIntegerToFloat", "DynamicQuantizeMatMul",
    "SkipLayerNormalization", "Conv", "ReorderInput", "ReorderOutput",
}


def _attr(node, key, default=None):
    from onnx import helper
    for a in node.attribute:
        if a.name == key:
            return helper.get_attribute_value(a)
    return default


def build_proxy(model, assume_batch):
    """Replace contrib ops with shape-equivalent standard ops; drop the ones we do not model.

    Returns (proxy_model, opaque_ops, assumed_dims). A dropped node leaves its outputs without a
    type, so every Transpose downstream of it is reported as UNKNOWN rather than guessed.
    ORT writes a STALE value_info into the dump (pre-optimization tensor names); it is deleted here
    and every shape is re-derived, because matching the dump's nodes to it gives wrong shapes.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    new, extra, opaque = [], [], collections.Counter()
    k = [0]

    def t(p):
        k[0] += 1
        return f"__px_{p}_{k[0]}"

    for n in model.graph.node:
        op, dom, o = n.op_type, n.domain, n.output
        contrib = dom not in ("", "ai.onnx")
        if not contrib:
            c = onnx.NodeProto()
            c.CopyFrom(n)
            new.append(c)
            continue
        if op in ("QLinearAdd", "QLinearMul"):
            new.append(helper.make_node(op[7:], [n.input[0], n.input[3]], [o[0]]))
        elif op in ("QLinearSigmoid", "QLinearSoftmax", "QuickGelu", "Gelu", "FastGelu",
                    "SkipLayerNormalization"):
            new.append(helper.make_node("Identity", [n.input[0]], [o[0]]))
        elif op == "BiasGelu":
            new.append(helper.make_node("Add", [n.input[0], n.input[1]], [o[0]]))
        elif op in ("MatMulIntegerToFloat", "DynamicQuantizeMatMul"):
            a, b = t("m"), t("m")
            new += [helper.make_node("Cast", [n.input[0]], [a], to=TensorProto.FLOAT),
                    helper.make_node("Cast", [n.input[1]], [b], to=TensorProto.FLOAT),
                    helper.make_node("MatMul", [a, b], [o[0]])]
        elif op == "QLinearConcat":
            new.append(helper.make_node("Concat", [n.input[i] for i in range(2, len(n.input), 3)],
                                        [o[0]], axis=_attr(n, "axis")))
        elif op == "QLinearGlobalAveragePool":
            a, b = t("g"), t("g")
            seq = [helper.make_node("Cast", [n.input[0]], [a], to=TensorProto.FLOAT)]
            if _attr(n, "channels_last", 0):
                c = t("g")
                seq.append(helper.make_node("Transpose", [a], [c], perm=[0, 3, 1, 2]))
                a = c
            seq.append(helper.make_node("GlobalAveragePool", [a], [b]))
            if _attr(n, "channels_last", 0):
                c = t("g")
                seq.append(helper.make_node("Transpose", [b], [c], perm=[0, 2, 3, 1]))
                b = c
            seq.append(helper.make_node("Cast", [b], [o[0]], to=TensorProto.UINT8))
            new += seq
        elif op in ("QGemm", "FusedGemm"):
            q = op == "QGemm"
            a, b, c = t("q"), t("q"), t("q")
            bi = n.input[3] if q else n.input[1]
            qo = q and len(n.input) > 7 and n.input[7] != ""
            new += [helper.make_node("Cast", [n.input[0]], [a], to=TensorProto.FLOAT),
                    helper.make_node("Cast", [bi], [b], to=TensorProto.FLOAT),
                    helper.make_node("Gemm", [a, b], [c if qo else o[0]],
                                     transA=_attr(n, "transA", 0), transB=_attr(n, "transB", 0))]
            if qo:
                new.append(helper.make_node("Cast", [c], [o[0]], to=TensorProto.UINT8))
        elif op in ("QLinearConv", "FusedConv") or (dom == "com.microsoft.nchwc" and op == "Conv"):
            q = op == "QLinearConv"
            w = n.input[3] if q else n.input[1]
            cl = _attr(n, "channels_last", 0) if q else 0
            kw = {kk: _attr(n, kk) for kk in ("strides", "pads", "dilations", "group", "kernel_shape")
                  if _attr(n, kk) is not None}
            xf, wf, y = t("c"), t("c"), t("c")
            seq = [helper.make_node("Cast", [n.input[0]], [xf], to=TensorProto.FLOAT),
                   helper.make_node("Cast", [w], [wf], to=TensorProto.FLOAT)]
            if cl:
                xt = t("c")
                seq.append(helper.make_node("Transpose", [xf], [xt], perm=[0, 3, 1, 2]))
                xf = xt
            seq.append(helper.make_node("Conv", [xf, wf], [y], **kw))
            if cl:
                yt = t("c")
                seq.append(helper.make_node("Transpose", [y], [yt], perm=[0, 2, 3, 1]))
                y = yt
            seq.append(helper.make_node("Cast", [y], [o[0]],
                                        to=TensorProto.UINT8 if q else TensorProto.FLOAT))
            new += seq
        elif dom == "com.microsoft.nchwc" and op == "ReorderInput":
            new.append(helper.make_node("Transpose", [n.input[0]], [o[0]], perm=[0, 3, 1, 2])
                       if _attr(n, "channels_last", 0)
                       else helper.make_node("Identity", [n.input[0]], [o[0]]))
        elif dom == "com.microsoft.nchwc" and op == "ReorderOutput":
            s, e, a = t("r"), t("r"), t("r")
            extra += [numpy_helper.from_array(np.array([0], np.int64), s),
                      numpy_helper.from_array(np.array([_attr(n, "channels")], np.int64), e),
                      numpy_helper.from_array(np.array([1], np.int64), a)]
            if _attr(n, "channels_last", 0):
                c = t("r")
                new += [helper.make_node("Slice", [n.input[0], s, e, a], [c]),
                        helper.make_node("Transpose", [c], [o[0]], perm=[0, 2, 3, 1])]
            else:
                new.append(helper.make_node("Slice", [n.input[0], s, e, a], [o[0]]))
        else:
            opaque[f"{dom}:{op}"] += 1        # dropped: downstream shapes become UNKNOWN

    p = onnx.ModelProto()
    p.CopyFrom(model)
    del p.graph.node[:]
    p.graph.node.extend(new)
    p.graph.initializer.extend(extra)
    del p.graph.value_info[:]                 # stale in ORT dumps; re-derive everything
    assumed = []
    for i in p.graph.input:
        d = i.type.tensor_type.shape.dim
        for j, dim in enumerate(d):
            if not dim.HasField("dim_value"):
                if j == 0 and assume_batch is not None:
                    assumed.append(f"{i.name}[0]={assume_batch}"
                                   + (f" (was '{dim.dim_param}')" if dim.dim_param else ""))
                    dim.ClearField("dim_param")
                    dim.dim_value = assume_batch
                else:
                    assumed.append(f"{i.name}[{j}]=UNKNOWN"
                                   + (f" ('{dim.dim_param}')" if dim.dim_param else ""))
    keep = [x for x in p.opset_import if x.domain in ("", "ai.onnx")]
    del p.opset_import[:]
    p.opset_import.extend(keep)
    return p, dict(sorted(opaque.items())), assumed


def executed_transposes(dump_path, assume_batch=1):
    """[(node_index, name, in_shape|None, perm|None, elem_type)] for EVERY Transpose in the dump."""
    import onnx
    from onnx import TensorProto, shape_inference

    model = onnx.load(str(dump_path))
    proxy, opaque, assumed = build_proxy(model, assume_batch)
    types = {}
    try:
        inf = shape_inference.infer_shapes(proxy, strict_mode=False, data_prop=True)
        graph = inf.graph
    except Exception as exc:                                  # noqa: BLE001 - report, never guess
        print(f"  ! shape inference failed ({type(exc).__name__}: {exc}); shapes UNKNOWN")
        graph = proxy.graph
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        tt = vi.type.tensor_type
        types[vi.name] = (tt.elem_type,
                          [d.dim_value if d.HasField("dim_value") else None for d in tt.shape.dim])
    for init in graph.initializer:
        types.setdefault(init.name, (init.data_type, list(init.dims)))

    rows = []
    for idx, n in enumerate(model.graph.node):
        if n.op_type != "Transpose":
            continue
        et, shape = types.get(n.input[0], (0, None))
        perm = _attr(n, "perm")
        perm = list(perm) if perm is not None else (
            list(reversed(range(len(shape)))) if shape is not None else None)
        rows.append({"node_index": idx, "name": n.name, "shape": shape, "perm": perm,
                     "elem_type": et,
                     "elem_type_name": TensorProto.DataType.Name(et) if et else "UNKNOWN"})
    n_total = sum(1 for n in model.graph.node if n.op_type == "Transpose")
    return rows, {"opaque_ops": opaque, "assumed_dims": assumed, "n_transposes": n_total,
                  "n_nodes": len(model.graph.node)}


def analyse_model(dump_path, assume_batch=1, label=None):
    """Full per-model analysis: node rows (8-bit Transposes only) plus a summary dict."""
    rows, meta = executed_transposes(dump_path, assume_batch)
    node_rows, unresolved, flagged, disp = [], 0, 0, {DUMP_ORT_VERSION: 0, "1.24.4": 0}
    for r in rows:
        shape, perm = r["shape"], r["perm"]
        eb = ELEM_BYTES.get(r["elem_type"])
        # An empty shape list is NOT a known shape: a rank-0 record with a rank-4 perm appears in
        # third-party exports (seen in shufflenet-v2-12-qdq and resnet50-v1-12-qdq), and indexing
        # dims[p] with it used to raise IndexError instead of reporting the node as unresolved.
        known = (shape is not None and perm is not None and eb is not None
                 and None not in shape and len(shape) > max(perm, default=-1))
        if not known:
            unresolved += 1
            node_rows.append({**r, "resolved": False, "plan": None})
            continue
        plan = transpose_plan(list(shape), list(perm), eb)
        for v in disp:
            disp[v] += plan[f"dispatches_{v}"]
        if plan["elem_bits"] != FLAG_RULE["element_bits"]:
            continue                                   # not 8-bit: counted for dispatches, not listed
        flagged += 1 if is_flagged(plan) else 0
        node_rows.append({**r, "resolved": True, "plan": plan})
    eight_bit = [x for x in node_rows if x["resolved"]]
    added = disp["1.24.4"] - disp[DUMP_ORT_VERSION]
    summary = {
        "model": label or Path(dump_path).name,
        "dump": str(dump_path),
        "n_nodes_executed": meta["n_nodes"],
        "n_transposes_executed": meta["n_transposes"],
        "n_8bit_transposes": len(eight_bit),
        "n_flagged": flagged,
        "n_unresolved_transposes": unresolved,
        f"dispatches_{DUMP_ORT_VERSION}_{FLAG_RULE['threads']}T": disp[DUMP_ORT_VERSION],
        f"dispatches_1.24.4_{FLAG_RULE['threads']}T": disp["1.24.4"],
        "added_dispatches": added,
        "prediction": model_prediction(added),
        "opaque_ops": ";".join(f"{k}x{v}" for k, v in meta["opaque_ops"].items()),
        "assumed_dims": ";".join(meta["assumed_dims"]),
    }
    return node_rows, summary


# ------------------------------------------------------------------------------------------------
# Step 1: dump the executed graph (onnxruntime only, no onnx package needed)
# ------------------------------------------------------------------------------------------------
def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def dump_name(src, version):
    """Unique dump file name for a source path (many projects call every file p00_int8.onnx)."""
    src = Path(src).resolve()
    tag = hashlib.sha1(str(src).lower().encode("utf-8")).hexdigest()[:8]
    parent = src.parent.parent.name or src.parent.name or "src"
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in f"{parent}_{src.stem}")
    return f"opt_{version}_{safe}_{tag}.onnx"


def cmd_dump(args):
    import onnxruntime as ort
    ver = ort.__version__
    if ver != DUMP_ORT_VERSION and not args.allow_ort_version:
        sys.exit(f"ERROR: this interpreter has onnxruntime {ver}, the dump must come from "
                 f"{DUMP_ORT_VERSION} (use {DEFAULT_DUMP_PYTHON}), or pass --allow-ort-version.")
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    free = {}
    for item in args.free_dim or []:
        name, _, val = item.partition("=")
        free[name] = int(val)
    for src in args.models:
        src = Path(src)
        out = work / dump_name(src, ver)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.optimized_model_filepath = str(out)
        so.intra_op_num_threads = args.intra_op
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        for name, val in free.items():
            so.add_free_dimension_override_by_name(name, val)
        ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])
        side = out.with_suffix(".provenance.json")
        side.write_text(json.dumps({
            "source": str(src), "source_sha256": sha256(src), "dump": str(out),
            "dump_sha256": sha256(out), "onnxruntime": ver, "intra_op_num_threads": args.intra_op,
            "graph_optimization_level": "ORT_ENABLE_ALL", "free_dim_overrides": free,
            "tool_sha256": sha256(Path(__file__)),
        }, indent=2), encoding="utf-8")
        print(f"dumped {src} -> {out}  (onnxruntime {ver})")
    return 0


# ------------------------------------------------------------------------------------------------
# Step 2: analyse (onnx only)
# ------------------------------------------------------------------------------------------------
NODE_FIELDS = ["model", "node_index", "node_name", "input_shape", "perm", "elem_type",
               "moved_block_bytes", "code_path", "loops", "M", "N",
               f"path_{DUMP_ORT_VERSION}", "path_1.24.4",
               f"dispatches_{DUMP_ORT_VERSION}", "dispatches_1.24.4", "flagged"]
SUMMARY_FIELDS = ["model", "n_nodes_executed", "n_transposes_executed", "n_8bit_transposes",
                  "n_flagged", "n_unresolved_transposes",
                  f"dispatches_{DUMP_ORT_VERSION}_{FLAG_RULE['threads']}T",
                  f"dispatches_1.24.4_{FLAG_RULE['threads']}T", "added_dispatches", "prediction",
                  "opaque_ops", "assumed_dims", "dump"]


def node_csv_row(model, r):
    p = r["plan"]
    if p is None:
        return {"model": model, "node_index": r["node_index"], "node_name": r["name"],
                "input_shape": "UNKNOWN" if r["shape"] is None else str(r["shape"]),
                "perm": "UNKNOWN" if r["perm"] is None else str(r["perm"]),
                "elem_type": r["elem_type_name"], "moved_block_bytes": "UNKNOWN",
                "code_path": "UNRESOLVED", "loops": "", "M": "", "N": "",
                f"path_{DUMP_ORT_VERSION}": "UNKNOWN", "path_1.24.4": "UNKNOWN",
                f"dispatches_{DUMP_ORT_VERSION}": "", "dispatches_1.24.4": "", "flagged": "unknown"}
    return {"model": model, "node_index": r["node_index"], "node_name": r["name"],
            "input_shape": str(r["shape"]), "perm": str(r["perm"]),
            "elem_type": r["elem_type_name"], "moved_block_bytes": p["moved_block_bytes"],
            "code_path": p["path"], "loops": p["loops"], "M": p["M"], "N": p["N"],
            f"path_{DUMP_ORT_VERSION}": path_string(p, DUMP_ORT_VERSION),
            "path_1.24.4": path_string(p, "1.24.4"),
            f"dispatches_{DUMP_ORT_VERSION}": p[f"dispatches_{DUMP_ORT_VERSION}"],
            "dispatches_1.24.4": p["dispatches_1.24.4"],
            "flagged": "yes" if is_flagged(p) else "no"}


def provenance_of(dump):
    side = Path(dump).with_suffix(".provenance.json")
    if not side.exists():
        return None
    try:
        return json.loads(side.read_text(encoding="utf-8"))
    except Exception:                                          # noqa: BLE001
        return None


def write_csvs(prefix, all_nodes, summaries):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    nodes_p = prefix.with_name(prefix.name + "_nodes.csv")
    summ_p = prefix.with_name(prefix.name + "_summary.csv")
    with open(nodes_p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, NODE_FIELDS)
        w.writeheader()
        w.writerows(all_nodes)
    with open(summ_p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(summaries)
    return nodes_p, summ_p


def run_analysis(dumps, assume_batch, labels=None):
    all_nodes, summaries, details = [], [], {}
    for i, d in enumerate(dumps):
        label = (labels or {}).get(str(d)) or Path(d).stem
        prov = provenance_of(d)
        if prov is None:
            print(f"WARNING: {d} has no provenance sidecar; cannot confirm the dump came from "
                  f"onnxruntime {DUMP_ORT_VERSION}.")
        elif prov.get("onnxruntime") != DUMP_ORT_VERSION:
            print(f"WARNING: {d} was dumped by onnxruntime {prov.get('onnxruntime')}, "
                  f"not {DUMP_ORT_VERSION}.")
        rows, summary = analyse_model(d, assume_batch, label)
        summary["source_sha256"] = (prov or {}).get("source_sha256", "")
        all_nodes += [node_csv_row(label, r) for r in rows]
        summaries.append(summary)
        details[label] = rows
    return all_nodes, summaries, details


def print_summaries(summaries):
    t = FLAG_RULE["threads"]
    old_key = "dispatches_{}_{}T".format(DUMP_ORT_VERSION, t)
    new_key = "dispatches_1.24.4_{}T".format(t)
    print()
    print("{:<52} {:>8} {:>8} {:>6} {:>11} {:>11} {:>7}  {}".format(
        "model", "8bit Tr", "flagged", "unres", "disp " + DUMP_ORT_VERSION, "disp 1.24.4",
        "added", "prediction"))
    for s in summaries:
        print("{:<52} {:>8} {:>8} {:>6} {:>11} {:>11} {:>7}  {}".format(
            s["model"][:52], s["n_8bit_transposes"], s["n_flagged"],
            s["n_unresolved_transposes"], s[old_key], s[new_key], s["added_dispatches"],
            s["prediction"]))
        if s["opaque_ops"]:
            print(f"{'':<52} opaque ops (shapes unknown downstream): {s['opaque_ops']}")
        if s["assumed_dims"]:
            print(f"{'':<52} dim assumptions: {s['assumed_dims']}")


def cmd_compare(args):
    """Same source model, dumped by both releases: is the EXECUTED graph the same graph?

    The prediction assumes 1.24.4 executes the same Transposes as 1.23.2 and only dispatches them
    differently (true for the reference file, Experiment 23a). For an unseen model that has to be
    checked, or a difference in the optimizer would be mistaken for the dispatch effect.
    """
    old, new = Path(args.dump_old), Path(args.dump_new)
    po, pn = provenance_of(old), provenance_of(new)
    for p, d, want in ((po, old, DUMP_ORT_VERSION), (pn, new, None)):
        if p is None:
            print(f"WARNING: {d} has no provenance sidecar.")
        elif want and p.get("onnxruntime") != want:
            print(f"WARNING: {d} was dumped by onnxruntime {p.get('onnxruntime')}, not {want}.")
    if po and pn and po.get("source_sha256") != pn.get("source_sha256"):
        print("WARNING: the two dumps do not come from the same source file "
              f"({po.get('source_sha256', '')[:12]} vs {pn.get('source_sha256', '')[:12]}).")
    ro, _ = executed_transposes(old, args.assume_batch)
    rn, _ = executed_transposes(new, args.assume_batch)

    def key(r):
        return (r["node_index"], r["name"], str(r["shape"]), str(r["perm"]), r["elem_type_name"])

    same = [key(r) for r in ro] == [key(r) for r in rn]
    print(f"[{'PASS' if same else 'FAIL'}] executed Transpose tables identical across the two "
          f"dumps ({len(ro)} vs {len(rn)} Transposes)")
    if not same:
        ko, kn = {k[0]: k for k in map(key, ro)}, {k[0]: k for k in map(key, rn)}
        for i in sorted(set(ko) | set(kn)):
            if ko.get(i) != kn.get(i):
                print(f"       node {i}: {po and 'old'} {ko.get(i)}  !=  new {kn.get(i)}")
        print("       the prediction does not apply unchanged: the 1.24.4 optimizer produced a "
              "different graph, so a measured difference would not be the dispatch change alone.")
    return 0 if same else 1


def cmd_analyse(args):
    all_nodes, summaries, _ = run_analysis(args.dumps, args.assume_batch)
    print_summaries(summaries)
    if args.csv:
        n, s = write_csvs(args.csv, all_nodes, summaries)
        print(f"\nwrote {n}\nwrote {s}")
    return 0


# ------------------------------------------------------------------------------------------------
# validate: re-derive the reference file's table and diff it against the recorded manifest
# ------------------------------------------------------------------------------------------------
VALIDATION_MODELS = {
    "mobilevit_xxs_64_int8 (steps)": ROOT / "results/classification_mobilevit/models/p00_int8.onnx",
    "deit_tiny_int8 (control)": ROOT / "results/classification_deit/models/p00_int8.onnx",
    "mobilenetv2_int8 (control)": ROOT / "results/classification/models/p00_int8.onnx",
}
REFERENCE_KEY = "mobilevit_xxs_64_int8 (steps)"


def cmd_validate(args):
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    models = {k: Path(v) for k, v in VALIDATION_MODELS.items()}
    missing = [str(v) for v in models.values() if not v.exists()]
    if missing:
        sys.exit("ERROR: missing local model(s): " + ", ".join(missing))

    dumps, labels = [], {}
    need = [v for v in models.values() if not (work / dump_name(v, DUMP_ORT_VERSION)).exists()]
    if need:
        py = Path(args.dump_python)
        if not py.exists():
            sys.exit(f"ERROR: no dumps in {work} and no ORT {DUMP_ORT_VERSION} interpreter at {py}.\n"
                     f"Run step 1 yourself:\n  {py} {Path(__file__)} dump --work-dir {work} "
                     + " ".join(str(v) for v in need))
        cmd = [str(py), str(Path(__file__).resolve()), "dump", "--work-dir", str(work)] + [str(v) for v in need]
        print("step 1 (ORT %s): %s" % (DUMP_ORT_VERSION, " ".join(cmd)))
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip())
        if r.returncode != 0:
            sys.exit(f"ERROR: dump step failed ({r.returncode}).")
    for k, v in models.items():
        d = work / dump_name(v, DUMP_ORT_VERSION)
        dumps.append(d)
        labels[str(d)] = k

    all_nodes, summaries, details = run_analysis(dumps, args.assume_batch, labels)
    print_summaries(summaries)

    ok = True
    print("\n" + "=" * 96)
    print("VALIDATION against results/transpose_probe/manifest.json (Experiment 27/27b ground truth)")
    print("=" * 96)
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    want_all = man["variants"]["tp_u8_all"]["transpose_paths_4T"]
    want_w2 = man["variants"]["tp_u8_w2"]["transpose_paths_4T"]
    want_disp = man["variants"]["tp_u8_all"]["predicted_dispatches"]
    ref_rows = [r for r in details[REFERENCE_KEY] if r["resolved"]]
    ref_sum = [s for s in summaries if s["model"] == REFERENCE_KEY][0]

    # (a) count of 8-bit Transposes
    n_ok = ref_sum["n_8bit_transposes"] == len(want_all) == 56
    ok &= n_ok
    print(f"[{'PASS' if n_ok else 'FAIL'}] 8-bit Transposes in the executed graph: "
          f"computed {ref_sum['n_8bit_transposes']}, manifest {len(want_all)}, expected 56")

    # (b) node-for-node: index, shape, perm and BOTH release paths, all 56 rows
    diffs = []
    for got, want in zip(ref_rows, want_all):
        p = got["plan"]
        g = (got["node_index"], list(got["shape"]), list(got["perm"]),
             path_string(p, DUMP_ORT_VERSION), path_string(p, "1.24.4"))
        w = (want["idx"], list(want["shape"]), list(want["perm"]),
             want[DUMP_ORT_VERSION], want["1.24.4"])
        if g != w:
            diffs.append((got["name"], g, w))
    rows_ok = len(ref_rows) == len(want_all) and not diffs
    ok &= rows_ok
    print(f"[{'PASS' if rows_ok else 'FAIL'}] all {len(want_all)} rows agree on "
          f"(node index, input shape, perm, {DUMP_ORT_VERSION} path, 1.24.4 path)")
    for name, g, w in diffs[:20]:
        print(f"       DISAGREEMENT at node '{name}'\n         computed: {g}\n         manifest: {w}")

    # (c) the flagged set is exactly the four recorded 2-byte-block nodes
    got_flag = [(r["node_index"], list(r["shape"]), list(r["perm"]), r["name"])
                for r in ref_rows if is_flagged(r["plan"])]
    want_flag = [(w["idx"], list(w["shape"]), list(w["perm"])) for w in want_w2]
    flag_ok = [g[:3] for g in got_flag] == want_flag and len(got_flag) == 4
    ok &= flag_ok
    print(f"[{'PASS' if flag_ok else 'FAIL'}] flagged set == the four manifest tp_u8_w2 nodes "
          f"(computed {len(got_flag)}, manifest {len(want_flag)})")
    for idx, shape, perm, name in got_flag:
        p = [r for r in ref_rows if r["node_index"] == idx][0]["plan"]
        print(f"       flagged: idx {idx:>4} {name}  shape={shape} perm={perm} "
              f"block={p['moved_block_bytes']}B loops={p['loops']} M={p['M']} N={p['N']} "
              f"-> +{p['added_dispatches']} dispatches")
    if not flag_ok:
        print(f"       computed: {[g[:3] for g in got_flag]}\n       manifest: {want_flag}")

    # (d) total dispatches per release, against the manifest's registered prediction
    t = str(FLAG_RULE["threads"])
    got_d = {DUMP_ORT_VERSION: ref_sum[f"dispatches_{DUMP_ORT_VERSION}_{t}T"],
             "1.24.4": ref_sum[f"dispatches_1.24.4_{t}T"]}
    want_d = {v: want_disp[v][t] for v in (DUMP_ORT_VERSION, "1.24.4")}
    d_ok = got_d == want_d
    ok &= d_ok
    print(f"[{'PASS' if d_ok else 'FAIL'}] thread-pool dispatches at {t}T: computed {got_d}, "
          f"manifest {want_d} (added {got_d['1.24.4'] - got_d[DUMP_ORT_VERSION]})")

    # (e) predictions, against what the measurements on record say each file does
    expected = {REFERENCE_KEY: "steps", "deit_tiny_int8 (control)": "does not step",
                "mobilenetv2_int8 (control)": "does not step"}
    for s in summaries:
        good = s["prediction"] == expected[s["model"]]
        ok &= good
        print(f"[{'PASS' if good else 'FAIL'}] {s['model']}: predicted \"{s['prediction']}\", "
              f"measured record says \"{expected[s['model']]}\" "
              f"({s['n_flagged']} flagged, {s['added_dispatches']} added dispatches, "
              f"{s['n_unresolved_transposes']} unresolved)")

    print("=" * 96)
    print(f"VALIDATION: {'PASS' if ok else 'FAIL'}")
    if args.csv:
        n, s = write_csvs(args.csv, all_nodes, summaries)
        print(f"wrote {n}\nwrote {s}")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__ % {"threads": FLAG_RULE["threads"], "dumpver": DUMP_ORT_VERSION},
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dump", help=f"step 1: write executed graphs (RUN WITH ORT {DUMP_ORT_VERSION})")
    d.add_argument("models", nargs="+")
    d.add_argument("--work-dir", required=True)
    d.add_argument("--intra-op", type=int, default=FLAG_RULE["threads"])
    d.add_argument("--free-dim", action="append", metavar="NAME=VALUE",
                   help="pin a symbolic input dimension before optimization (repeatable)")
    d.add_argument("--allow-ort-version", action="store_true",
                   help="dump with an onnxruntime other than %s (recorded in the sidecar)" % DUMP_ORT_VERSION)
    d.set_defaults(func=cmd_dump)

    a = sub.add_parser("analyse", help="step 2: read the dumps and predict (RUN WITH THE PROJECT VENV)")
    a.add_argument("dumps", nargs="+")
    a.add_argument("--csv", help="output prefix; writes <prefix>_nodes.csv and <prefix>_summary.csv")
    a.add_argument("--assume-batch", type=int, default=1,
                   help="value assumed for a symbolic leading input dimension (default 1); "
                        "pass -1 to assume nothing and report those Transposes as unresolved")
    a.set_defaults(func=cmd_analyse)

    c = sub.add_parser("compare", help="same model dumped by both releases: is the executed graph "
                                       "the same? (run with the project venv)")
    c.add_argument("dump_old", help=f"dump made by onnxruntime {DUMP_ORT_VERSION}")
    c.add_argument("dump_new", help="dump made by onnxruntime 1.24.4")
    c.add_argument("--assume-batch", type=int, default=1)
    c.set_defaults(func=cmd_compare)

    v = sub.add_parser("validate", help="check the tool against the recorded Experiment 27/27b table")
    v.add_argument("--work-dir", required=True)
    v.add_argument("--dump-python", default=str(DEFAULT_DUMP_PYTHON))
    v.add_argument("--csv")
    v.add_argument("--assume-batch", type=int, default=1)
    v.set_defaults(func=cmd_validate)

    args = ap.parse_args(argv)
    if getattr(args, "assume_batch", 1) is not None and getattr(args, "assume_batch", 1) < 0:
        args.assume_batch = None
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
