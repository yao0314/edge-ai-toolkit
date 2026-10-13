"""Experiment 27: is the 1.24.4 step of static MobileViT-XXS the 8-bit Transpose? (isolation probe)

The static MobileViT-XXS file (64 px) runs +31% slower at 4 threads on ONNX Runtime 1.24.4 than on
1.23.2 (none at 1 thread), on a node-for-node identical executed graph (23a), with extra parallel
work (23b, dC/dW = 3.1). The per-op profile (23c) put 62% of the delta on Transpose, below the
registered 70% threshold, so the registered verdict is "thread-pool-level". Experiment 24 found no
step in the MatMul-only static arms, whose Transposes stay in float.

This script builds micro models that isolate the executed graph's Transposes. Every model has ONE
float32 input of shape 1x3x64x64 (what ort_version_bisect.py feeds), so the existing harness times
them unchanged. Building uses onnx only; no onnxruntime import happens in this process.

Source-level prediction (desk research 2026-09-19, see results/transpose_probe/manifest.json):
between v1.23.2 and v1.24.4 the CPU Transpose kernel, MLAS transpose (x86 part) and the intra-op
thread pool are unchanged, except that commit 42869ca (PR #25680) added
`has_mlas_transpose<uint16_t>` in core/framework/transpose_helper.cc. A single-axis transpose whose
moved block is 2 bytes wide then calls MlasTranspose (multi-threaded, one MlasExecuteThreaded per
outer loop) instead of a scalar loop. In the static graph, four 8-bit Transposes have 2-byte blocks
([256,2,4,2], [256,4,2,2], [160,2,2,2] x2, perm [0,2,1,3]); at 4 threads they issue 832 thread-pool
dispatches per inference on 1.24.4 and none on 1.23.2. The float copies of the same Transposes have
8-byte blocks and stay scalar on both releases.

Variants (all under results/transpose_probe/models/):
  tp_u8_all    the static graph's 56 executed uint8 Transposes (shapes, perms, dtype), Q ... DQ wrapper
  tp_u8_w2     the same skeleton, only the 4 Transposes with 2-byte blocks
  tp_u8_rest   the same skeleton, the other 52 Transposes
  tp_u8_ctrl   the same skeleton, no Transpose (wrapper, slices, reshapes, concat only)
  tp_f32_all   the FP32 graph's 48 executed float Transposes in the same (float) skeleton
  tp_f32_ctrl  the float skeleton, no Transpose
  tp_f16_w2    4 float16 Transposes with 2-byte elements and the same loop structure as tp_u8_w2
  tp_f32_w4    4 float32 Transposes with 4-byte elements, same loop structure (MLAS path on BOTH releases)

    python scripts/paper/transpose_probe.py                      # build + check + print the command
    python scripts/paper/transpose_probe.py --check-dump INT8_OPT.onnx FP32_OPT.onnx
    python scripts/paper/transpose_probe.py --verify-executed    # RUNNER, before timing (uses side venvs)
    python scripts/paper/transpose_probe.py --analyse results/paper/transpose_probe.csv [more.csv]

The decision rule is registered in the design note and applied mechanically by --analyse.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

ROOT = Path(__file__).resolve().parents[2]
PROBE_DIR = ROOT / "results" / "transpose_probe"
MODEL_DIR = PROBE_DIR / "models"
SCRATCH = ("C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Desktop-SEC/"
           "103ae351-d709-4d02-91fc-0b8da660fe58/scratchpad")
VERSIONS = ("1.23.2", "1.24.4")
INPUT_SHAPE = [1, 3, 64, 64]
SEED = 27

# Reference step of the full static model at 4 threads, 1.24.4 - 1.23.2 (ms). Primary: medians of
# ort_version_bisect.csv (2.037 - 1.553). The other two observations are reported as sensitivity.
S_REF = 0.48
S_SENS = {"ort_version_bisect.csv": 0.484, "spin_version.csv (spin on)": 0.469,
          "coverage_mobilevit.csv": 0.570}
FLOOR = 0.07  # between-session latency floor used by every registered rule

# Every Transpose of the executed static graph (opt_1.23.2_p00_int8.onnx, node order), input shape
# with the batch pinned to 1 (proxy shape inference through the contrib ops; all inputs are UINT8),
# perm, and whether the executed FP32 graph (opt_1.23.2_p00_fp32.onnx) holds the same (shape, perm)
# as a FLOAT Transpose. The 8 False rows are the NHWC layout Transposes around QLinearConv.
# Re-derive with --check-dump.
TRANSPOSES = [
    (1, 'Transpose_token_191', [1, 3, 64, 64], [0, 2, 3, 1], False),
    (47, 'Transpose_token_258', [1, 8, 8, 64], [0, 3, 1, 2], False),
    (49, '/stages/stages.2/stages.2.1/Transpose', [256, 2, 4, 2], [0, 2, 1, 3], True),
    (60, '/stages/stages.2/stages.2.1/Transpose_1', [1, 64, 16, 4], [0, 3, 2, 1], True),
    (84, 'Transpose_token_357', [4, 16, 4, 16], [0, 2, 1, 3], True),
    (86, 'Transpose_token_355', [4, 16, 4, 16], [0, 2, 1, 3], True),
    (98, '/stages/stages.2/stages.2.1/transformer/transformer.0/attn/Transpose_1', [4, 16, 4, 16], [0, 2, 3, 1], True),
    (103, '/stages/stages.2/stages.2.1/transformer/transformer.0/attn/Transpose_2', [4, 4, 16, 16], [0, 2, 1, 3], True),
    (140, 'Transpose_token_361', [4, 16, 4, 16], [0, 2, 1, 3], True),
    (142, 'Transpose_token_359', [4, 16, 4, 16], [0, 2, 1, 3], True),
    (154, '/stages/stages.2/stages.2.1/transformer/transformer.1/attn/Transpose_1', [4, 16, 4, 16], [0, 2, 3, 1], True),
    (159, '/stages/stages.2/stages.2.1/transformer/transformer.1/attn/Transpose_2', [4, 4, 16, 16], [0, 2, 1, 3], True),
    (184, '/stages/stages.2/stages.2.1/Transpose_2', [1, 4, 16, 64], [0, 3, 2, 1], True),
    (186, '/stages/stages.2/stages.2.1/Transpose_3', [256, 4, 2, 2], [0, 2, 1, 3], True),
    (189, 'Transpose_token_261', [1, 64, 8, 8], [0, 2, 3, 1], False),
    (208, 'Transpose_token_285', [1, 4, 4, 80], [0, 3, 1, 2], False),
    (220, '/stages/stages.3/stages.3.1/Transpose', [160, 2, 2, 2], [0, 2, 1, 3], True),
    (225, '/stages/stages.3/stages.3.1/Transpose_1', [1, 80, 4, 4], [0, 3, 2, 1], True),
    (249, 'Transpose_token_376', [4, 4, 4, 20], [0, 2, 1, 3], True),
    (251, 'Transpose_token_374', [4, 4, 4, 20], [0, 2, 1, 3], True),
    (263, '/stages/stages.3/stages.3.1/transformer/transformer.0/attn/Transpose_1', [4, 4, 4, 20], [0, 2, 3, 1], True),
    (268, '/stages/stages.3/stages.3.1/transformer/transformer.0/attn/Transpose_2', [4, 4, 4, 20], [0, 2, 1, 3], True),
    (305, 'Transpose_token_380', [4, 4, 4, 20], [0, 2, 1, 3], True),
    (307, 'Transpose_token_378', [4, 4, 4, 20], [0, 2, 1, 3], True),
    (319, '/stages/stages.3/stages.3.1/transformer/transformer.1/attn/Transpose_1', [4, 4, 4, 20], [0, 2, 3, 1], True),
    (324, '/stages/stages.3/stages.3.1/transformer/transformer.1/attn/Transpose_2', [4, 4, 4, 20], [0, 2, 1, 3], True),
    (361, 'Transpose_token_384', [4, 4, 4, 20], [0, 2, 1, 3], True),
    (363, 'Transpose_token_382', [4, 4, 4, 20], [0, 2, 1, 3], True),
    (375, '/stages/stages.3/stages.3.1/transformer/transformer.2/attn/Transpose_1', [4, 4, 4, 20], [0, 2, 3, 1], True),
    (380, '/stages/stages.3/stages.3.1/transformer/transformer.2/attn/Transpose_2', [4, 4, 4, 20], [0, 2, 1, 3], True),
    (417, 'Transpose_token_388', [4, 4, 4, 20], [0, 2, 1, 3], True),
    (419, 'Transpose_token_386', [4, 4, 4, 20], [0, 2, 1, 3], True),
    (431, '/stages/stages.3/stages.3.1/transformer/transformer.3/attn/Transpose_1', [4, 4, 4, 20], [0, 2, 3, 1], True),
    (436, '/stages/stages.3/stages.3.1/transformer/transformer.3/attn/Transpose_2', [4, 4, 4, 20], [0, 2, 1, 3], True),
    (461, '/stages/stages.3/stages.3.1/Transpose_2', [1, 4, 4, 80], [0, 3, 2, 1], True),
    (463, '/stages/stages.3/stages.3.1/Transpose_3', [160, 2, 2, 2], [0, 2, 1, 3], True),
    (465, 'Transpose_token_288', [1, 80, 4, 4], [0, 2, 3, 1], False),
    (484, 'Transpose_token_312', [1, 2, 2, 96], [0, 3, 1, 2], False),
    (486, '/stages/stages.4/stages.4.1/Transpose', [96, 2, 1, 2], [0, 2, 1, 3], True),
    (497, '/stages/stages.4/stages.4.1/Transpose_1', [1, 96, 1, 4], [0, 3, 2, 1], True),
    (521, 'Transpose_token_403', [4, 1, 4, 24], [0, 2, 1, 3], True),
    (523, 'Transpose_token_401', [4, 1, 4, 24], [0, 2, 1, 3], True),
    (535, '/stages/stages.4/stages.4.1/transformer/transformer.0/attn/Transpose_1', [4, 1, 4, 24], [0, 2, 3, 1], True),
    (540, '/stages/stages.4/stages.4.1/transformer/transformer.0/attn/Transpose_2', [4, 4, 1, 24], [0, 2, 1, 3], True),
    (577, 'Transpose_token_407', [4, 1, 4, 24], [0, 2, 1, 3], True),
    (579, 'Transpose_token_405', [4, 1, 4, 24], [0, 2, 1, 3], True),
    (591, '/stages/stages.4/stages.4.1/transformer/transformer.1/attn/Transpose_1', [4, 1, 4, 24], [0, 2, 3, 1], True),
    (596, '/stages/stages.4/stages.4.1/transformer/transformer.1/attn/Transpose_2', [4, 4, 1, 24], [0, 2, 1, 3], True),
    (633, 'Transpose_token_411', [4, 1, 4, 24], [0, 2, 1, 3], True),
    (635, 'Transpose_token_409', [4, 1, 4, 24], [0, 2, 1, 3], True),
    (647, '/stages/stages.4/stages.4.1/transformer/transformer.2/attn/Transpose_1', [4, 1, 4, 24], [0, 2, 3, 1], True),
    (652, '/stages/stages.4/stages.4.1/transformer/transformer.2/attn/Transpose_2', [4, 4, 1, 24], [0, 2, 1, 3], True),
    (677, '/stages/stages.4/stages.4.1/Transpose_2', [1, 4, 1, 96], [0, 3, 2, 1], True),
    (679, '/stages/stages.4/stages.4.1/Transpose_3', [96, 1, 2, 2], [0, 2, 1, 3], True),
    (682, 'Transpose_token_315', [1, 96, 2, 2], [0, 2, 3, 1], False),
    (691, 'Transpose_token_324', [1, 2, 2, 320], [0, 3, 1, 2], False),
]
W2_IDX = (49, 186, 220, 463)  # the uint8 Transposes whose moved block is 2 bytes wide
# Same loop structure as W2_IDX (outer loops, M, N) with ONE element per moved block.
MECH = [(49, [256, 2, 4], [0, 2, 1]), (186, [256, 4, 2], [0, 2, 1]),
        (220, [160, 2, 2], [0, 2, 1]), (463, [160, 2, 2], [0, 2, 1])]

VARIANTS = {
    # name: (skeleton, element type, predicate on a TRANSPOSES row / MECH row)
    "tp_u8_all": ("full", "u8", lambda r: True),
    "tp_u8_w2": ("full", "u8", lambda r: r[0] in W2_IDX),
    "tp_u8_rest": ("full", "u8", lambda r: r[0] not in W2_IDX),
    "tp_u8_ctrl": ("full", "u8", lambda r: False),
    "tp_f32_all": ("full", "f32", lambda r: r[4]),
    "tp_f32_ctrl": ("full", "f32", lambda r: False),
    "tp_f16_w2": ("mech", "f16", lambda r: True),
    "tp_f32_w4": ("mech", "f32", lambda r: True),
}
ESIZE = {"u8": 1, "f16": 2, "f32": 4}


# ----------------------------------------------------------------------------------------------
# Static prediction: ORT's CPU Transpose dispatch (transpose.cc TransposeImpl + transpose_helper.cc),
# identical at v1.23.2 and v1.24.4 except has_mlas_transpose<uint16_t> (PR #25680).
# ----------------------------------------------------------------------------------------------
MLAS_BYTES = {"1.23.2": {1, 4}, "1.24.4": {1, 2, 4}}


def _is_reshape(perm, dims):
    last = 0
    for p in perm:
        if dims[p] == 1:
            continue
        if p < last:
            return False
        last = p
    return True


def _single_axis(perm):
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


def predict(dims, perm, esize, threads, version):
    """Code path and thread-pool dispatches per call of one Transpose."""
    if _is_reshape(perm, dims):
        return {"path": "reshape-copy", "dispatches": 0}
    sa = _single_axis(perm)
    if sa is None:
        return {"path": "untyped (serial)", "dispatches": 0}
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
    nbytes = block * esize
    if nbytes in MLAS_BYTES[version]:
        return {"path": f"MlasTranspose<{8 * nbytes}-bit> {kind} loops={loops} M={m} N={n}",
                "dispatches": loops if (threads > 1 and m > 1) else 0}
    return {"path": f"scalar {kind} {8 * nbytes}-bit loops={loops}", "dispatches": 0}


# ----------------------------------------------------------------------------------------------
# Model builder (onnx.helper only)
# ----------------------------------------------------------------------------------------------
def variant_rows(name):
    skel, et, pred = VARIANTS[name]
    if skel == "full":
        return [(r[0], r[1], r[2], r[3], pred(r)) for r in TRANSPOSES]
    return [(idx, f"mech_of_{idx}", shp, perm, pred(None)) for idx, shp, perm in MECH]


def build(name):
    skel, et, _ = VARIANTS[name]
    rows = variant_rows(name)
    nodes, inits = [], []

    def const(cname, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr), cname))
        return cname

    elem = {"u8": TensorProto.UINT8, "f16": TensorProto.FLOAT16, "f32": TensorProto.FLOAT}[et]
    src = "x"
    if et == "u8":
        const("q_scale", np.array(0.02, np.float32))
        const("q_zp", np.array(128, np.uint8))
        nodes.append(helper.make_node("QuantizeLinear", ["x", "q_scale", "q_zp"], ["xq"], name="Q_in"))
        src = "xq"
    nodes.append(helper.make_node("Reshape", [src, const("flat_all", np.array([12288], np.int64))],
                                  ["src_flat0"], name="flatten_in"))
    src = "src_flat0"
    if et == "f16":
        nodes.append(helper.make_node("Cast", [src], ["src_flat"], to=TensorProto.FLOAT16, name="cast_in"))
        src = "src_flat"
    axes0 = const("axes0", np.array([0], np.int64))
    parts = []
    for b, (idx, oname, shape, perm, has_t) in enumerate(rows):
        n = math.prod(shape)
        if n == 12288:
            cur = src
        else:
            # distinct offsets keep common-subexpression elimination from merging equal branches
            start = b + 1 if skel == "mech" else b
            assert start + n <= 12288
            cur = f"b{b:02d}_slice"
            nodes.append(helper.make_node(
                "Slice", [src, const(f"b{b:02d}_start", np.array([start], np.int64)),
                          const(f"b{b:02d}_end", np.array([start + n], np.int64)), axes0],
                [cur], name=f"b{b:02d}_Slice"))
        nodes.append(helper.make_node("Reshape", [cur, const(f"b{b:02d}_shape", np.array(shape, np.int64))],
                                      [f"b{b:02d}_in"], name=f"b{b:02d}_Reshape_in"))
        cur = f"b{b:02d}_in"
        if has_t:
            nodes.append(helper.make_node("Transpose", [cur], [f"b{b:02d}_t"], perm=list(perm),
                                          name=f"T{idx:03d}|{oname}"))
            cur = f"b{b:02d}_t"
        nodes.append(helper.make_node("Reshape", [cur, const(f"b{b:02d}_flat", np.array([n], np.int64))],
                                      [f"b{b:02d}_out"], name=f"b{b:02d}_Reshape_out"))
        parts.append(f"b{b:02d}_out")
    total = sum(math.prod(r[2]) for r in rows)
    if et == "u8":
        nodes.append(helper.make_node("Concat", parts, ["cat"], axis=0, name="concat"))
        nodes.append(helper.make_node("DequantizeLinear", ["cat", "q_scale", "q_zp"], ["y"], name="DQ_out"))
    elif et == "f16":
        nodes.append(helper.make_node("Concat", parts, ["cat"], axis=0, name="concat"))
        nodes.append(helper.make_node("Cast", ["cat"], ["y"], to=TensorProto.FLOAT, name="cast_out"))
    else:
        nodes.append(helper.make_node("Concat", parts, ["y"], axis=0, name="concat"))
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, INPUT_SHAPE)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [total])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)],
                              producer_name="transpose_probe.py")
    model.ir_version = 8
    model.doc_string = (f"Experiment 27 transpose probe, variant {name}: {sum(r[4] for r in rows)} "
                        f"{et} Transposes in a {skel} skeleton of {len(rows)} branches.")
    return model, rows, elem


def expected_transposes(name):
    _, et, _ = VARIANTS[name]
    return [(list(r[2]), list(r[3]), et) for r in variant_rows(name) if r[4]]


# ----------------------------------------------------------------------------------------------
# Graph inspection (onnx only): Transposes of a graph with input dtype and static input shape
# ----------------------------------------------------------------------------------------------
def _attr(n, key, default=None):
    for a in n.attribute:
        if a.name == key:
            return helper.get_attribute_value(a)
    return default


def _proxy(m):
    """Standard-op stand-ins for the contrib ops of an ORT-optimized graph, so onnx shape inference
    can run through it (same output shapes and dtypes, tensor names untouched)."""
    new, extra = [], []
    k = [0]

    def t(p):
        k[0] += 1
        return f"__px_{p}_{k[0]}"

    for n in m.graph.node:
        op, dom, o = n.op_type, n.domain, n.output
        if dom == "com.microsoft" and op in ("QLinearAdd", "QLinearMul"):
            new.append(helper.make_node(op[7:], [n.input[0], n.input[3]], [o[0]]))
        elif dom == "com.microsoft" and op in ("QLinearSigmoid", "QLinearSoftmax", "QuickGelu"):
            new.append(helper.make_node("Identity", [n.input[0]], [o[0]]))
        elif dom == "com.microsoft" and op == "QLinearConcat":
            new.append(helper.make_node("Concat", [n.input[i] for i in range(2, len(n.input), 3)], [o[0]],
                                        axis=_attr(n, "axis")))
        elif dom == "com.microsoft" and op == "QLinearGlobalAveragePool":
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
        elif dom == "com.microsoft" and op == "QGemm":
            a, b, c = t("q"), t("q"), t("q")
            qo = len(n.input) > 7 and n.input[7] != ""
            new += [helper.make_node("Cast", [n.input[0]], [a], to=TensorProto.FLOAT),
                    helper.make_node("Cast", [n.input[3]], [b], to=TensorProto.FLOAT),
                    helper.make_node("Gemm", [a, b], [c if qo else o[0]],
                                     transA=_attr(n, "transA", 0), transB=_attr(n, "transB", 0))]
            if qo:
                new.append(helper.make_node("Cast", [c], [o[0]], to=TensorProto.UINT8))
        elif (dom == "com.microsoft" and op == "QLinearConv") or (dom == "com.microsoft.nchwc" and op == "Conv"):
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
            seq.append(helper.make_node("Cast", [y], [o[0]], to=TensorProto.UINT8 if q else TensorProto.FLOAT))
            new += seq
        elif dom == "com.microsoft.nchwc" and op == "ReorderInput":
            new.append(helper.make_node("Transpose", [n.input[0]], [o[0]], perm=[0, 3, 1, 2])
                       if _attr(n, "channels_last", 0) else helper.make_node("Identity", [n.input[0]], [o[0]]))
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
        elif dom not in ("", "ai.onnx"):
            raise SystemExit(f"proxy: unhandled contrib op {dom}:{op}")
        else:
            c = onnx.NodeProto()
            c.CopyFrom(n)
            new.append(c)
    p = onnx.ModelProto()
    p.CopyFrom(m)
    del p.graph.node[:]
    p.graph.node.extend(new)
    p.graph.initializer.extend(extra)
    del p.graph.value_info[:]
    for i in p.graph.input:
        d = i.type.tensor_type.shape.dim
        if len(d) == 4 and not d[0].HasField("dim_value"):
            d[0].ClearField("dim_param")
            d[0].dim_value = 1
    keep = [o for o in p.opset_import if o.domain in ("", "ai.onnx")]
    del p.opset_import[:]
    p.opset_import.extend(keep)
    return p


def graph_transposes(path):
    """[(in_shape, perm, dtype_tag, node_name)] in node order, plus the (domain, op) census."""
    m = onnx.load(str(path))
    inf = shape_inference.infer_shapes(_proxy(m), strict_mode=False, data_prop=True)
    ti = {}
    for vi in list(inf.graph.value_info) + list(inf.graph.input) + list(inf.graph.output):
        tt = vi.type.tensor_type
        ti[vi.name] = (tt.elem_type, [d.dim_value if d.HasField("dim_value") else None for d in tt.shape.dim])
    tag = {TensorProto.UINT8: "u8", TensorProto.FLOAT16: "f16", TensorProto.FLOAT: "f32"}
    out = []
    for n in m.graph.node:
        if n.op_type == "Transpose":
            dt, shp = ti.get(n.input[0], (0, None))
            out.append((shp, list(_attr(n, "perm")), tag.get(dt, TensorProto.DataType.Name(dt)), n.name))
    census = collections.Counter(f"{n.domain or 'ai.onnx'}:{n.op_type}" for n in m.graph.node)
    return out, dict(sorted(census.items()))


def check_dump(int8_opt, fp32_opt):
    got8, _ = graph_transposes(int8_opt)
    want8 = [(r[2], r[3], "u8") for r in TRANSPOSES]
    ok8 = [(g[0], g[1], g[2]) for g in got8] == want8
    gotf, _ = graph_transposes(fp32_opt)
    wantf = collections.Counter((tuple(r[2]), tuple(r[3]), "f32") for r in TRANSPOSES if r[4])
    okf = collections.Counter((tuple(g[0]), tuple(g[1]), g[2]) for g in gotf) == wantf
    print(f"static graph: {len(got8)} Transposes, table {'MATCHES' if ok8 else 'DIFFERS'}")
    print(f"fp32 graph:   {len(gotf)} Transposes, table (fp32 rows, as a multiset) {'MATCHES' if okf else 'DIFFERS'}")
    return ok8 and okf


# ----------------------------------------------------------------------------------------------
# Pre-timing check of the EXECUTED micro graphs (runner only: runs the side venvs' onnxruntime)
# ----------------------------------------------------------------------------------------------
DUMP_CHILD = textwrap.dedent('''
    import json, sys
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.optimized_model_filepath = sys.argv[2]
    so.intra_op_num_threads = int(sys.argv[3])
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    ort.InferenceSession(sys.argv[1], so, providers=["CPUExecutionProvider"])
    print(json.dumps({"ort": ort.__version__}))
''')


def executed_matches(got, want):
    """got: graph_transposes() rows of an executed graph; want: expected_transposes().
    Transposes that only move size-1 axes may legally become Reshapes (static shapes); every other
    designed Transpose must be executed as designed (shape, perm, dtype), and nothing else may
    appear. Multisets, because the optimizer may reorder independent branches."""
    def key(s, p, e):
        return (tuple(s) if s is not None else None, tuple(p), e)

    got_t = collections.Counter(key(*g[:3]) for g in got)
    want_t = collections.Counter(key(*w) for w in want)
    essential = collections.Counter({k: c for k, c in want_t.items() if not _is_reshape(k[1], k[0])})
    got_ess = collections.Counter({k: c for k, c in got_t.items()
                                   if k[0] is not None and None not in k[0] and not _is_reshape(k[1], k[0])})
    return got_ess == essential and not (got_t - want_t)


def resolve_interpreters(scratch, versions):
    cands = {d.name: d / "Scripts" / "python.exe" for d in sorted(Path(scratch).glob("ort*"))
             if (d / "Scripts" / "python.exe").exists()}
    cands["project"] = ROOT / ".venv" / "Scripts" / "python.exe"
    found = {}
    for env, py in cands.items():
        r = subprocess.run([str(py), "-c", "import onnxruntime as o; print(o.__version__)"],
                           capture_output=True, text=True)
        v = r.stdout.strip()
        if r.returncode == 0 and v in versions:
            if v in found:
                raise SystemExit(f"two interpreters report {v}: {found[v]} and {py}")
            found[v] = py
    missing = set(versions) - set(found)
    if missing:
        raise SystemExit(f"no interpreter reports {sorted(missing)} under {scratch}")
    return found


def verify_executed(scratch, threads=4):
    pys = resolve_interpreters(scratch, VERSIONS)
    out_dir = PROBE_DIR / "executed"
    out_dir.mkdir(parents=True, exist_ok=True)
    child = out_dir / "_dump_child.py"
    child.write_text(DUMP_CHILD, encoding="utf-8")
    report, all_ok = {}, True
    for name in VARIANTS:
        want = expected_transposes(name)
        per_v = {}
        for v, py in pys.items():
            dst = out_dir / f"opt_{v}_{name}.onnx"
            r = subprocess.run([str(py), str(child), str(MODEL_DIR / f"{name}.onnx"), str(dst), str(threads)],
                               capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                per_v[v] = {"error": r.stderr.strip()[-300:]}
                all_ok = False
                continue
            got, census = graph_transposes(dst)
            ok = executed_matches(got, want)
            got_t = got
            per_v[v] = {"transposes": len(got_t), "designed": len(want), "essential_ok": ok,
                        "reshape_equivalent_converted": len(want) - len(got_t), "census": census}
            all_ok &= ok
        cens = [per_v[v].get("census") for v in pys]
        same = all(c == cens[0] for c in cens) and cens[0] is not None
        all_ok &= same
        report[name] = {"per_version": per_v, "census_identical_across_releases": same}
        print(f"  {name:12s} " + "  ".join(
            f"{v}: {per_v[v].get('transposes', 'ERR')}/{len(want)} T, "
            f"{'ok' if per_v[v].get('essential_ok') else 'FAIL'}" for v in pys)
            + f"  census {'identical' if same else 'DIFFERS'}")
    (PROBE_DIR / "executed_check.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nexecuted-graph check: {'PASS' if all_ok else 'FAIL - do not time; fix the builder first'}")
    return all_ok


# ----------------------------------------------------------------------------------------------
# Registered decision rule, applied mechanically
# ----------------------------------------------------------------------------------------------
def analyse(csvs, gate=0.95, out_path=None):
    import pandas as pd

    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    df = df[df["variant"].isin(VARIANTS.keys()) & df["ort"].isin(VERSIONS)]
    df["ort"] = df["ort"].astype(str)
    thr ={"loc": 0.70 * S_REF, "ref": 0.15 * S_REF, "small": FLOOR * S_REF}

    def cells(d):
        return {k: g["p50_ms"].tolist() for k, g in d.groupby(["variant", "ort", "threads"])}

    def run(d, label):
        c = cells(d)
        med = {k: float(np.median(v)) for k, v in c.items()}

        def m(v, r, t):
            return med.get((v, r, t), float("nan"))

        def D(v, t):
            return m(v, "1.24.4", t) - m(v, "1.23.2", t)

        def contrast(v, ctrl, t):
            est = D(v, t) - D(ctrl, t)
            keys = [(v, "1.24.4", t), (v, "1.23.2", t), (ctrl, "1.24.4", t), (ctrl, "1.23.2", t)]
            if any(k not in c for k in keys):
                return est, float("nan"), float("nan")
            vals = [a - b - (cc - dd) for a, b, cc, dd in itertools.product(*[c[k] for k in keys])]
            return est, min(vals), max(vals)

        def band(x):
            return ("LOCALISED" if x >= thr["loc"] else "PARTIAL" if x >= thr["ref"] else "REFUTED")

        res = {"label": label, "n_rows": int(len(d))}
        print(f"\n=== {label}: {len(d)} measurements")
        print(f"{'variant':12s} {'thr':>3s} {'1.23.2':>8s} {'1.24.4':>8s} {'D ms':>8s} {'D %':>7s} {'n':>5s}")
        for v in VARIANTS:
            for t in (1, 4):
                n = f"{len(c.get((v, '1.23.2', t), []))}/{len(c.get((v, '1.24.4', t), []))}"
                print(f"{v:12s} {t:3d} {m(v, '1.23.2', t):8.4f} {m(v, '1.24.4', t):8.4f} {D(v, t):+8.4f} "
                      f"{100 * D(v, t) / m(v, '1.23.2', t):+6.1f}% {n:>5s}")
        # validity: control drift
        v2 = all(abs(D(ctrl, t)) <= max(FLOOR * m(ctrl, "1.23.2", t), thr["small"])
                 for ctrl in ("tp_u8_ctrl", "tp_f32_ctrl") for t in (1, 4))
        du8, lo, hi = contrast("tp_u8_all", "tp_u8_ctrl", 4)
        du8_1, _, _ = contrast("tp_u8_all", "tp_u8_ctrl", 1)
        df32, _, _ = contrast("tp_f32_all", "tp_f32_ctrl", 4)
        above_floor = D("tp_u8_all", 4) > FLOOR * m("tp_u8_all", "1.23.2", 4)
        res.update({"V2_controls_stable": v2, "delta_u8_4T": du8, "delta_u8_4T_envelope": [lo, hi],
                    "delta_u8_1T": du8_1, "delta_f32_4T": df32, "u8_all_above_own_floor": above_floor,
                    "share_of_step": {k: du8 / s for k, s in S_SENS.items()}})
        print(f"\ncontrols stable (V2): {v2}")
        print(f"delta_u8 4T = {du8:+.4f} ms  (81-combination session envelope {lo:+.4f} .. {hi:+.4f}); "
              f"share of the {S_REF} ms step {100 * du8 / S_REF:.0f}%  "
              f"[{', '.join(f'{100 * du8 / s:.0f}% of {s}' for s in S_SENS.values())}]")
        print(f"delta_u8 1T = {du8_1:+.4f} ms   delta_f32 4T = {df32:+.4f} ms   "
              f"u8_all step above its own 7% floor: {above_floor}")
        if not v2:
            p = "INCONCLUSIVE (control drift at or above the floor)"
        elif not above_floor or du8 < thr["ref"]:
            p = "REFUTED"
        elif du8_1 >= thr["small"]:
            p = "INCONCLUSIVE (the probe also steps at 1 thread; the full model does not)"
        elif df32 >= thr["small"]:
            p = "INCONCLUSIVE (float Transposes step too; not 8-bit-specific)"
        else:
            p = band(du8)
        firm = (not math.isnan(lo)) and band(lo) == band(hi) == band(du8)
        if p in ("LOCALISED", "PARTIAL", "REFUTED") and not firm:
            p += " (PROVISIONAL: envelope crosses a band edge -> registered extension)"
        if p.startswith("LOCALISED") and du8 > 1.5 * S_REF:
            p += " [exceeds the step: state 'reproduces', no share above 100%]"
        res["P"] = p
        print(f"PRIMARY VERDICT: {p}")

        mech = "not evaluated"
        if p.startswith(("LOCALISED", "PARTIAL")):
            dw2, _, _ = contrast("tp_u8_w2", "tp_u8_ctrl", 4)
            dw2_1, _, _ = contrast("tp_u8_w2", "tp_u8_ctrl", 1)
            drest, _, _ = contrast("tp_u8_rest", "tp_u8_ctrl", 4)
            df16, df32w4 = D("tp_f16_w2", 4), D("tp_f32_w4", 4)
            cost = (m("tp_f32_w4", "1.23.2", 4) - m("tp_f32_w4", "1.23.2", 1)) / 832.0
            res.update({"delta_w2_4T": dw2, "delta_w2_1T": dw2_1, "delta_rest_4T": drest,
                        "D_f16_w2_4T": df16, "D_f32_w4_4T": df32w4, "per_dispatch_ms_f32_w4": cost,
                        "additivity": D("tp_u8_w2", 4) + D("tp_u8_rest", 4) - D("tp_u8_ctrl", 4)
                        - D("tp_u8_all", 4)})
            print(f"\ndelta_w2 4T = {dw2:+.4f} ms ({100 * dw2 / du8:.0f}% of delta_u8), 1T {dw2_1:+.4f}; "
                  f"delta_rest 4T = {drest:+.4f} ms ({100 * drest / du8:.0f}%)")
            print(f"D f16_w2 4T = {df16:+.4f} ms   D f32_w4 4T = {df32w4:+.4f} ms   "
                  f"f32_w4 (4T-1T)/832 at 1.23.2 = {1000 * cost:.2f} us/dispatch -> predicts "
                  f"{832 * cost:.3f} ms for the w2 step (ratio {dw2 / (832 * cost) if cost > 0 else float('nan'):.2f})")
            if dw2 < 0.30 * du8 or abs(df32w4) >= 0.5 * dw2 or df16 < thr["small"]:
                mech = "M-REFUTED (the 16-bit MLAS routing of PR #25680 does not carry the step)"
            elif (dw2 >= 0.70 * du8 and drest < 0.30 * du8 and df16 >= 0.50 * dw2
                  and abs(df32w4) < thr["small"] and dw2_1 < thr["small"]):
                mech = "M-CONFIRMED (2-byte-block Transposes routed to per-loop threaded MlasTranspose)"
            else:
                mech = "M-MIXED (report the shares; code-level mechanism stays a hypothesis)"
        res["M"] = mech
        print(f"MECHANISM VERDICT: {mech}")
        return res

    ungated = run(df, "ungated (as the paper reports)")
    gated = run(df[df["core_delivery"] >= gate], f"gated (core delivery >= {gate})")
    low = df[df["core_delivery"] < gate].groupby("variant").size().to_dict()
    same = ungated["P"].split(" (")[0] == gated["P"].split(" (")[0] and ungated["M"][:8] == gated["M"][:8]
    final = ungated["P"] if same else "INCONCLUSIVE (gated and ungated verdicts differ)"
    print(f"\ncells below the gate by variant: {low or 'none'}")
    print(f"FINAL: {final} | {ungated['M'] if same else 'mechanism not stated'}")
    out = {"final": final, "mechanism": ungated["M"] if same else None, "ungated": ungated, "gated": gated,
           "below_gate": low, "thresholds_ms": thr, "S_ref_ms": S_REF, "csv": [str(c) for c in csvs]}
    out_path = Path(out_path) if out_path else PROBE_DIR / "verdict.json"
    out_path.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"wrote {out_path}")
    return out


# ----------------------------------------------------------------------------------------------
def timing_command(scratch):
    return ("\"" + str(ROOT / ".venv" / "Scripts" / "python.exe").replace("\\", "/") + "\" "
            "scripts/paper/ort_version_bisect.py "
            f"--scratch \"{scratch}\" --versions {','.join(VERSIONS)} --threads 1,4 --repeats 3 "
            f"--seed {SEED} --families \"tprobe=results/transpose_probe/models:3,64,64\" "
            f"--variants {','.join(VARIANTS)} --out results/paper/transpose_probe.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-dump", nargs=2, metavar=("INT8_OPT", "FP32_OPT"),
                    help="re-derive the Transpose table from ORT-optimized dumps and compare")
    ap.add_argument("--verify-executed", action="store_true",
                    help="RUNNER, before timing: dump the micro models' executed graphs under both releases")
    ap.add_argument("--analyse", nargs="+", metavar="CSV", help="apply the registered decision rule")
    ap.add_argument("--verdict-out", default=None, help="where --analyse writes its JSON verdict")
    ap.add_argument("--scratch", default=SCRATCH)
    a = ap.parse_args()

    if a.check_dump:
        return 0 if check_dump(*a.check_dump) else 1
    if a.verify_executed:
        return 0 if verify_executed(a.scratch) else 1
    if a.analyse:
        analyse([Path(c) for c in a.analyse], out_path=a.verdict_out)
        return 0

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {"experiment": 27, "seed": SEED, "input_shape": INPUT_SHAPE, "versions": VERSIONS,
                "S_ref_ms": S_REF, "variants": {}}
    print(f"{'variant':12s} {'T':>3s} {'nodes':>5s}  predicted thread-pool dispatches / inference "
          f"(1.23.2 -> 1.24.4) at 1T | 4T")
    for name in VARIANTS:
        model, rows, _ = build(name)
        onnx.checker.check_model(model, full_check=True)
        path = MODEL_DIR / f"{name}.onnx"
        onnx.save(model, str(path))
        onnx.checker.check_model(str(path), full_check=True)
        got, census = graph_transposes(path)
        want = expected_transposes(name)
        assert [(g[0], g[1], g[2]) for g in got] == want, f"{name}: built graph differs from design"
        et = VARIANTS[name][1]
        pred = {v: {t: sum(predict(r[2], r[3], ESIZE[et], t, v)["dispatches"] for r in rows if r[4])
                    for t in (1, 4)} for v in VERSIONS}
        paths = [{"idx": r[0], "shape": r[2], "perm": r[3],
                  **{v: predict(r[2], r[3], ESIZE[et], 4, v)["path"] for v in VERSIONS}}
                 for r in rows if r[4]]
        manifest["variants"][name] = {
            "file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "dtype": et,
            "skeleton": VARIANTS[name][0], "transposes": len(want), "census": census,
            "predicted_dispatches": pred, "transpose_paths_4T": paths}
        print(f"{name:12s} {len(want):3d} {len(model.graph.node):5d}  "
              f"{pred['1.23.2'][1]} -> {pred['1.24.4'][1]} | {pred['1.23.2'][4]} -> {pred['1.24.4'][4]}")
    (PROBE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"\nwrote {len(VARIANTS)} models to {MODEL_DIR.relative_to(ROOT)} (onnx.checker full_check passed)")
    print(f"wrote {(PROBE_DIR / 'manifest.json').relative_to(ROOT)}")
    print("\nBefore timing (runner): python scripts/paper/transpose_probe.py --verify-executed")
    print("\nTiming command (from the project root; 8 variants x 2 releases x 2 thread counts x 3 "
          "sessions = 96 cells, ~7-8 min):\n")
    print(timing_command(a.scratch))
    print("\nThen: python scripts/paper/transpose_probe.py --analyse results/paper/transpose_probe.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
