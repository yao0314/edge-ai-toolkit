"""Experiment W: build the moved-block width probes, and verify them before anything is timed.

Binding protocol: results/paper/w_design_registered.md, written by an independent designer before any
timing existed. This file implements the BUILD side of it and times nothing; it does not import
onnxruntime at all.

The one shape template, from §2.1 of the registration:

    dims = [L, 2, N, K], perm = [0, 2, 1, 3]

`transpose_probe._single_axis([0,2,1,3])` gives ("out", 2, 1), so loops = L, M = 2, N = dims[2] and
the moved block is `K * elem_bytes` wide. Width is therefore set by K and the element type alone,
and the ladder holds **8192 bytes moved per node, four nodes per file, L and M identical** in every
variant: N = 16/W absorbs the width. That is the whole point — a naive sweep changes the tensor size
along with the width and then measures size.

    .venv/Scripts/python.exe scripts/paper/w_probe.py --build --out-root results/w_probe
    .venv/Scripts/python.exe scripts/paper/w_probe.py --verify-static --json results/expW/preflight/static.json
    .venv/Scripts/python.exe scripts/paper/w_probe.py --freeze --json results/expW/preflight/hashes.json

No RNG anywhere: the same command must reproduce byte-identical files (V8).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper, save_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transpose_probe import INPUT_SHAPE, predict  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PERM = [0, 2, 1, 3]
R_BRANCHES = 4
ESIZE = {"u8": 1, "f16": 2, "f32": 4}
ELEM = {"u8": TensorProto.UINT8, "f16": TensorProto.FLOAT16, "f32": TensorProto.FLOAT}

# §3.1 and §3.2, verbatim. (name, dtype, dims, has_transpose, added dispatches per file at 4 threads)
MAIN = [
    ("m_u8_w1",   "u8",  [256, 2, 16, 1], True,     0),
    ("m_u8_w2",   "u8",  [256, 2,  8, 2], True,  1024),
    ("m_u8_w4",   "u8",  [256, 2,  4, 4], True,     0),
    ("m_u8_w8",   "u8",  [256, 2,  2, 8], True,     0),
    ("m_f16_w2",  "f16", [256, 2,  8, 1], True,  1024),
    ("m_f16_w4",  "f16", [256, 2,  4, 2], True,     0),
    ("m_f16_w8",  "f16", [256, 2,  2, 4], True,     0),
    ("m_f32_w4",  "f32", [256, 2,  4, 1], True,     0),
    ("m_f32_w8",  "f32", [256, 2,  2, 2], True,     0),
    ("ctrl_u8",   "u8",  [256, 2,  8, 2], False,    0),
    ("ctrl_f16",  "f16", [256, 2,  8, 1], False,    0),
    ("ctrl_f32",  "f32", [256, 2,  4, 1], False,    0),
]
DOSE = [
    ("d_u8_w1_L64",    "u8", [64,   2, 64, 1], True,     0),
    ("d_u8_w1_L256",   "u8", [256,  2, 16, 1], True,     0),
    ("d_u8_w1_L1024",  "u8", [1024, 2,  4, 1], True,     0),
    ("d_u8_w2_L64",    "u8", [64,   2, 32, 2], True,   256),
    ("d_u8_w2_L256",   "u8", [256,  2,  8, 2], True,  1024),
    ("d_u8_w2_L1024",  "u8", [1024, 2,  2, 2], True,  4096),
    ("d_ctrl_u8",      "u8", [256,  2,  8, 2], False,    0),
]
FAMILIES = {"main": MAIN, "dose": DOSE}
BYTES_PER_NODE = 8192          # held identical across the whole ladder (§2.2)
BEFORE, AFTER = "1.23.2", "1.24.4"


def width(dtype: str, dims: list[int]) -> int:
    return dims[3] * ESIZE[dtype]


def build(name: str, dtype: str, dims: list[int], has_t: bool):
    """The §2.4 skeleton, parameterised. Same form as transpose_probe.build, which 27/27b validated."""
    n = math.prod(dims)
    nodes, inits = [], []

    def const(cname, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr), cname))
        return cname

    src = "x"
    if dtype == "u8":
        const("q_scale", np.array(0.02, np.float32))
        const("q_zp", np.array(128, np.uint8))
        nodes.append(helper.make_node("QuantizeLinear", ["x", "q_scale", "q_zp"], ["xq"], name="Q_in"))
        src = "xq"
    nodes.append(helper.make_node("Reshape", [src, const("flat_all", np.array([12288], np.int64))],
                                  ["src_flat0"], name="flatten_in"))
    src = "src_flat0"
    if dtype == "f16":
        nodes.append(helper.make_node("Cast", [src], ["src_flat"], to=TensorProto.FLOAT16,
                                      name="cast_in"))
        src = "src_flat"
    axes0 = const("axes0", np.array([0], np.int64))
    parts = []
    for b in range(R_BRANCHES):
        start = b                      # distinct starts defeat common-subexpression elimination
        assert start + n <= 12288, f"{name}: branch {b} needs {start + n} of 12288 elements"
        cur = f"b{b:02d}_slice"
        nodes.append(helper.make_node(
            "Slice", [src, const(f"b{b:02d}_start", np.array([start], np.int64)),
                      const(f"b{b:02d}_end", np.array([start + n], np.int64)), axes0],
            [cur], name=f"b{b:02d}_Slice"))
        nodes.append(helper.make_node("Reshape", [cur, const(f"b{b:02d}_shape", np.array(dims, np.int64))],
                                      [f"b{b:02d}_in"], name=f"b{b:02d}_Reshape_in"))
        cur = f"b{b:02d}_in"
        if has_t:
            nodes.append(helper.make_node("Transpose", [cur], [f"b{b:02d}_t"], perm=list(PERM),
                                          name=f"T{b}|{name}"))
            cur = f"b{b:02d}_t"
        nodes.append(helper.make_node("Reshape", [cur, const(f"b{b:02d}_flat", np.array([n], np.int64))],
                                      [f"b{b:02d}_out"], name=f"b{b:02d}_Reshape_out"))
        parts.append(f"b{b:02d}_out")
    total = R_BRANCHES * n
    if dtype == "u8":
        nodes.append(helper.make_node("Concat", parts, ["cat"], axis=0, name="concat"))
        nodes.append(helper.make_node("DequantizeLinear", ["cat", "q_scale", "q_zp"], ["y"], name="DQ_out"))
    elif dtype == "f16":
        nodes.append(helper.make_node("Concat", parts, ["cat"], axis=0, name="concat"))
        nodes.append(helper.make_node("Cast", ["cat"], ["y"], to=TensorProto.FLOAT, name="cast_out"))
    else:
        nodes.append(helper.make_node("Concat", parts, ["y"], axis=0, name="concat"))
    graph = helper.make_graph(
        nodes, name,
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, INPUT_SHAPE)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [total])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)],
                              producer_name="w_probe.py")
    model.ir_version = 8
    model.doc_string = (
        f"Experiment W probe {name}: {R_BRANCHES if has_t else 0} {dtype} Transposes, dims={dims}, "
        f"perm={PERM}, moved block {width(dtype, dims)} B, {n * ESIZE[dtype]} bytes moved per node.")
    return model


def rows_for(fam: str) -> list[dict]:
    out = []
    for name, dtype, dims, has_t, added4 in FAMILIES[fam]:
        n = math.prod(dims)
        row = {"family": fam, "variant": name, "dtype": dtype, "dims": dims,
               "has_transpose": has_t, "elem_bytes": ESIZE[dtype], "elements_per_node": n,
               "bytes_per_node": n * ESIZE[dtype], "width_bytes": width(dtype, dims),
               "registered_added_dispatches_4t": added4}
        if has_t:
            for v in (BEFORE, AFTER):
                for t in (1, 2, 4):
                    p = predict(dims, PERM, ESIZE[dtype], t, v)
                    row[f"path_{v}"] = p["path"]
                    row[f"disp_{v}_t{t}"] = p["dispatches"]
        out.append(row)
    return out


def do_build(out_root: Path) -> int:
    man = {"experiment": "W", "protocol": "results/paper/w_design_registered.md",
           "perm": PERM, "branches": R_BRANCHES, "input_shape": INPUT_SHAPE,
           "bytes_moved_per_node": BYTES_PER_NODE, "files": []}
    for fam in FAMILIES:
        d = out_root / fam
        d.mkdir(parents=True, exist_ok=True)
        for name, dtype, dims, has_t, added4 in FAMILIES[fam]:
            n = math.prod(dims)
            assert n * ESIZE[dtype] == BYTES_PER_NODE, \
                f"{name}: {n * ESIZE[dtype]} bytes per node, the ladder requires {BYTES_PER_NODE}"
            p = d / f"{name}.onnx"
            save_model(build(name, dtype, dims, has_t), str(p))
            man["files"].append({"family": fam, "variant": name, "path": str(p.relative_to(ROOT)),
                                 "dtype": dtype, "dims": dims, "has_transpose": has_t,
                                 "width_bytes": width(dtype, dims),
                                 "bytes_per_node": n * ESIZE[dtype],
                                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                 "size_bytes": p.stat().st_size})
            print("built %-16s %-4s dims=%-16s W=%d B  %d bytes/node  %s"
                  % (name, dtype, dims, width(dtype, dims), n * ESIZE[dtype],
                     "with Transpose" if has_t else "CONTROL, no Transpose"))
    (out_root / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print("\n%d files + manifest -> %s" % (len(man["files"]), out_root))
    return 0


def do_verify_static(json_path: Path | None) -> int:
    """V1: re-derive the registered table through transpose_probe.predict() and compare."""
    ok, report = True, []
    for fam in FAMILIES:
        for row in rows_for(fam):
            if not row["has_transpose"]:
                report.append({**row, "verdict": "CONTROL"})
                continue
            gain = R_BRANCHES * (row[f"disp_{AFTER}_t4"] - row[f"disp_{BEFORE}_t4"])
            agree = gain == row["registered_added_dispatches_4t"]
            iso = row["bytes_per_node"] == BYTES_PER_NODE
            ok = ok and agree and iso
            report.append({**row, "added_dispatches_4t_derived": gain,
                           "verdict": "OK" if (agree and iso) else "MISMATCH"})
            print("%-16s W=%d B  %-32s -> %-32s  added/file 4T: derived %5d registered %5d  %s"
                  % (row["variant"], row["width_bytes"],
                     row[f"path_{BEFORE}"].split(" loops")[0],
                     row[f"path_{AFTER}"].split(" loops")[0],
                     gain, row["registered_added_dispatches_4t"],
                     "OK" if agree else "MISMATCH"))
    movers = [r["variant"] for r in report if r.get("added_dispatches_4t_derived", 0) > 0]
    widths = sorted({r["width_bytes"] for r in report if r["has_transpose"]})
    print("\nfiles that gain dispatches at 1.23.2 -> 1.24.4: %s" % (", ".join(movers) or "none"))
    print("widths present: %s bytes | bytes moved per node: %d in every file"
          % (widths, BYTES_PER_NODE))
    only_w2 = all(width_of(v) == 2 for v in movers) if movers else False
    print("every gainer is a 2-byte file: %s" % only_w2)
    ok = ok and only_w2
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps({"ok": ok, "only_two_byte_files_gain": only_w2,
                                         "bytes_per_node": BYTES_PER_NODE, "rows": report},
                                        indent=1), encoding="utf-8")
        print("wrote %s" % json_path)
    print("\nV1 %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 3


def width_of(variant: str) -> int:
    for fam in FAMILIES:
        for name, dtype, dims, _, _ in FAMILIES[fam]:
            if name == variant:
                return width(dtype, dims)
    return -1


def do_freeze(out_root: Path, json_path: Path | None) -> int:
    man = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    frozen = {"frozen_utc": None, "files": {}}
    from datetime import datetime, timezone
    frozen["frozen_utc"] = datetime.now(timezone.utc).isoformat()
    bad = []
    for f in man["files"]:
        p = ROOT / f["path"]
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        frozen["files"][f["variant"]] = h
        if h != f["sha256"]:
            bad.append(f["variant"])
    frozen["matches_build"] = not bad
    frozen["changed_since_build"] = bad
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(frozen, indent=1), encoding="utf-8")
        print("wrote %s" % json_path)
    print("%d hashes frozen; changed since build: %s" % (len(frozen["files"]), bad or "none"))
    return 0 if not bad else 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify-static", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--out-root", default=str(ROOT / "results" / "w_probe"))
    ap.add_argument("--json")
    a = ap.parse_args()
    out_root = Path(a.out_root)
    out_root = out_root if out_root.is_absolute() else (ROOT / out_root)
    j = Path(a.json) if a.json else None
    if a.build:
        return do_build(out_root)
    if a.verify_static:
        return do_verify_static(j)
    if a.freeze:
        return do_freeze(out_root, j)
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
