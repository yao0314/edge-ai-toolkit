"""Experiment E2: the counterfactual that Experiment 27b's registered limitation asks for.

27b showed SUFFICIENCY IN ISOLATION: micro-models containing only the four uint8 Transposes whose
moved block is two bytes wide reproduce +0.417 ms of the full model's +0.48 ms 1.23.2 -> 1.24.4 step
at four threads. That does not show those four nodes are what slows the FULL model, so the paper may
say "traced to a code-level cause" and may not say "root cause".

E2 rewrites those four nodes IN THE FULL STATIC FILE so that they no longer take the two-byte
dispatch path, and times the rewritten file against the unmodified one on both releases.

    Nothing in this file times anything and nothing imports onnxruntime in the builder. The
    --verify-* subcommands shell out to the side venvs, load each model ONCE and dump or compare.

Models written to results/e2_counterfactual/models/ (all four timed under identical harness
treatment, from one directory, in one randomised run):

  p00_int8          byte-identical copy of results/classification_mobilevit/models/p00_int8.onnx
  p00_int8_e2       THE COUNTERFACTUAL. The four two-byte-block Transposes only.
  p00_int8_sham     MATCHED CONTROL. Same four nodes, same graph edit, block stays two bytes.
  p00_int8_plac     PLACEBO. Same graph edit on four Transposes that are NOT in the two-byte class.

The edit, per site (zero new nodes, one new initializer, one changed attribute):

  ... -> Reshape(shape = Concat(a, b, c, d)) -> Q -> DQ -> Transpose(perm) -> Q -> DQ -> Reshape ...

  * append a rank-1 int64 initializer [1] to that Reshape's dedicated shape Concat, so the tensor
    reaching the Transpose has one extra trailing axis of length 1;
  * change the Transpose's perm to the rank-5 perm named below.

  e2    perm [0,2,1,3] -> [0,2,1,4,3]  no longer a single-axis move: TransposeImpl falls through to
                                       DoUntypedTranspose, which takes no thread pool at all.
  sham  perm [0,2,1,3] -> [0,2,1,3,4]  still a single-axis move with a two-byte block: same
                                       MlasTranspose<uint16_t> calls, same 832 dispatches at 1.24.4.
  plac  perm [0,3,2,1] -> [0,3,2,1,4]  on the four [0,3,2,1] Transposes, which are already on the
                                       untyped path; the executed kernel parameters are unchanged.

Every rewrite is an exact identity on the data: appending a length-1 axis and permuting it does not
move a byte, so the flattened output is the same sequence, and the downstream Reshape is unchanged.

  python scripts/paper/e2_counterfactual.py --build
  python scripts/paper/e2_counterfactual.py --verify-static
  python scripts/paper/e2_counterfactual.py --verify-outputs  --scratch <scratchpad>
  python scripts/paper/e2_counterfactual.py --verify-executed --scratch <scratchpad>
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.paper.transpose_probe import (  # noqa: E402
    _is_reshape, _single_axis, graph_transposes, predict,
)

SRC = ROOT / "results" / "classification_mobilevit" / "models" / "p00_int8.onnx"
OUT_DIR = ROOT / "results" / "e2_counterfactual"
MODEL_DIR = OUT_DIR / "models"
VERSIONS = ("1.23.2", "1.24.4")
INPUT_SHAPE = (1, 3, 64, 64)

# The four source Transposes whose executed uint8 block is two bytes wide (Exp 27 census rows
# 3, 14, 17, 36; optimized-graph indices 49, 186, 220, 463).
TARGETS = [
    "/stages/stages.2/stages.2.1/Transpose",
    "/stages/stages.2/stages.2.1/Transpose_3",
    "/stages/stages.3/stages.3.1/Transpose",
    "/stages/stages.3/stages.3.1/Transpose_3",
]
# Four Transposes that are NOT in the two-byte class: the [0,3,2,1] folds of the same two MobileViT
# blocks (census rows 4, 13, 18, 35). They move 10,752 bytes in total, exactly as the targets do.
PLACEBO = [
    "/stages/stages.2/stages.2.1/Transpose_1",
    "/stages/stages.2/stages.2.1/Transpose_2",
    "/stages/stages.3/stages.3.1/Transpose_1",
    "/stages/stages.3/stages.3.1/Transpose_2",
]

# variant -> (nodes to edit, old perm, new perm)
VARIANTS = {
    "p00_int8_e2":   (TARGETS, [0, 2, 1, 3], [0, 2, 1, 4, 3]),
    "p00_int8_sham": (TARGETS, [0, 2, 1, 3], [0, 2, 1, 3, 4]),
    "p00_int8_plac": (PLACEBO, [0, 3, 2, 1], [0, 3, 2, 1, 4]),
}


# ----------------------------------------------------------------------------------------------
# builder (onnx only)
# ----------------------------------------------------------------------------------------------
def _perm(n):
    return list([a.ints for a in n.attribute if a.name == "perm"][0])


def _set_perm(n, perm):
    for a in n.attribute:
        if a.name == "perm":
            del a.ints[:]
            a.ints.extend(perm)
            return
    raise SystemExit(f"{n.name} has no perm attribute")


def _site(g, tname):
    """Transpose node, the Reshape that feeds it through Q/DQ, and that Reshape's shape Concat.

    Fails loudly unless the site has exactly the shape this edit assumes."""
    prod = {o: n for n in g.node for o in n.output}
    cons = collections.defaultdict(list)
    for n in g.node:
        for i in n.input:
            cons[i].append(n)
    t = [n for n in g.node if n.name == tname]
    if len(t) != 1:
        raise SystemExit(f"{tname}: found {len(t)} nodes")
    t = t[0]
    dq = prod[t.input[0]]
    q = prod[dq.input[0]]
    r = prod[q.input[0]]
    c = prod.get(r.input[1])
    if (dq.op_type, q.op_type, r.op_type) != ("DequantizeLinear", "QuantizeLinear", "Reshape"):
        raise SystemExit(f"{tname}: producer chain is {r.op_type}->{q.op_type}->{dq.op_type}")
    if c is None or c.op_type != "Concat":
        raise SystemExit(f"{tname}: shape tensor is not produced by a Concat")
    if [a.i for a in c.attribute if a.name == "axis"] != [0]:
        raise SystemExit(f"{tname}: shape Concat is not on axis 0")
    if len(cons[c.output[0]]) != 1 or cons[c.output[0]][0] is not r:
        raise SystemExit(f"{tname}: shape Concat feeds {len(cons[c.output[0]])} nodes, not just its Reshape")
    if len(cons[r.output[0]]) != 1 or len(cons[t.output[0]]) != 1:
        raise SystemExit(f"{tname}: Reshape or Transpose output has fan-out > 1")
    return t, r, c


def _restale(g, names):
    """The rank of the tensors on an edited path changes, so the file's stored value_info for them
    is stale. Drop those entries and put back whatever onnx shape inference derives for them, so the
    rewritten file carries the same kind of annotation as the original (nothing else is touched)."""
    keep = [v for v in g.value_info if v.name not in names]
    names = {v.name for v in g.value_info if v.name in names}   # restore exactly what we drop
    dropped = len(g.value_info) - len(keep)
    del g.value_info[:]
    g.value_info.extend(keep)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    try:
        inf = shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)
    except Exception:
        return dropped, 0
    back = [v for v in inf.graph.value_info if v.name in names]
    g.value_info.extend(back)
    return dropped, len(back)


def build(verbose=True):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    base = onnx.load(str(SRC))
    onnx.checker.check_model(base, full_check=True)

    shutil.copyfile(SRC, MODEL_DIR / "p00_int8.onnx")
    manifest = {"source": str(SRC.relative_to(ROOT)).replace("\\", "/"),
                "input_shape": list(INPUT_SHAPE),
                "sha256": {"p00_int8": _sha(MODEL_DIR / "p00_int8.onnx")},
                "edits": {}, "predicted_dispatches_4t": {}}

    for name, (nodes, old, new) in VARIANTS.items():
        m = onnx.ModelProto()
        m.CopyFrom(base)
        g = m.graph
        edits, stale = [], set()
        for k, tname in enumerate(nodes):
            t, r, c = _site(g, tname)
            if _perm(t) != old:
                raise SystemExit(f"{name}: {tname} perm is {_perm(t)}, expected {old}")
            init_name = f"e2_axis1_{name}_{k}"
            g.initializer.append(numpy_helper.from_array(np.array([1], np.int64), init_name))
            c.input.append(init_name)          # trailing length-1 axis on the Reshape target shape
            _set_perm(t, new)
            edits.append({"transpose": tname, "perm": [old, new], "concat": c.name,
                          "reshape": r.name, "new_initializer": init_name})
            # every tensor between the edited Reshape and the Reshape that restores rank 4
            cons = collections.defaultdict(list)
            for nn in g.node:
                for i in nn.input:
                    cons[i].append(nn)
            stale.add(c.output[0])
            cur = r
            while True:
                stale.add(cur.output[0])
                nxt = cons[cur.output[0]][0]
                if nxt.op_type == "Reshape":
                    break
                cur = nxt
        dropped, restored = _restale(g, stale)
        if verbose:
            print(f"  {name}: {len(edits)} sites; value_info re-derived for "
                  f"{restored}/{dropped} rank-changed tensors")
        onnx.checker.check_model(m, full_check=True)
        p = MODEL_DIR / f"{name}.onnx"
        onnx.save(m, str(p))
        manifest["edits"][name] = edits
        manifest["sha256"][name] = _sha(p)
        if verbose:
            print(f"  wrote {p.relative_to(ROOT)}  ({p.stat().st_size} bytes, sha {manifest['sha256'][name][:12]})")
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------------------------
# static verification (onnx only): shapes, paths, predicted dispatches, graph diff vs the original
# ----------------------------------------------------------------------------------------------
def _src_transposes(path):
    """[(name, in_dtype, in_dims, perm)] of a SOURCE graph, shapes from onnx shape inference with
    the batch dimension pinned to 1."""
    m = onnx.load(str(path))
    d0 = m.graph.input[0].type.tensor_type.shape.dim[0]
    d0.ClearField("dim_param")
    d0.dim_value = 1
    inf = shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)
    vi = {v.name: v for v in list(inf.graph.value_info) + list(inf.graph.input) + list(inf.graph.output)}
    out = []
    for n in m.graph.node:
        if n.op_type != "Transpose":
            continue
        v = vi.get(n.input[0])
        tt = v.type.tensor_type if v is not None else None
        dims = ([d.dim_value if d.HasField("dim_value") else None for d in tt.shape.dim]
                if tt is not None else None)
        et = TensorProto.DataType.Name(tt.elem_type) if tt is not None else "?"
        out.append((n.name, et, dims, _perm(n)))
    return out


def _census(path):
    m = onnx.load(str(path))
    return collections.Counter(f"{n.domain or 'ai.onnx'}:{n.op_type}" for n in m.graph.node)


def _node_ids(path):
    m = onnx.load(str(path))
    return collections.Counter((n.domain or "ai.onnx", n.op_type, n.name) for n in m.graph.node)


def verify_static():
    """Everything that can be checked without a runtime."""
    ok = True
    base_c = _census(MODEL_DIR / "p00_int8.onnx")
    base_t = {r[0]: r for r in _src_transposes(MODEL_DIR / "p00_int8.onnx")}
    report = {}
    for name, (nodes, old, new) in VARIANTS.items():
        p = MODEL_DIR / f"{name}.onnx"
        c = _census(p)
        got_t = {r[0]: r for r in _src_transposes(p)}
        same_census = c == base_c
        changed = sorted(k for k in base_t if base_t[k][3] != got_t[k][3] or base_t[k][2] != got_t[k][2])
        only_intended = changed == sorted(nodes)
        ok &= same_census and only_intended
        report[name] = {"node_census_identical_to_original": same_census,
                        "source_transposes_changed": changed,
                        "only_intended_changed": only_intended, "sites": {}}
        print(f"\n{name}: node census {'identical' if same_census else 'DIFFERS'}; "
              f"{len(changed)} source Transpose(s) changed, "
              f"{'exactly the intended ones' if only_intended else 'NOT the intended set'}")
        for k in sorted(set(changed) | set(nodes)):
            b, gsub = base_t[k], got_t[k]
            row = {}
            for tag, (dt, dims, perm) in (("before", b[1:]), ("after", gsub[1:])):
                # the executed tensor is uint8 (the QDQ transformer moves the Transpose onto the
                # quantized tensor); the source dtype is FLOAT because it sits between DQ and Q
                pr = {v: predict(dims, perm, 1, 4, v) for v in VERSIONS}
                row[tag] = {"dims": dims, "perm": perm,
                            "path_1.23.2": pr["1.23.2"]["path"], "path_1.24.4": pr["1.24.4"]["path"],
                            "dispatches": {v: pr[v]["dispatches"] for v in VERSIONS},
                            "bytes": int(np.prod(dims))}
            report[name]["sites"][k] = row
            print(f"   {k}")
            for tag in ("before", "after"):
                r = row[tag]
                print(f"      {tag:6s} dims={r['dims']} perm={r['perm']}  "
                      f"{r['dispatches']['1.23.2']:4d} -> {r['dispatches']['1.24.4']:4d} dispatches   "
                      f"[{r['path_1.24.4']}]")
    (OUT_DIR / "static_check.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nstatic check: {'PASS' if ok else 'FAIL'}")
    return ok


# ----------------------------------------------------------------------------------------------
# runtime verification: one session load per (model, release); no timing
# ----------------------------------------------------------------------------------------------
DUMP_CHILD = textwrap.dedent('''
    import json, sys
    import numpy as np, onnxruntime as ort
    model, dump, threads, level, npy_in, npy_out = sys.argv[1:7]
    so = ort.SessionOptions()
    so.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL if level == "all"
                                   else ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
    if dump:
        so.optimized_model_filepath = dump
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.log_severity_level = 3
    s = ort.InferenceSession(model, so, providers=["CPUExecutionProvider"])
    out = None
    if npy_in:
        x = np.load(npy_in)
        name = s.get_inputs()[0].name
        res = [s.run(None, {name: x[i:i + 1]})[0] for i in range(x.shape[0])]
        out = np.stack(res)
        np.save(npy_out, out)
    print(json.dumps({"ort": ort.__version__,
                      "out_shape": None if out is None else list(out.shape)}))
''')


def _interpreters(scratch):
    cands = {d.name: d / "Scripts" / "python.exe" for d in sorted(Path(scratch).glob("ort*"))
             if (d / "Scripts" / "python.exe").exists()}
    cands["project"] = ROOT / ".venv" / "Scripts" / "python.exe"
    found = {}
    for env, py in cands.items():
        r = subprocess.run([str(py), "-c", "import onnxruntime as o; print(o.__version__)"],
                           capture_output=True, text=True)
        v = r.stdout.strip()
        if r.returncode == 0 and v in VERSIONS and v not in found:
            found[v] = py
    missing = set(VERSIONS) - set(found)
    if missing:
        raise SystemExit(f"no interpreter reports {sorted(missing)} under {scratch}")
    return found


def _run_child(py, work, model, dump="", threads=4, level="all", npy_in="", npy_out=""):
    child = work / "_e2_child.py"
    child.write_text(DUMP_CHILD, encoding="utf-8")
    r = subprocess.run([str(py), str(child), str(model), str(dump), str(threads), level,
                        str(npy_in), str(npy_out)], capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise SystemExit(f"child failed for {model} ({level}): {r.stderr.strip()[-400:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def verify_outputs(scratch, n=8, seed=2026):
    """(4a) The rewritten graphs must compute the same thing as the original."""
    pys = _interpreters(scratch)
    work = OUT_DIR / "verify"
    work.mkdir(parents=True, exist_ok=True)
    x = np.random.default_rng(seed).standard_normal((n, *INPUT_SHAPE[1:])).astype(np.float32)
    npy_in = work / "inputs.npy"
    np.save(npy_in, x)
    report, ok = {}, True
    for v, py in pys.items():
        for level in ("disabled", "all"):
            ref = None
            for name in ["p00_int8"] + list(VARIANTS):
                o = work / f"out_{v}_{level}_{name}.npy"
                _run_child(py, work, MODEL_DIR / f"{name}.onnx", threads=4, level=level,
                           npy_in=npy_in, npy_out=o)
                a = np.load(o)
                if name == "p00_int8":
                    ref = a
                    continue
                dmax = float(np.max(np.abs(a - ref)))
                bit = bool(np.array_equal(a, ref))
                same_argmax = bool(np.array_equal(a.reshape(n, -1).argmax(1),
                                                  ref.reshape(n, -1).argmax(1)))
                report[f"{v}|{level}|{name}"] = {"bit_identical": bit, "max_abs_diff": dmax,
                                                 "argmax_identical": same_argmax}
                ok &= bit
                print(f"  {v} opt={level:8s} {name:14s} bit-identical={bit} "
                      f"max|d|={dmax:.3e} argmax={same_argmax}")
    (OUT_DIR / "output_check.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\noutput check: {'PASS (bit-identical everywhere)' if ok else 'FAIL'}")
    return ok


def verify_executed(scratch, threads=4):
    """(4b, 4c) Dump each optimized graph under both releases; diff against the original's; count
    the thread-pool dispatches the probe's prediction logic gives for the executed graph."""
    pys = _interpreters(scratch)
    work = OUT_DIR / "executed"
    work.mkdir(parents=True, exist_ok=True)
    dumps, report, ok = {}, {}, True
    for v, py in pys.items():
        for name in ["p00_int8"] + list(VARIANTS):
            dst = work / f"opt_{v}_{name}.onnx"
            _run_child(py, work, MODEL_DIR / f"{name}.onnx", dump=dst, threads=threads, level="all")
            dumps[(v, name)] = dst

    def summarise(v, name):
        ts, census = graph_transposes(dumps[(v, name)])
        rows, disp = [], {"1.23.2": 0, "1.24.4": 0}
        for shp, perm, tag, nm in ts:
            esize = {"u8": 1, "f16": 2, "f32": 4}.get(tag)
            p = {u: predict(shp, perm, esize, threads, u) for u in VERSIONS} if (
                shp and None not in shp and esize) else None
            if p:
                for u in VERSIONS:
                    disp[u] += p[u]["dispatches"]
            rows.append({"name": nm, "dims": shp, "perm": perm, "dtype": tag,
                         "path_1.23.2": p and p["1.23.2"]["path"],
                         "path_1.24.4": p and p["1.24.4"]["path"],
                         "dispatches_1.23.2": p and p["1.23.2"]["dispatches"],
                         "dispatches_1.24.4": p and p["1.24.4"]["dispatches"],
                         "bytes": None if not shp or None in shp else int(np.prod(shp))})
        return rows, dict(sorted(census.items())), disp

    for v in pys:
        base_ids = _node_ids(dumps[(v, "p00_int8")])
        base_rows, base_census, base_disp = summarise(v, "p00_int8")
        report.setdefault("p00_int8", {})[v] = {
            "n_nodes": sum(base_census.values()), "n_transposes": len(base_rows),
            "predicted_dispatches_this_release": base_disp[v],
            "transpose_bytes": sum(r["bytes"] or 0 for r in base_rows)}
        bkey = {r["name"]: r for r in base_rows}
        for name, (nodes, _old, _new) in VARIANTS.items():
            rows, census, disp = summarise(v, name)
            key = {r["name"]: r for r in rows}
            added = sorted(set(census) | set(base_census),
                           key=lambda k: -(census.get(k, 0) - base_census.get(k, 0)))
            census_delta = {k: census.get(k, 0) - base_census.get(k, 0)
                            for k in added if census.get(k, 0) != base_census.get(k, 0)}
            changed = sorted(k for k in set(key) | set(bkey)
                             if k not in bkey or k not in key
                             or (key[k]["dims"], key[k]["perm"]) != (bkey[k]["dims"], bkey[k]["perm"]))
            only_intended = changed == sorted(nodes)
            same_names = set(key) == set(bkey)
            ok &= only_intended and same_names
            ids = _node_ids(dumps[(v, name)])
            report.setdefault(name, {})[v] = {
                "nodes_added_vs_original": [list(k) for k in (ids - base_ids)],
                "nodes_removed_vs_original": [list(k) for k in (base_ids - ids)],
                "n_nodes": sum(census.values()),
                "node_count_delta_vs_original": sum(census.values()) - sum(base_census.values()),
                "op_census_delta_vs_original": census_delta,
                "n_transposes": len(rows),
                "transposes_changed": changed, "only_intended_changed": only_intended,
                "same_transpose_name_set": same_names,
                "predicted_dispatches_this_release": disp[v],
                "dispatch_delta_vs_original": disp[v] - base_disp[v],
                "transpose_bytes": sum(r["bytes"] or 0 for r in rows),
                "transpose_bytes_delta": (sum(r["bytes"] or 0 for r in rows)
                                          - sum(r["bytes"] or 0 for r in base_rows)),
                "changed_sites": {k: {"before": bkey.get(k), "after": key.get(k)} for k in changed},
            }
            print(f"  {v} {name:14s} nodes {sum(census.values()):4d} "
                  f"({sum(census.values()) - sum(base_census.values()):+d})  "
                  f"census delta {census_delta or '{}'}  "
                  f"Transposes changed {len(changed)} "
                  f"{'(exactly intended)' if only_intended else '(NOT AS INTENDED)'}  "
                  f"dispatches {base_disp[v]} -> {disp[v]}")
    # the executed graph must still be the SAME graph on both releases, as it is for the original
    for name in ["p00_int8"] + list(VARIANTS):
        a, b = (_node_ids(dumps[(v, name)]) for v in VERSIONS)
        same = a == b
        report[name]["node_ids_identical_across_releases"] = same
        ok &= same
        print(f"  {name:14s} optimized node-name multiset identical across releases: {same}")
    (OUT_DIR / "executed_check.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nexecuted-graph check: {'PASS' if ok else 'FAIL - do not time'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify-static", action="store_true")
    ap.add_argument("--verify-outputs", action="store_true")
    ap.add_argument("--verify-executed", action="store_true")
    ap.add_argument("--scratch", default=None)
    a = ap.parse_args()
    rc = 0
    if a.build:
        build()
    if a.verify_static:
        rc |= 0 if verify_static() else 1
    if a.verify_outputs:
        rc |= 0 if verify_outputs(a.scratch) else 1
    if a.verify_executed:
        rc |= 0 if verify_executed(a.scratch) else 1
    if not any([a.build, a.verify_static, a.verify_outputs, a.verify_executed]):
        ap.print_help()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
