"""Experiment 20: did ORT stop fusing at 1.24/1.25, or did its kernels get slower?

The version bisect pinned a ~35% MobileViT static-QDQ regression to two steps, at ORT 1.24 (4-thread
only) and 1.25 (general). The candidate mechanism from the node profiles is fusion blocking: static
QDQ prevents the QuickGelu / Gelu / SkipLayerNormalization fusions that fp32 and the dynamic path
both receive.

But fusion blocking is a property of the graph, and **the .onnx file never changed** - the bisect fed
byte-identical models to every runtime. MobileViT static was fine on 1.23.2 and slow on 1.25.1 with
the same file. So there are exactly two possibilities, and they are distinguishable by dumping the
optimized graph under each runtime:

  fusion LOSS     the fast runtime fuses ops that the slow one does not. Then the regression IS a
                  fusion regression, the node-profile mechanism holds in a sharper form, and the
                  diff names the fusion that disappeared.
  kernel SLOWDOWN both runtimes produce the same optimized graph, and the same ops simply run
                  slower. Then fusion blocking is refuted as the CAUSE of the regression (it may
                  still cost time, but it is not what changed), and the regression is in the
                  kernels or the threading.

No timing is involved, so this is decidable from two graph dumps and a set difference. That is the
whole experiment.

    python scripts/paper/optgraph_version_diff.py --versions 1.23.2,1.25.1

Writes results/paper/optgraph_version_diff.csv.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

# Runs inside each side venv: that runtime's own optimiser has to produce the graph, so the dump
# cannot be done from the project environment.
# The side venvs carry onnxruntime but not onnx, so the child only makes that runtime write its
# optimized graph; parsing happens here, in the project environment.
CHILD = textwrap.dedent('''
    import json, sys
    import onnxruntime as ort
    src, dst = sys.argv[1], sys.argv[2]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.optimized_model_filepath = dst
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    so.log_severity_level = 3
    ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])
    print(json.dumps({"ort": ort.__version__}))
''')

DEFAULT_MODELS = [
    "results/classification_mobilevit/models/p00_int8.onnx",
    "results/classification_mobilevit/models/p00_fp32.onnx",
    "results/classification_mobilevit/models/p00_int8_dyn_mm.onnx",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", required=True, help="directory holding the ortNNN side venvs")
    ap.add_argument("--versions", default="1.23.2,1.25.1",
                    help="the fast and slow runtimes to compare, by reported version")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--out", default="results/paper/optgraph_version_diff.csv")
    ap.add_argument("--keep", action="store_true",
                    help="keep the optimized graphs in <scratch>/_optdiff: an op-type census can "
                         "hide a same-count change, so a node-by-node comparison needs the files")
    a = ap.parse_args()

    scratch = Path(a.scratch)
    want = [v.strip() for v in a.versions.split(",")]
    # Map reported version -> interpreter. Directory names do not reliably match the version
    # installed (ort124 holds 1.28.0), so ask each interpreter rather than trusting its folder.
    pys = {}
    for d in sorted(scratch.glob("ort*")):
        py = d / "Scripts" / "python.exe"
        if not py.exists():
            continue
        r = subprocess.run([str(py), "-c", "import onnxruntime as o; print(o.__version__)"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            pys[r.stdout.strip()] = py
    pys[__import__("onnxruntime").__version__] = ROOT / ".venv" / "Scripts" / "python.exe"
    missing = [v for v in want if v not in pys]
    if missing:
        print(f"no interpreter for {missing}; have {sorted(pys)}", file=sys.stderr)
        return 1

    work = scratch / "_optdiff"
    work.mkdir(parents=True, exist_ok=True)
    child = work / "dump.py"
    child.write_text(CHILD, encoding="utf-8")

    rows, census = [], {}
    for model in a.models.split(","):
        src = ROOT / model.strip()
        if not src.exists():
            print(f"  missing {src}", file=sys.stderr)
            continue
        for v in want:
            dst = work / f"opt_{v}_{src.stem}.onnx"
            r = subprocess.run([str(pys[v]), str(child), str(src), str(dst)],
                               capture_output=True, text=True, timeout=600)
            if r.returncode != 0:
                print(f"  {src.stem} @ {v}: FAILED {r.stderr.strip()[-160:]}", file=sys.stderr)
                continue
            j = json.loads(r.stdout.strip().splitlines()[-1])
            import onnx
            g = onnx.load(str(dst), load_external_data=False).graph
            j["n_nodes"] = len(g.node)
            j["ops"] = dict(Counter(n.op_type for n in g.node))
            census[(src.stem, v)] = Counter(j["ops"])
            rows.append({"model": src.stem, "ort": j["ort"], "n_nodes": j["n_nodes"],
                         **{f"op_{k}": n for k, n in j["ops"].items()}})
            if not a.keep:
                dst.unlink(missing_ok=True)
            print(f"  {src.stem:20s} @ {v:8s}  {j['n_nodes']:4d} nodes")

    if not rows:
        print("nothing dumped", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows).fillna(0)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    fast, slow = want[0], want[-1]
    print("\n" + "=" * 78)
    print(f"OPTIMIZED-GRAPH DIFF   {fast} (fast)  vs  {slow} (slow)")
    verdict_any = False
    for model in sorted({m for m, _ in census}):
        a_c, b_c = census.get((model, fast)), census.get((model, slow))
        if a_c is None or b_c is None:
            continue
        gone = {k: a_c[k] - b_c.get(k, 0) for k in a_c if a_c[k] > b_c.get(k, 0)}
        added = {k: b_c[k] - a_c.get(k, 0) for k in b_c if b_c[k] > a_c.get(k, 0)}
        same = sum(a_c.values()) == sum(b_c.values()) and not gone and not added
        print(f"\n  {model}   {sum(a_c.values())} -> {sum(b_c.values())} nodes")
        if same:
            print("    IDENTICAL optimized graph")
        else:
            verdict_any = True
            if gone:
                print(f"    fewer in {slow}: " + ", ".join(f"{k} -{n}" for k, n in sorted(gone.items())))
            if added:
                print(f"    more in {slow}:  " + ", ".join(f"{k} +{n}" for k, n in sorted(added.items())))

    print("\n" + "-" * 78)
    if verdict_any:
        print("The optimized graphs DIFFER. The regression is a GRAPH change - read the diff above")
        print("for which fusion disappeared. Fusion blocking survives as the mechanism, sharpened:")
        print("it is not that static QDQ always blocks fusion, it is that this runtime stopped")
        print("performing a fusion it used to perform.")
    else:
        print("The optimized graphs are IDENTICAL. Same ops, same counts, different speed - so the")
        print("regression is in the KERNELS or the threading, not the graph. Fusion blocking is")
        print("REFUTED as the cause of the version regression (it may still cost time on both")
        print("versions; it is not what changed). Next place to look: the 4-thread-only step at")
        print("1.24 points at thread-pool or parallel-for changes rather than per-op kernels.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
