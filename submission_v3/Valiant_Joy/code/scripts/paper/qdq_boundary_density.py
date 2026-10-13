"""Q/DQ boundary density: a static predictor of whether INT8 will pay off.

Experiment 1 of results/paper/EXPERIMENT_PLAN.md.

The hypothesis is that what costs time is not HOW MANY ops were quantized but how many times the
graph CROSSES between int8 and fp32 regions, because each crossing is a quantize/dequantize
round-trip that performs no useful arithmetic.

So the quantity measured here is not a node count. It is:

  round_trips   a DequantizeLinear whose producer is a QuantizeLinear, with no compute between:
                pure overhead, the tensor leaves int8 and comes straight back to fp32
  islands       DQ -> compute -> Q: a genuine stretch of int8 arithmetic, which is what we want
  boundary_ops  compute nodes with at least one quantized and one float input/output edge

Three denominators are computed and ALL are reported. Picking the one that correlates after seeing
the data would manufacture the result, so the pre-registered primary is per_mac.

CORRECTION, 2026-09-16 (experiment 14). The first version of this script counted the graph stored
in the .onnx FILE. ORT's QDQSelectorActionTransformer fuses Q/DQ pairs into QLinear* kernels during
graph optimisation and deletes roughly 90% of them before the first inference - MobileViT 416/599
-> 40/97, DeiT 442/593 -> 63/102, ResNet50 74/182 -> 1/1. Every density figure produced on the file
graph was therefore counting nodes that never execute, and recounted on the executed graph the
variable points the OTHER WAY: the faster family runs more surviving Q/DQ than the slower one.

The fraction kept is model-specific, so this is not a constant-factor correction - it reverses
orderings. `--graph executed` is now the default view and `--graph both` prints the size of the
error per model, because that difference is itself the most useful thing this script produces.

    python scripts/paper/qdq_boundary_density.py

Writes results/paper/qdq_density.csv (one row per .onnx).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import onnx  # noqa: E402
import pandas as pd  # noqa: E402

Q, DQ = "QuantizeLinear", "DequantizeLinear"
# Ops that do real arithmetic; everything else (Reshape, Transpose, Concat...) is plumbing and must
# not count as an int8 "island", or a graph that only reshapes between Q/DQ would look productive.
COMPUTE = {"Conv", "ConvTranspose", "Gemm", "MatMul", "MatMulInteger", "QLinearConv",
           "QLinearMatMul", "ConvInteger", "Einsum", "LSTM", "GRU", "RNN", "Attention"}


def analyse(path: Path) -> dict | None:
    try:
        m = onnx.load(str(path), load_external_data=False)
    except Exception as e:
        print(f"  skip {path.name}: {e}", file=sys.stderr)
        return None
    g = m.graph
    nodes = list(g.node)
    producer = {}
    for n in nodes:
        for o in n.output:
            producer[o] = n
    consumers: dict = {}
    for n in nodes:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    ops = Counter(n.op_type for n in nodes)
    n_q, n_dq = ops[Q], ops[DQ]

    round_trips = 0
    for n in nodes:
        if n.op_type != DQ:
            continue
        p = producer.get(n.input[0])
        if p is not None and p.op_type == Q:
            round_trips += 1

    # A DQ that feeds compute starts an int8 island; count islands by their entry point.
    islands = 0
    for n in nodes:
        if n.op_type != DQ:
            continue
        if any(c.op_type in COMPUTE for c in consumers.get(n.output[0], [])):
            islands += 1

    boundary_ops = 0
    quantized_edges = set()
    for n in nodes:
        if n.op_type in (Q, DQ):
            quantized_edges.update(n.output)
    for n in nodes:
        if n.op_type not in COMPUTE:
            continue
        touches_q = any(i in quantized_edges for i in n.input)
        feeds_q = any(any(c.op_type == Q for c in consumers.get(o, [])) for o in n.output)
        if touches_q or feeds_q:
            boundary_ops += 1

    compute_nodes = sum(ops[k] for k in COMPUTE if k in ops)
    fp32_compute = [n.op_type for n in nodes
                    if n.op_type in COMPUTE and not any(i in quantized_edges for i in n.input)]

    # MACs from the ONNX graph, so the denominator does not need a torch model.
    try:
        from core.measure import count_macs_onnx
        macs = count_macs_onnx(path)
    except Exception:
        macs = float("nan")

    return {"file": path.name, "run": path.parent.parent.name,
            "variant": path.stem, "is_int8": "int8" in path.stem,
            "n_nodes": len(nodes), "n_compute": compute_nodes, "macs": macs,
            "n_quantize": n_q, "n_dequantize": n_dq,
            "round_trips": round_trips, "islands": islands, "boundary_ops": boundary_ops,
            "fp32_compute_left": len(fp32_compute),
            "fp32_compute_types": ",".join(sorted(set(fp32_compute))) or "",
            # pre-registered primary denominator: per MAC. The other two are reported for
            # transparency, not for selection.
            "boundary_per_gmac": boundary_ops / (macs / 1e9) if macs and macs == macs else float("nan"),
            "boundary_per_node": boundary_ops / len(nodes) if nodes else float("nan"),
            "boundary_per_compute": boundary_ops / compute_nodes if compute_nodes else float("nan"),
            "roundtrip_frac_of_qdq": round_trips / (n_q + n_dq) if (n_q + n_dq) else float("nan")}


def optimized_copy(path: Path, out_dir: Path):
    """Return the graph ORT will ACTUALLY execute, not the one stored in the file.

    This is the repair for the error that killed the first version of this experiment. ORT's
    QDQSelectorActionTransformer fuses QuantizeLinear/DequantizeLinear pairs into QLinear* kernels
    during graph optimisation, deleting roughly 90% of them before the first inference: measured
    416/599 -> 40/97 on MobileViT and 74/182 -> 1/1 on ResNet50. Every density figure this script
    produced from the file graph was therefore counting nodes that never run, and recounted on the
    executed graph the correlation with latency points the other way.

    Returns None when the runtime cannot load the model - a file-graph row is still worth having,
    clearly labelled, rather than dropping the model.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"opt_{path.stem}.onnx"
    try:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.optimized_model_filepath = str(dst)
        so.log_severity_level = 3
        # intra_op 1: optimisation is deterministic, and spinning up a pool here would compete with
        # whatever is being measured elsewhere.
        so.intra_op_num_threads = so.inter_op_num_threads = 1
        ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"  {path.stem}: could not produce an optimized graph ({type(e).__name__}: {e})",
              file=sys.stderr)
        return None
    return dst if dst.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="results/*/models/*.onnx")
    ap.add_argument("--out", default="results/paper/qdq_density.csv")
    ap.add_argument("--graph", choices=["file", "executed", "both"], default="both",
                    help="which graph to count on. 'file' is what this script originally did and "
                         "is wrong on its own: ORT deletes ~90%% of Q/DQ before inference.")
    a = ap.parse_args()

    files = sorted(ROOT.glob(a.glob))
    print(f"analysing {len(files)} ONNX file(s), graph={a.graph}")
    print("")
    opt_dir = ROOT / "results" / "paper" / "_optgraphs"
    rows = []
    for f in files:
        if a.graph in ("file", "both"):
            r = analyse(f)
            if r is not None:
                rows.append({**r, "graph": "file"})
        if a.graph in ("executed", "both"):
            o = optimized_copy(f, opt_dir)
            if o is not None:
                r = analyse(o)
                if r is not None:
                    # restore the ORIGINAL identity: analyse() read a temp file named opt_<stem>
                    rows.append({**r, "graph": "executed", "file": f.name,
                                 "variant": f.stem, "run": f.parent.parent.name})
                o.unlink(missing_ok=True)
    try:
        opt_dir.rmdir()
    except OSError:
        pass
    df = pd.DataFrame(rows)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    if "graph" in df and df["graph"].nunique() > 1:
        # The headline of this script is now the DIFFERENCE between the two graphs: it is the size
        # of the error that killed experiment 1, and it is model-specific, so it cannot be corrected
        # with a constant factor.
        key = [c for c in ("run", "variant") if c in df]
        piv = df[df["is_int8"]].pivot_table(index=key, columns="graph",
                                            values=["n_quantize", "n_dequantize", "boundary_ops"],
                                            aggfunc="first")
        print("Q/DQ nodes in the FILE vs in the graph ORT actually executes:")
        print(f"  {'model':40s} {'file Q':>7s} {'exec Q':>7s} {'file DQ':>8s} {'exec DQ':>8s} {'kept':>6s}")
        for idx, r in piv.iterrows():
            fq, eq = r.get(("n_quantize", "file")), r.get(("n_quantize", "executed"))
            fd, ed = r.get(("n_dequantize", "file")), r.get(("n_dequantize", "executed"))
            if pd.isna(fq) or pd.isna(eq):
                continue
            name = "/".join(str(x) for x in (idx if isinstance(idx, tuple) else (idx,)))
            kept = (eq + ed) / (fq + fd) if (fq + fd) else float("nan")
            print(f"  {name[:40]:40s} {fq:7.0f} {eq:7.0f} {fd:8.0f} {ed:8.0f} {kept:6.1%}")
        print("")
        print("Any density figure computed on the file graph counts nodes that never run.")
        print("The 'kept' fraction is model-specific, so this is not a constant-factor")
        print("correction: it can and does reverse the ordering between models.")
        print("")

    q = df[df["is_int8"]]
    if "graph" in q:
        q = q[q["graph"] == ("file" if a.graph == "file" else "executed")]
    print(f"{len(q)} INT8 model(s) on the {a.graph if a.graph != 'both' else 'executed'} graph:")
    print(f"  boundary_ops        {q['boundary_ops'].min():.0f} .. {q['boundary_ops'].max():.0f}")
    print(f"  round_trips         {q['round_trips'].min():.0f} .. {q['round_trips'].max():.0f}")
    print(f"  boundary_per_gmac   {q['boundary_per_gmac'].min():.1f} .. {q['boundary_per_gmac'].max():.1f}")
    print(f"  fp32 compute left   {q['fp32_compute_left'].min():.0f} .. {q['fp32_compute_left'].max():.0f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
