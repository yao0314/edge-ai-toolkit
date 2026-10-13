"""Experiment 15: is the INT8 penalty a property of these architectures, or of one ORT release?

This is the challengers' top-priority open question and it can invalidate everything else. Their
pilot (run at 24-52% background load, so not quotable) found MobileViT static INT8 is NOT slower
than fp32 on ORT 1.16.3 (1.006 at 4T) or 1.20.1 (1.056), and that DeiT dynamic is SLOWER than fp32
at 4 threads on 1.20.1 (0.912) - which contradicts the second half of the claim. Everything this
project has measured ran on 1.30.0.

If the penalty appears only on recent releases, the finding is a REGRESSION, not an architecture
property. That is a more useful result and a more publishable one, but it is a different paper, and
it is much better to know before the CNN extension run is built on top of the old framing.

PROTOCOL - the parts that are not optional, each one because something went wrong without it:

  same .onnx files everywhere   The quantizer is held fixed: every runtime is fed the SAME exported
                                model files. Re-quantizing per version would confound the runtime
                                with the quantizer and is the easiest way to get a wrong answer.
  one fresh subprocess per cell A single process leaks ORT thread pools between sessions.
  randomised order              core/sweep.py runs `for ratio: for precision:` with no shuffle, so
                                measurement order is perfectly nested inside variant. The
                                challengers showed that ordering alone reports two plan-identical
                                models as 7.30% apart at 3.9 sigma, collapsing to -0.72% when the
                                order is randomised. Order is shuffled here under a logged seed.
  repeats across sessions       Between-session variance is the dominant term, so repeats must be
                                separate PROCESSES, not more iterations inside one.
  core-delivery gate            other_load_pct is not a valid contamination alarm: over 232 stored
                                measurements it fires on 141 while correlating -0.057 (p=0.39) with
                                actual core delivery. The gate used here is
                                cpu_ms_per_inf / (threads * latency) >= 0.95, computed per cell.

    python scripts/paper/ort_version_bisect.py --repeats 3

Writes results/paper/ort_version_bisect.csv.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from core.cpu_topology import pick_cpus  # noqa: E402

CHILD = textwrap.dedent('''
    import json, sys, time
    import numpy as np, psutil, onnxruntime as ort
    mp, batch, t, aff, shape = (sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4],
                                [int(x) for x in sys.argv[5].split(",")])
    proc = psutil.Process()
    if aff:
        proc.cpu_affinity([int(c) for c in aff.split(",")])
    so = ort.SessionOptions()
    so.intra_op_num_threads = t
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    if len(sys.argv) > 6 and sys.argv[6]:
        so.add_session_config_entry("session.intra_op.allow_spinning", sys.argv[6])
    s = ort.InferenceSession(mp, so, providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    x = np.random.default_rng(0).standard_normal((batch, *shape)).astype(np.float32)
    feed = {name: x}
    warm = time.perf_counter()
    while time.perf_counter() - warm < 1.5:
        s.run(None, feed)
    lat = []
    c0, t0 = proc.cpu_times(), time.perf_counter()
    while time.perf_counter() - t0 < 2.0 or len(lat) < 40:
        a = time.perf_counter()
        s.run(None, feed)
        lat.append((time.perf_counter() - a) * 1e3)
    c1 = proc.cpu_times()
    cpu_ms = ((c1.user - c0.user) + (c1.system - c0.system)) * 1e3
    p50 = float(np.percentile(lat, 50))
    mi = proc.memory_info()
    peak_rss_mb = getattr(mi, "peak_wset", mi.rss) / 2**20   # peak_wset on Windows, rss elsewhere
    print(json.dumps({"ort": ort.__version__, "p50_ms": p50, "n": len(lat),
                      "peak_rss_mb": peak_rss_mb,
                      "cpu_ms_per_inf": cpu_ms / len(lat),
                      # 1.0 = every requested core actually delivered work
                      "core_delivery": (cpu_ms / len(lat)) / (t * p50) if p50 else None}))
''')

FAMILIES = {
    "mobilevit": ("results/classification_mobilevit/models", "3,64,64"),
    "deit": ("results/classification_deit/models", "3,32,32"),
}
VARIANTS = ["p00_fp32", "p00_int8", "p00_int8_dyn_mm"]

# Both settable from the CLI, because the question moved. Experiment 16 measured the fusion-blocking
# variants only on 1.30.0 - inside the regressed regime found by 15b - so MobileViT's result there
# could be about op types or about the 1.24/1.25 regression, and those are different findings.
# Pointing this script at the fusion_blocking model directories under a PRE-regression runtime
# separates them without needing torch or an adapter in the side venvs: the .onnx files already
# exist, and this script only has to time them.
#
#   --families "mobilevit=results/fusion_blocking_mobilevit/models:3,64,64"
#   --variants p00_fp32,p00_int8,p00_int8_static_cm,p00_int8_static_ex,p00_int8_dyn_mm
#   --versions 1.23.2,1.30.0


def parse_families(spec: str) -> dict:
    """"name=dir:shape,shape,shape" entries, comma-separated between entries is ambiguous with the
    shape, so entries are separated by ';'."""
    out = {}
    for entry in spec.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        name, rest = entry.split("=", 1)
        d, shape = rest.rsplit(":", 1)
        out[name.strip()] = (d.strip(), shape.strip())
    return out


def interpreters(scratch: Path, drop_duplicates: bool = False) -> dict:
    """Side venvs, one per ONNX Runtime release, plus the project venv. Keyed by DIRECTORY, not by
    claimed version: the directory names do not reliably match what is installed, so the version is
    taken from what the interpreter reports at run time.

    drop_duplicates: two directories can hold the SAME release - D:/sec_envs/ort1300 and the project
    venv are byte-identical - and that release would then be measured twice under two env_dir values,
    weighting it double in any map over releases. With this flag a directory whose onnxruntime
    binaries hash to a set already seen is dropped, and the drop is printed. Off by default, so every
    experiment that ran before keeps its behaviour."""
    found = {}
    for d in sorted(scratch.glob("ort*")):
        py = d / "Scripts" / "python.exe"
        if py.exists():
            found[d.name] = py
    found["project"] = ROOT / ".venv" / "Scripts" / "python.exe"
    if not drop_duplicates:
        return found

    import hashlib

    def fingerprint(py: Path) -> str:
        """SHA-256 over the runtime's own shipped binaries, not over the venv wrapper."""
        pkg = py.parent.parent / "Lib" / "site-packages" / "onnxruntime" / "capi"
        h = hashlib.sha256()
        for f in sorted(pkg.glob("*")):
            if f.is_file() and f.suffix.lower() in {".pyd", ".dll", ".so"}:
                h.update(f.name.encode())
                h.update(f.read_bytes())
        return h.hexdigest()

    kept, seen = {}, {}
    for name, py in found.items():
        try:
            fp = fingerprint(py)
        except Exception as e:                      # unreadable: keep it and let the run record it
            print(f"  [interpreters] {name}: could not fingerprint ({e}); kept")
            kept[name] = py
            continue
        if fp in seen:
            print(f"  [interpreters] {name} is byte-identical to {seen[fp]}; dropped")
            continue
        seen[fp] = name
        kept[name] = py
    return kept


def _status_row(env, fam, var, t, rep, sp, batch, status, detail=""):
    """A row that says an attempted session produced no measurement, so a reader can tell an empty
    cell from a cell that was never attempted. Only written under --status-rows."""
    return {"env_dir": env, "ort": "", "family": fam, "variant": var, "threads": t, "rep": rep,
            "spin": sp or "default", "batch": batch, "pinned_cpus": "", "p50_ms": float("nan"),
            "iters": 0, "cpu_ms_per_inf": float("nan"), "core_delivery": float("nan"),
            "peak_rss_mb": float("nan"),
            "status": status, "detail": detail}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", default=None, help="directory holding the ortNNN side venvs")
    ap.add_argument("--repeats", type=int, default=3, help="fresh SESSIONS per cell")
    ap.add_argument("--threads", default="1,4")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate", type=float, default=0.95, help="minimum core delivery to trust a cell")
    ap.add_argument("--drop-duplicate-envs", action="store_true",
                    help="drop a side venv whose onnxruntime binaries are byte-identical to one "
                         "already seen, so one release cannot be measured twice under two names")
    ap.add_argument("--status-rows", action="store_true",
                    help="write a row for EVERY attempted session, with a status column, instead "
                         "of silently skipping a missing file or a failed child")
    ap.add_argument("--exclude", default="",
                    help="';'-separated family:variant pairs to leave out of the grid, for cells "
                         "a release cannot run (e.g. 'mnv2:p00_int8_dyn_conv')")
    ap.add_argument("--families", default=None,
                    help="';'-separated name=dir:c,h,w entries; default is the two headline families")
    ap.add_argument("--variants", default=None, help="comma-separated variant stems")
    ap.add_argument("--versions", default=None,
                    help="comma-separated reported versions to restrict to (default: all found)")
    ap.add_argument("--out", default="results/paper/ort_version_bisect.csv")
    # With ORT's default spinning, idle intra-op threads burn CPU, so cpu time ~ threads x wall by
    # construction and says nothing about work done. Making spin a randomised factor lets the
    # spin-off cells carry that information. Core delivery below the gate is EXPECTED at spin=0
    # (threads really sleep), so the gate is not a contamination check for those cells.
    ap.add_argument("--spin", default="",
                    help="comma-separated session.intra_op.allow_spinning values, e.g. 1,0; "
                         "empty leaves the runtime default")
    a = ap.parse_args()

    scratch = Path(a.scratch) if a.scratch else Path(__file__).resolve().parent / "_venvs"
    pys = interpreters(scratch, drop_duplicates=a.drop_duplicate_envs)
    if len(pys) < 2:
        print(f"need at least two interpreters; found {list(pys)} under {scratch}", file=sys.stderr)
        print("build side venvs first, e.g.  uv venv ort120 && uv pip install -p ort120 "
              "onnxruntime==1.20.1 numpy psutil", file=sys.stderr)
        return 1
    child = scratch / "_bisect_child.py"
    child.parent.mkdir(parents=True, exist_ok=True)
    child.write_text(CHILD, encoding="utf-8")

    families = parse_families(a.families) if a.families else FAMILIES
    variants = [v.strip() for v in a.variants.split(",")] if a.variants else VARIANTS
    if a.versions:
        want = {v.strip() for v in a.versions.split(",")}
        # resolve by REPORTED version, since directory names do not track what is installed
        keep = {}
        for env, py in pys.items():
            r = subprocess.run([str(py), "-c", "import onnxruntime as o; print(o.__version__)"],
                               capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip() in want:
                keep[env] = py
        if not keep:
            print(f"no interpreter reports a version in {sorted(want)}", file=sys.stderr)
            return 1
        pys = keep
    threads = [int(t) for t in a.threads.split(",")]
    spins = [s.strip() for s in a.spin.split(",")] if a.spin else [""]
    cells = [(env, fam, var, t, rep, sp)
             for env in pys for fam in families for var in variants
             for t in threads for rep in range(a.repeats) for sp in spins]
    # Shuffle across EVERYTHING, so variant, thread count and version are all decorrelated from
    # position in time. Thermal drift then adds variance instead of bias.
    rng = random.Random(a.seed)
    rng.shuffle(cells)
    print(f"{len(pys)} interpreter(s), {len(cells)} measurements, seed {a.seed}, "
          f"randomised order\n")

    excluded = {tuple(x.split(":", 1)) for x in a.exclude.split(";") if ":" in x}
    if excluded:
        print(f"  excluding {len(excluded)} (family, variant) pairs: {sorted(excluded)}")
    rows = []
    for i, (env, fam, var, t, rep, sp) in enumerate(cells, 1):
        d, shape = families[fam]
        model = ROOT / d / f"{var}.onnx"
        if (fam, var) in excluded:
            if a.status_rows:
                rows.append(_status_row(env, fam, var, t, rep, sp, a.batch, "EXCLUDED"))
            continue
        if not model.exists():
            if a.status_rows:
                rows.append(_status_row(env, fam, var, t, rep, sp, a.batch, "NO_FILE"))
            continue
        aff = ",".join(map(str, pick_cpus(t, "auto") or []))
        r = subprocess.run([str(pys[env]), str(child), str(model), str(a.batch), str(t), aff, shape,
                            sp], capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            print(f"  [{i}/{len(cells)}] {env} {fam} {var} t{t} FAILED: {r.stderr.strip()[-120:]}")
            if a.status_rows:
                rows.append(_status_row(env, fam, var, t, rep, sp, a.batch, "FAILED",
                                        r.stderr.strip()[-200:]))
            continue
        j = json.loads(r.stdout.strip().splitlines()[-1])
        rows.append({"env_dir": env, "ort": j["ort"], "family": fam, "variant": var,
                     "threads": t, "rep": rep, "spin": sp or "default",
                     "batch": a.batch, "pinned_cpus": aff,
                     "p50_ms": j["p50_ms"], "iters": j["n"],
                     "cpu_ms_per_inf": j["cpu_ms_per_inf"],
                     "core_delivery": j["core_delivery"],
                     "peak_rss_mb": j.get("peak_rss_mb"),
                     **({"status": "OK", "detail": ""} if a.status_rows else {})})
        flag = "" if (j["core_delivery"] or 0) >= a.gate else "  LOW CORE DELIVERY"
        print(f"  [{i}/{len(cells)}] ort {j['ort']:8s} {fam:10s} {var:16s} t{t} r{rep} "
              f"{'spin' + sp + ' ' if sp else ''}"
              f"{j['p50_ms']:8.3f} ms  cd {j['core_delivery']:.2f}{flag}", flush=True)

    if not rows:
        print("no measurements completed", file=sys.stderr)
        return 1
    df = pd.DataFrame(rows)
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    if a.spin:
        # the summary below pools cells by version/variant/threads, which would mix spin settings
        print(f"\nwrote {out} ({len(df)} rows); spin is a factor - analyse per spin value")
        return 0

    bad = df[df["core_delivery"] < a.gate]
    if len(bad):
        print(f"\n{len(bad)}/{len(df)} measurements below the {a.gate} core-delivery gate; "
              f"they stay in the CSV but are excluded from the table below.")
    ok = df[df["core_delivery"] >= a.gate]

    med = ok.groupby(["ort", "family", "variant", "threads"])["p50_ms"].agg(["median", "std", "size"])
    print("\n" + "=" * 78)
    print("INT8 speedup vs the fp32 sibling, per ORT version (median over sessions)")
    print(f"{'ort':10s} {'family':11s} {'thr':>3s} {'static':>9s} {'dynamic':>9s}   "
          f"{'fp32 ms':>9s} {'n':>3s}")
    summary = []
    for (ort_v, fam, t), _ in med.groupby(level=["ort", "family", "threads"]):
        def m(v):
            try:
                return float(med.loc[(ort_v, fam, v, t), "median"])
            except KeyError:
                return None
        fp, st, dy = m("p00_fp32"), m("p00_int8"), m("p00_int8_dyn_mm")
        if not fp:
            continue
        n = int(med.loc[(ort_v, fam, "p00_fp32", t), "size"])
        s_x = fp / st if st else float("nan")
        d_x = fp / dy if dy else float("nan")
        summary.append({"ort": ort_v, "family": fam, "threads": t,
                        "static_speedup": s_x, "dynamic_speedup": d_x, "fp32_ms": fp})
        print(f"{ort_v:10s} {fam:11s} {t:3d} {s_x:9.3f} {d_x:9.3f}   {fp:9.3f} {n:3d}")

    s = pd.DataFrame(summary)
    if len(s) and s["ort"].nunique() > 1:
        print("\nDoes the penalty exist on every version?")
        for (fam, t), g in s.groupby(["family", "threads"]):
            lo, hi = g["static_speedup"].min(), g["static_speedup"].max()
            verdict = ("penalty on EVERY version" if hi < 1.0 else
                       "penalty on NO version" if lo > 1.0 else
                       "VERSION-DEPENDENT - the finding is a regression, not an architecture property")
            print(f"  {fam:11s} t{t}: static speedup {lo:.3f}-{hi:.3f}  ->  {verdict}")
        print("\nFALSIFICATION (pre-registered): if MobileViT static is at or above 1.0 on an older")
        print("release and below 1.0 only on recent ones, the architecture claim is refuted and")
        print("must be rewritten as an ONNX Runtime regression report.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
