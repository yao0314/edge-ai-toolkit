"""Write metrics.json in exactly the schema of the task sheet, section 8.

`scripts/make_metrics.py` emits a superset under our own names. This one emits the organisers'
structure and units verbatim, because they differ from ours in both: nested baseline/final blocks,
parameters in millions, size in MiB, MACs in billions. A key rename could not have done it.

Both blocks come from the same sweep, so baseline and final are measured the same way — the sheet
is explicit that the ratios are meaningless otherwise. `--baseline-pt` replaces only the baseline's
size with the provided checkpoint's own file size, for a judge who reads "size" as the artifact
they handed out rather than its ONNX export; the report should then say which was used.

  python scripts/sec_metrics.py --sweep results/<run>/sweep.csv --final <variant> --team "NAME"
  python scripts/sec_metrics.py ... --baseline-pt student_mobilenetv2_distilled.pt
"""
from __future__ import annotations

import argparse
import ast
import json
import platform
from pathlib import Path

import pandas as pd

MIB = 1024 * 1024


def num(v, default=None):
    """A JSON number, or the default. NaN means not measured and must not become 0.0."""
    if v is None or (isinstance(v, float) and v != v):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def block(row, size_bytes=None) -> dict:
    """One of the sheet's two measurement blocks, in its units."""
    size = size_bytes if size_bytes is not None else num(row.get("size_bytes"), 0.0)
    return {
        "accuracy": round(num(row.get("accuracy"), 0.0), 6),
        "params_million": round(num(row.get("params"), 0.0) / 1e6, 4),
        "size_mb": round(size / MIB, 4),
        "macs_giga": round(num(row.get("macs"), 0.0) / 1e9, 4),
        "latency_ms": round(num(row.get("latency_p50_ms"), 0.0), 4),
        "peak_ram_mb": round(peak_ram(row), 3),
        "joules_per_inference": round(num(row.get("energy_j"), 0.0), 6),
    }


def peak_ram(row) -> float:
    """Raw process peak RSS, which is what the organisers specified: no baseline subtraction.

    peak_rss_mb in a sweep CSV is NET of an empty ONNX Runtime session, and on this task the two
    differ by about threefold - reporting the delta would overstate the RAM reduction and would not
    match what a judge sampling process RSS sees. Older CSVs predate the un-suffixed
    peak_rss_total_mb column, so fall back to the primary thread count's copy; never fall back to
    peak_rss_mb, because silently reporting a different quantity under the same name is the whole
    failure this guards against.
    """
    v = num(row.get("peak_rss_total_mb"))
    if v is not None:
        return v
    p = row.get("primary_threads")
    if p is not None and p == p:
        v = num(row.get("peak_rss_total_mb_t%d" % int(p)))
        if v is not None:
            return v
    for key in row.index if hasattr(row, "index") else row:   # any per-thread copy, largest wins
        if str(key).startswith("peak_rss_total_mb_t"):
            v = num(row.get(key))
            if v is not None:
                return v
    raise SystemExit("no peak_rss_total_mb column in the sweep: re-measure, or the reported "
                     "peak RAM would be the empty-session delta rather than the process peak")


def worst_class(row) -> float:
    """The sheet asks for worst_class_accuracy; the adapter reports per-class recall."""
    v = num(row.get("worst_class_recall"))
    if v is not None:
        return round(v, 6)
    cells = row.get("per_class_recall")          # a CSV round trip makes it a string
    if isinstance(cells, str):
        try:
            vals = ast.literal_eval(cells)
            return round(float(min(vals)), 6)
        except Exception:
            return 0.0
    return 0.0


def codecarbon_mode(row) -> str:
    """"measured" only when CodeCarbon read hardware counters. Its own fallback is a TDP model,
    which the sheet says is not penalised but is weaker evidence - so it must not be overstated."""
    src = str(row.get("energy_source") or "")
    return "measured" if "rapl" in src.lower() else "tdp_estimate"


def techniques(row, cfg_hint: str | None, arch_change: bool = False) -> list:
    """What was actually done, for the sheet's techniques_applied field.

    `arch_change` is a flag rather than an inference. It used to be read off the variant name - any
    prefix before the pXX part meant a distilled student - which is true for a student ladder and
    silently false for any other prefix. Naming a KD-recovery run `kd_p60_int8` was enough to put
    "architecture change" into a submitted metrics.json for a model whose architecture never
    changed. A claim about what we did is not something to infer from a filename.
    """
    out = []
    if arch_change:
        out.append("architecture change distilled from the provided teacher")
    if num(row.get("prune_ratio"), 0):
        out.append("%s pruning (%.0f%% of channels)"
                   % (row.get("prune_mode") or "structured", 100 * num(row.get("prune_ratio"), 0)))
    if str(row.get("precision", "")).startswith("int8"):
        out.append("static INT8 quantization (QDQ, per-channel weights)")
    out += list(cfg_hint or [])
    return out or ["none applied"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", required=True, help="results/<run>/sweep.csv")
    ap.add_argument("--final", required=True, help="variant name of the submitted model")
    ap.add_argument("--baseline", default="p00_fp32", help="variant name of the provided baseline")
    ap.add_argument("--team", required=True)
    ap.add_argument("--baseline-pt", help="use this checkpoint's file size as the baseline size")
    ap.add_argument("--machine", help='e.g. "MSI GP65, Ryzen 7 4800H, 16 GB, CPU only"')
    ap.add_argument("--aihub", default="", help="key figures from the profile job")
    ap.add_argument("--note", action="append", help="one more technique to list; repeatable")
    ap.add_argument("--architecture-change", action="store_true",
                    help="the submitted model is a different architecture distilled from the teacher")
    ap.add_argument("--out", default="metrics.json")
    a = ap.parse_args()

    df = pd.read_csv(a.sweep)
    for name in (a.baseline, a.final):
        if not (df["variant"].astype(str) == name).any():
            raise SystemExit("variant %r not in %s. Present: %s"
                             % (name, a.sweep, ", ".join(df["variant"].astype(str))))
    base = df[df["variant"].astype(str) == a.baseline].iloc[0]
    final = df[df["variant"].astype(str) == a.final].iloc[0]

    pt = Path(a.baseline_pt).stat().st_size if a.baseline_pt else None
    report = {
        "team_name": a.team,
        "baseline": block(base, pt),
        "final": block(final),
        "worst_class_accuracy": worst_class(final),
        "codecarbon_mode": codecarbon_mode(final),
        "techniques_applied": techniques(final, a.note, a.architecture_change),
        "ai_hub_profile_summary": a.aihub or (
            "%s: %.2f ms, %.1f MB peak" % (final.get("aihub_device"),
                                           num(final.get("aihub_latency_ms"), 0.0),
                                           num(final.get("aihub_peak_memory_mb"), 0.0))
            if num(final.get("aihub_latency_ms")) else "not profiled"),
        "measurement_machine": a.machine or platform.platform(),
    }
    Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    b, f = report["baseline"], report["final"]
    print("wrote %s\n" % a.out)
    print("%-22s %14s %14s %10s" % ("", "baseline", "final", "ratio"))
    for k in ("accuracy", "params_million", "size_mb", "macs_giga", "latency_ms",
              "peak_ram_mb", "joules_per_inference"):
        r = "-" if not f[k] else "%.2fx" % (b[k] / f[k]) if b[k] else "-"
        print("%-22s %14.4f %14.4f %10s" % (k, b[k], f[k], r))
    print("\nworst_class_accuracy %.4f | codecarbon_mode %s"
          % (report["worst_class_accuracy"], report["codecarbon_mode"]))
    missing = [k for k, v in f.items() if not v]
    if missing:
        print("NOT MEASURED (written as 0.0, fix before submitting): " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
