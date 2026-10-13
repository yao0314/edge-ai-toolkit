"""Write the `metrics.json` the submission guidelines ask for, from a packaged deliverable.

The exact schema is in section 8 of the competition packet, which the pre-competition release does
not include (it jumps from 3.6 to 9) — it arrives with the starter kit at kickoff. So this emits a
SUPERSET: one field per thing the rubric scores, under names that say what they are, plus the
provenance needed for the Documentation & Reproducibility criterion. On the day, map the names
onto the official ones with --rename (a JSON file of {ours: theirs}) rather than rewriting this.

Every number is copied from a measured row; nothing here computes or estimates. A value the run
did not produce is written as null, never as a guess — an absent energy figure is a missing
measurement, and the rubric penalises unreproducible numbers more than missing ones.

  python scripts/make_metrics.py --deliverable deliverables/classification
  python scripts/make_metrics.py --deliverable deliverables/classification --rename official.json
"""
import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from _common import ROOT  # noqa: F401  (sets sys.path)

import pandas as pd


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _num(v):
    """A JSON number, or None. NaN is 'not measured', which must not become 0.0 or 'NaN'."""
    if v is None or (isinstance(v, float) and v != v):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return int(f) if f.is_integer() and abs(f) < 2**53 else f


def _ratio(base, got):
    base, got = _num(base), _num(got)
    return None if not base or not got else round(base / got, 4)


def build(deliverable: Path) -> dict:
    meta = json.loads((deliverable / "metadata.json").read_text(encoding="utf-8"))
    m, base = meta.get("metrics", {}), meta.get("baseline", {})

    row = {}
    sweep = deliverable / "sweep.csv"
    if sweep.exists():
        df = pd.read_csv(sweep)
        hit = df[df["variant"].astype(str) == str(meta.get("chosen"))]
        if len(hit):
            row = hit.iloc[0].to_dict()
    base_row = {}
    if sweep.exists() and base.get("variant"):
        hit = pd.read_csv(sweep)
        hit = hit[hit["variant"].astype(str) == str(base["variant"])]
        if len(hit):
            base_row = hit.iloc[0].to_dict()

    def g(key, default=None):
        return row.get(key, m.get(key, default))

    params, base_params = _num(g("params")), _num(base_row.get("params"))
    size, base_size = _num(g("size_bytes")), _num(base.get("size_bytes"))
    acc, base_acc = _num(g("accuracy")), _num(base.get("accuracy"))

    return {
        "schema": "edge-ai-toolkit/metrics/1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),

        "model": {
            "file": meta.get("files", {}).get("best"),
            "format": "onnx",
            "variant": meta.get("chosen"),
            "precision": g("precision"),
            "parameters": params,
            "size_bytes": size,
            "macs": _num(g("macs")),
        },
        "baseline": {
            "variant": base.get("variant"),
            "parameters": base_params,
            "size_bytes": base_size,
            "accuracy": base_acc,
        },
        # The rubric bands compression on the reduction in BOTH, so both ratios are reported and
        # `compression_x` is the weaker one — the half that decides the band.
        "compression": {
            "parameter_reduction_x": _ratio(base_params, params),
            "size_reduction_x": _ratio(base_size, size),
            "compression_x": min([r for r in (_ratio(base_params, params), _ratio(base_size, size))
                                  if r is not None], default=None),
        },
        "accuracy": {
            "metric": g("metric_name"),
            "value": acc,
            "baseline_value": base_acc,
            "retained_fraction": None if not (acc and base_acc) else round(acc / base_acc, 4),
            "validation_set": (meta.get("adapter") or {}).get("val_size"),
        },
        "resource": {
            "latency_p50_ms_1_thread": _num(g("latency_p50_ms_t1")),
            "latency_p50_ms_4_threads": _num(g("latency_p50_ms_t4")),
            "latency_p50_ms": _num(g("latency_p50_ms")),
            "peak_rss_mb": _num(g("peak_rss_mb")),
            "cpu_ms_per_inference": _num(g("cpu_ms_per_inf")),
            "primary_threads": _num(g("primary_threads")),
        },
        "energy": {
            # null means not measured. CodeCarbon reports a MODELLED figure where it cannot read
            # RAPL counters, so the source travels with the number and must not be dropped.
            "joules_per_inference": _num(g("energy_j")),
            "source": g("energy_source") or None,
        },
        "aihub": {
            "device": g("aihub_device"),
            "latency_ms": _num(g("aihub_latency_ms")),
            "peak_memory_mb": _num(g("aihub_peak_memory_mb")),
            "profile_job": g("aihub_profile_job"),
        },
        "quantization": {
            "int8_sqnr_db": _num(g("int8_sqnr_db")),       # below ~15 dB means the quantization broke
            "quantized_nodes": _num(g("int8_quantized_nodes")),
            "calibrate_method": g("calibrate_method"),
        },
        "reproducibility": {
            "git_commit": _git_commit(),
            "toolkit_version": meta.get("toolkit_version"),
            "seed": _num(meta.get("seed")),
            "environment": meta.get("env"),
            "preprocessing": meta.get("preprocessing"),
            "input": meta.get("input"),
            "platform": platform.platform(),
        },
    }


def rename(obj, mapping: dict):
    """Recursively rename keys. Lets the official schema be met without touching build()."""
    if isinstance(obj, dict):
        return {mapping.get(k, k): rename(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rename(v, mapping) for v in obj]
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deliverable", required=True, help="deliverables/<run> written by compress.py")
    ap.add_argument("--rename", help="JSON file of {our_key: official_key}")
    ap.add_argument("--out", help="default: <deliverable>/metrics.json")
    args = ap.parse_args()

    d = Path(args.deliverable)
    if not (d / "metadata.json").exists():
        raise SystemExit(f"{d}/metadata.json not found - run scripts/compress.py first")
    data = build(d)
    if args.rename:
        data = rename(data, json.loads(Path(args.rename).read_text(encoding="utf-8")))

    out = Path(args.out) if args.out else d / "metrics.json"
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")

    missing = [f"{sec}.{k}" for sec, body in data.items() if isinstance(body, dict)
               for k, v in body.items() if v is None]
    print(f"wrote {out}")
    if missing:
        print(f"not measured ({len(missing)}): " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
