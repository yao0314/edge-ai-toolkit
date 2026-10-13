"""Section 8 of the task sheet: write metrics.json in the official schema.

    python sec2026/make_metrics.py --baseline sec2026/results/baseline_onnx.json \
        --final sec2026/results/final_int8.json --team "<team>" \
        --technique "structured pruning (20%)" --technique "static INT8 quantization (QDQ)" \
        --aihub sec2026/out/p20/aihub_student_model_int8.json --out submission/metrics.json

--baseline and --final are measure.py results from the SAME machine with the same settings (the
sheet: otherwise the ratios mean nothing). The script refuses a pair measured on different machines
or with different thread/run settings, and prints the official baseline_metrics.json next to ours
so any disagreement is visible before submission. Units follow the sheet: params in millions, size
in MiB, MACs in giga, latency in ms, peak RAM in MB, joules per inference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

KEYS = ["accuracy", "params_million", "size_mb", "macs_giga", "latency_ms", "peak_ram_mb",
        "joules_per_inference"]


def block(m: dict) -> dict:
    acc = (m.get("accuracy") or {})
    if acc and not acc.get("full_val_split", True):
        raise SystemExit(f"{m['name']}: accuracy came from --val-limit, not the full val split")
    r = lambda v, n: None if v is None else round(float(v), n)
    return {"accuracy": r(acc.get("accuracy"), 6), "params_million": r(m["params"] / 1e6, 6),
            "size_mb": r(m["size_mb"], 4), "macs_giga": r(m["macs"] / 1e9, 6),
            "latency_ms": r(m.get("latency_ms"), 4), "peak_ram_mb": r(m.get("peak_ram_mb"), 3),
            "joules_per_inference": r(m.get("joules_per_inference"), 6)}


def same_rig(b: dict, f: dict) -> list:
    problems = []
    for k in ("cpu", "platform", "onnxruntime"):
        if b["machine"].get(k) != f["machine"].get(k):
            problems.append(f"machine.{k}: {b['machine'].get(k)} vs {f['machine'].get(k)}")
    for k in ("threads", "runs", "repeats", "energy_seconds"):
        if b["settings"].get(k) != f["settings"].get(k):
            problems.append(f"settings.{k}: {b['settings'].get(k)} vs {f['settings'].get(k)}")
    return problems


def ai_hub_summary(paths) -> str:
    parts = []
    for p in paths or []:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        s = d.get("summary", {})
        t, mem = s.get("inference_time_ms"), s.get("peak_memory_mb")
        parts.append(f"{Path(d['model']).name} on {d['device']}: "
                     + (f"{t:.3f} ms" if t is not None else "time n/a") + ", "
                     + (f"peak memory {mem:.2f} MB" if mem is not None else "peak memory n/a")
                     + f" (profile job {d.get('profile_job')})")
    return "; ".join(parts) if parts else "not profiled (AI Hub unavailable): local measurements only"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--final", required=True)
    ap.add_argument("--team", required=True)
    ap.add_argument("--technique", action="append", default=[])
    ap.add_argument("--aihub", action="append", help="aihub_profile.py output(s)")
    ap.add_argument("--machine", help="e.g. 'MSI Bravo 15, Ryzen 7 4800H, 16 GB, CPU only'")
    ap.add_argument("--kit")
    ap.add_argument("--out", default="metrics.json")
    ap.add_argument("--force", action="store_true", help="write even if the two runs differ in rig/settings")
    a = ap.parse_args(argv)

    import common

    b = json.loads(Path(a.baseline).read_text(encoding="utf-8"))
    f = json.loads(Path(a.final).read_text(encoding="utf-8"))
    problems = same_rig(b, f)
    if problems and not a.force:
        raise SystemExit("baseline and final were not measured the same way:\n  " + "\n  ".join(problems))
    modes = {b.get("codecarbon_mode"), f.get("codecarbon_mode")} - {None}
    mi = f["machine"]
    report = {
        "team_name": a.team,
        "baseline": block(b),
        "final": block(f),
        "worst_class_accuracy": round(float(f["accuracy"]["worst_class_accuracy"]), 6),
        "codecarbon_mode": "measured" if modes == {"measured"} else "tdp_estimate",
        "techniques_applied": a.technique,
        "ai_hub_profile_summary": ai_hub_summary(a.aihub),
        "measurement_machine": a.machine or f"{mi.get('cpu')}, {mi.get('ram_gb')} GB, CPU only, "
                                            f"{mi.get('platform')}, onnxruntime {mi.get('onnxruntime')}",
    }
    out = common.save_json(report, a.out)
    print(json.dumps(report, indent=2))

    off = common.kit_file(common.BASELINE_METRICS, a.kit, required=False)
    if off.exists():
        print(f"\nofficial {off.name} vs our baseline (ours / official):")
        o = json.loads(off.read_text(encoding="utf-8"))
        o = o.get("baseline", o)
        for k in KEYS:
            ov, mv = o.get(k), report["baseline"].get(k)
            ratio = f"{mv / ov:.3f}" if isinstance(ov, (int, float)) and ov and mv is not None else "-"
            print(f"  {k:22s} ours={mv}  official={ov}  ratio={ratio}")
    else:
        print(f"\n(no {off.name} in the kit yet: cannot cross-check the baseline)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
