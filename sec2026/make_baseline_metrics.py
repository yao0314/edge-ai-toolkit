"""Write baseline_metrics.json for OUR machine, from measure.py results of the official student.

    python sec2026/make_baseline_metrics.py \
        --pt-accuracy sec2026/results/baseline_pt_accuracy.json --pt-timing sec2026/results/baseline_pt_timing.json \
        --onnx-accuracy sec2026/results/baseline_onnx_accuracy.json --onnx-timing sec2026/results/baseline_onnx_timing.json

The organisers said verbally that each team produces this file on its own device; the figures in the
sheet's "Baseline figures" table are only an example of the fields. Accuracy and the static facts
(params, size, MACs) can be measured in one run and timing (latency, peak RAM, energy) in another,
on a quiet machine; this script merges them per model and writes:

  sec2026/results/baseline_pt.json, baseline_onnx.json   merged measure.py records (make_metrics.py input)
  <kit>/baseline_metrics.json                            the baseline block in the Section 8 schema

"baseline" is the official checkpoint exported to FP32 ONNX, because the submission is ONNX and the
sheet (5.4) asks for identical methods for baseline and final; the PyTorch checkpoint measured with
the sheet's Section 5.2/5.3 code is kept alongside as "baseline_pytorch_checkpoint".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from make_metrics import block  # noqa: E402

TIMING_KEYS = ["latency_ms", "peak_ram_mb", "cpu_ms_per_inference", "joules_per_inference",
               "codecarbon_mode", "codecarbon_cpu_mode", "sessions", "settings",
               "machine_cpu_load_pct_before"]


def merge(acc_path, timing_path) -> dict:
    acc = json.loads(Path(acc_path).read_text(encoding="utf-8"))
    tim = json.loads(Path(timing_path).read_text(encoding="utf-8"))
    if Path(acc["model"]).resolve() != Path(tim["model"]).resolve():
        raise SystemExit(f"different models: {acc['model']} vs {tim['model']}")
    for k in ("size_mb", "params", "macs"):
        if acc[k] != tim[k]:
            raise SystemExit(f"{k} differs between the two runs: {acc[k]} vs {tim[k]}")
    if acc["machine"].get("cpu") != tim["machine"].get("cpu"):
        raise SystemExit("accuracy and timing were measured on different machines")
    out = {**acc, **{k: tim[k] for k in TIMING_KEYS if k in tim}}
    out["timing_measured_at"] = tim["measured_at"]
    out["sources"] = {"accuracy": str(acc_path), "timing": str(timing_path)}
    return out


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(m: dict) -> dict:
    s = m["sessions"]
    return {"model_file": m["model"], "format": "ONNX (FP32, opset 13)" if m["model"].endswith(".onnx")
            else "PyTorch state_dict (torchvision mobilenet_v2, 10-class head)",
            "params_method": m["params_method"], "macs_method": m["macs_method"],
            "accuracy_n_images": m["accuracy"]["n"],
            "worst_class": m["accuracy"]["worst_class"],
            "per_class_recall": m["accuracy"]["per_class_recall"],
            "sessions": [{k: x.get(k) for k in ("latency_ms", "peak_ram_mb", "joules_per_inference",
                                                 "machine_cpu_load_pct_before", "codecarbon_cpu_mode",
                                                 "energy_runs", "energy_seconds")} for x in s],
            "statistic": f"median of {len(s)} sessions", "settings": m["settings"],
            "codecarbon_cpu_mode": m.get("codecarbon_cpu_mode")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt-accuracy", required=True)
    ap.add_argument("--pt-timing", required=True)
    ap.add_argument("--onnx-accuracy", required=True)
    ap.add_argument("--onnx-timing", required=True)
    ap.add_argument("--kit")
    ap.add_argument("--out", help="default: <kit>/baseline_metrics.json")
    a = ap.parse_args(argv)

    pt, ox = merge(a.pt_accuracy, a.pt_timing), merge(a.onnx_accuracy, a.onnx_timing)
    res = HERE / "results"
    common.save_json(pt, res / "baseline_pt.json")
    common.save_json(ox, res / "baseline_onnx.json")
    split = common.kit_file(common.SPLIT, a.kit)
    split_meta = json.loads(split.read_text(encoding="utf-8")).get("meta", {})
    student = common.kit_file(common.STUDENT, a.kit)
    mi = ox["machine"]
    rec = {
        "baseline": block(ox),
        "worst_class_accuracy": round(float(ox["accuracy"]["worst_class_accuracy"]), 6),
        "codecarbon_mode": ox["codecarbon_mode"],
        "measurement_machine": f"{mi.get('cpu')}, {mi.get('physical_cores')} cores / {mi.get('logical_cpus')} "
                               f"threads, {mi.get('ram_gb')} GB RAM, CPU only, {mi.get('platform')}",
        "baseline_pytorch_checkpoint": {**block(pt),
                                        "worst_class_accuracy": round(float(pt["accuracy"]["worst_class_accuracy"]), 6),
                                        "codecarbon_mode": pt["codecarbon_mode"]},
        "provenance": {
            "status": "team-generated on our own device, per the organisers' verbal instruction; the "
                      "task sheet's 'Baseline figures' table was an example of the fields, not values to copy",
            "student_checkpoint": {"file": student.name, "sha256": sha256(student)},
            "validation_split": {"file": split.name, "sha256": sha256(split), **{k: split_meta.get(k) for k in
                                 ("status", "method", "seed", "val_fraction", "n_train", "n_val")}},
            "software": {k: mi.get(k) for k in ("python", "torch", "onnxruntime")},
            "avx_vnni": mi.get("avx_vnni"),
            "method": {
                "accuracy": "top-1 and per-class recall on the full val split (Section 5.1 arithmetic)",
                "params_size_macs": "ONNX: weights+biases of Conv/Gemm and MACs from the graph; PyTorch: thop "
                                    "(Section 5.2); size = file bytes / 1024^2",
                "latency": "mean of 50 runs of a (1,3,224,224) input after warm-up (ONNX 10 runs, PyTorch 0 "
                           "as in Section 5.3), default runtime threads, fresh subprocess per session",
                "peak_ram": "process RSS sampled every 10 ms during the 50 timed runs (Section 5.3)",
                "energy": "CodeCarbon EmissionsTracker(measure_power_secs=1) around >= 30 s and >= 100 "
                          "inferences; joules = energy_consumed kWh * 3.6e6 / inferences (Section 5.5)",
                "statistic": "median of 3 sessions; whole-machine CPU load recorded before each session",
            },
            "baseline_onnx": describe(ox),
            "baseline_pytorch": describe(pt),
        },
    }
    out = Path(a.out) if a.out else common.kit_dir(a.kit) / common.BASELINE_METRICS
    common.save_json(rec, out)
    print(json.dumps({k: rec[k] for k in ("baseline", "worst_class_accuracy", "codecarbon_mode",
                                          "measurement_machine", "baseline_pytorch_checkpoint")}, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
