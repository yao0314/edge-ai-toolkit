"""Section 5 of the task sheet ("Measure First"), for the baseline and every candidate alike.

    python sec2026/measure.py --model starter_kit/student_mobilenetv2_distilled.pt --name baseline_pt
    python sec2026/measure.py --model sec2026/out/p00/student_model.onnx --name baseline_onnx
    python sec2026/measure.py --model sec2026/out/p00/student_model_int8.onnx --name p00_int8

Measures, with the sheet's own methods:
  5.1 top-1 accuracy and per-class recall on the official val split (the worst class is scored),
  5.2 file size (MiB, as the sheet divides by 1024*1024), parameters and MACs (thop for PyTorch,
      graph count for ONNX),
  5.3 latency (mean of 50 runs; ONNX after 10 warm-up runs) and peak RSS sampled every 10 ms
      while those runs execute,
  5.5 energy per inference with CodeCarbon, and which mode CodeCarbon actually ran in.

Latency, RAM and energy run in a fresh subprocess per repeat that loads only the model, so the
dataset and the evaluation loop are not counted as the model's memory, and a repeat is not warmed
by the previous one. The median over --repeats is reported, with every repeat kept. Measure on a
quiet machine (the script records whole-machine CPU load before it starts) and use the same
settings for the baseline and the final model: the sheet scores efficiency and energy as ratios
against our own baseline on our own machine.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

MEASURED_MODES = {"intel_rapl", "intel_power_gadget", "windows_emi", "powermetrics", "amd_smi"}


# ----------------------------------------------------------------------------- isolated probe
def _load_runner(path: str, threads: int | None):
    """Returns (run_once, kind). Imports only what this model needs."""
    import numpy as np

    if path.endswith(".onnx"):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
        return (lambda: sess.run(None, {name: dummy})), "onnx"
    import torch

    if threads:
        torch.set_num_threads(threads)
    import common

    model = common.build_student(path)
    dummy = torch.randn(1, 3, 224, 224)

    def run():
        with torch.no_grad():
            model(dummy)
    return run, "torch"


def probe(path: str, threads: int | None, runs: int, warmup: int, energy_s: float) -> dict:
    """One measurement session. Latency/RSS exactly as Section 5.3/5.4; energy as Section 5.5."""
    import threading

    import psutil

    run, kind = _load_runner(path, threads)
    process = psutil.Process(os.getpid())
    rss_after_load = process.memory_info().rss / 2**20
    for _ in range(warmup):
        run()

    peak = {"mb": process.memory_info().rss / 2**20}
    sampling = {"on": True}

    def sample_rss(interval_s=0.01):
        while sampling["on"]:
            peak["mb"] = max(peak["mb"], process.memory_info().rss / 2**20)
            time.sleep(interval_s)

    sampler = threading.Thread(target=sample_rss, daemon=True)
    sampler.start()
    cpu0 = process.cpu_times()
    start = time.perf_counter()
    for _ in range(runs):
        run()
    elapsed = time.perf_counter() - start
    cpu1 = process.cpu_times()
    sampling["on"] = False
    sampler.join()
    out = {"kind": kind, "latency_ms": elapsed / runs * 1000, "peak_ram_mb": peak["mb"],
           "rss_after_load_mb": rss_after_load,
           "cpu_ms_per_inference": ((cpu1.user + cpu1.system) - (cpu0.user + cpu0.system)) / runs * 1000}
    if energy_s > 0:
        out.update(_energy(run, energy_s))
    return out


def _energy(run, min_seconds: float) -> dict:
    """CodeCarbon as in Section 5.5, but the loop runs for at least `min_seconds` (and at least the
    sheet's 100 inferences): with measure_power_secs=1 a 100-inference loop of a few-ms model ends
    before CodeCarbon takes a single power reading."""
    try:
        import truststore  # this laptop's antivirus re-signs TLS; CodeCarbon looks up its location

        truststore.inject_into_ssl()
    except Exception:
        pass
    import logging

    from codecarbon import EmissionsTracker

    logging.getLogger("codecarbon").setLevel(logging.ERROR)
    tracker = EmissionsTracker(measure_power_secs=1, log_level="error", save_to_file=False)
    tracker.start()
    n, t0 = 0, time.perf_counter()
    while n < 100 or time.perf_counter() - t0 < min_seconds:
        run()
        n += 1
    tracker.stop()
    data = tracker.final_emissions_data
    modes = sorted({str(getattr(h, "_mode", "")) for h in getattr(tracker, "_hardware", [])
                    if type(h).__name__ == "CPU"})
    measured = any(m in MEASURED_MODES for m in modes)
    return {"joules_per_inference": data.energy_consumed * 3.6e6 / n, "energy_runs": n,
            "energy_seconds": time.perf_counter() - t0,
            "codecarbon_cpu_mode": ",".join(modes) or "unknown",
            "codecarbon_mode": "measured" if measured else "tdp_estimate",
            "cpu_power_w": getattr(data, "cpu_power", None)}


# ----------------------------------------------------------------------------- machine load
def machine_cpu_load() -> float | None:
    """Whole-machine CPU % right now. On Windows the processor performance counter (_Total), not
    LoadPercentage, which is not a machine-load figure."""
    if os.name == "nt":
        cmd = ["powershell", "-NoProfile", "-Command",
               "$a=@(); 1..3 | % { $a += (Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor "
               "-Filter \"Name='_Total'\").PercentProcessorTime; Start-Sleep -Milliseconds 700 }; "
               "($a | Measure-Object -Average).Average"]
        try:
            return float(subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip())
        except Exception:
            return None
    import psutil

    return psutil.cpu_percent(interval=3)


# ----------------------------------------------------------------------------- static facts
def static_facts(path: Path) -> dict:
    size_mb = path.stat().st_size / (1024 * 1024)
    if path.suffix == ".onnx":
        import export_quant as eq

        return {"size_mb": size_mb, "params": eq.onnx_params(path), "macs": eq.onnx_macs(path),
                "params_method": "ONNX graph (Conv/Gemm weights+bias)", "macs_method": "ONNX graph"}
    import torch
    from thop import profile

    import common

    model = common.build_student(path)
    macs, params = profile(model, inputs=(torch.randn(*common.INPUT_SHAPE),), verbose=False)
    return {"size_mb": size_mb, "params": int(params), "macs": int(macs),
            "params_method": "thop", "macs_method": "thop"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="a .pt (state_dict or pruned module) or .onnx; default: the kit's student")
    ap.add_argument("--kit")
    ap.add_argument("--name", help="results/<name>.json (default: from the file name)")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
    ap.add_argument("--threads", type=int, default=None,
                    help="intra-op threads; default = runtime default, as in the sheet's code")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=None, help="default: 10 for ONNX, 0 for PyTorch (as the sheet)")
    ap.add_argument("--energy-seconds", type=float, default=30.0, help="0 skips energy")
    ap.add_argument("--val-limit", type=int, default=None, help="quick look only; never report it")
    ap.add_argument("--skip-accuracy", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--_probe", nargs=5, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)

    if a._probe:   # child process: print one session as JSON
        path, threads, runs, warmup, energy_s = a._probe
        print("PROBE " + json.dumps(probe(path, int(threads) or None, int(runs), int(warmup), float(energy_s))))
        return

    import common

    path = Path(a.model) if a.model else common.kit_file(common.STUDENT, a.kit)
    name = a.name or path.stem
    warmup = a.warmup if a.warmup is not None else (10 if path.suffix == ".onnx" else 0)
    rec = {"name": name, "model": str(path), "measured_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
           "machine": common.machine_info(), "settings": {"threads": a.threads, "runs": a.runs,
           "warmup": warmup, "repeats": a.repeats, "energy_seconds": a.energy_seconds}}
    rec.update(static_facts(path))

    if not a.skip_accuracy:
        _, val = common.subsets(a.kit, limit_val=a.val_limit)
        t0 = time.time()
        acc = common.evaluate(common.predictor(path, a.threads), common.loader(val, a.batch_size))
        acc["seconds"] = round(time.time() - t0, 1)
        acc["full_val_split"] = a.val_limit is None
        rec["accuracy"] = acc
        print(f"accuracy {acc['accuracy']:.4f} | worst class {acc['worst_class']} "
              f"{acc['worst_class_accuracy']:.4f} | n={acc['n']}")

    load = machine_cpu_load()
    rec["machine_cpu_load_pct_before"] = load
    if load is not None and load > 15:
        print(f"WARNING: whole-machine CPU load {load:.0f}% (> 15%): timing and energy are suspect")
    sessions = []
    env = {**os.environ, "SEC_KIT": str(common.kit_dir(a.kit))}
    for i in range(a.repeats):
        pre_load = load if i == 0 else machine_cpu_load()   # whole-machine load just before this session
        cmd = [sys.executable, str(Path(__file__).resolve()), "--_probe", str(path.resolve()),
               str(a.threads or 0), str(a.runs), str(warmup), str(a.energy_seconds)]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        line = next((l for l in r.stdout.splitlines() if l.startswith("PROBE ")), None)
        if r.returncode or not line:
            raise RuntimeError(f"probe failed (exit {r.returncode}):\n{r.stderr[-3000:]}")
        s = json.loads(line[6:])
        s["machine_cpu_load_pct_before"] = pre_load
        sessions.append(s)
        print(f"session {i + 1}/{a.repeats}: load {pre_load}% | {s['latency_ms']:.2f} ms, "
              f"peak RSS {s['peak_ram_mb']:.1f} MB"
              + (f", {s['joules_per_inference']:.4f} J/inf ({s['codecarbon_mode']})"
                 if "joules_per_inference" in s else ""))
    med = lambda k: (statistics.median(s[k] for s in sessions)
                     if sessions and all(k in s for s in sessions) else None)
    rec.update({"latency_ms": med("latency_ms"), "peak_ram_mb": med("peak_ram_mb"),
                "cpu_ms_per_inference": med("cpu_ms_per_inference"),
                "joules_per_inference": med("joules_per_inference"),
                "codecarbon_mode": (sessions[0] if sessions else {}).get("codecarbon_mode"),
                "codecarbon_cpu_mode": (sessions[0] if sessions else {}).get("codecarbon_cpu_mode"),
                "sessions": sessions})
    out = common.save_json(rec, Path(a.out_dir) / f"{name}.json")
    print(f"size {rec['size_mb']:.2f} MB | params {rec['params'] / 1e6:.4f} M | MACs {rec['macs'] / 1e9:.4f} G"
          + (f" | latency {rec['latency_ms']:.2f} ms | peak RAM {rec['peak_ram_mb']:.1f} MB"
             if rec["latency_ms"] is not None else " | timing not run (--repeats 0)")
          + (f" | {rec['joules_per_inference']:.4f} J/inf ({rec['codecarbon_mode']})"
             if rec["joules_per_inference"] is not None else ""))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
