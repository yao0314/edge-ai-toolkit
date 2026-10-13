"""Energy (Joules / inference) hook.

We never *estimate* energy from a formula: without a real measurement source
energy_j stays empty (NaN). Providers:

  none       -> NaN (default)
  codecarbon -> run the model under CodeCarbon's tracker and divide the measured energy by the
                number of inferences. Named by the rubric (Software Track packet section 3.4/10.4),
                which scores Joules per inference locally and cross-checks against the AI Hub
                device profile. CodeCarbon reads RAPL on Intel Linux and Windows, and falls back
                to a TDP-fraction MODEL elsewhere - `energy_source` records which, because a
                modelled number is not a measurement and must not be reported as one.
  csv      -> look up a measured value: energy.csv has columns `variant,energy_j`
              (e.g. filled from a USB power meter or the organisers' tool)
  command  -> run an external measuring command; it must print the Joules/inference
              as the last number on stdout. Template fields: {onnx} {variant} {input}
              e.g.  "python tools/power_meter.py --model {onnx}"
  aihub    -> reserved for Qualcomm AI Hub profiling; AI Hub profiles report
              latency/memory, so this returns NaN unless an energy field is present.
"""
from __future__ import annotations

import logging
import math
import re
import shlex
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)
NAN = float("nan")


def _codecarbon(onnx_path, input_npy, cfg: dict) -> tuple[float, str]:
    """Joules per inference, measured over a fixed number of ONNX Runtime inferences.

    CodeCarbon reports kWh for the tracked block. Two things decide whether that number means
    anything:

    * Duration. The tracker samples power on a timer (`measure_power_secs`), so a block shorter
      than a few sample periods is noise, not a reading. We therefore run for `min_seconds` of
      wall clock and divide by the inferences actually completed, rather than for a fixed count:
      at 300 runs of a 5 ms model the whole block was 1.5 s, one or two samples, and INT8 came
      out *above* FP32.
    * Source. On Intel Linux (and Windows with a supported power library) CodeCarbon reads RAPL
      hardware counters; otherwise it falls back to a fraction-of-TDP MODEL, which is really just
      wall-clock time in disguise. The returned source string records which, so a modelled number
      is never reported as a measured one.

    It subtracts no idle baseline, so the figure includes whatever else the machine is doing: the
    same quiet-machine rule as every other measurement applies.
    """
    import time

    import numpy as np
    import onnxruntime as ort
    from codecarbon import EmissionsTracker

    min_seconds = float(cfg.get("min_seconds", 30))
    sample_secs = float(cfg.get("measure_power_secs", 1))
    runs = int(cfg.get("runs", 0))            # 0 = run by the clock (the default and the safe one)
    warmup = int(cfg.get("warmup", 20))
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(cfg.get("threads", 0)) or 0
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    if input_npy and Path(input_npy).exists():
        x = np.load(str(input_npy))
    else:
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
        x = np.zeros(shape, dtype=np.float32)
    feed = {inp.name: x}
    for _ in range(warmup):
        sess.run(None, feed)

    tracker = EmissionsTracker(measure_power_secs=sample_secs, save_to_file=False,
                               log_level="error", tracking_mode="process")
    n = 0
    tracker.start()
    try:
        if runs:
            for _ in range(runs):
                sess.run(None, feed)
            n = runs
        else:
            end = time.perf_counter() + min_seconds
            while time.perf_counter() < end:
                for _ in range(32):          # check the clock every 32, not every inference
                    sess.run(None, feed)
                n += 32
    finally:
        tracker.stop()
    data = tracker.final_emissions_data
    joules = float(data.energy_consumed) * 3.6e6 / max(n, 1)      # kWh -> J, per inference
    hw = "model"
    try:                                     # the private attribute name moves between versions
        if "rapl" in str(getattr(tracker, "_hardware", "")).lower():
            hw = "rapl"
    except Exception:
        pass
    if hw != "rapl" and float(getattr(data, "duration", 0)) < 3 * sample_secs:
        log.warning("codecarbon block was %.1fs at a %.1fs sample period: too short to mean anything",
                    getattr(data, "duration", 0), sample_secs)
    return joules, f"codecarbon:{hw}"


def measure_energy(onnx_path: str | Path, variant: str, input_npy: str | Path | None, cfg: dict | None,
                   extra: dict | None = None) -> tuple[float, str]:
    """Return (energy_j, source). energy_j is NaN when no real source is configured."""
    cfg = cfg or {}
    provider = (cfg.get("provider") or "none").lower()
    if provider == "none":
        return NAN, ""
    if provider == "codecarbon":
        return _codecarbon(onnx_path, input_npy, cfg)
    if provider == "csv":
        import pandas as pd

        path = Path(cfg["path"])
        if not path.exists():
            log.warning("energy csv %s not found", path)
            return NAN, ""
        df = pd.read_csv(path)
        hit = df[df["variant"] == variant]
        return (float(hit["energy_j"].iloc[0]), f"csv:{path.name}") if len(hit) else (NAN, "")
    if provider == "command":
        cmd = cfg["command"].format(onnx=onnx_path, variant=variant, input=input_npy or "")
        try:
            out = subprocess.run(shlex.split(cmd, posix=False), capture_output=True, text=True,
                                 timeout=float(cfg.get("timeout", 600)), check=True).stdout
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", out)
            return (float(nums[-1]), "command") if nums else (NAN, "")
        except Exception as e:  # never crash a sweep because of the meter
            log.warning("energy command failed: %s", e)
            return NAN, ""
    if provider == "aihub":
        val = (extra or {}).get("aihub_energy_j", NAN)
        return (val, "aihub") if not math.isnan(val) else (NAN, "")
    raise ValueError(f"unknown energy provider {provider!r}")
