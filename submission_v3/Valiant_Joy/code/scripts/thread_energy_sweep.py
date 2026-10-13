"""Latency and energy against thread count, for one model file.

Two questions, one sweep:

1. Is a low `core_delivery` on a small model contention, or the model? A 3 ms inference may simply
   not have enough work to keep four threads busy, in which case the figure is a property of the
   model and the 0.95 rule of thumb — written for contention — does not apply. If core_delivery
   climbs as the thread count falls, the threads were idling, not stolen.
2. Where does energy per inference bottom out? Energy ~ power x latency, so more threads cut the
   time but raise the power, and the minimum need not be at the thread count that minimises
   latency. The rubric scores Joules per inference separately from latency, so the two can and
   should be chosen separately.

  python scripts/thread_energy_sweep.py --onnx deliverables/<run>/best.onnx
  python scripts/thread_energy_sweep.py --onnx a.onnx --threads 1,2,4,8 --energy-seconds 20

Measure on a quiet machine: the whole point is to tell idle threads from stolen ones.
"""
import argparse
import json
from pathlib import Path

from _common import ROOT  # noqa: F401  (sets sys.path)

import numpy as np
import onnxruntime as ort

from core.energy import measure_energy
from core.measure import DEFAULT_MEASURE_CFG, _run_probe


def example_npy(onnx_path: Path) -> tuple[str, list]:
    s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = s.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    path = ROOT / ".cache_thread_sweep_input.npy"
    np.save(path, np.zeros(shape, dtype=np.float32))
    return str(path), shape


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--threads", default="1,2,4,8")
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--energy-seconds", type=float, default=20)
    ap.add_argument("--no-energy", action="store_true")
    ap.add_argument("--no-spinning", action="store_true",
                    help="required for core_delivery to mean anything: with spin-wait on, idle "
                         "threads burn CPU and the ratio reads above 1.0")
    ap.add_argument("--out", help="write the table as JSON here")
    args = ap.parse_args()

    onnx = Path(args.onnx)
    npy, shape = example_npy(onnx)
    counts = [int(t) for t in args.threads.split(",")]
    print(f"{onnx.name}  input {shape}\n")

    rows = []
    for t in counts:
        cfg = {**DEFAULT_MEASURE_CFG, "warmup": 10, "runs": args.runs, "min_time_s": 2.0,
               "spinning": not args.no_spinning, "affinity": "auto"}
        r = _run_probe(str(onnx), npy, t, cfg)
        lat, cpu = r.get("latency_p50_ms"), r.get("cpu_ms_per_inf")
        # The probe returns the two ingredients, not the ratio: measure() derives it per thread
        # count. NOTE it is only interpretable with spinning OFF - idle threads spin-wait and that
        # CPU time lands in cpu_ms, which is how this reads above 1.0 at four threads.
        row = {"threads": t, "latency_p50_ms": lat, "peak_rss_mb": r.get("peak_rss_mb"),
               "cpu_ms_per_inf": cpu,
               "core_delivery": (cpu / (t * lat)) if (lat and cpu) else None,
               "spinning": cfg["spinning"]}
        if not args.no_energy:
            j, src = measure_energy(onnx, f"t{t}", npy,
                                    {"provider": "codecarbon", "min_seconds": args.energy_seconds,
                                     "threads": t})
            row["energy_j"] = j
            row["energy_source"] = src
        rows.append(row)
        print("threads %-2d  p50 %7.3f ms  cpu %7.3f ms  core_delivery %.3f  rss %5.1f MB%s"
              % (t, row["latency_p50_ms"] or 0, row["cpu_ms_per_inf"] or 0,
                 row["core_delivery"] or 0, row["peak_rss_mb"] or 0,
                 "" if args.no_energy else "  energy %.5f J" % row["energy_j"]))

    best_lat = min(rows, key=lambda r: r["latency_p50_ms"] or 1e9)
    print(f"\nlowest latency : {best_lat['threads']} threads ({best_lat['latency_p50_ms']:.3f} ms)")
    if not args.no_energy:
        best_e = min(rows, key=lambda r: r["energy_j"] if r["energy_j"] == r["energy_j"] else 1e9)
        print(f"lowest energy  : {best_e['threads']} threads ({best_e['energy_j']:.5f} J, "
              f"{best_e['energy_source']})")
        if best_e["threads"] != best_lat["threads"]:
            print("  -> the two minima differ: latency and energy are separate choices here.")
    cd = {r["threads"]: r["core_delivery"] for r in rows}
    print("\ncore_delivery by thread count: "
          + ", ".join(f"{t}:{cd[t]:.3f}" for t in counts if cd.get(t) is not None))
    if args.no_spinning:
        print("  rising as threads fall means the threads were idle, not stolen - a property of "
              "the model, not a contaminated measurement.")
    else:
        print("  NOT interpretable: spin-wait is on, so idle threads' CPU time is counted as work "
              "and the ratio can exceed 1.0. Re-run with --no-spinning to read it.")

    if args.out:
        Path(args.out).write_text(json.dumps({"model": str(onnx), "rows": rows}, indent=2),
                                  encoding="utf-8")
        print(f"\nwrote {args.out}")
    Path(npy).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
