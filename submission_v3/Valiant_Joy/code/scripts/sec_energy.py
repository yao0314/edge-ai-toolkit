"""Measure Joules/inference for every variant of a finished sweep and write it back.

remeasure.py re-does latency, CPU, RAM and energy together; this does energy alone, because
energy is the one column the sweep left empty and re-measuring the rest would disturb numbers
that are already in the CSV. Section 5.5 of the task sheet measures energy with CodeCarbon on
the ONNX file, so that is what this runs.

Thread count matters: baseline and final must be measured identically or the ratio in
metrics.json means nothing (sheet, section 8). It therefore uses ONE thread count for every
row, defaulting to the sweep's primary so energy and the reported latency describe the same
configuration.

  python scripts/sec_energy.py --csv results/sec2026_real/sweep.csv --threads 4 --seconds 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from core.energy import measure_energy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--models", help="directory of <variant>.onnx (default <csv dir>/models)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--repeats", type=int, default=3, help="sessions per variant; the median is kept")
    ap.add_argument("--max-watts", type=float, default=115.0,
                    help="implied power above this is physically impossible and the reading is dropped")
    ap.add_argument("--only", nargs="*", help="limit to these variants")
    a = ap.parse_args()

    csv = Path(a.csv)
    models = Path(a.models) if a.models else csv.parent / "models"
    df = pd.read_csv(csv)
    for col in ("energy_j", "energy_source"):
        if col not in df:
            df[col] = None
    df["energy_source"] = df["energy_source"].astype(object)

    todo = [(i, str(r["variant"])) for i, r in df.iterrows()
            if (not a.only or str(r["variant"]) in a.only) and (models / f'{r["variant"]}.onnx').exists()]
    readings = {v: [] for _, v in todo}
    source = {}

    # Rotate the order each round. Measured back to back, a machine that warms up or a background
    # task that starts midway would otherwise land entirely on whichever variants sit in that part
    # of the list - which is exactly how the first attempt produced six impossible rows in a block.
    for r in range(a.repeats):
        order = todo[r % len(todo):] + todo[:r % len(todo)]
        for i, v in order:
            lat_ms = df.loc[i, "latency_p50_ms"]
            j, src = measure_energy(models / f"{v}.onnx", v, None,
                                    {"provider": "codecarbon", "min_seconds": a.seconds,
                                     "measure_power_secs": 1, "threads": a.threads})
            watts = j / (float(lat_ms) / 1000.0) if lat_ms and lat_ms == lat_ms else float("nan")
            bad = watts == watts and watts > a.max_watts
            if not bad:
                readings[v].append(j)
            source[v] = src
            print(f"round {r+1} {v:10s} {j:.4f} J/inf  ({watts:6.1f} W implied)"
                  f"{'  DROPPED: impossible' if bad else ''}", flush=True)

    print()
    for i, v in todo:
        vals = sorted(readings[v])
        if not vals:
            print(f"{v:10s} NO VALID READING")
            continue
        med = vals[len(vals) // 2] if len(vals) % 2 else (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2
        df.loc[i, "energy_j"] = med
        df.loc[i, "energy_source"] = source[v]
        spread = (max(vals) - min(vals)) / med * 100 if med else 0
        print(f"{v:10s} median {med:.4f} J/inf over {len(vals)}/{a.repeats} valid, spread {spread:.0f}%")

    df.to_csv(csv, index=False)
    print(f"\nwrote {csv} ({a.threads} thread(s), {a.seconds:.0f}s per variant)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
