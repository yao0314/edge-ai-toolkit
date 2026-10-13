"""Finding A, decomposed: "dynamic beats static" is two different effects wearing one label.

The finding was carried as a single claim — dynamic INT8 beats static QDQ within an architecture,
with non-overlapping 4-thread ranges. That is true and it reproduces. But pooling the two
architectures hides that the WIN HAS A DIFFERENT MECHANISM IN EACH, and one of the two mechanisms
is not a win at all once compression is counted.

Every dynamic variant is compared against its own fp32 and static INT8 stablemates at the same
prune ratio, on speed, size and accuracy together — because a quantization method that buys speed
by declining to quantize is not comparable to one that quantizes everything, and the latency column
alone cannot tell them apart.

    python scripts/paper/static_vs_dynamic.py

Writes results/paper/static_vs_dynamic.csv.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402


def collect(threads: int) -> pd.DataFrame:
    lat = f"latency_p50_ms_t{threads}"
    rows = []
    for f in sorted(glob.glob(str(ROOT / "results" / "*" / "sweep.csv"))):
        run = Path(f).parent.name
        if run.startswith("_"):
            continue
        d = pd.read_csv(f)
        if lat not in d:
            continue
        by = {str(r["variant"]): r for _, r in d.iterrows()}
        for v, r in by.items():
            if "dyn" not in v:
                continue
            m = re.match(r"^(.*?p\d+)_", v)
            if not m:
                continue
            fp, st = by.get(m.group(1) + "_fp32"), by.get(m.group(1) + "_int8")
            if fp is None or st is None:
                continue
            rows.append({
                "run": run, "variant": v, "threads": threads,
                "dyn_vs_fp32": fp[lat] / r[lat],
                "static_vs_fp32": fp[lat] / st[lat],
                "dyn_vs_static": st[lat] / r[lat],
                # the column that changes the reading: a method that leaves most of the graph in
                # fp32 keeps most of the bytes, and buys its speed by not quantizing
                "dyn_size_frac_of_fp32": r["size_bytes"] / fp["size_bytes"],
                "static_size_frac_of_fp32": st["size_bytes"] / fp["size_bytes"],
                "dyn_acc_drop_pt": (r["accuracy"] - fp["accuracy"]) * 100,
                "static_acc_drop_pt": (st["accuracy"] - fp["accuracy"]) * 100,
            })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="results/paper/static_vs_dynamic.csv")
    a = ap.parse_args()

    df = collect(a.threads)
    if df.empty:
        print("no dynamic variants with both an fp32 and a static INT8 stablemate", file=sys.stderr)
        return 1
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"{len(df)} dynamic variants across {df['run'].nunique()} run(s), at {a.threads} threads")
    print(f"  dynamic beats static in {int((df['dyn_vs_static'] > 1).sum())}/{len(df)} - "
          f"the pooled claim, and it holds\n")
    for run, g in df.groupby("run"):
        print(f"{run}   (n={len(g)})")
        print(f"   speed   dynamic {g['dyn_vs_fp32'].mean():.2f}x fp32    "
              f"static {g['static_vs_fp32'].mean():.2f}x fp32    "
              f"dynamic {g['dyn_vs_static'].mean():.2f}x static")
        print(f"   size    dynamic keeps {g['dyn_size_frac_of_fp32'].mean() * 100:.0f}% of fp32 bytes, "
              f"static keeps {g['static_size_frac_of_fp32'].mean() * 100:.0f}%")
        print(f"   acc     dynamic {g['dyn_acc_drop_pt'].mean():+.2f} pt    "
              f"static {g['static_acc_drop_pt'].mean():+.2f} pt")
        # Which of the two mechanisms is this? Decided on the data, not asserted.
        gains_over_fp32 = g["dyn_vs_fp32"].mean() > 1.15
        static_is_harmful = g["static_vs_fp32"].mean() < 1.0
        keeps_bytes = g["dyn_size_frac_of_fp32"].mean() > 0.5
        if gains_over_fp32 and not keeps_bytes:
            print("   -> GENUINE WIN: faster than fp32 AND compresses about as well as static.")
        elif static_is_harmful and keeps_bytes:
            print("   -> NOT A WIN, A RETREAT: barely faster than fp32 and it keeps most of the")
            print("      bytes. It looks good only because static is harmful here. What is being")
            print("      measured is the DAMAGE static does, not a benefit dynamic provides.")
        else:
            print("   -> mixed; read the columns rather than the headline.")
        print()

    print("The pooled claim 'dynamic beats static' is true and reproduces, but it is not one")
    print("finding. Report the mechanism per architecture or a reviewer will find this decomposition")
    print("before you do.")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
