"""The honest variant counts, de-duplicated by file content.

Three different numbers have been quoted for "how many quantized variants" during this work - 87,
37 and 16 - and all three were arithmetically correct answers to different questions. The paper can
only use one, and it has to be defined before a reviewer asks.

Worse, some variants are BYTE-IDENTICAL to each other. detection sets `calibrate_method: auto`,
auto selected `percentile 99.99`, and the config also declares an explicit `int8_pct` variant with
the same setting - so `320_p00_int8.onnx` and `320_p00_int8_pct.onnx` are the same file. That is
not a bug, but counting both inflates every prevalence figure computed over variants.

    python scripts/paper/variant_inventory.py

Writes results/paper/variant_inventory.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

HEADLINE = ["classification", "classification_deit", "classification_mobilevit", "anomaly",
            "detection", "detection_640", "stress_imagenette", "stress_resnet50"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/paper/variant_inventory.csv")
    a = ap.parse_args()

    rows = []
    for f in sorted((ROOT / "results").glob("*/models/*.onnx")):
        run = f.parent.parent.name
        rows.append({"run": run, "variant": f.stem, "headline_run": run in HEADLINE,
                     "is_quick": run.endswith("_quick"), "is_int8": "int8" in f.stem,
                     "size_bytes": f.stat().st_size,
                     "md5": hashlib.md5(f.read_bytes()).hexdigest()})
    df = pd.DataFrame(rows)
    df["dup_group_size"] = df.groupby("md5")["md5"].transform("size")
    df["is_dup_of_earlier"] = df.duplicated("md5", keep="first")
    df.to_csv(ROOT / a.out, index=False)

    def count(d, label):
        n_files = len(d)
        n_uniq = d["md5"].nunique()
        i8 = d[d["is_int8"]]
        print(f"  {label:44s} files {n_files:4d}   distinct {n_uniq:4d}   "
              f"INT8 files {len(i8):4d}   INT8 distinct {i8['md5'].nunique():4d}")

    print("Variant counts under each plausible definition\n")
    count(df, "everything under results/*/models/")
    count(df[~df["is_quick"]], "excluding *_quick smoke runs")
    count(df[df["headline_run"]], "the 8 headline (model x dataset) settings")
    count(df[df["headline_run"] & ~df["is_quick"]], "headline settings, no quick runs")

    print("\nByte-identical groups (counting both inflates every prevalence figure):")
    dups = df[df["dup_group_size"] > 1].sort_values(["md5", "run", "variant"])
    for md5, g in dups.groupby("md5"):
        names = ", ".join(f"{r.run}/{r.variant}" for r in g.itertuples())
        print(f"  {md5[:10]}  {names}")
    if dups.empty:
        print("  none")

    # Does the size each sweep.csv REPORTS still match the file on disk? Added after an audit found
    # 32 of 162 models disagreeing by a constant 242-247 bytes: the CSV records the size before
    # stamp_onnx rewrites the file with authorship metadata, so a model re-stamped after its sweep
    # drifts. The effect on compression ratios is tiny (anomaly 3.103x -> 3.095x) but it is exactly
    # the class of quiet accounting error this project has been bitten by, so it is now checked on
    # every inventory run rather than rediscovered.
    drift = []
    for f in sorted((ROOT / "results").glob("*/sweep.csv")):
        run = f.parent.name
        if run.startswith("_"):
            continue
        try:
            s = pd.read_csv(f)
        except Exception:
            continue
        if "size_bytes" not in s or "variant" not in s:
            continue
        for _, r in s.iterrows():
            m = f.parent / "models" / f"{r['variant']}.onnx"
            if not m.exists() or pd.isna(r["size_bytes"]):
                continue
            d = m.stat().st_size - int(r["size_bytes"])
            if d:
                drift.append({"run": run, "variant": r["variant"],
                              "csv_bytes": int(r["size_bytes"]), "disk_bytes": m.stat().st_size,
                              "delta_bytes": d, "pct": 100 * d / int(r["size_bytes"])})
    print("")
    if drift:
        dd = pd.DataFrame(drift)
        dd.to_csv(ROOT / "results" / "paper" / "size_drift.csv", index=False)
        print(f"SIZE DRIFT: {len(dd)} model(s) whose sweep.csv size differs from the file on disk")
        print(f"  offsets {sorted(dd['delta_bytes'].unique())[:6]}   worst {dd['pct'].abs().max():.2f}%")
        print(f"  runs: {', '.join(sorted(dd['run'].unique()))}")
        print("  Quote disk sizes, not CSV sizes, for anything in that list. Detail in "
              "results/paper/size_drift.csv")
    else:
        print("size drift: none - every sweep.csv size matches the file on disk")

    head = df[df["headline_run"]]
    print(f"\nRECOMMENDED DENOMINATOR for the paper: distinct INT8 models in the 8 headline "
          f"settings = {head[head['is_int8']]['md5'].nunique()}")
    print("State it that way. 'Variants' is ambiguous; 'distinct quantized models' is not, and it")
    print("is the number a reviewer can reproduce from the artifact.")
    print(f"\nwrote {ROOT / a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
