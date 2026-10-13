"""Create val_split.json: a reproducible, class-stratified train/val split of torchvision EuroSAT.

    python sec2026/make_split.py            # writes starter_kit/val_split.json

The task sheet asks for a fixed split ("val_split.json") but gives no seed, ratio or indices; the
organisers told teams verbally to produce the JSON files themselves. This is therefore OUR split,
not an official one, and the file says so. Method: indices follow torchvision.datasets.EuroSAT order
(ImageFolder: classes and file names sorted); within each class the indices are shuffled with
numpy.random.default_rng(SEED) and the first round(VAL_FRACTION * n) go to val. Re-running gives a
byte-identical file; its SHA-256 is printed for the report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

SEED = 42
VAL_FRACTION = 0.20
EUROSAT_ZIP_MD5 = "c8fa014336c82ac7804f0398fcb19387"   # torchvision's pinned EuroSAT.zip


def make_split(targets, seed: int = SEED, val_fraction: float = VAL_FRACTION) -> dict:
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed)
    train, val = [], []
    for c in sorted(set(targets.tolist())):
        idx = np.flatnonzero(targets == c)
        rng.shuffle(idx)
        k = int(round(val_fraction * len(idx)))
        val += idx[:k].tolist()
        train += idx[k:].tolist()
    return {"train": sorted(train), "val": sorted(val)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    ap.add_argument("--out", help="default: <kit>/val_split.json")
    a = ap.parse_args(argv)

    from torchvision import datasets

    ds = datasets.EuroSAT(root=common.kit_dir(a.kit) / "data", download=False)
    split = make_split(ds.targets, a.seed, a.val_fraction)
    assert not set(split["train"]) & set(split["val"]) and len(split["train"]) + len(split["val"]) == len(ds)
    per = lambda ix: {ds.classes[c]: n for c, n in sorted(Counter(ds.targets[i] for i in ix).items())}
    rec = {
        "train": split["train"], "val": split["val"],
        "meta": {
            "status": "team-generated (organisers said verbally that teams produce the JSON files); "
                      "NOT an official split",
            "method": "class-stratified random split; per class, numpy.random.default_rng(seed) shuffles "
                      "that class's indices and the first round(val_fraction*n) go to val",
            "seed": a.seed, "val_fraction": a.val_fraction,
            "index_order": "torchvision.datasets.EuroSAT (ImageFolder over eurosat/2750, sorted)",
            "dataset": {"name": "EuroSAT RGB", "n_images": len(ds), "classes": ds.classes,
                        "source_zip_md5": EUROSAT_ZIP_MD5},
            "n_train": len(split["train"]), "n_val": len(split["val"]),
            "train_per_class": per(split["train"]), "val_per_class": per(split["val"]),
            "generator": "sec2026/make_split.py",
        },
    }
    out = Path(a.out) if a.out else common.kit_dir(a.kit) / common.SPLIT
    text = json.dumps(rec, indent=1)
    out.write_text(text, encoding="utf-8")
    print(json.dumps(rec["meta"], indent=2))
    print(f"sha256 {hashlib.sha256(text.encode('utf-8')).hexdigest()}\n-> {out}")


if __name__ == "__main__":
    main()
