"""Download Imagenette (10 ImageNet classes, real photos) as an ImageFolder dataset.

  python scripts/fetch_imagenette.py            # imagenette2-320, ~325 MB

CIFAR-10 is 32x32; this exercises the adapter's `dataset: imagefolder` path with real-resolution images,
which is what an on-site dataset will look like. Creates data/imagenette/{train,val}/<class>/*.JPEG.
"""
import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

from _common import ROOT  # noqa: F401  (sets sys.path)

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--out", default=None, help="default data/imagenette")
    args = ap.parse_args()
    out = Path(args.out) if args.out else ROOT / "data" / "imagenette"
    if (out / "train").exists() and (out / "val").exists():
        print(f"already present: {out}")
        return

    tgz = ROOT / "data" / Path(args.url).name
    if not tgz.exists():
        print(f"downloading {args.url} (~325 MB)", flush=True)
        tmp = tgz.with_suffix(".part")
        with urllib.request.urlopen(args.url, timeout=300) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        tmp.replace(tgz)
    print("extracting ...", flush=True)
    with tarfile.open(tgz) as t:
        t.extractall(ROOT / "data", filter="data")
    src = ROOT / "data" / tgz.name.replace(".tgz", "")
    if src != out:
        if out.exists():
            shutil.rmtree(out)
        src.rename(out)
    n_train = sum(1 for _ in (out / "train").rglob("*.JPEG"))
    n_val = sum(1 for _ in (out / "val").rglob("*.JPEG"))
    classes = sorted(p.name for p in (out / "train").iterdir() if p.is_dir())
    print(f"done: {out}\n  {n_train} train / {n_val} val images, {len(classes)} classes")
    print("\nUse it with:  adapter_args: {dataset: imagefolder, data_root: data/imagenette, img_size: 224}")


if __name__ == "__main__":
    main()
