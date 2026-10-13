"""Download a COCO subset (val2017 or train2017) in YOLO format.

  python scripts/fetch_coco_val.py --n 500                          # unseen validation images
  python scripts/fetch_coco_val.py --split train2017 --n 5000       # real training data

Why: coco128 is a slice of COCO *train2017*, so a COCO-pretrained detector has already seen it, and its
64 images give ~±0.01 mAP - larger than the INT8 effect. And with only 64 training images, channel pruning
cannot be recovered by fine-tuning, which leaves the question "does pruning work with real data?" open.

Creates data/<out>/{images,labels}/<split-dir>/... . Downloads once and caches:
annotations_trainval2017.zip (241 MB, holds both instances_*.json) plus N images (~160 KB each).
"""
import argparse
import json
import random
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

from _common import ROOT  # noqa: F401  (sets sys.path)

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/{split}/{name}"


def fetch(url: str, dest: Path, desc: str = "", tries: int = 3) -> Path | None:
    """Download unless already present. Retries with backoff and returns None on give-up, so one reset
    connection cannot kill a 5000-file download (it happened at image 616)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if desc:
        print(f"downloading {desc} -> {dest}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=180) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            tmp.replace(dest)
            return dest
        except Exception as e:                      # connection reset, timeout, 5xx ...
            tmp.unlink(missing_ok=True)
            if attempt == tries:
                print(f"  giving up on {dest.name}: {type(e).__name__}: {e}", flush=True)
                return None
            time.sleep(2 ** attempt)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val2017", choices=["val2017", "train2017"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default data/coco_<val|train><N>")
    args = ap.parse_args()

    data = ROOT / "data"
    sub = "train" if args.split == "train2017" else "val"
    out = Path(args.out) if args.out else data / f"coco_{sub}{args.n}"
    if not out.is_absolute():
        out = ROOT / out
    img_dir, lab_dir = out / "images" / sub, out / "labels" / sub
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    ann_json = data / "coco_annotations" / f"instances_{args.split}.json"
    if not ann_json.exists():
        zip_path = fetch(ANN_URL, data / "annotations_trainval2017.zip", "COCO annotations (241 MB)")
        ann_json.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            with z.open(f"annotations/instances_{args.split}.json") as src, open(ann_json, "wb") as dst:
                shutil.copyfileobj(src, dst)
        print(f"extracted {ann_json.name} ({ann_json.stat().st_size / 2**20:.0f} MB)", flush=True)

    coco = json.loads(ann_json.read_text(encoding="utf-8"))
    # COCO category ids are 1..90 with gaps; detectors trained on COCO use the contiguous 0..79 order
    cat80 = {c["id"]: i for i, c in enumerate(sorted(coco["categories"], key=lambda c: c["id"]))}
    by_img = {}
    for a in coco["annotations"]:
        if not a.get("iscrowd"):
            by_img.setdefault(a["image_id"], []).append(a)

    images = [im for im in coco["images"] if by_img.get(im["id"])]      # keep images with objects
    random.Random(args.seed).shuffle(images)
    picked = images[: args.n]
    print(f"{len(coco['images'])} {args.split} images, {len(picked)} picked (seed {args.seed})", flush=True)

    # A killed process (not an exception) leaves its .part behind: the atomic tmp.replace() means a
    # partial file is never mistaken for a real image, but the leftovers pile up and confuse a
    # who-downloaded-what count, so sweep them before starting.
    stale = list(img_dir.rglob("*.part"))
    for f in stale:
        f.unlink(missing_ok=True)
    if stale:
        print(f"removed {len(stale)} stale .part file(s) from an interrupted run", flush=True)

    missing = 0
    for k, im in enumerate(picked, 1):
        dst = img_dir / im["file_name"]
        if not dst.exists() and fetch(IMG_URL.format(split=args.split, name=im["file_name"]), dst) is None:
            missing += 1
            continue                                # no image -> no label, so the pair stays consistent
        w, h = im["width"], im["height"]
        rows = []
        for a in by_img[im["id"]]:
            x, y, bw, bh = a["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            rows.append(f"{cat80[a['category_id']]} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} "
                        f"{bw / w:.6f} {bh / h:.6f}")
        (lab_dir / Path(im["file_name"]).with_suffix(".txt").name).write_text("\n".join(rows), encoding="utf-8")
        if k % 250 == 0:
            print(f"  {k}/{len(picked)} images", flush=True)

    have = {p.name for p in img_dir.iterdir()}
    picked = [im for im in picked if im["file_name"] in have]
    n_box = sum(len(by_img[im["id"]]) for im in picked)
    if missing:
        print(f"WARNING: {missing} image(s) could not be downloaded and were skipped", flush=True)
    print(f"\ndone: {out}\n  {len(picked)} images, {n_box} objects ({n_box / max(len(picked), 1):.1f} per image)")
    print("\nPoint the config at it:")
    print(f"  {'images' if sub == 'train' else 'val_images'}: {out.name}/images/{sub}")
    print(f"  {'labels' if sub == 'train' else 'val_labels'}: {out.name}/labels/{sub}")


if __name__ == "__main__":
    main()
