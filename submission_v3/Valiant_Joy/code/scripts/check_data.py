"""Verify a downloaded YOLO-format dataset before a long training run.

  python scripts/check_data.py data/coco_train5000 data/coco_val500 data/coco_val2000

Checks every image actually decodes, every image has a label, every label parses as
"cls cx cy w h" with normalised coordinates, and reports leftovers from interrupted downloads.
One truncated JPEG is enough to kill a multi-hour run partway through, and a download that was
killed (rather than raising) can leave an image whose label was never written.
"""
import argparse
import sys
from pathlib import Path

from _common import ROOT  # noqa: F401  (sets sys.path)

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def check(root: Path, deep: bool) -> int:
    from PIL import Image

    problems = []
    for split_dir in sorted((root / "images").iterdir()) if (root / "images").is_dir() else []:
        if not split_dir.is_dir():
            continue
        lab_dir = root / "labels" / split_dir.name
        imgs = sorted(f for f in split_dir.rglob("*") if f.suffix.lower() in EXTS)
        other = [f for f in split_dir.rglob("*") if f.is_file() and f.suffix.lower() not in EXTS]
        labs = {f.stem for f in lab_dir.glob("*.txt")} if lab_dir.is_dir() else set()

        n_box = 0
        for f in imgs:
            if f.stem not in labs:
                problems.append(f"{f.name}: no label file")
                continue
            try:
                rows = (lab_dir / f"{f.stem}.txt").read_text(encoding="utf-8").split("\n")
            except Exception as e:
                problems.append(f"{f.stem}.txt: unreadable ({e})")
                continue
            for ln, row in enumerate(rows, 1):
                if not row.strip():
                    continue
                parts = row.split()
                if len(parts) != 5:
                    problems.append(f"{f.stem}.txt:{ln}: {len(parts)} fields, expected 5")
                    continue
                try:
                    cls, *box = int(parts[0]), *(float(v) for v in parts[1:])
                except ValueError:
                    problems.append(f"{f.stem}.txt:{ln}: not numeric: {row!r}")
                    continue
                if cls < 0:
                    problems.append(f"{f.stem}.txt:{ln}: negative class {cls}")
                if not all(-0.01 <= v <= 1.01 for v in box):
                    problems.append(f"{f.stem}.txt:{ln}: box not normalised: {box}")
                n_box += 1
            if deep:
                try:
                    with Image.open(f) as im:
                        im.load()                     # load(), not verify(): actually decodes the pixels
                except Exception as e:
                    problems.append(f"{f.name}: does not decode ({type(e).__name__}: {e})")

        orphan_labels = labs - {f.stem for f in imgs}
        for stem in sorted(orphan_labels):
            problems.append(f"{stem}.txt: label with no image")
        print(f"  {split_dir.name}: {len(imgs)} images, {len(labs)} labels, {n_box} boxes"
              f"{' (pixels decoded)' if deep else ''}")
        if other:
            print(f"    {len(other)} non-image file(s), ignored by the adapter: "
                  f"{[f.name for f in other[:3]]}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+", help="dataset roots (each with images/<split> + labels/<split>)")
    ap.add_argument("--quick", action="store_true", help="skip decoding pixels (structure only)")
    args = ap.parse_args()

    bad = 0
    for r in args.roots:
        root = Path(r)
        root = root if root.is_absolute() else ROOT / root
        print(f"{root}:")
        if not root.is_dir():
            print("  MISSING")
            bad += 1
            continue
        problems = check(root, deep=not args.quick)
        for p in problems[:20]:
            print(f"    PROBLEM {p}")
        if len(problems) > 20:
            print(f"    ... and {len(problems) - 20} more")
        bad += len(problems)
    print("\nOK, no problems found" if not bad else f"\n{bad} problem(s) found")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
