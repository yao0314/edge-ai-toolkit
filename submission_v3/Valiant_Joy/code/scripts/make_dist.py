"""Package the toolkit for someone else to use, without the 13 GB of local working data.

The repo on disk is ~13 GB, of which the actual design is ~2.5 MB: .venv (6.5 G), data (3.0 G),
results (1.8 G, almost all .onnx/.pt), vendor (933 M of wheel bundles) and weights (749 M) are all
either rebuildable or downloadable. Shipping them would make the archive unusable to send and would
leak nothing of value.

  python scripts/make_dist.py                  # source + small result summaries + deliverables
  python scripts/make_dist.py --lean           # source only, no results, no models
  python scripts/make_dist.py --with-vendor linux-x86_64-py311   # add an offline wheel bundle

The archive always contains everything needed to RUN the toolkit: core/, adapters/, configs/,
scripts/, tests/, requirements, the README, CLAUDE.md and docs/ (the handover).
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIRS = ["core", "adapters", "configs", "scripts", "tests", "docs"]
ROOT_FILES = ["CLAUDE.md", "requirements.txt", "requirements-optional.txt", "README.md", "AUTHORS.md",
              ".gitignore", ".gitattributes"]
# results/ holds 1.8 GB of model files; these extensions are the summaries worth carrying (~4 MB).
RESULT_KEEP = {".csv", ".json", ".xlsx", ".md", ".png"}
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".venv", ".git", "node_modules"}
SKIP_SUFFIX = {".pyc", ".pyo"}


def _keep(p: Path) -> bool:
    return (not any(part in SKIP_DIRS for part in p.parts)) and p.suffix.lower() not in SKIP_SUFFIX


def add(z: zipfile.ZipFile, src: Path, arc: str, seen: list) -> None:
    z.write(src, arc)
    seen.append((arc, src.stat().st_size))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lean", action="store_true", help="source only: no results, no deliverables")
    ap.add_argument("--with-vendor", default=None, metavar="NAME",
                    help="also embed vendor/<NAME> (an offline wheel bundle; adds ~470 MB)")
    ap.add_argument("--no-checkpoints", action="store_true",
                    help="leave out weights/torch/hub/checkpoints (only safe with internet on site)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    name = "edge-ai-toolkit" + ("-lean" if a.lean else "")
    out = Path(a.out or ROOT.parent / f"{name}.zip")
    seen: list = []

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for d in SOURCE_DIRS:
            for f in sorted((ROOT / d).rglob("*")):
                if f.is_file() and _keep(f):
                    add(z, f, f"{name}/{f.relative_to(ROOT).as_posix()}", seen)
        # The ImageNet checkpoints torchvision downloads on demand. weights/ is excluded wholesale
        # as rebuildable, but these are only rebuildable WITH INTERNET, and the whole compression
        # strategy is a student started from them: without these files on an offline machine the
        # students train from scratch and land ~20 points below the accuracy bar. ~50 MB.
        if not a.no_checkpoints:
            ck = ROOT / "weights" / "torch" / "hub" / "checkpoints"
            for f in sorted(ck.glob("*.pth")) if ck.exists() else []:
                add(z, f, f"{name}/{f.relative_to(ROOT).as_posix()}", seen)

        for fn in ROOT_FILES:
            f = ROOT / fn
            if f.exists():
                add(z, f, f"{name}/{fn}", seen)
        if not a.lean:
            for f in sorted((ROOT / "results").rglob("*")):
                if f.is_file() and _keep(f) and f.suffix.lower() in RESULT_KEEP:
                    add(z, f, f"{name}/{f.relative_to(ROOT).as_posix()}", seen)
            d = ROOT / "deliverables"
            if d.exists():
                for f in sorted(d.rglob("*")):
                    if f.is_file() and _keep(f):
                        add(z, f, f"{name}/{f.relative_to(ROOT).as_posix()}", seen)
        if a.with_vendor:
            v = ROOT / "vendor" / a.with_vendor
            if not v.is_dir():
                print(f"no such bundle: {v}")
                return 2
            for f in sorted(v.rglob("*")):
                if f.is_file():
                    add(z, f, f"{name}/vendor/{a.with_vendor}/{f.name}", seen)

    raw = sum(s for _, s in seen)
    print(f"{out}")
    print(f"  {len(seen)} files, {raw / 2**20:.1f} MiB uncompressed -> {out.stat().st_size / 2**20:.1f} MiB zipped")
    top: dict = {}
    for arc, s in seen:
        top[arc.split("/")[1] if "/" in arc else "(root)"] = top.get(arc.split("/")[1] if "/" in arc else "(root)", 0) + s
    for k, s in sorted(top.items(), key=lambda kv: -kv[1]):
        print(f"    {k:18s} {s / 2**20:8.2f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
