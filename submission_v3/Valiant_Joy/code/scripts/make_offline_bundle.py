"""Build a vendored wheel bundle so the toolkit installs with NO network.

Why this exists: the competition sandbox is revealed on the day and may have no internet. A full
`pip install -r requirements.txt` needs ~700 MB even on the CPU index, and on Linux the DEFAULT
PyPI torch additionally drags in cuda-toolkit, nvidia-cudnn, nccl, nvshmem and triton - several GB
of CUDA that a CPU-only track will never execute. This script pins the CPU index for torch and
downloads wheels for a TARGET platform, which need not be this machine's.

  python scripts/make_offline_bundle.py --target linux-x86_64 --python 311
  python scripts/make_offline_bundle.py --target linux-aarch64 --python 311 --out bundle-arm

Then on the sandbox:

  pip install --no-index --find-links <bundle> -r <bundle>/requirements.txt

AFTER CHANGING requirements.txt, RE-RUN THIS SCRIPT. The bundle carries its own copy of
requirements.txt, and `pip install --no-index --find-links <bundle> -r <bundle>/requirements.txt`
reads THAT copy. Topping the directory up with a bare `pip download` leaves the copy stale, so a
newly pinned package sits in the bundle as an unused wheel and is silently not installed - which is
exactly how a missing openpyxl survived an otherwise successful offline install.

`--only-binary :all:` is deliberate: a source build on an unknown machine at a competition is a
failure mode, not a fallback. If a package has no wheel for the target, that is something to learn
NOW rather than on the day, so the download is allowed to fail loudly.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# pip needs the manylinux tag it will match against, not a friendly name.
TARGETS = {
    "linux-x86_64": ["manylinux2014_x86_64", "manylinux_2_28_x86_64", "linux_x86_64"],
    "linux-aarch64": ["manylinux2014_aarch64", "manylinux_2_28_aarch64", "linux_aarch64"],
    "win-amd64": ["win_amd64"],
    "macos-arm64": ["macosx_14_0_arm64", "macosx_11_0_arm64"],
}
# CPU-only torch lives here; without it Linux pulls the CUDA stack.
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=sorted(TARGETS), default="linux-x86_64")
    ap.add_argument("--python", default="311", help="target CPython, e.g. 311 or 312")
    ap.add_argument("--out", default=None, help="output directory (default: vendor/<target>-py<python>)")
    ap.add_argument("--requirements", default=str(ROOT / "requirements.txt"))
    a = ap.parse_args()

    # This project's .venv is built without pip, so sys.executable often cannot download anything.
    # Fail here with the fix rather than after an opaque "No module named pip".
    if subprocess.run([sys.executable, "-c", "import pip"], capture_output=True).returncode != 0:
        print(f"{sys.executable} has no pip. Run this with an interpreter that does, e.g.",
              file=sys.stderr)
        print(f"  py -3.11 scripts/make_offline_bundle.py --target {a.target} --python {a.python}",
              file=sys.stderr)
        return 2

    out = Path(a.out or ROOT / "vendor" / f"{a.target}-py{a.python}")
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "download", "-r", a.requirements, "-d", str(out),
           "--only-binary", ":all:", "--python-version", a.python,
           "--extra-index-url", TORCH_CPU_INDEX]
    for tag in TARGETS[a.target]:
        cmd += ["--platform", tag]

    print(f"target {a.target}, CPython {a.python} -> {out}")
    print("  " + " ".join(cmd[3:]) + "\n")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("\nDOWNLOAD FAILED. A package has no wheel for this target, or a pin is unsatisfiable.\n"
              "Fix it now - resolving it inside the competition sandbox is not an option.", file=sys.stderr)
        return r.returncode

    wheels = sorted(out.glob("*.whl")) + sorted(out.glob("*.tar.gz"))
    total = sum(f.stat().st_size for f in wheels)
    (out / "requirements.txt").write_text(Path(a.requirements).read_text(encoding="utf-8"), encoding="utf-8")
    (out / "INSTALL.md").write_text(
        f"# Offline install ({a.target}, CPython {a.python})\n\n"
        f"{len(wheels)} wheels, {total / 2**20:.0f} MiB. No network required.\n\n"
        "```bash\n"
        "python -m venv .venv && . .venv/bin/activate\n"
        "pip install --no-index --find-links . -r requirements.txt\n"
        "```\n\n"
        "torch here is the CPU build (`+cpu`): the default PyPI Linux wheel additionally installs\n"
        "cuda-toolkit / cudnn / nccl / triton, which this CPU-only toolchain never calls.\n",
        encoding="utf-8")
    print(f"\n{len(wheels)} file(s), {total / 2**20:.0f} MiB in {out}")
    print(f"install with:  pip install --no-index --find-links {out} -r {out / 'requirements.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
