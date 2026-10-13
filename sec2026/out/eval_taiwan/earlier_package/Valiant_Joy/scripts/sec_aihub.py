"""Compile + profile the baseline and the submitted model on Qualcomm AI Hub.

scripts/aihub_profile.py writes its results straight into sweep.csv, which cannot be used while
another measurement holds that file. This writes a standalone JSON instead, to be merged later.

Section 7 names "[PROFILE]" as a placeholder and says: "All teams should use the same device
profile, if provided. Otherwise, they can use their laptops." No profile was named, so the laptop
measurements are the sanctioned ones and this is a cross-check, which section 9 asks for under
Resource Efficiency only. Section 7 also says an unreachable service costs nothing provided the
local numbers are substituted and the substitution is stated.

The device is therefore chosen as a low-power embedded profile and documented, not reverse-engineered
from the Arduino UNO Q - that board belongs to the Hardware Deployment track, not this one. One trap
survives that change: "Arduino VENTUNO Q" is in the catalogue and is the obvious pick by name, and
it is a Dragonwing IQ8 / QCS8275 rated to 40 dense TOPS, which is not a constrained CPU target by
any reading.

  python scripts/sec_aihub.py --out results/sec2026_real/aihub.json \
      --models results/sec2026_real/models/p00_fp32.onnx results/sec2026_real/models/p40_int8.onnx
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import aihub

DEVICES = ["Dragonwing RB3 Gen 2 Vision Kit", "Dragonwing Q-6690 MTP", "QCS8550 (Proxy)"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compile-options", default="--target_runtime onnx --compute_unit cpu")
    a = ap.parse_args()

    import qai_hub as hub

    device = aihub.pick_device(DEVICES, client=hub)
    print(f"device: {device.name}", flush=True)

    out = {"device": device.name, "compile_options": a.compile_options, "models": {}}
    for m in a.models:
        name = Path(m).stem
        print(f"submitting {name} ...", flush=True)
        try:
            r = aihub.profile_onnx(m, (1, 3, 224, 224), device, client=hub, name=name,
                                   compile_options=a.compile_options)
            out["models"][name] = r
            print(f"  {name}: {r}", flush=True)
        except Exception as e:                    # a queue timeout must not lose the other model
            out["models"][name] = {"error": str(e)}
            print(f"  {name} FAILED: {e}", flush=True)
        Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
