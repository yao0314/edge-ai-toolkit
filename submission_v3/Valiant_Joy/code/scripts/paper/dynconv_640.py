"""Experiment 25 (K4): build the YOLOv8n-640 dynamic-Conv sibling, pre-registered in FINDINGS.md.

detection_640 has FP32 and static INT8 at 640 px but no dynamic-Conv arm, and the detection sweep
cannot produce one at this resolution without re-running the whole pipeline. Dynamic quantization
needs no calibration data, so the arm is built directly from the existing FP32 file with the same
settings detection_dyn used ({method: dynamic, op_types_to_quantize: [Conv, MatMul]}). mAP is not
measured; this arm exists only to be timed.

    python scripts/paper/dynconv_640.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.quantize import quantize_onnx  # noqa: E402

SRC = ROOT / "results" / "detection_640" / "models"
DST = ROOT / "results" / "detection_640_dyn" / "models"


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    for name in ("640_p00_fp32.onnx", "640_p00_int8.onnx"):
        shutil.copy2(SRC / name, DST / name)
    info: dict = {}
    out = quantize_onnx(DST / "640_p00_fp32.onnx", DST / "640_p00_int8_dyn_conv.onnx", None,
                        {"method": "dynamic", "op_types_to_quantize": ["Conv", "MatMul"]}, info=info)
    for p in sorted(DST.glob("*.onnx")):
        print(f"  {p.name:28s} {p.stat().st_size:>10,d} B")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
