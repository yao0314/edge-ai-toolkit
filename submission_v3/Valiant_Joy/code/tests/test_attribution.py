"""Authorship must stay in the code AND in the artefacts the toolkit produces.

This is attribution, not a lock: anyone editing the source can remove it. What it does guarantee is
that the name cannot vanish silently - the suite fails, and every model/report already produced keeps
its embedded metadata.
"""
import json

import onnx
import pytest

import core
from core.export import export_onnx, stamp_onnx
from core.quantize import quantize_onnx

AUTHOR = "楊元蓁 (Price Yang)"


def test_author_constant_is_intact():
    assert core.AUTHOR == core.__author__ == AUTHOR
    assert AUTHOR in core.PRODUCER and AUTHOR in core.__copyright__
    prov = core.provenance()
    assert prov["author"] == AUTHOR and AUTHOR in prov["produced_by"]


@pytest.mark.parametrize("path", ["AUTHORS.md", "README.md", "core/__init__.py"])
def test_author_named_in_repo_files(path):
    from core.config import ROOT

    assert AUTHOR.split(" (")[0] in (ROOT / path).read_text(encoding="utf-8")


def test_exported_and_quantized_models_carry_the_author(tmp_path, dummy, trained):
    fp32 = export_onnx(trained, dummy.example_input(), tmp_path / "m.onnx")
    q = quantize_onnx(fp32, tmp_path / "m_int8.onnx", dummy.load_data()["calib"], {"calib_batches": 2})
    for path in (fp32, q):
        m = onnx.load(str(path))
        meta = {p.key: p.value for p in m.metadata_props}
        assert meta["author"] == AUTHOR, f"{path.name} lost its author metadata"
        assert AUTHOR in m.producer_name
    assert len({p.key for p in onnx.load(str(stamp_onnx(fp32))).metadata_props}) == len(core.provenance())


def test_sweep_summary_and_reports_carry_the_author(tmp_path, dummy, trained):
    from core.sweep import run_sweep

    cfg = {"task": "dummy", "measure": {"threads": [1], "warmup": 1, "runs": 5, "min_time_s": 0.05},
           "sweep": {"prune_ratios": [0.0], "precisions": ["fp32"]}, "quantize": {"calib_batches": 1}}
    run_sweep(cfg, dummy, trained, tmp_path)
    from core.sweep import finalize
    import pandas as pd

    finalize(pd.read_csv(tmp_path / "sweep.csv"), cfg, tmp_path, dummy)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["author"] == AUTHOR and AUTHOR in summary["produced_by"]
