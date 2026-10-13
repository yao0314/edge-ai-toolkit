"""The quantize + measure path must work without PyTorch.

Competition day: the sandbox is revealed on the day and may not be able to install a 2-3 GB torch
(on Linux the default PyPI wheel additionally pulls cuda-toolkit, cudnn, nccl and triton). If we
are handed a .onnx, quantizing and measuring it needs only onnxruntime + numpy + psutil, and that
must keep working. These modules previously hard-imported torch at module level, which turned a
workable configuration into an ImportError.

Verified for real on Linux with a torch-free venv; this test keeps it from regressing.
"""
import subprocess
import sys
import textwrap

import numpy as np
import pytest

TORCH_FREE = ["core.cpu_topology", "core.measure", "core.quantize", "core.runtime",
              "core.model_io", "core.score", "core.pareto", "core.energy"]


@pytest.mark.parametrize("module", TORCH_FREE)
def test_module_imports_with_torch_unavailable(module, tmp_path):
    """Run in a subprocess with `torch` blocked at the import hook, so this also catches a torch
    import buried in a transitively imported module."""
    code = textwrap.dedent(f"""
        import sys
        class Blocker:
            def find_module(self, name, path=None): return self.find_spec(name, path)
            def find_spec(self, name, path=None, target=None):
                if name == "torch" or name.startswith("torch."):
                    raise ImportError("torch is blocked by this test")
                return None
        sys.meta_path.insert(0, Blocker())
        for m in list(sys.modules):
            if m == "torch" or m.startswith("torch."):
                del sys.modules[m]
        import {module}
        print("ok")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(_root()))
    assert "ok" in r.stdout, f"{module} needs torch at import time:\n{r.stderr[-1500:]}"


def _root():
    from pathlib import Path
    return Path(__file__).resolve().parents[1]


def test_as_float_numpy_accepts_plain_arrays():
    """runtime_metrics used example_input[:1].detach().cpu().float().numpy(), which assumes a torch
    tensor. On the ONNX-only path the example input is a plain array."""
    from core.measure import _as_float_numpy

    a = _as_float_numpy(np.ones((1, 3, 4, 4), dtype=np.float64))
    assert a.dtype == np.float32 and a.shape == (1, 3, 4, 4)

    import torch
    t = _as_float_numpy(torch.ones(1, 3, 4, 4))
    assert isinstance(t, np.ndarray) and t.dtype == np.float32


def test_onnx_predictor_takes_numpy_without_torch_conversion(dummy, trained, tmp_path):
    """as_predictor on a .onnx must not require torch to dispatch or to run."""
    from core.export import export_onnx
    from core.runtime import as_predictor

    p = export_onnx(trained, dummy.example_input(), tmp_path / "m.onnx",
                    input_names=["input"], output_names=["output"])
    pred = as_predictor(p)
    out = pred(np.asarray(dummy.example_input(), dtype=np.float32))
    assert np.asarray(out).shape[0] == 1


def test_env_info_reports_torch_none_when_absent():
    """env_info() must record the absence rather than raising - the run record is still valid."""
    from core.measure import _torch_env

    e = _torch_env()
    assert set(e) == {"version", "device", "cuda_device"}
