"""QAT must simulate quantization during training and leave a plain FP32 model behind."""
import torch

from core.export import export_onnx
from core.qat import FakeQuant, insert_fake_quant, observe, qat_finetune
from core.quantize import quantize_onnx


def test_fake_quant_rounds_forward_but_passes_gradients():
    fq = FakeQuant(bits=8)
    x = torch.linspace(-1, 1, 256, requires_grad=True)
    fq(x)                                   # observe the range first
    fq.observing = False
    y = fq(x)
    assert not torch.equal(y, x)            # values are snapped to the INT8 grid
    assert (y - x).abs().max() < (float(fq.hi) - float(fq.lo)) / 255      # ... by less than one step
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))                     # straight-through gradient


def test_disabled_or_unobserved_fake_quant_is_identity():
    fq = FakeQuant()
    x = torch.randn(10)
    fq.observing = False
    assert torch.equal(fq(x), x)            # no range observed yet -> identity
    fq.observing = True
    fq(x)                                   # observe
    fq.observing = False
    assert not torch.equal(fq(x), x)        # now it quantizes
    fq.enabled = False
    assert torch.equal(fq(x), x)            # explicitly disabled -> identity again


def test_insert_and_remove_leaves_the_model_unchanged(dummy, trained):
    x = dummy.example_input()
    with torch.no_grad():
        ref = trained(x).clone()
    handles, quants = insert_fake_quant(trained, skip=dummy.ignored_layers(trained))
    assert len(quants) >= 3
    observe(trained, dummy.load_data()["calib"], "cpu", batches=2)
    with torch.no_grad():
        quantized = trained(x)
    assert not torch.allclose(quantized, ref, atol=1e-6)    # quantization is actually simulated
    for h in handles:
        h.remove()
    with torch.no_grad():
        assert torch.allclose(trained(x), ref, atol=1e-6)   # ... and fully removable


def test_qat_finetune_returns_an_exportable_fp32_model(tmp_path, dummy, trained):
    model = qat_finetune(dummy, trained, epochs=1, cfg={"observe_batches": 2})
    assert not any(isinstance(m, FakeQuant) for m in model.modules())
    fp32 = export_onnx(model, dummy.example_input(), tmp_path / "qat.onnx")   # verifies ORT == PyTorch
    info = {}
    quantize_onnx(fp32, tmp_path / "qat_int8.onnx", dummy.load_data()["calib"], {"calib_batches": 2}, info=info)
    assert info["int8_quantized_nodes"] > 0
    assert dummy.evaluate(str(fp32))["score"] > 0.5
