"""Architecture compatibility matrix: every family must survive prune -> export -> INT8 -> ORT.

  depthwise  : MobileNetV2          residual : ResNet18
  attention  : MobileViT-XXS, DeiT-tiny (timm)
  multi-head detection with NMS-free export : YOLOv8n, YOLO11n (ultralytics, C2f / C2PSA)
  time series: 1D-CNN autoencoder
Random-init weights and random data: this checks the mechanics, not accuracy.
"""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from core.export import export_onnx
from core.prune import prune_model
from core.quantize import quantize_onnx
from core.runtime import as_predictor


def _mobilenet_v2():
    import torchvision.models as tvm
    m = tvm.mobilenet_v2(num_classes=10)
    return m, torch.randn(1, 3, 64, 64), [m.classifier[-1]]


def _resnet18():
    import torchvision.models as tvm
    m = tvm.resnet18(num_classes=10)
    return m, torch.randn(1, 3, 64, 64), [m.fc]


def _timm(name, **kw):
    timm = pytest.importorskip("timm")
    m = timm.create_model(name, num_classes=10, **kw)
    return m


def _mobilevit():
    m = _timm("mobilevit_xxs")
    return m, torch.randn(1, 3, 64, 64), [m.get_classifier()]


def _deit_tiny():
    m = _timm("deit_tiny_patch16_224", img_size=32, patch_size=4)
    return m, torch.randn(1, 3, 32, 32), [m.get_classifier()]


def _yolo(cfg):
    pytest.importorskip("ultralytics")
    from ultralytics.nn.tasks import DetectionModel

    from adapters.ultralytics_compat import make_prunable
    m = DetectionModel(cfg, nc=80, verbose=False).eval()
    m.model[-1].export = True  # decoded boxes+scores, NMS runs outside the graph
    make_prunable(m)
    return m, torch.randn(1, 3, 160, 160), [m.model[-1]]


def _conv_ae():
    from adapters.anomaly import ConvAE1d
    m = ConvAE1d(1, (32, 64))
    return m, torch.randn(1, 1, 144), [m.out]


FAMILIES = {
    "mobilenet_v2": _mobilenet_v2, "resnet18": _resnet18, "mobilevit_xxs": _mobilevit,
    "deit_tiny": _deit_tiny, "yolov8n": lambda: _yolo("yolov8n.yaml"),
    "yolo11n": lambda: _yolo("yolo11n.yaml"), "conv_ae1d": _conv_ae,
}


@pytest.mark.parametrize("family", list(FAMILIES))
def test_family_prune_export_int8(family, tmp_path):
    torch.manual_seed(0)
    model, x, ignored = FAMILIES[family]()
    model.eval()
    n0 = sum(p.numel() for p in model.parameters())
    pruned = prune_model(model, x, 0.3, ignored, round_to=8)
    assert sum(p.numel() for p in pruned.parameters()) < 0.85 * n0, "pruning removed too little"
    with torch.no_grad():
        y0, y1 = model(x), pruned(x)
    assert y0.shape == y1.shape  # output interface unchanged (classes / boxes / reconstruction)

    fp32 = export_onnx(pruned, x, tmp_path / f"{family}.onnx")  # verifies ORT == PyTorch
    calib = DataLoader(TensorDataset(torch.randn(8, *x.shape[1:])), batch_size=4)
    int8 = quantize_onnx(fp32, tmp_path / f"{family}_int8.onnx", calib, {"calib_batches": 2})
    out = as_predictor(str(int8))(x.numpy())
    assert out.shape == tuple(y1.shape)
    assert int8.stat().st_size < fp32.stat().st_size


def test_splitc2f_is_exact():
    pytest.importorskip("ultralytics")
    from ultralytics.nn.tasks import DetectionModel

    from adapters.ultralytics_compat import SplitC2f, make_prunable
    m = DetectionModel("yolov8n.yaml", nc=80, verbose=False).eval()
    x = torch.randn(1, 3, 160, 160)
    with torch.no_grad():
        a = m(x)[0]
        make_prunable(m)
        b = m(x)[0]
    assert any(isinstance(mod, SplitC2f) for mod in m.modules())
    assert torch.equal(a, b)
