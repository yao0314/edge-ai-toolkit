"""Parameter-count and ONNX-size ladder for candidate students.

The competition scores compression on parameter count AND on-disk size, both against the baseline,
with 10x needed for the top band. This enumerates candidates and prints the ratios before any
training time is spent, so the search space is known rather than guessed.
"""
import io, os, tempfile, warnings
import torch, torchvision.models as tvm

warnings.filterwarnings("ignore")
NUM_CLASSES, RES = 10, 224


def build(arch, width=None):
    kw = {"width_mult": width} if width and "mobilenet" in arch else {}
    m = getattr(tvm, arch)(weights=None, **kw)
    # retarget the classifier to 10 classes
    if hasattr(m, "fc"):
        m.fc = torch.nn.Linear(m.fc.in_features, NUM_CLASSES)
    elif hasattr(m, "classifier"):
        c = m.classifier
        last = c[-1] if isinstance(c, torch.nn.Sequential) else c
        new = torch.nn.Linear(last.in_features, NUM_CLASSES)
        if isinstance(c, torch.nn.Sequential):
            c[-1] = new
        else:
            m.classifier = new
    return m.eval()


def onnx_bytes(m, res=RES):
    f = os.path.join(tempfile.gettempdir(), "ladder.onnx")
    torch.onnx.export(m, torch.randn(1, 3, res, res), f, opset_version=17,
                      input_names=["x"], output_names=["y"], dynamo=False)
    n = os.path.getsize(f)
    os.remove(f)
    return n


CANDIDATES = [
    ("resnet50", None), ("resnet34", None), ("resnet18", None),
    ("mobilenet_v2", 1.0), ("mobilenet_v2", 0.75), ("mobilenet_v2", 0.5),
    ("mobilenet_v2", 0.35), ("mobilenet_v2", 0.25),
    ("mobilenet_v3_small", None), ("shufflenet_v2_x1_0", None),
    ("shufflenet_v2_x0_5", None), ("squeezenet1_1", None),
]

rows = []
for arch, w in CANDIDATES:
    try:
        m = build(arch, w)
    except Exception as e:
        print("skip %s: %s" % (arch, e)); continue
    p = sum(x.numel() for x in m.parameters())
    rows.append((("%s@%.2f" % (arch, w)) if w else arch, p, onnx_bytes(m)))

base_p, base_b = rows[0][1], rows[0][2]
print("基準 %s：%.2f M 參數、%.1f MB\n" % (rows[0][0], base_p / 1e6, base_b / 2**20))
print("%-24s %10s %9s %9s %9s %9s" % ("架構", "參數(M)", "參數倍率", "FP32 MB", "檔案倍率", "+INT8倍率"))
for name, p, b in rows:
    print("%-24s %10.2f %8.1fx %9.1f %8.1fx %8.1fx"
          % (name, p / 1e6, base_p / p, b / 2**20, base_b / b, base_b / (b / 4)))
print("\nCUDA:", torch.cuda.is_available(), "| threads:", torch.get_num_threads())
