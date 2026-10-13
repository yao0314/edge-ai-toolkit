# Independent re-measurement of `valiant_joy_Model.onnx` on a second machine

The figures in `metrics.json` were measured on the team's development laptop (i7-13620H, AVX-VNNI,
4 intra-op threads, CodeCarbon in TDP mode, the team's own split). We re-ran the same ONNX file on
the on-site laptop with a different, independently generated split, to show how the claims travel.

Machine: AMD Ryzen 7 4800H (8C/16T, AVX2, no AVX-VNNI), 16 GB, CPU only, Windows 11, ONNX Runtime
1.30.0, default runtime threads. Validation: class-stratified 80/20 split of torchvision EuroSAT,
seed 42, 5,400 images. Baseline = the provided student exported to FP32 ONNX, measured the same way.
Latency = mean of 50 runs after 10 warm-up; peak RAM = raw process RSS sampled every 10 ms; energy =
CodeCarbon, **measured** (`windows_emi`); medians of 3 sessions. Raw records: the JSON files here.

| | Baseline (FP32 ONNX) | valiant_joy_Model.onnx | Ratio |
|---|---|---|---|
| Top-1 accuracy | 99.33% | 96.48% | 97.1% retained |
| Worst-class accuracy | 96.40% (PermanentCrop) | 91.80% (PermanentCrop) | -4.6 pt |
| Parameters (ONNX graph, Conv/Gemm weights + biases) | 2.220 M | 0.394 M | 5.63x |
| File size | 8.51 MiB | 0.60 MiB | 14.07x |
| MACs (ONNX graph) | 0.300 G | 0.068 G | 4.42x |
| Latency | 6.49 ms | 3.48 ms | 1.86x |
| Peak RAM | 82.3 MB | 68.1 MB | 1.21x |
| Energy per inference | 0.1475 J | 0.0667 J | 2.21x |

The file accepts `(1, 3, 224, 224)` and returns `(1, 10)`; its graph is MobileNetV2's topology (52
convolutions, 17 depthwise, 10 residual adds) with fewer channels, i.e. the provided student after
structured pruning. Speed-ups are smaller on this CPU than on the i7-13620H (3.13x) and on AI Hub's
QCS6490 (8.94x): the INT8 kernels have no VNNI path here. Accuracy differs from `metrics.json`
because the split differs; on this split the retained accuracy is 97.1%.
