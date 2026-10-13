# Valiant Joy — the code, and the comparison it exists to let you re-run

> **Our claim is a comparison, so this repository is organised around the comparison rather than
> around the model.** We say the stopping point for compression is set by how well you recover, not
> by how hard you prune. That is only believable if you can run both recoveries yourself and watch
> the weakest class move. Four commands below do that. The submitted model falls out of the second
> one.

## The claim, as numbers

Every row here is a measured variant, INT8, against the provided student. Nothing is interpolated.

| prune | recovery | retained | weakest class | params | file | latency |
|---|---|---|---|---|---|---|
| 0 % | — | 99.81 % | 0.9384 | 1.00× | 3.40× | 1.95 ms |
| 10 % | hard labels | 100.24 % | 0.9734 | 1.21× | 4.03× | 1.85 ms |
| 20 % | hard labels | 100.13 % | 0.9803 | 1.51× | 4.90× | 1.74 ms |
| 30 % | hard labels | 99.50 % | 0.9630 | 1.96× | 6.14× | 1.59 ms |
| 40 % | hard labels | 98.73 % | 0.9528 | 2.61× | 7.81× | 1.47 ms |
| 50 % | hard labels | 98.04 % | 0.9322 | 3.80× | 10.56× | 1.18 ms |
| 50 % | **teacher** | 98.06 % | 0.9199 | 3.80× | 10.56× | 1.20 ms |
| 60 % | hard labels | 97.03 % | **0.8480** | 5.58× | 14.06× | 1.13 ms |
| **60 %** | **teacher** | **98.04 %** | **0.9261** | **5.58×** | **14.07×** | **1.12 ms** |
| 70 % | hard labels | 95.76 % | 0.8770 | 9.39× | 19.99× | 0.98 ms |
| 70 % | teacher | 97.18 % | 0.8850 | 9.39× | 20.01× | 1.00 ms |

Read the two 60 % rows against each other. Same cut, same architecture, same quantization, and a
weakest class that differs by seven and a half points depending on nothing but what the fine-tuning
loss was computed against. Then read the two 50 % rows, where the same change is worth nothing at
all. That non-monotonicity is the finding, and it is why the repository ships both ladders instead
of only the winning one.

## What is in this folder

| file | what it is |
|---|---|
| `valiant_joy_int8.onnx` | **the submission.** 60 % structured prune, KD recovery against the provided teacher, static INT8 QDQ. Input pinned to `(1, 3, 224, 224)`. |
| `valiant_joy_fp32.onnx` | the same pruned and recovered model before quantization, so the quantization step is checkable rather than asserted |
| `metrics.json` | section 8 schema. Baseline and final measured identically, same machine, same script |
| `baseline_metrics.json` | the provided student measured by us — the denominator of every ratio we claim |
| `val_split.json` | our 21,600 / 5,400 split over the 27,000 EuroSAT indices, disjoint |
| `tradeoff_curve.csv` | every measured variant in the table above. Each number in the report is a column of this file |
| `tradeoff.png` | the two panels the report argues from |
| `REPORT.md` | the 2-page technical report |

## The four commands

```bash
# 0. environment — Python 3.11, CPU-only torch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxscript onnxruntime thop codecarbon psutil qai-hub matplotlib

# 1. the hard-label ladder: prune 0-50 %, export, quantize, measure at each point
python scripts/compress.py --config configs/experiments/sec2026_real.yaml \
                           --model data/sec2026/student_mobilenetv2_distilled.pt
#    …and its extension past the cliff
python scripts/compress.py --config configs/experiments/sec2026_deep.yaml \
                           --model data/sec2026/student_mobilenetv2_distilled.pt

# 2. THE COMPARISON. Identical, except recovery distils from teacher_resnet50_eurosat.pt.
#    results/sec2026_kd/models/p60_int8.onnx is the submitted file.
python scripts/compress.py --config configs/experiments/sec2026_kd.yaml \
                           --model data/sec2026/student_mobilenetv2_distilled.pt

# 3. energy: seven CodeCarbon sessions per row, median kept, impossible readings dropped
python scripts/sec_energy.py --csv results/sec2026_kd/sweep.csv \
                             --threads 4 --seconds 30 --repeats 7 --only p60_int8

# 4. the AI Hub cross-check, then metrics.json and the figure
python scripts/sec_aihub.py   --out results/sec2026_kd/aihub.json \
       --models results/sec2026_real/models/p00_fp32.onnx results/sec2026_kd/models/p60_int8.onnx
python scripts/sec_metrics.py --sweep results/sec2026_real/sweep_with_kd.csv \
                              --final kd_p60_int8 --baseline p00_fp32 --team "Valiant Joy"
python scripts/sec_figure.py  --csv results/sec2026_real/sweep.csv \
       --deep-csv results/sec2026_deep/sweep.csv --kd-csv results/sec2026_kd/sweep.csv \
       --chosen p60_int8 --out tradeoff.png
```

The two configs in step 1 and step 2 differ in four lines: `finetune_mode: distill`,
`distill.teacher_checkpoint`, `teacher_kwargs`, and the epoch budget. Diff them; that diff is the
whole claim.

## Two more comparisons we ran and did not submit

Both are in the repository because the report cites them and because a result that did not work is
still a measurement.

**Per-layer prune allocation** (`configs/experiments/sec2026_alloc.yaml`). Instead of the same ratio
everywhere, probe each layer's sensitivity and solve for the least-damage allocation. At a matched
parameter count it is genuinely better — 0.9707 on the weakest class against a uniform 0.9630 — and
it **cannot go deep**: asked for 60 % of parameters it delivered 46.6 %, touching eleven of
forty-nine layers. The damage budget runs out. The parameters that are cheap to remove are a small,
bounded pool, so depth necessarily takes expensive ones, which is the same reason the weakest class
is what breaks first.

**The conservative submission** (`git checkout submission-v1 -- submission/`). The 40 % prune retains
98.73 %, lifts the weakest class to 0.9528 — above the baseline's own 0.9466 — and puts no class
under 0.95. It is a better model on the accuracy criterion and a worse one on the other three. We
kept it reachable rather than deleting it, because that trade is a judgement and not a fact.

## What reproduces exactly, and what does not

**Exactly.** The split (`val_split.json` is the file; the recipe that generated it is written into
`configs/experiments/sec2026_real.yaml`), the architecture at every prune ratio, parameter counts,
MACs, file sizes, the ONNX graphs, and accuracy including per-class recall. Seeds are set; export is
opset 13 with `dynamo=False` and a pinned batch dimension.

**Within noise.** Latency and peak RAM depend on the machine. Ours: 13th Gen Core i7-13620H, 16 GB,
Windows 11, ONNX Runtime 1.30.0, 4 intra-op threads, recorded in `metrics.json`. The *ratio* is what
section 9 scores and what should transfer.

**Not exactly, and the report says so.** Energy: no RAPL on this machine, so `codecarbon_mode` is
`tdp_estimate` and session spread is wide — implied power wandered between 46 W and 156 W, which is
why the script takes medians over seven sessions and drops readings implying more power than the
part can draw. The check that the medians are sound is not repetition but agreement: 62.2 W implied
for the baseline against 63.4 W for the submission, and a 3.07× energy ratio against a 3.13× latency
ratio, which is what the packet says a TDP-mode figure must look like. And recovery fine-tuning ran
on a GPU — eight KD epochs at about 40 s each. The code is unchanged on CPU and that step will
dominate the wall clock there.

## If you only want to check the claim

```bash
python -c "
import onnxruntime as ort, numpy as np
s = ort.InferenceSession('valiant_joy_int8.onnx', providers=['CPUExecutionProvider'])
print(s.get_inputs()[0].name, s.get_inputs()[0].shape)
print(s.run(None, {s.get_inputs()[0].name: np.zeros((1,3,224,224), np.float32)})[0].shape)
"
```

For accuracy, load the ONNX file in place of the PyTorch model in the packet's own `evaluate`, over
`Subset(full_dataset, json.load(open('val_split.json'))['val'])`, with
`Resize(224) → ToTensor → Normalize` and no crop. We get 0.9715 top-1 and 0.9261 on the weakest of
the ten classes.
