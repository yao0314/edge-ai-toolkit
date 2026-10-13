# Reproducing our SEC 2026 submission (EuroSAT, Software Track)

Everything starts from the provided `student_mobilenetv2_distilled.pt`. CPU only; Python 3.11.

## 0. Environment

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt "qai-hub==0.55.0" codecarbon
```

Put the Starter Kit in `starter_kit/` (`student_mobilenetv2_distilled.pt`, optional
`teacher_resnet50_eurosat.pt`). EuroSAT RGB goes to `starter_kit/data` (torchvision layout,
`eurosat/2750/<class>/*.jpg`); if it is not pre-cached:

```powershell
.venv\Scripts\python -c "from torchvision import datasets; datasets.EuroSAT(root='starter_kit/data', download=True)"
```

(torchvision verifies the archive's MD5, `c8fa014336c82ac7804f0398fcb19387`.)

## 1. Split and baseline (Section 5)

The organisers told teams verbally to produce `val_split.json` and `baseline_metrics.json` on their
own device. Our split is class-stratified 80/20 with seed 42 (21,600 train / 5,400 val); re-running
gives a byte-identical file.

```powershell
.venv\Scripts\python sec2026\make_split.py
.venv\Scripts\python sec2026\export_quant.py --out sec2026\out\p00 --fp32-only
.venv\Scripts\python sec2026\measure.py --model starter_kit\student_mobilenetv2_distilled.pt --name baseline_pt_accuracy --repeats 0
.venv\Scripts\python sec2026\measure.py --model sec2026\out\p00\student_model.onnx --name baseline_onnx_accuracy --repeats 0
# on a quiet machine (whole-machine CPU < 15 %, on AC power):
.venv\Scripts\python sec2026\measure.py --model starter_kit\student_mobilenetv2_distilled.pt --name baseline_pt_timing --skip-accuracy
.venv\Scripts\python sec2026\measure.py --model sec2026\out\p00\student_model.onnx --name baseline_onnx_timing --skip-accuracy
.venv\Scripts\python sec2026\make_baseline_metrics.py --pt-accuracy sec2026\results\baseline_pt_accuracy.json --pt-timing sec2026\results\baseline_pt_timing.json --onnx-accuracy sec2026\results\baseline_onnx_accuracy.json --onnx-timing sec2026\results\baseline_onnx_timing.json
```

## 2. Static INT8 of the unpruned student (Section 6.2)

```powershell
.venv\Scripts\python sec2026\export_quant.py --out sec2026\out\p00_int8_pct_rr --calib 256 --check 128 --method percentile --reduce-range
```

## 3. Structured pruning sweep with short recovery fine-tuning (Section 6.1)

```powershell
.venv\Scripts\python sec2026\prune_sweep.py --ratios 0.2 0.3 0.1 --ft-steps 600 --val-limit 1000 --no-int8 --out sec2026\out\sweep600
```

## 4. Final model (the submitted `student_model_int8.onnx`)

```powershell
# 20% structured pruning + BN re-estimation + 600-step recovery fine-tune (CPU, ~20 min)
.venv\Scripts\python sec2026\prune_sweep.py --ratios 0.2 --ft-steps 600 --val-limit 1000 --no-int8 --out sec2026\out\sweep600
# static INT8 (QDQ, U8S8, per-channel, reduce_range, percentile calibration on 128 train images)
.venv\Scripts\python sec2026\export_quant.py --model sec2026\out\sweep600\p20\pruned.pt --out sec2026\out\p20_int8 --calib 128 --check 128 --method percentile --reduce-range
# accuracy on the full val split, then latency / peak RAM / energy on a quiet machine
.venv\Scripts\python sec2026\measure.py --model sec2026\out\p20_int8\student_model_int8.onnx --name p20_int8_accuracy --repeats 0
.venv\Scripts\python sec2026\measure.py --model sec2026\out\p20_int8\student_model_int8.onnx --name p20_int8_timing --skip-accuracy
.venv\Scripts\python -c "import sys; sys.path.insert(0,'sec2026'); import common, make_baseline_metrics as mb; common.save_json(mb.merge('sec2026/results/p20_int8_accuracy.json','sec2026/results/p20_int8_timing.json'),'sec2026/results/p20_int8.json')"
# AI Hub (same device and runtime for baseline and final)
.venv\Scripts\python sec2026\aihub_profile.py --model sec2026\out\p00\student_model.onnx --device "Dragonwing RB3 Gen 2 Vision Kit" --tag _ortgt
.venv\Scripts\python sec2026\aihub_profile.py --model sec2026\out\p20_int8\student_model_int8.onnx --device "Dragonwing RB3 Gen 2 Vision Kit" --tag _ortgt
# metrics.json (Section 8) and the tradeoff plot
.venv\Scripts\python sec2026\make_metrics.py --baseline sec2026\results\baseline_onnx.json --final sec2026\results\p20_int8.json --team "Valiant Joy" --aihub sec2026\out\p00\aihub_student_model_ortgt.json --aihub sec2026\out\p20_int8\aihub_student_model_int8_ortgt.json --out metrics.json
.venv\Scripts\python sec2026\plot_tradeoff.py --out tradeoff.png
```

Fine-tuning is seeded (`torch.manual_seed(0)`), but CPU kernels are not bit-exact across machines, so a
re-run lands close to, not exactly on, our accuracy; `results/` holds every number we measured.
The `code/` folder is the project root: put the Starter Kit in `code/starter_kit/` and run the commands
from `code/`.