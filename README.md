# edge-ai-toolkit
**Original toolkit author and maintainer:** Price Yang
**SEC 2026 collaborating contributor:** Weiyao (David) Chang

See [AUTHORS.md](AUTHORS.md) for attribution details.

---

## SEC 2026 Competition Results — Team Valiant Joy

**3rd Place — Software Optimization Track** | Santa Clara University | October 13–15, 2026

Team Valiant Joy used this toolkit for **EuroSAT satellite-image classification**, optimizing a MobileNetV2 student model with **structured channel pruning**, **ResNet50-teacher knowledge distillation**, and **static INT8 quantization** for ONNX deployment.

### Reported optimization results

| Metric | Result | Measurement context |
| --- | ---: | --- |
| Competition placement | **3rd Place** | SEC 2026 Software Optimization Track |
| Parameter reduction | **5.58×** | Baseline vs. optimized model, team technical report |
| Baseline accuracy retained | **98.04%** | Ratio of optimized accuracy to baseline accuracy, team technical report |
| ONNX model-size reduction | **~14.5×** | Baseline vs. final INT8 ONNX, team technical report |
| CPU inference speedup | **8.94×** | Qualcomm AI Hub profiling on **QCS6490**, 27.969 ms → 3.127 ms |
| Peak memory reduction | **1.66×** | Qualcomm AI Hub profiling on **QCS6490**, 32.33 MB → 19.45 MB |

**Measurement note:** The QCS6490 results above are **team-run Qualcomm AI Hub profiles**, not measurements from a physical QCS6490 board and not organizer-certified performance figures. Local ONNX Runtime evaluations are a separate measurement context and should not be directly compared with AI Hub latency. Competition placement is the team's reported final result.

### Submission artifacts

- [Final submission package](submission_v3/Valiant_Joy/) — technical report, presentation, code, model files, and evaluation metrics
- [INT8 ONNX model](submission_v3/Valiant_Joy/valiant_joy_int8.onnx)
- [FP32 ONNX model](submission_v3/Valiant_Joy/valiant_joy_fp32.onnx)
- [Submission record](submission_v3/SUBMITTED.md) — identifies the files actually submitted

### Collaboration and attribution

The toolkit was originally developed by **Price Yang**. **Weiyao (David) Chang** contributed to the collaborative SEC 2026 model-optimization, evaluation, and competition-deliverable effort. These descriptions distinguish the toolkit's original authorship from the team's competition work; see [AUTHORS.md](AUTHORS.md). The contribution descriptions should be adjusted if the collaborators agree on a more detailed division of responsibilities.

---

---


A task-agnostic model compression and measurement toolchain for the **SEC 2026 Minimalist Edge AI Challenge, Software Optimization Track**.

**Original toolkit author and maintainer:** Price Yang. **SEC 2026 collaborating contributor:** Weiyao (David) Chang. See [AUTHORS.md](AUTHORS.md) for attribution details.

Input: a model plus a task adapter. Output: several compressed versions (structured pruning × INT8 quantization), every metric for each version, a Pareto plot, and a configurable total score.

> **Hard rule: `core/` contains no task-specific code.** Everything task-specific lives in `adapters/`. `tests/test_core_is_task_agnostic.py` checks this automatically.

---

## Competition facts: confirmed vs. unconfirmed

Sources: the [official page](https://acm-ieee-sec.org/2026/student_innovation_challenge.php) and the official flyer (checked 2026-09-11).

| Item | Status |
|---|---|
| Date / location | 2026/10/13–15, Santa Clara University (confirmed) |
| Software track format | "virtual development environment" / "virtual sandbox" / **"No hardware required"** (confirmed). **We will most likely have no physical UNO Q on site.** |
| Judging criteria | model accuracy, model compression, resource efficiency, energy parsimony (confirmed; **weights unknown**) |
| Target hardware | Arduino UNO Q (QRB2210, 4× Cortex-A53) → inference on CPU |
| Scoring formula / weights | **Unknown** → `score:` section in the YAML, equal weights by default |
| Competition environment (OS / Python / network / GPU) | **Unknown** → pure CPU + offline supported |
| Competition length | **Unknown** → every script has `--quick` (≤ 5 min) |
| Bringing our own code | **Unknown** → the toolchain has few dependencies and pinned versions, so it can be reinstalled on site |
| How energy is measured | **Unknown** → `energy_j` has a hook and stays empty by default. We never make up a number. |

---

## Installation

Tested with Python 3.11.9 on Windows 11 and an RTX 4060 (it also runs on CPU only).

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
# Windows GPU (optional): pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
```

The first run downloads datasets to `data/` and pretrained weights to `weights/` (`TORCH_HOME` points to `weights/torch`). **After that everything runs offline.** Uses the GPU automatically if present, otherwise the CPU.

## Quick start

```bash
# Phase 1: baseline measurement table
python scripts/measure.py --task classification --quick   # < 5 min, CPU is fine
python scripts/measure.py --task classification           # full (CIFAR-10, 15 epochs)

# measure a model file we were handed (see "Model format compatibility" below)
python scripts/measure.py --task classification --model given.onnx         # with accuracy
python scripts/measure.py --model given.tflite --input-shape 1,224,224,3   # no adapter: resources only

# Phase 2: parameter sweep + Pareto
python scripts/sweep.py --config configs/classification.yaml --quick
python scripts/sweep.py --config configs/classification.yaml [--resume]
python scripts/sweep.py --config configs/classification.yaml --model given.onnx   # ONNX only: skip pruning

# one command from model to submittable deliverable (sweep -> pick -> package)
python scripts/compress.py --task classification
python scripts/compress.py --config configs/detection.yaml --reuse   # reuse an existing sweep

# re-measure an existing sweep after changing measurement settings / moving to another machine (no retraining)
python scripts/remeasure.py --config configs/classification.yaml

# Excel report of every result (Summary / Candidates / All variants / Datasets / Method / Pitfalls / Limits)
python scripts/make_report_xlsx.py

# dead-code audit: unused imports/defs, config keys nothing reads, unreferenced modules
python scripts/check_dead_code.py

# verify a downloaded dataset before a long run (every image decodes, every label parses)
python scripts/check_data.py data/coco_train5000 data/coco_val500

# build an offline wheel bundle for a target platform (the sandbox may have no network)
python scripts/make_offline_bundle.py --target linux-x86_64 --python 311
python scripts/make_offline_bundle.py --target linux-aarch64 --python 311
#   then, on the sandbox:  pip install --no-index --find-links <bundle> -r <bundle>/requirements.txt

# package the toolkit to send elsewhere (excludes .venv / data / weights / vendor / result binaries)
python scripts/make_dist.py            # source + result summaries + deliverables (~46 MB zipped)
python scripts/make_dist.py --lean     # source only (~1 MB zipped)

# tests (core smoke tests, < 2 min, no downloads)
python -m pytest -q
```

Output goes to `results/<task>/` (`--quick` writes to `results/<task>_quick/`):
`baseline.csv/json`, `sweep.csv`, `summary.json`, `pareto_<x>.png`, and `models/*.onnx` (not committed to git).

---

## Architecture

```
core/                  # task-agnostic
  adapter.py           # TaskAdapter abstract interface
  measure.py           # metric row (accuracy/size/params/MACs/latency/RSS/CPU/energy)
  _probe.py            # isolated subprocess that measures latency/CPU/peak RSS
  energy.py            # energy hook (none | csv | command | aihub); never estimated
  export.py            # ONNX export + PyTorch vs. ORT output comparison
  quantize.py          # ONNX Runtime INT8 (static QDQ / dynamic), keep_fp32
  prune.py             # torch-pruning structured channel pruning + finetune
  distill.py           # knowledge distillation (soft target + hard label)
  sweep.py             # prune ratios × precisions → CSV + Pareto
  score.py             # normalize + weighted total score (weights from YAML)
  pareto.py            # Pareto front + plotting
  aihub.py             # (optional) Qualcomm AI Hub compile + profile
  model_io.py          # load any model file (.pt module/checkpoint/state_dict/TorchScript, .pt2, .onnx, .tflite)
  cpu_topology.py      # pin measurements to the fastest core class (avoids P/E-core noise)
  runtime.py           # torch / ONNX / TFLite shared predictor (auto NCHW↔NHWC, fixed batch)
  baseline.py          # baseline training / caching
  config.py            # YAML, --quick override, adapter loading
adapters/              # task-specific
  classification.py    # MobileNetV2 + CIFAR-10 (or any ImageFolder)
  anomaly.py           # 1D-CNN autoencoder + ECG5000 (or your own npz), AUC
scripts/               # CLI
configs/               # one YAML per task
tests/                 # pytest smoke tests (dummy adapter, no downloads)
```

### TaskAdapter

```python
class TaskAdapter(ABC):
    def load_data(self) -> dict            # {"train", "val", "calib"} DataLoaders
    def build_model(self) -> nn.Module     # baseline architecture
    def evaluate(self, model) -> dict      # {"metric_name", "score"}; model may be nn.Module / .onnx path / ORT session
    def example_input(self) -> Tensor      # batch=1
    def finetune(self, model, epochs) -> nn.Module
    def ignored_layers(self, model) -> list   # layers pruning must skip (output layer / task head)
    # optional: train_baseline, build_student, distill_loss, input_names, output_names, describe
```

Inside `evaluate`, call `core.runtime.as_predictor(model)` to get a uniform `f(batch) -> np.ndarray`, so the same code handles torch, ONNX and TFLite.

---

## Model format compatibility (the three on-site scenarios)

The organizers **have not announced** what they will hand out on site. The toolchain accepts any of these, and `core/model_io.py` detects the format automatically:

| What we are given | How to load it | Prune | INT8 | Measure |
|---|---|---|---|---|
| `.pt/.pth` full `nn.Module` | `--model x.pt` or `model: {path: x.pt}` in the YAML | ✅ | ✅ | ✅ |
| checkpoint dict (`{"model": module, ...}`, ultralytics style) | same as above; uses the `model`/`ema` key automatically | ✅ | ✅ | ✅ |
| `state_dict` | plus `--arch pkg.module:factory` (or `model.arch`); without it, the adapter's `build_model()` is used | ✅ | ✅ | ✅ |
| TorchScript (`torch.jit`) | `--model x.pt` (detected automatically) | ❌ skipped | ✅ | ✅ |
| `torch.export` (`.pt2`) | `--model x.pt2` | ❌ skipped | ✅ | ✅ |
| **ONNX only** | `--model x.onnx` | ❌ skipped | ✅ ORT | ✅ ORT |
| **TFLite** | `--model x.tflite` (needs `ai-edge-litert`, see `requirements-optional.txt`) | ❌ skipped | ✅ with `tf2onnx`+`tensorflow` installed (float TFLite → ONNX → ORT INT8; output is ONNX) | ✅ LiteRT |

- **Scenario A (we get a PyTorch model + data)**: write an adapter (copy an existing one) whose `load_data`/`evaluate` use the organizers' data and metric, then `--model` points at their weights. The full pruning + quantization + sweep flow works.
- **Scenario B (we only get ONNX/TFLite)**: `sweep.py --model x.onnx`. Prune ratios > 0 are marked `skipped` (not counted as failures), and quantization + measurement still run. Before we have time to write an adapter, `measure.py --model x --input-shape ...` measures resources only (accuracy stays empty, the input is random data).
- **Scenario C (we only get a task and data)**: use or adapt a built-in adapter (classification / anomaly / detection).
- A foreign model's input convention can differ from the adapter's data. **This is handled automatically in `core/runtime.py`**: NCHW↔NHWC transposition, fixed batch=1 (runs one sample at a time), int8/uint8 input quantization for TFLite. INT8 calibration gets the same handling.
- Without PyTorch weights you cannot prune. Only quantization and graph-level changes are possible. **Ask the organizers whether they provide PyTorch weights**; it decides the strategy.
- **`target_runtime` (reserved)**: the format you finally submit should decide the quantizer, the measuring runtime and the RAM baseline together. `onnx` is implemented. `tflite` (TFLite converter for INT8 + LiteRT for all measurements) waits until the organizers confirm the submission format. Setting it now raises `NotImplementedError` instead of being silently ignored.

---

## Platform compatibility

Verified by running, not by reading. Windows AMD64 and Linux x86_64 are both proven end-to-end
(full test suite, quantize, isolated probe, core pinning); Python 3.11 and 3.12 both work with no
code changes.

| | status |
|---|---|
| Windows AMD64, Python 3.11 | verified: full suite |
| Linux x86_64, Python 3.12 | verified: full suite, offline install from a vendored bundle |
| Linux aarch64 (ARM) | dependencies verified (every pin has an aarch64 wheel, nothing builds from source); **not run** |
| macOS arm64 | expected to work; **not run** |
| macOS x86_64 (Intel) | **unsupported** - torch, torchvision and onnxruntime publish no Intel-mac wheels at these pins, and no sdist |
| Alpine / musl | **unsupported** - onnxruntime publishes no musllinux wheel |
| Python 3.10 and older | **unsupported** - five pins set a 3.11 floor (onnxruntime, numpy, pandas, matplotlib, scikit-learn) |
| glibc older than 2.28 | **unsupported** - floor set by torch, torchvision, onnxruntime |

`core/cpu_topology.py` reads `cpu_capacity` first, which is the ARM big.LITTLE capacity file, so
core-class pinning is written for ARM as well as for Intel P/E cores; it falls back to no pinning
when the topology cannot be read. Known gap: nothing is cgroup-aware, so inside a container with a
restricted cpuset the probe sees the host's CPUs.

**Installing on Linux:** `pip install -r requirements.txt` pulls torch from PyPI, which on Linux
(including aarch64) drags in cuda-toolkit, cudnn, nccl, nvshmem and triton - roughly 1 GB of CUDA
this CPU-only toolchain never calls. `scripts/make_offline_bundle.py` pins the CPU index instead:
467 MiB total, 69 wheels, nothing built from source.

**Without PyTorch:** given a `.onnx`, INT8 quantization + measurement + scoring + the deliverable
folder all run on `onnxruntime + numpy + psutil` alone. Pruning, fine-tuning and torch->ONNX export
are the parts that need torch.

## Architecture compatibility (model families)

`tests/test_model_families.py` runs each family through prune → export → INT8 → ORT on every test run:

| Family | Example | Prune | INT8 | Notes |
|---|---|---|---|---|
| depthwise CNN | MobileNetV2 | ✅ | ✅ | the default classification baseline |
| residual CNN | ResNet18 | ✅ | ✅ | `configs/classification_resnet18.yaml` |
| conv + attention | MobileViT-XXS (timm) | ✅ | ⚠️ use dynamic MatMul INT8 | static INT8 breaks on its SiLU blocks (see below) |
| plain ViT | DeiT-tiny (timm) | ✅ | ✅ | attention pruned per head dim |
| detector (multi-output + NMS) | YOLOv8n / YOLO11n | ⚠️ see detection | ✅ with the output-tail rule | NMS stays outside the ONNX graph |
| time series | 1D-CNN autoencoder | ✅ | ✅ | `configs/anomaly.yaml` |

Two model-family quirks are handled automatically: ultralytics' `C2f`/`C3k2` blocks are rewritten to an exactly equivalent chunk-free form (`adapters/ultralytics_compat.SplitC2f`) because torch-pruning cannot track `chunk()`, and YOLO11's conv-`qkv` attention is excluded from pruning (torch-pruning hangs on it).

---

## Pitfalls found the hard way (they cost accuracy, silently)

Each of these produced a wrong or misleading result here, and each now has a guard or a default that prevents it:

| Pitfall | What happened | Now |
|---|---|---|
| **P/E-core scheduling** | The same model measured 0.9 ms on a P-core and 2.4 ms on an E-core, making pruning look 4× faster than it was | measurements are pinned to the fastest core class (`measure.affinity`) |
| **`round_to` over-pruning** | torch-pruning rounds kept channels *down*: "5%" removed 50% of 16-channel layers, top-1 0.684 → 0.117 | kept channels round *up*; `max_layer_cut` is reported |
| **No BN re-calibration** | Pruning invalidates BN running statistics: 25% pruning gave 0.099 top-1 | BN re-estimated on calibration data by default (0.099 → 0.657) |
| **INT8 on mixed-range outputs** | A detector's output concat (pixels + 0..1 scores) shared one scale: mAP 0.4095 → 0.1608 | the output tail stays FP32 by default (→ 0.3852, 0.4006 with percentile — **both on coco128**; on held-out val2017 minmax is the better of the two, see *Calibration*) |
| **INT8 on channel-imbalanced SiLU blocks** | MobileViT static INT8: 0.9165 → 0.668, and *slower* than FP32 | `int8_dyn_mm` variant (0.9161), or `auto_keep_fp32: search` |
| **"entropy" calibration** | ORT's default 128/128 bins make entropy silently identical to minmax | real 2048-bin entropy; percentile exposed |
| **Fine-tune LR too high** | lr 2e-4 on 64 images dragged the *unpruned* detector from 0.407 to 0.28 | detection default 2e-5 (control: 0.3951 after 3 epochs) |
| **Unverifiable INT8 damage** | Nothing warned that an INT8 model was broken until a metric was computed | every variant reports label-free `int8_sqnr_db` (< ~15 dB = broken) |
| **The score picked a bad model** | Equal weights make accuracy 25% of the score, so MobileViT's 50%-pruned variant (0.717 vs 0.917 top-1) ranked **first** | hard `max_accuracy_drop` before ranking + a candidate shortlist instead of one winner |
| **Speed-up quoted at one thread** | INT8 is 1.67× faster at 1 thread but 1.12× at 4 on MobileNetV2, and 0.66× (slower) for MobileViT static INT8 | 1T and 4T reported side by side; the UNO Q's 4 cores make 4T the target-like column |
| **Single-shot RAM numbers** | Peak RSS can be *higher* for INT8 on small models; one measurement cannot separate that from noise | `measure.repeats` (3) reports the median and the spread (`*_sd`), which showed the reversal is real (±0.05–0.17 MB) |
| **minmax calibration on an outlier-heavy model** | The 25%-pruned ResNet50's INT8 collapsed to top-1 0.468 (from its FP32 0.894) at 1.5 dB SQNR; `percentile 99.99` on the same model is **lossless** (0.8940, 13.6 dB). One config line, 42.6 accuracy points. Not predictable from the architecture: plain MobileNetV2 is fine on minmax (24.8 dB) | `calibrate_method: auto` quantizes with each candidate and keeps the best label-free per-channel SQNR |
| **`--quick` silently dropped every resource metric** | `primary_threads: 4` plus the `--quick` override `threads: [1]` left the un-suffixed columns NaN, which emptied `score_resource` and both the latency and RSS Pareto fronts - in the mode most likely to be used under time pressure | `primary_thread_count()` falls back to the nearest measured count and logs it; the CSV's `primary_threads` column records where the numbers came from |
| **A documented threshold nothing enforced** | "< ~15 dB = broken" existed only as prose and one column of a 70-column CSV, so a 1.5 dB model was produced and reported without comment | `quantize.sqnr_warn_db` (default 15, 0 disables) warns on the per-channel **minimum** - the mean hides a single destroyed channel group |
| **Training silently used the GPU** | PyTorch fine-tuning picks CUDA when a GPU is present, so the documented "`--quick` in < 5 min on CPU" had never actually been run CPU-only | `env_info()` records `torch_train_device`/`cuda_device`; verified: `--device cpu --quick` finishes in **52 s** |

---

## Metric definitions

| Column | How it is measured |
|---|---|
| `accuracy` | `adapter.evaluate(file)`. **Always evaluated on the deployed file** (.onnx / .tflite). `metric_name` records which metric it is (top1 / AUC / mAP). Empty when there is no adapter |
| `size_bytes` | model file size (including ONNX external data) |
| `params` | parameter count of the torch model (the INT8 version has the same count, only the bytes differ); ONNX-only files count Conv/Gemm/MatMul weights from the graph; empty for TFLite |
| `macs` | **[ultralytics-thop](https://github.com/ultralytics/thop)**, batch=1, multiply-accumulate count. Falls back to counting Conv/Gemm/MatMul directly from the ONNX graph when there is no torch model; empty for TFLite |
| `latency_p50/p90_ms[_tN]` | ONNX Runtime CPU EP (`.tflite` uses LiteRT), batch=1, warmup 10 then measure ≥100 runs (and ≥ `min_time_s` seconds); one column set per thread count (default 1 and 4). The unsuffixed columns = `primary_threads` (defaults to the first entry). **Pinned to fixed cores by default** (see below); `cpus_tN` records which CPUs were used |
| `cpu_util_pct[_tN]` | process CPU time / wall time over the measurement window (psutil; 100 = one full core). `cpu_util_pct_max` is the periodic sampling peak |
| `cpu_ms_per_inf[_tN]` | CPU time per inference (ms), the best proxy for energy |
| `peak_rss_mb[_tN]` | inference runs in an **isolated subprocess** that records max RSS (Windows `peak_wset` / Linux `ru_maxrss`), **minus the peak RSS of an "empty session" of the same runtime**. Includes the load-time transient. `peak_rss_total_mb` is the value before subtraction; `rss_steady_mb` is the RSS increase sampled every 2 ms after loading. The `rss_baseline` column records which baseline was used (see below) |
| `energy_j` | **Empty (NaN) unless a real measurement source is configured.** `measure.energy.provider`: `csv` (external power-meter results), `command` (runs your measurement script and reads the last number), `aihub` (reserved; AI Hub profiles **report no energy**) |

> ⚠️ **Latency, RAM and CPU measured on a dev machine are relative values.** Use them to compare versions, not as absolute UNO Q (Cortex-A53) numbers. Real device numbers need AI Hub profiling or on-device measurement.

> ⚠️ **Always say which thread count a speed-up refers to.** INT8's latency advantage shrinks as threads are added: on DeiT-tiny it is 1.8× at one thread but ~1.25× at four, and on MobileNetV2 4T it disappears entirely (0.496 vs 0.488 ms — a tie). The UNO Q has 4 Cortex-A53 cores, so the 4-thread column is the target-like one. Both are always reported (`latency_p50_ms_t1`, `latency_p50_ms_t4`).

> **The score uses `measure.primary_threads` (4 in the shipped configs).** The un-suffixed columns (`latency_p50_ms`, `peak_rss_mb`, `cpu_ms_per_inf`) are copies of that thread count's measurement, and they are what the score and the winner are computed from - so scoring matches the 4-core target instead of a single-thread run. `scripts/remeasure.py --rescore-only` re-derives them and re-scores without re-measuring.

> **Repeatability**: `measure.repeats` (3 in the shipped configs) runs each probe several times and reports the median plus the spread (`latency_p50_ms_sd_tN`, `peak_rss_mb_sd_tN`). Check the spread before quoting a small difference — peak RSS in particular can be *higher* for INT8 on small models (more QDQ nodes), and the sd column says whether that is real or noise.

**RAM baseline by runtime** (the runtime is decided from the file content: TFLite has `TFL3` at byte offset 4, so a wrong extension is still recognized):

| `rss_baseline` | When | Empty baseline |
|---|---|---|
| `empty_session` | ONNX | a single Identity-node ONNX model with the same input shape |
| `empty_model` | TFLite + TensorFlow installed | a one-op TFLite model with the same input shape/dtype (float: REVERSE_V2, int8: CAST; neither can be optimized away), generated in a subprocess and cached in `weights/_cache/` |
| `runtime_only` | TFLite without TensorFlow | LiteRT imported, no model loaded. **The interpreter's own overhead (~1 MB) counts toward the model.** In our test the same 3 KB model measured 1.28 MB, vs 0.25 MB with `empty_model` |

> **The model's RAM is not the toolchain's RAM.** Every `peak_rss_mb` above is the *model's*
> inference footprint in an isolated subprocess — the number that matters for deployment, and the
> largest measured here is 202 MB (ResNet50 FP32). *Producing* those variants costs far more:
> fine-tuning a mid-size CNN was measured at **~2.3 GB** in the trainer plus ~180 MB per dataloader
> worker. `host_peak_rss_mb` records it per variant. On a RAM-limited machine that is the binding
> constraint, not the optimised model — and the on-site sandbox spec is unknown, so lower
> `adapter_args.batch_size` / `num_workers` first if a run is killed.

> **Choosing the right RAM column**: `peak_rss_mb` is "the model's own footprint inside its runtime" (each runtime's overhead is subtracted separately). Use it to **compare versions within the same runtime**. To **compare ONNX vs TFLite**, or to match an official number that covers the whole process, use `peak_rss_total_mb`.

> ⚠️ **Hybrid CPUs (P-core + E-core) need core pinning.** On the i7-13620H the same model measured **0.9 ms on a P-core and 2.4 ms on an E-core**, a bigger gap than most compression gains. By default (`measure.affinity: auto`) the probe reads the core topology from the OS (Windows `GetSystemCpuSetInformation` / Linux sysfs), picks the highest-performance class, and puts one thread on each physical core (1T → CPU 0, 4T → CPU 0,2,4,6). `affinity: none` disables pinning; `[ids]` sets the cores by hand. The UNO Q (4× A53) is not a hybrid CPU, so this issue does not apply there.

---

## Example results

Measured on the dev machine: i7-13620H, Windows 11, ORT 1.30 CPU EP, **pinned to P-cores**. **Relative values.**
(Earlier versions of these tables had no core pinning, and their latencies were distorted by P/E-core scheduling. They were re-measured with `scripts/remeasure.py`.)

### Phase 1: classification baseline (`results/classification/baseline.csv`)

MobileNetV2 (ImageNet init, stride-1 stem), CIFAR-10 32×32, 15 epochs:

| variant | top1 | size | params | MACs | p50 latency 1T / 4T | peak RSS | CPU% (1T) | energy_j |
|---|---|---|---|---|---|---|---|---|
| baseline_fp32 | 0.9234 | 8.92 MB | 2.24 M | 26.6 M | 0.92 / 0.53 ms | 22.5 MB | 93 | (empty: no measurement source) |

### Phase 2: classification sweep (`results/classification/sweep.csv`, `pareto_*.png`)

`python scripts/sweep.py --config configs/classification.yaml`: 3 prune ratios × {fp32, int8}, 3 fine-tune epochs after pruning:

| variant | top1 | size | MACs | p50 1T / 4T | peak RSS | CPU ms/inf | score* | eligible** |
|---|---|---|---|---|---|---|---|---|
| p00_fp32 | 0.9234 | 8.51 MB | 26.6 M | 0.99 / 0.54 ms | 23.5 MB | 0.97 | 0.333 | yes |
| p00_int8 | 0.9220 | 2.50 MB | 26.6 M | 0.59 / 0.48 ms | 7.2 MB | 0.60 | **0.765** | yes |
| p25_fp32 | 0.8705 | 4.88 MB | 17.0 M | 0.64 / 0.43 ms | 12.9 MB | 0.64 | 0.558 | NO (−5.3 pt) |
| p25_int8 | 0.8715 | 1.53 MB | 17.0 M | 0.48 / 0.42 ms | 6.2 MB | 0.51 | 0.779 | NO (−5.2 pt) |
| p50_fp32 | 0.7463 | 2.25 MB | 8.1 M | 0.36 / 0.34 ms | 5.8 MB | 0.37 | 0.602 | NO |
| p50_int8 | 0.7435 | 825 KB | 8.1 M | 0.36 / 0.33 ms | 4.8 MB | 0.39 | 0.664 | NO |

\* Equal-weight placeholder score (energy has no data, so its weight was dropped). **The official formula is unknown.**
\*\* `eligible` = passes `score.max_accuracy_drop` (0.02). Ineligible variants are never proposed, even when they score higher — `p25_int8` scores 0.816 but costs 5.2 accuracy points.

Takeaways:
- **INT8 (U8S8) is nearly free**: accuracy drops 0.14 pt, size shrinks 3.4×, RSS drops 3.3×, latency **1.67× faster at 1 thread but only 1.12× at 4 threads**.
- **25% pruning + INT8 keeps 87.2%** at 1.53 MB (5.6× smaller) — usable only after the `round_to` and BN re-calibration fixes (0.845 before). With a 0.02 accuracy budget it is still out of bounds; raise `max_accuracy_drop`, or fine-tune longer, if a 5-point drop is acceptable on site.
- At 50% pruning the model is so small that INT8 stops speeding it up: QDQ overhead cancels the gain.

### Phase 3: anomaly detection sweep (`results/anomaly/`)

`python scripts/sweep.py --config configs/anomaly.yaml`. **Same command, only the config changes; not one line of `core/` was modified.**
1D-CNN autoencoder (channels [32, 64]); anomaly score = reconstruction MSE; metric = ROC AUC. Data: ECG5000; training uses 70% of the normal samples only; the test set = the remaining normal samples + all anomalies.

| variant | AUC | size | MACs | p50 1T / 4T | peak RSS | CPU ms/inf | score | eligible |
|---|---|---|---|---|---|---|---|---|
| p00_fp32 | 0.9817 | 193.5 KB | 3.04 M | 0.16 / 0.14 ms | 0.98 MB | 0.16 | 0.360 | yes |
| p00_int8 | 0.9815 | 62.3 KB | 3.04 M | 0.08 / 0.13 ms | 1.11 MB | 0.09 | 0.587 | yes |
| p25_fp32 | 0.9770 | 110.6 KB | 1.73 M | 0.11 / 0.11 ms | 0.81 MB | 0.11 | 0.552 | yes |
| p25_int8 | 0.9774 | 40.6 KB | 1.73 M | 0.07 / 0.11 ms | 1.03 MB | 0.07 | 0.643 | yes |
| p50_fp32 | 0.9698 | 51.2 KB | 0.79 M | 0.08 / 0.07 ms | 0.68 MB | 0.08 | **0.687** | yes |
| p50_int8 | 0.9699 | 24.7 KB | 0.79 M | 0.06 / 0.09 ms | 1.03 MB | 0.06 | 0.635 | yes |
| p75_fp32 | 0.9527 | 15.2 KB | 0.21 M | 0.05 / 0.05 ms | 0.34 MB | 0.05 | 0.669 | NO (−2.9 pt) |
| p75_int8 | 0.9524 | 15.2 KB | 0.21 M | 0.06 / 0.07 ms | 0.77 MB | 0.06 | 0.537 | NO |

At **4 threads** (the scoring setting, matching the UNO Q's 4 cores) the best eligible variant is `p50_fp32`, not an INT8 one: INT8's latency edge disappears here (0.086 vs 0.070 ms) and it costs more RAM, so 50% pruning at FP32 wins. At 1 thread the same data picks `p25_int8` - which is exactly why `primary_threads` matters.

Takeaways: **on very small models INT8 stops paying off**. With QDQ, per-channel scales, zero points and int32 biases are extra cost, so p75_int8 is no smaller than fp32 and its RSS is actually higher. The sweep and score pick this up automatically, so there is no need to assume "INT8 always wins."

At 75% pruning the model is so small that INT8 no longer shrinks it: the per-channel scales, zero points and int32 biases cost as much as the weights save (and the output-tail rule disables itself there — every conv is below its threshold — which `int8_quantized_nodes` in the CSV confirms). The sweep and the score pick this up, so there is no need to assume "INT8 always wins".

Using your own data on site: set `adapter_args.source: npz` + `npz_path`. The npz contains `X_train` (normal samples only), `X_test`, `y_test` (1 = anomaly), shaped `[N, L]` or `[N, C, L]`.
**MIMII**: not built in (the dataset is tens of GB). Convert the audio to log-mel frames `[N, n_mels, T]` and feed them through the npz path, with `C = n_mels`.

### Phase 4: object detection sweep (`results/detection/`)

YOLOv8n (COCO-pretrained). **Validation: 500 COCO val2017 images the model never trained on**
(`scripts/fetch_coco_val.py --n 500`); fine-tuning/calibration still uses the 128 coco128 images.
Metric: COCO mAP@[.5:.95]. The sweep also varies the **input resolution** (`sweep.adapter_variants`):

| variant | mAP50-95 | size | MACs | p50 1T / 4T | peak RSS | SQNR | score | eligible |
|---|---|---|---|---|---|---|---|---|
| 320_p00_fp32 | 0.3057 | 12.11 MB | 1.11 G | 29.2 / 9.92 ms | 26.2 MB | – | 0.339 | yes |
| **320_p00_int8** | 0.3010 | 3.38 MB | 1.11 G | 13.8 / 7.30 ms | 17.3 MB | 17.9 dB | **0.818** | yes |
| 320_p00_int8_pct | 0.2988 | 3.38 MB | 1.11 G | 14.4 / 7.92 ms | 17.5 MB | 19.2 dB | 0.796 | yes |
| 256_p00_fp32 | 0.2649 | 12.10 MB | 0.71 G | 18.6 / 6.64 ms | 26.4 MB | – | 0.318 | NO (−4.1 pt) |
| 256_p00_int8 | 0.2595 | 3.37 MB | 0.71 G | 9.54 / 5.50 ms | 15.4 MB | 15.9 dB | 0.768 | NO |
| 256_p00_int8_pct | 0.2585 | 3.37 MB | 0.71 G | 8.46 / 5.05 ms | 15.1 MB | 17.6 dB | 0.779 | NO |
| 192_p00_fp32 | 0.2040 | 12.09 MB | 0.40 G | 11.0 / 3.93 ms | 26.8 MB | – | 0.210 | NO (−10.2 pt) |
| 192_p00_int8 | 0.1976 | 3.36 MB | 0.40 G | 4.99 / 3.42 ms | 13.8 MB | 14.3 dB | 0.656 | NO |
| 192_p00_int8_pct | 0.1980 | 3.36 MB | 0.40 G | 4.81 / 2.93 ms | 14.2 MB | 16.7 dB | 0.664 | NO |

Two results worth keeping:

**1. The old coco128 number was inflated by 0.10 mAP.** The same FP32 model scored 0.4095 on the coco128
split and scores **0.3057** on unseen val2017 images — coco128 is a slice of COCO *train2017*, so the
pretrained detector had already seen it. Everything below is measured on the unseen set (±0.005 at 500
images).

**2. INT8 is cheap; lowering the resolution is not.** On this model INT8 costs **0.005 mAP** for 3.6×
smaller and 26% less 4-thread latency, while dropping 320 → 256 px saves 36% of the MACs but costs
**0.041 mAP** (13% relative), and 192 px costs 0.102. So resolution is a real lever, but a blunt one:
under a 2.5%-relative accuracy budget every reduced-resolution variant is rejected. It is worth keeping in
the sweep precisely because the trade-off is now measured instead of assumed — with a looser official
accuracy budget, 256 px + INT8 would be the fastest eligible point (5.05 ms vs 7.30 ms at 4 threads).

### Attention models: which INT8 works (`results/classification_mobilevit/`, `results/classification_deit/`)

Unpruned baselines, CIFAR-10 (MobileViT at 64 px, DeiT-tiny at 32 px with 4×4 patches):

| model | variant | top-1 | size | compression | p50 1T / 4T | peak RSS | SQNR | eligible |
|---|---|---|---|---|---|---|---|---|
| MobileViT-XXS | fp32 | 0.9165 | 3.83 MB | 1.00x | 1.46 / 1.41 ms | 9.99 MB | – | yes |
| | int8 (static) | 0.6680 | 1.71 MB | 2.24x | 1.58 / 2.13 ms | 15.6 MB | **3.0 dB** | NO (−24.9 pt) |
| | **int8_dyn_mm** | **0.9160** | 2.58 MB | 1.49x | 1.33 / 1.35 ms | 10.2 MB | 38.5 dB | yes |
| | **int8_search_pct** | 0.9122 | 2.22 MB | **1.72x** | 1.48 / 2.18 ms | 14.9 MB | 19.7 dB | yes |
| DeiT-tiny | fp32 | 0.9019 | 20.58 MB | 1.00x | 11.79 / 4.95 ms | 27.1 MB | – | yes |
| | int8 (static) | 0.8987 | 5.75 MB | 3.58x | 6.70 / 4.76 ms | 19.5 MB | 30+ dB | yes |
| | **int8_dyn_mm** | **0.9032** | 5.60 MB | 3.68x | 5.38 / 3.61 ms | 13.1 MB | 40+ dB | yes |

Same family (attention), opposite outcome: **static INT8 is fine on DeiT and broken on MobileViT**, because
MobileViT's damage comes from its SiLU inverted-residual convs, not from attention. `int8_sqnr_db` separates
the two before any labelled evaluation (3.0 dB vs 30+ dB).

**MobileViT is the weak case for compression**, and two options survive the accuracy budget:
`int8_dyn_mm` keeps only MatMul/Gemm in INT8 (1.49x, fastest, best score) while `int8_search_pct`
(percentile calibration + the label-free block search, which keeps the 7 worst SiLU blocks in FP32) reaches
**1.72x for −0.43 pt** but is ~1.6x slower at 4 threads and needs more RAM. Both are shortlisted; the score
picks `dyn_mm`, and a compression-weighted official formula would pick the other. Pushing further breaks the
budget: static INT8 with percentile alone is 2.24x at −5.2 pt, and dynamic INT8 that also covers Conv gives
2.82x but collapses to 0.513 — the same per-tensor activation problem, so it is not a way out.

Note the peak-RSS reversal: MobileViT **static INT8 needs more RAM than FP32** (15.6 vs 9.9 MB, ±0.05–0.17 over 3 runs, so it is real) because its QDQ graph adds ~400 Quantize/Dequantize nodes; `int8_dyn_mm` keeps RAM at the FP32 level. INT8 lowers RSS when weights dominate (MobileNetV2: 23.5 → 7.2 MB) and raises it when node overhead does (anomaly p50: 0.43 → 0.80 MB).

Same family (attention), opposite outcome: **static INT8 is fine on DeiT and broken on MobileViT**, because MobileViT's damage comes from its SiLU inverted-residual convs, not from attention. `int8_sqnr_db` separates the two cases before any labelled evaluation — 3.0 dB vs 30+ dB. `int8_dyn_mm` (dynamic INT8 on MatMul/Gemm only) is the safe default for attention models: it was the best variant for both, and the fastest for DeiT.

---

## Quantization / pruning / sweep / score

### Quantization `core/quantize.py` (wraps ONNX Runtime; no algorithm of our own)

**Which nodes stay FP32 — `auto_keep_fp32` (default `rules`)**. INT8 does not fail uniformly; it fails at specific places:
- **R1 output tail**: every node between the graph outputs and the last weighted op (Conv/Gemm/MatMul with ≥ `tail_min_params` weights) stays FP32. Those nodes carry almost no MACs, so it is nearly free. This is what makes detector INT8 usable: the output concatenates boxes in pixels (0–390) with scores in 0–1, and one shared INT8 scale (1.53) turned every score below 0.77 into 0 — mAP 0.4095 → 0.1608. With the rule: 0.3852 (0.4006 with percentile calibration) at the same size and speed. Both of those are coco128 numbers; re-measured on held-out val2017 the ordering flips (minmax 0.3010 vs percentile 0.2988), which is why `calib_auto_tie_db` now defaults to 0 — see *Calibration*.
- **R2 Concat range mismatch**: any Concat whose calibrated inputs differ in magnitude by more than `concat_range_ratio` (32×) stays FP32 — the general form of the same trap.
- `auto_keep_fp32: search` adds a **label-free** greedy search: blocks (module-path prefixes) are ranked by how much quantizing only that block hurts the output SQNR, and the worst ones are kept FP32 until `search_target_db` is reached. It needs no labels and no metric, so it works for any task (detector: 20.2 dB / mAP 0.3965 in 23 s).
- `keep_fp32` is still available for manual control, but **exclude contiguous blocks** (module prefixes). Excluding a single op or a whole op type usually changes nothing, because neighbouring nodes still quantize the shared tensors.

**Calibration — `calibrate_method` (default `auto`)**. The calibrator choice is not a detail: on the
25%-pruned ResNet50 it is worth 42.6 accuracy points.

| calibration | SQNR (mean / per-channel min) | top-1 | vs its FP32 (0.8940) |
|---|---|---|---|
| `minmax` | 1.5 / −1.5 dB | 0.4680 | **−0.4260** |
| `percentile` 99.99 | 13.6 / 10.0 dB | **0.8940** | **0.0000** (lossless) |
| `percentile` 99.9 | 8.4 / 6.3 dB | 0.8910 | −0.0031 |
| `entropy` | 1.0 / 0.5 dB | — | worst of the four |

`minmax` puts the outliers *inside* the range, and after pruning the outliers get worse, so every other
channel is crushed toward zero. Which models behave this way is **not** predictable from the architecture —
unpruned MobileNetV2 is perfectly happy on minmax (24.8 dB). And on site there are no labels to check
against. So `calibrate_method: auto` measures instead of guessing: it quantizes with each candidate in
`calib_auto_methods`, ranks them by the label-free **per-channel minimum** SQNR, keeps the winner's file,
and stops at the first candidate above `calib_auto_target_db` (15 dB, the same "not broken" floor as
`sqnr_warn_db`). The chosen method and the whole trail land in the CSV as `calib_method_chosen` and
`calib_auto_tried`. Because the SQNR probe already runs for every INT8 variant, a model whose first
candidate passes pays only one extra file copy; the worst case is one calibration pass per candidate.
Pin a fixed method to reproduce a result exactly.

Note the ordering above also validates the mechanism: ranking by SQNR (13.6 > 8.4 > 1.5) reproduced the
ranking by accuracy (0.8940 > 0.8910 > 0.4680) without using a single label.

Ranges are computed once, batch by batch (bounded memory), cached, and reused for every exclusion set — which is what makes the search affordable. `entropy` now uses 2048 histogram bins: ORT's `quantize_static` default (128 bins / 128 quantized bins) leaves a single threshold candidate, so "entropy" silently behaved exactly like minmax. `percentile` (`calib_percentile`, default 99.99) is often the best choice for SiLU/Swish networks. More minmax calibration data can make results *worse* (wider outlier ranges).

**Every variant reports `int8_sqnr_db`** — mean per-output-channel signal-to-quantization-noise on calibration data, label-free. It is computed per channel so a small-valued channel group (scores) cannot hide behind a large one (pixels). Below ~15 dB the INT8 model is broken: the detector scored 4.3 dB before the fix and 17.7–20.2 dB after. Use it on site to catch a bad INT8 conversion without waiting for a metric.

- `method: static`: QDQ INT8, per-channel weights, activations calibrated on `adapter.load_data()["calib"]` (`calib_batches` batches). Runs `quant_pre_process` first.
- `method: dynamic`: weights only. **CNNs get much slower with it** (44 ms vs 2.5 ms in our test, accuracy 0.80). Use it only for Linear/RNN-heavy models.
- `activation_type`: **defaults to `uint8` (U8S8)**. Measured on our x86: S8S8 QDQ *is slower than FP32* (2.86 vs 2.52 ms), U8S8 is 35% faster (1.63 ms), same accuracy. ARM supports both; re-measure on site.
- `keep_fp32: ["features.0", "re:classifier.*"]`: node-name substrings (or `re:` regexes) to keep in FP32. ONNX node names come from the torch module names. `keep_fp32_op_types` keeps whole op types in FP32.
- Any entry in `sweep.precisions` can be a dict that overrides quantization settings for that one variant, e.g. `{name: int8_pct, calibrate_method: percentile}` or `{name: int8_dyn_mm, method: dynamic, op_types_to_quantize: [MatMul, Gemm]}`.

### Pruning `core/prune.py`
- **Structured channel pruning only** (torch-pruning `MetaPruner`, which handles residual/depthwise/concat dependencies automatically). Settings: `importance` (l1/l2/random/bn_scale), `round_to`, `global_pruning`, `iterative_steps`, `attention_mode`.
- Layers from `adapter.ignored_layers()` are never pruned. After pruning, BN is re-calibrated and `adapter.finetune()` runs (or KD with `finetune_mode: distill`).
- **`round_to` never prunes more than the requested ratio.** torch-pruning rounds the *kept* channel count **down** to a multiple of `round_to`, which turns a 5% ratio into a 50% cut on a 16-channel layer. Kept channels are now rounded **up** instead, and each run logs (and reports in `max_layer_cut`) the largest fraction any single layer actually lost.
- **BatchNorm is re-calibrated after pruning** (`prune.bn_recalibrate`, on by default): running statistics are re-estimated on unlabelled calibration batches, because pruning removes channels whose mean activation every downstream BN was calibrated against. Adapters can override which data to use with `bn_calibration_loader()` — the detection adapter uses the *pretraining* resolution (640 px), since BN statistics are resolution-dependent.
- **Attention**: blocks with a fused `qkv` Linear and `num_heads` are pruned per head dim, and `num_heads`/`head_dim`/`attn_dim`/`scale` are kept in sync (only attributes change, so pickling and deepcopy keep working). Attention of an unsupported form (e.g. conv-`qkv`, which made torch-pruning hang) is excluded automatically and logged.
- **Pruning is exact, and there is a test that proves it**: `tests/test_prune_fixes.py` zeroes exactly the channels the pruner selects, prunes for real, and requires identical outputs. A producer/consumer index mismatch keeps the shapes valid but changes the numbers, and this catches it (verified on MobileNetV2, ResNet18, YOLOv8n and a toy net).

#### Pruning recovery: what actually limits it

Structured pruning compresses exactly as the arithmetic says. Pruning **alone** at 25%, measured on
four unrelated architectures:

| model | params | pruning-only size |
|---|---|---|
| MobileNetV2 | 2.24M → 1.28M | 1.74× |
| DeiT-tiny | 5.36M → 3.03M | 1.76× |
| 1D autoencoder | 0.05M → 0.03M | 1.75× |
| ResNet50 | 23.53M → 13.25M | 1.78× |

All four land on (1−r)², so the mechanism is sound. What decides whether the variant is *usable* is
recovery, and it splits cleanly by **capacity**, not by task:

| model | params | 25% pruning | verdict |
|---|---|---|---|
| DeiT-tiny | 5.36M | −0.64 pt | ✅ eligible → **6.22×** combined with dynamic INT8 |
| 1D autoencoder | 0.05M | −0.48 pt (−1.19 at 50%) | ✅ eligible → **7.84×** at 50% + INT8 |
| MobileNetV2 | 2.24M | −5.29 pt | ✗ |
| ResNet50 | 23.5M | −9.53 pt | ✗ (fine-tuned 1 epoch only) |
| YOLOv8n | 3.16M | −13.02 pt at 20% | ✗ |

**Two hypotheses for the failures were tested and both were wrong.** Round 1 gave the detector 5000
real training images and 10 fine-tune epochs (78× the data of coco128): 20% pruning still scored
0.1755 against a 0.3057 baseline, and the loss had *converged* (4.286 → 4.282 over the last epoch),
so it was not short of epochs. Round 2 raised the recovery LR 20× (2e-5 → 4e-4) and added 10%
warmup: 0.1728 — no change. So it is neither the epoch count nor the learning rate. What remains is
the loop itself (no mosaic, no EMA, no long schedule), which means the fix is the upstream trainer,
not a knob. `prune.recovery_lr_mult` and `finetune_warmup_frac` exist and default to no-ops.

**`round_to: 8` matters for latency, independently of accuracy.** Without it 54% of the detector's
conv layers ended at channel counts that are not multiples of 8 (115, 57, 51, …) and 10–20% pruning
made the model *slower* than the unpruned baseline — 11.96 ms and 10.26 ms against 9.36 ms, with
spreads of only 0.06–0.09 ms, so not noise. With `round_to: 8` the same 20% cut runs at 8.55 ms, 7%
faster than baseline.

**Current strategy**: quantization is the lever that works everywhere; pruning is worth trying when
the model has spare capacity (a large backbone, or a transformer) and is not worth the time on an
already-compact one.

#### `prune.mode: unstructured`

Magnitude pruning via `torch.nn.utils.prune`, available because "compression" may be scored on file
size. It zeroes individual weights, so shapes — and therefore latency and peak RAM — are unchanged:
on a CPU runtime without sparse kernels there is nothing to exploit. The benefit shows up only in
`size_bytes_gz` (90% sparsity on a small model: `size_bytes` identical, gzipped 4.25× smaller).
Structured stays the default because it is the only one of the two that moves latency and RAM.

The mask is deliberately kept live *through* fine-tuning and baked in afterwards. Baking it in first
silently undoes the pruning — every zeroed weight gets a non-zero gradient and drifts off zero;
measured, asking for 50/80/90% sparsity that way left 0.11/0.25/2.36% actual zeros in the export.

### Object detection `adapters/detection.py`
- ultralytics YOLOv8n / YOLO11n on any **YOLO-format** folder (`images/<split>` + `labels/<split>` with `cls cx cy w h`); default data is coco128 (auto-download). Metric: COCO mAP@[.5:.95] via pycocotools (cross-checked against ultralytics' own validator: the 0.02 difference at 256 px is explained by multi-label NMS, the resize filter and the AP implementation, not by an evaluator bug).
- **The ONNX graph stops at the decoded boxes + scores; NMS runs in `evaluate()`** (torchvision `batched_nms`), as it would in the deployed app. That keeps the graph plain conv ops and quantizable.
- **Pruning is off by default *for this dataset*, not as a rule.** The blocker is the recovery budget: coco128 leaves 64 training images, and channel pruning needs real retraining (the official Torch-Pruning YOLOv8 recipe uses ~160 epochs on full COCO). Measured here with every fix applied — protected early blocks, BN re-calibration at 640 px, lr 2e-5 — 10% pruning gives mAP 0.17 and 20% gives 0.05, and fine-tuning on 64 images recovers ~0.01. **On site with a real training set, turn it back on** (`prune_ratios: [0.0, 0.1, 0.2]`, more fine-tune epochs): pruning is the only lever that cuts MACs, which INT8 does not touch.
- Fine-tuning uses lr 2e-5: at 2e-4 the *unpruned* model fell from 0.407 to 0.28 on this data.

### Sweep `core/sweep.py`
- `prune_ratios × precisions`. Each prune ratio is pruned and fine-tuned only once; its fp32 and int8 variants share that model.
- A failed variant is recorded in the `error`/`traceback` columns and the sweep keeps going. CSV rows are written as soon as each variant finishes. `--resume` skips variants that already succeeded.
- Outputs: `sweep.csv` (including `score_*` and `pareto_<x>` columns), `pareto_<x>.png` (x set by `pareto_x`), `summary.json` (environment + config).

### Total score `core/score.py`
- `groups` defines which columns make up each official criterion. Each column is normalized (`minmax` across variants, or `baseline` as a ratio vs. the unpruned fp32 model), averaged within its group, then combined with `weights`.
- **Defaults are equal weights and marked "official formula unknown."** A group with no data (for example energy) is dropped and the remaining weights are renormalized; `score_note` records this.
- **A weighted score is not a selector.** With equal weights accuracy is only 25% of it, so a variant that lost 20 accuracy points can rank first for being small and fast (MobileViT's 50%-pruned variant: top-1 0.717 vs 0.916 — it ranked #1 before this guard existed). Two rules prevent that:
  - **`max_accuracy_drop` (absolute metric points, default 0.02), `max_accuracy_drop_rel` (a fraction of the baseline) and `min_accuracy` (absolute floor) are hard constraints applied before ranking; when several are set the strictest wins.** Pick the form that matches the metric's scale: 0.02 absolute is 2% of a 0.92 top-1 but ~5% of a 0.41 mAP, which is why the detection config uses `max_accuracy_drop: 0.01` plus `max_accuracy_drop_rel: 0.025`. Failing variants get `eligible = False` plus a reason, are excluded from "best", and are never proposed — whatever their score. They stay in the CSV so the trade-off remains visible.
  - **`candidates()` returns a shortlist, not one winner**: the top-scoring eligible variants plus every eligible variant on a Pareto front (best accuracy for its size / latency / RAM), each with a `why`. `scripts/sweep.py` prints it and `summary.json` stores it; a human picks the operating point.

### Deliverable `scripts/compress.py`
`python scripts/compress.py --config configs/<task>.yaml [--model given.pt] [--reuse] [--max-acc-drop 0.01]`
runs (or reuses) the sweep, applies the accuracy constraint, picks the highest-scoring **eligible** variant
and writes `deliverables/<run>/`:

| file | what it is |
|---|---|
| `best.onnx` | the model to submit |
| `best_fp32.onnx` | the same compression stage before INT8 - convert from this if another format is required |
| `best_model.pt` | the pruned PyTorch model, when pruning produced one |
| `REPORT.md` | one page: chosen variant, numbers vs the baseline (1T **and** 4T latency), why, what was rejected and why, how the numbers were measured, how to reproduce |
| `metadata.json`, `candidates.csv`, `sweep.csv` | machine-readable result, the shortlist, the full grid |

It refuses to write anything when no variant passes the constraint, and the report names every higher-scoring variant it rejected - so an over-compressed model cannot be handed in by accident.

### Qualcomm AI Hub (optional) `core/aihub.py`, `scripts/aihub_profile.py`
- `pip install -r requirements-optional.txt` (qai-hub 0.55.0; API checked against the official docs and source code, 2026-09).
- Token: read from the **environment variable `QAIHUB_API_TOKEN`** and passed in memory via `hub.Client(hub.ClientConfig(api_token=...))`. It is never written to code, config or git. If the variable is not set, the `~/.qai_hub/client.ini` written by `qai-hub configure` is used.
- **AI Hub has no UNO Q / QRB2210 device** (2026-09). `aihub.devices` in the YAML is a preference list and the first available device is used: `chipset:qualcomm-snapdragon-429` (also 4× Cortex-A53, the closest CPU) → `Dragonwing RB3 Gen 2 Vision Kit` → `QCS8550 (Proxy)`. The latter two are much faster than an A53, so their latency is optimistic.
- AI Hub profiles report latency (μs) and peak memory (bytes). **They do not report energy.**
- ⚠️ **Never run against a real account — the largest unverified part of this toolchain.** `python scripts/aihub_profile.py --check` verifies everything that does not need the network (package, token, device list, models ready to upload) and prints what is missing; with a token the real run is `--list-devices` and then `--config configs/<task>.yaml`.

---

## Known gap: only the ONNX Runtime path is complete

Everything measured in this repo runs through **ONNX Runtime on CPU**. If the competition sandbox scores
models on a different runtime, part of the toolchain has to be adapted, and the numbers above would have to
be re-measured there (`scripts/remeasure.py` does that without retraining).

| Path | State | What is missing |
|---|---|---|
| ONNX Runtime (CPU) | **complete** — export, INT8, measurement, scoring, deliverable | — |
| TFLite / LiteRT | measurement and evaluation work; `target_runtime: tflite` raises `NotImplementedError` | INT8 through the TFLite converter, an empty-model RSS baseline for it, and ONNX→TFLite export of the chosen variant (`onnx2tf`) |
| Qualcomm AI Hub | wrapper written against the documented API, device preference list in the YAML | **never run against a real account**; no UNO Q / QRB2210 device exists there, and it reports no energy |
| On-device (UNO Q / any Cortex-A board) | the probe is plain Python + ORT and would run there | a board, plus a USB power meter for energy |

Contingency if the sandbox is not ORT: keep FP32 ONNX as the interchange format (every deliverable ships
`best_fp32.onnx`), convert that to the required format, and re-run the *measurement* step in their runtime.
Compression itself (pruning, fine-tuning, distillation) is runtime-independent.

## Status

- [x] Phase 1: skeleton + measurement + classification adapter
- [x] Phase 2: quantization + pruning + sweep (+ optional AI Hub wrapper)
- [x] Phase 3: anomaly detection adapter (ECG5000 / npz, 1D-CNN AE, AUC)
- [x] Phase 4: object detection adapter (YOLOv8n/11 on YOLO-format data, COCO mAP, NMS outside the graph)
- [ ] Phase 5: distillation (`core/distill.py` exists, usable as `finetune_mode: distill`) + `scripts/report.py`
