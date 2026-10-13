What INT8 Post-Training Quantization Buys on a CPU: Runtime Version Moves the Speedup, Model Class Picks the Method

ABSTRACT: INT8 post-training quantization is judged by its speedup, size and accuracy, measured for a method applied to a model. We show that on a CPU the runtime release belongs in that description. On an x86 laptop CPU with AVX-VNNI, one set of byte-identical INT8 files was run on eleven ONNX Runtime releases. Release 1.24.4 made the static INT8 file of the hybrid MobileViT-XXS 31% slower at four threads on an identical executed graph and, in a second experiment, made the static file of YOLOv8n 28% faster. We trace the regression to four eight-bit Transpose operators, which reproduce it when timed alone, and to one routing change in the runtime, which sends them to a multi-threaded path and adds 832 thread-pool dispatches to every four-thread inference. At a fixed release the model class picks the method: static INT8 is 1.6-3.2x faster than FP32 on convolutional models at one thread, where dynamic INT8 is more than 12x slower, and dynamic INT8 wins on the attention and hybrid models. Against those gains the release is a first-order variable: an INT8 speedup is reproducible only together with its runtime version.

INDEX TERMS: Post-Training Quantization, INT8 Inference, CPU Inference, ONNX Runtime, Performance Regression, Reproducibility

## I. Introduction

Run one INT8 model file on two releases of ONNX Runtime [1], one minor version apart, on the same laptop CPU with the same four threads. On release 1.23.2 the statically quantized MobileViT-XXS file runs at the speed of its FP32 original; on 1.24.4 the same bytes run 31% slower (Fig. 1). The runtime builds the same executed graph on both releases, and the FP32 and dynamically quantized versions of the model do not move.

Post-training INT8 quantization (PTQ) is the standard way to make a trained network cheaper on a CPU [2, 3], and its benefit is reported as the speedup, size and accuracy of a method applied to a model. The runtime release that executes the file is not part of that report. A deployer who measures a speedup and later upgrades the runtime can ship a different speedup from the one measured: in the case above, an INT8 model slower than the FP32 model it was meant to replace.

We measure INT8 PTQ on an x86 laptop CPU with AVX-VNNI, the integer dot-product instructions that INT8 kernels rely on, for six models at one and four threads, and across eleven ONNX Runtime releases for one set of byte-identical files. The paper makes three contributions. First, it documents a whole-model INT8 regression on unchanged files over eleven releases; to our knowledge this is the first report of such a regression that separates the static and dynamic INT8 paths on the same bytes (Section IV). Second, it traces the regression to four eight-bit Transpose operators, which reproduce it when timed alone, and one routing change in the runtime, each step decided by a rule written down before the measurement (Section V); the same release makes another model faster, so the effect runs both ways (Section VI). Third, it measures what INT8 buys on this CPU when nothing changes, which sets the scale of the release effect and yields four recommendations for deployers (Section VII and Section IX).

## II. Background and Related Work

#### INT8 Quantization on CPUs.

Static quantization calibrates activation ranges offline; in the QDQ format, QuantizeLinear/DequantizeLinear pairs placed around each operator are fused by the runtime into integer kernels. Dynamic quantization stores INT8 weights and computes activation ranges at every inference. The runtime's documentation recommends static quantization for convolutional networks and dynamic quantization for transformers [4], a direction also observed on BERT-family models [5], MobileNetV2 [6], ResNet-18 [7] and vision transformers [8]. Per-channel PTQ keeps convolutional networks near FP32 accuracy [3, 9], while vision transformers and hybrids need dedicated methods [10, 11]. On x86, AVX-VNNI multiplies unsigned by signed bytes, so the fast path takes unsigned activations and signed weights (U8S8).

#### The Runtime as a Variable.

The framework tax [12] names the efficiency gap that the software stack, not the model, sets. DeltaNN varies framework, compiler and device for image recognition [13]; Pape et al. pin each of five LLM inference engines to one release [14], so the release is the variable they hold fixed. Weekly TFLite and ncnn builds on phones show commit-level latency cliffs [15]. In ONNX Runtime, release-to-release regressions are documented for single operators [16, 17] and, at 1.24, for FP32 convolutions on Apple M4 [18]. Shin [19] attributes the sign of the INT8 speedup to the CPU's dot-product instruction set, from one ResNet-50 on four CPUs under three releases, three of whose four cross-CPU contrasts also change the release. We hold the CPU fixed and vary the release.

#### Measuring Performance.

Compression results are hard to compare across papers [20], single-metric efficiency claims mislead [21], the best quantization method depends on the deployment backend [22], innocuous setup details bias measurements [23], and variability must be reported [24]. MLPerf Inference measures one long run per scenario [25]; because between-session variance dominates on our machine, we repeat isolated sessions in random order.

## III. Method

### A. Machine and Runtime

All measurements ran on an Intel Core i7-13620H laptop CPU with six performance cores (P-cores) and four efficiency cores, under Windows 11. It supports AVX2 and 256-bit AVX-VNNI but not AVX-512, AVX-512-VNNI or AMX, and ONNX Runtime's MLAS kernel library runs U8S8 matrix multiplication and convolution on its AVX-VNNI kernels. Inference used the CPU execution provider at batch 1 with one or four intra-op threads pinned to P-cores, one per physical core, because mixing core types makes latency bimodal. Execution was sequential, with all graph optimizations and the default spinning thread pool [26]. We call the graph that the runtime builds from a file and then runs the executed graph.

### B. Models and Files

Table I lists the six models: ResNet-50 [27], MobileNetV2 [28] and YOLOv8n [29]; a one-dimensional convolutional autoencoder trained on ECG5000 [30]; the attention model DeiT-tiny [31], with patch 4 at 32 px and 65 tokens; and the hybrid convolution-attention model MobileViT-XXS [32]. The data are Imagenette [33], a ten-class subset of ImageNet, CIFAR-10 [34], COCO val2017 [35] and ECG5000. Static files use QDQ with per-channel int8 weights and uint8 activations, calibrated offline with the method of best label-free signal-to-quantization-noise ratio (SQNR). Dynamic files quantize the operator types listed in Table I. All files come from the ONNX Runtime 1.30.0 quantizer at opset 17. Quality is the change against the FP32 file in points (pt) of top-1 accuracy, mAP or ROC AUC.

### C. Measurement

#### Comparing Releases.

Every release comparison runs each measurement in a fresh process: 1.5 s of warm-up, then at least 2 s and 40 iterations timed, reporting the median iteration latency. All measurements of one experiment are shuffled into a single random order under a logged seed, so release, model and variant are decorrelated from time and thermal drift, and each cell is the median of three sessions. A core-delivery ratio, CPU time per inference over thread count times wall time, flags sessions whose threads were off the CPU for part of the run; values are medians over all sessions unless marked otherwise, and recomputing them without flagged sessions changes no finding.

#### Comparing Methods within a Release.

The convolutional rows of Section VII measure each variant in three back-to-back processes in a fixed order and report their median. Reruns of byte-identical convolutional file pairs on 1.30.0 gave one-thread static speedups up to 13.6% apart (MobileNetV2 at 224 px: 1.52x and 1.73x).

#### Floors and Fixed Rules.

A latency difference below 7% is not interpreted: that is the largest gap observed between two files with identical executed graphs, and in the eleven-release comparison of Section IV a resampled gap between two medians of an unchanged file exceeds 3.3% in only 5% of draws. Quality differences are read against each task's resampling spread. Before each experiment of Section V we wrote the result that would confirm or reject its hypothesis into a dated log, which is part of the artifact.

## IV. One Release, Same Bytes, 31% Slower

Changing only the runtime release moved the INT8 speedup of an unchanged file from 0.976x on 1.23.2 to 0.631x on 1.25.1 at four threads (Fig. 1). One set of FP32, static and dynamic files of MobileViT-XXS and DeiT-tiny was loaded byte for byte by eleven releases, 1.16.3 to 1.30.0, in 396 measurements (Table II).

The fall has two causes, one release apart. At 1.24.4 the static file itself became 31% slower, 1.553 to 2.037 ms, and every session of 1.24.4 and later was slower than every session up to 1.23.2. At 1.25.1 the static file did not change while the latency of its FP32 sibling fell 15%, consistent with the layout changes of 1.25.0 [36], which the static file's integer kernels do not use; the same release's change to the unsigned-by-unsigned kernels that the static file's attention-score MatMuls use [37] explains neither step. At one thread only the second step is visible.

Across the first step the runtime builds the same executed graph: 698 nodes, node for node identical at 1.23.2, 1.24.4 and 1.25.1. The FP32 and dynamic files moved by less than 2%, and so did the static file at one thread. The regression is confined to the static path at more than one thread, and from 1.24.4 on, adding threads makes the static file slower.

The step is not noise. Across four independent timings of the same file, this one and three later runs (Section V and Section VI), every 1.24.4 session was slower than every 1.23.2 session (exact Mann-Whitney p = 7.4e-7 pooled, p = 1.25e-5 stratified by run), with a per-run factor of 1.25-1.35. A fifth timing in freshly rebuilt runtime environments, five sessions a side, reproduced it, with a factor of 1.31. The step is also specific to the hybrid's static file: DeiT-tiny's static file stays near parity with FP32 at four threads on every release, and the dynamic speedups of both models rose slightly over the releases.

## V. Where the Regression Comes From

Five explanations were tested in turn (Table III): three are rejected, one points at Transpose without reaching the share fixed in advance, and the fifth locates the regression. The executed graph did not change. The step survives with the thread pool's spinning disabled. A per-operator profile then points at Transpose, which carries 62% of the added time while the other operator types rose 9% together, short of the 70% fixed in advance for naming one operator. A static file that quantizes only the weight MatMuls does not step, while the full static file does in the same run, so we timed the Transposes alone.

Timed alone, the Transposes carry the regression (Fig. 2). A probe holding only the static graph's 56 uint8 Transposes, wrapped in quantize/dequantize and taken net of a Transpose-free control, reproduces 87% of the 0.484 ms step at four threads, in medians over the sessions that pass its registered core-delivery check; over all sessions it adds 0.437 ms, 77-107% of the step as measured in the four timings of Section IV. Four of the Transposes carry it: the unfold and fold Transposes of the MobileViT blocks, whose moved block, the run of contiguous bytes the Transpose copies as one unit, is two bytes wide. The other 52 add 0.007 ms. The same four loop shapes step in float16 and not in float32, and at one thread none of them steps.

One change in the runtime explains why exactly those four step. Pull request #25680 [38], first shipped in 1.24, enables MLAS's transpose for two-byte elements to speed up float16 inference. An eight-bit Transpose whose moved block is two bytes wide matches that element size, so from 1.24 it goes to MLAS's multi-threaded transpose as well, once per outer loop; by the v1.24.4 source that adds 832 thread-pool dispatches per inference at four threads, where 1.23.2 issues none for these four. At the probe's 0.417 ms for those four operators, each dispatch costs about half a microsecond. An optimization for float16 inference lands on INT8 files as a side effect.

## VI. Who Is Affected

The regression follows the model, and the same release runs the other way on a convolutional model (Fig. 3). A separate experiment, its reading fixed in advance, ran the FP32, static and dynamic files of six models on 1.23.2, 1.24.4, 1.25.1 and 1.30.0 at one and four threads in one random order, 462 sessions in all. A file counts as stepping if its static four-thread latency rises more than 7% at 1.24.4 and stays there through 1.30.0 while its FP32 and dynamic siblings move less than 7%.

Only the hybrid's static files step up. The 64 px file stepped again, by 25%. At 256 px the step grew to 46%, 4.43 ms against 0.41 ms at 64 px in the same run, so it grows with the input size rather than adding a fixed cost per inference, and it persists through 1.30.0. The FP32 and dynamic siblings of both files stayed within the floor. DeiT-tiny at 224 px, with 197 tokens, and the convolutional ResNet-50 and MobileNetV2 did not step.

YOLOv8n moved the other way. Its static file became 28% faster at four threads and 19% faster at one, while its FP32 sibling did not speed up, which raised its four-thread INT8 speedup from 1.05x to 1.58x. The same release that turns one model's parity with FP32 into a loss enlarges another model's gain. It also cut about 0.09 ms from every file of the autoencoder, FP32 and INT8 alike, the signature of a fixed per-call cost removed.

The nearest field report is a complaint of 15-20% higher CPU usage for a quantized model after upgrading to 1.24 on Windows, closed as stale after a maintainer asked for a per-kernel comparison [39]. Our static file matches it in release line, operating system and symptom, its CPU time per inference at four threads rising 27% from 1.22.1 to 1.24.4 with its siblings flat, and Section V gives the comparison that was asked for.

## VII. What INT8 Buys When Nothing Changes

A release effect of this size is of the same order as the effects INT8 is chosen for. Shin [19] attributes INT8 slowdowns of 1.65-1.76x to a missing dot-product instruction; on a CPU that has one, the runtime version alone took our static file from parity with FP32 on 1.23.2 to 1.54x slower on 1.30.0. This section measures, at a fixed release, what INT8 buys, so the release effect can be read against it.

### A. Static INT8 Pays on Convolutional Models

On the four convolutional models static INT8 was 1.63x (MobileNetV2) to 3.21x (ResNet-50) faster than FP32 at one thread, at 26-32% of the FP32 bytes, for at most 0.14 pt of top-1 accuracy, 0.02 pt of AUC and 0.69 pt of mAP (Table IV, Fig. 4). At four threads ResNet-50, MobileNetV2 and YOLOv8n kept 59-77% of that speedup, and the autoencoder, the smallest model, fell to 1.07x, within the floor. In all 31 static and FP32 pairs of the four original models, across 26 runs, the four-thread speedup was below the one-thread one, as Amdahl's law predicts when INT8 shortens the parallel arithmetic and not the serial part. The gain holds at deployment resolution: MobileNetV2 at 224 px gained 1.63x at one thread and 1.41x at four, and YOLOv8n at 640 px gained 2.26x and 1.63x.

### B. Dynamic INT8 on Convolutions Is More than 12x Slower

Applied to convolutions, as the quantizer's dynamic mode does by default, dynamic INT8 ran 12.6-16.8x slower than FP32 at one thread and 27-47x slower than the static file. The penalty does not shrink with input size: in a separate run of isolated sessions, MobileNetV2 paid 16.1x at 32 px and 18.2x at 224 px, and it holds on every release we tested that runs these files. The runtime executes 360 nodes for the dynamic MobileNetV2 against 69 for the static one, because each of its 52 ConvInteger nodes carries its own per-inference activation quantization and rescaling chain.

### C. On Attention and Hybrid Models the Ranking Reverses

On DeiT-tiny, dynamic INT8 beat static on every release, by 1.09-1.26x at one thread and 1.28-1.46x at four, at equal accuracy and size, and the ranking holds at 224 px. On the hybrid MobileViT-XXS, full static INT8 is slower than FP32 on 1.30.0 at both thread counts and 4.62 pt less accurate, while dynamic INT8 runs within 8% of FP32 speed and loses nothing: on the hybrid, INT8 buys mainly size. Part of that speed gap is the release effect of Section IV: up to 1.23.2 the full static and dynamic files ran within 7% of each other at four threads (Table II). A static file that quantizes exactly what dynamic quantizes loses only 0.07 pt, so the loss comes from what else full static quantizes; adding the attention-score MatMuls to that file costs 2.0 pt. At matched coverage, without them, the method still decides speed: static was slower than dynamic in all six release and thread-count cells tested, by 9-15% in five and 5% in the sixth (stratified exact rank test, p = 4.0e-5).

### D. The Default Activation Type Is Slower than FP32

The speedups above use U8S8. With signed activations and weights (S8S8), the documentation's default [4], INT8 was slower than FP32 on both models tried, in a run separate from Table IV (Fig. 5): at four threads MobileNetV2 fell from 1.09x to 0.47x and MobileViT-XXS from 0.70x to 0.39x. Recent guidance recommends U8S8 on x86 [40]; these measurements give the cost of leaving the default in place.

## VIII. Threats to Validity

One laptop CPU was measured under Windows. The routing change of Section V does not depend on the instruction set; what may differ across platforms is the cost of a thread-pool dispatch, which depends on the operating system's scheduler and on the count and type of cores. Servers with AVX-512-VNNI or AMX, AMD and ARM CPUs, and Linux were not tested. All runs used batch 1, and we measured four threads pinned to P-cores, not the runtime's unpinned default [26]. Each non-convolutional class is represented by one model; the files at 224 and 256 px carry random or 64 px weights and were timed only, and accuracy was measured on 1.30.0.

The mechanism of Section V is established in isolation, not by removal. A full-model counterfactual, rewriting the four Transposes so that their moved block is no longer two bytes wide, was attempted; its registered validity condition, that no variant show a one-thread release step above 0.034 ms, was not met, since a control stepped 0.0349 ms, so it returns no verdict, and a 1.24.4 build without the change was not run. The isolation probe's verdict comes from its second run: the first was inconclusive under its rule, whose core-delivery check cannot be met by cells that give the thread pool no work, and the second ran under a rule written before it that checks such cells for one fully delivered core instead. The probe covers the 64 px graph: the 256 px file's step, including a one-thread slowdown of 5.5% below the floor, and YOLOv8n's gain were not examined mechanistically.

## IX. Conclusion

On an x86 laptop CPU with AVX-VNNI, one ONNX Runtime release made one quantized model 31% slower on the same bytes and another model 28% faster. We traced the slowdown to four eight-bit Transposes that a float16 optimization sends to a multi-threaded path. Against what INT8 buys when nothing changes, the release is a first-order variable.

Four recommendations follow. Report every INT8 speedup with its runtime release, and measure it again after upgrading: the gain belongs to the file and the runtime together. Choose the method by model class: static INT8 for convolutional models, where dynamic INT8 was more than 12x slower than FP32, and dynamic INT8 for attention and hybrid models. Measure at the thread count you deploy: every static speedup in Table IV was smaller at four threads than at one, and at 64 px the regression needs more than one thread to appear. Check the activation type before trusting a default: on this CPU the quantizer's default left INT8 slower than FP32.

## References

[1] ONNX Runtime developers: ONNX Runtime. https://onnxruntime.ai

[2] Jacob, B., et al.: Quantization and training of neural networks for efficient integer-arithmetic-only inference. In: Proc. CVPR, pp. 2704-2713 (2018). https://doi.org/10.1109/CVPR.2018.00286

[3] Krishnamoorthi, R.: Quantizing deep convolutional networks for efficient inference: A whitepaper. arXiv:1806.08342 (2018)

[4] ONNX Runtime: Quantize ONNX models. https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

[5] Intel: Intel Neural Compressor 2.3, validated model list (ONNX Runtime 1.15.0). https://intel.github.io/neural-compressor/2.3/docs/source/validated_model_list.html

[6] Ahn, H., et al.: Performance characterization of using quantization for DNN inference on edge devices. In: Proc. IEEE ICFEC, pp. 1-6 (2023). https://doi.org/10.1109/ICFEC57925.2023.00009

[7] Upadhyaya, S.: int8-inference-lab: Post-training INT8 quantization benchmarked across PyTorch and ONNX Runtime. https://github.com/ShreyasUpadhyaya/int8-inference-lab

[8] Kluska, P., et al.: Beyond static and dynamic quantization -- hybrid quantization of vision transformers. In: Proc. BMVC (2024). https://bmva-archive.org.uk/bmvc/2024/papers/Paper_568/paper.pdf

[9] Nagel, M., et al.: A white paper on neural network quantization. arXiv:2106.08295 (2021)

[10] Liu, Z., et al.: Post-training quantization for vision transformer. In: Proc. NeurIPS (2021)

[11] Lee, J., et al.: Q-HyViT: Post-training quantization of hybrid vision transformers with bridge block reconstruction for IoT systems. IEEE Internet of Things Journal 11(22), 36384-36396 (2024). https://doi.org/10.1109/JIOT.2024.3403844

[12] Fernandez, J., et al.: The framework tax: Disparities between inference efficiency in NLP research and deployment. In: Proc. EMNLP, pp. 1588-1600 (2023). https://doi.org/10.18653/v1/2023.emnlp-main.98

[13] Louloudakis, N., et al.: DeltaNN: Assessing the impact of computational environment parameters on the performance of image recognition models. In: Proc. IEEE ICSME, pp. 414-424 (2023). https://doi.org/10.1109/ICSME58846.2023.00054

[14] Pape, D., Evertz, J., Schoenherr, L.: The silent hyperparameter: Quantifying the impact of inference backends on LLM reproducibility. arXiv:2605.19537 (2026)

[15] Zhang, Q., et al.: A comprehensive benchmark of deep learning libraries on mobile devices. In: Proc. WWW, pp. 3298-3307 (2022). https://doi.org/10.1145/3485447.3512148

[16] microsoft/onnxruntime: Issue #26293, [Performance] LRN latency 3x slower since 7c0c6fbe (v1.21.0) --- still present in v1.23.0 (2025). https://github.com/microsoft/onnxruntime/issues/26293

[17] microsoft/onnxruntime: Issue #27190, [Performance] Performance regression in AveragePool operator (2026). https://github.com/microsoft/onnxruntime/issues/27190

[18] microsoft/onnxruntime: Issue #27633, [Performance] CPU Conv kernel regression for large kernels (9x9) on macOS ARM64 in ORT 1.24.x (2026). https://github.com/microsoft/onnxruntime/issues/27633

[19] Shin, Y.: Is INT8 portable? A cross-platform measurement study of quantized inference on embedded and automotive accelerators. arXiv:2609.16085 (2026)

[20] Blalock, D., et al.: What is the state of neural network pruning? In: Proc. MLSys (2020)

[21] Dehghani, M., et al.: The efficiency misnomer. In: Proc. ICLR (2022)

[22] Li, Y., et al.: MQBench: Towards reproducible and deployable model quantization benchmark. In: Proc. NeurIPS Datasets and Benchmarks Track (2021). arXiv:2111.03759

[23] Mytkowicz, T., et al.: Producing wrong data without doing anything obviously wrong! In: Proc. ASPLOS, pp. 265-276 (2009). https://doi.org/10.1145/1508244.1508275

[24] Hoefler, T., Belli, R.: Scientific benchmarking of parallel computing systems: Twelve ways to tell the masses when reporting performance results. In: Proc. SC, pp. 1-12 (2015). https://doi.org/10.1145/2807591.2807644

[25] Reddi, V.J., et al.: MLPerf inference benchmark. In: Proc. ISCA, pp. 446-459 (2020). https://doi.org/10.1109/ISCA45697.2020.00045

[26] ONNX Runtime: Thread management. https://onnxruntime.ai/docs/performance/tune-performance/threading.html

[27] He, K., et al.: Deep residual learning for image recognition. In: Proc. CVPR, pp. 770-778 (2016). https://doi.org/10.1109/CVPR.2016.90

[28] Sandler, M., et al.: MobileNetV2: Inverted residuals and linear bottlenecks. In: Proc. CVPR, pp. 4510-4520 (2018). https://doi.org/10.1109/CVPR.2018.00474

[29] Ultralytics: YOLOv8, release v8.0.0 (2023). https://github.com/ultralytics/ultralytics

[30] Time Series Classification Website: ECG5000. https://www.timeseriesclassification.com/description.php?Dataset=ECG5000

[31] Touvron, H., et al.: Training data-efficient image transformers & distillation through attention. In: Proc. ICML, PMLR 139, pp. 10347-10357 (2021)

[32] Mehta, S., Rastegari, M.: MobileViT: Light-weight, general-purpose, and mobile-friendly vision transformer. In: Proc. ICLR (2022). arXiv:2110.02178

[33] Howard, J.: Imagenette. https://github.com/fastai/imagenette

[34] Krizhevsky, A.: Learning multiple layers of features from tiny images. Tech. rep., University of Toronto (2009)

[35] Lin, T.Y., et al.: Microsoft COCO: Common objects in context. In: Proc. ECCV, LNCS, vol. 8693, pp. 740-755 (2014). https://doi.org/10.1007/978-3-319-10602-1_48

[36] microsoft/onnxruntime: Pull request #27691, [CPU] Update the NCHWc transformer to handle more patterns (2026). https://github.com/microsoft/onnxruntime/pull/27691

[37] microsoft/onnxruntime: Pull request #27671, Fix MLAS qgemm dispatch and kernel regressions in quantized conv tests (2026). https://github.com/microsoft/onnxruntime/pull/27671

[38] microsoft/onnxruntime: Pull request #25680, FP16 inference performance improvement on CPU (2025). https://github.com/microsoft/onnxruntime/pull/25680

[39] microsoft/onnxruntime: Issue #27436, [Performance] (2026). https://github.com/microsoft/onnxruntime/issues/27436

[40] microsoft/onnxruntime: Pull request #28415, docs(quantization): clarify CPU parameter recommendations (2026). https://github.com/microsoft/onnxruntime/pull/28415
