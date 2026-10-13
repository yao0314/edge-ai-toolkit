# Claim inventory for the rewrite

2026-09-24. Every claim the paper may make, one line each, with its evidence. This is the input to the
rewrite; the submitted paper stays frozen and is not patched. Detail lives in `A_RESULTS.md`,
`B_RESULTS.md`, `W_RESULTS.md` and `SYNTHESIS.md` (Parts 1–2; its Part 3 is superseded).

**Corrected 2026-09-24, after the independent review of the rewrite** (five reviewers; every
correction re-derived from the raw files before editing): claims 9, 14, 17, 19, 20, 22, 23, 24, 25, 26,
27, 29, 33, 34, 36, 42, 43 and two lines of §8. Each is marked **[corrected]** and names what it
replaces.

**Core, in one sentence:** INT8 on a CPU buys speed, and with it a performance trap that can be read
off the file — a code path whose cost the runtime can change between releases.

---

## 1. What INT8 buys (submitted paper §IV)

1. Static INT8 makes convolutional models **1.6–3.2×** faster at one thread (1.30.0).
2. It buys less at four threads; a speedup must be reported at the deployed thread count.
3. Files shrink to **26–32 %** of FP32 for at most **0.86 pt** of quality.
4. Dynamic INT8 on convolutions is **more than 12× slower** than FP32 at one thread, on all three releases tested.
5. On the attention model the method ranking reverses.
6. On the hybrid model INT8 buys mainly size.
7. The quantizer's default activation type (S8S8 on x86) leaves INT8 slower than FP32.
8. Speed, size and quality rank the methods differently on the hybrid.

## 2. The trap

9. **[corrected]** A byte-identical file changes speed with the runtime release. MobileViT static INT8 at 64 px, four threads, 1.23.2 → 1.24.4, in four independent runs: **+31 %** (release sweep, the submitted paper's figure), **+32.8 %** (A2 run), **+40.3 %** (drift map), **+30.4 %** (replication); at 256 px **+43.9 %** (map) and **+44.5 %** (replication). The rewrite's first page uses the sweep's 31 % and quotes 30–40 % across four runs; 40.3 % is the largest of the four and is not the headline. *Replaces:* "+40.3 % … in a fresh environment" used as the headline beside +31 % without naming the runs.
10. The FP32 and dynamic siblings of the same weights do not move; the trap belongs to the static INT8 path.
11. At 64 px it needs more than one thread.
12. It persists: **+37.6 %** against 1.23.2 at 1.30.0, the newest release measured.
13. The optimized graph is node-for-node identical across the step (698 nodes).
14. **[corrected]** The explanations tested **for the 1.24.4 step** (submitted paper, Table 3), each against a result fixed in advance: the executed graph did not change (698 nodes); with spinning disabled the step persists (+23.7 %); a MatMul-only static file shows no step (−1 %) while the full file steps +35 % in the same run. A four-thread per-operator profile points at Transpose (62 % of the added time, other operator types +9 % together; short of the 70 % rule for naming one operator), and the 56 uint8 Transposes timed alone change by +0.421 ms, 87 % of the sweep's 0.484 ms step, and by +0.008 ms at one thread (Experiment 27b, the registered repeat; the first run, Experiment 27, read +0.423 ms and stayed INCONCLUSIVE under its gate). *Replaces:* "five rival explanations: spin policy, quantized MatMuls, QDQ density, fusion blocking, bandwidth". QDQ density, fusion blocking and bandwidth were tested on 1.30.0 as explanations of why the static path is slow on the hybrid; they are not evidence about the step.
15. A speedup has two moving terms: 1.25.1 sped up the FP32 baseline by 15–16 %.

## 3. The mechanism

16. PR #25680 routes 8-bit Transposes whose moved block is **two bytes** to MLAS's multi-threaded transpose.
17. **[corrected]** On constructed files where width is the only variable, only the two-byte files slow down. No 1-, 4- or 8-byte file slows; the 8-byte files (−7.0 to −10.2 %) and the Transpose-free controls (−4.2 to −7.8 %) got faster by the same release-wide saving of about 1.3 µs per inference, which crosses the 7 % floor only on 0.03 ms files, so under the MOVED rule taken literally the 8-byte files and `ctrl_f32` are MOVED (faster). Net of the controls every non-two-byte file is inside the floor. 34 of the 57 registered cell predictions (19 files × 3 thread counts) held as written; of the 23 misses, 12 are that saving, 5 are the two-byte files' one-thread serial cost (claim 22), 5 cleared the floor with overlapping ranges, and 1 is a one-byte dose file 9 % faster at one thread. *Second correction, 2026-09-24:* first written as "21 of the 36", which counted the main ladder only. *Replaces:* "1-, 4- and 8-byte files are flat".
18. The byte width governs, not the element count: two uint8 or one float16 per block agree within 1.2×.
19. **[corrected]** The added cost tracks the dispatch count: 16× the dispatches, 13.2× the **added** latency (the total latency grows 10.6×), at constant bytes moved; log–log slope 0.93 over the three points; 0.40–0.52 µs per dispatch. *Replaces:* "proportional", which the three points support only approximately.
20. **[corrected]** One dispatch costs **0.40–0.52 µs** in the probes. Charging the **whole** model step (drift-map run) to the new dispatches gives **0.80–1.03 µs**; charging only the four flagged Transposes' own change, timed in isolation (Experiment 27b: +0.417 ms over 832 dispatches; Experiment 27, inconclusive, read +0.427 ms), gives **0.50 µs**, the probes' level. The rest of the model's step lies outside those Transposes (the profile's +9 % over the other operator types) and is not predicted by the dispatch count. Two runs are also mixed in the 0.80 figure: the map's 64 px step is 0.664 ms (+40.3 %), the sweep's 0.484 ms (+31 %); 832 × 0.402–0.523 µs = 0.334–0.435 ms is 69–90 % of the sweep's step and half to two-thirds of the map's. Work per dispatch explains none of it: the probes' loops move 8–128 bytes at the same cost, the models' flagged loops 8–16 bytes at 64 px (12.9 on average) and 16–64 at 256 px (45). *Replaces:* "6.5 bytes per loop" (no source; the executed graph gives 12.9), "a model dispatch costs twice as much" and "it must come from context" (never measured).
21. The re-routed file lands on the always-routed file at matched dispatch count: **1.817 against 1.813 ms, 0.25 %**. The file was moved onto a slow path, not made slow.
22. **[corrected]** There is also a serial cost, read off the probe with the model's loop count: `d_u8_w2_L1024` has 4096 loops (four nodes × 1024) and costs +0.030 ms at one thread. The 256 px model has the same 4096 loops and changed by +1.401 ms (map) and +0.983 ms (replication) at one thread: **33–47×** what the serial cost gives. Per loop the serial cost is not constant across the dose files (13, 19 and 7 ns), so no per-loop figure is quoted. That slowdown is itself consistently positive — +5.5 % (submitted), +7.1 % (map), +5.2 % (replication), past the floor once — and remains unexplained. *Replaces:* "~29 ns per loop … 8.3–11.8× too small" (0.030 ms divided by 1024 instead of 4096 loops, then multiplied by 4096).

## 4. The screening criterion

23. **[corrected]** The condition is readable before any timing, from the executed graph the target release builds when it loads the file (session creation with `optimized_model_filepath`; no inference is run). *Replaces:* "readable from the file without running it", which reads as a check on the file graph alone.
24. **[corrected]** On 18 third-party static INT8 models (15 families, 9 publishers, 11 toolchains) the checker flagged none and none moved (+1.2 % to −4.8 % at four threads). Because nothing was flagged, "zero false alarms" and "specificity 18/18" hold by construction and carry no information. The informative result is that **every file the check called safe held still**, and the test has teeth: **15 of the 18 execute eight-bit Transposes** (moved blocks of 1, 32, 48, 49, 196 and 784 bytes, none of 2), so a check that flagged every eight-bit Transpose would have called 15 of them unsafe. The word is **criterion** (or screening rule), never *predictor*. *Replaces:* "zero false alarms, 18 negatives … this measures specificity only".
25. **[corrected]** The alarms are validated **by construction**, and the base is small: the check flags only eight-bit Transposes, so its alarms are tested on the four constructed **uint8** files with a two-byte block (three distinct constructions, `d_u8_w2_L256` replicating `m_u8_w2`), eight decision cells at two and four threads, and all eight moved, by **4.6× (`d_u8_w2_L64`, 256 dispatches, four threads) to 44× (`d_u8_w2_L1024`, 4096 dispatches, two threads)**; the float16 file with the same two-byte block also moved but is not among the check's alarms. *Second correction, 2026-09-24:* first written as "four distinct files, five with one replicate, ten decision cells", which counted the float16 file. The ratio grows with the dispatch count (claim 19). The paper states that count and where the range comes from.

## 5. How common, and how rare — stated together

26. **[revised — written as a claim, not left for a reviewer to find]** The pattern is **rare in the field**: none of the 18 public static INT8 files carried it, and most public "quantized" ONNX vision models are dynamically quantized, so they cannot hit it at all. In the drift map it accounts for **2 of the 59** single-release moves (3 counting the unexplained one-thread cell of the same file). This particular pattern touches few deployed files; the only field report found is an uncharacterized report of higher CPU use after 1.24 on Windows, closed as stale for want of a per-kernel comparison [40]. **[corrected]** *Replaces:* "which is also why it surfaced only once", a causal claim with no evidence behind it.
27. **[corrected]** **The class is not rare, and it is not INT8's alone.** On one machine a single-release upgrade moves an unchanged file by ≥7 % in **10.1–15.6 %** of pair-cells (file × thread count × step, N = 584). By variant: FP32 **9.8–19.6 %** (n = 112), static INT8 **9.3–13.3 %** (n = 400), dynamic INT8 **15.3–22.2 %** (n = 72). Against a fixed 1.23.2 baseline, 23.3–29.0 % of 630 later cells sit ≥7 % away. INT8's own exposure is that its speedup is a ratio of two latencies that releases move separately, on code paths FP32 does not take. The other 56 moves have other mechanisms. *Replaces:* "of single-release upgrades" (the unit is the pair-cell) and the unstated FP32 split.
28. **The paper explains one case to the microsecond and shows how to diagnose the rest.** Two further cases were diagnosed with the same procedure, and they fall on different branches of it. YOLOv8n's gain runs on an **identical executed graph** (283 nodes on both releases) and is **`Resize`**, 99 % of the delta, a 7.3× faster kernel. The largest third-party regression (+96.4 %) is **not** a kernel slowdown: on 1.22.1 the executed graph of `c3_299x299/b05` grows from **427 to 491 nodes** and loses every `QLinearConv`, `NhwcMaxPool` and layout `Transpose` — the quantized convolutions are no longer fused into integer kernels and run as float `Conv` wrapped in Quantize/Dequantize — and 1.23.2 restores the fusion. It is a **fusion fallback**.
29. **[corrected]** The one step that skips a minor version, 1.28.0 → 1.30.0 (76 pair-cells), moved more: 25.0–30.3 %. It is a single step, so "spanning" and "this release" cannot be separated. *Replaces:* "Multi-minor steps move more".
30. About a third of assessable moves never come back inside the floor.
31. Some releases move many files at once: 1.26.0 → 1.27.0 slows many, 1.28.0 → 1.30.0 speeds many up.

## 6. The diagnostic procedure

32. The same release that slowed MobileViT sped up YOLOv8n by 25–28 %.
33. **Three signatures, one procedure.** Compare the executed graphs first: if they differ, the release changed fusion (b05, 427 → 491 nodes). If they are identical and the file moves at one thread, a kernel changed (YOLOv8n, `Resize`). If they are identical and it moves far more at several threads than at one, thread-pool dispatch changed (MobileViT, `Transpose`, 698 nodes on both). A per-operator profile at a thread count where the file moves then points at the operator. **[corrected]** *Replaces:* "moves only at several threads" — a re-routing can also cost a little serially (the two-byte probes rose 0.003–0.030 ms at one thread), so "only" would misfile it — and an unqualified "names the operator" (see claim 34).
34. **[corrected]** Under `ORT_SEQUENTIAL` at one thread, node-time sums cover 88–99 % of wall time and name the operator outright (YOLOv8n: `Resize`, 97–99 % of the change). At four threads the profiler slows the file 7.5–8.4× and node sums cover 72–76 % of wall time; there the profile points at an operator (MobileViT: `Transpose`, 62 %) and an isolation probe confirms it (87 %). *Replaces:* the one-thread statement applied to all three cases.

## 7. Method

35. Four-thread session-to-session spread is **2.6×** the one-thread figure (9.0 % against 3.5 %); the 7 % floor is a four-thread floor.
36. **[corrected]** The decision rule has a **measured null**: one batch measured twice gives 1 false MOVED in 52 cells (2 %). That is a point estimate from one batch of the heaviest model, not a bound (its 95 % upper limit is about 10 %); it predicts about a dozen noise moves among the 59. The replication (claim 41) is the direct test. *Replaces:* the paper draft's "bounds the noise share".
37. Floor sensitivity: 7 % → 10.1–15.6 % (as registered), 10 % → 8.6–9.9 %, 12 % → 7.0–7.5 %. The registered figure is reported; the curve is shown beside it, not substituted.
38. The file graph is not the executed graph: ONNX Runtime deletes most Q/DQ nodes before inference.
39. ONNX Runtime 1.16.3 cannot run under numpy 2.x.
40. The submitted paper's smallest four-thread static gains (1.07×, 1.12×) sit inside the four-thread spread and cannot be read as gains.

## 7b. Independent replication (§5.5, P6)

41. **The map's moves reproduce.** All 114 moved and unresolved pair-cells re-measured in an independent run, five fresh sessions a side, seed 91, never pooled with the map: **69 of 78 MOVED replicate (88.5 %)**, and **28 of 28 with |Δ| ≥ 20 % (100 %)**; the primary denominator gives 52 of 59 (88.1 %). All three pre-registered P6 predictions hold (≥ 45 %, ≥ 80 % in the large band, large band above small); the refutation (< 25 %) does not fire.
42. **[corrected]** **What fails to replicate is small.** The nine NOT-REPLICATED cells are all 7.1–12.8 % moves and all kept their sign; one still cleared the floor with overlapping ranges (`b02` at one thread, 1.28.0 → 1.30.0: −12.4 % then −10.2 %, UNRESOLVED), and the other eight came back at 0.1–6.5 %. *Replaces:* "most of them the same sign but below the floor the second time". Of the 36 UNRESOLVED cells, 5 resolve as MOVED and 31 as STILL. Between-run difference on MOVED cells: median 1.7 pp, p90 5.2 pp, max 10.7 pp.
43. **The two files where A2 and the map disagreed both resolve in A2's favour**: `a03` (+0.0 %, −8.0 %, then −0.1 %) and `b02` (−2.9 %, +7.8 %, then −3.5 %). The criterion's safe calls survive the only two cases that looked like counter-examples. **[corrected]** *Replaces:* "zero false alarms" (see claim 24).

## 8. Not claimed

* No build-level revert; causality is shown by construction, not by removal.
* The 256 px one-thread slowdown is unexplained.
* One machine, one OS; the dispatch cost is a scheduler property.
* ~~The drift map is a single run.~~ **Resolved 2026-09-24:** the registered replication stage re-measured all 114 moved and unresolved pair-cells independently (claim 41).
* **[corrected]** The safe calls are tested on found files, the alarms on constructed ones; neither is quoted as both. *Replaces:* "specificity … sensitivity" (see claim 24).
* **[added 2026-09-24]** Drift is not claimed to be specific to INT8 (claim 27).
* **[added 2026-09-24]** Outputs were not compared across releases; releases were compared on speed only.
* Energy was not measured.
