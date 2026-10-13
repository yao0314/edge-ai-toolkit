# Rewrite plan — narrative first

## The one sentence a reader must remember

**Upgrading ONNX Runtime by one release made an INT8 model 31% slower on the same bytes, and we can
name the four operators and the one code change responsible; the same release made another model
28% faster.**

## The three pieces of evidence, in the order the paper presents them

1. **The phenomenon** (Section 4). Eleven releases, one set of byte-identical files. The static INT8
   MobileViT-XXS file's speedup falls from 0.976x to 0.631x at four threads, in two steps: at 1.24.4
   the INT8 file slows down, at 1.25.1 its FP32 baseline speeds up. The executed graph is identical
   across the first step; the FP32 and dynamic files do not move. → **Fig. 1** (the sweep), Table 3.
2. **The mechanism** (Section 5). Four candidate explanations are ruled out in turn; what remains is
   eight-bit Transposes. Timed alone they reproduce 87% of the step; four of them, whose moved block
   is two bytes wide, carry it; the same shapes in float16 step and in float32 do not; at one thread
   nothing steps. One code change in 1.24 routes exactly those Transposes to a multi-threaded path:
   832 thread-pool dispatches per inference instead of none. → **Table 2** (hypotheses), **Fig. 2**
   (localization).
3. **The scope** (Section 6). The regression grows with the input (46% at 256 px), spares four
   convolutional static files, and the same release makes YOLOv8n's static file 28% faster: the
   effect is real, model-dependent and runs both ways. → **Fig. 3** (new: one bar per file).

## What only rules out alternatives (kept short, placed where the doubt arises)

- Noise: every 1.24.4 session slower than every 1.23.2 session across four independent timings
  (exact Mann-Whitney p = 7.4e-7); a null-gap sweep bounds spurious gaps at 3.3% (p95); reproduced
  again in freshly rebuilt environments (V0).
- Graph change: the optimized graph is node-for-node identical at 1.23.2, 1.24.4 and 1.25.1.
- Spin policy: the step survives with spinning disabled.
- Quantized MatMuls: a static file quantizing only the weight MatMuls does not step.
- The FP32 and dynamic siblings of the stepping file do not step.

## What is background (Section 7, compressed, AFTER the main result)

What INT8 buys when nothing changes, so a reader can judge how large 31% is: static INT8 on
convolutions 1.6–3.2x at one thread and less at four; dynamic INT8 on convolutions more than 12x
slower; the ranking reverses on the attention model; on the hybrid INT8 buys mainly size; the
quantizer's default activation type makes INT8 slower than FP32. → Table 4, **Fig. 4**.

## Contributions, as the introduction states them

1. A whole-model INT8 regression on byte-identical files, measured over eleven releases and, to our
   knowledge, the first report that separates the static and dynamic INT8 paths on the same bytes.
2. Its cause located to four operators and one routing change, each step fixed in a dated log
   before it was measured.
3. A baseline of what INT8 buys on an AVX-VNNI laptop CPU, and three recommendations for deployers.

## Rules for every sentence

- Each paragraph opens with its claim; numbers and qualifiers follow.
- At most one number per sentence in the abstract and introduction; four numbers in the abstract.
- Experiments are named by the question they answer, never by log names (no "scope run", "coverage
  run", "V4a", "class-gated", "re-registered").
- Limitations are stated once, in Section 8. A result paragraph keeps only a qualifier without
  which a number would be misread.
- Every term is defined at first use. At most one parenthesis per sentence.
- Every figure has a caption a skimming reader can act on: what it shows, and what to see in it.

## Structure and page budget (target 12–13 of 15)

| § | Title | Carries | Pages |
|---|---|---|---|
| 1 | Introduction | the case, why it matters, three contributions | 1 |
| 2 | Background and Related Work | static/dynamic, QDQ, U8S8; prior work, no own results | 1 |
| 3 | Method | machine, models, the two protocols named by question, floors, fixed-in-advance rules | 1.25 |
| 4 | One Release, Same Bytes, 31% Slower | Fig. 1, Table 3 | 1.5 |
| 5 | Where the Regression Comes From | Table 2, Fig. 2, the code change | 2 |
| 6 | Who Is Affected | Fig. 3 | 1 |
| 7 | What INT8 Buys When Nothing Changes | Table 4, Fig. 4, the default activation type | 2 |
| 8 | Threats to Validity | every limitation, once | 0.5 |
| 9 | Conclusion | three recommendations | 0.5 |
|   | References | | 2 |
