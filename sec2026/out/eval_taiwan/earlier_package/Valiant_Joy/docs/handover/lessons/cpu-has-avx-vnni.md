---
name: cpu-has-avx-vnni
description: "The i7-13620H test laptop HAS AVX-VNNI (VEX); it only lacks AVX-512/AVX512-VNNI — never write \"no VNNI\""
metadata: 
  node_type: memory
  type: project
  originSessionId: 103ae351-d709-4d02-91fc-0b8da660fe58
  modified: 2026-09-18T06:46:22.138Z
---

CPUID on the measurement laptop (i7-13620H, Raptor Lake): AVX2 yes, AVX-VNNI (leaf 7.1 EAX[4]) yes, AVX512F no, AVX512-VNNI no, hybrid P/E yes. ORT's MLAS dispatches its AVX-VNNI U8S8 GEMM/conv kernels on this machine.

**Why:** the TAAI 2026 draft said "no VNNI" four times; independent review caught it on 2026-09-18, and EXPERIMENT_PLAN.md had already warned against the claim. Any scope sentence built on it ("may be specific to non-VNNI x86") was void.

**How to apply:** describe the CPU as "AVX2 + AVX-VNNI, no AVX-512". Check a hardware claim with a CPUID probe before it goes into text. Related: [[onnx-file-graph-is-not-the-executed-graph]].
