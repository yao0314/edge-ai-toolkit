---
name: onnx-canonical-output
description: SEC 2026 edge-ai-toolkit — ONNX is the canonical output format even if organisers want another format
metadata: 
  node_type: memory
  type: project
  originSessionId: 103ae351-d709-4d02-91fc-0b8da660fe58
  modified: 2026-09-11T17:06:39.630Z
---

The edge-ai-toolkit (C:\Users\User\Desktop\SEC\edge-ai-toolkit, for the SEC 2026 Minimalist Edge AI Challenge, Software Track, on-site 2026-10-13) always outputs ONNX. The user decided (2026-09-12) that this holds even if the organisers require another submission format, because converting from ONNX is easier.

**Why:** the submission format is still unknown. ONNX is the easiest hub format to convert from (AI Hub and onnx2tf both take it).

**How to apply:** don't build a TFLite-native output pipeline unless asked. `target_runtime: tflite` stays a reserved stub. If conversion is needed later, convert the *FP32* ONNX of the chosen variant and quantize with the target's own tool. Don't convert the QDQ INT8 ONNX. Related: [[sec2026-competition-unknowns]]
