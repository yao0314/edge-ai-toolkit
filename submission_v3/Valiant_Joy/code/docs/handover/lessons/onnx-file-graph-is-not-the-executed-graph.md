---
name: onnx-file-graph-is-not-the-executed-graph
description: ORT deletes ~90% of Q/DQ nodes during graph optimisation, so any statistic counted from the .onnx file describes a graph that never runs
metadata:
  type: reference
---

**Never compute a graph statistic from the `.onnx` file and relate it to runtime.** ONNX Runtime's
`QDQSelectorActionTransformer` fuses QuantizeLinear/DequantizeLinear pairs into QLinear* kernels
during graph optimisation, before the first inference. Measured 2026-09-16:

    MobileViT  416 Q / 599 DQ  ->  40 / 97          DeiT  442 / 593  ->  63 / 102
    ResNet50    74 Q / 182 DQ  ->   1 / 1

**Why:** this invalidated an entire experiment in edge-ai-toolkit. The Q/DQ boundary-density
mechanism was built on file-graph counts; recounted on the executed graph the variable points the
OTHER WAY - the faster model family runs more surviving Q/DQ than the slower one. The fraction kept
is model-specific (10% to 24% here), so it is not a constant-factor correction: it reverses
orderings between models.

**How to apply:** dump the real graph before counting anything -

```python
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.optimized_model_filepath = str(dst)
ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])   # writes dst
```

`scripts/paper/qdq_boundary_density.py --graph both` does this and prints the size of the error per
model. Related: [[adversarial-tests-need-a-separate-agent]] - independent challengers found this,
not me, and it was findable from a text editor for three days.
