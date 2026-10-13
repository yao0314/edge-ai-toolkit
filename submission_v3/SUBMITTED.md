# Official submission of Valiant Joy (SEC 2026 Software Track)

The three files submitted, unchanged:

| File | SHA-256 |
|---|---|
| `valiant_joy_int8.onnx` | `80456CBAE695C4EBADDE8AC422713A26CCE8CC312575A994E9D87883F3CCAA9A` |
| `metrics.json` | `914DC800D9903886EEC7F5329893EC43CDF0D197E5E6140A52E30235D88294F9` |
| `Valiant_Joy_documentation.zip` | `1AE8872365563D0062D368ACE7323A3838C52E6C195C7A4C3917FA726066C6DE` |

`Valiant_Joy/` is the extracted documentation zip (report PDF, pitch, split, baseline metrics,
tradeoff curve, FP32 and INT8 ONNX, and the code under `code/`).

Model: the provided MobileNetV2 student, 60% structured pruning, recovery distilled from the provided
ResNet-50, static INT8 (QDQ); designed in Taiwan on an i7-13620H.

Not submitted, kept as backups: `submission_v1` (INT8 only), `submission_v2` (20% pruning + INT8).
An earlier layout of the same package and the on-site re-measurement of the submitted model
(Ryzen 7 4800H, no AVX-VNNI: accuracy on both splits, latency, RAM, measured energy) are in
`sec2026/out/eval_taiwan/`. On the team's split the FP32 baseline reproduces exactly (99.09%); the
INT8 model gives 96.87% there versus 97.15% in metrics.json, a ~0.3-point CPU-dependent difference.
