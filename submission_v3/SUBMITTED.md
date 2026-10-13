# Official submission of Valiant Joy (SEC 2026 Software Track)

This folder is the package that was actually submitted, unchanged: `Valiant_Joy/` is the extracted
content of `Valiant_Joy_submission_AS_SUBMITTED.zip`.

- Submitted zip SHA-256: `707B23E4FD6D33356234000CFFEBED9A455AF91DF512ACD37B50E178A5CCB5C7`
- Model: `Valiant_Joy/submission/valiant_joy_int8.onnx` (SHA-256 starts `80456CBAE695C4EB`):
  the provided MobileNetV2 student, 60% structured pruning, recovery distilled from the provided
  ResNet-50, static INT8 (QDQ). Designed in Taiwan on an i7-13620H.
- Earlier candidates kept as backups, not submitted: `submission_v1` (INT8 only),
  `submission_v2` (20% pruning + INT8).
- Independent re-measurement of the submitted model on the on-site laptop (Ryzen 7 4800H, no
  AVX-VNNI): `sec2026/out/eval_taiwan/` (accuracy on both splits, latency / RAM / measured energy).
  On the team's split the FP32 baseline reproduces exactly (99.09%); the INT8 model gives 96.87%
  there versus 97.15% in metrics.json, a ~0.3-point CPU-dependent difference (no VNNI, no reduce_range).
