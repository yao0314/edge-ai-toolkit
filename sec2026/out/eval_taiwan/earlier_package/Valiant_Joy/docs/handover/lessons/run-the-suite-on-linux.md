---
name: run-the-suite-on-linux
description: Single-platform development masked three real bugs; run the test suite on WSL Linux before trusting a green Windows run
metadata:
  type: feedback
---

On 2026-09-14/15 three genuine defects in edge-ai-toolkit were invisible on Windows and appeared the
first time the suite ran on Linux. None was "a test we forgot to write" - every one had a test that
the Windows environment silently satisfied:

- `openpyxl` was missing from requirements.txt. `tests/test_selection.py` covered the Excel report,
  but the Windows `.venv` happened to have openpyxl, so it was green there and a clean install
  could not produce the Excel deliverable at all.
- `tests/test_formats.py` had a module-level `pytest.importorskip("ai_edge_litert")`. Without that
  optional extra the WHOLE file vanished - 15 tests, including the torch / TorchScript / .pt2 / ONNX
  ingestion tests that have nothing to do with TFLite - and the pytest summary did not mention them.
- Hidden behind that skip, `test_format_detected_from_content` genuinely failed on Linux.

**Why:** a passing suite on one OS is evidence about that OS. The competition sandbox is described
as a "virtual development environment" and is most likely Linux, which is precisely the platform
that had never been exercised.

**How to apply:** before claiming the toolkit is in good shape, run
`wsl.exe -e bash -c 'cd /mnt/c/.../edge-ai-toolkit && ~/eatk-v2/bin/python -m pytest tests/ -q -rs'`.
`-rs` matters: it prints the skip reasons, which is how a silently-vanishing test file becomes
visible. WSL venvs kept for this: `~/eatk` (no torch, for the degraded path) and `~/eatk-v2` (full).
Current baseline: Windows 167 passed, Linux 159 passed + 8 skipped (TFLite only).
See [[pruning-earns-nothing-within-budget]] for the results side.
