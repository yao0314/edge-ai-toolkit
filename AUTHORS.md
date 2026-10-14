# Authors

**楊元蓁 (Price Yang)** — author and maintainer of edge-ai-toolkit.

Copyright (c) 2026 楊元蓁 (Price Yang). All rights reserved.

Built for the ACM/IEEE SEC 2026 Student Innovation Challenge — Minimalist Edge AI Challenge,
Software Optimization Track.

## SEC 2026 collaborative contribution

**張維銚 (Weiyao / David Chang)** — contributor to the Team Valiant Joy SEC 2026 effort, collaborating on MobileNetV2 compression, knowledge distillation, INT8 model optimization, benchmarking, and preparation of competition deliverables.

**Team Valiant Joy** — 3rd Place, SEC 2026 Software Optimization Track.

The original toolkit authorship and the existing copyright notice above are preserved. This section acknowledges the competition collaboration; it does not assert that one contributor individually authored the entire toolkit or transfer ownership of pre-existing code. Specific contributions can be refined by mutual agreement.


## Where this attribution appears

The author is recorded in the places that travel with the work, not only in this file:

| Location | What carries it |
|---|---|
| `core/__init__.py` | `__author__`, `AUTHOR`, `PRODUCER`, `provenance()` — the single source used everywhere else |
| Exported models | every `.onnx` gets `producer_name` plus `author` / `copyright` / `toolkit` metadata entries (survives copying the model out of this repo) |
| `results/<run>/summary.json` | `produced_by`, `author` fields |
| `deliverables/<run>/REPORT.md`, `metadata.json` | author line in the report handed in with the model |
| `results/edge_ai_toolkit_report.xlsx` | workbook creator property and a footer line |
| Git history | commit author of every change |
| `tests/test_attribution.py` | fails if the attribution is removed from the code or from a produced artefact |

Note: source code can always be edited by whoever holds a copy — this is attribution, not a technical lock.
Git history and the metadata embedded inside already-produced model files are the durable records.
