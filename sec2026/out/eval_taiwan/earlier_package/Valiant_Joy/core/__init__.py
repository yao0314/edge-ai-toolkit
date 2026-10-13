"""Task-agnostic edge-AI compression toolkit core.

Author: 楊元蓁 (Price Yang)
Copyright (c) 2026 楊元蓁 (Price Yang). All rights reserved.

HARD RULE: nothing in this package may contain task-specific code
(datasets, model families, metric formulas). All of that lives in adapters/.
tests/test_core_is_task_agnostic.py enforces this.
"""

__author__ = "楊元蓁 (Price Yang)"
__copyright__ = "Copyright (c) 2026 楊元蓁 (Price Yang)"
__version__ = "0.5.0"

#: Stamped into every artefact this toolkit produces: exported ONNX models (producer/metadata),
#: sweep summaries, deliverable reports and the Excel report. tests/test_attribution.py checks it.
AUTHOR = __author__
TOOLKIT = "edge-ai-toolkit"
PRODUCER = f"{TOOLKIT} {__version__} by {AUTHOR}"


def provenance() -> dict:
    """Author/tool fields to embed in generated files."""
    return {"produced_by": PRODUCER, "author": AUTHOR, "toolkit": TOOLKIT, "toolkit_version": __version__,
            "copyright": __copyright__}
