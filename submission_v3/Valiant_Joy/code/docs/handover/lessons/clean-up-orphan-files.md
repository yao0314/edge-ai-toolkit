---
name: clean-up-orphan-files
description: "Delete scratch/orphan files created while debugging, every time, not at the end"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 103ae351-d709-4d02-91fc-0b8da660fe58
  modified: 2026-09-12T11:39:11.331Z
---

Every debugging or optimization round must account for the files it creates and delete them when
done — scratch probe models, duplicate ONNX exports, throwaway virtualenvs, interrupted-download
`.part` files, one-off patch scripts.

**Why:** on 2026-09-12 the session scratchpad had grown to **4.7 GB / 334 files over 1 MB** —
including a whole TensorFlow virtualenv (`tfenv`) and dozens of 13–94 MB ONNX probe exports from
calibration experiments — before the user noticed and asked for it to be cleaned. Nothing warned;
it just accumulated one debug round at a time. A killed (rather than failed) download also left an
orphan `.part`, which made an image/label count look mismatched (1409 images vs 1408 labels) and
briefly looked like a data-integrity bug.

**How to apply:**
- Delete scratch artifacts at the end of the round that created them, not "later".
- Before reporting a round complete, run an orphan sweep: `*.part *.tmp *.orig *.bak *.rej`,
  scratch `*.onnx`, and anything over ~1 MB in the scratchpad.
- Prefer `tempfile.TemporaryDirectory()` inside scripts so intermediates cannot outlive the run
  (this is what `_auto_calibrate` in `core/quantize.py` does for its candidate models).
- If a scratch file is real evidence for a claim, do not just keep it in the scratchpad — promote
  it into the repo with a name that says what it proves (e.g.
  `results/stress_resnet50/sweep_minmax_calibration.csv`), then wipe the rest.
- Never delete a `.part` belonging to a download that is still running; clear stale ones at the
  *start* of the next fetch instead. See [[onnx-canonical-output]] for why ONNX exports are the
  artifacts that pile up fastest here.
