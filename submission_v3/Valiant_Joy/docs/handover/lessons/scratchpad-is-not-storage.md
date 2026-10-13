---
name: scratchpad-is-not-storage
description: "Agents told to \"delete scratch files before finishing\" wiped the scratchpad, including all 10 ONNX Runtime side environments; keep irreplaceable state in the project"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 103ae351-d709-4d02-91fc-0b8da660fe58
  modified: 2026-09-21T15:49:06.523Z
---

On 2026-09-21 every ONNX Runtime side environment (ort116 ... ort1270, the ten venvs behind the paper's eleven-release sweep) and the helper script jess_status.py were found deleted from the session scratchpad. No backup and no pip wheel cache existed.

Cause: audit-agent prompts said "scratch files only under the scratchpad, deleted before you finish". At least one agent read that as "empty the scratchpad" and removed everything in it, including directories other agents and the main session depended on.

**Why:** "scratch" was defined by location, not by ownership, so an agent could not tell its own temporary files from the session's long-lived environments. Rebuilding the environments requires re-downloading wheels (needs the user's permission) and they will not be byte-identical to the ones the published measurements used.

**How to apply:**
- Keep anything that cannot be regenerated in one command (venvs, registered protocols, runner scripts, status tools) inside the project, e.g. `edge-ai-toolkit/.envs/`, never in the scratchpad.
- In agent prompts, give each agent its OWN scratch subdirectory (`<scratchpad>/<agent-label>/`) and say "delete only that directory"; name the paths that are off-limits.
- Record the exact package versions of every measurement environment (pip freeze) in results/, so a rebuilt environment can be compared with the one used.
Related: [[workflow-agents-leave-orphan-processes]], [[clean-up-orphan-files]].
