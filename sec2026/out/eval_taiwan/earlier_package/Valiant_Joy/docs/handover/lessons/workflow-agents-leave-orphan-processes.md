---
name: workflow-agents-leave-orphan-processes
description: "Workflow agents ran `find / -name llncs.cls`; the finds outlived their agents and loaded the CPU for hours"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 103ae351-d709-4d02-91fc-0b8da660fe58
  modified: 2026-09-18T19:46:33.298Z
---

On 2026-09-19, four paper-review agents each ran `find / -name llncs.cls` in Git Bash. Each agent gave up when its Bash call timed out, but the find.exe it started kept running. The four orphans used 4 cores for up to 3 hours and tripped the user's CPU-overload guard at 03:43, so the workflow had to be stopped mid-edit. `Stop-Process` did not kill the MSYS finds; `taskkill //F //PID` did.

**Why:** the user requires orphan discipline and asked that agents stop on CPU overload or overheating. Orphans also contaminate any latency measurement taken on this laptop.

**How to apply:**
- Every workflow agent prompt says: never search a whole drive or `/`; use `-maxdepth` or a known path; leave no background process running.
- After a workflow ends or is stopped, check for leftover `find.exe`, `python.exe` and `tectonic.exe`.
- Keep the CPU guard Monitor armed during long agent runs.

Related: [[clean-up-orphan-files]], [[adversarial-tests-need-a-separate-agent]].
