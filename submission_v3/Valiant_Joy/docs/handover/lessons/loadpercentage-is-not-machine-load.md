---
name: loadpercentage-is-not-machine-load
description: Win32_Processor.LoadPercentage is not machine-wide load on this laptop; it caused false CPU-overload alarms that stopped a timing run
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 103ae351-d709-4d02-91fc-0b8da660fe58
  modified: 2026-09-19T05:56:55.620Z
---

On the i7-13620H laptop (6P+4E, 16 logical CPUs), `Win32_Processor.LoadPercentage` jumped between 48% and 100% while the machine was 22-32% busy (`Win32_PerfFormattedData_PerfOS_Processor` `_Total` `PercentProcessorTime`; Idle 1048-1415 of 1600). A single 4-thread pinned timing child (python = 401 core-%) was enough to read "100%".

**Why:** the user's CPU-overload guard is built on this reading. On 2026-09-19 at 13:47 it fired on K1's own 4-thread cells and the K1 timing run was stopped for nothing. The 03:43 "overload" (four stray `find` processes, 88%) likely read high for the same reason.

**How to apply:** measure machine load with PerfOS_Processor `_Total` PercentProcessorTime, or with Idle / (100 x logical CPUs). On an alert, capture the top processes from `Win32_PerfFormattedData_PerfProc_Process`, which also shows protected processes such as Defender. Quiet-machine prechecks for timing runs should use `_Total` too. Related: [[other-load-pct-is-a-dead-gauge]], [[workflow-agents-leave-orphan-processes]].
