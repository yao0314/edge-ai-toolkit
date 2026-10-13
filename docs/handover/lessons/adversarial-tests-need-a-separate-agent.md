---
name: adversarial-tests-need-a-separate-agent
description: Adversarial experiments must always be designed by an independent challenger agent, never by the agent that formed the claim
metadata:
  type: feedback
---

**Adversarial experiments and adversarial reviews are ALWAYS designed by a separate challenger
agent.** Never by the agent that produced the claim being attacked — that includes me.

**Why:** an author designs the attacks they already expect to survive. If I could think of the
attack that kills my own claim, I would already have accounted for it, so the attacks I invent are
by construction the ones I can answer. The independence is not a formality; it is the only reason
the surviving claim means anything.

Concretely, on 2026-09-16 I wrote `scripts/paper/adversarial_int8.py` myself to attack my own two
INT8 findings, and the user caught it. The same session had already shown what independent
challengers find that I do not: one reproduced a proposed fix and discovered the FIX was wrong
(sorted dict keys reorder outputs); another re-counted a claim and found 28 sweeps had zero
failures, so the error-handling path had never run.

**How to apply:**
- Spawn the challenger through the Agent tool or a Workflow, with the claim stated as a conclusion
  to be refuted. Give it the same background and the same tools, but NOT my reasoning about why the
  claim holds — see [[challenger-independence-is-context-not-reasoning]] if that note exists, or
  the rule is: sync the context, isolate the reasoning. Only the structured conclusion crosses.
- Target objective is "refute this", not "evaluate this". Explicitly permit returning nothing.
- Use the same model tier. A weaker challenger rubber-stamps.
- My own attacks may be kept as a labelled baseline to compare the independent designs against —
  never as a substitute.
- **Say "design only, do not measure" when the machine is busy.** On 2026-09-16 five challengers
  were spawned while an overnight batch was running; they went straight to executing latency probes,
  several at once, and inflated both their own numbers and the batch's. Their own output gave it
  away - one cell read `2.289 ms (sd 2.441)`, an SD larger than the measurement. Three runs had to
  be quarantined and re-run. The designs were excellent; only the execution had to be thrown away.
  So: challengers design and return a ranked attack list, and the numbers are produced afterwards,
  sequentially, by one process. This does not weaken the independence rule - the DESIGN is what must
  be independent, and it still is.
- Healthy signal: a kill rate somewhere near half, and corrections on most items. 0% means the
  challenger is rubber-stamping; 100% means it is over-correcting to look useful.

Related: [[run-the-suite-on-linux]] (same family of lesson — my own green result was not evidence).
