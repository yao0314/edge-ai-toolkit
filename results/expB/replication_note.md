# Replication stage (§5.5) — the one deviation, recorded before launch

**2026-09-24, written before any replication session was run.**

The stage runs exactly as registered in `results/paper/b_design_registered.md` §5.5 — every MOVED and
every UNRESOLVED adjacent pair-cell of the map, five fresh sessions a side, one shuffled invocation per
model, seed 91, its own stratum, never pooled with the map — with **one** change:

| | registered | this run |
|---|---|---|
| time cap | 75 min | 240 min (`--replicate-cap-min 240`) |

**Why.** §5.5 fixes the cap "so the stage cannot overrun the night". Its only purpose is the overnight
time budget; it does not depend on any result. The stage now runs in daytime on a day of machine time
the user allotted, so the reason for the cap does not apply. The input list is
`results/expB/moves_sealed.json` (114 pair-cells: 78 MOVED, 36 UNRESOLVED; its b01 source hash matches
the map file on disk), and its cost under the stage's own estimator is 151 minutes for all 13 models.

**What this does not change.** The priority order is untouched: descending |Δ|, ties by (arm, model,
variant, threads, transition). The registered stage is therefore **a prefix of this run** — the first
five models, 54 pair-cells, 74.4 estimated minutes, are exactly what the 75-minute cap would have
tested, in the same order and with the same seed. Nothing registered is dropped or altered; the run only
continues past the point where the registered cap would have stopped. Both are reported: the registered
prefix as §5.5's stage, and the remainder as its extension, labelled.

Under the registered cap the NOT-TESTED models would have included `mv64` and `y256`. Those two already
have several independent timings each (the submitted paper's sweep, spin, coverage and scope runs; A2's
control; the map), so their replication here adds a further run rather than a first one.
