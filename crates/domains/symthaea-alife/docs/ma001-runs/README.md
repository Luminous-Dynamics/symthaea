# MA-001A run logs — preserved raw output

Per `ALIFE_MA001_PARTNER_CONDITIONED_POLICY_PLAN_2026-07-26.md`'s preregistration and Genesis's
own non-erasure precedent (`../genesis-runs/`): the confirmatory run's raw output, kept as run,
not just described in prose.

- **`ma001a-confirmatory-2026-07-26.txt`** — calibration (seed 9999) + 10 confirmatory seeds +
  swap intervention, `Ma001Config::default()` (population 100, 1200 ticks, 300-tick burn-in,
  100-tick shuffle epoch). Run via `cargo run -p symthaea-alife --example ma001_run`.

**Result** (see the plan doc's §10 for the full write-up): 0/10 confirmatory seeds pass the
calibrated margin. Divergence scores are small and nearly identical across all three conditions
(Bound/Shuffled/NoHistory, ~0.0023–0.0025). The swap intervention shows no systematic
history-following pattern (12 history-following vs. 14 identity-following among 247 classifiable
cases out of 1000 organism-instances; the remaining 753 were ineligible — mostly because there
was no meaningful pre-swap differentiation to test in the first place). Read against the plan's
interpretation ladder as **architecture limitation**, not "null from an unlucky threshold" or
"confounded" — the divergence scores' near-identical smallness across all three conditions,
combined with Genesis's own finding that a single-tick reachability test needed a *sustained*
500-tick two-organism setup just to detect any social-state sensitivity at all, points at the
current substrate genuinely not producing meaningful partner-conditioned policy differentiation
at this population scale, not at a broken measurement.

**One real bug found and fixed before this result was trustworthy**: the first run of this driver
used Genesis's own `4.0/n`-density-divided resource share, calibrated against populations starting
near 16 organisms (~0.25/organism). MA-001A's fixed population of 100 gave only ~0.04/organism
under that same formula — every organism in every condition starved to death, producing a
meaningless "0 alive, 0 eligible" run whose superficial "0/10 seeds... wait, actually passed
(0 >= 0)" result would have been actively misleading if reported at face value. Fixed by using a
flat, population-independent per-organism share (`0.25`, matching what sustains a population in
Genesis's own tests) — density-division exists to guard against unbounded growth in a
*reproducing* population, which MA-001A's fixed, non-reproducing design structurally cannot do, so
it was never actually needed here.
