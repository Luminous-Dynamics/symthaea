# UAL-P1 Live-Symthaea Diagnostic Pilot — Real Result

**Status: exploratory pilot result, NOT a formal UAL probe verdict.** Scoped and approved
via `/home/tstoltz/.claude/plans/ethereal-wandering-dewdrop.md` ("Live-Symthaea UAL
Diagnostic — P1 Reversal-Learning Pilot"), itself prompted by the question "should we
invest in making Symthaea better at UAL?" This pilot's purpose was to check cheaply whether
she already shows any hint of reversal-learning capacity in her real cognitive-loop
substrate before deciding whether to invest in building new capability. Recorded verbatim.

**Follow-up (2026-07-30)**: the block-collapse pattern below was root-caused via a
per-trial trace — see `SYMTHAEA_UAL_LIVE_DIAGNOSTIC_P1_COLLAPSE_TRACE_2026-07-30.md`. Short
version: the response trajectory shows a smooth, autonomous drift uncorrelated with the
trial-100 reversal, and `dopamine_effective` reads exactly `0.0000` on every trial
regardless of reward sign — most consistent with self-referential drift dominating over
reward-driven learning in this configuration, not real reversal learning. The numbers
below are unchanged and still the authoritative record of the raw result.

## What this is not

Unlike every other file in `benchmarks/ual/`, this result is **not** produced through the
`UalProbeReport`/preregistered-floor/power-analysis machinery, carries no
`Demonstrated`/`NotDemonstrated`/`Inconclusive` verdict, and licenses no claim about "UAL"
as a whole. It is a single exploratory run against `SystemUnderTest::LiveSymthaea` — the
first use of that previously-reserved variant.

## Source

- `crates/domains/symthaea-psych-bench/src/benchmarks/ual/p1_live_diagnostic.rs`
  (`P1LiveReversal`, `#[ignore]`d test `p1_live_reversal_diagnostic`).
- Driven via the already-existing `harness::live_runner::CognitiveLoopBenchmarkRunner`
  (used elsewhere for Butlin-suite live-integration tests) — zero production-code changes.

## Configuration

- `CognitiveLoopBenchmarkRunner::new("ual_p1_live_diagnostic_seed")` —
  `ConsciousnessProfile::Standard`, `async_training=false`, 100 warmup cycles.
- Real substrate dimension: **`state_dim()` = 256** — note this is `ConsciousnessProfile::
  Standard`'s configured CfC input dimension, not the canonical 16,384-D HDC figure
  documented elsewhere for the "full" production substrate. This pilot inherited whatever
  dimension the existing `CognitiveLoopBenchmarkRunner::new()` helper's profile choice
  uses; dimension was not deliberately chosen or varied.
- Response selection: deterministic argmax over cosine similarity of the CfC output
  against the two alternative hypervectors — **no exploration/temperature**, unlike the
  benchmark-local UAL learners' `softmax_choice`. This is a property of the existing
  `run_benchmark()` harness, not something built for this pilot.
- 200 trials: reward contingency favors arm 0 for trials 0-99, flips to favor arm 1 for
  trials 100-199 (single reversal, fixed change point).
- Single run, single seed (`config.seed=42`) — **not replicated across seeds**.

## Real result

```text
block 0 (trials   0- 49, pre-reversal,  arm0 correct):  accuracy=0.6400
block 1 (trials  50- 99, pre-reversal,  arm0 correct):  accuracy=0.0000
block 2 (trials 100-149, post-reversal, arm1 correct):  accuracy=0.4600
block 3 (trials 150-199, post-reversal, arm1 correct):  accuracy=0.0000

overall accuracy=0.2750; mean_reward=-0.1425; mean_prediction_error=0.3505; mean_psi=0.2725
```

## Interpretation (honest, deliberately bounded)

- **This does not show a clean reversal-learning signature.** The expected pattern (high
  accuracy pre-reversal, a dip immediately after the flip, recovery over subsequent trials)
  is not present.
- **The most striking feature is not about the reversal at all**: accuracy collapses from
  0.64 (block 0) to 0.00 (block 1) *within the unchanged pre-reversal contingency* — the
  correct answer (arm 0) did not change between these two blocks, yet the network's
  behavior went from mostly-correct to completely-wrong. Whatever drives trial-to-trial
  choice here does not appear to track the reward contingency in a stable way over even a
  single unchanging 100-trial block.
- **Overall accuracy (0.275) is below chance (0.5)** on a symmetric 2-alternative task —
  unusual, and worth flagging rather than glossing over, though not chased further in this
  pilot (would need a dedicated follow-up, e.g. checking whether the deterministic
  argmax-over-cosine-similarity readout is systematically anti-correlated with recent
  reward direction, or whether this is a single-seed artifact).
- `mean_reward=-0.1425` is exactly consistent with the observed accuracy
  (`0.275*0.8 + 0.725*(-0.5) = -0.1425`) — a basic internal-consistency check that passed,
  giving some confidence the harness itself is computing correctly even though the
  substantive result is unpromising.

## What this pilot does and does not tell us

- It does **not** demonstrate that live Symthaea lacks reversal-learning capacity in
  general — this is one run, one seed, one configuration (dim=256, Standard profile,
  deterministic no-exploration readout inherited from the existing harness). Any of those
  could be masking a real capacity that would show up under a different setup.
- It does **not** demonstrate that she has the capacity either — no clean positive
  signature was found.
- **It does not, by itself, provide encouraging evidence to justify escalating to a full
  preregistered live-Symthaea UAL probe track** (the kind of rigor the rest of `benchmarks/
  ual/` uses for P1/P2/P4a). Per the plan's own decision framework, a promising pilot
  result would have been the trigger to scope that; this result isn't promising enough to
  clear that bar on its own.
- The more interesting and tractable follow-up, if this line of work continues, is
  diagnosing the block 0 → block 1 within-contingency instability itself (a real, curious,
  reproducible-in-principle finding) rather than proceeding directly to P2/P4a live
  variants or to a bigger preregistered P1-live confirmatory run.

## What was NOT done

- No hyperparameter, dimension, or profile was changed after seeing this result.
- No additional seeds were run to "average out" the below-chance/unstable numbers.
- No attempt was made to explain away the block 0 → block 1 collapse in this pass — it's
  reported as an open, disclosed observation, not resolved.
