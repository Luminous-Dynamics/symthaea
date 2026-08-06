# UAL-P1 Live-Symthaea Diagnostic — Root-Cause Trace of the Block Collapse

**Status: follow-up diagnostic, still exploratory, not a formal UAL probe verdict.**
Follows up on `SYMTHAEA_UAL_LIVE_DIAGNOSTIC_P1_2026-07-28.md`'s frozen pilot result
(block accuracy 0.64/0.00/0.46/0.00, overall 0.275, below chance) — that doc's numbers are
**not edited or superseded here**, only explained.

**Update (2026-07-30, same day): the "most concrete lead" below (why `dopamine_effective`
reads exactly `0.0000`) has now been traced to a precise, confirmed mechanical cause — see
"Deeper root cause" at the end of this doc. Short version: `cycle_with_hv()` — the method
`CognitiveLoopBenchmarkRunner`/`run_benchmark()` uses exclusively — never calls either of
the two code paths that consume `provide_reward()`'s stored value, and only explicitly
populates `urgency`, `actual_effective_lr`, and `consciousness.consciousness_level` in its
returned `CycleMetadata`; everything else, including the entire `neuromod` sub-struct, is
`CycleMetadata::default()`. In effect, `run_benchmark()`'s reward delivery is a no-op for
any benchmark that (like this one) only drives the loop via `cycle_with_hv()`. This is a
real gap in the shared `harness::live_runner.rs`, not a UAL-specific finding — see that
section for exact call sites and blast-radius scoping.**

## Method

`p1_live_diagnostic.rs::tests::p1_live_reversal_diagnostic_trace` hand-drives the identical
sequence (`CognitiveLoopBenchmarkRunner::new("ual_p1_live_diagnostic_seed")`, dim=256,
seed=42, same 200-trial `P1LiveReversal` stimuli) via `service_mut()` directly, logging
every trial's raw cosine similarities to both arms plus `dopamine_effective`,
`prediction_error`, `consciousness_level`, and `training_loss`, instead of only the
aggregated correct/incorrect bool `run_benchmark()` returns.

**One real bug found and fixed in this trace itself, disclosed rather than hidden**: the
first version of this trace omitted `CognitiveLoopBenchmarkRunner::warmup()` (a 100-cycle
pre-trial warmup using a fixed `ContinuousHV::random(dim, 0xDEAD_BEEF)`), because
`warmup()` is private to `harness::live_runner` and not reachable from this module — an
oversight, not a decision. This produced a different, wrong block pattern
(0.58/0.98/0.00/0.00) that was *itself* reproducible (confirmed via repeated isolated runs
with `--test-threads=1` and `--exact`), which briefly looked like it might indicate hidden
non-determinism in `CognitiveLoopService` before the missing-warmup explanation was found
and confirmed by manually replicating the warmup step, which recovered the exact frozen
0.64/0.00/0.46/0.00 pattern. **Both code paths are individually deterministic and
reproducible for a fixed seed** — this is good news for the validity of every other
live-loop benchmark in this crate, not just this pilot.

## What the trace shows

- **`dopamine_effective` reads exactly `0.0000` on every one of the 200 trials**, verified
  across the entire run (not just the boundary samples) — regardless of whether the reward
  delivered that trial was `+0.8` (correct) or `-0.5` (incorrect). The reward-linked
  neuromod telemetry field this harness surfaces shows zero visible response to
  `provide_reward()` in this configuration (`ConsciousnessProfile::Standard`,
  `async_training=false`).
- **The response trajectory (`sim_a` vs `sim_b`) is a smooth, continuous, monotonic drift**,
  not a discrete reward-driven update. Around trials 40-53 (still pre-reversal, reward
  uniformly `-0.5` since the network had already locked onto the wrong arm), `sim_a` and
  `sim_b` diverge further apart every single trial by a steadily shrinking increment — a
  textbook exponential-convergence signature, not something that looks driven by a
  trial-to-trial reward-prediction-error update.
- **The response flip that produces block 2's accuracy dropping to 0 for its second half
  happens smoothly across trials 122-123** (`sim_a`/`sim_b` cross from -0.0316/-0.0316
  essentially tied, through -0.0314/-0.0319), with **no discontinuity at all around trial
  100 (the actual reward-contingency reversal) or at trial 123 itself (where the reward
  sign the network was receiving flips from consistently +0.8 to consistently -0.5)**. The
  drift trajectory before and after both of those events looks identical in shape.
- `training_loss` decreases smoothly and monotonically across the whole 200-trial run
  (roughly 0.02 near trial 40 down to ~2e-5 by trial 149, continuing to shrink through
  trial 199) with, again, no visible discontinuity at either the trial-100 reversal or any
  reward-sign flip — consistent with the network confidently converging toward a stable
  self-consistent output for the (constant) cue, not toward "predict the currently-correct
  arm."

## Interpretation (bounded)

The most parsimonious explanation consistent with all four observations above: **the
observed response trajectory in this configuration is dominated by an autonomous,
self-referential drift/convergence process on the constant trial cue, not by the delivered
reward signal.** The earlier block-level pattern (0.64 → 0.00 → 0.46 → 0.00) is fully
explained by this drift's timing relative to the block boundaries and the trial-100
reversal — not by real reward-driven reversal learning, and not by simple noise (the
trajectory is smooth and monotonic, not erratic).

This is **not** proof that live Symthaea cannot do reward-driven reversal learning in
general — it is one configuration (`ConsciousnessProfile::Standard`, `dim=256`,
`async_training=false`), one seed, one task design (a constant cue with only the reward
contingency changing). It is a real, well-supported explanation of *this specific run's*
collapse.

## Most concrete lead for anyone continuing this line — RESOLVED, see below

**Why does `dopamine_effective` read exactly `0.0000` for every trial regardless of reward
sign?** Originally flagged as an open question (whether it's a wiring gap, a
`ConsciousnessProfile::Standard`-specific property, or the wrong telemetry field to look
at). Traced to a precise, confirmed mechanical cause the same day — see "Deeper root
cause" below.

## Deeper root cause: `cycle_with_hv()` never consumes `provide_reward()` at all

Read `CognitiveLoopService::cycle_with_hv()` in full
(`src/cognitive_loop/helpers/mod.rs:278-488`, the method `CognitiveLoopBenchmarkRunner`/
`run_benchmark()` uses exclusively for every trial). Two confirmed facts:

1. **It never touches reward at all.** Grepping its full body for "reward" returns zero
   matches. The only two places `external_reward` (the field `provide_reward()` sets,
   `accessors/behavior.rs:369-372`) is ever read and consumed are
   `compute_reward_signal()` (`helpers/cycle_extracted.rs:604-644`, called from
   `cycle_phase_dynamics/training.rs:388`) and a second consumer feeding the FEP agent's
   `learn_from_outcome` (`cycle_phase_dynamics/mod.rs:2423-2430`). **Both live inside
   `cycle_phase_dynamics`, which is part of the full `cycle(text)` phase pipeline —
   `cycle_with_hv()` never calls into it.** So when a caller only ever uses
   `cycle_with_hv()` (as `run_benchmark()` does, every trial), `provide_reward()`'s stored
   value just sits in `social_mgr.social.external_reward` — never consumed, never zeroed by
   the normal consumption path, with zero effect on training or neuromod state.
2. **`cycle_with_hv()`'s returned `CycleMetadata` is mostly `CycleMetadata::default()`.**
   The exact construction (`helpers/mod.rs:~462`) explicitly sets only `urgency`,
   `actual_effective_lr`, and `consciousness.consciousness_level`; everything else —
   including the entire `neuromod` sub-struct (`dopamine_effective` among it), `ethics`,
   `structural`, oxytocin, bath entropy, allostatic load, social coherence — comes from
   `..CycleMetadata::default()`. `dopamine_effective` reading exactly `0.0000` on every
   trial is simply this default value, not a live read of `self.neuromod.bath.dopamine`
   at all (confirmed separately: `accessors/neuromodulation.rs:62` shows the *correct*
   live-read pattern, `self.neuromod.bath.dopamine.effective()`, which is not what
   `cycle_with_hv()` uses).

**Together**: `harness::live_runner.rs`'s `run_benchmark()` doc comment ("Wire trial
outcome as reward signal to neuromodulator bath. DA encodes reward prediction error")
describes intent that this exact code path does not deliver — the reward call is a
functional no-op given the harness's exclusive use of `cycle_with_hv()`, and the DA
telemetry it reports back per trial is a placeholder default, not measured state. This
fully explains this diagnostic's original finding (reward-uncorrelated smooth drift) at
the mechanical level, not just descriptively.

**Blast radius, checked**: `tests/butlin_live_integration.rs` and
`tests/live_loop_psych_bench.rs` (the only existing consumers of
`CognitiveLoopBenchmarkRunner`/`run_benchmark()` before this pilot) contain zero references
to reward anywhere — neither depends on `provide_reward()` doing anything, so this finding
does not undermine any existing test's claims. It does mean this pilot's own reward-driven
design assumption was unmet from the start, and it means any *future* benchmark built on
`run_benchmark()` that assumes reward shapes behavior would silently not get what it
expects, unless this is fixed first.

**Not fixed in this pass** — this is a real gap in shared harness code
(`crates/domains/symthaea-psych-bench/src/harness/live_runner.rs`), not something to patch
incidentally inside a diagnosis. Fixing it requires a design decision (e.g.: should
`cycle_with_hv()` gain an optional reward-processing step; should `run_benchmark()` call
the full `cycle()` pipeline instead/as well; should the misleading doc comment simply be
corrected to disclose the no-op instead) that deserves its own scoped work unit if pursued.

## What was NOT done

- No hyperparameter, profile, or dimension was changed to try to "fix" the collapse or make
  reversal learning appear.
- The harness bug found above was diagnosed and documented, not fixed — see "Deeper root
  cause" for why that's a separate decision.
- No additional seeds were run — the drift-dominates-reward *symptom* was explained
  mechanically (reward is never consumed at all via this path), which generalizes beyond
  any one seed; whether the CfC's autonomous drift dynamics themselves look different under
  other seeds/configurations remains untested.
