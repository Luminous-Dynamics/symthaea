# VCCR-001 — Bounded Closed-Loop Vocal Correction Trajectory Benchmark

**Date:** 10 November 2026  
**Status:** standalone trajectory evaluator and synthetic fixtures; no renderer or closed-loop controller performance is claimed  
**Owner:** [Issue #11482](https://github.com/Luminous-Dynamics/symthaea/issues/11482)

## Research question

Can a correction loop move a fixed musical performance into its declared acoustic acceptance region without overshooting, oscillating, stalling, changing protected properties, or making further edits after the evidence no longer justifies them?

## Paradigm shift: qualify the trajectory, not just the final waveform

A final render may satisfy a pitch or pronunciation target even though the system took unsafe or wasteful intermediate steps. Conversely, an individual correction may improve one metric while progressively damaging another.

VCCR-001 records the whole sequence:

~~~text
frozen source + baseline render
  -> bounded action / patch
  -> new render with parent-output identity
  -> independent target and collateral observations
  -> uncertainty-aware trajectory decision
  -> next bounded action OR stop
~~~

The evaluator consumes supplied receipts. It does not render, estimate a Jacobian, choose the next intervention, or create uncertainty intervals.

## Decision policy

Each target has its own accepted interval, minimum required progress and maximum allowed regression. A target measurement is:
- compliant when the full uncertainty interval is within the accepted region;
- needs correction when the interval is wholly outside it;
- indeterminate when it overlaps a boundary.

Each protected metric has a symmetric absolute-delta tolerance. Its interval must be wholly inside the tolerance on every iteration. A protected metric that is definitely outside makes the trajectory noncompliant; a boundary overlap makes the trajectory indeterminate.

At every correction:
- no previously compliant target may become noncompliant;
- every target's conservative distance-to-acceptance stays within its declared regression allowance;
- at least one unsatisfied target improves by its declared minimum or becomes compliant;
- the action norm stays within the predeclared maximum under a versioned action-norm profile that defines control coordinates and normalization;
- the output must bind to the previous iteration's exact output digest;
- source case and frozen context remain unchanged.

The first converged, indeterminate, noncompliant or stalled iteration is terminal. A later mutation in the same trace is rejected. If the maximum correction budget expires first, the result is ITERATION_LIMIT_REACHED, never converged.

This is a conservative engineering contract, not a universal control-theory optimality theorem. Different backends and metric spaces require separately calibrated distance, progress and action norms.

## Required experiment arms

On a fixed held-out set of supported controls, compare:
1. one-shot/manual correction;
2. fixed-step correction baseline;
3. bounded proportional correction;
4. regularized local-response/Jacobian correction;
5. constrained multi-metric correction;
6. no-op/sham controller;
7. wrong-sign and adversarial controller.

Keep the exact source performance, renderer/model/voicebank, analyzer, score, pronunciation, seed policy and correction budget matched. Where a stochastic renderer uses noise, preserve the seed/noise identity when the intended contrast requires it; record when exact pairing is unavailable.

## Mandatory challenge trajectories

- monotonic convergence into every target range;
- one target improves while another regresses beyond allowance;
- target overshoot and reversal;
- alternating two-step oscillation;
- no-op action with unchanged output digest;
- non-zero action with no measurable target progress;
- action magnitude beyond its declared bound;
- collateral metric definitely outside tolerance;
- collateral interval overlapping a tolerance boundary;
- target interval overlapping its acceptance boundary;
- changed backend/analyzer/context identity mid-trajectory;
- stale parent-output digest;
- missing, extra, duplicate or wrong-unit metric;
- an additional mutation after convergence or indeterminate evidence;
- correction budget exhausted before convergence.

Keep every case in the denominator.

## Metrics

Report separately:
- verified convergence rate;
- correction count distribution (median, p90 and maximum);
- per-target distance-to-acceptance before/after each step;
- maximum target regression by metric;
- collateral violation count and magnitude by metric;
- stalled, indeterminate, noncompliant and limit-reached fractions;
- false-convergence rate on injected wrong-output controls;
- correction action norms and output-change fractions;
- p50/p95 end-to-end latency, render time and compute cost;
- human-perceived requested-control compliance and naturalness on a blinded sample.

Do not collapse these into one weighted controller score.

## Statistical design

- Freeze controller policy and thresholds on development data.
- Use disjoint singer/song/session cases for calibration and confirmation.
- Treat singer/song/session as the sampling cluster; iterations and overlapping frames from one render are repeated measurements, not independent examples.
- Use paired source-case comparisons among controllers; report cluster-aware uncertainty.
- Use no-op repeats to calibrate collateral bounds before confirmation.
- Use simultaneous uncertainty across target and collateral metrics.
- Preserve every failed, empty, stalled, indeterminate and budget-exhausted trajectory in the denominator.
- Estimate sample size from a pilot and preregister primary metrics, minimum detectable effects and exclusion policy.

## Competitive positioning

Hume's Voice Controllability Leaderboard uses blinded human judgments to ask whether voice models follow instructions; it is speech-focused. SoulX-Singer-Eval provides a zero-shot singing baseline suite with SingMOS-Pro, Sheet-SSQA, MCD, WER/CER, speaker similarity and FFE/GPE/VDE. MusicJudge (Interspeech 2026) couples lyric correctness to pitch/rhythm in block-aligned assessment. These are useful adjacent benchmarks, but none of their headline scores measures closed-loop convergence, collateral preservation across correction iterations, or correct stopping under uncertain acoustic evidence.

VCCR therefore adds a new engineering qualification track rather than trying to replace MOS or a generation leaderboard. It should be evaluated alongside, not instead of, the industry's quality, linguistic, musical and human-preference measurements.

## Proposed held-out qualification targets

Targets to calibrate after the pilot, not observed results:
- at least 90% verified convergence for controls declared supported, progressing to 95%;
- zero protected-property violations in the verified-converged cohort;
- zero false convergence in synthetic no-op, wrong-direction and stale-output controls;
- zero actions accepted after convergence or indeterminate evidence;
- report correction-count p90 and all terminal-outcome fractions;
- improve correction count or total render cost over fixed-step correction at matched final-quality and preservation bounds.

A zero observed synthetic false-pass count does not imply zero real-world false-convergence risk. Report an upper confidence bound on the latter using independent source cases.

## Relation to existing Muse work

- #11401 VCRB-001 measures target response and acoustic cross-coupling.
- #11420 VNOC-001 calibrates no-op collateral tolerances.
- #11424 removes score-conditioned clipping from the primary F0 audit.
- #11425 qualifies independent F0 on held-out vocal references.
- #11456 / PR #11461 makes uncertain metric intervals indeterminate rather than passing.
- #11354 VELB-001 checks waveform locality.
- #11356 owns the future renderer-neutral patch and preservation receipt.
- #11402 is the planned versioned observation adapter.

## Claim ceiling

A synthetic VCCR pass establishes only that one supplied trajectory satisfies the contract under its declared metrics and intervals. It does not establish that the renderer follows real controls, the uncertainty method is calibrated, the controller generalizes, the audio is natural, or Muse is SOTA. Those require held-out real-vocal experiments and blinded human assessment.
