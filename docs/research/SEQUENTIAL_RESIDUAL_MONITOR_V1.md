# Sequential Residual Monitor v1

**Status:** synthetic research prototype; not a control-safety certificate or external validation result.

## Motivation

The score-preserving stealth-shift experiment shows a fundamental limitation: a detector cannot reliably detect a change that leaves every input available to that detector unchanged. An episode-level novelty score is useful when it carries shift information, but it is not a proof that the dynamics model remains valid.

This extension adds a second evidence channel: the sequence of normalized one-step prediction residuals. A shift that is invisible to the novelty score may still become observable after the real system takes an action and its measured transition disagrees with the frozen prediction. The monitor should then request a reachability fallback; the residual alarm itself must never be treated as proof that an action is safe.

## Sequential statistic

For a finite, non-negative normalized residual (r_t), update a one-sided cumulative statistic:

\[
g_t = \max(0,\ g_{t-1} + r_t - k), \qquad g_0 = 0,
\]

where (k\) is a predeclared allowance for nominal residual noise. At the episode boundary, calibrate one score per nominal calibration episode: the maximum (G = \max_t g_t\). Choose the threshold using the finite-sample order statistic

\[
q = G_{(\lceil (n+1)(1-\alpha)\rceil)}.
\]

The monitor latches an alarm when (g_t > q). It resets only at a declared episode boundary. Invalid, negative, or non-finite residual inputs return an explicit error-like `None` result without mutating state; callers must treat the loss of monitoring evidence as uncertified. A threshold cannot be produced when calibration is empty, invalid, or too small for the requested rank.

The episode-level maximum is important: calibrating a pointwise threshold and checking it repeatedly at every time step would not, by itself, control the probability of at least one false alarm over an episode. This procedure's nominal false-alarm interpretation still depends on a frozen score construction and exchangeable independent calibration and test episodes. It does not provide a distribution-free guarantee under arbitrary shift.

## Synthetic experiment

The Rust benchmark compares the existing episode-calibrated OOD score with the sequential residual monitor over five operating-point fixtures. For each fixture it runs 300 paired episode seeds under five conditions:

| Condition | OOD score changed? | Observed residual changed after step 12? | Purpose |
|---|---:|---:|---|
| nominal | No | No | Held-out false-alarm check |
| residual-only shift | No | +0.10 | Test sensitivity to weaker persistent evidence |
| observable strong shift | Yes | +0.30 | Positive control for both channels |
| stealth-score shift | No | +0.30 | Dynamics shift that the score-only monitor cannot see |
| sensor-blind shift | No | No | Negative control where both evidence channels are unchanged |

The same per-step nominal score and residual are reused for the matching operating point / episode seed across paired conditions. Therefore, the score-only alarms must match the nominal decisions in residual-only, stealth-score, and sensor-blind conditions; a strong score-visible shift is the positive control. The sensor-blind condition must retain the same residual alarm step as its nominal pair. These are fixture-accounting constraints, not a claim that the synthetic shift families represent all physical failures.

## Required interpretation

- Report episode-level false alarms and shift misses separately.
- Report detection delay from the first shifted time step, not only whether an alarm occurred.
- A residual-only alarm can only occur after enough observed transitions. Do not report it as pre-emptive protection for the first unobserved shifted transition.
- A sensor-blind shift is intentionally not detectable by the two provided signal channels. Detection requires additional independent evidence or a validated physical assumption.
- The benchmark should show both positive-control response and blind-spot behavior. It must not be tuned to require detection of a shift that the monitor cannot observe.
- This module has no action authority and does not modify a live Symthaea control loop. A production controller would need to combine the monitor with an independently verified multi-step fallback and a fail-closed transition on invalid monitoring state.

## Files and reproducibility

- `src/innovation_monitor.rs`: threshold calibration, CUSUM state, latched alarm, validation, reset, unit tests.
- `src/bin/allostatic-residual-monitor-bench.rs`: deterministic synthetic panel, paired policy-evidence streams, raw episode CSV and per-condition summary CSV.
- `qualify.sh`: formatting, tests, Clippy, byte-identical repetitions, row/schema checks, paired negative controls and output/source digests.

The panel has 7,500 episode rows (5 operating points × 5 conditions × 300 episodes) and 25 summary rows. Raw results retain the seed, first shift step, score and residual thresholds, and both alarm steps.

## Exit criteria before external claims

Run this exact-head Rust qualification first. Next, define task-specific normalized innovations from a pinned physics environment, freeze that residual transformation before calibration, and independently verify actual transitions. No detection rate from this synthetic scalar panel is evidence of SOTA performance, physical safety, or transfer to CartPole / Quadrotor.
