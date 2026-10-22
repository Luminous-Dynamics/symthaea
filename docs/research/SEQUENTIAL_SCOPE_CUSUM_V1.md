# Sequential CUSUM Monitoring for Task-Relevant Scope Drift V1

Status: isolated Rust prototype, benchmark runner and qualification gates. The exact-head run must pass before this branch is called qualified. Nothing here changes the live Symthaea control loop.

## Paradigm shift: monitor evidence over time

A one-step residual gate is good at a large abrupt model error, but it discards history. Small errors can be individually consistent with nominal noise while their *persistent excess* signals a changed system. This branch adds a cumulative-sum (CUSUM) monitor:

`S_t = max(0, S_(t-1) + r_t - k)`

where `r_t` is the task-relevant residual score and `k` is a declared per-step allowance. The alarm threshold is calibrated on the episode-maximum CUSUM statistic, separately from the pointwise residual threshold. These thresholds have different distributions and must not be interchanged.

This follows the general direction of sequential, anytime risk monitoring under unknown shift. Timans et al., UAI 2025, study testing-by-betting approaches for continuous risk-violation monitoring under dynamic distribution shifts ([paper](https://proceedings.mlr.press/v286/timans25a.html)). This prototype is a simpler calibrated CUSUM baseline; it is not a reproduction of that method.

## Implementation

- `ScopeCusumAccumulator` accumulates positive residual excess above a fixed allowance and can reset independently of the decision threshold.
- `ScopeCusumMonitor` consumes production `PredictionErrorScore` values and latches an alarm when the cumulative statistic exceeds its separately calibrated threshold. Like the pointwise scope monitor, it also supports an immediate false-safe override.
- The monitor is post-action: it recognizes a model error from the observed transition and can inform the *next* control decision. It cannot prevent the transition that already occurred.
- The benchmark compares pointwise residual alarms with CUSUM alarms on identical eight-step traces, at independently calibrated thresholds.
- Calibration uses 159 nominal calibration seeds; evaluation uses 300 held-out seeds for five operating-point labels and four conditions, resulting in 48,000 auditable step rows and 20 state/condition summary cells.

## Falsification fixture

The score generator has four conditions:
- **Nominal:** low residual noise plus rare single-step spikes.
- **Gradual shift:** residuals rise across the eight-step episode but are designed to remain individually below the pointwise threshold.
- **Abrupt shift:** the first three steps follow nominal noise, then residual magnitude jumps.
- **Stealth shift:** the full score sequence is paired exactly with the nominal sequence, while the scenario label represents an unobserved actual dynamics shift.

The stealth condition is not a detector-power test; it is a necessary negative control. When the monitored evidence is exactly the same as nominal, neither pointwise nor cumulative residual logic has new evidence to distinguish it.

The raw output includes the per-step score, the pointwise threshold and alarm, cumulative statistic and calibrated threshold, latched alarm, first alarm step, shift label, and whether every score in the episode remained below the pointwise threshold. The summary exposes alarms/misses, per-cell nominal false alarms, detection delay percentiles, which monitor was earlier when both detected, and how often CUSUM caught an episode whose every individual score was under the pointwise threshold.

## Acceptance gates (software fixture, not real-world claims)

- Two executions produce byte-identical raw and summary CSVs.
- Pointwise and CUSUM thresholds are calibrated separately from nominal episode maxima; the CUSUM threshold is computed on maximum cumulative statistics.
- Nominal false alarms stay at or below 10% per operating-state cell for both monitors.
- In the designed gradual-shift condition, every episode remains below the pointwise threshold, pointwise detection is zero, and CUSUM detects at least 80% of episodes.
- In the abrupt-shift control, each monitor detects at least 90% of episodes.
- In the stealth condition, residual scores and alarm decisions exactly match the nominal paired seeds; both monitors consequently miss the latent shift in most episodes.
- Raw schema, step coverage, CUSUM recurrence, paired-stealth equality, summary arithmetic, row counts and source/output hashes are checked.

The designed fixture deliberately favors the intended sequential signal to test that the implementation accumulates it correctly. The measured rates are not estimates of generalization. Independent episodes, held-out dynamics families and physical or high-fidelity control tasks are still required.

## How to compare with industry methods

The near-term test is not “beat SOTA” on a synthetic CUSUM stream. It is to compare pointwise residual, CUSUM, a calibrated conformal/e-value sequential baseline and ensemble-disagreement monitors at a matched nominal false-alarm rate. Keep the same action sequence, control authority, seed families and runtime budget.

Relevant external reference points include:
- [On Continuous Monitoring of Risk Violations under Unknown Shift (UAI 2025)](https://proceedings.mlr.press/v286/timans25a.html), focused on sequential risk monitoring under dynamic shift.
- [SODA-MPC (L4DC 2025)](https://proceedings.mlr.press/v283/contreras25a.html), which combines conformalized OOD monitoring with a reachability fallback in autonomous driving.
- [Robust OOD MPC via Conformalized System Level Synthesis (L4DC 2026)](https://proceedings.mlr.press/v331/srinivasan26a.html), which uses weighted conformal bounds, state/control-dependent covariance and optimized reachable sets.

The next credible benchmark should use a pinned Safe-Control-Gym or equivalent physical/high-fidelity task and separately hold out:
1. calibration episodes;
2. hyperparameter-selection disturbance families;
3. final evaluation disturbances, including gradual parameter drift, sensor degradation, delayed telemetry and shifts that are hard for the feature-space detector.

Report false-alarm probability and uncertainty, miss rate, detection delay, false-safe exposure before fallback, no-certified-action frequency, task reward, constraint violation severity, and worst-case runtime. Compare the pointwise and sequential methods at matched false-alarm budget. If CUSUM improves gradual-shift detection but worsens abrupt detection or nominal false alarms, publish that trade-off rather than choosing one pooled score.

## Remaining gaps

- The allowance is fixed rather than state-dependent; it needs sensitivity sweeps and validation on held-out nominal residual distributions.
- The CUSUM statistic currently accumulates a scalar normalized task residual. A later version should handle dependence explicitly and compare with testing-by-betting / e-value monitors.
- This prototype is not a controller or safety certificate. Its output is evidence for a supervisor; robust reachability remains the separate action-selection gate.
