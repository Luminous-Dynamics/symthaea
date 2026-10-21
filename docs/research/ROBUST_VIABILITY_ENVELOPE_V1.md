# Robust Viability Envelope V1

Status: isolated Rust research prototype and deterministic synthetic falsification harness; exact-head qualification is still required.

Follow-up to:
- [Allostatic control benchmark PR #8240](https://github.com/Luminous-Dynamics/symthaea/pull/8240)
- [Issue #8243: qualify allostatic viability control under model mismatch](https://github.com/Luminous-Dynamics/symthaea/issues/8243)

## Research question

Can the allostatic controller make more defensible safety decisions by representing state and action effects as explicit bounded uncertainty sets, instead of converting forecast confidence into a point forecast adjustment?

## Paradigm shift: from confidence score to uncertainty set

A confidence scalar is not a bound on model error. The prototype treats uncertainty as explicit intervals and adds episode-level split-conformal calibration for the interval radii:

- state envelope: reserve and allostatic-load intervals;
- action-effect envelope: possible reserve and load deltas under each action;
- viability limits: minimum reserve and maximum allowed load;
- authority: whether the controller is permitted to invoke emergency recovery;
- calibration: a frozen predictor is evaluated on independent calibration episodes, and four score families capture maximum absolute residuals over each episode for reserve state, load state, reserve action effect, and load action effect. Effect scores must cover every candidate action and timestep. Four-way Bonferroni-adjusted component coverage produces a joint state-and-action-effect envelope target.

For one-step interval propagation, a candidate action is robustly viable only when:

\[
\underline r_{t+1} \ge r_{\min}
\qquad\text{and}\qquad
\overline \ell_{t+1} \le \ell_{\max}.
\]

Here the lower next-reserve bound and upper next-load bound are computed over the declared interval set. State variables are clipped to their declared [0, 1] domain after interval addition, matching the synthetic simulator's saturation semantics.

The calibration executable contract is in `tools/allostatic-control-bench/src/calibration.rs`. For target joint coverage C across four residual families, it uses per-family coverage (1-(1-C)/4) and a finite split-conformal order statistic. If the requested rank exceeds the number of independent calibration episodes, it returns `None` rather than inventing a finite radius. That rule creates a useful minimum-sample-size gate: at least 39 exchangeable calibration episodes are required for a 90% joint-coverage target, and at least 79 for a 95% joint-coverage target.

These intervals must be calibrated with independent episode-level data or justified by a conservative physical argument in a real deployment. The current unit tests use synthetic scores and validate only the software contract; they do not establish empirical coverage. State and action-effect intervals are both widened by their calibrated residual radii. The finite-sample calibration statement assumes exchangeable calibration/test episodes and a predictor frozen before calibration. Distribution shift, serial leakage across split groups, under-bounded model classes, or a changing online predictor voids that interpretation.

## Implementation

Sources: `tools/allostatic-control-bench/src/robust_viability.rs`, `tools/allostatic-control-bench/src/calibration.rs`, and `tools/allostatic-control-bench/src/model_mismatch.rs`.

- `Interval` rejects nonfinite and inverted ranges.
- `calibrate_episode_radii` uses episode-level maximum residuals and Bonferroni-adjusts reserve-state, load-state, reserve-effect, and load-effect coverage to target simultaneous envelope coverage. Too few calibration episodes returns `None`; it never invents a finite radius.
- `model_mismatch.rs` keeps the nominal and true transition models separate, builds residual scores over all candidate actions and stress levels, and compares point certification, robust certification, and realized safety. Its synthetic calibration panel combines one-axis extrema and mixed eight-coordinate perturbations; it is a software falsification panel, not an empirical calibration dataset or a calibrated real-world dynamics model.
- `SafetyAvailabilityMetrics` reports candidate-action count, point/robust certifications, true-safe actions, false certifications, and false vetoes separately so a filter cannot look successful merely by rejecting every action.
- `StateEnvelope` constrains reserve/load intervals to [0, 1].
- `ActionEffectBounds` constrains one-step reserve/load deltas to [-1, 1].
- `ViabilityLimits` validates the reserve floor and load ceiling independently; they represent different axes and must not be compared to each other.
- `evaluate_action_set` returns Push/Maintain feasibility under the complete declared envelope. EmergencyRecover is included only when explicitly authorized.
- `RobustActionSet::filter` keeps a proposed action only if its whole next-state envelope is viable. It tries conservative authorized fallbacks and returns `None` when no candidate is certified viable; it does not label a best-effort action as safe.

The interval bounds use a rectangular uncertainty set: reserve and load changes are allowed to vary independently within their declared ranges. This is conservative when those effects are correlated and can unnecessarily shrink the safe action set. Polytopes or zonotopes that preserve validated correlations are a later optimization, not part of v1.

## Testable contracts

The combined unit tests cover:
1. Rejection of invalid/nonfinite/inverted intervals and invalid state bounds.
2. A point/midpoint-safe action rejected when its worst-case load effect exceeds the limit.
3. State-estimation uncertainty that straddles a reserve floor.
4. Monotonicity: widening an action's uncertainty bounds cannot make that action newly robustly viable.
5. Emergency-action authorization kept separate from physical feasibility.
6. No safe action is reported when every candidate violates at least one bound.
7. A safe proposal is preserved and unsafe proposals use only a fallback that is itself robustly viable.
8. Episode-level calibration uses the correct order statistic, rejects insufficient/nonfinite/mismatched samples, expands both state and action-effect bounds conservatively, and never shrinks an envelope when the residual radii increase.
9. A deliberately misspecified point shield can false-certify an unsafe action; a calibration envelope covering the held-out residual must reject it.
10. An explicitly out-of-calibration-family perturbation acts as a negative control, and safety certification is scored separately from false-veto/availability cost.

These tests are authored but do not count as passing until the exact-head Rust qualification workflow compiles and executes them.

## Qualification and comparison plan

First qualify the interval arithmetic and action-set semantics in the existing dependency-free Rust crate. The new `model_mismatch.rs` now supplies the first deterministic true-versus-estimated model falsification panel: zero error, both extrema of eight independent error coordinates, and mixed perturbation combinations. Extend it under issue #8243 with:

- true transition dynamics and shield-estimated dynamics must be distinct values;
- vary state-estimation width and action-effect uncertainty separately;
- report the robust safety–availability frontier across uncertainty width: false certification rate and false-veto rate versus task reward and available action count;
- test under-bounded envelopes as explicit negative controls (they should reveal why bounds must be calibrated);
- include missing telemetry, biased warnings, delayed warnings, and unannounced regime changes;
- compare no shield, point-estimate shield, interval-envelope shield, threshold control, and independently implemented MPC/control-barrier filters;
- report false vetoes, missed constraints, no-certified-action frequency, task reward, reserve, load, viable-action breadth, recovery and p95/worst-case outcomes;
- freeze held-out uncertainty scenarios before tuning and publish raw results plus deterministic provenance.

The existing benchmark can provide a local accounting harness. It now separates episodes with no qualifying disturbance from disturbed episodes whose recovery is right-censored; report recovery rates over disturbed episodes only. This is important because a calm episode cannot be counted as a failed recovery. For an external control comparison, adapt only where observation/action semantics match. Safe-Control-Gym includes symbolic safety constraints, injected disturbances, MPC variants and control-barrier safety filters; SafeRLEval recommends that safety evaluations report violation rate and severity alongside reward, rather than only average cost. References:
- [Safe-Control-Gym](https://github.com/learnsyslab/safe-control-gym)
- [SafeRLEval / Evaluation Metrics for Safe Reinforcement Learning (September 2026)](https://arxiv.org/abs/2609.15315)

This prototype does not reproduce or outperform those benchmarks. It introduces a small local testable hypothesis that must later be evaluated in the external environment.

## Proposed quantitative benchmark gates (targets, not current results)

The current Rust frontier runner emits 300 deterministically seeded profiles within the declared mismatch envelope and 32 out-of-envelope negative-control profiles, each across five stress levels and three candidate actions (4,980 action rows total). It must show zero false certifications on the in-envelope panel and report false-veto/action-availability costs. Because these are deterministic synthetic profiles rather than independent physical trials, they validate the software contract but do not support a statistical confidence claim.

- **Coverage:** evaluate the nominal 90% and 95% joint envelopes on untouched exchangeable episodes. Report empirical coverage and a binomial confidence interval separately for every perturbation cell; never use the same episodes to fit radii and claim test coverage.
- **Within-envelope safety (next empirical stage):** target zero false certifications in 300 genuinely independent held-out episodes per perturbation cell. With zero events, the rough 95% 'rule of three' upper bound is about 1% for that cell; report the interval, not just the zero count. Do not apply this statistical interpretation to the deterministic current grid.
- **Availability cost:** target no more than 5% false vetoes among truly safe candidate actions on in-envelope data and no more than a 5% task-reward drop versus the strongest safe baseline in nominal conditions. If these targets conflict with the risk bound, publish the frontier rather than hide the trade-off.
- **Shift robustness:** target at least 20% lower p95 normalized violation exposure than a point-estimate shield under the predeclared mismatch grid, at matched reward/authority budget. This is a go/no-go research target, not an asserted result.
- **Stress matrix:** pre-register reserve-drain, load-growth, and recovery mismatch at ±5/±10/±20%; state bias and noise; 0/5/20% missing telemetry; false, missed, delayed, and severity-biased warnings; plus fully unannounced regime shifts.
- **External comparison:** pin Safe-Control-Gym/Safety-Gymnasium commit, task, integration frequency, disturbance settings, controller and safety-filter variants, and tuning budget. Compare only where dynamics, action spaces, and constraint semantics can be mapped honestly.

## Acceptance gates for the next experiment

- Zero robustly certified actions may violate the declared one-step constraints when the true dynamics lie within the stated envelope.
- Widening the uncertainty envelope may shrink the certified action set or leave it unchanged, never add a new certified action.
- Under-bounded envelopes must demonstrate failures in explicit negative controls so certification assumptions are visible.
- Report loss of availability separately from violation: a conservative shield may be constraint-safe but too restrictive to be useful.
- Under held-out, calibrated model mismatch, compare task reward and false-veto cost with a point-estimate shield; do not claim the interval method is better merely because its safe set is smaller.
- Do not claim a universal safety guarantee for omitted model error, invalid sensor bounds, finite test suites, or physical systems.

## Scope boundary

This is an engineering model of capability/viability, not a model of subjective distress, biological allostasis, feeling, suffering, or consciousness. It must not replace Symthaea's task or prediction losses and must not change qualified interoception v0.1 semantics as part of this experiment.