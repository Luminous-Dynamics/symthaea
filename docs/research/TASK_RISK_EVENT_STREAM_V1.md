# Persistent Task-Risk Event Stream V1

Status: isolated synthetic research benchmark, not a live-control change. Exact-head qualification is required before the output can be called qualified.

## Why this is a different signal

The preceding e-process experiment monitored a normalized model-residual score. That is useful for testing whether dynamics prediction errors accumulate, but its risk null is a bound on expected residual loss; it is not a direct statement about constraint violations.

This extension monitors an outcome that is closer to the control objective: whether an episode contains at least one transition for which the predictor said the next state met reserve/load limits but the observed next state violated them. This event is called a false-safe episode in the upstream task-scope benchmark. It is a bounded binary loss:

- (L_i=1) if episode (i) contains an observed false-safe transition.
- (L_i=0) otherwise.

The monitor tests the explicit conditional-risk null (mathbb{E}[L_imidmathcal{F}_{i-1}]leqepsilon) with (epsilon=0.05). It is not an OOD classifier. The event is recorded after the transition, so the alarm cannot prevent that same transition; it can inform a supervisor about subsequent episodes.

## Persistent stream design

- The input is the task-scope raw CSV, 5 operating points × 4 conditions × 300 episodes = 6,000 episode records.
- For each operating-point/condition stream, the e-process **persists across all 300 episodes**. It resets only at the beginning of that evaluation stream, not every 8-step episode.
- Five operating points are treated as parallel monitors, with total (delta=0.05) allocated by the union bound. The e-value threshold is (K/delta=100), or (log(100)) in log space.
- The primary estimator is the fixed equal-weight mixture of three predeclared bet fractions ([0.25,0.50,0.75]) of (1/epsilon). A single fixed bet with fraction 0.50 is an ablation, evaluated on exactly the same binary event stream.
- The raw output preserves the first 19 task-scope columns and appends the observed false-safe event, mixture log-evidence and component logs, fixed-bet log-evidence, alert states, first alert episode and frozen risk parameters.
- The summary reports false-safe episode count/rate, first event, first mixture/fixed alarm, detection delay from first event, number of false-safe events observed by the alarm, and final evidence.

The four conditions are evaluation strata, not four simultaneous systems sharing one budget. The (delta=0.05) familywise statement here applies per 300-episode cohort across the five operating-point monitors under the stated conditional-risk null. It is **not** a global guarantee across all counterfactual condition families, repeated cohorts, or a deployment lifetime. A real continuously monitored system must maintain the evidence stream through its actual lifetime or allocate a fresh explicit error budget on reset.

## Important paired negative control

In the upstream task-scope fixture, strong-shift and stealth-shift cases have the same underlying synthetic actual dynamics and observation noise. Their feature-space OOD scores differ, but observed false-safe outcomes do not. This runner never consumes feature scores for risk-monitoring decisions.

The qualifier therefore requires strong and stealth streams to match for the event loss, mixture component logs, mixture/fixed evidence, alarm states and alarm indices. If those risk-stream results diverge, the outcome monitor has accidentally become dependent on an irrelevant detector signal or on condition-specific bookkeeping.

## Intended gates

- Exact 6,000-row/20-stream coverage, unique episode indices and preserved source columns.
- Every binary event is consistent with both the first false-safe transition index and the upstream override flag.
- The raw e-process components, mixture log-sum-exp, fixed-bet recurrence, threshold and alarm latching are recomputed independently in the qualifier.
- Summary event counts, rates, first-event/first-alarm indices, delay and final evidence agree with the raw stream.
- Nominal episodes have no false-safe events in this fixture and neither outcome-risk monitor alarms in nominal streams.
- Strong and stealth risk evidence matches exactly, despite feature-score differences.
- Repeated raw and summary outputs are byte-identical; their SHA-256 hashes and row counts are bound into the exact-head receipt.

These are deterministic software-fixture gates. They are not estimates of operational false-safe probability or evidence that the conditional-risk null holds outside the fixture.

## How this fits the research frontier

This is a useful bridge between statistical risk monitoring and runtime assurance, but the full research problem remains open:
- [Timans et al., UAI 2025](https://proceedings.mlr.press/v286/timans25a.html) develop testing-by-betting methods for monitoring bounded-risk violations under unknown shift. This implementation is a small predeclared mixture baseline, not a reproduction of the paper's full methods.
- [SODA-MPC, L4DC 2025](https://proceedings.mlr.press/v283/contreras25a.html) combine a conformalized OOD monitor with a reachability-based fallback in autonomous-driving simulation.
- [Srinivasan et al., L4DC 2026](https://proceedings.mlr.press/v331/srinivasan26a.html) integrate weighted conformal model-error bounds with state/control-dependent covariance and forward reachable sets for robust OOD MPC.

None of those links establishes that this synthetic scalar benchmark is competitive. A fair SOTA comparison requires a pinned physical or high-fidelity control task, the published dynamics/model/baselines, disjoint development and test disturbance families, matched false-alarm budgets, and metrics for task reward, constraint exposure, no-certified-action rate, detection delay and worst-case runtime.

## Next experiments

1. Sweep epsilon and mixture fractions on a preregistered development set, then freeze them before held-out evaluation.
2. Add event severity as a second bounded loss, while retaining event frequency as a separate reported metric.
3. Include autocorrelated disturbances, sensor faults, missing/delayed telemetry, and model changes that leave feature-score distributions unchanged.
4. Add uncertainty intervals across independent disturbance families; repeated episodes from the same synthetic transition template do not count as independent evidence of generalization.
5. Integrate the risk monitor with the reachability supervisor only after the monitor and action filter are separately validated.

The target is fewer task-critical unsafe outcomes at a matched false-alarm budget without unacceptable reward or latency loss. This branch makes that test more direct; it does not claim a SOTA win or real-world safety.
