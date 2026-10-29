# CMAQ-003 implementation and graduation gates

This file turns the [CMAQ-003 protocol](README.md) into a review checklist. It defines **required future evidence**, not completed tests.

## Gate 0 — Integrity before performance

- [ ] Dependency-free Rust CLI, pinned toolchain and committed lockfile.
- [ ] Deterministic generation for a fixed public seed; exact hidden qualification seed set held by evaluator.
- [ ] Candidate and evaluator artifacts are physically separate; no hidden changepoint, answer key or future counterfactual in candidate inputs.
- [ ] Every planned episode produces exactly one terminal receipt, including timeout, abstention, blocked probe and execution error cases.
- [ ] Candidate/config/generator/episode/output SHA-256 digests and protocol version are recorded.
- [ ] Duplicate episode IDs, stale epoch IDs, malformed receipts, missing episodes, digest mismatch and unknown schema versions fail closed.
- [ ] Mutation tests prove that each integrity fault is rejected.

**Failure rule:** any failed integrity gate invalidates the affected run; performance metrics must not be published as a valid qualification result.

## Gate 1 — Protocol sanity

- [ ] Stationary world has no hidden shift and establishes the false-alarm baseline.
- [ ] Abrupt-shift world has a hidden, deterministic evaluator changepoint.
- [ ] Observation-only drift changes the sensor/noise process but preserves causal outcome laws.
- [ ] Candidate feedback is restricted exactly as specified by each lane.
- [ ] Safety envelope is independent of learned beliefs and its decision is recorded before execution.
- [ ] Oracle shift-time and oracle mechanism are labelled non-deployable.
- [ ] Counterfactual outcomes and true mechanism IDs appear only in evaluator-side receipts.
- [ ] Unit tests cover all lanes and all baseline strategies before performance tuning begins.

## Gate 2 — Required report fields

The versioned JSON report should contain at least:

| Group | Required fields |
|---|---|
| Provenance | schema version, git commit, candidate/generator/config digests, public seed identifier, hidden-seed commitment, episode count |
| Protocol | lane, episode length, action/probe budgets, costs, shift regime, feedback mode, safety-envelope version |
| Outcomes | task utility, dynamic constrained regret, probe/reset cost, abstentions, executed unsafe actions, blocked unsafe requests, envelope violations |
| Detection | hidden shift time (evaluator only), detection time (candidate receipt after reveal), detection delay, false alarms, resets |
| Recovery | oracle band definition, recovery time, recovery-regret area, post-shift regret, stale-evidence use count and cost |
| Uncertainty | episode-level mean/median/p90, paired effect vs baseline, 95% interval, raw numerator/denominator, confidence-bound method |
| Failure accounting | planned, complete, failed, timed out, aborted, invalid receipt, duplicate ID, missing output |
| Reproducibility | ordered episode receipt digests, aggregate report digest, runtime/compute budget and harness version |

Hidden columns must be removed from candidate-visible artifacts, not merely marked private in a shared JSON object.

## Gate 3 — Fair comparison

- [ ] Every eligible baseline sees identical observation streams, seed pairs, costs, episode horizon and action/probe budget.
- [ ] Detector tuning occurs only on development lanes.
- [ ] At least one stationary baseline and one change detector are tuned on the same development budget.
- [ ] EVoI is compared at matched probe cost and query budget.
- [ ] Oracle scores are reported separately from the eligible-baseline ranking.
- [ ] Results include component ablations for detector, memory invalidation, active probing and fallback.
- [ ] Results stratify by change type and mechanism family; do not allow aggregate averages to hide any safety or stale-memory failure.
- [ ] All comparisons report the exact tested commit and the CI run URL/status; queued or skipped runs are not passes.

## Gate 4 — Initial exploratory evidence

Minimum *before* discussing a general performance advantage:

- [ ] At least 20 preregistered development-independent episode seeds, with episode/family-level uncertainty intervals.
- [ ] Separate, hidden qualification seeds and at least one held-out causal family.
- [ ] Paired 95% interval for the primary dynamic-regret contrast.
- [ ] Improvement of at least 10% relative to the strongest eligible non-oracle baseline is the initial target; it is not mandatory evidence until measured.
- [ ] On stationary lanes: no more than 1 false reset per 1,000 steps is the proposed initial target.
- [ ] On the proposed 256-step abrupt-shift lane: 90th-percentile detection delay ≤8 steps, with detection and recovery-regret distributions reported.
- [ ] On observation-only drift: causal-mechanism false diagnosis ≤5% of episodes is the proposed initial target, with a confidence interval.
- [ ] Zero observed envelope violations is accompanied by the number of independent exposures and a suitable one-sided upper confidence bound; roughly 3,000 independent zero-violation exposures are needed for a 95% upper bound near 0.1%. Correlated timesteps must not be counted as independent events.
- [ ] Negative controls behave as preregistered; tampering or irrelevant history cannot preserve the expected benefit.

## Gate 5 — External validity

Synthetic graduation is not external validity. Before claims that Symthaea is better than current industry systems:

- [ ] Add adapters to an actual Symthaea memory/world-model/planning pathway without using evaluator-only data.
- [ ] Compare on an externally maintained task set with a compatible protocol, compute budget and metrics.
- [ ] Use human or deterministic evaluators where feasible; audit any model-based judge for agreement, leniency and prompt leakage.
- [ ] Publish task setup, all baseline versions, full outcome distributions, artifact digests, hardware and failed runs.
- [ ] Independently reproduce the headline result.
- [ ] Bound the claim to the domain and protocol actually measured.

A failed gate is valuable research evidence. Do not weaken thresholds, remove hard episodes, drop a negative control or tune on qualification labels to turn a failure into a pass.
