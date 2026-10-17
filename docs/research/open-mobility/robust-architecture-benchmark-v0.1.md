# Open Mobility — Robust, Uncertainty-Aware Architecture Benchmark v0.1

**Status:** research protocol draft  
**Tracking issue:** [MOBILITY-ASSIST-009 / #7890](https://github.com/Luminous-Dynamics/symthaea/issues/7890)  
**Implementation:** draft [PR for interval-aware Pareto screening](https://github.com/Luminous-Dynamics/symthaea/pull/7892)  
**Point-estimate comparator:** [MOBILITY-ASSIST-008 / PR #7882](https://github.com/Luminous-Dynamics/symthaea/pull/7882)  
**Synthetic corpus:** [architecture-robustness-scenarios-v1.json](./fixtures/architecture-robustness-scenarios-v1.json)

## 1. Why the next shift is uncertainty-aware

A point-estimate Pareto frontier is a useful first filter, but the comparison can be brittle when task success, mass, energy, comfort, restriction or service time are estimated with uncertainty. A candidate can appear to dominate another because the report hides measurement error or person-to-person/context variation inside a single number.

The objective is not to replace trade-offs with a probability-flavoured winner. It is to distinguish:

- **Robust dominance:** candidate A is no worse than candidate B across the full declared uncertainty envelope for every required objective, and is strictly better on at least one.
- **Possible dominance:** a relationship could hold for at least one assignment within the declared independent intervals. This is a logical possibility, not a likelihood.
- **Unresolved comparison:** neither candidate robustly dominates the other. Keep both in the conservative frontier unless a user-defined constraint or further evidence resolves the trade-off.

Research motivating the change:

- A 2026 performance-evaluation review identifies fragmented evaluation indicators, inter-subject variability, and under-explored safety, stability, comfort and environmental adaptability: https://www.sciencedirect.com/science/article/pii/S0921889026003179
- A 2026 human-centred optimization perspective highlights the narrow scope of many human-in-the-loop approaches and the need to represent individual capacities and personalized training: https://pubmed.ncbi.nlm.nih.gov/42657862/
- A May 2026 user-preference-based tuning study explored 30.5 assistance settings on average, taking 10.9 ± 0.9 minutes for participants to reach a preferred setting. It motivates making preference and configuration burdens explicit; the small participant group does not establish a universal time target: https://www.nature.com/articles/s44385-026-00085-7
- ISO 18646-6:2026 adds a mechanical performance-testing reference for lower-limb wearable robots using an anthropomorphic test dummy. It complements this decision-support protocol; it does not validate interval estimates, user benefit or safety: https://www.iso.org/standard/88497.html

## 2. Formal comparison rules

Each candidate supplies one finite, closed interval for every objective in the point-estimate evaluator. V1 accepts only `declared_conservative_envelope` semantics.

For a minimizing metric, A is robustly no worse than B only if:

`upper(A) <= lower(B) + epsilon`

For a maximizing metric, A is robustly no worse than B only if:

`lower(A) >= upper(B) - epsilon`

A robust dominance claim requires the relevant no-worse condition on **all** metrics and a strict separation exceeding the numerical comparison tolerance on at least one metric. The tolerance is a floating-point comparison tolerance; it is not a measurement-uncertainty allowance.

The potential-dominance check reverses the bounds to ask whether there exists a compatible assignment inside the box intervals. Its answer is strictly logical under the declared interval model. It does not estimate frequency, probability, confidence or risk.

### Required integrity rules

- All 15 objective intervals appear exactly once per candidate.
- Every bound is finite; lower bound does not exceed upper bound.
- Non-negative metrics cannot have negative bounds; normalized task-success, movement-restriction and comfort bounds stay in [0, 1].
- Segment-mass envelopes must overlap the wearer-borne mass envelope within a 0.01 kg screening tolerance.
- The lower bound on total-system mass cannot understate the lower bound on wearer-borne plus off-board mass, allowing the same tolerance.
- Invalid inputs reject the comparison; the evaluator does not impute missing values or silently narrow an interval.
- Evidence level is copied exactly and never promoted.
- Output preserves caller candidate order and uses a canonical metric order for byte-stable replay.
- Reports remain separate from actuator authority and cannot establish a physical safety case.

The mass-envelope checks are necessary screens, not proof of joint physical consistency. The intervals are treated as independent, and cross-metric correlations are not estimated. A production metrology protocol must explicitly model dependencies if it intends to make stronger claims.

## 3. Proposed benchmark gates

These are engineering targets for the **software and evidence pipeline**, not clinical standards or claims about real suits.

| Gate | Proposed target | Failure interpretation |
|---|---|---|
| Fixture truth | 100% agreement with versioned expected robust frontier, dominance pairs and unresolved pairs | Reject release if a synthetic expected relation changes unexpectedly |
| False robust dominance | Zero false robust dominators across exhaustive objective-direction controls and at least 10,000 deterministic generated interval cases | A false positive invalidates the evaluator's central claim |
| Interval monotonicity | Widening a declared envelope must never create a new robust-dominance claim | Any counterexample is a correctness defect |
| Direction coverage | Every one of the 15 metrics tested as both the potential winning and blocking dimension where logically applicable | Untested dimensions remain unsupported |
| Missingness/integrity | 100% of missing, duplicate, inverted, non-finite and invalid normalized bounds rejected | Fail closed; never substitute zero |
| Mass integrity | 100% of deliberately inconsistent envelopes rejected by fixtures and property sweeps | Stop comparison rather than rank invalid mass denominators |
| Determinism | Identical request and schema produce byte-identical JSON across 100 replay iterations | Non-determinism breaks reproducible receipts |
| Claim ceiling | 0 paths that turn synthetic intervals into physical, clinical, regulatory or safety claims | Any claim promotion is a release blocker |

The current draft authors focused unit tests plus a fixture-replay test. The 10,000-case sweep, exhaustive dimension mutation, and interval-monotonicity property test are **future gates**, not tests claimed to have run or already passed.

## 4. Physical-data readiness targets

Do not use invented numbers from the fixture as mass, power, comfort or performance targets. Before real candidates enter this comparator:

1. **Publish the measurement model.** Define task success predicates, setup-ready state, energy measurement boundaries, repair task, comfort instrument and movement-restriction method. Record device serial/revision, operator, environment, calibration IDs and task-protocol digest.
2. **Carry the uncertainty budget.** For scales, load cells, electrical power meters, timing and sensors, document calibration, resolution, repeatability, drift, environmental sensitivity and propagation to reported bounds. Do not relabel a small-sample observed min/max as a conservative envelope.
3. **Check envelope coverage independently.** Hold out repeated measurements and predeclared operating conditions. Target 100% of the preregistered validation observations, with the measurement uncertainty budget applied, to fall inside the published envelope. Any miss requires invalidating/revising the envelope and preserving the failed receipt; it does not prove universal real-world coverage.
4. **Separate within-user and between-user variation.** Report individual trajectories and distributions, not only group averages. Sample size for human studies should be justified by the prespecified outcome and intended inference, not by a generic fixed n.
5. **Include failures and transitions.** Capture donning/doffing, restart, mode transitions, low energy, degraded sensors, obstacles, slopes, sit/stand, service/recalibration, stop requests and unplanned task exits. Successful steady-state walking alone is not sufficient.
6. **Predeclare meaningful trade-offs with intended users.** Keep Pareto candidates visible; only apply priorities that the user, clinician or task owner explicitly selected. Do not encode preference weights invisibly in a universal scalar.

## 5. Paradigm shifts this benchmark enables

- **Point value → bounded claim.** A value with unknown uncertainty cannot be used to declare superiority.
- **Universal winner → robust feasible set.** Keep candidates whose advantage is not stable across the declared envelope, then collect the evidence that separates them.
- **Device benchmark → user/task/context benchmark.** The same architecture can be viable for one task and unacceptable for another.
- **Component comparison → coupled whole-system accounting.** Preserve wearer, segment, off-board and total-system mass, energy, setup, interaction and repair burden together.
- **Optimizer output → auditable evidence receipt.** Every conclusion states the input schema, protocol, interval semantics, evidence level and claim ceiling.

The revolutionary opportunity is a shared, machine-checkable evidence contract for the entire design loop: candidate generation, bench measurement, user preference, lifecycle reliability and service experience can exchange evidence without allowing a better simulation score to masquerade as demonstrated real-world benefit.

## 6. Synthetic fixture interpretation

The versioned fixture has three fabricated candidates:

- `balanced-assist` robustly dominates the deliberately worse `overbuilt` candidate in the declared synthetic intervals.
- `mass-first` remains on the frontier because task-success/comfort ranges trade off against multiple mass, energy and service dimensions.
- Comparisons without robust dominance remain unresolved; the implementation does not average dimensions or assign a probability.

These values are generated only to test schema and decision logic. They are not measured products, engineering recommendations, safety limits, range predictions or clinical results.

## 7. Non-claims and next steps

This benchmark does not authenticate metrology, prove intervals contain future values, model correlated uncertainty, estimate probability of success, determine a clinically meaningful difference, recommend a particular architecture or issue actuator commands. It complements the point-estimate Pareto evaluator and mechanical standards testing.

Next implementation steps: add property-based monotonicity testing, objective-by-objective mutation controls, independent receipt hashing, and a measurement-data schema with calibration/provenance before introducing physical candidates.
