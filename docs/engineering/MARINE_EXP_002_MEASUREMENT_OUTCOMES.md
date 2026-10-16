# MARINE-EXP-002 — noisy, biased and adversarial measurement contract

Status: contract-first research scaffold; no measurement simulator, Rust integration, or benchmark result is claimed yet.

Parent: [MARINE-EXP-002 #7683](https://github.com/Luminous-Dynamics/symthaea/issues/7683)  
Previous selector: [PR #7661](https://github.com/Luminous-Dynamics/symthaea/pull/7661)  
Rights and physical uncertainty intake: [MARINE-PHYS-001 #7662](https://github.com/Luminous-Dynamics/symthaea/issues/7662)

## Why this is the next paradigm shift

The current experiment-selection simulator contracts intervals around hidden truth. It therefore guarantees coverage by construction and cannot measure false confidence, bias miss, correlated evidence, or whether a test result should be rejected. The next simulator must permit a measurement to contradict its prior interval, be inconclusive, be invalid, or expose an out-of-domain assumption.

Research in ship-design optimization already uses constraint-aware sample selection to improve surrogate construction; therefore the goal is not merely to rename active sampling. The distinction to test here is falsification-first evidence handling: explicit hard-constraint disposition, truth-coverage scoring, unknown-bias abstention, provenance and rights gates, and an audit trail for each selected measurement. See the [2026 constraint-aware ship-design sample-selection study](https://doi.org/10.1016/j.engappai.2025.112716). The ITTC repository is a community index of benchmark data, but notes that data property and link maintenance are not its responsibility; registry membership must not be interpreted as permission to redistribute a linked dataset ([ITTC benchmark repository](https://www.ittc.info/benchmark-repository/)).

## Contract split — do not collapse these types

1. **Policy-visible observation** — `schemas/marine_measurement_observation_v1.schema.json`. This allow-listed object contains only observed values/ranges, explicit outcome validity, method envelope, measurement-noise characterization, bias disposition, dependence, cost and provenance/rights state.
2. **Evaluator-only oracle key** — `schemas/marine_measurement_evaluator_key_v1.schema.json`. This separate object contains hidden truth, scenario split/family and injected noise/bias/fault labels. It must not be imported into or serialized alongside the policy request. The schema is a contract, not proof that an implementation respects the boundary; follow-up tests must enforce it.
3. **Frozen execution trace** — policy ID, exact revision, scenario/observation IDs, candidate catalog revision, decision order, abstentions and observed outcomes. Seal the trace before the evaluator joins it to oracle labels.
4. **Evaluator receipt** — join only after decisions are sealed; bind the scenario manifest, policy-visible corpus digest, oracle/evaluator revision, seed list, output trace and metric definitions.

Opaque scenario IDs must not encode truth, noise seeds, hold-out membership or fault assignments. Split manifests and oracle data remain evaluator-side.

## Outcome contract and fail-closed behavior

An observation is explicitly `valid`, `inconclusive`, or `invalid`. A valid numeric reading still does not automatically qualify as decision evidence: an unsupported method, failed calibration, domain mismatch, unknown unbounded systematic bias, unknown dependence, missing rights, or unknown experimental cost must not be given an optimistic information score. Depending on the missing field, the correct action is reject, abstain, or preserve `Indeterminate`.

Keep at least these concepts distinct:

- random repeatability/noise versus systematic bias/drift;
- measurement uncertainty versus numerical/discretization uncertainty;
- model discrepancy versus input/environment uncertainty;
- independent observations versus correlated observations;
- invalid test versus valid but inconclusive result;
- interval coverage versus interval sharpness;
- physical feasibility classification versus cost-to-resolution.

The ITTC publishes separate CFD uncertainty and validation procedures, including uncertainty analysis, validation benchmark data and practical application guidance. Those references reinforce why a small numerical residual is not a substitute for experimental validation. Do not promote the synthetic measurement contract into a hull or vessel qualification claim.

## Falsification matrix

The held-out suite should include at least: clean repeatable outcomes; heteroscedastic noise; heavy-tail/outlier cases; fixed offset and drift; correlated repeats; missing/censored/saturated sensors; failed calibration; method/domain mismatch; observations outside prior ranges; near-boundary cases; cheap irrelevant experiments; unsupported or rights-unclear methods; and unknown unbounded bias.

The generator must allow truth to fall outside the reported interval. Otherwise coverage is tautological. The selector gets no `truth_parameters`, applied-noise draw, applied-bias value, simulator seed, scenario family, or hold-out label.

## Proposed benchmark gates — targets, not results

- **No oracle leakage:** zero policy input paths to truth, disturbance values/seeds, hidden scenario family, or hold-out membership.
- **Critical false-favorable gate:** zero false `RobustFeasible` dispositions in the declared bounded-bias and injected-fault adversarial suite. Unknown unbounded bias, unsupported methods, failed calibration and domain mismatch must result in rejection/abstention or `Indeterminate`.
- **Coverage:** for methods that claim nominal 50%, 80%, 90% and 95% intervals, report empirical truth coverage with binomial confidence intervals, overall and per prespecified stratum. First target at least 10,000 held-out outcomes in each major stratum. Use 93–97% as a provisional 95%-interval calibration band for the declared in-distribution synthetic stratum; critical strata must not be hidden by pooling.
- **Coverage plus sharpness:** report coverage and average interval width together. A wider interval can improve coverage while making the decision useless; a narrow miss is not success.
- **Decision errors:** report false-favorable, false-unfavorable, indeterminate, abstention and budget-to-resolution rates separately for battery energy, water depth and charge power.
- **Dependence:** repeated correlated observations must not be scored as independent information without an explicit supported dependence model.
- **Held-out discipline:** commit scenario-family definitions, generator version, seed manifest and scoring rules before evaluation. Tuning and held-out scenario IDs must be disjoint.
- **Policy comparisons:** use equal-budget paired cases and report effect estimates with uncertainty. Hidden-truth oracles may be retained as explicitly clairvoyant diagnostic controls, but are excluded from claims of achievable selector performance.
- **Evidence integrity:** every receipt binds observation corpus, evaluator/oracle revision, selector revision, catalog, seeds, ordered choices, metric definitions and output. Hashes establish integrity only, not source authenticity or physical truth.

## Implementation order

1. Review the two strict schemas and the policy/oracle separation threat model before implementing outcomes.
2. Add separately typed policy-input and evaluator-only Rust structures; do not pass one combined scenario struct into policy code.
3. Implement deterministic but configurable noise/bias/correlation/fault families and record each generation choice in the evaluator-only manifest.
4. Add test seams that attempt oracle leakage, held-out label leakage, duplicate-correlated evidence, optimistic intervals after invalid measurements, and false favorable classifications.
5. Freeze held-out corpora and execute coverage/calibration gates; retain exact receipts and report failures rather than tuning away difficult strata.
6. Only after synthetic falsification passes, adapt rights-reviewed experimental records under #7662. Keep every physical uncertainty dimension separate.

## Claim ceiling

A pass establishes behavior on the declared synthetic outcome generator and evaluation strata only. It does not establish calibrated uncertainty for a real vessel, validated hydrodynamics, hull superiority, operational safety, seaworthiness, certification, or authority to operate.

## External anchors

- [Constraint-aware sample selection for ship-design optimization, Engineering Applications of Artificial Intelligence (2026)](https://doi.org/10.1016/j.engappai.2025.112716).
- [ITTC benchmark repository](https://www.ittc.info/benchmark-repository/) — reference index; independently verify exact case availability and data-use rights.
- [ITTC procedure index: CFD uncertainty analysis and resistance/propulsion validation guidance](https://www.ittc.info/search/?q=+7.5-03-02-).