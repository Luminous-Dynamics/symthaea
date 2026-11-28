# SEMI-OBSERVATION-TRACE-001 — joint observation-process trace v0.1

**Canonical issue:** #13913  
**Implementation branch:** `feat/semi-observation-path-trace-v1`  
**Status:** implementation with 17 authored unit tests and 3 independent-oracle integration tests; exact-head validation pending.

## Research thesis

There is no single "missing label" mechanism in semiconductor VM. A wafer's outcome may be absent from a model update for four fundamentally different reasons:

1. **Physical acquisition:** the process chose not to measure the wafer, the measurement is pending, or the physical acquisition failed.
2. **Evidence qualification:** a measurement exists, but its calibration/identity/value/revision evidence is eligible, UNKNOWN or BLOCKED.
3. **Controller feedback retention:** a qualified outcome may arrive after the online controller's finite retention window or capacity policy.
4. **Independent population audit:** a separate probability design may measure members of the nonqualified population using a separate reference method.

These are not exchangeable missingness events and do not share a universal propensity. A single "observed" boolean cannot represent them.

The v0.1 implementation composes the previously implemented K1-qualified label guard, τ-DACI, bounded-delay CA-OCRC and selection-risk estimator through explicit receipt digests and typed state.

## Four separate channels

### 1. Operational physical-metrology acquisition (budget M)

Each prediction round is one of:
- `UNSELECTED`;
- `SELECTED_PENDING`;
- `MEASURED`;
- `MEASUREMENT_UNAVAILABLE`.

The trace records acquisition design digest, declared inclusion probability when known, measurement-system identity, acquisition time, physical cost and latency. Unselected wafers cannot carry operational labels. Missing/unknown selection probability is not silently replaced with 1.

The declared operational acquisition budget counts selected, measured and unavailable attempts. It is separate from the online controller tracking capacity.

### 2. K1 evidence qualification

For a measured operational label, the trace binds:
- label ID and revision;
- full label-payload digest;
- measurement-event ID;
- K1 qualification report digest;
- corrected value and conservative uncertainty upper bound;
- calibration certificate and requalification report references;
- label and qualification-report availability times.

K1 state is `ELIGIBLE`, `UNKNOWN` or `BLOCKED`. Only ELIGIBLE labels may contribute an operational loss or a controller state update. UNKNOWN/BLOCKED evidence is never rehabilitated by inverse-probability weighting.

### 3. Online controller feedback tracking (capacity K)

One controller is declared per trace:
- `NONE`;
- `TAU_DACI`;
- `CA_OCRC`.

τ-DACI binds a fixed delay and rejects CA-OCRC-specific proxy/capacity fields. CA-OCRC binds its own policy digest, bounded-delay contract, proxy draw, proxy horizon, survival probability, capacity and occupancy. A CA-OCRC occupancy above K is rejected.

The feedback state is one of:
- `NOT_SUBMITTED`;
- `NO_PHYSICAL_OUTCOME` (prediction exists, but no operational physical label exists);
- `PENDING`;
- `RETAINED`;
- `RECEIVED_QUALIFIED`;
- `RECEIVED_UNQUALIFIED`;
- `CAPACITY_DENIED`;
- `EXPIRED`;
- `EVICTED`;
- `IMMEDIATE_TRIVIAL`.

The trace verifies receipt presence, K1 eligibility, the logical feedback round/delay, and the physical wall-clock ordering of label availability, K1 report availability and feedback receipt. An expired/evicted item may exist physically, but cannot mutate the online controller.

τ-DACI and CA-OCRC remain alternative controllers. The trace does not combine their recursions or risk guarantees.

### 4. Independent population audit (budget A)

The audit channel records its independent design, randomization provenance, inclusion probability, selection/measurement times, reference-method identity, measurement receipt, loss, physical cost and retention path:
- neither;
- online audit store;
- offline archive;
- both.

An audit method must be distinct from the operational measurement system for the same round. Only outcomes retained in the offline archive (or both stores) are passed to the offline Horvitz–Thompson population estimator. Online-only audit records are counted and costed but are not silently used as if available to the offline estimator.

The audit budget is separate from M and K.

## Population-risk calculation

The trace creates a complete, disjoint population partition:
[
mathcal P = mathcal Q dotcup mathcal U dotcup mathcal B dotcup mathcal N,
]
where (mathcal Q) is operationally qualified, (mathcal U) is UNKNOWN, (mathcal B) is BLOCKED, and (mathcal N) is unmeasured/non-applicable.

For losses in [0,1], the no-MAR partial-identification bounds are:
[
rac{sum_{iinmathcal Q} L_i}{N}
le R_{mathcal P} le
rac{sum_{iinmathcal Q}L_i + |mathcal U|+|mathcal B|+|mathcal N|}{N}.
]

The optional independent audit estimator is delegated to the already validated #13650 estimator and is included only under its declared independent Bernoulli design, positivity and independent reference-method assumptions. A missing or invalid audit result does not narrow the worst-case identification interval.

## What is deliberately not calculated

The module never multiplies:
- operational acquisition inclusion probability;
- CA-OCRC proxy survival probability;
- independent audit inclusion probability;
- K1 evidence status.

There is no general theorem that their product is a valid joint weight. The operational acquisition policy is outcome-dependent in realistic fabs; K1 invalidity is an evidence state, not a random sampling coin; and CA-OCRC capacity creates queue/retention dependence across rounds. Any joint estimator requires a separately derived design and dependence proof.

The report explicitly emits:
- `combined_ipw_estimate_emitted = false`;
- `combined_coverage_theorem_emitted = false`.

## Integrity and chronology invariants

- A committed population inventory is supplied before evaluation; observed trace rounds must match it exactly.
- Round, sample, measurement-event and prediction-receipt identities are unique.
- Event time, prediction time and round time are ordered in one declared clock domain.
- UNSELECTED and unavailable operational acquisition cannot carry operational K1 labels.
- MEASURED requires measurement system, design, cost and K1 disposition.
- ELIGIBLE K1 requires full label/revision/report identity, corrected value, uncertainty, certificate/requalification references and bounded loss.
- UNKNOWN/BLOCKED carries no eligible loss.
- Controller receipts and updates are mutually consistent with the single declared controller algorithm.
- Received feedback must be K1-qualified to update state and must arrive after both label and report availability.
- Fixed-delay τ-DACI feedback must match its exact fixed round schedule.
- CA-OCRC feedback must match its actual delay, d_max and sampled proxy window; per-round occupancy may not exceed K.
- Audit selection/measurement chronology, inclusion probability, retention and distinct reference-method identity are validated.
- Separate M and A budgets are enforced; K is checked per controller round.

## Independent validation

The module has **19 unit tests** covering:
- disjoint population partition and worst-case bounds;
- unselected-label rejection;
- independent/offline audit use and online-only exclusion;
- audit-method separation from operational measurement;
- fixed-delay τ-DACI receipt chronology;
- valid CA-OCRC K1 feedback and expired-feedback no-update;
- explicit NO_PHYSICAL_OUTCOME state distinct from capacity-denied, unsubmitted, pending, expired and evicted;
- explicit capacity-denial state distinct from unsubmitted, pending, expired and evicted;
- capacity K versus physical acquisition M;
- per-stage resource budgets;
- inventory omission;
- permutation-invariant trace/report digest.

Three integration-oracle tests independently check:
- population partition and worst-case bound arithmetic without calling the selection-risk estimator;
- committed inventory completeness;
- input-order-invariant trace/report digest.

Tests are authored; exact-head rustfmt, strict Clippy, all crate/reference tests and SEMI-BENCH-001 pinned chronology remain required.

## Preregistered joint synthetic tournament

Use the same population truth, prediction receipts and paired random seeds across methods.

- Population: N=10,000 rounds per seed.
- Physical acquisition rates M/N: 1%, 2%, 5%, 10%, 20%.
- Independent audit rates A/N over the nonqualified population: 0%, 0.1%, 0.5%, 1%, 2%, 5%.
- Controller tracking capacity K: 1, 2, 4, 8, 16, 32.
- Maximum bounded delay d_max: 0, 1, 2, 4, 8, 16.
- K1 outcomes: qualified/UNKNOWN/BLOCKED under MCAR, MAR and MNAR simulation controls.
- Common-cause calibration/reference failures, audit-method independence failures, out-of-order delivery and capacity saturation.
- τ-DACI fixed delay τ in {1,2,4,8}; CA-OCRC arbitrary-bounded-delay q_G=1 as a separate controller.

Run at least 100 paired seeds for the primary Monte Carlo comparison. Report confidence intervals and keep estimands separate.

### Required metrics

**Prediction / selected stream:** per-tool/chamber RMSE/MAE, empirical coverage/miscoverage, interval width and interval score.

**Population risk:** qualified subset risk, worst-case identification interval width, audit HT bias/variance/ESS and finite-sample interval coverage.

**Tracking:** retained/expired/evicted outcomes, occupancy, K violations, proxy-survival weights, delayed-label backlog and theorem-assumption de-certification.

**Resource:** operational physical measurement M/cost, independent audit A/cost, online feedback storage K/CPU/memory, delay distribution and update latency.

### Acceptance targets (hypotheses, not achieved results)

- 100% population inventory and disjoint partition completeness.
- Zero K1 UNKNOWN/BLOCKED controller updates.
- Zero confusion of M, K and A.
- Exact independent replay of τ-DACI and CA-OCRC receipts on their own method-specific lanes.
- Audit HT unbiased within Monte Carlo error only under correctly specified independent Bernoulli audit sampling.
- Zero population point claims from qualified-subset risk alone under incomplete outcomes.
- Zero combined IPW/coverage claims unless a separate derived joint estimator and theorem are added.

## Claim ceiling

This is a typed observation-process validator and synthetic benchmark harness. It does not prove that declared physical source records are authentic, that an audit reference instrument is physically traceable, or that the declared audit randomization was independently executed. Receipt digests are not signatures. It is not a combined coverage theorem, not a population estimator under arbitrary MNAR selection, and not a manufacturing/process-control authorization.
