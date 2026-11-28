# SEMI-VM-JOINT-TRACE-001 — paired metrology and online-risk replay v0.1

**Issue:** #14142  
**Implementation PR:** #14148 (draft)  
**Module:** `symthaea_semiconductor::benchmark::semi_vm_joint_risk_trace`  
**Current scope:** Mode A — paired controllers, shared acquisition stream.

## Research thesis

The semiconductor VM benchmark must maintain four separate ledgers:

1. **Physical metrology selection:** which wafers were actually measured, at what physical metrology cost, and under which acquisition policy.
2. **Label qualification:** whether each physical result is K1-qualified, UNKNOWN or BLOCKED, with revision, payload digest, measurement event, calibration and availability lineage.
3. **Controller feedback tracking:** which arrived outcomes each online conformal controller retained, evicted, expired or used, under its own policy and capacity.
4. **Independent population audit:** a randomized reference channel independent of operational qualification, used to estimate population loss separately from qualified-subset loss.

These are different random variables/resources. Feedback tracking capacity is not physical metrology capacity. K1 eligibility is not a sampling propensity. A conformal coverage result on delivered/qualified labels is not automatically a population-risk result.

## Composed, not conflated

The v1 trace validates the original receipts from three independently implemented components without rewriting their algorithm code:

- Fixed-delay τ-ACI from #13614: exact phase/delay schedule, frozen interval, K1 delivery, update and degenerate-resolution receipts.
- Bounded-delay CA-OCRC from #13708: original round input, policy, prediction receipt, feedback inputs/results, tracking occupancy/capacity and theorem-assumption state.
- Selection-adjusted population-risk bounds from #13650: qualified-only risk, worst-case partial-identification bounds and independent Bernoulli audit HT estimate/variance/confidence bounds.

Pinned input code heads used when composing this branch:
- tau-ACI: `9b7d22cbd5eb8aa22549c7d58edf480e8b128041`;
- CA-OCRC: `25bd7733e468f7bceb3d99f27e7e532fcc812393`;
- population risk: `ec432fe1934da712ad2e32323e2a057ce9579723`.

The shared benchmark does not transfer one method's guarantee to another. ACI is checked against its fixed-delay schedule; CA-OCRC against its bounded-delay/tracking policy; population inference is computed by its separate design-based estimator.

## Comparison scope

Each trace now declares one of two comparability claims:

- **Controller ablation:** every arm must use the same global-model artifact digest and interval-method digest. This isolates controller policy/feedback semantics.
- **End-to-end pipeline comparison:** model and interval artifacts may differ; the report labels the result accordingly and no controller-only attribution is permitted.

Both scopes still share one population, physical acquisition stream, K1 evidence and independent audit design in Mode A.

## v1 — Mode A paired shared acquisition

Each arm sees the same population manifest and source ledger, wafer ordering and measurement-event identities, physical-metrology selected flags and costs, K1 qualification states/revisions/payload/report digests, availability/report/delivery timestamps, acquisition-policy digest, physical-metrology budget and independent randomized audit design/draw/inclusion probabilities/reference receipts/**observed loss values**.

Arm-local state remains separate:
- global model artifact and interval-method identity;
- controller policy/config digest;
- ACI phase state or CA-OCRC tracking state;
- controller prediction/delivery/update/round receipts.

The current v1 implementation requires at least two arms and supports tau-ACI and CA-OCRC typed evidence. It does not yet execute either controller; it validates their frozen receipts and computes common selection-risk summaries.

## Evidence checks

### Population and selection
- unique non-zero population rounds, sample IDs and measurement event IDs;
- positive wafer event timestamps, strictly increasing with round order;
- every τ-ACI and CA-OCRC prediction event time exactly matches the shared population row event time;
- feedback delivery cannot precede the prediction issued in its delivery round;
- measured versus unmeasured outcome/selection consistency;
- metrology cost finite, nonnegative and zero for unselected rows;
- K1 label ID, nonzero revision, payload and report digests for measured outcomes;
- finite numeric values when present;
- label availability ≤ K1 report availability ≤ delivery;
- each arm's qualified/unknown/blocked/unmeasured partitions and qualified losses exactly match the common population;
- independent audit design provenance, positivity and non-empty audit sample;
- same audit design, inclusion probabilities, **losses**, round IDs and reference receipt digests across paired arms;
- physical cost ≤ equal declared metrology budget.

### tau-ACI
- actual policy object digest matches arm policy;
- interval-method digest and global model identity match;
- one prediction per population row; phase and due round match policy delay;
- prediction, update, K1 label and degenerate-resolution receipt digests recompute;
- exact report-attested label revision/payload/value/measurement-event and clock binding;
- label, K1-report and delivery availability match the shared row exactly;
- update miss bit recomputed against the frozen prediction interval;
- alpha-before/after is checked against the frozen τ-ACI recursion;
- every prediction has exactly one terminal path: delivered K1 feedback plus update result, missing/forfeit feedback, or deterministic full/empty interval resolution;
- missing feedback or invalid/non-qualified update cannot coexist with a fixed-delay guarantee claim;
- full/empty interval resolutions are included in the observed-miscoverage denominator and reported separately from physical feedback updates.

### CA-OCRC
- original round input digest and round report digest recompute;
- round input prediction is bound to the shared row, model artifact, interval method, acquisition flag and policy;
- CA policy object digest, tracking capacity and declared actual-delay contract match arm;
- round report's prediction receipt, capacity/occupancy and finite risk-state fields validate;
- every feedback input maps to exactly one result in its delivery round;
- K1 report, full label revision/payload/value, source wafer, delivery/report/label times and clock domain match the shared row;
- result origin round and actual delay match the source wafer round and delivery round when it was actually tracked; `NotTracked` cannot claim an origin/delay/weighted contribution;
- UNKNOWN/BLOCKED outcomes cannot be reported as updated;
- explicitly closed feedback horizon and unresolved tracking rounds are recorded;
- a theorem-intact claim is rejected when physical feedback was unselected/unqualified or unresolved at horizon closure.

## Risk output — separate numbers, separate meanings

For every arm the report includes:
- qualified-subset risk;
- worst-case partial-identification lower/upper bounds and width;
- independent randomized-audit HT population-risk estimate, standard error and conservative Hoeffding bounds;
- ACI observed miscoverage count / scored-outcome denominator plus held UNKNOWN, BLOCKED, rejected, missing and deterministic degenerate counts;
- CA-OCRC weighted feedback contribution, updated/held/blocked/not-tracked/rejected/duplicate counts, max occupancy, capacity and theorem-assumption state;
- physical measurement count/cost and K1 outcome counts for the shared population.

The audit estimate is not a conditional-on-every-wafer coverage guarantee. The partial-identification interval remains the conservative population statement absent stronger assumptions.

## Mixed-controller known-answer comparison

The test suite now includes a shared two-wafer paired trace with one fixed-delay τ-ACI arm and one bounded-delay CA-OCRC arm. Both arms use the same global-model artifact, interval method, acquisition policy, physical-metrology budget, K1-qualified physical label and independent audit draw.

The CA-OCRC stream admits the selected wafer with a one-round proxy delay, receives the K1-qualified label on the next round and records one successful weighted feedback update. The second wafer is physically unmeasured; this correctly invalidates the CA-OCRC complete-feedback theorem assumptions. The summary must therefore show the feedback result and `ca_ocrc_theorem_assumptions_intact=false`, not transfer the bounded-delay theorem to a selected/unmeasured population.

This is the first end-to-end shared-trace execution that compares both controller families rather than using two ACI arms as the paired fixture.

## Frozen fixture

The first known-answer fixture is synthetic and has two rounds:
- wafer 1: physically measured, K1-qualified, its frozen interval is missed, and delayed τ-ACI updates at the declared due round;
- wafer 2: unmeasured and receives a full-real-line interval after the previous update; its deterministic zero-miscoverage outcome is resolved without inventing a physical label;
- the independent audit samples wafer 2 with inclusion probability 1 and a separate reference receipt;
- two paired arms share the same trace, acquisition policy, physical budget and audit draw.

Eleven tests currently cover:
1. successful paired replay with separate physical/qualified/audit counts and ACI coverage;
2. numeric/K1 row mismatch rejection;
3. paired arms with different observed audit losses rejection;
4. physical metrology budget overrun rejection;
5. row/arm input-order-invariant trace digest;
6. unmeasured row cannot carry label/report/delivery evidence;
7. UNKNOWN/BLOCKED CA-OCRC feedback can remain visible without promoting it to an eligible update;
8. controller-only scope rejects different global-model or interval-method artifacts;
9. end-to-end scope accepts a distinct interval method and preserves the scope in the report;\n10. mixed τ-ACI/CA-OCRC arms replay on one shared population and audit, with the CA-OCRC theorem claim suspended when one wafer is unmeasured;
11. a paired ACI trace rejects delivery before the delivery-round prediction.

Exact-head rustfmt, strict Clippy, crate tests and SEMI-BENCH-001 Reference Oracle are pending for the current branch.

## v2 roadmap

**Mode B — acquisition policy comparison at equal physical-metrology cost:** allow different physical selections but one fixed population, per-wafer cost model and independent audit design. Compare population-risk and decision-quality frontiers; the audit must stay independent.

**Mode C — joint acquisition + risk controller + tracking policy:** allow both selection and controller arms to differ, publish a preregistered cost/risk Pareto frontier, and avoid a composite score unless utility weights are preregistered.

Later extensions:
- include delayed conformal calibration width/interval-score metrics alongside miscoverage;
- compare same-policy tau-ACI and CA-OCRC controller arms on identical streams;
- add GRAPE streaming adaptation as another arm after the exact GRAPE dataset conversion and label timing protocol are frozen;
- support tool/chamber stratum summaries and paired block bootstrap intervals;
- include partial audit nonresponse and two-phase audit design with known inclusion probabilities;
- report feedback delay-to-memory and physical-metrology delay/backlog separately.

## Claim ceiling

This is a synthetic shared-trace validation layer. It does not modify the mathematical ACI/CA-OCRC guarantees, prove the stated assumptions hold in a real fab, authenticate the audit issuer, establish source traceability, or prove yield/scrap/cycle-time improvement. It does not issue wafer disposition, recipe or APC/R2R actions, and has no equipment-control authority.
