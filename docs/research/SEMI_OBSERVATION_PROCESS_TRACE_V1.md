# SEMI-OBSERVATION-PROCESS-001 — joint observation-path trace v1

**Issue:** #13913  
**Implementation:** symthaea-semiconductor::benchmark::observation_process_trace  
**Base:** #13650 population-risk bounds + #13478 evidence-gated streaming adapter  
**Status:** Rust trace validator with 11 authored unit tests; exact-head validation pending.

## Why this exists

The semiconductor programme already has separate validated components:
- #13614 fixed-delay K1-gated tau-DACI;
- #13708 bounded-delay CA-OCRC with feedback capacity K and q_G=1;
- #13650 population-risk partial identification and independent Bernoulli audit estimator;
- #13478 delayed-label evidence-gated streaming updates;
- #13465 model/label lineage impact closure.

The remaining integration risk is not a missing optimizer. It is accidental conflation of the mechanisms that decide whether a label exists, whether it is trustworthy, whether an online controller retained it, and whether the population was independently audited.

## Four distinct processes

1. **Operational acquisition:** whether physical metrology was selected and returned a measurement.
2. **K1 qualification:** whether the measured evidence is eligible, UNKNOWN or BLOCKED.
3. **Online controller retention:** whether tau-DACI or CA-OCRC has a pending, retained, consumed, expired, evicted or non-admitted feedback record.
4. **Independent audit:** a distinct reference method that samples the nonqualified population under its own Bernoulli audit design.

These are separate fields in every population-round row. There is no generic "missing label" state and no universal propensity field.

## Implementation

The typed, sealed trace contract binds:
- population manifest and source ledger;
- one unique prediction receipt per population round;
- event/prediction/round chronology and clock domain;
- operational acquisition state, optional known inclusion probability, selection design, measurement-system identity, label/revision/payload/value/uncertainty and K1 report;
- exactly one controller mode per run (NONE, TAU_DACI or CA_OCRC), policy/delay/capacity declaration, prediction/feedback/degenerate receipt references, tracking state, actual/proxy delay and proxy survival;
- independent audit design, per-round inclusion probability, randomization receipt, separate reference-method receipt, bounded audit loss and archive receipt.

The validator checks row and trace digests, exact population partition, K1 report/value/revision/payload binding, controller transitions, controller-specific fields, audit-frame membership, positivity, per-round randomization decisions, independent reference-method identity and any linked #13650 population-risk report counts.

## Critical boundaries

- A K1 UNKNOWN/BLOCKED record may remain in an audit population but cannot be consumed by an online risk controller.
- A feedback outcome that is expired, evicted or not admitted is not an online controller update, even if its physical value later becomes available offline.
- An independent audit result is not silently inserted into operational ACI/CA-OCRC feedback.
- tau-DACI and CA-OCRC are mutually exclusive controller modes in one run; the trace never combines their alpha/threshold recursions.
- The CA-OCRC inverse-survival correction addresses its own feedback-retention mechanism under its bounded-delay assumptions. It does not correct operational metrology selection.
- The #13650 audit HT estimator addresses the declared independent audit design over nonqualified population units. It is not a controller coverage theorem.
- No product of operational propensity, audit probability and CA-OCRC proxy survival is calculated. Queue admission/eviction can couple tracking decisions; any combined inclusion estimator needs a separately derived joint design and independent oracle.

## Validation behavior

The current contract rejects:
- an unselected/unavailable/pending operational result carrying a label or K1 report;
- an eligible label whose corrected value or uncertainty differs from the exact payload;
- a consumed controller feedback without an eligible K1 report and update receipt;
- controller updates on UNKNOWN/BLOCKED evidence;
- controller proxy-delay fields in tau-DACI mode;
- CA-OCRC retained entries without positive capacity/proxy delay;
- expired/evicted feedback marked as updating;
- an audit result without a separate audit design, randomization receipt, reference method or bounded loss;
- audit measurement drawn from the operationally qualified frame;
- nonpositive/missing audit propensities;
- population-risk report counts inconsistent with the observation trace.

An incomplete population inventory returns UNKNOWN with explicit missing round IDs. It never means "no missing outcomes."

## Authored tests

Eleven Rust unit tests cover:
1. complete trace counts operational acquisition separately from controller state;
2. incomplete inventory returns UNKNOWN;
3. unselected rounds cannot carry operational labels/K1 receipts;
4. UNKNOWN K1 evidence cannot be marked consumed by tau-DACI;
5. eligible label value/revision/payload/report can be consumed by tau-DACI with the fixed delay declaration;
6. BLOCKED measured label can be represented but not consumed;
7. tau-DACI rejects CA-OCRC-only proxy fields;
8. CA-OCRC retained pending path requires capacity, selected physical measurement and proxy delay;
9. independent audit measurement must be separate from qualification and traceable;
10. an audit non-selection decision requires its per-round randomization receipt;
11. row-order permutation preserves the trace/report digest.

Exact-head rustfmt, strict Clippy, crate tests, independent reference checks and SEMI-BENCH-001 are still required.

## Benchmark protocol

Generate the same 10,000-round population truth for each method; use paired seeds and at least 100 Monte Carlo seeds.

Sweep:
- operational metrology acquisition rate M/N: 1%, 2%, 5%, 10%, 20%;
- independent audit rate A/N on nonqualified rounds: 0%, 0.1%, 0.5%, 1%, 2%, 5%;
- online feedback capacity K: 1, 2, 4, 8, 16, 32;
- maximum physical feedback delay: 0, 1, 2, 4, 8, 16;
- qualification MCAR/MAR/MNAR and common-cause calibration-revocation regimes;
- fixed-delay tau-DACI tau=1,2,4,8;
- bounded-delay CA-OCRC q_G=1 as a separate controller run.

Report separate estimands:
- selected qualified-stream coverage/miscoverage;
- qualified-subset risk;
- worst-case full-population risk interval;
- independent audit HT estimate, variance, effective sample size and confidence coverage;
- CA-OCRC capacity occupancy, proxy survival, feedback retention and theorem applicability;
- physical acquisition cost, audit cost, controller memory and update latency.

No single metric may stand in for all four processes.

## Claim ceiling

This module validates an observation-path ledger. It is not a combined coverage theorem, a new IPW estimator, an MNAR correction, a physical traceability proof, or equipment authority. Population-wide risk remains unidentified without assumptions or independent audit measurements; no theorem is emitted by this trace validator.
