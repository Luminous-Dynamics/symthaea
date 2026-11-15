# symthaea-decision-assessment

A small, non-authoritative Rust composition layer for engineering decision assessments.

## Purpose

Consume qualified outputs from existing owners and combine them into five non-compensatory axes:

1. Process admissibility.
2. Evidence-use validity.
3. Engineering applicability.
4. Causal justification.
5. Action-policy correctness.

It also qualifies the candidate's pre-outcome commitment and provides a paired ACT/HOLD evaluator with step-indexed runtime interruption timing.

## Boundary

This crate does **not**:

- authenticate signatures, SCITT statements, transparency proofs, or trust roots;
- establish source truth, physical correctness, or causal truth;
- implement a temporal ledger, evidence-use engine, applicability oracle, probability calibrator, or action/loss oracle;
- create engineering approval, requirement acceptance, release, manufacture, operation, or external-effect authority.

The caller supplies ReceiptQualification values only after the corresponding owner-native validators have run. The caller also supplies CommitmentQualification only after an independent commitment/timestamp owner has qualified the exact packet digest, timestamp integrity, authenticity, and outcome blinding. Missing owner qualification or commitment witness remains indeterminate.

## API outline

- evaluate_json(packet_bytes, owner_qualifications, commitment_qualification) returns an AssessmentReport with per-axis states, exact reason codes, preserved outcome state, and a hard effect_authority = none boundary.
- evaluate_act_hold_pair(should_act, should_hold) scores both sides of a paired scenario.
- ActHoldEpisode records the first observable stop-trigger step, first irreversible-effect step, and terminal safe-disposition step. This measures timely interruption rather than merely checking the final answer.

The candidate commitment uses the domain-separated canonical projection described in docs/research/ENGINEERING_DECISION_RUST_COMPOSITION_V0_1.md. Evaluator-authored axis findings, defeaters, counterfactual labels, outcome records, and assessment results are excluded from the candidate's pre-outcome commitment.

## Offline known-answer corpus

benchmarks/engineering-change-impact/act-hold-pairs-v0.1.json contains seven synthetic known-answer pairs:

- eligible ACT with pre-execution HOLD;
- always-abstain negative control;
- post-hoc abstention after an irreversible effect;
- action after a runtime-visible stop trigger;
- action without qualified authority;
- timely runtime interruption;
- unknown trigger/hold timing.

The unit test consumes this frozen fixture and compares all reported fields and reason-code sets. This is a deterministic synthetic regression suite, not a population reliability estimate or an external benchmark result.

## Build and test

From the workspace root:

    cargo test -p symthaea-decision-assessment
    cargo clippy -p symthaea-decision-assessment --all-targets -- -D warnings
    cargo fmt --all -- --check

These commands are the intended local verification path. Do not describe them as passed until run against the exact branch head and recorded in a reproducible receipt.


## Partial-order stop/effect event traces

The crate also exposes `evaluate_event_order` for qualified event traces. Unlike the existing single-harness ACT/HOLD step counter, this evaluator classifies event order as:

- `before`: the first event is established to precede the second;
- `after`: the first event is established to follow the second;
- `overlapping`: qualified clock intervals overlap without a decisive causal or same-domain sequence relation;
- `unknown`: no qualified source establishes an order;
- `inconsistent`: qualified ordering sources contradict one another.

In `EventOrderReport.effect_orders`, each relation is **effect relative to stop trigger**: `before` means the effect preceded the trigger, while `after` is a critical post-trigger effect. Overlapping, unknown or inconsistent order blocks further irreversible actions.

Operation start, cancellation request, cancellation acknowledgement, unknown outcome, exact reconciliation and retry are separate event kinds. A cancellation acknowledgement is not a terminal operation outcome. A retry requires an exact `not_applied` reconciliation ordered before that retry. If an operation outcome remains unknown, the operation remains pending and new mutations are blocked.

The evaluator consumes event/clock/causal-order and operation-outcome qualifications from their owners. A passing stop-disposition qualification is additionally bound to a non-empty receipt reference, SHA-256 digest, exact trigger event ID and exact discharge event ID; a mismatched or replayed binding fails closed. These fields are still caller-supplied owner qualification outputs, not authenticated by this crate. It does not authenticate signatures, qualify clocks, prove trace completeness, execute cancellation, authorize effects, or replace the event/history owners.

A stop trigger remains an active no-new-effect obligation until a separate `StopObligationDischarged` event is ordered after the trigger and its `stop_disposition` owner qualification is PASS. A discharge cannot be used while any started operation remains unreconciled. An effect, operation start or retry after the trigger is accepted only when it is definitely after that qualified discharge.

The frozen synthetic fixture is `benchmarks/engineering-change-impact/action-event-order-v0.1.json` (11 event traces). Its standalone Rust integration reference test also checks canonical trace digests, a separately expressed ordering relation, and invariance under reverse delivery order.

Run the composition tests from the workspace root:

```sh
cargo test -p symthaea-decision-assessment
cargo clippy -p symthaea-decision-assessment --all-targets -- -D warnings
cargo fmt --all -- --check
```

These tests are authored but are not qualified as passed until executed at the exact branch head. The event-order reference adds canonical event-trace digest checks, independent ordering reconstruction, delivery-order permutation tests, and malformed-graph/operation-terminal controls.


## Event-order benchmark protocol

The full methodology, baselines, metrics and claim boundaries for the distributed stop/effect evaluator are documented in [ENGINEERING_DECISION_EVENT_ORDER_V0_1.md](../../../docs/research/ENGINEERING_DECISION_EVENT_ORDER_V0_1.md). The protocol distinguishes effect-before-trigger from effect-after-trigger, uses five-way partial-order results, requires an independently bound stop-disposition receipt, and treats unknown operation outcomes as pending. The fixture has eleven synthetic traces. `action-event-order-mutations-v0.1.json` adds 26 preregistered mutation controls (not executed outcomes). The Rust reference suite now has twelve test functions, including canonical digest, independent ordering, delivery permutation, malformed trace, receipt-binding, operation chronology, cancellation-order and mutation-manifest integrity controls. These remain authored, not compiled/executed, until exact-head toolchain evidence is recorded.


## Event-order v0.2 contract kernel

The additive `event_order_v0_2` module introduces explicit time-basis descriptors, qualified interval comparison, bitemporal visibility at decision frontier K, attempt identity and exact start/terminal/retry event bindings, and independently scoped stop obligations. Stop discharges commit to the exact in-flight inventory frontier and canonical attempt-manifest digest. These are owner-qualified inputs, not signature verification or effect authority. See [ENGINEERING_DECISION_EVENT_ORDER_V0_2.md](../../../docs/research/ENGINEERING_DECISION_EVENT_ORDER_V0_2.md).

Generated properties live in `tests/event_order_v0_2_properties.rs`; they are authored, not executed, until exact-head Rust toolchain evidence is recorded.


## Event-order v0.3 deterministic replay harness

The additive `event_order_v0_3` module defines versioned typed schedules, per-domain virtual clock advances, event production bound to exact source clock basis/epoch/scale/unit and exact mapping-action identity, with unqualified/expired/stale-mapping fault schedules and causal edges, delivery/duplicate/drop and network partition faults, observation/durability, cancellation, stop-policy revisions, effects/reconciliation, process crash/restart, supersession, exact-repeat versus conflicting effect-commit, cancellation-ack and terminal-receipt dispositions, plus seeded stop/cancel/commit race generation. It provides clock/event-time regression dispositions scoped to a source clock epoch, exact trigger and policy-action references for stop discharge, canonical schedule hashing, deterministic seeded generation, a replay report and a delta-debug reducer that preserves structural validity and a caller-supplied counterexample predicate.

The schedule schema is [event-order-schedule-v0.3.schema.json](../../../docs/schemas/event-order-schedule-v0.3.schema.json). The benchmark methodology and claim ceiling are in [ENGINEERING_DECISION_EVENT_ORDER_V0_3.md](../../../docs/research/ENGINEERING_DECISION_EVENT_ORDER_V0_3.md). This scheduler is not an event-order safety oracle and does not authenticate owner receipts or authorize effects. Its next integration gate is an adapter to the v0.2 evaluator and a separately expressed differential oracle. Sixteen module unit tests and four Proptest properties (256 cases each configured) are authored, not executed, until exact-head Rust toolchain evidence is recorded.
