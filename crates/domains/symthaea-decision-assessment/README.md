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

The schedule schema is [event-order-schedule-v0.3.1.schema.json](../../../docs/schemas/event-order-schedule-v0.3.1.schema.json). A seven-action frozen example is [event-order-v0.3.1-clocked-observation-persistence.json](../../../benchmarks/engineering-change-impact/event-order-v0.3.1-clocked-observation-persistence.json). The benchmark methodology and claim ceiling are in [ENGINEERING_DECISION_EVENT_ORDER_V0_3.md](../../../docs/research/ENGINEERING_DECISION_EVENT_ORDER_V0_3.md). The separate reference model is [event-order-v0.3.1-independent-reference.mjs](../../../benchmarks/engineering-change-impact/event-order-v0.3.1-independent-reference.mjs), with its non-Rust result record [event-order-v0.3.1-independent-reference.json](../../../benchmarks/engineering-change-impact/event-order-v0.3.1-independent-reference.json). This scheduler is not an event-order safety oracle and does not authenticate owner receipts or authorize effects. Its next integration gate is an adapter to the v0.2 evaluator and a separately expressed differential oracle. At v0.3.1, 30 module unit tests and seven Proptest properties (256 cases each configured) were authored. The v0.3.2 stacked branch adds action-history coverage; see below.


## Event-order v0.3.1: qualified observer and durable clocks

The stacked follow-on [issue #12292](https://github.com/Luminous-Dynamics/symthaea/issues/12292) extends `EventObserved` and `EventPersisted` with independent source clock series, exact mapping-action references, local timestamps, receipt refs/digests and owner qualification. Simulator `at_ns` is scheduling time only; it is not copied into either qualified timestamp.

`generate_clocked_observation_persistence_schedule` exercises separate observer and durable-store clocks; `generate_capture_clock_fault_schedule` covers unqualified, expired, not-yet-valid and malformed-validity mappings on both domains. `project_bitemporal_event` accepts owner-normalized `QualifiedEventTime` resolutions keyed by the exact mapping action and source-local timestamp, checks receipt/source binding and clock validity, and then projects a v0.2 `BitemporalEventEnvelope`. Missing evidence is Indeterminate; substituted receipt identity or explicit owner failure is CriticalFailure. The adapter performs no clock normalization and no signature/trust-root verification.

The projection requires owner-qualified occurrence, first-observation and durable-record timestamps; enforces event-level observation/persistence cardinality; requires replay-applied production/observation/persistence actions; checks occurrence-before-observation and observation-before-durable-record ordering; and returns a schedule digest plus exact action/mapping/receipt sidecars. It does not yet reconstruct attempt lifecycle, scoped stop obligations, or a complete attempt inventory.

At this branch head, 30 module unit tests, seven Proptest properties (256 cases each configured), and one frozen integration fixture are authored, not executed. That is 1,792 potential property cases when run; no result is claimed without exact-head Rust execution.

## Event-order v0.3.2: owner-qualified action and stop-history projection

The stacked implementation is tracked by [issue #12316](https://github.com/Luminous-Dynamics/symthaea/issues/12316) and is based on the v0.3.1 bitemporal branch. Its versioned contract is [event-order-schedule-v0.3.2.schema.json](../../../docs/schemas/event-order-schedule-v0.3.2.schema.json); the v0.3.1 and original v0.3 schema files remain separate.

`ScheduledAction` carries optional exact `owner_time_ref` and typed `ActionEvidence` sidecars. Owner-time references must resolve to a prior clock-mapping action; for event-production, observation and persistence records the sidecar must match the action's native mapping ID and local timestamp. Simulator `at_ns` remains scheduler time and is never promoted to owner time.

The action projector consumes externally supplied owner-normalized `QualifiedEventTime` values and `OwnerReceiptAssessment` verdicts. Each receipt verdict is bound to the exact action ID, receipt reference and SHA-256 digest it actually checked, plus identity, digest, authenticity, scope and semantic gates. A `PASS` is not generated by the scheduler and is not a signature verifier.

`project_action_history` reconstructs v0.2 attempt starts, explicit retry ancestry, terminal reconciliation, effect-commit observations, scoped stop obligations, discharge receipts, and complete attempt-history inventory. A retry's predecessor-terminal reference is paired with the retry child's own start action (not the predecessor's start). Stop policy revisions may widen but not narrow the triggering scope; discharge and inventory scope checks use coverage, so a broader complete inventory may attest a narrower obligation. It then invokes the v0.2 lifecycle and start-eligibility evaluators. The inventory frontier must resolve to exactly one replay-applied produced event and precede its inventory attestation in owner time. A discharge must bind the exact trigger action, latest policy-revision action, scope, inventory frontier and attempt-manifest digest. Each attempt is bound to that exact discharge by the v0.2 per-attempt relation.

Cancellation acknowledgement remains a separate observation and never maps to `NOT_APPLIED`; its unique request and acknowledgement each require owner time, and the attempt must precede the request, which must precede the acknowledgement. An accepted acknowledgement with any effect commit is CriticalFailure. Retry is admissible only after the exact predecessor's qualified `NOT_APPLIED` terminal, with terminal-before-retry-start. An `APPLIED` terminal with an effect commit after it, or a `NOT_APPLIED` terminal with any linked effect commit, is CriticalFailure. Any combined projection failure suppresses the nested start-eligibility signal.

Stop discharge is evaluated against the exact policy generation current at that historical action, not the final policy after later revisions. A historically valid v1 discharge remains visible but does not clear a later v2 obligation; a widened undischarged policy retains the widened scope. The independent policy-generation reference run exercised 100,000 histories: 109,887 valid historical discharge instances, zero historical false rejects, zero stale/substituted discharge false accepts, zero scope-narrowing false accepts, and zero final-policy projection mismatches. These V8 checks are reference-model evidence, not Rust tests.


## Inventory frontier: snapshot plus append-only delta

A complete attempt inventory is a point-in-time commitment, not a digest of every record that exists later. The v0.2 compatibility API remains available, while `assess_start_under_stop_obligations_with_inventory_cut` separates:

- `inventory_snapshot_attempts`: the exact attempt state committed at the inventory frontier and checked against the attested manifest;
- `current_attempts`: the full currently known lineage used for lifecycle, requested-operation pending checks and exact stop-discharge relations.

The v0.3.2 projector reconstructs the snapshot using qualified owner time. Attempts that started before the frontier belong to the snapshot; terminal reconciliations strictly after it and stop-discharge relations whose discharge is after it are removed from the snapshot digest projection. A terminal or discharge already before the frontier remains in the snapshot. Attempts started after the frontier are an append-only delta and are not silently injected into the earlier manifest.

A post-frontier delta covered by a stop obligation is admissible only when its exact discharge is definitely before its start. A delta that began after the frontier but before discharge is a critical omission. A pending sibling after a global discharge can remain active without globally serializing an unrelated logical operation; a pending attempt for the requested logical operation still blocks eligibility. The full-history lifecycle report continues to expose the pending sibling as indeterminate even when the selected operation's start gate is safe.

Regression tests cover both sides of the cut, including post-discharge pending sibling admission and omitted pre-discharge attempt rejection. The frozen action-history manifest was corrected to exclude a future discharge relation; its new canonical v0.2 digest is `dd24fe30e2d4391a07629ba1e08990e5de7054bda2b393b62daeaac56bd3c865`.

An independent V8 reference model exercised 100,000 snapshot/delta cases across six families, with zero model violations: post-discharge pending sibling, omitted pre-discharge attempt, requested-operation pending attempt, ambiguous frontier order, post-cut terminal exclusion, and retention of already-completed stop relations. The reproducibility script and result are [here](../../../benchmarks/engineering-change-impact/event-order-v0.3.2-inventory-snapshot-delta-reference.mjs) and [here](../../../benchmarks/engineering-change-impact/event-order-v0.3.2-inventory-snapshot-delta-reference.json). This is reference-model evidence, not Rust execution or production reliability evidence.

The branch contains 86 authored Rust module unit tests across v0.2 and v0.3, eight Proptest properties (256 cases each configured; 2,048 potential property cases), and two integration tests. The frozen action-history known-answer input is [event-order-v0.3.2-action-history-known-answer.json](../../../benchmarks/engineering-change-impact/event-order-v0.3.2-action-history-known-answer.json); the seven-action observer/store fixture is [event-order-v0.3.2-clocked-observation-persistence.json](../../../benchmarks/engineering-change-impact/event-order-v0.3.2-clocked-observation-persistence.json). The projector returns schedule and replay SHA-256 digests plus exact terminal, stop-trigger/policy/discharge, inventory, effect and cancellation evidence sidecars. These are authored source counts and frozen inputs, not Rust execution. The independent JavaScript reference implementation is [script](../../../benchmarks/engineering-change-impact/event-order-v0.3.2-independent-reference.mjs) with its [v2 result record](../../../benchmarks/engineering-change-impact/event-order-v0.3.2-independent-reference.json). The policy-generation-specific model is [here](../../../benchmarks/engineering-change-impact/event-order-v0.3.2-stop-generation-reference.mjs), with its [100,000-history result](../../../benchmarks/engineering-change-impact/event-order-v0.3.2-stop-generation-reference.json). The recorded V8 model run includes 100,000 interval comparisons, 150,000 receipt-identity checks, 100,000 retry cases (20,000 child-start/predecessor binding variants), 90,000 stop/inventory cases (15,000 scope-expansion/coverage variants), 10,000 action-order permutations, and 100,000 cancellation/effect-terminal checks. Zero model violations were observed. The committed Node.js script is a reproducible implementation but was not itself executed in this run; none of these checks is Rust execution. No owner signature/trust-root implementation, physical-system validation, Rust compilation, clippy/fmt run or successful CI result is claimed.

