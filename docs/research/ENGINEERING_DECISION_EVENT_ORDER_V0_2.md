# Engineering Decision Event-Order v0.2 — Contract Kernel

**Status:** authored research specification and additive Rust source; compilation and execution pending.  
**Parent:** [Issue #12257](https://github.com/Luminous-Dynamics/symthaea/issues/12257).  
**v0.1 base:** [PR #12255](https://github.com/Luminous-Dynamics/symthaea/pull/12255).  
**Claim ceiling:** typed contract and source-level tests only; no execution authority, runtime assurance, performance result, or physical-safety claim.

## 1. Purpose

v0.2 adds an independent contract kernel without rewriting the v0.1 ledger or evaluator. The kernel consumes qualified owner outputs and determines whether three conditions are established:

1. event times are comparable in one explicitly named common basis;
2. execution attempts have a valid one-start / at-most-one-terminal lifecycle and retry lineage;
3. every applicable stop obligation has its own qualified, correctly bound discharge.

It also adds a bitemporal visibility query: what was durably knowable at decision frontier K.

## 2. Clock-basis qualification

A wall-clock value is not comparable merely because two sources both say PASS. A comparison requires:

- source clock-domain identity;
- target time-basis descriptor: identifier, epoch, scale and unit;
- mapping receipt reference and SHA-256 commitment;
- owner qualification of that mapping;
- mapping validity interval;
- bounded interval width.

Intervals with different time-basis IDs are UNKNOWN. An unqualified mapping is UNKNOWN; a failed qualification is INCONSISTENT. A mapping whose event interval falls outside its validity range is inconsistent input. Strict BEFORE/AFTER require non-touching closed intervals; touching intervals remain OVERLAPPING.

Causal edges and same-domain monotonic sequence evidence remain separate ordering sources. This kernel's time comparison does not manufacture causality or replace the v0.1 partial-order composition.

## 3. Bitemporal visibility

Each event separates:

- occurred_at: source/event time;
- first_observed_at: first collector observation;
- durably_recorded_at: evidence-store record time;
- supersedes_event_id: append-only correction/revocation relation.

An event is AVAILABLE_AT_K only when both first observation and durable recording are definitely BEFORE K in the qualified common basis. If both are AFTER K it is NOT_AVAILABLE_AT_K. Mixed, overlapping, unknown, inconsistent, or unqualified comparisons are INDETERMINATE_AT_K.

Occurrence time alone never proves that the event was knowable at K. First observation must not be definitely after durable recording. Corrections append a new record and preserve the historical record.

## 4. Attempt identity and reconciliation

The invariant-bearing unit is an execution attempt, not a logical operation.

- logical_operation_id groups attempts against one intended operation.
- attempt_id is unique for each execution attempt.
- An attempt record contains exactly one start event ID.
- It contains zero or one terminal reconciliation event ID.
- A retry links to exactly one predecessor attempt of the same logical operation.
- The predecessor must have one qualified PASS terminal reconciliation with outcome NOT_APPLIED.
- The ordering receipt must bind the exact predecessor terminal event ID and exact current retry-start event ID, and the predecessor terminal must be definitely BEFORE that start.
- APPLIED, missing, unqualified, overlapping, or unknown predecessor outcomes do not admit retry.

The report sorts pending IDs and reason codes, and validation builds its identity index independent of input delivery order.

## 5. Multiple scoped stop obligations

A stop obligation has its own identity, trigger, policy revision and scope (global or an explicit non-empty set of logical operation IDs).

Each discharge must independently bind:

- exact obligation ID;
- exact trigger event ID;
- exact policy revision;
- exact scope;
- receipt reference and digest;
- owner qualification PASS;
- trigger definitely BEFORE discharge.

A discharge receipt for obligation A cannot discharge obligation B. A scoped obligation cannot block unrelated operations. The discharge receipt also carries owner-native in-flight-reconciliation qualification for the exact scope at discharge time; FAIL is critical and an unresolved result blocks. This avoids confusing an operation that began after discharge with one that was already in flight when discharge was evaluated. The start gate separately blocks an unresolved prior attempt for the same logical operation. Multiple applicable obligations compose conjunctively: every one must be discharged.

start_permitted_by_stop_contract is an eligibility result only. It never grants effect authority and does not enforce a command at the executor boundary.

## 6. Tests authored in the v0.2 module

The source includes unit tests for:

- same-basis cross-domain comparison;
- mismatched-basis UNKNOWN;
- touching/overlapping intervals;
- unqualified and expired mappings;
- late-arriving event exclusion from historical frontier K;
- observed + recorded visibility at K;
- exact NOT_APPLIED retry lineage;
- retry before reconciliation;
- multiple roots and retry forks are rejected;
- retry ordering cannot be reused for a different pair of event IDs;
- terminal reconciliation binds the exact start event ID;
- in-flight inventory qualification binds a frontier event ID and canonical attempt-manifest digest;
- duplicate start and terminal cardinality;
- independent overlapping stop obligations;
- owner in-flight-inventory qualification required for discharge;
- unrelated pending attempts do not block independent scoped starts;
- unrelated operation outside a scoped stop;
- exact obligation/policy receipt binding;
- attempt-report invariance under delivery permutation.

These are authored tests, not executed results. The host in which this work was prepared has no cargo, rustc, rustfmt or nix executable. Exact-head CI remains separately reported and cannot be treated as local execution.

## 7. Research position and matched baselines

The kernel is not a new clock service, event ledger, transaction coordinator, executor, or PLM/ALM system of record.

Matched baselines should include:

1. wall-clock point sorting;
2. qualified interval-only comparison;
3. same-domain sequence-only ordering;
4. causal-graph-only ordering;
5. HLC/causal-only ordering;
6. v0.1 event-order evaluator;
7. v0.2 qualified time-basis + attempt + stop-obligation composition.

For transaction histories, use an established history checker such as Elle only on the subset whose consistency model and operation semantics match. Do not call this a general linearizability benchmark.

## 8. Qualification targets (not achieved results)

- zero false-safe outcomes on frozen and held-out hard negatives;
- 100% mismatch/unqualified clock-basis refusal;
- 100% one-start and at-most-one-terminal contract enforcement;
- 100% retry rejection without exact prior NOT_APPLIED and proven order;
- 100% independent discharge binding for overlapping stop obligations;
- 100% bitemporal exclusion of records not observed and durably recorded by K;
- at least 95% five-way order classification on blinded, independently adjudicated held-out traces.

Report confusion matrices, risk/coverage, disposition/reason code distribution, per-family errors, event throughput, comparison cost, decision latency, and lineage-aware sample counts. Mutation count is not an independent-trial denominator.

## 9. Next increments

1. Integrate the v0.2 time qualifier into the production partial-order comparator.
2. Add seed-recorded concurrent trace generation and deterministic shrinking.
3. Add a real owner-adapter fixture that verifies receipt bytes, signatures, trust roots, scope and policy revocation.
4. Add bitemporal correction/supersession replay and historical-at-K versus current-now tests.
5. Build a NIST IR 8536-compatible provenance demonstration over PLM, supplier, MES and physical-observation records.

## 10. Nonclaims

This kernel does not authenticate a clock mapping receipt; it consumes the owner qualification. It does not prove event completeness, physical sensor integrity, source honesty, physical safety, transaction linearizability or industrial superiority. Synthetic tests are regression controls, not population reliability evidence.


## 11. Generated property tests

The integration test target `tests/event_order_v0_2_properties.rs` uses Proptest for generated interval and attempt-chain properties, with shrinking of counterexamples. It covers:

- antisymmetry of the five-way comparator for valid same-basis intervals;
- translation invariance when both event intervals and their shared validity window shift by the same amount;
- attempt-lifecycle report invariance under reverse and rotated delivery orders for generated valid retry chains.

The properties complement the named unit cases; they do not simulate a real network, physical clock, executor, or owner verifier. On failure, preserve the Proptest seed and minimized counterexample with the exact commit. A passing property target is still not evidence of production clock calibration or physical event completeness.


The generated property target now also checks that an old occurrence timestamp cannot make late evidence available at K, and that one missing discharge blocks a start under any number of applicable overlapping obligations regardless of delivery order.
