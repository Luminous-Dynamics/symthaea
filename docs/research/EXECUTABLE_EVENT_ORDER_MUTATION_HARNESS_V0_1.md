# Distributed Event-Order Executable Mutation Harness v0.1

**Status:** authored research harness; Rust execution pending  
**Test target:** `crates/domains/symthaea-decision-assessment/tests/event_order_mutations.rs`  
**Claim ceiling:** test definitions and source-level binding only; no execution result or physical-safety claim.

## Purpose

The 26 frozen mutation controls are now connected to Rust integration tests. Each test selects a frozen base trace, applies one deterministic mutation, deserializes typed events and owner qualification, and calls the actual `evaluate_event_order` consumer. The tests assert expected disposition and reason-code prefix. Malformed traces must return `InvalidEventTrace`; event-delivery reversal must preserve the full serialized report; a valid post-discharge operation is retained as a positive control.

There is one Rust test per mutation control plus a manifest-completeness test. Authorship is not execution. The manifest remains `execution_status: not_run` until this target runs on a supported Rust toolchain.

## Mutation families

| Controls | Family | Challenge |
|---|---|---|
| 001–006 | Trace integrity | Duplicate identity, missing predecessor, cycle, invalid interval, duplicate start, duplicate terminal reconciliation |
| 007–009 | Temporal order | Remove decisive causal edge, remove qualified order sources, contradict qualified clock and causal order |
| 010–014 | Stop discharge | Missing discharge, discharge-before-trigger, malformed digest format, wrong trigger/discharge binding |
| 015–020 | Operation lifecycle | Cancellation request/ack order; APPLIED/NOT_APPLIED consistency; start/effect/outcome chronology |
| 021–023 | Retry and resume | Retry before exact reconciliation, retry before discharge, new start before discharge |
| 024 | Delivery invariance | Reverse event delivery and compare serialized report bytes |
| 025 | Qualification | Unqualified causal edges cannot resolve overlapping intervals |
| 026 | Positive control | Valid discharge followed by correctly ordered new operation remains admissible |

## Receipt-digest boundary

The consumer receives `stop_disposition_receipt_sha256` only after owner-native receipt validation. It checks presence, format and exact trigger/discharge event-ID binding. It does not possess receipt bytes or a trust root and therefore cannot independently detect substitution of one valid-looking digest for another. MUT-EVT-012 tests malformed 63-character digest format only. Content integrity, authenticity and digest-to-receipt binding remain upstream receipt-owner responsibilities.

A future owner-adapter suite should inject real receipt bytes, digest mismatches, signature failures, scope mismatches and trust-root revocation into the native verifier, then verify that the consumer receives `Fail` or `Indeterminate` as appropriate.

## Execution

From the workspace root:

```sh
cargo test -p symthaea-decision-assessment --test event_order_reference --test event_order_mutations
```

Record exact commit, `rustc -Vv`, `cargo -V`, test count, exit status and failure output. Run formatting and clippy separately. Preserve workflow states exactly as success, failure, cancelled, skipped or pending. Do not mark the preregistration passed because tests compile. A zero-failure synthetic suite is a regression gate, not a population reliability estimate.

## Next research gates

1. Deterministically generate traces across operation IDs, clock domains, interval widths, event-delivery permutations, missing telemetry and overlapping in-flight operations.
2. Check temporal properties: one start and at most one terminal reconciliation per operation; every reported order is effect-relative-to-trigger; no retry until exact NOT_APPLIED reconciliation; no post-trigger new operation or effect until exact discharge; no discharge while any operation remains unresolved.
3. Differential-check against an independently maintained executable specification and, only where applicable, established concurrency/history checkers. This is not a database linearizability benchmark and does not claim to outperform Spanner, Jepsen/Elle or Porcupine.
4. Add blinded, independently authored held-out traces and explicit cross-domain clock calibration/error models.
5. Build a real adapter demonstration spanning PLM/ALM, supplier, MES and physical observations while preserving native system-of-record authority.
