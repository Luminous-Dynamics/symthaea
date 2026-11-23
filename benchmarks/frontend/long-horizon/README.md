# SFEB-PILOT-INT-004 offline event reconciliation

This public-pilot fixture defines a deterministic, synthetic work-board event stream and two independent reference implementations. It has no live service, wall clock, credentials or third-party network dependency.

## Run the independent references

From the repository root:

```sh
node benchmarks/frontend/independent-oracle/reconcile-events.mjs
node benchmarks/frontend/independent-oracle/verify-event-trace.mjs --self-test
cargo run --manifest-path benchmarks/frontend/long-horizon/Cargo.toml
cargo test --manifest-path benchmarks/frontend/long-horizon/Cargo.toml
cargo run --manifest-path benchmarks/frontend/long-horizon/Cargo.toml -- --verify-event-trace
```

The committed Node.js scripts and Rust crate are authored source. Their CLI/test execution has not been completed in the current authoring environment because the local Rust toolchain is unavailable; the Node scripts are also not being represented as executed here. The separately executed in-memory V8 preflights are recorded in the research note.

## Fixture coverage

The fixture has seven golden cases, one pointer/keyboard metamorphic relation and seven negative controls. Across all golden traces it represents 25 event deliveries.

The cases cover:
- optimistic action rollback after a late authoritative event;
- permission revocation while a protected action is pending;
- duplicate delivery and exactly-once audit effects;
- out-of-order delivery and reconnect convergence;
- clarification before a protected mutation;
- pointer/keyboard semantic equivalence; and
- one composite trace that exercises all seven event-oracle policy families in a single replay, including a real permission revocation, a real clarification conflict, a duplicate delivery and a final acknowledged action.

The negative controls require rejection of: (1) a changed payload under a reused event identity; (2) a second identity claiming the same sequence; (3) an unauthorized actor; (4) an illegal transition whose rejected delivery leaves its identity and sequence free for a valid retry; (5) an excessive reorder gap; (6) an unknown event field; and (7) a second optimistic action for a job that already has one pending.

## Bounded replay and causal-order campaign

Event IDs and fields are strictly checked. The Rust raw-input parser rejects duplicate JSON object keys recursively. Sequence and entity-version values are restricted to safe positive integers. Out-of-order buffering is bounded to a 1,024-sequence window and 1,024 events. Event delivery is transactional: a rejected event cannot reserve its identity or sequence, alter the buffer/audit history, or poison a later valid retry.

The reference campaign exhausts all 720 delivery orders of the six precondition events. For every order, it then submits the post-clarification action, delivers its authoritative acknowledgement, and replays all seven event identities in reverse. This preserves the causal dependency between conflict resolution and acknowledgement; unconstrained permutations that deliver an acknowledgement before its action are not counted as valid executions.

## Independent event semantic verifier

The Rust CLI path `--verify-event-trace` consumes:
- the raw event trace and its strict schema;
- the exact frozen fixture bytes and digest;
- the frozen seven-rule event policy, whose SHA-256 is pinned in verifier source; and
- an independently implemented state reducer.

The report binds the raw trace, fixture, policy and rule/evidence digests. Worker-provided `disposition` and `observed_oracle_ids` are untrusted diagnostics and do not determine the derived result. The composite demonstration trace contains every event family needed to exercise permission-revocation and clarification rules non-vacuously.

This verifier is an offline event-semantics sub-verifier only. It is not a signed in-toto verifier statement, does not authenticate browser execution identity, does not inspect screenshot/accessibility/network artifacts, and does not qualify the Leptos candidate.

## Evidence boundary

The 2026-11-23 in-memory V8 checks are reference-model preflights, not executions of the committed Node.js CLI or Rust crate. Rust tests/fmt/Clippy, Leptos/WASM compilation, the work-board candidate, browser journeys, accessibility checks, isolated egress, signed execution identity and separate verifier workflow remain pending.

INT-004 is fixture-ready, not runnable or independently qualified. SFEB's task-family ledger correctly retains `runnable_task_count = 0`. This is public development material, not a hidden holdout or leaderboard result.
