# SFEB-PILOT-INT-004 offline event-reconciliation fixture

This directory contains a deterministic, synthetic event schedule and a standalone Rust reference-model crate for the dynamic work-board candidate.

## Run the independent reference

From the repository root:

```sh
node benchmarks/frontend/independent-oracle/reconcile-events.mjs
```

## Run the Rust reference and tests

```sh
cargo run --manifest-path benchmarks/frontend/long-horizon/Cargo.toml
cargo test --manifest-path benchmarks/frontend/long-horizon/Cargo.toml
```

The Rust crate is intentionally standalone so it can be run while the main workspace or browser runners are unavailable.

## Run the independent event semantic verifier

```sh
cargo run --manifest-path benchmarks/frontend/long-horizon/Cargo.toml -- --verify-event-trace
```

The Rust verifier reads the raw example trace, frozen fixture and pinned seven-rule policy. It binds the exact fixture bytes, checks the controller's frozen step sequence, replays delivered events through the independent state reducer and emits a structured report. Worker-provided `disposition` and `observed_oracle_ids` are untrusted diagnostics and do not affect the derived verdict. The policy digest is pinned in the verifier source. A separate dependency-free Node.js implementation and self-test are available:

```sh
node benchmarks/frontend/independent-oracle/verify-event-trace.mjs --self-test
```

The JavaScript verifier source is authored but has not been executed in this authoring environment. The earlier 7/7 V8 preflight was an independent in-memory implementation against the exact fixture/policy/trace bytes, not execution of either committed verifier file.

## Covered behaviors

The six golden scenarios exercise optimistic rollback after a late authoritative transition, permission revocation, duplicate delivery, out-of-order replay/reconnect, conflict clarification, and pointer/keyboard semantic equivalence. The reference harness also exhausts all 720 causally valid delivery orders for the six precondition events, then submit the post-clarification action and deliver its acknowledgement and replays every event identity in reverse after each order; every order must converge to the same state with exactly-once audit effects. Seven negative controls must be rejected: (1) changed payload under a reused event identity; (2) a second identity claiming the same sequence; (3) an unauthorized actor; (4) an illegal transition whose rejected delivery must leave the event identity and sequence free for a valid retry; (5) an excessive reorder gap; and (6) an unknown event field; and (7) a second optimistic action for a job that already has one pending.

Event IDs and fields are strictly checked, duplicate JSON object keys are rejected by the Rust input parser, integer sequence/version fields are bounded to the JavaScript-safe range, and out-of-order buffering is limited to a 1,024-sequence window and 1,024 pending events. Events are applied only in contiguous sequence order. Duplicate delivery of the same event identity and payload is idempotent. A changed payload under a previously seen identity is rejected. Pending actions remain visibly optimistic but are not authoritative state; the authoritative event either confirms them or invalidates them.

The report is scoped to deterministic event semantics. It is not a signed in-toto statement, does not authenticate browser execution, does not verify screenshots/accessibility/network artifacts and does not qualify the Leptos candidate.

## Evidence boundary

The fixture is public development material. A successful JavaScript run checks only the independent JavaScript model against the frozen fixture. Twelve Rust tests, including the 720-order causality-aware campaign, are authored but must be compiled and executed on a Rust-equipped host. No Leptos work-board candidate, browser stream adapter, Chromium journey, full keyboard/accessibility verification, isolated egress policy, signed execution statement, or independent trusted browser verifier is included. This fixture does not make INT-004 runnable or qualified.
