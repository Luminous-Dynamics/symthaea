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

## Covered behaviors

The six golden scenarios exercise optimistic rollback after a late authoritative transition, permission revocation, duplicate delivery, out-of-order replay/reconnect, conflict clarification, and pointer/keyboard semantic equivalence. Four negative controls must be rejected: changed payload under a reused event identity, a second identity claiming the same sequence, an unauthorized actor, and an illegal transition.

Events are applied only in contiguous sequence order. Duplicate delivery of the same event identity and payload is idempotent. A changed payload under a previously seen identity is rejected. Pending actions remain visibly optimistic but are not authoritative state; the authoritative event either confirms them or invalidates them.

## Evidence boundary

The fixture is public development material. A successful JavaScript run checks only the independent JavaScript model against the frozen fixture. Rust code and tests are authored but must be compiled and executed on a Rust-equipped host. No Leptos work-board candidate, browser stream adapter, Chromium journey, full keyboard/accessibility verification, isolated egress policy, signed execution statement, or independent trusted browser verifier is included. This fixture does not make INT-004 runnable or qualified.
