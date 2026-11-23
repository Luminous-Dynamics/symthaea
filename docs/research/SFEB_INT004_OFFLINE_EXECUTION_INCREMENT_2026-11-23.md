# SFEB INT-004 — offline event-reconciliation increment

**Date:** 2026-11-23  
**Parent:** [PR #13162](https://github.com/Luminous-Dynamics/symthaea/pull/13162)  
**Task:** SFEB-PILOT-INT-004  
**Status:** fixture authored; independent JavaScript model preflight passed; Rust execution and browser qualification pending

## What changed

INT-004 has moved from an unauthored brief to a fixture-ready public-pilot task. The new deterministic event schedule covers:

- a late authoritative status transition invalidating an optimistic local action;
- permission revocation invalidating a pending protected action;
- duplicate event delivery with exactly-once audit effects;
- out-of-order reconnect delivery buffered until the missing sequence arrives;
- conflict detection that blocks mutation until explicit clarification is resolved;
- equivalent semantic outcomes for pointer and keyboard action paths.

The fixture includes six golden scenarios, one pointer/keyboard metamorphic pair, and four negative controls. Event identity, sequence ownership, entity versions, actor permissions and state-machine transitions are independently represented. The event schedule has no live service, wall clock, credentials, or third-party network dependency.

## Executed preflight

The independent JavaScript model was executed against the fixture bytes:

- Golden scenarios: **6/6**
- Pointer/keyboard metamorphic relation: **1/1**
- Adversarial negative controls rejected: **4/4**
- Event deliveries represented in golden traces: **17**
- Fixture SHA-256: `b3c09e0a7c343c058d271e97c24f576ad64e2b52a5f8d84c354b7d570e26a703`

This is a reference-model result. It does not demonstrate that a browser candidate behaves correctly.

## Rust and browser qualification boundary

A standalone Rust reference model and five authored unit tests are included. Event delivery is transactional: a rejected event cannot reserve its identity or sequence, alter the buffer, or poison a later valid retry. They have **not** been compiled or executed in this authoring environment because `rustc`/`cargo` are unavailable. The task still requires a Leptos candidate, fixture-local streaming adapter, real pointer and keyboard browser journeys, full accessibility-tree checks, deterministic Chromium execution, egress isolation, and a verifier in a separate trust domain.

The task-family ledger records INT-004 as `fixture_ready`, `golden_cases_authored_rust_execution_pending`, and `browser_runner_pending`. The portfolio's `runnable_task_count` remains **0**. Public pilot cases are not hidden holdouts and are not used as comparative leaderboard evidence.

## Next implementation gates

1. Compile and run the standalone Rust reference on the pinned toolchain; resolve all discrepancies against the JS oracle.
2. Build a deterministic local event service with controllable delivery delay, duplication, disconnect and resume.
3. Implement the Leptos work-board candidate with a non-drag keyboard action path and accessible pending/rejected/accepted announcements.
4. Drive the candidate through a fresh isolated Chromium session and capture raw event, DOM, accessibility, layout and network evidence.
5. Replay the evidence in a separate verifier identity; derive verdicts from raw traces and frozen oracle policy, not candidate- or worker-supplied oracle IDs.

INT-004 complements rather than replaces L2-005. The first independently verified end-to-end engineering change remains the priority; both tasks should ultimately exercise the same evidence execution architecture.
