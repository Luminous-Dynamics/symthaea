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

The fixture includes six golden scenarios, one pointer/keyboard metamorphic pair, and six negative controls, including unsafe stream-gap and unknown-field attempts. Event identity, sequence ownership, entity versions, actor permissions and state-machine transitions are independently represented. The event schedule has no live service, wall clock, credentials, or third-party network dependency.

## Executed preflight

The independent JavaScript model was executed against the fixture bytes:

- Golden scenarios: **6/6**
- Pointer/keyboard metamorphic relation: **1/1**
- Adversarial negative controls rejected: **6/6**
- Event deliveries represented in golden traces: **17**
- Fixture SHA-256: `f9f642acd996300068f0b0dd9a8a428093bfec6fa1236d27d8ee774370e3669a`

This is a reference-model result. It does not demonstrate that a browser candidate behaves correctly.

## Rust and browser qualification boundary

A standalone Rust reference model, offline trace-to-oracle evaluator, pinned seven-rule policy and twelve authored unit tests are included. The event contract rejects duplicate JSON object keys, unknown event fields and unsafe numeric values, and caps out-of-order buffering at a 1,024-sequence window / 1,024 pending events. Event delivery is transactional: a rejected event cannot reserve its identity or sequence, alter the buffer, or poison a later valid retry. The Rust source and twelve tests have **not** been compiled or executed in this authoring environment because `rustc`/`cargo` are unavailable. The task still requires a Leptos candidate, fixture-local streaming adapter, real pointer and keyboard browser journeys, full accessibility-tree checks, deterministic Chromium execution, egress isolation, and a verifier in a separate trust domain.

The task-family ledger records INT-004 as `fixture_ready`, `golden_cases_authored_rust_execution_pending`, and `browser_runner_pending`. The portfolio's `runnable_task_count` remains **0**. Public pilot cases are not hidden holdouts and are not used as comparative leaderboard evidence.



## Independent event-oracle replay increment

Added a frozen task policy, raw execution-trace schema/example, verifier report schema and a standalone Rust trace-to-oracle path:
- `event-oracle-policy-v1.json` and schema: seven required rules, pinned by SHA-256 `dddaaa445881feabc3dc7d3599c8741004917190b2b2467db65cf1c42d7b6d5c` in the Rust verifier source.
- `event-execution-trace-v1.schema.json` and `examples/late-event-rollback-trace.json`: raw action/event steps bind to the exact fixture SHA-256.
- `event-oracle-verification-v1.schema.json`: structured event-semantics report, rule hashes and canonical evidence-event digests.
- `--verify-event-trace`: checks the frozen controller step sequence, replays event payloads through the reducer, derives seven semantic outcomes, and ignores worker `disposition` / `observed_oracle_ids`.
- `benchmarks/frontend/independent-oracle/verify-event-trace.mjs`: dependency-free Node.js implementation with a `--self-test` covering worker-claim invariance and altered-event rejection. The source is committed but has not been run as a Node CLI here.

The Rust verifier's raw JSON path now rejects duplicate object keys recursively before constructing `serde_json::Value`. A generated event-order campaign now exhausts all 5,040 permutations of the seven authoritative events; every order must converge to the same authoritative state, and reverse replay of all seven event identities must remain idempotent.

An independent in-memory V8 preflight against the exact repository fixture, policy and example trace passed **7/7 policy rules**. A separate in-memory V8 state-reducer campaign passed **5,040/5,040 event delivery permutations**, each followed by seven reverse-order duplicate replays. Mutating worker claims left all oracle outcomes unchanged; altering a frozen event payload was rejected. These are independent in-memory V8 checks against the exact fixture/policy/trace bytes, not executions of the committed Node.js CLI or Rust verifier. The permutation reducer is separately authored in the Rust test source and has not been compiler-executed. The verifier does not authenticate browser execution, produce a signed in-toto verifier statement or inspect browser screenshots/accessibility/network artifacts. It is an initial semantic replay component for #13247 only.

Trace SHA-256: `a6db51ee92df248161b15975027e65c13a475b20ae67a5e2b16ccb4b566d7fc0`.

## Next implementation gates

1. Compile and run the standalone Rust reference on the pinned toolchain; resolve all discrepancies against the JS oracle.
2. Build a deterministic local event service with controllable delivery delay, duplication, disconnect and resume.
3. Implement the Leptos work-board candidate with a non-drag keyboard action path and accessible pending/rejected/accepted announcements.
4. Drive the candidate through a fresh isolated Chromium session and capture raw event, DOM, accessibility, layout and network evidence.
5. Replay the evidence in a separate verifier identity; derive verdicts from raw traces and frozen oracle policy, not candidate- or worker-supplied oracle IDs.

INT-004 complements rather than replaces L2-005. The first independently verified end-to-end engineering change remains the priority; both tasks should ultimately exercise the same evidence execution architecture.
