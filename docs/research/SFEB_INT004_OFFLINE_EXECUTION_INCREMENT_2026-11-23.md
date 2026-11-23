# SFEB INT-004 — offline event reconciliation and oracle replay increment

**Date:** 2026-11-23  
**Parent:** [PR #13162](https://github.com/Luminous-Dynamics/symthaea/pull/13162)  
**Task:** SFEB-PILOT-INT-004  
**Status:** fixture-ready; in-memory V8 reference preflights executed; Rust, CLI and browser qualification pending

## What changed

INT-004 now has a deterministic synthetic work-board event fixture, independent JavaScript reference code, a standalone Rust reference crate, and an initial event-semantic verifier path. The fixture includes **seven golden cases**, one pointer/keyboard metamorphic relation, seven seeded negative controls, and 25 event deliveries.

The new composite golden case, `composite-independent-oracle-all-families`, was added specifically to avoid vacuous oracle success. Its raw trace includes:
- an optimistic action invalidated by a later authoritative transition;
- a pending protected action invalidated by explicit permission revocation;
- duplicate delivery of the same event identity and payload;
- owner reassignment and a subsequent authoritative state transition;
- a conflict event that blocks an ambiguous action;
- explicit conflict resolution; and
- a new action accepted only after resolution and acknowledged by the authoritative event stream.

The composite trace binds to exact fixture bytes with SHA-256:

`f93030379f3698680ff89703c42f29f10973e1deafb5e84610dbbdd24cbb2d79`

## Executed in-memory reference evidence

The in-memory V8 preflight campaign covers the seven golden cases, one pointer/keyboard metamorphic relation, seven negative controls and the causality-aware event replay campaign. The earlier six scenario checks and the new composite scenario were exercised against the checked-in fixture and expected outcomes.

The seven-rule composite verifier preflight is designed to make both permission revocation and clarification non-vacuous: the trace must contain the corresponding authoritative events and derive the expected rejected/accepted actions. The separate reference replay also checks that changing worker-supplied `disposition` / `observed_oracle_ids` cannot change the derived result, and that a modified authoritative event payload is rejected.

The event-order experiment enumerates **720 causally valid delivery permutations** of the six precondition events. Each permutation then submits the post-clarification action, delivers its authoritative acknowledgement, and replays all seven event identities in reverse. This is deliberately not an unconstrained 7! experiment: the acknowledgement is causally dependent on the user action.

These are in-memory reference-model results. They are not execution of the committed Node.js CLI or Rust crate and do not prove the Leptos candidate behaves correctly.

## Independent event-oracle replay source

The committed offline verifier path includes:
- a frozen seven-rule policy and schema, with policy digest pinned in the Rust verifier;
- a raw trace input schema and composite trace example;
- a structured event-oracle report schema;
- a Rust `--verify-event-trace` path that parses duplicate-key-free JSON, verifies exact fixture/policy bindings, replays the controller step sequence and exact authoritative event payloads, and derives results without consulting worker claims; and
- a dependency-free Node.js verifier source with a self-test for worker-claim invariance and altered-event rejection.

The verifier's scope is deliberately limited to deterministic event semantics. It is not a signed in-toto verifier statement, does not authenticate browser execution or workflow identity, and does not independently inspect screenshots, accessibility snapshots, layout artifacts or network-denial evidence.

## Rust and browser qualification boundary

The standalone Rust model/verifier source and **12 authored unit tests** are committed but have not been compiled or executed because this authoring environment lacks `cargo`/`rustc`. The committed Node.js CLI source is also execution-pending here. A Leptos work-board candidate and deterministic fixture-local event service still need implementation.

The task-family ledger records INT-004 as `fixture_ready`, `golden_cases_authored_rust_execution_pending`, and `browser_runner_pending`. The portfolio's `runnable_task_count` remains **0**. It is not a qualified benchmark task or a hidden holdout.

## Next implementation gates

1. Compile and run the standalone Rust reference/verifier on the pinned toolchain; resolve all differential failures against the JS reference.
2. Add full property and mutation campaigns over event identity, versions, permissions, idempotency, replay and recovery.
3. Implement the fixture-local streaming service with deterministic delay, duplication, disconnect and resume.
4. Implement the Leptos candidate with explicit keyboard alternatives and accessible pending/rejected/accepted announcements.
5. Execute fresh isolated Chromium journeys and capture raw event, DOM, accessibility, layout and network evidence.
6. Replay those artifacts in a separate verifier identity; derive outcomes from frozen policy and raw evidence, not worker-provided oracle IDs.
