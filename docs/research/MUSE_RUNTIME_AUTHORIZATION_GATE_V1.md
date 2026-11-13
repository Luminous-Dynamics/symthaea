# MUSE-RUNTIME-001 — One-shot authorization and unknown-outcome reconciliation

**Date:** 10 November 2026  
**Status:** pure Rust runtime transition contract and synthetic state-machine tests; no production renderer integration claimed  
**Owner:** https://github.com/Luminous-Dynamics/symthaea/issues/11540

## Research hypothesis

A content-bound one-shot authorization gate with an explicit OUTCOME_UNKNOWN state can prevent duplicate/stale audio mutations after timeouts and restarts, while preserving a replayable record of every authorized, dispatched, failed and committed edit attempt.

## State machine

1. AUTHORIZED — exact CQEP plan node, current registry, parent audio, active session, user authorization, operator, scope, preconditions, observers and budget all validated. The resource estimate is reserved.
2. DISPATCHED — dispatch receipt persisted before the external operator is invoked.
3. COMMITTED — exact output audio digest and execution receipt committed; only now does the runtime head advance.
4. FAILED_NO_MUTATION — definite failure with an external receipt proving no output was committed; the current head does not advance and a new attempt can be authorized.
5. OUTCOME_UNKNOWN — timeout, crash or ambiguous operator response after dispatch; no further mutation grant can be issued until reconciliation.

A timeout is not evidence of no mutation. The only transition out of OUTCOME_UNKNOWN is reconciliation tied to the original grant, plan/action and parent. Reconciliation either proves one exact committed output and advances the head, or proves no mutation and allows a new attempt.

An unused AUTHORIZED grant can be cancelled before dispatch with a durable cancellation receipt. Its resource reservation is not refunded in v1.

## V1 implementation

The crate tools/muse-runtime-authorization-gate:
- binds grant to plan digest/node, registry snapshot, exact parent audio, session ledger, action, operator ID/version/artifact and user authorization receipt;
- checks operator capabilities and registered locality;
- requires exact precondition receipt coverage;
- requires exact independent observer metric coverage against the frozen authority registry, including observer profile, evaluator artifact and calibration receipt;
- reserves compute, latency and risk budget once;
- rechecks exact audio, registry and session head before dispatch and before any outcome resolution;
- rejects reused/altered grants and wrong state transitions;
- supports cancellation of a grant that was authorized but never dispatched, retaining the cancellation receipt;
- blocks new grants during any AUTHORIZED, DISPATCHED or OUTCOME_UNKNOWN attempt;
- advances current audio only after a direct valid commit or verified reconciliation commit;
- retains definite failures and unknown outcomes in the attempt history.

## Safety boundary

The library is a transition validator, not a production transaction manager. It does not provide durable storage, host locking, database compare-and-swap, operator sandboxing, digital signature verification, model idempotency or exactly-once semantics. A production host must write the DISPATCHED ledger state durably before invoking the operator and use linearizable persistence/CAS to prevent races across processes. After an unknown result, the host must reconcile with the external operator/output store before authorizing another mutation.

Authorization and observer receipt digests are treated as externally verified evidence references. The library validates their syntax and binding; signature verification and provenance authentication remain the responsibility of the runtime authority.

## Synthetic validation matrix

Tests cover valid authorization/dispatch/commit, altered and reused grants, stale parent/registry/session, operator capability and locality, exact user-authorization binding, missing preconditions/observers, observer registry mismatch, budget reservation, unknown-outcome blocking, committed/no-mutation reconciliation, failure without no-mutation proof, cancellation before dispatch, commit-before-dispatch and wrong-parent resolution.

## Qualification targets

Targets, not observed results:
- 100% rejection of stale/altered/reused grants;
- zero mutation authorization while an outcome is unknown;
- zero audio-head advancement without exact commit evidence;
- 100% attempt retention;
- restart/replay reproduces the exact current head and unresolved attempt state;
- production host CAS prevents concurrent duplicate grants.

## Integration

CQEP-001 compiles a plan. VRCO-001 proposes a bounded control. This runtime gate is the sole authority to permit a mutating operator call. Independent post-render measurements then feed VCRB/VIA/VPTB/VELB, VCCR and VMEA. MMAE-TRACE preserves active constraints across multiple turns and finalizes only after final-output evidence.

## Claim boundary

Synthetic tests validate this crate's state-transition logic only. They do not qualify storage durability, host concurrency, operator behavior, signature trust, sandbox containment or actual audio quality.
