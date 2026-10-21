# First-Aid Workflow Trust State Model v0

**Status:** research model; not clinical validation, certification, or deployment authorization.  
**Implementation:** `xtask/src/first_aid_model.rs`  
**Claim ceiling:** finite abstract state/action exploration only. It does not prove clinical correctness or patient safety.

## Purpose

The trace checker validates JSON event histories. This model expresses a smaller state-transition system independently of the JSON parser and exhaustively explores its reachable abstract states. Its aim is to catch trust-state bugs that hand-authored traces might miss.

## Abstract state

Each state is the tuple:

`S = (session_active, loaded, verified, revoked_mask, uncertainty_pending, uncertainty_preserved)`

- `session_active`: whether the active workflow session is live.
- `loaded`: optional bundle identity, abstracted to A or B.
- `verified`: optional identity whose verification currently authorizes rendering.
- `revoked_mask`: monotonic bit set for bundle A and B.
- `uncertainty_pending`: an uncertainty was observed and remains unresolved for handoff.
- `uncertainty_preserved`: the pending uncertainty has been explicitly preserved for a handoff.

The two bundle identities are a finite abstraction for identity mismatch, not a model of actual CPIC or first-aid content. Verification actions vary validity state and digest verification. Session start, load, verify, revoke, interrupt, resume, uncertainty detection/preservation, handoff, and render actions comprise the action alphabet.

## Required invariants

1. **Verification precondition:** verification succeeds only inside an active session, for the currently loaded matching bundle, with valid bundle state, a verified digest, and no prior revocation.
2. **Render authority:** a render without a violation requires an active session, verification of that exact bundle, and absence of revocation.
3. **Revocation monotonicity:** no transition clears a recorded revocation.
4. **Session boundary:** starting a session clears loaded/verified bundle state. An unpreserved unresolved uncertainty produces a boundary violation rather than silently disappearing.
5. **Interruption and resume:** interruption clears verification; every resume clears verification even if the trace is malformed and lacks an interruption event.
6. **Handoff:** handing off while uncertainty is pending and unpreserved emits an uncertainty-loss violation.
7. **Unknown-state default:** the implementation must not treat an invalid digest, expired bundle, missing verification, or identity mismatch as verified.

## Exploration method

The checker performs breadth-first exploration from the initial state. For each distinct reachable state it applies each of the 16 declared abstract actions, checks the next state and violations against the invariants, and enqueues newly discovered states until reaching a fixed point. A finite fixed-point traversal avoids relying on sampled traces for this abstract model.

The exact count of reachable states and state/action transitions is printed by the Rust test run. The current source also asserts lower bounds on those counts to catch accidental removal of action coverage. The dedicated workflow binds the model source digest and its test outcome into the trace-contract receipt.

## What it does not establish

The model is intentionally small. It does not include concurrent event delivery, clock/freshness semantics, cryptographic implementation behavior, real network recovery, UI rendering, actual human interactions, clinical protocol correctness, variant interpretation, or patient outcomes. Its transition rules are authored and maintained in the same repository as the production checker. Differential agreement is therefore an implementation-diversity signal, not independent ground truth.

A stronger external claim would require an independently reviewed formal specification, a separately implemented oracle, held-out generated traces, qualified domain adjudication, and separate authorization for any clinical evaluation.
