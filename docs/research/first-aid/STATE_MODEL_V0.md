# First-Aid Workflow Trust State Model v0

**Status:** research model; not clinical validation, certification, or deployment authorization.  
**Implementation:** `xtask/src/first_aid_model.rs`  
**Claim ceiling:** finite abstract state/action exploration only. It does not prove clinical correctness or patient safety.

## Purpose

The trace checker validates JSON event histories. This model expresses a smaller state-transition system independently of the JSON parser and exhaustively explores its reachable abstract states. Its aim is to catch trust-state bugs that hand-authored traces might miss.

## Abstract state

Each state is the tuple:

`S = (session_active, connectivity, loaded, verified, revoked_mask, uncertainty_pending, uncertainty_preserved)`

- `session_active`: whether the active workflow session is live.
- `connectivity`: current modeled network state (`online`, `offline`, or `intermittent`), updated by explicit events and read when a network-dependent action occurs.
- `loaded`: optional bundle identity, abstracted to A or B.
- `verified`: optional identity whose verification currently authorizes rendering.
- `revoked_mask`: monotonic bit set for bundle A and B.
- `uncertainty_pending`: an uncertainty was observed and remains unresolved for handoff.
- `uncertainty_preserved`: the pending uncertainty has been explicitly preserved for a handoff.

The two bundle identities are a finite abstraction for identity mismatch, not a model of actual CPIC or first-aid content. Verification actions vary validity state and digest verification. Session start, load, verify, revoke, interrupt, resume, uncertainty detection/preservation, handoff, render, connectivity changes, a core-network-dependent action, six resource freshness/presentation combinations, four sensitive-data/consent combinations, and unknown normalization comprise the 31-action alphabet.

## Required invariants

1. **Verification precondition:** verification succeeds only inside an active session, for the currently loaded matching bundle, with valid bundle state, a verified digest, and no prior revocation.
2. **Render authority:** a render without a violation requires an active session, verification of that exact bundle, and absence of revocation.
3. **Revocation monotonicity:** no transition clears a recorded revocation.
4. **Session boundary:** starting a session clears loaded/verified bundle state. An unpreserved unresolved uncertainty produces a boundary violation rather than silently disappearing.
5. **Interruption and resume:** interruption clears verification; every resume clears verification even if the trace is malformed and lacks an interruption event.
6. **Handoff:** handing off while uncertainty is pending and unpreserved emits an uncertainty-loss violation.
7. **Unknown-state default:** the implementation must not treat an invalid digest, expired bundle, missing verification, or identity mismatch as verified.

## Explicit state invariants checked on every edge

For every reachable state and every one of the 31 actions, the test checks both the state before the action and the successor state:

- The revocation mask contains only the two modeled identity bits and is monotonic across every transition.
- A verified identity can exist only in an active session, must equal the currently loaded identity, and must not be revoked.
- Preserved uncertainty implies uncertainty is still pending until handoff consumes it.
- Starting a new session clears loaded/verified identity state; interruption and resume clear verification.
- Every verification action is checked against its exact preconditions, including matching identity, active session, validity state, digest result, and revocation status. Failed attempts must clear verification and emit the expected invalid-verification violation.
- Every render action is checked against exact violation precedence: inactive-session violation can co-occur with the render's identity/trust violation; a revoked render reports revocation rather than also emitting the generic missing-verification violation.
- Load clears prior verification, and revocation immediately invalidates verification for the same bundle.

These assertions strengthen the transition checker beyond a reachable-state count. The fixed count (252 states / 7,812 edges) is still used as a review trigger—not as a substitute for the invariant checks.

## Exploration method

The checker performs breadth-first exploration from the initial state. For each distinct reachable state it applies each of the 31 declared abstract actions, checks the next state and violations against the invariants, and enqueues newly discovered states until reaching a fixed point. A finite fixed-point traversal avoids relying on sampled traces for this abstract model.

The exact lifecycle/connectivity graph is pinned at 252 reachable abstract states and 7,812 state/action transitions; the Rust test prints both counts and fails if either changes. A changed count requires reviewing the state/action space rather than silently accepting a smaller graph.

The test suite now stores a breadth-first shortest event-history witness for each reachable abstract state. For every state it appends every one of the 31 actions, feeds each resulting history through the production JSON trace checker, and compares the ordered violation codes to those returned by replaying the abstract model. This gives 252 × 31 = 7,812 model-to-production edge comparisons, in addition to state invariants and the 18 orthogonal policy truth tables. A further bounded-sequence test appends every ordered pair of actions to each state's shortest witness, compares the full ordered violation list against the production checker, and covers 252 × 31 × 31 = 242,172 two-action suffix sequences. The CI test step extracts the actual comparison counts from test output, asserts 7,812 edges, 242,172 two-action suffixes, and 18 policy cases, and passes those values into trajectory receipt schema v1. The receipt gate rejects missing or wrong counts and requires `first-aid-trajectory-receipt-v1`. This instrumented evidence is still pending successful exact-head execution. It is a same-repository differential check: a matching result catches implementation divergence, but shared requirements and abstractions mean it is not an independent clinical oracle. The newest exact-head workflow must pass before these comparisons count as verified evidence.

A second exhaustive truth-table test covers 18 **orthogonal policy combinations** that are deliberately kept outside the lifecycle product graph to prevent a Cartesian state-space explosion. A further differential test feeds each of those 18 combinations to the production JSON trace checker and compares its result with the abstract policy helper:

- Six connectivity/requirement combinations: online, offline, or intermittent connectivity when the core workflow either does or does not require the network.
- Six resource freshness/presentation combinations: fresh, stale, or unknown resource state, each presented as confirmed or unconfirmed.
- Four data sensitivity/consent combinations: sensitive or non-sensitive payload, with or without consent scope.
- Two unknown-normalization cases: no normalization event vs. an explicit unknown-normalization event.

Connectivity state is also included in the reachable lifecycle graph: the model can switch online/offline/intermittent before a core-network-dependent action, and the production JSON checker is compared against that ordered history. The 18 orthogonal truth-table probes remain useful local checks for connectivity/requirement, freshness/presentation, consent, and unknown normalization, but they do not model real packet loss, retries, timeouts, clock-based freshness, consent-change semantics, or all cross-domain temporal interactions. The dedicated workflow binds the model source digest, model-specification digest, and test outcome into the trace-contract receipt.

## What it does not establish

The model is intentionally small. It does not include concurrent event delivery, clock/freshness semantics, cryptographic implementation behavior, real network recovery, UI rendering, actual human interactions, clinical protocol correctness, variant interpretation, or patient outcomes. Its transition rules are authored and maintained in the same repository as the production checker. Differential agreement is therefore an implementation-diversity signal, not independent ground truth.

A stronger external claim would require an independently reviewed formal specification, a separately implemented oracle, held-out generated traces, qualified domain adjudication, and separate authorization for any clinical evaluation.
