# Symthaea IoT Actuation Linearization

This crate establishes one non-portable, closure-scoped convergence point for privileged IoT actuation.

It consumes one `ComposedActuationEvidence`, then holds the independently mutable roots stable in this exact order:

1. atomic actuation-trust publication;
2. durable admission reservation;
3. durable semantic head;
4. current Xenia transport;
5. current admission-bound device reality; and
6. current post-semantic controller/interlock evidence.

Only after all six have converged does the crate read one common wall clock. The physical-effect envelope deadline is intersected with the transport, device-reality, and controller/interlock natural validity ceilings. A fixed 250 ms monotonic handoff ceiling starts only after that common wall-clock check succeeds.

`CurrentActuationAttempt<'a>` owns every held/current fence and borrows the consumed composition. It is non-clone and non-serializable. `ActuationLinearizer::with_current_attempt` uses a higher-ranked `FnOnce` scope, so the callback result cannot borrow the attempt or retain its locks after the scope ends.

## What this proves

Within the callback scope:

- the trust/policy publication cannot advance;
- the admission-reservation journal cannot advance;
- the semantic journal cannot advance;
- the exact Xenia transport proof remains current under the published transport trust head;
- the exact signed device appraisal remains current under the published device-reality trust/policy;
- the exact controller/interlock proof remains current under the published controller trust/policy;
- the durable admission and semantic checkpoint objects are byte-equivalent at the Rust-value level to those retained by composition;
- the common wall clock has not regressed behind any owner-local currentness check; and
- the common wall and monotonic handoff windows are still open.

## What this does not prove

This crate deliberately does **not**:

- perform a HAL/device write;
- expose a network transport;
- mint a reusable final permit or lease;
- serialize an actuation attempt;
- select transport, device-verifier, or controller keys;
- duplicate signature verification;
- prove that hardware actually executed the requested effect; or
- provide hardware anti-rollback for the durable journals.

A later HAL adapter should accept `CurrentActuationAttempt<'a>` by value, call `validate_dispatch_window_now()` as its final pre-effect operation, perform at most one narrowly typed physical effect, and return an evidence-bearing result rather than a reusable capability.

## Promotion status

Candidate only. Promotion requires the stacked parents (#489, #492, and #497) to settle, this exact-head workflow to pass, a real end-to-end linearization regression using the independently verified two-branch Xenia chain, and intentional final workspace `Cargo.lock` bookkeeping.
