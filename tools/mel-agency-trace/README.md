# MEL Agency Trace — reference evaluator

This standalone Rust crate is a deterministic reference evaluator for the proposal-to-decision lifecycle in [MEL-013 co-creative agency](https://github.com/Luminous-Dynamics/symthaea/issues/11319).

## Run

From the repository root:

    cargo run --manifest-path tools/mel-agency-trace/Cargo.toml -- tools/mel-agency-trace/fixtures/apply-approved-undo-v1.json
    cargo run --manifest-path tools/mel-agency-trace/Cargo.toml -- tools/mel-agency-trace/fixtures/suggest-only-v1.json

Run focused tests:

    cargo test --manifest-path tools/mel-agency-trace/Cargo.toml

The crate is an isolated nested Cargo workspace. It does not depend on the root Symthaea workspace or production Studio code.

## What is checked

- append-only event sequence and SHA-256 hash-chain integrity;
- proposal commitment over source, candidate, scope, protected set, requested change and edit-plan digests;
- preview and authorization bound to the exact same proposal commitment;
- explicit autonomy mode (suggest-only, preview-only, apply-approved or bounded exploration);
- authorization expiry and exact event reference;
- veto precedence over a pending authorization;
- no-op rejection;
- exact candidate identity at commit;
- preservation-verifier receipt reference;
- undo references the exact commit and restores the original source digest.

## Claim boundary

This is a reference trace evaluator, not a production transaction service. A passing trace does not authenticate an actor, cryptographically verify an external consent receipt, open or hash score/audio bytes, rerun a musical preservation verifier, prove a preview was actually shown, or write/undo a Studio artifact. Those remain responsibilities of the authenticated Studio/MEL-013A transaction path. The event chain proves internal trace consistency only when its producer and source inputs are independently trusted.
