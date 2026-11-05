# REA transparency proof harness

This isolated Rust crate implements an offline, bounded RFC 6962 SHA-256 Merkle verifier and a small stateful checkpoint monitor.

## Implemented

- RFC 6962 domain-separated leaf hashing: SHA-256(0x00 || leaf).
- RFC 6962 internal-node hashing: SHA-256(0x01 || left || right).
- Bounded Merkle inclusion proofs.
- Bounded append-only consistency proofs.
- Subject-kind binding that distinguishes raw artifact SHA-256, Code Transparency JWT SHA-256 and signed-statement SHA-256 even when digest bytes happen to be equal.
- Checkpoint rollback and same-size equivocation checks after an external signature-verification step.
- Witness quorum counted by distinct operator identity, not by number of certificates/keys; conflicting roots from one operator are rejected.

## Deliberately not implemented

- Network fetch, log-specific tile formats or receipt decoding.
- Checkpoint signature verification, certificate identity or trust-root selection.
- Publisher authentication or statement predicate truth.
- Durable checkpoint persistence. The caller must persist the accepted checkpoint atomically and reload it as the monitor's initial state.
- Android Binary Transparency, Rekor or SCITT provider adapters.

The verifier consumes proof material already acquired and bound to a specific log identity. A production adapter must bind the leaf preimage, VDS/hash scheme, log ID, tree/shard ID, checkpoint key, tree size and subject kind. Do not treat different providers' leaves as the same canonical statement without an explicit adapter.

## Claim ceiling

A successful inclusion check shows only that the supplied leaf preimage is included under the supplied root at a given index and tree size. A successful consistency check shows an append-only extension between two supplied roots under RFC 6962 assumptions. Before accepting a checkpoint, an external layer must validate its signature and log identity. Witnesses reduce split-view risk only when their operators are genuinely independent. None of these checks establishes that the leaf's signed statement is true, that a package is safe, or that a publisher is benevolent.

## Build and test

    cargo fmt --manifest-path tools/rea-transparency-harness/Cargo.toml -- --check
    cargo test --manifest-path tools/rea-transparency-harness/Cargo.toml --locked
    cargo clippy --manifest-path tools/rea-transparency-harness/Cargo.toml --locked --all-targets -- -D warnings

Tests cover inclusion and consistency for every prefix of generated trees through 32 leaves, mutated proofs, invalid indices, checkpoint rollback/equivocation, subject-kind confusion and witness independence.
