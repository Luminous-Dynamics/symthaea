# Agent Authority Composition Reference Model

**Status:** research prototype / synthetic benchmark oracle. Not production IAM, identity proof, cryptographic verification, sandboxing, or an authorization service.

This is a separate Cargo workspace. It provides early AACB fixtures while Symthaea's existing workstreams retain ownership of production authorization, execution isolation, trust roots, receipts, and evaluator custody.

Run with:

    cargo test --manifest-path research/agent-authority-composition/Cargo.toml

Current synthetic checks:
- sealed intent is an authorization ceiling;
- delegation attenuates resource/action/tenant/expiry scope;
- parent revocation invalidates descendants;
- source revocation invalidates downstream use of derived artifacts;
- sensitive + untrusted content cannot be sent externally without destination-specific synthetic declassification;
- collective-support metadata is non-authoritative;
- unknown remote effects cannot be automatically retried before reconciliation.

Limitations: fixture booleans, grants, labels and declassification values are trusted inputs. This does not authenticate issuers, compute artifact provenance, establish isolation, prove policy correctness, model a real destination, or establish end-to-end exactly-once effects. Tests are examples, not security certification.
