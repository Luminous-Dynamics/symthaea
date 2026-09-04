# Exact Xenia transport evidence

This crate preserves the exact canonical Xenia receipt and physical-effect payload bytes represented by an already opaque `VerifiedTransportEnvelope`.

It is deliberately an **ephemeral provenance capsule**, not current transport trust and not physical authority. Binding performs no signature verification and accepts no trust registry, verifier, clock, final permit, JIT lease, or HAL handle.

The capsule is non-clone and non-serializable. A process restart destroys the continuation evidence; a previously burned command cannot resume toward actuation from durable semantic state alone.

The next privileged stage must re-run the retained receipt through a fixed Ed25519 + ML-DSA-65 verifier under independently anchored **current** Xenia transport trust and must reject any transport-trust generation mismatch with the physical-effect lineage.
