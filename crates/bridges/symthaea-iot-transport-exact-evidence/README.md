# Exact Xenia transport evidence

This crate preserves the exact canonical Xenia receipt and physical-effect payload bytes represented by an already opaque `VerifiedTransportEnvelope`.

Binding is an **affine transition**: it consumes the original `VerifiedTransportEnvelope` by value while constructing the non-clone/non-serializable `ExactXeniaTransportEvidence`. Normal callers therefore cannot reuse the same in-process verified transport proof to mint multiple continuation capsules.

It is deliberately an **ephemeral provenance capsule**, not current transport trust and not physical authority. Binding performs no signature verification and accepts no trust registry, verifier, clock, final permit, JIT lease, or HAL handle.

A process restart destroys the continuation evidence; a previously burned command cannot resume toward actuation from durable semantic state alone.

The next privileged stage must consume this exact evidence, re-run the retained receipt through the fixed Ed25519 + ML-DSA-65 verifier under independently anchored **current** Xenia transport trust, and reject any transport-trust generation mismatch with the physical-effect lineage.
