# Cloned sender nonce lifecycle: assurance boundary

## Finding and question

A monotonic in-memory counter is not sufficient to guarantee AEAD nonce uniqueness if two live processes can begin from the same snapshot and use the same effective traffic key. A VM, filesystem, device, or process snapshot can copy the sender identity, key material, epoch, counter, and local reservation state. If both copies transmit, each can independently emit the same nonce under the same key.

RFC 8439 requires each 96-bit nonce to be different for each invocation using the same ChaCha20-Poly1305 key. Its security considerations explain that nonce reuse repeats the Poly1305 one-time key and keystream and can expose the XOR of plaintexts ([RFC 8439 §2.8 and §4](https://www.rfc-editor.org/rfc/rfc8439.html#section-4)). This model does not propose a new AEAD primitive; it makes the lifecycle assumption testable. The newly published [RFC 10033](https://www.rfc-editor.org/rfc/rfc10033.html) separately documents state reuse risks from process forks and VM cloning for stateful hash-based signatures, and recommends cryptographic hardware for rollback-resistant state. That RFC concerns a different primitive family, so it is corroborating lifecycle precedent—not a protocol specification for this mesh.

## Model

- MESH_CLONE_NONCE_LIFECYCLE.tla tracks a bounded sender counter, snapshot cloning, a fencing authority, and an abstract effective-key generation.
- mesh_clone_nonce_legacy.cfg disables fencing. TLC should find a counterexample to NoNonceReuse after the clone receives its snapshot and both copies emit counter zero under the same key-generation identifier.
- mesh_clone_nonce_strict.cfg enables exclusive fencing and fails closed on a repeated tuple. Acquiring a new generation also models a fresh effective traffic key and a reset counter. TLC should complete the strict bounded state space without violating TypeOK or NoNonceReuse.
- The CI workflow pins the TLA+ tools checksum, records exact source/config/tool hashes, checks that the legacy run violates the named invariant, and retains raw logs as an artifact.

## Required implementation architecture

This reference model is valid only if the real system enforces every abstraction it relies on:

1. **Unique authority:** generation numbers must come from durable, atomic, non-rollback state (or a genuinely non-clonable hardware source). An ordinary in-memory counter or copied local file is not that authority.
2. **Enforced fencing:** stale processes must be unable to emit after a newer lease is granted. A cooperative boolean or an unauthenticated self-asserted generation field is not enough.
3. **Key-domain separation:** the new generation must derive a genuinely distinct traffic key through a reviewed KDF and authenticated context, or the design must reserve disjoint nonce ranges under one key using a durable allocator. Merely changing a local epoch variable while keeping the same key is unsafe.
4. **Fail-closed recovery:** if the authority, durable state, generation uniqueness, or key derivation is unavailable or ambiguous, do not encrypt/send. Re-enrol with a new cryptographic identity or recover authoritative state.
5. **Bounded counters:** reaching the limit must stop transmission or force a new authenticated key generation before any nonce repeats. Counter wrap is never silent.

For disconnected radios, a central online fencing service may be impractical. The design must then use an actually enforceable alternative, such as a provisioned per-device key that cannot be copied with application state, a hardware monotonic counter, or a protocol whose key and nonce namespaces are independently and durably allocated. A UUID chosen at process restart is not a proof of uniqueness.

## Existing authority-kernel relationship

The exclusive-owner/fencing part should be treated as a concrete mesh refinement of the repository's existing SYM-FV-041D / ExclusiveLease conservation work (#6127), not as a second general-purpose lease theorem. The mesh-specific obligation is stronger than abstract exclusivity alone: the runtime must prove that the currently fenced owner is the only process able to reach the AEAD send path, and that a new generation actually changes the effective traffic-key namespace (or reserves a disjoint counter range). That refinement is not established by this model.

## What this proves—and does not

The intended claim is narrow: **the bounded state model can produce a clone/reuse counterexample for the legacy policy and preserve its modeled nonce-uniqueness invariant for the fenced reference policy**.

It does not prove that the repository's runtime implements the reference architecture, that a KDF yields independent keys, that the fencing service is available or uncompromised, that hardware state is non-clonable, or that all protocol nonce fields are correctly encoded. It is not TLC evidence until the pinned workflow has actually completed, and even a passing bounded model is not a cryptographic proof or an implementation audit.

This is designed as a falsifiable companion to the proposed clone-safety scenario family in the mesh security benchmark work. Keep all scenario/evidence status at proposed until implementation-level tests and independently reviewed receipts exist.
