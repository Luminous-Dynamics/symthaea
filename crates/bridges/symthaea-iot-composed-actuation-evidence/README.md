# Composed Actuation Evidence v0.1

`ComposedActuationEvidence` is a strictly non-authorizing correlation boundary for the corrected two-phase IoT actuation lineage.

It consumes three affine evidence objects by value:

1. `RevalidatedXeniaTransport` — exact portable Xenia receipt/payload re-verified under fixed current cryptography;
2. `PersistedSemanticAcceptance` — crash-durable admission reservation, trusted device appraisal and semantic acceptance; and
3. `VerifiedPostSemanticPhysicalInterlock` — controller/interlock evidence produced only after semantic persistence.

Composition re-derives both durable heads and requires the three branches to converge on the same exact physical envelope, signed transport receipt commitment, transport-trust generation, admission reservation, device appraisal object, semantic head, privileged controller challenge and controller statement/evidence commitments.

The resulting `composition_digest` additionally commits the exact transport-evidence lineage and selected transport key, device-reality trust/policy/key, controller trust/policy/key, semantic persistence time, device-reality expiry and physical-effect deadline.

The type intentionally contains no current trust registry, current fence, wall clock, monotonic clock, serialization guard, final permit, JIT lease, HAL capability, network/process handle or effect-occurrence claim.

A later privileged actuation boundary must still acquire per-device serialization first, re-read the durable admission and semantic heads, obtain all three owner-local current fences while serialization is held, enforce absolute and monotonic deadlines, and expose exactly one consumable HAL attempt.
