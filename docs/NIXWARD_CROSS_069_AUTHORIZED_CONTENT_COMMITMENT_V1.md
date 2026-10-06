# Nixward CROSS-069 — Authorized Definition Content Commitment V1

## Purpose

CROSS-069 closes the provenance gap between an observer-produced systemd definition-content digest and the authority identity that authorizes a service effect.

A content digest is not useful merely because it is cryptographically committed. The authorization chain must make it part of the exact effect identity.

## Authority binding

`NixServiceEffectContextV1` now contains `authorized_definition_content_digest` alongside:

- typed operation;
- canonical unit;
- authorized NixOS generation;
- exact pre-state digest;
- source-identity definition digest;
- optional pre-invocation identity;
- required stability contract.

The context digest commits the content commitment, so changing definition bytes changes the authorized service-effect identity.

## Provenance-aware admission

`NixServiceEffectAdmissionV1` is the orchestration waist between the observer and authorization.

It consumes `NixVerifiedSystemdDefinitionContentCommitmentV1`, not the raw serializable commitment.

Therefore the normal admission path is structurally:

`systemd observer -> verified content commitment -> service-effect admission -> typed intent -> authorization`

The authorization module does not import systemd observer types. This preserves the architectural separation between:

- observation;
- provenance-aware orchestration;
- authorization;
- execution transport.

## Post-state propagation

The content digest is carried forward into:

`effect context -> post-state expectation -> durable receipt`

The receipt requires the authorized and observed content commitment to match.

Changing the commitment therefore cannot preserve the same effect digest or a supposedly equivalent post-state receipt.

## Important distinction

The content digest proves what the bounded content-observer read and hashed. It does not by itself prove that systemd parsed exactly those bytes at every earlier moment.

The current bounded statement is:

`At the observation point, the exact source files reported by systemd were read twice through read-only descriptors, remained metadata-stable, and produced the committed content digest.`

The stronger statement:

`This byte sequence was definitely the exact byte sequence parsed by systemd during a particular historical load transaction.`

is outside the current claim ceiling and would require tighter kernel/systemd transactional evidence.

## Fail-closed conditions

The chain rejects:

- content digest with invalid shape;
- content commitment absent from a contextual service intent;
- content commitment changed after intent construction;
- expectation whose content commitment differs from the intent context;
- receipt whose observed content commitment differs from the authorized commitment;
- provenance-aware admission using raw/unverified content data.

## Relationship to CROSS-068

CROSS-068 establishes the observer-side byte commitment.
CROSS-069 establishes the authority-side identity binding.

Keeping these separate prevents the common mistake of treating a caller-provided digest as if it were an observation receipt.

## Qualification

Exact-head GitHub Actions remain authoritative. Mergeability, queued status, or local source inspection are not PASS evidence.