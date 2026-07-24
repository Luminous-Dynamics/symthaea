# Cross-Implementation Evidence Conformance Contract

## Acceptance dimensions

Every verification result reports these dimensions separately:

1. `Decoded`
2. `SchemaKnown`
3. `CanonicalEncodingMatched`
4. `IdentityMatched`
5. `StructureValid`
6. `ExpectedPolicyMatched`
7. `ExternalSignaturesAuthenticated`
8. `LineageValid`
9. `LogicalFreshnessValid`
10. `ConflictFreeForClaim`
11. `AuthorizationSatisfied`
12. `ClaimAccepted`

A failure at an earlier dimension must not be relabeled as a later policy or signature failure.

## Fixture classes

- **Positive:** valid artifacts with frozen canonical bytes and identities.
- **Boundary:** zero, maximum, empty, singleton, and threshold-edge values where allowed.
- **Mutation:** exactly one semantic field altered with outer hashes optionally recomputed.
- **Ambiguity:** duplicate, reordered, unknown, or architecture-dependent encodings.
- **Authority:** valid structure but wrong expected policy, signer, threshold, or purpose.
- **Lineage:** rollback, fork, missing predecessor, repeated checkpoint, cross-segment substitution.
- **Privacy:** fixtures proving private fields are rejected from public disclosure artifacts.

## Release gate

The complete fixture corpus is immutable within a schema version. Corrections require a new corpus version and an explicit migration note; expected outcomes may not be silently edited.
