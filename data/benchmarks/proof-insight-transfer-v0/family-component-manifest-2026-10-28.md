# Complete PITB component manifest

Status: prerequisite implementation and synthetic controls only. This does not claim the actual relationship inventory is complete, that the primary scorer consumes component IDs, or that PITB is statistically qualified.

The existing leakage report emits cross-split hard-link components for quarantine. It is not a complete clustering key because same-split groups and isolated families are absent. The builder maps every frozen family ID to a deterministic component ID. Confirmed and unresolved hard relations join components; dismissed hard relations and soft relations such as conceptual similarity do not. Every isolated family becomes a singleton. Component IDs are SHA-256 content addresses of sorted family IDs; the manifest binds the frozen split digest and exact relationship-inventory bytes.

## Fail-closed qualification

The output can be labelled `structurally_eligible_receipt_unverified` when the inventory declares complete coverage, supplies a receipt digest, and has no non-dismissed hard component crossing train/validation/test. That state is **not qualification**: the current builder only receives the digest, not the receipt bytes or an authenticated reviewer attestation. Accordingly, `qualified_for_confirmatory_resampling` remains false in every output from this version. An incomplete inventory or a cross-split hard component also remains unqualified.

Unresolved hard links are conservatively included. A digest binds bytes; it does not authenticate reviewer identities or establish that the coverage assertion is true. A later verifier must inspect the exact receipt bytes, reviewer evidence, and coverage scope before a new qualification decision can be issued.

The builder does not yet change the production scorer. The run config, validation power planner, and test bootstrap must bind and consume the same component-manifest digest and component-size vector. Only then can the minimum of 20 independent components be enforced; until receipt verification and scorer integration are complete, no confirmatory PITB result is qualified.
