# Continuity current-authorized coverage V1 — evidence scope

Date: 2026-09-11

## Frozen implementation subject

- Repository: `Luminous-Dynamics/symthaea`
- Branch: `architecture/continuity-current-authorized-coverage-v1`
- Direct parent: `architecture/continuity-verifier-adoption-authority-v1@5d9401be4bbc790e935ec5523283751970d48e0c`
- Exact code subject: `4009f1bfddcada4d0b2b412b2f90d50fae5dbc93`

## Historical identity preservation

This child does not alter the canonical identity construction of:

- `QualifiedExternalEffectCoverageV1`; or
- `QualifiedNoExternalEffectsCoverageV1`.

It only exposes previously retained verifier/profile identity fields read-only and composes those historical proofs with the fresh verifier-adoption/currentness theorem from the parent.

## Exact decision theorem

V1 deliberately admits no freshness TTL or implicit grace period. Before physical preparation, the exact historical coverage proof must be wrapped with the exact admitted verifier adoption and a rollback-resistant `CurrentAuthorizedVerifierProfileV1` proving that same adoption/profile remains current.

For both effectful and independently-proven-no-effects execution, V1 requires:

`coverage analysis time == fresh verifier-currentness anchor time == exact A->B commit time == protected post-intent journal-anchor time`.

The coverage verifier profile ID and verifier root epoch must equal both the admitted adoption profile and the freshly current profile. Coverage analysis must be at or after the exact adoption admission time and inside the adoption subject validity interval.

## Physical API sealing

The previous effectful `prepare_coverage_qualified_effect_execution(...)` and no-effects `prepare_proven_no_effects_execution(...)` constructors are crate-private in this child.

The intended public physical constructors are:

- `prepare_current_authorized_effect_execution(...)`; and
- `prepare_current_authorized_no_effects_execution(...)`.

Lower-level pending/ready types remain useful for internal composition/audit but cannot be constructed externally through the crate public API without the current-authorized gate.

## Persistent-evidence limitation

The new current-authorized wrapper is currently non-Serde type-level authority retained through the ready physical token. The existing rollback-resistant effect/no-effects commitments still bind the historical coverage proof, not the new `CurrentAuthorized*CoverageId`, adoption ID, and current-verifier proof ID.

Therefore this child does **not yet** establish that a post-crash auditor can reconstruct proof that the current-verifier gate itself was satisfied before mutation. A direct child must add a same-root protected current-verifier execution commitment tied to the exact journal anchor and existing effect/no-effects commitment.

## Qualification status

`NOT_ESTABLISHED`.

This record freezes source/design intent only. Exact-head formatter/compiler/clippy/test/security/workflow qualification must execute successfully before this branch may be represented as qualified evidence. Parent qualification is independently required.
