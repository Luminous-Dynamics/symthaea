# Symthaea Nutrition Source Registry

This crate is a small, deterministic gate between discovered food-composition records and downstream nutrition reasoning. It records declared source provenance and rights, checks source-review freshness, matches a fact to its declared source, and distinguishes assessability from legal permission and data quality.

It is not a food database, a nutrient calculator, a license attorney, a publisher authenticator, a laboratory verifier, a food-safety certificate, a crop planner, or clinical nutrition advice. Numeric nutrient aggregation belongs to Symtropy. Community and traditional-knowledge authorization belongs to Mycelix and the relevant rights holders.

## Core behavior

- Rights are separate explicit decisions for assessment use, caching, transformation, redistribution, commercial use and public fixtures. Unknown, denied and unresolved written-permission requirements fail closed.
- A granted-by-written-permission decision must reference the permission record.
- Source manifests are versioned and dated; expired, future-dated, revoked, withdrawn or review-required sources block use.
- Retained artifacts require a canonical lower-case SHA-256 plus the exact artifact bytes. A digest-shaped string by itself is not proof: verification recomputes SHA-256 over the supplied bytes.
- If a declared digest is supplied without its bytes, its integrity is Unverified and numeric use is blocked. Without a declared digest and with no retained artifact, the result explicitly says NoDigestClaimed, not Verified.
- Fact records bind source ID, source record/citation, food identity/state, nutrient definition, unit, integer scale and basis mass. A missing nutrient value is NotAssessable rather than zero.
- Approximate/geographic fallback rows are visibly QualifiedFallback. Unresolved food matches and unknown geographic matches block.
- Synthetic rows return SyntheticFixtureOnly and cannot be misrepresented as empirical source data.

No SAFOODS values or other real food-composition rows are included in this crate's fixtures; all fixtures use synthetic example bytes and identifiers. The actual source licensing and data quality still require human review. This validator checks declarations and supplied bytes, not whether those declarations are truthful or legally sufficient.

## Validation

Run focused checks from the repository root:

- cargo fmt --check -p symthaea-nutrition-source-registry
- cargo test -p symthaea-nutrition-source-registry
- cargo clippy -p symthaea-nutrition-source-registry --all-targets -- -D warnings

The source registry contract is tracked in [Symthaea issue #7481](https://github.com/Luminous-Dynamics/symthaea/issues/7481), with broader nutrition reasoning in [PR #7476](https://github.com/Luminous-Dynamics/symthaea/pull/7476). Exact-head checks must pass before the implementation is treated as validated.
