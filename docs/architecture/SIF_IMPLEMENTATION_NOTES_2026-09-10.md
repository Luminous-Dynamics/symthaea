# Solar Infrastructure Fabric — implementation notes

## Current tranche

Draft PR #1402 establishes SIF-001 only. The ontology is intentionally placed in `symthaea-engineering` as a thin composition vocabulary and is not yet re-exported by the crate root until the existing crate surface is reviewed and wired with compile-verified changes.

## Required next verification

Before promoting PR #1402 from draft:

1. wire `infrastructure` into `symthaea-engineering/src/lib.rs`;
2. run `cargo test -p symthaea-engineering --lib`;
3. run the workspace dependency/cycle checks;
4. verify serde is already an enabled dependency for the engineering crate;
5. confirm no canonical `AssetId`, resource, reservation, or authorization types already exist under another crate and should be reused instead;
6. add property tests for finite-value/resource-window invariants if the crate's current testing conventions use proptest;
7. verify the new vocabulary does not grant or imply runtime hardware authority.

SIF-002 and SIF-003 are tracked separately in issue #1403 so orbital-physics and composition changes do not get mixed into the ontology evidence line.
