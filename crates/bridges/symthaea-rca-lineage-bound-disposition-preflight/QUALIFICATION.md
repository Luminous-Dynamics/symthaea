# RCA-003b.3c Qualification Contract

Qualification target: `symthaea-rca-lineage-bound-disposition-preflight`

Status until hosted Actions pass: **IMPLEMENTED, NOT QUALIFIED**.

## Required theorem

```text
raw cross-artifact preflight
        !=
canonical-lineage-bound preflight
        !=
shadow disposition
        !=
canonical belief/workspace state
        !=
action authority
        !=
self-improvement promotion
```

## Required positive checks

Hosted qualification must establish:

1. raw preflight case id/proposition/scope/local-lineage reference exactly match the supplied bound case;
2. supplied evidence-witness slot IDs exactly equal those already bound by raw preflight;
3. runtime case lineage is reconstructed as unique `RootObservation` nodes plus `Transformation` candidate children;
4. the reconstructed validated graph receives #578 canonical evidence-lineage identity;
5. every supplied evidence witness carries exactly that canonical lineage generation;
6. subset/superset/alternate complete graph generations fail closed even when selected local roots could match;
7. changing only the legacy/local graph reference does not define canonical generation;
8. binding identity covers exact raw preflight ID plus exact canonical lineage generation;
9. issued binding has private fields and no `Deserialize` path;
10. no threshold/disposition/downstream-authority logic exists;
11. rustfmt, tests, and strict Clippy pass.

## Required negative checks

Qualification must fail if this crate:

- accepts a supplied witness whose ID differs from raw preflight;
- trusts `structural_case.lineage_graph_id()` as canonical witness authority;
- compares a witness generation to the local graph reference instead of #578 canonical identity;
- permits a witness from a different complete lineage generation;
- changes the V1 root→transformation reconstruction semantics without a new profile;
- gains witness threshold comparison, relation-strength arithmetic, disposition logic, belief/workspace/action, or self-improvement promotion authority;
- makes the issued lineage-bound preflight deserializable.

## Evidence tier

A green focused workflow establishes compilation/tests/lints and freezes this exact generation-binding boundary. It does not qualify a disposition algorithm; no disposition algorithm belongs in this tranche.
