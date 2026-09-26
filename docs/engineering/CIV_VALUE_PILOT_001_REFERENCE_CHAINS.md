# CIV-VALUE-PILOT-001 — Synthetic Terrestrial Resource-to-Service Reference Chains

Parent: CIV-VALUE-000 #5946  
Matrix: CIV-VALUE-000A #5950 / draft PR #5951  
Companion subjects: CIV-MIDSTREAM-001 #5948, CIV-SERVICE-001 #5949  
Issue: #5954

## Purpose

Freeze seven contrasting synthetic terrestrial chains that exercise the complete resource-to-service projection without introducing real industrial performance numbers, process recipes, operating procedures, site facts, or physical authority.

The benchmark asks whether evidence and ownership remain separated across:

```text
primary/input feedstock
-> refining / material grade
-> intermediate / component
-> manufactured asset
-> installed system
-> operational service
-> maintenance / replacement
-> circular return
```

The fixtures are intentionally synthetic. They test composition semantics, not real process feasibility.

## Ownership discipline

The fixture stage labels are benchmark roles only. They do not replace canonical owners:

- PIE #1604 owns neutral resource/process/grade/utility/equipment semantics.
- MFG-PROC #5686 owns manufacturing process/capability/recipe/plan semantics.
- CIV-MIDSTREAM #5948 owns the terrestrial projection across refining/intermediate/component/productive-equipment closure.
- CIV-BOOT #5774/#5782 owns structural and multi-generation productive capability.
- MFG-LIFE #5705 owns lifecycle/circular engineering projections.
- SEP #5199 and CRITMAT #5200 own recovery/separation/critical-material intervention semantics.
- CIV-SERVICE #5949 owns the installed-asset-to-service dependency projection.
- Mycelix Manufacturing/CIRC-MFG #3095 owns real lots, inventory, facilities, work orders, logistics, maintenance and circularity facts.
- Domain programs own their physics, qualification and domain-specific claim ceilings.

No fixture may manufacture a fact owned by another plane.

## Frozen chains

### T1 — water-service infrastructure

Preserves the distinction between treatment/service infrastructure and potable-water safety or capacity.

### T2 — electric-power distribution/service

Preserves the distinction between conductor/generation availability and cables, transformers, switchgear, control/storage components and continuous electrical service.

### T3 — productive machine + repair service

Preserves the distinction between machine structure, precision/mechatronic/tooling closure, one successful repair and multi-generation productive closure.

### T4 — compute/network service

Preserves the distinction between board/system assembly, semiconductor/package closure, installed compute/network assets and resilient communications/data service.

### T5 — food/cold-chain infrastructure

Preserves the distinction between production quantity, food safety/nutrition, storage/cold-chain integrity, logistics and service continuity.

### T6 — health diagnostic/manufacturing infrastructure

Preserves the distinction between material/reagent/component/device manufacturing and clinical efficacy, diagnostic validity, sterility, regulatory approval or health-service sufficiency.

### T7 — circular critical-material route

Preserves the distinction between recyclable/end-of-life articles, recovered constituents, purified/refined material, specification-grade secondary feedstock and qualified component/material reuse.

## Frozen data artifact

Path:

`docs/release/evidence/civ-value-pilot-001-reference-chains-v1.json`

Schema:

`civ-value-pilot-001-reference-chains-v1`

Chain count:

`7`

Stage-role count per chain:

`8`

Adversarial-case count:

`10`

Canonical SHA-256:

`3bf4501f80c73274f50fa64c03f1a6dcc3bb5a81f022b0bd4cc5d791d477f815`

## Shared invariants

```text
feedstock present
!= refining route present

refined material present
!= required precision/component route present

asset present
!= required service dependencies present

service observed once
!= continuity established

stocked imported spare
!= multi-generation renewal

recovered constituent
!= qualified secondary feedstock

Mycelix operational fact
!= engineering/process qualification

engineering/closure model result
!= Mycelix operational fact

benchmark result
!= procurement / allocation / execution authority
```

## Synthetic adversarial cases

The frozen corpus contains ten negative cases:

1. feedstock without refining -> downstream component unavailable;
2. refined material without a required precision component -> asset route blocked;
3. asset without utility/consumable dependency -> service unavailable;
4. one service episode without maintenance/spares -> continuity unresolved;
5. G1 stocked imports without G2 renewal -> multi-generation closure absent;
6. recovered constituent without grade evidence -> qualified re-entry blocked;
7. local alternative sufficient for one bounded profile but not another -> profile-relative sufficiency only;
8. Mycelix inventory/work-order fact without engineering evidence -> engineering qualification unestablished;
9. model/closure result without operational fact -> operational fact unestablished;
10. any benchmark/model result -> zero procurement, allocation or physical execution authority.

## No hidden service promotion

Each service fixture carries explicit service-dependency domain IDs. The existence of an installed asset does not implicitly satisfy those dependencies.

A future adapter may resolve actual service dependencies only from qualified/canonical owners. Missing/unknown dependencies remain missing/unknown.

## No hidden circularity promotion

Circular return is a stage role, not proof of closed-loop reuse.

A returned article or recovered constituent may require sorting, preprocessing, purification/refining, grade/currentness evidence and component/material qualification before it can re-enter a manufacturing route.

## First qualification sequence

1. freeze these exact seven chains and ten adversarial cases;
2. add an independent stdlib validator for digest/schema/chain IDs/stage ordering/owner syntax/adversarial outcomes;
3. only after the domain-matrix and chain validators execute successfully, audit thin adapters into current qualified PIE/CIV-BOOT/MFG owners;
4. map Mycelix operational facts through an explicit bridge without duplicating operational state;
5. later add site/profile-specific data only when evidence exists.

## Claim ceiling

This benchmark establishes no real:
- reserves or recoverability;
- plant/process capability or yield;
- material grade/purity;
- component qualification;
- installed-system performance;
- water/food/health safety;
- grid/data/logistics/service reliability;
- industrial scale/economics;
- workforce sufficiency;
- environmental superiority;
- resilience/self-sufficiency;
- procurement/resource allocation;
- physical execution authority.
