# CIV-DOMAIN-GAP-001A — Canonical Owner-Coverage Audit for Critical Productive Domains

Parent issue: CIV-DOMAIN-GAP-001 #5960  
Parent matrix implementation: CIV-VALUE-000A PR #5951  
Exact parent head: `acc9db77d7522f7213781020f9a0880815234699`  
Implementation issue: #5971

## Purpose

Freeze one deterministic, reviewable ownership audit across the 27 domain IDs already defined by CIV-VALUE-000A.

This tranche does **not** create another industrial ontology, process engine, component catalog, observation system, closure engine, service model, operations database, or execution layer.

Its only question is:

> For each already-frozen domain, which canonical owners cover each layer, what domain-specific theorem remains missing, and is a dedicated root actually justified?

The parent domain IDs and critical-to tags remain authoritative in:

`docs/release/evidence/civ-value-000-critical-domain-matrix-v1.json`

This child does not rename, reorder, or extend those 27 domain identities.

## Exact parent binding

- parent schema: `civ-value-000-critical-domain-matrix-v1`
- parent canonical SHA-256: `67abb09708153f10879d06121213dc1dae0f5de558adae857e670fee51cc5ae6`
- parent PR: #5951
- parent exact head: `acc9db77d7522f7213781020f9a0880815234699`

The owner audit is invalid if evaluated against another silently changed parent matrix.

## Ownership layers

Every domain row records independently:

1. upstream/resource owner refs;
2. materials/refining owner refs;
3. manufacturing/process owner refs;
4. component/product owner refs;
5. service owner refs;
6. physical observation/metrology owner refs;
7. lifecycle/circularity owner refs;
8. Mycelix operational-projection refs;
9. owner-coverage disposition;
10. missing domain-specific theorem(s);
11. dedicated root refs where justified;
12. shared-owner profile refs;
13. conservative evidence-maturity state;
14. claim ceiling.

These fields are **references to canonical owners**, not copied implementations.

## Coverage dispositions

The matrix uses only bounded descriptive dispositions:

- `ExistingDedicatedOwner`
- `ExistingSharedOwnersSufficient`
- `SharedOwnersNeedDomainProfile`
- `DedicatedDomainRootOpened`
- `DedicatedDomainLikelyMissing`
- `AuditUnresolved`

They are not ranks or maturity scores.

Current frozen counts:

- `ExistingDedicatedOwner`: 3
- `ExistingSharedOwnersSufficient`: 3
- `SharedOwnersNeedDomainProfile`: 10
- `DedicatedDomainRootOpened`: 7
- `DedicatedDomainLikelyMissing`: 4
- `AuditUnresolved`: 0

## New dedicated roots bound by this audit

The current domain-gap tranche has opened exactly these bounded specializations:

- PROD-EQP-000 #5964 — machine tools, tooling, workholding and productive-equipment reproduction;
- IND-COMP-000 #5965 — precision mechanical and fluid-component functional qualification;
- WATER-MFG-000 #5966 — water/sanitation/wastewater infrastructure engineering;
- AGRI-MFG-000 #5967 — agriculture/food/cold-chain infrastructure engineering;
- GRID-MFG-000 #5968 — power-distribution equipment manufacturing;
- HEALTH-MFG-000 #5969 — health-product manufacturing infrastructure;
- CHEM-MFG-000 #5970 — industrial chemical/process-gas/fertilizer-intermediate manufacturing architecture.

These roots do not inherit ownership of generic quantities, observations, materials, processes, lifecycle, closure, service, Mycelix operations, or physical authority.

## Strong anti-duplication theorem

The audit explicitly rejects:

`domain row exists -> create monolithic domain stack`

and also rejects:

`generic process/component support exists -> domain functional qualification already exists`

The intended architecture is:

shared canonical owners  
+ exact domain-specific transformation/function/qualification semantics  
+ explicit service/operational projection  
= bounded domain coverage

A new root is justified only for the missing domain theorem.

## Important findings

### Productive equipment and industrial components

PROD-EQP #5964 and IND-COMP #5965 close two high-fan-out semantic gaps that generic MFG-PROC and ENG-DEVICE do not close by themselves:

`machine frame / component geometry`
`!=`
`qualified productive or functional capability`

The resulting evidence still remains exact-profile relative.

### Human-essential infrastructure

WATER-MFG #5966, AGRI-MFG #5967 and HEALTH-MFG #5969 now provide domain engineering/manufacturing boundaries without inheriting safety, clinical, public-service or regulatory authority.

### Electrical distribution

GRID-MFG #5968 fills the conductor/material -> distribution-equipment qualification seam, while ENG-MAG/ENG-DEVICE/power owners retain their generic physics/subsystem ownership.

### Industrial chemistry

CHEM-MFG #5970 provides grade/process-chain/plant-dependency semantics while explicitly excluding hazardous recipes and physical plant control.

## Domains that still appear to need a profile rather than a new root

The frozen audit currently classifies ten rows as `SharedOwnersNeedDomainProfile`.

Examples include:

- integrated energy systems beyond grid equipment;
- construction/shelter system composition;
- mining/beneficiation;
- metallurgy/foundry/heat-treatment qualification;
- polymers/elastomers;
- glass/ceramics outside cement;
- electromechanical power beyond distribution equipment;
- building services;
- communications/data infrastructure;
- emergency/resilience composition.

This means the next step is **not** automatically to open ten roots. First try a narrow profile over existing owners.

## Likely missing dedicated domains

Four rows remain `DedicatedDomainLikelyMissing` in this source audit:

- wood/pulp/paper/packaging;
- textiles/hygiene;
- electrochemical storage manufacturing beyond materials discovery;
- transport/vehicle/logistics hardware qualification.

This is still an architectural hypothesis, not permission to create those programs immediately.

A future root requires the #5960 launch rule: explicit service/function need, proof shared owners are insufficient, domain-specific theorem, composition plan, bounded first campaign, and explicit safety/authority boundary.

## Evidence maturity is deliberately unresolved

Every row uses:

`evidence_maturity = AuditUnresolved`

with the explicit rule:

`issue / roadmap / source exists != physical or qualified evidence`

This avoids a common architecture-audit mistake where the existence of a rich issue graph is silently converted into real capability maturity.

A later maturity audit must bind exact source/qualification/bench/physical evidence from the owning subjects.

## Frozen data artifact

Path:

`docs/release/evidence/civ-domain-gap-001a-owner-coverage-matrix-v1.json`

Schema:

`civ-domain-gap-001a-owner-coverage-matrix-v1`

Domain count:

`27`

Canonical UTF-8 compact sorted-key JSON + final newline SHA-256:

`45056ab59a16137b48325338a552b109b40eaef263936af96fb52eec1fab892a`

## Hostile cases for the independent validator

A later stdlib-only validator should reject or flag at minimum:

1. domain ID not present in the exact parent matrix;
2. missing or changed parent SHA;
3. critical-to tags not equal to parent tags;
4. duplicate domain row;
5. missing ownership layer;
6. `ExistingDedicatedOwner` without an actual dedicated root ref;
7. `DedicatedDomainRootOpened` with no dedicated root ref;
8. new root claiming canonical ownership already assigned to MFG-PROC/FIELD/CIV-BOOT/Mycelix/etc.;
9. generic MFG-PROC alone presented as domain functional qualification;
10. pilot issue/roadmap presence presented as physical evidence maturity;
11. owner refs reordered if the schema declares canonical ordering;
12. unknown coverage disposition;
13. universal priority/self-sufficiency/readiness score added;
14. owner-coverage result used to authorize procurement, resource allocation or physical execution.

## Next composition target

After this source/data freeze, CIV-VALUE-PILOT-002 #5963 should consume the owner matrix by reference.

For each of its seven synthetic terrestrial chains, every stage should identify:

- exact domain owner;
- exact upstream/midstream/component/service owner;
- utility dependency owner;
- unresolved owner gap, if any.

This lets the benchmark distinguish:

`chain blocked by missing physical capability`
from
`chain blocked because no domain owner has yet defined the required theorem`.

Those are different failures and should remain different.

## Claim ceiling

This tranche establishes only a deterministic architecture/ownership audit.

It establishes no real:

- reserve or recoverability;
- material grade or process yield;
- industrial capacity;
- equipment/component performance;
- service capacity or continuity;
- economic viability;
- workforce sufficiency;
- safety or regulatory compliance;
- potable-water, food, clinical, grid, transport or building authority;
- resilience or self-sufficiency;
- societal priority;
- funding/procurement/resource-allocation recommendation;
- physical execution authority.
