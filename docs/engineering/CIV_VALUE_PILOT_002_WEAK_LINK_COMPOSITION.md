# CIV-VALUE-PILOT-002 — Upstream, Utility, Owner, and Service Weak-Link Composition

Parent benchmark: CIV-VALUE-PILOT-001 #5954 / draft PR #5955  
Owner audit: CIV-DOMAIN-GAP-001A #5971 / draft PR #5975  
Architecture parents: CIV-UPSTREAM #5958, CIV-MIDSTREAM #5948, CIV-UTILITY #5959, CIV-SERVICE #5949  
Issue: #5963

## Purpose

Extend the seven PILOT-001 terrestrial reference chains so they no longer begin from magically available feedstock and no longer treat utilities or architectural ownership as invisible assumptions.

This tranche is still synthetic documentation/data only. It adds no real reserves, process parameters, facility facts, resource allocation, procurement or physical execution.

The stronger chain is:

`resource occurrence`
→ `recoverability / accessibility`
→ `acquisition / harvest + custody`
→ `beneficiation / preparation`
→ `refining / specification grade`
→ `intermediate / component`
→ `productive / manufacturing asset`
→ `installed / commissioned system`
→ `operational service bundle`
→ `maintenance / spares / calibration / tooling renewal`
→ `circular recovery`
→ `secondary-feedstock requalification`

Utilities cut across those stages. They are not a fake thirteenth material stage.

## Exact parent bindings

### PILOT-001

- PR: #5955
- exact head: `d85bab23ff192d83337d4576cf4fb8f718c303a6`
- schema: `civ-value-pilot-001-reference-chains-v1`
- canonical SHA-256: `3bf4501f80c73274f50fa64c03f1a6dcc3bb5a81f022b0bd4cc5d791d477f815`

### Owner matrix

- PR: #5975
- exact repaired head: `7fc655b12eb6decf1fb77a4055fa7b20741bba79`
- schema: `civ-domain-gap-001a-owner-coverage-matrix-v1`
- canonical SHA-256: `45056ab59a16137b48325338a552b109b40eaef263936af96fb52eec1fab892a`

The owner matrix is referenced as an exact sibling subject; this PR does not copy its 27-domain ontology.

## New distinction: capability gap versus owner gap

PILOT-002 explicitly separates:

`owner-defined capability exists but evidence/capability is unresolved`

from

`the domain-specific theorem/owner is itself not yet adequately defined`.

Those are different failures.

A physical/import blocker must not create a new ontology, and a missing architecture owner must not be reported as proof that the corresponding real-world capability is unavailable.

## Frozen weak-link dispositions

The corpus retains bounded descriptive states rather than one readiness score:

- `OwnerDefinedCapabilityUnresolved`
- `DomainOwnerGap`
- `FeedstockUnavailable`
- `UpstreamEvidenceUnresolved`
- `GradeOrPurityUnresolved`
- `MidstreamCapabilityUnavailable`
- `ComponentDependencyUnavailable`
- `UtilityEnvelopeUnresolved`
- `UtilityCapacityOrTimingInsufficient`
- `InstalledAssetUnqualified`
- `ServiceReachableButContinuityUnresolved`
- `MaintenanceOrRenewalUnresolved`
- `CircularReentryUnqualified`
- `EvidenceInsufficientOrStale`
- `ProfileBoundedAlternativeOnly`
- `FullyRepresentedSyntheticRouteUnderProfile`
- `NoProcurementAllocationOrExecutionAuthority`

No disposition is a priority, maturity, investment, resilience or self-sufficiency score.

## Cross-cutting utility profiles

The synthetic corpus includes references for:

- electric power;
- thermal service;
- water service;
- process gas / process chemical supply;
- vacuum / pressure service;
- refrigeration;
- communications/data;
- metrology/calibration;
- waste treatment.

Every utility profile points to CIV-UTILITY #5959 plus the relevant owner-matrix domain IDs. No numeric utility envelope is invented here.

## Seven extended chains

### T1 — water-service infrastructure

Adds source occurrence/recoverability, intake/custody, pretreatment, commissioning, maintenance/media/calibration renewal, residual recovery and secondary-input requalification.

No potable-water safety, pathogen-removal, public-service-capacity or continuity claim.

### T2 — electric-power distribution/service

Adds upstream mineral/material stages, commissioning, renewal and circular re-entry while preserving electrochemical-storage manufacturing as an explicit owner-gap candidate.

`electrolyte/material research != cell/pack manufacturing owner`.

No equipment rating, grid reliability, grid-code, protection or energization claim.

### T3 — productive machine + repair service

Adds recoverability/acquisition, material preparation, explicit productive-machine commissioning, metrology/tooling renewal and secondary-feedstock return.

`machine frame exists != productive machine capability != G2+ renewal`.

### T4 — compute/network service

Adds upstream material/feedstock stages and separates commissioned compute/network assets from operational communications/data service and repair/cooling renewal.

`working server != semiconductor closure != resilient service`.

### T5 — food/cold-chain infrastructure

Adds agricultural input accessibility, water dependencies, packaging grades/components, logistics commissioning, refrigeration/packaging renewal and recovery.

The existing owner audit makes wood/pulp/paper/packaging and transport/logistics explicit owner-gap candidates rather than hidden assumptions.

No food-safety, nutrition, cold-chain-integrity or supply-sufficiency claim.

### T6 — health diagnostic/manufacturing infrastructure

Adds upstream feedstock stages, commissioning, cold-chain/logistics/calibration renewal and disposal/requalification while preserving transport/logistics as an owner-gap candidate.

No clinical, diagnostic, sterility, regulatory or patient-safety authority.

### T7 — circular critical-material route

Expands end-of-life material into recoverability, custody, preprocessing, refining/grade, manufacturing, service context, repeated recovery and next-generation secondary-feedstock requalification.

`recovered constituent != specification-grade secondary feedstock`.

## Owner-gap candidates are not shortage claims

The corpus currently exposes these owner-gap candidates because #5975 classifies them `DedicatedDomainLikelyMissing`:

- electrochemical storage manufacturing in T2;
- wood/pulp/paper/packaging in T5;
- transport/logistics hardware in T5 and T6.

This means only that the architecture lacks a sufficiently specific dedicated domain theorem under the current audit.

It does **not** mean batteries, packaging or transportation are physically unavailable in any real location.

## Shared utility common mode

PILOT-002 adds an adversarial case where otherwise distinct chains share one utility failure root.

Examples may include power, metrology, refrigeration, communications or waste treatment.

The result exposes common dependency structure only:

`shared high-fan-out dependency != automatic priority or investment recommendation`.

## Frozen adversarial corpus

The data artifact contains 17 deterministic negative cases covering:

1. resource occurrence without representative assay/evidence;
2. acquired feedstock without refining;
3. refined commodity without specification grade;
4. complete materials/components but unresolved process utility;
5. sufficient average energy but inadequate instantaneous capacity/timing;
6. nominal utility redundancy with one common failure root;
7. asset without commissioning/calibration;
8. one service episode without renewal;
9. G0/G1 imported consumable stock without local renewal;
10. recovered constituent without secondary-feedstock qualification;
11. bounded local substitute valid for one profile only;
12. source/lot change with stale downstream evidence;
13. Mycelix operational fact without engineering qualification;
14. Symthaea model/closure result without operational fact;
15. shared utility cross-chain common mode;
16. missing domain theorem distinguished from physical shortage;
17. zero procurement/allocation/execution authority from every disposition.

## Frozen data artifact

Path:

`docs/release/evidence/civ-value-pilot-002-weak-link-chains-v1.json`

Schema:

`civ-value-pilot-002-weak-link-chains-v1`

- chain count: 7
- stage-role count per chain: 12
- utility profiles: 9
- adversarial cases: 17
- canonical UTF-8 compact sorted-key JSON + final newline SHA-256:
  `b263c517c244ab11d38312c7ac28f11eea7151564d6527e551e60d3fa9745a47`
- Git blob:
  `b431242c2bfad4dfd4f3949e3cc429c4db819cd0`

The stage representation is compact: each `stages[i]` entry binds a synthetic subject suffix plus domain codes, while `i` is interpreted through the frozen `stage_roles` vector. This avoids copying the owner ontology while preserving exact stage order.

## Independent validation target

A later stdlib-only validator should hard-bind both parent subjects and reject at least:

- changed parent digest/head;
- missing/reordered/duplicated chain IDs;
- any chain not containing exactly 12 stages in the frozen role order;
- unknown domain or utility code;
- owner-gap candidate not classified `DedicatedDomainLikelyMissing` by the exact owner matrix;
- unknown weak-link disposition;
- missing cross-chain utility common-mode adversarial case;
- any scalar priority/readiness/resilience/self-sufficiency score;
- any authority field stronger than analysis-only.

## Claim ceiling

This source/data freeze establishes only a synthetic composition benchmark.

It establishes no real:

- resource reserve or recoverability;
- assay, grade, purity or process yield;
- component, machine or utility capability;
- service safety, quality, capacity or continuity;
- food/water/health/grid/communications regulatory or public-service sufficiency;
- economics, workforce, sustainability, resilience or self-sufficiency;
- societal priority, procurement or resource allocation;
- facility operation or physical execution authority.
