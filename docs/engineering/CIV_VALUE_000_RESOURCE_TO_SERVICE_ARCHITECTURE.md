# CIV-VALUE-000A — Terrestrial Resource-to-Service Architecture and Critical-Domain Matrix

Parent: CIV-VALUE-000 #5946  
Companion subjects: CIV-DOMAIN-001 #5947, CIV-MIDSTREAM-001 #5948, CIV-SERVICE-001 #5949  
Implementation issue: #5950

## Purpose

Freeze the first Earth-facing projection that connects primary resource/refining/manufacturing capability to end products and operational services without creating a second PIE, CIV-BOOT, manufacturing, lifecycle, or Mycelix operations engine.

This tranche is documentation/data only. It introduces no production Rust, process recipe, physical control, procurement, resource allocation, economic policy, or public-service authority.

## Ownership

### Symthaea engineering/model layer

- PIE #1604 remains the neutral resource/process/grade/utility/equipment industrial-ecology precedent.
- CIV-BOOT #5774 and #5781/#5782/#5783 own productive/reproductive closure, multi-generation envelope renewal, and frontier/sensitivity planning.
- MFG-PROC #5686 owns manufacturing process/capability/recipe/plan semantics.
- MFG-LIFE #5705 owns lifecycle/circular engineering semantics.
- SEP #5199 and CRITMAT #5200 own selective separation/recovery and critical-material intervention semantics.
- Domain programs own domain physics, design, qualification, and claim ceilings.
- CIV-VALUE only projects these owners into terrestrial resource-to-service dependency views.

### Mycelix operational layer

Mycelix Manufacturing/Supply Chain/Circularity remain the intended owners of real-world:
- lots, batches, units and inventories;
- facilities, resources and supplier offerings;
- routings, work orders and production records;
- custody/provenance and logistics;
- installed assets, maintenance/service events and circular return/recovery events.

CIV-VALUE may consume explicit mapped Mycelix facts later. It is not another ERP/MES, inventory ledger, logistics system, marketplace, maintenance database, or community allocation authority.

## Resource-to-service chain

The reference projection is:

```text
resource occurrence
-> recoverable primary feedstock
-> acquisition / harvest
-> beneficiation / preparation
-> refining / separation / purification
-> qualified material grade
-> intermediate material / component
-> manufactured assembly / asset
-> installed system
-> operational service
-> maintenance / repair / refurbishment / remanufacture
-> recovered component / secondary feedstock
-> manufacturing
```

Every arrow is a proposition that requires evidence from the appropriate owner.

Core non-equivalences:

```text
resource occurrence != recoverable feedstock
primary feedstock != refined commodity
refined constituent != specification-grade material
material grade != functional component
component != assembled asset
asset manufactured != asset installed
asset installed != operational service
service operational once != service continuity
service continuity != productive closure
repairable != locally reproducible
recyclable != recovered != specification-grade secondary feedstock
```

## Four coordinated views

### 1. Material and value transformation

Use PIE, ENG-MAT, MFG-PROC, SEP/CRITMAT and domain-specific process physics to preserve:
- material/state/grade;
- process transformations;
- utilities and equipment;
- coproducts, byproducts, waste and recycle;
- explicit unknowns and unresolved grade/quality.

### 2. Productive capability and renewal

Use CIV-BOOT to preserve distinctions among:
- operationally reachable;
- locally reproducible;
- import-dependent;
- unavailable;
- maintainable/repairable;
- metrology/tooling renewable;
- multi-generation envelope preserved or degraded.

A capability graph result does not prove physical feasibility, adequate quantity, economics, safety or current qualification.

### 3. Operational reality

Use Mycelix for actual lots, inventory, suppliers, facilities, work, shipments, production, installation, maintenance, service and circularity events.

Planning/model evidence cannot manufacture an operational fact.

### 4. Operational service

CIV-SERVICE #5949 projects installed assets into service dependency bundles. A service may require:
- installed assets/infrastructure;
- utilities;
- consumables/feedstocks;
- procedures/workforce/skill refs where modeled;
- maintenance and spares;
- metrology/quality/currentness;
- logistics/storage/cold chain;
- communications/data/software;
- domain-owned safety/regulatory/authority evidence.

```text
product exists
!= service exists
!= service capacity sufficient
!= service continuous
!= service productively closed
```

## Domain classification

The frozen matrix uses three non-ranked tags:

- `human_essential`
- `symthaea_continuity`
- `shared_industrial_multiplier`

A domain can carry more than one tag. These tags identify dependency roles only.

They are not:
- a moral ranking;
- a political or economic priority;
- a recommended investment order;
- a scarcity claim;
- a resilience score;
- a self-sufficiency score;
- resource-allocation authority.

Cross-domain fan-out may be reported by CIV-BOOT as a structural consequence, but high fan-out does not by itself mean a capability should be funded or built.

## Why the midstream is first-class

CIV-MIDSTREAM #5948 prevents raw material availability from laundering into component availability.

Examples:

```text
steel available != qualified bearing available
copper available != transformer available
silicon available != semiconductor die available
rare-earth oxide available != qualified magnet available
polymer resin available != qualified seal / hose / insulation
```

Midstream routes can depend on material grade, separations/refining, heat treatment, precision processing, surface condition, tooling, metrology, calibration, consumables, assembly and qualification simultaneously.

Productive equipment is also a midstream multiplier:

```text
machine frame exists
!= machine commissioned
!= machine productive capability qualified
!= tooling renewable
!= metrology renewable
!= G2+ capability envelope preserved
```

## Human essential-service coverage

The matrix covers resource/productive dependencies for water/sanitation, food/cold chain, energy, shelter/buildings, clothing/hygiene, health-product infrastructure, mobility/logistics, communications/data and waste/recovery.

The matrix does not establish:
- potable-water or food safety;
- clinical efficacy;
- medical-device or medicine regulatory approval;
- building occupancy approval;
- grid reliability;
- transport safety;
- communications availability;
- public-service adequacy.

Those claims remain with qualified domain authorities/evidence.

## Symthaea physical continuity

Symthaea continuity is decomposed across:
- compute/controller hardware;
- memory/storage/networking;
- power conversion/distribution/storage;
- cooling/thermal hardware;
- sensors/metrology/calibration;
- motors/actuators/drives;
- machine tools/tooling/workholding;
- bearings/gears/seals and other precision components;
- electronics/PCB/passives/cables/connectors;
- semiconductor/component supply;
- mechanical structures/enclosures;
- repair/diagnostic equipment;
- optics where required;
- software/firmware build/deployment dependencies through canonical owners.

```text
runtime available
!= repairable
!= replaceable
!= locally reproducible
!= multi-generation productive closure
```

## Frozen matrix

Path:

`docs/release/evidence/civ-value-000-critical-domain-matrix-v1.json`

Schema:

`civ-value-000-critical-domain-matrix-v1`

Domain count:

`27`

Canonical SHA-256:

`67abb09708153f10879d06121213dc1dae0f5de558adae857e670fee51cc5ae6`

The matrix includes stable domain IDs/names, non-ranked classification tags, represented value-chain roles, representative dependency categories, existing-owner references, and explicit evidence-boundary profiles through its compact codebook.

## First implementation/qualification sequence

1. freeze this architecture + matrix;
2. independent stdlib validator for exact digest/schema/domain IDs/tag vocabulary/no-score rules;
3. audit adapters against qualified PIE/CIV-BOOT/MFG owner surfaces;
4. Mycelix bridge maps operational facts without redefining truth;
5. synthetic terrestrial reference chains;
6. only later exact site/resource/facility facts and physical evidence.

No production adapter should silently promote a coverage-matrix entry into real industrial capability.

## Required hostile cases for later validation

A future independent reference corpus/validator should reject or preserve at least:

1. domain tag treated as priority score;
2. resource occurrence relabeled refined material;
3. refined element relabeled qualified component;
4. local PCB assembly relabeled semiconductor closure;
5. machine frame relabeled productive machine;
6. installed asset relabeled operational service without required service dependencies;
7. one successful service episode relabeled continuity;
8. imported spare stock relabeled multi-generation reproduction;
9. recovered constituent relabeled specification-grade secondary feedstock without grade evidence;
10. repeated derived facts counted as independent operational evidence;
11. Mycelix inventory fact treated as physical process qualification without explicit mapping;
12. CIV-BOOT reachability treated as procurement/resource-allocation authority;
13. health-product manufacturing capability treated as clinical/regulatory evidence;
14. local water-treatment equipment treated as potable-water evidence;
15. same material counted twice across reuse and recycling;
16. domain matrix or planner output used to command physical infrastructure.

## Claim ceiling

This tranche freezes a coverage/ownership architecture and deterministic data matrix only.

It establishes no real:
- resource reserve or recoverability;
- manufacturing capacity/yield;
- supply security;
- economic viability;
- labor or skills sufficiency;
- service sufficiency or resilience;
- health/water/food safety;
- regulatory compliance;
- environmental superiority;
- self-sufficiency;
- procurement/resource allocation;
- physical execution authority.
