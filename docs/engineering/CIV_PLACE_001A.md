# CIV-PLACE-001A — home-to-city place composition contract

**Status:** source/fixture tranche only. No production authority.

## Scope

CIV-PLACE-001A defines a narrow composition boundary above existing Symthaea engineering owners. It represents **which exact systems make up a place, how they connect, which dependencies are shared, what services cross the place boundary, and which changes reopen downstream evidence**.

It deliberately does not become a second building, utility, service, evidence, twin, governance, or actuator ontology.

### Existing owners

- BUILT-ENV #6032 — built-environment realization.
- CIV-UTILITY #5959 — utility/consumable dependencies.
- CIV-SERVICE #5949 — installed asset to operational-service composition.
- ENG-COUPLING #6173 / #6180 / #6181 — cross-domain transfer semantics.
- SE-OBS / twin #6125 — configuration-bound twin snapshots.
- ENG-COMMUNITY #6154 / #6156 — stewardship/operator projection.
- ENG-SYMBIOSIS #6146 — resource cascades and fallback independence.
- ENG-TOOLS #6131 — read-only engineering projection.
- IoT authority/dispatch lineage — #299 onward through the current physical-effect hardening chain.
- Mycelix Capital-to-Commons #912 / #985 / #1020 / #1081 / #1350 — economic, service, stewardship and currentness facts.

## Core theorem

```
canonical domain facts
+ exact configuration generations
+ explicit interfaces/dependencies
+ service contracts
+ spatial references
+ currentness/change lineage
-> PlaceSnapshotV1
```

But:

```
PlaceSnapshot
!= engineering qualification
!= occupancy/permit approval
!= utility approval
!= public-service sufficiency
!= safety case
!= physical actuation authority
```

## Place hierarchy

The hierarchy is composition, not governance:

```
Dwelling -> Building -> Block/Site -> Campus/Cluster
        -> Neighborhood/District -> City/Regional System
```

A home can consume a district service without becoming subordinate to the district software.

## PlaceSnapshotV1

Bind:

1. exact place identity and profile generation;
2. spatial/reference-frame identity;
3. exact configuration generation;
4. parent/child place refs;
5. opaque refs to canonical member systems;
6. exact cross-place interface/coupling refs;
7. typed dependency edges;
8. common-mode groups;
9. service-boundary projections;
10. isolation/degradation profile refs;
11. currentness/change-impact dependencies;
12. publication/privacy class and claim ceiling.

The place object must not copy claim-bearing domain payloads.

### Dependency classes

```
PhysicalFlow
InformationFlow
ControlDependency
ServiceDependency
MaintenanceDependency
CommonModeDependency
AuthorityDependency
SpatialAccessDependency
```

The place layer owns the relationship; the canonical domain owner defines the meaning of the underlying resource or authority.

### Common-mode law

```
three feeds
    + one shared upstream transformer
    != three independent sources
```

Common-mode groups are therefore first-class rather than an afterthought.

### Spatial law

```
nearby != connected
connected != authorized
```

Visual proximity must never manufacture a physical dependency.

### Service law

```
asset exists
!= service reachable
!= service qualified
!= service currently available
!= service resilient
```

Service eligibility/currentness remains with canonical service/evidence owners.

### Degradation law

A fallback can only produce the state declared by its own capacity/currentness profile:

```
FullService
DegradedService
IsolatedModule
SafeShutdown
Unknown
```

Redundancy labels do not imply graceful degradation.

## Change-impact law

A material configuration change produces a deterministic dependency cone.

For example:

```
transformer G4 -> G5
  -> affected feeder interfaces
  -> affected building service projections
  -> affected currentness/twin snapshots
  -> affected continuity/capacity claims
  -> affected physical-control bindings when device identity changes
```

Evidence outside the cone remains reusable when its exact dependency bindings prove it unaffected.

## Cyber-physical boundary

The place layer can reference physical-control capabilities but cannot mint them:

```
place proposal
 -> engineering/evidence evaluation
 -> Mycelix authority/provenance
 -> exact cyber-physical admission
 -> execution/runtime lineage
 -> physical effect
```

Thus:

```
connectivity != authority
intelligence != authority
composition != authority
```

## Profiles

### HomeProfile

Dwelling/building + thermal/HVAC + electrical/solar/storage + water + appliances + local observation + maintenance/service dependencies.

### NeighborhoodProfile

Homes/buildings + shared energy/water/thermal/data + access/service corridors + common-mode groups + local resilience/islanding + stewardship refs + shared maintenance.

### CityProfile

Districts/campuses + utility systems + mobility/logistics + public/service infrastructure + emergency/resilience interfaces + cross-district dependencies + external authority refs.

CityProfile remains compositional; it is not a universal municipal operating system.

## Standards projection boundary

The research review found useful complementary standards rather than one canonical external ontology:

- NIST CPS/smart-city work emphasizes interoperable, replicable, scalable and trustworthy CPS.
- NIST's current building-security work explicitly connects machine-readable Digital Building Profiles, digital twins and building-service cybersecurity.
- Brick provides machine-readable relationships among physical, logical and virtual building entities; Brick 1.4 also uses ASHRAE 223 connection semantics for topology.
- OGC API - Connected Systems provides a standards-based bridge among systems, deployments, observations and commands in geospatial context.
- Project Haystack provides semantic tagging/data models for homes, buildings, factories and cities.
- BACnet remains a building automation/control interoperability protocol.
- OpenADR 3 provides standardized grid/DER demand-response signaling.

These are projection/interoperability surfaces, not evidence or authority.

## Synthetic ladder

The frozen source fixture contains:

- **P0 home:** PV, battery, grid, HVAC, water, sensors, maintenance.
- **P1 block:** four homes sharing transformer, water branch and data branch.
- **P2 neighborhood:** community battery, water/thermal nodes, mobility and maintenance.
- **P3 district:** district heat, water/wastewater, grid and mobility interfaces.
- **P4 city:** multiple districts plus city water/grid/mobility/emergency interfaces.

The first fixture is synthetic and intentionally non-site-specific.

## A1 oracle contract

The independent oracle must:

1. parse only the frozen JSON with Python stdlib;
2. require byte-canonical JSON serialization;
3. verify the exact fixture SHA-256;
4. validate IDs, interface references, duplicate IDs and common-mode membership;
5. derive place identity independently;
6. derive each hostile-case disposition without importing Symthaea production code;
7. compare derived dispositions with the frozen expected values;
8. emit deterministic machine-readable output;
9. never turn a synthetic PASS into a real-world authority claim.

Fixture SHA-256:

```
89fc3954b8c99e7ea7e38041bb4dac24594dedc45b12e1609a7f3ec9156665b7
```

## Claim ceiling

This tranche can establish only deterministic semantics for the exact synthetic place/dependency/service/currentness corpus. It cannot establish structural safety, occupancy, utility reliability, public-service sufficiency, municipal authority, ecological benefit, economic viability, or physical actuation authority.

## External references

- NIST Cyber-Physical Systems / Smart Cities: https://www.nist.gov/programs-projects/cyber-physical-systemsinternet-things-smart-cities
- NIST Cybersecurity for Building Systems: https://www.nist.gov/programs-projects/cybersecurity-building-systems
- NIST Building Digitization and Semantic Interoperability: https://www.nist.gov/programs-projects/building-digitization-and-semantic-interoperability
- Brick: https://brickschema.org/
- Project Haystack: https://project-haystack.org/
- OGC API - Connected Systems: https://www.ogc.org/standards/ogc-api-connected-systems/
- ASHRAE BACnet: https://www.ashrae.org/technical-resources/standards-and-guidelines/standards-addenda
- OpenADR: https://www.openadr.org/specification
