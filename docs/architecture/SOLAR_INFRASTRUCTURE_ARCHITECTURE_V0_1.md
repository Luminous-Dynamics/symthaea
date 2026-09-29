# Solar Infrastructure Architecture v0.1

Status: architecture baseline
Date: 2026-09-10

## Purpose

Define a thin, interoperable infrastructure layer for off-Earth systems without creating a monolithic space stack or granting cognitive systems direct physical authority.

## Design laws

1. Interoperable before optimized.
2. Serviceable before disposable.
3. Intelligence is not authority.
4. Local deterministic controllers retain immediate physical authority.
5. Shared infrastructure should replace unnecessary duplicated infrastructure.
6. Evidence must accompany authority escalation.
7. Infrastructure should increasingly build infrastructure.

## Layering

Human and organizational intent flows into Mycelix for identity, authority, ownership, agreements, provenance, resource allocation, and coordination. Symthaea performs planning, prediction, diagnosis, simulation, optimization, and trade-space search. Proposed actions then pass through deterministic safety and interface validation before any local controller may execute them.

```text
human / organizational intent
            |
            v
         Mycelix
 identity / authority / provenance / coordination
            |
            v
         Symthaea
 planning / prediction / optimization / diagnosis
            |
            v
   digital twin + evidence
            |
            v
 verified execution boundary
 hard constraints / interlocks / validation
            |
            v
      local controllers
 power / robotics / logistics / ISRU / comms / transport
```

## Scope boundaries

This architecture does not replace CCSDS networking, LunaNet, flight software, certified local controllers, structural solvers, CFD solvers, electrical solvers, or other domain-specific truth models. Instead it composes them through typed capability and evidence boundaries.

Symtropy remains the fast deterministic world/systems simulation substrate. High-fidelity structural, fluid, circuit, and multibody studies remain behind existing simulation bridges. `symthaea-orbital` remains the orbital-mechanics and orbital-operations domain. `symthaea-digital-twin`, `symthaea-formal-safety`, `symthaea-engineering`, `symthaea-hal`, and `symthaea-fabrication-kernel` remain authoritative for their existing concerns.

## Canonical infrastructure vocabulary

The first implementation slice should standardize these concepts:

- `AssetId`: stable identity for a physical or virtual infrastructure asset.
- `ResourceKind`: energy, power, water, oxygen, propellant, bandwidth, compute, storage, cargo capacity, thermal rejection, habitation volume, material, or extensible domain-specific resource.
- `ServiceCapability`: a bounded service an asset offers, including capacity, availability, constraints, and interface requirements.
- `InterfaceRef`: a reference to an external or internal interoperability contract rather than an embedded replacement standard.
- `Observation`: time-bound measured state with source and uncertainty.
- `Hazard`: an identified unsafe condition with severity and evidence references.
- `Reservation`: a temporary resource allocation that does not itself imply actuator authority.
- `CommandProposal`: a non-authoritative proposed action.
- `Authorization`: a narrowly scoped authority grant bound to subject, operation, validity window, and policy/evidence context.
- `ExecutionReceipt`: evidence that a local controller accepted, rejected, or completed an authorized operation.

## Authority invariant

A `CommandProposal` must never be executable solely because Symthaea produced it. The expected progression is:

```text
proposal
  -> domain feasibility validation
  -> operational constraint validation
  -> safety validation
  -> external authority validation
  -> local-controller acceptance
  -> execution receipt
```

Every transition is explicit and fail-closed. Missing evidence, stale authority, ambiguous interface compatibility, or uncertainty outside an allowed envelope must prevent promotion to the next stage.

## Reference infrastructure graph

Infrastructure should be modeled as nodes and service edges rather than by hard-coding transportation technologies.

A transport edge advertises capacity, accepted cargo classes, timing/window constraints, energy or propellant use, latency, reliability, risk, interface requirements, and availability. The edge may represent a rover, rail line, reusable lander, electric tug, mass driver, elevator, or future transport system.

The same abstraction should work for power, communications, maintenance, manufacturing, storage, and resource transfer.

## External interoperability boundary

Adapters should describe compatibility with external standards and service families, not reimplement them. Initial adapter targets include CCSDS/DTN networking concepts, LunaNet service descriptions, and NASA deep-space interoperability interface families. Any conformance claim must distinguish `declared`, `modeled`, `tested`, and `qualified` evidence states.

## Reference scenario

The initial benchmark is intentionally small:

- two independent power sources;
- one storage installation;
- ten cargo/maintenance robots;
- two excavators;
- one ISRU demonstrator;
- one communications relay;
- one navigation source;
- one warehouse;
- one landing zone;
- one habitat;
- one spare-parts depot.

The benchmark must include seeded failure campaigns covering communications loss, navigation degradation, power-source loss, wheel/actuator failure, battery degradation, dust contamination, inventory mismatch, ISRU underperformance, route obstruction, thermal excursion, sensor disagreement, and concurrent failures.

Success means the settlement preserves explicit safety invariants while continuing useful operation. It does not mean that Symthaea controls every asset.

## Optimization objective

Architecture search should return Pareto fronts rather than a single hidden scalar ranking. Required dimensions include Earth-imported mass, energy, cost, risk, reliability, repairability, throughput, latency, local productive capacity, and uncertainty.

A useful derived metric is off-Earth productive multiplication:

```text
productive_multiplication = durable_increment_in_off_earth_productive_capacity
                            / earth_imported_mass_or_equivalent_seed_resource
```

The metric is diagnostic, not authoritative, and must never hide the underlying Pareto dimensions.

## Planned implementation tranches

- SIF-001: architecture record and thin infrastructure ontology.
- SIF-002: cislunar orbital foundations: explicit frames/units, lunar/solar third-body support, rotating Earth-Moon frame, CR3BP primitives, and L1/L2 calculations.
- SIF-003: compose existing digital-twin, formal-safety, observability, engineering, and simulation-bridge capabilities without duplication.
- SIF-004: external standards/conformance adapters.
- SIF-005: lunar logistics reference benchmark.
- SIF-006: deterministic seeded fault campaign and unsafe-action rejection tests.
- SIF-007+: Mycelix resource coordination, trade-space search, high-throughput transport studies, and bootstrap experiments.

## Non-goals

- no custom LunaNet or DTN replacement;
- no second orbital-mechanics library;
- no monolithic `symthaea-space-everything` crate;
- no flexible-body solver added to Symtropy core solely for a lunar elevator;
- no learned model receives unrestricted actuator authority;
- no claimed real-world qualification from simulation-only evidence;
- no infrastructure interface depends on Symthaea consciousness/HDC internals.

The interface layer must remain usable by Symthaea, conventional planning software, deterministic controllers, human operators, and other independent implementations.
