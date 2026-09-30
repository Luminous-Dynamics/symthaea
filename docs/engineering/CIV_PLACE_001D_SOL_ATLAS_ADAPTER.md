# CIV-PLACE-001D — Sol Atlas PlacePlan adapter

Status: executable adapter tranche

Parent: #6615  
Base kernel: #6618 / CIV-PLACE-001B-B  
External planner contract: Luminous-Dynamics/luminous-dynamics#3482

## 1. Boundary

The adapter is a narrow translation boundary:

Sol Atlas PlacePlanV1
        ↓
strict adapter DTO
        ↓
qualified projection bindings
        ↓
CIV-PLACE dependency/currentness kernel
        ↓
read-only PlaceEvaluationV1

The adapter does not create a new spatial ontology and does not import geometry semantics into the kernel.

It does not:
- infer currentness from provider release presence;
- infer dependency from containment or proximity;
- infer independence from missing edges;
- create permits or authority;
- emit commands or actuation;
- turn a rendered object into physical evidence.

## 2. Why the adapter has two inputs

A PlacePlanV1 describes a user-authored scenario. It does not, by itself, establish current physical state.

Therefore the adapter requires explicit SolAtlasNodeBindingV1 and SolAtlasProjectionBindingV1 records.

PlacePlan
  = proposed composition + declared assumptions

NodeBinding
  = explicit currentness/state projection

ProjectionBinding
  = completeness/convergence/contradiction state

This prevents a plan from silently upgrading itself into a qualified world-state projection.

## 3. Dependency direction

Sol Atlas expresses:

home → transformer

as “the home depends on the transformer”.

CIV-PLACE's closure kernel traverses upstream dependencies as:

transformer → home

The adapter performs exactly one explicit direction inversion:

source = PlacePlan.to_id
target = PlacePlan.from_id

This is covered by an executable test so a future refactor cannot accidentally invert the relation twice.

## 4. Dependency discovery

A new hardening rule is carried through the adapter:

dependency_discovery = Complete
        → closure may support a service disposition

dependency_discovery = Partial / Unknown / Conflicted
        → UnresolvedDependencies

This is important because:

> an incomplete dependency graph is not evidence of independence.

The kernel still allows a separate positive independence witness to establish IndependentWitnessed; missing discovery alone never establishes it.

## 5. Determinism

The adapter canonicalizes:
- external references;
- source snapshots;
- non-goals;
- elements;
- element external references;
- dependencies;
- assumptions;
- node bindings;
- projection bindings;
- service bindings.

The projection receipt binds:
- adapter profile;
- exact plan ID/version;
- canonical plan SHA-256;
- projected dependency identities;
- service identities;
- kernel schema version;
- plan claim ceiling.

A permutation of semantically unordered input collections therefore cannot change the projected contract.

## 6. Qualification fixture

docs/engineering/fixtures/sol-atlas-place-001d.json contains:
- one site;
- one neighborhood;
- one block;
- eight homes;
- one shared transformer;
- one shared water node;
- explicit common-mode groups;
- explicit currentness/projection bindings;
- three service questions.

The fixture is synthetic and exists only to qualify deterministic adapter behavior.

## 7. Differential gate

scripts/diff_sol_atlas_place_001d.py independently re-derives the projection and evaluation surface in Python and compares it to the Rust oracle.

The current corpus includes:
1. nominal scenario;
2. permutation of semantically unordered collections;
3. stale home currentness;
4. partial source projection;
5. incomplete dependency discovery;
6. unrelated geometry revision.

The intent is not to prove a real site safe. It is to prove that the exact declared adapter contract is translated consistently.

## 8. Standards boundary

Sol Atlas remains the spatial experience; external standards remain adapters/interchange surfaces.

CityGML 3.0 is a semantic conceptual model for virtual 3D city/landscape models and explicitly covers urban planning, architectural design, environmental/energy/mobility simulation and disaster management.

OGC API Features provides modular feature access, while OGC API Tiles provides reusable tiled-data access.

OGC API Connected Systems is particularly relevant at the boundary between static system descriptions and dynamic observations, events and commands. The adapter must keep command-shaped external data separate from local authority.

IFC 4.3.2.0 is the current official IFC release, while Overture's current data release is 2026-09-23.1 and its schema is v2.0.0. Overture's September patch specifically corrected GERS IDs for building and building-part features, reinforcing why exact release identity belongs in provenance.

## 9. Home-engineering consequence

The adapter is deliberately agnostic to home physics. Home engineering stays in HomeDesignScenarioV1 on the Sol Atlas side.

The current research baseline supports keeping envelope, ventilation/IAQ, moisture, thermal comfort and human-experience dimensions separate rather than collapsing them into a single score. Passive House identifies insulation, high-performance windows, airtightness, thermal-bridge control and ventilation with heat recovery as core principles; ASHRAE currently lists Standard 55-2023 and Standard 62.2-2025; EPA Indoor AirPlus Version 2 spans moisture, radon, HVAC/ventilation, pollutant control and materials.

## 10. Mycelix/Holochain boundary

The qualification boundary should preserve Holochain's distinction between deterministic validation and incomplete dependency retrieval.

Holochain documents validation as deterministic/pure and explicitly represents unavailable validation dependencies as UnresolvedDependencies; validation receipts are not a reliable measure of current DHT availability.

That maps cleanly to the Place architecture:

evidence exists
    ≠
currentness established
    ≠
engineering qualification
    ≠
approval
    ≠
physical effect

## 11. Next tranche

The next useful step is not another renderer.

It is to make the adapter consume real PlacePlan fixtures from Sol Atlas and produce the exact read-only evaluation projection required by the Engineer Workbench:

Place Studio
  → PlacePlan
  → qualified projection
  → CIV-PLACE
  → evaluation
  → evidence/currentness/change cone
  → read-only Engineer Workbench

Then add the neighborhood replay corpus so a transformer failure, water degradation, communications partition or maintenance loss can be replayed from the same canonical place artifact.