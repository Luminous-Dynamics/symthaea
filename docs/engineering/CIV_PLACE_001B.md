# CIV-PLACE-001B — typed dependency/currentness composition engine

**Status:** independent reference-engine tranche. Read-only. No production authority.

## Purpose

CIV-PLACE-001B turns the 001A composition contract into a bounded, deterministic dependency/currentness reference model.

It answers:

> Given one exact place/service question, one bounded semantic projection, and one declared rule profile, which dependencies are in scope, what currentness is provable, which common-mode groups remain visible, and what service disposition follows?

It does not own the persistent knowledge graph, domain truth, observations, utility/service authority, governance, authorization, physical control, device drivers, or municipal policy.

## Architecture boundary

    Mycelix Holochain DKG
            |
            | qualified projection / semantic closure
            v
    bounded engineering input
            |
            v
    CIV-PLACE-001B reference engine
            |
            +--> DependencyClosureV1
            +--> currentness disposition
            +--> common-mode closure
            +--> PlaceEvaluationV1
            |
            v
    read-only engineering projection

The raw DKG is not traversed as though it were an acyclic engineering graph.

Holochain's public data model uses agent source chains and a validating distributed hash table. Peers can continue operating during network disruption and later heal distributed state, so a retrieved local/partial view must remain representable as incomplete rather than being promoted to globally-current engineering state.

## Core types

### DependencyEdgeV1

Each dependency edge binds:

- stable edge identity;
- exact source and target references;
- dependency class;
- required/optional semantics;
- common-mode group where applicable.

A dependency edge means the relationship is declared. It does not establish that the underlying system is physically adequate or currently operational.

### CommonModeGroupV1

Common-mode groups retain shared failure domains.

    A -> transformer
    B -> transformer
           |
        feeder

does not become two independent feeds.

The model distinguishes:

    SharedDependency
    IndependentWitnessed
    IndependenceUnknown

The absence of a discovered shared dependency is not evidence of independence. Independence requires a positive, scoped witness whose projection is complete and non-conflicted.

### CurrentnessWitnessV1

Currentness is a compound property. The reference model separates:

- semantic projection completeness;
- convergence state;
- contradiction state;
- node currentness;
- service currentness requirements.

Therefore:

    retrievable != current
    current != qualified
    qualified != authorized

and:

    partial / partitioned -> PartiallyAvailable

Historical evidence remains evidence of historical state but cannot satisfy a currentness requirement after a material change.

### DependencyClosureV1

A closure is computed only for a declared service question.

It records:

- root;
- selected nodes;
- selected dependency edges;
- common-mode groups;
- deterministic traversal boundary;
- bounded-cycle behavior.

The underlying knowledge substrate may contain cycles. The task-specific projection terminates deterministically with a visited set and declared resource bound.

### PlaceEvaluationV1

The read-only evaluation contains:

- service identity;
- service disposition;
- dependency closure;
- currentness state;
- exact reasons for blocking/degradation/conflict.

It contains no command, actuator, authorization, or mutation capability.

## Service disposition rules

For this reference tranche:

    Conflicted input
        -> Conflicted

    incomplete currentness projection
        -> Blocked

    unavailable required dependency
        -> Unavailable

    unavailable optional dependency
        -> DegradedService

    all required dependencies current/available
        -> FullService

This is a small reference theorem, not a real-world reliability model.

## Determinism requirements

The oracle requires:

1. byte-canonical JSON;
2. exact SHA-256 fixture commitment;
3. stable identifier ordering;
4. dependency-order permutation invariance;
5. deterministic cycle termination;
6. irrelevant-material invariance;
7. explicit contradiction preservation;
8. no confidence/attestation promotion of engineering authority.

## DKG / epistemic boundary

A DKG claim, attestation, or confidence value can be part of the lineage used to build a qualified projection.

It cannot by itself create:

    physical truth
    currentness proof
    independence proof
    safety qualification
    authorization
    actuation authority

This is important for a peer-to-peer substrate: a local node's knowledge can be internally valid while still incomplete.

## Interoperability boundary

External standards remain projections:

- OGC API - Connected Systems v1.0 for system/deployment/dynamic-data/command exchange;
- Brick / ASHRAE 223 for building topology and connection semantics;
- BACnet for building automation interoperability;
- Project Haystack for semantic tagging/modeling;
- OpenADR for demand-response signaling.

A mapping must bind its standard, version, profile, mapping artifact, and exact semantic status.

A label-only mapping is ProjectionIncomplete rather than inferred semantic equivalence.

Likewise:

    command schema != authorization
    adapter acceptance != physical effect

## Frozen hostile corpus

The 001B fixture contains 18 hostile cases covering:

1. absence-of-evidence independence;
2. positive independence witnesses;
3. partial DKG views;
4. late-arriving semantic dependencies;
5. irrelevant DKG material;
6. attestation mutation;
7. confidence mutation;
8. contradictory claims;
9. historical-not-current evidence;
10. partitioned/non-converged views;
11. nested common-mode groups;
12. maintenance loss;
13. unresolved fallback currentness;
14. label-only interoperability mappings;
15. command-shaped objects without authority;
16. governance-only changes;
17. historical evidence after material change;
18. cyclic dependency closure.

The corpus includes both positive and negative evidence so the engine cannot pass merely by rejecting everything.

## Qualification

The independent oracle emits a deterministic machine-readable PASS containing the exact fixture digest, service count, hostile-case count, permutation result, cycle result, and authority claim.

A PASS establishes only deterministic behavior of the frozen synthetic reference model.

It does not establish:

- physical infrastructure resilience;
- safety;
- structural adequacy;
- utility reliability;
- public-service sufficiency;
- regulatory approval;
- municipal authority;
- economic viability;
- physical-control authority.

## Next tranche

    001B-A  independent reference oracle      <- this tranche
    001B-B  typed Rust leaf kernel
    001B-C  differential Rust/Python corpus
    001B-D  synthetic neighborhood replay
    001C    read-only Engineer Workbench
    001D    qualified cyber-physical adapter

The Rust kernel should remain leaf-like and side-effect-free. It should consume qualified projections rather than importing the Mycelix DKG implementation directly.

## Fixture commitment

    38bd385f1bd7a53daf46ff4f9af3cefaf0542d1348d588fdc1c46bca90d8f46a
