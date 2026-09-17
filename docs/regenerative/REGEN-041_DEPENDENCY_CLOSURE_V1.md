# REGEN-041 — Local Dependency Closure Contract v1

Status: preregistration / architecture boundary only

Parent: REGEN-040 essential regenerative services

Program: Luminous-Dynamics/mycelix#940

## Purpose

Define dependency closure for regenerative resilience without equating resilience
with autarky, locality, or universal self-sufficiency.

For an externally adopted service profile and declared time horizon, REGEN-041
asks whether every transitive dependency terminates in an explicit,
identity-bearing provision path, bounded reserve, qualified substitute /
fallback, or an unresolved leaf.

The absence of hidden dependencies is the target. "Everything must be local"
is not.

## Governing theorem

```text
adopted service requirement
+ explicit transitive dependency graph
+ identity-bearing provision paths
+ bounded stocks / reserves / capacities
+ explicit substitutes / fallbacks
+ explicit time horizon
= bounded dependency-closure assessment
```

not self-sufficiency, autarky, resource entitlement, procurement authority,
allocation authority, service guarantee, or physical execution.

## Closure != locality

The roadmap phrase "local dependency closure" is retained as a program label,
but the scientific property is dependency closure under a declared locality /
externality classification.

```text
local != closed
external != open
```

A local source with one hidden upstream dependency is not closed. A geographically
external source with an explicit supply path, reserve, fallback, failure domains,
and evidence may be fully represented.

REGEN-041 never rewards locality merely for being local.

## Explicit leaf states

A dependency leaf should terminate in a declared state such as:

```text
ProvisionedLocal
ProvisionedExternal
CoveredByReserve
CoveredByQualifiedSubstitute
NotRequiredInScenario
Unresolved
Unsupported
```

Every leaf must have a visible disposition. A missing branch is never interpreted
as provisioned.

## Transitive closure

Service dependencies are traversed recursively, including fallback dependencies.

```text
food service
 -> cold storage
    -> electricity
       -> grid intertie
          -> external generation / contract / unknown
```

Checking only the first edge is insufficient.

## Time horizon

Closure is always horizon-bound. A reserve that covers one day is not evidence
for a thirty-day horizon.

```text
DependencyClosureQuery {
    service_profile_id,
    scenario_id,
    horizon,
    support,
}
```

Reserve and capacity witnesses must use compatible time bases.

## Stock, rate, and service distinctions

A stock and a consumption rate are not directly comparable. Coverage calculations
are valid only when commodity identity, unit basis, conversion losses, demand
semantics, and availability are explicit.

```text
gross stock
!= allocatable stock
!= protected reserve
!= service-usable reserve
```

REGEN-041 references authoritative eligibility / commitment state instead of
overriding it.

## Protected reserve

The existing subterranean logistics system demonstrates a useful principle:
discretionary work is refused before consuming a protected return reserve.

REGEN-041 generalizes the principle without adopting its mission-specific policy.

```text
usable_for_declared_service
=
qualified_available
- prior_commitments
- protected_reserves_not_releasable_for_this_service
```

The actual reserve policy remains externally adopted.

## Shared reserve firewall

The same reserve cannot independently close multiple simultaneous service
dependencies unless an explicit contention model demonstrates enough capacity.

```text
10 units reserve
 -> service A claims 10
 -> service B claims 10
```

does not establish 20 units of closure.

One reserve witness is not unlimited independent fallback capacity.

## Provision paths

A dependency may be provisioned by an external or local path.

```text
ProvisionPath {
    dependency_id,
    source_id,
    route_or_interface_id,
    capacity,
    lead_time,
    time_support,
    evidence_refs,
    common_cause_domains,
    epistemic_class,
}
```

Transport, network, supplier, and receiving dependencies remain explicit where
material.

## Lead time

Capacity alone does not prove continuity.

```text
quantity available
!= quantity available before service floor fails
```

Lead time must be compared with reserve coverage and REGEN-040 outage semantics.

## Substitution

```text
substitute exists
!= substitute compatible
!= substitute sufficiently capacious
!= substitute available in time
```

REGEN-043 may later explore substitution frontiers. REGEN-041 only evaluates a
declared substitute path.

## Skill and repair dependencies

Not every dependency is material. The graph may contain operator skill,
maintenance capability, repair tooling, software, spare parts, documentation,
communications, or authorization.

```text
knowledge available
!= trained person available
!= tool available
!= spare available
!= repair achievable in horizon
```

A spare without skill / tooling does not close a repair dependency.

## Common-cause domains

REGEN-040 multi-domain common-cause semantics apply transitively. Nominally
different sources can share a supplier, transport corridor, watershed, power
source, network, geography, software image, workforce, facility, or authorization
dependency.

Nominal source count is not effective independence.

## Cycles

A dependency cycle is not automatically impossible if explicit stocks and time
semantics make it physical. But a bare cycle is not closure.

```text
A depends on B
B depends on A
```

requires an initialization / stock / ordering witness or remains unresolved.
REGEN-039 stock-and-time semantics should be reused.

## External imports are first-class

External imports remain visible rather than being hidden to inflate apparent
self-sufficiency.

The report may distinguish local, external, reserve-covered, substituted, and
unresolved dependencies. It does not turn those classes into a moral ranking.

```text
local != resilient
external != fragile
```

## Closure report

```text
DependencyClosureReport {
    query,
    closed_leaves,
    reserve_covered_leaves,
    external_leaves,
    substituted_leaves,
    unresolved_leaves,
    unsupported_leaves,
    shared_bottlenecks,
    horizon_shortfalls,
    evidence_gaps,
}
```

There is no universal closure percentage required by the contract. Essential
unresolved dependencies must never disappear behind an aggregate number.

## Missing-data firewalls

```text
missing demand != zero demand
supplier exists != unlimited supply
candidate supplier != contracted supply
contracted supply != delivered supply
```

Capacity, lead time, commitment, transport, receiving capability, and delivery
state remain separate propositions.

## Authority firewall

REGEN-041 cannot:

- reserve or purchase a resource;
- create a contract;
- transfer ownership;
- release a protected reserve;
- authorize extraction;
- override ecological retention;
- dispatch infrastructure;
- allocate essential services.

It evaluates dependency representation only.

## First executable candidate

Use one REGEN-040 service with a small dependency DAG and prove:

1. recursive traversal terminates;
2. every leaf has an explicit disposition;
3. missing leaves remain unresolved;
4. reserve coverage is horizon-aware;
5. one reserve cannot be double counted across concurrent demands;
6. nominal fallbacks sharing a common cause are not independent;
7. explicit external supply is not penalized merely for being external;
8. external supply with unknown capacity / lead time remains unresolved;
9. a cycle without stock/time semantics is rejected or unresolved;
10. no authority-bearing action is emitted.

## Adversarial fixtures

Qualification should include hidden transitive dependencies, insufficient
reserve horizon, shared reserve double counting, listing-as-contract,
contract-as-delivery, shared transport corridors, incompatible substitutes,
spares without repair capability, knowledge offered as skill, cycles without
initial stock, missing demand treated as zero, missing loss treated as zero,
protected reserves treated as discretionary, and closure reports presented as
procurement authority.

## Relationships

REGEN-039 supplies stock / flow / external-boundary semantics.

REGEN-040 supplies adopted service floors, outage, fallback, and common-cause
semantics.

REGEN-041 composes them into a transitive dependency question.

REGEN-042 should perturb the same dependency identities under compound shocks.
REGEN-043 should deepen substitution semantics. REGEN-044 should deepen repair /
skill dependencies.

No downstream phase may silently rebuild a second dependency graph.

## Promotion gate

An executable REGEN-041 candidate should not be called qualified until its exact
subject demonstrates:

- frozen toolchain / code lineage;
- compile / test / strict lint;
- deterministic traversal;
- exact service / dependency / provision identities;
- explicit time horizon;
- unit-safe stock/rate coverage;
- reserve double-count rejection;
- common-cause propagation;
- unresolved-frontier preservation;
- external provision without locality bias;
- cycle handling with stock/time semantics;
- no authority-bearing output;
- postflight immutability.

## Deliberate non-claims

REGEN-041 does not establish self-sufficiency, autarky, optimal localization,
supply guarantees, contractual enforceability, actual delivery, equitable
allocation, resource ownership, economic feasibility, emergency authority,
procurement authority, or physical control.

It establishes only whether a declared service dependency graph has an explicit,
evidence-bearing frontier under a stated scenario and time horizon.
