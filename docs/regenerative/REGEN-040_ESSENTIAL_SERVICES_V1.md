# REGEN-040 — Essential Regenerative Services Contract v1

Status: preregistration / architecture boundary only

Parent: REGEN-039 settlement metabolism graph

Program: Luminous-Dynamics/mycelix#940

## Purpose

Define the first service-continuity boundary for regenerative resilience without
allowing Symthaea to decide which human/community services are "essential," and
without collapsing resilience into a single score.

REGEN-040 evaluates an externally adopted service profile against explicit
availability, dependency, fallback, capacity, outage, recovery, and evidence
state.

It does not govern, allocate resources, or control infrastructure.

## Governing theorem

```text
adopted service requirement
+ current service evidence
+ dependency evidence
+ fallback evidence
+ common-cause dependency model
+ declared outage / recovery objectives
= bounded service-continuity assessment
```

not:

```text
service entitlement
public policy
resource allocation
infrastructure authority
resilience superiority
physical action
```

## Essentiality is externally adopted

Symthaea must not infer that a service is essential from:

- model centrality;
- economic value;
- frequency of use;
- network degree;
- predicted harm;
- AI confidence;
- historical importance.

The service profile must identify the source of adopted essentiality.

Conceptually:

```text
ServiceRequirement {
    service_id,
    adopted_profile_ref,
    criticality,
    minimum_service_floor,
    maximum_outage,
    recovery_objective,
    dependency_requirements,
    fallback_requirements,
}
```

The model evaluates the profile. It does not create the profile's social
authority.

## Service != resource

A service is an outcome / capability delivered over time.

It must not be confused with one input stock.

```text
water stock != water service
food stock != food service
energy stock != energy service
repair parts != repair service
```

A service may require multiple resources, infrastructure components, skills,
locations, and dependencies.

## Availability state

A first-class availability state should preserve uncertainty.

Suggested states:

```text
Available
Degraded
Unavailable
Unknown
```

Normative:

```text
Unknown != Available
Unknown != Unavailable
```

Unknown evidence must remain visible.

## Service floor

A binary up/down state is insufficient for many settlement services.

An adopted profile may declare a minimum service floor in a service-specific
unit or vector.

Examples are deliberately not hard-coded into the core.

Conceptually:

```text
ServiceFloor {
    metric_id,
    minimum_acceptable,
    unit_basis,
    population_or_support,
    time_support,
}
```

A fallback counts only if it can satisfy the declared floor under the assessed
scenario.

## Existing resilience precedent

`symthaea-helicopter::service_resilience` already demonstrates useful general
ideas:

- explicit criticality;
- dependency graphs;
- maximum outage;
- recovery objectives;
- evidence-bearing observations;
- fallback declarations;
- cycle detection.

REGEN-040 reuses the conceptual discipline, not the aviation-specific authority
or service identities.

## Important strengthening: fallback viability

A declared fallback is not automatically a working fallback.

Normative:

```text
fallback declared
!= fallback available
!= fallback independent
!= fallback capacity sufficient
!= fallback service floor satisfied
```

A fallback should count only when all required conditions are demonstrated.

At minimum:

1. the fallback identity exists;
2. current fallback availability is known and acceptable;
3. its own dependencies are acceptable;
4. it has adequate modeled / observed capacity for the adopted service floor;
5. activation / transfer assumptions are explicit;
6. it is sufficiently independent of the failed primary path;
7. supporting evidence is present.

A boolean `fallback_active=true` is not enough.

## Dependency graph

Each service may depend on:

- another service;
- physical infrastructure;
- a resource stock / flow;
- a skill / operator capability;
- a supplier;
- a network;
- a location;
- an external source.

REGEN-040 should preserve dependency identity rather than flatten all
dependencies into one availability percentage.

Dependency cycles must be explicit and either supported by a time / stock model
or rejected as unresolved.

## Common-cause failure

Nominal redundancy is not independence.

The existing helicopter common-cause model demonstrates this correctly: two
different assets can fail together through shared power, software, cooling,
environment, or configuration.

Settlement continuity needs a more open domain vocabulary.

Conceptually:

```text
FailureDomainRef {
    domain_type,
    domain_id,
}
```

Potential domain types may include:

- geography;
- watershed / source;
- electrical supply;
- communications;
- software / control stack;
- supplier;
- transport corridor;
- workforce / skill;
- fuel / material source;
- environmental exposure;
- shared facility;
- governance / authorization dependency.

The model must not assume two providers are independent merely because their
service IDs differ.

## Multi-domain independence

One string failure-domain label is insufficient.

A primary and fallback may be independent on one axis and coupled on another.

Conceptually:

```text
DependencyDomainSet {
    domains: Set<FailureDomainRef>
}
```

Independence assessment should identify the overlapping domains rather than
produce a bare boolean.

## Capacity

A surviving fallback that cannot meet the adopted service floor is degraded or
insufficient, not "available enough" by implication.

Capacity should retain:

- metric identity;
- unit;
- interval;
- support / beneficiary scope;
- observed / modeled / assumed epistemic class.

```text
fallback exists
!= fallback has required capacity
```

## Time semantics

Continuity requires time.

REGEN-040 must distinguish:

- outage start;
- current duration;
- recovery time;
- recovery objective;
- maximum tolerated outage;
- service degradation interval.

A service that recovers after its objective can be "recovered" and still have
missed the continuity requirement.

## Recovery objective != physical promise

A recovery objective is a profile requirement.

It is not proof that recovery is technically or operationally achievable.

```text
declared recovery target
!= qualified recovery procedure
!= recovery execution
```

## Evidence

Every availability / capacity / dependency / recovery statement should point to
evidence or be explicitly modeled / assumed.

Conceptually:

```text
ServiceObservation {
    service_id,
    availability,
    capacity_state,
    observation_time,
    evidence_refs,
    unresolved_fields,
}
```

Missing evidence should make the assessment incomplete / unresolved rather than
silently healthy.

## Service-continuity result

There is no universal resilience score.

A report should preserve service-level findings.

Conceptually:

```text
ServiceContinuityReport {
    profile_id,
    scenario_id,
    service_assessments,
    dependency_issues,
    common_cause_issues,
    fallback_issues,
    outage_issues,
    unresolved_evidence,
}
```

Each service assessment may expose:

```text
MeetsDeclaredFloor
DegradedButAboveFloor
BelowDeclaredFloor
Unavailable
Unknown
```

The exact names may evolve.

## No weighted collapse

REGEN-040 must not create:

```text
resilience_score = weighted_sum(all_services)
```

A high optional-service score must never offset failure of an adopted essential
service.

Plural service outcomes remain plural.

## Local / external service

An externally supplied service is not inferior by definition.

The model may distinguish dependency location or source, but:

```text
local != resilient
external != fragile
```

Resilience depends on actual dependencies, reserves, fallbacks, failure domains,
and recovery evidence.

## Relationship to REGEN-039

REGEN-039 may provide modeled stocks, flows, service outputs, and dependency
topology.

REGEN-040 evaluates whether those modeled / observed capabilities satisfy an
adopted service requirement under a scenario.

It must preserve whether each input came from:

- observation;
- upstream projection;
- scenario assumption.

A modeled service output is not relabeled as observed availability.

## Relationship to Mycelix authority

Mycelix / local institutions may own:

- service definitions;
- rights;
- obligations;
- adopted minimum floors;
- emergency policy;
- resource allocation.

REGEN-040 evaluates exact referenced profiles.

It does not adopt them.

## First executable candidate

The smallest useful implementation should remain synthetic.

Suggested scope:

1. define service identity / criticality / adopted-profile reference;
2. define availability with `Unknown`;
3. define one service-specific capacity metric;
4. define explicit dependencies;
5. define sets of common-cause domain references;
6. define one primary and one fallback path;
7. require evidence for both;
8. prove a fallback in a shared failure domain does not count as independent;
9. prove an unavailable fallback does not count merely because
   `fallback_active=true`;
10. prove an under-capacity fallback does not satisfy the floor;
11. prove an independent, available, sufficient fallback preserves the declared
    service floor;
12. emit a service-level report without an aggregate resilience score.

## Adversarial fixtures

Qualification should include:

- missing required service;
- unknown availability reported as available;
- fallback active flag with fallback unavailable;
- fallback adequate in capacity but sharing the failed source;
- fallback independent in geography but sharing power;
- two providers sharing one supplier;
- service dependency unavailable;
- dependency observation missing;
- recovery objective missed but service now recovered;
- outage start missing;
- fallback capacity measured on wrong time interval;
- optional services healthy while an essential floor fails;
- externally supplied service incorrectly penalized merely for being external;
- high model confidence presented as service evidence;
- continuity report presented as resource-allocation authority.

Each shortcut must reject or remain explicitly unresolved.

## Relationship to REGEN-041

REGEN-041 local dependency closure should operate on the explicit service /
dependency graph created here.

It must not redefine service criticality or minimum floors.

## Relationship to REGEN-042 / 047

Compound-shock and adversarial common-mode campaigns should reuse the same
service / failure-domain identities.

A shock campaign changes availability / capacity / dependencies.

It does not change what the service profile means.

## Promotion gate

An executable REGEN-040 candidate should not be described as qualified until
its exact subject demonstrates:

- frozen toolchain / code lineage;
- compile / test / strict lint;
- deterministic replay;
- exact service / profile / metric identities;
- explicit `Unknown` state;
- dependency cycle handling;
- fallback availability verification;
- fallback capacity verification;
- multi-domain common-cause assessment;
- outage / recovery objective semantics;
- no aggregate resilience score;
- no authority-bearing output;
- postflight immutability.

## Deliberate non-claims

REGEN-040 does not establish:

- universal essential-service definitions;
- human survival thresholds;
- legal service obligations;
- emergency authority;
- infrastructure safety;
- actual fallback operability;
- resource ownership;
- service entitlement;
- equitable allocation;
- economic feasibility;
- settlement resilience superiority;
- physical control.

It establishes only an evidence-bearing contract for assessing an externally
adopted service-continuity profile.
