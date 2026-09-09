# Civilization Continuity Profile V1

Status: architecture profile; no runtime authority.

## Purpose

Civilization continuity extends the existing Symthaea continuity theorem from computing transitions to
material, human, institutional, knowledge, energy, and infrastructure capabilities.

The objective is not to preserve a frozen civilization or to promise recovery after arbitrary collapse.
The objective is to make capability loss, substitution, degradation, and recovery explicit enough to be
reasoned about, verified, exercised, and governed.

The core invariant is:

```text
CapabilityDefinition
    !=
CapabilityRealization
    !=
CurrentAvailability
    !=
VerifiedSufficiency
    !=
Authority
```

A recovery analysis is also not an action:

```text
RecoveryAnalysis
    != work order
    != purchase authorization
    != human assignment
    != machine execution
```

Human or community governance decides what actually happens.

## Relationship to the existing continuity kernel

This profile is additive. It does not change the V1 byte contract of
`ContinuityRequirementV1`, `ContinuityContractV1`, verification evidence, qualified witnesses,
verifier-profile adoption, or any authority-bearing lineage.

The existing free-form `ContinuityRequirementV1::capability` remains unchanged. A later bridge may bind
that field to a canonical capability identity only through an explicitly versioned contract.

Capability identity is also distinct from operational subject identity. A capability describes what must
remain possible. A subject describes the thing, service, site, fleet, or other scope whose continuity is
being considered. A later realization layer may bind the two without making either identity an authority
statement.

## Threat model

The profile is intended to describe graceful degradation under combinations of:

- local or regional infrastructure loss;
- prolonged network partition;
- supplier and logistics loss;
- machine, tool, energy, or material loss;
- specialist loss and skill succession gaps;
- knowledge-copy, format, or toolchain loss;
- correlated failures hidden behind apparently redundant resources;
- divergent schemas, institutions, and histories after long partition;
- partial recovery where lower-service substitutes are available before full restoration.

It does not assume that all failures are independent, that all dependencies form a DAG, or that the
highest-technology realization is always the best recovery path.

## Reference degradation levels

C0-C5 are contextual continuity states. They are not intrinsic properties of a capability. Exact
thresholds belong to an adopted continuity profile for a specific community or system.

| Level | Reference meaning |
| --- | --- |
| C0 | Nominal operation: ordinary infrastructure, connectivity, supply, and specialist access are available. |
| C1 | Local degradation: one or more local dependencies fail, but ordinary regional substitutes and support remain available. |
| C2 | Regional degradation: important infrastructure, suppliers, or communications are impaired and local substitution becomes necessary. |
| C3 | Prolonged partition: the community must maintain critical services from a substantially local dependency closure for an extended period. |
| C4 | Deep capability degradation: advanced industrial or technical dependencies are unavailable and lower-technology realizations may be required. |
| C5 | Humane continuity floor: preserve minimum life, dignity, agency, essential knowledge, and recoverability while higher capability tiers are absent. |

A capability can have different obligations at different levels. For example, an advanced diagnostic
service may be mandatory at C0, desirable at C2, and unavailable-but-acceptable at C4, while sterile
wound care may remain mandatory throughout.

Therefore:

```text
CapabilityDefinition != ContinuityPolicy
```

The C0-C5 policy is contextual and belongs in a later `ContinuityProfileV1`, not in `CapabilityId`.

## Capability identity and definition identity

Capability graphs must permit cycles. A definition may depend on a capability that eventually depends
back on the first capability.

For that reason, V1 separates a stable semantic capability identity from the exact identity of one
definition revision:

```text
CapabilityId
    = stable namespace + logical capability key

CapabilityDefinitionId
    = exact canonical definition revision
```

If `CapabilityId` were the content hash of its full dependency definition, cyclic definitions would
require recursive hashes. Separating the stable key from the exact definition revision avoids that
self-reference while still making every definition revision independently verifiable.

Neither identity proves that the capability exists, is available, is sufficient, or is authorized.

## AND/OR cyclic prerequisite model

Civilization is not a simple DAG.

A capability can require all of several prerequisites while accepting alternatives for another
prerequisite:

```text
make-pump-shaft =
    suitable-material
    AND
    (lathe OR mill OR qualified-manual-machining-path)
    AND
    dimensional-metrology
    AND
    qualified-operator
```

V1 therefore needs three pure requirement forms:

```text
Leaf(CapabilityId)
AllOf([...])
AnyOf([...])
```

Requirement expressions are declarations only. They do not prove the referenced capabilities exist or
that an `AnyOf` branch is currently usable.

Cycles are valid. Closed-world graph validation must resolve every referenced `CapabilityId`, but must
not reject a graph merely because it contains strongly connected components. SCC analysis and recovery
frontiers belong to a later tranche.

## Realization boundary

A canonical capability definition is not evidence that a community can realize it.

A later realization claim may bind a capability to evidence-bearing local realizers such as:

```text
machine
+ operator skill
+ process
+ metrology
+ materials
+ energy
+ knowledge
```

The realization layer must preserve these distinctions:

```text
declared machine capability
    != evidence-backed realization claim
    != current availability
    != verified service sufficiency
```

A broken lathe, expired calibration, unavailable operator, missing material, or failed power supply can
invalidate current availability without changing the canonical capability definition.

## Correlated failure domains

Raw counts are not resilience.

Three operators, archives, pumps, or generators that all depend on the same building, grid feed,
institution, machine, supply source, cryptographic root, or storage system may represent one effective
failure domain.

Continuity projections should therefore be able to reason about typed failure-domain references such as:

```text
geography
power
institution
machine
knowledge-copy
supply-chain
authority-root
```

A later projection may report both total count and independent failure-domain count. This profile does
not define privacy-sensitive identity disclosure or a universal independence metric.

## Human fallback and human standing

Continuity is a decision-support and verification architecture, not a coercion engine.

The system may identify:

- a critical skill bottleneck;
- a fragile succession path;
- a lower-service fallback;
- a recovery multiplier;
- a dangerous dependency concentration.

It may not infer from those facts that a particular person must work, train, move, surrender resources,
accept a policy, or lose standing.

Cross-cutting invariant:

```text
uncertainty or machine failure
    may remove machine privilege
    must not silently erase baseline human standing
```

Optimization must never treat coercive labor, deprivation of rights, involuntary assignment, or
authority escalation as recovery shortcuts.

When automation, models, identity infrastructure, or networks fail, the architecture should retain a
human-governed path for critical decisions.

## Closed-world graph snapshots

A validated capability graph snapshot should eventually establish only that:

- each definition is structurally and canonically valid;
- each `CapabilityId` has at most one definition in that snapshot;
- every referenced capability is defined in that snapshot;
- definition ordering and snapshot identity are deterministic;
- cycles are permitted and preserved.

It does not establish that any capability is locally realized, currently available, sufficient for an
adopted service obligation, safe to operate, or authorized.

The validated graph form should be a non-Serde wrapper so untrusted deserialization cannot directly
construct a downstream-trusted graph.

## Recovery semantics

The first recovery compiler should remain advisory and evidence-grounded. It should be able to ask:

```text
What required capabilities are missing?
Which prerequisite sets block them?
Which alternative paths are structurally reachable?
Which retained or restored capabilities unlock the largest downstream region?
```

Later planning may consider vectors such as time, energy, materials, skill-hours, tooling, ecological
burden, risk, and uncertainty. These dimensions should remain visible rather than being collapsed into
one universal scalar score.

A simulated recovery result is scenario evidence, not a forecast.

## Human-rights floor

Every continuity profile must preserve a non-optimizable human floor. At minimum, recovery machinery
must not gain authority to:

- compel labor or training;
- revoke baseline personhood or standing because credentials, models, or networks fail;
- hide uncertainty behind a deterministic recommendation;
- rewrite historical provenance to simplify reunion;
- turn a continuity obligation into ownership of a person, community, or resource;
- silently convert advisory analysis into executable authority.

More demanding communities may adopt stronger floors. This V1 specifies a minimum asymmetry, not a
complete political constitution.

## Failure-domain and degradation testing

Future destructive qualification should progressively remove assumptions such as Internet access, DNS,
package registries, cloud APIs, a global clock, peer availability, clean storage, key continuity,
specialist continuity, machine availability, energy budget, supply items, and schema convergence.

The qualification question is not merely "does the software still run?"

Useful measures include:

```text
required capability survival
recovery frontier size
critical unmet obligations
human-rights-floor preservation
historical provenance retention
knowledge retention
skill succession
reunion conflict count
recovery time
dependency concentration
```

Claims must remain no stronger than demonstrated evidence.

## First end-to-end proving scenario

The first proving scenario should be deliberately small:

> A community workshop loses a critical water-pump capability. Given actual machines, BOM/process
> information, skills, energy/material availability, and preserved knowledge, can Symthaea identify
> missing prerequisites and several evidence-backed restoration paths without issuing work,
> purchasing, human-assignment, or execution authority?

Candidate outcomes might include an unavailable OEM replacement, a blocked local-machining route, a
reachable alternate-pump adapter, and a lower-throughput emergency fallback.

This scenario is large enough to exercise AND/OR prerequisites, substitutions, currentness, failure
domains, human skill, and service-level degradation without pretending that the system has modeled all
of civilization.

## Exact non-claims for this profile

This architecture does not:

- predict civilizational collapse or recovery time;
- assert that a capability definition is true or locally realizable;
- mutate the existing continuity V1 wire contract;
- define capability realization, currentness, or authority wire formats;
- define SCC or recovery-frontier algorithms;
- define Mycelix Manufacturing, Craft, Praxis, Knowledge, Energy, or Commons adapters;
- define DTN reliability or reunion semantics;
- create a new archival container format;
- authorize work, procurement, migration, execution, governance, or human assignment;
- claim that C0-C5 thresholds are universal across communities.

## Immediate tranche boundary

The implementation sequence is intentionally narrow:

```text
CC-00  this architecture profile
CC-01  canonical capability identity + AND/OR requirement definitions
CC-02  closed-world canonical graph validation
CC-03  SCCs + recovery frontier
CC-04  contextual C0-C5 continuity profile
```

The first three tranches establish the language. Recovery algorithms come only after the graph can be
validated exactly.
