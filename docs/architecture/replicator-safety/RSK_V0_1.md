# Replicator Safety Kernel (RSK) v0.1

**Status:** constitutional design tranche

**Scope:** abstract authority and safety semantics for autonomous fabrication and future replicating systems. This document does **not** specify molecular, biological, or physical replication mechanisms.

## 1. Purpose

RSK exists to answer one question:

> How can a system be permitted to create another system without allowing physical reproduction to create, inherit, or amplify authority?

The constitutional rule is:

> **Replication does not imply authority.**

Equivalently:

```text
PhysicalCreation(child) != Authorization(child)
```

The default descendant authority is empty:

```text
Authority(child) = NONE
```

A descendant may acquire bounded authority only through an explicit external grant evaluated against current independent safety conditions.

RSK is intended to sit beside the Symthaea Fabrication Kernel. The Fabrication Kernel remains responsible for qualified execution; RSK adds descendant lineage, bounded replication authority, resource authority, containment assumptions, negative containment, and safety-case binding.

## 2. Non-goals

RSK v0.1 does not:

- implement self-replicating machinery;
- implement molecular or biological replication;
- claim software evidence establishes physical safety;
- make Symthaea or any AI a single point of safety;
- treat a cryptographic signature as proof of physical containment;
- authorize open-environment replication by default;
- convert uncertainty or unavailable monitoring into permission.

## 3. Risk classes

| Class | Capability | Default posture |
|---|---|---|
| R0 | No replication | Existing fabrication safety controls |
| R1 | Bounded manufacture of non-replicating descendants | Provenance + ordinary containment |
| R2 | Bounded self-copy in a controlled environment | RSK authority + independent monitoring |
| R3 | Multi-generation bounded replication in containment | High-assurance RSK + explicit safety case + quorum |
| R4 | Adaptive/evolvable bounded replication | Exceptional research-only posture; no implicit release authority |
| R5 | Open-environment or potentially unbounded replication | No default deployment authority |

Risk classification does not itself grant authority.

## 4. Constitutional invariants

These invariants are normative. Implementations must fail closed if they cannot establish them.

1. **No automatic authority inheritance.** A descendant receives no authority because its parent manufactured it.
2. **No population authority amplification.** Creating more descendants cannot increase aggregate authority.
3. **Explicit replication bounds.** Replication authority is bounded by population, depth, time, resource scope, and environment.
4. **No silent capability escalation.** Descendant capabilities cannot exceed the intersection of parent constraints and an explicit external grant.
5. **Lineage-aware revocation.** Revocation propagates to descendants when policy declares a lineage scope.
6. **Unknown lineage fails closed.** Missing, ambiguous, or unverifiable lineage is not treated as valid lineage.
7. **Containment drift invalidates authority.** Departure from qualified environmental assumptions terminates dependent authority.
8. **Monitor failure cannot increase autonomy.** Missing, stale, contradictory, or unavailable monitoring cannot convert denial into permission.
9. **Evidence expiry invalidates dependent authority.** Safety evidence is scoped and fresh, not timeless.
10. **Independent failure domains.** Physical, resource, computational, and governance controls must not intentionally share a single common point of failure.
11. **No self-minted external authorization.** A physical artifact cannot unilaterally create the authority required for its own external release or replication.
12. **High-consequence release requires independent authorization.** R4/R5 release cannot be approved solely by the system developer, operator, or controlled system itself.

## 5. Authorization equation

A replication decision is positive only when every required positive predicate is established and no negative containment fact is present:

```text
ALLOW = D & L & E & B & P & M & Q & G & !X
```

Where:

- `D` — design qualification is current;
- `L` — descendant lineage is known and valid;
- `E` — the current environment is inside the qualified containment envelope;
- `B` — population/depth/time/resource budget remains;
- `P` — policy and dependent evidence are current;
- `M` — required independent monitoring is healthy and fresh;
- `Q` — the risk-class-specific external quorum is satisfied;
- `G` — an explicit grant exists for this exact subject and requested capability;
- `X` — any applicable negative containment fact exists.

`X` has precedence over all positive authority. Positive permission never cancels quarantine, revocation, stale safety evidence, containment violation, or another explicit negative fact.

Unknown is not true. Unavailable is not true. Timeout is not true.

## 6. Authority model

### 6.1 Physical creation and authorization are separate events

A parent may be authorized to manufacture a child. That manufacturing event creates a lineage fact, not an authority grant.

```text
Create(parent, child)
  -> ChildExists(child)
  -> ParentOf(child) = parent
  -> Authority(child) = NONE
```

A later authorization is a distinct event:

```text
Grant(external_authority, child, requested_capability, bounded_scope)
```

The grant is evaluated against the current RSK predicates. It cannot be inferred from parent authority.

### 6.2 Capability intersection

If bounded descendant authority is allowed, the maximum effective capability is:

```text
Effective(child)
  = ParentConstraint(parent)
  INTERSECT ExternalGrant(child)
  INTERSECT CurrentPolicy
  INTERSECT CurrentContainment
```

No term may widen another term.

### 6.3 Authority is not conserved by copying

Duplicating an authorized artifact does not duplicate its grant. Grants are subject-bound, scope-bound, evidence-bound, freshness-bound, and revocable.

## 7. Core data concepts for the Rust tranche

RSK v0.1 expects the next implementation PR to introduce abstract types equivalent to:

- `ReplicationSubjectId` — identity of the artifact whose replication authority is evaluated;
- `DescendantLineageId` — physical descendant lineage, deliberately distinct from Fabrication Kernel deployment `ReleaseLineage`;
- `ReplicationGrantId` — explicit external authorization identifier;
- `ReplicationCapability` — requested replication-related authority;
- `ReplicationBudget` — remaining children/population, lineage depth, resource units, and expiry;
- `ContainmentEnvelope` — qualified environmental assumptions and their evidence digest;
- `SafetyCaseRef` — immutable reference to the exact evidence/claims supporting current authority;
- `MonitoringSnapshot` — fresh independent observation state;
- `ReleaseQuorumEvidence` — evidence that the required independent authorities approved the requested scope;
- `NegativeContainment` — quarantine, revocation, drift, stale evidence, compromised authority, or another overriding deny fact;
- `ReplicationDecision` — `Allow` or `Deny { reasons }`, with complete auditable reasons.

These types must not overload the existing deployment-release lineage vocabulary.

## 8. Budget semantics

Replication authority is quantitatively bounded. At minimum a grant can constrain:

```text
max_direct_children
max_total_descendants
max_lineage_depth
max_resource_units
not_before
expires_at
containment_envelope
```

Budget consumption is monotonic within a grant generation. Reauthorization creates a new grant/evidence lineage; it does not silently refill the old grant.

A budget cannot be increased by a descendant, by population growth, by replay, or by loss of connectivity.

## 9. Lineage semantics

A descendant record binds at least:

```text
child_subject
parent_subject
lineage_id
creation_event
qualified_design_digest
creation_authority_digest
```

Unknown parents, cycles, ambiguous parents, subject reuse, or broken ancestry are denial conditions until independently resolved.

Policy may revoke:

- one subject;
- one branch of descendants;
- an entire lineage;
- all descendants produced under a compromised grant or evidence root.

Lineage revocation is a negative containment fact and therefore dominates positive grants.

## 10. Containment envelopes

A `ContainmentEnvelope` describes the conditions under which qualification evidence is valid. It is not merely a location string.

It may bind abstract facts such as:

- qualified facility/environment identity;
- required independent monitors;
- permitted operating mode;
- resource authority scope;
- maximum population/depth class;
- evidence freshness requirements;
- required fallback/containment state.

If the implementation cannot establish that the current state remains within the envelope, dependent replication authority is denied.

## 11. Runtime-assurance boundary

Symthaea may perform simulation, anomaly detection, counterfactual reasoning, safety-case analysis, and operator assistance. Those functions are advisory to the hard RSK authorization boundary.

The trusted containment path must remain able to deny or revoke authority if Symthaea is unavailable, partitioned, compromised, stale, or wrong.

A high-capability reasoner therefore cannot be the only component standing between a fault and unconstrained replication.

## 12. Safety-case binding

Every non-trivial replication grant references the exact safety case on which it depends.

A safety case should be able to represent:

```text
claim
assumptions
evidence
counterevidence
experiment/model provenance
mitigation
residual uncertainty
supersession
expiry/freshness
authorization consequence
```

The intended Mycelix epistemic-DKG integration is evidence-bearing rather than certificate-bearing: a grant depends on explicit claims and assumptions, and invalidating a required assumption invalidates the dependent grant.

RSK itself should depend on a compact verified safety-case snapshot/digest at the runtime boundary rather than requiring the runtime kernel to reason over the entire DKG.

## 13. Decision precedence

Recommended evaluation order:

1. explicit negative containment / quarantine;
2. subject and grant binding;
3. lineage validity;
4. design qualification and safety-case freshness;
5. containment-envelope validity;
6. budget availability;
7. independent monitor health/freshness;
8. required quorum;
9. capability intersection;
10. allow.

Implementations should return all determinable denial reasons for audit while preserving negative-containment precedence.

## 14. Required property tests for Rust integration

The first executable RSK tranche should prove at least:

1. creating a child from an authorized parent gives the child no grant;
2. N children cannot mint or amplify authority;
3. an explicit grant for subject A cannot authorize subject B;
4. zero population/resource/depth/time budget denies;
5. unknown or cyclic lineage denies;
6. containment drift denies an otherwise valid grant;
7. monitor outage/staleness denies and never increases autonomy;
8. safety-case expiry denies;
9. quarantine/revocation dominates an otherwise all-green decision;
10. descendant effective capabilities are a subset of both parent constraints and external grant;
11. replayed grant/budget state cannot restore consumed authority;
12. required external quorum cannot be satisfied solely by the controlled subject/developer role.

Property-based tests should include race/replay permutations and generated lineage trees, not only example unit tests.

## 15. Formal-model obligations

The accompanying abstract model focuses on constitutional properties rather than physical replication.

Minimum invariants:

- `CreationNeverGrantsAuthority`
- `QuarantineDominates`
- `UnknownLineageCannotAuthorize`
- `MonitorFailureCannotAuthorize`
- `BudgetNeverNegative`
- `GrantSubjectBinding`
- `NoSelfMintedGrant`

Later models should cover descendant-tree revocation, concurrent budget consumption, replay, quorum compromise, and reauthorization generations.

## 16. PR sequence

### PR 1 — constitutional core

- this specification;
- abstract formal model;
- no runtime behavior change.

### PR 2 — Rust authority types and pure decision engine

- add RSK module/crate beside Fabrication Kernel;
- pure deterministic `evaluate_replication_authority`;
- no physical replication hooks;
- unit + property tests for the twelve invariants.

### PR 3 — lineage and budget ledger

- append-only descendant-lineage events;
- monotonic budget consumption;
- replay/race resistance;
- lineage-scoped negative containment.

### PR 4 — safety-case adapter

- bind verified compact safety-case snapshots to authority;
- Mycelix epistemic-DKG adapter remains outside the minimal runtime trust core;
- expiry/supersession/counterevidence invalidate dependent grants.

### PR 5 — abstract adversarial simulator

- abstract resources and descendants only;
- generation growth, drift, partitions, stale monitors, compromised grants, evidence expiry, and containment changes;
- no molecular/biological implementation.

### PR 6 — ordinary-fabrication exercise

- validate semantics against printers, virtual robot cells, and self-driving-lab simulations before considering any higher-risk domain.

## 17. Acceptance gate for RSK v0.1

RSK v0.1 is not accepted because a demo succeeds. It is accepted only when:

- the invariants are represented as executable properties;
- the abstract formal model finds no counterexample within its declared bounds;
- negative containment demonstrably dominates positive authority;
- physical creation demonstrably produces no authority;
- monitor/evidence loss fails closed;
- safety claims explicitly state their assumptions and scope;
- no runtime dependency on Symthaea is required to preserve hard replication bounds.

## 18. Foundational statement

RSK should remain reusable across autonomous factories, robots building robots, software-controlled manufacturing, self-driving laboratories, spacecraft industry, engineered systems, and any future domain in which one autonomous system can create another.

Its central constitutional commitment is deliberately simple:

> **The ability to reproduce is never, by itself, a source of power.**
