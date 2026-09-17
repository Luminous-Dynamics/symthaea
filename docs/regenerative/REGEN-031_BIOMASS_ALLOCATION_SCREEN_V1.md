# REGEN-031 — Biomass Allocation Screen v1

Status: preregistration / architecture-only

Parent: REGEN-030 terrestrial driver projection boundary
Program: Luminous-Dynamics/mycelix#940

Parent ProductHead:

```text
9dc582e4f6e390860015de20456346a913a0e0a2
```

This document creates no feedstock eligibility, property right, ecological truth, optimization mandate, reservation, consumption record, process command, or physical action.

---

## 1. Purpose

Freeze the first deterministic Symthaea screening layer for biomass allocation without duplicating Mycelix's evidence, rights, contamination, ecological-retention, reservation, or execution authorities.

The central theorem is:

```text
qualified upstream hard-gate state
+ exact material/accounting scope
+ explicit competing-use candidates
+ explicit hard allocation constraints
+ deterministic screening rules
= bounded model-screen result
```

not:

```text
= new feedstock eligibility
= ownership / removal right
= contamination clearance
= ecological permission
= accepted reservation
= physical consumption
= optimal social allocation
= authority
```

---

## 2. Upstream ownership boundary

Mycelix remains authoritative for the regenerative evidence propositions it owns or composes, including the REGEN biomass chain:

```text
resource occurrence
material quantity / basis
state snapshot
material partition
ecological-function eligibility
rights / custody references
quality / contamination propositions
process-scoped feedstock assessment
reservation acceptance
consumption accounting
```

Symthaea REGEN-031 MUST NOT silently reconstruct these propositions from raw values.

If an upstream proposition is `Unresolved`, the model screen cannot upgrade it to positive eligibility merely because a simulated allocation looks attractive.

If an upstream hard gate is `Ineligible` or violated, no soft objective may restore that candidate.

---

## 3. Screened state versus authoritative state

REGEN-031 output is recommendation/modeling state only.

Provisional v1 outcome vocabulary:

```rust
pub enum BiomassScreenOutcome {
    PassesDeclaredScreen(BiomassScreenResult),
    FailsDeclaredScreen(ScreenFailure),
    Unresolved(ScreenUnresolved),
}
```

The phrase `PassesDeclaredScreen` is intentional. It MUST NOT be shortened in external interfaces to a generic `Eligible`, `Approved`, `Safe`, or `Available` flag.

A positive screen result means only:

> Under the exact declared inputs, assumptions, and constraints, this candidate survives this deterministic model screen.

---

## 4. Input identity

Every screen attempt must bind the exact upstream state used.

At minimum:

```text
screen_ref
source_evidence_snapshot_ref
source_biomass_lot_ref
source_state_snapshot_ref
source_feedstock_assessment_ref
source_material_partition_ref
source_ecology_profile_ref
source_process_profile_ref
mass_basis
screen_profile_ref
```

If later execution consumes ProductFrozen Mycelix biomass types, the exact ProductFrozen dependency identity must also be bound.

The initial executable REGEN-031 MUST wait for the REGEN-011D1 -> REGEN-011E biomass type-state/ProductFrozen path. It must not target the pre-D1 reservation API.

---

## 5. Hard-gate inheritance

The screen receives hard-gate outcomes; it does not invent them.

Examples:

```text
feedstock assessment unresolved
    -> screen unresolved / excluded

feedstock assessment ineligible
    -> screen fails before optimization

ecological hard constraint violated
    -> screen fails before optimization

rights/custody unresolved
    -> screen unresolved / excluded

contamination proposition unresolved where required by target use
    -> screen unresolved / excluded
```

No numeric benefit estimate may override these states.

---

## 6. Competing uses are first-class

A regenerative biomass system must not assume that material is "waste" simply because one candidate process can consume it.

Candidate uses may include, for example:

```text
retain in place
mulch / soil-cover function
habitat / ecological function
animal bedding or other existing use
compost pathway
pyrolysis pathway
material / manufacturing pathway
energy-service pathway
other declared local use
unresolved / reserve
```

The candidate-use vocabulary is profile-bound rather than universal.

A use being modeled does not establish that the use is safe, legal, socially preferred, or physically available.

---

## 7. Explicit material closure

The deterministic screen must preserve exact material accounting.

For one exact mass basis:

```text
screenable quantity
=
sum(candidate allocations)
+ explicit unallocated reserve
+ explicit unresolved residual
```

The screen MUST NOT:

- erase residual material;
- mix as-received and dry-matter quantities without a qualified conversion;
- allocate the same mass twice;
- use gross occurrence as the allocation denominator when upstream state exposes a narrower qualified quantity;
- treat a reservation or modeled allocation as physical consumption.

---

## 8. Allocation request versus accepted model allocation

The first executable API should use type-state semantics similar to the biomass reservation correction.

```text
AllocationCandidate
      |
      v
hard-gate + scope + mass + conflict evaluation
      |
      v
ScreenedAllocationSet
```

A raw candidate is not a screened allocation.

A failed candidate set returns no partial `ScreenedAllocationSet` unless the screen profile explicitly defines independent per-candidate evaluation and reports failures separately.

The default v1 theorem should prefer atomic whole-set validation when a shared finite material pool is being allocated.

---

## 9. Hard constraints before soft dimensions

The ordering is normative:

```text
identity / evidence validity
-> upstream hard-gate state
-> mass-basis compatibility
-> finite material capacity
-> ecological / reserved floors inherited from upstream profiles
-> competing-use incompatibilities
-> declared service constraints
-> only then soft comparison
```

Soft dimensions may include:

```text
expected useful output
expected information value
cost
local dependency reduction
energy-service opportunity
carbon-related model outputs
labor burden
repair burden
transport burden
```

These dimensions MUST remain plural in v1.

There is no universal weighted `biomass_value_score` or `sustainability_score`.

---

## 10. Locality firewall

Local use is not automatically preferred.

```text
local use
!= lower ecological burden
!= safer use
!= more resilient use
!= socially preferred use
!= better opportunity cost
```

A local pathway may contribute to resilience when it closes a consequential dependency or creates independent optionality, but that proposition belongs to the Phase-E service/dependency analysis rather than being baked into the biomass screen.

---

## 11. Carbon firewall

Carbon-related outputs remain separate from biomass allocation legitimacy.

```text
modeled carbon retention
!= carbon removal
!= credit eligibility
!= ecological permission
!= agronomic suitability
```

A candidate with a favorable modeled carbon dimension cannot bypass a failed ecological, rights, contamination, material, or service constraint.

Climate/carbon authority remains outside REGEN-031.

---

## 12. Ecology-model firewall

Symthaea may later use `symthaea-earth-system` and `symthaea-ecology` projections to explore consequences of different retention/allocation scenarios.

Those projections are model evidence only.

They MUST NOT replace the upstream adopted ecological-obligation profile.

In particular:

```text
model predicts low ecological impact
!= hard ecological obligation satisfied

model predicts high productivity
!= biomass removable

population/ecology baseline stable in simulation
!= real habitat obligation satisfied
```

The screen may narrow candidates in response to model risk or uncertainty. It may not widen upstream ecological permission.

---

## 13. Uncertainty and unresolved state

`Unresolved` is a first-class result.

Examples include:

- missing required upstream hard-gate evidence;
- incompatible or missing mass basis;
- unresolved competing-use reservation;
- unknown material remainder;
- insufficient ecological-service evidence;
- required quality/contamination proposition unavailable;
- projection input outside a declared model validity domain.

The screen MUST NOT translate unresolved input into a conservative-looking numeric penalty and then proceed as if the candidate were eligible.

---

## 14. Proposed v1 type surface

Names remain provisional until executable implementation review.

```rust
pub struct BiomassAllocationCandidate {
    pub candidate_ref: ExactRef,
    pub use_profile_ref: ExactRef,
    pub requested_quantity: ExactMass,
    pub prerequisite_refs: Vec<ExactRef>,
}

pub struct BiomassScreenScope {
    pub lot_ref: ExactRef,
    pub state_snapshot_ref: ExactRef,
    pub feedstock_assessment_ref: ExactRef,
    pub material_partition_ref: ExactRef,
    pub evidence_snapshot_ref: ExactRef,
    pub mass_basis_ref: ExactRef,
    pub screen_profile_ref: ExactRef,
}

pub struct ScreenedAllocation {
    pub candidate_ref: ExactRef,
    pub screened_quantity: ExactMass,
    pub retained_constraint_refs: Vec<ExactRef>,
}

pub struct BiomassScreenResult {
    pub scope: BiomassScreenScope,
    pub allocations: Vec<ScreenedAllocation>,
    pub unallocated_reserve: ExactMass,
    pub unresolved_residual: ExactMass,
}
```

No public constructor should be available for positive screened results if that would allow callers to bypass the evaluator.

---

## 15. First executable target

The first REGEN-031 candidate should be a dependency-light pure function over synthetic fixtures.

It should NOT initially depend on:

- HDC/LTC/LLM inference;
- Holochain runtime;
- marketplace or finance runtime;
- actuator/device interfaces;
- online data fetch;
- adaptive optimization;
- physical process control.

Recommended evaluator shape:

```rust
fn screen_biomass_allocation(
    scope: &QualifiedBiomassScreenScope,
    candidates: &[BiomassAllocationCandidate],
    constraints: &AllocationConstraintProfile,
) -> Result<BiomassScreenOutcome, ScreenError>;
```

---

## 16. Minimum qualification propositions

The first executable campaign should cover at least:

1. exact input identity retained;
2. positive upstream hard-gate state can enter the screen;
3. upstream `Ineligible` cannot be overridden;
4. upstream `Unresolved` remains unresolved;
5. mixed mass bases rejected;
6. zero/overflow arithmetic handled explicitly;
7. candidate set cannot overallocate finite quantity;
8. same mass cannot be allocated to two mutually exclusive uses;
9. retained/unallocated floor remains explicit;
10. unresolved residual remains explicit;
11. competing-use reservation is not silently discarded;
12. hard constraints execute before soft comparison;
13. favorable cost cannot override a hard gate;
14. favorable locality cannot override a hard gate;
15. favorable carbon output cannot override a hard gate;
16. favorable model confidence cannot override a hard gate;
17. model projection cannot manufacture ecological permission;
18. screen result does not create Mycelix reservation acceptance;
19. screen result does not create physical consumption;
20. screen result carries no command/actuator authority;
21. deterministic fixture replay is stable;
22. malformed identity substitution fails closed;
23. missing required proposition does not become zero/default;
24. null/no-use / retain-in-place remains a valid candidate when profile allows it.

---

## 17. Relationship to REGEN-032

Pyrolysis accounting may consume a biomass allocation proposal only after the separate Mycelix consumption path establishes actual process input.

The intended chain is:

```text
Mycelix ProductFrozen feedstock state
      -> Symthaea REGEN-031 model screen
      -> recommendation / candidate allocation
      -> independent adoption / reservation authority
      -> AcceptedReservation
      -> evidence-bound physical consumption accounting
      -> REGEN-032 pyrolysis accounting input
```

The screen itself MUST NOT jump directly to REGEN-032 as if modeled allocation were consumed feedstock.

---

## 18. Deliberate non-claims

REGEN-031 establishes no:

- feedstock ownership or custody;
- ecological permission;
- contamination safety;
- process suitability beyond exact upstream propositions;
- optimal community allocation;
- universal waste hierarchy;
- carbon-removal claim;
- local-resilience superiority;
- reservation acceptance;
- material consumption;
- process execution;
- physical actuation.

It freezes only a deterministic recommendation-level allocation screen that can narrow already-qualified options without widening authority or evidence.