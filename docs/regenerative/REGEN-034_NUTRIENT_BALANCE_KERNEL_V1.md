# REGEN-034 — Nutrient Balance Kernel v1

Status: preregistration / architecture-only

Parent: REGEN-033 compost accounting
Program: Luminous-Dynamics/mycelix#940

Parent ProductHead:

```text
efa30c85872c2b3c2d251c0580b0c41d031f76c9
```

Existing exact Symthaea nutrient-oracle anchor on the current REGEN-030 base:

```text
crates/domains/symthaea-earth-system/src/nutrient.rs
beddb69b35f784184d689d42488255ee6b4c3d1d
```

This document creates no fertilizer recommendation, plant-availability claim, crop-uptake observation, discharge permit, agronomic suitability, or physical action.

---

## 1. Purpose

Freeze the first deterministic nutrient-balance kernel around Symthaea's existing exact two-pool conservation model while preserving Mycelix evidence identity and refusing to turn model pools into observed or agronomic truth.

Core theorem:

```text
identified nutrient + compatible units
+ exact initial modeled stocks
+ explicit input / deposition assumptions or evidence
+ identified turnover / loss parameters
+ deterministic two-pool model
= bounded nutrient-state projection + exact budget residual
```

not:

```text
= measured nutrient state
= plant-available nutrient
= actual crop uptake
= fertilizer equivalence
= environmental safety
= agronomic recommendation
```

---

## 2. Existing oracle to reuse

`symthaea-earth-system::TwoPoolNutrientCycle` already provides:

- an organic nutrient pool;
- a mineral nutrient pool;
- mineralization;
- externally supplied deposition/input;
- modeled biological uptake;
- modeled leaching;
- exact constant-input state solutions;
- exact cumulative uptake/leaching integrals;
- explicit nutrient-budget residual;
- finite/non-negative domain guards.

REGEN-034 SHOULD reuse this baseline rather than write a second nutrient turnover model.

The existing model is intentionally generic: the caller decides whether the conserved quantity represents N, P, or another nutrient and supplies compatible units.

---

## 3. Model-state / evidence firewall

The following are distinct propositions:

```text
observed total nutrient
observed chemical form
modeled organic pool
modeled mineral pool
measured extractable / available fraction
modeled uptake flux
measured plant uptake
leaching model output
measured environmental loss
```

REGEN-034 MUST NOT collapse them.

In particular:

```text
model mineral_pool
!= measured plant-available nutrient

model uptake_flux
!= measured crop uptake

model leaching_flux
!= observed groundwater / runoff loss
```

A model state can support a prediction or comparison, not rewrite observational evidence.

---

## 4. Nutrient identity is explicit

No unlabeled generic scalar enters the kernel.

Each invocation must bind an exact nutrient identity/profile, for example:

```text
N-total
P-total
K-total
other declared conserved element / quantity
```

The identity also binds the unit/basis and conversion rules used by that invocation.

The v1 kernel MUST NOT assume that the same turnover parameters or pools are physically meaningful for every nutrient simply because the equations are mathematically reusable.

---

## 5. Unit and basis discipline

Inputs must use one explicit compatible quantity basis per ledger.

Examples of dimensions/bases that cannot be silently mixed include:

```text
mass
mass / area
concentration
moles
mass / dry-matter mass
mass / wet mass
stock per modeled cell
flux per time
```

Any conversion requires an exact identified rule and source support.

A concentration does not become a stock without the required support volume/mass/area relation.

A stock does not become a flux without a declared time basis or process model.

---

## 6. Evidence and scenario inputs remain distinct

REGEN-034 should support at least two modes:

```text
ScenarioProjection
EvidenceConditionedProjection
```

### ScenarioProjection

Initial stocks, rates, inputs, deposition or losses may be caller-declared assumptions.

### EvidenceConditionedProjection

One or more values are explicitly derived from qualified evidence bindings and conversion lineage.

In both modes, every non-observed parameter/assumption remains identified.

A scenario input never gains observational status because it appears in a successful conservation ledger.

---

## 7. Conservation theorem

For the existing two-pool model:

```text
final_total - initial_total
= cumulative_organic_input
+ cumulative_deposition
- cumulative_modeled_uptake
- cumulative_modeled_leaching
+ budget_residual
```

The first REGEN-034 executable theorem should preserve this residual as an explicit numerical oracle.

No balancing term may be silently adjusted to force closure.

If a later extension adds other sources/sinks, the accounting identity itself changes and requires a new profile/version.

---

## 8. Availability firewall

REGEN-020 in Mycelix already freezes the principle:

```text
total nutrient presence != plant availability
```

REGEN-034 inherits that rule.

No projection may map `mineral_pool` directly to a universal agronomic `available` field.

Any later availability proposition requires an explicit adopted model/profile that identifies:

```text
nutrient identity
chemical form / extraction interpretation
soil/media context
spatial support
time support
method/model identity
uncertainty
validity domain
```

---

## 9. Uptake firewall

The existing `uptake_rate * mineral_pool` term is a model sink.

It is not a universal crop-uptake model.

REGEN-034 therefore distinguishes:

```text
ModeledNutrientSink::BiologicalUptake
```

from any future observed or crop-specific uptake evidence.

A crop/nutrition projection in REGEN-038 may consume the modeled sink only through an explicit adapter whose assumptions are visible.

---

## 10. Loss firewall

Similarly, modeled leaching is a ledger sink, not environmental discharge evidence.

```text
model leaching
!= measured leachate
!= groundwater contamination
!= watershed load
!= legal discharge violation
```

Those propositions require separate hydrological/spatial/evidence bridges.

A favorable low modeled leaching term cannot create environmental permission.

---

## 11. Recovered nutrient / circularity firewall

Recovered nutrients can enter the model only as identified input streams.

```text
nutrient recovered
!= nutrient available at target site
!= nutrient safe
!= nutrient legally usable
!= nutrient applied
!= nutrient taken up
```

REGEN-034 cannot double-count one recovered stock in multiple simulated destinations unless the scenario explicitly models mutually exclusive alternatives.

Actual allocation/reservation remains outside this kernel.

---

## 12. Compost / amendment bridge

REGEN-033 may produce evidence/model references describing nutrient stocks retained in a compost/co-compost material.

REGEN-034 may project nutrient turnover only after an explicit bridge declares how those material stocks map into the model's organic/mineral initial conditions or inputs.

That bridge is not automatic.

```text
compost total N
!= soil organic-pool N
!= mineral-pool N
```

The same applies to biochar-associated nutrients or other amendments.

---

## 13. Proposed v1 type partition

Names are provisional.

```rust
pub struct NutrientIdentity {
    pub nutrient_ref: ExactRef,
    pub unit_basis_ref: ExactRef,
}

pub struct NutrientProjectionScope {
    pub subject_ref: ExactRef,
    pub evidence_snapshot_ref: ExactRef,
    pub model_ref: ExactRef,
    pub parameter_set_ref: ExactRef,
    pub nutrient: NutrientIdentity,
    pub time_basis_ref: ExactRef,
}

pub enum NutrientInputOrigin {
    Evidence(ExactRef),
    Derived(ExactRef),
    ScenarioAssumption(ExactRef),
}

pub struct NutrientStateProjection {
    pub organic_pool: Quantity,
    pub mineral_pool: Quantity,
    pub model_uptake: Quantity,
    pub model_leaching: Quantity,
    pub budget_residual: Quantity,
}
```

Model state/output names must retain `model` / `projection` semantics where needed to prevent observational overclaiming.

---

## 14. Parameter identity

Turnover/loss parameters must never be anonymous defaults in downstream evidence.

Each result binds:

```text
mineralization_rate
uptake_rate
leaching_rate
parameter_set_ref
parameter_source / assumption identity
validity domain where known
```

The built-in mathematical model may have example parameters in tests or callers, but REGEN-034 should not establish one universal N/P/K parameter set.

---

## 15. Exact solution as independent oracle

The existing exact constant-input solution is a major qualification advantage.

The first implementation should use it to independently test any stepped/integrated execution path.

Candidate properties:

```text
state(t=0) = initial_state
budget residual ~ 0 within declared floating tolerance
non-negative state preserved for validated inputs
numerical trajectory converges toward exact solution
identical exact inputs -> deterministic exact result
```

Floating tolerances must be scale-aware and declared rather than selected after observing failure.

---

## 16. Uncertainty

Rates, initial states, deposition, inputs, conversion rules and model structure can all be uncertain.

V1 should preserve deterministic alternatives or bounded ensembles rather than collapsing them prematurely.

```text
central parameter set
!= true parameter set

ensemble member count
!= probability
```

Parameter sensitivity should remain visible in downstream trial design.

---

## 17. First executable target

Recommended first implementation:

```text
frozen synthetic nutrient identity
+ exact initial organic/mineral stocks
+ constant organic input
+ constant deposition
+ exact parameter set
-> TwoPoolNutrientCycle::exact_sample
-> identity-bearing projection receipt
-> independent budget oracle assertion
```

No HDC/LTC/LLM inference belongs in this first theorem.

No adaptive fertilizer recommendation belongs in this first theorem.

---

## 18. Minimum qualification propositions

At least:

1. nutrient identity retained;
2. unit/basis identity retained;
3. model/parameter-set identity retained;
4. t=0 exact identity;
5. exact solution replay deterministic;
6. exact conservation residual within frozen tolerance;
7. negative/non-finite state rejected;
8. negative invalid rates rejected;
9. incompatible unit/basis conversion rejected;
10. concentration cannot silently become stock;
11. scenario assumption remains tagged;
12. derived evidence retains derivation lineage;
13. model mineral pool does not become observed availability;
14. modeled uptake does not become measured crop uptake;
15. modeled leaching does not become observed environmental loss;
16. recovered nutrient cannot be allocated twice in one mutually exclusive scenario;
17. missing data does not become zero/default;
18. nutrient identity substitution changes lineage / fails exact expectation;
19. alternate parameter sets remain separate projections;
20. model output cannot mint application authority;
21. favorable nutrient projection cannot override contamination/ecology/rights gates;
22. output contains no actuator/application command;
23. numerical and exact solutions agree under the declared convergence campaign;
24. unknown/not-evaluable remains distinct from zero.

---

## 19. Relationship to later REGEN models

```text
REGEN-020 Mycelix nutrient stock/flow evidence
       |
       v
REGEN-030 projection boundary
       |
       v
REGEN-034 nutrient balance kernel
      / \
     v   v
REGEN-035 soil response   REGEN-038 crop/nutrition service
     |
     v
field-trial prediction vs observation
```

The downstream models consume model state as model state; they do not retroactively turn it into observation.

---

## 20. Deliberate non-claims

REGEN-034 establishes no:

- calibrated N, P or K soil model;
- universal mineralization rate;
- plant-available nutrient measurement;
- fertilizer equivalence;
- crop nutrient demand;
- measured plant uptake;
- watershed loss or contamination;
- nutrient-application recommendation;
- legal/agronomic suitability;
- process or actuator authority.

It freezes only the identity-preserving conservation kernel required for later terrestrial nutrient projections.