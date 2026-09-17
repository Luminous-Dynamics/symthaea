# REGEN-033 — Compost / Co-Compost Accounting Boundary v1

Status: preregistration / architecture-only

Parent: REGEN-032 pyrolysis accounting
Program: Luminous-Dynamics/mycelix#940

Parent ProductHead:

```text
0ab0e556efd451df8c3a6d299bf6a96f6ead035e
```

This document creates no compost operating recipe, sanitation claim, maturity claim, pathogen clearance, agronomic suitability, process authority, or physical action.

---

## 1. Purpose

Freeze a deterministic evidence/conservation boundary for compost and co-compost modeling before any predictive or adaptive process intelligence is allowed to participate.

Core theorem:

```text
evidence-bound consumed inputs
+ explicit process-observation lineage
+ explicit material/nutrient accounting
+ identified model assumptions
= bounded compost accounting result
```

not:

```text
= mature compost
= stable compost
= pathogen-safe compost
= contaminant-safe compost
= agronomically suitable amendment
= process authorization
```

---

## 2. Existing decomposition predictor classification

Current Symthaea contains `symthaea-circular::DecompositionPredictor`.

That predictor currently embeds heuristic assumptions including:

- waste-category-specific base half-lives;
- Q10 temperature response around a fixed reference;
- Gaussian moisture response around a fixed optimum;
- a simple oxygen multiplier;
- a fixed C:N response rule.

REGEN-033 does **not** adopt those constants as normative compost science.

Until separately qualified against a frozen empirical protocol, the existing predictor may be used only as an explicitly identified:

```text
HeuristicShadowModel
```

Its output is hypothesis/model evidence only.

In particular:

```text
predicted decomposition percent
!= measured mass loss
!= stabilization
!= maturity
!= sanitation
!= agronomic suitability
```

A future validated replacement does not rewrite old predictions; it creates a new model lineage.

---

## 3. Physical input boundary

A compost model consumes evidence-bound material input records, not allocations or reservations.

```text
material candidate
-> accepted reservation / disposition authority
-> evidence-bound physical consumption or transfer
-> compost process-input record
-> REGEN-033 accounting
```

The initial synthetic campaign may use clearly identified synthetic input fixtures.

No model output may manufacture evidence that material was physically added to a compost batch.

---

## 4. Batch identity boundary

Real compost and co-compost batch identity remains a Mycelix evidence concern.

A Symthaea model run binds exact external batch/process references but does not mint a physical batch merely by simulating one.

The following remain distinct:

```text
input material lot
process run
compost batch
co-compost child batch
model projection
observed batch outcome
```

Blending/co-composting creates a new material subject; it does not inherit every positive property of its parents.

---

## 5. Material ledger

For one exact compatible accounting basis:

```text
sum(consumed inputs)
=
identified compost material
+ identified separated/recovered material
+ measured or derived mass loss
+ leachate / liquid loss where represented
+ gaseous loss where represented
+ unresolved residual
```

Unknown loss is never forced to zero.

No negative stream is valid.

Mixed wet/dry bases require explicit derivation.

Measured total mass loss does not identify its chemical or biological pathway without additional evidence.

---

## 6. Nutrient and carbon ledgers remain separate

Total mass closure does not establish nutrient or carbon closure.

Where evidence supports them, v1 may carry parallel ledgers such as:

```text
C input = C retained + identified C loss + unresolved C
N input = N retained + identified N loss + unresolved N
P input = P retained + identified P loss + unresolved P
K input = K retained + identified K loss + unresolved K
```

But:

```text
total N != plant-available N
retained C != stable soil C
mass loss != CO2 emission
mass loss != decomposition completion
```

Nutrient identity, chemical form, availability, volatilization, leaching and later plant uptake remain separate propositions.

REGEN-034 owns the general nutrient-balance kernel.

---

## 7. Process observations

Process observations may include exact evidence references for quantities such as temperature, moisture, oxygen, mass, pH, or other declared measurements.

REGEN-033 does not prescribe a universal required sensor list or universal target range.

Each process profile declares:

```text
required observation roles
units
spatial / sampling support
temporal support
measurement or derivation requirements
validity / currentness rules
```

A missing process observation remains missing; it does not become an assumed ideal condition.

---

## 8. Time firewall

Elapsed time is not maturity.

```text
N days elapsed
!= decomposition complete
!= biologically stable
!= sanitized
!= ready for soil application
```

A model may use time as one input variable, but any maturity/stability proposition requires an exact adopted evidence/profile contract.

---

## 9. Maturity, stability, sanitation and contamination are distinct

REGEN-033 explicitly separates:

```text
material accounting
process history
maturity / stability evidence
pathogen / sanitation evidence
chemical contamination evidence
physical contaminant evidence
agronomic suitability
```

Passing one does not imply the others.

A process-history profile cannot convert missing sanitation or contaminant evidence into safety.

---

## 10. Scenario versus reconciliation modes

As in REGEN-032, v1 should distinguish:

```text
ScenarioAccounting
EvidenceReconciliation
```

### ScenarioAccounting

Uses declared assumptions / model profiles to explore possible trajectories or balances.

### EvidenceReconciliation

Uses actual process/batch evidence to calculate closure and residuals.

Scenario output is never relabeled as observed process evidence.

---

## 11. Proposed v1 type partition

Names are provisional.

```rust
pub enum CompostAccountingMode {
    Scenario,
    EvidenceReconciliation,
}

pub struct CompostInputRecord {
    pub source_material_ref: ExactRef,
    pub consumption_or_transfer_ref: ExactRef,
    pub quantity: ExactMass,
    pub basis_ref: ExactRef,
    pub evidence_ref: ExactRef,
}

pub struct CompostProcessScope {
    pub process_ref: ExactRef,
    pub batch_ref: ExactRef,
    pub evidence_snapshot_ref: ExactRef,
    pub accounting_profile_ref: ExactRef,
    pub model_ref: Option<ExactRef>,
}

pub struct CompostMaterialLedger {
    pub input_total: ExactMass,
    pub retained_material: ExactMass,
    pub identified_losses: Vec<MaterialStream>,
    pub unresolved_residual: ExactMass,
}

pub struct CompostAccountingResult {
    pub mode: CompostAccountingMode,
    pub scope: CompostProcessScope,
    pub material: CompostMaterialLedger,
    pub nutrient_ledgers: Vec<NutrientLedger>,
    pub process_observation_refs: Vec<ExactRef>,
    pub assumptions: Vec<ExactRef>,
}
```

No positive maturity/safety type is minted by this accounting result.

---

## 12. Co-composting firewall

Combining biochar and compost creates a new modeled/material subject.

```text
qualified biochar parent
+ qualified compost parent
!= qualified co-compost child
```

The child requires its own identity, formation/process lineage, evidence snapshot and later suitability assessment.

Parent concentrations, contamination states, quality classes or agronomic claims are not inherited by max/union.

The initial model may calculate mixture arithmetic only when units/bases are compatible and constituent quantities are explicit.

---

## 13. No universal recipe

REGEN-033 does not encode universal recommendations for:

- C:N ratio;
- moisture;
- oxygen;
- temperature;
- turning frequency;
- duration;
- inoculation;
- pile geometry;
- feedstock proportions.

Such values may appear in a versioned experimental/process profile, but profile identity and evidence basis must remain visible.

A heuristic predictor's hard-coded values do not become the REGEN constitutional defaults.

---

## 14. First executable target

The first executable candidate should be smaller than the existing heuristic decomposition predictor.

Recommended first theorem:

```text
exact evidence-bound input masses
+ exact identified retained/output mass
+ identified measured/derived losses
= material closure + explicit unresolved residual
```

Then add, separately:

1. multi-input blend/co-compost arithmetic;
2. nutrient sub-ledgers;
3. process-observation bindings;
4. scenario-vs-evidence typing;
5. shadow-model comparison against the existing decomposition predictor;
6. independent conservation oracle.

Do not make the heuristic decomposition predictor load-bearing for the first ProductFrozen theorem.

---

## 15. Minimum qualification propositions

At minimum:

1. exact input identities retained;
2. allocation/reservation alone rejected as physical input;
3. same-basis material closure correct;
4. mixed mass bases rejected;
5. overallocated output rejected;
6. unresolved residual explicit;
7. missing loss does not become zero;
8. scenario assumptions remain assumptions;
9. evidence reconciliation remains evidence-bound;
10. predicted decomposition does not create measured mass loss;
11. elapsed time does not create maturity;
12. maturity does not create sanitation;
13. sanitation does not create contamination clearance;
14. chemical clearance does not create agronomic suitability;
15. child blend does not inherit strongest parent state;
16. nutrient totals remain separate from nutrient availability;
17. deterministic arithmetic fixture replay stable;
18. independent conservation oracle parity;
19. malformed batch/input substitution rejected;
20. model profile identity retained;
21. heuristic shadow model cannot create authority;
22. result contains no process-control command;
23. result cannot mint real batch observation evidence;
24. unknown/not-assessed remains distinct from zero/pass.

---

## 16. Relationship to downstream work

```text
Mycelix material / residue eligibility
      -> evidence-bound process input
      -> REGEN-033 compost accounting
      -> Mycelix REGEN-013 / 014 batch lineage
      -> REGEN-016 contamination evidence
      -> REGEN-017 agronomic suitability
      -> REGEN-015 / 050+ field-trial intelligence
```

Modeling never bypasses physical evidence or authoritative hard gates.

---

## 17. Deliberate non-claims

REGEN-033 establishes no:

- validated decomposition kinetics;
- universal compost recipe;
- process completion;
- maturity or stability;
- pathogen destruction;
- contaminant safety;
- fertilizer equivalence;
- agronomic suitability;
- carbon-removal outcome;
- process execution authority;
- physical actuation.

It freezes only the accounting/evidence semantics required before compost models can become scientifically load-bearing.