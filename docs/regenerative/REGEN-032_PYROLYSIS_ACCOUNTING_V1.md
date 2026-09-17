# REGEN-032 — Pyrolysis Accounting Boundary v1

Status: preregistration / architecture-only

Parent: REGEN-031 biomass allocation screen
Program: Luminous-Dynamics/mycelix#940

Parent ProductHead:

```text
9674b137e6c3d5b2c7cc7e8756b4f17908aacca2
```

This document defines a deterministic accounting boundary only. It is not a pyrolysis operating procedure, reactor model, safety case, emissions permit, biochar certification, carbon-removal methodology, or process-control system.

---

## 1. Purpose

Freeze the first Symthaea pyrolysis model as an inspectable conservation ledger rather than a premature reaction-kinetics or equipment-control model.

The core theorem is:

```text
evidence-bound consumed biomass input
+ explicit transformation assumptions or measured process evidence
+ explicit material/energy accounting rules
= bounded pyrolysis accounting result
```

not:

```text
= validated pyrolysis chemistry
= safe reactor settings
= process execution
= qualified biochar
= agronomic suitability
= carbon removal
= carbon credit
```

---

## 2. Existing substrate and reuse boundary

Symthaea already contains useful but deliberately bounded scientific/engineering primitives.

### `symthaea-thermofluids`

Current scope includes textbook fluid/thermal relations such as:

- Carnot efficiency;
- Fourier conduction;
- Newton cooling;
- engine work;
- continuity / Bernoulli / Reynolds / Darcy-Weisbach relations.

REGEN-032 MAY reuse those relations when the exact assumptions and validity domain are declared.

It MUST NOT infer from their existence that Symthaea has a validated pyrolysis reactor model.

### `symthaea-organic-chemistry`

Current scope is structural organic chemistry / cheminformatics: SMILES, molecular formula, mass composition, groups and simple structural properties.

It explicitly does not provide reaction mechanisms.

REGEN-032 therefore MUST NOT pretend that structural molecule utilities establish biomass thermal-decomposition kinetics, product distributions, tar chemistry, reaction pathways, or emissions.

---

## 3. Physical-evidence handoff

REGEN-032 must not treat a modeled allocation or accepted reservation as process input already consumed.

The required physical-evidence chain is:

```text
qualified feedstock state
-> model allocation / recommendation
-> independent adoption / reservation authority
-> AcceptedReservation
-> evidence-bound biomass consumption accounting
-> exact process-input record
-> REGEN-032 accounting
```

The Mycelix REGEN-011C consumption theorem therefore precedes executable use of REGEN-032 for real batch evidence.

For synthetic campaigns, clearly identified synthetic consumed-input fixtures are allowed.

---

## 4. Two explicit operating modes

The first model should distinguish two kinds of accounting use.

### 4.1 Forward scenario accounting

Inputs include declared transformation assumptions such as output fractions, moisture handling, external heat requirement, auxiliary energy, or recovery fractions.

Outputs are projections/scenarios, not observations.

### 4.2 Evidence reconciliation

Inputs include observed or derived evidence for consumed feedstock and one or more measured output/coproduct streams.

The model computes explicit budget closure and unresolved residual.

A reconciliation result does not magically identify the chemistry of an unexplained residual.

These two modes MUST NOT share an unlabeled `PyrolysisResult` that hides whether values were assumed or measured.

---

## 5. Material ledger

At one exact compatible material basis, the accounting identity is:

```text
consumed input
=
solid char output
+ condensable / liquid output
+ gaseous output
+ recovered other material
+ measured / declared process loss
+ unresolved residual
```

Not every implementation must distinguish every coproduct immediately, but any omitted fraction MUST remain visible as unresolved residual rather than being forced into char, gas, or zero.

No negative stream is valid.

No stream may exceed the available consumed input.

Exact arithmetic or an explicitly bounded residual tolerance must be used; tolerance itself is profile-bound.

---

## 6. Basis firewall

The model MUST NOT silently mix:

```text
as-received mass
dry-matter mass
water mass
ash / mineral fraction
carbon mass
total organic matter mass
```

Conversions require explicit evidence or declared scenario assumptions.

For example:

```text
wet biomass mass
!= dry biomass mass

char mass
!= carbon mass

carbon retained in char
!= atmospheric CO2 removed
```

A dry-matter conversion may be used only when its moisture evidence / derivation lineage is identified.

---

## 7. Carbon ledger is separate from total-mass ledger

If carbon accounting is performed, it is a second explicit conservation surface.

```text
input carbon
=
char carbon
+ liquid carbon
+ gas carbon
+ other identified carbon
+ unresolved carbon residual
```

The model MUST NOT infer carbon fractions from total mass unless a declared composition assumption/evidence source provides them.

Likewise:

```text
char carbon retained in modeled output
!= durable carbon removal
!= net greenhouse-gas benefit
!= creditable carbon removal
```

Those require separate lifecycle, durability, counterfactual, leakage, emissions, and methodology propositions owned by later REGEN / Climate layers.

---

## 8. Energy ledger

Energy accounting remains distinct from material accounting.

A first v1 ledger may include:

```text
external energy input
feedstock sensible/latent terms if explicitly modeled
process heat requirement assumption
recovered heat
chemical-energy coproduct estimate if explicitly supported
auxiliary electricity/fuel
unrecovered / unresolved energy
```

The following distinctions are mandatory:

```text
heat produced
!= recoverable heat
!= useful heat captured
!= useful heat delivered to a compatible load
```

Thermofluid primitives can be used only when their assumptions are applicable and recorded.

REGEN-037 later owns the heat/resource-quality service bridge.

---

## 9. No universal yield model in v1

REGEN-032 MUST NOT hard-code a universal mapping such as:

```text
temperature + time -> fixed char yield
```

without a separately reviewed and validated model.

Real product distribution can depend on feedstock, moisture, particle geometry, heating rate, residence conditions, equipment, pressure, atmosphere and other factors.

V1 therefore permits:

- exact measured output evidence;
- declared scenario fractions;
- bounded empirical profiles whose identity and validity domain are explicit;
- uncertainty envelopes / alternative scenarios.

It does not silently generalize them beyond their declared domain.

---

## 10. No operating recipe or actuator boundary

REGEN-032 contains no normative reactor operating instructions.

It MUST NOT produce:

- ignition commands;
- temperature setpoints for real equipment;
- valve, blower, fuel, auger or pump commands;
- emergency-shutdown logic;
- operator bypasses;
- claims of safe residence time / temperature;
- device control messages.

A future physical pyrolysis system would require a separate device/process safety and authority theorem outside this accounting core.

---

## 11. Proposed v1 type partition

Names are provisional, semantics are not.

```rust
pub enum PyrolysisAccountingMode {
    Scenario,
    EvidenceReconciliation,
}

pub struct PyrolysisInputScope {
    pub process_run_ref: ExactRef,
    pub consumption_record_ref: ExactRef,
    pub feedstock_lot_ref: ExactRef,
    pub input_basis_ref: ExactRef,
    pub evidence_snapshot_ref: ExactRef,
    pub accounting_profile_ref: ExactRef,
}

pub struct MaterialStream {
    pub stream_ref: ExactRef,
    pub stream_class: MaterialStreamClass,
    pub quantity: ExactMass,
    pub evidence_or_assumption_ref: ExactRef,
}

pub struct PyrolysisMaterialLedger {
    pub input: ExactMass,
    pub outputs: Vec<MaterialStream>,
    pub unresolved_residual: ExactMass,
}

pub struct PyrolysisEnergyLedger {
    pub energy_inputs: Vec<EnergyTerm>,
    pub useful_outputs: Vec<EnergyTerm>,
    pub rejected_or_unrecovered: Vec<EnergyTerm>,
    pub unresolved: Option<EnergyTerm>,
}

pub struct PyrolysisAccountingResult {
    pub mode: PyrolysisAccountingMode,
    pub scope: PyrolysisInputScope,
    pub material: PyrolysisMaterialLedger,
    pub carbon: Option<CarbonLedger>,
    pub energy: Option<PyrolysisEnergyLedger>,
    pub model_profile_ref: ExactRef,
}
```

Scenario and evidence terms should be separately typed or tagged strongly enough that downstream consumers cannot mistake assumptions for measurements.

---

## 12. Output identity and Mycelix batch lineage

A Symthaea accounting result does not mint a real `BiocharBatchId` by itself.

Mycelix REGEN-012 owns real biochar batch/process evidence lineage.

The clean relation is:

```text
Symthaea projection/accounting output
    -> model prediction / reconciliation evidence

physical process + observed outputs
    -> Mycelix batch lineage
```

Prediction and observed batch identity remain separate until explicitly compared.

---

## 13. Unknown coproduct firewall

Unknown output is first-class.

The model MUST reject these shortcuts:

```text
unmeasured gas = zero
unmeasured condensate = zero
unknown process loss = zero
unobserved emission = zero
missing ash/mineral accounting = zero
```

A closed ledger may include an `unresolved_residual`; it must not claim chemical identity for that residual without evidence.

---

## 14. Uncertainty

Uncertainty can enter through feedstock quantity, composition, moisture, declared yields, measured outputs, heat requirements, conversion efficiencies or other inputs.

V1 may use:

- bounded low/central/high scenarios;
- exact deterministic alternatives;
- explicitly identified ensemble members.

It MUST NOT treat an uncalibrated ensemble as a probability distribution.

A single central ledger does not erase uncertainty branches.

---

## 15. First executable target

The first executable REGEN-032 should be a pure deterministic accounting crate or module over synthetic fixtures.

Recommended first theorem:

```text
one exact dry-basis consumed-input quantity
+ caller-declared non-negative output partition
= exact material closure with explicit residual
```

Then add, in separate increments:

1. evidence-vs-assumption typed inputs;
2. carbon sub-ledger;
3. simple energy sub-ledger;
4. independent conservation oracle;
5. malformed / overallocated stream attacks.

Do not begin with a reactor simulator.

---

## 16. Minimum qualification propositions

The first executable campaign should cover at least:

1. exact consumed-input identity retained;
2. accepted reservation without consumption evidence is rejected as physical input;
3. same-basis material streams close exactly;
4. output over-allocation is rejected;
5. negative/non-finite quantity is rejected where applicable;
6. mixed mass basis is rejected;
7. wet-to-dry conversion requires explicit derivation;
8. unknown residual stays explicit;
9. scenario term remains identifiable as assumption;
10. observed term remains identifiable as evidence;
11. scenario result cannot masquerade as measured batch outcome;
12. char mass cannot masquerade as carbon mass;
13. char carbon cannot create a carbon-removal claim;
14. material closure does not imply energy closure;
15. recovered heat does not imply useful delivered heat;
16. favorable modeled yield does not create process suitability;
17. no output mints a Mycelix BiocharBatchId;
18. no output grants process execution authority;
19. no output contains actuator commands;
20. malformed identity substitution fails closed;
21. deterministic fixture replay is stable;
22. independent arithmetic oracle agrees with material residual;
23. uncertainty alternatives remain separately identified;
24. missing coproduct measurement does not become zero.

---

## 17. Relationship to later REGEN models

```text
REGEN-031 qualified model allocation
        |
        v
independent adoption + accepted reservation
        |
        v
Mycelix REGEN-011C consumption evidence
        |
        v
REGEN-032 pyrolysis accounting
       / \
      v   v
char projection   heat projection
      |             |
      v             v
Mycelix REGEN-012   REGEN-037 heat/resource-quality bridge
batch evidence
```

REGEN-032 does not own either downstream authority.

---

## 18. Deliberate non-claims

REGEN-032 establishes no:

- validated pyrolysis reaction mechanism;
- universal product yield;
- safe operating temperature, time, pressure, heating rate, or atmosphere;
- reactor design certification;
- emissions compliance;
- operator competence;
- qualified biochar;
- contamination safety;
- agronomic suitability;
- durable carbon removal;
- carbon-credit eligibility;
- process execution authority;
- physical actuation.

It freezes only the material/carbon/energy accounting semantics needed to make later regenerative process models inspectable and falsifiable.