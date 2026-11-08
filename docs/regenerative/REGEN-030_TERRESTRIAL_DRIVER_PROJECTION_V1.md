# REGEN-030 — Terrestrial Driver Projection Boundary v1

Status: preregistration / architecture-only

Program: Luminous-Dynamics/mycelix#940
Research profile: Luminous-Dynamics/symthaea#3252 / REGEN-001

Frozen Symthaea base for this preregistration:

```text
2f3fa459ab99da3f64da1a6aae26de7c2972c112
```

Relevant existing Earth-system anchors at that base:

```text
crates/domains/symthaea-earth-system/src/lib.rs       ec4b2c8e54a291a5e2e6a01e0bf7e65a6ef30b17
crates/domains/symthaea-earth-system/README.md        b2f5fbe357743db2682d8cebd22e6b93a2330733
crates/domains/symthaea-earth-system/COUPLING.md      e0628f12db1f76dd59b7f63b20626e9a8fc9aa8e
crates/domains/symthaea-earth-system/MODEL_CARDS.md   9f43ae071da905230b4c89c0f23ee71a938f6071
```

Qualified Mycelix shared-evidence admission anchor available to the REGEN program:

```text
REGEN-019D ProductHead
872151cae7bee995b6bfff7867fc40178ad4759e
```

This document creates no executable model, agronomic claim, authority, or physical action.

---

## 1. Purpose

Freeze the smallest terrestrial modeling waist between Mycelix evidence and Symthaea's existing reduced-order Earth-system models.

The roadmap historically calls REGEN-030 a **"terrestrial PIE projection"**. The current Symthaea tree does not expose a canonical `PIE` implementation API under that name. Therefore v1 treats that phrase as a program label only and MUST NOT invent `Pie*` types or refactor unrelated infrastructure merely to satisfy the acronym.

The executable target should instead be a thin adapter around the current, explicit Earth-system driver and model-card contracts.

The central theorem is:

```text
qualified evidence snapshot
+ explicit projection profile
+ explicit assumptions
+ identified deterministic model
+ identified parameter set
= modeled terrestrial projection
```

not:

```text
= environmental observation
= ecological truth
= agronomic suitability
= intervention recommendation
= authority
= physical action
```

---

## 2. Existing substrate to reuse

`symthaea-earth-system` is already a dependency-light reduced-order scientific baseline. Its current public surface includes deterministic or bounded models for:

- climate forcing and thermal response;
- conserved land-water storage;
- soil-carbon turnover;
- nutrient turnover;
- finite-interval productivity accounting;
- dependency-neutral temperature, hydrology, soil-carbon, nutrient, and productivity driver export;
- deterministic ensembles and exact/analytic numerical oracles where available.

REGEN-030 MUST compose these surfaces rather than build a second terrestrial physics or biogeochemistry stack.

The existing coupling contract is also normative for this program: Earth-system and ecology remain independently owned, timestamps and integration intervals are explicit, uncertainty is not silently collapsed, model identity remains visible, and driver export does not silently infer ecological meaning.

---

## 3. Three-layer boundary

REGEN-030 freezes three distinct data categories.

### 3.1 Evidence snapshot

An evidence snapshot identifies the exact external observations and validated lineages consumed by one projection attempt.

It preserves, by reference or verified commitment:

```text
subject identity
observation / computed-product identity
phenomenon
EvidenceClass
value + unit where present
uncertainty
spatial support
valid temporal support
provenance / lineage
retrieval or evidence-snapshot identity
```

The adapter MUST NOT duplicate Mycelix PEF as a second observation schema.

### 3.2 Projection input

A projection input is an explicit interpretation of evidence for one model invocation.

Examples of roles include:

```text
InitialState
BoundaryCondition
ModelParameterEvidence
CalibrationEvidence
ExternalForcing
```

A value does not acquire one of these roles merely because its unit is numerically compatible.

Every evidence-to-input interpretation MUST identify the projection profile/rule that performed the mapping.

### 3.3 Projection output

A projection output is model-generated state or a dependency-neutral driver record.

It MUST carry enough identity to answer:

```text
which model?
which model version / code identity?
which parameter set?
which input evidence snapshot?
which assumptions?
which time grid / interval convention?
which ensemble member or uncertainty branch?
which projection profile?
```

A projection output MUST NOT be serialized back into an observational evidence class without a new explicit evidence proposition.

---

## 4. Proposed v1 type surface

Names are provisional until executable implementation review, but the semantic partition is frozen.

```rust
pub struct TerrestrialEvidenceSnapshotRef {
    pub snapshot_ref: ExactRef,
    pub source_system: ExactRef,
    pub source_revision: ExactRef,
}

pub enum ProjectionInputRole {
    InitialState,
    BoundaryCondition,
    ModelParameterEvidence,
    CalibrationEvidence,
    ExternalForcing,
}

pub struct ProjectionInputBinding {
    pub role: ProjectionInputRole,
    pub source_evidence_ref: ExactRef,
    pub expected_phenomenon: ExactRef,
    pub expected_unit: ExactRef,
    pub conversion_rule_ref: Option<ExactRef>,
}

pub struct ProjectionAssumption {
    pub assumption_ref: ExactRef,
    pub proposition: ExactRef,
    pub rationale_ref: ExactRef,
}

pub struct TerrestrialProjectionProfile {
    pub profile_ref: ExactRef,
    pub model_ref: ExactRef,
    pub parameter_set_ref: ExactRef,
    pub required_inputs: Vec<ProjectionInputBinding>,
    pub assumptions: Vec<ProjectionAssumption>,
    pub time_grid_ref: ExactRef,
}

pub struct TerrestrialProjectionFrame<T> {
    pub projection_ref: ExactRef,
    pub evidence_snapshot_ref: ExactRef,
    pub projection_profile_ref: ExactRef,
    pub model_ref: ExactRef,
    pub parameter_set_ref: ExactRef,
    pub elapsed_time: f64,
    pub uncertainty_member_ref: Option<ExactRef>,
    pub payload: T,
}
```

The executable design MAY use stronger unit-safe wrappers than the sketch above. It SHOULD avoid free-form strings where an existing canonical Symthaea/Mycelix identity already exists.

No v1 type contains authorization, actuator, procurement, reservation, governance, or physical-control capability.

---

## 5. Evidence / assumption firewall

The following are distinct:

```text
Observed value
Derived evidence product
Calibrated parameter
Scenario assumption
Default illustrative parameter
Model state
Projection output
```

They MUST NOT share an unlabeled scalar path.

In particular:

```text
missing observation != zero
missing observation != model default
model default != measured value
scenario assumption != evidence
calibration fit != universal parameter truth
projection != observation
```

If a required projection input is missing, v1 should return a typed unresolved/not-evaluable outcome unless the projection profile explicitly declares a scenario assumption.

A scenario assumption must remain visibly an assumption in downstream lineage.

---

## 6. Unit and support discipline

REGEN-030 MUST NOT perform hidden unit, spatial-support, temporal-support, or basis conversions.

Any conversion that changes meaning requires an identified deterministic rule, for example:

```text
mass / area normalization
wet-basis -> dry-matter basis
instantaneous -> interval aggregate
site observation -> modeled cell input
concentration -> stock
stock -> flux
calendar duration -> SI seconds
```

Numerical compatibility is not semantic compatibility.

A model input is admissible only when the exact projection profile establishes the required relation between evidence support and model support.

---

## 7. Time and event ordering

The existing Earth-system coupling contract already requires explicit time alignment; REGEN-030 inherits that rule.

A projection record MUST identify its time convention and MUST NOT silently treat:

```text
observation time
forcing-event time
integration-boundary time
model update time
field intervention time
```

as simultaneous.

Event-aligned Earth-system trajectories should retain generated breakpoints instead of being resampled onto an assumed uniform grid without an explicit transformation.

---

## 8. Uncertainty and ensembles

Uncertainty MUST remain inspectable.

Allowed v1 patterns include:

```text
bounded low / central / high evaluations
deterministic Cartesian ensemble members
identified parameter alternatives
identified structural-model alternatives
```

Not allowed:

```text
ensemble mean -> observation
central estimate -> certainty
member count -> probability
unweighted ensemble -> calibrated probability distribution
```

If probabilities are later introduced, their statistical interpretation requires its own qualified theorem.

---

## 9. Model-identity firewall

Current Earth-system models intentionally include alternative structural hypotheses and illustrative parameterizations.

Therefore every projection MUST distinguish at least:

```text
model family
model/code revision
parameter-set identity
calibration identity if any
projection-profile identity
```

For example, output from the reversible two-box carbon oracle cannot be silently merged with output from the configurable three-reservoir model under one generic `carbon_state` label.

Likewise illustrative soil, nutrient, hydrology, and productivity defaults MUST NOT be represented as observationally calibrated site models.

---

## 10. Driver export rather than semantic overreach

REGEN-030 should prefer the existing dependency-neutral driver exports when crossing model boundaries.

Current useful Earth-system driver families include:

```text
TemperatureDriverSample
HydrologyDriverSample
SoilCarbonDriverSample
NutrientDriverSample
ProductivityDriverSample
LatitudeBandDriverSample
```

Their receiving adapter MUST NOT silently manufacture:

```text
habitat suitability
crop yield
soil health
biomass availability
water rights
feedstock eligibility
agronomic recommendation
resilience score
```

Those propositions belong to later REGEN models/contracts with their own assumptions and evidence.

---

## 11. First deterministic executable target

The first REGEN-030 implementation SHOULD be intentionally small.

Recommended target:

```text
frozen synthetic Mycelix-style evidence fixtures
        |
        v
explicit projection-input bindings
        |
        v
one or more existing Earth-system deterministic kernels
        |
        v
identity-bearing terrestrial driver frames
        |
        v
machine-readable projection receipt
```

The first campaign should not introduce HDC, LTC, LLM selection, adaptive policy learning, online calibration, or physical control.

Candidate first kernels are those that already expose strong conservation/oracle properties:

- `HydrologyBucket`;
- `TwoPoolSoilCarbon`;
- `TwoPoolNutrientCycle`;
- `EcosystemProductivityModel`.

This choice aligns the regenerative program with existing conservation evidence while keeping the adapter scientifically inspectable.

---

## 12. Minimum qualification matrix

The first executable REGEN-030 candidate should include at least the following proposition families.

### Identity / lineage

1. exact evidence snapshot is retained;
2. exact projection profile is retained;
3. exact model and parameter-set identity are retained;
4. changing one source observation identity changes projection lineage;
5. model alternatives cannot share an unlabeled output identity.

### Evidence / assumption separation

6. missing required evidence fails closed;
7. explicit scenario assumption is allowed only when declared;
8. assumption remains tagged downstream;
9. model default cannot masquerade as observation;
10. derived input requires its declared derivation lineage.

### Units / support

11. incompatible units are rejected;
12. unqualified support conversion is rejected;
13. explicit deterministic conversion is replayable;
14. temporal-support mismatch is not silently accepted.

### Determinism / numerics

15. same exact inputs/profile/model produce byte-stable canonical result payloads where serialization is part of the theorem;
16. non-finite inputs fail closed;
17. model domain guards propagate as typed projection failure;
18. known Earth-system conservation/oracle residuals remain within declared tolerance.

### Uncertainty

19. ensemble members remain individually identified;
20. ensemble mean does not gain observational status;
21. unknown/not-evaluable remains distinct from zero.

### Authority / claims

22. projection types contain no authority or execution capability;
23. high predicted benefit cannot bypass a missing hard input;
24. projected output cannot be accepted as observed outcome without a separate evidence record;
25. projection cannot directly construct a REGEN-054 recommendation or physical command.

---

## 13. Relationship to downstream Phase D

REGEN-030 is the common projection waist for later deterministic models.

```text
Mycelix evidence
      |
      v
REGEN-030 terrestrial projection boundary
      |
      +--> REGEN-031 biomass-allocation screen
      +--> REGEN-032 pyrolysis accounting
      +--> REGEN-033 compost/co-compost accounting
      +--> REGEN-034 nutrient-balance kernel
      +--> REGEN-035 soil-response shadow model
      +--> REGEN-036 water-balance shadow model
      +--> REGEN-037 heat/resource-quality bridge
      +--> REGEN-038 crop/nutrition service projection
      +--> REGEN-039 settlement metabolism graph
```

REGEN-039 MUST compose already-qualified kernels; it must not become a second hidden authority or monolithic world model.

---

## 14. Relationship to experiment intelligence

Phase F must consume bounded, identity-bearing Phase-D outputs rather than bypassing deterministic baselines.

The intended dependency direction is:

```text
observations / evidence
        -> REGEN-030..039 deterministic modeling
        -> bounded uncertainty-bearing predictions
        -> REGEN-050 candidate generation
        -> REGEN-051 information analysis
        -> REGEN-052/053 plural response analysis
        -> REGEN-054 recommendation only
        -> independent adoption / authority boundary
```

No Phase-F model confidence may rewrite the evidence snapshot, model identity, hard eligibility inputs, or authority state inherited from upstream systems.

---

## 15. Relationship to field evidence

Prediction and observation remain separate lineages.

A later trial should bind:

```text
pre-trial evidence snapshot
projection profile + model identity
frozen prediction
field protocol
actual observations
prediction-vs-observation comparison
```

The observed outcome MUST NOT rewrite the historical prediction in place. Model updates create a new model/projection lineage.

Null and adverse outcomes remain first-class evidence.

---

## 16. Deliberate non-claims

REGEN-030 v1 establishes no:

- calibrated site-specific soil model;
- calibrated crop model;
- universal biomass recovery fraction;
- biochar or compost efficacy claim;
- groundwater or catchment model;
- food-security prediction;
- habitat or biodiversity model;
- carbon-removal or carbon-credit claim;
- legal or property-right determination;
- irrigation or process-control recommendation;
- model superiority claim;
- intervention authorization;
- equipment, device, or actuator authority;
- autonomous physical action.

It freezes only the evidence-to-model-to-driver boundary required to make later REGEN modeling scientifically inspectable and replayable.

---

## 17. Promotion gate

Do not proceed directly from this architecture note to a large terrestrial simulator.

The next executable rung is earned only by a small adapter candidate that:

```text
1. uses existing Earth-system kernels unchanged where possible;
2. preserves exact evidence / assumption / model identities;
3. has deterministic fixtures;
4. passes exact-head format / test / strict lint;
5. demonstrates at least one independent conservation/oracle check;
6. rejects malformed identity/unit/support substitutions;
7. emits no recommendation or authority.
```

Only after that baseline qualifies should REGEN-031 and the later Phase-D models consume it.
