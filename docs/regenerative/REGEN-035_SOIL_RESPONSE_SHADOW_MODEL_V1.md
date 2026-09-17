# REGEN-035 — Soil-Response Shadow Model Contract v1

Status: preregistration / architecture boundary only

Parent: REGEN-034 nutrient-balance kernel

Program: Luminous-Dynamics/mycelix#940

## Purpose

Define the first scientific boundary for predicting soil / crop response to a
regenerative treatment without collapsing heterogeneous observations, model
states, treatment identity, and agronomic conclusions into one "soil health"
number.

REGEN-035 is a shadow-model contract. It may compare explicit model predictions
with later observations. It does not establish agronomic efficacy, suitability,
safety, causal effect, recommendation authority, or physical-action authority.

## Governing theorem

```text
identified site / plot / treatment context
+ explicit baseline evidence
+ exact treatment identity
+ explicit endpoint definitions
+ identified deterministic model components
+ explicit parameter / calibration provenance
= endpoint-specific shadow predictions
```

not:

```text
soil truth
crop truth
treatment efficacy
causal attribution
universal amendment benefit
agronomic suitability
recommendation authority
physical action
```

## Core semantic firewall

The following propositions are intentionally distinct:

```text
measured soil state
!= modeled soil state
!= modeled treatment response
!= observed treatment response
!= causal treatment effect
!= agronomic suitability
!= recommendation
!= authority
```

Likewise:

```text
more soil carbon
!= improved soil function
!= improved crop response
!= improved nutrition
!= improved ecosystem outcome
```

and:

```text
biochar applied
!= biochar beneficial
compost applied
!= compost beneficial
nutrient retained
!= nutrient plant-available
water retained
!= water agronomically useful
```

## Why no universal soil-health scalar

A single scalar would erase important disagreement between endpoints.

A treatment may plausibly improve one modeled or measured endpoint while
worsening or leaving unchanged another. REGEN-035 therefore represents a soil
response as an explicit vector of independently identified endpoints.

Conceptually:

```text
SoilResponseVector {
    endpoint_predictions: [EndpointPrediction],
    unresolved_endpoints: [EndpointId],
    model_context,
    treatment_context,
    baseline_context,
}
```

There is no canonical:

```text
soil_health_score: f64
```

in REGEN-035.

Any later ranking or decision layer must state which endpoint set, weighting,
constraints, rights, and uncertainty policy it uses. It may not smuggle those
choices into the scientific model.

## Endpoint identity

Every predicted or observed endpoint must carry an exact semantic identity,
units, support, timing, and interpretation.

Illustrative endpoint classes include:

- water / moisture response;
- soil-carbon stock or flux response;
- nutrient stock / turnover response;
- physical soil-property response;
- chemical soil-property response;
- biological / ecological response;
- crop establishment / growth / yield response;
- adverse / contamination / loss response.

These are endpoint classes, not claims that Symthaea presently contains
validated models for every member.

The first executable REGEN-035 candidate should use only endpoint families for
which deterministic kernels and exact contracts already exist.

## Existing deterministic substrate

REGEN-035 should compose existing Symthaea models rather than create a second
soil simulator.

### Hydrology

`symthaea-earth-system::HydrologyBucket` provides a conserved finite-storage
water bucket and dependency-neutral hydrology driver export.

This can support a modeled water-storage endpoint only when the caller declares
the support, units, forcing, parameter set, and validity assumptions.

It does not by itself establish:

- field capacity;
- infiltration;
- groundwater recharge;
- plant-available water;
- root-zone water;
- irrigation sufficiency;
- amendment-induced water retention.

### Soil carbon

`symthaea-earth-system::TwoPoolSoilCarbon` provides an exact two-pool carbon
turnover baseline with a closed carbon budget.

Its built-in `illustrative()` parameters are explicitly illustrative.

Therefore:

```text
TwoPoolSoilCarbon projection
!= measured soil organic carbon
!= amendment persistence
!= biochar permanence
!= carbon removal
```

A treatment comparison must identify how the treatment changes model inputs or
parameters. No amendment-specific effect is inferred merely because the model
contains soil carbon.

### Nutrients

REGEN-034 freezes the nutrient balance around
`symthaea-earth-system::TwoPoolNutrientCycle`.

That model may contribute nutrient-state predictions to REGEN-035, but:

```text
mineral pool
!= measured plant-available nutrient

modeled uptake
!= measured crop uptake
```

Any mapping from measured soil chemistry or amendment analysis into modeled
organic / mineral stocks is a separate explicit adapter.

### Productivity

`symthaea-earth-system::EcosystemProductivityModel` is a finite-interval carbon
ledger with environmental and nutrient ceilings.

It is not a vegetation-dynamics model and must not be treated as a crop-yield
oracle.

Therefore:

```text
modeled productivity ledger
!= crop biomass observation
!= harvested yield
!= nutritional service
```

### Ecology couplings

`symthaea-ecology::HydroLogisticEnvironmentCoupling` and
`NutrientLogisticEnvironmentCoupling` expose explicit, replaceable response
functions.

Their moisture response, half-saturation, floors, and growth / capacity
exponents are model parameters, not universal agronomic laws.

They may be used as transparent hypothesis models only when:

1. every parameter set has an identity and provenance;
2. the biological subject / guild / crop context is explicit;
3. the driver support and time units are compatible;
4. the resulting predictions remain labeled model outputs;
5. no parameter default is presented as field calibration.

## Agribot quarantine

The current `symthaea-agribot::SimpleAgribotSimulator` contains convenient
engineering-simulator channels including:

- `CROP_HEALTH`;
- `YIELD_FORECAST`;
- `SOIL_NUTRIENTS`;
- `SOIL_EXHAUSTION`;
- `DROUGHT_RISK`;
- `WATERLOGGING_RISK`.

Its first-pass simulator updates crop health and yield forecast using authored
bounded arithmetic over simulator channels.

Those formulas are useful for robotics / control experiments. They are not a
field-science model and are outside the REGEN-035 scientific evidence line.

Normative firewall:

```text
Agribot simulator crop_health
!= measured crop health
!= REGEN soil-response prediction

Agribot simulator yield_forecast
!= agronomic yield model
!= field-trial prediction
```

REGEN-035 must not import those channels as scientific endpoint evidence.

## Treatment identity

Every shadow prediction must bind to an exact treatment identity.

A treatment identity should distinguish at least:

- treatment batch / material identity;
- amount / concentration / loading representation where relevant;
- placement / application representation where relevant;
- treatment timing;
- co-treatment identities;
- control / comparator identity;
- site / plot identity;
- baseline observation set;
- evidence lineage.

This is an identity contract, not an operating recipe.

REGEN-035 defines no universal amendment amount, application method, timing, or
field procedure.

## Baseline and comparator semantics

A treatment response is never represented as a bare post-treatment state.

Conceptually:

```text
ResponseComparison {
    baseline_or_control,
    treatment,
    endpoint,
    predicted_control_trajectory,
    predicted_treatment_trajectory,
    predicted_difference,
    uncertainty_or_sensitivity,
}
```

The first executable implementation may be simpler, but the semantic distinction
must remain.

Required propositions:

```text
post-treatment state
!= change from baseline
!= difference from control
!= causal treatment effect
```

## Null and adverse predictions

The model must preserve:

- predicted no effect;
- predicted adverse effect;
- endpoint disagreement;
- model failure;
- missing inputs;
- unsupported endpoint;
- parameter-range violation.

A treatment model is not allowed to encode "regenerative" as a positive
directional prior that prevents null or adverse predictions.

Normative requirement:

```text
treatment identity must not imply positive response
```

## Heterogeneous response

REGEN-035 must be context-conditioned.

A later observation that a treatment was beneficial at one site does not become
a universal treatment effect.

Relevant context may include:

- soil / substrate state;
- climate / weather drivers;
- hydrologic state;
- nutrient state;
- crop / organism / guild identity;
- treatment batch;
- management history;
- time horizon;
- spatial support;
- measurement method.

The exact context schema should grow only as evidence requires.

## Evidence / model separation

Inputs should carry an epistemic class.

Conceptually:

```text
InputValue<T> =
    Observed {
        value,
        evidence_ref,
        support,
        uncertainty,
    }
  | Derived {
        value,
        derivation_ref,
        source_evidence_refs,
    }
  | ScenarioAssumption {
        value,
        assumption_id,
    }
  | CalibratedParameter {
        value,
        calibration_id,
        validity_domain,
    }
```

No constructor should silently convert a missing observation into a
scenario assumption.

Missing data remains missing.

## Prediction receipt

A REGEN-035 prediction should be replayable.

Conceptually:

```text
SoilResponsePrediction {
    prediction_id,
    subject_context_id,
    treatment_context_id,
    comparator_context_id,
    endpoint_id,
    model_id,
    parameter_set_id,
    input_evidence_refs,
    assumption_refs,
    baseline_state_ref,
    horizon,
    predicted_value_or_trajectory,
    uncertainty_or_sensitivity,
    unresolved_inputs,
}
```

The receipt is model evidence, not environmental observation.

## Prediction vs observation

Later REGEN-086 work should compare predictions with independently recorded
observations.

The observation must not be overwritten by the model prediction, and the model
must not be retroactively relabeled as correct merely because one endpoint
moves in the expected direction.

Conceptually:

```text
PredictionComparison {
    prediction_ref,
    observation_ref,
    endpoint_identity_match,
    support_match,
    timing_match,
    residual,
    declared_metric,
}
```

If identities, support, or timing do not match, the comparison is unresolved
rather than coerced.

## Causal boundary

REGEN-035 is predictive / comparative.

It does not independently establish causal treatment effects.

Causal claims require the later field-trial / observational evidence program,
including REGEN-015 and the integrated proving campaign.

Normative:

```text
prediction accuracy
!= causal identification

before-after difference
!= causal effect

correlation
!= treatment efficacy
```

## Uncertainty

REGEN-035 should prefer bounded sensitivity, deterministic ensembles, or
explicit scenario sets before learned probabilistic uncertainty.

If a probability distribution is introduced later, its interpretation must be
declared. A deterministic parameter sweep must not be mislabeled as a
probability distribution.

## Parameter promotion

A parameter set may progress through states such as:

```text
Illustrative
-> LiteratureBound
-> ContextCalibrated
-> IndependentlyChecked
-> Replicated
```

The exact promotion taxonomy may evolve, but code must never upgrade a parameter
set merely because it produced a desirable prediction.

## First executable candidate

The first REGEN-035 executable increment should be intentionally small.

Suggested scope:

1. define endpoint IDs and model / parameter identities;
2. define observed / assumed / calibrated input classes;
3. construct a synthetic baseline context;
4. compose an existing deterministic hydrology, nutrient, or soil-carbon kernel;
5. define a synthetic treatment scenario only by an explicit perturbation to a
   declared model input or parameter;
6. compute control and treatment projections;
7. emit their endpoint-specific difference;
8. retain all assumptions and evidence references;
9. prove that a null perturbation produces a null predicted difference;
10. prove that unsupported / missing inputs fail unresolved rather than becoming
    positive default values.

Do not begin with a multi-endpoint learned model.

## Adversarial fixtures

The qualification corpus should include:

- treatment name suggests "regenerative" but perturbation is zero;
- treatment produces an adverse modeled endpoint;
- treatment improves one endpoint and worsens another;
- missing baseline evidence;
- incompatible units;
- mismatched spatial support;
- mismatched time support;
- wrong treatment-batch identity;
- parameter set outside validity domain;
- model endpoint does not match requested endpoint;
- Agribot simulator value offered as field evidence;
- illustrative parameter set presented as calibrated;
- prediction offered as observation;
- before-after change offered as causal effect.

Every such shortcut must be rejected or remain explicitly unresolved.

## Relationship to REGEN-015

REGEN-015 owns the field-trial contract.

REGEN-035 supplies predictions and explicit endpoint definitions that a trial may
choose to test.

It does not choose the trial endpoint after seeing the result.

The desired sequence is:

```text
frozen endpoint + model + parameter set
-> frozen prediction
-> frozen trial protocol
-> observation
-> prediction-vs-observation comparison
-> replication / revision
```

## Relationship to Phase F

Experiment intelligence may eventually use REGEN-035 uncertainty, disagreement,
or sensitivity to propose informative trials.

That produces a recommendation only.

```text
high expected information gain
!= permission to experiment
!= permission to apply treatment
!= physical-action authority
```

## No authority

REGEN-035 contains no:

- farm-management authority;
- treatment prescription;
- actuator command;
- irrigation authority;
- dosing authority;
- equipment authority;
- procurement authority;
- carbon-credit authority;
- ecological-rights override.

Symthaea remains a model / comparison / experiment-proposal layer.

## Promotion gate for REGEN-035 implementation

An executable candidate should not be described as a qualified soil-response
model until its exact subject demonstrates at least:

- Rust compile / test / strict lint on a frozen toolchain;
- deterministic replay;
- exact subject / parameter / endpoint identity;
- unit and support checks;
- null-perturbation identity;
- explicit adverse outcome support;
- missing-input fail-unresolved behavior;
- independent arithmetic / analytic oracle where applicable;
- no Agribot scientific-evidence shortcut;
- no authority-bearing output;
- postflight immutability.

## Deliberate non-claims

This contract does not establish:

- biochar efficacy;
- compost efficacy;
- Terra Preta equivalence;
- yield improvement;
- soil restoration;
- drought resilience;
- fertilizer replacement;
- nutrient availability;
- food safety;
- contaminant safety;
- carbon sequestration;
- carbon-credit eligibility;
- ecological net benefit;
- farm recommendation;
- causal treatment effect;
- autonomous physical control.

It establishes only the boundary for making those questions experimentally
legible without pretending the answers are already known.
