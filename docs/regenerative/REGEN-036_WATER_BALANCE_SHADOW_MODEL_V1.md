# REGEN-036 — Water-Balance Shadow Model Contract v1

Status: preregistration / architecture boundary only

Parent: REGEN-035 soil-response shadow model

Program: Luminous-Dynamics/mycelix#940

## Purpose

Define the first deterministic water-balance shadow model for the REGEN program
without turning modeled water storage into water rights, water quality,
irrigation authority, drought diagnosis, or agronomic truth.

REGEN-036 composes Symthaea's existing conserved land-water baseline and
dependency-neutral hydrology driver exports. It remains a modeling layer.

## Governing theorem

```text
identified spatial / temporal support
+ explicit initial modeled water storage
+ explicit forcing evidence or assumptions
+ identified hydrology model
+ identified parameter set
= bounded water-balance projection
```

not:

```text
measured field hydrology
water-source authority
water quality
water allocation right
irrigation permission
plant-available water
drought diagnosis
physical actuation
```

## Existing exact substrate

The first implementation target is:

`symthaea-earth-system::HydrologyBucket`

The current model is a conserved single-bucket baseline with:

- finite storage capacity in mm water equivalent;
- constant precipitation forcing;
- storage-limited evapotranspiration;
- overflow runoff;
- exact constant-forcing trajectories;
- an exact cumulative water-budget residual.

The current model explicitly does not resolve:

- snow;
- groundwater;
- infiltration fronts;
- vegetation;
- routed catchments.

Those exclusions remain normative REGEN-036 non-claims.

## Conservation invariant

For the existing v1 bucket:

```text
initial storage
+ cumulative precipitation
- cumulative evapotranspiration
- cumulative runoff
- final storage
= budget residual
```

A qualified deterministic fixture should drive the residual to the declared
numerical tolerance.

Conservation closure proves the model's accounting contract only.

```text
water-budget closure
!= hydrologic realism
!= agronomic sufficiency
!= irrigation suitability
```

## Evidence / assumption classes

Every input should retain its epistemic class.

Conceptually:

```text
WaterModelInput<T> =
    Observed {
        value,
        evidence_ref,
        spatial_support,
        temporal_support,
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

Missing observations must not become zero precipitation, zero uncertainty, or
default storage.

## Parameter identity

At minimum, a water-balance projection must bind:

- model identity;
- model version / exact code subject;
- bucket capacity;
- potential evapotranspiration parameter;
- initial storage;
- forcing identity;
- spatial support;
- time unit;
- projection horizon;
- parameter-set identity.

The built model parameters are not universal field constants.

## Critical semantic firewalls

The current `HydrologyBucket` exports:

- `storage_mm`;
- `soil_moisture_fraction`;
- actual evapotranspiration under the model;
- runoff under the model;
- water-deficit fraction.

REGEN-036 freezes:

```text
modeled storage_mm
!= measured soil-water storage

modeled soil_moisture_fraction
!= volumetric water content
!= plant-available water
!= root-zone water availability

modeled runoff
!= measured watershed runoff
!= erosion
!= nutrient export

modeled evapotranspiration
!= measured ET
!= crop water requirement
```

The driver's `water_deficit_fraction = 1 - soil_moisture_fraction` is a model
coordinate. It is not a drought-severity index.

## Water authority firewall

REGEN-036 has no authority to establish:

- water-source ownership;
- withdrawal rights;
- allocation rights;
- potable / irrigation water quality;
- watershed permissions;
- environmental-flow obligations;
- irrigation authorization;
- pump control;
- valve control.

Any authoritative water evidence remains external and is referenced by exact
identity.

```text
modeled water available
!= legally / socially available water
!= safe water
!= authorized water
```

## Irrigation semantics

The current v1 exact model treats precipitation as its external water input.

REGEN-036 must not silently encode irrigation as precipitation just to make an
existing API fit.

A later irrigation extension should represent irrigation as a distinct input
flux with its own:

- source identity;
- quantity evidence;
- quality evidence where relevant;
- authority reference;
- timing;
- delivery efficiency / loss assumptions;
- spatial support.

The model may account for an already-declared irrigation flux. It may not
authorize one.

## Treatment-response semantics

REGEN-035 may compare a control water-balance projection with a treatment
projection.

An amendment does not automatically change bucket capacity, evapotranspiration,
or runoff parameters.

Normative:

```text
amendment identity
!= increased water-holding capacity
!= reduced drought stress
!= reduced irrigation demand
```

Any treatment-induced parameter perturbation must be explicit and carry its own
evidence, calibration, or scenario-assumption identity.

## Endpoint set

The first REGEN-036 output should remain physically narrow.

Suggested endpoint identities:

- modeled storage trajectory;
- modeled soil-moisture fraction;
- modeled cumulative evapotranspiration;
- modeled cumulative runoff;
- modeled water-budget residual.

A later endpoint may be added only with an explicit model and semantic contract.

No single `water_resilience_score` belongs in REGEN-036.

## Time and support alignment

Hydrology is support-sensitive.

A precipitation record, storage observation, and treatment plot must not be
combined merely because their units are compatible.

Every comparison should declare:

- spatial support;
- temporal support;
- aggregation method;
- timezone / interval convention where observational data requires it;
- missing intervals;
- whether forcing is measured, interpolated, or assumed.

A mismatch yields unresolved comparison rather than silent coercion.

## Driver export boundary

`hydrology_drivers` deliberately exports dependency-neutral water states.

That is the correct REGEN boundary.

The downstream ecology layer may map soil-moisture fraction into a response
function only with an identified coupling / parameter set.

REGEN-036 itself does not assign biological meaning.

## Drought and waterlogging

The model may expose dry or saturated modeled states.

It does not infer a universal drought or waterlogging diagnosis.

Such interpretations depend on crop / ecosystem / soil / infrastructure
context and require a separate explicit profile.

```text
low modeled storage
!= drought diagnosis

bucket saturation
!= harmful waterlogging
```

## Null / adverse outcomes

REGEN-036 must preserve:

- null treatment effect;
- reduced modeled storage;
- increased modeled runoff;
- increased modeled evapotranspiration;
- endpoint disagreement;
- unsupported comparison;
- missing forcing;
- invalid parameter domain.

"Regenerative" treatment identity must never force a positive water outcome.

## Prediction receipt

Conceptually:

```text
WaterBalancePrediction {
    prediction_id,
    subject_context_id,
    model_id,
    parameter_set_id,
    forcing_refs,
    assumption_refs,
    initial_state_ref,
    spatial_support,
    temporal_support,
    horizon,
    endpoint_trajectory_refs,
    budget_residual,
    unresolved_inputs,
}
```

The receipt is model evidence, not environmental observation.

## First executable candidate

The smallest valid implementation should:

1. bind a frozen `HydrologyBucket` parameter set;
2. bind synthetic initial storage and constant precipitation;
3. compute the exact trajectory;
4. export dependency-neutral hydrology drivers;
5. verify exact water-budget closure;
6. verify monotonic time;
7. verify storage / moisture bounds;
8. verify zero-time identity;
9. verify a null treatment perturbation produces identical trajectories;
10. reject incompatible units / support / missing required inputs before model
    execution.

Do not begin with remote sensing, learned hydrology, irrigation optimization, or
crop-water recommendations.

## Independent oracle

The existing exact constant-precipitation solution should remain the oracle for
any later numerical integrator.

A later learned or high-dimensional water model earns promotion only if it
demonstrates declared improvement over the deterministic baseline on frozen
evidence without losing conservation / identity guarantees.

## Adversarial fixtures

Qualification should include:

- precipitation evidence from the wrong site;
- initial storage from a mismatched support;
- bucket capacity with no parameter provenance;
- negative precipitation;
- storage outside capacity;
- time-unit mismatch;
- irrigation quantity mislabeled as precipitation;
- modeled moisture offered as measured moisture;
- modeled runoff offered as watershed observation;
- modeled deficit offered as drought diagnosis;
- external water right omitted while output is presented as irrigation
  permission;
- treatment name implies improved retention but parameter perturbation is zero;
- favorable water projection offered as agronomic recommendation.

Every shortcut must reject or remain explicitly unresolved.

## Relationship to REGEN-023

REGEN-023 is the Mycelix-side regenerative irrigation-water binding.

REGEN-036 is the Symthaea modeling counterpart.

The intended relationship is:

```text
authoritative source / quality / quantity / permission evidence
                    |
                    v
          exact evidence references
                    |
                    v
         REGEN-036 water model
                    |
                    v
        modeled water endpoints
```

The arrow never reverses.

A Symthaea water projection cannot mint source, quality, allocation, or
authority evidence.

## Relationship to REGEN-035

REGEN-035 may consume REGEN-036 outputs as one endpoint family.

It must preserve the model identity and cannot relabel moisture fraction as crop
response.

## Relationship to Phase F

Experiment intelligence may use REGEN-036 sensitivity or disagreement to
propose measurements / trials.

```text
informative water experiment
!= irrigation permission
!= pump command
```

## Promotion gate

An executable REGEN-036 candidate should not be described as qualified until its
exact subject demonstrates:

- frozen Rust / toolchain lineage;
- compile / test / strict lint;
- deterministic replay;
- exact model / parameter / forcing identities;
- exact water-budget closure;
- unit and support checks;
- null-perturbation identity;
- missing-input fail-unresolved behavior;
- independent exact-solution parity;
- no biological-meaning shortcut;
- no water-authority output;
- postflight immutability.

## Deliberate non-claims

REGEN-036 does not establish:

- site hydrology calibration;
- soil field capacity;
- infiltration rate;
- groundwater recharge;
- irrigation requirement;
- drought resilience;
- crop-water demand;
- amendment water-retention efficacy;
- watershed runoff;
- erosion;
- water quality;
- water rights;
- environmental-flow compliance;
- irrigation authority;
- autonomous irrigation control.

It establishes only a conservation-first water-balance modeling boundary.
