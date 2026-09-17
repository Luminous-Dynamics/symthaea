# REGEN-038 — Crop / Nutrition Service Projection Contract v1

Status: preregistration / architecture boundary only

Parent: REGEN-037 thermal-service compatibility

Program: Luminous-Dynamics/mycelix#940

## Purpose

Define an explicit modeling boundary from terrestrial productivity evidence and
assumptions to crop / food / nutritional-service projections without collapsing
ecosystem carbon, crop yield, harvested food, edible food, stored food, and
nutritional service into one number.

REGEN-038 is a projection layer only.

It creates no agronomic recommendation, dietary recommendation, food-safety
claim, procurement authority, or physical action.

## Governing theorem

```text
identified biological / crop context
+ identified productivity projection
+ explicit biomass / harvest conversion
+ explicit harvest / storage loss assumptions or evidence
+ explicit edible-fraction evidence
+ explicit nutrient-profile evidence
= bounded crop / nutrition service projection
```

not:

```text
measured yield
food available to people
food safe to eat
nutrition delivered
diet adequacy
agronomic recommendation
authority
```

## Core semantic chain

The following stages are intentionally distinct:

```text
gross primary production
!= net primary production
!= retained biomass carbon
!= crop biomass
!= harvestable biomass
!= harvested mass
!= edible mass
!= stored edible mass
!= delivered food
!= consumed food
!= nutritional service
```

No constructor may skip those distinctions silently.

## Existing productivity substrate

`symthaea-earth-system::EcosystemProductivityModel` is an auditable
finite-interval carbon ledger.

It exposes:

- potential gross primary productivity;
- environmental and nutrient ceilings;
- gross primary production;
- autotrophic respiration;
- net primary production;
- retained biomass carbon;
- litter carbon;
- nutrient uptake;
- remaining modeled mineral nutrient;
- carbon / nutrient budget residuals.

Its own module contract states that it is a finite-interval ledger rather than a
vegetation model.

REGEN-038 preserves that boundary.

```text
retained_biomass_carbon
!= crop biomass
!= harvested yield
```

## No Agribot yield shortcut

The current `SimpleAgribotSimulator` carries a bounded `YIELD_FORECAST`
channel for robotics / stewardship simulation.

REGEN-038 must not import that value as agronomic evidence or a crop-yield model.

Normative:

```text
Agribot YIELD_FORECAST
!= REGEN crop-yield projection
!= field yield observation
```

## Crop / biological identity

Any crop-specific projection must bind an exact biological context.

Potential identity dimensions include:

- species;
- cultivar / variety where relevant;
- production system;
- site / plot;
- growth interval;
- planting / establishment context;
- environmental context;
- management context.

The schema should remain as small as evidence permits.

A generic "crop" label must not silently imply one carbon-to-yield mapping.

## Biomass conversion

A later crop-specific projection may map retained biomass carbon into biomass
only through an explicit conversion / partition model.

Conceptually:

```text
BiomassConversion {
    model_id,
    parameter_set_id,
    carbon_fraction,
    allocation_fractions,
    evidence_or_assumption_refs,
    validity_domain,
}
```

The exact form may evolve.

The invariant is:

```text
carbon stock
!= dry biomass
!= fresh biomass
```

Moisture / water mass must remain explicit where mass conversions require it.

## Harvest index / harvestable fraction

Harvestable production must be a separate transformation.

```text
biomass produced
!= harvestable biomass
!= harvested biomass
```

A harvest fraction or harvest-index-like parameter must carry:

- biological context;
- production context;
- evidence or calibration identity;
- uncertainty / validity domain.

No default "typical crop" value belongs in the scientific core.

## Harvest losses

Harvest loss is first-class.

Potential distinctions include:

- field loss;
- damage;
- uncollected biomass;
- handling loss.

REGEN-026 on the Mycelix side owns harvest / storage-loss observations.

REGEN-038 may consume exact evidence references or explicit scenario
assumptions.

Unknown loss is not zero loss.

## Edible fraction

Harvested mass is not automatically food.

```text
harvested mass
!= edible mass
```

Edible fraction must be explicit and context-bound.

Non-edible coproducts remain visible because they may feed other regenerative
loops, but their presence does not create a food-service claim.

## Storage and spoilage

Stored edible mass should preserve explicit losses over time.

```text
harvested edible mass
!= food available at later time
```

Storage assumptions may include:

- duration;
- preservation mode;
- measured or assumed loss;
- spoilage / discard;
- processing loss.

REGEN-038 v1 should not invent storage kinetics unless an independently
identified model exists.

A scenario loss factor remains a scenario assumption.

## Nutrient-profile boundary

Symthaea's existing `symthaea-culinary::NutrientProfile` follows a useful
epistemic pattern: nutrient totals are supplied by the caller from an external
source rather than inferred from ingredient names.

REGEN-038 should preserve that pattern.

Nutrient composition must come from:

- a lab analysis;
- a recognized composition dataset;
- a declared recipe / product accounting source;
- another exact evidence source;
- or an explicit synthetic fixture in tests.

It must not be inferred from crop carbon.

```text
edible mass
!= nutrient profile
```

## Energy / calorie firewall

The culinary module's Atwater arithmetic can calculate energy from a declared
macronutrient profile.

That is a narrow accounting formula.

```text
Atwater kcal
!= food quality
!= dietary adequacy
!= individual health recommendation
```

REGEN-038 must not produce personalized diet or medical advice.

## Micronutrients

The current culinary `NutrientProfile` is not a complete micronutrient
database.

REGEN-038 must not claim complete nutritional service from an incomplete profile.

A nutrition service vector should preserve exactly which nutrient dimensions are
known and which are absent.

Missing micronutrient data is not zero micronutrient content.

## Nutritional-service vector

There is no universal `nutrition_score`.

Conceptually:

```text
NutritionServiceProjection {
    edible_mass,
    service_interval,
    nutrient_profile_id,
    declared_nutrient_totals,
    known_losses,
    unresolved_nutrients,
    evidence_refs,
    assumption_refs,
}
```

A later decision layer may use explicit nutritional requirements, but REGEN-038
itself does not collapse the vector into one welfare or diet score.

## Seasonal service

Food service is temporal.

A crop that produces a large annual harvest is not equivalent to uniform
year-round availability.

REGEN-038 should preserve:

- harvest interval;
- storage interval;
- service interval;
- loss timing.

Temporal aggregation must be explicit.

## Population / demand firewall

REGEN-038 may project a supply-side service vector.

It does not infer:

- household demand;
- population dietary need;
- equitable allocation;
- distribution access;
- affordability;
- preference;
- cultural suitability.

Those are separate coordination / economic / social layers.

## Food-safety firewall

Nutrient presence does not establish safety.

```text
edible classification
!= food safety
```

REGEN-038 does not establish:

- contaminant clearance;
- pathogen safety;
- allergen safety;
- storage safety;
- regulatory compliance.

Those propositions require their own evidence.

## Quantity conservation

Every mass transformation should close its declared boundary.

Illustratively:

```text
harvestable biomass
=
harvested mass
+ field / collection loss
+ unresolved residual
```

and:

```text
harvested edible mass
=
stored / delivered edible mass
+ storage / handling loss
+ unresolved residual
```

Do not force unrelated mass bases together.

Dry mass, fresh mass, carbon mass, and edible mass remain distinct.

## Evidence / assumption typing

Every conversion should preserve whether it is:

- observed;
- derived;
- calibrated;
- literature-bound;
- scenario-assumed;
- synthetic fixture.

A desirable nutrition outcome must not cause an assumption to be relabeled as
evidence.

## Prediction receipt

Conceptually:

```text
CropNutritionProjection {
    projection_id,
    crop_context_id,
    productivity_projection_ref,
    biomass_conversion_ref,
    harvest_conversion_ref,
    loss_evidence_refs,
    edible_fraction_ref,
    nutrient_profile_ref,
    service_interval,
    output_vector,
    unresolved_fields,
}
```

The receipt is model evidence.

## Null / adverse outcomes

REGEN-038 must preserve:

- zero harvest;
- harvest loss;
- storage loss;
- low edible fraction;
- unfavorable nutrient profile;
- endpoint disagreement;
- insufficient evidence;
- unsupported biological conversion.

"Regenerative" identity may not impose positive food / nutrition output.

## First executable candidate

The first implementation should be deliberately synthetic and conservation-first.

Suggested scope:

1. consume one frozen `ProductivityLedger`;
2. bind one synthetic crop-context identity;
3. apply one explicitly synthetic carbon-to-biomass conversion;
4. apply one explicitly synthetic harvestable fraction;
5. apply one explicit harvest-loss fraction;
6. apply one explicit edible fraction;
7. bind one caller-supplied synthetic nutrient profile;
8. compute a service vector;
9. close every declared mass ledger;
10. prove zero harvest fraction produces zero harvested service;
11. prove missing nutrient-profile evidence remains unresolved;
12. reject Agribot `YIELD_FORECAST` as a scientific input.

This proves pipeline semantics, not crop science.

## Adversarial fixtures

Qualification should include:

- retained biomass carbon offered directly as harvested yield;
- dry biomass offered as fresh mass with no water conversion;
- harvested mass offered as edible mass;
- unknown storage loss set to zero;
- nutrient composition inferred from carbon mass;
- missing micronutrients represented as measured zeros;
- Agribot yield forecast offered as field evidence;
- wrong crop / cultivar conversion parameters;
- parameter set outside declared validity domain;
- annual yield presented as uniform daily service;
- nutrient profile offered as food-safety evidence;
- Atwater energy offered as diet adequacy;
- favorable service projection offered as allocation authority.

Every shortcut must reject or remain explicitly unresolved.

## Relationship to REGEN-021 / REGEN-026

Mycelix REGEN-021 owns the crop / nutritional-service evidence contract.

REGEN-026 owns harvest / storage-loss observations.

REGEN-038 is the Symthaea projection counterpart.

The model must reference authoritative evidence identities rather than duplicate
them.

## Relationship to REGEN-039

Settlement metabolism may consume crop / nutrition service projections only
through exact identity-bearing service records.

It must not recreate a yield or nutrition shortcut.

## Relationship to Phase F

Experiment intelligence may use uncertainty in conversion, loss, or response to
propose measurements / trials.

```text
informative crop trial
!= permission to plant
!= permission to apply treatment
!= food allocation authority
```

## Promotion gate

An executable REGEN-038 candidate should not be described as qualified crop /
nutrition modeling until its exact subject demonstrates:

- frozen toolchain / code lineage;
- compile / test / strict lint;
- deterministic replay;
- exact crop / model / parameter identities;
- explicit mass bases;
- mass-ledger closure;
- explicit loss handling;
- missing-data preservation;
- no Agribot yield shortcut;
- no carbon-to-nutrition shortcut;
- no food-safety / diet-authority output;
- postflight immutability.

## Deliberate non-claims

REGEN-038 does not establish:

- crop yield calibration;
- agronomic efficacy;
- harvest performance;
- food availability;
- food safety;
- nutritional adequacy;
- individual dietary needs;
- micronutrient completeness;
- storage safety;
- equitable allocation;
- economic affordability;
- farming recommendations;
- autonomous agricultural control.

It establishes only a typed, evidence-preserving crop / nutrition service
projection boundary.
