# CF-024: nutrition delivery per constrained resource

**Status:** benchmark contract proposal; not evaluated. No nutrient-composition observations, dietary adequacy estimates, health effects or field improvements are claimed.

## Why this is a paradigm shift

The usual crop objective is yield per area or revenue per area. A nutrition-oriented system should ask a more consequential question:

> How much nutritionally meaningful, safe, edible food reaches a person, per unit of scarce water, land, energy, money and time?

Crop yield, nutrient concentration, nutrient mass harvested, edible mass after losses, nutrient retention during storage/preparation, and bioavailable nutrient delivery are different quantities. A crop can have a high nutrient concentration and still deliver little total nutrient mass if edible yield is low. Conversely, high mass yield can hide dilution of minerals or protein. Nutrient concentration alone does not prove bioavailability; food matrix, preparation, digestibility, antinutritional factors and portion sizes can all matter.

Recent work supports evaluating crop nutritional resilience jointly across productivity, nutrient density, bioavailability and dietary diversity, instead of treating yield as a complete food-security proxy ([Dissanayaka et al., *From yield to nutrition* (2026)](https://www.frontiersin.org/journals/nutrition/articles/10.3389/fnut.2026.1987519/abstract)). A 2025 meta-analysis measured elevated-CO₂ effects across 43 crops and 32 nutrients/antinutrients using 5,324 entries covering 29,524 observation pairs, illustrating both the breadth of evidence and the need to preserve crop/nutrient-specific response structure ([ter Haar et al., *Global Change Biology*](https://pmc.ncbi.nlm.nih.gov/articles/PMC12616468/)). A 2026 review also identifies crop-, nutrient- and stress-specific variation rather than a uniform direction of nutritional change ([Bello et al., *Agriculture*](https://doi.org/10.3390/agriculture16111220)).

These papers do not establish that CropFoundry can improve dietary outcomes. They motivate a new evaluation target and a careful evidence boundary.

## Objective and constraints

For nutrient `n`, first align the mass with the stage where the concentration was measured:

```text
nutrient_mass_at_assay_stage_mg =
  concentration_mg_per_kg * edible_mass_kg_at_concentration_stage

if concentration_stage == measured_at_serving:
  delivered_nutrient_mass_mg = nutrient_mass_at_assay_stage_mg
  # a retention factor is not applied; the assay already represents the served state

if concentration_stage == measured_before_preparation:
  delivered_nutrient_mass_mg =
    nutrient_mass_at_assay_stage_mg * documented_retention_fraction
  # retention factor and its evidence are required

nutrient_mg_per_serving =
  delivered_nutrient_mass_mg / explicitly_declared_servings_in_batch
```

A further bioavailability adjustment can only be used where its factor, source, applicability, and uncertainty are explicitly declared. It must not be silently applied as if it were directly measured. Retention is applied to nutrient mass from the assay-stage edible mass; do not multiply a pre-preparation concentration by a post-preparation mass. Batch-level delivered mass and per-serving delivered mass are separate outputs.

The objective should be a vector before it becomes a scalar: amounts delivered for each declared nutrient, target-requirement coverage for each defined population group, dietary diversity, edible-serving count, cost, water, land, energy, labor and uncertainty. Hard constraints include food-safety gates, resource caps, and any predeclared upper intake limits. Missing nutrient assays are **unknown**, not zero and not imputed from a neighboring crop. Literature factors and modeled quantities stay visibly separate from direct measurements.

Do not optimize an opaque aggregate nutrition score. If decision-makers later choose weighted trade-offs, record the weights, their provenance, the hard constraints, and a Pareto frontier so users can see which nutrients or groups are losing out.

## Executable nutrient-accounting prototype

A local-only accounting CLI has been added to test explicit nutrient/mass/resource arithmetic and fail closed on unsupported assumptions. It accepts a small JSON record rather than downloading or selecting a dataset:

```bash
cargo run -q -p xtask -- cropfoundry-nutrition-accounting \
  --input /path/to/nutrition-accounting-input-v0.json \
  --out reports/cropfoundry/nutrition-accounting-receipt.json
cargo test -p xtask cropfoundry_nutrition
```

The input schema requires the edible mass to correspond to the exact stage of the concentration assay, distinguishes fresh-edible from dry-edible mass basis, and rejects a mismatched basis. Concentration measured before preparation requires an explicit retention fraction and a separate evidence URI. Batch-level nutrient delivery is distinct from per-serving mass, which is only reported when a serving count is explicitly supplied. Bioavailability is calculated only when an explicit factor and its own evidence URI are provided; unknown factors remain unknown. Direct assays require a method, evidence URI and source artifact digest. The receipt retains the exact input snapshot and its SHA-256, keeps literature/model estimates separate from direct assays, and reports nutrient mass per water, land, energy and currency units only when those resource measurements exist.

The prototype is a calculator and provenance-contract test, not a validated nutrition model. Its synthetic CI fixture uses invented values only to exercise the implementation; it cannot establish a crop nutrient value, nutrient adequacy, health outcome, food safety or farm recommendation.

## Reference data path: South Africa first, global comparator second

For a South African use case, begin with [SAFOODS](https://safoods.mrc.ac.za/) as the local food-composition reference and access path. Its public site describes 1,857 foods and a free web view of energy and selected macronutrients/fibre; its [2024 sixth-edition composition tables](https://safoods.mrc.ac.za/products.html) cover 36 nutrients across 16 food groups and are sold as an electronic publication ([site details](https://safoods.mrc.ac.za/composition.html)). The free view may support a restricted schema/matching pilot, but it does not expose every nutrient needed for CF-024. The detailed table's research-use, purchase, redistribution and model-training terms need clarification before acquisition or reuse. SAFOODS values are reference means, not exact nutrient assays for a user's cultivar, lot, farm or preparation; some source values are not South African in origin.

[USDA FoodData Central](https://fdc.nal.usda.gov/api-guide/) provides a global comparator and its official API guide states that FDC data are CC0 1.0. It is useful for testing parsers, food identifiers, nutrient tags and provenance, but FDC includes different data types and food/processing states and cannot stand in for local South African assay data. Any snapshot should preserve the FDC data type, food ID, reference, nutrient basis and preparation state.

Follow [FAO/INFOODS standards and guidelines](https://www.fao.org/infoods/infoods/standards-guidelines/en/) for food matching, units/denominators, nomenclature and data quality. Keep analytical, calculated, borrowed/literature and imputed values distinct at nutrient-record level; never average across unmatched food states or fresh/dry bases by convenience. These sources are now recorded as DS-015 (SAFOODS) and DS-016 (USDA FDC), both training-disabled in the source manifest.

## Proposed baseline suite

1. **Mass-yield baseline:** maximize edible harvested mass under equal water, land, energy and budget caps.
2. **Nutrient-density baseline:** rank/allocate using directly assayed concentration on a consistent fresh- or dry-mass basis. This is a useful but incomplete baseline.
3. **Lowest-cost nutrition basket:** meet the declared requirements with the lowest fully loaded cost using only nutrients, edible fractions, serving sizes and retention factors that have valid source evidence.
4. **Resource-constrained diverse basket:** minimize declared requirement shortfall and cost while maintaining a preregistered diversity floor and respecting the same resource budgets.
5. **Robust version:** optimize under uncertainty intervals for assays, seasonal yields, storage loss, price, nutrient-retention and bioavailability factors. Report worst-case and expected shortfalls separately.

Every baseline must be compared on the same resource envelope. Report per-nutrient and per-group outcomes, not only a total score.

## Proposed targets

- **Data contract:** every nutrient value has an assay/source, units, edible-portion definition, fresh/dry basis, sample count, uncertainty and version; every processing/retention factor has source evidence and matching crop/process conditions.
- **Accounting correctness:** reproduce known nutrient-mass calculations from raw inputs; no negative edible mass, untraceable unit conversion, double-counted batch, or silent replacement of missing evidence.
- **Fair comparison:** beat simple equal-resource baselines on the preregistered nutrient-delivery objective without violating hard requirements, water, land, energy, affordability or diversity constraints.
- **Robustness:** publish sensitivity to plausible yield, price, loss, retention, and bioavailability ranges. Report which nutrients/population groups cease to meet requirements under stress and data uncertainty.
- **Provisional outcome target:** after the dataset, population requirements, resource envelope and analysis plan are independently reviewed, investigate at least a 10% improvement in the preregistered delivered-nutrient objective, with no new predeclared requirement shortfalls and no increase in cost per adequate serving. This is a proposed research target—not an established benchmark standard, forecast, or current result.

No model may claim nutritional adequacy for a population from crop composition alone. Dietary patterns, serving sizes, preparation, individual needs, food access, safety and public-health context require appropriate domain review.

## Required split and provenance discipline

- Pin data-source and assay versions before analysis; capture retrieval time and artifact digest where authorized.
- Keep all measurements from a crop lot/batch in one split. Separate train/test by time, lot and site as required by the question; images, assays and repeated measurements of the same lot are not independent samples.
- Distinguish direct assay, literature-derived factor, model estimate and unknown in the schema. A single numeric field must not erase that epistemic distinction.
- Report missingness by nutrient, crop, site, season, processing state and population group. Do not impute nutrition facts from visual appearance.
- Lock serving definitions, target population requirements, diversity criteria, hard limits, weighting and evaluation metrics before inspecting test outcomes.
- Use sensitivity and negative controls: vary the mass basis, remove unsupported retention factors, perturb price/yield assumptions, and check whether the recommended basket changes under plausible measurement uncertainty.
- Require independent nutritionist/agronomist review of the accounting rules and a prospective evaluation before making real-world nutrition impact claims.

## Integration path

- **Symthaea:** evidence-aware data structures, uncertainty accounting, optimization/decision policies and reproducible evaluation receipts.
- **Mycelix:** versioned provenance, permission and claim states for assays, food lots and producer/community data.
- **Sol-Atlas:** water/land/climate/context layers with explicit spatial-temporal uncertainty, never as a substitute for nutrient assays.
- **Symtropy:** scenario visualization and resource trade-off exploration; synthetic scenes are not biological evidence.

CF-024 is blocked until source-backed nutrient assays, edible/serving definitions, storage/preparation retention factors, bioavailability evidence where used, and population-specific requirements are pinned. No training is authorized by this benchmark document.
