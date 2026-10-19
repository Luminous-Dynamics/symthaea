# CF-024: nutritional throughput and source-backed uncertainty

**Status:** accounting prototype with an executable synthetic fixture; CF-024 remains *not evaluable* as an agricultural outcome until crop-specific composition, preparation, retention and population requirements are sourced and reviewed.

## Paradigm shift: optimize delivered nutrition, not harvested mass

A food system can maximize tonnes harvested while delivering relatively little of the nutrient a population needs. Crop choice, edible fraction, storage, preparation, waste, cost and water boundary all change the amount of nutrient that reaches a meal. CF-024 asks how to compare crop/portfolio choices on measured nutrient mass delivered per constrained resource, while preserving nutrient-specific constraints and affordability—not a single generic yield score.

That distinction is empirically important. A systematic review covering 67 studies across seven conventionally bred biofortified crops found that post-harvest handling affects provitamin A, iron and zinc retention, with mineral retention varying by processing method; it also identifies bioaccessibility/bioavailability measurement as a gap ([Huey et al., *Nature Food*, 2023](https://doi.org/10.1038/s43016-023-00874-y)). The nutritional-water-productivity concept similarly shifts the accounting question from crop mass per water unit to nutrient output per water unit, while results remain crop- and location-dependent ([FAO AGRIS record](https://agris.fao.org/search/en/providers/122535/records/65ded3c36eef00c2cea1225d)).

## Current executable: one explicit nutrient/batch accounting receipt

Run from the workspace root with a JSON input that conforms to `cropfoundry-nutrition-accounting-input-v1`:

```bash
cargo run -q -p xtask -- cropfoundry-nutrition-accounting \
  --input /path/to/nutrient-accounting-input.json \
  --out reports/cropfoundry/nutrient-accounting-receipt.json
```

The command records the input's own SHA-256 and emits a `cropfoundry-nutrition-accounting-receipt-v1` receipt with nutrient mass per batch and declared serving, plus ratios per litre of water, square metre of land, kWh and currency unit when the corresponding inputs are known. Every supplied resource amount must be paired with a declared boundary: water (for example, irrigation applied, total applied water, crop consumptive use, or a named water-footprint boundary), land (allocated cultivated area, harvested area or lifecycle), energy (on-farm, irrigation pumping, post-harvest or lifecycle), and cost (farm gate, production, post-harvest or delivered-to-serving). The receipt echoes those boundaries beside each ratio. An amount without its boundary—or a boundary without its amount—fails closed; ratios are not comparable when boundary labels differ. The versioned machine-readable contract is [nutrition-accounting-input-v1.schema.json](./nutrition-accounting-input-v1.schema.json); runtime deserialization rejects unknown fields. Legacy v0 point-only inputs remain accepted; v0 inputs cannot include the new interval fields. A missing resource measurement stays unknown; a zero/absent denominator does not produce an infinite or invented efficiency. Resource ratios are comparable only when the water/land/energy/cost boundaries, unit definitions, crop stage and allocation rules match.

#The checked-in [nutrition-accounting-example-v1.json](./nutrition-accounting-example-v1.json) is a **synthetic fixture only**. Its assay, digest, reference URIs, concentration, mass, resource inputs and uncertainty bounds are illustrative fake values for deterministic tests. Never cite or reuse them as measurements, crop-composition data, nutrition guidance or evidence for a farm decision.

## Accounting equations

For concentration `C` in mg/kg of the stated edible mass basis and edible mass `M` in kg at the same stage:

```text
nutrient_mass_at_stage_mg = C_mg_per_kg * M_kg
```

When concentration is measured at serving, this measured product is the delivered nutrient mass and no retention factor is applied. When concentration is measured before preparation, the prototype requires an explicit, evidence-linked retention factor `r`:

```text
nutrient_mass_at_serving_mg = C_mg_per_kg * M_kg * r
```

A bioavailability fraction `b` is optional and must have its own evidence URI. When supplied, the output is explicitly *estimated bioavailable nutrient mass* `nutrient_mass_at_serving_mg * b`, never relabelled as a direct assay. Missing `b` produces no bioavailability estimate; it is not treated as 1. Direct assays require an assay method, concentration evidence URI and a 64-character source digest. The digest's syntax is checked; this CLI does not authenticate it against an independent artifact registry.

## New uncertainty interval support

Point estimates can hide a range of outcomes and are particularly misleading when concentration, edible mass, processing retention and bioavailability are taken from different sources. Input schema v1 optionally accepts paired lower/upper bounds for:

- nutrient concentration (mg/kg);
- edible mass at the same measurement stage (kg);
- retention fraction when concentration is measured before preparation;
- bioavailability fraction when a bioavailability estimate is desired.

Any interval requires both bounds for each field, a declared `uncertainty_interval_kind` (`confidence_interval95`, `credible_interval95`, `measurement_tolerance`, `sensitivity_bounds`, or `observed_range`) and an `uncertainty_evidence_uri`. A point value must lie inside its bounds. Pre-preparation interval calculations require retention bounds; concentration-at-serving inputs must not include retention. The receipt propagates conservative endpoint products to nutrient-mass and, if bounded, bioavailable-mass intervals.

No interval is generated when numeric bounds are absent: a lone point is **not assumed to have zero uncertainty**. Conversely, the term “95%” is not justified merely because an interval is present; the cited source/method must support that interpretation. Endpoint multiplication is interval arithmetic, not a guarantee of statistical coverage and not a substitute for covariance-aware uncertainty propagation when input factors are dependent.

## Evidence tiers and source gates

1. **Data contract:** verify source terms, exact artifact digest, nutrient identity, assay method, units, fresh/dry mass basis, edible portion, crop/cultivar where known, preparation stage, resource boundaries and uncertainty semantics.
2. **Accounting reproduction:** independently recalculate nutrient mass and resource ratios from source-backed records; retain direct, literature-derived and model-estimated composition as distinct provenance classes.
3. **Comparable evaluation:** compare transparent baselines—harvest-mass efficiency, measured nutrient density, cost-per-nutrient, and nutrient output per matching water/land/energy boundary—under the same resource caps. Report each nutrient separately and preserve Pareto tradeoffs.
4. **Dietary adequacy:** only after population group, official nutrient-requirement reference, edible/served portion distributions, diet context and safety/upper-limit constraints are pinned may a separately reviewed layer estimate requirements met. A mg count is not itself a diet-adequacy or health outcome.
5. **Field validation:** a crop portfolio or cultivation recommendation requires agronomically credible trials plus affordability, preparation, waste and access evidence. Neither direct nutrient concentration nor an accounting pass establishes absorption, deficiency prevention, health impact or local transfer.

## Provisional research targets (not achieved measurements)

- Reproduce the nutrient mass balance exactly from a pinned input and independent recalculation.
- Emit no nutrient mass, bioavailability or resource-efficiency metric when required inputs, unit basis or evidence are missing.
- Compare strategies under equal, explicitly defined water, land, energy and budget constraints; report nutrient-specific outcomes and uncertainty, not only a weighted average.
- Keep the CF-024 proposed target of at least 10% improvement in a predeclared nutrient-delivery objective *conditional* on a versioned dataset and population assumptions; do not reduce any predeclared nutrient adequacy constraint, dietary-diversity constraint, safety limit or access requirement to reach the scalar target.

## Data-source notes

The public South African Food Data System (SAFOODS) is a key local composition-reference discovery path, but its free web view exposes only selected nutrients and the broader 2024 tables are paid. The official FAQ says written SAMRC permission is required before any part of the South African Food Database is reproduced ([SAFOODS FAQ](https://safoods.mrc.ac.za/faq.html)). The site says values are means per 100 g edible food, are affected by natural variation, and are not all South African in origin ([composition notes](https://safoods.mrc.ac.za/composition.html)). Do not bulk ingest or reproduce records without permission, and do not treat national mean values as cultivar/lot assays or direct assay evidence. USDA FoodData Central (DS-016) is a separate CC0 reference candidate; it can provide a transparent comparative baseline but cannot stand in for South African food composition, local cultivar/preparation data or direct assays. Both remain training-disabled in the source manifest.

The accounting receipt is intentionally not a safety certificate, diet prescription, clinical claim, or model-training authorization.
