# Nutrient accounting qualification benchmark v1

**Status:** contract draft with synthetic fixture and test cases. No real food-composition accuracy or nutrient-retention performance is claimed.

## Paradigm shift

Separate three things which recipe calculators often blur:
1. **Ingredient nutrient amount:** composition per 100 g edible ingredient multiplied by the actual edible ingredient mass.
2. **Nutrient transformation:** nutrient-specific retention or loss for the actual food/process combination, applied once to amount, not to a concentration that is then also yield-adjusted.
3. **Final concentration:** retained nutrient amount divided by the independently measured or qualified final recipe mass.

The final recipe mass and nutrient amount are distinct physical quantities. A yield factor changes concentration via the denominator; it must not be applied a second time to nutrient total. Added ingredients must be represented explicitly. Unknown composition is not zero. An evidence-qualified result must be based on versioned composition, mass, final-yield and retention evidence; arithmetic self-consistency alone cannot qualify scientific truth.

## Research basis

- FAO's procedure for calculating multi-ingredient food composition explicitly separates edible-portion corrections, cooked-ingredient/yield adjustments, nutrient totals, recipe-level weight and retention adjustments, and final values per 100 g or serving: [FAO Food Composition Data](https://www.fao.org/4/y4705e/y4705e23.htm).
- FAO's discussion of composite prepared dishes notes that calculated results depend on known raw-to-cooked yield and that retention-factor estimates should carry lower confidence than representative analytical values: [FAO: Calculation of composition of composite prepared dishes](https://www.fao.org/4/y4705e/y4705e15.htm).
- EuroFIR distinguishes a weight yield factor from a nutrient retention factor, recommends recipe-level yield adjustment and ingredient-level retention factors, and emphasizes detailed documentation of recipe, source data, factors and calculation method: [EuroFIR recipe calculation method](https://www.eurofir.org/how-do-recipes-and-composite-foods-come-to-their-nutritional-values/) and [guideline paper](https://pubmed.ncbi.nlm.nih.gov/28867099/).
- EuroFIR's step-by-step guideline says to exclude cooking-medium water used to boil rice, pasta, potatoes and legumes from the recipe ingredient calculation, while separately determining cooked weight and using ingredient-specific retention factors: [EuroFIR recipe guideline, steps 1–8](https://www.eurofir.eu/bacchus/wp-content/uploads/2015/12/EUROFIR-RECIPE-GUIDELINE_FINAL.pdf). Accordingly, the synthetic fixture uses the measured final dish mass as the concentration denominator and excludes cooking water as an ingredient nutrient source.
- USDA Foundation Foods records sample/analytical context, refuse/edible portion information, and edible-portion weight conventions. Portion weights are data-type-specific; missing nutrients can mean “not analyzed,” not an established zero: [USDA FoodData Central Foundation Foods documentation](https://fdc.nal.usda.gov/Foundation_Foods_Documentation/).
- The EuroFIR guideline also recommends checking sums of major components plus water for plausible closure around 100 g per 100 g edible portion. This contract does not yet implement that composition-closure diagnostic; it is a planned independent consistency check, not a substitute for nutrient-specific laboratory validation: [EuroFIR recipe guideline PDF](https://www.eurofir.eu/bacchus/wp-content/uploads/2015/12/EUROFIR-RECIPE-GUIDELINE_FINAL.pdf).

These sources motivate a source-pinned calculation chain, but the v1 contract is not a claim of complete implementation or formal compliance with the cited guidance.

## Synthetic conformance targets

- 100% agreement with independently enumerated exact arithmetic and interval arithmetic.
- Zero silently omitted ingredients, missing nutrient values, inconsistent nutrient units, duplicate ingredient/nutrient IDs or duplicate retention factors.
- Zero accepted output tampering, invalid bounds or source chain gaps.
- Zero qualification from synthetic, unreviewed or merely reviewed inputs; never infer evidence quality from matching numbers.
- Zero retention factors applied across a mismatched process ID or incompatible input/output food state.
- Zero double-application of recipe yield to retained nutrient total.
- Deterministic outputs under ingredient, nutrient and JSON property order permutations.
- All tested nutrient outputs include a traceable input contribution for every ingredient. A zero contribution must be explicit and evidence-backed, not implied by a missing row.

## Empirical qualification plan

Assemble a rights-reviewed dataset that pairs ingredient composition data with measured prepared-recipe analyses and full preparation protocols. A representative validation study should record ingredient edible masses, preparation states, cooking conditions, final recipe mass, component-specific retention evidence, laboratory methods, sample batch, replicates, and source/version/licence. Analytical reference results need independent provenance and should be held out by recipe and process, not by random duplicated rows.

Evaluate:
- absolute and relative error for each nutrient per 100 g finished food and per serving;
- signed bias, median error, 90th/95th percentile error and confidence intervals;
- empirical coverage and width of 90%/95% predictive intervals;
- nutrient-family and process-specific performance (water-soluble vitamins, minerals, protein, fat, carbohydrate, sodium);
- performance by regional staple, dish type, cooking method, equipment, data source and food-composition basis;
- mass-balance residuals, missing-value handling, abstention, evidence age, reviewer effort and repeatability.

Provisional target for recipe/process strata with qualified component data:
- median absolute relative error ≤10% for nutrients above analytical quantification thresholds;
- 95th percentile absolute relative error ≤25%;
- nominal 95% prediction intervals achieve 90–98% measured coverage with interval width reported;
- mass-balance diagnostics and retention-factor provenance available for 100% of qualified results;
- zero silently qualified outputs where ingredients, nutrient source values, process factors or final mass are missing or out of scope.

These are experimental targets, not universal guarantees. Nutrient-specific targets may need adjustment before test-set lock according to analytical variability, quantification limits, and intended use. Any clinical or regulated nutrition-labelling use needs a separate validation and governance process.

## Data and source governance

- Reuse canonical NUTRI-SOURCE-001 (#7481) for permissions, release pinning and freshness rather than duplicating source-registry policy.
- Keep nutrition values and retention factors as separate sourced claims; cite component definitions, analytic method, denominator, source release and applicable process.
- Obtain explicit redistribution permission for local food-composition data where required.
- Do not use a global average factor as a substitute for unmeasured local food, kitchen, or process characteristics without a recorded applicability decision.
- Keep regional capability coverage visible, including explicit unsupported/unknown states, rather than treating gaps as zero nutrients.

## Scope exclusions and risks

This contract only supports nonnegative ingredient nutrient values and retention factors from 0 to 1. Fortification/nutrient additions must currently be represented as explicit recipe ingredients with their own composition. More complex gains/losses and material streams (discarded cooking liquid, drained oil, bone/peel waste, absorption/migration) require a subsequent transformation ledger.

The validator checks internal arithmetic and source metadata shape; it does not fetch or authenticate data, establish source rights, assess medical dietary needs, guarantee allergen safety, or prove empirical accuracy. Rust compilation and execution must be verified independently before this can be treated as an executable implementation.
