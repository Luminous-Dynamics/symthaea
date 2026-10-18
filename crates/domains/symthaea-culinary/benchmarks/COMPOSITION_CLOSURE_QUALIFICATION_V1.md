# Proximate composition closure qualification benchmark v1

**Status:** synthetic diagnostic and proposed benchmark protocol. It does not validate component-level analytical correctness, recipe-process mass closure, or nutrition-label suitability.

## Research basis

- EuroFIR recommends summing fat, carbohydrate, fibre, protein, ash/salt, alcohol and water for composition data expressed per 100 g edible portion. Preferred closure is 97–103 g; acceptable closure is 95–105 g. [EuroFIR Recipe Calculation Guideline](https://www.eurofir.org/wp-admin/wp-content/uploads/2015/12/EUROFIR-RECIPE-GUIDELINE_FINAL_v171030.pdf).
- FAO's food-composition guidance says proximate components should ideally sum to 100 g and permits a 97–103 g range in practice, with out-of-range data re-examined for protein factors, carbohydrate expression, units, denominators and source quality. [FAO Food Composition Data](https://www.fao.org/4/y4705e/y4705e15.htm).
- USDA Foundation Foods records analytical/sample context, values on an edible portion basis, and limits of quantification; in some cases values are recorded as zero alongside a nonzero LOQ. A zero in a source therefore needs its source semantics preserved rather than automatically treated as an exact absence. [USDA Foundation Foods Documentation](https://fdc.nal.usda.gov/Foundation_Foods_Documentation/).
- FAO/INFOODS publishes separate guidance for food matching and for converting units, denominators and expressions, reinforcing that food identity, quantity basis and composition closure are different validation questions. [FAO/INFOODS standards and guidelines](https://www.fao.org/infoods/infoods/standards-guidelines/en/).

## Implemented contract

A profile is evaluated on a fixed basis of grams per 100 g edible fresh-weight food and one declared food state. The caller chooses one non-overlapping carbohydrate convention:
- available carbohydrate plus dietary fibre; or
- total carbohydrate (already including fibre) without a second fibre row.

Each supplied component pins source, version, known-at time, evidence status, unit, denominator basis and food state. Components below quantification limit are represented as [0, LOQ] with no point estimate. Not-analysed, not-reported, unsupported-basis and conflicting-source cases must be represented explicitly and produce an indeterminate result; their absence is never silently treated as zero.

Disposition is conservative over the summed interval:
- within_preferred: the entire interval is inside [97, 103];
- within_acceptable: the entire interval is inside [95, 105] but not entirely inside [97, 103];
- outside_acceptable: the entire interval is below 95 or above 105;
- indeterminate: the interval straddles an acceptance boundary or a required component is unavailable.

This is a screening rule for profile plausibility, not proof that the values are correct. Components can be wrong and still sum to 100.

## Acceptance targets

- 100% agreement between Rust implementation and independently enumerated synthetic cases.
- Zero hidden missing values, fibre/carbohydrate double counting, basis/state mismatches, or tampered component IDs.
- Zero promotion of missing or below-quantification values into a precise point sum.
- Monotone uncertainty handling: widening an interval must not strengthen the closure disposition.
- 100% traceability of supplied values; unavailable evidence must remain explicit.
- All classifications report the component denominator and food state; pooled performance cannot mask a food-class or source gap.

For empirical evaluation, stratify by food class, data source, analytic method, preparation state, carbohydrate expression, and regional composition database. Report disposition versus expert data-quality review, reviewer workload, false reassurance rate, and distribution of closure deviation. Do not optimize thresholds on the held-out set: treat the cited bands as methodology-based screening rules, not an adjustable classifier.

## Boundary with material-flow accounting

This contract is not a process mass-balance engine. Shared PIE-001 (issue #1606) owns interval-aware process-boundary mass closure. Culinary should expose explicit process streams for ingredient inputs, added oil/water, retained product, discarded liquid/fat, trimming, food left on utensils, and waste/by-products through that shared API. The mass-balance layer should distinguish exact closure, possible-with-uncertainty and impossible-within-bounds; a wide interval must never count as demonstrated conservation.

## Qualification limits

The fixture and examples are synthetic. Static schema/fixture parsing and an independent calculation oracle do not execute the Rust implementation or perform full JSON Schema validation. No empirical accuracy, food safety, clinical suitability or regulatory labelling claim is made.
