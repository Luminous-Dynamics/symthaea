# Process-trace ↔ nutrient-accounting binding qualification benchmark v1

**Status:** contract benchmark with synthetic fixtures; not empirical nutrition qualification.

## Motivation

FAO describes two separable requirements for prepared-food nutrient calculation: changes to ingredient mass/edible portions and changes to nutrient content during preparation. EuroFIR likewise distinguishes weight yield from nutrient retention and recommends applying them at different levels. This bridge prevents the calculator from using a finished-mass record or ingredient-weight record that refers to a different process trace.

References:
- FAO recipe calculations: https://www.fao.org/4/y4705e/y4705e23.htm
- EuroFIR recipe calculation procedure: https://www.eurofir.org/report-on-collection-of-rules-on-use-of-recipe-calculation-procedures-including-the-use-of-yield-and-retention-factors-for-imputing-nutrient-values-for-composite-foods/
- USDA Foundation Foods source/sample/LOQ context: https://fdc.nal.usda.gov/Foundation_Foods_Documentation/

## Acceptance targets

1. 100% of nutrient-accounting ingredient records map to one unique trace ingredient stream.
2. Zero process-ID, food-state, final-mass, ingredient-mass, or evidence-reference mismatches accepted.
3. Zero ingredient/seasoning/added-oil boundary streams silently omitted from the nutrient inputs.
4. Exact equality of the recorded mass interval and source evidence across a binding; the adapter must not recalculate, round, convert or silently substitute them.
5. 100% agreement on fixed positive and negative controls before qualification.
6. The result means only the records are cross-consistent. It does not establish source authenticity, analytical correctness, retention-factor validity, dietary adequacy, safety, or label compliance.
7. Rust formatting, compilation, tests and Clippy must run on the exact PR head. Missing/skipped checks are not passing evidence.

## Synthetic scenario

Two declared nutrient-bearing inputs (rice 150 g and peas 50 g) are matched to two external trace streams. Water is an explicit process input; the finished 260 g edible product and 20 g evaporated-water stream are explicit outputs. The point values total 280 g on each boundary, but all records are synthetic and this bridge does not compute closure.

## Future real-world benchmark

For a preregistered set of weighed batches, compare ingredient and final mass recorded independently by the nutrition worksheet and process trace. Report unmatched-record rates, state/process identity conflicts, mass interval differences, evidence-identity disagreements, and their causes by cooking method. Keep measured product mass, predicted retention and lab-assayed composition as separate evidence channels.
