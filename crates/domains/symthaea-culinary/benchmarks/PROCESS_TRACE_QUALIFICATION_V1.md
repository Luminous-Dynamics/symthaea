# Culinary process trace qualification benchmark v1

**Status:** proposed contract benchmark with synthetic fixtures. No empirical cooking-performance claim is made.

## Why this is a step change

Ingredient lists and finished recipe weights do not expose where material went. A process trace keeps ingredient additions, intermediate transfers, retained product, trimming, discarded liquid/fat, evaporation, and utensil residue visible. It also distinguishes a quantity that is unknown from a measured zero. This creates a reusable bridge from Culinary to the shared Planetary Industrial Ecology process-boundary interface without duplicating its mass-balance engine.

EuroFIR separates recipe-level weight yield from ingredient-level nutrient retention; FAO's recipe-calculation guidance likewise separates edible-portion adjustments, cooking effects, final yield, and nutrient-per-mass reporting. These remain separate transformations in the overall architecture:
- EuroFIR: https://www.eurofir.org/report-on-collection-of-rules-on-use-of-recipe-calculation-procedures-including-the-use-of-yield-and-retention-factors-for-imputing-nutrient-values-for-composite-foods/
- FAO: https://www.fao.org/4/y4705e/y4705e23.htm
- USDA Foundation Foods source-context and LOQ documentation: https://fdc.nal.usda.gov/Foundation_Foods_Documentation/

## Acceptance targets

1. 100% of stream references resolve, with exactly one declared producer and consumer when applicable.
2. Zero orphan streams, dangling step references, duplicate stream IDs, duplicate step IDs, self-loops, or backward-in-time internal flows.
3. 100% of streams carry source, version, timestamp, and evidence-status metadata; synthetic sources cannot be relabelled as qualified.
4. Unknown mass never becomes zero. Any unknown stream quantity blocks mass-balance export readiness and retains a human-readable reason.
5. Any boundary inventory state other than claimed_complete blocks export readiness; that status remains an assertion, not an independently verified fact.
6. Stream-record permutation does not change the validity summary.
7. The structural report exposes no balanced, unbalanced, or conservation-pass field. Only PIE-001 may make those classifications.
8. A permanent Rust-native suite and an independent oracle should agree on all synthetic positive/negative cases; target 100% agreement before qualification.
9. Exact-head CI, formatter, compilation, tests and Clippy must execute successfully before treating the Rust implementation as qualified. Skipped, queued, stale, or absent checks are not passes.

## Next empirical benchmark tranche

Use an independently audited set of actual cooking batches with calibrated input/output scales and preregistered boundaries. Measure inventory completeness, instrument resolution, missing-stream rate, input/output interval overlap, and attribution of residuals to evaporation, discarded liquids/fats, trimming, residues, and scale error. Report per-process and per-food-class results; do not tune tolerances on the final evaluation set.

After trace integrity is qualified, connect the explicit stream quantities to PIE-001 and evaluate its mass-status classifications against an independent calculation oracle. Separately assess nutrient prediction against laboratory assays, energy demand against instrumented measurements, waste reduction against weighed disposal, and sensory outcomes with blinded human evaluation. None of those claims follows from a structurally valid trace.
