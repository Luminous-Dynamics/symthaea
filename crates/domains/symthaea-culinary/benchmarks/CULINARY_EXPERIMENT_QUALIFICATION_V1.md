# Culinary paired-experiment qualification benchmark v1

**Status:** contract and synthetic fixture only. No real cooking experiment has been performed and no outcome benefit is claimed.

## Paradigm shift

Move from ranking generated recipes by a composite score to conducting preregistered, paired experiments with explicit outcomes, controls, execution order and provenance. Yield, discarded material, energy, time, nutrition and sensory acceptance remain distinct measures; improving one must not silently conceal regressions in another.

The new `validate_culinary_experiment_v1` function validates the protocol and record structure. It does not compute a p-value, claim causality, estimate statistical power, certify food safety or authenticate evidence sources.

## Acceptance gates

1. Every block contains exactly one control and one variant run.
2. Every block's randomized assignment is a permutation of the declared conditions and matches the recorded execution order.
3. Preregistration timestamp strictly precedes each run's start; run start precedes its end, and paired runs do not overlap in the declared order.
4. Run IDs, block IDs and outcome IDs are unique within their applicable scope.
5. Every run references a process trace and has exactly one record for each declared outcome.
6. Units must match preregistered outcome definitions; numeric outcomes require finite ordered lower/value/upper bounds.
7. Unknown values must remain null and carry an explicit reason; estimated values cannot be counted as observed outcomes.
8. Synthetic sources cannot be marked qualified. Any estimated, unknown or unqualified outcome blocks empirical paired-analysis readiness.
9. Outcome dimensions and directions remain separate. No universal score or statistical significance claim is emitted.
10. Schema validation, formatting, compilation, Rust tests and Clippy must run against the exact head before qualification. Queued, skipped or absent checks are not pass evidence.

## Research protocol for physical trials

- Freeze protocol bytes and preregistration before the first run; retain the SHA-256 and time.
- Use paired blocks to compare baseline and variant under a documented material batch, appliance, vessel, geometry, operator and environmental context. Record what was controlled and what varied.
- Randomize run order and retain the declared seed/method; verify that the actual sequence matches the preregistration.
- Capture raw input masses, prepared edible yield, discard fractions, energy, elapsed time and any process observations. Keep instrument calibration and measurement uncertainty.
- Collect sensory ratings under an appropriate blinded or anonymized presentation plan; record sample size, recruitment and test design rather than treating a handful of ratings as population evidence.
- Determine sample size/power and analysis methods before data collection. Preserve all blocks, including failed and incomplete runs.
- Separate exploratory trials, validation data and confirmatory test data. Do not use the same results for both tuning and claims of independent performance.
- Report per-outcome paired differences and uncertainty with the analysis plan disclosed; adjust inference if multiple outcomes or repeated comparisons require it.

FAO guidance emphasizes food identity, methods, sampling, quality assurance, cooked yield and the limits of calculated composition estimates. ISO 11136 describes controlled consumer hedonic testing and its study-design considerations. These sources inform the data-collection requirements; they do not validate this software or any future experiment:
- https://www.fao.org/4/y4705e/y4705e15.htm
- https://www.fao.org/4/y4705e/y4705e23.htm
- https://www.iso.org/obp/ui?_escaped_fragment_=iso%3Astd%3Aiso%3A11136%3Aed-1%3Av1%3Aen

## Fixture

`fixtures/culinary-experiment-v1/synthetic-paired-cooking-study.json` contains two paired blocks, four runs and twelve outcome records. All outcome records are marked estimated and all sources are explicitly synthetic. This is intentionally valid but not ready for empirical analysis.

## Qualification boundary

The validator reports data-structure conformance and declared evidence status only. It does not authenticate source files, check whether the randomization process was performed correctly beyond the declared order, establish causality, confirm food safety, or prove benefits to households.