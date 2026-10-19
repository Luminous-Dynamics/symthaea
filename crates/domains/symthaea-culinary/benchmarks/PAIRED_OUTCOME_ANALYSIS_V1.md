# Paired outcome analysis qualification benchmark v1

**Status:** descriptive analysis contract and synthetic test fixture. The fixture is not real cooking data and no effect or benefit is claimed.

## Paradigm shift

Move from "which recipe has the best overall score?" to "what changed, in which preregistered outcome, against which paired baseline, under what evidence and measurement uncertainty?"

The `analyze_paired_outcomes_v1` function:
- revalidates the experiment and refuses analysis unless every declared measurement is `observed` with an `evidence_status` of `qualified`;
- computes each block's `variant - control` contrast for every preregistered outcome;
- propagates interval bounds conservatively: `difference.lower = variant.lower - control.upper`; `difference.upper = variant.upper - control.lower`;
- reports mean paired contrast, mean propagated measurement bounds, minimum/maximum observed pair contrast and directional counts based on point estimates;
- reports every declared outcome separately, using the primary outcome only as a declared emphasis rather than suppressing secondary outcomes;
- emits no composite winner, p-value, statistical confidence interval, causal estimate or safety verdict.

The experiment's analysis plan must freeze the primary outcome, estimand, aggregation, missing-data policy, interpretation of bounds, requirement to report all declared outcomes, and the explicit statement that population inference is not estimated by this V1 analysis.

## Acceptance gates

1. Estimated, unknown, or not-qualified outcome data cannot produce an analysis report.
2. The analysis plan must reference a declared primary outcome and match all supported V1 semantics; any attempt at post-hoc imputation must fail.
3. Every declared outcome is reported exactly once with its unit and direction, preserving outcomes rather than averaging unlike units.
4. Each paired contrast matches the declared control and variant, independent of input array ordering.
5. For every pair, `lower_difference <= difference <= upper_difference`; aggregate measurement bounds must preserve the same ordering.
6. The sign convention is invariant and explicit: `variant - control` (positive means the variant value is higher, not necessarily better).
7. For `maximize`, `minimize`, and `target`, descriptive directional counts follow the predeclared direction. `observe` outcomes emit no better/worse tally.
8. Measurement-bound outputs must never be labelled confidence intervals or population-effect bounds.
9. Non-finite arithmetic/aggregate overflow fails closed.
10. Rust tests, formatting, compilation and Clippy must run against the exact head before qualification. Queued, skipped and absent CI are not pass evidence.

## Statistical interpretation

This first increment is deliberately descriptive. Propagated intervals quantify only the interval values recorded per measurement and their arithmetic combination. They do not capture between-batch sampling variability, uncontrolled confounding, instrument bias, food heterogeneity, operator effects, or the uncertainty of generalizing to future kitchens.

The next statistical layer should be an explicitly preregistered, design-matched procedure for sampling uncertainty—chosen before outcome inspection—along with sample-size/power planning, sensitivity analysis for missingness, and multiplicity policy. Do not infer that a small exploratory experiment demonstrates population benefit. CONSORT 2025 emphasizes defining primary/secondary outcomes as specific measurement variables and reporting estimated effects with precision; CONSORT's pilot/feasibility guidance likewise emphasizes uncertainty and the feasibility purpose of pilot studies. This module is an earlier contract layer, not a full trial-analysis implementation:
- https://doi.org/10.1001/jama.2025.4347
- https://doi.org/10.1136/bmj.i5239

## Synthetic oracle expectations

The bundled synthetic fixture contains two paired blocks and three dimensions. Its records are estimated/synthetic and must be rejected as empirical input. The test-only observed-path mutation checks arithmetic under declared observed/qualified labels but cannot authenticate the invented source locators.

Expected point differences after that test-only mutation:
- Prepared yield: +6 g in each block; mean paired difference +6 g.
- Discarded material: -3 g in each block; mean paired difference -3 g.
- Energy input: +5 Wh and +4 Wh; mean paired difference +4.5 Wh.

These numbers are fabricated regression oracles, not culinary results. The measurement intervals in the fixture are also invented. No confidence interval or significance test is computed.

## Qualification boundary

The analyzer trusts the upstream contract's declared qualified labels only. It cannot authenticate evidence, verify that randomization was physically followed, prove causality, estimate population effects, confirm safety, or establish sensory/nutritional benefit. Empirical qualification requires real source artifacts, audited measurements, a precommitted protocol and analysis plan, and exact-head software verification.