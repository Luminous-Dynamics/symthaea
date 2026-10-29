# PITB interval-method comparison under component-size stress

**Date:** 2026-10-29  
**Status:** executed independent synthetic calibration. This is not a real PITB analysis, not a production Rust output, and not an independently approved inference method.

## Question

The prior component-size screen showed that finite-cluster inference can vary with the number and size distribution of connected components. This follow-on experiment asks whether the current whole-component percentile bootstrap has an obvious alternative under the same synthetic conditions.

We compare:

1. **Whole-component target-weighted percentile bootstrap:** resample connected components with replacement and recompute the target-weighted effect ratio in each resample.
2. **Weighted cluster-sandwich t interval:** a CR1-style cluster score variance with a t critical value and df = number of components − 1. It omits CR2/leverage correction and is included as a diagnostic comparator, not as an endorsed production method.

Both use a target-weighted treatment-minus-control estimate, 95% two-sided intervals, and the same registered +10 percentage-point observed-effect / positive-lower-bound utility rule. The generating process centers the true target-weighted treatment effect at zero, while allowing effect heterogeneity to correlate with component size.

## Reproduction

```bash
python3 scripts/research/pitb_inference_method_comparison_v0.py \
  --studies 1000 \
  --bootstrap-reps 299 \
  --seed 20261029 \
  --out /tmp/pitb-inference-method-comparison.json
```

This run used Python 3.13.5, NumPy 2.3.5 and SciPy 1.17.0. It covered six synthetic component-size profiles and two effect-structure mechanisms (12 cells), producing paired summaries for both interval methods (24 method rows). There were 1,000 simulated studies per cell and 299 component bootstrap resamples per bootstrap-method study. The exact generated JSON report is available as a companion execution artifact for this work session; no actual PITB component manifest or outcomes were loaded.

## Results

Coverage is conditional on an informative interval; interval availability was 100% in all 24 rows. Rejection and false-gate rates are unconditional. Monte Carlo Wilson intervals quantify the noise in the simulation rate only.

| Component profile | Components / targets | Method | Coverage | Lower-bound rejection under null | False +10pp gate |
|---|---:|---|---:|---:|---:|
| Balanced | 40 / 160 | Percentile bootstrap | 94.6% | 2.4% | 0.6% |
| Balanced | 40 / 160 | Weighted sandwich t | 96.0% | 2.3% | 0.6% |
| Moderate tail | 40 / 84 | Percentile bootstrap | 93.8% | 2.0% | 1.4% |
| Moderate tail | 40 / 84 | Weighted sandwich t | 95.1% | 2.0% | 1.3% |
| Heavy tail | 40 / 120 | Percentile bootstrap | 93.5% | 2.0% | 0.9% |
| Heavy tail | 40 / 120 | Weighted sandwich t | 94.9% | 2.0% | 1.0% |
| One giant component | 40 / 100 | Percentile bootstrap | 92.5% | 3.5% | 1.8% |
| One giant component | 40 / 100 | Weighted sandwich t | 92.6% | 4.2% | 2.1% |
| Few components | 20 / 60 | Percentile bootstrap | 93.9% | 2.3% | 2.0% |
| Few components | 20 / 60 | Weighted sandwich t | 94.4% | 2.7% | 2.5% |
| More components | 60 / 180 | Percentile bootstrap | 93.3% | 3.0% | 0.5% |
| More components | 60 / 180 | Weighted sandwich t | 94.8% | 2.6% | 0.5% |

The table uses the size-independent-effect cell for each profile. The 12 size-correlated cells are retained in the machine-readable result as well; across **all** 12 cells the coverage range was 91.4–94.6% for the percentile bootstrap and 92.6–96.0% for the sandwich t method. The maximum observed point rejection rate was 3.5% for the percentile method and 4.2% for the sandwich t method. For the one-giant-component, size-independent case, the t method's rejection rate was 4.2% with a 95% Monte Carlo Wilson interval of 3.12–5.63%. Across the 24 method-by-cell rows, the maximum false primary-gate rate was 2.7% (weighted sandwich t, few components with size-correlated effects); its 95% Monte Carlo Wilson interval was 1.86–3.90%.

These per-cell results **do not establish** the registered simultaneous 5% false-claim requirement. The observed lower-bound rejection is also not the complete primary utility gate. Calibration decisions must use the predeclared simultaneous stress grid, not a favorable selected cell.

## Interpretation

There is no universal winner in this screen:

- The weighted sandwich t interval often had closer-to-nominal coverage in the moderate/heavy-tail profiles and in the 20-component size-correlated case.
- Under one giant component with independent effects, the sandwich t interval was narrower (18.86 versus 21.74 percentage points on average), but its null lower-bound rejection rate rose from 3.5% to 4.2%. A narrower interval is not inherently an improvement if coverage/calibration degrades.
- The percentile bootstrap showed worse conditional coverage in some cells, including 91.4% in the 20-component size-correlated case, but lower rejection in several of the difficult cases.
- With 20 components, the average interval widths were around 25.3–25.8 percentage points: the uncertainty is still large even with 60 target rows.

This supports a methodological paradigm shift: **do not select an interval estimator from one preferred Monte Carlo table or from the real test results.** Freeze the estimand first, then compare candidate inference methods using a preregistered generative grid, simultaneous error control and independent review. A CR2-style method or randomization-based test can be considered only if its assumptions match the sampling/assignment design; neither was implemented here.

## Decisions

1. Keep the current percentile bootstrap as provisional, not certified; do not replace it with the CR1-style t interval based on this screen.
2. Extend calibration to selective missingness and alternate component-size generating processes; report interval availability and all denominator definitions.
3. After the independent relationship audit, run the frozen eligible component-size vector through the same method grid and include a largest-component-share / size-CV stress envelope.
4. Ask an independent statistician to review the target-weighted estimand, interval estimators and simultaneous calibration gate before unsealing test outcomes.
5. Retain the primary benchmark endpoint as exact-target semantic-gate success against the strongest validation-selected baseline. This statistical screen does not answer whether typed insight transfer improves proof success.

## Evidence boundary

The research script was executed locally with the listed settings and syntax-checked. It does not use actual PITB target outcomes, does not verify the production Rust analyzer, does not authenticate the real dependency manifest and does not establish model or SOTA performance. No exact-head CI pass is claimed.
