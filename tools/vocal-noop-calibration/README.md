# Vocal no-op tolerance calibration

Dependency-free Rust calibration of acoustic collateral tolerances from repeated no-op measurements.

## Why this exists

A control-response evaluator needs a tolerance for every collateral metric. Choosing thresholds after inspecting confirmatory results produces optimistic evaluations. Choosing them from three or four repeats provides little tail coverage. Calibrating ten metrics at 95% confidence each also does not provide 95% confidence for the complete ten-metric family.

This tool makes the calibration policy explicit before confirmatory evaluation.

## Statistical method

For each metric, collect absolute differences from matched no-op repeats across independent, representative calibration units. Use the sample maximum as a one-sided, distribution-free upper tolerance bound.

For target population coverage p and sample size n, the sample maximum's confidence of covering at least p of the metric's distribution is:

~~~text
1 - p^n
~~~

Therefore the required sample count is:

~~~text
n >= ln(1 - confidence) / ln(p)
~~~

For multiple metrics, the tool applies a Bonferroni allocation of family-wise error across the metric set. It does not assume the metrics are statistically independent.

Examples for 95% population coverage:

| Metric count | Family-wise confidence | Minimum independent units |
|---:|---:|---:|
| 1 | 95% | 59 |
| 10 | 95% | 104 |
| 20 | 95% | 117 |

The calculated statistical bound is then multiplied by an explicitly declared safety margin (minimum 1.0) and compared with an explicitly declared engineering floor. The resulting calibrated tolerance is never smaller than the observed maximum.

## Input contract

- A metric specification declares a stable metric ID, unit, safety multiplier and minimum engineering floor.
- Every observation contains a calibration-unit ID, metric ID, unit and finite non-negative absolute no-op delta.
- Each metric must be measured over the same set of unique calibration-unit IDs.
- The calibration unit is the independent sampling unit, not an arbitrary frame or repeated window from the same take.
- Missing metrics, duplicate units, extra metrics, unit mismatches, negative values and non-finite values fail closed.
- The family-wise confidence and population coverage must both be strictly between zero and one.
- The tool rejects data sets below the mathematically required sample count.

The tool cannot establish that calibration units are actually independent or representative. The experiment owner must define and document the sampling unit (for example, held-out source take × singer × register × backend/voicebank stratum) and avoid pseudo-replication from overlapping frames or repeated measurements on one take.

## Output

The report includes the requested coverage and family-wise confidence, per-metric confidence allocation, calibration-unit IDs, sample count, required sample count, maximum observed no-op delta, safety multiplier, engineering floor and final tolerance.

The family-wise confidence lower bound uses the Bonferroni union bound. It is conservative and does not require independence between different acoustic metrics.

## Important limitations

This is a distribution-free tolerance-bound calculation for independent, representative calibration units. It does not prove the distribution is stationary across backend versions, voices, registers, languages or analysis pipelines. Calibrate separate strata when those conditions change.

Do not use the confirmatory/test split to calibrate thresholds. Freeze and hash this calibration report before running control interventions. A changed analyzer, renderer artifact, voicebank, preprocessing policy, sample rate or metric definition invalidates the corresponding calibration.

This tool calibrates engineering tolerances from no-op variability. It does not measure naturalness, intelligibility, control success or SOTA performance.

## Run locally

~~~sh
cargo +1.96.0 fmt --check --manifest-path tools/vocal-noop-calibration/Cargo.toml
cargo +1.96.0 test --manifest-path tools/vocal-noop-calibration/Cargo.toml
~~~
