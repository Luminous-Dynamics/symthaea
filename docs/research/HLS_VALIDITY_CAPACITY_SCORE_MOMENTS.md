# HLS validity-memory score-moment diagnostics

This note defines the measurement layer that connects the frozen capacity sweep to the pre-result analytic null model.

## Why accuracy and winner margin are not enough

The primary capacity sweep records cleanup accuracy and the gap between the largest and second-largest candidate score.

That **winner margin** answers:

> How decisively did cleanup choose its winner?

It does not answer:

> How far was the correct target above the strongest distractor?

A confidently wrong prediction can have a large positive winner margin.

The diagnostic therefore adds the signed **true margin**

`target_score - max_distractor_score`.

It is positive when the target outranks every distractor, zero at a tie, and negative when at least one distractor scores higher.

## Independent reconstruction

`measure_validity_capacity_score_moments()` accepts the same public `ValidityCapacityPlan` object as the primary runner but independently reconstructs:

- temporal-axis frequencies;
- key-role codebook;
- candidate-role codebook;
- deterministic span assignments;
- archived validity spans.

The public smoke contract requires bit-identical parity with the primary runner for:

- case and seed identity;
- correct count;
- total queries;
- accuracy;
- mean winner margin;
- smallest winner margin;
- spans written;
- represented key-checkpoint facts.

This makes the diagnostic a second implementation of the synthetic measurement path rather than a post-hoc transformation of the primary output.

## Added score observables

For every query the diagnostic evaluates two direct scores in addition to ordinary cleanup:

1. the correct target candidate;
2. one deterministic seed-hashed candidate guaranteed to differ from the target.

The probe distractor is selected from all non-target candidates by a deterministic hash of seed, relation-key index, and checkpoint. It is not selected according to its observed score.

For both score streams the diagnostic records:

- count;
- mean;
- population variance.

The null model predicts approximately:

`E[target_score] = 1`

`E[distractor_score] = 0`

with the predeclared target/distractor variances from `ValidityCapacityNullModel`.

The diagnostic also reports empirical-to-null variance ratios.

These ratios are descriptive. The query scores within one archive are correlated, and the null model explicitly assumes more independence than the real representation provides.

## Strongest distractor without candidate-wide rescoring

Ordinary cleanup already identifies the top two candidate scores.

If the target wins:

`max_distractor = second_score`.

If the target does not win:

`max_distractor = best_score`.

Thus the signed true margin can be computed from one extra target score without repeating all candidate correlations.

The additional seed-hashed distractor score exists specifically to estimate the single-distractor score moments predicted by the null model.

## No second seed-selection path

The diagnostic has no independent `research_v0` plan. It consumes `ValidityCapacityPlan::research_v0()` directly.

Therefore the frozen 27 cases and seeds `31001..=31005` remain the only research selection surface. Duplicate seeds are rejected before measurement.

## Interpretation

Useful comparisons include:

- empirical target mean versus `1`;
- empirical probe-distractor mean versus `0`;
- empirical target variance / null target variance;
- empirical distractor variance / null distractor variance;
- signed true-margin distribution versus ordinary winner-margin distribution;
- whether variance-ratio departures track codebook coherence, span length, or realized semantic-change density from the primary capacity observation.

A variance ratio near one is not by itself proof of the null model. Conversely, a large departure is useful evidence about which independence assumptions fail.

## Scientific boundary

This diagnostic does not add or remove research cases, does not tune thresholds, and does not define success after seeing data.

Its purpose is to make the existing preregistered sweep capable of testing the theory that was frozen before result interpretation.
