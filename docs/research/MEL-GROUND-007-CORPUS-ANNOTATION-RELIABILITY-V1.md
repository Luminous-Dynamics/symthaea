# MEL-GROUND-007 — Corpus-level multi-annotator reliability v1

**Status:** synthetic-first reliability adapter; not a musician-population result  
**Date:** 26 November 2026  
**Implementation:** `tools/mel-temporal-grounding-qualifier/src/temporal_reliability.rs`

## 1. Two distinct kinds of agreement

This profile intentionally separates:

1. **Disposition reliability** — whether annotators classify an item as Present, Absent, Ambiguous or NotEstimable.
2. **Temporal interval agreement** — whether two Present annotations mark the same event occurrences at comparable times.

The first is measured with nominal Krippendorff alpha over item-level categorical ratings. The second is reported as a symmetric pairwise event F1, matched IoU and occurrence-count agreement at a fixed set of IoU thresholds. They are not collapsed into one score.

The implementation does **not** calculate Krippendorff's alpha for unitizing (α_U). That is a separate reliability family for the boundaries of units on a continuous timeline. The current temporal comparison is an explicitly descriptive use of the existing production scorer.

## 2. Nominal disposition alpha

For each item with (m_u ge 2) independent ratings, the evaluator forms ordered coder-pair coincidences with weight (1/(m_u-1)), excluding self-pairs. Ratings are the four explicit states:

- Present
- Absent
- Ambiguous
- NotEstimable

Ambiguous and NotEstimable remain categories in this disposition reliability matrix; they are not silently converted to absent or dropped. The chance-corrected coefficient is:

[
alpha = 1 - rac{D_o}{D_e}
]

where (D_o) is observed nominal disagreement from the coincidence matrix and (D_e) is expected disagreement from the pooled category marginals. If expected disagreement is zero (no category variation), alpha is reported as `null`, not invented as 0 or 1.

The report includes raw rating count, pairable item count, category counts, observed disagreement, expected disagreement and alpha.

## 3. Temporal interval agreement

For each item and each pair of Present annotators, the evaluator uses the **production temporal-grounding scorer** in both directions:

- left annotation as truth, right as prediction;
- right annotation as truth, left as prediction.

The two F1 values are averaged to form a symmetric pair score. The process is repeated at fixed minimum-IoU thresholds 0.25, 0.50, 0.75 and 0.90. It reports the number of Present/Present pairs, mean symmetric event F1, matched interval IoU, and exact occurrence-count agreement.

This deliberately preserves two orthogonal failure modes:
- same occurrence count but different event localization;
- wrong occurrence count even if one event overlaps well.

It does not create a merged reference or run a second interval matcher.

## 4. Synthetic acceptance fixture

The committed four-item fixture contains nine ratings across three source-work clusters. Its hand-calculated nominal disposition result is:

- ratings: 9;
- category counts: Present 3, Absent 4, Ambiguous 2, NotEstimable 0;
- observed disagreement: (4/9 approx 0.444444);
- expected disagreement: (13/18 approx 0.722222);
- nominal alpha: (5/13 approx 0.384615).

Three Present annotators on one item contribute three interval pairs. Their intervals are constructed so symmetric event F1 is 1.0 at IoU thresholds 0.25, 0.50 and 0.75, and (1/3) at 0.90. This verifies threshold sensitivity and pairwise production-scorer reuse.

The fixture is a deterministic arithmetic/control test only. It does not estimate real annotation reliability.

## 5. Claim boundaries

The report always declares:
- `consensus_ground_truth_created = false`;
- `model_performance_scored = false`;
- `population_confidence_interval_computed = false`;
- `source_work_bootstrap_computed = false`;
- `unitizing_alpha_computed = false`.

A small corpus-level alpha is descriptive of the supplied annotation sample. It is not a confidence interval and does not prove the panel represents a musician population. Confirmatory inference needs a preregistered sampling plan, adequate independent source-work clusters and a separately designed source-work bootstrap.

The corpus manifest and annotator digests are identifiers, not authentication. Human identity, consent, licensing and adjudication remain governed by the human annotation programme.

## 6. Research basis

Krippendorff's nominal alpha and alpha for unitizing continuous data are distinct methods. This implementation only implements nominal alpha for the categorical disposition task, and keeps temporal boundary agreement as a separately named score profile.

References:
- Krippendorff, *On the Reliability of Unitizing Continuous Data* (1995): https://doi.org/10.2307/271061
- Krippendorff, Mathet, Bouvry & Widlöcher, *On the reliability of unitizing textual continua: Further developments* (2016): https://doi.org/10.1007/s11135-015-0266-1
- qdaR guide to unitizing alpha and its distinction from categorical alpha: https://qdar.zotqda.org/latest/en/guide/unitizing.html
