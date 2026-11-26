# MEL-GROUND-007 — Corpus temporal annotation reliability v1

**Canonical implementation:** [PR #13742](https://github.com/Luminous-Dynamics/symthaea/pull/13742)  
**Owner:** [Issue #13676](https://github.com/Luminous-Dynamics/symthaea/issues/13676)  
**Date:** 26 November 2026

## Measurement model

This protocol keeps three levels separate:

1. **Disposition reliability:** nominal Krippendorff alpha over Present, Absent, Ambiguous and NotEstimable item labels.
2. **Event-set agreement:** symmetric annotator-pair event F1, matched IoU and exact event-count agreement, using the production one-to-one matcher at a frozen increasing IoU threshold sweep.
3. **Boundary precision:** start/end absolute error distributions (MAE/P95) from matched pairs. This is a descriptive boundary comparison, not Krippendorff's unitizing alpha.

The coefficient is for categorical disposition only. Temporal boundaries are not discretized into arbitrary bins or merged into a synthetic consensus.

## Missing-rater design

Every item contains:
- a frozen roster of assigned annotators, represented by pseudonymous SHA-256 identity commitments;
- returned independent annotation records;
- one explicit nonresponse record for every assigned rater without a returned annotation.

The allowed nonresponse reasons are NotReturned, Declined, Withdrawn, TechnicalFailure and Incomplete. The roster is a partition: each assigned rater must occur exactly once as a returned annotation or a nonresponse. Duplicate or out-of-roster identities, overlaps, and unaccounted assignments are rejected.

Missing ratings are not a fifth disposition category and never become Absent. They are excluded from alpha's observed rating matrix but remain in the report's assigned/returned/missing denominators. Ambiguous and NotEstimable are returned categorical annotations and remain distinct categories in the alpha calculation.

The corpus report includes global assignment coverage, complete/partial item counts, missingness by reason, per-rater coverage across query items, and full per-item rosters, annotations and nonresponse records. Only pseudonymous identities are allowed in public records; the digest is an identity commitment, not proof of consent or authentication.

## Source-work uncertainty

Bootstrap resampling groups every query-item from a source-work lineage and resamples lineages as clusters. The observed annotator panel is treated as fixed. Thus the interval describes source-work variation for this rater panel; it does not generalize to an unseen annotator population.

A 30-source-work floor is an exploratory reporting warning, not a substitute for power analysis or representative sampling. Three synthetic source works are only a deterministic implementation test.

## Synthetic fixture

The fixture contains four items, three source-work lineages, nine returned labels and ten assigned rater-item slots:
- one assigned rater has a TechnicalFailure nonresponse;
- the other returned annotations include all four categorical dispositions;
- the nonresponse remains outside the disposition counts.

The nominal-alpha fixture expectations remain unchanged because only returned records enter alpha:
- observed disagreement 4/9;
- expected disagreement 3/4;
- nominal alpha 11/27, approximately 0.4074.

The CLI additionally asserts total assigned slots 10, returned records 9, missing slots 1 and coverage 0.9.

## Claim ceiling

This evaluator does not adjudicate ground truth, create consensus intervals, rank annotators, authenticate the roster/manifest, measure population reliability, or infer model confidence. Its source-work interval is conditional on the observed rater panel. Musician consent, identity custody, sampling and adjudication governance remain separate.

The source files and synthetic CLI fixture are qualified by the Rust 1.96 nested harness; no human study or model performance improvement is claimed.
