# SYM-RSI-SEM-003 — Preregistration Amendment 1

Status: **FROZEN BEFORE SEM-003 MEASUREMENT**

Parent preregistration:

`af8162162e13a2b3c95fc70a0af9b62a3183ac4f`

This amendment clarifies only secondary/descriptive estimators that the original preregistration named but did not fully specify. It does **not** change any SEM-003 primary success criterion, dataset size, context dimension, alpha, candidate-set rule, support boundary, or claim boundary.

No SEM-003 benchmark result had been executed or observed when this clarification was authored. Earlier unregistered harness drafts are construction history only and are not measurement evidence.

## Why this amendment exists

Static implementation review found that the original document required reporting:

- clean/ambiguous/unrelated set-size histograms;
- top-1/top-2 similarity margin by regime;
- empirical-null retrieval specificity by regime;
- exact tie counts;
- raw OOD similarity diagnostics;

but did not freeze the exact summary/calibration mechanics for those descriptive quantities.

Leaving those choices implicit would create unnecessary researcher degrees of freedom even though none of them controls the primary disposition. This amendment freezes them before any SEM-003 measurement is accepted.

## Frozen empirical-null specificity diagnostic

Specificity is descriptive only and has **zero candidate-set admission authority** in SEM-003.

For each numeric context dimension, build one `SemanticNullCalibration` from the exact SEM-003 candidate-memory bank for that dimension using:

- `max_reference_contexts_per_dimension = 64`;
- `min_null_pairs_per_dimension = 128`;
- the exact existing `SemanticContextEncoder` configuration inherited from the parent preregistration.

The calibration corpus consists only of candidate-memory contexts. No calibration query, clean query, ambiguous query, unrelated query, OOD query, held-out label, candidate-set threshold, or observed benchmark outcome may contribute to this empirical null.

For each supported evaluation query:

1. find the raw top-1 candidate similarity under the same candidate bank used by set construction;
2. assess that similarity against the matching-dimensional empirical null;
3. record `retrieval_specificity` as the existing conservative empirical tail-rank diagnostic.

This value remains explicitly **not** a p-value, probability, confidence score, evidence weight, or admission threshold.

OOD queries may have raw top-1 similarity and empirical specificity computed only as post-rejection diagnostics. Their operational result remains `UnsupportedQuery` before candidate-set construction.

## Frozen set-size histograms

For each of these held-out regimes independently:

- clean;
- ambiguous;
- unrelated;

report an aggregate integer histogram over returned candidate-set size.

A histogram is represented canonically as ascending `(set_size, count)` bins. Zero-count bins are omitted. Counts must sum exactly to the frozen number of queries in that regime.

OOD queries have no operational candidate set and therefore do not contribute to a set-size histogram.

## Frozen margin summaries

For each supported held-out regime independently report:

- mean top-1/top-2 similarity margin;
- p95 top-1/top-2 similarity margin;
- exact tie count, where an exact tie means the top two `f32` similarity values have identical IEEE-754 bit patterns.

When fewer than two candidates exist in a bank, margin is undefined; however the frozen SEM-003 candidate bank always contains 64 same-dimensional candidates, so this case is an integrity failure rather than a benchmark outcome.

Margin remains descriptive in SEM-003. It does not alter set membership or disposition.

## Frozen specificity summaries

For clean, ambiguous, and unrelated held-out regimes independently report:

- mean retrieval specificity;
- p05 retrieval specificity;
- p50 retrieval specificity;
- p95 retrieval specificity.

Percentiles use deterministic nearest-rank indexing over values sorted ascending:

`rank = ceil(N * p)`

clamped to `[1, N]`, returning `sorted[rank - 1]`.

For OOD diagnostics report:

- mean raw top-1 similarity;
- maximum raw top-1 similarity;
- mean empirical retrieval specificity;
- maximum empirical retrieval specificity;
- support rejection count/rate.

These OOD values remain diagnostic after operational rejection and cannot rescue or improve SEM-003 disposition.

## Exact tie reporting

Exact top-1/top-2 ties are counted separately for:

- clean;
- ambiguous;
- unrelated;
- OOD diagnostic ranking.

Tie counts are descriptive only. No post-hoc tie threshold is introduced.

## Receipt binding

The eventual SEM-003 receipt-integrity digest must bind, in addition to the parent preregistration requirements:

- this amendment identity;
- the semantic-null calibration identity;
- all three set-size histograms;
- all margin summary fields and exact tie counts;
- all specificity summary fields;
- OOD raw-similarity/specificity diagnostic summaries.

A future public protocol wrapper must bind both the parent preregistration SHA and this amendment SHA before a SEM-003 receipt may be described as protocol-bound.

## Unchanged primary criteria

The original eight primary criteria remain byte-for-byte unchanged in meaning:

1. aggregate clean target coverage `>= 0.90`;
2. every dimension clean target coverage `>= 0.85`;
3. aggregate clean mean set size `<= 2.0`;
4. ambiguous at-least-one-parent inclusion `>= 0.90`;
5. ambiguous dual-parent inclusion `>= 0.70`;
6. ambiguous singleton forced-choice rate `<= 0.20`;
7. unrelated non-empty set rate `<= 0.10`;
8. OOD support rejection rate `= 1.0`.

No descriptive metric introduced here may be promoted into a primary endpoint after results are observed.

## Claim boundary unchanged

SEM-003 remains a synthetic retrieval experiment only.

Candidate-set membership, similarity, specificity, margin, support status, set size, and any combination of those quantities do not establish truth, empirical validation, confidence authority, consciousness, or general recursive self-improvement.
