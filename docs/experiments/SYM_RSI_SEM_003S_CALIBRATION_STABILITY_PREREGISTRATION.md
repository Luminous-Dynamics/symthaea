# SYM-RSI-SEM-003S — Calibration Stability Audit

Status: **PREREGISTERED — NO SEM-003/SEM-003S MEASUREMENT CLAIM YET**

Parent SEM-003 implementation subject:

`df3eedea80cb21ffc7f4896b64ff6480ea8d2680`

Parent SEM-003 preregistration:

`af8162162e13a2b3c95fc70a0af9b62a3183ac4f`

Parent SEM-003 descriptive-estimator amendment:

`418f057387eacf2747edba6e8efaffca05d4b04c`

SEM-003S is a separate deterministic robustness audit. It does **not** change SEM-003's candidate-set rule, alpha, dataset, support boundary, eight primary criteria, canonical full-calibration disposition, or claim boundary.

No accepted SEM-003 benchmark result had been executed or observed when this stability protocol was frozen.

## Question

Is a SEM-003 result robust to modest deterministic changes in which labelled calibration examples determine the per-dimension nonconformity threshold?

The canonical SEM-003 result uses all 32 calibration queries per dimension. SEM-003S asks whether that result depends on one unusually favorable subset of those 32 observations.

The audit deliberately introduces **no new scalar quality cutoff**. Instead it reuses the already-preregistered SEM-003 success criteria under eight deterministic leave-one-fold-out recalibrations.

## Core distinction

```text
canonical SEM-003 PASS
    !=
calibration-stable SEM-003 PASS
```

A canonical PASS remains the canonical SEM-003 result even if SEM-003S later detects calibration fragility.

SEM-003S may restrict **promotion of the retrieval mechanism** but may not rewrite, rescue, reverse, or relabel the canonical SEM-003 receipt.

## Inputs

SEM-003S consumes:

1. one canonical `Sem3Receipt` produced by the frozen SEM-003 implementation subject;
2. the same deterministic SEM-003 candidate bank;
3. the same 32 calibration queries per dimension;
4. the same clean, ambiguous, unrelated, and OOD held-out query corpora.

The supplied canonical receipt must:

- pass its own receipt-integrity validation;
- bind the exact SEM-003 preregistration SHA;
- bind the exact SEM-003 Amendment 1 SHA;
- bind the exact primary implementation subject `df3eedea80cb21ffc7f4896b64ff6480ea8d2680`;
- bind the exact candidate/calibration/clean/ambiguous/unrelated/OOD corpus digests reconstructed by the stability auditor.

Any mismatch is an integrity failure, not a robustness result.

## Protected-partition boundary

SEM-003S is synthetic semantic-memory evaluation only.

It must not execute, inspect, or consume protected SYM-RSI partitions:

- 201–204;
- 301–304;
- 401–404;
- 1201–1204.

## Deterministic fold construction

For each context dimension independently:

1. take the exact 32 SEM-003 calibration queries for that dimension;
2. compute each query's collision-resistant `ContextIdentity` from the query context itself;
3. sort queries ascending by `ContextIdentity.exact_digest` byte order;
4. assign canonical sorted position `i` to fold `i mod 8`.

This produces exactly eight folds containing exactly four calibration queries each.

Fold construction must not use:

- HDC similarity;
- true-source nonconformity;
- candidate-set size;
- clean/ambiguous/unrelated/OOD outcomes;
- SEM-003 disposition;
- any metric observed after semantic encoding.

Changing fold count, ordering rule, or assignment rule requires a new SEM-003S protocol identity.

## Fold-specific recalibration

For fold `f in {0,...,7}` and dimension `D`:

- exclude exactly the four calibration queries assigned to fold `f`;
- retain the other 28 calibration queries;
- compute each retained true-source nonconformity exactly as SEM-003 does:

`a_i = 1 - similarity(query_i, true_source_i)`

- compute `Q_(D,f)` using the unchanged SEM-003 split-calibration rule:

`k = ceil((n + 1) * (1 - alpha))`

with:

- `n = 28`;
- `alpha = 0.10`;
- index clamped to `[1,n]`;
- threshold equal to sorted score `k - 1`.

No fold-specific retuning, smoothing, interpolation, averaging, margin correction, or empirical-null term is allowed.

## Held-out evaluation under each fold

For each fold, run the unchanged SEM-003 candidate-set rule over the exact frozen held-out corpora using the fold-specific thresholds `Q_(D,f)`:

`candidate_set_f(q) = { c : dimension(c)=dimension(q) and 1-similarity(q,c) <= Q_(D,f) }`

Operational OOD behavior remains unchanged:

- support validation happens before candidate-set construction;
- unsupported OOD queries return no operational set;
- OOD rejection is still required to equal 1.0.

The same eight SEM-003 primary criteria are then recomputed for each fold:

1. aggregate clean target coverage `>= 0.90`;
2. every dimension clean target coverage `>= 0.85`;
3. aggregate clean mean set size `<= 2.0`;
4. ambiguous at-least-one-parent inclusion `>= 0.90`;
5. ambiguous dual-parent inclusion `>= 0.70`;
6. ambiguous singleton forced-choice rate `<= 0.20`;
7. unrelated non-empty set rate `<= 0.10`;
8. OOD support rejection rate `= 1.0`.

Fold disposition uses the exact SEM-003 disposition logic. SEM-003S does not define alternate pass thresholds.

## Stability disposition

SEM-003S has four dispositions.

### `StablePass`

Requires:

- canonical SEM-003 disposition is `Pass`;
- all eight fold-specific dispositions are `Pass`;
- all lineage/corpus/integrity checks pass.

This is the only SEM-003S disposition that authorizes describing the canonical SEM-003 result as **stable under the frozen deterministic calibration perturbation audit**.

### `PrimaryPassCalibrationFragile`

Canonical SEM-003 disposition is `Pass`, but at least one fold-specific disposition is not `Pass`.

The canonical PASS remains unchanged, but live-default promotion should remain blocked until the calibration dependence is understood and a new preregistered mechanism/version is evaluated.

### `PrimaryNotPass`

Canonical SEM-003 disposition is not `Pass`.

SEM-003S may still report fold diagnostics, but cannot rescue the primary result.

### `IntegrityFailure`

Any protocol, subject, corpus, receipt, fold-construction, threshold-reconstruction, support, or accounting mismatch.

Integrity failure cannot be interpreted as evidence for or against semantic retrieval quality.

## Omitted-fold calibration diagnostics

For each `(dimension, fold)` report descriptively:

- fold ID;
- included calibration count = 28;
- omitted calibration count = 4;
- fold-specific threshold `Q_(D,f)`;
- omitted-fold true-source coverage using `Q_(D,f)`;
- minimum/maximum/mean omitted true-source nonconformity.

These diagnostics do not affect SEM-003S disposition except through the fold-specific held-out SEM-003 criteria above.

They exist to distinguish threshold instability from held-out candidate-set instability.

## Threshold stability summaries

For each dimension report across the eight recalibrations:

- canonical full-calibration `Q_D`;
- minimum fold threshold;
- maximum fold threshold;
- mean fold threshold;
- absolute threshold range;
- maximum absolute deviation from canonical `Q_D`.

No threshold-range cutoff is introduced in v1.

## Candidate-set drift diagnostics

For each held-out supported query and fold, compare the fold-specific candidate set with the canonical full-calibration candidate set for the same query.

Report separately for clean, ambiguous, and unrelated regimes:

- exact-set agreement rate;
- mean symmetric-difference size;
- maximum symmetric-difference size;
- mean absolute set-size difference;
- maximum absolute set-size difference.

Set comparison uses exact `ContextIdentity` membership only.

These drift quantities are descriptive. They do not override the fold disposition rule.

OOD queries have no operational candidate set and therefore do not participate in candidate-set drift summaries.

## Fold receipt surface

Each fold receipt must bind at least:

- fold ID;
- exact fold-membership digest;
- all three fold-specific `Q_(D,f)` thresholds;
- all eight SEM-003 primary endpoint values;
- all eight guard booleans;
- fold disposition;
- candidate-set drift diagnostics;
- omitted-fold calibration diagnostics;
- fold evidence digest.

## Top-level SEM-003S receipt

The top-level receipt must bind:

- SEM-003S schema/version;
- this preregistration SHA;
- parent SEM-003 preregistration SHA;
- parent SEM-003 Amendment 1 SHA;
- exact primary implementation subject SHA;
- exact stability implementation subject SHA;
- complete canonical primary `Sem3Receipt` evidence digest;
- candidate/calibration/held-out corpus digests;
- all eight fold receipts in ascending fold order;
- per-dimension threshold stability summaries;
- overall stability disposition;
- complete top-level evidence digest.

Mutation of any fold membership, threshold, metric, drift diagnostic, primary lineage field, or disposition must change the top-level digest.

## No formal statistical guarantee

SEM-003S is a deterministic finite-corpus sensitivity analysis.

It does not establish:

- distribution-free conformal coverage;
- bootstrap confidence intervals;
- sampling-distribution validity;
- exchangeability;
- population-level calibration;
- robustness to arbitrary dataset shift.

Its claim is only about sensitivity to the exact frozen eight-way calibration perturbation procedure.

## Promotion boundary

A future live semantic-memory integration should require, at minimum:

- canonical SEM-003 `Pass`;
- SEM-003S `StablePass`;
- exact provenance resolution remains mandatory;
- candidate sets remain search-only;
- no set membership or set size becomes evidence/confidence authority.

Even `StablePass` does not authorize:

- belief-confidence promotion;
- empirical validation;
- generated-evidence promotion;
- action authority;
- scientific truth claims;
- consciousness claims;
- general recursive-self-improvement claims.

## Failure discipline

After SEM-003/SEM-003S outcomes are observed, forbidden under v1:

- changing fold assignment;
- dropping an inconvenient fold;
- changing alpha;
- changing the quantile rule;
- changing primary success criteria;
- tuning perturbation magnitude from fold results;
- editing candidate/query corpora;
- replacing `all eight folds pass` with a majority rule.

A demonstrated representation or calibration defect must be preserved and addressed under a new protocol/version and new untouched evaluation lineage.