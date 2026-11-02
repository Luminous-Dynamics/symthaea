# HARTH subject-held-out result receipts v0.1

Tracking: [#9850](https://github.com/Luminous-Dynamics/symthaea/issues/9850)  
Implementation branch: `research/harth-loso-receipt-v0.1`  
Parent: [HARTH streaming ingestion PR #9847](https://github.com/Luminous-Dynamics/symthaea/pull/9847)

## Purpose

A streaming parser preserves source-family identity. This module makes the next step enforceable: a HARTH result must be based on exactly 22 participant-held-out folds, retain per-class confusion counts, and report participant-level macro-F1 rather than pooled-window accuracy.

It does not implement a classifier or execute SelfPAB, HART or Symthaea.

## Label tracks

`HarthLabelTrack` defines two non-interchangeable tasks:

- **Native12:** the twelve documented HARTH activity codes remain distinct.
- **SelfPabComparable10:** the two inactive cycling classes are merged into their active counterparts: 130→13 and 140→14.

The selected label track is bound into the receipt digest. A 12-class score must never be presented as a direct comparison against SelfPAB's ten-class result.

## Fold contract

The receipt requires:
- exactly 22 distinct test-subject source-family digests;
- one held-out fold for every subject digest;
- exactly 21 training subject digests in every fold;
- each fold's training set exactly equals the other 21 test-subject identities;
- no test/train overlap, duplicate identity, missing held-out subject or duplicate fold;
- a square row-major confusion matrix whose dimension is fixed by the label track;
- at least one scored test sample per participant.

All 22 fold records and each training identity list are sorted canonically before receipt creation.

## Metric semantics

Confusion matrix rows are true class and columns are predicted class. For each declared class:

`F1 = 2TP / (2TP + FP + FN)`

A class absent from both truth and prediction has an undefined per-class F1 and is excluded from that participant's macro average. A class predicted but never present in truth remains in the macro average with F1 = 0. This matches macro-F1 over the observed label union; the precise scoring semantics are explicit rather than hidden.

The receipt reports:
- per-subject frame count;
- per-class F1 and true support;
- supported class count;
- participant macro-F1;
- mean of the 22 participant macro-F1 scores;
- descriptive standard error across the 22 subject scores;
- worst participant macro-F1;
- total test sample count.

The mean is an unweighted participant mean, not a pooled frame-weighted score. The standard error is descriptive; a candidate-vs-baseline comparison must use fold-matched subject deltas and a paired confidence interval/test. Do not infer significance from overlapping or non-overlapping marginal confidence intervals.

## Provenance

The canonical receipt binds:
- parent dataset-manifest SHA-256;
- preprocessing SHA-256;
- label-track identity;
- candidate ID, exact source revision, dependency-lock digest, built-artifact digest and configuration digest;
- all 22 sorted subject fold identities, training identities and confusion counts;
- recomputed fold aggregate and receipt digest.

A receipt hash is an integrity commitment, not proof that a trusted evaluator generated the confusion counts or that test labels were isolated. Confirmatory model evaluation still requires the separate candidate/evaluator process boundary specified by the edge benchmark bridge.

## Comparator alignment

### SelfPAB

SelfPAB (Logacjov et al., 2024) reports 81.3 ± 1.3% mean F1 on HARTH v1.2 with 22-subject LOSO and a ten-class mapping that merges inactive cycling labels. Its predictions are produced on one-second frames with half-second overlap. A direct comparison requires the matching label mapping, cadence, aggregation, preprocessing and subject split.

### HART

HART (2026) reports aggregate results across 14 datasets (8 in-distribution and 6 OOD). Its aggregate F1 values are not HARTH-specific. Pin its pretraining source inventory and subject overlap before assigning a HARTH run to an ID or OOD lane.

The external papers' scores are reference values, not results reproduced by this code.

## Tests and status

Eleven synthetic tests cover label mapping, valid 22-fold receipt creation, 10-class dimensions, test/train leakage, duplicate held-out subjects, missing training subjects, invalid matrix shape/empty denominator, confusion-derived F1, receipt tamper detection and canonical sorting.

**Execution status:** tests are authored but have not been compiled or run. No model, HARTH data archive or baseline has been executed. No accuracy or SOTA claim is made.
