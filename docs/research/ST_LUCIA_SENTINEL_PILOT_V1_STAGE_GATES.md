# St. Lucia Sentinel Pilot v1 — empirical stage gates

Status: **pre-run amendment**

This document narrows claim scope for the first real Sentinel pilot. A result may be promoted only to the highest gate whose evidence requirements are actually satisfied.

## R0 — real-data provenance and deterministic perception plumbing

Inputs:

- one deterministically selected Sentinel-2 L2A item;
- one deterministically paired Sentinel-1 GRD item where available.

R0 may establish only:

- reproducible catalogue discovery;
- exact source/asset identity and content digests;
- deterministic provider decoding into a canonical raster payload;
- exact CRS/grid/pixel support;
- explicit NoData and mask semantics;
- deterministic reviewed optical/SAR feature computation;
- cross-run byte and feature reproducibility;
- resource/runtime measurements.

R0 does **not** establish wetland-classification accuracy or future-state prediction unless an independent reference-outcome lineage is separately frozen and qualified.

## R1 — independently labeled perception benchmark

R1 requires attributable reference labels, field evidence, or another prospectively justified reference dataset.

Labels produced from the same Sentinel preprocessing chain may be useful diagnostics but must not be presented as independent ground truth.

If adequate independent labels are unavailable, record:

```text
reference-label-lineage-insufficient
```

and stop before making an accuracy claim.

For a classification-style R1:

- enforce Training -> Calibration/model selection -> sealed Evaluation;
- enforce spatial/acquisition/time leakage controls;
- include a literature-strength RF baseline and normally SVM/CART/KNN;
- report overall accuracy, per-class precision/recall/F1, confusion matrix, Quantity Disagreement and Allocation Disagreement where applicable;
- preserve wetland boundary/transitional zones as an explicit diagnostic slice;
- retain null and conventional-model-win outcomes.

## R2 — temporal wetland forecasting

Same-time land-cover classification is not forecasting.

R2 requires a time-ordered observation series and a rolling-origin/prequential design:

```text
history through t
    ↓
forecast t+h
    ↓
commit
    ↓
reveal future observation
    ↓
score
```

Requirements include:

- explicit forecast horizon;
- forecast-before-reveal ordering;
- proper scoring or task-specific preregistered forecast metrics;
- coverage/abstention reporting;
- out-of-time Evaluation;
- calibration;
- direct replication on a distinct temporal and/or geographic lineage.

## R3 — semantic downlink

Semantic downlink is a separate scientific experiment.

The primary question is whether semantic prioritization increases mission-relevant information per transmitted byte/joule under a frozen conventional baseline ladder:

```text
A = conventional codec
B = A + ordinary cloud/change/ROI prioritization
C = B + Symthaea/HDC semantic prioritization
```

C must beat B to receive scientific credit.

A successful R0/R1/R2 result does not imply R3 succeeds.

## Claim ladder

The following implications are forbidden:

```text
reproducible bytes/features
    != accurate wetland classification

accurate wetland classification
    != future-state prediction

future-state prediction
    != improved satellite compression
```

Each transition requires its own frozen protocol, evidence and evaluation.
