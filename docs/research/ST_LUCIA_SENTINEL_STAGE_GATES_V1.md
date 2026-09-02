# St. Lucia Sentinel Pilot v1 — empirical stage gates

This document prevents stronger scientific claims from being inferred from a weaker completed stage.

## R0 — real-data provenance and deterministic perception plumbing

R0 asks whether the selected real Sentinel products can move through a reproducible, content-addressed perception pipeline.

Before R0 begins, product IDs must be selected only by the preregistered discovery protocol and executable runner in:

```text
docs/research/ST_LUCIA_SENTINEL_PILOT_V1.md
docs/research/ST_LUCIA_STAC_DISCOVERY_EXECUTION_V1.md
scripts/research/st_lucia_stac_discovery.py
```

The discovery run stops before asset download for a human/evidence review of protocol identity, raw page hashes, exhaustive pagination and deterministic candidate ordering.

R0 may demonstrate:

- exact catalogue/product discovery provenance;
- exact source-asset hashes;
- canonical raster decoding;
- exact pixel/window provenance;
- explicit dtype/endian/band/NoData/mask semantics;
- deterministic optical/SAR feature construction;
- cross-run reproduction of those outputs.

R0 does **not** establish wetland classification accuracy, forecasting skill, semantic compression benefit, or subsurface inference.

## R1 — independently labelled perception / classification

R1 requires an independently attributable reference-label lineage. Sentinel-derived pseudo-labels cannot be used to claim independent Sentinel classification accuracy.

If no defensible reference-label lineage exists, report:

```text
reference-label-lineage-insufficient
```

rather than promoting R0 outputs into a classification claim.

Where classification is supported, compare against the frozen conventional baseline floor including RF and normally SVM/CART/KNN, with Training -> Calibration/model selection -> sealed Evaluation separation.

## R2 — temporal wetland forecasting

R2 requires a multi-timepoint design in which a model receives observations through time `t`, issues and commits a forecast for `t+h`, and only then receives the future verification observation.

Same-time land-cover classification is not forecasting.

R2 reports proper score, calibration, coverage/abstention, baseline comparisons and replication on new time/geography.

## R3 — semantic downlink

R3 is a separate communication-efficiency experiment.

The minimum contest remains:

```text
A conventional codec
B A + ordinary cloud/change/ROI prioritisation
C B + Symthaea/HDC semantic prioritisation
```

C must beat B on preregistered mission-relevant information per transmitted byte/joule. A perception or forecasting win does not imply an R3 win.

## Promotion rule

Each stage may advance only on evidence appropriate to that stage. A lower-stage success must never be described with a higher-stage claim.
