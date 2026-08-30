# PR validation notes

This file records the intended review boundary for the initial research-split tranche.

## In scope

- content-addressed train/calibration/evaluation assignments;
- multi-dimension group separation;
- forward-time embargo validation;
- structural leakage detection;
- explicit attributable separation evidence;
- deserialization through semantic validation + digest verification;
- overflow-safe temporal diagnostics.

## Out of scope

- automatic spatial-block construction;
- estimation of autocorrelation length;
- automatic buffer selection;
- geospatial coordinate geometry;
- random split generation;
- claims of statistical independence;
- Sentinel-specific product semantics;
- model training or scoring.

Those belong in follow-up domain adapters and experiments. Keeping them out of this crate allows the primitive to be reused across remote sensing, longitudinal studies, repeated subjects, robotics episodes, and other autocorrelated datasets.
