# symthaea-model-ensemble

Evidence-preserving comparison of multiple models for one counterfactual outcome.

## Core rule

An ensemble is not a license to average disagreement into certainty.

Each model prediction retains:

- model family, version, and optional artifact digest;
- scenario and outcome-dimension identity;
- point/interval estimate and units;
- applicability / OOD status;
- calibration summary and validation-dataset digest;
- assumptions and source references.

Pairwise agreement is evaluated only under an explicit caller-supplied `AgreementPolicy`. There is no default tolerance, hidden model weight, synthesized consensus prediction, or universal trust score.

## OOD handling

Out-of-distribution, near-boundary, and unknown-applicability models remain visible in the full report. `in_distribution_predictions()` is only a filtered review view; it does not erase the other predictions.

## Calibration

Calibration statistics are evidence about prior predictive behavior. They are not permission to treat a model as correct in a new regime. Calibration retains its validation dataset digest and sample count.

## Non-scope

This crate does not:

- average model predictions;
- learn ensemble weights;
- choose a winning model;
- convert calibration into authority;
- rank interventions;
- execute observations or actions.

Later experiments may compare explicit ensemble algorithms, but any synthesized prediction must be a new evidence-bearing object with method/version/validation rather than a silent property of this report.
