# symthaea-forecast-calibration

Append-only forecast verification and calibration contracts for Planetary Perception.

## Core rule

Predictions earn credibility by surviving contact with later evidence.

A forecast records its model/version, target, issue time, validity window, prediction, assumptions, and optional artifact digest **before** the target window is evaluated. Later verification refers to that immutable forecast id and requires explicit `EvidenceStage::Verification` support.

There is no delete/replace operation for bad forecasts in this crate.

## Metrics

Numeric forecasts retain:

- signed error;
- absolute error;
- squared error;
- interval hit/miss when an interval was supplied.

Binary probability forecasts use Brier score.

Model/target reports expose:

- total forecasts;
- verified forecasts;
- pending forecasts;
- MAE / RMSE for numeric forecasts;
- empirical interval coverage;
- mean Brier score for binary forecasts.

There is intentionally no composite `trust_score`.

## Boundaries

- A pending forecast is not silently excluded from history.
- A missed interval is retained alongside a hit.
- Verification evidence must be explicitly marked as verification rather than an ordinary supporting observation.
- Calibration describes historical predictive performance; it does not grant authority to act.
- Different targets are calibrated separately so incompatible units/phenomena are not averaged together.

## Future work

A later tranche can bind these reports into `symthaea-model-ensemble` applicability/calibration views and compare calibration across regimes, regions, horizons, and out-of-distribution transitions.
