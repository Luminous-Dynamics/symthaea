# Aviation Forecast Evaluation Contract

**Status:** Initial Rust metrics implementation on a research branch; compile/test execution pending.  
**Tracking:** [AVIATION-ENTERPRISE-014 / issue #8862](https://github.com/Luminous-Dynamics/symthaea/issues/8862).  
**Implementation:** \`aviation_forecast_metrics.rs\` in \`symthaea-operations-research\`.  
**Claim ceiling:** reproducible descriptive scoring and timing-contract checks. This does not demonstrate causal policy value, airline savings, or operational safety.

## The first paradigm shift: make the prediction time part of the task definition

A forecast benchmark is not operationally meaningful until it states when the prediction is made relative to the event it predicts. Two tasks can use the same data and report the same RMSE while answering materially different questions:

- **Prospective operational forecast:** the prediction and every input feature are available no later than the start of the target window. For taxi-out duration, that means the forecast decision occurs before or at the off-block/start time, not after the aircraft has taken off.
- **Retrospective challenge reconstruction:** the task may reveal the target-window endpoint at prediction/ranking time. This can be a valid retrospective benchmark, but it is not evidence that the same output could have informed a pre-event decision.

This distinction is directly relevant to the PRC Data Challenge 2026. The official challenge page states that the model is trained on movements from 2025 and ranked using RMSE for January and July 2026. The openly published \`moe-a-m/prc2026\` solution describes its ranking-time setup as having take-off known while estimating the recorded off-block time. That is a legitimate challenge setup to reproduce faithfully, but it must be labelled as retrospective reconstruction rather than an operational prediction available before taxi-out starts.

Sources:
- Official challenge scope and metric: https://prc-data-challenge-2026.netlify.app/index.html
- Open implementation and its stated ranking-time features/results: https://github.com/moe-a-m/prc2026

## Public comparison point — reported, not independently verified here

The open solution reports:
- Public-board RMSE: **268.25 seconds** for its v11 recipe.
- Local January 2025 holdout RMSE: **334.2 seconds**.
- Local July 2025 holdout RMSE: **300.8 seconds**.
- Leave-one-month-out validation over all of 2025: **250.06 seconds**.

These figures refer to different evaluation partitions and should not be compared as if they measured the same records. The public-board result is a reported comparator, not a universal state-of-the-art number. A Symthaea claim requires reproducing target construction, features, preprocessing, split, and evaluator on the same data.

## Shared record contract

Each scored row carries:
- a unique record ID and a dependence-cluster ID such as disruption episode or airport-day;
- a cohort ID, usually airport or airport-season;
- one explicit Validation or Test partition;
- decision time, target-window start/end, and target-observation time;
- actual and predicted duration in seconds;
- an explicitly declared feature schema and each feature's availability timestamp; and
- optional central prediction interval with a nominal coverage level.

The module requires a run context with syntactically valid digests for the dataset snapshot, prediction artifact, model/configuration, feature schema, evaluation protocol, and evaluator commit; it preserves those identifiers in the report. It rejects Training/Calibration rows, mixed partitions, duplicate record IDs, invalid chronology, target windows inconsistent with the labelled duration by more than one second, predictions issued after target start in prospective mode, and any model input with missing or future availability. The caller must compute these digests from the actual files; syntax checks alone do not prove the digest matches a file. A declared intercept-only baseline may use an explicitly declared empty feature schema; an undeclared schema is an error.

The one-second target-window tolerance accommodates timestamp precision differences. If a data source's target is not directly the difference between the specified target-window endpoints, define a different target contract rather than silently relaxing this check.

## Metrics returned

For the overall test partition and each cohort (sorted deterministically by ID), the scorer returns:
- RMSE in seconds — comparable to the PRC challenge's primary metric when the same evaluation records and target definition are used;
- MAE;
- mean signed error, where positive means over-prediction;
- P90 absolute error using linear/type-7 interpolation;
- when an interval is supplied consistently for every row: nominal coverage, empirical coverage, mean interval width, and mean central-interval Winkler score.

The interval score uses the nominal central coverage \(1-\alpha\): interval width plus a penalty of \(2/\alpha\) times the miss distance when the observation lies outside the interval. Lower is better. Empirical coverage alone is not enough: a uselessly wide interval can cover nearly everything, so report coverage and width together.

Cohort results prevent a large airport from hiding failure at a smaller or more disrupted airport. Cluster counts are exposed so a later inference layer can resample whole disruption episodes/days rather than pretending correlated flights are independent. This module does not yet compute confidence intervals or run cluster bootstrap inference.

## Evaluation protocol

1. Freeze the exact data snapshot, licensing/access conditions, schema, target construction, feature availability semantics, and output partition.
2. Record the model's decision time independently from the target endpoint.
3. For a prospective operational claim, require prediction time no later than target-window start and reject all unavailable features.
4. Keep Validation and Test scoring separate. A Test score does not prove that the test set was never viewed; use the existing calibration/split-integrity and no-test-tuning audits.
5. Compare global median, airport × local-time/day/season historical baselines, persistence where meaningful, regularized linear regression, LightGBM/CatBoost, and the published open challenge recipe where reproduction and licensing permit.
6. Publish overall and per-airport metrics, errors by season/weather/capacity regime where supported, interval coverage/width, and the exact scored record IDs or a digest-backed score ledger.
7. Do not infer decision utility from RMSE. Feed the time-valid forecasts into the separate event-stream/recovery harness tracked by issue #8603 and evaluate policy value under matched scenarios and compute budgets.

## Test coverage defined

The module defines unit tests for provenance binding/invalid hashes, point metrics, per-cohort sorting, prospective-versus-retrospective timing, future/missing feature times, target-field leakage, partition separation, duplicate IDs, interval coverage/width/Winkler score, mixed interval outputs, explicit intercept-only schema, non-finite predictions, and target-window consistency.

A separately compiled test oracle recomputes RMSE, MAE, bias, and P90 absolute error from raw fixtures without using production metric helpers. It also checks the prospective timing gate, feature-time leakage rejection, deterministic cohort ordering, and partition separation.

These are test definitions, not passing evidence. The code must be formatted and built with the exact tested commit before any claim that the tests pass. No real-data experiment has been executed as part of this change.
