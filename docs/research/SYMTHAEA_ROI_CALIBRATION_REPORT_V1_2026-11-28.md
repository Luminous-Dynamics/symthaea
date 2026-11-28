# Symthaea ROI calibration report v1 — 2026-11-28

## Why this exists

The current ROI expected-value ranker accepts caller-supplied calibration metadata (model digest, task family, sample count, held-out ECE and a report digest). That is not enough by itself: a caller could provide an ECE that does not agree with the underlying reliability evidence, reuse the same dataset for calibration and evaluation, or bind a report to a different model.

This module adds an internal-consistency gate before ROI estimates enter the existing ranker.

## Implemented

Source: `crates/domains/symthaea-soma/src/roi_calibration.rs`.

The report records:
- model and task-family identity;
- distinct calibration and held-out evaluation dataset digests;
- distinct split digests;
- calibration and evaluation sample counts;
- reliability bins with sample count, mean confidence, observed accuracy and mean Brier contribution;
- reported ECE and Brier score;
- a domain-separated digest over the complete report.

The verifier recomputes:

```
ECE = sum_i(n_i * abs(confidence_i - accuracy_i)) / N
Brier = sum_i(n_i * mean_brier_i) / N
```

The report is rejected for duplicate calibration/evaluation dataset or split identities, insufficient sample counts, invalid bins, bin-count mismatch, inconsistent metrics or digest mismatch.

The binding function then requires each ROI estimate to match exactly one verified report by report ID, report digest, model digest, task family, calibration sample count and ECE. Missing or ambiguous reports fail closed.

## Important limits

- A report digest establishes deterministic content identity, not that its data are authentic, licensed, representative or honestly collected.
- This report does not implement bootstrap confidence intervals, subgroup calibration, temporal drift detection or adaptive recalibration.
- ECE is binning-sensitive and can hide local or subgroup miscalibration. It must be paired with Brier score, reliability diagrams, calibration slope/intercept and risk-coverage curves in published evaluation.
- The calibration dataset and evaluation dataset must be separate, and splits should also be speaker/session-disjoint for audiovisual interaction.
- A valid calibration report does not authorize camera capture, model inference or data retention.

## Research upgrades

1. Add cluster bootstrap intervals by speaker/session rather than treating frames as independent.
2. Report calibration by task family, modality, demographic/appearance condition, lighting, device and temporal regime where collection consent and sample sizes permit.
3. Add calibration drift alarms and require a new report/model digest after material model or preprocessing changes.
4. Evaluate decision calibration, not only token/frame confidence: confidence in expected loss reduction must be calibrated against realized task loss.
5. Compare uncertainty methods under matched compute: deep ensembles, temperature scaling, isotonic regression, conformal risk control and selective prediction.
6. Measure ROI allocation regret against an oracle policy and random/fixed-ROI baselines under equal pixel, latency and energy budgets.

## Proposed acceptance targets (not results)

| Gate | Target |
|---|---:|
| Calibration/evaluation leakage | 0 |
| Model/task-family/report binding mismatches admitted | 0 |
| ECE | <=0.05 overall and per preregistered task family |
| Brier score | Report with confidence interval; no universal threshold |
| Calibration slope | 0.9–1.1 |
| Selective risk at 80% coverage | <=0.10 normalized decision loss |
| ROI allocation regret vs oracle | <=10% |
| Matched-budget decision quality lift vs surprise-only | >=5 pp |

## Qualification status

The report verifier, recomputation logic, bindings and source-level tests are authored. They have not been compiled or run. No calibration dataset/report or empirical ROI result exists. The source is a deterministic contract, not a benchmark result.
