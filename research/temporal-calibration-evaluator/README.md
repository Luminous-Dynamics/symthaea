# Temporal Calibration Evaluator (evaluator side)

**Package:** `symthaea-temporal-calibration-evaluator`  
**Tracking:** NEURO-SILICON-010 / #9872  
**Status:** research evaluator; ten Rust test functions authored, not executed.

## Purpose

Compare candidate-predicted reductions in wrong-belief, abstention and missing-output exposure with realized evaluator outcomes from valid paired interventions. Fit a separate additive bias correction per exposure axis on an earlier fit partition, then evaluate raw and corrected values on a later holdout partition.

## Anti-leakage rules

- Each sample binds the candidate model digest, evaluator feedback sequence, episode ID, forecast issue time, time feedback became available, the validated paired-intervention digest and the finalized receipt digest.
- Model digest must match across fit and holdout partitions.
- Sequence IDs and forecast issue times must strictly increase within each partition.
- Episode IDs, paired-intervention commitments and finalized receipt commitments cannot repeat.
- Every outcome must become available strictly after its forecast.
- The latest fit feedback availability must be strictly earlier than the earliest holdout forecast origin.
- The evaluator reports a correction and holdout metrics; it does not release hidden current-episode outcomes to candidate code.

The crate consumes already-authorized evaluator records. It does not verify Ed25519 signatures or witness quorums itself and must stay inside the evaluator boundary.

## Metrics and claim ceiling

For each of the three axes, report fitted bias, raw holdout bias, corrected holdout bias, raw MAE and corrected MAE. The correction is an additive mean residual. MAE is descriptive: it is not a confidence interval, calibrated probability, causal effect estimate or guarantee that correction helps. A negative result is a valid qualification outcome.

Causal action-value claims still require randomized paired interventions; this diagnostic only measures prediction calibration against recorded outcomes.

## Local checks

```sh
cargo test --manifest-path research/temporal-calibration-evaluator/Cargo.toml --locked
```
