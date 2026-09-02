# VART-WORLD-CREATIVE-001 — Independent Calibration Reconstruction v1

Status: confirmatory analysis/qualification contract. It does not authorize a calibration-improvement claim by itself.

## Principle

Prediction error is not authoritative merely because the runtime writes a `CalibrationLedger` or scalar `prediction_error_magnitude`.

The independent verifier reconstructs calibration from the prospectively closed `RevisionHypothesis` and the post-revisit `RevisionOutcome`.

Runtime calibration ledgers remain useful diagnostics, but scientific claim admission uses the independently reconstructed values.

## Canonical hypothesis/outcome fields

For each complete trial:

`RevisionHypothesis.expected_effects` is a JSON object mapping preregistered metric/channel IDs to finite numeric predicted deltas.

`RevisionOutcome.actual_effects` is a JSON object mapping measured metric/channel IDs to finite numeric observed deltas.

Every predicted channel must have a corresponding actual value. Outcomes may contain additional measured channels, but they are not retroactively promoted into the prospective prediction family.

## Per-channel reconstruction

For each predicted channel `c`:

- `signed_error[c] = actual_effects[c] - expected_effects[c]`
- `absolute_error[c] = abs(signed_error[c])`
- `squared_error[c] = signed_error[c]^2`

These values are reported separately by channel. They are not a world-quality aggregate.

## Scalar prediction-error diagnostic

If the runtime exports `prediction_error_magnitude`, v1 defines it as:

`sqrt(sum(squared_error[c] for c in sorted(predicted_channels)))`

This is `l2_over_declared_effects_v1`.

The scalar is a calibration diagnostic only. It cannot substitute for per-channel reporting and cannot be used as `world_quality`, `creative_score`, or an equivalent primary aggregate.

## Calibration receipt

Each complete trial exports `calibration_receipt` and binds its raw-byte digest as `calibration_receipt_sha256`.

The receipt contains:

- `schema = "symthaea.vart-world-creative-001.calibration-receipt.v1"`
- `experiment_id`
- `trial_id`
- `revision_hypothesis_sha256`
- `revision_outcome_sha256`
- `error_metric = "l2_over_declared_effects_v1"`
- `expected_effects`
- `actual_effects`
- `signed_error`
- `absolute_error`
- `squared_error`
- `prediction_error_magnitude`

The independent verifier recomputes every numeric field.

## Longitudinal reconstruction

Calibration trends are computed within persistent world identity, never by treating revisions as independent worlds.

For each world/channel with sufficient preregistered repeated revisions, the verifier reconstructs the ordered `absolute_error` sequence by revision index. Trend estimation and the minimum number of revisions are frozen in the analysis/calibration contract before confirmatory execution.

Worlds whose calibration worsens remain in the population summary.

## Separation of claim families

A system can improve the world while becoming less calibrated, or become well calibrated while making poor interventions. Therefore:

- world-improvement results do not prove calibration improvement;
- calibration improvement does not prove world improvement;
- both claim families retain separate uncertainty and multiplicity treatment.

## Required rejection classes

- `CALIBRATION_EVIDENCE_INCOMPLETE`
- `CALIBRATION_NONFINITE_VALUE`
- `CALIBRATION_RECONSTRUCTION_MISMATCH`
- `CALIBRATION_SCALAR_MISMATCH`
- `CALIBRATION_CONTRACT_MISMATCH`

## Claim ceiling

Passing reconstruction establishes only that prediction errors are faithfully derived from prospective predictions and retrospective measurements. A claim that calibration *improves over time* additionally requires the preregistered longitudinal estimator, sufficient repeated worlds/revisions, uncertainty analysis, and the frozen claim threshold.
