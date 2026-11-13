# EV SOC streaming reference baseline v0.1

## Purpose

A causal Rust 1 Hz coulomb-counting reference with explicit asset, calibration and capacity evidence. The estimator receives one sample at a time. Evaluation truth is stored separately in the fixture and is not part of the estimator API.

This is a software and synthetic-known-truth baseline, not a production BMS, SOC ground-truth source or physical validation result.

## Causal input contract

Each update receives only the current sample:

- timestamp in milliseconds;
- current in amperes, positive for charge and negative for discharge;
- terminal voltage and temperature;
- explicit validity flags for current, voltage and temperature.

The estimator has no future-sample, target-SOC, fixture-label or true-sensor-bias input. Exactly one 1,000 ms step is accepted. Invalid sensors, non-finite readings, voltage/temperature outside the declared domain or a timestamp gap force ABSTAIN. An abstention leaves the state and accepted timestamp unchanged; recovery requires a new initialization/calibration boundary. Missing samples are never silently interpolated.

Voltage and temperature are applicability gates in v0.1. They are not used as OCV anchors or thermal corrections.

## Reference estimator

Let measured current be I_m and independently calibrated current bias be b:

I_hat = I_m - b

For dt = 1 second and capacity Q in Ah:

- charge, I_hat >= 0: delta_SOC = I_hat * eta_charge * dt / (3600 * Q);
- discharge, I_hat < 0: delta_SOC = I_hat * dt / (3600 * Q * eta_discharge).

The point estimate is a simple coulomb counter. It does not attempt OCV correction, hysteresis compensation, online capacity identification, temperature compensation or model adaptation.

## Deterministic uncertainty bound

The implementation propagates a conservative interval from declared hard bounds for initial SOC, nominal capacity, current measurement error and residual current-bias calibration error. At each sample it evaluates the interval-corner combinations of corrected current and capacity and accumulates the minimum and maximum possible SOC increment under the fixed efficiency assumptions.

This is a deterministic bound conditional on those declared hard bounds and fixed efficiencies. It is not a probabilistic 90% or 95% confidence/credible interval, and no empirical coverage claim is made. The current measurement and bias uncertainty radii are added conservatively; no independence assumption is used to shrink them.

## Evidence binding

Every estimate carries exact vehicle and pack references, configuration digest, sensor-calibration digest, nominal-capacity reference digest, current-bias calibration digest and parameter digest. A digest establishes byte identity, not calibration competence or physical truth.

## Frozen synthetic corpus

The versioned corpus contains six known-truth cases:

1. positive-current charge;
2. negative-current discharge;
3. independent current-bias calibration;
4. invalid current sensor;
5. temperature outside the declared domain;
6. a missing 1 Hz sample / timestamp gap.

True current and true SOC are separate evaluation-only fixture fields. They are used by the corpus auditor after the estimator update, never passed to the estimator. The auditor checks expected dispositions, deterministic interval containment, exact abstention reasons, non-mutation on abstention and case-order-canonical corpus identity. Temporal sample order remains meaningful and is not permuted.

## Next qualification gates

Adapt this reference interface to the exact McMaster Battery SOC Benchmark 1 Hz submission boundary and reproduce its published weighted score before comparing more complex estimators. Then add an independent test harness for initialization error, current-sensor offsets, temperature transitions, aging, clock drift and missing telemetry.

The public McMaster weighted score and single-dataset NASA RMSE use different protocols and must not be ranked as if they were one leaderboard.

## Claim ceiling

This crate does not prove physical SOC accuracy, cross-cell/chemistry transfer, automotive functional safety, BMS suitability, battery safety or permission to charge/discharge. It must not command contactors, chargers, inverters or thermal systems.
