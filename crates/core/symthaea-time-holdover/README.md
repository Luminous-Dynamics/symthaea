# Symthaea Time Holdover

`symthaea-time-holdover` converts an **accepted, evidence-bound calibration** into a finite-window `ClockTransformReceipt` only when explicit holdover assumptions are supplied.

It is not an oscillator monitor, synchronization daemon, or source of trust. It is an evidence-composition layer.

## Why another layer exists

A tight calibration interval answers a narrow question about the calibration exchanges that produced it. It does not prove that the source-to-target clock offset remains unchanged forever.

Holdover requires additional claims about:

- source-clock continuity;
- continuity of the source-to-target mapping;
- target-clock continuity;
- a bound on relative offset drift;
- a finite source-time validity interval;
- any fixed model error not already covered by calibration or drift.

These claims remain explicit instead of being silently inferred from a successful calibration.

## Relative drift, not clock quality

`max_relative_drift_ppb` bounds the rate at which the **source-to-target offset** may change.

It is not a generic quality score and does not describe either oscillator independently.

For a distance `dt_us` from a reference interval, additional drift uncertainty is:

`ceil(dt_us * max_relative_drift_ppb / 1_000_000_000)` microseconds.

The implementation uses checked/conservative integer arithmetic and fails if the resulting uncertainty cannot be represented safely.

## Calibration evidence is transported to the anchor first

The accepted calibration consensus is not treated as though it were measured at an arbitrary midpoint in time.

The holdover derivation:

1. verifies the complete `CalibrationDecisionBundle`;
2. requires the requested holdover window to contain every source-side calibration timestamp;
3. chooses a deterministic source anchor at the midpoint of the full source calibration envelope;
4. for each calibration exchange, transports that exchange's offset interval to the anchor using the declared relative-drift bound;
5. intersects the transported intervals at the anchor;
6. uses the midpoint of that anchor interval as the transform anchor offset and its radius as calibration uncertainty at the anchor;
7. grows uncertainty from the anchor to the farthest holdover endpoint;
8. adds explicit fixed model error;
9. derives the finite-window `ClockTransformReceipt`.

An exchange's offset interval is treated as applicable across that exchange's source-side send/receive interval under the calibration model's approximately-constant-offset assumption. Drift is charged only for distance outside that interval when transporting it to the common anchor.

## Total uncertainty

The transform-wide uncertainty is conservatively composed from:

- radius of the anchor-admissible calibration interval;
- worst-case relative drift from the anchor to either validity endpoint;
- explicit fixed model error.

No averaging or independence assumption is used.

## Continuity is three separate claims

Strict derivation requires all of:

- source continuity = `Continuous`;
- mapping continuity = `Continuous`;
- target continuity = `Continuous`.

One claim cannot substitute for another.

## Self-verifying composition

`BoundedHoldoverTransform` retains:

- the exact `CalibrationDecisionBundle`;
- the complete `HoldoverClaim`;
- the derived `ClockTransformReceipt`.

Deserialization and `verify_self()` rederive the transform. A stored transform whose anchors, validity interval, identities, continuity state, or uncertainty no longer match the underlying evidence and holdover claim is rejected.

## Important evidence boundary

The holdover claim is still a **claim container**.

This crate does not prove:

- oscillator stability;
- PTP/NTP/GPS/PPS correctness;
- hardware timestamp authenticity;
- source or target clock authenticity;
- temperature-compensated oscillator performance;
- independence of calibration samples;
- that omitted calibration evidence does not exist;
- global wall-clock truth.

A future hardware/monitoring layer must justify the drift and continuity claims. The exact calibration bundle also retains its own explicit completeness limitation.

## Intended sensor use

For physical chemosensation, a future acquisition adapter can attach generic `TimeIntegrityReceipt`s, validate calibration evidence, apply a use-specific calibration policy, bind the exact evidence set, derive a bounded holdover transform, normalize device-local timestamps, and finally perform uncertainty-aware temporal fusion.

No olfactory/gustatory root channel should be activated merely because this timing representation exists.
