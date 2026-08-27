# Symthaea Time Calibration Bundle

`symthaea-time-calibration-bundle` binds a clock-calibration policy decision to the exact calibration evidence records used to produce it.

## Why

A `CalibrationDecisionReceipt` can prove that its frozen policy yields its stored decision for its stored interval. By itself, however, it cannot prove which four-timestamp exchanges produced that interval.

This crate closes that internal linkage gap.

## Verification chain

A `CalibrationDecisionBundle` contains:

- one `CalibrationDecisionReceipt`;
- the exact direct `ClockCalibrationEvidence` records used for the decision.

`verify_self()` performs:

1. decision-receipt self-verification;
2. structural evidence limits;
3. duplicate evidence rejection;
4. self-verification of every calibration evidence record;
5. recomputation of the calibration consensus by interval intersection;
6. rerun of the frozen decision policy over that consensus;
7. exact comparison with the stored decision receipt.

A valid exchange therefore cannot be relabeled onto another clock pair, a forged interval cannot be substituted, and removing evidence that changes the consensus/decision is detectable.

## Bounded direct evidence

A single bundle accepts at most 256 direct calibration records.

This is a structural/wire bound, not a scientific sample-size recommendation. Larger studies should chunk or content-address their evidence instead of placing an unbounded vector into one message.

The deserializer enforces the cap while reading the sequence rather than allocating an arbitrarily large vector first.

## Duplicate records

Byte/semantic-equal `ClockCalibrationEvidence` records are rejected as duplicates.

This prevents a caller from inflating the visible record count by repeating one exchange. The bundle does not use record count as confidence in any case.

## Accepted estimates

`accepted_estimate()` is available only after the entire bundle verifies and the decision is `Accepted`.

The returned midpoint + radius remains a convenience projection, not independent timing authority and not a clock transform.

## Important completeness boundary

This bundle proves:

> the stored decision is exactly what the attached evidence set produces under the stored policy.

It does **not** prove:

> no other eligible calibration evidence was omitted before the bundle was constructed.

Detecting selective omission requires an external completeness mechanism such as:

- preregistered run IDs;
- an append-only acquisition ledger;
- a content-addressed evidence-root commitment;
- signed acquisition manifests;
- or another independently auditable registry.

That should be a later layer rather than being falsely implied by this bundle.

## Intended timing stack

1. `symthaea-time-integrity` — timestamp domain/epoch/continuity/error claims;
2. `symthaea-time-calibration` — four-timestamp offset intervals;
3. `symthaea-time-calibration-policy` — Accepted / Rejected / Inconclusive decision;
4. this crate — exact evidence-set binding;
5. future completeness + holdover/drift evidence;
6. `symthaea-time-normalization` — validated transform representation/application.
