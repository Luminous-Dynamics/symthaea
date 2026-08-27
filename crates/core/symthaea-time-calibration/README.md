# Symthaea Time Calibration

`symthaea-time-calibration` derives conservative clock-offset intervals from raw four-timestamp request/response exchanges.

It is an **evidence derivation layer**, not a clock synchronization protocol.

## Four timestamps

For one exchange:

- `t1`: source sends request
- `t2`: target receives request
- `t3`: target sends response
- `t4`: source receives response

Define source-to-target clock offset as:

`theta = target_time - source_time`

With non-negative one-way delay and approximately constant offset over the exchange:

- `theta <= t2 - t1`
- `theta >= t3 - t4`

The crate does **not** assume symmetric network delay.

## Endpoint uncertainty

Each timestamp carries its own `TimeIntegrityReceipt`.

If the four finite timestamp error bounds are `e1..e4`, the admissible offset interval becomes:

- `lower = t3 - t4 - e3 - e4`
- `upper = t2 - t1 + e2 + e1`

If `lower > upper`, no offset satisfies the exchange under the declared assumptions and calibration fails closed.

## Required temporal evidence

A valid calibration exchange requires:

- explicit source and target clock epochs;
- source send/receive in one source domain + epoch;
- target receive/send in one target domain + epoch;
- `ContinuityStatus::Continuous` on all four timestamps;
- finite uncertainty on all four timestamps;
- local event ordering that remains physically possible after uncertainty is considered.

If source and target claim the exact same domain and epoch, the derived interval must contain zero. Otherwise the calibration evidence contradicts the declared timebase identity.

## Multiple exchanges

Multiple compatible exchanges are combined only by interval intersection.

Example:

- exchange A admits `[430, 530] us`
- exchange B admits `[480, 560] us`
- combined evidence admits `[480, 530] us`

If intervals are disjoint, the evidence is rejected as mutually inconsistent.

There is no averaging, Gaussian model, independence assumption, or confidence inflation from sample count in this crate.

## Relationship to normalization

This crate deliberately does **not** mint `ClockTransformReceipt`s.

The intended pipeline is:

1. `symthaea-time-integrity` describes timestamp identity, epoch, continuity, and error bounds;
2. `symthaea-time-calibration` derives an admissible offset interval from raw exchange evidence;
3. a later preregistered/policy layer decides whether that interval is sufficiently tight for a specific use;
4. only then may `symthaea-time-normalization` receive an offset anchor + bounded transform uncertainty.

This separation prevents a calibration observation from automatically upgrading itself into synchronization authority.

## Non-claims

This crate does not provide or prove:

- NTP/PTP/GPS/PPS synchronization;
- symmetric delay;
- constant oscillator rate outside one exchange;
- independence of repeated exchanges;
- hardware timestamp authenticity;
- network-path authenticity;
- oscillator qualification;
- a trusted global clock;
- statistical confidence from sample count.

A four-timestamp interval is only as sound as the attached timing receipts and the stated non-negative-delay / approximately-constant-offset assumptions.
