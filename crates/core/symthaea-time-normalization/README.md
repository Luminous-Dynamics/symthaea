# Symthaea Time Normalization

`symthaea-time-normalization` converts timestamp evidence between declared clock domains without erasing where the source time came from or how uncertain the conversion is.

## Why this exists

A timestamp integer is not a clock identity. A device-monotonic counter, Unix time, GPS/PPS time, mesh time, and a rebooted device counter can all be represented as `u64` microseconds while having different semantics.

`symthaea-time-integrity` establishes the evidence carried by one timestamp. This crate adds the next step: an explicit, finite-window transform receipt that can normalize one clock domain into another while propagating uncertainty.

## v1 transform model

The first model is a rate-1 offset mapping anchored by one source/target pair:

```text
source_anchor_us  ─────────► target_anchor_us
       │                           │
       └──── equal-rate delta ─────┘
```

It is valid only over a declared source-time interval. Drift, calibration residual, network delay asymmetry, fit error, oscillator instability, or other synchronization error must be covered by the transform uncertainty bound.

The v1 model deliberately does **not** pretend a single offset remains globally valid.

## Evidence required for strict normalization

`normalize_timestamp_us(...)` fails closed unless all of these are true:

1. the source `ClockDomainId` matches the transform;
2. the source has an explicit `ClockEpochId` matching the transform;
3. the source timestamp is inside the calibrated validity interval;
4. source continuity is `Continuous`;
5. transform-mapping continuity is `Continuous`;
6. target-timebase continuity is `Continuous`;
7. source timestamp uncertainty is finite;
8. transform uncertainty is finite;
9. the mapped target timestamp is representable as `u64`.

The constructor additionally proves that the **entire declared validity interval** maps without target underflow/overflow.

## No provenance laundering

A normalized point retains:

- the original source timestamp;
- the original `TimeIntegrityReceipt`;
- the complete `ClockTransformReceipt`;
- the derived target timestamp;
- the derived target `TimeIntegrityReceipt`.

Source-domain sequence numbers are not copied into the target receipt. A sequence counter meaningful to one device clock is not automatically meaningful in another clock domain.

## Uncertainty propagation

For source error bound `e_source` and transform error bound `e_transform`:

```text
target_error <= e_source + e_transform
```

The implementation uses saturating addition. It does not assume statistical independence and does not shrink uncertainty through averaging.

## Threshold decisions are ternary

A bounded separation is an interval, not one exact number. For a maximum allowed skew `T`:

```text
window.maximum <= T
    -> DefinitelyWithin

window.minimum > T
    -> DefinitelyOutside

otherwise
    -> Ambiguous
```

This is intentional. There is no convenience boolean that silently turns an uncertainty-overlapping threshold into a pass or fail.

## Non-claims

This crate does **not** implement or prove:

- NTP, PTP, GPS/PPS, mesh-time, or hardware synchronization;
- oscillator qualification or drift estimation;
- network-delay estimation;
- authentication or authorization of transform producers;
- hardware timestamping;
- clock-source trust;
- a global wall-clock truth.

A transform receipt is a claim container. A synchronization/acquisition system must establish why its continuity and error-bound claims deserve to be accepted.

## Intended consumers

The contract is domain-neutral. Expected consumers include:

- chemosensation multi-device acquisition;
- camera/IMU fusion;
- distributed audio;
- robotics;
- mesh/agent observations;
- scientific instruments;
- replay/recording systems.

For chemosensation specifically, physical multi-nose or smell+taste temporal fusion should eventually normalize acquisition timestamps into a common bounded time domain before interpreting temporal skew.

## Validation

```text
cargo test -p symthaea-time-normalization
cargo clippy -p symthaea-time-normalization --all-targets -- -D warnings
cargo test -p symthaea-time-normalization --doc
```
