# symthaea-time-integrity

Domain-neutral temporal evidence contracts for Symthaea.

This crate does **not** synchronize clocks. It records what an acquisition or synchronization layer claims about timestamp identity, continuity, and uncertainty, and it provides conservative comparison helpers that fail closed when those claims are insufficient.

## Why this exists

A numeric timestamp alone is not enough evidence for temporal fusion. Before two observations are treated as simultaneous or ordered, a consumer may need to know:

- which clock/timebase produced each timestamp;
- whether both observations belong to the same continuity epoch;
- whether continuity checks passed;
- whether a finite timestamp-error bound exists.

That distinction matters for sensor fusion, robotics, distributed cognition, replay, and chemosensation. A device uptime counter must never become Unix time merely because both are represented as `u64` microseconds.

## Contract

A `TimeIntegrityReceipt` combines a reported timestamp's clock-domain identity with optional continuity-epoch provenance, continuity status, and timestamp uncertainty.

Two comparison levels are intentionally different:

- `declared_separation_us` is weak. It requires the declarations to agree on clock domain and on whatever epoch metadata is present. Two receipts that both omit an epoch may still be compared nominally, but the result carries no reset/reboot continuity guarantee.
- `bounded_separation_window_us` is strict. Both receipts must provide explicit matching `ClockEpochId`s, claim `ContinuityStatus::Continuous`, and provide finite uncertainty bounds before a conservative real-separation interval is returned.

`ClockDomainId::unix_epoch()` is explicit. An absent or different clock identity never implicitly means Unix time.

## Why strict comparison requires an explicit epoch

`clock_epoch = None` means epoch provenance was not supplied. It must not become equivalent to a known shared reboot/continuity epoch merely because both sides happen to omit the field.

A producer therefore cannot upgrade an epoch-less timestamp into strict physical-fusion evidence simply by setting continuity to `Continuous` and supplying a finite error number.

`TimeIntegrityReceipt::supports_bounded_comparison()` enforces the same rule as `bounded_separation_window_us()`:

- explicit epoch present;
- continuity established;
- finite uncertainty present.

The comparison still separately verifies that the two explicit epochs and clock domains actually match.

## Two levels of temporal claim

`declared_separation_us` answers a weak question: *if these two producer declarations are accepted, what is their nominal numeric separation?*

It checks clock-domain and declared epoch identity before arithmetic but does not claim the clocks are accurate or synchronized. When both epochs are absent, the result is deliberately weak and must not be used as a strict reset-safe timing guarantee.

`bounded_separation_window_us` answers the stronger question: *under explicit epoch, continuity, and uncertainty claims, what interval can safely contain the real absolute separation?*

For timestamps `t1`, `t2` and error bounds `e1`, `e2`, the nominal separation is `|t1 - t2|`; the two error bounds are conservatively added; the minimum separation saturates at zero; and the maximum separation saturates at `u64::MAX` rather than wrapping.

## Evidence boundary

A `TimeIntegrityReceipt` is a **claim container**, not proof by itself.

It does not establish:

- authenticity or authorization of the producer;
- PTP/NTP/mesh synchronization;
- oscillator qualification;
- hardware timestamp provenance;
- monotonicity unless a producer has actually checked it;
- an uncertainty bound unless a producer has actually established one.

Consumers that care about those properties must obtain the receipt from an appropriate acquisition, synchronization, or integrity-monitoring subsystem.

## Relationship to existing Symthaea timing work

Symthaea already contains adjacent mechanisms:

- mesh-time consensus models peer offset, RTT, stratum, drift, and quality;
- HAL time-integrity work has explored source identity, reboot epochs, monotonicity, missing samples, age, discontinuity, and rate-error checks.

This crate is intentionally smaller. Those mechanisms may produce or validate `TimeIntegrityReceipt`s later; they should not be duplicated inside individual sensor domains.

## Chemosensation migration intent

The chemosensation clock-domain work predates this generic extraction. A later integration tranche should adapt or replace `ChemicalClockDomainId` with `ClockDomainId` and attach `TimeIntegrityReceipt` at the acquisition boundary.

Physical multi-device smell/taste fusion should require strict bounded temporal evidence rather than treating equal clock-domain strings as synchronization proof.

## Validation

Run:

```sh
cargo test -p symthaea-time-integrity
cargo clippy -p symthaea-time-integrity --all-targets -- -D warnings
cargo test -p symthaea-time-integrity --doc
```

The crate is an evidence contract, not a synchronization implementation.
