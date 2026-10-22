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

```text
reported timestamp
      +
ClockDomainId
      +
optional ClockEpochId
      +
ContinuityStatus
      +
TimeUncertainty
      |
      v
TimeIntegrityReceipt
      |
      +--> declared_separation_us
      |      same declared clock + epoch only
      |      no accuracy/synchronization claim
      |
      `--> bounded_separation_window_us
             same clock + epoch
             continuity established
             finite uncertainty on both sides
             => conservative [minimum, maximum] separation
```

`ClockDomainId::unix_epoch()` is explicit. An absent or different clock identity never implicitly means Unix time.

## Two levels of temporal claim

`declared_separation_us` answers a weak question: *if these two producer declarations are accepted, what is their nominal numeric separation?*

It checks clock-domain and epoch identity before arithmetic but does not claim the clocks are accurate or synchronized.

`bounded_separation_window_us` answers the stronger question: *under these explicit continuity and uncertainty claims, what interval can safely contain the real absolute separation?*

For timestamps `t1`, `t2` and error bounds `e1`, `e2`:

```text
nominal = |t1 - t2|
combined_error = e1 + e2
minimum = max(0, nominal - combined_error)
maximum = nominal + combined_error
```

Arithmetic saturates instead of wrapping.

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

Physical multi-device smell/taste fusion should require bounded temporal evidence rather than treating equal clock-domain strings as synchronization proof.

## Validation

Run:

```sh
cargo test -p symthaea-time-integrity
cargo clippy -p symthaea-time-integrity --all-targets -- -D warnings
cargo test -p symthaea-time-integrity --doc
```

The crate is an evidence contract, not a synchronization implementation.
