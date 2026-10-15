# Space Exosuit PLSS telemetry evidence and replay v0.1

**Status:** simulation-only software evidence pipeline. Not a qualified telemetry subsystem, medical monitor, exposure evaluator, or life-support controller.

**Implementation:** [SX-025 draft PR #7482](https://github.com/Luminous-Dynamics/symthaea/pull/7482), stacked on [SX-024 mass-balance PR #7463](https://github.com/Luminous-Dynamics/symthaea/pull/7463).

## Purpose and boundary

SX-024 emits a uniformly mixed *modeled bulk-circuit* CO2 partial-pressure estimate. That estimate is not a sensor observation and does not predict inspired CO2 (PICO2). This module retains that distinction by fusing observations separately by metric and provenance class:

- `Instrument`: value emitted by a named sensor source, not automatically calibrated or trustworthy.
- `ModelEstimate`: a model's estimate of a state.
- `ModelPrediction`: a model's prediction.

A model value cannot fill a missing instrument lane. Even `Fresh` means only that the entry passed this software's domain, age, confidence, and agreement gates; it is not a statement that the atmosphere is safe.

The available channels are bulk-circuit CO2 partial pressure, oxygen partial pressure, circuit pressure, gas temperature, ventilation flow, and relative humidity. Units are checked against the channel. Bounds in code exist to reject malformed values, not to encode human exposure or qualified hardware limits.

## Deterministic fusion contract

For each metric and provenance lane, the fuser:

1. rejects invalid fields and unit mismatches;
2. rejects future-dated observations into a distinct `FutureDated` result when no other exclusion reason exists;
3. excludes observations older than the configured maximum age and observations below minimum confidence;
4. retains only the latest eligible sample per source;
5. reports contradictory same-source payloads at the same timestamp as `Conflicted`;
6. reports cross-source disagreement beyond configured tolerance plus declared absolute uncertainty as `Conflicted`;
7. emits no point estimate or interval for missing, invalid, stale, future-dated, low-confidence, insufficient, or conflicted evidence.

Source IDs are sorted and result metrics use a fixed metric/provenance order. Repeated observations from one source are not counted as independent sources. The configured uncertainty envelope is not a calibrated statistical confidence interval.

## Versioned adversarial corpus

The fixture `crates/domains/symthaea-exoskeleton/tests/fixtures/plss_telemetry_v1.json` is the v1 corpus: 12 observations, 11 exact metric/provenance outcome expectations, and a fixed schema version. Replay asserts all 18 metric/provenance result lanes (six metrics × three provenance classes) are emitted, pins accepted-source and invalid/future/stale/low-confidence counts, and requires conflicted outputs to contain no point estimate or uncertainty bounds. The fixture covers:

- agreeing CO2 instruments kept separate from a widely different model estimate;
- conflicting oxygen instruments with no fused point estimate;
- contradictory same-source circuit-pressure readings at the same timestamp;
- stale measured temperature but fresh model-estimated temperature;
- a future-dated ventilation observation;
- a relative-humidity unit mismatch;
- low-confidence predicted humidity;
- missing metric/provenance combinations.

The integration test `tests/plss_telemetry_replay.rs` replays fixture expectations. Its receipt verifier has a separate implementation of the defined fingerprint encoding so it does not simply call the production digest helpers to validate themselves. It verifies order independence and recomputes the input, result, and envelope fingerprints from the exported report and receipt.

Run the targeted replay on a configured Rust workspace:

```sh
cargo test -p symthaea-exoskeleton --test plss_telemetry_replay
cargo test -p symthaea-exoskeleton plss_telemetry
```

The source revision used inside the test is deliberately synthetic and only exercises revision binding. Actual evidence runs must pass the exact checked-out Git commit SHA to the receipt API and retain the corpus, configuration, output report, counters, and receipt together.

## Fingerprint encoding

The SHA-256 input, result, and receipt fingerprints have distinct domain strings. Integer and IEEE-754 f64 bit values are encoded as little-endian u64; strings are encoded as a little-endian u64 byte length followed by UTF-8 bytes; optional floating-point values have a one-byte presence marker. Input observations are sorted by metric, provenance, source ID, timestamp, unit, value, uncertainty, and confidence before hashing.

These hashes help detect accidental or deliberate changes to a corpus/result under the chosen encoding, but they do **not** authenticate sensors, prove that source IDs are true, attest the executable binary, or validate the caller-supplied revision. Fused output is not a digital signature or tamper-proof provenance chain. If receipts must cross a trust boundary, the release process must independently validate Git object identity and sign/attest the complete receipt outside this module.

## Research and validation ladder

NASA-STD-3001 Volume 2, §11.3 separately addresses suit atmospheric monitoring and inspired-CO2 requirements. NASA's 2026 standardized measurement-method memorandum is about evaluating inspired CO2 by an established in-suit procedure. Integrated PLSS test-bed reports characterize interacting subsystems and boundary conditions. Consequently, code-level replay is necessary for software evidence, but it is not an atmospheric calibration or human-suit validation.

References:

- [NASA-STD-3001 Volume 2, §11.3 Spacesuit Atmosphere](https://www.nasa.gov/reference/11-0-spacesuits-vol-2/)
- [NASA/TM-20260001545, standardized measurement method for CO2 partial pressure within pressure suits](https://ntrs.nasa.gov/citations/20260001545)
- [NASA PLSS 1.0 breadboard development and results](https://ntrs.nasa.gov/citations/20120004300)
- [NASA PLSS 1.0 development and testing report](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20120007410.pdf)

Before quantitative use, record calibrated sensor identities and uncertainty, preserve raw time series, characterize component maps and capacity/breakthrough, test integrated gas-loop boundary conditions, validate inspired CO2 with the standardized method, and have an independent reviewer replay versioned measurements against the exact source SHA. Exact-head CI must finish before claiming the implementation is verified; local tests and passing software CI alone still do not establish human safety.
