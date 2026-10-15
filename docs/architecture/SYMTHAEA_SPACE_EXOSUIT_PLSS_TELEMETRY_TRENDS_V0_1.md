# Space Exosuit PLSS telemetry trends v0.1

**Status:** report-only, simulation-tagged software. Not a suit controller, qualified telemetry subsystem, life-support alarm, atmosphere safety estimator, or inspired-CO2 (PICO2) model.

**Research track:** [SX-026 — deterministic PLSS telemetry trend analysis](https://github.com/Luminous-Dynamics/symthaea/issues/7495).

**Implementation:** [SX-026 draft PR #7498](https://github.com/Luminous-Dynamics/symthaea/pull/7498), stacked on [SX-025 PR #7482](https://github.com/Luminous-Dynamics/symthaea/pull/7482).

## Why trend analysis is separate

NASA-STD-3001 Volume 2 Revision F, §6.1 explicitly calls for environmental and suit-monitoring data formats compatible with temporal trend analysis. Its rationale includes atmosphere constituents, temperature, humidity, water, suit pressure, ppO2, ppCO2, metabolic rate, and data measurement rate. §6.2 stresses that atmosphere constituents interact and must be considered together. NASA PLSS 1.0 integrated test reports describe system-level anomalies and boundary behavior involving CO2/water-vapor saturation, ventilation fan control, and thermal valve response.

- [NASA-STD-3001 Volume 2 §6.1, Trend Analysis of Environmental and Suit Data](https://www.nasa.gov/reference/6-0-natural-and-induced-environments-vol-2/)
- [NASA active standard record, Revision F, dated July 14, 2026](https://standards.nasa.gov/node/237)
- [NASA PLSS 1.0 development and test results](https://ntrs.nasa.gov/citations/20120004300)
- [NASA standardized inspired-CO2 measurement method, 2026](https://ntrs.nasa.gov/citations/20260001545)

The current implementation deliberately reports endpoint trends only. It does not infer inspired CO2 from bulk-circuit estimates, interpret a trend as a physiological exposure limit, or authorize control actions.

## Trend contract

- Every one of the six telemetry metrics has a lane for each provenance class: instrument, model estimate, and model prediction. The 18 lanes are emitted in fixed order even when empty.
- Samples are analyzed separately by metric, provenance class, and source ID. The module never averages separate source histories into one synthetic slope.
- The configured window, minimum number of samples, minimum confidence, minimum temporal coverage, and maximum internal gap are all explicit and included in the result fingerprint.
- Invalid units/domain values, future-dated observations, pre-window observations, and low-confidence observations are counted and excluded. If a lane has only one such class, it reports the specific status; mixed reasons remain InsufficientEvidence.
- Identical same-time samples from one source collapse to one sample. Contradictory values from one source at the same timestamp are Conflicted; no slope or rate interval is emitted.
- With enough valid samples, adequate observed span, and no gap exceeding the maximum, the rate is the endpoint secant rate: (last value − first value) / elapsed seconds. It is not a fitted regression trend and makes no assumption of continuous sampling between timestamps.
- The reported lower/upper rate bounds are arithmetic endpoint envelopes computed from the two endpoint absolute uncertainties. They are not statistical confidence intervals or calibration claims.
- Every eligible source trend also reports observed minimum/maximum and maximum deviation from the endpoint secant, so a benign start/end rate cannot hide a transient spike. This deviation is descriptive evidence, not an alarm threshold.
- Large final disagreement between independent sources marks the aggregate lane Conflicted. Individual source histories remain separately visible for diagnosis.
- Missing, sparse, conflict-only, invalid, future-only, out-of-window-only, and low-confidence-only lanes do not receive an invented zero rate.

## Deterministic receipts

The report receipt contains the caller-supplied source revision, observation and filter counts, canonical input SHA-256, result/config SHA-256, and a domain-separated SHA-256 envelope binding the revision, counters, and both digests. Numeric fields are serialized as little-endian IEEE-754 f64 bit patterns; strings are UTF-8 bytes with an explicit byte length. Input samples are canonically sorted before hashing.

The SHA-256 digests help identify changes and permit independent replay. They do not authenticate sensors, attest a binary, validate source IDs, or prove that a caller supplied a truthful source revision. The integration test's source SHA is explicitly synthetic and exercises binding semantics only.

## Versioned replay corpus

The file at crates/domains/symthaea-exoskeleton/tests/fixtures/plss_telemetry_trend_v1.json is corpus v1: 21 observations and 12 exact metric/provenance expectations. It exercises:

- two individually coherent CO2 sensor trends whose final readings disagree;
- an intermediate CO2 spike that leaves the endpoint rate benign but produces a large maximum deviation diagnostic;
- a separately retained modeled CO2 trend;
- same-source timestamp collision with contradictory pressure readings;
- a valid temperature trend with known endpoint rate;
- a future-only ventilation lane, invalid humidity units, out-of-window humidity, low-confidence humidity prediction, a sparse oxygen lane, and wholly missing lanes.

The tests/plss_telemetry_trend_replay.rs integration test checks exact statuses, source IDs, rates, full 18-lane output, and pinned observation/filter counts. It reimplements the fingerprint encoding separately from the production module and verifies input-order invariance.

On a configured Rust workspace, run:

```sh
cargo test -p symthaea-exoskeleton plss_telemetry_trend
cargo test -p symthaea-exoskeleton --test plss_telemetry_trend_replay
```

These tests demonstrate deterministic software behavior, not whether the configured limits or measurements are physically representative.

## Quantitative validation gates

Before using trends for engineering decisions, compare versioned outputs against independently calibrated bench time series. Preserve raw sample values, timestamps, source/calibration identifiers, units, uncertainty, gaps, reset markers, and environmental operating point. Review sample-rate adequacy and aliasing risk. Exercise the integrated loop across metabolic loads, ventilation flow, pressure, temperature, moisture loading, consumable exhaustion, sensor delays, timestamp collisions, and instrument faults. Trend thresholds and alert/control policy must be separately specified, hazard-analyzed, validated, and independently reviewed; none is implemented here.
