# Space Exosuit PLSS boundary-approach projections v0.1

**Status:** Simulation-only, deterministic, report-only research software. This is not an alarm, an exposure-limit evaluator, a physiological predictor, a life-support controller, or a suit qualification artifact.

**Research track:** [SX-027 — uncertainty-aware boundary-approach projections](https://github.com/Luminous-Dynamics/symthaea/issues/7500).

**Implementation:** Draft work builds on [SX-026 trend analysis](https://github.com/Luminous-Dynamics/symthaea/pull/7498), itself stacked on [SX-025 telemetry evidence](https://github.com/Luminous-Dynamics/symthaea/pull/7482).

## Why this step matters

NASA-STD-3001 Volume 2 Revision F is active and dated July 14, 2026. Section 6.1 says suit/environmental telemetry should support temporal trend analyses to help anticipate harmful conditions and troubleshoot previously recorded behavior. Section 6.2 says atmosphere variables interact and must be considered together. NASA's Diagnostics & Prognostics group also treats fault detection, diagnostics, prognostics, and prognostic decision-making as distinct research problems.

References:
- NASA-STD-3001 Volume 2, current record: https://standards.nasa.gov/node/237
- NASA §6.1 trend analysis and §6.2 atmosphere interaction: https://www.nasa.gov/reference/6-0-natural-and-induced-environments-vol-2/
- NASA Diagnostics & Prognostics Group: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/diagnostics-and-prognostics-group/
- NASA standardized inspired-CO2 method (measurement-method reference, not a threshold source): https://ntrs.nasa.gov/citations/20260001545

SX-026 emits source-separated endpoint-rate envelopes, explicit coverage/gaps, uncertainty, excursion diagnostics, and conflict states. SX-027 adds a constrained question: given a specific source, a specific threshold value, and a version identifier supplied by the caller, what *candidate time interval* would a local linear extrapolation imply?

## Projection contract

- The caller supplies metric, provenance class, exact source ID, upper/lower direction, boundary value, threshold-set revision (40/64 hexadecimal characters), and a positive maximum projection horizon.
- The boundary value must be valid for the telemetry metric's unit/domain. The revision links to external policy/configuration provenance; it does **not** authenticate, endorse, validate, or qualify that boundary.
- Before projecting, the API verifies that the supplied SX-026 trend report matches its receipt digest, counts, source revision, and receipt envelope.
- A projection uses only one exact metric/provenance/source lane. A model estimate is not combined with an instrument, and predictions are not relabeled as observations.
- Both the aggregate lane and selected source trend must be Fresh. A conflicted aggregate lane suppresses numeric projections even where one individual source has a plausible slope. Sparse, missing, invalid, out-of-window, low-confidence, or conflicting evidence never becomes a reassuring forecast.
- The module orients lower-boundary problems by negating value and rate, then applies the same calculation as an upper-boundary crossing.
- If the current value-uncertainty envelope is already past or overlaps a boundary, no future crossing interval is manufactured.
- When the complete rate envelope points toward the boundary, the earliest time uses the near side of the current value envelope and the fastest admissible endpoint rate; the latest uses the far side and the slowest admissible positive rate. If some admissible rates point toward the boundary while others do not (including zero), only a possible earliest crossing is reported and the result is DirectionUncertain. If the greatest admissible rate is exactly zero, the result is NotApproaching rather than MovingAway.
- A candidate interval is CandidateWithinHorizon only when its latest crossing estimate fits inside the configured horizon. If the interval runs beyond that horizon, the result is CandidateBeyondHorizon. Such times remain mathematical candidates, not guarantees.
- These calculations are arithmetic endpoint envelopes, not probability distributions or statistical confidence intervals. They may be conservative and effectively count current endpoint uncertainty both in the rate envelope and current-value envelope. They do not model nonlinearity, sensor lag, future workload, scrubber saturation, transient dynamics between samples, or causal coupling among atmospheric variables.
- No output issues alerts, changes mission state, actuates hardware, infers inspired CO2 (PICO2), or makes claims about human exposure safety.

## Deterministic receipt

The boundary receipt binds the input trend-result digest, caller-supplied source revision, canonical boundary set, each projection/status/output, and counts using domain-separated SHA-256 hashes. Numeric values are represented by little-endian IEEE-754 f64 bits, strings are UTF-8 with explicit byte lengths, and boundary definitions are canonically sorted before hashing. The replay test reimplements the digest encoding independently.

Fingerprints reveal input/output changes; they are not signatures and do not authenticate sensors, prove that a threshold revision is truthful, or attest to a deployed binary.

## Versioned replay corpus

Corpus: `crates/domains/symthaea-exoskeleton/tests/fixtures/plss_telemetry_boundary_v1.json`.

It includes upper and lower directions, boundary already beyond current data, a threshold inside the endpoint uncertainty envelope, candidates inside and outside the horizon, and missing-source behavior. Additional unit tests cover rate intervals that include zero, a flat zero-rate control, moving-away trajectories, source conflicts, altered SX-026 reports/receipts, invalid thresholds, and boundary input-order invariance.

Run on a configured Rust workspace:

```sh
cargo test -p symthaea-exoskeleton plss_telemetry_boundary
cargo test -p symthaea-exoskeleton --test plss_telemetry_boundary_replay
```

These tests establish deterministic software behavior only. They do not show that any particular slope, threshold, or extrapolation is physically valid.

## Evidence ladder before quantitative use

1. Review and independently approve the specific threshold source, revision, units, and intended operating context.
2. Replay against independently calibrated bench time series with real sampling rates, timestamps, sensor/calibration IDs, uncertainty, resets, dropouts, saturation, and delay.
3. Compare candidate horizon estimates to withheld time segments and boundary crossings; report false negatives, false positives, interval coverage, and sensitivity to noise and gaps.
4. Exercise coupled faults: workload change, ventilation degradation, CO2/moisture loading, thermal change, sensor lag, reset, and contradictory channels. Keep measured inspired-CO2 assessment distinct from bulk-circuit pressure trends.
5. Obtain independent engineering review and the program's required verification/qualification process before using results for operational decisions.

The paradigm change is **from plotting a slope to preserving the provenance and uncertainty of a decision-relevant projection**—while retaining the authority boundary that software extrapolation does not become a safety limit.
