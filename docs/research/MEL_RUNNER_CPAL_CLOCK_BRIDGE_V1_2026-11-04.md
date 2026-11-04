# MEL-CLOCK-002 — Runner/CPAL clock bridge v1

**Status:** pure Rust contract, deterministic fitter, strict schemas and synthetic fixtures authored; runtime callback wiring, Rust compilation and device testing are not yet qualified.

## Objective

Bridge the runner monotonic clock used by turn-taking/effect receipts to the callback clock of one CPAL output stream. This enables a later composition with MEL-CLOCK-001's CPAL stream-time → output-frame fit, without assuming shared clock origins.

The v1 mapping is:

`runner monotonic time ~= offset + rate * CPAL callback-stream time`

Its inverse estimates the CPAL stream-clock instant corresponding to a runner timestamp.

## Exact clock-origin contract

`RunnerMonotonicClockOriginV1` is a copyable wrapper around one `std::time::Instant` origin. Construct it once per session and use the exact same value in:
- the streaming turn-taking receipt producer;
- the action/effect receipt producer;
- the CPAL callback clock-pair capture.

Persist only session-relative nanoseconds, never `Instant` internals. `runner_origin_id` names this exact origin; `runner_clock_domain_id` names its timebase; `session_id` binds it to the musical session. A process restart or new origin requires a new descriptor and fit segment.

## Callback-pair capture

At callback entry, bracket the observation of `OutputStreamTimestamp.callback` with runner time samples from the shared origin:

1. sample runner monotonic time before reading/normalizing the CPAL callback timestamp;
2. read CPAL's callback timestamp and normalize it relative to the first callback timestamp in that stream;
3. sample runner monotonic time immediately after;
4. enqueue a fixed-size `RunnerCpalClockPairV1` to a bounded SPSC ring;
5. increment the pair index even when the record cannot be retained, so gaps remain visible.

The callback record is `Copy` and contains only integers, an enum and a fixed 32-byte descriptor-identity digest. The identity digest is computed off the real-time thread. No allocation, mutex, blocking call, formatting or logging is needed for the pair record.

The before/after bracket is an observed software capture interval. It is not a guarantee about backend callback-start error or absolute CPAL timestamp accuracy. Fit residuals are not converted to accuracy bounds.

## Identity and replay

The bridge descriptor binds:
- session ID;
- runner clock-domain ID;
- exact runner-origin ID;
- CPAL stream-instance ID;
- SHA-256 digest of the matching CPAL stream descriptor/configuration.

A canonical identity digest is copied into every fixed-size pair. The fitter rejects any pair whose stream or descriptor digest does not match. This prevents a compact callback ring from being replayed under a different session or origin.

The fit receipt binds the exact serialized pair vector and normalized fit policy with SHA-256. Missing pair records before or between observations are counted; they are never filled with zero timestamps. Trailing record drops remain available from the live capture counter and must be included in the enclosing evidence manifest.

## Fit and rejection rules

The deterministic OLS fitter:
- requires the policy minimum pair count (never fewer than two);
- requires strictly increasing pair indices, CPAL callback times, and runner bracket endpoints;
- rejects reversed or over-wide capture brackets;
- rejects mixed stream/session/origin identity;
- estimates rate drift relative to 1 ns / 1 ns;
- reports median, p95 and maximum absolute fit residuals;
- rejects drift/residuals outside the frozen policy;
- retains the observed validity interval and a bounded extrapolation horizon;
- reports `point_estimate_only` and leaves uncertainty absent in v1.

The fitter never silently removes outliers. A device restart, new stream instance, runner-origin reset or timestamp discontinuity requires a new segment.

## Composition with output-frame fit

A caller may:
1. map a runner timestamp to a CPAL callback-stream time using `cpal_time_at_runner_time`;
2. pass that CPAL stream time to MEL-CLOCK-001's output-frame fit, which maps the CPAL stream clock to cumulative output frames.

Both fits must match the same session/stream identity and remain inside their respective validity windows.

**This composition is a point estimate only in v1.** It must not be labelled calibrated latency, absolute timing accuracy, DAC output or acoustic observation. A physical output claim still requires independent electrical/acoustic loopback.

## Synthetic fixtures

The checked-in fixtures cover:
- exact nominal mapping;
- +25 ppm and -25 ppm in-policy drift;
- missing pair records with continuous clock progression;
- over-wide observation bracket rejection;
- descriptor/origin identity mismatch rejection;
- out-of-policy drift rejection;
- residual outlier rejection without silently dropping the record.

The unit suite also rejects runner-clock regression even when CPAL callback time continues increasing.

These are conformance fixtures, not device measurements.

## Qualification targets

- 100% acceptance of valid exact and in-policy drift fixtures.
- 100% rejection of identity mismatch, malformed bracket, non-monotonic clock and over-policy drift/residual cases.
- Zero extrapolation beyond the declared validity horizon.
- Exact replay of fit results from the same descriptor, pair bytes and policy.
- No upgrade from point estimate to accuracy/latency claim without propagated uncertainty and independent device validation.

## Integration remaining

- Wire the same `RunnerMonotonicClockOriginV1` instance into the #10309 receipt producer and #10345/#10327 CPAL callback.
- Add bounded SPSC capture and non-real-time drain to the #10345 runtime branch.
- Bind the pair-set receipt to the turn-taking session and action-effect ledger (#10382).
- Implement uncertainty propagation or preserve point-estimate-only status.
- Run Rust tests/format/clippy on an exact head, then test actual CPAL devices and independent loopback.

No Rust test execution, CI pass, physical device measurement or acoustic latency result is claimed by this v1 source slice.
