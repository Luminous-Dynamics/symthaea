# MEL-CLOCK-002 — Runner/CPAL clock bridge v1

**Status:** pure Rust contract, deterministic fitter, strict schemas and synthetic fixtures authored; runtime callback wiring, Rust compilation and device testing are not yet qualified.

## Objective

Bridge the runner monotonic clock used by turn-taking/effect receipts directly to the **CPAL predicted-playback clock** of one output stream. Retain CPAL callback time as a separate diagnostic. This allows direct composition with MEL-CLOCK-001's playback-stream-time → output-frame fit without assuming shared clock origins.

The v1 mapping is:

`runner monotonic time ~= offset + rate * CPAL playback-stream time`

Its inverse estimates the CPAL stream-clock instant corresponding to a runner timestamp.

## Exact clock-origin contract

`RunnerMonotonicClockOriginV1` is a copyable wrapper around one `std::time::Instant` origin. Construct it once per session and use the exact same value in:
- the streaming turn-taking receipt producer;
- the action/effect receipt producer;
- the CPAL callback clock-pair capture.

Persist only session-relative nanoseconds, never `Instant` internals. `runner_origin_id` names this exact origin; `runner_clock_domain_id` names its timebase; `session_id` binds it to the musical session. A process restart or new origin requires a new descriptor and fit segment.

## Callback-pair capture

At callback entry, bracket the observation and normalization of both CPAL timestamps with runner time samples from the shared origin. Regress against `OutputStreamTimestamp.playback`; retain `OutputStreamTimestamp.callback` for monotonicity/diagnostic comparison.

1. sample runner monotonic time before reading/normalizing the CPAL timestamp pair;
2. read both `OutputStreamTimestamp.callback` and `OutputStreamTimestamp.playback`, normalizing both to the same first-callback stream origin;
3. sample runner monotonic time immediately after;
4. persist both CPAL timestamps; fit runner monotonic time against the **playback** timestamp, not callback invocation time;
4. enqueue a fixed-size `RunnerCpalClockPairV1` to a bounded SPSC ring;
5. increment the pair index even when the record cannot be retained, so gaps remain visible.

The callback record is `Copy` and contains only integers, an enum and a fixed 32-byte descriptor-identity digest. The identity digest is computed off the real-time thread. No allocation, mutex, blocking call, formatting or logging is needed for the pair record.

The callback timestamp remains a diagnostic and must also be monotonic. The before/after bracket is an observed software capture interval. It is not a guarantee about backend callback-start error, predicted playback accuracy or DAC time. Fit residuals are not converted to accuracy bounds.

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

The module now provides `estimate_output_frame_at_runner_time_v1`, which composes:
1. runner monotonic time → CPAL predicted-playback stream time;
2. CPAL predicted-playback stream time → cumulative output frame.

It checks stream instance, CPAL configuration digest, output-fit descriptor validity and both fit validity windows. The callback timestamp is retained in each pair as a distinct diagnostic; it is **not** substituted for the playback timestamp.

**This composition is a point estimate only in v1.** It must not be labelled calibrated latency, absolute timing accuracy, DAC output or acoustic observation. A physical output claim still requires independent electrical/acoustic loopback.

## Synthetic fixtures

The checked-in fixtures cover:
- exact nominal mapping to the predicted-playback clock;
- +25 ppm and -25 ppm in-policy drift;
- callback-time monotonicity and predicted-playback-time monotonicity as distinct checks;
- exact runner-to-predicted-playback-to-output-frame composition with matching descriptor identity;
- exact runner → predicted-playback → output-frame composition with matching stream/configuration identity;
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
