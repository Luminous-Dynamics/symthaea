# MEL-CLOCK-002 — Runner/CPAL/output-frame clock composition v1

**Status:** pure Rust clock-origin contract, three-stage point-estimate composition, strict schemas, nine runner-clock fixtures, and 22 authored Rust tests are committed. Runtime callback wiring, compilation, and device/loopback qualification remain outstanding.

## Objective

Build a replayable mapping from the runner monotonic clock used by turn-taking/effect receipts to the cumulative output-frame coordinate used by MEL-CLOCK-001, without treating CPAL callback and predicted-playback timestamps as the same instant.

The correct chain has three distinct relations:

`runner monotonic time → CPAL callback time → CPAL predicted-playback time → output frame`

1. **Runner → callback:** simultaneous clock observation. The runner's `Instant` is sampled around the CPAL callback timestamp read.
2. **Callback → predicted playback:** a scheduling relation learned from CPAL's paired callback/playback timestamps on the exact output-anchor vector.
3. **Predicted playback → output frame:** MEL-CLOCK-001's existing fit from CPAL playback time to the cumulative frame index.

The second relation is not another shared-origin clock calibration. It is a measured callback-to-scheduled-playback mapping and must not be replaced with nominal buffer duration.

## Exact clock-origin contract

`RunnerMonotonicClockOriginV1` is a copyable wrapper around one `std::time::Instant` origin. Construct it once per session and use the exact same value in:
- the streaming turn-taking receipt producer;
- the action/effect receipt producer;
- the CPAL callback timestamp observer.

Persist only session-relative integer nanoseconds, never `Instant` internals. `runner_origin_id` names this exact origin; `runner_clock_domain_id` names its timebase; `session_id` binds it to the musical session. A process restart or new origin requires a new descriptor and fit segment.

## Runner-to-callback observation

At callback entry, bracket the read/normalization of CPAL's `OutputStreamTimestamp.callback` and `OutputStreamTimestamp.playback` with runner monotonic samples. The OLS fit in `RunnerCpalClockFitV1` uses the **callback** timestamp because it represents the instant at which the callback was invoked. The paired playback timestamp is retained but is not treated as simultaneous with the runner clock.

Each fixed-size `RunnerCpalClockPairV1` includes:
- session/runner-origin/stream identity digest;
- pair index;
- normalized CPAL callback timestamp;
- normalized CPAL predicted-playback timestamp;
- runner timestamp immediately before the CPAL timestamp read;
- runner timestamp immediately after it.

The callback record is `Copy` and contains only fixed-size values. The identity digest is computed off the real-time thread. No allocation, mutex, blocking call, formatting or logging is needed to construct/enqueue a pair.

The before/after bracket is an observed software capture interval. It is not a certified bound on backend timestamp error, callback-start accuracy, playback accuracy or DAC timing.

## Callback-to-predicted-playback scheduling fit

`fit_cpal_callback_playback_v1` consumes the exact `OutputClockAnchorV1` vector that MEL-CLOCK-001 uses for its output-frame fit.

It estimates:

`predicted playback stream time ~= offset + rate * callback stream time`

The fit:
- validates descriptor, stream instance, sample rate and channel count;
- validates callback/playback delta consistency;
- reports slope deviation from unity separately from runner/CPAL clock drift (both timestamps share one CPAL stream clock);
- requires strictly increasing anchor index, callback time and playback time;
- rejects invalid frame counts, mixed anchor sources, excessive drift and residuals;
- retains the exact serialized anchor-vector SHA-256 and fit-policy SHA-256;
- carries explicit calibration validity and bounded extrapolation;
- reports `point_estimate_only`; uncertainty remains absent.

This is a scheduling relation, not a device-independent physical clock relation. A changed or substituted anchor record changes the anchor-set digest, even if its event timing happens to remain numerically identical.

## Three-stage composition

`estimate_output_frame_at_runner_time_v1` composes:
1. runner monotonic time → CPAL callback time using `RunnerCpalClockFitV1`;
2. CPAL callback time → predicted playback time using `CpalCallbackPlaybackFitV1`;
3. predicted playback time → cumulative output frame using `OutputClockFitBundleV1`.

Before composition it requires:
- matching stream-instance IDs across all three receipts;
- matching CPAL configuration digest;
- valid schema and stream descriptors;
- the callback-to-playback fit's `anchor_set_sha256` to equal the output-frame bundle's anchor-set digest;
- the query to remain within all fit-validity windows and extrapolation horizons.

No shared clock origin is assumed between runner and CPAL. No nominal callback-buffer duration is used as a substitute for the callback-to-playback anchors.

The returned estimate includes the callback time, predicted-playback time, output-frame estimate and playback-time integer quantization delta. All remain point estimates until uncertainty propagation is qualified.

**Claim boundary:** this is not a calibrated absolute latency measurement, DAC observation or acoustic effect. CPAL's playback timestamp is a backend prediction. A physical-output claim still requires an independent electrical/acoustic loopback capture with its own clock, calibration and artifact provenance.

## Synthetic conformance and adversarial coverage

The nine runner-clock fixtures cover exact mapping, ±25 ppm drift, missing pair records, wide bracket rejection, descriptor mismatch, out-of-policy drift, residual outlier and predicted-playback-clock regression while callback time remains increasing.

The Rust test module additionally covers:
- direct runner-to-callback fitting and runner-clock regression;
- callback-to-playback exact scheduling fit;
- callback-to-playback residual outlier rejection;
- exact three-stage runner → callback → playback → output-frame composition;
- rejection of a scheduling fit whose anchor-set digest differs from the output-frame bundle;
- stream/configuration identity mismatch;
- validity expiry, dropped observations, digest binding and point-estimate-only status.

The exact synthetic composition checks runner time 120 ms → callback time 20 ms → predicted playback 25 ms → output frame 960. These are authored tests and controlled fixture expectations—not executed Rust results or physical measurements.

## Qualification gates

- 100% acceptance of valid exact and in-policy synthetic cases.
- 100% rejection of identity mismatch, malformed/wide brackets, non-monotonic callback/playback/runner clocks, out-of-policy clock-rate/scheduling-slope deviation and residuals, expired calibration and anchor-set substitution.
- Zero extrapolation beyond each declared validity horizon.
- Exact replay from the same descriptor, pair/anchor bytes and fit policies.
- No upgrade from point estimate to timing-accuracy claim without propagated uncertainty and independent device validation.

## Integration remaining

- Wire the same `RunnerMonotonicClockOriginV1` into #10309 receipt production and the #10345/#10327 CPAL callback.
- Capture both CPAL timestamps and runner brackets into a bounded SPSC ring; drain/fit outside the callback.
- Use the exact output-anchor vector for both callback-to-playback fitting and MEL-CLOCK-001 output-frame fitting.
- Bind the resulting receipts to the turn-taking session and action-effect ledger (#10382).
- Qualify uncertainty propagation or preserve point-estimate-only status.
- Run rustfmt, compilation, tests and feature configurations on an exact head, then test physical CPAL devices and independent loopback.

No Rust test execution, CI pass, physical-device measurement or acoustic-latency result is claimed by this v1 source slice.
