# MEL-CLOCK-001 — Output-clock anchor and fit contract v1

**Status:** research/implementation v1; no device measurements claimed.

## Scope

This slice defines the persisted anchor and deterministic fit contract. It maps CPAL playback-stream nanoseconds to cumulative output sample-frame position within one continuous stream segment.

Each callback anchor binds:
- exact output frame index and frame count represented by the callback buffer;
- CPAL callback timestamp;
- CPAL predicted playback timestamp;
- callback-to-predicted-playback interval;
- stream-instance and sample-rate identity;
- optional externally justified timestamp uncertainty.

## Clock semantics

CPAL 0.17.1 documents OutputStreamTimestamp.callback as the callback invocation time and playback as the predicted time the written block will be delivered to the device. StreamInstant is monotonic for one stream lifetime; stream origins may differ.

Consequently:
- Never compare absolute StreamInstant values across streams.
- Represent times relative to the first callback timestamp within one stream instance.
- Associate each playback timestamp with the first frame index represented by that callback buffer.
- Increment the frame index by the number of output frames supplied to each callback, including underrun-filled silence.
- Start a new segment after frame discontinuity, stream restart, sample-rate change or timestamp reset.

## Fit and rejection rules

Use a frozen ordinary-least-squares fit over one ordered window:

    frame = reference_frame + frames_per_ns * (playback_ns - reference_playback_ns)

The nominal sample rate is metadata and a drift reference, not a replacement for the measured stream clock. The fit reports drift in ppm and absolute residual quantiles. It rejects:
- fewer than the policy minimum anchors;
- invalid callback/playback deltas;
- mixed stream IDs or sample rates;
- duplicate or regressing playback times;
- non-contiguous output frame indices;
- excessive drift;
- any residual outside the frozen threshold;
- queries beyond the valid interval plus explicit extrapolation horizon.

The fitter does not silently remove outliers. A discontinuity or suspicious anchor requires a new segment or a separately justified regrade.

## Accuracy boundary

CPAL supplies predicted playback timestamps, not a universal accuracy bound and not a measurement of physical DAC output. A low fit residual demonstrates internal consistency of the anchor series; it does not establish absolute timestamp accuracy.

Raw per-anchor timestamp uncertainty metadata is retained, but v1 does not yet propagate those bounds through regression slope and intercept estimation. Therefore reported_uncertainty_frames remains absent and status is point_estimate_only even when anchors carry declared bounds. This deliberately avoids converting a fit residual or incomplete error-propagation model into an accuracy guarantee.

Electrical/acoustic output latency still requires independent loopback observation. Mapping CPAL stream time to runner monotonic time requires a separate set of paired observations and a separately qualified cross-clock fit.

## Synthetic fixtures

- clock-exact-v1.json: nominal 48 kHz.
- clock-drift-minus-25ppm-v1.json: small negative drift.
- clock-outlier-v1.json: timestamp outlier that must be rejected.
- clock-discontinuity-v1.json: output frame gap that requires a new segment.

These fixtures are synthetic conformance cases, not device traces or performance results.

## Acceptance targets

- 100% acceptance of valid exact-clock fixtures.
- 100% estimation of declared in-policy drift fixtures within test tolerance.
- 100% rejection of injected discontinuities and out-of-policy residuals.
- Zero extrapolation outside the declared validity horizon.
- Zero absolute-accuracy claim until both source uncertainty and regression-parameter uncertainty propagation are qualified.
