# MEL Output Observation Closure v1

**Status:** executable evaluator contract with a synthetic known-answer fixture and synthetic negative controls. No live audio loopback, MIDI monitor, or independent observer has been connected.

## Why this exists

A missing event in a partial recording is not evidence that the event did not occur. Earlier predictive-transparency scoring could treat the absence of a realization record as a negative presence observation. This revision separates three states:

- **Detector-reported present:** the independent event-observer manifest contains a matching event in a structurally complete capture window.
- **Detector-reported absent:** the event-observer manifest contains no matching event and the capture window is structurally complete for the entire forecast target interval.
- **Unobserved:** the capture is absent, incomplete, truncated, gapped, duplicated, misaligned, or otherwise not eligible to establish absence.

Only the first two states enter the presence Brier score or event-level observation metrics. These are detector-reported labels, not independently established ground truth; detector sensitivity and specificity remain separate. The report exposes output-observation structural eligibility separately from forecast-lifecycle integrity. The third remains missing evidence; it is never silently converted to probability zero. Even a positive event listed in a producer-side realization record is not promoted to a scored observation when the independent capture is incomplete.

## Closure contract

Each forecast may have one independent observation-closure record. The record binds the capture plus an independent event-observer identity, implementation/configuration digests, a qualification-receipt reference, an observed-event-manifest digest, and the observed event records themselves. The report calls this structural eligibility, not authenticated observation. It reports raw supplied event-record counts separately from the count of event records admitted by complete closure. When a producer realization record exists, it must point to that exact capture ID and event-manifest digest; the independent capture and observer manifest remain sufficient even when the producer emitted no realization record. The manifest event list is carried explicitly as `observation_closures[].observed_events`; `realizations[].events` are producer-side output claims and are never used as observation ground truth. The record binds:

- forecast and capture IDs;
- capture source kind: rendered-audio loopback or MIDI output monitor;
- capture-observer identity distinct from the output producer;
- event-observer identity distinct from the output producer (the event detector may be a separate service from the capture recorder);
- producer identity, capture clock domain and target-window coverage;
- capture start/end, terminal watermark and close timestamp;
- sequence range, expected/observed/missing/duplicate record counts;
- audio sample rate, channels, expected/observed/dropped sample frames and xrun count; or
- MIDI monitor heartbeat cadence and expected/observed/missing heartbeat counts;
- capture-route/configuration, capture-payload, observer-build and closure-receipt SHA-256 digests.

A complete window is derived, not asserted by a Boolean field. It requires the capture to cover the full forecast target interval, a terminal watermark at or beyond the complete captured interval, closure after the complete capture interval, matching clock domain, independent observer/producer identities, contiguous sequence accounting, and no missing or duplicated records. Audio additionally requires exact expected frame coverage and zero dropped frames/xruns. MIDI monitoring requires the complete scheduled heartbeat series.

An observation closure with valid field shapes but failed coverage checks is retained as an incomplete result. Its negative labels are not scored. A mismatched capture ID or event-manifest digest when a producer realization is supplied, producer-as-event-observer claim, malformed accounting, impossible ranges, unsupported source kinds, unknown forecast IDs, duplicate closure records and invalid digests are ineligible or rejected. Producer realization metadata is optional and cannot veto or establish the independent observation.

## What this does not prove

- SHA-256 field validation checks syntax and referential provenance shape; this version does not recompute the event-manifest digest from the embedded event list, authenticate a signer, or prove that a recorder actually captured the stated bytes.
- Complete capture does not prove the sensitivity or specificity of the event detector. This evaluator binds an event-observer qualification reference and manifest digest but does not verify the referenced qualification artifact, its signatures, or the detector's measured sensitivity/specificity. Detector qualification remains a separate gate requiring task-, instrument-, event- and signal-condition-specific validation. The synthetic event manifest is not decoded from the synthetic waveform fixture.
- A loopback capture proves only the measured output path; it does not establish what a musician heard unless acoustic position, monitoring chain and relevant propagation are independently measured.
- MIDI output monitoring and rendered-audio loopback are different observation modes and must be reported separately.
- Synthetic fixtures prove evaluator behavior only, not live Melothaea performance.

## Required negative controls

The tests require incomplete/unobserved outcomes for:
- missing closure record;
- capture end before the target window ends;
- terminal watermark before target-window closure;
- dropped audio sample frames or an audio xrun;
- missing MIDI heartbeat;
- clock-domain mismatch;
- producer certifying its own output;
- event detector distinct from the producer but separate from the capture-recorder identity; and
- duplicate closure records or malformed receipt digests.

A complete capture with an empty `observed_events` manifest must produce an explicit detector-reported absence and a scored Brier sample, even when the producer realization contains an event record or is absent altogether. Conversely, an observer event can be scored when the producer-side realization record or its event list is absent.

## Prospective qualification targets

These are research targets, not achieved results:

| Measure | Initial target | Stretch target |
|---|---:|---:|
| Scored forecasts with complete closure | 100% | 100% |
| False-negative score on incomplete capture | 0 | 0 |
| Negative controls incorrectly admitted as complete | 0 | 0 |
| Complete capture-window frame/heartbeat accounting | 100% | 100% |
| Qualified observation-window yield in live sessions | >=99% | >=99.9% |
| Event-detector sensitivity lower 95% CI, within declared scope | >=0.98 | >=0.995 |
| Event-detector false-positive upper 95% CI, within declared scope | <=0.02 | <=0.005 |

The zero-tolerance gates concern evidence eligibility, not musical performance. For event detection, confidence bounds must be computed on independent held-out material with instrument, register, overlap, SNR, room, codec, dynamics and output-route strata. A detector cannot be qualified on clean synthetic tones and then silently generalized to dense live mixes.

## Next integration boundary

1. Connect the real runner/CPAL output path to an independent audio loopback recorder and MIDI monitor.
2. Bind stream identity, sample rate, buffer size, device/route identity, monotonic clock, xrun/drop counters and end-of-window watermark into each capture receipt.
3. Test silence, underrun, buffer wrap, reconnect, device change, clock drift, partial write, process crash and deliberately truncated capture.
4. Qualify onset/event detectors independently before using their negatives for forecast calibration.
5. Only then run musician-held-out anticipation and collaboration studies.

The auditor must continue to report MIDI emission latency separately from acoustic audible-onset latency. A complete MIDI event stream is not a substitute for an audio loopback measurement.
