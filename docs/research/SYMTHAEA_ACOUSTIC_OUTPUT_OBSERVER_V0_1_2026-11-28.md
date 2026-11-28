# Symthaea Acoustic Output Observer v0.1 — 2026-11-28

**Status:** proposed protocol; no observer implementation, calibration, acoustic measurement, or hardware qualification is claimed.  
**Parent work:** #14183 adapter integration, #14213 passported pilot, #14256 acoustic stop qualification.  
**Relationship:** complementary to the compositor presentation observer. It measures acoustic output; it does not replace visual presentation evidence.

## 1. Objective

Prove the difference between a stop request being accepted by software and agent-generated sound actually ceasing at a declared acoustic measurement point.

The observer must be outside the audio adapter's self-report path. It must bind each measurement window to the exact run, semantic event, action segment, stop trigger, output stream, policy version, sensor, calibration record, clock domain, and raw-window digest.

An acoustic observation establishes only what the qualified sensor measured at its declared location and threshold. It does not prove that a particular person heard, understood, or attended to the sound.

## 2. Evidence states — never collapse them

| State | Evidence | Does not establish |
|---|---|---|
| StopRequested | Policy/turn controller emitted a cancellation, expiry, or revocation event | Adapter or device response |
| AdapterAccepted | Adapter accepted the stop request for a bound segment | Driver buffer drained or sound ceased |
| DeviceDrainComplete | Device/driver reports that queued frames drained | Physical speaker silence |
| LoopbackSilent | Independent post-mixer/device loopback is below a frozen digital threshold | Physical transducer, amplifier, Bluetooth, or room output is silent |
| AcousticCessationObserved | Calibrated external sensor reports no attributable agent output above a pre-registered threshold for a pre-registered quiet dwell | Human audibility or perception in every environment |
| OutcomeUnknown | Coverage, attribution, calibration, or clock evidence is insufficient | Success |
| ObserverFault | Sensor/observer failure is positively detected | Silence |

A device-drain callback, empty software queue, stream close, mute request, adapter acknowledgement, or missing observation must never mint an acoustic stop receipt.

## 3. Two measurement lanes

### Lane A — software/device loopback

On Linux, PipeWire sink-monitor capture can provide a useful first local measurement path. PipeWire documents capture from sink monitor ports; its loopback module links capture and playback streams. This is evidence at a software/device boundary, not a microphone measurement of the speaker output.

Reference: https://pipewire.pages.freedesktop.org/pipewire/page_module_loopback.html

Record the PipeWire server version, graph topology, sink identity, monitor source identity, sample format/rate, channel map, buffering/quantum, xruns, dropped samples, and whether the observer is a separate process. Treat graph reconnection or a sink switch as a new output stream until re-bound.

ALSA loopback or a speaker-line monitor may provide additional device-path baselines. Preserve the exact point at which each capture occurs; do not call all of these “physical output”.

### Lane B — physical acoustic measurement

Use a measurement microphone or sound-level meter connected to a separate recorder/device. Record:

- sensor manufacturer/model and serial pseudonym or digest;
- calibration certificate/record digest and validity;
- measurement position, orientation, distance, enclosure, and room;
- sample rate, bit depth, channel, frequency weighting, time/integration weighting, and measurement window;
- background-noise estimate and validity;
- clipping/overload and missing-sample counters;
- pre-registered detection threshold and quiet dwell;
- source-attribution method and confidence/ambiguity.

IEC 61672-1 describes performance specifications for sound-level meters; IEC 60942:2017 describes sound calibrators. These references inform instrument selection and calibration records; citing a standard is not a claim that a device conforms to it.

References:
- https://webstore.iec.ch/en/publication/5708
- https://webstore.iec.ch/en/publication/30045

Do not use an ordinary laptop microphone as a calibrated measurement instrument without a traceable calibration procedure and a documented uncertainty budget.

## 4. Detection and attribution

Thresholds must be frozen before the held-out run. A protocol must define the metric and unit rather than store an unqualified “silence” boolean.

For loopback, use a digital metric appropriate to the captured stream, such as peak/RMS, sample count, or correlation against the expected output stream. Bind measurements to the device stream and action/segment identifiers.

For physical acoustics, define the measurement quantity and setup explicitly: for example, frequency-weighted sound level over a specified integration window, or correlation against the known emitted waveform under a calibrated capture geometry. Record ambient noise and the attribution method. A microphone reading below the noise floor, a clipped capture, ambiguous overlapping speech, or a sensor that lost coverage is not proof of silence.

A no-output claim is eligible for consideration only when:
- the complete observation interval is present and sample/drop counters reconcile;
- the sensor calibration is valid for the measurement lane;
- the clock is continuous and comparable to the stop trigger with bounded uncertainty;
- the noise floor is valid and the frozen detection threshold is usable;
- the sensor is not clipped or overloaded;
- attribution is sufficiently resolved for the declared lane;
- the below-threshold condition persists for the frozen quiet dwell.

Eligibility is computed by the verifier from the record and protocol. The observer must not self-assert a trusted “stopped” result.

## 5. Clock and latency model

Keep host scheduler, audio-device frame counter, loopback capture, and external recorder clocks distinct. Never subtract timestamps from different domains without a measured mapping.

A clock mapping must include:
- source and destination clock IDs;
- calibration method and calibration record digest;
- measured offset and drift bound;
- mapping validity interval;
- uncertainty at the time of each observation;
- reset/discontinuity detection.

For a stop trigger at mapped time t_s with uncertainty u_s and the final attributable output sample at t_o with uncertainty u_o, report the latency interval:

- lower bound = max(0, (t_o - t_s) - (u_s + u_o));
- upper bound = (t_o - t_s) + (u_s + u_o).

If the intervals cannot be compared safely, latency is unavailable. Do not report zero, drop the sample, or substitute adapter acknowledgement time.

## 6. Observation record

The companion Draft 2020-12 schema is SYMTHAEA_ACOUSTIC_OUTPUT_OBSERVER_V1_2026-11-28.schema.json.

Each record binds:
- run ID, observation ID, monotonic observation sequence;
- observer ID, observer build digest, process/device trust boundary;
- output lane and sensor type;
- sensor calibration, location, measurement method, and frozen threshold;
- clock ID, timebase, uncertainty, drift bound, and clock-mapping digest;
- sample interval and sample/drop counts;
- signal state, measurement value/unit, threshold, clipping and attribution;
- coverage, noise-floor validity, clock continuity and calibration status;
- semantic event, action, segment, stop trigger, stream and policy-version bindings;
- raw-window digest and previous-record digest.

The record is evidence input, not a verdict. A trace verifier must enforce cross-field semantics, event ordering, monotonic sequences, clock continuity, complete coverage, and the stop-state machine. JSON Schema validation alone cannot establish those properties.

## 7. Stop-state machine

Maintain separate per-channel states:

1. Active — output segment may produce samples under a live permit.
2. StopRequested — cancellation/revocation/expiry is recorded; new segment dispatch is prohibited.
3. AdapterAccepted — stop command accepted for the exact stream and segment.
4. DeviceDrainComplete — device output queue is reported drained.
5. LoopbackSilent — the loopback observer has satisfied its frozen digital silence rule.
6. AcousticCessationObserved — the physical acoustic observer has satisfied its frozen detection and dwell rule.
7. OutcomeUnknown — the system cannot establish a terminal result.
8. Failed — known adapter/device/observer failure.

LoopbackSilent and AcousticCessationObserved are different evidence claims. A loopback result must never be promoted to physical acoustic cessation.

For a multi-channel action, every active channel must independently reach its required terminal evidence state. A successful audio stop does not imply viseme, visual, gesture, haptic, or locomotion cessation. The global action is not Stopped while any in-scope output channel remains active or unresolved.

A later silence window cannot erase an attributable output event observed after revocation. Such an event is a policy/output violation, even if the output eventually stops.

## 8. Required metrics

Report per model/configuration/device/OS/driver/transport stratum:

- interrupt-to-stop-request;
- interrupt-to-adapter-acceptance;
- interrupt-to-device-drain;
- interrupt-to-loopback-silent;
- interrupt-to-acoustic-cessation;
- residual acoustic energy after the stop trigger;
- p50/p90/p95/p99 stop latency and uncertainty bounds;
- false-stopped rate (receipt says stopped while qualified output persists);
- missed-stop rate;
- post-revocation output events;
- observer-unknown and observer-fault rates;
- queue depth, audio buffer duration, dropped samples, and CPU/energy overhead.

Do not combine wired, Bluetooth, browser-remote, and hosted output into one latency distribution. Their output paths and measurement boundaries differ.

## 9. Fault-injection matrix

At minimum:
- interrupt during speech prepare, onset, sustain, and release;
- stop between audio chunks and during a partially consumed chunk;
- long queued output, driver buffer drain, and adapter backpressure;
- simultaneous human-resolution, turn-yield, consent revocation, and expiry;
- device switch, Bluetooth buffering, sink reconnection, and channel remapping;
- underrun, overrun, dropped samples, capture restart, recorder disconnect;
- microphone clipping, overload, ambient-noise spike, and overlapping human speech;
- clock regression, suspend/resume, stale mapping, and recorder restart;
- duplicate, delayed, reordered, replayed, or mismatched observation records.

Negative cases must produce failure or OutcomeUnknown, never a successful stop.

## 10. First experimental sequence

1. Freeze one Linux device profile, exact sink, sample rate, channel map, buffer quantum, and test waveform.
2. Measure the same action through adapter timestamps, device-drain reporting, and PipeWire sink-monitor capture.
3. Add a calibrated external microphone/recorder at a fixed documented location; establish calibration and clock mapping.
4. Run paired stop trials for each interruption phase and buffer/backpressure condition.
5. Inject observer faults and confirm that false stop receipts are impossible.
6. Repeat on a second output class (for example, Bluetooth) as a separate stratum.
7. Publish per-trial records, denominators, uncertainty, failures, raw-window digests, and the exact Benchmark Passport.

Initial qualification target: zero false-Stopped outcomes in the deterministic and fault-injection suite, with every ambiguous or incomplete observation remaining OutcomeUnknown. Set numerical p95/p99 product SLOs only after the first named-device baseline and measurement uncertainty are available. A stretch target may be declared in advance for a specific device profile, but must not be reported as achieved.

## 11. Privacy and retention

Prefer local processing and retain only signal measurements, event bindings, digests, calibration/clock records, and minimized failure metadata. Raw microphone windows can contain voices and bystanders; capture requires explicit consent and a documented retention/deletion scope. Do not publish raw samples unless the license and consent explicitly permit it. A digest proves byte identity only when the bytes are available; it does not authenticate the recording source or prove consent.

## 12. Qualification boundary

This protocol is authored design work. It does not establish that an acoustic observer exists, that PipeWire/ALSA integration is implemented, that a microphone is calibrated, that clocks are comparable, or that any output-stop latency has been measured. The existing action scheduler and receipt source contracts remain uncompiled/unexecuted until exact-head tooling and tests are available.
