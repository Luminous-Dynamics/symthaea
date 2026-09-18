# Visual Observation Identity v2

Status: VIS-000R refinement stacked on VIS-000.

## Why this refinement exists

VIS-000 correctly separates `Observed`, `Remembered`, `Inferred`, `Predicted`, `Simulated`, and `Counterfactual`, but its first observation identity shape used only:

```text
(source_id, frame_id, captured_at_us)
```

That is insufficient for long-lived embodied cognition:

- a capture process can restart and reuse frame numbers;
- a camera can reconnect and restart a device-local counter;
- two timestamps measured in microseconds may belong to incomparable clock domains;
- numeric timestamp equality does not prove shared acquisition time.

VIS-000R fixes this before object memory, multi-camera fusion, or world-state persistence depend on the weaker shape.

## Exact observation identity

A visual observation is now identified by:

```text
VisualStreamRef {
    source_id,
    stream_epoch,
}
+
frame_id
+
captured_at_us
+
VisualCaptureClock
```

`source_id` and `stream_epoch` are both non-zero and validated during construction and deserialization.

The source owner is responsible for minting stable source identity and a fresh stream epoch whenever frame/timestamp counters may restart or become ambiguous. Vision code must not derive the epoch from frame number, wall-clock time, model output, or crop position.

## Clock domains

`VisualCaptureClock` is explicit:

- `UnixEpoch`
- `StreamMonotonic`
- `DeviceLocal`
- `Unspecified`

The clock label describes the semantics of the numeric timestamp. It does **not** prove clock synchronization quality, authenticity, oscillator stability, or a bound on timestamp error.

## Ordering rules

### Frame ordering

`frame_ordering()` is defined only when two observations belong to the same exact `(source_id, stream_epoch)`.

A restarted stream that reuses `frame_id = 42` is not continuous with the prior stream's frame 42.

### Timestamp ordering

`timestamp_ordering()` is intentionally conservative:

- `UnixEpoch` may be numerically compared across streams, while making no synchronization-accuracy claim;
- `StreamMonotonic` may be compared only inside one exact stream epoch;
- `DeviceLocal` may be compared only inside one exact stream epoch;
- `Unspecified` is never numerically ordered by the provenance API.

Consumers requiring bounded cross-device timing must use a separately qualified synchronization/time-integrity receipt rather than treating equal clock labels as synchronization proof.

## Wire invariants

`VisualStreamRef` fields are private. Deserialization rejects:

- `source_id = 0`;
- `stream_epoch = 0`.

`VisualObservationRef` fields are private and can only contain a validated stream reference.

VIS-000's fail-closed `VisualEvidence` deserialization remains unchanged: generated/inferred/recalled state cannot populate the direct-observation slot contrary to its origin.

## Authority boundary

Observation identity and clock semantics are evidence metadata only. They grant no camera movement, robotics, motor, actuation, targeting, execution, or physical authority.

## Migration rule

VIS-001 structured foveation evidence must be refined to consume a caller-owned `VisualStreamRef` plus the correct capture clock semantics rather than a source id alone.

Later camera/screen/phone capture owners should mint the stream reference at the actual observation boundary and propagate it through foveation, tracking, memory, world-state, prediction verification, telemetry, and language-facing provenance.

## Nonclaims

VIS-000R does not establish synchronized cameras, calibrated clock error, globally unique hardware identity, improved recognition, better tracking, or stronger world-model accuracy. It prevents weaker identity/time assumptions from silently becoming those stronger claims.
