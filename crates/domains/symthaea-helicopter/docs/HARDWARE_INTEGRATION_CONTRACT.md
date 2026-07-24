# Hardware Integration Contract

Physical helicopter output is available only through
`HelicopterHardwareBridge::new_physical`. The constructor rejects any backend
that declares itself simulation-only. There is no automatic simulator fallback.

Before arming, the bridge requires:

- a physical backend and passing backend self-test;
- a mission-bound authority token with bounded lifetime;
- finite monotonic time and an explicit command watchdog;
- fresh, finite, strictly sequenced sensor frames;
- finite, strictly sequenced actuator command frames.

Any stale sensor, expired authority, sequence regression, time regression,
watchdog gap, invalid command, or I/O failure moves the bridge to `Faulted`,
invokes the backend's physical output-disarm operation, and clears authority.

The trait is the crate-side adapter boundary for `symthaea-hal`. A concrete HAL
adapter must prove its units, frame conventions, clock source, output-disarm
semantics, actuator feedback, and backend self-test before flight use.

## Sensor clock discipline

Physical sensor timestamps must be normalized into the host monotonic domain
before freshness or ordering decisions are made. `SensorClockDiscipline`
provides a bounded affine clock model, explicit lock state, offset/drift
evidence, and fail-closed stale/future rejection. A backend must not claim
freshness by comparing unrelated clock epochs directly.

## Multi-rate measurement snapshots

Corrected measurements remain asynchronous. `MultiRateSensorBus` retains each
channel timeline, bounds interpolation, rejects sequence/time regressions, and
requires fresh IMU, radar-altimeter, rotor-tachometer, and powertrain channels
before a control snapshot is considered complete. Optional sources may disappear
without being silently substituted for required sources.

## Real-time deadline evidence

Hardware qualification must provide `ControlCycleTiming` observations from the
actual scheduling and I/O boundary. `RealtimeControlMonitor` records start
jitter, execution time, sensor-to-actuator latency, individual deadline misses,
and consecutive-miss unsafe state. Simulator loop frequency or average
throughput is not accepted as evidence that physical control deadlines were met.


## Fault-containment evidence

A second lane is not accepted as redundancy evidence by name alone. The
`FaultContainmentArchitecture` must declare lane power, sensors, computers,
actuator paths, shared propulsion, and required dependency edges. Release
evidence must enumerate single-component faults that remove critical services
and identify cross-zone propagation from common power, buses, or shared
actuators. This architectural analysis does not replace physical FMEA/FMEDA.
