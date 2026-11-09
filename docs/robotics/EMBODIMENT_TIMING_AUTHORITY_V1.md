# Embodiment Timing Authority v1

**Status:** implementation contract; qualification targets below are not measured results.  
**Related:** [PR #7634](https://github.com/Luminous-Dynamics/symthaea/pull/7634), [issue #1284](https://github.com/Luminous-Dynamics/symthaea/issues/1284), [physical-time benchmark contract](https://github.com/Luminous-Dynamics/symthaea/blob/fix/substrate-energy-rate-budget-v1/docs/robotics/PHYSICAL_TIME_AND_ENERGY_BENCHMARKS_V1.md).

## Principle

The component that owns the physical or simulated plant owns the authority to define the duration of its transition. Cognition can request a step or supply a command; cognition does not get to assert elapsed plant time merely because its scheduler ticked.

The bridge must declare one of these states:

- `Unspecified`: unqualified; the cognitive loop refuses to step it. This is the default for new implementations.
- `ConfiguredSimulation`: the bridge explicitly declares that its plant is an in-process model whose transition duration is the simulator/profile's configured `dt`.
- `BackendMeasuredClockRequired`: the bridge must own the clock measurement and return timing evidence for the specific transition. The generic configured-simulation adapter cannot satisfy this authority.

## Contracts

### Configured simulation

A simulation-time owner may use `step_with_timing` only with a valid `ConfiguredSimulationStep` record. The record's duration is model time, not wall duration. Cognitive tau, scheduler cadence, and device execution speed must not change it.

Existing in-process model bridges have been explicitly marked `ConfiguredSimulation`. This is a declaration about the current implementation's timing ownership, not a claim of real-hardware qualification.

### Backend-measured timing

A bridge that declares `BackendMeasuredClockRequired` must implement `step_with_backend_timing` and return both the result and the timing record actually used. The cognitive loop validates that:
- the timing record itself is valid;
- its step index matches the requested step;
- its source is a measured monotonic or device clock, not configured simulation time.

The bridge must additionally establish that the clock domain and epoch are authoritative for its plant. Constructors validate record shape; they cannot prove clock trustworthiness. Do not subtract timestamps across domains or epochs. Preserve the source timestamp separately from receive/arrival timestamps and command age. A validated record returned for an accepted step is also attached to the serialized `CycleMetadata.plant_step_timing` field so the evidence is available to replay/qualification tooling instead of existing only in debug logs; the field is omitted when no authorized timing record was accepted.

The trait's default backend-measured method returns an error. A hardware implementation cannot silently fall back to configured cognitive `dt`.

### Unspecified and event-driven backends

An implementation that has not declared its timing semantics is refused before stepping. External interfaces such as phone/browser/desktop automation need a separate event/transport timing contract; action execution latency is not automatically plant model time. They must not be marked `ConfiguredSimulation` merely to restore a call path.

## Qualification tests

1. An unspecified bridge receives zero calls.
2. A bridge that requires backend time does not fall back to configured simulation dt.
3. A measured-time return with configured-simulation source, a mismatched step index, missing domain/epoch, reversed interval, or inconsistent duration is rejected.
4. For a valid measured interval, reported dt must equal the source timestamp difference and retain its clock domain/epoch.
5. Changing cognitive substrate/tau while holding the plant profile, initial state, seed and applied command trace constant leaves plant model-time advancement exactly invariant.
6. Serialized `CycleMetadata.plant_step_timing` round-trips the accepted record without rewriting source/domain/epoch/duration and omits the field when no authorized timing record was accepted.
7. A feature-gated cognitive-loop integration test runs biological and photonic cognitive substrates with the same seed/input trace and simulator config, confirms tau differs, and verifies both per-cycle timing records retain identical configured dt and step indices. The test does not claim identical trajectories when cognition produces different commands.
8. For each bridge, record authority kind, source of timing evidence, supported modes, simulator/hardware profile, and negative-control result in a versioned receipt.

## Migration rule

Do not mass-change every bridge to `ConfiguredSimulation` simply because its name describes a robot. Inspect the actual implementation. Physics simulators may explicitly own configured model time; real hardware must own or consume qualified measured timing; external/event-driven devices need a separate timing contract. If ownership cannot be established, leave the authority `Unspecified` and fail closed.

## Methodology references

- ROS 2 control distinguishes steady/monotonic time for hardware `read/update/write` loop periods from ROS time used in simulation/message semantics: https://control.ros.org/rolling/doc/ros2_control/controller_manager/doc/clocks.rst
- ROS time may pause or jump backward, so it cannot be treated as interchangeable with monotonic physical elapsed time: https://github.com/ros2/design/blob/gh-pages/articles/130_ros_time.md
- NIST robotics agility work emphasizes measured reaction to unexpected events, re-tasking and combined virtual/real testbeds: https://www.nist.gov/programs-projects/agility-performance-robotic-systems
