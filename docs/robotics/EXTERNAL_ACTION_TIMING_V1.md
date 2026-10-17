# External Action Timing V1

**Status:** implementation proposal / first adapter integration. No latency or hardware qualification result is claimed.

## Why this is a separate contract

A phone UI, desktop, browser, or remote tool environment is event-driven. It is not a continuously integrated physics plant. Keep at least four notions separate:

- cognitive-cycle duration;
- host monotonic transport/dispatch duration;
- device/source event time, when the source truly provides it;
- time since an observation was captured, which is only computable when trustworthy source-time semantics exist.

A successful ADB process exit is evidence about the host-side dispatch call, not proof that the Android application reached the intended postcondition. A screenshot returned over ADB has a host receive time; this adapter currently does **not** receive an authoritative timestamp for when the depicted UI state was captured. Therefore source time, observation age relative to source, and UI-settled time stay unknown.

## Receipt contract

ExternalActionTimingV1 is separate from PlantStepTimingV1 and has its own validator and ExternalActionEmbodimentStepV1.

Every record binds:

- schema version and caller-assigned step identity;
- action identity/type;
- host monotonic clock domain and epoch;
- device-session epoch;
- screenshot request start/return and success disposition;
- host receive timestamp and observation identity when bytes were returned;
- optional source timestamp only when accompanied by source clock domain and epoch;
- action-dispatch call start/return and success disposition when dispatch was attempted.

Unknown source time must remain unknown. Host receive time must never be copied into the source-time field to make age appear computable. Host monotonic values may be compared only within the same host clock epoch. Device-session epoch is a separate lineage boundary and is advanced when ADB capture failure and later recovery indicate disconnection/reconnection, or on an explicit bridge reset.

This receipt is instrumentation/evidence, not an actuator permission, safety certificate, application-level completion acknowledgement, or a claim that the environment is fresh. The general cognitive-loop dispatcher still needs a dedicated event-driven scheduling integration; this first tranche exposes the distinct bridge contract while leaving the existing fail-closed plant-time path intact.

## Frozen qualification corpus

Use an injectable fake-ADB/event-source implementation for deterministic CI. Freeze event traces, expected disposition, and an exact corpus hash.

| Case | Required disposition |
|---|---|
| normal capture + dispatch | preserve request/dispatch intervals and success separately |
| source timestamp unavailable | keep source timestamp/domain/epoch all absent |
| source timestamp present without domain or epoch | reject receipt |
| screenshot timeout / ADB error | record failed request; no observation ID or receive timestamp |
| delayed dispatch return | report measured host dispatch interval; do not call it plant dt |
| UI action process returns success but postcondition is absent | dispatch success may be true; goal/postcondition must remain unconfirmed |
| duplicate or late observation | do not present as fresh solely because host receive time is new |
| disconnect then reconnect | device-session epoch changes; prohibit cross-epoch age/latency subtraction |
| process/host clock epoch changes | prohibit cross-epoch monotonic timestamp subtraction |
| partial/reversed intervals | reject record |
| wrong step index | reject record |
| serialized fixture replay | same fixture + pinned identities yields byte-identical canonical receipt and rejection code |

## Proposed benchmarks (targets, not results)

1. **Contract negative controls:** 100% rejection of invalid schema, partial intervals, reversed intervals, missing action/device/clock identities, mismatched step IDs, and source timestamps without provenance.
2. **Source-time honesty:** 100% of source-less observations retain absent source timestamps; 0 synthetic source timestamps derived from host receive time.
3. **Epoch isolation:** 0 cross-epoch timestamp subtractions after disconnect/reconnect or host-clock restart.
4. **Replay determinism:** 100% byte-identical canonical receipts and dispositions for a frozen fake event corpus with fixed IDs and event trace.
5. **Telemetry separation:** publish p50/p95/p99/p99.9/max for cognitive cycles, capture/request latency, action-dispatch latency, observation receive age where valid, timeouts, dispatch errors, postcondition confirmation, and recovery. Never collapse them into a single score.
6. **Fault recovery:** report detection-to-reconnect and reconnect-to-first-valid-observation latency, with p50/p95/p99 and the exact injected fault.
7. **Real-device qualification:** repeat across named device/OS/ADB profiles; include device model, Android build, ADB version, transport, test-app state, and host/kernel profile. Do not promote fake-ADB timing into physical or fleet-wide claims.

No universal p99 milliseconds target is asserted for phone UI operations: transport, animation, load, and application-specific postconditions vary. Define SLOs per task/device profile and report misses rather than hiding them.

## Relevant methodology

- ROS 2 control documents separate host steady time for hardware-facing loops from ROS time semantics used for simulation/controller interfaces: https://github.com/ros-controls/ros2_control/blob/master/controller_manager/doc/clocks.rst
- ROS time design explicitly separates system, ROS, and steady clocks: https://github.com/ros2/design/blob/gh-pages/articles/130_ros_time.md
- NIST robotic agility work motivates task-level measures including reaction, re-tasking, and response to unexpected events: https://www.nist.gov/programs-projects/agility-performance-robotic-systems

## Limits / next tranche

- This contract does not infer device-side capture time from ADB transport timing.
- A successful ADB command does not establish that the intended application state was reached.
- Device epoch transitions based on observed ADB capture failure/recovery are best-effort session evidence, not a certified device clock epoch.
- Next tranche: wire step_event_driven into a dedicated cognitive-loop dispatcher and serialized CycleMetadata field, add injectable ADB tests, and test stale/duplicate observation admission before allowing event-driven adapters into normal orchestration.
