# ROS 2 Platform Support Profile

**Profile date:** 2026-10-15  
**Status:** proposed support policy; no new distribution is qualified by this document alone.

This profile makes the ROS 2 bridge's platform intent explicit. It is not a statement that the optional `rclrs` dependency, message bindings, simulator integration, or physical hardware path has passed on any distro/OS combination.

**Relationship to the existing robotics roadmap:** this document owns distro/OS support policy only. Continue to use [Universal Embodiment v2 qualification and dependency gates (#1288)](https://github.com/Luminous-Dynamics/symthaea/issues/1288) and its existing simulator/ROS/HIL issue train for implementation and qualification. This file does not introduce a parallel simulator or HIL qualification program.

## Recommended distribution policy

| ROS 2 distribution | Official support horizon | Bridge policy | Evidence still required |
|---|---|---|---|
| Lyrical Luth | Through May 2031 | Forward target for long-lived deployments; start CI qualification on Ubuntu 26.04 amd64 and arm64 | Exact-head feature build, publisher/subscriber integration, message semantics, shutdown and fault tests |
| Jazzy Jalisco | Through May 2029 | Compatibility target for the established Ubuntu 24.04 device ecosystem | The same exact-head tests, with Jazzy-specific dependency and RMW versions pinned |
| Humble Hawksbill | Through May 2027 | Legacy-only compatibility when an existing customer/device requires it; do not choose it for a new long-lived deployment without a documented exception | Exact-head test run plus an explicit migration plan |
| Kilted Kaiju | Through December 2026 | Do not invest in a new qualified lane by default this late in its support window | Customer-specific exception and migration decision |
| Iron Irwini and older | End of life | No new support claim; provide migration guidance instead | None for new qualification work |

Lifecycle dates are sourced from the [official ROS 2 distribution table](https://github.com/ros2/ros2_documentation/blob/rolling/source/Releases.rst). Check that table before changing this policy: lifecycle status is time-sensitive.

## Operating-system and packaging policy

1. **Primary qualification environments:** use the official Tier 1 OS for the selected ROS 2 distribution. For Lyrical that means Ubuntu 26.04 amd64/arm64 for the Linux lane; for Jazzy use its documented Tier 1 environment, Ubuntu 24.04 amd64/arm64.
2. **NixOS developer hosts:** keep Rust tools, pinned source, project utilities, and reproducible non-ROS inputs in Nix where practical. Do not describe NixOS itself as an officially supported ROS Tier 1 platform. Run ROS integration tests inside a pinned, supported Linux container or VM first; document any native overlay as a separate, non-upstream packaging path.
3. **Containers are not proof of hardware support:** container smoke tests can qualify userspace integration, not kernel drivers, USB/device permissions, real-time scheduling, bus timing, motor authority, or physical safety.
4. **Pin the full test subject:** record distro, OS image digest, architecture, Rust toolchain, `rclrs` version, RMW implementation/version, bridge feature set, and exact Git commit SHA. A distro name by itself is not a reproducible environment.
5. **No implicit support expansion:** introducing a message crate, compiling with default features, or passing a simulator test does not qualify the optional `ros2` feature or any physical robot.

ROS 2's current Lyrical platform matrix lists Ubuntu 26.04 amd64/arm64 as Tier 1 and Ubuntu 24.04 as Tier 3 for Lyrical; on Tier 3 platforms ROS must be built from source. See the [official Lyrical supported-platform matrix](https://github.com/ros2/ros2_documentation/blob/rolling/source/Releases/lyrical/supported-platforms.rst).

## Qualification levels

Use explicit evidence levels rather than a single ambiguous “supported” label:

- **Declared:** intended target only; no test result implied.
- **Build-qualified:** the exact commit builds with the `ros2` feature in the pinned environment.
- **Simulation-integrated:** a deterministic local publisher/subscriber test passes with the required message schemas and the selected RMW.
- **Fault-qualified:** negative tests demonstrate that malformed, stale, missing, reordered, or non-finite observations are rejected or preserved as unknown according to the contract; transport loss and shutdown remain observable.
- **HIL-observed:** a named physical article and exact device/firmware/configuration are tested with instrument and calibration lineage recorded.
- **Deployment-qualified:** task-specific commissioning, field acceptance, operational constraints, and rollback evidence exist for the named article and workcell.

Each level requires the evidence of the preceding levels plus its own proof. A queued, skipped, stale, or unrelated workflow is not a pass; a result is valid only for the exact tested commit and environment.

## Minimum integration contract

Before exposing a distro as integration-qualified, test at least:

- sensor/proprioception ingestion with explicit units, reference frames, timestamp semantics, and freshness;
- command/request types separated from measured effects (a published command is not evidence that a motor latched it or that the robot moved);
- unknown/missing data kept distinct from measured zero;
- NaN/infinite values, invalid frames/units, stale samples, duplicated or out-of-order samples, and transport disconnects;
- lifecycle startup/shutdown and loss-of-peer behavior;
- replay of a frozen synthetic transcript with deterministic expected outputs;
- ROS-free default build remains available when the optional `ros2` feature is disabled.

For actuation, cognition and planning remain advisory. An independent controller, hardware interlocks, emergency stop, bounded envelope, and existing Symthaea authority/HAL gates must retain final authority. Do not route free-form model output directly to motor commands.

## Recommended CI sequence

1. Preserve the default ROS-free workspace build.
2. Add an isolated Jazzy bridge-feature build in a pinned Ubuntu 24.04 environment.
3. Add a separate Lyrical bridge-feature build in a pinned Ubuntu 26.04 environment.
4. Run simulated message-contract, lifecycle, malformed-input, disconnect, and replay tests for each lane.
5. Add a second RMW and hardware-in-the-loop only after the first lane is reproducible.
6. Publish a machine-readable receipt bound to the commit SHA and environment digest; report unrun tests as unrun.

Do not make distro support conditional on unrelated workspace-wide CI when a path-scoped bridge lane can provide stronger, faster evidence. Conversely, a green bridge lane does not qualify the entire Symthaea robotics stack.

## Commercial product boundary

This profile supports a software-first integration and qualification offer: onboard a customer's existing ROS 2 robot, establish a replayable baseline, report compatibility and sensor/transport faults, and produce a traceable hand-off package. It is deliberately hardware-vendor-neutral. It does **not** claim that Symthaea is already qualified for arbitrary robots, that a simulation result establishes physical safety, or that revenue/product-market fit has been demonstrated.
