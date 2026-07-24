# Exoskeleton ↔ HAL integration architecture

## Decision

`symthaea-hal` is the canonical low-level hardware boundary, but its PCA9685
`ServoOutput` is **not** the exoskeleton actuator backend.

The exoskeleton produces an `ActuatorCommandFrame<21>` containing:

- explicit actuator mode;
- physical torque in N·m;
- monotonic sequence and timestamps;
- a short validity deadline;
- calibration revision and digest;
- one disabled reserved channel.

The HAL then independently enforces:

1. frame structure and finiteness;
2. freshness and monotonic sequence;
3. calibration identity;
4. hardware capability limits;
5. final command watchdog;
6. fail-closed output disablement.

The exoskeleton safety kernel remains the high-level, body-aware authority. The
HAL checks are defense in depth and may only reduce or reject its output.

## Why `HumanoidCommand` is not used

The existing humanoid/PCA9685 route names its vector `torques`, but converts the
values directly into target servo angles. A load-bearing exoskeleton cannot
safely treat normalized position as torque. The compatibility API remains for
bench-top humanoid rigs and is explicitly documented as a legacy compatibility path; new integrations must use the typed frame contract.

## Real hardware gate

`RealTransportUnavailable` is intentional. It must remain until a force-capable
backend can prove all of the following:

- closed-loop torque or impedance control at the actuator;
- measured current and joint torque feedback;
- independent hardware e-stop and power contactor;
- watchdog that removes drive power without host cooperation;
- identified firmware and calibration artifact;
- HIL fault-injection evidence;
- passive/backdrivable behavior after power removal.
