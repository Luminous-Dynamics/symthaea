# Symthaea Exoskeleton Evidence Gates

This file defines software evidence gates. Passing them does **not** certify a
human-wearable device and does not authorize physical actuation.

## Gate S0 — fail-closed startup

A new or reset safety kernel is disarmed. No consciousness score, learned
output, or command can arm it. The emitted command is zero torque, zero
stiffness, and zero damping.

## Gate S1 — independent fault handling

Emergency stop, watchdog expiry, stale sensors, non-finite observations,
non-finite commands, low battery, and joint-envelope violations remove motor
authority downstream of all learned systems. Emergency stop is latched and a
separate clear-plus-arm ceremony is required.

## Gate S2 — mechanical envelope

Every command is checked for per-joint magnitude, torque slew, soft travel
limits, velocity direction, aggregate mechanical power, stiffness, and damping.
Consciousness and FEP can only reduce the resulting authority ceiling.

## Gate C0 — one canonical control frame

Training and live embodiment consume the same fused representation of cognitive
intent, measured proprioception, disturbance-observer intent, and FEP state.
Privileged simulator human torques are replaced by the observer estimate before
encoding and training.

## Gate P0 — auditable dynamics

The simple plant records human, learned-assist, impedance, gravity, passive
damping, and net torque separately. Their sum must reconstruct net torque.
Battery draw is based on simulated actuator work plus idle power.

## Gate P1 — adversarial plant campaign

The deterministic campaign covers trips, fatigue, added payload, actuator
degradation, noisy intent estimation, torque saturation, and slew limiting. All
states and torque traces must remain finite.

## Gate F0 — full-frame force honesty

Human torque and exoskeleton torque are applied explicitly to all 20 modeled
joints. Phi never scales gravity, contacts, friction, or human forces. The model
currently remains a reduced-order articulated research simulator, not a
validated human biomechanics model.

## Gate H0 — transport honesty

The mock command sink validates channel count, finite normalized commands, and
watchdog behavior. Real I2C is intentionally refused until a calibrated
`symthaea-hal` adapter, hardware e-stop, current sensing, thermal limits,
command acknowledgements, and hardware-in-the-loop evidence exist.

## Required external work before human trials

- Formal hazard analysis and risk controls.
- Independent hardware safety controller and normally-safe power stage.
- Mechanical stops, torque/current limits, thermal protection, and emergency
  release.
- Calibrated sensors with timestamp and plausibility checks.
- Hardware-in-the-loop fault injection.
- Validated multibody/contact model and benchtop test rig.
- Applicable regulatory, ethics, and institutional approvals.

## Gate A0 — governed authority input

Raw Phi cannot directly increase assistance. Upgrades require finite, fresh,
high-confidence samples, hysteresis, and sustained dwell. Quality loss removes
authority immediately and upgrades occur one tier at a time.

## Gate K0 — validated calibration

Profile identity, revision, polarity, scale, neutral pose, travel limits, torque
ratings, and anthropometry must be internally consistent. Calibration can derate
hardware limits but cannot increase them.

## Gate RT0 — deterministic timing contract

Control frames use exact sequence continuity and monotonic integer timestamps.
Stale samples, period violations, time regressions, and exhausted deadline
budgets are explicit faults.

## Gate D0 — unified fault containment

All active faults resolve to one most-restrictive degraded mode. Critical power,
thermal, calibration, watchdog, battery, and emergency-stop faults latch until a
separate post-inspection clear ceremony.

## Gate T0 — applied-command replay

The fixed-capacity trace records the command after safety filtering together
with authority and safety reasons. Replay checks sequence, time, finite values,
torque bounds, and complete removal of torque and impedance at zero authority.
