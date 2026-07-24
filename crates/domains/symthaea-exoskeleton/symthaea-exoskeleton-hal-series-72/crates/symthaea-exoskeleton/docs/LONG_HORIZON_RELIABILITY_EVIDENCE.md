# Long-Horizon Reliability Evidence

Instantaneous torque, velocity, watchdog, and passivity limits are necessary but
not sufficient for a wearable robot. Release evidence must also cover cumulative
runtime, direction reversals, mechanical energy throughput, repeated recoverable
faults, deadline misses, and thermal exposure.

## Required counters

The reliability monitor records monotonic interval counters for:

- operating time and executed control ticks;
- per-joint torque-direction reversals above a deadband;
- absolute mechanical energy throughput;
- real-time deadline misses;
- recoverable fault occurrences; and
- peak actuator temperature.

Counters must be retained in the evidence trace and, for hardware, persisted in
a power-loss-safe maintenance record. Resetting software state is not proof that
mechanical wear disappeared.

## Runtime policy

1. **Nominal:** all mission-profile quantities remain below derating thresholds.
2. **Derated:** authority is capped while the session is brought to a controlled
   stop or inspected.
3. **Maintenance required:** authority is zero until an authorized maintenance
   ceremony records the physical inspection and replacement state.
4. **Shutdown required:** critical faults or thermal shutdown thresholds remove
   authority immediately and latch the independent power path where available.

## Evidence campaign

A release candidate should complete accelerated soak and fault-injection runs
covering at least:

- the maximum intended continuous session plus margin;
- repeated start/stop and direction-reversal cycles;
- hot and cold environmental envelopes;
- battery sag and transport latency;
- intermittent encoder, IMU, current, and temperature faults;
- passivity-tank depletion and recharge cycles;
- consent expiry and operator-triggered revocation; and
- restart/replay continuity of maintenance counters.

The embodiment applies the resulting authority ceiling after the independent
safety kernel and passivity supervisor. Maintenance-required or shutdown states
therefore replace the final command with a fully backdrivable command and disarm
the safety kernel before the plant or HAL can consume it.

The compact software campaign in this crate verifies policy behavior. It is not
a substitute for component endurance data, HIL soak tests, or physical teardown
inspection.
