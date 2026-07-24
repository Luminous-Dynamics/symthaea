# Series 36 Release Gates

Series 36 closes continuity hazards that remain after nominal command safety is established. Passing software tests is not authorization for powered human-worn use.

## Clock integrity

- Every safety-relevant source has a fixed identity and reboot epoch.
- Replay, reorder, backward time, epoch change, rate drift, source substitution, and stale synchronization are injected.
- No untrusted or synchronizing clock can authorize motion.
- Independent hardware clock behavior is validated under thermal and supply variation.

## Restart and persistent state

- Clean, watchdog, brownout, panic, storage-corruption, and unknown reset paths are exercised.
- No reset restores a session, arm state, actuator command, or nonzero authority.
- Power-fail injection is performed at every journal-write boundary.
- Journal sequence, hash link, authentication, rollback floors, and accumulated maintenance counters survive interrupted writes.
- Watchdog, brownout, panic, and unclean resets require physical inspection before Standby.

## Calibration change control

- Calibration revisions are monotonic and previous/candidate digests are verified.
- Polarity changes, travel expansion, torque increase, actuator-scale increase, oversized zero/scale changes, reviewer reuse, and evidence rebinding are rejected.
- Bench, fit, and wearer-acknowledgement evidence is independently retained.
- A calibration update cannot bypass new-hardware enrollment requirements.

## Human-interface load

- Pressure, shear, slip, temperature, sensor loss, stale data, sequence loss, and calibration mismatch are tested per attachment.
- Left/right load asymmetry is exercised across representative gait and recovery maneuvers.
- Hard limits latch zero authority and require a physical skin/fit/attachment inspection.
- Sensor thresholds are established by validated hardware and qualified human-factors expertise; software defaults are not clinical limits.

## Integrated release evidence

- The Series 36 deterministic campaign passes in native and target builds.
- HIL uses the production clock source, persistent storage, actuator interlock, interface sensors, and deployment trust store.
- Endurance testing demonstrates that resets cannot erase maintenance debt.
- Independent safety review confirms that the production entry point consumes every readiness signal and uses the minimum authority ceiling.
- All evidence is bound to source revision, artifact digest, runtime configuration, hardware manifest, firmware, calibration, wearer-fit protocol revision, and test-equipment calibration.
