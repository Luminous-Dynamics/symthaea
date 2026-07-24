# Human-worn hardware-in-the-loop release gates

No human-worn powered test is permitted until every gate has durable evidence.

## Command contract

- [ ] Torque and impedance commands remain in SI units end to end.
- [ ] Every frame carries a monotonic sequence and expiry.
- [ ] Duplicate, reordered, future, and expired frames are rejected.
- [ ] Calibration digest mismatch disables the backend.
- [ ] Unsupported mode and capability excess disable the backend.

## Independent shutdown

- [ ] Physical e-stop removes actuator power without software cooperation.
- [ ] HAL watchdog removes output after missed frames.
- [ ] Host process exit, panic, I/O error, and deadline miss produce passive state.
- [ ] Power-cycle returns disarmed and requires a new enable ceremony.

## Sensor and actuator evidence

- [ ] Dual-channel joint angle disagreement is fault-injected.
- [ ] Current, torque, encoder, and temperature faults are fault-injected.
- [ ] Reversed polarity and wrong-joint mapping are detected before arming.
- [ ] Maximum torque, slew, velocity, power, and thermal envelopes are measured.
- [ ] Backdrivability is measured with drive power removed.

## Test progression

1. pure mock transport;
2. controller-in-the-loop replay;
3. powered actuator on rigid fixture;
4. instrumented dummy load;
5. suspended exoskeleton without wearer;
6. unpowered wearer fit test;
7. powered wearer test under formal protocol.
