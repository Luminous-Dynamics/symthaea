# Actuator Diagnostics and Predictive Power Envelope

## Physical feedback contract

A command transmission is not accepted as evidence of execution. Each actuator feedback frame carries:

- a monotonic sequence and sample timestamp;
- the calibration revision and digest;
- measured position, velocity, torque, current, temperature, enable state, and validity.

The residual monitor detects persistent tracking error, unresponsiveness, polarity mismatch, unavailable feedback, runaway torque, overcurrent, and overtemperature. Critical conditions request a latched shutdown.

## Electrical and thermal envelope

The predictive power supervisor uses a conservative first-order thermal model, measured temperature when available, current limits, bus-voltage headroom, pack current, and state of charge. It can only derate or shut down.

Missing temperature feedback imposes a conservative per-channel ceiling. The model is downstream of drive-local current limits and is not a replacement for fuses, thermal cut-outs, contactors, or battery-management protections.

## Release evidence

Real hardware release requires traceable calibration, independent torque measurement, current-sensor calibration, thermal soak tests, locked-rotor testing behind a physical barrier, undervoltage testing, contactor-drop verification, and proof that all critical diagnoses disable power within the allocated fault-tolerant time interval.
