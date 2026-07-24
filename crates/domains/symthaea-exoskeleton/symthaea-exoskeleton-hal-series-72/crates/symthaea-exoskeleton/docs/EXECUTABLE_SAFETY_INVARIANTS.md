# Executable Safety Invariants

Series 43 adds a final-state monitor for contradictions that cannot be detected
inside one subsystem alone. The monitor observes the applied command, lifecycle,
independent safety lease, atomic commit, trust inputs, E-stop state, and measured
mechanical power.

Any violation latches zero authority and requires inspection acknowledgement.
Clearing the monitor does not arm the exoskeleton or enable actuator power.

Required invariants include:

- actuator power exists only in `Active` or explicitly derated `Degraded` state;
- the body-coupled safety kernel, co-processor lease, and atomic commit all agree;
- invalid time, consent, deployment, calibration, or watchdog state implies zero authority;
- E-stop implies zero command and zero power;
- safe-stop behavior never injects positive mechanical power;
- reported actuator power agrees with torque times measured velocity;
- all commands, velocities, authority values, and power values are finite.

This executable monitor supplements rather than replaces formal analysis,
independent hardware interlocks, HIL testing, and physical validation.
