# Kinetic multi-rate sensor/actuator interface v0.1

This contract freezes what a controller is actually allowed to observe and command:
- diagnostic measurement operator, sample period, transport latency, timestamp age, jitter, noise and dropout policy;
- actuator update period, delivery delay, bandwidth, amplitude, slew-rate and saturation policy;
- controller update period, worst-case execution deadline, maximum admissible observation age and fallback;
- explicit prohibition on hidden full-state access in deployment-score runs.

A schema-valid contract is not a stabilizability proof. It only makes the interface assumptions explicit and machine-checkable. Future qualification must compare observability and controllability on the same unstable-mode/QoI subspace, include delay and noise, and evaluate closed-loop behavior against a full-state oracle separately.

The checked-in example is a synthetic linear interface fixture only; it is not a kinetic simulation, real actuator specification, or closed-loop stability result. Research owner: [#13484](https://github.com/Luminous-Dynamics/symthaea/issues/13484).
