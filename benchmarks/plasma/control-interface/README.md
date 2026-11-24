# Kinetic multi-rate sensor/actuator interface v0.1

This contract freezes what a controller is actually allowed to observe and command:
- diagnostic measurement operator, sample period, transport latency, timestamp age, jitter, noise and dropout policy;
- actuator update period, delivery delay, bandwidth, amplitude, slew-rate and saturation policy;
- controller update period, worst-case execution deadline, maximum admissible observation age and fallback;
- explicit prohibition on hidden full-state access in deployment-score runs.

A schema-valid contract is not a stabilizability proof. It only makes the interface assumptions explicit and machine-checkable. Future qualification must compare observability and controllability on the same unstable-mode/QoI subspace, include delay and noise, and evaluate closed-loop behavior against a full-state oracle separately.

The checked-in example is a synthetic linear interface fixture only; it is not a kinetic simulation, real actuator specification, or closed-loop stability result. Research owner: [#13484](https://github.com/Luminous-Dynamics/symthaea/issues/13484).


## Semantic timing audit

The Rust crate exposes `validate_sensor_actuator_timing_contract_json` and the `sensor_actuator_contract_audit` CLI. Beyond schema shape, it checks:
- diagnostic timestamp-age bound covers sampling period + transport latency + jitter + clock skew;
- every diagnostic's declared age is within the controller's maximum allowed observation age;
- controller WCET plus clock skew fits its deadline, and the deadline fits its update period;
- diagnostic and actuator channel IDs are unique;
- hidden full-state access is rejected;
- synthetic fixtures cannot claim closed-loop execution or stability.

The timing summary also reports:
- worst-case actuator command latency = update-period wait + delivery delay + clock skew;
- the minimum effective bandwidth across actuator channels, limited by both physical actuator bandwidth and each channel's digital command Nyquist rate;
- a conservative observation-to-actuation latency estimate = maximum diagnostic age + controller WCET + maximum actuator command latency.

For the synthetic fixture these are 26.5 ms actuator command latency, 20 Hz minimum effective actuator bandwidth, and 71.5 ms conservative loop latency. These are derived from the synthetic contract only, not measurements of real hardware.

The tests include an asynchronous two-diagnostic / one-actuator fixture and negative controls for stale timestamps, missed execution deadlines, hidden state and duplicate channel identity. These are timing-contract checks, not a proof of closed-loop stability or sufficient controllability.
