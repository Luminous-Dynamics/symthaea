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
- minimum effective actuator bandwidth, limited by physical actuator bandwidth and digital command Nyquist rate;
- minimum effective diagnostic bandwidth, limited by sensor bandwidth and sampling Nyquist rate;
- conservative observation-to-actuation latency = maximum diagnostic age + controller WCET + maximum actuator command latency;
- diagnostic and actuator bandwidth multiples relative to a declared fastest target instability frequency;
- closed-loop latency as a fraction of the target instability e-folding time;
- an `interface_timescale_qualified` disposition against caller-declared minimum bandwidth multiple and maximum latency/e-fold fraction.

The timescale qualification is an interface-feasibility check, **not a stability proof**. It does not assert that the instability is observable/controllable from the chosen channels, that an external field can suppress it, or that the actuator model is physically valid. The target frequency, e-folding time and margins must come from an independently qualified dispersion / reference analysis for the exact cohort. A failed timescale qualification is useful: the interface is too slow or too bandwidth-limited for the declared mode, and the controller must not be evaluated as though it had an instantaneous full-state loop.

For the synthetic fixture: actuator command latency 26.5 ms; effective actuator bandwidth 20 Hz; effective diagnostic bandwidth 20 Hz; target mode frequency 5 Hz; diagnostic/actuator bandwidth multiples 4; e-fold time 100 ms; conservative loop latency 71.5 ms = 0.715 e-folds. With required bandwidth multiple 2 and maximum latency fraction 0.8, the synthetic interface timescale check passes. These are only derived synthetic values, not measured hardware timing or a closed-loop result.

The tests include an asynchronous two-diagnostic / one-actuator fixture and negative controls for stale timestamps, missed execution deadlines, hidden state and duplicate channel identity. These are timing-contract checks, not a proof of closed-loop stability or sufficient controllability.


## Coupled unstable-subspace authority

The coupled-mode owner is [#13564](https://github.com/Luminous-Dynamics/symthaea/issues/13564). The Rust primitive `evaluate_coupled_two_mode_authority` extends the scalar and two-quadrature oracles to a manufactured real two-state system:

`x[k+1] = A*x[k] + B*u[k]`, `y_j[k] = C_j*x[k] + noise_j[k]`.

It computes the actual finite-horizon observability Gramian from both diagnostic channels, whitened by their independent noise variances, and the controllability Gramian from the single actuator direction on its declared update/delivery schedule. Samples or commands arriving after the horizon do not count. Both eigenvalues of the declared 2-state transition must be unstable; this keeps the reported authority restricted to the declared unstable two-state subspace.

The fixture contains three matched known-answer cases:
1. **Positive:** distinct unstable growth factors, independent diagnostic channels and one actuator direction. The sampled dynamics make the pair both observable and controllable.
2. **Sensor-rank negative:** two collinear diagnostic rows and a delivery delay that leaves only one sample; the system remains controllable but is not observable.
3. **Actuator-rank negative:** repeated unstable eigenvalues, full independent observation, but one actuator direction; the system is observable but not controllable.

The checks report the smallest/largest Gramian eigenvalues and condition numbers, plus the Boolean authority disposition. They are exact finite-horizon linear-system checks only. They do not establish Vlasov–Poisson observability, nonlinear stabilizability, actuator feasibility or closed-loop stability. The matrix must eventually be supplied by a separately qualified linearization / dispersion analysis for the exact kinetic cohort.
