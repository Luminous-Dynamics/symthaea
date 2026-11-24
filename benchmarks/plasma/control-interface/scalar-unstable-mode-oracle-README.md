# Scalar unstable-mode observation and authority oracle v0.1

This fixture is the first manufactured MROAC mode-level reference. It uses the discrete scalar system

`x[k+1] = exp(gamma*dt) * x[k] + b*u[k]`

with asynchronous sampled diagnostics and delayed actuator commands.

## Exact scalar observability information

A diagnostic sample at step k observes `y_k = c*x[k] + noise` with noise standard deviation sigma. Its Fisher-information contribution for the initial mode amplitude is

`(c/sigma)^2 * exp(2*gamma*dt*k)`.

Only samples that arrive within the declared finite horizon are included. Their sum is the scalar noise-whitened observability Gramian / Fisher information.

## Exact scalar controllability Gramian

A command sent at step j is applied at step `j*update_stride + actuator_delay`. Its unit-energy contribution to the final state at horizon H is

`b^2 * exp(2*gamma*dt*(H - 1 - applied_step))`.

The finite-horizon controllability Gramian sums those terms for commands applied before the final transition.

## Known-answer fixture

- gamma = 0.5 s^-1; frequency = 5 Hz; dt = 10 ms; horizon = 10 steps.
- Diagnostic gain 2, noise sigma 0.1, sample stride 2, delivery delay 1 step.
- Actuator gain 0.5, update stride 2, delivery delay 1 step.
- 5 available measurement samples and 5 actuator commands.
- Observability Fisher information = 2082.4542913757846.
- Controllability Gramian = 1.3015339321098656.
- Effective diagnostic bandwidth 25 Hz, effective actuator bandwidth 20 Hz.
- Conservative loop latency 65 ms = 0.0325 e-folds at gamma = 0.5 s^-1.

The scalar mode is qualified against the declared 2x bandwidth and 0.1 e-fold latency margins. The Rust function `evaluate_scalar_unstable_mode_interface` computes these results and rejects non-finite growth, excessive horizons and invalid noise / schedules.

## Scope

This is an analytic scalar-mode oracle only. It does not prove stabilizability, controllability of a multi-mode kinetic system, closed-loop stability, nonlinear Vlasov response, or any hardware capability. The scalar Gramians are a manufactured first step for #13484 and must be extended to multi-mode, non-normal and partially observed linearizations before physical controller claims.
