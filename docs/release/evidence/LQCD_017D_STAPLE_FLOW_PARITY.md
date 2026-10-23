# LQCD-017D direct-staple flow parity evidence

## Subject

`scripts/lqcd-topology-staple-parity.py`

Executed subject SHA-256:

`1d041728777a9d121590516765e4a372efbd61131b6e358bbc43257c4657995d`

The subject is Python-standard-library only and imports the independent finite-difference topology/flow oracle in the same branch. It does not import Rust or Symthaea implementation code.

## Analytic gradient under test

For a link `U_mu(x)` and the six-staple matrix `H` oriented so the local Wilson trace is `Re Tr(U H)`, define `X = U H`. With the Gell-Mann normalization `Tr(lambda_a lambda_b)=2 delta_ab` and Wilson action at beta=1,

`dS/dtheta_a = Im Tr(lambda_a X) / 3`

for the left perturbation `U -> exp(i theta_a lambda_a) U`.

The subject compares that direct expression against the LQCD-017D central finite-difference gradient for **every link and every one of the eight generators** on the explicit nontrivial `2^4` fixture.

## Executed result

```text
ok
max_gradient_abs_error=1.007493422022776e-09
max_flow_link_abs_error=1.0852447386312045e-12
analytic_action=8.1318570997778501
reference_action=8.1318570997827067
analytic_q=-0.00041190783630270052
reference_q=-0.0004119078363035159
```

The flow comparison uses the same `dt=1e-3`: one step is applied with the analytic staple gradient and independently with the finite-difference gradient. The resulting full gauge fields agree to about `1.1e-12` in the maximum complex link-entry difference.

## Authority boundary

This establishes the small-step algebra/parity target for an optimized six-staple Wilson-action flow on this fixture. It does **not** establish:

- higher-order integration accuracy at finite flow time;
- acceptable production step sizes;
- physical `t0` or `w0`;
- integer topology after smoothing;
- topology tunneling;
- continuum observables.

A production optimized flow must still preserve SU(3), pass exact-head Rust CI, retain a reference fallback for degenerate extents, and be tested over multi-step trajectories rather than relying on this one-step parity result alone.
