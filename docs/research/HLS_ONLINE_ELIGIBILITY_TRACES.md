# HLS exact online eligibility traces

This note records the online-learning result for the diagonal Holographic Liquid State (HLS) cell.

## Prior art boundary

Efficient online recurrent gradients for diagonal recurrences are established prior art. Recurrent Trace Units (NeurIPS 2024) exploit diagonal recurrence to make real-time recurrent learning efficient, and e-prop / diagonal-RTRL families use forward eligibility traces to avoid storing a full BPTT trajectory.

The HLS claim is therefore narrower:

> The nonlinear, irregular-time, binding-equivariant diagonal HLS recurrence admits exact per-coordinate forward sensitivities for its six local parameter fields whenever the global cross-coordinate norm limiter does not activate.

This is not a claim that Symthaea invented RTRL or eligibility traces.

## Recurrence

For one HLS coordinate,

`h_next = F(h, x, dt; theta)`

with local parameters

`theta = {w_r, w_x, w_tau, w_gh, w_gx, b_g}`.

Because diagonal HLS has no cross-coordinate recurrence in this tranche, the exact forward sensitivity for local parameter `p` obeys

`E_p,next = (dF/dh) * E_p + dF/dp`.

The implementation carries six eligibility scalars per coordinate. For dimension `D`:

- eligibility storage is `6D` scalars;
- trace update work is `O(D)` for the fixed six-field parameterization;
- storage is independent of sequence length `T`.

A current loss derivative `dL/dh` is contracted locally with the traces to produce `dL/dtheta`.

## Continuous-time terms

The derivative includes the HLS liquid timescale path, not only the recurrent equilibrium path. In particular it differentiates through:

- magnitude-conditioned gate control;
- log-space liquid timescale control;
- `alpha = 1 - exp(-dt/tau)` including the implementation's exponent floor;
- the exact piecewise rational/clamped `fast_tanh` used by the forward cell.

The trace therefore covers irregular event intervals directly.

## Exactness boundary

The optional global L2 state limiter rescales the full vector when active. Its Jacobian contains cross-coordinate terms, so a purely coordinate-local trace would no longer be exact.

`step_with_eligibility` therefore computes the candidate state first. If the global limiter would activate, it returns `GlobalNormBoundWouldActivate` before mutating either cell state or eligibility state.

For bounded odd activations such as HLS tanh, experiments can set `state_norm_limit = infinity` when exact online gradients are required.

Contextual HLS is also outside the exact local theorem because invariant remote context makes coordinate `i` depend on magnitudes from other coordinates. Its gradients should be treated as a separate sparse/structured-Jacobian research tranche.

## Qualification

The implementation is tested in three ways:

1. traced and ordinary forward trajectories must be bit-identical when the exactness conditions hold;
2. analytic online gradients are compared with central finite differences through a multi-step irregular-time sequence for all six parameter families;
3. activation of the global norm-bound coupling must fail closed without changing either state or trace.

## Next experiment

The next learning experiment should use the online trace with a deliberately simple differentiable readout/loss. It should compare:

1. frozen diagonal HLS;
2. online-trained diagonal HLS;
3. the same readout trained with recurrent parameters frozen;
4. later, a matched diagonal recurrent baseline using online RTRL.

The state-tracking benchmark split, HDC codec, and query-only control should remain fixed so gains are attributable to recurrent learning rather than task leakage.
