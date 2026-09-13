# HLS trainable-parameter contract

This note records the learning boundary for the theorem-bearing Holographic Liquid State (HLS) cell.

## Structural theorem

For bipolar role binding `B_r(x) = r ⊙ x`, the HLS forward map is constructed so that

`F(B_r(h), B_r(x), dt) = B_r(F(h, x, dt))`

up to floating-point roundoff.

The proof depends on architecture, not on the numerical values of the trainable coefficients:

1. recurrent/input transforms are diagonal;
2. gate and liquid-timescale controls depend on magnitudes;
3. the activation is odd;
4. global norm bounding is invariant under bipolar sign binding.

Therefore all six diagonal fields may be trained without weakening the forward symmetry law:

- recurrent weight;
- input weight;
- tau-state weight;
- gate-state weight;
- gate-input weight;
- gate bias.

For dimension `D`, the cell exposes exactly `6D` trainable recurrent scalars.

## Atomic mutation rule

`HlsParameters` is the complete parameter snapshot. `set_parameters` validates every field before mutating any live field. A malformed dimension or non-finite scalar must fail closed.

Optimizer deltas use

`theta_next = clamp(theta + eta * delta, -B, +B)`

with finite step size `eta` and explicit positive absolute bound `B`. The candidate snapshot is completed and validated before installation.

## What this does not establish

This tranche does not claim that a particular learning algorithm improves HLS performance. It only establishes a safe and theorem-compatible parameter surface so optimizers can be compared independently.

The next learning tranche should therefore keep the following separate:

1. HLS parameterization;
2. optimizer/search algorithm;
3. readout training;
4. benchmark data split;
5. symmetry qualification after training.

A performance gain is only attributable to learned HLS dynamics if the same state-tracking codec/readout protocol and held-out world splits are retained.
