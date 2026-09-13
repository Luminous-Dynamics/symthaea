# HLS fixed associative readout

This note records the differentiable decoder used to turn the state-tracking task into a learning signal for exact diagonal-HLS eligibility traces.

## Design constraint

The decoder must not become a second model that can solve the task while recurrent state remains weak. The first learning experiment therefore uses **no trainable decoder parameters**.

The state-tracking codec supplies a deterministic bipolar unitary query key `k(q)` composed from:

- query type;
- target entity/object identity;
- historical lag role.

The recurrent state is treated as an HDC associative memory. Prediction is

`y = k(q) ⊙ h`.

Because `k(q)` is bipolar unitary, the map is self-inverse, norm preserving, and has a diagonal Jacobian whose entries are exactly the query-key signs.

If an ideal memory contains

`h = k(q) ⊙ answer`,

then the same key recovers

`k(q) ⊙ h = answer`.

## Loss

The target is the benchmark's deterministic answer hypervector `t`. The fixed decoder uses an epsilon-regularized cosine objective:

`ny = sqrt(y·y + eps^2)`

`nt = sqrt(t·t + eps^2)`

`c = (y·t)/(ny nt)`

`L = 1 - c`.

The exact prediction-space gradient is

`dL/dy = -t/(ny nt) + (y·t)y/(ny^3 nt)`.

Since `y = k ⊙ h`,

`dL/dh = k ⊙ dL/dy`.

That current-state learning signal is contracted with `HlsEligibilityTrace` to obtain gradients for all six recurrent HLS parameter fields.

## Why this is a useful first decoder

- zero learned decoder weights;
- O(D) query cost;
- exact analytic state gradient;
- native HDC binding/unbinding semantics;
- same answer codebook already used by the model-agnostic benchmark;
- decoder capacity cannot grow independently of recurrent state.

This does not establish that the query-key algebra is the best possible HDC readout. It is deliberately a low-capacity diagnostic head.

## Qualification

The implementation requires:

1. a perfect synthetic association `key ⊙ answer` to decode the exact answer with near-zero loss;
2. finite loss/gradient at an all-zero recurrent state;
3. analytic `dL/dh` to agree with central finite differences at multiple state coordinates;
4. the resulting learning signal to contract through exact HLS eligibility traces into a full `6D` recurrent parameter gradient.

## Next experiment

For an exact episodic-gradient experiment, keep HLS parameters fixed throughout one generated world:

1. reset recurrent state and eligibility trace;
2. stream the observable initialization prefix and mutation events with `step_with_eligibility`;
3. at each scored query compute the fixed associative loss and `dL/dh`;
4. accumulate `dL/dtheta = E^T dL/dh`;
5. update HLS parameters only after the episode completes;
6. evaluate on held-out world seeds without parameter updates.

Updating recurrent parameters in the middle of the same trace would change the parameter trajectory being differentiated and should not be described as the same exact fixed-parameter episode gradient without additional derivation.
