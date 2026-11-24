# Topology-aware kinetic coordinate lifting v0.1

This benchmark contract separates periodic and open physical domains.

## Periodic torus lift

For normalized coordinate xi modulo one, the periodic Fourier-character lift emits, for harmonics k=1..K:

- cos(2*pi*k*xi)
- sin(2*pi*k*xi)

The raw coordinate xi is deliberately omitted from this periodic feature vector. The feature map and all derivatives are periodic at the seam. A finite harmonic budget is part of the declared representation.

## Open-boundary lift

For an open interval, retain:
- normalized physical coordinate xi;
- distance to the lower boundary xi;
- distance to the upper boundary 1-xi.

The two faces remain distinct. Inputs outside [0,1] are rejected by the current Rust primitive; no periodic wrapping is performed.

## Known-answer fixture

The JSON includes:
- exact seam equality for xi=0 and xi=1;
- first-derivative seam equality for harmonics one and two;
- a global translation example with expected Fourier-feature rotation;
- a near-seam continuity bound;
- open-boundary endpoint features that remain distinct.

The Rust crate tests these analytic properties. This contract does not train a particle-field operator, measure physical field equivariance, or qualify a plasma simulation. Physical boundary type must be supplied by the cohort configuration, never inferred from a finite sample's observed particle extrema.


## Weighted Fourier empirical-measure embedding

The Rust crate provides `WeightedFourierMeasureEmbedding`. For a weighted particle measure

`mu_N = (1 / sum_i w_i) * sum_i w_i delta_(xi_i)`

it computes the first K complex Fourier moments

`m_k = sum_i (w_i / sum_j w_j) * exp(i * 2*pi*k*xi_i),  k=1..K`.

This gives a representation-aware measurement layer with four useful algebraic properties:
- particle permutation invariance;
- exact co-located particle split/merge invariance when the children’s weights sum to the parent weight;
- invariance to globally rescaling every particle weight by the same positive factor;
- global torus translation equivariance: a shift by delta multiplies moment k by exp(i*2*pi*k*delta).

The implementation max-scales weights before normalization to avoid overflow, uses compensated summation for each quadrature, and imposes both a particle-count cap and a harmonic-times-particle work budget.

A checked-in synthetic known-answer fixture demonstrates the first four moments for positions [0, 1/4, 1/2, 3/4] and weights [0.1, 0.2, 0.3, 0.4]. The focused workflow validates its schema and the Rust unit test checks the expected complex values.

**Important limitation:** a finite set of Fourier moments is not an injective representation of an arbitrary particle measure. It is a band-limited observable embedding, not a lossless microscopic state or substitute for high-k / echo response qualification. Compare its storage and future-response distortion through KIRD rather than calling it a complete kinetic reconstruction.
