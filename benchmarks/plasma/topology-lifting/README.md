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
