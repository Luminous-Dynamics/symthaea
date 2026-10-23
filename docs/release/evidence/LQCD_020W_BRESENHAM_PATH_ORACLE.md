# LQCD-020W — generalized Bresenham off-axis path oracle

Independent standard-library Python subject for the bounded 3D generalized-Bresenham geometry used by later off-axis Wilson transport.

Exact executed subject SHA-256:

`c270fe43785b1c45cfd7e03acb4ac583782eb5e83193bf1fa977a6358242fdb9`

Canonical result SHA-256:

`884142f2645067e312380be738b618e85bc8bb727cbf6a3b0ea675858852c285`

## Qualified geometry

- one major-axis loop with independent error accumulators for the two remaining axes;
- deterministic major-axis tie break by axis index;
- every emitted step is one signed axial lattice link;
- the emitted path endpoint equals the requested displacement;
- path length equals the Manhattan length `|dx|+|dy|+|dz|`;
- reversing and sign-negating the exact emitted sequence gives the geometric return path;
- signed/permuted cubic-orbit orientations preserve Manhattan path length and exhaustive shortest-path multiplicity;
- exact local decision sequences are frozen for `(5,3,0)` and `(5,3,2)`.

## Cost regression

This oracle deliberately compares geometry work only; it does not claim that a single Bresenham transporter equals the exhaustive shortest-path average numerically.

- `(7,7,0)`: cubic-orbit bounded work = `168` link steps; exhaustive shortest-path symmetrization would require `576576` link steps (`3432` shortest paths per orientation).
- `(6,6,6)`: cubic-orbit bounded work = `144` link steps; exhaustive shortest-path symmetrization would require `2470051584` link steps (`17153136` shortest paths per orientation).

The bounded construction therefore makes the benchmark-scale off-axis families computationally tractable while leaving the exhaustive LQCD-020M operator available as a small-vector semantic/oracle path.

## Scientific boundary

This subject qualifies path geometry and deterministic cost only. It does not implement SU(3) transport, prove gauge invariance, establish equivalence to the exhaustive operator, restore rotational symmetry, determine a static potential, or produce a string tension. The production Wilson operator must separately qualify gauge covariance/invariance and its exact temporal-link authority boundary.
