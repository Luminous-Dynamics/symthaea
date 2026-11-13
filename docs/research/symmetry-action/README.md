# Symmetry action representation-covariance fixtures

This directory begins A0 of [SYMM-ACTION-001](https://github.com/Luminous-Dynamics/symthaea/issues/11416).

## Frozen fixture

`representation-covariance-v1.json` records a canonical 2D rotation generator and a rational, non-orthogonal change of coordinates. It freezes:

- the original generator and Euclidean metric;
- the transformed generator (A'=DAD^{-1});
- the transformed metric (G'=D^{-T}GD^{-1});
- a scalar-function family and one shared state in both coordinate systems;
- canonical and transformed metric-aware squared-cosine reference values;
- a wrong-metric negative control.

The coordinate convention is (x'=Dx). Covectors therefore transform as row vectors (p'=pD^{-1}), while tangent vectors transform as (v'=Dv). The metric-aware normalized residual is

[
ho_G^2 = rac{(p v)^2}{(pG^{-1}p^T)(v^TGv)}.
]

## What this fixture tests

The same geometric action must produce the same residual in both coordinate representations. The conjugated generator is not generally skew under the identity matrix, so the transformed metric is part of the action declaration, not optional metadata.

The near-symmetry values are calibration negatives. In particular, the (epsilon=10^{-4}) sample is below an illustrative (10^{-6}) threshold. The analytic supremum is

[
sup_{(x,y)
e 0}ho^2 = rac{epsilon^2}{(2+epsilon)^2},qquad epsilon>-1.
]

This shows why a fixed residual tolerance encodes a measurable acceptance band. The exact polynomial lane should reject nonzero (epsilon) for supported polynomials; this residual fixture is especially useful for calibrating unsupported/non-polynomial sampled cases.

## Evidence limitations

The JSON decimals are analytic-reference calculations evaluated in floating point, not Rust test output. This fixture does not establish a false-positive rate, prove a sampled candidate, validate a vector field, establish a physical momentum map, or qualify a simulator. Future test harnesses should compare both representations using the same exact fixture and should report raw residuals, conditioning, skipped samples and the complete evidence profile.
