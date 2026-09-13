# LQCD-020P — Declared lattice-potential fit family

This branch is the Rust production counterpart to independent oracle PR #2439 (LQCD-020O).

It evaluates three preregisterable correlated linear models on exactly the same potential points and covariance:

1. `V0 + sigma*r - e*C_lat + l*(C_lat - 1/r)` with `V0, sigma, e, l` free;
2. the same model with `e = pi/12` fixed and `l` free;
3. `V0 + sigma*r - (pi/12)*C_lat` with `l = 0`.

The module returns all three fits side-by-side and deliberately exposes no `best_model`, automatic model ranking, fit-range search, point dropping, covariance regularization, or model averaging.

The regression fixture is pinned to the exact LQCD-020O numerical targets. Passing Rust exact-head CI remains required before this implementation may be treated as executable-qualified.

Scientific boundary: this is model-family algebra/statistics qualification only. It does not establish a physical fit range, string tension, scale, continuum limit, or benchmark reproduction.
