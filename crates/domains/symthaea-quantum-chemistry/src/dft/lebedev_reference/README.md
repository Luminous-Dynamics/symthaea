# Lebedev quadrature reference data (Phase Q5c, 2026-07-17)

`lebedev_011.txt` is the raw, unmodified degree-11 (N=50) Lebedev quadrature rule, fetched via
`curl` (not `WebFetch`, not memorized) from John Burkardt's "Sphere Lebedev Rule" dataset (Florida
State University), the standard, widely-mirrored public-domain republication of Lebedev & Laikov's
original tables:

```
https://people.sc.fsu.edu/~jburkardt/datasets/sphere_lebedev_rule/lebedev_011.txt
```

Format: one row per point, `phi_degrees theta_degrees weight`, where
`x = sin(theta)*cos(phi), y = sin(theta)*sin(phi), z = cos(theta)`.

Vendored for provenance/reproducibility -- `dft/grid.rs::lebedev_50()`'s constants (weights and the
`p`, `q` orbit-generator values) were derived from this file. Verified before use: 50 rows, weights
sum to `1.000000000000002` (~1, floating-point exact).

References:
- Lebedev, V. I. (1976). Zh. Vychisl. Mat. Mat. Fiz. 16, 293-306.
- Lebedev, V. I. & Laikov, D. N. (1999). Dokl. Math. 59, 477-481.
