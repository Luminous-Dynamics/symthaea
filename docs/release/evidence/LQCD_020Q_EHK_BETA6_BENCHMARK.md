# LQCD-020Q — EHK β=6.0 Wilson-action benchmark extraction

Independent external source: Edwards, Heller, Klassen, *Accurate Scale Determinations for the Wilson Gauge Action*, arXiv:hep-lat/9711003.

Exact executed standard-library subject SHA-256:

`d79e3cae4a9eb0bd590c7889130861a71599d1e0c3ffd2ffa616597be2b4dc12`

Canonical extracted-result SHA-256:

`cc7a16813bd175b56cc9242425ec4c1f235a9f54483efe9dc173b87fbd3b28b2`

Frozen source facts for β=6.0:

- Wilson pure-gauge action;
- lattice volume `16^3 × 32`;
- 4000 configurations;
- Cabibbo–Marinari SU(2)-subgroup heat-bath plus microcanonical over-relaxation, typically 3:1 per sweep;
- spatial APE smearing with approximately `epsilon * n = 4`, reported to give at least 85% ground-state overlap even at large separation;
- measured off-axis vector families based on `(1,0,0)`, `(1,1,0)`, `(1,1,1)`, and `(2,1,0)`;
- string-tension analysis using 4-parameter, fixed-`e=pi/12` 3-parameter, and fixed-`e=pi/12,l=0` 2-parameter potential fits;
- quoted `a*sqrt(sigma) = 0.2189(9)`;
- quoted `r0/a = 5.369(9)`, `r4/a = 8.831(21)`, `r6/a = 10.89(3)`.

The executable extraction additionally records algebraic derived values `sigma*a^2 = 0.04791721` and `r0*sqrt(sigma) = 1.1752741`. The propagated errors stored for these derived combinations use simple linear/uncorrelated propagation only and are **not** source-quoted covariance-aware uncertainties.

## Authority boundary

This PR freezes an external benchmark target; it does not reproduce the benchmark, authorize a Symthaea ensemble, or establish a physical scale. A later campaign must bind this exact extraction hash and independently earn equilibrium, autocorrelation, topology, static-potential, fit-family, finite-volume, and continuum evidence.
