# Curved causal-set order-statistic inversion v0.1

Date: 2026-11-08  
Status: deterministic QMC calibration + finite-sample synthetic example; exact-head CI pending  
Parent: [CST-CURV-001 #11096](https://github.com/Luminous-Dynamics/symthaea/issues/11096)  
Programme: [CTGP PR #10802](https://github.com/Luminous-Dynamics/symthaea/pull/10802)

## Scientific boundary

The current curved sampler models the conformal 1+1D FRW metric

`ds² = a(eta)²(-deta² + dx²)`, `a(eta)=exp(kappa*eta²/2)`

over a symmetric Alexandrov diamond of conformal-time extent T. Its causal relation in null coordinates is the same componentwise order as flat 1+1D Minkowski because conformal rescaling preserves the light cones. Curvature affects the sampling measure through `sqrt(|g|)=exp(kappa*eta²)`.

A scale-free order statistic on a rescaled diamond cannot identify kappa and T independently. The dimensionless combination is

`q = kappa * T²`.

The new estimator therefore returns q. It must not report a dimensionful curvature unless T is independently calibrated.

The inference is conditional on the declared metric, the Alexandrov region and a volume-faithful fixed-cardinality (Poisson conditioned on N) sprinkling model. Detector censoring, eta-dependent selection or an unknown region shape can bias the order statistic and must be modelled separately.

## Executable primitive

`symthaea_causal_set_lab::curved_order_statistics` provides:

- `qmc_expected_comparable_pair_fraction(q, P)`: deterministic Halton/QMC reference for the expected fraction of distinct event pairs that are causally related.
- `estimate_dimensionless_curvature(f_observed, q_max, P, iterations)`: finite-QMC monotone calibration and bisection over q.
- `CurvedOrderFractionReference` and `DimensionlessCurvatureEstimate`: outputs retain q, the predicted/observed fraction, quadrature point count and iteration count.
- A weighted two-dimensional dominance calculation using a Fenwick tree, with O(P log P) work.
- An independent O(P²) pair-loop implementation inside tests as a differential oracle.

This is a finite deterministic reference calculation. It is not an exact analytic expectation. The inverse routine does not return a finite-sample confidence interval; that interval must be estimated separately from independent causal-set sprinklings.

## Why the Fenwick method is exact for the declared QMC point set

For two points in normalized null coordinates `(u,v)∈[0,1]²`, strict causal order is equivalent to strict componentwise order. Sort all QMC points by u. For each point, the Fenwick tree stores weights of points with smaller u, indexed by compressed v coordinate. A prefix query sums weights with smaller v; multiplying by the current point weight counts each comparable unordered pair once. The final numerator doubles this weight sum to account for both pair orientations, while the denominator is the total weight of all ordered distinct pairs.

The independent pair-loop tests check equality on fixed point sets. This is an algorithmic identity for those points, not a physical validation result.

## Validation protocol

The unit tests cover:
- Fenwick QMC versus the pair-loop oracle at q=0, 0.5, 2 and 4;
- flat q=0 reference near the 1+1D Minkowski value 1/2;
- scale rescaling `(kappa,T) -> (16*kappa,T/4)` preserving q and the reference;
- inverse recovery of a q value from its own finite-QMC reference;
- curved-versus-flat reference separation;
- invalid QMC counts, negative q and out-of-range observed fractions.

The executable `curved_order_inversion` uses one deterministic volume-faithful synthetic sprinkling, compares its empirical order fraction with flat and curved QMC references, estimates q, and displays the rescaling degeneracy.

## Required next benchmark

This v0.1 does not establish finite-sample uncertainty or a SOTA estimator. The next experiment should:
1. freeze q grid and independent seed corpus;
2. repeat each q with 30+ independent seeds and multiple event counts;
3. report bias, RMSE, empirical 90/95% interval coverage and reference quadrature convergence;
4. compare the flat Myrheim–Meyer estimate as a deliberately misspecified negative control;
5. inject eta-dependent thinning / detector selection and show the resulting bias;
6. refine QMC points (e.g. 2,048 -> 8,192 -> 32,768) to separate reference discretization error from sampling error;
7. report runtime and peak memory for Fenwick versus pair-loop methods.

The calibration should abstain on a non-monotone finite-QMC curve or an observed value outside the declared q interval. A production estimator must also propagate calibration/reference uncertainty rather than treating the QMC output as exact.

## Claim ceiling

A passing test establishes a numerical reference and a conditional inverse for the specified conformal 1+1D FRW volume-sprinkling model. It does not establish causal-set dynamics, manifold recovery, a continuum limit, spacetime discreteness or quantum gravity.
