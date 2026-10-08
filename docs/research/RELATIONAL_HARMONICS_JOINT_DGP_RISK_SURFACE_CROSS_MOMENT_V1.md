# RH-006 Joint-DGP Finite-Sample Risk Surface — v1

## Status

Research diagnostic only.

- applicability = `not-approved-for-execution`
- selection = `stop-assumption-failure`
- formal p-value = disabled
- formal confidence interval = disabled

## Main result

The prior endogeneity stress showed that the independence-based finite-sample equal-risk root can cease to be an equal-risk point when one relational feature innovation shares shocks with the outcome innovation.

The new derivation makes the reason explicit.

For the declared shared-shock construction,

`eta_t = sqrt(1 - omega^2) z_t + omega q_t`

and

`e = L eta`.

Conditional on the shared innovation path `q`,

`E[e | q] = omega L q`

and

`Var(e | q] = (1 - omega^2) L L'`.

For a fixed realized feature path and fixed rolling ridge operators, write the incremental relational mean as

`mu(gamma) = mu0 + Z gamma`.

Then the aggregated finite-sample squared-loss difference remains exactly quadratic in `gamma`:

`Delta(gamma; omega) = gamma' Q gamma + (b0 + omega b1)' gamma + c0 + omega c1 + omega^2 c2`.

Here `b1`, `c1`, and `c2` are the explicit conditional-mean / cross-moment contributions. The quadratic curvature matrix `Q` is unchanged by `omega` for this particular shared-shock construction.

Thus the uncovered problem is not loss of polynomial geometry. It is an enlarged nuisance state required to locate the correct null surface.

## Global topology

For a ray `gamma = t u`, the discriminant is

`D(u; omega) = u' R(omega) u`

with

`R(omega) = b(omega)b(omega)' - 4 c(omega) Q`.

A 40-path deterministic execution using the prior endogeneity seed `20261080` gives:

| omega | R signature on all 40 paths | median largest eigenvalue | median middle eigenvalue | median smallest eigenvalue |
|---:|---|---:|---:|---:|
| 0.00 | `+++` | 2.106e-4 | 1.493e-4 | 1.766e-5 |
| 0.25 | `+++` | 1.934e-4 | 4.035e-5 | 4.643e-6 |
| 0.50 | `--+` | 1.476e-4 | -3.619e-5 | -2.994e-4 |
| 0.75 | `--+` | 8.264e-5 | -1.150e-4 | -8.581e-4 |
| 0.90 | `--+` | 3.979e-5 | -1.856e-4 | -1.301e-3 |

The lower two eigenvalues cross zero together because at `c(omega)=0` the discriminant matrix becomes `bb'`, which is rank one.

The pathwise `c(omega)=0` crossing quantiles are:

`0.2724, 0.2779, 0.2924, 0.3052, 0.3165`

for the 0%, 10%, 50%, 90%, and 100% quantiles.

This gives a stronger statement than the two-ray stress:

> The joint-DGP null surface changes topology from all-direction real-root support to a cone-supported real-root regime at moderate endogeneity.

## Directional versus global null support

Both predeclared weak and strong rays have 100% root support at `omega=0` and `0.25`, then 0% support at `omega=0.5`, `0.75`, and `0.9` across the 40 paths.

But the full discriminant matrix still has a positive eigenvalue at `omega=0.9` on every path.

Therefore:

`direction-specific null support failure != global null-surface disappearance`.

The supported equal-risk set has migrated into a proper coefficient-direction cone.

That is scientifically important because a directional scientific estimand can become undefined under a joint DGP even while a global composite equal-risk surface remains nonempty.

## Endogeneity displacement at the old root

Evaluating the joint surface at the `omega=0` independence-based root gives:

| Direction | omega=0.25 | omega=0.50 | omega=0.75 | omega=0.90 |
|---|---:|---:|---:|---:|
| Weak | -3.091e-4 | 7.107e-4 | 3.060e-3 | **5.107e-3** |
| Strong | 1.832e-3 | 4.993e-3 | 9.482e-3 | **1.281e-2** |

The sign convention is unchanged: positive means the relational model has lower MSE.

This reproduces the qualitative result of the earlier Monte Carlo stress while providing the finite-sample algebra that generates the displacement.

## Exact mechanics check

The analytic conditional surface was checked against direct outcome simulation on 4 feature paths, 2 directions, 3 omega values, and 1,000 outcome draws per cell: 24 checks.

- maximum absolute discrepancy = `3.67e-5`
- mean absolute discrepancy = `1.26e-5`
- discrepancy RMSE = `1.63e-5`

This is an identity/mechanics check only. It does not establish statistical size, bootstrap validity, empirical predictive performance, or validity of the Rust implementation.

## Implication for the next inference bridge

The next problem is now well-posed:

1. Define the scientifically admissible joint feature/outcome nuisance class.
2. Decide whether that class is parametric/restricted or broad enough to require uniform treatment.
3. Construct a resampler that preserves that joint law and refits the exact rolling estimator when required.
4. Stratify calibration over at least joint cross-moment strength, discriminant topology, Q curvature, branch conditioning, and effect radius.
5. Only then test prospective null size and weak-identification robustness.

The current literature remains methodological precedent, not a plug-in proof. Giacomini–White frames inference around forecasting methods and finite-sample estimator behavior; Zhu–Timmermann show that rolling-window conditional-accuracy nulls can fail and can distort unconditional accuracy testing; weak-identification-robust bootstrap work warns that ordinary supremum/average bootstrap constructions need not remain valid under weak identification. https://doi.org/10.1111/j.1468-0262.2006.00718.x https://arxiv.org/abs/2006.03238 https://www.cambridge.org/core/journals/econometric-theory/article/abs/weakidentification-robust-wild-bootstrap-applied-to-a-consistent-model-specification-test/3C8BC38490511C3A0597BBAC15836BFF

## Reproducibility

Primary script:

`scripts/research/rh006_joint_dgp_risk_surface_cross_moment_v2.py`

- Git blob SHA-1 = `9938bebabf9e537b419664ce363d96125f43a3c9`
- SHA-256 = `a13aff5c28010504afb80d82bb697dcc08ac40c1eb93742f3b0ca206de6f9fa8`

Dependency:

`scripts/research/rh006_joint_feature_outcome_endogeneity_splitmix64.py`

- Git blob SHA-1 = `5e956ffa3dcf632d81566bd63f92893939f2fd5d`
- SHA-256 = `ec255396810ad8caeb10ceb286ffd5e41f7cab384cd5a2f9777955497f15c178`

Execution:

- paths = 40
- seed = `20261080`
- geometry = 48 / 4 / 16 / 24 / 16
- ridge = `1e-4`
- omega cells = 0, .25, .5, .75, .9
- identity check = 4 paths × 2 directions × 3 omega values × 1,000 outcomes

## Nonclaims

No formal p-value.

No confidence interval.

No uniform or least-favorable size guarantee.

No empirical validation.

No exact-Rust implementation qualification.

No claim that the polynomial-in-omega form or topology transition is universal across arbitrary endogenous DGPs.
