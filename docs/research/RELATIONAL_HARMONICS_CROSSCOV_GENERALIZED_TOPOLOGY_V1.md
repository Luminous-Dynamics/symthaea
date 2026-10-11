# RH-006 Cross-Covariance Generalization and Exact Null-Surface Topology — v1

Status: research diagnostic only.

- applicability = not-approved-for-execution
- selection = stop-assumption-failure
- formal p-value = disabled
- formal confidence interval = disabled

## 1. Generalization

The scalar shared-shock stress is a finite-rank feature/outcome cross-covariance construction.

For a normalized latent feature-driver process q:

eta = sqrt(1 - omega^2) z + omega q
e = L eta

Conditioning on q:

E[e | q] = omega L q
Var(e | q) = (1 - omega^2) L L'

With fixed realized feature paths and fixed rolling ridge operators, squared forecast loss remains quadratic in the incremental relational coefficient vector:

Delta(gamma; omega)
= gamma' Q gamma + (b0 + omega b1)' gamma + c0 + omega c1 + omega^2 c2

The executed family spans five deterministic cross-covariance structures:

- single relational channel
- multi-relational channel mixture
- common + relational mixture
- anti-relational mixture
- one-step lagged relational channel

This is a finite-rank stress family, not a claim to exhaust every possible feature/disturbance cross-covariance operator.

## 2. Exact topology theorem when Q is positive definite

For a ray gamma = t u:

A = u' Q u
B = b' u
C = c
D(u) = B^2 - 4 A C

and the discriminant matrix is

R = b b' - 4 c Q.

When Q is positive definite:

### c < 0

D(u) is positive for every nonzero direction. Because C/A is negative, every ray has exactly one positive and one negative root.

Global discriminant support therefore needs no directional grid in this regime.

### c = 0

R = b b'. The nonzero positive branch exists on the hemisphere b'u < 0, with b'u = 0 as the tangential boundary.

### c > 0

R has at most one positive eigenvalue.

Using Q^(-1/2) R Q^(-1/2) = v v' - 4 c I, with v = Q^(-1/2) b, a positive-discriminant direction exists exactly when:

b' Q^(-1) b > 4 c.

When this holds, real-root support is a cone. Positive roots additionally require b'u < 0.

When it fails, no real-root direction exists.

For weak or indefinite Q, this theorem is deliberately not applied; the existing eigenstructure and stratified weak-identification machinery remain necessary.

## 3. Executed generalized cross-covariance stress

Execution:

- 40 feature paths per family
- seed = 20261080
- omega = 0, .25, .5, .75, .9
- five cross-covariance families
- RH-006 geometry 48 / 4 / 16 / 24 / 16
- ridge = 1e-4

The theorem classification matched the independently computed discriminant-matrix spectrum on all 1000 topology cells.

## 4. Topology result

For contemporaneous cross-covariance families, the transition to cone-supported geometry is common:

| Family | omega=.25 | omega=.50 | omega=.75 | omega=.90 |
|---|---|---|---|---|
| single_rel | 39 all-direction / 1 cone | 40 cone | 40 cone | 40 cone |
| multi_rel | 40 all-direction | 40 cone | 40 cone | 40 cone |
| mixed_common_rel | 40 all-direction | 40 cone | 40 cone | 40 cone |
| anti_rel | 39 all-direction / 1 cone | 40 cone | 40 cone | 40 cone |

The lagged relational family is materially different:

| Family | omega=.50 | omega=.75 | omega=.90 |
|---|---|---|---|
| lag_rel | 40 all-direction | 25 all-direction / 15 cone | 10 all-direction / 30 cone |

Thus at omega=.90 the lagged family retains all-direction topology on 25% of paths.

The correct conclusion is:

Cross-covariance geometry, including channel and lag structure, controls the null-surface transition. A scalar endogeneity magnitude is not sufficient to classify topology.

## 5. Direction-specific support is non-universal

At omega=.90:

- contemporaneous families can lose both predeclared weak and strong ray support;
- the lagged family retains weak-ray support on 87.5% of paths and strong-ray support on 25%.

Therefore a universal rule such as “high endogeneity makes the scientific ray undefined” is false.

The scientific direction and the joint nuisance geometry must remain explicit.

## 6. Direct mechanics check

The executed local checker performed:

5 cross-covariance families × 3 feature paths × 2 directions × 3 omega values = 90 direct Monte Carlo identity checks.

Each cell used 250 outcome draws.

Results:

- maximum absolute analytic-vs-Monte-Carlo discrepancy = 1.4778778e-4
- mean absolute discrepancy = 2.9440427e-5

The theorem-vs-spectrum match rate was 100% on all 1000 topology cells.

These are mechanics checks, not statistical size validation.

## 7. Architectural consequence

The joint nuisance state should not be reduced to one scalar endogeneity coefficient.

The eventual calibration state is better represented as:

cross-channel weights
+ cross-lag structure
+ disturbance covariance
+ Q curvature
+ discriminant topology
+ branch conditioning
+ effect radius

This suggests a two-layer inference architecture:

1. Exact conditional geometry whenever Q is positive definite.
2. Numerical/eigenstructure treatment only where Q is weak or indefinite.

That reduces arbitrary-grid dependence without pretending weak identification is solved.

## 8. Literature relevance

Endogenous predictive regressors are known to create bias and size problems, especially with persistent predictors and long-horizon forecasting. This makes joint feature/outcome dependence a substantive forecasting concern. See Cai and Wang, Journal of Econometrics (2014).

Forecast-evaluation simulations also find that several bootstrap methods can over-reject with highly persistent data, reinforcing the need to validate the complete forecasting procedure rather than import a generic bootstrap.

Recent weak-identification work similarly recommends exploiting polynomial geometry rather than brute-force grid inversion for confidence-set construction.

## 9. Evidence boundary

This artifact does not authorize inference.

It does not establish:

- a p-value
- a confidence interval
- a least-favorable critical value
- uniform size control
- empirical predictive validity
- validity for arbitrary endogenous DGPs
- validity of the Symthaea Rust implementation

The applicability selector remains stop-assumption-failure.

## 10. Reproducibility

Executed local mechanics source:

rh006_crosscov_generalized_topology_splitmix64.py

Local execution SHA-256:

ca1418e05874fa6ba7af4b5e7ac7cdb5a83f3514dd91857dbd2bc92b4a6d60f7

Local execution result SHA-256:

8ea4421f1b02fbc90dc8dda24242fa385128ae6adb6ebfade1bb630716e2e3c6

The repository copy is committed separately. Byte identity between the local execution source and repository copy was not established, so the evidence receipt distinguishes those identities.