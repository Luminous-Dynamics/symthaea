# Projective vector betting with predictable covariance envelopes

**Status:** draft extension of PR #9712 (history-dependent scalar betting). This construction is model-dependent. It does not replace the arbitrary-dependence affine merger in #9695.

## Conditional validity

Fix an intersection S and let A_t = S intersect I_t be its active coordinates. Under H_S, assume each active coordinate has conditional mean theta_i,t <= 0 and the vector score has a predictable conditional sub-Gaussian mgf envelope V_A,t:

```
E[ exp(lambda^T (Z_A,t - theta_A,t)) | F_(t-1) ]
    <= exp(0.5 * lambda^T V_A,t lambda)
```

for predictable lambda >= 0 in the supported domain. Then the one-step factor

```
X_S,t = exp(lambda_A,t^T Z_A,t - 0.5 * lambda_A,t^T V_A,t lambda_A,t)
```

has conditional expectation at most one. Its product is an intersection test supermartingale/e-process.

For a conditionally Gaussian score with mean theta and covariance Sigma, it is sufficient that V_A,t - Sigma_A,t is positive semidefinite. For more general noise, the conditional mgf domination itself is the assumption; covariance domination alone is not enough.

## Projectivity contract

- The active set A_t is predictable: activation at t is decided from information through t-1.
- Each coordinate's lambda_i,t is chosen from its own history through t-1 (and public predictable metadata), not from the current-round score or unrelated future arrivals.
- V_A,t is the principal submatrix of one globally declared predictable envelope. Appending an unrelated coordinate cannot change an existing principal submatrix.
- A dormant coordinate contributes factor one (lambda_i,t=0). When it activates, it starts contributing only according to the predeclared activation rule.
- For a family of experts valid under the same conditional mgf assumption, a fixed ex-ante mixture with weights summing to one remains an e-process. The pointwise maximum of experts is not an e-process in general.

The code checks finite values, dimensions, matrix symmetry and PSD. It cannot prove that an envelope covers the true conditional mgf; that is a separately sourced/model-validated assumption.

## Dimension-free projective covariance envelope

For the equicorrelation family

```
Sigma_d(rho) = (1-rho) I_d + rho * 11^T,
rho in [rho_min, rho_max]
```

a tight envelope computed separately at each dimension is **not** projectively consistent: increasing d may increase its diagonal slack and change the old principal submatrix. That invalidates prefix equality even if each individual dimension's matrix dominates its covariance family.

The corrected single global kernel is

```
V_d = (1-rho_min) I_d + rho_max * 11^T.
```

For every rho in the declared interval,

```
V_d - Sigma_d(rho)
  = (rho-rho_min) I_d + (rho_max-rho) 11^T >= 0.
```

Both coefficients are nonnegative and 11^T is positive semidefinite, so this proves the matrix domination for every finite d. Every principal submatrix is exactly the same formula at the smaller dimension. For [rho_min,rho_max]=[0.2,0.8],

```
V_d = 0.8 I_d + 0.8 * 11^T
diag(V_d) = 1.6
```

This is an infinite projective kernel represented by two scalars, not a dense matrix. For any vector x,

```
V_d x = 0.8 x + 0.8 * sum(x) * 1
x^T V_d x = 0.8 * sum(x_i^2) + 0.8 * sum(x_i)^2
```

Both operations cost O(d) time and O(1) kernel parameters. The implementation is `ProjectiveEquicorrelationKernel`; the dense matrix form is retained only as a small-dimension reference and test oracle.

This corrects the earlier dimension-specific envelope `Sigma(0.5)+1.2I`. That earlier matrix was valid at d=5, but recomputing the slack at a different dimension changed the previous principal block. The old benchmark rates are superseded by the projective-kernel results below.

## Reproducible benchmark

Run:

```bash
python validation/benchmark_projective_vector_betting.py --seed 9751 --streams 20000 --dimension 5 --horizon 60 --output vector-betting-results.json
```

Optional dependency: NumPy. The runner samples the Gaussian equicorrelation model with a common Gaussian factor and independent coordinate noise; it uses the analytic O(d) projective quadratic rather than a dense covariance multiplication. It records Wilson 95% intervals and includes:
1. vector-coordinate intersection bet;
2. shared scalar bet;
3. fixed 50/50 vector/scalar mixture;
4. fixed equal-weight coordinate-e-process mixture (valid under arbitrary cross-coordinate dependence when marginal factors are valid);
5. an all-null predictable nested-arrival diagnostic.

The runner rejects correlation settings outside the declared [0.2,0.8] envelope. The envelope is dimension-free and prefix-consistent.

For seed 9751, 20,000 streams per fixed-intersection cell, d=5, horizon 60 and alpha=.05, at true rho=.5:

| Scenario | Vector bet | Scalar bet | Fixed 50/50 | Coordinate mixture |
|---|---:|---:|---:|---:|
| All-null crossing | 0.060% [0.034, 0.105] | 0.020% [0.008, 0.051] | 0.040% [0.020, 0.079] | 0.010% [0.003, 0.036] |
| Sparse alternative (0,0,0,0,0.8) | 96.050% [95.771, 96.311] | 0.785% [0.672, 0.917] | 93.835% [93.493, 94.160] | 93.330% [92.976, 93.668] |
| Dense alternative (0.45 in every coordinate) | 28.410% [27.789, 29.039] | 26.220% [25.615, 26.834] | 27.920% [27.303, 28.546] | 46.480% [45.790, 47.172] |

Rates are proportions of streams whose intersection process crossed 1/alpha at any point in the 60-round horizon. Brackets are Wilson 95% binomial intervals. Across true rho=.2,.5,.8:
- vector sparse-alternative crossings: 96.735%, 96.050%, 96.150%;
- vector dense-alternative crossings: 22.655%, 28.410%, 30.555%;
- coordinate-mixture dense-alternative crossings: 53.740%, 46.480%, 39.350%.

The global kernel's lower idiosyncratic diagonal slack improves sparse power while the common-factor term protects against high positive dependence. It also makes the fixed coordinate mixture a much stronger dense-signal portfolio. The methods still trade off: the joint vector bet wins on sparse signals; the coordinate mixture is better on the dense equal-strength example. The fixed 50/50 combination is not uniformly best.

For the particular all-null nested-arrival policy (activate the next coordinate if the most recently active coordinate's prior sample mean is positive after at least five observations), the reference run recorded 1, 1 and 0 crossings in 20,000 streams at rho=.2,.5,.8 respectively (0.005%, 0.005%, 0%). The Wilson upper 95% bounds are 0.0283%, 0.0283% and 0.0192%. This is a finite simulation diagnostic, not proof of calibration.

## Scope and limitations

- The theorem's guarantee requires the declared conditional multivariate mgf domination, coordinatewise null means <=0, and predictable bet/arrival rules.
- Checking V is PSD does not prove that it covers the unknown conditional law. For Gaussian scores, domination of conditional covariance is sufficient; for non-Gaussian scores, the mgf assumption must be separately justified.
- The coordinate mixture is an arbitrary-dependence e-process under valid marginal coordinate factors. It is a relevant conservative comparator, not the same merger as the global affine closure oracle in #9695.
- This is not a closed-FDR, stopped-FDR, FWER or SupFDR result. It does not establish a SOTA improvement.
- The benchmark is exploratory and not executed from a repository checkout in this environment. The test files are authored; no CI pass is claimed.

## Required next experiments

- Block-sparse alternatives; heterogeneous signal strengths; sign-mixed and adversarial means.
- Conditional non-Gaussian sub-Gaussian noise and time-varying covariance envelopes.
- Model misspecification at and outside the declared envelope; invalid envelope must fail closed.
- Nested random arrivals with exact prefix equality and arrival-policy replay.
- Matched comparison against #9695 affine fallback, #9684 countable mixture, #9712 scalar betting, and 2026 online e-closure/compound-e-value methods.
- Only after intersection-level calibration, wire local tests through the actual dynamic closure and its distinct stopped-FDR/SupFDR adjusters.

## Files

- validation/projective_vector_betting.py
- validation/test_projective_vector_betting.py
- validation/benchmark_projective_vector_betting.py
- validation/test_benchmark_projective_vector_betting.py
