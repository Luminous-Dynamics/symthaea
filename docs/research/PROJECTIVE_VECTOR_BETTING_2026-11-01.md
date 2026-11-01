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

## Covariance-envelope example

For d-dimensional equicorrelation covariance Sigma(rho)=(1-rho)I+rho*11^T and rho in [rho_min,rho_max], choose reference rho_0 in the interval and define

```
delta = max(0, (d-1)*(rho_max-rho_0), rho_0-rho_min)
V = Sigma(rho_0) + delta*I
```

The eigenvalues of Sigma(rho)-Sigma(rho_0) are (rho-rho_0)*(d-1) on span(1) and -(rho-rho_0) on its orthogonal complement, so V dominates the entire interval. For d=5, [0.2,0.8], and rho_0=0.5, delta=1.2 and V=Sigma(0.5)+1.2I. Independent numerical checks covered dimensions 2 through 9, three rho intervals, and 101 grid points per interval.

## Reproducible benchmark

Run:

```bash
python validation/benchmark_projective_vector_betting.py --seed 9751 --streams 20000 --dimension 5 --horizon 60 --output vector-betting-results.json
```

Optional dependency: NumPy. The runner uses a fixed seed, records Wilson 95% intervals, and includes:
1. vector-coordinate intersection bet;
2. shared scalar bet;
3. fixed 50/50 vector/scalar mixture;
4. fixed equal-weight coordinate-e-process mixture (arbitrary cross-coordinate dependence);
5. an all-null predictable nested-arrival diagnostic.

The declared envelope V=Sigma(0.5)+1.2I covers true equicorrelations rho in [0.2,0.8].

For seed 9751, 20,000 streams per fixed-intersection cell, d=5, horizon 60, alpha=.05, and true rho=.5:

| Scenario | Vector bet | Scalar bet | Fixed 50/50 | Coordinate mixture |
|---|---:|---:|---:|---:|
| All-null crossing | 0.065% [0.038, 0.111] | 0.020% [0.008, 0.051] | 0.035% [0.017, 0.072] | 0.000% [0, 0.019] |
| Sparse alternative (0,0,0,0,0.8) | 82.800% [82.271, 83.317] | 1.210% [1.068, 1.371] | 73.930% [73.317, 74.534] | 65.130% [64.467, 65.788] |
| Dense alternative (0.45 in every coordinate) | 31.740% [31.098, 32.389] | 41.315% [40.634, 41.999] | 38.025% [37.355, 38.700] | 19.035% [18.497, 19.585] |

Rates are proportions of streams crossing 1/alpha at any point during the 60-round horizon. Brackets are Wilson 95% binomial intervals. Across rho=.2,.5,.8, vector sparse-alternative crossing rates were 81.88%, 82.80%, 84.235%; vector dense-alternative rates were 21.94%, 31.74%, 38.505%. Shared-scalar dense rates were 42.785%, 41.315%, 41.185%. The fixed portfolio dense rates were 34.81%, 38.025%, 40.11%; coordinate-mixture dense rates were 19.94%, 19.035%, 17.90%.

The vector bet is much stronger on the sparse alternative; the scalar bet is stronger on the dense alternative. The fixed portfolio gives a middle ground while preserving validity under the common envelope. These are intersection-level diagnostics, not multiple-testing power claims.

For the particular all-null nested-arrival rule in the runner (activate the next coordinate if the most recently active coordinate's prior sample mean is positive after at least five observations), the local reference run recorded 0 crossings in 20,000 streams at each rho=.2,.5,.8. Each cell has a Wilson upper 95% bound of about 0.0192%. This is a finite simulation diagnostic, not a proof of calibration.

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
