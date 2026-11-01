# Projective vector betting with predictable covariance envelopes

**Status:** draft extension of PR #9712 (history-dependent scalar betting). The construction is model-dependent. It does not replace the arbitrary-dependence affine merger in #9695.

## Conditional validity

Fix an intersection S and let A_t = S intersect I_t be the active coordinates. Under H_S, assume each active coordinate has conditional mean theta_i,t <= 0 and the vector score has a predictable conditional sub-Gaussian mgf envelope V_A,t:

$$
\\mathbb E[\\exp(\\lambda^T(Z_{A_t,t}-\\theta_{A_t,t}))\\mid\\mathcal F_{t-1}]
\\le \\exp(\\tfrac12\\lambda^T V_{A_t,t}\\lambda)
$$

for predictable lambda >= 0 in the supported domain. Then the one-step factor

$$
X_{S,t}=\\exp\\left(
\\lambda_{A_t,t}^T Z_{A_t,t}
-\\tfrac12\\lambda_{A_t,t}^T V_{A_t,t}\\lambda_{A_t,t}
\\right)
$$

has conditional expectation at most one. Its product is an intersection test supermartingale/e-process.

For a conditionally Gaussian score with mean theta and covariance Sigma, it is sufficient that V_A,t - Sigma_A,t is positive semidefinite. For more general noise, the conditional mgf domination itself is the assumption; covariance domination alone is not enough.

## Projectivity contract

- The active set A_t is predictable: activation at t is decided from information through t-1.
- Each coordinate's lambda_i,t is chosen from its own history through t-1 (and public predictable metadata), not from the current-round score or unrelated future arrivals.
- V_A,t is the principal submatrix of one globally declared predictable envelope. Appending an unrelated coordinate cannot change an existing principal submatrix.
- A dormant coordinate contributes factor one (lambda_i,t=0). When it activates, it starts contributing only according to the predeclared activation rule.
- For a family of experts valid under the same conditional mgf assumption, a fixed ex-ante mixture with weights summing to one remains an e-process. The pointwise maximum of experts is not an e-process in general.

The present code checks finite values, dimensions, matrix symmetry and PSD. It cannot prove that an envelope covers the true conditional mgf; that is a separately sourced/model-validated assumption.

## Covariance-envelope example

For d-dimensional equicorrelation covariance Sigma(rho)=(1-rho)I+rho 11^T and rho in [rho_min,rho_max], select reference rho_0 in the interval and define

$$
\\delta=\\max\\{(d-1)(\\rho_{max}-\\rho_0),\\rho_0-\\rho_{min},0\\},\\qquad
V=\\Sigma(\\rho_0)+\\delta I.
$$

The eigenvalues of Sigma(rho)-Sigma(rho_0) are (rho-rho_0)(d-1) on span(1) and -(rho-rho_0) on its orthogonal complement, so V dominates the entire interval. For d=5, [0.2,0.8], and rho_0=0.5, delta=1.2 and V=Sigma(0.5)+1.2I.

## Independent exploratory simulation

A vectorized Gaussian experiment (d=5, horizon 60, 30,000 streams per cell, alpha=.05, 5-step warmup, coordinate-local predictable bet, cap 1.5 and threshold 1.5/sqrt(n)) compared:
1. vector-coordinate intersection bet;
2. shared scalar bet;
3. fixed 50/50 mixture of vector and scalar e-processes;
4. arbitrary-dependence affine coordinate mixture.

The same declared envelope V=Sigma(0.5)+1.2I covered true equicorrelations rho in [0.2,0.8].

At true rho=0.5, estimated threshold-crossing rates were:

| Scenario | Vector | Scalar | Fixed portfolio | Affine fallback |
|---|---:|---:|---:|---:|
| All-null | 0.077% | 0.047% | 0.050% | 0.000% |
| One sparse signal, mean (0,0,0,0,0.8) | 83.27% | 1.31% | 74.39% | 65.65% |
| Dense signal, all means 0.45 | 32.26% | 42.10% | 38.35% | 19.76% |

Across true rho=0.2, 0.5 and 0.8 with the same envelope, vector sparse-signal power was 81.88%, 83.27% and 83.96%; dense-signal power was 21.97%, 32.26% and 38.41%. Scalar betting was better on dense alternatives. A predictable-arrival all-null experiment (30,000 streams, 60 rounds) had estimated crossing rates 0.013%, 0.017% and 0.110% at those three correlations.

These are single-intersection Gaussian diagnostics from an independent local implementation, not repository test results, not closed-FDR results, and not SupFDR results. They do not establish validity; validity follows only under the conditional mgf contract and predictable betting/activation assumptions.

## Required next experiments

- Block-sparse alternatives; heterogeneous signal strengths; sign-mixed and adversarial means.
- Conditional non-Gaussian sub-Gaussian noise and time-varying covariance envelopes.
- Model misspecification at and outside the declared envelope; invalid envelope must fail closed.
- Nested random arrivals with exact prefix equality and arrival-policy replay.
- Matched comparison against #9695 affine fallback, #9684 countable mixture, #9712 scalar betting, and 2026 online e-closure/compound-e-value methods.
- Only after intersection-level calibration, wire these local tests through the actual dynamic closure and its distinct stopped-FDR/SupFDR adjusters.

## Files

- validation/projective_vector_betting.py
- validation/test_projective_vector_betting.py
