# RH-006 K Identifiability and Observable Cross-Covariance — v1

Status: research diagnostic only.

- applicability = not-approved-for-execution
- selection = stop-assumption-failure
- formal inference = disabled
- formal p-value = disabled
- formal confidence interval = disabled

## 1. Why K itself needs an identifiability gate

The general joint-DGP layer uses

    E[e|q] = Kq
    Var(e|q) = Sigma_e - K Sigma_q K'

When the driver q is latent, K is not invariant to a change of latent coordinates. For any invertible A,

    q* = A q
    Sigma_q* = A Sigma_q A'
    K* = K A^{-1}

produce the same conditional mean and conditional covariance. Thus a structural K can only be called identified after a measurement/factor normalization has been declared.

## 2. Observable cross-covariance is the more stable binding

Define Xi = Cov(e,q) = K Sigma_q. The block covariance is [[Sigma_q, Xi'], [Xi, Sigma_e]].

For positive definite marginal covariances, the normalized cross-covariance C = Sigma_e^{-1/2} Xi Sigma_q^{-1/2} has singular values equal to canonical correlation strengths. The admissibility condition Sigma_e - K Sigma_q K' >= 0 is equivalent to the contraction condition ||C||_2 <= 1.

Accordingly:

- when q is observed, bind Xi and its canonical spectrum rather than a coordinate-dependent raw K;
- when q is latent, bind the observational equivalence class plus an explicit measurement/factor normalization before treating K as a structural nuisance coordinate.

The Schur-complement form is the standard covariance-positivity condition; normalized conditional-covariance/correlation operators provide the corresponding contraction representation. citeturn603819search0turn603819search2

## 3. Executed algebraic audit

Design: 40 deterministic trials; latent dimension 6; outcome dimension 5; observed feature dimension 4; canonical contraction strength 0.70; seed 20261011.

Each trial applies a non-orthogonal invertible latent-coordinate transformation with condition number capped below 25. A second construction co-transforms an observed feature loading and K, producing the same observed covariance for (F,e).

Exact local execution:

- script SHA-256 = 6c9c63b8ae18c572aa17dfe448b148666e1c56712260bef06e600910d13de870;
- full result bytes SHA-256 = ad379f2c8d3d464523bfb89e6af17d5a6a7d6b14cf831f65104f322805302fdd;
- canonical result payload SHA-256 = 321594ec1c150c0810e22d19ec213f0cf872e369e55e32e8567cd19af1f27e17.

Maximum observed errors:

- conditional covariance invariance: 9.95e-14;
- conditional mean invariance: 1.14e-15;
- canonical-spectrum invariance: 2.32e-14;
- latent-factor observed-covariance equivalence: 3.55e-13.

The median change in raw K operator norm under latent-coordinate transformation was 0.397, with a maximum of 4.175, while the median canonical strength remained 0.7000000000000002.

These are identifiability/mechanics results, not inference validation.

## 4. Scientific consequence

The nuisance architecture should distinguish three levels:

1. Structural coordinate: K, usable only after a declared latent normalization.
2. Observable cross-moment: Xi = K Sigma_q, when the driver innovations are actually observed or recoverable.
3. Invariant canonical dependence: the singular spectrum of Sigma_e^{-1/2} Xi Sigma_q^{-1/2}.

For latent q, the correct inferential nuisance object is not a single matrix K but an observational equivalence class constrained by the measurement model.

This is a stronger gate than merely adding more K geometries: it prevents a coordinate choice from being mistaken for scientific identification.

## 5. Literature relevance

Schur complements characterize positivity of block covariance matrices, while normalized cross-covariance/correlation operators provide invariant contraction representations. Canonical block-matrix methods are also useful for covariance parameterization and regularization. citeturn603819search0turn603819search1turn603819search2

## 6. Evidence boundary

This artifact does not authorize inference.

It does not establish:
- a formal p-value;
- a confidence interval;
- uniform or least-favorable size control;
- identification of latent K from real RH-006 observables;
- validity for arbitrary latent-factor/measurement models;
- empirical predictive validity;
- exact validation of the Symthaea Rust implementation.

The selector remains: applicability = not-approved-for-execution; selection = stop-assumption-failure.

## 7. Next gate

The next useful experiment is an observable-versus-latent nuisance bridge: specify what RH-006 actually observes about the feature-generating process; determine whether Xi is estimable from observed quantities; if not, define a measurement model and an identified quotient of the latent K class; then simulate estimation error in that identified nuisance coordinate and re-estimate the nuisance inside every bootstrap draw.

Formal inference remains closed until this bridge is established.