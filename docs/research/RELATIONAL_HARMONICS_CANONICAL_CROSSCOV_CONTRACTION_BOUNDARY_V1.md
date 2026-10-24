# RH-006 Canonical Cross-Covariance Contraction Boundary — v1

Status: research diagnostic only.

## 1. Exact admissibility parameterization

Let Xi = Cov(e,q) = K Sigma_q and assume Sigma_e and Sigma_q are positive definite.

Define C = Sigma_e^{-1/2} Xi Sigma_q^{-1/2}. Then the block covariance is positive semidefinite exactly when

    ||C||_2 <= 1.

Equivalently, the admissible cross-covariance class can be parameterized as

    Xi = Sigma_e^{1/2} C Sigma_q^{1/2},  ||C||_2 <= 1.

For invertible Sigma_q this gives

    K = Sigma_e^{1/2} C Sigma_q^{-1/2}.

This turns operator admissibility into a fail-closed spectral-norm contraction test.

## 2. Boundary execution

Executed design: 60 deterministic trials, latent dimension 7, outcome dimension 5, seed 20261012, target strengths 0, 0.35, 0.85, 0.99, 1.0, and 1.01.

Results:

- strengths through 0.99 were admissible in 100% of trials;
- strength 1.0 was on the PSD boundary to floating-point tolerance, with minimum conditional-covariance eigenvalues ranging from -1.61e-14 to +1.61e-14;
- strength 1.01 was inadmissible in 100% of trials, with minimum conditional-covariance eigenvalues ranging from -0.1741 to -0.0110;
- the canonical singular-value strength matched the target to approximately 1e-14.

Exact local execution:

- script SHA-256: 098091ebc4d308b1774c105dc1994ee33d6c829f04cc09f8a89f0b9b228afcc4;
- result bytes SHA-256: 366abb7791dd09ad412d17e3b491217b813222ea1b6264296e46fa181e49a79f;
- result payload SHA-256: 0b3b3bca142c1dbd75f565f81b99e795605a4786ea3c1bd05f647d85aeb92b1f.

## 3. Scientific consequence

The nuisance strength coordinate now has a principled domain: [0,1]. The endpoint 1 is not an ordinary interior nuisance value; it is a covariance-completion boundary where conditional variance can become singular.

Thus any future calibration grid that approaches strength 1 must be stratified separately. Values above 1 must fail closed rather than being clipped silently back into the admissible region.

## 4. Evidence boundary

This is an admissibility/mechanics result, not statistical inference.

It establishes no p-value, confidence interval, size guarantee, empirical validation, latent-K identification, or Rust implementation qualification.

References: covariance Schur-complement positivity and normalized cross-covariance/correlation operator representations.