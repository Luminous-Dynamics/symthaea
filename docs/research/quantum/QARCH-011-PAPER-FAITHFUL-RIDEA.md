# QARCH-011 — Paper-equation R-IDeA primitives and sign audit

**Issue:** [Symthaea #9088](https://github.com/Luminous-Dynamics/symthaea/issues/9088)  
**Sign audit:** [QARCH-012 #9092](https://github.com/Luminous-Dynamics/symthaea/issues/9092)  
**Branch:** research/qarch-011-paper-faithful-ridea-20261030  
**Code:** crates/domains/symthaea-quantum-comp/src/rida_exact.rs  
**Fixture:** docs/research/quantum/benchmarks/QARCH-011-RIDA-EXACT-ANALYTIC.json

## Scope

This implementation creates small, dependency-free Rust primitives for the printed R-IDeA equations and a separate evaluator-only diagnostic for the sign of the squared-error interaction. It does not train the proxy predictor g, prove a tau2 bound, estimate EIG, qualify an optical model, or establish photonic hardware performance.

The reference is Tang, Sloman & Kaski, *Representative, Informative, and De-Amplifying: Requirements for Robust Bayesian Active Learning under Model Misspecification*, AISTATS 2026 / PMLR 300. The full camera-ready HTML is available at https://arxiv.org/html/2506.07805v3.

The authors' released code was inspected at pinned commit a72868b4a00f43945ba724b8f71be9181f97865e. Its plotting.py compute_mmd function returns the biased empirical squared RBF-MMD; the PK BOED code's proxy fitting objective combines MSE and an absolute-gap hinge and its disagreement weight uses a sigmoid on the absolute gap. We use that as a reproduction reference, not as a source of unreviewed defaults.

## Implemented equations

### Biased empirical squared MMD

For samples P and Q:

[
\widehat{\mathrm{MMD}}^2(P,Q)=
\frac{1}{m^2}\sum_{i,i'}k(p_i,p_{i'})
+\frac{1}{n^2}\sum_{j,j'}k(q_j,q_{j'})
-\frac{2}{mn}\sum_{i,j}k(p_i,q_j)
]

[
k(p,q)=\exp\left(-\frac{\|p-q\|^2}{2\sigma^2}\right).
]

This is the Appendix D estimator, including diagonal terms, with sigma=1 in the paper's experiments. R-I uses the squared-MMD ratio consistently, matching the authors' code. The code rejects empty sets, mismatched dimensions, non-finite points and invalid bandwidths. Small negative round-off is clamped only within a declared tolerance. A zero baseline MMD is rejected rather than replaced by an arbitrary epsilon.

### R-I score

[
\mathrm{R-I}(\xi_t)=\mathrm{EIG}(\xi_t)
\left(1-\lambda
\frac{\widehat{\mathrm{MMD}}^2(h_{t-1}\cup\{\xi_t\},d_{test})}
{\widehat{\mathrm{MMD}}^2(h_{t-1},d_{test})}\right).
]

The implementation requires a non-empty history; the caller must declare an initialization design before the first adaptive step. It preserves negative acquisition scores rather than clamping them to zero. Lambda is finite and non-negative but otherwise reported as an explicit input.

### Printed proxy objective and DeA gate

The paper's proxy objective (Eq. 11) is evaluated as:

[
\mathcal{L}(g)=\frac{1}{n}\sum_i(g(\xi_i)-y_i)^2
+\frac{1}{n}\sum_i\max(0,\tau-|\hat f^{(n)}(\xi_i)-g(\xi_i)|).
]

The primitive evaluates this objective for fixed predictions; it does not train g. The paper's optional optimizer-specific weighting and regularization are not included in this exact equation primitive.

The printed DeA gate is computed with a numerically stable sigmoid:

[
\mathrm{DeA}(\xi)=\sigma\left(
\frac{|\hat f^{(n)}(\xi)-g(\xi)|-\tau}{\kappa}
\right).
]

The threshold formula is:

[
\tau=\tau_1+\tau_2,\qquad
\tau_1=\frac{\tau_0}{\mathbb{B}_\infty}+c\mathbb{B}_\infty,
\quad c\ge 2,\;\tau_0\ge0,\;\tau_2\ge0.
]

B_infinity=0 is handled as an explicit well-specified state, where the de-amplification gate is neutral. This avoids dividing by zero. A finite tau2 must come from a separately verified proxy-error bound; this module does not create that certificate.

## Important theoretical audit: the magnitude gate is not a signed certificate

The paper's Proposition 1 squared-error identity contains a signed interaction:

[
(\hat f-f^\star)^2=
(\bar f-f^\star)^2+(\hat f-\bar f)^2+
2(\bar f-f^\star)(\hat f-\bar f).
]

For A=(f_hat-bar_f)(bar_f-f_star), A<0 means the cross-term offsets the sum of the two squared components; A>0 means it reinforces that sum. Yet the printed approximate de-amplifying region and DeA gate are based on an absolute prediction/proxy gap, which does not carry the sign of A.

The frozen negative control sets f_star=0.5, bar_f=1, B_infinity=0.5, tau0=0, c=2, tau2=0, giving tau=1. For f_hat=-1 and f_hat=3, the absolute gap is 2 in both cases, so the printed sigmoid gate assigns the same weight 0.7310586. But A is -1 in the first case and +1 in the second; their actual squared errors are 2.25 and 6.25. The same magnitude gate therefore cannot distinguish an offsetting interaction from a reinforcing one.

This is not just an arbitrary predictor pair. The fixture supplies a finite model-class witness: the true regression output is 0.5 at the single design, and the model class contains constant predictors -1, 1 and 3. The constant-one model is the unique best-in-class predictor (squared error 0.25; B_infinity=0.5). With admissible noisy outcomes y=-2 or y=3, the constant-minus-one or constant-three model respectively becomes the empirical risk minimizer. Thus both sign cases are compatible with a finite-class ERM setting, bounded outcomes y_infinity=3, and the same printed tau threshold.

This is why the module exposes a separate evaluate_signed_interaction_oracle function. It requires f_star and is explicitly evaluator-only; it must never be passed into deployed experiment selection. The gate function reproduces the printed magnitude formula but its return value is not a certificate that the signed interaction is beneficial.

The issue [QARCH-012 #9092](https://github.com/Luminous-Dynamics/symthaea/issues/9092) tracks an independent audit of the sign/inequality relationship in Proposition 1, Theorem 1, Theorem 2 and Appendix C. Until that audit is resolved, do not claim that the printed magnitude gate alone guarantees de-amplification.

## Frozen known-answer coverage

The JSON fixture pins:
- MMD^2 for identical empirical samples and a one-unit singleton separation;
- duplicate-sample invariance of the biased estimator;
- an R-I ratio with independently calculated before/after MMD^2;
- lambda=0 reducing R-I to EIG;
- finite and well-specified threshold branches;
- sigmoid and proxy-objective known answers;
- the signed offsetting/reinforcing magnitude-gate counterexample.

These are analytic/synthetic software fixtures, not paper experiment reproductions, not calibrated photonic data and not a quantum-advantage benchmark.

## Next research gates

1. Resolve QARCH-012's mathematical sign/region audit before describing DeA as a theoretically valid de-amplification certificate.
2. Train g with a transparent optimizer interface and independently verify its holdout error bound tau2; do not treat training loss as a supremum error bound.
3. Reproduce the paper's polynomial regression, source localization and pharmacokinetic experiments using pinned code/data and a faithful protocol. The original work uses 20 runs/10 designs for polynomial regression, 30 steps for source localization and 10 steps for PK; report MSE and MMD separately.
4. Compare Random, EIG/BAD, R-I, printed R-IDeA, R-IDeA-oracle as evaluator-only diagnostic, QARCH-010's distinct proxy, and signed-error controls under identical seeds and budgets.
5. Only after independent reproduction, adapt the method to a photonic categorical likelihood model. The paper's regression target is real-valued; its proxy objective and theory do not automatically transfer to click/no-click likelihood calibration.

## Validation boundary

Sixteen Rust unit tests are authored in this module. The module is registered and publicly exported from the quantum-computation crate. The JSON fixture was parsed and its scalar values were independently recomputed. Rust compilation/tests have not been run in this environment because cargo and rustc are unavailable. No exact-head CI result is claimed here.