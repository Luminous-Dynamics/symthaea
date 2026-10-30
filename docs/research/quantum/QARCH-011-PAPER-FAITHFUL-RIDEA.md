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

For A=(f_hat-bar_f)(bar_f-f_star), A<0 means the cross-term offsets the sum of the two squared components; A>0 means it reinforces that sum. The camera-ready Section 3.2 explicitly describes positive alignment as amplification and negative alignment as de-amplification. However, Theorem 2 claims the approximate absolute-gap region is a subset of the signed-product de-amplifying region. Appendix C subset relation (iii) tries to establish that implication from a difference of absolute values, without establishing that the two underlying signed errors have the same sign. This is a concrete gap in the stated proof, not merely a limitation of the smooth DeA approximation.

The frozen negative control uses a two-point domain with true function (0.5, -0.1), target weights (0.25, 0.75), and finite model class f_L=(-1,0.5), bar_f=(1,0.5), f_H=(3,0.5). The best-in-class predictor is bar_f, with target risk 0.3325; the global misspecification bound is B_infinity=0.6, attained at the second point, while the tested point has |bar_f-f_star|=0.5 < B_infinity. With tau0=0, c=2 and tau2=0, tau1=1.2. At the tested point, f_L and f_H both have |f_hat-bar_f|=2 and both satisfy the claimed approximate-region threshold. Yet A is -1 for f_L (pointwise squared error 2.25) and +1 for f_H (pointwise squared error 6.25). Their target risks are 0.8325 and 1.8325, respectively. Both predictors are empirical risk minimizers for admissible one-observation samples at the tested point: y=-2 selects f_L, while y=3 selects f_H. The observations are bounded by 3 and arise from symmetric noise of +/-2.5 around the true mean 0.5. The printed sigmoid DeA weights are identical in the two cases. This is a finite-model-class witness satisfying the strict pointwise misspecification bound at the tested point; it contradicts the stated set inclusion as printed.

This is why the module exposes a separate evaluate_signed_interaction_oracle function. It requires f_star and is explicitly evaluator-only; it must never be passed into deployed experiment selection. The gate function reproduces the printed magnitude formula but its return value is not a certificate that the signed interaction is beneficial.

The audit in [QARCH-012 #9092](https://github.com/Luminous-Dynamics/symthaea/issues/9092) has isolated a concrete failure in Appendix C subset relation (iii): the proof converts an absolute-value separation into a signed product without proving sign alignment. The finite-model-class witness above satisfies the boundedness assumptions and has a strict pointwise misspecification bound, but violates the claimed inclusion. Treat the printed magnitude gate as an implemented heuristic pending a corrected theorem or an additional explicit sign assumption. Do not claim the published theorem currently justifies it as a de-amplification guarantee.

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