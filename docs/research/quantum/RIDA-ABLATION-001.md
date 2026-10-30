# QARCH-010 — Representative, informative, and de-amplifying experiment design

**Issue:** [#9067](https://github.com/Luminous-Dynamics/symthaea/issues/9067)  
**Depends on:** QARCH-005/006/007/008/009  
**Code:** `crates/domains/symthaea-quantum-comp/src/representative_experiment.rs`  
**Fixture:** `docs/research/quantum/benchmarks/QARCH-010-RIDA-ABLATION.json`

## Research motivation

Tang, Sloman and Kaski's AISTATS 2026 paper, [Representative, Informative, and De-Amplifying: Requirements for Robust Bayesian Active Learning under Model Misspecification](https://proceedings.mlr.press/v300/tang26d.html), analyzes how covariate shift and misspecification can amplify generalization error. Their R-IDeA acquisition criterion combines representativeness, informativeness and de-amplification, and reports improvements over several one- and two-component alternatives.

This work is a research comparator. **The implementation below is not the published R-IDeA formula and does not claim to reproduce its theorem or empirical results.** It is a small, auditable R-I-D proxy over the existing QARCH-005 finite decision problem. The purpose is to establish measurable component/ablation contracts before attempting a faithful implementation of the paper's exact method.

## Implemented component definitions

For candidate experiment (i), let (p_T(r)) be the preregistered target operating-regime distribution, (p_i(r)) the candidate's measurement-regime mix, and (e(r)in[0,1]) the validation error of the current predictive model in regime (r). All regime labels and orders are explicit.

**Representativeness**

[
R_i = 1 - operatorname{TV}(p_T,p_i)
= 1 - \frac{1}{2}\sum_r |p_T(r)-p_i(r)|.
]

**Decision informativeness**

[
I_i = \frac{\operatorname{EVSI}_i / c_i}
{\max_j(\operatorname{EVSI}_j / c_j)}.
]

This uses QARCH-005's decision value per declared cost, not entropy alone. It is normalized within the frozen candidate pool; raw EVSI/cost and information gain/cost remain in the report. If all candidates have zero EVSI, every normalized informativeness score is zero.

**Validation-error distribution-gap proxy**

[
E_T=\sum_r p_T(r)e(r),\qquad
E_i=\sum_r p_i(r)e(r)
]
[
G_i=|E_T-E_i|,\qquad D_i=1-G_i.
]

(D_i) is bounded in [0,1] because the declared validation errors are in [0,1]. It measures a particular validation-error covariate-distribution discrepancy. It is **not** a proof of de-amplification and can fail if the regime partition, validation errors, or target distribution fail to represent deployment.

**Transparent paired and full ablations**

[
S_{RI}=\sqrt{R_i I_i},\quad
S_{RD}=\sqrt{R_i D_i},\quad
S_{ID}=\sqrt{I_i D_i},\quad
S_{RID}=\sqrt[3]{R_i I_i D_i}.
]

The geometric means ensure no component can be silently hidden by an arbitrary linear weight. They are an experimental design choice for this proxy, not a claim that the paper uses these exact equations.

## Crosswalk to the actual AISTATS 2026 R-IDeA method

The paper's de-amplification term is not a validation-error distribution gap. It defines a signed interaction between estimation error and misspecification error. Since the true data-generating function is unknown, the method trains a proxy predictor (g) to approximate the best-in-class predictor while maintaining disagreement from the current predictor.

The paper's acquisition structure is:

[
\mathrm{R-I}(\xi_t)=\mathrm{EIG}(\xi_t)
\left(1-\lambda\frac{\mathrm{MMD}(h_{t-1}\cup\{\xi_t\},d_{test})}
{\mathrm{MMD}(h_{t-1},d_{test})}\right)
]

[
\mathrm{DeA}(\xi_t)=\sigma\left(
\frac{|\hat f^{(n)}(\xi_t)-g(\xi_t)|-\tau}{\kappa}
\right),
\qquad
\mathrm{R\text{-}IDeA}(\xi_t)=\mathrm{R-I}(\xi_t)\,\mathrm{DeA}(\xi_t).
]

The proxy is trained with a data-fit term plus a disagreement hinge penalty. Its threshold is linked to the paper's bound through (\tau=\tau_1+\tau_2), (\tau_1=\tau_0/\mathbb{B}_\infty+c\mathbb{B}_\infty), (c\geq2), and a proxy error bound (\tau_2). The paper's experiments use squared MMD with a Gaussian/RBF kernel and fixed bandwidth 1. These equations are from the AISTATS 2026 camera-ready paper, not the proxy implemented in this PR. [PMLR paper](https://proceedings.mlr.press/v300/tang26d.html) · [arXiv HTML](https://arxiv.org/html/2506.07805v3).

**Why this PR remains a proxy:** the current Rust API scores finite experiment candidates with QARCH-005 EVSI/cost, target-regime total variation, and a validation-error distribution gap. It does not fit (g), compute the paper's MMD robust ratio, or estimate the signed error-amplification interaction. The proxy is retained as a small independently auditable screening baseline; [QARCH-011 #9088](https://github.com/Luminous-Dynamics/symthaea/issues/9088) tracks the paper-faithful implementation and its separate validation gates.

An important lesson from the paper is that lower covariate shift alone is not sufficient: its source-localization and pharmacokinetic experiments report cases in which R-IDeA has higher MMD/covariate shift yet lower generalization error, consistent with de-amplifying designs. Thus this proxy must never be judged by representativeness or MMD alone; downstream prediction error and decision regret are essential.

## Frozen analytical toy case

The fixture uses a two-hypothesis decision task, three target operating regimes, four candidate sampling designs, and an independently identified synthetic validation artifact. The evaluator-only artifact has a different identifier and is not accepted by the policy API.

| Candidate | EVSI/cost | Representativeness R | Informativeness I | De-amplification proxy D |
|---|---:|---:|---:|---:|
| A — high information, out-of-distribution | 0.90 | 0.30 | 1.00 | 0.44 |
| B — representative, low information | 0.30 | 1.00 | 0.33 | 1.00 |
| C — balanced R-I-D | 0.60 | 0.90 | 0.67 | 0.94 |
| D — alternative regime mix | 0.45 | 0.70 | 0.50 | 0.91 |

The ablations deliberately disagree: information-only selects A; representativeness-only and de-amplification-only select B; the full proxy selects C. This is a synthetic unit fixture designed to check the implementation and expose tradeoffs. It does not establish that C is better under real held-out truth, and it should not be used to tune a policy before a broader preregistered suite is frozen.

### Target-distribution shift control

A second frozen case changes only the preregistered target distribution to [0.1, 0.2, 0.7]. The full proxy then selects A (score ≈0.8759), rather than C. This is intentional: the objective has changed because the target operating population has changed. It is not a universal statement that A is now physically better. The same candidate profiles, validation errors, EVSI values and separate evaluator artifact are retained.

## Validation and leakage controls

- Target-regime probabilities and candidate sampling probabilities must be valid distributions.
- Validation-error vectors must match regime dimensions and lie in [0,1].
- Every candidate must have exactly one profile; missing, unknown or duplicate profile IDs fail closed.
- Every candidate binds an upstream eligibility evidence digest. Ineligible candidates carry no numerical sampling/validation vectors, are listed separately, and do not affect EVSI normalization or policy rankings.
- Validation and evaluator-only artifact digests must be different.
- Evaluator-only truth does not enter the ranking API.
- Scenario and profile order must not change policy rankings.
- All component-only, pairwise and full rankings are emitted; no private aggregate score replaces them.

Different digests alone do not prove row-level or session-level separation. The eventual verifier must resolve the underlying immutable manifests and establish the actual split policy (for example by device, configuration, physical session or operating regime) before physical evidence can be credited.

## Industry comparison and next benchmark gates

QOBLIB's 2026 library provides ten model-independent optimization problem classes, reproducible instances, and strong classical solver baselines; its live repository reports 1,264 instances and 3,226 submissions across 27 organizations. Its guidance is directly relevant to application benchmarking: improve classical baselines as well as quantum methods and report negative results. [Paper](https://www.nature.com/articles/s43588-026-00991-1) · [Live library](https://zib-aopt.github.io/QOBLIB/).

Open QBench provides hybrid, high-level and compiled benchmark layers with gate-based, photonic and annealing compatibility. QARCH-010 should complement these rather than substitute an internal acquisition score for application-level performance. [Open QBench](https://quantum.psnc.pl/en/openqbench/).

Proposed gates (not achieved):
- 100% agreement with frozen analytic component fixtures and exact deterministic rankings.
- Zero false accepts for invalid regime distributions, validation/evaluation artifact overlap, missing profiles and invalid validation errors.
- Compare every ablation on the same frozen candidate pool and downstream decision-regret suite.
- Report mean, worst-case and tail regret; utility; cost; target-distribution coverage; predictive adequacy; unsupported outcome mass; and nominal performance sacrificed.
- Select policy components/normalization only on a development/validation split; score the final choice on evaluator-only scenarios untouched by policy selection.
- Later integrate QSIM-002 calibrated likelihoods and physical measurements, keeping analytic, simulated, calibrated and measured evidence distinct.

A SOTA claim requires matching workload, strongest applicable classical and quantum competitors, equal quality gates, end-to-end costs, reproducible hardware/software details and independent held-out evidence. This module currently measures algorithm-selection components only.

## Current implementation boundary

The Rust module is registered in the quantum-computation crate and exposes all component metrics plus seven policy rankings (three individual, three pairwise, one full). It explicitly excludes NotCalibrated, StaleCalibration, OutOfDomain and InsufficientEvidence candidates before ranking; ineligible profiles must not carry numerical score inputs. Twelve unit tests are authored. The frozen JSON fixture contains a default-regime case, a target-shift case, and a stale-calibration exclusion negative control. Rust compilation/tests have not been run in this environment; no passing CI or physical validation is claimed.