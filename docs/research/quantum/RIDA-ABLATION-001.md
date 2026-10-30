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

## Frozen analytical toy case

The fixture uses a two-hypothesis decision task, three target operating regimes, four candidate sampling designs, and an independently identified synthetic validation artifact. The evaluator-only artifact has a different identifier and is not accepted by the policy API.

| Candidate | EVSI/cost | Representativeness R | Informativeness I | De-amplification proxy D |
|---|---:|---:|---:|---:|
| A — high information, out-of-distribution | 0.90 | 0.30 | 1.00 | 0.44 |
| B — representative, low information | 0.30 | 1.00 | 0.33 | 1.00 |
| C — balanced R-I-D | 0.60 | 0.90 | 0.67 | 0.94 |
| D — alternative regime mix | 0.45 | 0.70 | 0.50 | 0.91 |

The ablations deliberately disagree: information-only selects A; representativeness-only and de-amplification-only select B; the full proxy selects C. This is a synthetic unit fixture designed to check the implementation and expose tradeoffs. It does not establish that C is better under real held-out truth, and it should not be used to tune a policy before a broader preregistered suite is frozen.

## Validation and leakage controls

- Target-regime probabilities and candidate sampling probabilities must be valid distributions.
- Validation-error vectors must match regime dimensions and lie in [0,1].
- Every candidate must have exactly one profile; missing, unknown or duplicate profile IDs fail closed.
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

The Rust module is registered in the quantum-computation crate and exposes all component metrics plus seven policy rankings (three individual, three pairwise, one full). Seven unit tests are authored. The frozen JSON fixture is a synthetic known-answer reference. Rust compilation/tests have not been run in this environment; no passing CI or physical validation is claimed.
