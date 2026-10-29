# QARCH-007: held-out predictive adequacy before active experiment selection

**Status:** implementation draft; seven Rust tests authored, not yet executed  
**Code:** \`crates/domains/symthaea-quantum-comp/src/predictive_adequacy.rs\`  
**Benchmark fixture:** \`docs/research/quantum/benchmarks/QARCH-007-PREDICTIVE-ADEQUACY.json\`  
**Related:** QARCH-005 decision-focused experiment selection; QARCH-006 finite-ensemble robust selection; Symtropy QSIM-001.

## Why this matters

QARCH-006 makes experiment selection robust over an explicitly supplied finite model ensemble. But all members of the ensemble may be wrong in the same direction. Agreement between models is not evidence that the ensemble covers the real data-generating process.

Recent work makes this a concrete benchmark concern:
- Tang, Sloman and Kaski, *Representative, Informative, and De-Amplifying: Requirements for Robust Bayesian Active Learning under Model Misspecification*, AISTATS 2026, shows that information-only active learning can amplify errors under model misspecification and proposes a criterion that combines representativeness, information and de-amplification ([paper](https://proceedings.mlr.press/v300/tang26d.html)).
- Harikumar et al., *Robust Bayesian Decision Making under Adversarial Uncertainty*, UAI 2026, studies decision-focused experimental design where hidden effects can make the nominally best experiment misleading ([paper](https://proceedings.mlr.press/v337/harikumar26a.html)).
- Abdulsamad et al., *Maximin Robust Bayesian Experimental Design*, 2026 preprint, studies a max-min formulation with information-theoretic ambiguity constraints and finite-sample bounds ([preprint](https://arxiv.org/abs/2603.14094)).
- For the quantum program, IonQ's 2026 application-level benchmarking framework includes 13 benchmark families and emphasizes measurable quality, timing, energy and time-to-solution ([framework](https://www.ionq.com/resources/benchmarking-white-paper?trk=article-ssr-frontend-pulse_little-text-block)).

QARCH-007 does not claim these ideas are new. It adds a small, explicit predictive check to the Luminous pipeline so an unvalidated likelihood model cannot silently steer an architecture decision.

## What the module does

Given a declared outcome set, predicted categorical probabilities, and held-out observation counts, it reports:

- empirical outcome frequencies;
- observed total-variation distance between predicted and empirical distributions;
- multiclass Brier score and mean log loss;
- an explicit infinite-log-loss diagnostic when a model assigns zero probability to an observed outcome;
- a finite-sample TV radius and upper bound only when the stated IID categorical assumption is accepted;
- an evidence provenance class and a three-way policy result: Pass, Reject, or InsufficientEvidence.

The three-way gate matters. Too few observations or unknown/correlated sampling is **insufficient evidence**, not a pass. A systematic mismatch or an impossible observed outcome is a rejection under a sufficient holdout. A simulated fixture that passes remains simulated; it is not promoted to physical model validation.

The module validates probability distributions without silently renormalizing them, requires matching outcome/prediction/count dimensions, checks unique labels and non-empty model/version/workload metadata, and checks count accumulation for overflow.

## Finite-sample bound and assumption

For \(K \geq 2\) categorical outcomes and \(n\) IID observations, use the Weissman-style concentration bound:

\[
P\left(\operatorname{TV}(\hat p,p)\geq r\right)
\leq (2^K-2)e^{-2nr^2}.
\]

At confidence \(1-\alpha\), the radius is

\[
r_{n,K,\alpha}
=\min\left(1,
\sqrt{\frac{\log(2^K-2)+\log(1/\alpha)}{2n}}\right).
\]

By the triangle inequality, the model-to-true-distribution TV is bounded by observed prediction-to-empirical TV plus this radius, capped at one. The implementation computes the logarithmic cardinality term stably rather than directly constructing \(2^K\).

This is only a finite-sample bound under its stated IID categorical assumptions. Adaptive sampling, correlated outcomes, distribution shift, missing outcome categories, or a post-selected holdout can invalidate the stated interpretation. Those cases return InsufficientEvidence until a suitable design-specific method is supplied.

## Gate semantics

A policy is frozen before the holdout is evaluated and includes:
- minimum observation count;
- confidence level;
- maximum allowed TV upper bound;
- maximum allowed multiclass Brier score;
- optional maximum mean log loss.

The result is:
- **InsufficientEvidence** if the sample is too small or the IID premise is not established;
- **Reject** if the model assigns zero probability to an observed outcome or a predeclared metric threshold fails;
- **Pass** only if the evidence gate is applicable and all configured thresholds pass.

A Pass refers only to this particular model/version, workload fingerprint, holdout, and policy. It does not prove physical correctness outside that domain or globally calibrate the model. Report individual metrics; do not hide failure behind one aggregate score.

## How it connects the stack

### Symtropy QSIM-001
Symtropy can create deterministic physical-context scenarios and perturbation sweeps within a declared classical model. The outcomes of those runs can check consistency against analytic references or an independent simulator. They cannot be called physical calibration data unless compared with measurements.

### QARCH-001 and QARCH-002
Photonic path/loss and adaptive fusion models may emit outcome-probability hypotheses, but those probabilities need explicit physical meaning and provenance. If repeated fusion outcomes are state-dependent or adaptive, do not pretend they are IID draws from a single fixed categorical distribution.

### QARCH-005 and QARCH-006
Run the predictive-adequacy gate on the declared likelihood models and relevant held-out observations before relying on the model scores to rank experiments. On rejection, investigate model discrepancy or recalibrate. On insufficient evidence, request a discriminating experiment or abstain. Do not automatically convert a failed model into a different unvalidated likelihood.

### QARCH-003
The final quantum-versus-classical comparison still requires matched accepted output quality, end-to-end time/cost/energy where measurable, retries, verification and a credible classical baseline. A predictive model that passes this gate has not demonstrated quantum utility.

## Benchmark plan

### Deterministic contract fixtures
1. Exact 0.5/0.5 prediction and 50/50 held-out counts: observed TV is 0; Brier score is 0.5.
2. Prediction 0.9/0.1 and observations 10/90: observed TV is 0.8; policy must reject.
3. Prediction 1/0 with one observed event in the zero-probability category: infinite-log-loss diagnostic and reject.
4. Fewer than the preregistered minimum number of observations: InsufficientEvidence.
5. Correlated/unknown sampling: no IID radius and InsufficientEvidence.
6. Invalid distribution, duplicate labels, dimension mismatch, non-finite data and u64 count overflow: fail closed.
7. One category with probability 1: zero TV and zero IID sampling radius.

### Model-calibration experiments
- Freeze model versions, workload fingerprints, outcomes, thresholds and holdout before evaluating a model.
- Report per-model Brier/log loss, observed and upper-bound TV, sample size, provenance and validity regime.
- Plot predictive metrics and acceptance rate against sample size and operating regime.
- Keep devices, calibration runs, or environmental regimes held out as groups; avoid leaking adjacent traces across fit and holdout splits.
- Include distribution shift and deliberately misspecified noise controls.
- Compare active decision regret and experiment cost with the gate on and off. If the gate causes abstention, report it as such rather than counting it as a wrong choice or hiding it.
- Compare with proper scoring rules and suitable calibration tests for the actual data-generating process; this finite categorical gate is a starting contract, not a universal calibration suite.

## Proposed targets

- 100% agreement with independent analytic fixtures within numerical tolerance.
- Zero false accepts in invalid-input and zero-probability-observation negative controls.
- No Pass when there is insufficient data or the stated sampling assumption is not justified.
- On held-out physical data, report calibration and decision regret by operating regime, including OOD degradation.
- On a preregistered robustness study, improve or preserve decision quality relative to ungated nominal and robust experiment selectors, reporting abstention and extra calibration cost.

These are target criteria, not achieved results or a claim of superiority to any industry benchmark.

## Current boundary

Seven Rust tests are authored. The current environment has not executed them. The module does not collect backend data, infer a likelihood from raw photon events, test IID assumptions automatically, produce an OOD certificate, or prove a physical model true. It validates a specific finite categorical prediction against a specific held-out count record with explicit assumptions.
