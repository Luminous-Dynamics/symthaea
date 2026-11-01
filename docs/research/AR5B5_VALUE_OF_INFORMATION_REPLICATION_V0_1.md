# AR-5B5 Cost-Aware Value-of-Information Replication v0.1

**Status:** research protocol / implementation queue; no sampling controller or VOI advantage is claimed  
**Tracking:** [Issue #9696](https://github.com/Luminous-Dynamics/symthaea/issues/9696)  
**Dependencies:** #9659 noisy/expensive MOO; #9663 noisy evidence/cost receipts; #9692 EventTape vector/noisy outcomes; PR #9429 anytime replay and confidence analysis

## 1. Research objective

Choose the next experiment that most improves a preregistered multi-objective decision per unit of all-in cost.

A fixed number of replicates per candidate wastes budget when evaluations have different noise, cost and decision relevance. A widest-confidence-interval heuristic can help in some settings, but it is not value of information (VOI): it can spend samples on uncertainty that cannot change a Pareto, feasibility or target decision.

The goal is a decision-aware controller that chooses candidate/objective/replicate actions, explains why they were chosen, and proves that every requested, failed, repeated or verified measurement is included in the resource ledger.

## 2. Keep statistical contracts separate

Two sampling modes must not be conflated:

1. **Bounded-observation confidence analysis.** The existing Rust module uses time-uniform Hoeffding intervals from frozen observation bounds and summable alpha spending. Its coverage claim is conditional on independently sampled bounded observations under the declared sampling process. It does not require a Gaussian posterior.
2. **Model-based VOI.** Expected value of sample information requires a predictive distribution for the next observation and a decision-loss model. A Gaussian process, heteroskedastic likelihood, bootstrap or other model must be explicit, frozen, validated and receipt-bound. The model must not be smuggled into the Hoeffding interval or described as assumption-free.

When noise assumptions cannot be supported, the controller must fall back to a declared conservative policy or stop; it must not emit an unjustified EVSI estimate.

## 3. Decision value, not interval width

Let D be the available decision set, theta the unknown objective state, and L(d, theta) a preregistered decision loss. The current Bayes decision is:

\[
d^* = \arg\min_{d\in D} \mathbb{E}[L(d,\theta)\mid \mathcal{E}]
\]

For a sampling action a (candidate, objective, protocol and next replicate), the expected value of sample information is:

\[
\operatorname{EVSI}(a) =
\min_{d\in D}\mathbb{E}[L(d,\theta)\mid \mathcal{E}]
-
\mathbb{E}_{Y\mid \mathcal{E},a}
\left[
\min_{d\in D}
\mathbb{E}[L(d,\theta)\mid \mathcal{E},Y,a]
\right]
\]

The action value should include its complete cost:

\[
\operatorname{NetVOI}(a)=
\operatorname{EVSI}(a)
-\lambda_{\mathrm{cpu}} C_{\mathrm{cpu}}(a)
-\lambda_{\mathrm{wall}} C_{\mathrm{wall}}(a)
-\lambda_{\mathrm{energy}} C_{\mathrm{energy}}(a)
-\lambda_{\mathrm{memory}} C_{\mathrm{memory}}(a)
-\lambda_{\mathrm{verify}} C_{\mathrm{verify}}(a)
\]

The pricing vector and units are frozen before outcomes are visible. If no defensible price vector exists, report a Pareto frontier of decision value versus resource vector instead of choosing arbitrary weights.

Decision losses to study separately:
- false elimination of a truly Pareto-relevant candidate;
- false retention of a certainly dominated candidate;
- unsafe or constraint-violating selection;
- expected hypervolume regret at the final budget;
- wrong decision about crossing a preregistered target.

Do not average safety or feasibility into an ordinary quality score.

## 4. Decoupled objective measurements

For objective j on candidate x, the action can be a single-objective measurement rather than a full vector replicate. This is useful when quality, energy, memory, latency or simulator measurements have different costs.

Requirements:
- record which objective was sampled and why;
- retain objective-specific seed/provenance and evidence;
- model cross-objective noise correlation when observations share physical conditions or sample batches;
- represent missing observations explicitly; never impute a free measurement;
- prevent data-dependent missingness from being treated as independent random sampling;
- include setup, model-update and acquisition-optimization work in controller cost.

A joint vector sample and a decoupled sample are distinct experimental actions and must not share a cost or evidence interpretation by default.

## 5. Baselines and ablations

Minimum policy set:
1. Uniform fixed replication.
2. Random candidate/objective allocation.
3. Largest interval width (heuristic, not VOI).
4. Largest normalized interval width per estimated measurement cost.
5. Successive halving / racing / best-arm-identification baselines.
6. Exact model-based EVSI on tractable small problems (oracle policy).
7. Scalable approximate EVSI / decision-focused uncertainty-reduction policy.
8. qNEHVI and qNParEGO where the declared model and task support a Bayesian MOO comparison.
9. Sobol/random design as a simple non-adaptive control.

Ablate decision-loss definition, posterior/noise model, candidate archive, cross-objective correlation model, cost model and acquisition optimization budget.

## 6. Synthetic and adversarial benchmark design

Start with tractable known-mean, bounded-noise cases so exact decision loss and expected VOI can be computed or approximated by high-replication oracle simulation. Include:
- noisy ZDT-like and DTLZ-like fronts;
- heterogeneous variance per candidate and objective;
- heterogeneous evaluator costs and timeouts;
- candidates with intervals that overlap but cannot change the final Pareto set;
- candidates whose one uncertain objective determines feasibility;
- a delayed-breakthrough case where early narrow intervals are misleading;
- serial-correlation and shared-physical-sample controls;
- cross-objective-correlation controls;
- regime drift / changing means;
- informative missingness and evaluator failure;
- seed reuse/collision and falsely labelled independent-sample controls.

Use the same frozen candidate offers, decision loss, objective bounds, noise process, measurement-cost model and maximum campaign budget for all policies. Separate policy-training tasks from sealed evaluation landscapes.

## 7. Required measurements

Per action:
- candidate/objective/replicate selected;
- predicted EVSI or uncertainty-reduction estimate and model digest;
- expected and measured incremental resource vector;
- acquisition-controller computation cost;
- outcome, raw evidence digest, verifier decision and retry linkage.

Per campaign:
- final Pareto front and hypervolume / IGD+ / additive epsilon;
- decision-regret distribution;
- false-dominance and false-feasibility rates;
- confidence coverage and invalid-model detection rate;
- target-attainment distribution;
- total replicates, controller overhead and all-in resource vector;
- cost to first independently verified correct decision;
- worst-case behavior across held-out landscape families.

Compare at equal all-in resource budgets and equal objective-observation budgets. Do not compare only at equal successful samples: failed samples and controller overhead are part of the experiment.

## 8. Acceptance gates

- EVSI on small finite cases matches a high-accuracy reference calculation within a preregistered tolerance.
- The approximate policy agrees with the exact policy within a declared regret/decision-error tolerance.
- Width-based and random controls are included; a VOI claim must exceed them under paired trials.
- The controller's compute/optimization overhead is charged.
- A policy that misstates its noise assumptions is rejected by an independent verifier.
- Correlated-noise, drift and informative-missingness controls cause explicit model-invalid/coverage-warning outcomes.
- Every action and cost can be reconstructed from the campaign event chain.
- Sealed landscape-family transfer remains separate from tuning results.
- Results include paired confidence intervals and multiplicity-aware comparisons across policy families.

## 9. Current implementation boundary

PR #9429 currently supplies a bounded-observation noisy Pareto analysis primitive and a replay bridge from noisy replicate events into AnytimeHistory. It does not implement the controller, model-based EVSI, decoupled objective measurement policy, or prove a value-of-information advantage. The direct campaign-level EventTape vector/noisy action variants are tracked separately in #9692.

## 10. External reference families

- BoTorch noisy parallel multi-objective BO documents qEHVI, qNEHVI and qNParEGO and recommends qNEHVI for noisy/batch settings: https://botorch.org/docs/v0.17.2/tutorials/multi_objective_bo
- BoTorch constrained noisy MOO adds qNEHVI/qParEGO with explicit constraints: https://botorch.org/docs/tutorials/constrained_multi_objective_bo
- COCO official archives include noisy BBOB, bi-objective, constrained, mixed-integer and large-scale black-box test suites: https://coco-platform.org/data-archive/index.html

No result from these external references transfers automatically to this Rust implementation. Reproduce task definitions, noise models, reference data, batch rules, compute costs and run protocols before making comparative claims.
