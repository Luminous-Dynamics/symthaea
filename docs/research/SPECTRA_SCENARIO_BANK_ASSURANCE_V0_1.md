# SPECTRA Scenario-Bank Assurance — Research Specification v0.1

**Status:** Research specification; no scenario-coverage guarantee or empirical result.
**Research issue:** [#10618 — qualify scenario-bank coverage and hidden-world robustness](https://github.com/Luminous-Dynamics/symthaea/issues/10618)
**Programme:** [#10490 — robust goal-directed sequential design](https://github.com/Luminous-Dynamics/symthaea/issues/10490)
**Implementation context:** [#10599 — anytime robust simulation allocation](https://github.com/Luminous-Dynamics/symthaea/pull/10599)

## 1. Objective

Qualify the scientific scope of SPECTRA's scenario bank.

A Monte Carlo confidence sequence can correctly bound the score mean of a supplied simulator. A robust maximin or lower-tail CVaR certificate can correctly compare actions over a frozen finite scenario set. Neither result, by itself, establishes that the frozen scenarios include the plausible worlds relevant to the scientific target.

The next assurance layer must therefore distinguish **uncertainty inside the declared world set** from **uncertainty about whether the declared world set is adequate**.

This work evaluates the scenario bank as a scientific assumption. It does not promise universal protection against unenumerated data-generating processes.

## 2. Four non-interchangeable uncertainty layers

Every SPECTRA report and certificate should enumerate these scopes separately:

| Layer | Question | Evidence owner |
|---|---|---|
| Monte Carlo uncertainty | How uncertain is the simulated score mean for a supplied action × scenario generator? | #10599 / #10598 |
| Scenario-weight uncertainty | How uncertain are the relative weights assigned to represented scenarios? | This specification |
| Scenario-coverage uncertainty | Which plausible mechanisms or interactions may be missing from the bank? | This specification / hidden evaluator |
| Within-scenario model uncertainty | Does the fitted observation model describe held-out data and the target population? | #10555 / #10582 / #10602 |

Never combine these into one confidence interval. The uncertainty objects have different targets, assumptions, and calibration procedures.

## 3. Scope and certificate binding

A robust selection certificate must bind the exact:
- action-set snapshot and digest;
- scenario-set snapshot and digest;
- normalized scenario weights or ambiguity-set specification;
- robust objective and its parameters (including CVaR tail mass);
- utility ranges, epsilon and familywise alpha;
- score-generator, PRNG and method versions;
- score receipt, selection receipt and independent-verifier version;
- scenario-bank development/evaluation manifest identifiers.

A certificate remains valid only for that frozen scope. If a scenario is added, removed, redefined, or reweighted, the original certificate must remain an immutable historical result for its original scope. The changed scope requires a new manifest, new receipt and fresh certification.

Do not silently extend the original familywise alpha allocation to an unbounded sequence of scenario additions. If adaptive scenario discovery is eventually required, specify a separate scenario-epoch multiplicity or e-process construction and prove its coverage before enabling it.

## 4. Scenario-weight uncertainty

Compare at least four models:

1. **Fixed weights:** current normalized finite scenario weights.
2. **Weight polytope:** predeclared intervals and linear constraints on plausible scenario weights.
3. **Contamination neighborhood:** a bounded mass may be reassigned adversarially within a named support.
4. **Distributional ambiguity:** a named metric and radius over scenario distributions, only if the metric and radius have an empirical or scientific calibration rationale.

For each candidate action, compute the worst-case robust utility over the frozen admissible weight set. Evaluate Maximin and lower-tail CVaR separately; do not assume they are interchangeable.

Freeze the ambiguity set and radius before opening confirmatory seeds. Report the complete sensitivity curve across preregistered radii. Do not select the radius that produces the most favorable result after observing evaluation outcomes.

A mathematically convenient Wasserstein or KL ball is not automatically a scientifically justified uncertainty set. Record how its support, metric, and radius were selected and what population or mechanism they represent.

## 5. Hidden-world evaluation

Maintain two distinct scenario sources:

### 5.1 Development bank

Publicly available to policy and allocator developers. It may be used for implementation, debugging, and exploratory tuning. Every policy and benchmark configuration trained or tuned against it must record that exposure.

### 5.2 Confirmatory evaluator bank

Sealed or otherwise inaccessible to policy tuning until configurations are frozen. Generated independently from the policy's development bank and covering predeclared alternative mechanisms and parameter ranges.

At minimum include both single-mechanism omissions and interaction stresses:

- criterion drift × lapse probability;
- participant/session heterogeneity × evidence-dependent missingness;
- heavy-tailed evidence × criterion shift;
- RT quantization × timeout selection;
- class imbalance × missing-not-at-random response;
- deployment-target covariate shift;
- stimulus-dependent lapses;
- non-decision-time shift and RT-channel corruption.

Include near-boundary and severity ladders, not only extreme cases. Preserve the generator version, parameter envelope, random-stream provenance and evaluator budget.

A red-team search may search within a bounded mechanism family for adverse worlds. Preserve the search algorithm, candidate count, compute budget and selection procedure. An adversarially discovered synthetic world is a falsification case, not automatically evidence that the world is scientifically plausible. Assign it an evaluator-only label and require a separate rationale or external support before treating it as part of a plausible target population.

The policy and allocator must not receive evaluator-only scenario labels, oracle parameters, hidden outcome scores, or future evaluation seeds.

## 6. False-robustness estimand

Let `C` denote the event that the frozen-bank certificate passes. Let `R_H` be robust regret under the separately generated hidden evaluator distribution or a named hidden stress family. Let `tau` be a material regret threshold fixed before evaluation.

Report the conditional false-robustness rate:

`P(R_H > tau | C)`

and the unconditional joint event:

`P(C and R_H > tau)`.

Do not report the conditional rate alone: its denominator may be small or zero. Include Monte Carlo uncertainty for both, the certificate rate, and every hidden-scenario family.

Also report:
- frozen-bank certified regret;
- hidden-bank expected and worst-scenario regret;
- regret under weight perturbations;
- stress detection recall and false-alarm rate;
- false-adequacy / false-confidence rate;
- unsupported and non-evaluable scenario counts;
- compute spent on stress discovery versus score refinement;
- selection changes after scenario-bank expansion.

The oracle is used only by the evaluator after the policy's action, model and receipt have been frozen.

## 7. Relationship to model adequacy and de-amplification

Scenario-bank assurance is not a replacement for held-out predictive model criticism or known-truth inferential-risk evaluation.

- #10582 evaluates whether a fitted observation model predicts held-out observations under its split assumptions.
- #10602 evaluates downstream target-distribution inferential risk under known-truth misspecification.
- This specification evaluates whether the finite scenario bank and its weights support the robustness scope claimed by SPECTRA.

A model-adequacy veto cannot be overridden by a favorable information, representativeness, or utility score. A scenario-bank certificate cannot be interpreted as evidence of human-population representativeness unless its target-population sampling assumptions have been separately justified.

## 8. Benchmarks

Compare policies under equal simulation budgets and, in separate human-study planning simulations, equal participant-time budgets:

- fixed-weight Maximin;
- fixed-weight lower-tail CVaR;
- weight-polytope robust Maximin/CVaR;
- contamination-neighborhood robust objectives;
- distributionally robust variants when radius calibration is supported;
- robust policies with randomized audit/sentinel floors;
- a frozen-bank-only policy as a deliberate negative control;
- an oracle evaluator-aware policy as an unattainable upper reference only.

Measure:
- (PCS_\epsilon) over the frozen bank;
- false-robustness conditional and joint rates under hidden worlds;
- worst-hidden-world regret and worst-decile regret;
- detection recall / false-alarm frontier;
- sensitivity to weights, ambiguity radius and tail mass;
- certificate invalidation and recertification behavior after bank mutation;
- action-selection stability and policy regret;
- Monte Carlo and participant-time budgets separately;
- failure-inclusive availability, unsupported regimes and exact replay.

The benchmark should include development/confirmatory leakage controls, action/scenario-count scaling, seed replays, tied and near-tied candidates, and hidden interaction stresses. Report each mechanism family independently before any aggregate.

## 9. Proposed qualification targets — not results

- 100% exact binding of every certificate to its scenario/weight manifest.
- Zero silent carry-forward after scenario or weight mutation.
- No use of evaluator-only scenario data by the policy or allocator.
- At most 5% false-robustness in each preregistered supported hidden-world family, with Monte Carlo confidence bounds.
- At least 90% detection of material hidden stress where that family is identifiable from available observables, at a fixed preregistered false-alarm ceiling.
- No superiority claim based solely on frozen-bank PCS or in-bank maximin/CVaR.
- 100% reporting of failed, unsupported and non-evaluable regimes.

The numeric targets are aspirations. Feasibility and statistical precision must be established in a development campaign before they are locked for confirmatory evaluation.

## 10. Research comparators

- Gao, Xiao, Zhou & Chen (Automatica 2017), [Robust ranking and selection with optimal computing budget allocation](https://doi.org/10.1016/j.automatica.2017.03.019): worst-case design performance over a finite input-uncertainty set, with R-OCBA.
- Wan, Li & Hong (Asia-Pacific Journal of Operational Research 2025), [New Additive OCBA Procedures for Robust Ranking and Selection](https://doi.org/10.1142/S0217595925400032): fixed-budget robust R&S under ambiguity sets, minimizing probability of incorrect selection.
- Tang, Sloman & Kaski (AISTATS/PMLR 2026), [Representative, Informative, and De-Amplifying](https://proceedings.mlr.press/v300/tang26d.html): representativeness, information and error de-amplification under misspecification and shift.
- Callahan & Catanach (AISTATS/PMLR 2026), [On the Misinformation in a Statistical Experiment](https://proceedings.mlr.press/v300/callahan26a.html): informative experimental designs can create confident but incorrect inference under joint misspecification.
- Wen et al. (ICML/PMLR 2026), [Robust Sequential Experimental Design for A/B Testing](https://proceedings.mlr.press/v306/wen26i.html): robust sequential allocation and worst-case MSE, with a distinct treatment-effect estimand.

These are serious comparators, not evidence that SPECTRA has reproduced or surpassed them.

## 11. Execution sequence

1. Add immutable scenario/weight manifest binding to the evaluator and report schema; do not alter the existing Monte Carlo confidence sequence.
2. Add a scope-mutation negative control: mutate one scenario, weight, ambiguity radius, objective parameter or evaluator version and require the old certificate to remain scoped to its old manifest and the new claim to require recertification.
3. Establish public development and independent confirmatory scenario generators with strict oracle-data separation.
4. Implement fixed-weight, weight-polytope and contamination baselines before more complex distributional ambiguity.
5. Add hidden-world false-robustness metrics to #10601's locked campaign.
6. Integrate #10602 target-risk outputs and #10582 adequacy dispositions without merging their estimands.
7. Run locked campaigns and publish complete scenario-wise outcomes, uncertainty and failure dispositions.

## Non-goals

- Universal guarantees over unenumerated worlds.
- Treating synthetic adversarial stress as real-world prevalence.
- Mixing Monte Carlo coverage with scenario-bank coverage.
- Retrofitting new scenarios into old certificates.
- Human-subject collection or operational authorization.
