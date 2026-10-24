# SE-OPT: Decision-Value and Experiment-Selection Benchmark Contract

**Date:** 2026-10-17  
**Status:** proposed benchmark contract; no selector performance is claimed  
**Tracking:** [SE-OPT-BENCH-001 #7803](https://github.com/Luminous-Dynamics/symthaea/issues/7803)  
**Composes:** [SE-OPT #3686](https://github.com/Luminous-Dynamics/symthaea/issues/3686), [robust frontier #7787 / draft PR #7788](https://github.com/Luminous-Dynamics/symthaea/pull/7788), [evidence-acquisition planner #5596](https://github.com/Luminous-Dynamics/symthaea/issues/5596), [acoustic experiments #7739](https://github.com/Luminous-Dynamics/symthaea/issues/7739), and [plasma multifidelity experiments #7685](https://github.com/Luminous-Dynamics/symthaea/issues/7685).

## 1. The paradigm shift

A frontier says which candidates remain non-dominated under a declared relation. It does not tell us which uncertainty to investigate next. A generic acquisition policy can also be busy but strategically useless: it may reduce model variance or increase hypervolume while leaving the decision unchanged.

The next shared capability should be judged by **decision impact per constrained evaluation budget**, not by information gain alone. Its role is to compare and audit policies implemented by existing domain owners—not to become a new universal optimizer, evidence store, autonomous experiment authority, or engineering approval system.

Core distinctions:

- A low-uncertainty prediction can still be biased.
- High uncertainty does not imply high decision value.
- A candidate being on a robust frontier does not make it the preferred design.
- An experiment's modeled value does not grant permission to perform it.
- Synthetic oracle agreement does not qualify a physical model.

## 2. Explicit decision model

A benchmark instance declares:

- a finite set (or explicitly bounded set) of design alternatives;
- hard feasibility predicates and their evidence/assessment status;
- objective directions, units, reference conditions and measurement/model identities;
- an explicit preference policy (hard thresholds first; utility or lexicographic priorities only when stated);
- uncertainty representation and provenance;
- proposed experiment actions, possible outcomes, declared observation model, fidelity, cost and failure modes;
- the evaluation budget and held-out scenarios.

For a world/scenario $\omega$, let $F(\omega)$ be the feasible designs under the declared hard constraints and $L(d,\omega;\pi)$ the loss of design $d$ under explicit preference policy $\pi$.

When at least one design is feasible, the scenario regret of decision $d$ is

$$
R(d,\omega;\pi)=L(d,\omega;\pi)-\min_{d'\in F(\omega)}L(d',\omega;\pi).
$$

A world with no feasible design is reported as an infeasible/abstention outcome and constraint violation—not silently folded into a convenient finite regret value.

Two uncertainty regimes must remain distinct:

1. **Distribution-backed:** if scenario probabilities have a defensible, named source and calibration status, report expected regret and expected value of sample information (EVSI). A posterior is a model artifact, not physical truth.
2. **Distribution-free:** if only intervals or a scenario set $\Omega$ are justified, report worst-case regret, robust feasibility, and bounds on possible decision changes. Do not infer probabilities from interval widths or endpoint combinations.

Expected regret reduction is

$$
\Delta_e = \mathbb{E}[R_{\mathrm{before}}]-\mathbb{E}[R_{\mathrm{after}\ e}],
$$

only when the outcome probabilities and observation/update model for experiment $e$ are explicitly supplied. For set-based uncertainty, compute a separately labeled worst-case regret change over the declared set and possible experiment outcomes. Do not blend those values with compute time, dollars, safety risk, or privacy risk into an unannounced scalar. Report value and cost separately; ratios such as regret reduction per cost are secondary diagnostics with units and assumptions shown.

If all available experiments are decision-irrelevant, the correct result may be zero modeled value and abstention. The evaluator must not reward action for action's sake.

## 3. Benchmark problem families

The first release should include deterministic, synthetic cases with known ground truth and independently implemented reference answers:

1. **Irrelevant uncertainty:** one metric has large uncertainty but cannot alter feasibility or the chosen design.
2. **Decision boundary:** a small uncertain quantity can reverse the decision; a targeted measurement should distinguish the alternatives.
3. **Biased low-fidelity model:** cheap predictions are precise but systematically biased in a localized part of the design space.
4. **Constraint cliff:** small prediction errors cross a hard feasibility boundary; measure false-feasible and missed-feasible outcomes separately.
5. **Dependence trap:** per-objective or per-input intervals appear to permit an outcome, but a declared joint set excludes some combinations. Rectangular interval logic must not claim correlation-aware certainty.
6. **No-value control:** every available experiment has zero decision impact. Correct behavior is abstention or a documented zero-value ranking.
7. **Missing / failed / censored evaluation:** preserve every attempt and its terminal disposition; never impute a failed observation as a favorable score.
8. **Model misspecification:** an apparently calibrated surrogate fails on a held-out regime; measure robustness to the shift and abstention behavior.
9. **Preference perturbation:** change explicit stakeholder thresholds or priority rules; report recommendation changes and regret while preserving the full objective frontier.
10. **Equal / near-equal candidates:** the right answer may remain unresolved; the system must not fabricate a unique winner.

Synthetic tasks qualify algorithm semantics only. Later engineering cases must attach domain model identities, validity envelopes, uncertainty-calibration evidence and independent held-out validation.

## 4. Policy baselines and fair comparison

At identical initial evidence, with paired seeds and identical cost budgets, compare:

- seeded random selection;
- space-filling / grid selection;
- highest predicted uncertainty;
- entropy/information-gain selection, only where a probability model is defined;
- hypervolume-based multi-objective acquisition, where its reference point and assumptions are declared;
- fixed low-/high-fidelity allocation;
- decision-aware expected-regret reduction under a declared distribution;
- distribution-free worst-case-regret or robust-feasibility reduction under a declared set.

Do not require every algorithm to live inside the dependency-free Rust operations-research crate. The benchmark contract can compare versioned policy adapters, but the adapter output must use the same receipts, budget accounting, and scoring oracle.

### Leakage-resistant evaluation sequence

1. Freeze problem generator, policy versions, scenario split, budget, metrics and statistical analysis before the scored run.
2. Expose only the initial evidence and the permitted observation/update contract to each policy.
3. Let the policy commit to its next action before the evaluator reveals that action's hidden high-fidelity truth or held-out outcome.
4. Record requested, attempted, successful, failed, timed-out, censored and abstained evaluations; actual cost/time; fidelity; config/input hashes; and the reason for selection.
5. Reveal hidden truth only to a separate scorer after the policy's action commitment. Keep policy-selection code unable to call or inspect the scoring oracle.
6. Evaluate final decisions against held-out worlds, not only the same synthetic draws used to construct the policy or tune its hyperparameters.

## 5. Metrics: report a vector, not a leaderboard scalar

Report these independently:

- normalized expected decision regret, only for distribution-backed tests;
- worst-case regret and regret bounds, for set-based tests;
- false-feasible and missed-feasible rates;
- decision reversal / stability rate across held-out scenarios and declared preference profiles;
- frontier precision/recall or hypervolume metrics where their definitions fit the case;
- evaluations and high-fidelity calls consumed, direct cost, solver time/wall time;
- failed / timed-out / censored / missing call rates and abstention rate;
- interval coverage or probabilistic calibration, where the data design actually permits that assessment;
- decision-impact reduction per unit cost, with its normalization stated;
- independent known-answer correctness and oracle-leakage checks.

Hypervolume is useful but insufficient: its value depends on a declared reference point and it may not indicate whether the next experiment changes the decision relevant to the current mission. Entropy reduction and posterior variance are diagnostics, not decision success. Do not average incomparable quantities into a composite leaderboard score.

## 6. Negative controls and adversarial checks

At minimum:

- shuffle experiment labels or disconnect proposed outcomes from the targeted uncertainty;
- include an experiment that measures a highly uncertain but decision-irrelevant quantity;
- inject model bias near a hard constraint or decision boundary;
- perturb or miscalibrate the supplied probabilities;
- provide only intervals and ensure no expected-value/probability output is emitted;
- constrain objective correlation and verify that rectangular interval claims are not described as joint certainty;
- test no-feasible-design and all-evaluations-failed cases;
- ensure the policy cannot read evaluator truth or held-out scenarios;
- permute input order and verify deterministic reports for deterministic policies;
- change preference policy and prove the original frontier remains available.

Any negative control that reports unsupported probability, robust feasibility, or a fabricated positive decision value is a correctness failure, even if average regret improves elsewhere.

## 7. Proposed gates and performance targets

These are acceptance targets to test—not measured results or claims of superiority.

**Correctness gates**
- 100% known-answer agreement for deterministic reference cases.
- Zero false robust-dominance and false-feasible claims on preregistered synthetic ground-truth cases.
- 100% accounting of every evaluation action and terminal disposition.
- Zero scorer-oracle leakage.
- Zero probability claims where only set-valued uncertainty is provided.
- Exact reproducibility for deterministic fixtures and deterministic seeded runs.

**Decision-quality target**
- On preregistered decision-relevant families, target at least 20% lower median normalized cumulative regret than the strongest simple baseline at the same cost budget, across at least 20 problem instances and 10 paired seeds. Report paired effect sizes and confidence intervals; treat this as a target, not as a universal guarantee.
- On the no-value control, decision-value estimates should be zero within declared numerical tolerance, and any non-zero claim needs a traceable reason.
- On constraint-cliff tests, zero false-feasible classifications for the deterministic exact-reference contract. Probabilistic real-world false-feasible rates need a separate sample-size / confidence-bound protocol.

If the decision-aware policy fails to beat the baselines, publish the failure and keep the simpler policy. Benchmark design must not be tuned after inspecting hidden results without versioning and preregistering a new protocol.

## 8. Research context and integration

This contract builds on—not replaces—the research and implementation roles of existing methods:

- De Temmerman & Verbeke (2025), [uncertainty quantification and robust Pareto modelling for manufacturing](https://www.nature.com/articles/s41598-025-97508-z), motivates separating uncertainty modelling from a nominal Pareto result.
- Jiang & Li (2025), [trading off quality and uncertainty in batch Bayesian optimization](https://ojs.aaai.org/index.php/AAAI/article/view/34909), illustrates that acquisition can explicitly trade off exploitation and uncertainty; Symthaea must still evaluate whether that trade-off helps the target decision.
- Zhou, Wang, Lucor & Sauret (published 1 Oct 2026), [variance-constrained robust Bayesian optimization for multi-objective design](https://doi.org/10.1016/j.ress.2026.112359), is a very recent peer-reviewed method that separates expected performance and variability and permits explicit stability tolerances. Symthaea should not claim novelty for robust-frontier analysis or uncertainty-aware acquisition alone; its testable differentiator should be cross-domain decision-regret evaluation, interval/distribution semantics, leakage-resistant benchmarking, and explicit abstention when information cannot change a decision.
- Fan et al. (2024), [UAI/PMLR multi-fidelity Bayesian optimization](https://proceedings.mlr.press/v244/fan24a.html), is a reference for comparing fidelity-aware strategies rather than assuming every expensive model call has the same value.
- Tavana, Santos-Arteaga & Di Caprio (2018), [value of information as verification and regret prevention in algorithmic search](https://doi.org/10.1016/j.ins.2018.03.032), provides an earlier formal basis for valuing information by whether it confirms a good choice or prevents a regrettable one. The benchmark should make that decision linkage measurable under current engineering evidence contracts.
- [OpenMDAO's official documentation](https://openmdao.org/) is a useful precedent for connecting multidisciplinary models and variable-fidelity tools. It does not by itself provide the provenance, epistemic-status, or evidence-admission semantics required here.

### Ownership boundaries

- Robust frontier reports relations from supplied intervals; it does not select experiments.
- SE-OPT (#3686) owns the shared design-space, objective/constraint, uncertainty and tool-adapter program.
- The evidence-acquisition planner (#5596) owns authorization-bounded evidence requests and acquisition policy semantics.
- Acoustic (#7739), plasma (#7685), aircraft, and other domain tracks provide domain-specific evaluators, costs, constraints, and calibration evidence.
- This document specifies how policies are evaluated consistently; it does not declare any policy correct before the benchmark is run.

## 9. Implementation sequence

1. Freeze the data/receipt schema and synthetic challenge definitions as versioned fixtures.
2. Implement an independent small-instance scoring oracle and hand-calculated known answers.
3. Add paired-budget baseline runners and leakage controls.
4. Run the suite before tuning the decision-aware policy; publish raw per-case receipts and aggregate metrics.
5. Implement or integrate a reusable policy only after the baseline reveals a clear gap.
6. Validate on independent domain data with uncertainty calibration and physical/tool qualification kept as separate gates.

**Non-goal:** no aircraft performance claim, physical validation, autonomous experiment execution, or airworthiness decision is established by this benchmark.
