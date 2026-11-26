# Robust and Pareto-Aware Scientific Decision Design (RPSD) v1

**Status:** research protocol and exact small-instance references
**Parent:** Decision-Sufficient Scientific Intelligence v1
**Implementation:** [scripts/research/sdcb_robust_pareto.rs](../../scripts/research/sdcb_robust_pareto.rs)
**Robust fixture:** [SDCB_ROBUST_DECISION_EPISODE_V1.example.json](SDCB_ROBUST_DECISION_EPISODE_V1.example.json)
**Robust expected result:** [SDCB_ROBUST_DECISION_EPISODE_V1.expected.json](SDCB_ROBUST_DECISION_EPISODE_V1.expected.json)
**Pareto fixture:** [SDCB_PARETO_DECISION_EPISODE_V1.example.json](SDCB_PARETO_DECISION_EPISODE_V1.example.json)
**Pareto expected result:** [SDCB_PARETO_DECISION_EPISODE_V1.expected.json](SDCB_PARETO_DECISION_EPISODE_V1.expected.json)

## 1. Why another layer is needed

DSSI v1 makes two assumptions that are rarely safe in real scientific work: likelihoods are known, and the task owner supplies one utility schedule that can be treated as exact. Standard Bayesian optimal experimental design can become self-reinforcing when its assumed model is misspecified; model errors can distort both posterior inference and the acquisition function.

RPSD adds two complementary contracts:

1. **Finite ambiguity-set robust EVSI:** choose experiments against multiple plausible complete model scenarios, without assigning unsupported probabilities to the scenarios.
2. **Pareto and hard-constraint decision support:** retain non-dominated action/experiment policies across objectives, while enforcing hard requirements separately rather than hiding violations inside a weighted score.

These are established research ideas individually. RPSD's project contribution is to make them small, exact, inspectable reference contracts that compose with DSSI and EGBT: one decision state, multiple model scenarios, explicit safety constraints, and evidence-bound observations.

## 2. Robust experiment selection over an ambiguity set

Let the hypothesis prior be p(h). The uncertainty set contains complete scenarios s in S. Each scenario supplies:

- a utility function U_s(a,h);
- an experiment likelihood P_s(o | h,e).

The scenarios are plausible alternatives, not a probability distribution. No scenario weights are silently invented.

The current robust action is:

\[ V_{robust,0}=\max_{a\in A}\min_{s\in S}\sum_h p(h)U_s(a,h). \]

An experiment policy maps an observed outcome to an action, pi: O_e -> A. The policy must be shared across scenarios: the agent does not get to observe which hidden scenario is true. Its value in scenario s is:

\[ V_s(\pi,e)=\sum_h\sum_o p(h)P_s(o\mid h,e)U_s(\pi(o),h). \]

The robust post-experiment value and gain are:

\[ V_{robust,e}=\max_{\pi:O_e\to A}\min_{s\in S}V_s(\pi,e),\qquad RIG(e)=V_{robust,e}-V_{robust,0}. \]

Net robust gain subtracts cost after explicitly converting cost to utility units. This is a finite-scenario, maximin criterion; it is not a distributionally robust guarantee outside the declared ambiguity set.

### Important modeling choices and limitations

- **Scenario coherence:** each scenario represents one complete joint assumption about utility and the observation process. The implementation accepts a utility matrix per scenario and a likelihood matrix per scenario/experiment.
- **No scenario probabilities:** the objective takes the worst case over the finite set. It does not average scenarios.
- **Common policy:** the outcome-to-action mapping cannot depend on the hidden scenario ID.
- **Deterministic policies only:** randomized policies are not enumerated. The result is therefore an exact optimum over pure policies, not necessarily the optimum over mixed policies.
- **Finite support only:** robustness is only with respect to the supplied scenarios. Missing a plausible failure mode invalidates any broader guarantee.
- **Conservatism is visible:** report the whole scenario-value vector, not only the minimum.

## 3. Robust reference counterexample

Two hypotheses are equally likely. Action A is correct under h1 and action B under h2; the current maximin value is 0.5.

Two candidate probes are compared over two equally unweighted plausible likelihood scenarios:

| Probe | Scenario 1 | Scenario 2 | Worst-case improvement |
|---|---|---|---:|
| brittle probe | 95% accurate signal | 95% reversed signal | 0.00 |
| robust probe | 70% accurate signal | 70% accurate signal | 0.20 |

The brittle probe looks excellent under one assumed model and misleading under another. Because the agent cannot observe which scenario is true, the common policy cannot exploit the apparent signal reversal. Its robust value is only the stop value. The robust probe yields value 0.70 in both scenarios, for robust gain 0.20. At cost rate 0.10 per unit and one cost unit, net robust gain is 0.10.

This is a hand-checkable demonstration of model-risk sensitivity, not an empirical scientific-agent result.

## 4. Pareto decisions without reward scalarization

Let each hypothesis/action pair have a vector of objectives U_j(a,h). The reference assumes every objective is represented in a maximize direction. For example, represent risk as negative loss or as safety margin; do not mix opposite directions without transforming and documenting them.

The expected objective vector for action a is:

\[ \mathbf{V}(a)=\sum_h p(h)\mathbf{U}(a,h). \]

Action a dominates b if it is no worse on every objective and strictly better on at least one. The Pareto set retains all non-dominated actions.

Hard constraints are not Pareto objectives that can be traded away. RPSD allows a lower bound for an objective to be required under every hypothesis with nonzero prior mass:

\[ U_j(a,h)\geq m_j \quad \forall h:p(h)>0. \]

An action violating a hard floor under even one plausible hypothesis is excluded before Pareto ranking, even if its posterior-average score looks acceptable. This is a conservative finite-hypothesis safety check, not a complete safety case.

The experiment-policy extension enumerates outcome-to-action maps and reports non-dominated expected objective vectors plus cost. Costs are minimized as a separate coordinate; no weighted scalar score is created.

## 5. Pareto reference counterexample

Two hypotheses are equally likely. There are three actions with two objectives: throughput and safety margin.

| Action | Throughput | Safety margin h1 | Safety margin h2 |
|---|---:|---:|---:|
| fast | 10 | -4 | 1 |
| balanced | 8 | -1 | -1 |
| safe | 6 | 0 | 0 |

The fast action has expected safety margin -1.5, which is above a declared average floor of -2. But it violates the floor under h1. RPSD excludes it because hard constraints are checked per plausible hypothesis, not only on the average.

The remaining Pareto options are balanced (8,-1) and safe (6,0). An illustrative two-outcome, uninformative experiment can produce the intermediate policy vector (7,-0.5), but this does not make the measurement scientifically informative; the cost dimension and stop baseline must remain visible when comparing policies.

## 6. Reference implementation

The standard-library-only Rust reference implements:

- maximin current action across a finite ambiguity set;
- exact enumeration of outcome-contingent pure policies shared across scenarios;
- worst-case robust experiment gain and cost-aware abstention;
- scenario-level outcome value reporting;
- Pareto frontiers of actions and outcome-contingent experiment policies;
- non-compensatory per-hypothesis lower-bound constraints;
- cost as a separate minimized Pareto coordinate;
- deterministic tie-breaking, validation, and search-space limits.

Limits are deliberately explicit: robust ambiguity sets are capped at 8 scenarios, 5 actions, 4 outcomes and 8 experiment candidates; Pareto enumeration is capped at 8 actions, 6 objectives, 4 outcomes and 8 candidates. These bounds are safeguards against accidental combinatorial explosion, not performance claims.

**17 Rust tests are authored. They have not been compiled or executed in this environment because rustc/cargo are unavailable.** Independent JavaScript calculations were used to check the two frozen examples; this is not Rust execution.

## 7. Benchmarks and external comparisons

Do not claim robust or Pareto superiority from these analytic fixtures. Integrate RPSD as an additional reporting layer into external scientific method and active discovery tasks.

| Benchmark / prior art | What it contributes | RPSD / SDCB addition |
|---|---|---|
| petri-bench (working draft v0.5, 2026) | Procedural mystery worlds, controlled experiments, inference, method-rigor scoring, call budgets, p-hacking/probe-only audits, no LLM judge in headline score | Add likelihood/utility misspecification, ambiguity-set robustness, decision regret and hard safety constraints; preserve its original score |
| Beyond Accuracy (2026) | Decomposes belief updates, hypothesis discrimination, experiment planning and intervention prediction | Add robust values across alternative mechanism/observation models and decision outcomes |
| ActiveSciBench / LLM-AutoSciLab | Budgeted mechanism discovery in kinetics and gene-regulatory networks | Evaluate robust experimental choices under outliers, alternative noise models and model misspecification |
| Autonomous Science Bench | Real-lab design-build-test-learn campaigns; separate discovery and unmeasured-design learning | Add cost/safety Pareto fronts and source-bound evidence transitions |
| Active Causal Discovery Benchmark | Budgeted interventions, CPDAG/DAG ceilings and identifiability limits | Report robust intervention value and per-scenario failure, not only graph-recovery scores |
| CoDBench | Physical-dynamics model quality, generalization and resource comparisons | Use as a physical predictor suite; apply RPSD only when downstream action/utility is explicitly defined |

### petri-bench deserves immediate attention

petri-bench is particularly relevant because it scores how the scientific conclusion was earned, rather than only whether the final answer is correct. Its current working draft v0.5 reports a standardized 540-episode sweep across six frontier models, 30 procedural tasks and three episodes per task. The report separates correctness, method rigor and efficiency and audits p-hacking and probe-only output matching. It reports that no evaluated model solved more than half the tasks in that snapshot, while its curated-value reference solver performed much better; the blind-value ablation materially narrows the interpretation of that gap. The report explicitly retracts an earlier claim about the contribution of value knowledge. Treat these as working-draft benchmark results, not universal rankings.

SDCB should add challenges that extend rather than duplicate petri-bench:

1. blind target metric and full metric-vector identification to prevent output-matching shortcuts;
2. factorial and interaction identification with explicit oracle ceilings;
3. phase-boundary localization and regime-change detection;
4. noise-family and model-misspecification shifts;
5. action/utility uncertainty and preference perturbations;
6. hard safety constraints and robust experiment portfolios;
7. independent evidence receipts and posterior/decision replay.

## 8. Benchmark metrics and proposed targets

Report each axis separately:

- worst-case robust gain across the frozen ambiguity set;
- scenario regret and tail scenario value;
- robust-vs-nominal decision disagreement;
- Pareto hypervolume only with a frozen reference point, plus coverage and diversity;
- hard-constraint violation rate (must be zero on frozen safety controls);
- experiment cost and budget efficiency;
- sensitivity to adding/removing plausible scenarios;
- decision and mechanism calibration;
- evidence integrity and independent replay.

Project targets, not industry standards or achieved results:

- 100% correct robust choice on hand-computable finite ambiguity fixtures;
- zero hard-constraint violations on the declared safety-negative-control suite;
- report the entire scenario-value vector for every robust choice;
- no single scalar aggregate used to hide Pareto trade-offs;
- beat nominal EIG/EVSI baselines on preregistered misspecification families without materially regressing in well-specified families;
- maintain a Pareto set whose hypervolume and diversity beat single-objective scalarized baselines under matched evaluations;
- reproduce across five or more seeds and at least one held-out scenario family before external claims.

## 9. Research programme and next steps

1. Replace hand-authored scenario sets with a versioned perturbation generator for omitted variables, noise misspecification, covariate shift, parameter drift, and alternative mechanisms.
2. Add generalized Bayesian/Gibbs updates and compare them with standard Bayes under controlled outliers and incorrect noise assumptions.
3. Add model discrepancy detection and a deliberate exploration budget for representative/de-amplifying designs.
4. Extend pure robust policies with mixed policies via a checked linear-program solver; retain the pure reference as an independent oracle.
5. Add distributional ambiguity sets with a declared distance metric and radius sensitivity curve.
6. Add multi-objective sequential policies and compare against exact small-instance Pareto oracles.
7. Integrate Mycelix-style identity/authority receipts only as provenance and authorization; never treat identity consensus as scientific truth.
8. Keep analytic, synthetic, simulated, and physical observations in distinct evidence classes.

## 10. References

1. Barlas, Sloman & Kaski, Robust Experimental Design via Generalised Bayesian Inference (2025), https://arxiv.org/abs/2511.07671. Introduces generalized Bayesian optimal design and Gibbs expected information gain for outliers and incorrect noise assumptions.
2. Tang, Sloman & Kaski, Generalization Analysis for Bayesian Optimal Experiment Design under Model Misspecification (2025), https://arxiv.org/abs/2506.07805. Studies representativeness, error de-amplification and generalization under misspecification.
3. Rainforth et al., Modern Bayesian Experimental Design (2024), https://doi.org/10.1017/S0962492924000023. Broad treatment of Bayesian/decision-theoretic design and sequential non-myopic methods.
4. Aqeeli, Leelawat & Shorthouse, Novelty-aware evolutionary Bayesian optimisation for multi-objective discovery science (2026), https://doi.org/10.1039/D6DD00134C.
5. petri-bench report, working draft v0.5 (2026), https://petri-labs.org/bench/report; benchmark portal https://petri-labs.org/bench/; archived artifact DOI 10.5281/zenodo.20618024.
6. Deb, Shah & Krishnan, Active Causal Discovery Benchmark (2026), https://arxiv.org/abs/2609.31675.

## 11. Claim boundary

This is an exact finite reference for a declared ambiguity set and finite objective model. It does not guarantee robustness to scenarios that were not included, prove physical safety, authenticate observations, or establish SOTA performance. No Rust compilation, model inference, external benchmark run, or physical experiment has been performed.