# Decision-Sufficient Scientific Intelligence (DSSI) v1

**Status:** research protocol and analytic reference implementation
**Parent:** SDCB evidence-gated belief transitions v1
**Implementation:** [one-step decision value](../../scripts/research/sdcb_decision_value.rs) and [exact finite-horizon oracle](../../scripts/research/sdcb_nonmyopic_oracle.rs)
**Design fixture:** [SDCB_DECISION_VALUE_EPISODE_V1.example.json](SDCB_DECISION_VALUE_EPISODE_V1.example.json)
**Schema:** [SDCB_DECISION_VALUE_EPISODE_V1.schema.json](SDCB_DECISION_VALUE_EPISODE_V1.schema.json)

## 1. Research thesis

A scientific system should not maximize information in the abstract. It should acquire the information that can improve a declared decision, while retaining enough uncertainty to avoid unsupported conclusions.

An experiment can sharply distinguish two hypotheses yet have no effect on the best available action. Another experiment can provide less total information but move the posterior across a decision boundary. The latter is more useful when the purpose of the campaign is to select an action.

DSSI adds a decision layer to SDCB:

**hypotheses → decision model → candidate experiments → expected decision value → verified observation → evidence-gated belief update → revised decision**

This is not a claim that expected value of sample information or Bayesian optimal experimental design is new. Decision-theoretic and Bayesian optimal experimental design are established fields. The project contribution is to make decision value, decision sufficiency, identifiability, and evidence-gated state transitions one auditable interface that can be tested inside scientific-agent benchmarks.

## 2. The mathematical distinction

Let hypotheses be H = {h_1, ..., h_n}, with posterior p(h). Let A be the available action set and U(a,h) a declared utility for action a under hypothesis h.

The current Bayes action and its value are:

\[ V_0(p) = \max_{a\in A}\sum_h p(h)U(a,h). \]

For experiment e with outcomes o and likelihood P(o | h,e), define the posterior p(h | o,e). Its expected post-experiment decision value is:

\[ V_e(p) = \sum_o P(o|e)\max_{a\in A}\sum_h p(h|o,e)U(a,h). \]

The gross expected value of sample information is:

\[ \operatorname{EVSI}(e)=V_e(p)-V_0(p). \]

Net value subtracts experiment cost after explicitly converting that cost to the same utility unit:

\[ \operatorname{NetEVSI}(e)=\operatorname{EVSI}(e)-\lambda c(e). \]

The v1 reference uses a scalar cost-rate lambda for transparent finite fixtures. Real deployments should preserve separate money, time, energy, safety, and opportunity constraints; collapsing those into one scalar without an explicit policy is not acceptable.

By contrast, expected information gain is:

\[ \operatorname{EIG}(e)=H(H)-\mathbb{E}_{o|e}[H(H|o,e)]. \]

EIG measures information about hypotheses. EVSI measures expected improvement in the best declared decision. These objectives can rank experiments differently.

## 3. Frozen counterexample: more information can be less useful

Consider three hypotheses with prior p = (0.3, 0.3, 0.4) and two actions:

- Under h1 and h2, action A has utility 1 and B has utility 0.
- Under h3, action A has utility 0 and B has utility 1.

The current expected values are A = 0.6 and B = 0.4, so A is selected.

Two experiments are available:

| Experiment | P(o1|h1) | P(o1|h2) | P(o1|h3) | Cost units |
|---|---:|---:|---:|---:|
| Hypothesis-split probe | 1.0 | 0.0 | 0.5 | 1 |
| Decision-boundary probe | 0.8 | 0.8 | 0.2 | 1 |

The hypothesis-split probe distinguishes h1 from h2 but those hypotheses have identical utilities for every available action. Its EIG is higher, yet its EVSI is zero.

The decision-boundary probe produces:

- Outcome 1 with probability 0.56, posterior P(h3|o1) = 1/7, and best expected utility 6/7.
- Outcome 2 with probability 0.44, posterior P(h3|o2) = 8/11, and best expected utility 8/11.

Its expected post-experiment value is 0.8, so EVSI = 0.2. This is a hand-checkable case in which the lower-information probe is more valuable for the declared decision.

This is a demonstration fixture, not empirical evidence that DSSI outperforms a benchmark.

## 4. Decision-sufficient hypothesis classes

Two hypotheses are decision-equivalent for a fixed action set when their utility vectors are identical:

\[ h_i \sim_A h_j \iff U(a,h_i)=U(a,h_j)\quad\forall a\in A. \]

Probability mass can be aggregated within each equivalence class without changing any current expected action utility. The resulting quotient is decision-sufficient for choosing among the declared actions.

Important limits:

- Decision equivalence is not observational equivalence, predictive equivalence, causal equivalence, or ontological identity.
- If the action set or utility model changes, recompute the partition.
- Hypotheses in one decision class may still make different predictions and may matter for future decisions.
- The reference code uses exact finite utility-row equality. It does not silently quantize near-equal utilities.

This suggests a useful architecture: Symthaea need not always reconstruct every hidden detail. It should preserve distinctions that can affect the current decision, while retaining unresolved hypotheses for future objectives.

## 5. New paradigm shifts

### 5.1 Decision-sufficient state, not reconstruction at all costs

Build a task-relative quotient of the hypothesis space around declared actions and utilities. Keep predictive state separate so compression does not erase future scientific value.

### 5.2 Falsification targeted at decision boundaries

Search for experiments where plausible hypotheses disagree about which action is best. This is stronger than searching only for the largest disagreement in parameter space.

### 5.3 Explicit value-of-information budgets

Treat experiments as scarce resources. Compare decision gain with their actual money/time/energy/risk costs, and abstain when the expected benefit does not justify the cost.

### 5.4 Separate epistemic and instrumental progress

Report information gained, mechanism identified, decision changed, expected utility improved, and evidence verified as distinct events. Do not turn all progress into one confidence number.

### 5.5 Evidence-gated decision transitions

Bind a decision proposal to the decision model digest, action set, utility schedule, hypothesis-set digest, current belief version, and experiment proposal. The verified observation then updates the belief ledger; the new action is recomputed from that state. A utility change is a new decision context, not a silent reinterpretation of old evidence.

## 6. Reference implementation

The standalone Rust reference implements:

- validation of finite posterior distributions, action identifiers, utility matrices, and likelihood rows;
- Bayes-optimal current action selection with deterministic tie-breaking;
- exact decision-equivalence partitioning by identical utility vectors;
- expected information gain in nats;
- EVSI over a finite outcome space;
- net-EVSI experiment selection with an explicit cost-to-utility conversion;
- abstention when all experiments have zero or negative net decision value;
- negative controls for high information with no decision value, cost overruns, malformed probabilities, and duplicate candidate IDs.

The source has 15 authored Rust tests and uses only the Rust standard library. It has not been compiled or executed in this environment because rustc/cargo are unavailable. The checked-in Draft 2020-12 episode schema passed meta-schema validation, and the example episode passed schema validation plus independent semantic checks for prior normalization, utility dimensions, unique hypothesis IDs, likelihood dimensions, and likelihood row sums. A separate numerical reference calculation was used to check the frozen three-hypothesis example; this is not Rust execution.

## 7. Benchmark strategy and comparison

Do not make a new standalone leaderboard the primary objective. Add DSSI as a parallel measurement layer to existing scientific-discovery tasks where the task defines an actionable objective.

| Benchmark | Existing capability to preserve | DSSI extension |
|---|---|---|
| Beyond Accuracy | Belief updating, hypothesis discrimination, experiment selection, intervention prediction | Score decision-value gain in addition to mechanism identification; report decision regret and irrelevant-information spend |
| ActiveSciBench / LLM-AutoSciLab | Adaptive mechanism discovery across enzyme kinetics and gene-regulatory networks | Where the task has explicit utility, compare EIG, EVSI, and realized downstream decision improvement |
| Autonomous Science Bench | Real-lab campaigns with separate measured-design discovery and unmeasured-design learning gates | Keep both gates; add decision utility, utility-model provenance, and cost/risk constrained experiment selection |
| Active Causal Discovery Benchmark | Skeleton, CPDAG/DAG and intervention-budget evaluation | Distinguish graph recovery from identifying the causal features needed for a declared intervention decision |
| CoDBench | Physical dynamics prediction and generalization | Add decision-value analysis only for tasks with explicit downstream control/optimization objectives |

Upstream definitions and scores must remain unchanged. Use pinned task versions and adapters, and report DSSI metrics as separate columns.

## 8. Required baselines

1. Random selection and fixed space-filling designs.
2. Maximum expected information gain.
3. Variance reduction or uncertainty sampling where well-defined.
4. Greedy one-step EVSI.
5. Exact finite-horizon dynamic-programming oracle on tractable synthetic cases.
6. Bayesian optimal experimental design using the benchmark's own declared objective.
7. Human or domain-expert design under the same information and budget, where a valid controlled study is possible.

Match observation access, candidate sets, costs, prior information, tool permission, and stopping rules. Compare at equal experiment budget and equal deployment cost separately.

## 9. Metrics and provisional targets

Report separately:

- final decision utility and normalized decision regret against a known oracle;
- cumulative regret over experiment rounds;
- gross EVSI, net EVSI, and realized decision improvement;
- EIG, to expose information/decision-value divergence;
- wasted-information spend: cost spent on experiments with zero measured decision value;
- decision-reversal hit rate: fraction of experiments that reveal a needed action change before the decision deadline;
- mechanism identification conditioned on identifiability;
- calibration, selective risk, and false mechanism claims;
- experiment budget, money, time, energy, latency, and failure/recovery cost;
- evidence receipt validity and independently replayable belief transitions.

Project targets, not standards or achieved results:

- exact agreement with hand-computable EVSI fixtures within 1e-12;
- zero decision-equivalent false distinctions on exact utility-row fixtures;
- select the decision-boundary probe over the higher-EIG irrelevant probe in the frozen counterexample;
- abstain when every candidate has non-positive net EVSI;
- beat random and max-EIG baselines on decision regret under identical small-system budgets before scaling;
- reproduce across at least five seeds and one held-out generating family before a general claim;
- no unauthorized belief updates in the EGBT negative-control suite.

Do not claim superiority from the counterexample alone. A genuine advance requires matched baseline runs on upstream tasks and independent reproduction.

## 10. Statistical and governance requirements

- Freeze the action set, utility matrix, cost conversion, prior, candidate experiment set, and scoring rules before hidden evaluation.
- Utility values require explicit domain-owner provenance and sensitivity analysis. The agent must not choose its own reward function to justify its preferred experiment.
- Report results across multiple plausible utility schedules when stakeholder preferences are uncertain.
- Include safety constraints as non-compensatory feasibility gates; a high expected utility cannot trade away a hard safety constraint.
- Separate train/validation/test by mechanism family and parameter regime, not only random episode rows.
- Include utility perturbations, observational-equivalence pairs, no-op probes, misleading evidence, and candidate-set leakage controls.
- Use paired tasks and common seeds for baseline comparisons; report per-family confidence intervals and bootstrap at the task-family level.
- Keep discovery, learning, decision quality, evidence integrity, and resource use as separate gates.

## 11. Exact non-myopic adaptive experiment oracle

One-step EVSI can still be myopic. An experiment may have zero immediate decision value but reveal which follow-up experiment is worth running. DSSI now includes a separate exact finite-horizon dynamic-programming oracle for small finite cases.

For belief state p, remaining experiments E, budget b, and horizon k, the Bellman recurrence is:

\[ V(p,E,b,k)=\max\left\{V_0(p),\max_{e\in E,\ c(e)\le b}\left[-\lambda c(e)+\sum_o P(o|p,e)V(p_{e,o},E\\setminus\\{e\},b-c(e),k-1)\right]\right\}. \]

The policy is adaptive: after each outcome it can choose a different next experiment, or stop. Each candidate is used at most once. Costs are charged at every selected experiment node. The implementation rejects duplicate candidate IDs and caps the exact candidate set at 12 to keep the reference oracle bounded and auditable.

### Sequential counterexample

Use prior p = (0.4, 0.4, 0.2), with actions A and B. Hypotheses h1/h2 both assign utility (1,0), while h3 assigns (0,1). The current best value is 0.8.

Two experiments:

| Experiment | P(o1|h1) | P(o1|h2) | P(o1|h3) | Cost units |
|---|---:|---:|---:|---:|
| screen-a | 0.75 | 1.0 | 0.5 | 1 |
| screen-b | 0.5 | 0.5 | 1.0 | 1 |

Each experiment alone has zero myopic EVSI. A one-step EVSI planner therefore abstains. An exact two-step adaptive policy reaches expected terminal value 0.85, a gain of 0.05 over stopping. With budget one, the value remains 0.8. This fixture establishes why evaluation must include policy value over a sequence, not only the immediate score of the first experiment.

The finite-horizon oracle is intentionally an exhaustive small-instance comparator. It is not a scalable general-purpose planner or an achieved Symthaea model capability. Its role is to give us an exact ground truth against which approximate HDC/CfC, Bayesian, or language-agent experiment policies can be evaluated.

## 12. Limitations and next extensions

The one-step decision-value module assumes supplied likelihoods and utilities. The exact non-myopic oracle handles only small finite candidate sets (at most 12) and is exponential in the horizon/outcome branching. Both assume a finite hypothesis set and known likelihoods; neither learns likelihoods or utility models, solves continuous design problems, models utility ambiguity, or authenticates evidence.

Next research increments:

1. Finite-horizon non-myopic planning against an exact dynamic-programming oracle.
2. Robust EVSI under posterior and likelihood misspecification.
3. Multi-objective Pareto decision value without arbitrary scalarization.
4. Safe experiment design with hard constraints, chance constraints, and intervention authority.
5. Utility elicitation and preference uncertainty with human/domain-owner provenance.
6. Adapter evaluation inside Beyond Accuracy, ActiveSciBench, and ACDB, subject to licenses and available interfaces.

## 13. References

1. Kuroki, Mansouri & Schwaller, Beyond Accuracy: A Diagnostic Benchmark for Hypothesis-Driven Experiment Planning in LLM Agents (22 Nov 2026), https://arxiv.org/abs/2611.29467 and https://schwallergroup.github.io/beyond-accuracy/.
2. Kabra et al., LLM-AutoSciLab / ActiveSciBench (NeurIPS 2026), https://scientific-discovery.github.io/llm-autoscilab-project/.
3. Wen & Liu, Autonomous Science Bench 0.1 (2026), https://asb.open-ssi.org/.
4. Deb, Shah & Krishnan, Active Causal Discovery Benchmark (16 Sep 2026), https://arxiv.org/abs/2609.31675.
5. Alexanderian, Optimal experimental design: Formulations and computations, Acta Numerica 33 (2024), https://doi.org/10.1017/S0962492924000023. The survey reviews Bayesian/decision-theoretic criteria and emerging non-myopic sequential design.

## 14. Claim boundary

This protocol and analytic code do not establish that Symthaea outperforms SOTA, makes real-world scientific discoveries, or improves physical outcomes. No external benchmark has been run. Utility-aware experiment selection is an established research area; the proposed contribution is the testable integration of task-relative decision sufficiency with SDCB's evidence-gated belief transitions.