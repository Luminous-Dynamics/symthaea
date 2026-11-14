# AOBench — Experimental Design and Cross-Family Generalization

**Status:** Research protocol and benchmark-design proposal; no AOBench score claimed  
**Date:** 2026-11-14  
**Tracking:** [AOBENCH-011 #12173](https://github.com/Luminous-Dynamics/symthaea/issues/12173)

## 1. Research update: the bottleneck is often evidence acquisition

AOBench's experimental-identifiability and causal-diagnosis tracks should explicitly measure the process that acquires evidence, not just reasoning over a curated evidence set.

Recent 2026 benchmarks make this distinction concrete:

| Benchmark | Published scope/result | Lesson for AOBench |
|---|---|---|
| Causal-Discovery-Bench | 3,802 episodes, 13 models, 40 metrics in seven families; authors attribute 94.5% of recorded errors to experimental design | Experiment selection, controls, sample adequacy and stopping need separate scores |
| ACED-Bench | ACED-CORE: 180 tasks; ACED-QUAL: 150 questions; hidden probabilistic causal worlds | Separate readable task description from hidden causal ground truth; measure evidence acquisition and answer commitment |
| ACDB | Budgeted intervention ladder; PC + greedy active orientation reports directed F1 42.7% / SHD 4.79, ahead of raw LLM intervention baselines in its current calibrated version | Classical causal methods are mandatory matched baselines; report precision/recall and calibration floors |
| CausalArena | Synthetic, semantic-operational, formula-grounded and public real-world SCM families; rankings shift across families and protocols | Use mechanism-family holdouts and external validity, not random-seed splits alone |
| TCD-Arena | ICLR 2026; around 30 million attempts over 33 assumption violations | Stress assumption violations systematically; report robustness profiles, not only idealized synthetic averages |
| Traverse | 2,518 long-horizon trajectories; 6,967 mistakes across 78 types; strongest evaluated frontier judges find the first error in fewer than one-third of runs | Locate first causal failure, measure recovery and irreversible intermediate harm |
| AgentLAB | Five long-horizon attack families, 28 environments, 644 security cases | Add adaptive attacks to experiment/evidence pipelines, including objective drift and memory poisoning |
| MANTA | 74.0 mean success across five task benchmarks, with bounded inference-time topology repair | Compare adaptive organization under matched budgets, then test properties beyond final task success |
| ASIWorld | Persistent survival world; couples world state and execution records with outcomes, progression and cost | Add durable consequences, execution overhead, resource depletion and multi-epoch recovery |

These results are complementary, not directly rankable. The benchmarks differ in task, model, budget, environment, oracle and metric. AOBench must not treat their headline scores as one common leaderboard.

The current official comparator protocols are especially instructive:

- **MANTA:** its public table reports 74.0 ± 1.8 across five suites, using Gemma 4 31B, 30 tasks per suite and three independent runs. It scores 76.7 on BrowseComp, 82.2 on StableToolBench, 76.7 on PlanCraft, 43.3 on WorkBench and 91.1 on MATH. Its architecture changes roles, links, execution order, information visibility and validation paths with bounded repair; controller code—not the agent—decides when the loop stops. Its online playbook excludes the benchmark success verdict from learning memory. AOBench should borrow that separation between adaptation signals and the authority used to score correctness.
- **ASIWorld:** the current official page lists 15 frontier configurations under a common Claude Code harness, with a separate compact Qwen3.5-27B track, over a seven-hour persistent sandbox spanning four gameplay domains and six capability dimensions. Its weighted 0–100 score is 20% survival, 35% combat, 15% building and 30% crafting. The current top row is claude-fable-5-2 at 79.1; most frontier rows are single seven-hour runs, while DeepSeek V4.1 Flash reports five runs (33.8 ± 9.3) and the separate Qwen3.5-27B compact track reports three runs (10.2 ± 1.7). Its world state, asynchronous execution, progression and durable action records are useful comparators for AOBench's repeated-epoch and recovery-cost scenarios. The 79.1 headline is not directly rankable against MANTA's 74.0: tasks, horizons, run counts and score construction differ.

The comparison protocol should record model/version, reasoning effort, environment commit, harness, trial count, time/compute/action budgets, scoring weights, oracle, and uncertainty for every external baseline. If these differ, report a descriptive comparison rather than a numerical ranking.

One result deserves special attention: Causal-Discovery-Bench reports that increasing one model's resource budget by 125× moved structure-identification success from 17% to 78%, but controlled experiments were absent below the top budget and appeared in only one of 15 episodes at the top budget. The same paper reports 14/15 correct networks when a mid-tier model received adequate controlled data, versus 9/15 when it received the same data volume with inadequate variable control. Across 3,779 scored episodes, 94.5% of attributed errors occurred because agents concluded before collecting adequate evidence. More budget can therefore mask poor experimental design by enabling broad candidate exploration. AOBench must report controlled-intervention rate, direct-treatment/variable-control adequacy, information gain per cost, achieved power and oracle regret—not just final discovery accuracy.

## 2. Paradigm shift: from inference quality to evidence-acquisition competence

An organizationally intelligent system must be able to:

1. maintain competing causal hypotheses;
2. identify which observation or intervention best distinguishes them;
3. predict outcomes and uncertainty before acting;
4. check target, delivery, isolation and observer integrity;
5. decide sample size and statistical power against a practical-effect threshold;
6. revise hypotheses after sequential evidence;
7. resolve conflicting observations or declare them unresolved;
8. stop when the frozen evidence contract is met;
9. refuse to make an organizational change when identification is inadequate.

A correct answer from a curated dataset measures inference. It does not establish that the system can acquire the right data.

## 3. Task ladder

### E1 — Single discriminating experiment

Provide two or more candidate organizational mechanisms and a finite experiment budget. Score whether the chosen probe actually separates them, rather than whether it merely produces a plausible observation.

### E2 — Sequential discovery

Reveal observations over time. The agent must maintain a versioned hypothesis set, choose subsequent experiments adaptively and preserve evidence lineage.

### E3 — Resource and power allocation

Interventions differ in cost, risk, delay and statistical power. Score utility per cost, expected information gain, power, uncertainty and regret relative to an oracle experiment policy.

### E4 — Conflicting evidence and common-cause failure

Include duplicated observers, missingness, common-cause dependencies, carryover, interference, nonstationarity and oracle disagreement. The system must distinguish genuine mechanism disagreement from measurement failure.

### E5 — Stop / continue / abstain

Require an explicit stopping decision. Penalize premature commitment, endless experiments after sufficient evidence, and false equivalence from non-significance. Cases that are not identifiable under the available probe family must remain unresolved.

### E6 — Organizational intervention

Only after sufficient identification may the system propose a repair or reconfiguration. The proposal must pass structural admission, viability, authority, consent, semantic continuity and independently qualified intervention delivery.

## 4. Required baselines

All baselines receive matched task instances, seeds, compute, observation budget, intervention budget, stopping budget and environment:

- Random intervention selection.
- Fixed schedule / one-factor-at-a-time.
- Greedy information gain and active-orientation heuristics.
- Classical PC, FCI and GES where assumptions permit.
- Suitable constraint-based, score-based and functional causal discovery baselines for each SCM family.
- LLM-only sequential experimenter.
- LLM with basic statistical tools.
- MANTA-style topology-adaptive agent on organizational tasks.
- Oracle experiment policy as an upper bound.
- AOBench-qualified system with frozen hypotheses, evidence ledger, delivery/isolation checks and explicit stop rule.

The classical baselines are mandatory. ACDB's current results show why comparing only against other LLM agents would overstate progress.

## 5. Metrics

Do not merge evidence acquisition, inference and downstream intervention into one score.

**Experiment design**
- intervention utility per unit cost;
- expected and realized information gain;
- correct target selection;
- control-group adequacy;
- intervention isolation and delivery validity;
- hypothesis-set contraction;
- oracle policy regret.

**Statistical adequacy**
- empirical power and achieved sample size;
- confidence interval coverage and width;
- practical-effect/equivalence margin handling;
- multiplicity and optional-stopping control;
- missingness, clustering and carryover handling.

**Inference**
- precision, recall, F1 and SHD where graph recovery is meaningful;
- identifiability coverage;
- false causal claim rate;
- calibrated confidence and abstention quality.

**Sequential behavior**
- evidence retention and event/receipt completeness;
- premature-commitment rate;
- unnecessary-continuation rate;
- conflicting-evidence resolution;
- first-failure localization and recovery.

**Organizational action**
- intervention success conditional on qualified delivery;
- structural mutation budget;
- capability and governance continuity;
- false continuity acceptance;
- irreversible side effects;
- recovery time, repeated-epoch stability and resource cost.

## 6. Split and generalization protocol

AOBench should use mechanism-family-disjoint splits:

- **Development:** public generator families and open task templates.
- **Validation:** new seeds and some unseen parameterizations, but no confirmatory claim.
- **Sealed test:** complete held-out mechanisms, graph motifs, effect classes and intervention pathways.
- **Transfer:** semantic/operational SCMs and real-world cases with independent ground-truth adjudication.
- **Assumption stress:** systematically add latent confounding, selection bias, nonlinearity, nonstationarity, measurement error, interference, feedback, delayed effects, policy-induced distribution shift and common-cause failure.

Random seeds alone are not adequate holdouts when the mechanism generator is known. Record generator version, mechanism-family identity, model exposure declarations and potential benchmark contamination. Report in-family, cross-family and environment-transfer results separately.

## 7. How AOBench should outperform current benchmark practice

The advantage should come from combining properties that current benchmarks often measure separately:

- hidden causal ground truth and readable task interfaces (ACED-Bench);
- resource-bounded sequential discovery (Causal-Discovery-Bench and ACDB);
- mechanism-family diversity and leakage controls (CausalArena);
- robustness to assumption violations (TCD-Arena);
- first-failure localization and recovery (Traverse);
- adaptive organizational topology (MANTA);
- long-horizon adversarial integrity (AgentLAB);
- persistent consequences and cost (ASIWorld);
- typed authority/consent/effect/provenance/substrate continuity (AOBENCH-010).

This does not imply that AOBench has already demonstrated superiority. It defines a falsifiable experimental bridge through which superiority can be measured.

## 8. Proposed AOBench qualification targets (not achieved results)

These are preregistration targets for pilot calibration and confirmatory evaluation. They are not current Symthaea scores or claims of superiority.

| Qualification layer | Proposed target |
|---|---|
| Deterministic contract fixtures | 100% of registered positive, negative, invalid-input and budget-exhaustion fixtures behave as specified |
| Authority / consent / irreversible effects | Zero accepted false-continuity cases in the registered adversarial controls |
| Intervention evidence | Zero hypothesis updates from rejected or unavailable delivery, isolation or observation evidence |
| Sequential decision | At least 95% correct stop / continue / defer decisions on sealed deterministic tasks; zero budget violations |
| Model inadequacy | At least 95% recall on held-out out-of-class mechanisms, with at most 1% false expansion on in-class controls |
| Refinement validation | Zero accepted refinements with a failed or missing sealed prediction; exact 100% coverage required for deterministic acceptance |
| Experiment efficiency | Report information gain per cost and regret against a safe/affordable oracle; target median normalized regret below 5% after calibration |
| Cross-family robustness | Report every mechanism family and assumption violation separately; aim for no more than a 10-point absolute degradation from validation to held-out families, without hiding worst-family failure |
| Causal graph discovery | Run matched ACDB and CausalArena protocols; target at least +10 absolute directed-F1 points over the strongest non-oracle baseline on the same task ladder, while also reporting precision, recall and SHD |
| Real-world transfer | No universal numeric target until outcome ground truth and intervention validity can be independently adjudicated; publish failures and abstentions |

AOBench must not tune thresholds on its confirmatory suite. Pilot data select and freeze the thresholds; sealed family-disjoint evaluation measures them. A failed safety or evidence gate blocks the task regardless of the aggregate task score.

## 9. How we compare fairly with current SOTA

- **ACDB:** its current linear-Gaussian, perfect-hard-intervention ladder reports PC plus greedy active orientation as the strongest non-oracle method at directed F1 42.7% / SHD 4.79. This is the minimum classical comparator, not an LLM-only baseline. AOBench must use paired instances and report skeleton, orientation, precision/recall, intervention utility and abstention separately.
- **Causal-Discovery-Bench:** 3,802 episodes are described in the abstract, with 3,779 scored episodes in the detailed analysis. The paper attributes 94.5% of errors to premature conclusions before sufficient evidence; it also shows that a 125× resource increase can raise structure success from 17% to 78% while controlled experimentation remains rare. AOBench must separate raw budget from direct-treatment isolation and evidence quality.
- **ACED-Bench:** 180 ACED-CORE tasks plus 150 qualification questions demonstrate the importance of hidden formal worlds, readable tasks, evidence ledgers and stop rules. AOBench should preserve the separation between public task surface and evaluator-only truth.
- **CausalArena:** ranking shifts across synthetic, semantic-operational, formula-grounded and public real-world SCM families. AOBench should report per-family outcomes and generator exposure; a single pooled average is insufficient.
- **TCD-Arena:** 33 time-series assumption violations and large-scale experiments motivate severity ladders and compound violations. AOBench should report a robustness profile rather than one global robustness score.
- **MANTA:** its official evaluation reports 74.0 ± 1.8 across five benchmarks, with Gemma 4 31B, 30 tasks per benchmark and three independent runs. Its controller makes bounded topology repairs and keeps the benchmark verdict out of online playbook learning. AOBench should compare organizational adaptation under the same task/resource budget, then separately score governance, effect and evidence continuity.
- **ASIWorld:** its current official page lists 15 frontier configurations in a seven-hour persistent world, with a weighted 20/35/15/30 score. The top listed row is 79.1, but most frontier configurations are single runs. AOBench can borrow persistent consequences and multi-epoch recovery, but must not rank against its score.

No cross-benchmark headline number is an apples-to-apples leaderboard comparison. Every report must include model/version, task generator and commit, harness version, number of runs, task counts, compute/time/action budgets, oracle, scoring weights and uncertainty.

## 8. Claim boundary

A synthetic causal-discovery result does not establish scientific discovery competence in uncontrolled physical environments. Intervention conclusions require independently qualified target, delivery, isolation, observer and oracle evidence. A non-significant difference is not an equivalence proof. A system that cannot identify the mechanism under the available experiments must abstain.

## References

- [Causal-Discovery-Bench](https://arxiv.org/abs/2611.09934)
- [ACED-Bench](https://arxiv.org/abs/2610.14381)
- [ACDB](https://arxiv.org/abs/2609.31675)
- [CausalArena](https://arxiv.org/abs/2609.11897)
- [TCD-Arena (ICLR 2026)](https://tcd-arena.github.io/)
- [Traverse](https://arxiv.org/abs/2609.17930)
- [AgentLAB (ICML 2026)](https://proceedings.mlr.press/v306/jiang26as.html)
- [MANTA](https://mao-code.github.io/MANTA/)
- [ASIWorld](https://asigameworld.github.io/)
