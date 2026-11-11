# Luminous Dynamics — evidence-gated intelligence benchmark crosswalk

**Version:** 0.1  
**Date:** 11 November 2026  
**Status:** research positioning and target proposal; no cross-benchmark score or superiority claim

This matrix is a research decision aid, not a leaderboard. The benchmarks below measure different constructs; numbers must not be compared as if they were a common metric. Report each outcome separately and preserve all denominators, uncertainty, model/harness revisions and compute budgets.

## External benchmark landscape

| Area | Primary comparator and reported scale/result | What it measures | What it does not establish |
|---|---|---|---|
| Active causal investigation | [ACED-Bench v2](https://arxiv.org/abs/2610.14381): ACED-Repair uses 540 test worlds / 54 selected evaluation worlds, 9 causal archetypes, 15 experiments and 32 turns. Published model acceptance spans 0–43%. | End-to-end experiment choice, evidence retention, causal repair, dose/combinations and commitment. | General scientific agency or production intervention safety. The 54 eval worlds are selected from the 540 test inventory, not an independent additional sample. |
| Active causal discovery | [ACDB](https://arxiv.org/abs/2609.31675): PC + greedy active orientation reports 42.7% directed F1 and SHD 4.79; raw active LLM results include 31.7% and 22.9%. Its random-DAG floor moves from 23.6% to 16.9% after a density probe. | Graph recovery under budgeted interventions, with skeleton, orientation and efficiency separated. | Causal repair utility or the validity of a single distribution-blind random baseline. |
| Sequential causal experiment design | [Causal-Discovery-Bench](https://arxiv.org/abs/2611.09934): 3,802 episodes, 13 models from 5 vendors, 40 metrics in 7 families; authors attribute 94.5% of evaluated errors to experiment design rather than inference. | Sequential experiment design, resource allocation, statistical power, conflicting evidence and stopping. | General scientific discovery or results outside its configurable SCM sandbox. |
| Data-driven causal graph repair | [Heuresis](https://arxiv.org/abs/2610.22846): anonymous simulated datasets across six mechanism families; paper reports GPT-6.1 Sol +14 adjacency-F1 points and Claude Opus 5.5 +7.4 on its hardest family over the best selected algorithm. | Separates algorithm routing from additional data-dependent edge discovery, with edge-level change provenance. | Independent reproduction yet; the paper says code/data will be released. Treat as a method comparator until artifacts are available. |
| Causal reasoning under realistic bias | [CausalGame](https://proceedings.mlr.press/v306/chen26l.html): 14 interactive game settings; 29 frontier models. | Experimental protocol design under hidden confounders, selection bias and noisy measurements. | Real-world causal identification or scientific repair competence outside its game tasks. |
| Full-cycle scientific insight rediscovery | [FIRE-Bench](https://proceedings.mlr.press/v306/wang26d.html), ICML 2026: agents must independently design and execute experiments and reproduce documented findings; strongest reported frontier agent remains below 50 F1 with high run-to-run variance. | End-to-end experiment design, execution and evidence-backed rediscovery. | Novel scientific discovery or external experimental validation beyond the selected reproducible findings. |
| Data-driven science workflow | [ScienceAgentBench](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f12b4df26344f3be803c06b555252efe-Abstract-Conference.html): 102 tasks from 44 peer-reviewed studies, expert-validated by 9 subject-matter experts; best reported agent solved 32.4% independently (34.3% with expert-provided knowledge). | Verified data-analysis programs, execution results and costs across real scientific tasks. | Autonomous experiment design or full scientific discovery; its output contract is a Python program. |
| Capability-oriented scientific analysis | [SDABench](https://arxiv.org/abs/2607.11079): 527 real-data and 6,000 synthetic instances, 6 capabilities and 5 scientific domains, 15 models. | Descriptive, exploratory, inferential, predictive, causal and mechanistic analysis; identifies assumption-selection and latent-process failures. | Real experimental science or sequential intervention competence; preprint/synthetic tasks must be audited before reuse. |
| Paired abstention competence | [AgentAbstain](https://arxiv.org/abs/2607.10059): 263 paired tasks, 42 MCP environments, 8 scenarios, 17 frontier models / 4 harnesses; leading paired accuracy 59.5%. | Both the should-act and should-abstain twin must be correct. | Enforcement by an independent authorization layer. |
| Timely stopping | [Agentic Abstention (NeurIPS 2026)](https://agentic-abstention.github.io/): 28,350 instructions across WebShop, Terminal-Bench 2.0 and interactive QA; reported CONVOLVE improvement 26.7%→57.4% AbsRec@1 on a held-out WebShop subset. | Whether an agent stops at the earliest justified point, rather than eventually abstaining after more actions. | Paired act/abstain competence or tool-effect authorization. Its evaluation coverage varies by experiment. |
| Policy-context authorization | [AuthGateBench](https://arxiv.org/abs/2610.16426): 1,400 policy-guided instances across five failure modes, with identical action/trajectory contrasts where request context flips allow/block. | Whether a gate reasons over explicit request/authority context rather than action appearance alone. | A deterministic external gate's enforcement integrity. |
| Realistic authorization overreach | [AuthorizationBench](https://arxiv.org/abs/2610.27548): 286 executable tasks and 6,864 trajectories across four models, two reasoning settings and three harnesses; 19.2% out-of-authorization behavior in its tested mix. | Whether agents exceed delegated scope, including without conventional harmful outcomes. | Deployed-population incident rates or deterministic gate performance. |
| Runtime governance layer | [AgentGovBench](https://github.com/agentic-control-plane/agentgovbench): 48 scenarios; the public 46/48 score is the benchmark maintainer's own product, ACP, and it is the only product scored so far. | Identity, policy, delegation, rate limits, audit, tenant isolation and failure behavior. | Independent vendor-neutral SOTA comparison. Use it as scenario taxonomy until third-party submissions exist. |
| Human/LLM deliberation | [DelibSim](https://openreview.net/pdf/83aaab07477b27ac43d15c5aba397e6d003ae242.pdf): 1,980 five-agent sessions, 11 model setups, 12 topics; AQuA 2.94 vs human reference 2.98, perspective diversity 6.5 vs 18.8. | Procedural discourse quality and deliberative-reason outcomes, distinct from each other. | Real human collective intelligence, legitimacy or population representativeness. |
| Human facilitation effects | [FAccT 2026 study](https://deepmind.google/research/publications/224297/): N=879; no significant consensus or participation-equity improvement, while selected charity allocations shifted up to 5.5 percentage points. | Real group outcomes and participant perceptions under facilitation. | That perceived inclusion or process satisfaction means power/equity improved. |
| Private reflection | [AI-guided self-reflection study](https://arxiv.org/abs/2610.37862): randomized N=1,173 across 391 groups; DQI effect d=0.31 vs no reflection, d=0.27 vs unassisted reflection. | Non-directive reflection before group discussion. | Live facilitation effects or generalization beyond the tested U.S. sample. |

## Proposed Luminous target ladder

These are **proposed research targets**, not achieved scores. Numeric targets become confirmatory only after baseline reproduction, split audit, preregistered hypotheses, power analysis and independent evaluation.

### A. Causal investigation — ACED-Repair

**Near-term smoke gate**
- Exact pinned upstream inventory and index hashes qualify.
- Zero candidate access to hidden SCM/oracle/grader data in the isolated test environment.
- Replay all official metrics without modifying their definitions.
- Log each predicted experiment value *before* observation and preserve every tool call, failure, timeout and incomplete case.

**Confirmatory target**
- Beat the strongest matched official agent by a preregistered margin on accepted repair rate and/or benefit recovered.
- Reduce intervention regret and experiments wasted on non-decisive evidence.
- Improve multi-cause, dose-window, synergy and subgroup-policy performance.
- No regression in the worst archetype or worst domain.
- Generate a new hidden-seed confirmatory split; do not use the 54-world selected subset as if it were independent of the 540-world parent inventory.

### B. Causal discovery — ACDB

**Stretch target:** exceed 50% directed F1 and SHD below 4.0 on the same ACDB ladder, while beating density/degree-matched nulls by a preregistered margin.

Required safeguards: skeleton F1 separate from directed F1; CPDAG/Markov-equivalence-aware scoring; precision and recall together; intervention budget and cost; exact small-world regret; uncertainty/abstention; held-out graph families, confounding, latent variables and distribution shifts. A high random baseline must trigger a benchmark audit, not a victory declaration.

### A2. Causal discovery — new comparator expansion

The October/November 2026 landscape makes it clear that causal discovery needs more than one score:
- **Sequential acquisition:** Causal-Discovery-Bench tests four escalating levels—from choosing one experiment through sequential exploration, resource-aware experimentation and conflicting-evidence resolution. In its 3,802-episode study, the authors attribute 94.5% of evaluated mistakes to experiment design rather than inference. This is the strongest argument for prospective EIG/EVSI traces and experiment-selection ablations.
- **Algorithm selection versus data use:** Heuresis distinguishes gains from routing to a better existing algorithm from gains due to additional data-dependent edge repair. Future comparisons should report the best fixed algorithm, best algorithm router and evidence-driven agent separately, and audit every edge added or removed against the observation that justified it. Its current reported advantages are preprint results; code/data availability must be rechecked before reproduction.
- **Bias-resistant causal judgment:** CausalGame adds hidden confounding, selection bias and noisy measurement under interactive protocol design. Use it as a robustness crosswalk, not an exact replacement for ACDB or ACED.
- **End-to-end scientific work:** FIRE-Bench evaluates rediscovery of documented findings using independently designed and executed experiments; ScienceAgentBench and SDABench cover separate real-data analysis capabilities. Do not merge them into a single “AI scientist” rank.

**New Luminous experimental ladder:** (1) correct inference on supplied curated evidence; (2) select one decision-relevant experiment; (3) sequentially choose and revise experiments under a finite budget; (4) detect conflicting evidence and stop/reconcile; (5) design and execute a complete rediscovery/repair trajectory; (6) verify the final conclusion against a hidden oracle and independently recorded artifacts. Report stage transitions and failure attribution, not only a terminal answer.

### B2a. Complementary information and adaptive sequential lookahead

A myopic experiment policy can stop even when a useful *sequence* exists: two individually decision-irrelevant probes may jointly resolve the action. Add a bounded exact policy-tree baseline for small discrete worlds. It should be compared against greedy EIG/EVSI and robust EVSI with identical priors, utilities, likelihood models, candidate probes, costs, risk, horizon and information access. Include synthetic complementary-partition controls where each first probe has zero one-step EVSI but the two-probe sequence has positive value. Report policy-tree size, branch coverage, expected probe count/cost, sequence regret and exact-vs-approximation gap as the horizon expands. This is bounded finite-state enumeration, not a general POMDP or a claim of global optimality.

### B2. Model-uncertain experiment selection

Nominal EVSI assumes a sufficiently specified prior, action-utility table and observation likelihood model. That is fragile when the agent's causal mechanism or measurement reliability is uncertain. Add a separate **robust EVSI** arm: evaluate each proposed experiment over a preregistered finite set of plausible decision models, then select using the worst-case net EVSI. Do not average the scenarios unless defensible probabilities are independently calibrated.

Compare:
- entropy-only EIG;
- nominal EVSI;
- risk-adjusted nominal EVSI;
- finite-set worst-case robust EVSI;
- random-valid and no-experiment controls.

Report the best / worst / mean-descriptive scenario value, minimum net EVSI, selection reversals, regret under each realized held-out model, stop rate and performance as the uncertainty set expands. The uncertainty set must be frozen before the trial and include plausible prior, likelihood and utility variation; a model cannot choose a convenient set after it sees the observations. This is a Luminous research extension, not an upstream ACED metric or reported SOTA result.

### B3. ACED-Query and statistical measurement validity

ACED-Query supplies three official agent architectures—Active, Active+Code and Modular—under a four-outer-turn protocol, with a separate guess-only baseline. Core and Guess tracks must be reported separately and payload identity must be controlled by SHA-256, not basename. Preserve the upstream evaluator and its LLM extraction step for official reproduction, while adding a deterministic-parser diagnostic lane and independent adjudication of a stratified sample.

The upstream Active agent displays categorical distribution shifts using fixed absolute thresholds. Symthaea's diagnostic implementation reports total-variation distance, per-category effect sizes and Wilson/Newcombe intervals, with an externally supplied familywise critical value. Compare fixed-threshold alerts against the preregistered interval/measurement-validity arm across low and high sample sizes, rare categories and multi-category outcomes. Keep distribution shift distinct from identified causal effect and preserve the intervention-assignment mechanism.

### C. Abstention and runtime stop

**Stretch target:** exceed 70% paired act/abstain accuracy on a regenerated, independently authored suite of at least 500 pairs, across multiple harnesses, while preserving actionable completion.

Also report AbsRec@1, premature-stop rate, post-trigger tool calls, time-to-justified-stop, failed-tool reconciliation and any effect attempted after the stop condition first became observable. Do not collapse paired accuracy and timely stopping into one number.

### D. Authorization and actual effect enforcement

Separate four rates:
1. model unauthorized-effect requests;
2. external-gate vetoes;
3. unauthorized adapter executions;
4. independently witnessed state changes.

**Bounded-environment target:** zero unauthorized adapter executions in at least 10,000 diverse negative attempts, with at least 10,000 authorized positive controls and false-denial below 1%; independent replay of every receipt. If zero violations are seen in 10,000 independent trials, the simple rule-of-three upper 95% bound is approximately 0.03%; clustering and scenario dependence require wider, cluster-aware intervals.

Never allow a good gate score to hide a model that constantly requests unauthorized effects, or a good model score to substitute for gate enforcement. Add identity substitution, confused deputy, delegation depth, purpose drift, target substitution, stale pre-state, replay, shared-limit burst, gate outage and audit-loss cases.

### E. Deliberation and collective sovereignty

Do not optimize consensus as the primary objective. Keep process quality, reasoning quality, perspective diversity, minority-position retention, participant influence, actual allocation/outcome changes, trust and contestability separate.

**Research target:** improve DQI/AQuA and evidence-warranted revision while retaining diversity within the relevant human-reference distribution; keep high-salience minority positions traceable and appealable; detect unwarranted directional steering against randomized controls. Require a sealed-monologue control to separate peer-interaction effects from private reflection or stochastic stance drift. Do not claim persona agents represent a population without independent survey calibration.

Human trials require applicable ethics review, informed consent, privacy controls, preregistration and independent human adjudication.

## Industry positioning and actual differentiation

- **Model vendors:** generally optimize task completion and model-level refusal. Luminous should show whether the action was within delegated purpose and scope, whether an external gate enforced that decision, and what the independent observer actually witnessed.
- **Agent runtime/governance vendors:** increasingly expose identity, policy, delegation and audit controls. Luminous should compete on reproducible receipts, negative controls, exact-head verification, cross-runtime conformance and independent effect evidence—not marketing feature counts or self-scored leaderboards.
- **Causal/agent research:** often reports final answer accuracy, graph F1 or acceptance. Luminous should expose the full investigation trajectory: what was believed, what observation could change the action, which experiment was selected, what was learned, why it stopped and what the outcome was.
- **Deliberation systems:** often emphasize consensus, summary quality or user satisfaction. Luminous should preserve plurality, individual voice, counterargument provenance, minority views, appeal and measured outcome influence.

The OECD's 2026 survey of 36 member governments reports only 11/36 with post-deployment AI audits, 10/36 measuring AI-use impacts and 8/36 with citizen feedback/complaint mechanisms. This is evidence of a governance implementation gap in that OECD sample, not a claim about all governments. It motivates outcome-linked audits and participant feedback as core infrastructure rather than optional dashboards: https://www.oecd.org/en/publications/digital-government-outlook_0496b2bc-en/full-report/adopting-and-governing-ai-in-government_7ef312a9.html

## Rules for honest SOTA comparison

1. Pin model snapshot, harness, prompt, tools, decoding settings, budgets, date and dataset SHA.
2. Run matched baselines and include classical, simple heuristic, always-act, always-abstain and deterministic-oracle controls where applicable.
3. Keep tasks/pairs atomic across splits; use held-out task templates, mechanisms, domains and generated seeds.
4. Report point estimates, confidence intervals, complete denominators, missing/timeout/invalid cases, worst-group outcomes, cost and latency.
5. Reproduce upstream metrics unchanged; Luminous-specific metrics are additional columns.
6. Treat vendor self-evaluations as design references until independent submissions exist.
7. Never combine causal skill, abstention, authorization enforcement and deliberative legitimacy into one universal score.

**Claim ceiling:** this matrix proposes comparisons and targets. It does not establish any Luminous SOTA result, model trial, human-study outcome or deployment qualification.
