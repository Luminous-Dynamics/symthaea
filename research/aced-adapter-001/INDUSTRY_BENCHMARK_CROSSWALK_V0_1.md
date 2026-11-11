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
