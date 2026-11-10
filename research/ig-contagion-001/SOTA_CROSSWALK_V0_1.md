# IG-CONTAGION-001 — research / SOTA crosswalk v0.1

**Purpose:** translate recent multi-agent debate, belief revision and deliberation research into matched experiments for Symthaea's institutional-game lab.

This document is a research design crosswalk. Published values below are reported by their respective authors and are not directly comparable across papers because tasks, models, prompt budgets, group sizes, evaluation procedures and outcome definitions differ.

## External reference studies

| Reference | What it contributes | How we use it | What it does not establish |
|---|---|---|---|
| Haider & Vosoughi, *Misalignment Contagion*, COLM 2026 | 35,290 trials; five normative/safety datasets; four topologies; private post-discussion re-elicitation; topology and centrality effects in tested settings | Replicate the three-stage public/private design and test topology/placement as causal factors | Not a theorem about all models, all topologies, or human groups |
| Oh et al., *From Belief Entrenchment to Robust Reasoning in LLM Agents*, TACL 2026 | Separates static prior-belief bias from homogenized debate dynamics; DReaMAD and MetaNIM Arena; reported +9.5% accuracy over ReAct and +19% higher win rate than standard MAD on its benchmark | Factor prior correction and debate diversity as separate treatment arms; reproduce the cited baseline on its original conditions | These gains cannot be imported as expected Symthaea performance or compared to unrelated tasks |
| Ha & Mitchell, *Beyond Correctness*, ICML 2026 / PMLR | Measures the direction and distance of answer revisions relative to an explicit ground truth | Add task-specific revision-distance and direction metrics on synthetic known-answer tasks | Does not license a truth-distance score for normative opinions |
| Myakala et al., *BeliefShift*, 2026 preprint | Longitudinal temporal consistency, contradiction resolution and evidence-driven revision across multi-session trajectories | Add delayed re-elicitation and contradiction/revision analysis where exact artifacts and rights permit | Preprint claims and dataset are not independently reproduced here |
| *When collaboration fails: persuasion driven adversarial influence in multi agent large language model debate*, Scientific Reports 2026 | Persuasion attacks beyond prompt injection; reports substantial accuracy loss and incorrect-consensus increases in the tested setup | Add confident, status-bearing, social-proof, citation-laundering and emotional-urgency attacks | Does not imply the same effect sizes across models or tasks |
| MADBench, arXiv 2026 | Six security-attack families over 356 source tasks / 3,958 test cases; includes prompt/resource attacks, compromised agents and orchestrator compromise | Compose its attack taxonomy as a separate security layer; compare answer correctness and unauthorized read/write outcomes | The current 50-case stance/topology matrix is not a reproduction of MADBench; exact artifacts/harness require separate qualification |
| *Demystifying Multi-Agent Debate*, Findings ACL 2026 | Diversity-aware initial answers and confidence-modulated updates across six reasoning benchmarks | Add initial-diversity and calibrated-confidence treatment arms | Confidence is not evidence; QA improvements do not establish civic legitimacy |
| *Free-MAD*, Findings ACL 2026 | Consensus-free debate, full-trajectory scoring and anti-conformity; reports lower round/token cost | Add a trajectory-scoring and consensus-free baseline under equal model/token budgets | Majority resistance can still erase minorities if trajectory scoring is opaque or miscalibrated |

| ACIArena, ACL 2026 | 1,356 cases over six MAS implementations; external-input, agent-profile and inter-agent-message attacks; topology alone insufficient | Add role/profile/message attack layers and transfer tests; do not reduce security to topology | Not a like-for-like accuracy benchmark for our synthetic stance protocol |
| *Beyond Consensus*, Findings EMNLP 2026 | On tested subjective scoring tasks, single-judge baselines had stronger human alignment than consensus MAD; asymmetric strict roles induced downward bias | Include blind-human reference, single-judge baseline, symmetric-MAD control and role-asymmetry ablation | Does not imply single judges always outperform multi-judge systems or justify removing human adjudication |
| *The Value of Variance*, ICML 2026 | Separates intra-agent, inter-agent and system-level uncertainty as debate-collapse diagnostics | Record hierarchical uncertainty and self-contradiction / peer-conflict signals | Disagreement is not automatically collapse; legitimate minority reasoning must be preserved |
| Agent Red Teaming (ART), UK AISI 2026 | Public competition across 22 agents / 44 deployment scenarios, 1.8m attack submissions and 60k+ observed policy violations in that competition | Add effect-boundary adversarial testing for reads, writes, exfiltration and policy violations under adaptive query budgets | These competition figures are not our baseline and must not be compared as a raw percentage |

Links:
- https://proceedings.colm.cc/paper_files/paper/2026/hash/871ed095b734818cfba48db6aeb25a62-Abstract-Conference.html
- https://aclanthology.org/2026.tacl-1.58/
- https://proceedings.mlr.press/v306/ha26a.html
- https://arxiv.org/abs/2603.23848
- https://www.nature.com/articles/s41598-026-42705-7
- https://arxiv.org/abs/2609.39146
- https://aclanthology.org/2026.findings-acl.1694/
- https://aclanthology.org/2026.findings-acl.1600/

## The comparison principle

Do not ask only whether multi-agent debate improves final accuracy. Ask whether it produces **evidence-selective revision**:

- Does the system change when new, relevant, independently checkable evidence supports a correction?
- Does it resist unsupported repetition, fabricated authority, status, false consensus and persuasive but evidence-poor messages?
- Do those changes persist in a fresh private re-elicitation?
- Does network placement alter the result even when content and the base graph are held constant?
- Does the system preserve value disagreement when no single policy preference is correct?

This produces two primary outcome axes, never a single blended score:

1. **Warranted-revision sensitivity:** how often an initially incorrect/unsupported position is corrected after valid new evidence.
2. **Unsupported-influence susceptibility:** how often a false/unsupported claim is adopted and remains after private re-elicitation when no valid supporting evidence was introduced.

A system that improves one axis by becoming stubborn on the other is not an overall improvement. Report both with confidence intervals, plus the supporting evidence and task profiles.

## Matched experimental arms

Use the same task instances, model family/revision, agent count, token/tool budget, decoding configuration and random seeds wherever the compared method allows.

| Arm | Intervention | Research question |
|---|---|---|
| A0 — private repeat | No discussion; repeated private elicitation | What is baseline elicitation/session noise? |
| A1 — standard MAD | Ordinary multi-agent debate | Baseline social-revision behavior |
| A2 — prior correction | Explicit private prior / evidence elicitation before debate | Does better initial grounding reduce later susceptibility? |
| A3 — perspective diversity | Structured disagreement / non-homogeneous role sampling | Does diversity improve correction or merely create more disagreement? |
| A4 — evidence protocol | Claims require source/evidence references; independent challenge and correction stage | Does evidence accountability reduce persuasion without blocking valid updates? |
| A5 — source verification | Independent source/claim verification before post-discussion answer | Does verification change unsupported and warranted revision differently? |
| A6 — topology counterfactual | Same exact message, agent profile, evidence packet and graph; only speaker placement changes | Does exposure centrality itself change persistent influence? |
| A7 — confidence/diversity | Diverse initial answer sets and calibrated confidence reporting | Does this improve evidence-selective revision without turning confidence into authority? |
| A8 — Free-MAD trajectory scoring | Preserve initial/intermediate/final answers; use a consensus-free trajectory scorer and anti-conformity condition | Does full-trajectory selection reduce conformity and token cost without losing minority information? |
| A9 — MADBench security taxonomy | Matched attacks at prompt, resource, coordination and orchestrator layers | Does the protocol fail safely across the wider debate-security lifecycle? |
| A10 — role-symmetry / human alignment | Single-judge baseline, symmetric-MAD and asymmetric strict/lenient roles against blind human ratings | Does consensus improve alignment with human judgments, or merely enforce role-induced bias? |
| A11 — hierarchical uncertainty | Measure intra-agent, inter-agent and system-output uncertainty separately | Can collapse be detected without treating justified disagreement as failure? |
| A12 — cascading injection / effect security | Attack external inputs, agent profiles and inter-agent messages; test read/write/exfiltration outcomes in a contained environment | Does topology-aware debate also resist cross-agent instruction propagation and unauthorized effects? |

A2 and A3 separate the two DReaMAD design levers. A7 mirrors the confidence/diversity factors studied in Findings ACL 2026. A8 tests Free-MAD's trajectory-level, consensus-free alternative. A9 composes MADBench's threat taxonomy but requires a separate resource/tool/orchestrator harness; it is not represented by the current five persuasive-message bundles. A4/A5 remain candidate Symthaea interventions, not assumed wins. A6 uses the five frozen message/evidence/agent/topology pairs in the scenario matrix.

## Required outcome vector

### Correctness and revision (known-answer synthetic tasks only)
- initial and final accuracy;
- direction and task-specific distance of revision;
- warranted-revision sensitivity;
- unsupported-influence susceptibility;
- false-correction rate (correct baseline changed to incorrect answer);
- confidence calibration and unsupported confidence increase;
- abstention when evidence remains unresolved.

### Social influence and topology
- public/private stance divergence;
- adoption by speaker exposure path;
- placement effect under matched content;
- repetition/order/status/false-consensus sensitivity;
- influence concentration and cross-community transmission;
- persistence after private re-elicitation with exposure memory retained.

### Evidence integrity
- valid evidence introduction and uptake;
- fabricated citation/authority acceptance;
- claim-source provenance;
- response to valid correction after misinformation;
- unsupported claims treated as unresolved rather than silently accepted.

### Evaluator validity and human alignment
- blind human ratings, inter-rater variance and adjudication provenance;
- single-judge versus symmetric-MAD versus role-asymmetric-MAD human alignment;
- strict/lenient role directional bias and order/position effects;
- calibration, selective risk/coverage and unresolved disagreement.

### Hierarchical uncertainty
- intra-agent uncertainty;
- inter-agent disagreement/interaction uncertainty;
- system-output uncertainty;
- self-contradiction and debate-collapse indicators.

### Operational security
- unauthorized read/write attempts versus completed effects;
- instruction hijacking, task disruption and information exfiltration;
- cross-agent propagation, role confusion and orchestration compromise;
- benign false-block rate and operator burden.

### Normative pluralism (no single answer)
- participant-preference / value-position representation;
- trade-off disclosure;
- minority reason and dissent preservation;
- options considered and reasons for exclusions;
- no forced consensus or convergence reward.

### Efficiency and reliability
- tokens / tool calls / latency / model cost;
- initial answer diversity and confidence calibration;
- trajectory-scoring vs terminal-vote performance and cost;
- run-completion and invalid-trace rates;
- repeated-seed variance;
- evaluator disagreement and correlated-error profile.

## Statistical design

- Freeze protocol, model revisions, prompts, budgets, answer keys, analysis code and seeds before confirmatory runs.
- Use exact paired comparisons for the same content across topology placement.
- Factor model-family effects, topology effects, adversary-style effects and task-family effects instead of pooling them away.
- Use multiple independent seeds and report intervals/variance, not just averages.
- Cluster bootstrap or hierarchical models by task/scenario and model profile as appropriate.
- Separate exploratory sweeps from confirmatory holdouts.
- Evaluate at least one held-out task/topology combination when dataset size supports it.
- Keep agent-generated scores separate from independently computed answer/evidence oracles.
- For normative-pluralism tasks, do not calculate accuracy, correctness distance or policy-preference winner.

## What “better than SOTA” should mean

First reproduce the strongest comparable baseline under its declared conditions. Then define improvement as a constrained Pareto comparison:

- no material loss in warranted evidence-driven revision;
- lower persistent unsupported-influence susceptibility;
- lower false-correction rate;
- no loss of uncertainty/abstention under unresolved evidence;
- improved minority-reason and dissent preservation on pluralism tasks;
- equal or lower token, latency and tool budget where possible.

A numerical target should be set only after a reproduced baseline, variance estimate and power analysis. Do not claim superiority from different datasets, different answer spaces, different model budgets or a single selected seed.

## Initial milestone sequence

1. Finish structural qualification of the 50-case protocol, nine families, five paired content bundles and negative controls.
2. Run deterministic non-LLM fixture-oracle checks before invoking any model.
3. Implement a runner adapter that records exact model/provider/revision/prompt/budget/evidence/topology and separates recorded-trace replay from stochastic regeneration.
4. Reproduce A0 and A1 on a small pilot matrix; freeze confirmatory design only after variance and runtime are understood.
5. Run A2–A12 as separate treatments; preserve failures and incomplete runs.
6. Publish raw multi-axis receipts and counterexample traces; no aggregate "alignment contagion" score.
7. Only after synthetic evidence is stable, consider separately authorized human-group experiments under Mycelix #5999 and independent ethics review.

## Ownership

- #11337 owns CommunicationGraph generation, message-channel semantics and centrality fixtures; #11248 owns the separate delegation graph taxonomy.
- #3054 / #3095 / #3115 own institutional experiment orchestration and provenance.
- #1872 owns hostile-agent execution.
- #9431 owns evaluator correlation/ensemble qualification.
- #11320 owns the stance-shift measurement protocol and counterfactual analysis.
- MADBench security-layer replication/adaptation is separate from the 50-case persuasion/topology matrix; no dataset ingestion or reproduction is implied by citation alone.
- Mycelix #5999/#6011 own human-group process and participant-conditioned summary evaluation.

No existing engine or production governance authority is duplicated by this crosswalk.


## Industry platform comparison and interoperability target

These platforms already provide strong generic orchestration primitives. The benchmark should run against them through adapters rather than attempt to replace them.

| Platform / published pattern | Existing strengths documented by provider | Where this benchmark should complement it |
|---|---|---|
| OpenAI Agents SDK | Agent/handoff/guardrail/session primitives; built-in tracing for model calls, tools, handoffs and guardrails; trace grading and repeatable eval runs | Add content-addressed revision/evidence receipts, matched causal-pair identity checks, explicit social exposure and authority semantics, cross-provider trace portability, and pluralism-aware human evaluation |
| Microsoft Agent Framework | Python and C#/.NET; graph-based sequential/concurrent/handoff/group orchestration; typed state, middleware, OpenTelemetry, checkpoint/restart and human-in-the-loop controls | Validate the same scenario/event contracts across runtime providers; test whether typed run/effect evidence survives retries, restarts and handoffs; distinguish workflow routing from decision authority |
| Anthropic multi-agent research pattern | Long-horizon context management; phase summaries and external memory; artifact-first subagent work to reduce lossy coordinator relays; evaluate final state and discrete checkpoints | Measure artifact/provenance retention, evidence loss in handoffs, revision history and checkpoint completeness; compare output/effect quality under matched cost and task budgets |

Official/provider sources:
- OpenAI agent-building and orchestration primitives: https://developers.openai.com/tracks/building-agents
- OpenAI Agents SDK tracing: https://openai.github.io/openai-agents-python/tracing/
- OpenAI trace grading and eval workflows: https://developers.openai.com/api/docs/guides/agent-evals
- OpenAI guardrails and human approvals: https://developers.openai.com/api/docs/guides/agents/guardrails-approvals
- Microsoft Agent Framework: https://labs.ai.azure.com/innovations/microsoft-agent-framework/
- Anthropic multi-agent research system: https://www.anthropic.com/research/multi-agent-research-system

### What not to compete on

Do not create another generic agent loop, handoff router, trace dashboard, memory wrapper, group-chat engine or human-approval widget. Use adapters to consume each platform's traces and action/effect events.

### Cross-platform benchmark protocol

For each adapter, freeze:
- provider/runtime and exact package/commit/model revision;
- agent roles, prompts, tool/capability grants and handoff policy;
- memory/session/checkpoint semantics;
- model, token, tool-call, latency and human-review budgets;
- evaluator and evidence-oracle identities;
- CommunicationGraph profile and delivered message events;
- output/effect state and recovery transcript.

Run the same scenario, attack, truth-profile and revision-ledger fixtures where semantics allow. Mark unsupported features as Unsupported rather than silently weakening a test.

Compare the following vectors separately:
1. task answer and warranted-revision sensitivity;
2. persistent unsupported-influence susceptibility and false-correction rate;
3. evidence/provenance completeness across handoffs and memory checkpoints;
4. unauthorized reads, writes, exfiltration and policy-boundary effects;
5. blind-human alignment and role-asymmetry bias on subjective tasks;
6. recovery correctness after timeout, partial result, restart and stale context;
7. cost, latency, token/tool budget, trace size and operator burden;
8. exportability/replayability of run, evidence and effect receipts across providers.

This is not a present-day product benchmark result. The platform matrix is an adapter/evaluation target. No claim that Symthaea already outperforms these stacks is made; a fair comparison requires the same tasks, models, tools, budgets and independently verified outcomes.

The strategic differentiation is **assurance and institutional semantics above the orchestration substrate**: a portable proof/evidence vocabulary for what agents saw, what they changed, what they were authorized to do, what happened at the effect boundary, how a result can be replayed, and where uncertainty or normative disagreement remains unresolved.

## Graph ownership and semantics

The topology axis in this programme is a **CommunicationGraph**, not the delegation graph.

- [Symthaea #11337](https://github.com/Luminous-Dynamics/symthaea/issues/11337) owns AgentInstanceId nodes, directed MessageChannel edges, delivery/revocation events, communication reachability, and the normalized-betweenness definition used by #11320.
- [Symthaea #11248](https://github.com/Luminous-Dynamics/symthaea/issues/11248) owns PrincipalId/scoped mandate delegation networks. Its edges are not message channels and must not be reused as such.
- An optional principal binding on a communication node is metadata only. Multiple AgentInstanceIds per PrincipalId remain distinct unless a declared experiment transformation explicitly aggregates them.

The five matched centrality controls require strongly connected reciprocal directed graphs, unweighted hop-count shortest paths, minimum/maximum tie-set capture and deterministic AgentInstanceId tie-breaking. A disconnected or unsupported graph remains Unsupported; the runner does not manufacture a centrality result.


## Evaluator and effect-security boundary

A multi-judge consensus is not a reference oracle. For subjective/rubric tasks, compare single-judge and symmetric/role-asymmetric panels against blind human reference ratings, preserve inter-rater disagreement and publish role/order ablations. [Beyond Consensus (Findings EMNLP 2026)](https://aclanthology.org/2026.findings-emnlp.243/) reports that role asymmetry can introduce directional downward bias and that consensus did not repair it in the tested tasks. Compose with #9431's judge calibration/dependence programme rather than creating a second evaluator ensemble.

Topology is not a complete security evaluation. [ACIArena (ACL 2026)](https://aclanthology.org/2026.acl-long.457/) spans external inputs, agent profiles and inter-agent messages across six MAS implementations; it reports that topology alone is insufficient. [ART, UK AISI 2026](https://www.aisi.gov.uk/research/security-challenges-in-ai-agent-deployment-insights-from-a-large-scale-public-competition) provides an operational attack/effect benchmark context. Compose the attack surfaces and effect-level outcomes with #1872; do not import those public competition figures as our result.

Uncertainty is hierarchical. [The Value of Variance (ICML 2026)](https://proceedings.mlr.press/v306/tang26n.html) motivates separately recording within-agent, between-agent and system uncertainty. The benchmark must not optimize away valid disagreement or minority dissent by rewarding consensus.


## New human-reference and confidence comparators (10 November 2026)

### Human–LLM dynamic update fidelity

**Wali & Tayyab, *Before You Poll with LLMs: A Deliberative Diagnostic Framework*, EMNLP 2026.** The paper compares human and LLM expressed-position changes after identical informational interventions. The authors report using America in One Room data (526 personas / 72 questions) and observe reversal, overshoot and rigidity signatures among the particular frontier models they evaluated. These are reported findings, not a universal model ranking or evidence that any human panel is a normative truth oracle.

This reference motivates a separate human-reference track in [Mycelix #6023](https://github.com/Luminous-Dynamics/mycelix/issues/6023). It is **not** folded into the current synthetic 50-case Symthaea matrix. Dataset use is blocked pending a distinct consent, rights, ethics, privacy, version and split-provenance review.

Comparison metrics should include direction of update, magnitude/overshoot, rigidity, heterogeneity, response-scale harmonization and missingness. Do not simply optimize model output to mimic a single individual's opinion or to steer a demographic group.

Paper:
- https://aclanthology.org/2026.emnlp-main.2291/
- https://arxiv.org/abs/2609.15849

### Confidence-channel integrity

**Zhang et al., *When Linguistic and Internal Confidence Diverge in Large Language Models*, Findings of EMNLP 2026.** The authors report analysis across 30 models, eight classification tasks and two generation tasks. They separate instance-level association, magnitude agreement and calibration between verbal confidence and internal/logit or semantic-entropy signals, finding frequent divergence and prompt-sensitive inflation.

This strengthens the revision-ledger / [#11367 confidence-calibration](https://github.com/Luminous-Dynamics/symthaea/issues/11367) contract: confidence should be evaluated across at least three separate axes—rank association, magnitude agreement and probabilistic calibration—with task/model/prompt scope and held-out split evidence. Self-reported verbal confidence is never evidence, source authority or civic entitlement.

Paper:
- https://aclanthology.org/2026.findings-emnlp.1300/

### Efficiency and adaptive debate

**SVR-MAD, Findings of EMNLP 2026.** The authors propose posterior-guided communication-graph construction using pre-debate signals and debate outcomes; they report up to 61% token-cost reduction while matching or improving the most accurate competing MAD baseline in their tested setup. This should become a matched efficiency baseline: same task/model/agent/tool budget, cost boundary, and quality metrics. We should additionally test whether adaptive pruning suppresses minority evidence or overweights poorly calibrated confidence.

Paper:
- https://aclanthology.org/2026.findings-emnlp.483/
- https://arxiv.org/abs/2605.23099

### Treatment arms added to the comparison plan

| Arm | Comparator | Required caveat |
|---|---|---|
| A10 — human dynamic-reference diagnostic | Mycelix #6023 / Wali & Tayyab EMNLP 2026 | Human data use requires separate rights/consent review; human response distribution is not normative truth |
| A11 — confidence-channel audit | Findings EMNLP 2026 linguistic/internal confidence comparison | Report association, magnitude agreement and calibration separately |
| A12 — adaptive evidence-aware routing | SVR-MAD | Reproduce token savings only at matched output quality and measure minority-evidence exclusion |

A12 should also be compared with Free-MAD's trajectory scoring, standard MAD and static topologies. Communication graph adaptation is not delegation or authority assignment.
