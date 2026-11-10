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

A2 and A3 separate the two DReaMAD design levers. A4/A5 are candidate Symthaea interventions, not assumed wins. A6 uses the five frozen content/topology pairs in the scenario matrix.

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
5. Run A2–A9 as separate treatments; preserve failures and incomplete runs.
6. Publish raw multi-axis receipts and counterexample traces; no aggregate "alignment contagion" score.
7. Only after synthetic evidence is stable, consider separately authorized human-group experiments under Mycelix #5999 and independent ethics review.

## Ownership

- #11248 owns deterministic graph generation.
- #3054 / #3095 / #3115 own institutional experiment orchestration and provenance.
- #1872 owns hostile-agent execution.
- #9431 owns evaluator correlation/ensemble qualification.
- #11320 owns the stance-shift measurement protocol and counterfactual analysis.
- Mycelix #5999/#6011 own human-group process and participant-conditioned summary evaluation.

No existing engine or production governance authority is duplicated by this crosswalk.


## Graph ownership and semantics

The topology axis in this programme is a **CommunicationGraph**, not the delegation graph.

- [Symthaea #11337](https://github.com/Luminous-Dynamics/symthaea/issues/11337) owns AgentInstanceId nodes, directed MessageChannel edges, delivery/revocation events, communication reachability, and the normalized-betweenness definition used by #11320.
- [Symthaea #11248](https://github.com/Luminous-Dynamics/symthaea/issues/11248) owns PrincipalId/scoped mandate delegation networks. Its edges are not message channels and must not be reused as such.
- An optional principal binding on a communication node is metadata only. Multiple AgentInstanceIds per PrincipalId remain distinct unless a declared experiment transformation explicitly aggregates them.

The five matched centrality controls require strongly connected reciprocal directed graphs, unweighted hop-count shortest paths, minimum/maximum tie-set capture and deterministic AgentInstanceId tie-breaking. A disconnected or unsupported graph remains Unsupported; the runner does not manufacture a centrality result.
