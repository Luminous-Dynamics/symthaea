# Collaboration Gap, Relay Inference, and Team-Diversity Protocol

**Status:** research extension proposal, 2026-11-15  
**Applies to:** SympoeticBench human-AI collaboration and Luminous multi-agent systems  
**Evidence status:** adjacent published research exists; this protocol has not yet been empirically evaluated.

## 1. Why add this protocol?

Solo capability does not reliably predict collaborative capability.

The COLM 2026 work *The Collaboration Gap* evaluates 32 leading open- and closed-source models in solo, homogeneous-pair, and heterogeneous-pair conditions. It reports that collaboration can substantially reduce performance relative to solo ability, and that starting with the stronger agent can improve outcomes through a relay-inference strategy. This is a direct warning against inferring team performance from model leaderboards.

A separate May 2026 study, *Multi-agent AI systems outperform human teams in creativity*, reports 4,541 ideas from multi-agent LLM teams and 341 ideas from human teams across six tasks. The reported LLM-team advantage was driven by novelty while usefulness remained comparable. This is important adjacent evidence, but it does not establish that a human-AI team is better than a human team, nor that AI-only team output preserves human agency.

CHI 2026 work comparing model-led and reflective human-led co-creation found that interaction design changes quality, diversity, and perceived ownership. A 2026 meta-analysis found a small overall homogenisation effect, with substantial task-dependent heterogeneity.

The combined implication: evaluate the *team configuration and interaction policy*, not only the component models.

## 2. Distinct constructs

### 2.1 Collaboration gap

For normalized task quality in [0, 1]:

**collaboration_gap = max(Q(A), Q(B)) - Q(A+B)**

- Positive: the pair underperforms its best solo member.
- Zero: the pair matches the best solo member.
- Negative: the pair exhibits synergy over the best solo member.

For human-AI work, use matched human-only, AI-only, and human-AI conditions. For AI-agent experiments, record each agent's solo score and evaluate homogeneous and heterogeneous pairings separately.

Do not treat a team score as evidence of the individual's capability, and do not treat a solo score as evidence of team competence.

### 2.2 Partner-order sensitivity

**order_sensitivity = abs(Q(A→B) - Q(B→A))**

Randomize who begins, who receives the initial brief, which partner synthesizes, and whether a handoff occurs. A high value means performance is strongly dependent on order. Report the full order-specific results; do not conceal the asymmetry in an average.

### 2.3 Handoff recovery

**handoff_recovery_rate = successful_post_handoff_tasks / attempted_valid_handoffs**

A valid handoff must include the goal, constraints, supporting evidence, assumptions, unresolved questions, and the authority boundary. A handoff is successful only when a blinded evaluator confirms that the receiving agent preserves the critical information and completes the assigned continuation without inventing prior work.

### 2.4 Role complementarity

Measure whether role specialization produces non-redundant work that improves the joint artifact. Count neither role labels nor number of agents as evidence of complementarity. Use contribution traces, artifact changes, independent outcome scoring, and counterfactual ablations.

### 2.5 Team diversity

Measure both within-team exploration and between-team output diversity at matched quality. Novelty alone is insufficient; random or irrelevant outputs must not be rewarded. Report novelty, usefulness, feasibility, and quality-matched diversity separately.

## 3. Experimental matrix

Where feasible, use matched tasks and information budgets across the following conditions:

1. Agent A solo.
2. Agent B solo.
3. Stronger agent first, then handoff to the other agent.
4. Weaker agent first, then handoff to the stronger agent.
5. Homogeneous pair (same model/configuration).
6. Heterogeneous pair (different strengths or tools).
7. Human-only.
8. AI-only.
9. Conventional human-AI assistant interaction.
10. Reflective human-led interaction.
11. Reciprocal human-AI co-creation.
12. Human-AI team with multiple specialist agents.
13. Human-human reference team.

Not every task supports every condition. Report omitted conditions and avoid cross-condition claims where information, time, tools, or role permissions differ.

## 4. Relay and handoff ablations

For each eligible task, run these variants:

- No handoff: the initial agent completes the task.
- Blind handoff: the receiving agent gets the raw task and partial outputs but no structured handoff.
- Structured handoff: the receiving agent gets a versioned handoff contract.
- Strong-first structured relay.
- Weak-first structured relay.
- Role-specialized parallel work followed by independent synthesis.
- Parallel work with a structured disagreement-resolution stage.

The order of these conditions should be counterbalanced. Do not infer a universal strongest-first rule from one domain or model pairing.

## 5. Handoff contract

A handoff event should include:

- Current task and subtask identifiers.
- Current goal and accepted constraints.
- Evidence references and evidence quality.
- Explicit assumptions.
- Decisions already made and who authorized them.
- Open questions and disagreements.
- Known failure modes and work not yet performed.
- Authority and tool permissions of the receiving agent.
- Artifact version and content digest.

The receiving agent should acknowledge the handoff, identify missing information, and state which parts it will preserve. A handoff is not complete merely because a message was transmitted.

## 6. New failure injection suite

Add controlled tests for:

- Lost constraint across handoff.
- Stale or contradictory evidence.
- Duplicate work across agents.
- False claim that a prior agent ran a test.
- Incorrect role or authority inheritance.
- Unavailable specialist after partial work.
- Conflicting plans with a hidden shared assumption.
- Overconfident synthesis that hides unresolved disagreement.
- Handoff message truncated or reordered.
- Tool outcome unknown at handoff boundary.

A receiving agent must not inherit authority merely because a previous agent had it. Permissions are principal-bound and must be re-authorized for the recipient.

Unknown tool outcomes remain unresolved until independently reconciled. A subsequent state-changing tool call is blocked while the relevant execution state is unknown.

## 7. Adaptive intervention policy

A future SympoeticBench track should measure whether a coordinator selects the right collaborative move:

- Ask a clarifying question.
- Offer an optional idea.
- Challenge an assumption.
- Delegate to a specialist.
- Summarize shared ground.
- Preserve an unresolved disagreement.
- Defer or stop intervening.

This is a policy-selection problem, not a verbosity contest. Measure intervention appropriateness, unnecessary intervention rate, time-to-clarification, downstream rework, participant agency, and missed opportunities to challenge.

A system should be rewarded for *not* intervening when the participant's intent is clear and no material risk or opportunity is present.

## 8. Reporting

Every report should include:

- Solo score for each agent or participant.
- All pairing and order-specific scores.
- Collaboration-gap and synergy estimates with uncertainty intervals.
- Handoff recovery and information-retention scores.
- Role contribution and counterfactual impact.
- Diversity and usefulness distributions.
- Cost, latency, human time, and communication burden.
- Hard failures, authority violations, and unresolved unknown outcomes.
- Model, prompt, tool, adapter, environment, and scenario digests.

Do not publish one team leaderboard number without the solo baselines and pair configuration.

## 9. Research hypotheses

- **H1:** Solo capability rankings will not perfectly predict collaborative rankings.
- **H2:** Structured handoffs will reduce information loss relative to blind handoffs.
- **H3:** Partner order will interact with role complementarity and task difficulty.
- **H4:** Multi-agent generation may increase novelty, while human participation and reflective interaction may better preserve agency and ownership.
- **H5:** Adaptive intervention policies will outperform static always-ask, always-suggest, or always-generate policies on a joint outcome/agency frontier.

These are hypotheses. The benchmark must allow them to fail.

## 10. External references

- Davidson et al., *The Collaboration Gap*, COLM 2026, https://www.microsoft.com/en-us/research/publication/the-collaboration-gap/
- Hu et al. (2026), *Multi-agent AI systems outperform human teams in creativity*, https://arxiv.org/abs/2605.17885
- Maier, Schneider, and Feuerriegel (2026), *Partnering with Generative AI*, CHI 2026, https://doi.org/10.1145/3772318.3791185
- de Rooij and Biskjaer (2026), *Does generative AI make us think alike?*, https://doi.org/10.1080/0144929X.2026.2726451

SympoeticBench should build on these results and test the intersection of collaboration gap, human agency, team diversity, provenance, consent, and long-horizon co-creation.
