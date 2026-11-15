# Benchmark Landscape and Positioning

**Research snapshot:** 15 November 2026  
**Purpose:** choose fair comparators and identify what SympoeticBench must add rather than duplicate.

## 1. Adjacent benchmarks and studies

| Work | What it measures well | Public scale / design reported | What SympoeticBench should add |
|---|---|---|---|
| AOBench v0.4.1 | Role-aware HPC agent operations, tool sequence, grounding, RBAC, workflow, robustness, cost and latency | 88 tasks, 29 environment bundles, five roles, ten QCATs, seven weighted dimensions | Do not duplicate operations testing. Use as an external operations baseline; add real consequence, concurrency, and site-specific qualification only as separately named extensions |
| Common Ground Benchmark (2026) | Shared goals, referential coordination, joint action, situation awareness and repair | Interactive collaborative puzzle with human-AI validation | Broaden beyond puzzle coordination to creative, engineering, scientific, pluralistic and longitudinal collaboration |
| AgentWorld (2026) | Long-horizon collaboration, role asymmetry, resource sharing, communication and causal contribution | 100 human-annotated tasks plus 100 augmented variants; 3–20 agents; 50+ interaction rounds | Extend causal contribution analysis to human-AI dyads and multi-stakeholder creation, with individual agency and consent measures |
| Human Creativity Benchmark (2026) | Separates professional convergence on verifiable quality from legitimate divergence in taste | 15,000 professional judgments across five creative domains and three workflow phases | Preserve rater disagreement and evaluate the interaction process, not only the artifact |
| CHI 2026 co-creation experiment | Compares model-led rewriting with reflective human-led interaction and measures quality, diversity and ownership | Randomized study with 486 participants and a separate replication with 640 participants | Turn the interaction-mode finding into repeatable, cross-domain, longitudinal benchmark tasks |
| Symthaea Psych-Bench | Broad cognitive and psychometric architectural tests | Existing suite with synthetic and externally grounded components | Keep as a separate cognitive suite; do not relabel it as evidence of human-AI collaboration |

## 2. Findings that shape our design

The 2026 common-ground benchmark formalizes shared assumptions, goals, situation awareness, joint action and repair as core conditions for collaboration. This is a strong reference for SympoeticBench's common-ground and repair tasks.

AgentWorld reports long-horizon failures in communication, role tracking and shared-plan maintenance, and introduces Causal Collaboration Effectiveness to ask whether agent actions actually contributed to a team's outcome. SympoeticBench should adopt the general lesson that activity volume is not contribution, while measuring individual agency and artifact provenance in addition to group completion.

The Human Creativity Benchmark argues that professional agreement and disagreement carry different information: technical correctness can invite convergence, while aesthetic direction and conceptual risk legitimately diverge. SympoeticBench should not collapse all rater disagreement into noise or force one universally preferred output.

The CHI 2026 experiment found that model-led rewriting improved idea quality while reducing idea diversity and perceived ownership; reflective human-led conditions also improved quality while preserving diversity and ownership. This motivates matched interaction conditions rather than a single generic "AI-assisted" condition.

A 2026 systematic review and meta-analysis of 19 studies and 61 effect sizes reported a small but statistically significant homogenisation effect associated with generative-AI use (reported d = 0.334). This makes population-level diversity a first-class outcome, not a cosmetic novelty score.

## 3. The distinctive research gap

No benchmark should claim that the adjacent work is absent. The opportunity is to integrate its strongest constructs into one evidence-qualified human-AI co-creation protocol:

1. Task-level outcome quality and constraint satisfaction.
2. Synergy against matched human-only and AI-only controls.
3. Participant agency and control across the process.
4. Common ground, uncertainty, and repair.
5. Constructive divergence and preservation of plural outcomes.
6. Individual and population-level diversity at matched quality.
7. Event-level contribution provenance and counterfactual contribution impact.
8. Consent scope, revocation, reuse limits, and truthful memory reporting.
9. Longitudinal adaptation across repeated sessions.
10. Cost, latency, human time, workload, accessibility, and rework.

The benchmark's contribution is the integrated measurement protocol and reproducible corpus, not a claim that any one of these constructs is newly invented.

## 4. How to compare fairly

- Reproduce an adjacent benchmark's published task and protocol before claiming superiority on that construct.
- Pin benchmark release, task split, judge, model snapshot, agent harness, and resource budget.
- Use identical or explicitly matched interaction time, information, tools, and participant instructions.
- Report all conditions and per-domain performance, not only a chosen aggregate.
- Use confidence intervals and prospective power calculations.
- Preserve expert disagreement where the task has legitimate plural outcomes.
- Separate direct benchmark results from local extensions and simulated proxies.
- Do not claim "better than SOTA" from a new score on a different task or rubric.

## 5. Research targets

The first goal is construct validity and reproducibility, not leaderboard rank. A later competitive claim should require:

- At least one faithful replication on Common Ground-style tasks.
- A matched interaction-mode comparison informed by the CHI 2026 protocol.
- A creative-diversity evaluation consistent with HCB's separation of convergence and divergence.
- A long-horizon contribution analysis comparable in spirit to AgentWorld's CCE.
- An external human study with preregistered outcomes and blinded raters.
- An independent replication on a held-out scenario set.

A SympoeticBench result should be considered stronger than a baseline only when it improves the intended outcome without violating its declared non-inferiority or hard-gate constraints.
