# EGDDM competitive research and paradigm shifts — 2026-11-14

**Purpose:** define a fair, falsifiable research position for Evidence-Grounded Dynamical Development Maps (EGDDM), and identify what would justify claims of improvement over current repository-intelligence and coding-agent systems.

**Claim status:** research synthesis and prospective experimental plan. No competitive benchmark has been run by this document.

## 1. Positioning: do not claim that a graph is the breakthrough

The field already includes systems that join repository structure, development activity, temporal information and provenance. CoReGraph is an especially direct overlap: its 2026 architecture explicitly combines technical dependencies, organizational context from development activity, temporal modelling and provenance metadata. A claim that EGDDM is novel because it has a repository graph plus history plus provenance would therefore be weak.

EGDDM should instead be evaluated as a **decision-centric engineering intelligence system** with five coupled obligations:

1. reconstruct the exact information state that a particular actor could legitimately use at a decision point;
2. identify the smallest additional observation that can resolve a consequential uncertainty;
3. predict the effects of a proposed intervention, not merely correlate co-changes or dependency paths;
4. choose act, request evidence, defer or abstain under explicit constraints and authority; and
5. measure whether the recommendation led to independently accepted capability gains, lower maintenance burden and fewer regressions.

The repository map is an input to this system, not the product or the success metric.

## 2. Paradigm shifts

### 2.1 From repository snapshots to epistemic-temporal state

A static map asks what is in the repository. A temporal repository asks how it changed. EGDDM must additionally ask what a named actor could have known at the time, what the ledger had actually recorded, and what later correction changes about the present interpretation of the past.

D2 v0.1 separates effective/valid time, source observation time, per-actor availability and ledger recording time. Later versions should add uncertainty intervals for unsynchronised clocks, causal/partial ordering for concurrent actors, and authenticated external checkpoints.

### 2.2 From maximal retrieval to decision-sufficient evidence

More context is not automatically better. The system should rank candidate observations by expected reduction in decision loss per unit of acquisition cost, latency, review burden and risk. It should request the cheapest discriminating observation when the current evidence cannot justify an action.

The benchmark must compare a fixed-budget retriever against task-specific retrieval baselines and against an evidence-acquisition policy. It must report both coverage and false-positive context, not just Recall@K.

### 2.3 From dependency graphs to intervention-tested causal claims

A dependency edge, shared owner, co-change or temporal precedence is not proof that one component caused another to fail. Each consequential causal claim should include:
- an explicit hypothesis and competing explanations;
- the intervention or natural experiment that could distinguish them;
- a preregistered predicted outcome and uncertainty range;
- a frozen pre-intervention repository/configuration world;
- independently observed post-intervention results; and
- residual confounders and limitations.

The system should be rewarded for preserving an unresolved causal relation as unknown, not for filling every graph edge.

### 2.4 From patch success to longitudinal engineering value

A patch can pass its immediate tests while degrading the system's next maintenance task. Every recommendation therefore needs an outcome trail: intended capability, affected obligations, acceptance evidence, later regressions, rollback/repair, human review effort, compute and elapsed time.

The primary value metric should be **risk-adjusted, independently accepted capability gain per engineering/reviewer hour**, with regression burden and irreversible effects reported separately. Never hide a critical safety or authority violation in a weighted average.

### 2.5 From agent autonomy to evidence-bounded agency

The action policy should have at least four explicit outputs: act, acquire evidence, defer, and abstain. A recommendation is not authorized merely because its technical prediction is confident. Authority, scope, evidence freshness, execution isolation and reversibility remain hard constraints.

Evaluate this with paired cases differing in exactly one decisive fact: sufficient versus insufficient evidence, authorized versus unauthorized actor, active versus inactive configuration, current versus stale contract, or reversible versus irreversible effect.

### 2.6 From central graph authority to federated, polycentric evidence

Each repository retains authority over its own source, policy and domain records. Federation edges start as hypotheses or contract candidates. They become qualified only when both producer and consumer provide evidence tied to exact revisions, configuration worlds, contract blobs and independently checked conformance receipts.

A central map may discover a relationship; it cannot grant authority to another repository.

## 3. Competitive landscape and what each system actually measures

| System / benchmark | Published or documented scope | EGDDM response |
|---|---|---|
| Aider RepoMap | Symbol/dependency-oriented context map chosen under a token budget | Use as a retrieval/context baseline; compare task-specific context yield under equal budgets |
| rust-analyzer / SCIP / CodeQL | Language-aware semantic navigation and static analysis | Consume qualified outputs; do not relabel syntax-only extraction as compiler-resolved |
| CoReGraph (2026) | Code dependencies, development activity, time and provenance in an extended repository KG | Treat as direct architecture overlap; differentiate on actor-time admissibility, decisions, intervention outcomes and action/defer |
| Agent Retrieval Bench | 427 samples across 25 repos; code2test, comment2context, trace2code, edit2ripple, natural no-gold and wrong-repository controls | Reproduce exact official version; add actor-time, configuration and evidence-sufficiency controls |
| CORE-Bench | 180,000+ queries and 106,000 broader-context relevance labels | Use as a large-scale context-retrieval baseline; report query-family results separately |
| SWE-Explore | 848 issues across 203 repositories and 10 languages, ranking code regions under a line budget | Reproduce its ranking/line-budget evaluation; add temporally admissible and decision-sufficient context |
| SWE-Time (2026) | 1,661 cases across 65 repos; retrospective and prospective temporal repository reasoning | Reproduce its temporal tasks and splits; keep DTEA admissibility separate from its regression/forward-change tasks |
| ATime-Consistent (2026) | Repository context frozen at T0 and future merged PRs evaluated after T0; matched with/without repository context | Adopt time-consistent split design and matched A/B control to limit temporal leakage and prompt confounding |
| TEMPO (2026) | 1,730 temporal retrieval queries across 13 domains | Reproduce temporal retrieval measures; do not confuse time-aware retrieval with actor-authorized decision evidence |
| SWE-CI (2026) | Longitudinal maintenance tasks over histories averaging 233 days and 71 commits | Add sustained evolution and cumulative regression checks |
| SWE-Maintain (2026) | 100 three-event maintenance episodes across 50 Python repositories; 1,996 paired later events. Agent-History reduces later-task resolution by 5.6 pp on average (14.0 pp when all earlier tests must still pass); the disadvantage disappears when inherited behavior already passes the earlier tests (+2.7 pp, 95% CI [-0.5, +6.1]) | Measure whether EGDDM preserves accumulated behavior and improves future maintainability; distinguish inherited test failures from code-quality effects |
| FeatureLoop (2026) | 61 human-reviewed tasks from 34 repositories with evolving requirements and cumulative checks | Add multi-stage capability evolution with human review and preservation of earlier requirements |
| SWE-Milestone | Continuous development across a milestone DAG and real release histories; its public leaderboard exposes resolve, precision, recall, cost, output-token and elapsed-time metrics | Use milestone-level quality and cumulative acceptance as an outcome measure; compare cost and time, not score alone |
| MLOps-Bench (2026) | 212 validated production-ML tasks across 31 repositories, six lifecycle stages plus end-to-end; the published ACL edition reports 27.3% task solves and 67.1% of tests passed for its best evaluated agent | Use as a lifecycle/production engineering baseline where the repository domain overlaps; record paper version because an earlier arXiv version reports different figures |
| SWE Atlas | 284-task suite spanning codebase Q&A, test writing and refactoring | Compare engineering-loop performance, not only edit success |
| Artificial Analysis Coding Agent Index v2.1 | Equal-weight composite of FrontierCode-AA 2.0 (132 tasks), Terminal-Bench 4.1 (66 tasks) and SWE-Atlas-QnA-AA (124 tasks), with efficiency reporting | Treat as an industry reference, not as a substitute for EGDDM's evidence, temporal and authority tracks |
| AgentAbstain | 263 paired tasks across 42 environments, explicitly testing act/abstain contrasts | Adapt paired one-delta design to engineering evidence, authority and irreversible-effect decisions |

### Benchmark version discipline

Several named benchmarks have multiple versions or paper/leaderboard datasets. The Agent Retrieval Bench paper's expanded 427-sample evaluation is not interchangeable with its separate 225-sample V1 leaderboard. SWE-Time results are harness-dependent; do not compare one agent's Claude Code number against another agent's Codex number as a single ranking. MLOps-Bench and other evolving projects must be cited with their exact artifact, paper version and scoring protocol.

Before any comparative claim, freeze the dataset release, sample IDs, base commits/trees, train/test boundary, model and agent harness, prompts, tools, tokenizer, context budget, execution environment, grader and scoring rules. Reproduce the published benchmark before adding EGDDM-specific extensions.

## 4. Proposed DynaMapBench evaluation family

Do not produce one aggregate score. A strong retrieval score cannot compensate for temporal leakage, unauthorized action or a false qualification claim.

| Track | Core question | Prospective target |
|---|---|---|
| D1: source fidelity | Does each claimed source fact match an authenticated source object and declared authority level? | ≥95% precision/recall on independently adjudicated syntax/source gold; 100% rejection of planted source identity mismatches |
| D2: temporal replay | Can the system reconstruct valid-time, actor-knowledge and ledger-as-recorded states without hindsight leakage? | 100% deterministic fixture outcomes; 100% rejection of explicit future-availability/tampered-chain controls; incremental and clean replay identical |
| D2: decision admissibility | Was each consulted item available, in scope and valid for this actor at the decision? | ≥95% precision/recall on independently adjudicated real cases; zero accepted planted future/unauthorized controls |
| D3: retrieval | Does the system return the useful regions within the same token/line budget? | Reproduce external baselines first; then target ≥5 percentage-point line-recall gain over reproduced K=5 baseline with ≤1 pp precision loss |
| D4: change impact | Does predicted affected scope match independently labelled downstream obligations and tests? | ≥95% recall, ≥80% precision as an initial pilot; zero missed critical obligations in explicit controls |
| D4: planning | Are all hard constraints, prerequisites, authority and isolation requirements preserved? | Zero hard-constraint violations; ≥95% prerequisite precision/recall |
| D5: causal interventions | Do preregistered predictions outperform no-change, dependency-only and co-change baselines under held-out interventions? | Lower prediction error with calibrated uncertainty and no increase in unsupported causal claims; set numeric threshold after corpus pilot |
| D6: act/defer | Does the system act when justified and defer when one decisive evidence/authority condition is missing? | ≥75% paired accuracy on an engineering-specific pilot; zero unauthorized irreversible effects |
| D7: longitudinal outcomes | Do recommendations improve later capability and maintenance outcomes? | Initial hypothesis: ≥10 pp improvement in independently accepted capability and ≥20% lower review/experiment cost, subject to powered paired intervals |
| D8: federation | Are cross-project relations truly compatible under both owners' pinned contracts and configurations? | ≥95% critical mismatch recall; zero promotion without two-sided receipts |

These are research targets, not observed results, commercial promises or established industry thresholds. Security-critical planted controls should require zero observed false acceptance, but publish the denominator and a confidence bound; a small zero-error set does not prove zero risk.

## 5. Matched baseline matrix

The confirmatory evaluation should compare:

1. lexical search (ripgrep/BM25) with a fixed context budget;
2. dense embedding retrieval and reciprocal-rank fusion;
3. Aider-style RepoMap;
4. syntax/manifest-only projection;
5. compiler-resolved index where a trusted build/configuration exists;
6. temporal-only retrieval;
7. graph-guided retrieval without actor/authority constraints;
8. CoReGraph-style technical + organizational + temporal KG baseline;
9. an evidence-admissibility-only ablation;
10. EGDDM full system;
11. an expert-human reference arm.

Use identical source snapshots, time cutoffs, tools, budgets and task context. For end-to-end comparisons, give every system the same editing/execution harness or score retrieval separately. Report model-free retrieval metrics separately from model-agent completion metrics.

Required ablations: remove valid-time, actor availability, ledger-recording time, configuration-world pinning, source identity pins, independent custody, no-gold controls, causal interventions, hard constraints and abstention. If removing a feature does not change its corresponding error class, the design has not demonstrated that the feature matters.

## 6. Experimental validity and measurement

- Repository-family holdout and forward-time holdout; do not split sibling revisions across train and test.
- Freeze pre-change source snapshots before deriving future tasks.
- Keep labels, hidden outcomes and evaluator credentials outside the candidate's context.
- Independently adjudicate relevant, forbidden, unknown and authority-required evidence.
- Pair cases that differ by one decisive condition.
- Use a separately implemented reference oracle for D2 and D4; do not let the candidate validate its own output.
- Report micro and repository-macro metrics, precision/recall, F1, calibration, abstention coverage, no-gold false-return rate, critical miss rate, cost, latency, tokens, memory, reviewer time and regression burden.
- Use bootstrap intervals clustered by repository, and publish exact denominators and exclusions.
- For D5 and D7, preregister hypotheses, controls, stopping rules and minimum detectable effects; measure outcomes independently of the recommending system.

### Causal outcome design

A recommendation is not a causal effect. For D7, use randomized assignment where possible: eligible tasks are assigned to EGDDM-assisted or matched baseline workflows, with identical human review/acceptance and execution safety. Where randomization is not feasible, use a preregistered matched-pair or stepped-wedge design and document confounders. Compare accepted capability gain, critical regressions, maintenance burden, time-to-correction and reviewer effort. Avoid counting merged PRs or test passes as capability without independent obligation-level verification.

## 7. Priority engineering roadmap

1. Qualify the D1 capture chain: authenticate exact commit-to-tree and tree-to-blob relationships, reject truncated inventories, and independently verify source content digests.
2. Finish D2 as an independently tested event ledger with actor availability evidence, append-only corrections, partial-order/multi-actor handling, signed checkpoints and replay equivalence.
3. Reproduce Agent Retrieval Bench, CORE-Bench and SWE-Explore before changing retrieval strategy.
4. Add intervention-based D4/D5 tasks with held-out source/configuration mutations and independent downstream oracles.
5. Add paired act/defer and minimum-evidence acquisition tasks.
6. Run a controlled D7 pilot that tests whether recommendations actually improve the repository over multiple milestones.

## 8. Research references

- CoReGraph (2026), *A Knowledge Graph Approach for Software Repository Analysis*: https://doi.org/10.1016/j.procs.2026.09.172
- Agent Retrieval Bench (2026): https://arxiv.org/abs/2607.24882 and https://agent-retrieval-bench.github.io/
- CORE-Bench (2026): https://aclanthology.org/2026.emnlp-main.1604/
- SWE-Explore (2026): https://arxiv.org/abs/2606.07297
- SWE-Time (2026): https://arxiv.org/abs/2611.12083
- ATime-Consistent repository-level evaluation (2026): https://arxiv.org/abs/2603.26137
- TEMPO (2026): https://aclanthology.org/2026.emnlp-main.291/
- SWE-CI (2026): https://arxiv.org/abs/2603.03823
- SWE-Maintain (2026): https://arxiv.org/abs/2610.14768
- FeatureLoop (2026): https://arxiv.org/abs/2610.35488
- SWE Atlas (2026): https://scale.com/blog/swe-atlas-complete
- Artificial Analysis Coding Agent Index v2.1 methodology: https://artificialanalysis.ai/methodology/coding-agents-benchmarking
- AgentAbstain (2026): https://arxiv.org/abs/2607.10059 and https://agentabstain.github.io/
- SWE-Milestone: https://swe-milestone.com/
- MLOps-Bench (published ACL Findings edition): https://aclanthology.org/2026.findings-emnlp.1402/

The central scientific claim remains unproven until EGDDM is compared under these matched protocols. A map that is more detailed is not necessarily a better engineering system. A better system makes fewer unsupported claims, asks for less but better evidence, predicts effects more accurately, preserves hard boundaries, and improves independently verified outcomes over time.
