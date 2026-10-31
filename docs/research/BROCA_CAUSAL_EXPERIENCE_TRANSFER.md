# Broca: Causal Experience Transfer Across Sessions

**Status:** research protocol / architecture proposal  
**Tracking:** [#9516](https://github.com/Luminous-Dynamics/symthaea/issues/9516)  
**Date:** 2026-10-31

## 1. Research thesis

The next frontier for Broca is not another recall benchmark. It is whether an agent can learn from its own interactions with an environment, preserve the parts of experience that matter, and use that knowledge to make better decisions in later dependent tasks.

Three 2026 results make this a timely, testable direction:

- **MemoryArena (ICML 2026)** evaluates interdependent multi-session Memory-Agent-Environment loops. An agent must distill earlier actions and feedback into memory and use it for later dependent subtasks. The authors report that systems near saturation on LoCoMo can still perform poorly in this agentic setting.
- **AMA-Bench (ICML 2026)** evaluates long-horizon agent trajectories composed of states, actions, observations and tool outputs. It identifies missing causal/objective information and lossy similarity retrieval as important limitations. AMA-Agent reports 57.22% average accuracy and an 11.16 percentage-point gain over its strongest baseline using causality graphs and tool-augmented retrieval.
- **UltraHorizon (ICML 2026)** studies partially observable, very long exploration tasks. Its heavy setting averages 200K+ trajectory tokens and 400+ tool calls; standard configurations exceed 35K tokens and 60 tool calls. The paper reports persistent gaps between current agents and humans, and finds that simple scaling does not close them.

This is not a generic MemoryArena adapter or a replacement for CMAQ-001 (#8877), the cost-quality benchmark (#8427), or proof-carrying actions (#9490). It defines the missing bridge: **causal experience formation and transfer across dependent tasks**, with independent ground truth for what an action actually caused.

## 2. Paradigm shift: from storing events to learning qualified transitions

An event log says what happened. A causal memory must represent what changed, under which conditions, what the agent did, what was merely observed, what alternative explanations remain, and when the learned relationship is safe to reuse.

Do not equate:

~~~text
action preceded outcome         != action caused outcome
correlated transition           != identified causal effect
successful episode              != reusable policy
prediction confidence           != model applicability
memory retrieval                != transfer
one-step prediction accuracy    != long-horizon task improvement
~~~

The core object is a **qualified experience transition**:

- prior state/belief and its provenance;
- selected action and exact parameters;
- objective, constraints and policy context;
- prediction distribution and model digest, recorded before acting;
- observed post-state and reconciled outcome receipt;
- intervention/observation classification and confounder assumptions;
- source/session/task lineage;
- causal claim with confidence and identification status;
- validity domain, regime and expiry/trigger conditions;
- downstream tasks and decisions that consumed the transition;
- correction, retraction and alternative explanations.

An action followed by an outcome does not establish causation. Where identification is unavailable, store an observed association with explicit uncertainty, not a causal rule.

## 3. Proposed architecture

1. **Append-only interaction trace.** Preserve state snapshots, action intents, dispatched commands, tool outputs and environment observations. Distinguish proposed, authorized, dispatched, accepted and completed.
2. **Outcome reconciliation.** Bind each claimed outcome to observed environment state, not solely to the agent's own generated report.
3. **Experience abstraction.** Derive candidate transition schemas, preconditions, effects, costs and failure modes from trace evidence. Each abstraction references exact source events.
4. **Causal qualification.** Distinguish randomized/interventional evidence, quasi-experimental evidence, observational evidence, simulator ground truth and unverified hypotheses. Record assumptions and identified estimands.
5. **Objective and constraint memory.** Preserve goal hierarchy, hard constraints, resource budgets, dependencies and preference validity—not only entity/fact summaries.
6. **Policy-relevant retrieval.** Query by current state, objective, action preconditions, expected effects, failure signatures and regime, as well as semantic similarity.
7. **Prediction before action.** Record predicted transition and uncertainty before acting. Compare later with reconciled outcome; never backfill the prediction after observing the result.
8. **Counterfactual transfer.** Reuse experience in a later task only if preconditions and model applicability hold. Otherwise abstain or acquire more evidence.
9. **Lifecycle repair.** Outcome corrections, changed rules and retractions invalidate dependent causal abstractions and policies. Preserve alternatives only when independently supported.
10. **Proof-carrying policy use.** A future action binds to the qualified transition/model evidence it relies on, then passes the separate action-authorization contract in #9490.

## 4. MACE benchmark proposal

**MACE: Memory-Agent-Causal Environment** is a proposed protocol, not a claim of an existing implemented suite. Reuse official MemoryArena and AMA-Bench examples/adapters where licensing permits, then add a deterministic symbolic simulator where ground truth and interventions are controlled.

### Episode design

Each scenario is a multi-session chain with explicit dependencies:

- Session A exposes partial state and a task objective. The agent chooses actions, observes outcomes and records experience.
- A hidden simulator oracle retains the actual transition function and causal graph. The agent is not given the future task solution.
- Memory consolidation runs under a fixed storage/write/token budget. Decisions, rejected memories and derived rules are receipted.
- Session B introduces a dependent task requiring useful transfer from Session A. A relevant regime variable may be preserved or changed.
- Session C introduces a delayed correction, retraction, new constraint or contradictory observation.
- The agent must decide whether to reuse, adapt, seek more evidence, request confirmation or abstain.
- A held-out final task tests transfer to a new but structurally related environment.

### Scenario families

1. Causal vs correlational decoy: action and outcome co-occur, but a hidden confounder explains the association.
2. Objective dependence: the same transition has different utility under different goal/constraint configurations.
3. Delayed consequence: short-term reward is positive but later cost reverses the decision.
4. Partial observability: the action result depends on a latent state; uncertainty must be preserved.
5. Regime shift: previously valid transition rules change after an environment intervention.
6. Intervention vs observation: observational trajectories cannot justify the same causal claims as controlled interventions.
7. Failure memory: unsuccessful actions are informative; retain what failed and under which preconditions.
8. Alternative explanation: multiple causal models fit current data; preserve model uncertainty.
9. Outcome misattribution: a tool reports success, but external state did not change as expected.
10. Objective drift: goals or hard constraints change; old policies must not be reused as if objectives were unchanged.
11. Memory poisoning: a false agent-generated explanation is inserted into a trace; it must not become trusted causal knowledge.
12. Cross-session transfer: relevant information is distributed across sessions and requires complete evidence closure.
13. Compression under budget: compress memory and test downstream decision preservation, not only token reduction.
14. Irrelevant-update control: unrelated events must not alter the selected policy or claimed causal relation.

Every decisive intervention has a matched irrelevant control. Freeze seed, environment generator, transition oracle, intervention schedule and hidden test labels by digest before model runs.

### Stage-resolved memory lifecycle: write, maintain, retrieve, use

WorldMemArena (arXiv:2605.29341) evaluates 400 multi-session multimodal tasks and separates the memory lifecycle into four observable stages: write, maintain, retrieve and use. It includes Lifelong Evolution and Agentic Execution regimes and reports that stronger memory extraction/storage alone does not guarantee better downstream use; visual evidence can be degraded when reduced to captions, and performance is brittle on GUI/embodied trajectories.

MACE should therefore publish four separate stage scorecards:

| Stage | What is evaluated | Required diagnostics |
|---|---|---|
| Write | Which experience atoms and transition claims enter persistent memory | evidence-point precision/recall, action/outcome separation, provenance completeness, causal-claim calibration, memory-write cost |
| Maintain | Whether memory survives corrections, regime changes and policy updates correctly | freshness accuracy, stale-memory exposure, invalidation latency, alternative-support retention, interference/poisoning rate |
| Retrieve | Whether all decision-relevant support is found at the moment it is needed | required-evidence recall, proof closure, distractor exclusion, session coverage, modality-preserving retrieval |
| Use | Whether retrieved memory changes the decision correctly | action ranking, dependent-task completion, regret reduction, safe abstention, outcome reconciliation |

Do not combine these stages into a single average. A weak writer can starve a strong retriever; a weak maintainer can invalidate an otherwise correct memory; a weak user can retrieve everything and still act incorrectly. Preserve per-stage error attribution and link each downstream error to the earliest upstream defect that caused it.

For multimodal extensions, retain image/frame/object/region/time references and renderer/model lineage. A caption is a derived representation, not a replacement for the original visual observation when spatial, temporal or procedural details determine the action.

### Continual improvement: progressive experience ladders

The Continual Learning Exam (CLE, arXiv:2610.33956, submitted 25 October 2026) introduces 120 expert-authored tasks arranged as five-level ladders (600 levels total), spanning 89 professional roles and 12 occupational domains. Its best reported configuration completes 47.7% of levels and all five levels in 23.3% of tasks; diagnostic feedback plus one revision improves median completion by 5.3 percentage points. These are preprint-specific results, not a general workplace-success rate or a matched Broca comparison.

MACE should add a **progressive transfer lane**:
1. Level 1 teaches a bounded skill through a task with observable outcome.
2. Levels 2–4 change constraints, tools, scale and relevant regime variables while reusing prior experience.
3. Level 5 is a held-out transfer task with an altered surface form and at least one explicit negative-transfer trap.
4. Compare no memory, raw episode replay, episodic summaries, objective-aware memory, causal transition memory, and causal + proof/lifecycle memory.
5. Add a distinct feedback arm: diagnostic feedback and at most one revision. Record it separately from persistent-memory improvement so feedback effects are not misattributed to memory.

Measure the learning curve across levels, retained improvement after distractor sessions, transfer to unpracticed variants, recovery after correction, negative transfer under changed preconditions, and total resources spent per newly acquired capability. Require improvement to persist after the feedback message and immediate context have been removed; otherwise the result may be in-context revision rather than durable learning.

Official adapters should preserve the original CLE and WorldMemArena evaluators. MACE-specific synthetic tasks are a separate controlled track and must not be pooled into their official scores.

## 5. Baselines and ablations

Within each paired comparison, freeze backbone, tool schema, environment seed, observation budget and action budget.

1. No persistent memory.
2. Full raw trajectory replay / full-context ceiling.
3. Latest-state key-value memory.
4. Episodic summaries.
5. BM25, dense and hybrid similarity retrieval.
6. Objective-aware retrieval without causal links.
7. Causal graph without proof/provenance gates.
8. Proof/provenance graph without policy-aware retrieval.
9. Broca candidate: bitemporal proof DAG + causal qualification + objective-aware retrieval + prediction/outcome reconciliation.
10. Oracle transition model / oracle memory as an upper bound only.

Ablate causal edges, objective records, outcome receipts, bitemporal state, regime checks, proof validation, memory compression and active evidence acquisition one at a time.

## 6. Metrics

### Task and policy quality
- dependent-subtask completion and final-chain success;
- cumulative reward/cost and regret versus no-memory and oracle policies;
- action-ranking accuracy and safe-action ranking;
- policy transfer under unchanged and shifted regimes;
- recovery time and regret after correction or model shift.

### Causal and model quality
- transition prediction log loss / Brier score;
- calibration and interval coverage;
- cause/effect edge precision, recall and orientation accuracy on oracle worlds;
- intervention-effect error and counterfactual ranking accuracy;
- false causal claim rate and unsupported-generalization rate;
- model applicability / out-of-domain detection and selective risk.

### Memory quality and lifecycle
- provenance coverage and exact event lineage;
- useful-memory retention and irrelevant-memory invariance;
- memory-induced policy lift versus matched no-memory;
- obsolete-policy / stale-transition reuse;
- invalidation correctness and alternative-support preservation;
- downstream decision agreement after compression;
- write/consolidation cost, memory bytes and replay complexity.

### Operational efficiency
- tokens, model/tool calls, retrieval latency;
- p50/p95/p99 end-to-end latency;
- memory bytes including indexes, journals, tombstones and proof metadata;
- CPU/GPU time and energy/cost per completed dependent task.

Publish vector scorecards and Pareto frontiers. Do not use one weighted average to hide increases in unsafe actions, false causal claims or stale-policy reuse.

## 7. Proposed qualification gates (targets, not results)

- 100% oracle agreement on finite deterministic causal and lifecycle fixtures.
- Zero accepted causal certificates whose assumptions, intervention identity or lineage are mutated.
- Zero stale, revoked or out-of-scope transition policies used after invalidation is known.
- Zero claimed action effects without reconciled outcome evidence.
- 100% prediction-before-action receipt coverage on the controlled simulator track.
- On matched dependent tasks, at least +10 percentage points in final-chain success over no-memory, with paired cluster-aware confidence intervals excluding zero; stretch target +15 pp.
- **AMA-Bench:** reproduce AMA-Agent's paper protocol and 57.22% reference first; then target at least 62.2% under the same exact split, answer/evaluator setup and action/tool budget, with paired uncertainty excluding zero. The published 57.22% and +11.16 pp belong to AMA-Bench only; do not compare with conversational QA scores.
- **Continual Learning Exam:** reproduce the paper's best 47.7% level-completion result first; then target at least 52.7% of progressive levels and at least 28% of tasks completing all five levels, with equal feedback/revision budget and no-memory / replay / memory ablations. These are stretch targets against a preprint result, not expected deployment rates.
- **WorldMemArena:** publish separate write, maintain, retrieve and use results. Target a measurable improvement in each stage under the same split/backbone/reader budget, while keeping worst-stratum performance non-inferior within a pre-registered 1 percentage point. A stronger write score may not compensate for weaker maintenance or use.
- At equal action/model-call budget, improve cumulative regret over episodic-summary and similarity-retrieval baselines.
- Under regime shift, lower worst-group regret than unguarded causal-graph reuse, without unacceptable abstention inflation.
- Compression preserves at least 99% of decisions on a frozen proof-sensitive query panel at 10x memory reduction; individually explain all divergent cases. This is a stretch target, not a current result.
- Publish per-trajectory receipts, exact commit, model/provider revisions, environment seed digests and raw predictions/outcomes.

Small demonstrations such as 20-task runs are useful debugging evidence, not sufficient superiority claims. Report independent scenario/trajectory cluster counts, paired uncertainty and worst-group performance.

## 8. Relationship to existing programme

- #8877 / PR #8877: causal memory-to-action scorer and intervention protocol. MACE supplies interdependent multi-session memory formation/transfer tasks and reuses its action/outcome receipt discipline.
- #8427 / PR #8428: cost-quality frontier and matched baseline matrix.
- #9490 / PR #9491: proof-carrying tool arguments and separate effect authorization.
- #9426: proof-carrying answer claims.
- #9427: persistent bitemporal proof DAG and replay.
- #9447: adversarial epistemic conformance suite.
- #9294: industry baseline ledger.

Do not create a parallel action-authority kernel or another generic scorecard. The missing research object is the causal experience transition and its downstream transfer validity.

## 9. Deliverables

- Benchmark protocol and versioned trace/transition/outcome schemas.
- Synthetic causal-world generator with independent hidden oracle.
- MemoryArena and AMA-Bench adapters and official-score-preserving runners.
- Deterministic scenario corpus with decisive/irrelevant intervention pairs.
- Baseline matrix and ablation runner.
- Machine-readable trajectory receipts and independent evaluator.
- Rust-first, dependency-light simulator/protocol core.

## 10. Executable MACE-001 v0.1 evaluator

The branch now contains a dependency-free Rust evaluator package at `research/mace-001`:

- Rust 1.96.0 is pinned and the package has an isolated Cargo workspace with no third-party dependencies.
- Agent-visible task/payload TSV and evaluator-only oracle TSV are separate inputs.
- `audit` enforces one-to-one case coverage, opaque IDs, evaluator-label leakage sentinels, task/payload binding, exact required evidence markers, one anchor plus one irrelevant control per pair, identical paired prompts, identical shared state/objective/budget fields, exactly one declared irrelevant context addition, exactly one appended irrelevant history event, and agreement with the independent structural-causal reference model.
- The SCM reference contains four deterministic toy mechanisms: a common-cause confounder whose valve intervention effect is zero; a pump-fixed valve intervention with four noiseless replications per arm; a cooling-induced thermal regime shift; and a relay command that is accepted without changing the observed external state.
- `score` separately reports causal-status, transfer, action and completion agreement, plus four lifecycle-stage agreement counts: write, maintain, retrieve and use.
- Stage safety violations are non-compensatory. Examples include storing a qualified causal rule where only association is supported, retaining a transition after a relevant regime change, retrieving a transition as currently applicable when it needs a regime warning, or reusing an experience where requalification/reconciliation is required.
- The report distinguishes `hard_gate_pass` (no defined epistemic safety violation) from `full_conformance_pass` (hard gates pass **and** all four core decisions plus all four lifecycle stages match on every case). A safe under-retrieval or other safe-but-incorrect result may pass the safety gate but cannot pass full conformance.
- The output receipt binds the exact input/oracle/candidate bytes with SHA-256.
- Non-compensatory gates cannot be passed by a high average on the other dimensions.
- `src/scm_oracle.rs` supplies an independent deterministic reference mechanism for the four smoke families: common-cause confounding, pump-fixed valve intervention, cooling regime shift and accepted-but-unrealized relay action. Audit rejects evaluator labels that disagree with this model.
- The controlled-intervention fixture includes four deterministic repetitions per arm. This is a noiseless simulator contract, not a statistical estimate for a real physical system.
- A synthetic perfect-control trace validates the scorer's positive path; it is not a Broca output or a benchmark result.
- A path-filtered GitHub Actions workflow runs unit tests, audit and the synthetic score control.

The Rust unit suite contains 23 main-evaluator tests and 5 SCM tests, including mutations for false causal claims, invalid transfer, outcome overclaim, safe-but-incorrect decisions, relevant current-state changes in a control, extra irrelevant metadata, changed irrelevant events, missing intervention replications, anchor/control inconsistency, missing/duplicate/unknown rows, label leakage, prompt mismatch, answer-key/SCM disagreement, malformed candidate data and SHA-256 standard test vectors.

**Verification boundary:** code, workflow and tests are committed, but the Rust suite has not been compiled or executed in this session. The GitHub Actions workflow is an exact-head validation path; until a completed non-skipped run is observed, no Rust test pass is claimed. The v0.1 parser validates the compact TSV/payload contract but does not run a general JSON Schema validator. The candidate's prediction-before-action flag and record ID are assertions; they are not authenticated timestamps or signed receipts.

## 11. Claim boundary

This is a research specification plus a first executable conformance evaluator. It does not establish that current Broca learns causal effects from its own actions, that observational logs identify causality, that a causal graph guarantees policy transfer, or that Broca beats AMA-Agent or MemoryArena baselines. SOTA claims require matched official evaluation and independently reproduced outcomes.

## References

- MemoryArena, ICML 2026: https://proceedings.mlr.press/v306/he26am.html
- AMA-Bench, ICML 2026: https://proceedings.mlr.press/v306/zhao26bs.html
- AMA-Bench project: https://ama-bench.github.io/
- UltraHorizon, ICML 2026: https://proceedings.mlr.press/v306/luo26ai.html
- WorldMemArena, arXiv 2026: https://arxiv.org/abs/2605.29341
- Continual Learning Exam (CLE), arXiv 2026: https://arxiv.org/abs/2610.33956
