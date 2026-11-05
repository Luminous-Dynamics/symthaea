# Symthaea Recursive Self-Improvement Programme v1

**Date:** 2026-11-04  
**Status:** Research and engineering contract; proposed benchmark targets are not measured results.  
**Implementation companion:** `src/consciousness/recursive_improvement/candidate_archive.rs`

## Executive thesis

Symthaea should be developed as an experimentally governed evolutionary system, not as one increasingly autonomous self-editing agent.

A genuine recursive-self-improvement claim requires more than self-reflection, tool use, memory accumulation, or a code-writing loop. A candidate must improve a measurable capability; the evidence must survive independent evaluation; the accepted candidate must become the parent of a later improvement; and the improvement process itself must eventually become more effective.

The programme therefore separates five research objects:

1. **Capability improvement:** does a candidate solve more or harder problems?
2. **Agent improvement:** does changing the agent scaffold improve its research or execution behaviour?
3. **Algorithm improvement:** does changing the learning or optimization algorithm improve the capability/compute frontier?
4. **Meta-improvement:** does the system become better at discovering, testing, and selecting improvements?
5. **Governance and retention:** are gains independently established without losing protected capabilities or weakening the evaluation boundary?

These are distinct claims and must have distinct evidence.

## 1. RSI claim ladder

Use the narrowest label supported by the evidence.

| Level | Name | Minimum evidence |
|---|---|---|
| R0 | Adaptive system | Learns from observations or experience within a fixed architecture |
| R1 | Self-refining system | Changes prompts, memory, policies, tools, or repair rules and measures the effect |
| R2 | Self-modifying agent | Modifies executable agent/scaffold code; candidate changes are tested independently |
| R3 | Recursive agent improvement | An accepted successor becomes the parent of another independently evaluated improvement |
| R4 | Meta-recursive improvement | The system improves its own improvement process and shows better discovery efficiency |
| R5 | Algorithmic RSI | It modifies learning/training/optimization algorithms and transfers the gains to subsequently trained systems |

No level implies consciousness, sentience, general intelligence, or safety. Φ and other consciousness-relevant indicators are not substitutes for task performance, generalization, calibration, or safety evidence.

## 2. First implemented substrate: context-bound Pareto archive

The companion Rust module supplies a deliberately small first vertical slice.

### What it enforces

- Every archive belongs to one immutable evaluation context.
- The context binds benchmark identity/version, partition, split, evaluator digest, corpus digest, metric-schema digest, resource-budget digest, and seed-manifest digest.
- Candidate IDs are unique.
- Every candidate can declare zero, one, or multiple parents; all parents must already exist in the archive, so the stored multi-parent lineage is an acyclic DAG by construction.
- Candidate source-tree and result-receipt digests must be canonical 64-character lowercase hexadecimal strings.
- Capability vectors are non-empty, finite, normalized to [0, 1], and use a consistent set of metric IDs.
- Higher is better for every normalized capability dimension.
- Dominated candidates are retained for auditability; only the frontier view changes.
- Trade-offs remain visible. The archive does not invent a composite winner.

### What it deliberately does not do

- It does not run an evaluator or verify the truth of benchmark measurements.
- It does not prove that a digest is authentic. A digest is an identity commitment, not a signature or proof.
- It does not protect a sealed holdout from being read or reused. The partition field is metadata, not access control.
- It does not authorize promotion, deployment, network access, or source mutation.
- It does not establish that an archive frontier candidate is safe, statistically superior, or generally more intelligent.

Those powers require separate, independently operated components.

### Why Pareto retention matters

A single weighted score allows a coding gain to conceal a reasoning regression, a generalization loss, or a safety regression. A Pareto archive preserves candidates that are non-dominated across a frozen capability vector. A coding specialist and a more general candidate can both survive as useful stepping stones.

The archive is intentionally scoped to one exact evaluation context. Results from a different dataset, split, evaluator, budget, metric schema, or seed manifest belong in a different archive and must not be compared as if they were matched.

## 3. Paradigm shifts to prioritize

### 3.1 From a champion chain to an open-ended candidate population

Maintain a historical archive and a non-dominated frontier. Sample parent candidates from the frontier and, selectively, from historically useful stepping stones. Keep diversity; do not always mutate only the latest candidate.

### 3.2 From self-grading to evaluator independence

The candidate generator must not control the final evaluator, protected test data, scoring weights, acceptance thresholds, or trusted verification tools. Development feedback may be visible; selection and final holdout evidence must be controlled independently.

### 3.3 From scalar performance to a capability vector

The frozen metric schema should include, where relevant:

- coding and repair correctness;
- reasoning and mathematical problem solving;
- scientific hypothesis quality and experimental validity;
- out-of-distribution generalization;
- calibration and uncertainty quality;
- capability retention / catastrophic forgetting;
- safety-policy and constraint adherence;
- normalized time, compute, memory, and energy efficiency.

Normalize metrics before the experiment and bind their definitions to the metric-schema digest. Retain raw costs alongside normalized values in the underlying evidence; normalization must not hide absolute resource consumption.

### 3.4 From task improvement to improvement efficiency

Measure how much independently confirmed capability gain is obtained per unit of compute, time, and energy. Also measure experiment yield: valid hypotheses, evaluated candidates, accepted candidates, and useful frontier discoveries per fixed budget.

### 3.5 From one-shot self-modification to causal scientific search

Every change should have a falsifiable hypothesis, a predicted failure mode, a controlled intervention, a matched baseline, negative controls, and a retained result—even when the result is negative. Distinguish correlation in telemetry from an intervention that caused the improvement.

### 3.6 From capability accumulation to capability preservation

An accepted coding gain does not justify losing memory reliability, calibration, reasoning, safety constraints, or another protected capability. Keep the full regression vector and retain prior candidates.

## 4. Benchmark programme

Use multiple complementary task families. Do not reduce the programme to SWE-bench.

| Track | Research question | Main measurement |
|---|---|---|
| RSI-CODE | Can Symthaea improve its own coding agent? | Hidden issue resolution, patch correctness, regression rate, repair attempts, cost |
| RSI-ARCH | Can it improve search, planning, memory, tool selection, and context management? | Held-out task success, recovery, generalization, time-to-solution |
| RSI-ALGO | Can it discover better learning or optimization algorithms? | Capability at matched training/inference compute |
| RSI-SCIENCE | Can it improve scientific research behaviour? | Hypothesis quality, experimental validity, replication, discovery yield |
| RSI-META | Can it improve its own improvement process? | Valid discoveries per fixed budget, evaluation efficiency, improvement yield |
| RSI-RETENTION | Does it preserve previously demonstrated capabilities? | Per-domain regression and forgetting profile |

### Required baselines

Every experiment should compare, under the same model/backend and resource budget where possible:

1. **Frozen baseline:** the original agent never changes.
2. **Fixed meta-agent:** the original agent repeatedly edits candidates but does not improve its own editing process.
3. **Single-champion chain:** each candidate is derived from the latest selected candidate only.
4. **Archive-based search:** retain diverse non-dominated candidates and explore from the archive.
5. **Human-engineered reference:** a competent, independently constructed agent or algorithm.
6. **Ablations:** remove memory, causal review, self-selected evaluation targets, candidate archive, or other claimed mechanisms one at a time.

This isolates whether progress comes from recursive modification, a stronger fixed base model, more test-time compute, data accumulation, prompt changes, or open-ended diversity.

### External reference points

- **Darwin Gödel Machine:** archive-based self-modification and empirical selection; reported 20% to 50% on SWE-bench and 14.2% to 30.7% on Polyglot.
- **HyperAgents / DGM-H (Meta, March 2026):** combines task-agent and meta-agent in one editable program, so the improvement procedure itself can evolve. It reports gains across coding, paper review, robotics reward design and Olympiad-math grading. This is the closest direct comparator for the proposed R4 meta-improvement target.
- **Mendel Gödel Machine (August 2026 preprint):** adds reaction-norm mutation from multi-task trajectories and cross-lineage hybridization from another lineage's results on the same task. This motivates controlled comparisons of single-trajectory mutation, multi-task mutation and cross-lineage transfer; reported results remain a preprint and should be independently reproduced.
- **AIDE² (September 2026):** a self-modifying research agent with seven reported successive improvements in an eight-day autonomous run and gains on four held-out task families.
- **AI4AI-Bench:** isolates training-algorithm redesign; the paper reports a best normalized score of 0.250 across ten algorithm-design tasks. Keep this distinct from task-agent and meta-policy improvement.
- **RSIBench-Data:** shows that continued search can regress after a prior best; checkpoint retention and best-so-far selection are essential.
- **SWE-bench Verified:** useful only as a secondary coding reference. Results depend on model/scaffold configuration and the benchmark has known contamination and test-quality limitations.

Published results are context, not directly comparable Symthaea scores. Match task versions, models, scaffolds, budgets, and evaluator protocols before making a leaderboard claim.

## 5. Metrics and proposed gates

The primary result is a vector of paired, per-task outcomes, not one opaque composite score.

### Primary measurements

- **Target capability delta:** candidate minus parent on a preregistered target dimension.
- **Transfer delta:** candidate minus parent on held-out task families and out-of-distribution tasks.
- **Retention floor:** worst protected capability change, not only average retention.
- **Discovery yield:** independently confirmed improvements per fixed compute/time budget.
- **Improvement efficiency:** target capability gain per normalized resource budget, with raw compute/time/energy reported.
- **Regression burden:** count and severity of failed tasks, broken invariants, and lost capabilities.
- **Search diversity:** number of distinct non-dominated candidates and distinct mechanism families retained.
- **Meta-improvement:** change in discovery yield and cost efficiency across successive generations.

### Proposed initial gates

These are goals to preregister, not claims that Symthaea already meets them.

| Gate | Proposed criterion |
|---|---|
| Evidence integrity | 100% of measured candidates bind exact source, evaluator, corpus, metric schema, budgets, and seeds |
| Matched comparison | Parent and child use the same benchmark context and case set |
| Capability improvement | Positive target-dimension effect on an independent selection/confirmation set |
| Retention | No material regression on any protected capability; publish the full per-domain vector |
| Generalization | Improvement transfers to at least one held-out task family not used for candidate selection |
| Recursive closure | At least two accepted successor generations, each evaluated against its direct parent |
| Meta-improvement | Later generations improve discovery yield or capability gain per budget against a fixed-meta-agent baseline |
| Reproducibility | Confirm the primary finding across independent seeds and retain all candidate/evaluator identities |

For small paired test sets, report exact paired tests and confidence intervals; do not infer equivalence from a non-significant result. Correct for repeated candidate selection and multiple comparisons. A final holdout should be used only through a separately controlled evaluator service.

## 6. Benchmark protocol

Each candidate experiment should bind:

1. Candidate source-tree digest and parent candidate identity.
2. Exact benchmark ID/version, split, and per-case IDs.
3. Evaluator source/binary digest and trusted toolchain identity.
4. Corpus and seed-manifest digests.
5. Metric-schema digest, including direction and normalization.
6. Resource-budget digest, including model, tokens, accelerator, runtime, memory and energy accounting.
7. Raw per-case outcomes, failures, and resource use.
8. Receipt digest plus an independently retained signature or append-only provenance record.

### Data separation

- Development data: freely inspectable and used for debugging.
- Validation data: used for candidate selection under a fixed protocol.
- Sealed holdout: never exposed to the candidate generator; accessed only by the independent evaluator.

A digest does not establish that a dataset was actually hidden. Enforce access using separate credentials/processes, restricted filesystem and network boundaries, evaluator logs, and one-shot or tightly budgeted holdout access. Record every attempted evaluation, not only successful ones.

### Anti-gaming controls

- Candidate code cannot modify the evaluator, benchmark corpus, acceptance policy, test launcher, or trusted verifier.
- Include negative controls and deliberately broken candidates.
- Compare against a compute-matched frozen baseline.
- Use fresh task families to test for memorization.
- Preserve unsuccessful candidates and all attempted results.
- Require raw per-case data and resource accounting; aggregate scores alone are insufficient.
- Re-run key outcomes with a structurally independent evaluator where feasible.

## 7. Safety and autonomy boundary

Start with sandboxed research runs, not unrestricted self-modification of production Symthaea.

The candidate may propose changes to its agent, memory, search policy, representation, training algorithm, and experiment strategy. It may not grant itself new privileges or rewrite the trusted evaluator, benchmark access controls, promotion policy, safety kernel, signing keys, or audit ledger.

Promotion and deployment remain separate from candidate generation. Candidate generation can be autonomous within a bounded sandbox; production promotion requires independent qualification and an explicit authorization boundary.

Preserve reversible snapshots, deterministic builds where possible, resource caps, network-egress restrictions, and a complete event log. A failed or ambiguous qualification is not a pass.

## 8. Engineering sequence

### Phase 0 — Archive and contract

- [x] Add a validating Rust representation for candidate identities and capability vectors.
- [x] Bind an archive to one exact evaluation context.
- [x] Retain dominated candidates while exposing a non-dominated frontier.
- [x] Reject inconsistent metric dimensions, duplicate/self/missing parents, and out-of-order lineage.
- [x] Represent multi-parent hybrid candidates with canonical parent ordering; compare a hybrid separately against each declared parent.
- [ ] Run exact-head Rust formatting, compilation, tests, and Clippy; no runner result is inferred from source creation.

### Phase 1 — Receipt and paired-evaluation kernel

- [x] Add an in-memory paired-analysis adapter over the shared benchmark-run receipt contract (`paired_evaluation.rs`); do not introduce a competing persisted receipt schema.
- [x] Require exact evaluation context, candidate/parent identity, source-tree digest, shared receipt digest, case IDs, and normalized metric dimensions in the in-memory view.
- [x] Derive archive capability summaries from sorted per-case evidence rather than caller-supplied aggregate scores.
- [x] Report per-metric paired wins/losses/ties, mean paired delta, and exact one-sided sign-test p-values.
- [x] Reject empty/duplicate/mismatched cases, metric-schema drift, context mismatch, incorrect parent lineage, and invalid tie tolerances.
- [ ] Add authenticated evaluator receipts and retained raw outcomes; digests alone do not authenticate a measurement.
- [ ] Add confidence intervals, multiplicity correction, practical-effect thresholds, and non-inferiority gates.
- [ ] Add an independently operated sealed-holdout evaluator/access boundary. The partition enum is descriptive metadata, not access control.

**Evidence integration:** the adapter must consume validated task-level rows from the shared benchmark-run receipt contract (#10314) and the applicable DSSE/regrade verification pipeline (#10167/#10370). It is not an evidence format or authenticator of its own.

**Statistical limitation:** the exact sign test measures directional consistency among non-tied paired cases under an independent-pair-sign assumption. It does not measure effect magnitude (the mean delta is reported separately), establish practical significance, correct for repeated adaptive selection or multiple metrics, prove equivalence when non-significant, or authorize promotion. Cases clustered by repository, task family, project, or run require independent-group or cluster-aware confirmation; counting correlated cases as independent can overstate evidence.

### Phase 2 — Controlled coding-agent evolution

- Connect the archive to the coding agent through a candidate adapter.
- Keep generated repairs and code edits in isolated worktrees/containers.
- Compare mutation candidates with their parent; compare multi-parent hybrids separately with every declared parent and the frozen baseline.
- Promote only after independent tests, retention gates, and held-out evaluation.
- Keep a fixed-meta-agent and single-champion baseline.

### Phase 3 — Meta-improvement

- [ ] Execute the dedicated matched-budget programme in [SYM-RSI-META-001 #10497](https://github.com/Luminous-Dynamics/symthaea/issues/10497).
- Let Symthaea improve experiment selection, hypothesis generation, causal review, and memory/context management.
- Compare clonal/single-trajectory mutation, reaction-norm mutation over multiple task histories, cross-lineage hybridization, archive-based fixed-meta search, and meta-evolving archive search.
- Measure independently confirmed discoveries per fixed total budget, transfer of the meta-level change, capability retention, and best-so-far versus final-candidate quality.
- Retain all candidates on a Pareto frontier rather than selecting by one blended score.

### Phase 4 — Algorithmic RSI

- Introduce bounded experiments that modify training objectives, update rules, representation learning, and architecture search.
- Train from scratch under identical data and compute budgets.
- Compare the resulting models on held-out tasks, not only training loss.
- Only claim algorithmic RSI when the improved learning procedure produces a better successor system.

## 9. Immediate next work

1. Implement the paired per-case comparison and statistical qualification kernel.
2. Design a sealed evaluator adapter that Symthaea cannot edit or query directly.
3. Connect the candidate archive to the existing replay-grounded SYM-RSI experiment branch without assuming draft PRs are integrated into main.
4. Wire the coding-agent self-modification path only after candidate isolation and evaluator separation exist.
5. Run a small native Rust benchmark suite first; expand to external coding and algorithm-design benchmarks when the measurement protocol is stable.

## Claim discipline

Source code, tests authored, tests executed, benchmark measured, independently reproduced, and generalized are separate evidence states.

No queued workflow is a pass. No source-authored receipt proves the evaluator was honest. No digest alone proves authenticity. No higher Φ, internal coherence signal, or self-reported confidence is evidence of task capability. No single benchmark improvement is proof of recursive self-improvement.

The programme's target is simple to state and difficult to fake:

> A successor must independently outperform its parent, preserve the capabilities we require, generalize beyond the selection tasks, and become demonstrably better at finding the next improvement.


## 10. Cluster-aware statistical qualification — issue #10513

The original paired adapter retains per-case wins/losses/ties and its exact sign test, but that test assumes independent non-tied case-pair signs. Cases from one repository, project, task family, or run are often correlated; treating them as independent can create pseudo-replication.

The additive `ClusteredPairedComparison` view addresses this without changing the shared receipt format:

- The caller supplies a case-ID → cluster-ID map that must cover exactly the matched case set.
- Cluster IDs are validated for non-empty, non-control-character identifiers.
- Paired deltas are averaged within each cluster before directional wins/losses/ties are counted.
- Both pooled case-weighted mean delta and equal-cluster-weighted mean delta are reported. These are different estimands and must not be conflated.
- The exact one-sided sign-test diagnostic operates on non-tied cluster means. It is withheld when fewer than two non-tied clusters remain. This adapter reports point estimates and a directional exact-sign diagnostic; it does **not** yet produce cluster-bootstrap confidence intervals, uncertainty for Pareto frontier membership, or practical-effect/non-inferiority decisions. Those are owned by the existing paired uncertainty issue #8438, while broader task-population and crossed-family uncertainty belongs to #9433. Do not duplicate those statistical backends in the RSI module.
- Holm step-down family-wise adjustment uses the full predeclared metric-schema size for this one comparison. Metrics without enough independent clusters to produce a p-value have no reported adjusted p-value, but still count conservatively as p=1 in the family size. This does **not** correct across candidates, adaptive search, generations, repeated runs, or a broader preregistered family.

The cluster mapping must originate in the frozen task-family/corpus manifest and be covered by the shared receipt/corpus identity. The in-memory adapter validates mapping shape; it cannot authenticate mapping provenance or prove that clusters are genuinely independent. The independent unit must be chosen before confirmation, not after inspecting outcomes.

This does not replace the original per-case results. It exposes a second analysis at a defensible independent unit, reducing the risk that many correlated cases from one project dominate the evidence. Small numbers of independent clusters remain low-power; cluster-aware analysis does not manufacture sample size. Until #8438 supplies qualified cluster-level intervals and #9433 supplies defensible target-population inference where appropriate, G3/G4 claims remain descriptive/directional and cannot be upgraded to population-generalization claims.

### Exact evidence boundary

- This is analysis code, not a new persisted receipt schema.
- The input rows and cluster map still require upstream validation through #10314 / #10167 / #10370.
- Holm adjustment is local to one metric family and one comparison. Sequential candidate selection and generation-level inference remain a separate preregistered confirmation problem.
- No p-value is a practical-effect threshold, non-inferiority result, or promotion authorization.
- Tests are authored in the PR but remain unexecuted until exact-head Rust qualification is available.


## 11. Newest field comparator: RSI-Env (arXiv:2611.01736, 2026-11-01)

RSI-Env is the newest directly relevant public comparator located for this research pass. It models a system as model parameters, executable harness, and persistent research state, with resources managed by an external Host Controller. The evolving target system itself becomes the Improver in subsequent rounds. It varies research autonomy across Open-General, Capability-Directed, and Benchmark-Guided settings, and keeps controller-side validity, versioning, and final capability evaluation outside the editable environment.

The paper reports seven tasks from Qwen3.8-27B with a 30-hour allocation split into ten three-hour rounds. Self's Open-General run changes GPQA Diamond from 65.2% to 73.2% and AIME 2026 from 53.3% to 56.7%, but also changes SimpleQA from 82.0% to 77.2% and Terminal-Bench 2.1 pass@3 from 12.4% to 9.0%. The Capability-Directed Knowledge run raises SimpleQA from 82.0% to 88.0% while GPQA Diamond falls from 65.2% to 53.0%. The paper's Knowledge Self case study reports successive SimpleQA versions at 82.0%, 85.0%, and 88.0%.

These are author-reported preprint results, not independently reproduced measurements. The authors explicitly report uneven cross-capability retention. The Self runs use roughly 25–29 hours in the Open-General and Capability-Directed cases, while the external Teacher runs use roughly 1–7 hours under the same 30-hour allocation; the improvers are also different models. Therefore the paper is a strong architecture/evaluation comparator, but its Self-vs-Teacher scores are not a clean compute-matched leaderboard. Its three-trial avg@3/pass@3 summaries also do not substitute for clustered, multi-seed uncertainty estimates. The reported H1/M1 Knowledge continuation shows a successor doing further work and being followed by a further gain; it is not a controlled no-recursion ablation proving recursion caused the second gain.

### RSI evaluation should be a matrix, not a single ladder

Keep the R0–R5 claim ladder for claim discipline, but organize experiments across orthogonal axes:

1. **Editable system surface:** harness/scaffold; memory and tool policy; model weights; training objective/update algorithm; experiment-selection and meta-improvement procedure.
2. **Research autonomy:** Benchmark-Guided; Capability-Directed; Open-General. Freeze the autonomy condition independently of the task domain.
3. **Inheritance depth:** one accepted update; successive inherited updates; inherited changes to the improvement process; changes to the algorithm that trains successor models.
4. **Evidence sovereignty:** external controller; exact candidate/source/evaluator identities; access-controlled confirmation; independent regrade; matched total resource accounting.

This prevents conflating harness improvement with algorithmic RSI and prevents a benchmark-specific gain from being called general RSI. It also makes direct comparison to RSI-Env, HyperAgents, AIDE², and AI4AI-Bench more precise: each should be compared only on the overlapping system surface, autonomy condition, inheritance depth, and budget regime.

### Symthaea's differentiating target

Do not try to beat RSI-Env by maximizing one headline benchmark. Target **capability-preserving open-ended research autonomy**: an evolving model/harness/meta-process that can formulate goals, propose falsifiable interventions, retain failed and successful research state, and independently transfer gains while keeping every protected capability above its pre-registered floor. Report the full vector, worst regression, best-so-far and final checkpoint, per-generation cost, and independently confirmed discoveries per matched total budget.

Immediate implications:

- Add the three autonomy settings to the R4 meta-RSI protocol, with identical infrastructure and controlled evaluation permissions.
- Treat model-weight updates and harness edits as separate treatments and as a coupled treatment; do not attribute a gain to the wrong edit surface.
- Preserve a frozen external Host Controller for candidate registration, source identity, resource limits, sealed evaluation, and rollback. The candidate must not modify that controller.
- Evaluate full capability retention on independent units and fresh task families. Use the cluster-aware view from issue #10513 as an analysis adapter, not as authentication.
- Compare against the existing deterministic SYM-RSI-001 replay substrate as a low-cost mechanism baseline, then test transfer into real R&D/coding domains. The old replay stack remains a separate draft lineage and is not integrated into `main`.

Source: Qu et al., *RSI-Env: An Open-Ended Environment for Recursive Self-Improvement of Agents*, arXiv:2611.01736v1, 2026-11-01. https://arxiv.org/abs/2611.01736


## 12. Generalization ladder and external RSI benchmark matrix

SelfImprove-Bench (Xu et al., arXiv:2611.02797v1, 2026-11-02) gives a useful explicit transfer ladder. Symthaea should preserve its R0–R5 claim ladder but score each system update on the following generalization levels:

- **G0 — Replay:** improvement on the same replayed task instances.
- **G1 — Same-task transfer:** held-out instances from the same task/game at the training difficulty.
- **G2 — Difficulty transfer:** harder held-out instances of the same task/game.
- **G3 — Within-capability transfer:** different tasks in the same capability category.
- **G4 — Cross-capability transfer:** tasks in a different capability category.

SelfImprove-Bench reports that these levels do not move monotonically: G1 improvement may fail on harder G2 instances; within-category transfer can be directional; and rankings between methods can reverse across transfer sets. Its HCI (Headroom-Closed Index) is HCI = (successor mean − baseline mean) / (reference mean − baseline mean). Always publish raw baseline/successor/reference scores and per-task results alongside HCI. HCI is normalized headroom closed, not an effect size or confidence interval; it can exceed 100% where successor performance surpasses the declared reference. The benchmark's controllable games support causal/mechanism analysis, but do not substitute for engineering or scientific transfer.

### External benchmark tracks — separate scorecards

| Track | What it tests | Use in Symthaea RSI |
|---|---|---|
| SelfImprove-Bench G0–G4 | Replay, same-task holdout, harder tasks, within-capability and cross-capability transfer | Primary generalization mechanism track; preserve all G-level results |
| RSI-Env | Open-General, Capability-Directed and Benchmark-Guided model/harness self-improvement under an external controller | Autonomy-mode and capability-retention comparison |
| AIDE² selection + held-out R&D tasks | Repeated research-agent rewrites across ML engineering, heuristic algorithms and harness engineering | R3 repeated improvement and independent transfer |
| AIRS-Bench | 20 research-science tasks from ML papers across language modelling, mathematics, bioinformatics and time series; idea→experiment→analysis→refinement | Broad scientific-research lifecycle track |
| FIRE-Bench | Rediscovery of established ML research findings through independently designed experiments and evidence-backed conclusions | Scientific correctness/replicability track, not prose-quality judging |
| AI4AI-Bench | Actual training-algorithm redesign in 10 frozen research repositories under a four-hour design window and clean-start training evaluation | R5 algorithmic RSI; separate algorithm changes from harness, data and hyperparameter changes |
| Coding Agent Index v2.0 components | FrontierCode v1.1, Terminal-Bench 4.0 and SWE-Atlas-QnA-AA | External software-engineering transfer; report each suite plus cost/reliability, not only index |

### Prospective qualification targets (not measured results)

These are research acceptance targets to preregister before any confirmatory run; they are not claims about current Symthaea performance:

1. **R3 repeated improvement:** at least five successive independently accepted improvements, each evaluated under the same fixed total per-candidate budget; retain all rejected proposals and best-so-far checkpoints.
2. **Transfer:** positive raw held-out delta at G1 and G2; positive aggregate G3 and G4 with a cluster-aware lower confidence bound above zero across a preregistered minimum of five independent task families/categories. Publish the full distribution; do not claim success from a single aggregate or one lucky domain.
3. **Capability retention:** every protected metric stays above a pre-registered non-inferiority floor and per-metric regression margin. Example margins must be justified per metric and frozen before selection; do not universalize a 2-point margin across incomparable scales.
4. **R4 meta-improvement:** more independent, transfer-confirmed improvements per total resource budget than frozen-meta, champion-only, fixed-meta archive, and matched random/human-search baselines. Resource accounting includes failed proposals, evaluations, training, inference, data access and retries.
5. **R5 algorithmic improvement:** the patch must alter the learning objective, supervision, update rule or another declared learning mechanism; a harness, checkpointing, data-budget or hyperparameter-only change is a separate treatment. Train from a clean start and evaluate with a fixed hidden grader. The initial external aim is to exceed AI4AI-Bench's reported best normalized score of 0.250 under its exact frozen protocol, but no comparison is valid unless tasks, base models, B300-hour budgets and evaluation procedure match.
6. **Research reliability:** on FIRE-Bench/AIRS-Bench, report experiment validity, independent evidence correctness, replication, uncertainty, unsupported claims and run variance—not only final score or LLM-judge writing quality.
7. **Engineering transfer:** compare the same foundation model across a frozen baseline harness and Symthaea-enabled harness on FrontierCode, Terminal-Bench 4.0 and SWE-Atlas-QnA-AA, holding model version, reasoning setting, attempts, tools, budget and environment constant. Report per-suite pass@1 and pass@3 as appropriate, cost, tokens, wall-clock, refusals, reward-hack flags and verified patch quality.

### Paradigm shift

Treat improvement as a **causal research process with an audited transfer frontier**, not a chain of increasingly high scores. Each proposal must declare the mechanism it changes, predicted effects and possible regressions before evaluation; retain the complete trajectory, resource use and negative findings; and attribute observed changes to model, harness, memory, algorithm and meta-policy edits using matched ablations. The external controller—not the evolving agent—owns candidate identity, task release, resource caps, evaluator, independent regrade, holdout access and rollback.

Sources:
- Xu et al., *SelfImprove-Bench: Evaluating Generalization in Self-Improving Agents*, arXiv:2611.02797v1 (2026-11-02): https://arxiv.org/abs/2611.02797
- Meta AI, *AIRS-Bench* (2026-02-10): https://ai.meta.com/research/publications/airs-bench-a-suite-of-tasks-for-frontier-ai-research-science-agents/
- Wang et al., *FIRE-Bench*, PMLR 306 (2026): https://proceedings.mlr.press/v306/wang26d.html
- Chi et al., *AI4AI-Bench*, arXiv:2608.20318 (2026-08-20): https://arxiv.org/abs/2608.20318
- Artificial Analysis Coding Agent Index v2.0 methodology (2026-10): https://artificialanalysis.ai/methodology/coding-agents-benchmarking/


## 13. Research-agent throughput and evaluation-noise controls: AIRA₂

Meta AI's AIRA₂ (published 2026-04-16) is an important industry comparator for the R3/R4 research-agent substrate. Its authors identify three bottlenecks: synchronous single-GPU execution, noisy validation-based selection over long search horizons, and fixed single-turn operators. Their architecture combines an asynchronous multi-GPU worker pool, Hidden Consistent Evaluation (HCE), and ReAct agents that dynamically scope actions and debug interactively. The Meta publication reports MLE-bench-30 mean Percentile Rank of 81.5% at 24 hours and 83.1% at 72 hours, versus 72.7% for the strongest baseline; on AIRS-Bench it exceeds human state-of-the-art on 6 of 20 tasks. These are the paper's own results on its own protocol, not Symthaea measurements or directly transferable scores. [Source: Meta AI publication, 2026-04-16](https://ai.meta.com/research/publications/aira-overcoming-bottlenecks-in-ai-research-agents/).

### What Symthaea should test rather than merely imitate

1. **Asynchronous experiment orchestration:** submit independent candidate/probe jobs without serially blocking on the slowest evaluation. Preserve exact candidate and experiment identities, bounded queues, cancellation, retry lineage, resource attribution, and deterministic collection order. Workers may propose and execute only within the external controller's permission; they cannot edit the evaluator or trust roots.
2. **Hidden-consistent evaluation:** distinguish cheap/public development feedback from a frozen, independently controlled confirmation signal. Repeat or aggregate noisy evaluations with a predeclared estimator; retain every replicate and its resource use. A score used for candidate selection must not also be presented as independent confirmation. Measure selection regret and false-winner rate over long search horizons, not just evaluator repeatability.
3. **Interactive research operators:** compare fixed single-turn edit/score operators with ReAct-style diagnose → formulate a falsifiable hypothesis → choose a bounded probe → inspect evidence → debug/refine. Each step should emit a trace with the hypothesis, predicted observation, selected probe, actual result, and whether the hypothesis was falsified.
4. **Joint ablation:** test async pool only, HCE only, interactive operator only, all three, and frozen baseline. Use the same foundation model, accessible data, total GPU/CPU hours, candidate-generation budget, evaluator calls, and allowed toolset. A second operational comparison may fix wall-clock instead, but must report the extra compute/energy consumed. Never compare equal wall-clock with unequal compute as if it were a compute-matched result.

### RSI-specific measurements

- Independently confirmed, held-out improvements per total compute-hour and per wall-clock hour (separate scorecards).
- Best-so-far trajectory, final checkpoint, number of candidates proposed/evaluated/retained/rejected, and improvement survival after fresh evaluation.
- Worker utilization, queue delay, GPU/CPU hours, peak memory, total tokens, energy and cost; retries and failed proposals remain in the denominator.
- Evaluation noise, replicate variance, HCE repeatability, false-winner rate, selection regret and post-selection reversal rate.
- Interactive operator's hypothesis falsification rate, useful-probe rate, debugging iterations, patch collateral and independently verified outcome.
- Transfer across SelfImprove-Bench G0–G4 and AIRS-Bench task families, without using final confirmation items during search.

### Hardware-aware execution strategy

The local Symthaea development target includes a single RTX 2070 Laptop GPU (~8 GB VRAM). It should not claim AIRA₂'s multi-GPU throughput scaling from asynchronous scheduling alone. An honest first implementation can overlap CPU-side static analysis, repository indexing, fixture generation, scoring and artifact checks while keeping GPU inference/training concurrency bounded by measured VRAM. For any remote/multi-GPU experiment, charge all workers and queue retries to the same all-in budget. Publish quality-versus-compute and quality-versus-wall-clock curves separately.

### Proposed target

Under a frozen, repeated-run research task suite, the combined system should yield more **independently confirmed generalizing improvements per fixed total compute budget** than both a synchronous interactive baseline and an asynchronous fixed-operator baseline, with no protected capability regression. In the equal-wall-clock track it should improve time-to-first-confirmed-gain without hiding increased compute, energy or cost. No numeric superiority threshold should be frozen until the baseline variance and practical effect are measured on a development set; the confirmatory threshold and power plan must then be preregistered.

This complements #10497 (meta-RSI comparison), #8438 (paired/cluster uncertainty), #9433 (population generalization), and #10521/#10533 (validated receipt composition). It does not create a second scheduler, receipt format, evaluator or runner-isolation boundary.


## 14. Complete task-universe composition and RSIGym comparison

The RSI paired adapter now has a second, additive in-memory layer: `CompositeCandidateEvaluationView` and `CompositePairedComparison`.

This layer addresses a denominator failure the original matched-case adapter cannot detect: if both parent and successor omit the same eligible task, the two remaining case sets still match. The composite view therefore requires every metric-specific run to contain exactly one outcome for every task in the frozen eligible set. It rejects duplicate, missing and unexpected rows, mismatched composite manifests, and mismatched parent/successor task universes.

Each task retains an explicit disposition: passed, failed, refused, abstained, unsupported, timed out, tool failed, ungraded, not attempted, or in progress. Numeric normalized grades are optional only for non-scored dispositions; they are never imputed as zero by this adapter. Per-metric output reports eligible count, scored count, score coverage, mean among scored cases, disposition counts, paired scored count, one-sided score counts, both-unscored count, paired wins/losses/ties, mean paired delta and the full disposition-transition table.

The paired mean is calculated only on tasks with grades on both arms. It is therefore a descriptive pairwise-observed estimand, not a complete-population effect. Coverage and unmatched grades are displayed separately. The new view emits no p-value; #8438 owns paired cluster-bootstrap uncertainty and Pareto-frontier stability, while #9433 owns population and task-family inference.

**Claim boundary:** this constructor checks task-universe completeness and numeric/disposition consistency, but it still accepts caller-projected rows. It does not invoke #10314's canonical receipt verifier, authenticate component run digests, verify DSSE/regrade, enforce isolation, or grant sealed-holdout access. The next integration step is a narrow adapter that projects validated receipt outputs into these types under #10521 / #10370. Until then, the type is a fail-closed structural analysis boundary, not proof of provenance.

### RSIGym: a necessary new comparator

RSIGym (arXiv:2610.10310, 2026-10-07) is another important 2026 comparator that should be in the R4/R5 test matrix. It provides reusable services for training, inference, rollout, evaluation and sandbox execution, with shared budget and permission control. Its Data, Harness and Joint tracks separate data changes, execution-harness changes and their combination. It defines RSI-Index as the mean fraction of remaining performance gap closed across five benchmarks spanning software engineering, terminal interaction, mathematics, scientific reasoning and skill tasks.

The authors report a best RSI-Index of 0.4809 for Opus 5 in independent Joint runs under a $500 platform-service budget per benchmark run. They report selected systems improving SWE-bench Verified from 17.67% to 50.33% and AIME from 31.67% to 97.78%. These are author-reported preprint results, not independently reproduced measurements. Its RSI-Index uses its own task set, ceiling definitions and budget accounting; it is not directly comparable with AI4AI-Bench's algorithm-only normalized score, RSI-Env's seven-task experiments, or SelfImprove-Bench's G0–G4 transfer levels.

RSIGym gives us a useful experimental factorization: Data-only, Harness-only, Joint, plus a frozen control. Symthaea should add a distinct Algorithm track to separate learning-rule changes from data/harness changes. All four should share total resource caps, exact model identity, task permissions, independent evaluator, clean-start successor training where required, and the same external controller. Report each benchmark lane and cost separately; never rank the systems only by an averaged RSI-Index.

### Immediate experimental design consequence

The next complete-evidence experiment should cross these factors rather than run a single undifferentiated self-improvement loop:

- **Edit surface:** Data, Harness, Learning Algorithm, Meta-Improvement Procedure, and Joint combinations.
- **Autonomy:** Benchmark-Guided, Capability-Directed, Open-General.
- **Transfer:** SelfImprove-Bench G0–G4, with real software/science families for G3/G4.
- **Inheritance:** one update, repeated inherited updates, and successor changes to the improvement procedure.
- **Evidence:** complete eligible task universe, all dispositions, matched total budget, external evaluator, qualified regrade, cluster uncertainty and non-inferiority floors.

Source: Meng et al., *RSIGym: A Flexible Environment for Recursive Self-Improvement*, arXiv:2610.10310, 2026-10-07. https://arxiv.org/abs/2610.10310
