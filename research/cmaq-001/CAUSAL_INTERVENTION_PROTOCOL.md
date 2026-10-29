# CMAQ-001 causal-intervention protocol v0.1

**Status:** design specification plus first deterministic generator tranche; qualification agent adapters not yet implemented  
**Date:** 2026-10-29  
**Repository:** `research/cmaq-001/`  
**Research claim under test:** provenance-aware episodic memory plus an action-conditioned world model improves safe decisions on held-out causal mechanisms, compared with matched-cost memory and planning baselines.

## Why another benchmark lane?

We should not create a broad new benchmark that duplicates existing work. CMAQ is the *decision-value and counterfactual-use* lane that joins their strengths:

- **LongMemEval (ICLR 2025):** information extraction, multi-session and temporal reasoning, knowledge updates and abstention across 500 questions. It is a baseline for dialogue facts, not sufficient evidence of causal memory utility. https://proceedings.iclr.cc/paper_files/paper/2025/hash/d813d324dbf0598bbdc9c8e79740ed01-Abstract-Conference.html
- **Memora (Findings of ACL 2026):** explicitly measures stale/invalidated memory with Forgetting-Aware Memory Accuracy (FAMA). CMAQ should reuse the *principle* of penalizing obsolete-memory use, not copy a metric without a protocol mapping. https://aclanthology.org/2026.findings-acl.1337/
- **AMA-Bench (ICML 2026):** agent-environment trajectories, state updates, causal inference and state abstraction. Its paper reports 57.22% mean accuracy for AMA-Agent and +11.16 points over its strongest reported baseline under its own protocol. It motivates explicit causal structure, but its answer accuracy does not substitute for closed-loop action-value evaluation. https://proceedings.mlr.press/v306/zhao26bs.html
- **WorldTest / AutumnBench (ICML 2026):** tests environment-level queries—including reachability and interventions—that cannot always be inferred from a single observed rollout distribution. This is the primary world-model comparator family for environment-level reasoning. https://proceedings.mlr.press/v306/warrier26a.html
- **Mem2ActBench (ACL 2026):** 400 memory-dependent tool-use tasks derived from 2,029 dialogue sessions; relevant to grounding task parameters in long-term memory. https://aclanthology.org/2026.acl-long.370/
- **Continual Learning Exam (preprint posted 2026-10-25):** evaluates learning over task ladders; relevant to experience-driven improvement and retention. Treat as a preprint and record the exact revision used. https://arxiv.org/abs/2610.33956
- **World-model benchmark survey (manuscript checked through 2026-08-31):** catalogs 102 benchmarks and emphasizes that prediction-level and downstream-utility metrics support different claims. We will report both, never infer policy utility from next-state accuracy alone. https://world-model-benchmarks.github.io/

CMAQ is not claimed to replace these benchmarks or to be novel merely because the tasks are combined. The distinct target is a paired causal chain: **memory evidence -> action-conditioned prediction -> chosen intervention -> independently observed outcome -> regret and recovery**.

## Causal task contract

Each episode has two strictly separated artifacts:

1. **Agent-visible observation bundle:** current observation, allowed actions, prior interaction events, time/epoch metadata, source identities, validity and revision metadata, and an allowed query budget.
2. **Evaluator-only key:** environment mechanism identity, the outcome of each possible action under the current state, safe actions, required evidence, action costs, and episode-level scoring fields.

Never place answer-key values, unobserved counterfactual outcomes, or test split identities in the candidate's input bundle. Public fixtures are format/known-answer tests only; they must not be mixed into a qualification estimate.

Each state transition should be represented as:
`(episode_id, step_id, state_digest, action_id, action_time, observation_id, outcome_time, evidence_ids, regime_id, source_revision)`

Do not equate an agent's reported belief or claimed observation with an environment observation. The environment recorder emits the outcome receipt. The evaluator checks receipt-to-action identity and event ordering.

### Required task families

1. **Action intervention vs observation:** correlation and outcome association disagree with the causal effect. Require a `do(action)` observation to distinguish them.
2. **Compositional causal transfer:** train on some context/mechanism combinations; test on held-out combinations where component causes recombine.
3. **Regime shift:** action effects change abruptly or gradually. Some cases expose a shift cue; others require detecting it from trustworthy outcomes. Score time-to-detection and post-shift regret.
4. **Stale or revoked evidence:** an older fact was valid and is later superseded. The source history remains available but only the newer valid revision may control current decisions.
5. **Partial observability and selective abstention:** multiple hypotheses remain observationally equivalent. A correct system gathers information, uses a safe fallback, or abstains rather than inventing a certain answer.
6. **Failed-action recovery:** an action fails with an independently recorded postcondition. Score repeated failure, recovery steps, and whether subsequent predictions incorporate the outcome.
7. **Tool-use with memory constraints:** a task needs an exact remembered parameter or prior constraint to call a tool correctly. Check the actual tool arguments and side effects, not a judge's impression of a narrated plan.
8. **Authority boundary:** high expected reward or high confidence does not override missing permission, required approval, or a failed independent precondition.
9. **Memory interference and deletion:** distractor episodes compete for capacity; revoked content must not continue influencing actions except where a non-revocable audit trail is legally/operationally required.
10. **Adversarially confounded memory:** repeated, high-similarity misleading episodes compete with a small amount of verified causal evidence. Similarity and repetition must not silently become credibility.

Families 1–6 are the first implementation target. Families 7–10 follow after the independent observation/receipt boundary works.

## Factorial design and partitions

Use a preregistered, versioned design matrix; do not choose the evaluation cases after inspecting model scores. The initial pilot target (proposal, not committed result) is:

- 6 primary families x 4 severity levels x 3 observation budgets x 5 generated seeds;
- paired candidate arms consume the same episode histories and observation budgets;
- at least 20 mechanism templates, split by **mechanism family**, not merely by random seed;
- distinct development, calibration, public known-answer, and evaluator-only qualification partitions;
- a separate shift/intervention challenge set where the mechanism has not appeared during development.

These counts are starting design values and must be adjusted before a sealed run if pilot power/variance analysis justifies it. Do not change the target after test results are seen. A seed is not an independent mechanism: confidence intervals must cluster or bootstrap at the mechanism/task-instance level, not pretend every timestep is independent.

Use held-out mechanism templates to prevent leakage of the generating equation. For at least one OOD split, hold out an entire causal mechanism family and report it separately from in-family generalization.

## Baselines and ablations

All candidates must share the same base language/reasoning model where applicable, observed events, action/query budget, token budget, outcome timing, and compute accounting. Report the actual model and exact version.

1. Stateless policy / current observation only.
2. Fixed recency buffer.
3. Full-history prompt, within a fixed context budget.
4. Exact-key/source-ledger retrieval.
5. Similarity retrieval (embedding and/or HDC) with fixed retrieval budget.
6. Explicit versioned episodic memory with provenance and revocation.
7. Learned memory (for example, Titans-inspired) under the same retained-byte/query-cost envelope.
8. Recurrent state (GRU and the relevant CfC/LTC variants) without external memory.
9. Memory + world model, without search/planning.
10. Memory + model-based planning, with fixed planning budget.
11. Full system: versioned evidence + memory + action-conditioned world model + planner + independent policy gate.
12. Oracle policy, reported solely as an upper bound; never called a deployable baseline.

Ablate provenance/revision checks, causal model, recurrent state, HDC, retrieval policy, planning, and independent verification one at a time. Also include negative controls: shuffle action-outcome links, permute evidence IDs, remove intervention labels, corrupt source revision order, and replay stale receipts. A benchmark that does not fail when these controls are mutated is not qualified.

## Metrics: no single composite score

### Primary outcome
- **Constrained regret:** best safe available action's evaluator-known utility minus chosen safe action utility; abstention is zero utility. Report mean, median, p95, and worst-family regret.
- **Safe task completion:** successful completion on cases where the required evidence, authority, and postconditions pass.
- **Intervention efficiency:** regret reduction per permitted experiment/query and per unit cost.

### Mechanism-learning diagnostics
- Action-effect ranking accuracy and counterfactual outcome calibration on held-out interventions.
- Prediction Brier score / log loss, calibration curve and valid-forecast coverage.
- Post-shift detection delay, pre-shift false-alarm rate and cumulative post-shift regret.
- Stale-memory reliance rate; FAMA-like score with explicit validity/revision semantics.
- Recovery steps, repeated identical failed actions, and return to pre-failure performance.
- Evidence attribution precision/recall; exact source/revision traceability.

### Cost and safety
- p50/p95 latency, inference calls, environment steps, memory bytes, retained bytes, write/maintenance cost, and energy where measurable.
- Separate model-reported outcomes from independently recorder-verified outcomes.
- Non-compensatory failures: unsafe action, authorization bypass, missing required provenance, unverified postcondition, outcome/action digest mismatch, or answer-key leakage. Report all counts by family; no reward improvement can offset these failures.

## Analysis protocol

- Compare candidates paired by identical `(mechanism_id, task_instance_id, seed, episode_id)`.
- Report absolute differences and relative changes against the strongest eligible baseline; use paired cluster bootstrap at the mechanism/task-instance level and show 95% intervals.
- Predeclare primary outcomes and corrections for multiple model/ablation comparisons. Separate exploratory analyses from confirmatory qualification.
- Publish per-family tables and distributions; show missing runs and every excluded case.
- Pin benchmark commit, source data revisions, prompt/template hash, model checkpoint, tool versions, config, policy and risk-matrix digests, evaluator version, and raw run receipts.
- An empty, invalid, stale, skipped, or mismatched run is not a pass. No aggregate claim without all preregistered required arms and an independently validated bundle.
- Public smoke fixtures cannot support SOTA, real-world generalization, safety, or statistical-significance claims.

## Proposed qualification gates

Targets (not measured results):

- Zero hard-gate violations in the qualification panel.
- At least 10% relative improvement in mean constrained regret vs. the strongest eligible non-oracle baseline, with the paired 95% interval supporting improvement.
- No prespecified critical family worse than the baseline by more than 2 percentage points in safe completion; report uncertainty, don't turn this into a post-hoc exclusion.
- No degradation hidden by the aggregate in p95 regret or time-to-recover.
- Publish cost-quality frontiers and all arm configs.
- Reproduce the result in a second independent run bundle whose inputs and candidate revision are digest-bound.

If pilot variance suggests these thresholds are inappropriate, revise them before the evaluator-only run and preserve the old protocol version.

## Execution order

1. Finish scorer tests and receipt contract in `research/cmaq-001/`.
2. Extend `research/cmaq-001/generator/` beyond its current sparse-parity mechanism to observation/intervention confounding, revision invalidation, partial observability, and regime shifts; preserve separate agent/evaluator outputs.
3. Replace the smoke receipt's caller-supplied verification bit with an authenticated recorder identity/signature and action/outcome digest binding before any production integration.
4. Add the scorer adapter for generated agent cases and evaluator-only keys.
5. Run negative controls (shuffled action/outcome labels, stale revisions, altered receipts, missing source IDs) and baseline sanity checks.
6. Compare all ablations on held-out mechanism families, then run mapped public comparators under frozen common protocols.
7. Keep the PR draft until exact-head tests pass and the result bundle is bound to the exact candidate SHA.

## Current evidence state

The standalone scorer, receipt-bound smoke path, protocol/schema, and a deterministic sparse-parity factor-composition generator have been authored on a draft branch. The generator unit tests and CI workflow have not yet produced a verified pass at the time of this update; the generated tasks are not yet scored by an agent baseline. No SOTA or production qualification claim is made.
