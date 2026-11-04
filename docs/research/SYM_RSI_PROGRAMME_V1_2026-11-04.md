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
- A child can only reference a parent already present in the archive, so the stored lineage is acyclic by construction.
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
- **AIDE²:** a self-modifying research agent with seven reported successive improvements in an eight-day autonomous run and gains on four held-out task families.
- **AI4AI-Bench:** isolates training-algorithm redesign; the paper reports a best normalized score of 0.250 across ten algorithm-design tasks.
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
- [x] Reject inconsistent metric dimensions and broken parent lineage.
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

**Statistical limitation:** the exact sign test measures directional consistency among non-tied paired cases. It does not measure effect magnitude (the mean delta is reported separately), establish practical significance, correct for repeated adaptive selection, prove equivalence when non-significant, or authorize promotion.

### Phase 2 — Controlled coding-agent evolution

- Connect the archive to the coding agent through a candidate adapter.
- Keep generated repairs and code edits in isolated worktrees/containers.
- Compare the candidate with its direct parent and frozen baseline.
- Promote only after independent tests, retention gates, and held-out evaluation.
- Keep a fixed-meta-agent and single-champion baseline.

### Phase 3 — Meta-improvement

- Let Symthaea improve experiment selection, hypothesis generation, causal review, and memory/context management.
- Measure discovery yield and quality per fixed budget.
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
