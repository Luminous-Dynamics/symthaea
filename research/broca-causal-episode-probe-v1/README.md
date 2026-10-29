# Broca causal-episode retrieval probe v1

**Status:** research fixture and evaluator; not an official benchmark, not an end-to-end answer score, and not evidence of causal truth.

This probe is a small adversarial companion to the LongMemEval-V2 BM25 worker. It asks whether retrieval returns the complete evidence chain for event questions, rather than one lexically similar sentence. It also exposes stale-state and plausible-decoy retrieval.

The benchmark is intentionally separate from the official LongMemEval-V2 run. It adds no production dependencies and does not modify retrieval behavior.

## Why this experiment

AMA-Bench reports that causal-graph construction plus tool-assisted retrieval improves its own long-horizon agent-memory task; it reports 57.22% accuracy and an 11.16 percentage-point improvement over its strongest baseline in that benchmark setup. That result motivates a hypothesis, not a transferable score target. LongMemEval-V2, AMA-Bench and this focused probe use different task and evaluation contracts.

The hypothesis is:

> At a matched context budget, an episode-aware retriever should improve complete support-chain recall for action → observation → outcome questions, while reducing stale-state support, compared with lexical BM25 alone.

The falsifier is simple: if an episode-aware arm fails to improve complete-chain retrieval on these fixed probes, or only improves by returning more evidence, its extra complexity has not earned its place.

## First controlled ablation: same/adjacent-step expansion

`episode_expand.py` is a deliberately simple non-graph baseline. It takes the ranked BM25 candidate pool, expands each seed to source fields from the same trajectory step and one adjacent step, uses a fixed action/observation/thought/URL priority, removes duplicates and stops at exactly k. It reads trajectory content and BM25 audit candidates, **not** support labels or gold answers. Use `test_episode_expand.py` for negative controls.

This is not a causal graph and does not solve currentness. The stale-address adversarial test intentionally demonstrates that neighboring-step expansion can reintroduce a retired fact. That failure is informative: currentness must be represented by explicit update/supersession semantics, not inferred from proximity or the highest step index. Report both complete-chain improvement and stale-support regressions.

## What is in the package

- `fixtures/trajectories.jsonl`: five small web-agent trajectories with explicit action and observation fields.
- `fixtures/questions.jsonl`: paired queries annotated with required support, stale/forbidden support, and plausible distractors using source JSON Pointers.
- `run_probe.py`: feeds the fixtures to the actual Rust `broca-lme-worker`, collects its ranked audit records, and writes a worker/config/dataset receipt.
- `score_probe.py`: strict, dependency-free scoring for required-support recall, ordinary and stale-aware complete chains, first-support reciprocal rank, stale-support hits, and distractor retrieval.
- `fixtures/mutations/eligibility-ledger.jsonl` and `eligibility_filter.py`: digest-bound eligibility-first ablation with an annotated-intent upper-bound mode and a transparent keyword baseline; unknown source pointers fail closed.
- `test_eligibility_filter.py`: tests for eligibility policy, the oracle/heuristic distinction, fixed-budget selection, abstention, fail-closed behavior, and receipt integrity.
- `test_mutation_fixtures.py`: tests for pointer coverage, currentness negatives, provenance reversions, and the synthetic causal/temporal labels.
- `fixtures/mutations/dependency_snapshots.jsonl` and `dependency_cascade.py`: seven synthetic snapshots of a dependency graph and a reference truth-maintenance evaluator that classifies historical, stale-dependent, unknown-dependent, and current claims.
- `test_dependency_cascade.py`: 17 tests covering direct/transitive invalidation, partial downstream refresh, stale-value absence, A → B → A provenance, full-chain restoration, historical auditability, a 1,500-node dependency chain, order invariance, and fail-closed graph validation.
- `test_score_probe.py`: offline unit tests for scoring and fail-closed dataset matching.

The worker must be built from the exact Symthaea checkout under test. A probe run is not a substitute for the official LongMemEval-V2 harness.

## Run

From the Symthaea repository root, after the worker compiles:

    python3 research/broca-causal-episode-probe-v1/run_probe.py \
      --worker target/debug/broca-lme-worker \
      --k 5 --candidate-k 20 \
      --output /tmp/broca-causal-probe-candidates.jsonl

    # Baseline uses the first five BM25 candidates.
    python3 research/broca-causal-episode-probe-v1/score_probe.py \
      --predictions /tmp/broca-causal-probe-candidates.jsonl \
      --k 5 --output /tmp/broca-bm25-score.json

    # Ablation expands a BM25 seed into same/adjacent trajectory steps.
    python3 research/broca-causal-episode-probe-v1/episode_expand.py \
      --predictions /tmp/broca-causal-probe-candidates.jsonl \
      --k 5 --window 1 \
      --output /tmp/broca-episode-expanded.jsonl
    python3 research/broca-causal-episode-probe-v1/score_probe.py \
      --predictions /tmp/broca-episode-expanded.jsonl \
      --k 5 --output /tmp/broca-episode-expanded-score.json

    python3 research/broca-causal-episode-probe-v1/test_score_probe.py
    python3 research/broca-causal-episode-probe-v1/test_episode_expand.py
    python3 research/broca-causal-episode-probe-v1/test_lifecycle_probe.py
    python3 research/broca-causal-episode-probe-v1/test_analyze_paired_runs.py
    python3 research/broca-causal-episode-probe-v1/test_mutation_fixtures.py
    python3 research/broca-causal-episode-probe-v1/test_eligibility_filter.py

    # Verify the explicit supersede operation through two clean worker processes:
    python3 research/broca-causal-episode-probe-v1/run_lifecycle_probe.py \
      --worker target/debug/broca-lme-worker \
      --output /tmp/broca-supersession-probe.json

The runner writes a sibling `.manifest.json` containing the worker SHA-256, fixture hashes, scope, candidate pool, evaluation budget, protocol, and question count. The candidate pool (`--candidate-k`) must be at least as large as the final budget (`--k`). The baseline is scored using the first k BM25 results. The episode-expansion arm requires that sibling manifest, verifies the candidate prediction digest and fixture digest, and carries forward the worker SHA-256 and BM25 settings into its own receipt. It then adds indexed evidence from the same or adjacent trajectory steps until the identical k-item budget is reached. Preserve all manifests with every metric.

## Metrics and interpretation

- **Required-support recall@k:** fraction of annotated required source pointers present in the top-k audit.
- **Complete-chain rate@k:** fraction of questions for which every required support pointer appears in the top-k audit, even if stale/forbidden evidence is also present.
- **Stale-aware complete-chain rate@k:** fraction of questions for which every required pointer is present and no annotated forbidden pointer appears in the same top-k audit. This is the primary diagnostic metric; report ordinary complete-chain rate as well so the cost of stale evidence remains visible. This local metric is not the official FAMA score.
- **First-support MRR@k:** reciprocal rank of the first required support item, averaged across questions.
- **Forbidden/stale hit rate@k:** fraction of annotated stale pointers retrieved in top-k; also report the fraction of questions with any stale hit.
- **Distractor retrieval rate@k:** fraction of annotated plausible-but-non-supporting pointers retrieved in top-k. These are reported, not automatically treated as critical failures: a useful verifier may retrieve a decoy in order to refute it.

A source-pointer hit demonstrates that a span was retrieved; it does **not** prove that the span is true or causally sufficient. The labels in this tiny synthetic corpus are for retrieval diagnostics. Do not tune on it and then call it held-out evidence.

## Research sequence

1. Run BM25 at k = 1, 3, 5, and 10 with fixed k1 = 1.2 and b = 0.75.
2. Add an episode-expansion arm that retrieves the same trajectory's linked action/observation neighbors after a primary hit. Count every added item against the same k budget.
3. Add an explicit typed transition graph only after the episode-expansion ablation is measured.
4. Compare the arms on official LongMemEval-V2 and one independent action-oriented benchmark (e.g. AMA-Bench or Mem2ActBench); do not average their scores together.
5. Run the mutation lane separately: supersede a fact, save/load memory, rebuild the index, and prove the old pointer is no longer eligible. That is a governance test, not a causal-chain score.

## Dependency cascade and absence probe

MEME: Multi-entity & Evolving Memory Evaluation isolates dependency cascade and absence—capabilities that static fact-recall metrics can miss. Its authors report very low default average accuracy on those tasks in their 100-episode study; the result is benchmark-specific, but it motivates our separate probe: https://arxiv.org/abs/2605.12477

The new reference evaluator checks seven snapshots: an initial current dependency chain; an upstream endpoint revision; partial service refresh; complete refresh; an A → B → A endpoint reversion; direct-dependent refresh while the worker remains stale; and final full-chain refresh. It computes a current view only from the latest revisions whose dependency references all point to latest/current dependencies. Stale values remain auditable and are excluded from the current view; missing dependencies become `unknown_dependency`; dependency cycles and duplicate revisions fail closed. The implementation uses iterative topological processing rather than recursion, and the report binds the input and evaluator-source SHA-256. A 1,500-node chain test guards against recursion-limit failures.

```sh
python3 research/broca-causal-episode-probe-v1/dependency_cascade.py \
  --input research/broca-causal-episode-probe-v1/fixtures/mutations/dependency_snapshots.jsonl \
  --output /tmp/broca-dependency-cascade-report.json
python3 research/broca-causal-episode-probe-v1/test_dependency_cascade.py
```

This is a reference semantics and synthetic diagnostic, not a production worker integration or a benchmark score. Next, port the semantics into a Rust truth-maintenance implementation and test update transactions, rollback, persisted event replay, and index rebuilds against these expected transitions.

## New benchmark lanes to plan for

- **MEME Cascade / Absence / Deletion** is the immediate dependency-validity lane. The MEME authors evaluate six memory systems on 100 controlled episodes, using GPT-4.1-mini as the internal and answering model with approximately 32K filler tokens; they report 3% average Cascade accuracy and 1% Absence accuracy across tested defaults. This is a benchmark-specific result, not a universal field average. The tlc research page separately reports its own MEME results of 0.628 Cascade, 0.42 Absence and 0.574 overall; treat those as provider-reported and reproduce against the official harness before setting a definitive baseline.
- **COLLABMEM (EMNLP 2026)** evaluates collaborative memory operations: which role gets information, provenance, access control, routing, conflict resolution and retention. Its released description includes 58 hand-authored scenarios / 1,118 core probes and a scaled split of 174 scenarios / 3,359 probes. This is a natural future lane for Mycelix-integrated, scope-aware Broca memory.
- **MemoryArena (ICML 2026)** evaluates memory in interdependent multi-session agent-environment loops, connecting memory formation with later action.
- **MemoryAgentBench (ICLR 2026)** covers accurate retrieval, test-time learning, long-range understanding and selective forgetting.
- **Memory-R1 (ACL 2026)** is the comparator for learned ADD / UPDATE / DELETE / NOOP memory-management policies. Any learned writer must remain subordinate to provenance, scope, revocation and dependency invariants.
- **Mem-Gallery** and **DMV-Bench (EMNLP 2026)** extend the research toward multimodal long-term memory and incidental visual cues. They belong after the existing worker's visual retrieval limitations are explicitly addressed.

Do not blend scores from these benchmarks into one mean. Use their native evaluation protocols and fixed reader/settings.

## Eligibility-first currentness ablation

The mutation suite includes a versioned policy ledger covering all 107 string-valued source pointers exactly once and a 13th composite query that asks for both current and prior state. The CLI verifies that coverage against the source trajectories. Each question has a manually annotated retrieval intent, including `current_and_history` for questions that request both present and historical state. The filter runs after BM25 candidate generation but **before final top-k selection**, preserves the BM25 order among eligible candidates, fails closed for unannotated pointers, and emits a receipt binding the candidate manifest, worker hash, fixture hashes, policy-ledger hash, filter-code SHA-256, intent mode, policy and output digest.

Run both modes against the same candidate pool:

```sh
# Oracle upper bound: use the manually annotated retrieval intent.
python3 research/broca-causal-episode-probe-v1/eligibility_filter.py \
  --predictions /tmp/broca-mutation-candidates.jsonl \
  --questions research/broca-causal-episode-probe-v1/fixtures/mutations/questions.jsonl \
  --trajectories research/broca-causal-episode-probe-v1/fixtures/mutations/trajectories.jsonl \
  --ledger research/broca-causal-episode-probe-v1/fixtures/mutations/eligibility-ledger.jsonl \
  --intent-source annotated --k 5 --output /tmp/broca-mutation-eligible.jsonl

# Transparent keyword baseline; an unknown intent abstains with an empty context.
python3 research/broca-causal-episode-probe-v1/eligibility_filter.py \
  --predictions /tmp/broca-mutation-candidates.jsonl \
  --questions research/broca-causal-episode-probe-v1/fixtures/mutations/questions.jsonl \
  --trajectories research/broca-causal-episode-probe-v1/fixtures/mutations/trajectories.jsonl \
  --ledger research/broca-causal-episode-probe-v1/fixtures/mutations/eligibility-ledger.jsonl \
  --intent-source heuristic --k 5 --output /tmp/broca-mutation-heuristic.jsonl

python3 research/broca-causal-episode-probe-v1/score_probe.py \
  --questions research/broca-causal-episode-probe-v1/fixtures/mutations/questions.jsonl \
  --predictions /tmp/broca-mutation-eligible.jsonl --k 5
python3 research/broca-causal-episode-probe-v1/score_probe.py \
  --questions research/broca-causal-episode-probe-v1/fixtures/mutations/questions.jsonl \
  --predictions /tmp/broca-mutation-heuristic.jsonl --k 5
```

The annotated mode is an **oracle-intent upper bound**; it is not a production-system win. The heuristic mode uses simple keyword rules and records the matched rule, per-question intent predictions, accuracy, and unknowns in the receipt. The rules were authored against this tiny fixture, so the current 13 prompts are not held out. Even perfect intent accuracy on these prompts would only be a fixture smoke check, not generalization evidence. The filter never reads required/forbidden support labels to select candidates. Next, evaluate intent prediction on independently authored, held-out queries and report per-class precision/recall, confusion matrix, abstention, policy coverage, retained candidate count, stale-aware quality, latency and lifecycle cost.

## Paired official-run decision analysis

`analyze_paired_runs.py` makes the preregistered quality/efficiency gates executable once a baseline and candidate produce per-question JSONL. Both files must contain exactly one row for every matching `(domain, question_id)` pair, with the same `group_id` in both arms and these fields:

```json
{"domain":"web","question_id":"q-001","group_id":"session-07","status":"success","correct":true,"query_latency_ms":210.5,"context_tokens":816,"lifecycle_cost_units":0.42}
```

`status` must be `success`; failed or missing rows are not silently discarded. `lifecycle_cost_units` must use the same declared units for both arms and should include the agreed allocation of write/index/background cost, not query latency alone. The script groups and pairs question outcomes, runs a deterministic 95% cluster bootstrap by `group_id`, computes per-domain and aggregate accuracy deltas, latency p50/p95/p99, context size, lifecycle-cost reduction, and machine-readable gate results.

Example:

    python3 research/broca-causal-episode-probe-v1/analyze_paired_runs.py \
      --baseline runs/baseline/per_question.jsonl \
      --candidate runs/broca/per_question.jsonl \
      --bootstrap-replicates 20000 --seed 20261028 \
      --output runs/paired-analysis.json

The default quality gate requires at least +3 percentage points and a 95% clustered-bootstrap interval whose lower bound is above zero. The alternative efficiency gate requires the accuracy interval's lower bound to be above -1 percentage point and observed lifecycle-cost reduction of at least 25%. These are internal preregistered project gates. A numerical pass alone is not an official SOTA claim; the upstream harness, reader/evaluator identity, exact split and run receipts still must be verified. The analysis tool is authored but not executed in the current environment.

## Explicit currentness transition probe

`run_lifecycle_probe.py` drives the compiled JSONL worker twice with the exact same trajectory and BM25 configuration. The first process discovers stable evidence IDs for the old and replacement address spans. The second process inserts the same trajectory, sends the worker's explicit `supersede` operation, then queries both the old and current address.

The runner fails closed unless:
- both clean processes report the same canonical source digest;
- the supersede acknowledgement names the exact old and replacement IDs;
- the old evidence ID disappears from post-transition search;
- the replacement remains retrievable;
- returned context and audit arrays have matching cardinality.

This tests a recorded explicit update event, not automatic extraction of facts from natural language. The benchmark's naive episode expansion can still retrieve a retired span because it does not consult lifecycle state. Do not claim the system autonomously inferred the update. The runner writes a JSON receipt containing the worker/fixture digests and hashes of both protocol transcripts.

## Claim ceiling

This package establishes a reproducible probe protocol only. It does not establish causal inference, real-world generalization, official LongMemEval-V2 improvement, competitive SOTA, agent task success, or safe action selection.
