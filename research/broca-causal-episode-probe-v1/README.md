# Broca causal-episode retrieval probe v1

**Status:** research fixture and evaluator; not an official benchmark, not an end-to-end answer score, and not evidence of causal truth.

This probe is a small adversarial companion to the LongMemEval-V2 BM25 worker. It asks whether retrieval returns the complete evidence chain for event questions, rather than one lexically similar sentence. It also exposes stale-state and plausible-decoy retrieval.

The benchmark is intentionally separate from the official LongMemEval-V2 run. It adds no production dependencies and does not modify retrieval behavior.

## Why this experiment

AMA-Bench reports that causal-graph construction plus tool-assisted retrieval improves its own long-horizon agent-memory task; it reports 57.22% accuracy and an 11.16 percentage-point improvement over its strongest baseline in that benchmark setup. That result motivates a hypothesis, not a transferable score target. LongMemEval-V2, AMA-Bench and this focused probe use different task and evaluation contracts.

The hypothesis is:

> At a matched context budget, an episode-aware retriever should improve complete support-chain recall for action → observation → outcome questions, while reducing stale-state support, compared with lexical BM25 alone.

The falsifier is simple: if an episode-aware arm fails to improve complete-chain retrieval on these fixed probes, or only improves by returning more evidence, its extra complexity has not earned its place.

## What is in the package

- `fixtures/trajectories.jsonl`: five small web-agent trajectories with explicit action and observation fields.
- `fixtures/questions.jsonl`: paired queries annotated with required support, stale/forbidden support, and plausible distractors using source JSON Pointers.
- `run_probe.py`: feeds the fixtures to the actual Rust `broca-lme-worker`, collects its ranked audit records, and writes a worker/config/dataset receipt.
- `score_probe.py`: strict, dependency-free scoring for complete-chain retrieval, support recall, first-support reciprocal rank, stale-support hits, and distractor retrieval.
- `test_score_probe.py`: offline unit tests for scoring and fail-closed dataset matching.

The worker must be built from the exact Symthaea checkout under test. A probe run is not a substitute for the official LongMemEval-V2 harness.

## Run

From the Symthaea repository root, after the worker compiles:

    python3 research/broca-causal-episode-probe-v1/run_probe.py \
      --worker target/debug/broca-lme-worker \
      --k 5 \
      --output /tmp/broca-causal-probe-predictions.jsonl

    python3 research/broca-causal-episode-probe-v1/score_probe.py \
      --predictions /tmp/broca-causal-probe-predictions.jsonl \
      --k 5

    python3 research/broca-causal-episode-probe-v1/test_score_probe.py

The runner writes a sibling `.manifest.json` containing the worker SHA-256, fixture hashes, scope, top-k, protocol, and question count. Preserve these artifacts with any reported metric. To compare methods fairly, each arm must use the identical fixtures and question set and the same top-k evidence budget.

## Metrics and interpretation

- **Required-support recall@k:** fraction of annotated required source pointers present in the top-k audit.
- **Complete-chain rate@k:** fraction of questions for which every required support pointer appears in the top-k audit. This is the primary probe metric.
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

## Claim ceiling

This package establishes a reproducible probe protocol only. It does not establish causal inference, real-world generalization, official LongMemEval-V2 improvement, competitive SOTA, agent task success, or safe action selection.
