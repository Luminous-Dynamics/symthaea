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
- `score_probe.py`: strict, dependency-free scoring for complete-chain retrieval, support recall, first-support reciprocal rank, stale-support hits, and distractor retrieval.
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

    # Verify the explicit supersede operation through two clean worker processes:
    python3 research/broca-causal-episode-probe-v1/run_lifecycle_probe.py \
      --worker target/debug/broca-lme-worker \
      --output /tmp/broca-supersession-probe.json

The runner writes a sibling `.manifest.json` containing the worker SHA-256, fixture hashes, scope, candidate pool, evaluation budget, protocol, and question count. The candidate pool (`--candidate-k`) must be at least as large as the final budget (`--k`). The baseline is scored using the first k BM25 results. The episode-expansion arm requires that sibling manifest, verifies the candidate prediction digest and fixture digest, and carries forward the worker SHA-256 and BM25 settings into its own receipt. It then adds indexed evidence from the same or adjacent trajectory steps until the identical k-item budget is reached. Preserve all manifests with every metric.

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
