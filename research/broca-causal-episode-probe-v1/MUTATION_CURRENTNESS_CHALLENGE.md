# Broca Memory Mutation and Currentness Challenge v1

**Status:** synthetic retrieval diagnostic; not an official benchmark, not a measure of end-to-end answer accuracy, and not proof of causal correctness.

This suite isolates a high-value weakness of long-term agent memory: the answer can contain the right words and still be wrong because the retrieved evidence is obsolete, tentative, revoked, or a stale record of a value that later became current again.

## Hypothesis

A memory system should retrieve evidence that is **valid for the question's temporal and epistemic intent**, not merely evidence with high lexical overlap. It should distinguish:
- current state from historical state;
- proposed state from approved/committed state;
- active facts from explicitly revoked facts;
- a causal chain from a plausible but non-causal intervening event; and
- a currently valid fact from an old record that happens to contain the same value (the ABA/reversion case).

The ABA case matters: if an office address changes A → B → A, an old A record may match the current answer text but is not the record establishing the current state. Provenance and validity cannot be replaced by string equality.

## Contents and execution

- `fixtures/mutations/trajectories.jsonl`: 9 synthetic event histories.
- `fixtures/mutations/questions.jsonl`: 12 questions with source-pointer ground truth for required, forbidden/stale, and distractor evidence.
- `run_probe.py`: accepts optional fixture paths while preserving the original defaults.
- `score_probe.py`: scores complete-support rate, required-support recall, forbidden/stale hits, and distractor retrieval at the same fixed top-k budget.

From the repository root, after compiling the Rust worker:

```sh
python3 research/broca-causal-episode-probe-v1/run_probe.py \
  --worker target/debug/broca-lme-worker \
  --trajectory-fixture research/broca-causal-episode-probe-v1/fixtures/mutations/trajectories.jsonl \
  --question-fixture research/broca-causal-episode-probe-v1/fixtures/mutations/questions.jsonl \
  --candidate-k 20 --k 5 \
  --output /tmp/broca-mutation-candidates.jsonl

python3 research/broca-causal-episode-probe-v1/score_probe.py \
  --questions research/broca-causal-episode-probe-v1/fixtures/mutations/questions.jsonl \
  --predictions /tmp/broca-mutation-candidates.jsonl --k 5 \
  --output /tmp/broca-mutation-score.json
```

Always preserve the generated manifest. It binds the worker binary hash, fixture hashes, retrieval settings, and prediction digest. Keep top-k fixed when comparing BM25, episode expansion, and any future semantic/HDC/graph arm.

## What this does and does not test

This suite measures retrieval support selection against manually authored synthetic labels. The `revocation` question tests whether a text retriever returns the revocation record; it does **not** test the worker's explicit tombstone API. The existing lifecycle probe remains the separate check for the explicit `supersede` operation. Likewise, the incident fixture tests retrieval of action/observation evidence, not causal identification in an intervention study.

The labels are intentionally visible because this is a development diagnostic; it is not a hidden test set and must not be used to claim SOTA. Before publishing a benchmark result, grow the held-out set, have a second reviewer adjudicate every support pointer, add paraphrase variants, and include a blind negative-control partition.

## Required result table

Report every method at the same top-k and with the same corpus:
- complete support-chain rate;
- required-support recall;
- question-level stale/forbidden hit rate;
- forbidden-support item hit rate;
- distractor retrieval rate;
- p50/p95/p99 retrieval latency;
- context size and total write/index/query cost.

Do not collapse these into a single accuracy number. A method that raises recall by flooding context or reintroducing stale facts has not won.

## Research gates

1. **Integrity:** every gold pointer resolves; no missing/duplicate predictions; worker/fixtures/config/predictions are receipt-bound.
2. **Quality:** use grouped paired tests on a larger, independently reviewed set; the current 12-question fixture is diagnostic only.
3. **Currentness:** zero forbidden-hit observations in this small suite is only a smoke test, never a zero-risk claim.
4. **Efficiency:** compare at matched context budgets and publish the entire lifecycle cost, not just query latency.
5. **Generalization:** freeze a held-out set before tuning; test natural-language updates, contradictory sources, delayed updates, and A → B → A reversions separately.

## Next architecture implication

Treat memory as a revisioned evidence ledger:
- claims are linked to the source event, scope, effective interval, and lifecycle status;
- supersession changes which claim is authoritative without erasing provenance;
- retrieval first resolves scope, validity, and lifecycle, then ranks eligible evidence;
- multi-hop reconstruction expands only from eligible seeds;
- uncertainty or conflicting eligible claims should trigger verification or abstention, not a confident guess.

This is a testable architectural hypothesis, not a claim that the current BM25 worker implements all of it.
