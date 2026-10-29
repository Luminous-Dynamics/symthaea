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
- `score_probe.py`: scores required-support recall, ordinary complete-chain rate, **stale-aware complete-chain rate**, forbidden/stale hits, and distractor retrieval at the same fixed top-k budget.
- `fixtures/mutations/eligibility-ledger.jsonl`: explicit status policy for every string source pointer (current, historical, revoked, tentative, not-applied, causal, or contextual).
- `eligibility_filter.py`: provenance-bound eligibility-first ablation, with `--intent-source annotated` (oracle upper bound) and `--intent-source heuristic` (transparent keyword baseline). Both filter a BM25 candidate pool before final top-k selection; unknown policy pointers fail closed and unknown heuristic intent produces an empty context.

From the repository root, after compiling the Rust worker:

```sh
python3 research/broca-causal-episode-probe-v1/run_probe.py \
  --worker target/debug/broca-lme-worker \
  --trajectory-fixture research/broca-causal-episode-probe-v1/fixtures/mutations/trajectories.jsonl \
  --question-fixture research/broca-causal-episode-probe-v1/fixtures/mutations/questions.jsonl \
  --candidate-k 20 --k 5 \
  --output /tmp/broca-mutation-candidates.jsonl

# Baseline: take the first five BM25 candidates.
python3 research/broca-causal-episode-probe-v1/score_probe.py \
  --questions research/broca-causal-episode-probe-v1/fixtures/mutations/questions.jsonl \
  --predictions /tmp/broca-mutation-candidates.jsonl --k 5 \
  --output /tmp/broca-mutation-bm25-score.json

# Research upper bound: filter by explicit source eligibility before final top-k.
python3 research/broca-causal-episode-probe-v1/eligibility_filter.py \
  --predictions /tmp/broca-mutation-candidates.jsonl \
  --questions research/broca-causal-episode-probe-v1/fixtures/mutations/questions.jsonl \
  --trajectories research/broca-causal-episode-probe-v1/fixtures/mutations/trajectories.jsonl \
  --ledger research/broca-causal-episode-probe-v1/fixtures/mutations/eligibility-ledger.jsonl \
  --intent-source annotated --k 5 --output /tmp/broca-mutation-eligible.jsonl

# Compare the transparent keyword-intent baseline against the annotated upper bound.
python3 research/broca-causal-episode-probe-v1/eligibility_filter.py \
  --predictions /tmp/broca-mutation-candidates.jsonl \
  --questions research/broca-causal-episode-probe-v1/fixtures/mutations/questions.jsonl \
  --trajectories research/broca-causal-episode-probe-v1/fixtures/mutations/trajectories.jsonl \
  --ledger research/broca-causal-episode-probe-v1/fixtures/mutations/eligibility-ledger.jsonl \
  --intent-source heuristic --k 5 --output /tmp/broca-mutation-heuristic.jsonl

python3 research/broca-causal-episode-probe-v1/score_probe.py \
  --questions research/broca-causal-episode-probe-v1/fixtures/mutations/questions.jsonl \
  --predictions /tmp/broca-mutation-eligible.jsonl --k 5 \
  --output /tmp/broca-mutation-eligible-score.json

python3 research/broca-causal-episode-probe-v1/score_probe.py \
  --questions research/broca-causal-episode-probe-v1/fixtures/mutations/questions.jsonl \
  --predictions /tmp/broca-mutation-heuristic.jsonl --k 5 \
  --output /tmp/broca-mutation-heuristic-score.json

python3 research/broca-causal-episode-probe-v1/test_eligibility_filter.py
python3 research/broca-causal-episode-probe-v1/test_mutation_fixtures.py
```

Always preserve the generated manifest. It binds the worker binary hash, fixture hashes, retrieval settings, and prediction digest. Keep top-k fixed when comparing BM25, episode expansion, and any future semantic/HDC/graph arm.

## What this does and does not test

This suite measures retrieval support selection against manually authored synthetic labels. The `revocation` question tests whether a text retriever returns the revocation record; it does **not** test the worker's explicit tombstone API. The existing lifecycle probe remains the separate check for the explicit `supersede` operation. Likewise, the incident fixture tests retrieval of action/observation evidence, not causal identification in an intervention study.

### Eligibility-first ablation and its limits

The eligibility arm uses the same BM25 candidate pool as the baseline, then filters before truncating to the same final top-k. The ledger contains policy metadata for every string-valued source pointer; missing entries are rejected rather than passed through. Its receipt binds the original candidate receipt, worker hash, source fixture hashes, eligibility-ledger digest, policy version, and output digest.

Question `retrieval_intent` is manually annotated and therefore an **oracle input** in annotated mode. This isolates whether a correct temporal/epistemic eligibility policy can prevent stale evidence from consuming the answer budget; it is an upper bound, not evidence that Broca can infer intent correctly. Do not compare its result to BM25 as a production-system win.

Heuristic mode applies fixed, transparent keyword rules and logs the predicted intent, matched rule, and comparison with the annotated intent in the receipt. It is a cheap control, not a learned classifier; the rules are hand-authored against this tiny fixture. Its current 12 prompts are not held out, so any apparent perfect intent accuracy here is only a fixture smoke check, not evidence of generalization. Unknown intent abstains with an empty context. A production version must be evaluated on an independently authored, held-out intent set, reporting per-class precision/recall, confusion matrix, abstention, policy coverage, retained candidate count, and downstream stale-aware answer quality.

For `current_state`, old records are filtered; for `historical_state`, the system must preserve access to historical evidence; revocation questions retrieve the revocation event, not the prior active claim; proposal-history questions may retrieve tentative/not-applied events; causal-chain questions can retain failed interventions needed to interpret an outcome. This explicitly separates lifecycle state from the question's intent.

The labels are intentionally visible because this is a development diagnostic; it is not a hidden test set and must not be used to claim SOTA. Before publishing a benchmark result, grow the held-out set, have a second reviewer adjudicate every support pointer, add paraphrase variants, and include a blind negative-control partition.

## Required result table

Report every method at the same top-k and with the same corpus:
- **ordinary complete-chain rate:** all required evidence is present, regardless of whether stale evidence is also present;
- **stale-aware complete-chain rate:** all required evidence is present and no annotated forbidden evidence appears in the same top-k context;
- required-support recall;
- question-level stale/forbidden hit rate;
- forbidden-support item hit rate;
- distractor retrieval rate;
- p50/p95/p99 retrieval latency;
- context size and total write/index/query cost.

The stale-aware complete-chain metric is a local diagnostic gate inspired by the problem targeted by Memora's Forgetting-Aware Memory Accuracy (FAMA); it is **not** the official FAMA formula and must not be labeled FAMA. FAMA research reports that memory systems frequently reuse invalid memories and fail to reconcile evolving facts. Compare against Memora/FAMA separately if the official implementation and evaluation setup are available. <https://aclanthology.org/2026.findings-acl.1337/>

Do not collapse these into a single accuracy number. A method that raises ordinary recall by returning both the current answer and obsolete evidence has not won.

## Research gates

1. **Integrity:** every gold pointer resolves; no missing/duplicate predictions; worker/fixtures/config/predictions are receipt-bound.
2. **Quality:** use grouped paired tests on a larger, independently reviewed set; the current 12-question fixture is diagnostic only.
3. **Currentness:** zero forbidden-hit observations in this small suite is only a smoke test, never a zero-risk claim.
4. **Efficiency:** compare at matched context budgets and publish the entire lifecycle cost, not just query latency.
5. **Generalization:** freeze a held-out set before tuning; test natural-language updates, contradictory sources, delayed updates, and A → B → A reversions separately.

## Why the evaluation needs two chain metrics

The ordinary complete-chain metric can reward a retrieval set that contains every required item **and** a superseded item. The stale-aware metric deliberately makes that case fail, while preserving the separate forbidden-hit rates so severity and frequency remain visible. Keep both metrics: the gap between them quantifies cases where recall alone hides a currentness defect.

This distinction mirrors a wider finding in recent memory evaluation: temporal update and forgetting are separate capabilities, not just retrieval recall. Memora/FAMA evaluates reuse of obsolete or invalidated memories; MRAgent evaluates active reconstruction from associative evidence; Mem2ActBench checks whether memory translates into correct tool selection and argument grounding. Use them as distinct comparison lanes rather than combining their scores.

## External benchmark context

Keep the research lanes separate:
- [LongMemEval-V2](https://github.com/xiaowu0162/LongMemEval-V2): official end-to-end answer accuracy and query-latency comparison.
- [Memora / FAMA (Findings of ACL 2026)](https://aclanthology.org/2026.findings-acl.1337/): evaluates remembering, reasoning and recommending across extended conversations; FAMA explicitly penalizes reuse of obsolete or invalidated memory. Our stale-aware chain metric is not its official formula.
- [MRAgent (ICML 2026)](https://proceedings.mlr.press/v306/ji26d.html): tests active associative reconstruction from evolving intermediate evidence rather than static top-k retrieval.
- [Mem2ActBench (ACL 2026)](https://aclanthology.org/2026.acl-long.370/): evaluates whether memory improves tool choice and argument grounding in task-oriented agents.

### Competitive snapshot and comparison discipline

- The [official LongMemEval-V2 page](https://xiaowu0162.github.io/longmemeval-v2/) reports AgentRunbook-C V2 Online at 75.6% accuracy and 130.5 s query latency on its small tier, with LAFS gain +2.8231. Its V2 small GPT-5.4-mini medium-reasoning setting is reported separately at 72.73% and 56.68 s. These are different configurations; use the exact row and run manifest.
- [Supermemory's published LongMemEval-S report](https://supermemory.ai/research/longmembench/) reports 97% Recall@20 over 500 questions with a GPT-4o judge and an aggregation strategy. This is a vendor-published result on LongMemEval-S, and **retrieval Recall@20 is not end-to-end LongMemEval-V2 answer accuracy**. Do not compare the headline percentages directly.
- The [Agent Memory Leaderboard Cycle 1](https://agentmemoryleaderboard.ai/home) reports MemoraX at 58.02 overall, with 60.00 for temporal/event reasoning and 51.19 for memory governance in that leaderboard's rubric. These dimensions are useful because they expose failure modes not captured by fact recall alone, but they are not LongMemEval percentages.

- The [Cycle 2 public preview](https://agentmemories.ai/evaluation), checked 2026-10-29, currently lists MemoraX v0.5 (submitted API) at 62.00% Overall Resolved, with claude-mem v12.1.1 and MemOS v2.0.1 at 52.00%. The site says final Cycle 2 results will be released in mid-November, after its October 31 materials deadline and November 4 evaluation close. Treat the current rows as a live preview, not the final ranking; its Overall Resolved metric is distinct from Cycle 1's aggregate and from LongMemEval answer accuracy.

The Cycle 2 materials deadline currently listed by the leaderboard is **October 31, 2026 at 23:59 UTC+8 (October 31 at 17:59 SAST)**. A submission requires a stable participant-hosted Add/Search API and complete materials. This repository-only probe is not submission-ready; do not claim entry or rush an unqualified endpoint merely to hit the deadline.

Newer work also points to two useful architecture comparisons:

- [Memora: A Harmonic Memory Representation (ICML 2026)](https://proceedings.mlr.press/v306/xia26k.html) organizes concrete values under abstractions and connects them with cue anchors; its authors report state-of-the-art results on LoCoMo and LongMemEval. This is a direct comparison for Broca's future abstraction-versus-specificity and cue-expansion arms—not a result transferable to our synthetic fixtures.
- [Memory-R1 (ACL 2026)](https://aclanthology.org/2026.acl-long.583/) separates a memory manager (ADD/UPDATE/DELETE/NOOP) from an answer agent and uses outcome-driven reinforcement learning; the paper reports cross-benchmark generalization from 152 training QA pairs. Broca should compare learned write/update policies only after strict provenance and lifecycle invariants are in place; a learned policy must not override revocation or scope rules.
- [Mem-Gallery (ACL 2026)](https://aclanthology.org/2026.acl-long.1892/) evaluates multimodal long-term memory across extraction/adaptation, reasoning and knowledge management, and reports persistent organization and efficiency challenges. This is a later benchmark lane because the current BM25 worker explicitly does not support visual retrieval.

A useful field-level hypothesis is that temporal validity should be a hard eligibility predicate before any lexical, dense, or graph-based ranking. Reconstruction can expand only from eligible evidence; a verifier may inspect rejected stale evidence to explain a contradiction, but must not pass it to the answer context as current fact. Test this by holding the retrieval budget and reader fixed and reporting both quality and stale-evidence violation rates.

## Next architecture implication

Treat memory as a revisioned evidence ledger:
- claims are linked to the source event, scope, effective interval, and lifecycle status;
- supersession changes which claim is authoritative without erasing provenance;
- retrieval first resolves scope, validity, and lifecycle, then ranks eligible evidence;
- multi-hop reconstruction expands only from eligible seeds;
- uncertainty or conflicting eligible claims should trigger verification or abstention, not a confident guess.

This is a testable architectural hypothesis, not a claim that the current BM25 worker implements all of it.
