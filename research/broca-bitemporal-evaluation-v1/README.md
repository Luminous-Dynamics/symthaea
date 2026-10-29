# Broca Bitemporal Evaluation v1

**Status:** research-only reference semantics and synthetic regression suite. This is not wired into the production worker and is not an official benchmark submission.

## Research question

Can a memory system distinguish these two questions without temporal leakage?

1. **As known then:** What answer was justified by the events recorded by a historical knowledge cutoff?
2. **As corrected now:** Given what is known now, what value applied at an earlier point in the world?

A single “current” / “historical” label cannot answer both. Store at least two times for a memory claim:

- **Valid time:** when the claim applies in the represented world.
- **Transaction/record time:** when the memory system received the claim.

A late-arriving correction may have an old valid time and a new transaction time. The resolver must not leak that correction into an answer about what the system knew before it arrived.

## Executable reference policy

The Python reference consumes event and query JSONL. Every claim has an immutable event ID, subject, attribute, value, half-open valid interval [valid_from, valid_to), recorded_at with explicit UTC offset, source, and non-negative authority rank.

For a (subject, attribute, valid_at, known_at) query:

1. Exclude events recorded after known_at.
2. Exclude events whose valid interval does not contain valid_at.
3. Among remaining claims, choose the latest valid_from.
4. If claims share that effective start, choose the highest authority rank.
5. If the surviving highest-ranked claims disagree, return **conflict** with no value. Never use file order or arrival order as an unannounced tie-break.
6. If no claim survives, return **unknown**. Do not convert missing evidence into a negative fact.

The output reports eligible, selected, and shadowed event IDs; selected source/value/time/rank provenance; the resolution rule; and SHA-256 digests for the exact event file, query file, and evaluator source. It reports fixture accuracy only as a synthetic consistency check.

This is a deliberately small policy, not a universal truth oracle. The precedence policy is explicit and therefore falsifiable. A production design also needs typed retractions/deletions, source authentication, authority-policy versioning, source scope, clock uncertainty, event corrections, policy migrations, and atomic index/replay semantics. Those are not silently approximated here.

## Run

\`\`\`sh
python3 research/broca-bitemporal-evaluation-v1/bitemporal_reference.py \\
  --events research/broca-bitemporal-evaluation-v1/fixtures/events.jsonl \\
  --queries research/broca-bitemporal-evaluation-v1/fixtures/queries.jsonl \\
  --output /tmp/broca-bitemporal-report.json

python3 research/broca-bitemporal-evaluation-v1/test_bitemporal_reference.py
\`\`\`

The 11 event records and 11 questions include:
- a value change that is not visible before its record time;
- a backdated correction received late, so as-known and as-corrected answers differ;
- a half-open validity boundary;
- an authority tie-break at the same effective time;
- same-rank disagreement that must produce an explicit conflict;
- redundant same-value witnesses whose provenance must be retained;
- no-evidence abstention; and
- a cutoff exactly before and exactly at a change's recorded time.

The supplied gold labels are synthetic and intentionally visible. Never treat 11/11 on this fixture as a quality estimate or tune this tiny fixture and call it held out.

## Paradigm shift and integration hypothesis

Treat memory as a **temporal evidence ledger**, not a mutable string store:

- Append immutable, provenance-bearing events rather than silently overwriting a value.
- Answer with a two-dimensional time query (valid_at, known_at).
- Represent **conflict** and **unknown** as first-class states rather than forcing a plausible value.
- Keep value selection separate from evidence provenance and policy authority.
- Make downstream retrieval consume only the evidence state that is valid for the requested intent.
- Preserve an auditable “why this answer” trace, including what was visible, selected, and shadowed.

The immediate integration experiment should be a Rust implementation behind a feature gate, then transactional updates/replay/reindexing against the same golden cases. Do not replace the existing retrieval path until the Rust implementation has exact semantic parity and rollback/rebuild tests.

## Benchmark plan and gates

Do not combine unlike benchmarks into one score.

| Lane | What it can establish | Required reporting |
|---|---|---|
| This bitemporal probe | Correct temporal cutoffs, late correction, authority tie, fail-closed conflict and unknown | Exact fixture accuracy, state accuracy, conflict/unknown rates, event provenance, hashes |
| MEME Cascade / Absence / Deletion | Multi-entity update and dependency reasoning, including absence and deletion | Official harness/version, per-task score, fixed reader/model and context/cost settings |
| LongMemEval-V2 / AgentRunbook-C V2 | End-to-end use of long histories for agent question answering | Official split, accuracy, query latency, reader identity and matching context budget |
| Agent Memory Leaderboard | Standardized multi-category textual/coding/multimodal memory assessment | Official track/round and system API settings; don't conflate preview submissions with released results |
| COLLABMEM | Role-aware sharing, provenance, routing, conflict resolution and retention | Official task splits; separate authorization/routing errors from answer errors |

Relevant public references:
- MEME, *Multi-entity & Evolving Memory Evaluation* (2026): https://arxiv.org/abs/2605.12477
- LongMemEval-V2 official implementation and AgentRunbook-C V2: https://github.com/xiaowu0162/LongMemEval-V2
- Agent Memory Leaderboard / Cycle 2 information: https://agentmemories.ai/evaluation
- COLLABMEM, EMNLP 2026: https://aclanthology.org/2026.emnlp-main.1194/

MEME's paper reports very low default average accuracy on its Cascade and Absence tasks in its tested configurations. This motivates the dependency lane but is not a universal baseline. The Agent Memory Leaderboard page currently previews Cycle 2 submissions separately from the released Cycle 1 table; use published, protocol-matched results when comparing systems. COLLABMEM explicitly treats provenance and authority as separate operational dimensions, so this temporal probe should eventually be paired with its official tasks rather than substituted for them.

### Decision gates for a real system

These are proposed internal gates, not existing results:

1. **Semantic correctness:** 100% on checked-in deterministic cases, including adversarial boundary/cutoff cases, plus property tests for input-order invariance.
2. **No unsafe selection:** 0 arbitrary winners on unresolved equal-authority conflicts; conflict must remain explicit through retrieval and answer construction.
3. **Temporal leakage:** 0 post-cutoff events in the as-known view, verified by negative controls on every case.
4. **Durability:** update, restart, replay and index rebuild produce the same query result and receipt for the same immutable inputs.
5. **End-to-end utility:** use a paired official LongMemEval-V2 or AgentRunbook-C run; require a preregistered quality or lifecycle-cost improvement with grouped uncertainty, identical reader/context budget, and exact run receipts.
6. **Cost and latency:** report p50/p95/p99 query latency, write/index/replay costs, evidence count, context tokens, abstention and conflict rates. A better accuracy number that silently spends more context or compute is not a fair comparison.

## Competitive boundary

This is a new reference-evaluation artifact, not an SOTA claim. The synthetic fixture is not an external leaderboard score, the resolver is not integrated into Broca, and no official MEME/LongMemEval/COLLABMEM run has been conducted in this package. Compare against industry only after identical harness, data split, reader, budget, and protocol are pinned.
