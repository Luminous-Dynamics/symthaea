# Broca exhaustive event enumeration and speaker attribution

**Protocol:** `BROCA-EXACT-LEXICAL-ENUMERATION-V1`  
**Proof policy:** `BROCA-COMPLETE-ENUMERATION-PROOF-V1` (request-bound enumeration requirement; lifecycle action remains V2)  
**Status:** implementation and deterministic conformance tests authored; no Rust compilation or BEAM results claimed.

## Why this is a separate capability

Long-context memory benchmarks expose two related but distinct failures:

1. **Incomplete enumeration.** A top-k retriever can return plausible matching fragments but omit further matching events. The agent then answers a count or 'all instances' question as though the returned subset were exhaustive.
2. **Speaker misattribution.** Assistant-generated advice or summaries can be mistaken for statements made by the user. A relevant sentence is not necessarily evidence of who said it.

Mastra's 29 October 2026 BEAM analysis describes both failure modes: its 10M multi-session reasoning score is 25% (20-question ability slice), and it reports retrieval stopping after a few closest notes even when more matches exist; it also describes assistant advice being attributed to the user. These diagnoses motivate completeness and authorship as first-class proof requirements, not another reranking score.

## Exact enumeration semantics

`EvidenceIndex::enumerate_exact_lexical_all_terms_at_snapshot` scans every eligible text record in one EvidenceIndex state and returns all records containing every normalized query term. There is no top-k result cap, BM25 cutoff or score-based truncation.

The receipt commits to:

- algorithm version, normalized sorted query terms, required speaker role and scope;
- `valid_at` and `known_at`, plus the exact EvidenceIndex state digest;
- total scanned records and eligible text records;
- complete matching record count, distinct trajectory/session count and user/assistant/unknown speaker counts;
- sorted matching evidence IDs and a digest of the full match metadata.

`verify_exact_lexical_enumeration` rescans the index and compares the entire result. Omitted or injected IDs, changed source metadata, speaker changes, counter mutations, query changes and index-state drift fail closed.

**Important limit:** this proves completeness for the exact lexical all-terms predicate over the records in that bound index snapshot. It does not prove semantic recall for a natural-language question. Synonyms, paraphrases, implicit event classes, extraction failures and facts never represented in the index remain outside the guarantee. A query/predicate planner must translate the natural-language request into a validated event predicate before this receipt can support a user-facing count.

The current reference scan is O(N) in the number of indexed records. This is intentional for the correctness baseline. It is not yet a 10M-token latency solution.

## Speaker provenance

The AML ingestion contract stores `session_id` as `EvidenceSource.trajectory_id`, and encodes the original message role in the canonical source step ID (`user-N` or `assistant-N`). The speaker parser accepts only those two exact prefixes with a numeric suffix. Legacy or other source formats return `None`; unknown is not silently promoted to User.

At proof admission, speaker role is derived from the source-step ID on the snapshot-bound `LexicalEvidenceHit`. The caller cannot attach a free-form speaker label. `ProofEvidenceCandidate` and `ProofPacketEvidenceRef` retain the derived role; `VerifiedReaderEvidence` preserves it outside reader-visible text.

`ProofObligation.minimum_user_authored_evidence` counts only dependencies whose verified `source_speaker` is exactly `User`. Assistant and unknown evidence do not count. A user-only enumeration (`required_speaker_role = User`) automatically raises the proof's user-authored floor to the entire enumerated match count.

This is provenance attribution, not natural-language entailment. It says which source role produced the indexed text; it does not prove the text is true, nor whether an answer claim is logically entailed by it.

## Complete-enumeration proof binding

The supported count/all-instances path is:

1. Build the index at the requested scope, `valid_at` and `known_at`.
2. Enumerate the complete exact predicate, optionally filtering to User or Assistant.
3. Verify the enumeration against that same index state.
4. `EvidenceIndex::materialize_exact_lexical_enumeration_hits` re-verifies the receipt and materializes every hit with its source/session/step metadata.
5. `admit_complete_enumeration_for_proof_budget` requires a trusted admission envelope for every hit, matches the snapshot, source pointer, trajectory, step ID, digests, authorization, validity and token cost, then derives session and speaker provenance.
6. `ProofObligation::bind_verified_complete_enumeration` verifies the enumeration again and stores both its receipt digest and the exact required evidence ID set. It creates a conjunctive proof alternative containing every match.
7. The exact optimizer rejects any alternative omitting one required ID. Its admissible lower bound excludes incomplete alternatives.
8. The packet verifier and independent exhaustive optimality verifier re-check the full required set, user-authored floor, session diversity, source-family diversity, conflicts, temporal eligibility and packet digests.
9. The reader receives only verified source excerpts; speaker/session metadata remains in the control-plane context.

The request digest commits to the enumeration receipt digest and full required ID set. The packet digest commits to the request digest and exact evidence references. Thus a smaller top-k subset cannot be substituted for the full set and still pass the proof verifier.

**Zero-match case:** the current builder rejects an empty enumeration because there is not yet a dedicated proof-of-absence capability. Broca must abstain or use another justified route; a zero count is not authorized merely by an empty hit list.

## Non-substitution invariants

| Requirement | What counts | What does not count |
|---|---|---|
| Complete enumeration | Every ID returned by the verified exact predicate | Top-k, a relevance threshold, or a model's confidence that it has enough context |
| User-authored evidence | A snapshot-bound source step classified as `user-N` | `assistant-N`, `turn-N`, missing role metadata or a caller-supplied speaker label |
| Session diversity | Distinct trusted source trajectory/session IDs | More revisions from one session |
| Source independence | Distinct trusted source-family IDs | Different sessions containing copied material from one source family |
| Temporal admissibility | Same `valid_at` / `known_at` and eligible source revision | Current-state overwrite or stale evidence selected only by similarity |

## Metrics

Report these separately rather than collapsing them into one memory score:

- **Exact predicate completeness:** missing-match rate, injected-match rate, exact match-set equality and receipt verification rate. Conformance target: 100% exact set equality and 0 accepted omissions/injections.
- **Count exactness:** absolute count error, exact count accuracy and overcount/undercount rates, separated by 1, 2–5, 6–20, and 20+ matching events.
- **Speaker attribution:** user/assistant/unknown confusion matrix; user precision, recall and F1; assistant-to-user false attribution rate. Deterministic parser target: 100% correct on canonical AML role IDs, unknown IDs never counted as User.
- **Cross-session closure:** number of distinct sessions in the complete match set and percentage of required claims whose verified proof contains all matching IDs across the requested sessions.
- **Proof soundness:** invalid complete-enumeration certificates, omitted required evidence IDs, mismatched index digests and invalid speaker-role bindings. Target: zero in deterministic conformance and adversarial mutation suites.
- **Action quality:** correct ANSWER / RETRIEVE_MORE / WARN_STALE / EXPLAIN_CONFLICT / ABSTAIN action, with false-answer rate on incomplete, unknown-speaker, stale or conflicted records.
- **Cost:** records scanned, posting-list entries visited, candidate count, evidence tokens, proof packet tokens, p50/p95/p99 latency, CPU time and cost per exact supported count.

Also report answer quality with fixed reader/answer model, prompt, tokenizer and judge. Retain raw answers and per-question receipts. Use a second blinded judge or human review sensitivity analysis; the same saved answers have been reported to vary by 10.9 percentage points between two BEAM judges.

## BEAM and industry qualification

BEAM (ICLR 2026) includes 100 conversations and 2,000 validated questions over 100K, 500K, 1M and 10M-token history tiers. Report all ten ability scores; do not hide multi-session reasoning inside the aggregate.

Published vendor-run BEAM values are not directly comparable because answer models, judges, prompts and configurations differ. Examples:

- past.dev reports 85.03% aggregate at 10M and 28.17% multi-session reasoning on its 20-question 10M ability slice.
- Mastra reports 66.4% two-run mean at 10M (±5.8 points) and 25% multi-session reasoning on its 20-question 10M slice. Its main answer model is Claude Opus 5.5 and judge Gemini 3 Flash.
- OmniMemEval's shared-harness cohort uses GPT-4.1-mini answers and GPT-4o-mini judging; its 10M results include LCM-X 63.32%, Hindsight 59.75% and MemOS 56.75%. This harness is maintained by MemTensor, maker of MemOS, and is not independent certification.

These are separate comparison classes. Do not compare a vendor self-report to a shared-harness score as though they were one controlled leaderboard. See [INDUSTRY_BASELINE_LEDGER.md](INDUSTRY_BASELINE_LEDGER.md) and [industry-baseline-ledger-v1.json](industry-baseline-ledger-v1.json).

### Prospective Broca targets

These are engineering targets, not current results:

- 100% complete ID-set agreement and zero accepted partial enumerations for every exact-predicate fixture;
- 100% user/assistant attribution on canonical AML source steps; zero unknown-role records satisfying user-only constraints;
- 100% soundness on packet, receipt, index-state, source-step, role, session and source-family mutation tests;
- at least 95% exact count accuracy on a human-validated count-query split, while reporting exact lexical coverage separately from semantic predicate coverage;
- at least 70% BEAM 10M multi-session reasoning on the official split, with no statistically significant regression in other BEAM ability slices;
- p95 full-scan latency as a measured baseline, then an exact postings-based enumerator with the same result set and 5–10x lower p95 at 10M history scale (prospective engineering target; do not trade completeness for speed);
- report answer-level paired deltas, cluster confidence intervals, judge sensitivity and realized all-in token/latency costs.

## Next paradigm shift: predicate planning plus exact postings

The reference enumerator deliberately scans the index. The production direction is an explicit query predicate compiler:

1. Extract typed event predicates (actor, action, object, attributes, temporal range, scope and required speaker role) from count/all-instance questions.
2. Normalize aliases and synonyms with a versioned, auditable lexicon. Preserve which normalization rule fired; do not silently convert semantic similarity into exact evidence.
3. Execute the predicate over an exact inverted index with per-token postings, role/session metadata and bitemporal eligibility. Intersect postings, then apply exact predicate verification.
4. Produce a receipt containing the predicate digest, ontology/lexicon version, index snapshot, posting segments scanned, candidate count, result IDs and independent verification digest.
5. For predicates whose semantic normalization is uncertain, return a bounded/unknown result and retrieve more or abstain rather than calling lexical completeness semantic completeness.

This shifts memory evaluation from 'did top-k return a plausible fact?' to 'did the system enumerate and prove every event satisfying a declared predicate, with correct authorship and temporal provenance?'

## Limitations

- Exact lexical enumeration is O(N) and currently scans the in-memory EvidenceIndex; persistent/incremental posting lists are not implemented.
- Semantic predicate extraction, synonym/alias coverage and event ontology are not implemented.
- Zero-match proof-of-absence is not implemented; the complete-enumeration binder rejects empty result sets.
- AML `user-N` / `assistant-N` is the only speaker mapping currently recognized. Other sources must define a trusted source-step convention or remain Unknown.
- The source-family admission value still requires its own trusted identity/independence policy.
- The receipt is a digest commitment, not a signature, proof of correctness of the upstream source, or protection against an index built from an incomplete source corpus.
- Synthetic fixtures and authored Rust tests are not BEAM benchmark measurements. Rust tests have not been compiled or executed in the current environment.

## References

- BEAM, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/d7f0cfa0fe759b033d5262e1bb7d4065-Abstract-Conference.html
- Mastra Observational Memory BEAM report (29 October 2026): https://mastra.ai/research/observational-memory-beam
- OmniMemEval canonical results: https://github.com/MemTensor/OmniMemEval/blob/main/docs/user_memory/results.md
- past.dev BEAM results: https://www.past.dev/benchmarks/beam

