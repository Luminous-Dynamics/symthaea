# Broca cross-session proof coverage

**Protocol version:** BROCA-CROSS-SESSION-PROOF-V1  
**Status:** proof optimizer / independent verifier / reader and lifecycle provenance changes authored; Rust tests authored, not compiled or executed here.  
**Benchmark motivation:** BEAM multi-session reasoning at 100K–10M-token histories.

## Paradigm shift: cross-session proof closure, not longer context windows

Long-history memory should not be measured only by whether a relevant fragment can be retrieved from a large history. A difficult BEAM-style query often requires several premises distributed across different conversation sessions, with updates, contradictions and temporal constraints. The useful unit is the required claim and its complete proof dependency set.

Broca therefore treats cross-session evidence assembly as a proof-selection constraint. Each required conclusion may specify both:

- a minimum number of distinct **source families**, which expresses evidence independence; and
- a minimum number of distinct **trusted sessions**, which expresses cross-conversation coverage.

Those are separate axes. A different session does not automatically make a source independent, and two independent source families inside one conversation do not demonstrate multi-session reasoning.

## Exact data path

1. **Ingestion and provenance.** The ingestion adapter binds a retrieved evidence record to an immutable source revision and a trusted session identity. In the AML Text/Coding ingestion adapter, the record's `trajectory_id` is copied from the Add request's `session_id`; other ingestion paths must provide their own trusted session binding. Session identity must not be guessed from message text or timestamps.
2. **Retrieval and admission.** BM25 returns candidate evidence. The separate proof-admission envelope carries the canonical session ID alongside source revision/family, source pointer and digest, scope, bitemporal fields, authorization decision, conflict edges and exact rendered token cost.
3. **Proof request.** Each `ProofObligation` has `minimum_independent_families` and `minimum_distinct_sessions`. A zero session threshold preserves the earlier behavior; positive values require exact session provenance for the selected dependency set.
4. **Exact optimizer.** The optimizer independently counts source families and session IDs for every candidate proof alternative. Missing session provenance or insufficient sessions rejects that alternative. The admissible remaining-cost lower bound includes the same session constraint, so pruning cannot incorrectly remove a valid or optimal cross-session proof.
5. **Packet and independent verifier.** The proof packet carries the session ID for every selected evidence reference. The separate exhaustive optimality verifier independently recomputes both diversity constraints and rejects a packet whose source/session metadata, dependencies, cost or digest do not match.
6. **Reader hand-off.** The verified reader context retains source session IDs as control-plane provenance while exposing only exact proof excerpts as reader-visible text. A canonical `context_sha256` commits to the complete context (including session IDs, dependencies, exact excerpts and token accounting); post-build mutation fails `verify_internal_consistency()`. This digest detects mutation but is not a signature or authority token.
7. **Lifecycle action.** The opaque `VerifiedLifecycleProofToken` binds each required conclusion to its selected justification, exact evidence dependencies and exact distinct session set. A lifecycle claim that changes its session set cannot pass the final answer capability check.

## Non-substitution invariants

| Evidence pattern | Source-family count | Session count | Meets 2 families + 2 sessions? |
|---|---:|---:|---|
| Two independent sources in one session | 2 | 1 | No |
| One source family represented in two sessions | 1 | 2 | No |
| Two independent source families in two sessions | 2 | 2 | Yes, subject to temporal, authorization, conflict and proof checks |
| Two sources with missing session metadata | Depends on family IDs | Unknown | No; missing provenance is not evidence of session diversity |

Distinct revisions from the same source family do not count as independent families. The same session repeated over many revisions counts as one session. Session provenance is never inferred from a retrieval rank, source family, message content or number of retrieved documents.

## Qualification fixture corpus

The versioned [cross-session-proof-fixtures-v1.json](cross-session-proof-fixtures-v1.json) file contains four deterministic cases:

1. Valid proof with two independent source families across two sessions: accepted.
2. Two independent source families in one session: rejected for insufficient distinct sessions.
3. Two different sessions carrying one source family: rejected for insufficient independent source families.
4. Missing session provenance under a positive session requirement: rejected.

The Rust test suite reads the checked-in file and runs both the optimizer and the independent verifier on supported packets. These are synthetic conformance cases, not benchmark observations.

## BEAM connection and industry comparison

BEAM (ICLR 2026) evaluates long histories over 100 conversations and 2,000 validated questions, with history-size slices up to 10M tokens. It includes multi-session reasoning, event ordering, contradiction resolution, knowledge update, temporal reasoning and abstention. Its LIGHT architecture combines episodic memory, working memory and a scratchpad and reports 3.5–12.69 percentage-point improvements over its strongest baselines depending on backbone.

A published past.dev BEAM 10M run reports 85.03% aggregate accuracy across 200 questions and 10 conversations, but 28.17% multi-session reasoning on a much smaller 20-question slice. Provider values on the published comparison are separate runs with differing models, judges and configurations; they are not a matched head-to-head. The 28.17% result is an important diagnostic target, not a precise estimate of general performance.

Broca must first reproduce the official BEAM splits, task labels, answer model and judge settings. It should publish the ten ability scores separately, especially multi-session reasoning, contradiction resolution, event ordering, knowledge update, temporal reasoning and abstention. Never hide a low multi-session score inside a high aggregate.

## Metrics and prospective targets

Report these separately:

- **Session provenance completeness:** share of required evidence candidates with trusted session identity; zero unknown-session candidates may satisfy a positive threshold.
- **Cross-session proof coverage:** proportion of answerable required claims with a complete selected dependency set spanning the specified number of unique sessions.
- **Source independence:** number of distinct source-family IDs per selected proof, independently of session count.
- **False proof rate:** number of accepted packets that violate the session or source-family threshold; deterministic test target is zero.
- **Multi-session answer/action correctness:** official BEAM multi-session score and action-conditioned lifecycle score, reported by dataset split.
- **Temporal and conflict correctness:** valid_at / known_at replay, supersession/retraction decisions, conflict precision/recall and stale-use rate.
- **Cost and latency:** evidence tokens, packet tokens, number of sessions touched, p50/p95/p99 latency, retrieval calls and cost per complete cross-session proof.

**Prospective engineering target:** at least 70% on the official BEAM 10M multi-session slice, with no statistically significant regression in the other ability slices, zero invalid proof certificates in deterministic conformance tests, and explicit action correctness for stale/conflicted/unanswerable cases. The 70% figure is an ambition, not a reported Broca result or a claim of parity with another system.

Add 20M and 50M-token histories only as clearly labelled Broca stress tests; do not mix them into official BEAM scores.

## Current limitations

- The current proof solver can enforce the declared per-claim session threshold, but the production claim extractor / planner is not yet wired to derive those thresholds from arbitrary natural-language questions.
- The AML adapter's `trajectory_id == session_id` mapping is specific to its current ingestion contract. Other source types need a trusted session resolver and should not invent a session identifier.
- Session count is a provenance constraint, not proof of semantic entailment or causal independence. Two sessions may repeat the same underlying source; source-family and conflict constraints remain necessary.
- The fixtures are synthetic. The official BEAM dataset has not been run and no score has been measured.
- Rust tests are authored but not compiled or executed here; no runner success is claimed.

## References

- BEAM, ICLR 2026: https://proceedings.iclr.cc/paper_files/paper/2026/hash/d7f0cfa0fe759b033d5262e1bb7d4065-Abstract-Conference.html
- BEAM published results and ability breakdown: https://www.past.dev/benchmarks/beam
- Vera-Life lifecycle-action benchmark: https://aclanthology.org/2026.groundlm-1.18/
- Broca lifecycle action protocol: [LIFECYCLE_ACTION_POLICY.md](LIFECYCLE_ACTION_POLICY.md)
- Broca retrieval and proof benchmark protocol: [DERIVED_STATE_BENCHMARK.md](DERIVED_STATE_BENCHMARK.md)

