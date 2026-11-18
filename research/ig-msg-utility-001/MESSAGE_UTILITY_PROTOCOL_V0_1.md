# IG-MSG-UTILITY-001 — protocol v0.1

## Question

Can a multi-agent system reduce redundant receiver context while preserving every piece of evidence necessary for correct decisions, independent review and provenance?

## Direct comparator

You et al., *MessageSieve: Runtime Message Pruning for Accurate and Efficient LLM-Based Multi-Agent Communication*, EMNLP 2026: https://aclanthology.org/2026.emnlp-main.2179/

The reported approach acts on message instances before receiver prompt construction. It uses lightweight state cards and local message comparisons to retain full text, a compact evidence representation or omit a message. Its reported 19–49% token reduction and +2.59–4.10 macro-average improvements are specific to the paper's tested two backbones and five benchmarks; reproduce before comparing.

## Treatment arms

1. Dense Full-MAS.
2. Simple token truncation.
3. Agent/edge pruning (separate structural baseline).
4. Round-robin / last-message-only.
5. MessageSieve reproduction where released artifacts and license permit.
6. Evidence-aware message-level sieve.
7. Compact evidence atoms with original transcript digests.
8. Targeted recipient pull for missing evidence.
9. Dissent-preserving policy for minority/negative facts.
10. Symthaea hybrid with #12762 discovery, #12775 integration and #11383 deliberation control.

Match task, model revision, prompt/tool scope, group size, evidence assignment, total tokens, tool calls and deadline. Count every selector, state-card, summarizer, synthesizer and audit call. Use identical backbones within a comparison; then repeat across model families.

## Message semantics

Every message carries:
- immutable message ID and digest;
- evidence atom ID and originating evidence-root ID;
- source agent and source receipt;
- polarity, scope, freshness and uncertainty;
- authorization/recipient policy;
- transformation (original, compact, omitted);
- token count before and after transformation;
- audit reference to the original payload.

An omitted message is not deleted from the audit record. Re-forwarding or paraphrasing an existing root cannot create a new independent observation.

## Metrics

**Outcome:** task accuracy, correct abstention, decisive evidence recall, negative evidence use, warranted minority revision, source/entity/attribute/polarity integration.

**Efficiency:** total input/output tokens for all model calls, unique evidence roots per 1,000 tokens, decisive evidence recall per 1,000 tokens, redundant-content ratio, p50/p95 latency, cost per correct verified decision.

**Fidelity:** source and root retention, semantic field preservation, digest traceability, omission/compression errors, freshness/scope/uncertainty errors.

**Privacy/fairness:** sensitive attribute leakage, receiver-state inference, unauthorized disclosure, cross-episode linkage, per-agent evidence access parity.

**Control:** send/compact/omit/pull/stop appropriateness and verified marginal value of the next transmission.

## Negative controls

- Duplicate messages with different IDs but the same evidence root.
- Echo chains A→B→C→A.
- Unique minority-held decisive evidence.
- Unique negative/disconfirming evidence.
- Polarity reversal.
- Missing source, lineage, caveat, scope, uncertainty or freshness.
- Stale evidence made to appear current.
- Unauthorized facts both retained and omitted.
- Missing original digest.
- A compact representation whose token cost exceeds the original.
- A required evidence atom omitted.

## Proposed targets (internal, not achieved)

- ≥20% total-token reduction vs dense messaging at a predeclared non-inferiority margin no larger than 2 percentage points of task accuracy.
- ≥15% reduction in cost per correct verified decision, counting all auxiliary model calls.
- 100% duplicate-root invariance and provenance preservation in deterministic fixtures.
- No statistically supported loss of decisive-evidence recall, negative-fact use or warranted minority revision.
- Held-out task-family and model-family transfer.
- Full quality/cost/latency/privacy Pareto frontier.

The non-inferiority margin, sample size, multiplicity policy and model budgets must be frozen before confirmatory evaluation. Do not claim differential privacy from token pruning or minimized messages.
