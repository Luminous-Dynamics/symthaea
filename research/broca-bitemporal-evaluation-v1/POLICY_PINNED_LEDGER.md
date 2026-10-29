# Policy-Pinned Evidence Ledger v1

**Status:** research-only reference semantics and synthetic fixtures. Not integrated with production Broca, and not an official benchmark score.

## Why this lane exists

The first bitemporal prototype placed authority_rank inside each evidence event. That is unsuitable as a trust boundary: an untrusted event can simply claim a higher rank. This follow-on moves rank, allowed attributes, and retraction authority into a separately versioned policy registry.

The event schema therefore has no authority field. Adding one is a strict-schema error. An event's source is admitted only if it was registered and authorized for that attribute under the policy that existed when the event was recorded. At query time, current authority ranking and scope are derived from the latest policy version recorded by the query's known_at cutoff.

This creates two explicit policy times:
- **Admission policy:** was this source permitted to record/retract the event when it was written?
- **Query policy:** what authority ordering and scope did the system know at this query's transaction-time cutoff?

The policy JSONL's bytes are included in the receipt digest. A digest proves which policy file was evaluated, not that the registry owner was authentic. Cryptographic writer identity, signed admissions, key rotation and distributed consensus are not implemented here.

## Retractions are append-only events

Assertions and retractions use different exact schemas. A retraction names the immutable event ID it targets. It becomes visible only at and after its own recorded_at timestamp; as-known queries before that timestamp still see the evidence. Retraction authorization is checked using the policy version available at the retraction's recorded time, so a later policy change does not retroactively authorize a formerly unauthorized retraction or invalidate a previously admitted one.

The prototype rejects dangling targets, duplicate IDs, multiple retractions of the same target, and a retraction recorded earlier than its target. Revocation/retraction is not physical erasure: the audit trail remains available, while the target is removed from the eligible evidence set after the authorized retraction becomes known.

A production system still needs a reviewed reinstatement/correction protocol, cryptographic source authentication, tenant/entity scope, deletion requirements, policy signatures, atomic storage/replay, concurrency control and rebuild parity.

## Synthetic fixtures

The fixtures/pinned-ledger corpus contains:
- **2 authority-policy versions**, with an exact March 15 boundary at which incident-review authority changes;
- **9 append-only events**, including a late correction, two competing sources, an authorized retraction, a fake out-of-scope assertion, an unregistered source, an unauthorized retraction, and a properly scoped note;
- **12 temporal queries**, including exact policy-change and retraction boundaries, before/after comparisons, and a source-scope control.

The intended result progression for the January 20 endpoint query is:
1. Before the late assertions are recorded, answer db-primary-a.internal.
2. After E003 is recorded but under policy v1, answer db-primary-c.internal.
3. After policy v2 is recorded, incident-review outranks operations for the same valid-time start, so answer db-primary-d.internal.
4. After the authorized retraction R005 is recorded, return to db-primary-c.internal; unauthorized R006 must not remove E003.

This is not a universal truth policy. It is a deterministic test of versioned authority and transactional retraction semantics.

## Run

From the repository root:

~~~sh
python3 research/broca-bitemporal-evaluation-v1/policy_pinned_ledger.py \
  --output /tmp/broca-policy-pinned-ledger-report.json

python3 research/broca-bitemporal-evaluation-v1/test_policy_pinned_ledger.py
~~~

The report includes exact file digests for policies, events, queries and evaluator source, the policy version selected per question, event admission decisions, eligible/selected evidence IDs, applied retractions and exclusion reasons.

## Metamorphic and adversarial properties to add next

The existing metamorphic audit covers the earlier bitemporal resolver. Port equivalent perturbation tests to this policy-pinned lane:
- input ordering never selects a different winner;
- adding an event with a future recorded_at never changes an earlier as-known query;
- adding a lower-authority source cannot override an authorized higher-authority claim;
- an unknown or out-of-scope source cannot introduce a claim;
- an unauthorized retraction cannot remove admitted evidence;
- a valid retraction is absent just before its recorded_at and applied at/after its cutoff;
- policy changes are invisible before their recorded_at;
- adding authority_rank into an event schema is rejected;
- report digests change when policies/events/queries change.

The 12-query fixture accuracy is a consistency check only. The authored Python tests and CLI have not yet been executed from a full repository checkout. Independent mirrors must be described as such; they are not official benchmark results.

## Industry comparison

Use each official benchmark for the question it measures; keep scores separate:
- [MEME](https://arxiv.org/abs/2605.12477): evolving multi-entity memory, including cascade, absence and deletion;
- [LongMemEval-V2 / AgentRunbook-C V2](https://github.com/xiaowu0162/LongMemEval-V2): long-history answer quality and latency;
- [COLLABMEM](https://aclanthology.org/2026.emnlp-main.1194/): scope-aware sharing, provenance, routing and conflict management;
- [Agent Memory Leaderboard](https://agentmemories.ai/api-guide): standardized temporal reasoning, governance, privacy and other tracks;
- [STATE-Bench](https://opensource.microsoft.com/blog/2026/05/19/introducing-state-bench-a-benchmark-for-ai-agent-memory/): end-to-end stateful agent-task success, reliability, cost and user experience;
- [MemEvolve / EvolveLab](https://proceedings.mlr.press/v306/zhang26fa.html): modular memory architectures and meta-evolution under a unified benchmark substrate;
- [TAME / Trust-Memevo](https://aclanthology.org/2026.emnlp-main.2100/): trust degradation and memory misevolution during continual adaptation.

A meaningful claim requires fixed and disclosed reader/model settings, split and context budgets; official scoring; per-question receipts; paired statistics clustered by scenario/session; and cost including ingestion, indexing, replay and querying. The current synthetic numbers are not comparable to any leaderboard score and do not establish state-of-the-art performance.

## Proposed gates

1. Exact semantic tests pass, including policy time, retraction time, scope and authority boundaries.
2. Zero admission of unregistered/out-of-scope sources; zero use of event-supplied authority.
3. Zero unauthorized retraction effects and zero future-event leakage.
4. Identical semantics after update, restart, replay, index rebuild and rollback.
5. A paired official end-to-end benchmark shows either the preregistered accuracy improvement with grouped uncertainty, or a lifecycle-cost reduction at a preregistered non-inferiority margin.
6. No production promotion without authenticated source identity and policy distribution; JSON hashes alone are not authentication.


## Rust-side API

The branch also exports a typed research module at crates/domains/symthaea-broca/src/memory_governance.rs. It uses the crate's existing serde, chrono and sha2 dependencies; it is separate from the JSONL worker protocol and must not be treated as production-integrated.

The Rust API exposes strict event JSON parsing, source/policy types, a validated PolicyPinnedLedger, a query object, resolution state, selected evidence provenance, applied retractions, excluded-event reasons, and a deterministic digest over normalized policy versions. The module contains 17 authored unit tests.

Once the workspace Rust toolchain is available, run:

~~~sh
cargo test -p symthaea-broca memory_governance --lib
~~~

No Rust compilation or Cargo test result is claimed in the current research environment. Before integration, require Rust/Python fixture parity, authenticated writer identity, signed policy distribution, and update/restart/replay/index-rebuild tests. The existing worker's JSONL request protocol is unchanged.
