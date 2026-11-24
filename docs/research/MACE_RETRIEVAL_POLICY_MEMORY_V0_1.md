# MACE retrieval-policy memory v0.1

**Status:** Rust reference prototype and authored unit tests. The module has not been compiled or executed in the current environment. This is not a benchmark submission, hosted service, durable store, or production authorization system.

## 1. Research hypothesis

Persistent agent memory should contain two distinct planes:

1. **World/evidence memory** retains source records, facts, events, trajectory states, screenshots and their rights, lineage, freshness and lifecycle qualification.
2. **Retrieval-policy memory** retains reusable procedures for acquiring evidence: tool/action templates, productive search sequences, bounded failed-search observations, domain applicability, environment/tool revisions, measured cost and expiry.

The second plane is not a substitute for the first. A reusable strategy may say “inspect the order index before opening an individual order record.” It cannot establish that a particular order exists, make a source authoritative, grant access or override rights/lifecycle checks.

The method is motivated by the official LongMemEval-V2 AgentRunbook-C V2 Online approach, which consolidates reusable search strategies from prior interactions. Its reported operating point is +2.8231 LAFS gain, 75.6% Small accuracy and 130.5 seconds average query latency. These are external benchmark results, not MACE results. See:
- https://xiaowu0162.github.io/longmemeval-v2/agentrunbook-c-v2/
- https://xiaowu0162.github.io/longmemeval-v2/

## 2. Label-blind input boundary

A retrieval trace receipt may contain only:

- run ID and opaque query-invocation ID;
- domain, environment revision digest and tool-profile digest;
- exact query digest (never query text);
- optional candidate-set digest;
- ordered tool IDs, exact tool revision digests and parameterized action-template digests;
- operation outcomes, IDs of evidence actually observed, elapsed time and inspected bytes;
- stop reason, tool-call budget, actual calls and timestamp.

It has no field for a question ID, task category, gold answer, evaluator judgment, hidden attack assignment, correctness label, target action or benchmark score. A retrieval strategy signature excludes query content and retrieved evidence IDs. The promoted note stores supporting trace receipt digests, not the trace's facts or payloads.

The candidate process can report only what it actually executed and observed. It must not receive the evaluator's view of whether its final answer was correct. The old V1 `evidence_sufficient` label was overloaded: it meant only that the candidate-side retrieval process had a non-empty candidate evidence set. It never established independent admissibility, decision sufficiency or correctness. New sidecar observations use `candidates_returned`; the legacy label is accepted only when replaying older durable V1 receipts.

## 3. Trace validation

RetrievalPolicyMemoryV1::record_trace checks:

- non-empty, canonical run/invocation/domain identities;
- valid SHA-256 digests for query, environment, tool profile, executable revisions and action templates;
- bounded, non-empty operation sequences and exact operation/tool-call count agreement (256 tool calls maximum, 10,000 evidence IDs maximum per trace, 256-byte run/invocation/tool identifiers, 512-byte domain IDs and 1,024-byte evidence IDs);
- operation evidence IDs consistent with the declared outcome;
- aggregate elapsed nanoseconds and inspected bytes exactly equal the sum of operation measurements;
- acquisition-status constraints: `candidates_returned` requires a candidate-set digest and observed evidence; `no_candidate_evidence` cannot carry either; budget exhaustion requires the full declared budget; tool-error termination requires the final operation to have failed. Legacy `evidence_sufficient` receipts retain their original digest and decoding semantics but are never treated as decision evidence.

The trace receipt is a deterministic, version-domain-separated SHA-256 commitment. Replaying the same (run, opaque invocation ID) with the same receipt is idempotent. Rebinding that invocation ID to a different trace is rejected.

## 4. Promotion requires independent decision outcomes

Raw retrieval traces are acquisition observations only. A non-empty candidate set, repeated successful tool call, or repeated query pattern is not evidence that candidates were admissible, useful to a declared `DecisionNeed`, sufficient for the downstream reader, or causally responsible for a better decision.

Accordingly, `RetrievalPolicyMemoryV1::promote_strategy` now fails closed for candidate-return observations with `DecisionOutcomeReceiptRequired`. Legacy V1 `evidence_sufficient` receipts remain decodable for restart/replay compatibility but are subject to the same rejection. Empty-result, tool-error, budget-exhausted and abstained traces remain non-promotable as well.

The required future promotion chain is separate and invocation-bound:

1. A decision-scoped qualification receipt admits/rejects/marks-indeterminate each candidate under an explicit scope, purpose, authority, lifecycle snapshot and verifier profile.
2. A decision-sufficiency receipt binds the exact qualified-evidence digest and declared `DecisionNeed`.
3. A separately governed outcome receipt binds the decision and observed result without exposing evaluator truth to retrieval-policy memory.
4. A pre-registered paired held-out evaluation demonstrates positive decision utility at matched resource budgets, with task-family regression and security guardrails.
5. Only then may an exploratory strategy note become policy-eligible through a distinct promotion path.

Three retrieval observations may support a future exploratory hypothesis; they are not a policy-eligibility threshold and do not establish causal utility. This patch deliberately disables the previous trace-only promotion route rather than silently substituting a weaker proxy.

The strategy signature continues to identify an ordered tool/action template sequence, not source evidence, query text or answer content. A strategy note must carry the exact receipt digests and applicability envelope used by the independent promotion service.

## 5. Negative transfer and expiry

A note is applicable only if all of these match exactly:

- domain ID;
- environment revision digest;
- tool-profile digest;
- current time within [promoted_at, expires_at).

A site redesign, API revision, tool upgrade, permissions change, schema change or benchmark-domain change therefore invalidates reuse by default. The prototype does not silently generalize a strategy across a changed environment. A future version may add explicit revalidation receipts, but it should never remove the exact-match fallback.

## 6. Required benchmark ablation

Compare the following under the same pinned reader, judge, dataset and resource budget:

| Arm | Evidence/world memory | Retrieval-policy memory | Question answered |
|---|---|---|---|
| A | None | None | No-memory control |
| B | Raw BM25/dense | None | Retrieval-only |
| C | None | Strategy only | Can policy help without stored facts? |
| D | Facts/events/notes | None | Structured world memory |
| E | Facts/events/notes | Strategy | Dual-memory effect |
| F | Raw source + independently qualified evidence | Label-blind strategy with lineage/expiry | Full MACE design |
| G | AgentRunbook-C V2 Online | Native online strategy notes | Strong external policy-memory baseline |

Report these separately:

- cold-start and warmed-up accuracy / LAFS / latency;
- query order and independently randomized order permutations;
- leave-one-domain-out and leave-one-tool/API-version-out transfer;
- tool calls, search actions, inspected bytes/tokens and latency distributions;
- offline consolidation CPU/time/storage cost and online amortized cost;
- strategy-note count/growth, expiry rate and revalidation work;
- harmful negative transfer from stale notes;
- privacy and cross-domain leakage;
- downstream evidence admissibility and decision regret.

Mandatory controls:
- no strategy consolidation;
- random strategy-note selection;
- notes from the wrong domain;
- notes from an older environment/tool revision;
- repeated copies of the same query;
- a bounded unsuccessful search followed by a different query.

Never present warmed-up results as cold-start. Do not fold consolidation cost out of TCO or latency. Do not promote on answer labels or use the evaluator to select notes.

## 7. Security and governance boundaries

This local module includes an in-memory policy engine plus a local append-only receipt-store adapter. It does not provide:
- an independently administered or rollback-resistant trust anchor;
- multi-process atomic promotion;
- cryptographic issuer authentication;
- authorization to call tools or read sources;
- proof that a tool actually returned the cited source ID;
- evidence qualification, semantic truth or decision sufficiency;
- persistence of promoted strategy notes (trace receipts can be replayed, but promotion must be recomputed);
- defense against an attacker who can rewrite or roll back the trace store and its digests.

Production still requires separately administered witness custody, authenticated source-observation receipts, tenant-aware authorization, access-controlled note namespaces and replayable consolidation manifests. The local file adapter writes canonical receipts create-once via temporary-file fsync and atomic hard-link publication, restricts the directory to 0700 and files to 0600 on Unix, and replays strict receipt JSON into the in-memory policy store on restart. It shares the benchmark workspace/backup domain and therefore does not defend against whole-workspace rollback. Tool execution must independently enforce rights/scope regardless of strategy-note content.

A strategy note is advisory and untrusted. It may propose an action sequence; the execution layer must re-authorize every actual tool call and the evidence layer must qualify every result.

## 8. LongMemEval-V2 trace capture integration

The companion V2 sidecar now emits one policy trace for each official query invocation:

- tool: `trajectory.bm25.search`;
- tool revision: deterministic backend execution-profile digest;
- action template: SHA-256 of the BM25 search operation and its pinned domain/top-k/window/neighbor/k1/b/tokenizer parameters;
- observed evidence IDs: the candidate-evidence graph records produced for that query;
- stop reason: candidate-side evidence sufficient or no candidate evidence;
- latency: measured end-to-end sidecar query work;
- inspected bytes: explicitly incomplete because the current BM25 index does not expose posting/index bytes read.

The official opaque query invocation ID is used as the trace key. The Rust sidecar and Python shim share the same `retrieval_traces/` directory and invocation-derived filename. Rust publishes the canonical JSON receipt create-once through a temporary file, fsync and atomic hard link, then reloads all receipts into `RetrievalPolicyMemoryV1` at sidecar startup. Python independently recomputes the receipt SHA-256, verifies the exact persisted bytes, and checks that each trace pairs with one candidate-evidence receipt carrying the same candidate-set digest. Restore fails closed when either saved directory is missing and checks trace stop reason, operation/evidence consistency and resource-accounting completeness. Rust and Python share a pinned receipt-hash reference vector.

This is **single-operation trace capture**, not multi-step strategy learning. The V2 BM25 adapter does not yet choose alternate tools or automatically promote a search sequence. Persisted traces are restored across sidecar restarts and can support later re-promotion; strategy notes themselves are not yet persisted. The local trace store shares the benchmark workspace's rollback domain, so restoring the whole workspace can roll back the trace history too. Cross-query consolidation, persistent note storage and independent witness custody remain future work.

## 9. Durable receipt storage

The Rust reference file store is in `research/broca-mace-transfer-bench-v0-1/aml-adapter/src/retrieval_policy_store.rs`. It strictly decodes canonical trace JSON, binds filenames to the opaque invocation ID, enforces file/count bounds, rejects symlinks/non-regular entries and replays receipts into the policy store. Five tests cover restart replay, create-once idempotency/conflict, ambiguous post-publication in-memory binding recovery, tamper rejection and interrupted pending files. The store is a local conformance adapter, not an external witness service.

## 10. Rust API

The reference API is in:

research/broca-mace-transfer-bench-v0-1/aml-adapter/src/retrieval_policy_memory.rs

- record_trace(trace) validates and records a trace; returns its receipt digest.
- promote_strategy(refs, algorithm_revision, promoted_at, expires_at) deterministically creates a note from supporting traces.
- applicable_notes(domain, environment_revision, tool_profile, now) returns only currently valid notes matching every scope/version constraint.

The Rust unit tests cover invocation idempotency/rebinding, candidate-return status validation, legacy V1 replay, refusal to promote acquisition-only traces, duplicate query exclusion, domain/environment/tool-profile/strategy isolation, promotion-time bounds, resource accounting, invalid acquisition claims, insufficient search budgets, input-size limits, explicit byte-accounting completeness and a cross-language retrieval-trace receipt hash vector. These tests are authored; they remain unexecuted until a pinned Rust toolchain is available.

## 11. Next implementation stages

1. Compile, rustfmt, Clippy and execute the 13 policy-memory and 4 file-store tests at the exact PR head.
2. Add canonical schema and replay receipts for promoted strategy notes, then atomically persist those notes against the supporting trace digests.
3. Replace the single BM25 operation with a multi-step, label-blind tool/controller trace using the official opaque invocation ID and no evaluator metadata.
4. Add actual posting/index byte and RSS instrumentation; never infer zero from missing telemetry.
5. Run the seven-arm ablation with cold/warm and randomized order; compare AgentRunbook-C V2 Online using its native protocol.
6. Move trace and note custody into a separately administered witness/policy service, add independent source-observation receipts and qualification checks, and test whole-workspace rollback separately from local process restart.

**No MACE benchmark result is claimed by this research prototype.**
