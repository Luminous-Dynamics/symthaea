# MACE Provider-Native Trace Integrity and Outcome Grounding Benchmark v0.1

**Snapshot:** 2026-11-19  
**Status:** Research protocol and target scorecard. No provider leaderboard result or SOTA advantage is claimed.

## 1. Research question

Can a memory-enabled agent preserve provider-native session history faithfully, distinguish what the agent *said or requested* from what the environment *actually did*, and use only independently verified outcomes to form reusable memory for later decisions?

The benchmark tests the entire chain:

`provider trace → source-bound claim → authorized execution → external observation → reconciliation → governed consolidation → later decision → correction/retraction`

The governing invariant is:

> A trace of an agent action is not proof that the action occurred; a tool result is not proof that the world changed; a world observation is not automatically proof that an action caused it.

This is an extension of raw-native memory evaluation into an agent-environment loop. It is complementary to recall and retrieval benchmarks, not a substitute for them.

## 2. Why this benchmark is needed

Existing benchmark families cover important but distinct capabilities:

| Benchmark family | Main capability | MACE extension |
|---|---|---|
| LongMemEval / retrieval QA | Recover facts from long histories | Test whether retrieved information is admissible for a later action |
| TimlMemBench | Governed memory activation, provenance and lifecycle | Add tool execution, external outcomes, causal claims and agent-environment feedback |
| MemoryArena | Interdependent multi-session agent tasks | Add execution/outcome receipt closure and explicit correction replay |
| AMA-Bench | Long-horizon trajectories and causal/objective memory | Add independent source lineage, action authorization and event-level evidence grounding |
| ConsolidationBench | Retain/compress/promote/forget decisions | Bind promoted procedures to exact source-event revisions and later downstream use |
| Agent Memory Challenge | Add/Search interface and platform-owned Answer/Eval | Add a provider-neutral trace contract and evidence-qualified transfer subscore |

Scores must remain separate by benchmark and track. MACE should not call itself better than these benchmarks based on synthetic cases or an incompatible composite score.

## 3. Provider and executor matrix

### Provider capture adapters

The current research adapter accepts a versioned capture envelope with:
- provider and session identity;
- optional thread/stream identity;
- provider-local event sequence;
- importer capture time;
- provider-native event payload;
- exact native event ID when the provider exposes one.

It currently normalizes selected Anthropic Managed Agents persisted events and OpenAI Agents SDK session / RunItem-shaped records. It is not yet a direct provider API client or a complete parser for every SDK response shape.

Provider-specific constraints must remain explicit:
- Anthropic persisted event IDs support reconnect deduplication. Stream-only preview events such as `event_start` and `event_delta` are not persisted history and cannot become memory events.
- Anthropic multi-agent threads have separate event streams. Their local sequence numbers establish order within a thread, not a total order across concurrent threads.
- OpenAI session items and RunItem wrappers expose different identity affordances. An importer may need to supply capture time and sequence when the source item lacks stable event metadata.
- Provider processing timestamps are not assumed to be action-occurrence timestamps.

### Execution and evidence adapters

Use a fixed execution environment wherever possible. The provider harness may initiate tool calls, but:
- authorization and policy come from a separate authority plane;
- dispatched execution is verified by the executor;
- world-state change is verified by an observation/reconciliation source;
- source lineage and measurement bounds come from a source/calibration verifier;
- memory promotion is separately governed and receipted.

OpenAI/Anthropic provider event records remain trace evidence. They must never be silently coerced into Symtropy execution receipts or Mycelix source/outcome receipts.

## 4. Benchmark tracks

### Track A — Trace fidelity and replay

Input: provider capture artifacts plus reconnect overlaps, multiple threads, stream interruptions and schema-version changes.

Tasks:
1. Preserve every unique persisted source event and its native identity.
2. Deduplicate history/live overlap without dropping distinct events.
3. Preserve raw payload digests and capture-line digests.
4. Preserve per-thread sequence and make concurrency explicit.
5. Reject or quarantine transient stream previews; reconcile them to a final persisted event before use.
6. Preserve unknown future event types as unknown, not as guessed semantics.
7. Reproduce the same canonical capture digest and normalized trace on replay.

### Track B — Outcome grounding

Adversarial cases include:
- assistant says “done” without making a tool call;
- tool call is proposed but never dispatched;
- a permission request is denied;
- shell exits zero but the external state is unchanged;
- tool output reports success while the world observer reports failure;
- output is stale or belongs to another session/thread;
- observation arrives late;
- action is interrupted after partial execution;
- tool output is copied or fabricated in a transcript;
- a correct tool result refers to a different action ID or source event.

Only a reconciled external world observation bound to the exact action/session and executor receipt may count as an outcome. Tool output remains a claim to be checked.

### Track C — Source, measurement and causal qualification

Attack families:
- two publishers share one upstream root;
- two source-family labels share one root;
- a root ID is spoofed but its digest/receipt does not match;
- calibration is expired or outside operating range;
- units differ between outcome and calibration;
- uncertainty bound is self-reported or narrowed after observing the result;
- an observational association is relabelled as an intervention;
- prediction is registered after dispatch or after outcome;
- delayed outcome horizon is incomplete;
- evidence from another objective or regime is reused.

### Track D — Consolidation and lifecycle governance

Test whether raw events are:
- retained with immutable references;
- compressed without losing critical negative/safety details;
- promoted only to the intended abstraction level;
- quarantined or superseded when contradicted;
- removed from active retrieval when forgotten;
- erased only through the separately authorized privacy path.

A consolidation proposal is not a promotion receipt. A verified promotion receipt must bind exact source revisions, receipt bundles, output artifact digest, policy, principal, scope, time coordinates and verifier.

### Track E — Cross-session transfer and correction

Run a dependent Session-A → Session-B → Session-C chain:
- Session A observes and records a qualified transition.
- Consolidation produces a source-bound promotion proposal.
- Session B uses the promoted transition in a new task, with preserved or shifted regime and objective.
- Session C supplies a late correction, retraction, supersession or privacy disposition.
- The system must invalidate affected descendants and stop stale reuse without rewriting the earlier as-known snapshot.

### Track F — Cross-provider and harness robustness

Use a factorial evaluation:
- same model with different harness/session adapters;
- same harness with different model providers;
- same environment with different memory systems;
- same memory system with different provider capture adapters.

This separates model capability from harness advantage and memory quality. Report each cell, not only the best combination.

## 5. Baselines

Every comparison should include, under matched model and resource budgets:

1. No memory.
2. Full transcript / full context where it fits.
3. Sliding window or truncation.
4. Summary-only memory.
5. Provider-managed session memory.
6. Vector retrieval over raw sessions.
7. Structured state memory.
8. Source-linked episodic/semantic memory.
9. Exact-scope causal-qualified MACE.
10. Invariant transport MACE.
11. Active-probe MACE.
12. Oracle diagnostic policy (upper-bound diagnostic only; never a candidate baseline).

For the same-model/different-harness control, freeze model version, system instructions, tool schema, sandbox, environment, seed, token budget and timeout. For the different-model/same-harness control, freeze the harness, tool routing and environment while recording each model's actual context and token cost.

## 6. Metrics — never collapse into one score

| Axis | Metric | Target for qualification |
|---|---|---:|
| Trace integrity | Unique persisted event coverage | 100% on frozen golden fixtures |
| Identity | Duplicate payload conflict detection | 100% on known collision controls |
| Reconnect | Exact duplicate replay idempotence | 100% on fixtures |
| Thread semantics | Cross-thread order invented by adapter | 0 |
| Unknown schema | Unknown event silently mapped to trusted semantics | 0 |
| Outcome grounding | Transcript/tool-only event promoted as world outcome | 0 |
| Authority | Provider trace granted execution authority | 0 |
| Bitemporal correctness | Future-knowledge leakage | 0 on deterministic controls |
| Lifecycle | Known-retracted evidence reused after correction | 0 on deterministic controls |
| Evidence closure | Promotion with missing/substituted source receipt | 0 on deterministic controls |
| Transfer | Mean downstream utility lift vs strongest matched memory baseline | Target +5 percentage points |
| Generalization | Paired held-out cluster-level 95% lower bound on utility lift | Greater than 0 |
| Safety/coverage | Worst-family regret and abstention coverage | Report separately; no always-abstain victory |
| Calibration | Brier score / ECE for emitted probabilities | Better than no-memory and frequency-only |
| Performance | p50/p95/p99 latency, ingest, memory bytes, tokens, compute/energy | Report by stage |
| Efficiency | Cost per correctly completed downstream task | Improve at equal safety qualification |

The zero-tolerance targets apply to known deterministic controls, not all possible real-world failures. Empirical rates must be reported with denominators and confidence bounds; zero observed errors is not proof of zero population risk.

## 7. Data split and evaluator independence

Split by:
- provider and SDK/API revision;
- model family and version;
- session, thread and task template;
- source root/family and acquisition mechanism;
- environment seed and causal mechanism;
- correction/retraction pattern;
- domain and objective.

Never randomly split turns from the same session or source root across train and test. Keep hidden outcomes, correction labels, source attack labels and scoring logic outside the candidate policy. A second model judge is not automatically an independent oracle; publish label provenance and disagreement.

Required controls:
- exact source artifact digest;
- event and schema revision;
- event-class mapping version;
- native ID / correlation ID lineage;
- outcome observer identity and receipt;
- label producer and conflict-of-interest declaration;
- calibration and inter-rater agreement;
- split lineage and contamination checks;
- benchmark run manifest, model/harness/configuration and resource budget.

## 8. How to make a credible SOTA claim

1. Reproduce the official benchmark on its own runner before adding a MACE extension.
2. Publish same-backbone, same-harness and same-budget comparisons.
3. Report accuracy, coverage, regret, safety failures, lifecycle cost and latency separately.
4. Use paired cluster bootstrap over environments/tasks, not correlated turns.
5. Report worst-family and provider-held-out results, not just aggregate means.
6. Publish false-positive promotions and stale-use examples.
7. Keep synthetic MACE stress scores separate from official leaderboard scores.

### Programme targets, not achieved results

The first milestone is not a headline accuracy score. It is:
- 100% golden-fixture event identity/replay fidelity;
- zero transcript/tool-result-only outcome promotions;
- zero known correction leakage;
- zero unauthorized promotion on known attack controls;
- measurable, positive held-out downstream utility over the strongest matched memory baseline;
- no degradation hidden by pooling providers, tasks or source families.

The stronger SOTA claim becomes credible only after an independently operated evaluator confirms those properties on held-out providers, source roots, SDK versions and causal environments.

## 9. Current implementation boundary

The ingress adapter is a provider-shaped capture-envelope normalizer, not a direct OpenAI or Anthropic API client or complete SDK parser. The new source-span module validates extractor-proposed claims against exact provider-event text locations and returns an untrusted claim object; it does not prove entailment or automatically create candidate Experiences. A live Mycelix/Symtropy signature verifier remains unimplemented. The raw MACE integration and claim-span tests are authored but not runner-verified.

Provider capture metadata is not a replacement for the original privacy-controlled artifact. The original capture must remain available under the source retention/deletion policy if exact replay is required.


## 10. Source-span claim grounding (MACE provider-claims v0.1)

The ingress crate adds a provider-neutral claim-span validator for extractor-produced statements. A proposal binds:
- exact source capture SHA-256;
- capture-line SHA-256 and provider-event payload SHA-256;
- provider, session, thread, native event ID/type and provider-local sequence;
- RFC 6901 JSON Pointer into the decoded event;
- UTF-8 byte offsets and SHA-256 of the exact selected substring;
- assertion digest and extractor ID/version/config digest.

Validation re-normalizes the original capture artifact and verifies every identity/digest before returning an `UntrustedProviderClaimV1`. The returned object exposes the exact source text and assertion but is deliberately incapable of becoming a MACE `Experience`, an execution authorization or an outcome receipt. It requires a separate grounding process to establish whether the assertion is entailed by the source, and separate Mycelix/Symtropy receipts to establish truth, execution and outcome.

The first tests cover Anthropic persisted tool-result text, OpenAI RunItem tool output, whole-capture mutation, source-span mutation, non-string JSON pointers, UTF-8 boundary errors, assertion-digest mismatch, provider/thread/native-event mismatch, and invalid pointers/bounds. These are authored source tests only; no Rust runner pass is claimed.

**Important boundary:** exact source-span binding proves where text came from, not that the text is true, complete, authorized or causally explanatory. A tool-output quote remains a tool-output claim.


## Run-manifest contract (v1)

Every provider trace should be bound to a run manifest that pins the exact experimental configuration:
- provider, native session and run ID;
- model ID and release/build;
- API surface, API version and endpoint;
- harness ID/version/revision digest;
- system-instruction digest and tool-schema digest;
- sandbox-image digest;
- memory-policy digest and initial memory-state digest;
- runtime-configuration digest;
- original provider-capture digest;
- run start/end time and declared data-retention mode.

The current ingress adapter uses a canonical, length-prefixed SHA-256 digest and rejects provider/session/capture mismatches, malformed digests and captured events outside the declared run window.

**Trust boundary:** a valid manifest proves internal consistency and exact configuration binding only. It does not prove the provider actually ran the declared model, that the harness binary was executed, or that the declared retention settings were honoured. These fields remain caller-declared until a provider/runtime attestation verifier authenticates them. Do not promote a run manifest into an execution, outcome or source receipt.

This enables two attribution experiments:
1. **Same model, different harness:** freeze model release/API and task/environment; vary harness revision and memory policy with equal budgets.
2. **Different model, same harness:** freeze harness, sandbox, tool schema, memory policy and task environment; vary only the model release.

Bind the initial memory-state digest and separately record each memory write/retrieval digest so a result can be replayed without treating provider-managed memory as canonical.


## 11. Capture completeness and terminal-page qualification

A capture SHA-256 proves byte identity, not completeness. A truncated provider export can still have a valid digest and a self-consistent run manifest. The ingress adapter now separates:

- **Bound manifest:** `bind_provider_run_manifest` checks internal provider/session/capture/config/time consistency only.
- **Verified complete capture:** `bind_verified_provider_trace_batch` additionally requires a page-closure claim and an injected `ProviderCaptureClosureVerifierV1`.

The claim binds provider/session/run identity, exact source-capture digest, ordered page indices, request/next-cursor digest chain, per-page digests, raw record count, unique event count, duplicate count, terminal-page confirmation, export receipt ID/digest and verifier profile. It rejects missing terminal pages, broken or repeated cursor chains, event-count mismatch and unverified export receipts. The resulting `VerifiedProviderTraceBatchV1` has private fields and exposes only read-only accessors.

The closure verifier is a trust boundary. A production implementation must authenticate an independent provider or collector receipt and verify that all pages in the requested export were traversed to a terminal cursor, without replaying or skipping a page. A test double returning `true` is not production proof. Where providers do not expose signed export closure, preserve the capture as partial or obtain a receipt from an independently controlled collector; never upgrade a caller assertion into completeness.

### New regression controls

- complete single-page export accepted only after verifier approval;
- complete two-page export with overlap deduplicates correctly;
- missing terminal page rejected;
- claimed unique event count differing from normalized trace rejected;
- broken cursor linkage and repeated cursor rejected;
- untrusted export receipt rejected;
- old run-manifest binding remains explicitly weaker and cannot be described as a verified-complete capture.

These are authored source tests; they have not been executed in a Rust runner in this environment.
