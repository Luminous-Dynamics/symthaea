# AR-5B5 Anytime Evaluation History Implementation v0.1

**Status:** Rust implementation committed; compilation, formatting, tests, clippy, and external parity checks not yet demonstrated  
**Module:** \`crates/domains/symthaea-architecture-engineering/src/pareto_anytime.rs\`  
**Export:** \`pub mod pareto_anytime;\` in the architecture-engineering crate  
**Protocol:** \`AR5B5_ANYTIME_EVIDENCE_COST_ATTAINMENT_V0_1.md\`

## 1. Implemented surface

The first code tranche is optimizer-agnostic. It provides a shared event-history and target-attainment interface that can be emitted by NSGA-II, MOEA/D-DE, SMS-EMOA, or a future optimizer without making their internal algorithms depend on one another.

### Frozen header and provenance

\`AnytimeHistoryHeader\` binds:

- run, algorithm, evaluator, and verifier identities;
- algorithm, evaluator, and verifier build SHA-256 digests;
- execution-environment digest;
- problem ID and seed;
- ordered objective specifications and scale anchors;
- hypervolume reference point;
- canonicalized target plan and reference Pareto front;
- cost-model ID/digest;
- objective-evaluation attempt budget;
- required resource meters.

The header itself is SHA-256 bound. The target plan is independently digestible.

### Hash-chained event history

\`AnytimeHistory::append\` emits contiguous one-based event ordinals and a SHA-256 chain rooted in the header digest. Each event binds its time interval, resource vector, action payload, and previous-event digest.

Supported action types:

- successful, failed, invalid, and timed-out candidate evaluations;
- candidate verification promotions;
- explicit candidate verification rejections;
- proposal/measurement/tool/other auxiliary actions.

A failed attempt is retained and charged. Candidate retries must have contiguous attempt numbers and retain the same decision digest. A decision digest cannot be admitted under two different candidate IDs, preventing duplicate bytes from inflating archive size. A canonical candidate may be successfully admitted only once in v0.1; repeated measurements must use a separate run/replication identity until an explicit replicate-aggregation contract is implemented.

### Evidence levels

The recorded levels are:

- Recorded
- ReplayVerified
- ArtifactVerified
- HeldOutVerified
- Confirmatory

A target declares the minimum level required for inclusion in its target-specific frontier. This allows ordinary optimizer-quality targets and held-out evidence targets to be reported separately.

Important boundary: the history module validates the structure and digest references of verification events. It does **not** retrieve candidate artifact bytes, validate cryptographic verifier signatures, or independently prove that a named verifier performed the declared check. Evidence levels are assertions recorded in the chain until an artifact/receipt resolver and verifier-authenticity layer independently validate the underlying evidence. Do not use a HeldOutVerified label alone as a scientific superiority claim.

### Resource accounting

\`ResourceCost\` stores twelve integer meter fields. Additive values (CPU, worker-time, accelerator-time, energy, tokens, calls, and normalized charge) sum with checked arithmetic. Peak-memory fields aggregate by maximum. Unknown values propagate as unknown and disqualify a target's strict resource-cost qualification whenever the meter is declared required.

Elapsed wall time is the monotonic completion timestamp of the run/event prefix, not the sum of action durations. Worker-time remains separate from elapsed time.

### Archive-prefix metrics and target receipts

For each event prefix and each declared evidence level, the module reconstructs the candidate set and calls the shared Pareto analysis implementation. It calculates:

- hypervolume against the frozen reference point;
- IGD+ against the frozen reference front;
- additive epsilon against the frozen reference front.

An indicator is unavailable when the eligible feasible archive is empty; unavailable IGD+/epsilon does not satisfy a target. Hypervolume is zero for an empty feasible archive.

Each target has one final outcome:

- Reached, with the first crossing event ordinal, objective-evaluation-attempt count, elapsed time, cumulative resource vector, archive digest, and resource-qualification flag;
- NotReachedByBudget, with final metric, final archive digest, final resource vector, elapsed time, and resource-qualification flag.

The receipt binds the header digest, target-plan digest, final history-chain tip, all target results, and total resources. \`verify_anytime_run_receipt\` reconstructs the receipt from the exact event history and rejects substituted fields.

## 2. Initial authored tests

The module contains tests for:

1. exact target-prefix crossing and HeldOutVerified gating;
2. failed retry charging and unknown required meter disqualification;
3. a dominated candidate not increasing hypervolume;
4. verifier rejection being retained and charged without promoting evidence level;
5. duplicate decision bytes being rejected under a second candidate ID;
6. overlapping event intervals and mismatched verifier evidence being rejected;
7. target reference-front ordering being canonicalized to one digest;
8. receipt tampering being rejected by deterministic reconstruction.

These tests are authored only. They have not been compiled or executed in this environment.

## 3. Explicit v0.1 limitations

- Sequential accounting only. Any overlapping event intervals are rejected; parallel, batched, and asynchronous work is not qualified.
- The three optimizer runners do not yet emit this event history. The module API exists, but evaluation-history instrumentation is a separate integration task.
- No candidate artifact-byte resolver, immutable object-store retrieval, verifier signature check, or independent evidence-bundle validator is present yet. The event chain detects post-capture alteration but cannot prove that submitted resource counters are truthful; resource receipt attestation remains a separate integration.
- One successful measurement per candidate ID per run. Replicate aggregation is not yet defined.
- No multi-run empirical attainment surface or cluster-level target-cost inference yet.
- Exact hypervolume target analysis inherits the shared 2–4 objective limit.
- No measured optimization result, benchmark win, PlatEMO parity, or SOTA claim is established.

## 4. Next integration issue

The next engineering task is to add a deterministic evaluation-history hook to NSGA-II, MOEA/D-DE, and SMS-EMOA without changing their proposal/selection behavior. Each optimizer must retain every evaluated decision/objective vector and emit the same candidate IDs, decision digests, objective-evaluation ordinals, and resource/action events.

The resulting histories should then be passed through \`analyze_anytime_history\` and \`verify_anytime_run_receipt\` under the same frozen target plan. Do not retrofit a trace from final_population; it cannot reconstruct candidates discarded before the end of the run.

## 5. Exact-head acceptance commands

\`\`\`bash
cargo fmt --all -- --check
cargo test -p symthaea-architecture-engineering pareto_anytime -- --nocapture
cargo test -p symthaea-architecture-engineering pareto_objectives -- --nocapture
cargo test -p symthaea-architecture-engineering
cargo clippy -p symthaea-architecture-engineering --all-targets -- -D warnings
\`\`\`

Then perform deterministic two-run replay from clean state and cross-check target metrics/reference-front behavior against the pinned benchmark implementation. GitHub workflow conclusions marked skipped are not passes.


## 6. Separate artifact-byte verification primitive

The content-addressed byte verification foundation is now in \`crates/domains/symthaea-architecture-engineering/src/evidence_artifact.rs\`, exported from the crate.

It provides canonical run/candidate/source-commit-bound artifact manifests; exact byte-length and SHA-256 verification for a complete resolved byte set; a verifier-build-bound byte receipt; and deterministic receipt reconstruction. Its authored tests cover byte mutation, missing/extra/duplicate artifacts, canonical ordering and tampered receipts.

This primitive receives bytes from its caller. It does not retrieve from CAS/Holochain/object storage, authenticate a remote store, authenticate digital signatures, or establish semantic correctness. Integration with the anytime verifier promotion and bounded CAS retrieval is tracked separately in [Issue #9608](https://github.com/Luminous-Dynamics/symthaea/issues/9608). The implementation and limits are documented in \`AR5B5_EVIDENCE_ARTIFACT_BYTES_V0_1.md\`.

All tests remain authored and unexecuted; no byte-verification or science-validity pass is claimed.


### Terminal verifier rejection

A candidate-verification rejection is now terminal for that candidate identity within the run. A later attempt to promote the same candidate is rejected by history validation. A contradictory verifier outcome after a prior promotion invalidates the history rather than allowing an already-qualified candidate to be silently reclassified. Use a new candidate/run/replication identity for changed bytes or an independent re-evaluation. A regression test covers rejection followed by attempted held-out promotion.


### Incremental append and restart behavior

The history writer now maintains a private in-memory validation state. A newly constructed history validates its frozen header once; each subsequent append verifies only the new event transition, updates bounded hash/index state, and appends the event. It no longer replays the entire prefix twice per action.

The validation cache is excluded from serialization. When a retained history is deserialized, the next append first reconstructs the cache by replaying the complete chain, then resumes incrementally. The public history/header/event fields are exposed through immutable accessors so callers cannot mutate the prefix behind the cache. Full \`validate()\` and analysis still replay every event independently.

Expected complexity: incremental appends are O(log n) for indexed identity checks plus event hashing; a full audit is O(n log n). An authored regression test serializes and resumes a history, then requires the resumed and uninterrupted streams and receipts to match exactly. No runtime benchmark or measured throughput claim is yet available.


### Prefix-analysis complexity boundary

The writer now appends incrementally, and the analyzer avoids recomputing target metrics for failed/invalid/timed-out attempts, verifier rejections, and auxiliary events that do not change evidence-level eligibility. It still performs batch Pareto analysis when an eligible candidate/promotion changes a target archive. Repeated full batch extraction at every successful candidate can become expensive for large runs.

Issue #9610 tracks incremental evidence-level nondominated archives and prefix metrics. The existing batch Pareto evaluator must remain the independent oracle; randomized prefix-equivalence tests are required before using the incremental path for benchmark claims. No measured analysis speedup is claimed.


### Empty-run resource semantics

An empty history now reports unknown resource totals and fails strict resource qualification rather than presenting zero cost. The first retained event establishes the measured accumulation baseline. A not-started/unexecuted run is not evidence of free computation. An authored test covers the empty-history receipt.


## 7. Optimizer trace integration checkpoint

The in-repository reference runners now expose additive history-enabled entry points:

- \`run_nsga2_zdt1_with_history\` / \`run_nsga2_dtlz2_with_history\`;
- \`run_moead_zdt1_with_history\` / \`run_moead_dtlz2_with_history\`;
- \`run_smsemoa_zdt1_with_history\` / \`run_smsemoa_dtlz2_with_history\`.

Each runner records every initial and offspring objective evaluation at the point of evaluation, before environmental selection can discard it. Existing runner signatures remain unchanged and call the common private implementation with trace capture disabled. Authored equality tests compare complete final run reports with capture on/off and require history event count and evaluator-call count to equal the optimizer's declared objective-evaluation count.

The first adapter deliberately uses a **synthetic objective-evaluation-unit cost model**, not measured hardware:
- exactly one evaluator call and one normalized synthetic charge unit per objective-evaluation attempt;
- CPU time, GPU time, energy, memory and other physical meters remain unknown;
- history header must bind the exact canonical synthetic cost-model ID and digest;
- fresh optimizer history must be empty and run-specific;
- timed/untimed modes cannot be mixed.

This is a mechanism-matched trace test, not a resource-efficiency result. A real execution adapter must provide actual monotonic timing and EventTape/OS/hardware measurement receipts.

## 8. Exact timing semantics

Event timestamps and elapsed receipt fields are optional. A run must use one clock mode throughout:
- **Timed:** each event supplies both start and completion monotonic timestamps; the first event starts at zero; subsequent events are non-overlapping and ordered.
- **Untimed:** every event has both timestamps absent; the resulting run and target receipts carry \`elapsed_wall_millis = None\`.

Partial timestamps and transitions between timed and untimed event modes are rejected. This prevents synthetic deterministic runners from emitting zero milliseconds and being misread as zero latency. Zero-length intervals are permitted only when supplied as actual monotonic measurements by the caller; they are not generated as a synthetic placeholder.

## 9. Artifact promotion integration checkpoint

\`append_artifact_verified_promotion\` reconstructs the byte receipt from the frozen manifest and exact supplied artifact bytes, then checks:
- manifest run ID and source commit equal the anytime history header;
- manifest candidate ID refers to a successful candidate in that history;
- manifest candidate-evidence digest equals the successful evaluation evidence digest;
- byte-receipt verifier ID/build equal the frozen history verifier identity/build;
- all artifact bytes, lengths, manifest digest, and receipt fields reconstruct.

Only then does it append an \`ArtifactVerified\` event with the verifier receipt digest and charged resource vector. \`resolve_and_append_artifact_verified_promotion\` composes this with the explicit bounded resolver interface in \`evidence_artifact.rs\`.

These APIs establish artifact byte integrity only. They do not establish semantic correctness, held-out scientific validity, authenticated verifier signatures, or resource-meter truth. EventTape/OS resource attestation and domain-specific verification remain separate integration steps.
