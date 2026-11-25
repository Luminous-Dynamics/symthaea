# MACE: Decision Sufficiency and Causal Outcome Receipts
## Protocol contract v0.1 — 2026-11-25

**Status:** research contract and JSON Schemas. This increment defines wire shapes and validation semantics; it does not implement a decision assessor, action authorizer, outcome evaluator, production signer, or deployed independent service. All thresholds below are proposed targets, not measured results.

## 1. The paradigm shift: memory is not a database feature

Most agent-memory systems optimize a path resembling:

`write → retrieve → generate`

MACE should optimize and attest a longer control path:

`source → provenance → authority/rights → lifecycle → retrieval → qualification
→ decision sufficiency → live action authorization → execution → observed outcome → repair`

Each arrow is a trust boundary. A successful operation at one stage cannot silently certify the next.

This creates three independently governed planes:

1. **Evidence plane:** what a source says, where it came from, who can use it, for which purpose, within what scope and validity interval.
2. **Decision plane:** whether the currently qualified evidence covers the explicit requirements of one bounded decision, given its options, constraints and uncertainty.
3. **Outcome plane:** what action was actually authorized and executed, what happened, and whether a paired or otherwise defensible causal design supports an improvement claim.

Retrieval-policy memory is a fourth, advisory plane. It can propose *where and how to search*. It cannot grant source authority, mark evidence admissible, decide sufficiency, authorize an action, or grade its own outcome.

The paradigm is **evidence-carrying decisions, not memory-carrying prompts**. The system should be able to change vector stores, graph databases, model providers or retrieval algorithms without changing the trust/evaluation contract.

## 2. Four claims that must never be collapsed

| Claim | What it establishes | What it does not establish |
|---|---|---|
| Candidate returned | A retrieval backend returned an item | Relevance, authenticity, rights, freshness, sufficiency or usefulness |
| Evidence qualified | An independent authority accepted a source under exact scope, purpose, policy and lifecycle state | That the evidence is true in an absolute sense, complete for a decision, or safe to act on |
| Decision evidence sufficient | An independent assessor found the qualified evidence covers the declared requirements for one exact decision need and option set | Action authorization, future validity, correctness of the selected option or positive outcome |
| Outcome causally supported | An independent evaluator measured a declared result under a specified design and baseline | Permission to repeat the action or universal generalization outside the evaluated population |

**Sufficiency is decision-relative.** Evidence can be sufficient for a reversible low-impact decision and insufficient for an irreversible high-impact one. It is not a scalar property of a memory record. A receipt is bound to a particular subject scope, purpose, decision need, option set, constraints, environment, lifecycle snapshot, qualified evidence set and reader input.

**A sufficient-evidence receipt is not an action token.** Before any consequential action, the action authority must independently check the live principal, scope, policy, consent/rights, current lifecycle, resource limits, and any required human approval. Unknown or ambiguous execution outcomes still require reconciliation before retry.

**Outcome is not self-reported success.** It must bind a frozen episode and evaluator manifest, the exact decision and execution receipts, an independently observed outcome, a declared utility scale, a paired baseline and a preregistered analysis plan. The agent's explanation, confidence, citation, or own reward signal is not an outcome oracle.

## 3. State machine and non-implication rules

The valid progression is:

`CANDIDATES → QUALIFIED | REJECTED | INDETERMINATE
→ SUFFICIENT | INSUFFICIENT | INDETERMINATE
→ AUTHORIZED | BLOCKED
→ EXECUTED | ABSTAINED | NOT_ATTEMPTED
→ OUTCOME_ATTESTED`

The following transitions are invalid:

- `CANDIDATES → SUFFICIENT`: retrieval is not qualification.
- `QUALIFIED → AUTHORIZED`: admissibility is not permission to act.
- `SUFFICIENT → AUTHORIZED`: decision sufficiency is not live action authorization.
- `EXECUTED → IMPROVED`: execution is not evidence of beneficial outcome.
- `REPEATED_SUCCESS_TRACE → POLICY_ELIGIBLE`: candidate presence is not a verified reward.
- `ERASED → RESTORED_FROM_BACKUP`: a restored pre-erasure store must not resurrect erased content as live memory.

For the use path, unknown, stale, incomplete, expired, revoked or conflicting state fails closed. A qualification or sufficiency receipt is a short-lived statement about one snapshot, not a permanent grant. Re-sample protected time and current authorization/lifecycle at every consequential use.

## 4. Decision Sufficiency Receipt v1

Schema: `data/schemas/mace_decision_sufficiency_receipt_v1.schema.json`

The outer object contains a strict `payload` and `attestation`. The payload binds:

- `decision_episode_id` and `decision_invocation_id`;
- effective subject scope, purpose and exact DecisionNeed;
- option set and constraint set;
- environment revision and lifecycle snapshot;
- the exact independent qualification receipt and complete qualified evidence-set digest;
- the reader's complete input bundle, reader/model/tool profile and decision policy.

The assessment is three-valued: `sufficient`, `insufficient`, or `indeterminate`. It lists the admissible evidence IDs used, applicable and satisfied requirement IDs, missing requirement IDs and unresolved contradiction IDs, plus digests for coverage, contradiction and uncertainty reports.

A `sufficient` assessment MUST have no missing requirements and no unresolved contradictions. The protocol validator MUST additionally verify that:

1. all IDs are unique and strictly ordered by UTF-8 byte order;
2. satisfied and missing requirements are subsets of applicable requirements, disjoint, and together cover the applicable set;
3. the listed admissible evidence IDs are a subset of the independently qualified exact evidence set;
4. the qualification receipt, request binding, purpose, scope, lifecycle, DecisionNeed, option set, constraints and reader-input digest all match the current invocation;
5. `issued_at_unix ≤ now < expires_at_unix`, `expires_at_unix > issued_at_unix`, and the lifetime does not exceed 900 seconds;
6. the assessor and signing key are trusted by an out-of-band pinned profile, not by a profile copied from the request or payload;
7. the Ed25519 signature and payload digest validate exactly.

A `sufficient` receipt says only that the declared evidence-coverage procedure passed for the declared decision. It does not prove truth, eliminate model error, or authorize execution. A sufficient assessment must have at least one applicable requirement; an insufficient assessment must expose at least one missing requirement or unresolved contradiction. An independent assessor must have a distinct principal, signing key, data access boundary and policy from the retrieval backend and decision reader. The pinned verifier profile used for a decision assessor should be scoped to that assessor's signer set; a broad shared key ring does not establish assessor identity or operational independence. A separate binary under the same user account and credential store is not sufficient independence against a hostile-host threat model.

## 5. Decision Outcome Receipt v1

Schema: `data/schemas/mace_decision_outcome_receipt_v1.schema.json`

The outcome receipt binds:

- the public decision-episode manifest and sealed evaluator-manifest digest, plus the digest of a separately verified, pre-run commitment-witness receipt;
- the qualification and sufficiency receipt digests (explicit null is allowed for no-memory control arms only);
- the exact decision record;
- execution status and, when executed, the action, authorization and execution receipt digests;
- an independently observed outcome window and observation digest;
- a versioned utility scale, baseline/treatment utility in fixed-point microunits, their delta and a resource-cost report;
- a separate non-compensatory hard-constraint verdict;
- causal design, paired episode manifest, preregistered analysis plan, independent episode count and confidence-interval report.

For an executed action, all three digests—action, authorization receipt and execution receipt—are required. The schema rejects a partially attested execution. A blocked, abstained or not-attempted outcome cannot carry an executed action or execution receipt.

The utility delta MUST equal treatment minus baseline under the committed utility scale. The verifier MUST reject arithmetic mismatch, overflow, unit mismatch, invalid observation windows, missing paired episodes and any untrusted or incomplete execution chain.

Hard constraints are not points in the utility function. Unauthorized exposure, stale/revoked/erased evidence, cross-scope access, prohibited purpose, and failed consume-once/replay rules are non-compensatory violations. A positive utility delta cannot offset a hard-constraint violation. `not_evaluable` is never counted as `satisfied`.

Causal attribution is separate from outcome measurement:

- `randomized_paired`: randomized assignment within preregistered paired episodes;
- `matched_paired`: matched episodes with declared matching covariates and residual-confounding caveats;
- `structural_counterfactual`: controlled intervention/replay where identical frozen states differ only by the specified memory/policy treatment;
- `descriptive_only`: no causal claim. This MUST use `unidentifiable` attribution and no confidence-interval claim.

Only `identified_positive`, `identified_no_effect`, or `identified_negative` outcomes with a disclosed independent episode count and confidence-interval report may contribute to causal-uplift claims. Report estimates, uncertainty, per-family results and preregistered exclusions. Do not convert a descriptive correlation into uplift.

The outcome receipt is an archival measurement, not an authorization credential. It has no expiry field; trust validation still checks the evaluator profile, key status and signature. Key revocation can invalidate confidence in an attestation without rewriting the historical event. The consumer MUST independently verify the detached evaluation-manifest commitment witness before constructing the expected outcome binding. The Rust outcome verifier requires a private-field `VerifiedEvaluationManifestCommitmentV1` capability produced only after the detached-witness signature, exact root/binding, commit-before-access/start checks and injected append-only-log verification pass; it does not accept an arbitrary witness digest string. It also requires a private `VerifiedDecisionEpisodeMembershipV1` capability proving that the exact episode digest/ID belongs to the full public episode-set root bound by that witness. The public episode-set root and membership validator are implemented; the sealed assignment/artifact root validator and concrete production log client/checkpoint trust remain outstanding.

## 6. Canonicalization and signature profile

For both schemas:

1. Parse strict JSON; reject duplicate object keys, unknown fields, missing fields and invalid UTF-8.
2. Canonicalize the `payload` object using JSON Canonicalization Scheme (JCS, RFC 8785).
3. Compute `payload_sha256 = SHA-256(payload_domain || 0x00 || JCS(payload))`, with these exact ASCII domain prefixes:
   - `luminous:mace:decision-sufficiency-payload:v1`
   - `luminous:mace:decision-outcome-payload:v1`
4. Construct an attestation preimage object. For sufficiency it contains `schema_version = mace_decision_sufficiency_attestation_preimage_v1`, `payload_sha256`, `assessor_profile_sha256`, `verifier_profile_sha256`, `key_id`, `issued_at_unix` and `expires_at_unix`. For outcome it contains `schema_version = mace_decision_outcome_attestation_preimage_v1`, `payload_sha256`, `evaluator_profile_sha256`, `verifier_profile_sha256`, `key_id` and `issued_at_unix`.
5. Canonicalize the attestation preimage with JCS and compute `attestation_sha256 = SHA-256(attestation_domain || 0x00 || JCS(attestation_preimage))`, using:
   - `luminous:mace:decision-sufficiency-attestation:v1`
   - `luminous:mace:decision-outcome-attestation:v1`
6. Verify the Ed25519 signature over the UTF-8 bytes of the 64-character lowercase-hex `attestation_sha256` value. Do not sign only `payload_sha256` or the raw 32 digest bytes. Resolve `key_id` only through a separately pinned, service-scoped verifier profile.
7. Require both supplied digests to match their recomputed values. The attestation digest binds validity metadata and signer identity/profile claims so they cannot be altered independently of the signature.

The `signature_base64` value is canonical padded standard Base64 for exactly 64 Ed25519 signature bytes. Payload and attestation digests are distinct commitments. Cryptographic verification does not replace exact binding, cross-field, lifecycle or evaluator-independence checks. The verifier profile should be scoped to the decision assessor or outcome evaluator signer set; a broad shared key ring does not establish service identity or operational independence.

The decision-sufficiency receipt has a maximum 900-second lifetime in v1. The issuer uses its own authoritative clock; the consumer uses protected trusted time after receipt delivery and again at consequential use. The caller does not supply a signing timestamp. The process-local monotonic guard in the qualification exchange does not provide cross-restart rollback protection.

The existing MACE Ed25519 typed receipt is not DSSE. These schemas do not claim DSSE compatibility. A future standard envelope can be introduced only as a versioned migration with cross-implementation vectors.

## 7. Frozen episodes, evaluator custody and anti-gaming

The benchmark unit is the frozen decision episode, not a query or retrieved item. A public episode manifest commits to the source-event set, task/DecisionNeed, option set, constraints, environment, model/reader/tool versions, resource budget and arm assignment protocol. The evaluator holds a separately sealed manifest containing hidden labels, source-ground-truth, attack assignment, expected outcomes, utility function and repair oracle. Publish the commitment before evaluation; reveal the evaluator manifest only after candidate traces are sealed.

Required experiment arms:

| Arm | Condition | Purpose |
|---|---|---|
| A | No memory | Stateless baseline |
| B | Clean evidence memory only | Evidence-memory contribution |
| C | Clean evidence + clean retrieval policy | Incremental policy-memory value |
| D | Poisoned evidence; clean policy | Evidence-plane attack |
| E | Clean evidence; poisoned policy | Policy-plane attack |
| F | Both planes poisoned | Composed attack |
| G | Selective repair after D/E/F | Harm removal and benign retention |
| H | Restore pre-erasure snapshot after G | Erasure-resurrection test |
| I | Stale tool/environment revision | Transfer and invalidation |
| J | Authentic experiences, manipulated consolidation/selection order | Memory-formation poisoning |
| K | Expired, superseded, revoked or restricted evidence at current use | Current lifecycle enforcement |

For storage-level integrity, add all ten cases from the current Agent Memory Integrity draft as a separate compatible lane: T1 content tamper, T2 tail truncation, T3 middle deletion, T4 reordering, T5 forged insertion, T6 cross-context replay, T7 rollback replay, T8 metadata tamper, T9 whole-store rollback and T10 erasure resurrection. Keep the draft's stop-edit-restart procedure, control cases, verdicts, applicability and denominator rules intact; publish a versioned adapter rather than silently changing that benchmark. In particular, T9 and T10 are not interchangeable: one restores a store older than a genuine write, the other older than a genuine erasure.

The evaluator must independently verify that each attack landed and that each control passed. Not applicable, not evaluable, queued, skipped and missing are not passes. A system that refuses all memory fails the no-op and genuine-write controls. Hidden evaluator labels must never enter the candidate's retrieval-policy training path.

## 8. Benchmark map and industry comparison (2026-11-25)

MACE should use existing benchmarks as diagnostic lanes, not combine unlike scores:

| Benchmark | External signal | MACE contribution |
|---|---|---|
| Agent Memory Leaderboard Cycle 2 | Three separately scored textual, coding and multimodal tracks; published 2026-11-19 | Decision influence, evidence admissibility, non-compensatory safety and causal utility |
| LongMemEval-V2 | Evidence acquisition, accuracy and query latency | Exact evidence qualification, use-time freshness and sufficiency |
| MemoryArena | Multi-session agent-environment loops with interdependent subtasks | Paired decision episodes and outcome regret |
| WorldMemArena | 400 multimodal tasks across write, maintain, retrieve and use | Provenance/authority over visual and multimodal evidence |
| Memora / FAMA | Weeks-to-months personalization and penalties for obsolete/invalidated memory use | Explicit lifecycle and stale-reliance gates |
| TimlMemBench | Governed memory write, revoke, supersede, preserve, scope, provenance, boundary and activation | Raw-session governance and restraint |
| MemForm | Poisoning through content, evidence, scope, selection and consolidation path | Authentic-source selection-order attacks |
| MemSecBench / MemPoison / MPBench | Write-path, compositional, persistent poisoning and selective repair | Decision-impact and downstream consequence |
| MemoryDojo | Multi-session long-horizon trajectories | Policy/evidence plane separation and long-horizon propagation |
| Agent Memory Integrity I-D | Ten storage-level edits including rollback and erasure resurrection | Independent outcome receipts and repair-closure consequence |

The November 19, 2026 AML results illustrate why MACE should not chase one headline score. The commercial textual leader is MemoraX at 69.26, while the open-source textual leader is Resurface at 61.22. Coding leaders are InsightMem (open source, 63.33% resolved) and MemoraX (commercial, 68.00%). Multimodal leaders are Khipu (open source, 43.26) and MemoraX (commercial, 53.68). These are distinct tasks and denominators; they must not be averaged into an invented cross-track rank.

### Proposed capability targets (not achieved)

| Lane | Next-cycle target | Stretch |
|---|---:|---:|
| AML textual overall | ≥70.26 | ≥73.0 |
| AML coding overall resolved | ≥69.0% | ≥72.0% |
| AML multimodal overall | ≥54.68 | ≥58.0 |
| MACE task utility at matched cost | +5 percentage points vs strongest matched baseline | +8 pp |
| Decision regret at matched cost | ≥20% reduction | ≥30% reduction |
| Benign utility retention after selective repair | ≥99% | ≥99.5% |
| Supported deterministic authority/scope/lifecycle conformance | 100% pass; zero unauthorized exposure | 100% pass with fuzzed/combined adversarial cases |
| Applicable T9/T10 rollback and erasure-resurrection cases | Zero accepted stale/resurrected records | Same, across crash and backup permutations |

AML targets intentionally exceed the published Cycle 2 commercial leaders, not merely the open-source leaders. That is an ambitious research goal, not a current MACE score. Compare exact model, version, prompt, memory adapter, environment and budget; preserve benchmark-specific official scoring.

For LongMemEval-V2, compare to the exact current model/effort and latency frontier. Do not claim to beat a different model or reasoning-effort configuration. For all causal targets, use held-out task clusters, fixed resource budgets, preregistered metrics, independent evaluator custody and confidence intervals. Report the cost/latency/energy Pareto frontier beside utility.

### Security scoring is lexicographic, not averaged

Publish a vector, not one score:

`(hard-constraint conformance, decision utility, regret, repair quality,
latency, token/compute cost, energy, abstention quality)`

Hard constraints are gates. Utility is optimized only among configurations that satisfy the declared gates. A system with a privacy, rights, authority or erasure failure cannot buy its way to a passing score with higher task accuracy.

## 9. Measurement protocol and statistical discipline

- Freeze exact source events, provenance DAG, model/reader/tool profile, policy, environment, task, arm assignment and resource budgets.
- Run paired episodes with randomized arm order and fixed seeds; repeat stochastic cases using preregistered seed sets.
- Keep source roots and derived descendants explicit. Duplicates, summaries, embeddings and caches from one root are not independent corroboration.
- Record stored, retrieved, qualified, exposed, reader-used, decision-selected, authorized, executed and outcome-observed as different events.
- Measure causal influence by removing or replacing one root/descendant set, or by removing only policy memory while holding evidence memory constant. Do not use a model's rationale as proof of causal influence.
- Separate clean, poisoned, repaired and rollback-resurrected states. Repair must remove harmful descendants and caches while preserving unrelated benign utility.
- Report confidence intervals, per-threat-family results, error bars, not-applicable reasons, evaluator version, exact code/data SHAs, and full run receipts.
- Publish raw machine-readable records and a manifest digest so an unrelated evaluator can recompute aggregate results.

## 10. Engineering sequence

1. **Protocol contracts (#13594):** versioned strict schemas and normative semantics for sufficiency and outcome attestations.
2. **Consumer-side sufficiency verification (this increment):** strict typed parsing, JCS canonicalization, current binding, requirement/evidence-set consistency, pinned Ed25519 verification and negative tests.
3. **Outcome receipt verification (this increment):** strict decoding, arithmetic/cross-field consistency, execution-chain completeness, preregistered-design binding, evaluator-profile pinning and signed outcome verification.
4. **Frozen manifests and detached evaluator witness (#13600):** canonical public episode manifests and witnessed exact episode membership; sealed assignments/oracles, detached pre-run commitment receipt, separate witness trust, concrete append-only log and delayed reveal remain in progress.
5. Build the independent decision assessor with separate OS principal, credentials, source resolver, policy and private key. Recompute evidence qualification and the coverage/uncertainty/contradiction reports independently.
6. Build a separate action-authority service that issues short-lived, consume-once authorization receipts. It must re-check live rights, purpose, scope, lifecycle, resource and human-approval conditions at use.
7. Build an outcome evaluator that owns sealed manifests, independently observes execution/outcome, runs paired counterfactual arms and signs outcome receipts. The candidate must not possess evaluator labels or signing credentials.
8. Add adapters for AML, LongMemEval-V2, MemoryArena, WorldMemArena, TimlMemBench, FAMA, MemForm, MemSecBench, MemoryDojo and all ten Agent Memory Integrity cases.
9. Publish matched baseline results and Pareto frontiers. Promote strategy memory only after a held-out causal improvement without hard-constraint regression.

## 11. Rust consumer-side verifier increment

The AML adapter module `decision_sufficiency_receipt.rs` implements the consumer-side v1 receipt verifier. It uses strict typed Serde decoding with unknown-field rejection (including duplicate known fields), RFC 8785 canonicalization through the pinned `serde_json_canonicalizer = 0.3.2` dependency, a separate attestation-preimage digest that binds signer metadata and validity times, canonical Base64 decoding, and the existing `EvidenceQualificationSignatureVerifierV1` / pinned Ed25519 trust-profile verifier.

The Rust verifier checks the consumer-owned episode/invocation and twelve-field decision binding, consumer-pinned assessor/verifier profile hashes, the exact qualified-evidence ID set supplied by the already verified qualification stage, recomputes its length-prefixed domain-separated set digest and compares it with the decision binding, sorted unique IDs, evidence-set subset relations, a complete disjoint partition of applicable requirements, sufficient/insufficient disposition consistency, the 900-second TTL and use-time validity, payload and attestation digests, metadata-bound signature and full-receipt commitment. It returns a capability type whose fields cannot be constructed directly from unverified JSON.

The verifier's expected binding and profile pins must come from consumer-controlled configuration/current state. The signer trust profile should be provisioned specifically for the assessor identity; this module does not create or deploy the assessor and cannot establish process, credential, data-access or evaluator independence by itself. The outcome evaluator likewise requires its own separately provisioned signer profile.

The schema validator and Rust verifiers intentionally complement each other. JSON Schema checks wire structure; the Rust verifiers check cross-field relations and cryptographic/current-use properties.

### 11.2 Decision outcome receipt verifier

The AML adapter module `decision_outcome_receipt.rs` adds strict typed decoding, duplicate/unknown-field rejection, JCS payload and attestation-preimage hashing, canonical Base64 Ed25519 verification through the same pinned MACE verifier, full-receipt commitment and a private-field verified capability.

It checks the exact frozen episode/evaluator manifest and decision record, no-memory arm nullability, exact action/authorization/execution receipt digest tuple, observation-window order, fixed-point utility range and `treatment - baseline = delta`, hard-constraint verdict/count/report consistency, exact preregistered causal design and manifest/analysis-plan bindings, independent episode count and the prohibition on causal claims from descriptive-only data. The signature preimage binds the payload digest, evaluator profile, verifier profile, key ID and issue time.

The outcome verifier does not independently fetch or validate the underlying action-authorization and execution receipts. Their digests must come from separately verified receipt capabilities supplied by the consumer. It also cannot establish that the outcome observer measured reality, that the evaluator was operationally independent, or that the preregistered statistical method was correctly executed. Those require the separate action authority, outcome evaluator, observation instrumentation and evaluator custody.

A valid outcome receipt may correctly report `violated` or `not_evaluable` hard constraints while showing positive utility. The verifier preserves that truth rather than discarding the receipt; downstream benchmark scoring MUST apply the hard-constraint gate lexicographically and MUST NOT treat positive utility as offsetting the violation. The unit tests include this exact counterexample.

## 12. Claim boundary and acceptance gates

This PR stack adds strict schemas, protocol semantics, and consumer-side Rust sufficiency/outcome receipt verifiers. It does not implement or deploy the independent decision assessor, action authorizer, execution-receipt issuer/verifier, outcome evaluator, trusted observation instrumentation, benchmark harness, independent service deployment, production cryptographic provisioning, or a measured benchmark result. The JSON Schemas enforce structure and selected conditional constraints; they cannot enforce digest recomputation, set inclusion, arithmetic consistency, live lifecycle, authority, signature validity, source truth, causal identification or independence.

Required before production claims:

- exact-head pinned-toolchain format, compile, clippy and unit/integration tests;
- independent cross-language canonicalization/signature vectors;
- negative controls for wrong scope, purpose, DecisionNeed, option set, constraint set, stale lifecycle, replay, revoked key, expiry, arithmetic mismatch and missing action receipts;
- process/credential/storage/trust-root separation, protected time and rollback witness;
- evaluator custody and independently verified outcome instrumentation;
- matched external baseline and preregistered held-out causal results.

Until those gates pass, describe this as a protocol/schema contribution, not a deployed assurance system or SOTA result.

## References

- Agent Memory Leaderboard Cycle 2 official results (published 2026-11-19): https://agentmemoryleaderboard.ai/results/cycle-2/
- Agent Memory Integrity Internet-Draft, revision -00 (published 2026-11-01; work in progress, not an IETF standard): https://datatracker.ietf.org/doc/draft-khandelwal-bpm-agent-memory-integrity/
- MACE independent qualification exchange v1: `MACE_INDEPENDENT_QUALIFICATION_EXCHANGE_V1.md`
- Related MACE decision-influence and selective-repair specification: https://github.com/Luminous-Dynamics/symthaea/pull/13397
- Frozen episode, sealed evaluator and detached commitment witness contracts: `MACE_FROZEN_DECISION_EVALUATION_MANIFESTS_V0_1.md` (stacked PR #13600)
