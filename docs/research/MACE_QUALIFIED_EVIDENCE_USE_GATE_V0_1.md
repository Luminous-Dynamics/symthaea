# MACE: Decision-Scoped Evidence Qualification Gate
## Rust boundary contract v0.1 — 2026-11-23

**Status:** Rust interface prototype and authored unit tests. Tests, compilation, formatting and integration are pending because the current authoring environment does not provide Rust tooling. This is not a production source-authentication or authorization service.

## 1. Purpose

MACE now has a durable, label-blind retrieval trace receipt path. A retrieval trace can prove, subject to the local persistence boundary, that a query invocation ran a particular tool/action sequence and produced a particular candidate-set commitment.

That does not prove that the candidates are admissible for a requested decision.

The missing boundary is a **decision-scoped qualification receipt** issued by an independently administered verifier. This PR adds a Rust API contract for that boundary in:

`research/broca-mace-transfer-bench-v0-1/aml-adapter/src/evidence_qualification_gate.rs`

It is deliberately separate from retrieval-policy memory. A search strategy can propose actions; the strategy itself cannot issue qualification receipts, authorize evidence or convert a non-empty candidate result into admissible support.

## 2. Scope of the new Rust module

The module defines:

- `EvidenceQualificationReceiptV1`: verifier result bound to one query invocation and decision need.
- `EvidenceQualificationEntryV1`: explicit per-candidate disposition and basis receipt digest.
- `EvidenceQualificationBindingV1`: consumer-supplied expected context, not copied from the receipt.
- `EvidenceQualificationSignatureVerifierV1`: external cryptographic verifier interface.
- `VerifiedEvidenceQualificationV1`: a capability-style token whose fields are private and which is created only after signature verification, freshness checking and exact binding.

The consumer binding includes:

- an effective subject/tenant authorization-scope commitment (a digest of a canonical scope manifest, not a raw user or tenant identifier);

The consuming runtime must supply a trusted current time; callers cannot treat the token as perpetual after verification.

- run ID, opaque query invocation ID and domain;
- environment revision and tool-profile digests;
- exact query digest;
- candidate-set digest and independently recomputed sorted candidate-ID-set digest;
- DecisionNeed digest and purpose digest;
- qualification-policy digest and lifecycle-snapshot digest;
- verifier-profile digest and current consumption time.

The receipt includes those bindings, signer key ID, issuance/expiry times, an explicit disposition for every candidate and opaque signature bytes. The signing digest uses a versioned, domain-separated encoding. Candidate IDs and entries must be unique, complete and canonically sorted.

## 3. Three-valued evidence disposition

Each candidate receives exactly one disposition:

| Disposition | Meaning | Permitted use |
|---|---|---|
| `Admissible` | Passed the independent qualification profile for this exact DecisionNeed, purpose and lifecycle snapshot | May be considered by downstream decision logic, subject to live re-authorization |
| `Rejected` | Failed one or more applicable qualification rules | Must not enter the admissible evidence set |
| `Indeterminate` | Evidence or verification is incomplete, conflicting, unavailable or unresolved | Must not enter the admissible evidence set |

An empty admissible set is a valid verified outcome but is **not permission to proceed**. The caller should abstain, seek another authorized observation or ask for clarification.

Admissibility is not semantic truth. A receipt may establish that a source was authentic, authorized, rights-compliant and current for a purpose without proving that the underlying proposition is true. Semantic support, contradiction handling and decision sufficiency remain separate checks.

## 4. Fail-closed binding requirements

The Rust gate rejects:

- missing, duplicate, noncanonical or extra candidate IDs;
- incomplete per-candidate qualification dispositions;
- candidate-set digest or exact candidate-ID-set mismatch;
- query, invocation, run, domain, environment or tool-profile mismatch;
- DecisionNeed, purpose, policy, lifecycle-snapshot or verifier-profile mismatch;
- malformed SHA-256 identifiers and invalid receipt shape;
- future-issued, expired, over-age or overlong-lived receipts;
- empty or oversized signatures;
- signatures rejected by the external verifier.

The exact candidate-ID-set digest is recomputed by the module over the sorted evidence IDs and compared with the signed receipt. The broader candidate-set digest still needs to be independently recomputed from the canonical candidate-evidence envelope by the production qualification service. The module does not pretend that a caller-provided digest authenticates source bytes.

## 5. External verifier boundary

The trait is an integration seam, not a cryptographic implementation. A production implementation must:

1. Resolve the verifier profile to separately governed trust roots.
2. Authenticate the receipt signer and reject unknown, expired or revoked keys.
3. Verify the signature over the exact domain-separated payload digest.
4. Independently recompute the candidate-set digest from the canonical evidence envelope.
5. Inspect source origin, issuer authority, purpose, rights, scope, lineage, freshness and lifecycle state.
6. Emit a receipt bound to the exact DecisionNeed, purpose, environment, tool profile and lifecycle snapshot.
7. Preserve an independently verifiable receipt record outside the candidate backend's authority.

The downstream executor must perform live authorization and lifecycle checks again at consequential use. A qualification token is purpose- and time-scoped; it is not a standing access credential.

No signature algorithm, key custody, issuer registry, remote source-observation API, rights engine, tenant-isolation service or external rollback witness is implemented by this module. The subject-scope digest must commit to the effective authorization-scope manifest; it must not be treated as a raw user/tenant identifier or as a substitute for live access control.

## 6. Test inventory

Thirteen Rust unit tests are authored in the module:

1. Only independently admissible evidence IDs cross the gate.
2. DecisionNeed binding cannot be changed.
3. Purpose binding cannot be changed.
4. Expired receipts are rejected at consumption time.
5. Missing candidate dispositions fail closed.
6. Duplicate candidate IDs are rejected.
7. Altered signatures fail closed.
8. Environment revision cannot be rebound.
9. A zero-admissible outcome remains an abstention signal.
10. Candidate-ID-set commitment must match the exact sorted IDs.
11. A verified qualification token cannot be used after expiry.
12. Consumer/tenant scope cannot be rebound.
13. Three candidate-producing traces may promote an advisory search strategy, while the independent evidence gate still rejects the poisoned candidate and yields no admissible evidence.

The module's test verifier is intentionally a deterministic test double; it is not a production signature scheme.

**Execution status:** unexecuted. The environment currently has no `rustc`, `cargo` or `rustfmt`. No compile, test, clippy, schema, integration or CI pass is claimed.

## 7. Relationship to the current MACE architecture

The intended flow is:

`BM25 / dense / graph / timeline / HDC / other backend`
`  -> candidate evidence envelope`
`  -> exact candidate-set verification`
`  -> independent decision-scoped qualification receipt`
`  -> verified admissible evidence IDs`
`  -> decision-sufficiency / uncertainty / abstention policy`
`  -> consequential executor with live authorization`

Retrieval-policy memory remains parallel and advisory:

`label-blind trace -> possible strategy note -> suggested search action`

It does not feed the qualification service an authority decision. It merely influences which sources to inspect. The service must assess the resulting source evidence independently.

The current LongMemEval-V2 sidecar still performs one BM25 search operation per query. This module is not wired into that sidecar and does not yet implement the multi-step retrieval controller.

## 8. Competitive and benchmark implications

The gate addresses a gap that ordinary retrieval scores do not measure: whether evidence returned by a memory backend is independently admissible for a specific use.

It complements, rather than replaces, the current external protocols:

- **Agent Memory Leaderboard Cycle 2** measures separate textual, multimodal and coding memory tracks. Its published open-method leaders are Resurface 61.22 textual, Khipu 43.26 multimodal and InsightMem 63.33% coding-resolved; commercial leaders are 69.26, 53.68 and 68.00%, respectively. These are different track metrics and divisions, not one pooled score.
- **LongMemEval-V2** measures answer accuracy versus memory-query latency using its LAFS frontier. The current published Small table includes AgentRunbook-C V2 Online at +2.8231 LAFS / 75.6% accuracy / 130.5s; OPS-Agentic-Memory-v1 at 81.4% / 153.6s but +0.3223 LAFS; and Perpetual Recall V3 at 53.2% / 1.96s and +1.0955 LAFS. This illustrates why MACE must report the full frontier and not optimize only peak accuracy.
- **MemSecBench** measures lifecycle poisoning through Write–Execute–Forget and selective repair. Its paper reports 84.2% poison persistence, 50.3% full Write–Execute chain success and 56.1% selective repair across its evaluated configurations.
- **MemoryDojo** studies poisoning propagation across 155 multi-session trajectories in four domains and five memory systems.
- **MemPoison** contributes 1,227 cases and 3,681 trigger tasks, covering L1 direct, L2 compositional and L3 context-triggered poisoning.
- **MPBench** studies persistent memory poisoning through multiple write channels and structural vulnerabilities.
- The **Agent Memory Integrity Internet-Draft** is an individual Internet-Draft, not an IETF-endorsed standard. Its T9 whole-store rollback and T10 erasure-resurrection tests remain important but are specifically about at-rest integrity, not semantic correctness or evidence admissibility.

MACE should keep the original external benchmark protocols separate and add its own assurance dimensions: unauthorized exposure, inadmissible evidence influence, decision regret, harmful influence removal, benign utility retention and repair closure.

Other current benchmarks expose additional gaps that the MACE comparison matrix should cover:

| Benchmark | Scope and distinctive measurement | MACE implication |
|---|---|---|
| MemoryArena (ICML 2026) | Interdependent multi-session Memory-Agent-Environment tasks across web navigation, preference-constrained planning, progressive information search and sequential formal reasoning | Evaluate memory acquisition and later action together; a dialogue-recall score is not enough |
| AMA-Bench (ICML 2026) | 207 samples, 2,482 QA pairs, six domains and four capabilities: recall, causal inference, state updating and state abstraction | Add causal and state-update baselines, not just recall/retrieval |
| WorldMemArena (2026) | 400 multi-session multimodal tasks and an observable write/maintain/retrieve/use lifecycle | Preserve original images, screenshots, states and action feedback; do not reduce all visual evidence to captions |
| StateMemBench (2026) | 234 multi-session state-tracking scenarios; separates current-state, superseded-state and other failures | Make supersession and bitemporal current-at-use semantics first-class baselines |
| Memora / FAMA (Findings ACL 2026) | Weeks-to-months personalized histories; remembering, reasoning and recommending; Forgetting-Aware Memory Accuracy penalizes reliance on obsolete memory | Score harmful retention and stale-memory use, not just successful recall |
| TimlMemBench (Findings EMNLP 2026) | 208 reviewed bilingual samples, 1,566 sessions, 1,715 questions and 1,821 governed-use/activation cases | Test revocation, scope, provenance recovery, activation and restraint under raw-native memory ingestion |

These are distinct tasks and scoring protocols. Do not pool their scores with AML, LongMemEval-V2 or each other. Use them as separate benchmark adapters and as guides to MACE's own cross-protocol assurance axes. In particular, StateMem's reported gains from explicit supersession/state structure and TimlMemBench's raw-native governance failures reinforce that MACE needs current-at-use qualification, not only high-quality storage and recall.

## 9. Next integration milestones

1. Add a versioned JSON wire schema and canonical serializer/deserializer for this receipt.
2. Implement a signature verifier adapter with separate trust-profile and revocation handling.
3. Connect the canonical candidate-evidence envelope to a qualification service that independently verifies parent closure and source-observation receipts.
4. Add a V2 sidecar protocol operation that obtains the qualification receipt before exposing candidates to a consequential reader.
5. Make missing, expired, mismatched or indeterminate qualifications result in no admissible evidence and a visible abstention/repair path.
6. Add current-at-use revalidation, correction/revocation propagation and a bitemporal lifecycle snapshot.
7. Add external witness custody for T9/T10; do not count local filesystem persistence as rollback protection.
8. Run the adversarial decision episodes in #13397 and matched baselines in #13261 under independently isolated evaluator custody in #13287.

## 10. Claim boundary

This PR adds a Rust interface prototype and authored unit tests only. It does not qualify any actual source, authenticate an issuer, authorize a tool call, secure a benchmark sandbox, prevent whole-store rollback, prove erasure, change the LongMemEval-V2 result, or establish any SOTA advantage. The next proof must include a real external verifier, executed tests, an exact-head integration run and independent counterfactual evaluation.
