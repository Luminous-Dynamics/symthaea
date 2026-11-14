# Engineering Decision Correctness — Owner Compatibility Audit v0.1

**Date:** 2026-11-14  
**Status:** source-level inspection; integration not qualified  
**Issue:** [#12039](https://github.com/Luminous-Dynamics/symthaea/issues/12039)  
**Protocol:** [ENGINEERING_DECISION_CORRECTNESS_BENCHMARK_V0_1.md](ENGINEERING_DECISION_CORRECTNESS_BENCHMARK_V0_1.md)  
**Audit rule:** an open PR branch is a source reference, not a current-main dependency or a passing implementation.

## 1. Executive result

The capability portfolio is broader than the engineering benchmark integration. The owner audit found usable semantic contracts for bitemporal evidence, evidence-use validity, lifecycle actions, risk-sensitive utility, retrieval planning, decision receipts and trust verification.

However, no inspected artifact currently supplies one qualified **engineering decision assessment** joining all of those axes. Several reusable implementations are on long-lived draft branches substantially diverged from the current main branch. They may guide interface design, but must not be imported or described as current-main qualified without reconciliation and exact-head validation.

The safest immediate implementation is a pure consumer of versioned owner outputs. The benchmark should not create a second time kernel, source ledger, use-validity engine, action-loss oracle, calibration engine, retrieval runner, authorization service or trust root.

## 2. Exact source and branch inventory

The branch comparisons below were read against the repository `main` ref during this audit. Counts are commit counts, not code-size estimates. Branches can move; re-run the comparison before integration.

| Owner / PR | Exact branch and head | Base / divergence from current main | Source-level finding | Qualification state |
|---|---|---|---|---|
| Symthaea #11996 | `research/eng-change-gold-temporal-main-20261114` @ `96c1b1627a19266d67233e884d51338255373bc4` | main; observed +12 / -5 | V1.1 temporal gold contract, T/K, source identity/bytes, publication, receipt, ingestion, materialization, run context, rights and causal-frontier fields | Open draft; no hosted status contexts; schema engine and semantic evaluator outstanding |
| Symthaea #9138 | `feat/broca-use-validity-2026-10-30` @ `693b29b79f2b6ade10e361011d6adab0ebbd1c29` | Stacked on `feat/broca-bitemporal-ledger-2026-10-30`; observed +234 / -253 vs main | Typed decision-specific evidence-use evaluator and deterministic integrity certificate | Open draft; 26 authored tests; not current-main integrated or executed |
| Symthaea #9323 | `feat/broca-lifecycle-action-controller-2026-10-30` @ `f7c84fd1110b89dc669f550e163e68fe2d332ba7` | Stacked on `feat/broca-lexical-proof-candidate-admission-2026-10-30`; observed +351 / -253 vs main | Claim lifecycle ledger, proof-token boundary and deterministic answer/warn/conflict/retrieve/abstain policy | Open draft; not current-main integrated or executed |
| Symthaea #11253 | `research/sym-fin-time-001-rust-kernel` @ `cbd15b7d1155e3cd7411e7600006550206013a03` | main; observed +29 / -51 | Typed bitemporal evaluator with historical-reconstruction vs prospective-streaming modes, explicit availability and rights proofs | Open draft; finance-scoped; not current-main qualified |
| Symthaea #11550 | `research/ig-risk-action-001-v0.1` @ `0cab03f77a37092dc7508df4b01bb982827e9861` | main; observed +19 / -39 | 40 synthetic action/loss cases, exact rational utility oracle and five actions | Open draft; synthetic known-answer only; not a reusable public library API |
| Symthaea #10314 | `feat/benchmark-run-receipt-consolidated-20261103` @ `a671269d2804b20415957453713cc5de2a6d2158` | main; observed +42 / -178 | Generic Python Draft 2020-12 schema/instance CLI, duplicate-key rejection and pinned CI dependencies; separate Rust receipt validator | Open draft; no actual V1.1 invocation recorded |
| Symthaea #10651 | `feat/know-retrieve-anchor-contract-v0-1` @ `6b134a2f0886fac2f1656472cddc5c8c92f71500` | Stacked on `research/esoterica-knowledge-fabric-v0` | Candidate-only retrieval plan and anchor/constraint schema | Plan is not authorized execution or proof of actual model visibility; #10816 owns runtime join |
| Mycelix #5771 | `research/pulse-authority-lifecycle-v1-20261104` @ `94142014cba0bbc3b81fca4ee05fbba0a292568e` | main; observed +453 / -17 | Multi-conductor causal-history audit, DSSE signature checks, root-signed trust-policy code path and explicit incomplete SCITT path | Open draft; large divergent branch; verifier claim ceilings must be preserved |
| Mycelix #6107 | `research/supplychain-interop-2026-11-13` @ `fca96a40ed08c0c09f79b30edfe350ea016ed3e2` | main; observed +27 / -1 | 27 synthetic EPCIS/supply-chain scenarios, including temporal availability and frozen coverage denominators | Open draft; corpus unadjudicated and unscored |

For PRs #9138, #10314, #11550, #11996 and #12041, the combined-status query returned an empty status collection at the inspected heads. **Empty status is not a passing status.** PR metadata can report mergeability even while a branch is substantially diverged; compare against current main and inspect the exact commit ancestry before relying on it.

## 3. Reusable interfaces and claim ceilings

### 3.1 Evidence-use validity — #9138

Source: `crates/domains/symthaea-broca/src/memory_use_validity.rs` on the PR head.

The exact public request has:
- `decision_id`, `requested_action`, `scope`;
- `valid_at: NaiveDate` (day granularity);
- `known_at: DateTime<Utc>` (UTC instant);
- `rule_id`, `rule_version`, `rule_sha256`.

The minimal `EvidenceSourceRef` has `revision_id`, payload SHA-256, scope, `valid_from`, exclusive `valid_to`, `recorded_at` and `retracted_at`. The output certificate binds the request, canonical source revisions, governing evidence, governing-state digest, assessment and certificate digest. Main entry points are `assess_evidence_use` and `verify_evidence_use_certificate`.

Statuses include Admissible, AdmissibleWithLimitations, ExploratoryOnly, NotComparable, MissingOrFailed, Contested, NotKnownAtQueryTime, Retracted, OutsideValidity and OutOfScope.

**Important limitations:** this reference projection is not the source ledger; day-level `valid_at` cannot express all instant-level validity intervals; `recorded_at` is not proof of actor-specific availability, downstream materialization or run-context inclusion; source authentication, policy provenance and authorization remain external. The certificate digest is deterministic tamper detection, not a signature. It has 26 authored tests, but they are not compiled/executed on the inspected PR head.

### 3.2 Lifecycle action policy — #9323

Source: `crates/domains/symthaea-broca/src/memory_lifecycle_policy.rs` on the PR head.

The claim ledger distinguishes SupportedCurrent, StaleOnly, Conflicted, Missing, Unauthorized and ProvenanceInvalid. The lifecycle policy request binds decision ID, principal, scope, valid/known timestamps, proof-verification state, optional packet digest, retrieval progress and required claims. The result binds the action, rationale, proof status, claim ledger and an action digest.

Actions are Answer, WarnStale, ExplainConflict, RetrieveMore and Abstain. A VerifiedOptimal proof state requires an opaque `VerifiedLifecycleProofToken` from the independent proof verifier; a caller-supplied status alone is insufficient.

**Engineering gap:** this policy is query/claim-lifecycle oriented. It does not define engineering-change dispositions, false-safe/false-block loss matrices, physical verification obligations or post-decision outcome scoring. Consume its statuses/proof result where compatible; do not replace #10543's engineering disposition vocabulary.

### 3.3 Bitemporal evidence frontier — #11253

Source: `benchmarks/financial-time-frontier/src/lib.rs` on the PR head.

The core types distinguish:
- `EvaluationMode::HistoricalReconstruction` from `ProspectiveStreaming`;
- `valid_from` / exclusive `valid_until`;
- public `published_at`, actor-scoped `known_at` and local `retrieved_at`;
- rights disposition plus a principal/profile/time-bound `RightsProof`;
- `AvailabilityProof::Verified`, Missing and Unverifiable, with exact instant, principal and verifier identity/build;
- source revisions, supersession and provenance-family identity.

`EvaluationContext` binds mode, principal, rights profile, decision frontier and separate target time. The evaluator exposes per-evidence and bundle evaluation, returning Pass / Fail / Indeterminate and typed reasons.

This is the closest bitemporal consumer model inspected, but it is a finance-scoped branch, not a generic production kernel. Its distinct-provenance-family count is explicitly **not proof of source independence**. Its branch is 51 commits behind main in the observed comparison; do not import without reconciliation.

### 3.4 Retrieval plan vs actual visibility — #10651 and #10816

PR #10651's `retrieval-plan-v0.1` schema binds the original query digest, anchors, constraints and planned steps. The schema explicitly states that it does not authorize retrieval or network access.

Issue #10816 owns the missing plan-to-authorized-execution-to-capture-to-index-materialization-to-query-visible join. It references existing Mycelix execution receipts, evidence views and frontier/materialization owners. Until that runtime join produces a qualified receipt, a planned query, a repository record time or an index watermark must not be promoted into actual candidate visibility.

### 3.5 Loss-sensitive action oracle — #11550

Source: `research/ig-risk-action-001/qualifier/src/main.rs` on the PR head.

The CLI validates 40 synthetic fixtures across four task families and five actions: Answer, Abstain, RequestEvidence, Escalate and Defer. It uses bounded exact rational arithmetic for expected utility, checks declared utility values, enforces a five-action set, and verifies five paired wrong-answer-penalty changes.

Its probabilities are explicitly from `SyntheticScenarioOracle`. The functions are private to a binary crate; this is not currently a public library interface. The embedded `validate_schema` checks a bounded top-level schema identity/shape; it is not a full Draft 2020-12 engine. The PR has 11 authored tests on its inspected head; no CI status contexts were returned.

**Engineering gap:** consume the arithmetic/action result as a benchmark reference, or later expose a small versioned library/CLI receipt. Do not copy its oracle into #12039 or treat synthetic probabilities as empirical engineering-risk estimates.

### 3.6 Generic Draft 2020-12 validator — #10314

The reusable command is `tools/benchmark-run-receipt-harness/validate_schema.py`, with `--schema` and `--instance`. It uses `jsonschema.Draft202012Validator`, format checking, rejects duplicate JSON object keys and blocks external schema references. The CI lock list includes `jsonschema==4.26.0`.

The Rust `benchmark-run-receipt-validator` crate is a separate semantic validator for benchmark run receipts; it is not itself the generic JSON Schema engine. #12017 should reuse the Python CLI for the V1.1 structural run, recording exact script/dependency revision and input hashes. The #10314 branch is 178 commits behind main in the observed comparison, so reconcile first. No V1.1 schema/instance engine run is claimed.

### 3.7 Temporal gold contract — #11996

The V1.1 schema separates:
- target-world time T and decision frontier K;
- single-source ordering from multi-source causal partial order/hybrid-clock modes;
- source-version identity from byte capture;
- evidence validity interval and its evidence reference;
- source publication, issuer availability, canonical record receipt, acquisition/ingestion, derived materialization and run-context inclusion;
- rights at decision from rights at replay/query;
- typed outcome and assessor adjudication.

Its claim ceiling is `temporal_admissibility_only_no_source_truth_or_authority`. A schema-valid record is not a semantic pass, gold adjudication, source-truth claim or engineering approval. The mutation suite has 13 structural negatives and 28 semantic negatives (41 unique negative IDs), plus one positive control.

### 3.8 Mycelix trust and causal history — #5771

At the inspected PR head, `tools/pulse-signed-evidence-audit/src/main.rs` contains an optional pinned-root policy verification path: it verifies the exact policy bytes against an out-of-band root pin and Ed25519 signature before parsing and checking policy identity/epoch. This is stronger than a digest alone, but the root pin itself remains a relying-party provisioned input.

The verifier checks DSSE signatures over exact DSSE PAE bytes and binds issuer/subject/trace/event identity. It validates causal relations against authenticated endpoints and preserves explicit incomplete findings.

**Hard claim ceiling:** SCITT COSE statement and receipt bytes/metadata are parsed and their digest commitments checked, but COSE signatures, transparency-service receipt authenticity and VDS proofs are not verified; the tool emits `SCITT_COSE_VERIFICATION_UNIMPLEMENTED` / INCOMPLETE. The code also retains incomplete causal-edge semantics where a complete causal basis is unavailable. It does not prove physical event truth, conductor honesty, global participant consensus or effect authorization. The multi-conductor audit README explicitly limits its synthetic reachability/performance work and says topological position is never substituted for a reachability proof.

### 3.9 Supply-chain corpus — Mycelix #6107

The exact JSON contains 27 unique synthetic scenarios. SC-TIME-007 separates event occurrence validity from decision-time admissibility; SC-TIME-008 freezes required temporal bins and source-role denominators. The corpus is marked unadjudicated/unscored, and does not prove physical event truth, GS1 conformance or real-world completeness.

## 4. Integration decision for #12039

The first evaluator adapter should not attempt to translate every owner's internal model into one universal schema. It should build a narrow, versioned **engineering decision assessment** over exact owner outputs:

1. V1.1 temporal/admissibility result and exact source/version/byte references.
2. Actual candidate-visible run-context evidence from #10816, not merely a planned retrieval.
3. Evidence-use result from #9112 only after the source and branch are reconciled; preserve its day-granularity semantics.
4. Engineering claim applicability/disposition and mandatory obligations from #10543.
5. Causal-frontier and trust results from Mycelix, preserving INCOMPLETE for unsupported SCITT/COSE and unproven source truth.
6. Loss/action result from #11453/#11550 only after the action oracle has an explicit versioned machine-readable interface and its synthetic probability source is clearly identified.
7. Later outcome observation from the domain system, with censoring/missingness and observation horizon explicit.

The consumer may emit PASS, FAIL or INDETERMINATE per axis. It must not turn any owner's INCOMPLETE/UNKNOWN into PASS, and it cannot grant approval, release, manufacturing, operation or external-effect authority.

## 5. Two additional paradigm shifts to add before candidate trials

### 5.1 Pre-outcome decision commitment (prequential replay)

Before exposing post-decision outcomes, seal the candidate's exact decision, legal action set, evidence manifest, T/K, governing policy, loss matrix, assumptions and build digest. Only then reveal later corrections and outcomes to the evaluator. A later decision is a new episode with a new K; it does not rewrite the earlier packet.

This is the strongest protection against hindsight rationalization. Score decision-time justification against the sealed information set, and realized outcome separately after the outcome horizon.

### 5.2 Decision-minimal sufficient proof and value of information

Measure not only whether a decision has enough evidence, but whether it requested the smallest discriminating evidence set and imposed only necessary requalification. Reuse the lifecycle proof-token/optimality work (#9323), action-acquisition semantics (#11453/#11550) and engineering obligations (#10543) where compatible.

Proposed metrics:
- sufficient-proof coverage (all mandatory claims supported);
- minimum-cost proof gap, only where exact search finishes and the feasible proof space is complete;
- unnecessary-evidence burden and unnecessary requalification obligations;
- value-of-information prediction error only when acquisition outcomes have an independently qualified distribution;
- decision delay cost versus reduction in risk/uncertainty.

Do not claim proof optimality after a search limit, incomplete action set, unknown cost or incomplete source inventory. Return SearchLimitReached / BudgetInsufficient / INDETERMINATE instead.

## 6. Runner-independent next steps

1. Reconcile the long-lived draft branches with current main; pin compatibility commits and inspect tests at the reconciled heads.
2. Run the existing #10314 Draft 2020-12 CLI against the V1.1 schema and synthetic instance at the exact combined revision.
3. Define a machine-readable adapter contract that references owner receipts without copying owner state or losing temporal precision.
4. Add a small, non-overlapping engineering decision pair corpus for pre-outcome commitment, decision flip/invariance, proof-minimality and requalification.
5. Implement the Rust consumer only after the source and type audit is accepted.
6. Run the offline reference checker and mutations; then seek independent engineering assessors.

## 7. Nonclaims

This audit is source inspection, not compile/test evidence, semantic qualification, benchmark execution, independent adjudication, vendor trial or physical validation. A branch being open, mergeable, having authored tests or containing a verifier does not establish current-main integration or successful qualification. All performance targets remain preregistered hypotheses.

## 8. Additional benchmark and standards research

The protocol now includes AgentAbstain (arXiv:2607.10059 / NeurIPS 2026 Oral) as the closest paired agentic abstention comparator: 263 paired tasks, 42 executable environments, eight abstention scenarios and 59.5% best paired accuracy in its own harness/task protocol. Its post-hoc abstention failure motivates a specific engineering metric: zero critical irreversible actions before the system resolves a runtime trigger. This is not a directly comparable Symthaea score.

The protocol also adopts a defeater-first assurance direction. ISO/IEC/IEEE 15026-2:2022 standardizes assurance-case structure and terminology but does not guarantee content quality. ISO/PAS 8800:2024 is scoped to safety-related AI in series-production road vehicles; it is a reference for assurance reasoning, not a directly applicable engineering-change benchmark standard. The ICO's argument-based assurance guidance explains the claim/argument/evidence structure and the role of context, assumptions and defeaters.

NIST IR 8536 became final on 9 September 2026. It reinforces interoperable cross-ecosystem traceability, cryptographic linkage and selective disclosure, but does not establish engineering decision correctness or physical truth.

Sources:
- AgentAbstain paper: https://arxiv.org/abs/2607.10059
- AgentAbstain project: https://agentabstain.github.io/
- ISO/IEC/IEEE 15026-2:2022: https://www.iso.org/standard/80625.html
- ISO/PAS 8800:2024: https://www.iso.org/standard/83303.html
- NIST IR 8536 final: https://csrc.nist.gov/pubs/ir/8536/final
- Structured assurance cases: https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/artificial-intelligence/explaining-decisions-made-with-artificial-intelligence/annexe-5-argument-based-assurance-cases/

These additions are research design inputs only. No external benchmark was rerun and no standards conformity is claimed.
