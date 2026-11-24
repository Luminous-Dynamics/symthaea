# MACE evidence receipt chain v2 — protocol contract v0.1

**Status:** research contract and draft JSON Schemas. These schemas are not yet wired into the sidecar, qualification service, sufficiency evaluator or policy-promotion runtime. No benchmark result or security qualification is claimed.

## 1. Why a receipt chain, not a single confidence score

MACE must not collapse four different propositions into one boolean:

1. **Acquisition:** a retrieval adapter returned candidate references.
2. **Qualification:** an independently administered verifier decided which exact records are admissible for a scoped purpose.
3. **Decision sufficiency:** a decision-need-specific evaluator determined whether the qualified evidence meets declared requirements and a calibrated risk bound.
4. **Outcome:** an independently observed execution/world/evaluator event recorded what followed the action.

These are different questions, authorities, times and failure domains. A successful acquisition can yield no admissible evidence. A fully qualified batch can still be insufficient for a particular decision. A sufficient evidence set does not authorize an action. A recorded outcome does not prove that the memory policy caused the outcome.

## 2. Receipt graph and trust boundaries

```text
untrusted retrieval adapter
    |
    | RetrievalAcquisitionReceiptV2
    | candidates_returned / no_candidate_evidence / budget_exhausted /
    | tool_error / abstained
    v
independent qualification authority
    |
    | EvidenceQualificationReceiptV1
    | exact candidate IDs + purpose + scope + lifecycle + verifier profile
    v
decision-sufficiency authority
    |
    | DecisionSufficiencyReceiptV1
    | exact qualified set + DecisionNeed + calibration + risk bound
    v
separate action authorization and consume-once execution
    |
    | ActionExecutionReceipt (existing execution/authorization seam)
    v
independent world observer or sealed benchmark evaluator
    |
    | DecisionOutcomeReceiptV1
    | action + observed world state + evaluator/metric profile
    v
sealed paired evaluator / policy auditor
    |
    | PolicyPromotionCertificateV1
    | paired held-out uplift + cost parity + family/security guardrails
    v
exploratory note -> policy eligible
```

No receipt inherits the authority of its parent. Each consumer verifies exact bindings, signer profile, time/lifecycle state and its own purpose. Live action authorization remains outside memory and must be evaluated at the moment of use.

## 3. Shared invocation binding

The shared binding tuple is:

- `run_id`, opaque `query_invocation_id`, `domain_id`;
- `environment_revision_sha256`, `tool_profile_sha256`;
- `query_sha256`, `candidate_set_sha256`.

Every downstream receipt repeats the applicable tuple and adds its own decision/scope/policy inputs. Every digest is over the exact canonical typed payload, using a domain-separated digest namespace. A receipt that names a different invocation, candidate set, environment, tool profile, purpose, subject scope or DecisionNeed is rejected, even if its own signature is valid.

The acquisition V2 digest domain is `luminous:mace:retrieval-acquisition:v2\0`; it is deliberately distinct from the V1 retrieval-trace digest domain. Rust and Python must have fixed, shared golden vectors for every status, including empty results and mixed-channel partial outcomes.

## 4. Receipt responsibilities and required fields

### 4.1 Retrieval acquisition V2

Schema: `data/schemas/mace_retrieval_acquisition_receipt_v2.schema.json`.

The candidate-side adapter may assert only what it actually executed and observed: ordered operations, tool/action-template identity, candidate evidence IDs, elapsed time and explicitly complete/incomplete resource accounting. The terminal status is an acquisition status, never an admissibility, sufficiency or success label.

- `candidates_returned`: at least one operation observed candidate IDs and the receipt binds a non-empty candidate-set digest.
- `no_candidate_evidence`: no operation observed candidates and candidate-set digest is null.
- `budget_exhausted`: the controller stopped because its declared acquisition budget was consumed.
- `tool_error`: a tool operation failed; partial candidates may still exist and must remain visible as partial acquisition.
- `abstained`: the acquisition controller deliberately stopped without asserting that evidence was sufficient.

The acquisition receipt is an integrity commitment, not an identity signature or proof a tool returned truthful records. The adapter cannot supply gold answers, hidden labels, evaluator keys, decision correctness, qualification outcomes or promotion status.

### 4.2 Independent evidence qualification V1

Reuse the existing `mace_evidence_qualification_receipt_v1` contract introduced with the decision-scoped qualification gate (PR #13410 / #13436). It binds the candidate set and exact evidence-ID set to subject scope, domain, environment, tool profile, query, DecisionNeed, purpose, qualification policy, lifecycle snapshot, verifier profile, key identity and validity interval. Each candidate receives one disposition: `admissible`, `rejected` or `indeterminate`.

Qualification must be independently issued. A candidate backend must not sign its own admissibility. The signature is verified against an out-of-band trust profile with explicit key version, validity and revocation state. A digest match alone does not establish issuer identity, source authority, semantic truth, or rights.

### 4.3 Decision-sufficiency V1

Schema: `data/schemas/mace_decision_sufficiency_receipt_v1.schema.json`.

This receipt binds the exact qualification receipt and qualified-evidence set to one DecisionNeed, purpose, subject scope, policy, calibration profile and verifier identity. It reports requirement-level dispositions (`met`, `unmet`, `unknown`) and an explicit status (`sufficient`, `insufficient`, `indeterminate`, `abstained`).

A `sufficient` receipt requires all declared requirements to be met, a calibration receipt, a confidence level, and a risk upper bound no greater than the declared risk limit. JSON Schema enforces the shape; the runtime verifier must recompute the evidence/receipt digests and enforce numeric comparisons, expiry, lifecycle freshness and signer trust. No generic model-confidence scalar can replace this decision-specific certificate.

### 4.4 Decision outcome V1

Schema: `data/schemas/mace_decision_outcome_receipt_v1.schema.json`.

This receipt binds the decision/action digest, action-execution receipt, observed-world receipt, sufficiency receipt, DecisionNeed, environment and metric/evaluator profile. The issuer is an independent world observer or sealed evaluator, not the retrieval adapter. An outcome may be `observed`, `not_observable`, `invalidated` or `unknown`. A utility value is permitted only for an observed outcome.

Outcome receipts may contain evaluator-sensitive results. They must remain in the evaluator/policy-audit plane; candidate-facing retrieval memory receives neither gold answers nor task labels. A promotion auditor may consume evaluator-owned receipts through a narrow authenticated interface and emit only the aggregate eligibility certificate.

### 4.5 Policy promotion certificate V1

Schema: `data/schemas/mace_policy_promotion_certificate_v1.schema.json`.

A strategy may be policy-eligible only after a pre-registered, sealed, paired held-out comparison produces an independently signed certificate. The certificate binds the strategy, candidate and baseline profiles, task distribution, evaluation manifest and exact outcome-receipt set. An eligible result requires at least 30 paired blocks, 10 independent clusters, a positive lower confidence bound, matched resource budgets, passing security/selection/distribution-shift controls, and no task-family regression above the configured 20,000-micro-unit guardrail.

Those numeric floors are **initial protocol defaults**, not universal statistical guarantees. They must be pre-registered for the benchmark/task distribution and may need to be stricter after power analysis. A cluster count of one is non-estimable; rows or sessions sharing one underlying world/user/source are not independent clusters.

A promotion certificate is not an action permission, source-authority statement, or perpetual validity token. It has an issuer profile, issue/expiry interval and revocation state. Policy use still rechecks applicability and safety at runtime.

## 5. V1 migration rules

Legacy `mace_retrieval_trace_receipt_v1` records remain immutable historical inputs.

- Decode and verify the original V1 bytes and V1 digest exactly; do not rewrite their receipt digest.
- Map legacy `evidence_sufficient` only to the internal compatibility interpretation `legacy_candidates_returned`.
- Never synthesize a V2 sufficiency, qualification, outcome or promotion receipt from V1 fields.
- A migration record, if needed, references the original V1 receipt digest and records the migration implementation/version. It does not upgrade the underlying claim.
- New retrieval queries emit only `mace_retrieval_acquisition_receipt_v2` and its V2 digest domain.
- Restart replay must handle V1 legacy and V2 acquisition receipts without permitting invocation rebinding or cross-version digest confusion.

## 6. Canonicalization, signing and replay

- Apply strict JSON Schema validation with `additionalProperties: false` at every typed object boundary.
- Define one canonical typed encoding per schema version; do not assume arbitrary JSON serializers produce canonical bytes.
- Use domain-separated SHA-256 over the versioned canonical payload.
- The V0.1 schemas carry detached Ed25519 signature bytes in the same explicit byte-array shape as the existing qualification receipt; these arrays are **not** DSSE or COSE envelopes. For cross-system publication, add a separately versioned DSSE/in-toto or COSE envelope and schema. Do not silently mix `COSE_Sign` and `COSE_Sign1` or reinterpret raw signature bytes as a standard envelope.
- Provision trusted verifier/evaluator keys out-of-band. Verify issuer identity, key status, profile version, purpose, scope, validity and revocation at consumption.
- Make create-once publication idempotent. An ambiguous write/acknowledgement outcome is `unknown` and blocks further mutation until independent reconciliation.
- A local hash chain or same-workspace signature file is not rollback resistance. Qualification, outcome custody and promotion authority need a separate administrative and backup/restore failure domain.

## 7. Required negative controls

1. Candidate result exists but every item is rejected or indeterminate.
2. A qualified set exists but at least one required DecisionNeed predicate remains unknown.
3. Sufficient receipt has a mismatched candidate digest, wrong purpose/scope, stale lifecycle snapshot, expired key, revoked key or mismatched verifier profile.
4. Decision action occurs after the sufficiency receipt expires or after evidence correction/revocation.
5. Outcome receipt binds a different action, invocation, environment or observed-world record.
6. Outcome is unknown/not observable but a caller tries to promote it as success.
7. V1 legacy receipt is replayed, relabelled or submitted as V2 sufficiency.
8. Receipt is replayed into a different tenant, subject, domain, policy or tool-profile context.
9. Whole-store rollback restores deleted evidence or a revoked signer/profile.
10. Promotion evaluation has duplicate pairs, incomplete arms, unbalanced budgets, too few independent clusters, non-estimable intervals, a negative worst-family guardrail or failed security controls.
11. Candidate process attempts to read evaluator gold labels, outcome utility, pairing keys or promotion certificate signing keys.
12. Ambiguous external witness/receipt publication is retried as though no mutation occurred.

## 8. Benchmark and SOTA qualification

Do not collapse unrelated benchmark results into one MACE score. Keep at least five independent evidence lanes:

- **LongMemEval-V2:** static state, dynamic state, workflow knowledge, environment gotchas and premise awareness; report accuracy/LAFS, latency and budget frontier.
- **AMA-Bench:** causal/relational retrieval and tool-use; reproduce the paper protocol and resolve the paper-vs-live-site score discrepancy before comparison.
- **MemoryArena:** multi-session agent-environment completion, not just recall.
- **Agent Memory Leaderboard:** separate Textual, Coding and Multimodal tracks; also report per-capability subscores and exact resource budgets.
- **MemSecBench / MemoryDojo / MemPoison:** poisoning persistence, downstream execution, propagation, selective repair and erasure/rollback controls.

MACE-specific measures should include: decision regret vs full-evidence oracle; compression-regret gap; selective risk/coverage; calibration; abstention utility; candidate-to-qualified yield; evidence freshness and revocation latency; decision influence under source-root knockout; task-family worst case; repair precision/recall; poisoned-memory action rate; erasure resurrection rate; p50/p95/p99 latency; cold-build and warm query cost; tokens, CPU, RSS, storage and network I/O.

Current competitive anchors (Agent Memory Leaderboard Cycle 2, 2026-11-19) are open leaders 61.22 textual, 43.26 multimodal and 63.33% coding; commercial leaders are 69.26, 53.68 and 68.00%, respectively. These are separate track scores, not a blended baseline. Gap priorities from the open-to-commercial comparison are multimodal retrieval (+16.45 points), textual personalization (+15.88), textual streaming (+13.46), textual epistemic safety/privacy (+13.39), and coding bug-fix resolution (+8.08). These are research priorities, not claims that MACE already closes the gaps.

A proposed target ladder is:
- **Phase I:** beat each open-method track leader by at least 5 points on a reproducible matched submission.
- **Phase II:** beat each commercial track leader by at least 2 points where the track permits a fair comparable submission.
- In every phase, do not trade away safety, worst-family performance, evidence governance or cost parity to raise a headline average.

The numerical targets are stretch goals; no MACE score is measured in this contract. A SOTA claim requires official harness execution, pinned reader/judge/model/data/environment, paired confidence intervals, worst-family reporting, independent evaluation and reproducible receipts.

## 9. Implementation sequence

1. Land the acquisition-status containment patch and retain immutable V1 replay.
2. Implement the actual V2 acquisition struct, V2 digest and Rust/Python golden vectors; update durable storage to load both versions.
3. Connect the independent qualification service and verify signed qualification receipts at use.
4. Implement decision-sufficiency receipts with calibration-only threshold fitting and risk/coverage reporting.
5. Integrate independent action/outcome receipts across the Symtropy execution/observation boundary.
6. Implement sealed paired evaluation and policy-promotion certificates; keep exploratory notes non-authoritative.
7. Run the official benchmark suites and adversarial controls. Only then consider marking this protocol implementation complete.

## 10. Current claim boundary

This PR adds schemas and an implementation contract only. It does not claim that these receipt issuers, consumers, stores, migration adapters, calibration estimators or promotion services are implemented. Schema validity is not cryptographic verification; signatures are not semantic truth; evidence sufficiency is not action authorization; and observed outcome is not proof of causal policy improvement.
