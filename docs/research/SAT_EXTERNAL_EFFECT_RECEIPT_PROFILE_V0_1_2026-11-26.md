# SAT External Effect Receipt and Reconciliation Profile v0.1 — 2026-11-26

**Status:** research profile and fixture specification; implementation and execution pending.  
**Parent:** SAT Qualification Infrastructure v0.1 and SAT Trace Verifier v0.1.  
**Purpose:** bind a frozen execution attempt to authenticated, independently corroborated evidence of the external effect that actually occurred.

## 1. Executive thesis

SAT must distinguish the truth of an external effect from the agent's report, the connector's response, the executor's trace, and the transparency log's inclusion receipt.

A transparency receipt can prove that a statement was signed and recorded. It does not prove that the statement accurately describes a bank ledger, actuator, cloud resource, email destination, or other external system. A valid signature over a false or incomplete statement remains a false statement.

This profile introduces a portable evidence envelope for the relationship between:

1. a frozen intent and exact effect binding;
2. one unique execution attempt;
3. the source system's signed outcome evidence;
4. independent read-back evidence of external state;
5. an independent verifier's reconciliation decision.

It composes with the existing SAT event contract and verifier. It does not replace SCITT, Rekor, provider-specific receipts, capability authorization, or source-of-truth systems. It does not create a new transparency ledger or cryptographic primitive.

## 2. Threat model and trust boundary

Assume any of the following may be faulty, compromised, misleading, stale, or colluding: the agent, model provider, prompt and memory, tool adapter, orchestration layer, executor, event store, connector response, one observer, one transparency service, one signing key, and the qualification subject itself.

The verifier is expected to operate outside the subject's administrative control for an L3+ claim. Trust roots and acceptance policy are selected independently of the subject. A subject-provided receipt is an untrusted input until its signature, key status, statement identity, and corroborating evidence have been checked.

The profile distinguishes four facts:

- **Submission:** a particular attempt was sent to an external boundary.
- **Acknowledgement:** a connector or sink claims to have accepted or processed the request.
- **Observed state:** an authoritative or independent source reports the resulting external state.
- **Correspondence:** the independently verified observed state matches the exact authorized effect, including the sink, operation, resource, purpose, parameters, payload commitment, and attempt identity.

A connector success response is not independent read-back. Two different `source_id` strings do not establish independence. A cryptographically valid statement does not establish source-system correctness. The assurance report must preserve these boundaries.

## 3. Wire artifacts

### 3.1 Effect attempt record

`schemas/sat-effect-attempt-v0.1.schema.json` defines a frozen attempt record. It binds:

- `run_id` and `manifest_digest` to the frozen SAT run;
- `attempt_id` to exactly one execution attempt;
- `idempotency_key` to stable intent across safe retries;
- `intent_digest` to the initiating intent;
- `authority_decision_digest` and `disclosure_decision_digest` to separately recorded decisions;
- `effect_binding` to `sink_id`, `action`, `resource_id`, `purpose`, `parameter_digest`, labels (unique and UTF-16 sorted) and a domain-separated binding digest;
- `submitted_at` to the attempt's declared submission time.

The stable idempotency key is not an attempt identifier. A safe retry preserves the intent, idempotency key and exact-effect binding while receiving a fresh `attempt_id`. A receipt for attempt A cannot settle attempt B, even when both share an idempotency key.

The attempt's `effect_binding.binding_digest` is recomputed over the `effect_binding` object with the `binding_digest` field removed. The digest is `SHA-256(UTF8("SAT-EFFECT-ATTEMPT-BINDING-v0.1\n") || canonical_json(binding_without_digest))`, encoded as lowercase `sha256:` hex. The canonicalization follows the existing restricted SAT v0.1 JSON profile: duplicate keys and non-integer/unsafe numbers are rejected, object keys are sorted by UTF-16 code units, arrays retain order, and strings are not Unicode-normalized. This is a versioned SAT profile, not a claim of full RFC 8785 conformance. The example vector is `sha256:558cfd260809c2c99a32da1019c3bd360e4b2aca56f26f5008cf02144a0e3077`.

### 3.2 External effect receipt

`schemas/sat-effect-receipt-v0.1.schema.json` defines the receipt envelope. It repeats the identity commitments needed to detect cross-run, cross-manifest, cross-intent and cross-attempt replay. The `observed_outcome` is one of `applied`, `not_applied`, `partial`, `unknown` or `rejected`.

For `applied` and `partial`, the receipt must include an observed effect projection with its sink, operation, resource, purpose, parameter and payload commitments, external-effect identifier commitment, projection schema and projection digest. For `not_applied`, `unknown` and `rejected`, the receipt must not invent an observed applied effect.

Each source attestation binds the same run, manifest, attempt, idempotency key, intent and exact-effect digest. It names the source system and version, source role, source event sequence, observation time, outcome, projection digest, signature envelope, signing key, key-status evidence, timestamp evidence, trust policy and independence-basis evidence reference.

References are content-addressed or otherwise retrievable by the independent evaluator. A reference's presence is not proof that its contents exist or are authentic.

### 3.3 Verification result is computed, not self-asserted

The receipt contains evidence inputs, not a field declaring itself qualified. The verifier computes a structured result with at least:

- `receipt_integrity`: valid / invalid / unavailable;
- `attempt_binding`: exact / mismatch / stale / replayed;
- `source_authenticity`: verified / invalid / unknown;
- `source_independence`: satisfied / not_satisfied / unverified;
- `outcome_reconciliation`: matched_applied / confirmed_not_applied / mismatched_effect / partial / unknown / conflicting / insufficient_evidence;
- `mutation_disposition`: terminal / hold_reconcile / reject_new_mutation;
- `qualification_effect`: supports_claim / adverse_evidence / inconclusive.

No single boolean may collapse these dimensions. A valid receipt for an unauthorized or wrong effect is adverse evidence, not a verification pass. Missing evidence is inconclusive, not evidence that the effect did not occur.

## 4. Normative reconciliation invariants

A conforming verifier implementation MUST enforce the following invariants. JSON Schema checks structure; it cannot enforce these cross-record and stateful rules by itself.

1. **Frozen-run binding:** receipt and every source attestation match the expected run ID, manifest digest and frozen trust-policy digest.
2. **Attempt binding:** receipt matches one known attempt ID and its idempotency key, intent digest and exact-effect binding digest.
3. **No stale settlement:** evidence for an earlier attempt cannot settle a later attempt, including a retry using the same idempotency key.
4. **No intent rebinding:** one idempotency key cannot be reused with a different intent or exact-effect binding.
5. **No duplicate terminal settlement:** an attempt has at most one accepted terminal outcome; contradictory terminal observations move the case to `conflicting` and trigger investigation.
6. **Unknown means hold:** an `unknown` or ambiguous outcome blocks another mutation until independent reconciliation establishes a safe disposition. A transport timeout is not proof of non-execution.
7. **Retry only after non-application:** a fresh attempt may follow a verified `not_applied` result only when the exact intent and idempotency binding are preserved and the retry policy permits it.
8. **Partial is not success:** a partial effect is explicitly represented and reconciled; it cannot be normalized to `applied` or `not_applied`.
9. **Exact effect comparison:** a verified applied effect is compared against every field in the authorized exact-effect binding. A mismatch is a system failure, even where the sink receipt is authentic.
10. **Independent corroboration:** high-impact effect classes require the source roles and independent-source quorum in the trust policy. An execution connector's response alone cannot satisfy a policy requiring independent read-back.
11. **Source sequence monotonicity:** source event sequences and source-state versions cannot silently roll back. Persist prior accepted sequence/checkpoint state outside the subject.
12. **Key lifecycle:** signing keys, certificates, revocation status, key usage, algorithm constraints and trusted time evidence are checked under a versioned policy. The verifier records whether a revocation is a key retirement, ordinary revocation or suspected compromise and applies explicit effective-time semantics.
13. **No implied physical truth:** signatures authenticate a statement from an identity; they do not prove the sensor was calibrated, the ledger was honest, or a physical event occurred. Those are separate source-integrity and measurement-assurance claims.
14. **Evidence custody:** all referenced signed statements, proofs, policy snapshots and source records are retrievable or explicitly reported unavailable. The subject cannot silently replace a missing artifact with a newly generated one.
15. **Privacy minimization:** receipts carry commitments and minimum metadata, not raw prompts, credentials, personal data or full sensitive payloads. Low-entropy hashes can be guessed; use a reviewed, versioned salted/opaque commitment scheme when required and preserve salt custody for authorized re-verification.

## 5. Independence policy

Independence is a governance and systems property, not a count of labels or public keys.

A trust policy should declare the required evidence roles for each effect class. Example:

| Effect class | Minimum evidence roles | Example authoritative/read-back sources |
|---|---|---|
| Payment / financial transfer | Sink acknowledgement plus independent ledger or statement read-back | Payment provider transaction record plus separately operated ledger/export |
| Cloud mutation | Provider operation receipt plus independent resource-state read | Cloud API receipt plus account-scoped control-plane state query |
| Email / data disclosure | Delivery or egress receipt plus independent destination/egress evidence | Mail service delivery record plus recipient/egress audit evidence |
| Physical actuation | Controller acknowledgement plus independent sensor/physical-state observation | Robot controller log plus separately sampled encoder, force or vision evidence |

The policy should record legal/operator identity, ownership or administrative control, cloud and key custody, software supply chain, network path, source of truth, conflicts of interest and incident disclosure. Distinct keys under one administrator or one cloud tenancy may be cryptographically distinct but operationally dependent.

A verifier can check that independence evidence satisfies declared, independently approved policy rules. It cannot prove the absence of undisclosed common control from signatures alone. Report `independence_asserted` separately from `independence_qualified`.

## 6. Transparency integration and adjacent work

Use SCITT signed statements and receipts or Sigstore transparency mechanisms as transport and publication layers where appropriate. Keep the SAT effect-receipt profile as the semantic binding layer: it defines what an external outcome statement must commit to, while the selected transparency service defines how the signed statement is registered and proven included.

RFC 9943 defines the SCITT architecture. The SCITT Reference APIs draft defines interoperable HTTP resources for interacting with a transparency service. Sigstore Rekor documents signed checkpoints, inclusion and consistency proofs, and the need for clients to persist accepted checkpoints. These standards and implementations should be consumed rather than copied.

The October 2026 individual Internet-Draft *Causal Provenance Receipts for Agent Sessions: A SCITT Profile* is directly adjacent: it addresses session lifecycle and resumption provenance. SAT should cite and interoperate with that work. SAT's distinct target is qualification of authorized external effects, independent outcome correspondence, adversarial trust dependencies, and ongoing system assurance—not a competing general-purpose session receipt.

No claim is made that SCITT or Rekor proves an external physical or financial outcome merely by logging a statement. The origin and reliability of each source attestation remain part of the assurance case.

## 7. Failure taxonomy

| Condition | Receipt integrity | Reconciliation | SAT interpretation |
|---|---|---|---|
| Valid signature, exact effect, policy-required independent read-back | Valid | Matched applied | Supports the bounded effect claim |
| Valid signature, wrong recipient/amount/resource/operation | Valid | Mismatched effect | Adverse system evidence; not qualified for the affected gate |
| Connector acknowledgement only where independent read-back is required | Valid or unverified | Insufficient evidence | Inconclusive; do not claim the effect was independently confirmed |
| Timeout / ambiguous response | Not necessarily invalid | Unknown | Hold and reconcile; no blind retry |
| Old attempt receipt presented for fresh attempt | May be valid for old attempt | Stale / replayed | Reject as evidence for the new attempt |
| Cross-run or changed-manifest receipt | May be valid elsewhere | Identity mismatch | Reject for this run |
| Contradictory source outcomes | May be individually valid | Conflicting | Hold; investigate source reliability or incident |
| Invalid signature, revoked/compromised key under policy, or malformed proof | Invalid or unknown | Not eligible | Invalid evidence; cannot support qualification |
| Valid `not_applied`, exact retry intent, fresh attempt ID | Valid | Confirmed not applied | Retry may be permitted by policy |
| Partial effect | Valid or unknown | Partial | Requires compensation/recovery plan and explicit residual-risk record |

## 8. SAT effect-truth benchmark

`fixtures/sat-effect-reconciliation-v0.1.jsonl` is a public development fixture catalog. It defines positive controls and fault scenarios; it is not an executed campaign or a measured result.

The campaign should use frozen service emulators and, later, independently operated real sinks. Preserve source-native result fields and IDs. Run paired benign controls alongside attacks. Attackers should adapt after learning public protocol behavior, but sealed source records and adjudication labels must remain unavailable to the subject.

Required fixture families:

- receipt and statement substitution;
- wrong sink, resource, operation, purpose, amount and payload;
- attempt replay, idempotency-key rebinding and cross-run replay;
- unknown, partial, delayed and duplicate outcomes;
- contradictory connector and source-of-truth observations;
- source sequence rollback and stale read-back;
- invalid signatures, wrong key usage, stale trust roots and revocation/compromise timing;
- same-operator “independent” witnesses and shared key custody;
- missing evidence, unavailable references and privacy leakage;
- retry-after-confirmed-non-application versus retry-after-unknown.

### Proposed metrics and research targets

| Measure | Initial target | Interpretation |
|---|---:|---|
| Unauthorized verified applied effects accepted as safe | 0 | Hard non-compensatory gate |
| Exact-effect mismatch accepted as matched | 0 | Hard non-compensatory gate |
| Stale/cross-run receipt settling another attempt | 0 | Hard non-compensatory gate |
| Unknown outcome followed by an unapproved new mutation | 0 | Hard non-compensatory gate |
| Invalid signatures/proofs accepted on finite negative controls | 0 | Finite conformance target, not universal security proof |
| Missing required independent read-back reported as qualified | 0 | Hard evidence-completeness gate |
| Correctly matched benign outcomes held unnecessarily | ≤1% | Research target; report confidence interval and strata |
| Independently evidenced critical-effect coverage | 100% | Target for the declared critical effect scope |
| Receipt-to-reconciliation latency | p50/p95 reported | Do not optimize by weakening checks |
| Evidence verification overhead | p50/p95 time and cost | Include all monitor, witness and auditor costs |

A zero-failure result must include the denominator and an upper confidence bound. Correlated attempts, repeated environments and adaptive attackers are not independent Bernoulli trials. Use task-family/attacker/environment clustering and preregister the statistical unit.

## 9. Matched benchmark strategy

SAT should add this evidence layer around native upstream tasks, not modify those tasks and report the result as their original score.

- **AgentDojo:** retain its 97 tasks and 629 security test cases as a prompt-injection and tool-use robustness testbed; add exact-effect receipts and source read-back for external state changes.
- **AgentHarm:** retain its 110 malicious tasks and 440 augmented variants as a harmful-capability testbed; record whether a harmful action crossed a real or faithfully emulated effect boundary.
- **DelegateGuard:** reproduce purpose-bound attenuable capabilities and revocation semantics as an authority comparator; SAT adds outcome correspondence and evidence qualification.
- **AuditBench:** preserve hidden-behavior discovery, blind positive/negative targets and evidence-adjudicated auditor metrics; an audit finding is not ground truth until behavior is independently established.
- **ControlArena / SHADE-Arena:** preserve the control protocol, monitor and attack setup; wrap trajectories in SAT evidence and report protocol overhead and independently observed effects.
- **LADDER / physical control:** preserve task horizons and success scoring; add controller/sensor evidence, actuator-authority checks, revocation and physical-state correspondence.

An upstream result is `reproduced` only when its native protocol is followed. Otherwise label it `adapted baseline` or `external reference`. Model, harness, attacker budget, environment, grader and compute differences must be disclosed.

## 10. Integration plan

1. **Contract:** add the attempt and receipt schemas to the SAT qualification package and cross-reference the existing event contract.
2. **Rust verifier:** add a read-only effect-receipt parser and cross-record reconciliation state machine to `symthaea-sat-trace-verifier`; do not grant it authority to execute or retry actions.
3. **Cryptographic adapters:** add explicit provider adapters for the chosen SCITT and Rekor profiles. Verify signed statement envelopes, key identity/usage/status, trusted time, inclusion and consistency proofs against persisted pins. Keep provider-specific parsers separate from shared policy semantics.
4. **Independent source adapters:** add test emulators first, then read-only provider/ledger/actuator observations. Record each adapter's source-of-truth limitations.
5. **Owner integrations:** consume existing CogSec, INTX, Authority and Symtropy owners. Do not introduce duplicate authorization or payment state machines. `unknown outcome => hold and reconcile` remains the cross-system invariant.
6. **Campaign:** execute fixtures with independent regrading, paired benign controls, adaptive attackers and preregistered statistics.

## 11. Evidence status and claim limits

At authoring time:

- profile, schemas, examples and fixture IDs are source artifacts;
- schema/fixture validation is a local development check only;
- no Rust implementation of this profile has been compiled or executed;
- no signature, source receipt, provider proof, revocation snapshot or witness independence has been authenticated;
- no upstream benchmark reproduction, external sink experiment, independent audit or production qualification has been performed.

The strongest permitted claim is that SAT has a specified contract and test plan for independently reconciling external effect evidence. It is not yet a functioning cryptographic effect-receipt verifier.

## References

- [IETF RFC 9943 — An Architecture for Trustworthy and Transparent Digital Supply Chains](https://www.rfc-editor.org/rfc/rfc9943.html), June 2026.
- [SCITT Reference APIs, draft-ietf-scitt-scrapi-11](https://datatracker.ietf.org/doc/html/draft-ietf-scitt-scrapi-11), work in progress.
- [Sigstore Rekor architecture](https://github.com/sigstore/architecture-docs/blob/main/rekor-spec.md) and [Rekor security model](https://docs.sigstore.dev/about/security/).
- [Causal Provenance Receipts for Agent Sessions: A SCITT Profile, draft-bailey-scitt-agent-session-receipt-00](https://datatracker.ietf.org/doc/html/draft-bailey-scitt-agent-session-receipt-00), individual Internet-Draft, work in progress.
- [AgentDojo](https://papers.neurips.cc/paper_files/paper/2024/file/97091a5177d8dc64b1da8bf3e1f6fb54-Paper-Datasets_and_Benchmarks_Track.pdf), NeurIPS 2024 Datasets and Benchmarks.
- [AgentHarm](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c493d23af93118975cdbc32cbe7323f5-Abstract-Conference.html), ICLR 2025.
- [DelegateGuard](https://arxiv.org/abs/2610.16005), October 2026 preprint.
- [AuditBench](https://alignment.anthropic.com/2026/auditbench/), March 2026.
- [ControlArena](https://control-arena.aisi.org.uk/), UK AI Security Institute and Redwood Research.
- [LADDER](https://ladderrobotics.ai/research), long-horizon robot manipulation benchmark.