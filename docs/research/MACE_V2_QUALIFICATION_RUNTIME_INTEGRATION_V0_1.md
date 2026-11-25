# MACE V2 — Qualification Runtime Integration Contract v0.1

**Status:** Draft integration contract. The V2 acquisition receipt, pinned Ed25519 verifier, and fail-closed qualified-candidate projection are composable library components on the current draft branch. The LongMemEval-V2 sidecar is **not yet wired to consume a qualification receipt**, and no independent qualification service is implemented by this change.

## 1. The trust-boundary rule

The candidate backend is an untrusted evidence producer. It can report what it searched and which candidate records it materialized. Its receipt is an unsigned observation, not an assertion that the candidates are authentic, authorized, true, admissible, or sufficient.

The required chain is:

`Acquisition observation → independent qualification → exact-batch projection → decision sufficiency → live action authorization → observed outcome`

Each arrow is a separate contract. No stage may silently grant the next stage's authority.

### Acquisition observation

The V2 receipt binds a run, invocation, query, tool profile, operations, observed IDs, status, and candidate-set commitment. It can establish consistency of the candidate backend's own report. It cannot establish source authenticity, source authority, rights, or semantic correctness.

The sidecar-specific binding check additionally requires:
- candidate backend revision to match the receipt's tool-profile revision;
- candidate scope and purpose to match the LMEv2 domain contract;
- every direct hit and neighbor expansion to occur in an acquisition operation's observed-ID set;
- every source-state record to be referenced by a returned record;
- every screenshot record to have exactly one source-state parent.

These invariants establish internal lineage consistency. A well-formed lineage can still originate from false, unauthorized, or stale source material.

### Independent qualification

A separately operated qualifier must recompute the exact candidate batch and independently inspect source-observation receipts, issuer authority, rights/purpose, provenance ancestry, and current lifecycle status. It issues a signed `EvidenceQualificationReceiptV1` only for the exact candidate set and use context.

The candidate backend must not hold the qualifier's private key, choose its own trust profile, self-assert its own admissibility, or transform its acquisition trace into a qualification receipt.

The qualification receipt binds:
- run, invocation, domain, subject-scope commitment;
- environment and tool-profile revisions;
- query and complete candidate-set commitments;
- the canonical candidate evidence-ID set;
- DecisionNeed, purpose, qualification policy and lifecycle snapshot;
- verifier profile, signer key ID, issuance and expiry;
- one ordered disposition and basis-receipt digest per candidate.

A valid signature authenticates the qualifier's statement. It does not prove the source statement is true; the qualifier must have an independently defensible basis for its disposition.

### Pinned verifier and projection

The current library implementation provides:
- a pinned Ed25519 leaf-key profile with an out-of-band expected profile digest and minimum profile version;
- strict signature verification, key-ID derivation, validity-window checks and rejection of unknown/revoked keys;
- a verified capability that retains the complete signed binding;
- a use-time method that requires the caller to re-present the full binding and trusted current time;
- a pure projection that rebuilds the candidate batch, checks exact identity and candidate-ID-set equality, removes rejected/indeterminate evidence, closes over required parents, rejects cycles, and returns an explicit abstention when nothing remains admissible.

The profile digest and minimum profile version must come from a trust channel outside the candidate backend and the memory-store rollback domain. Reading a profile and its purported pin from the same mutable store is not an independent trust bootstrap.

The verified capability also retains the signer key ID, signed payload digest, and signature. At every consumption, the caller must supply a freshly provisioned verifier, and the capability re-checks the signature against that verifier at the supplied trusted current time. A newly revoked key or a changed trust-profile pin therefore fails closed; a rotation to a new profile invalidates receipts that bind the old profile and requires re-qualification/re-issuance under the new profile. Passing a cached old verifier defeats this property, so freshness and monotonicity of the verifier-profile source remain an external relying-party obligation.

### Standard DSSE / in-toto transport (library support on the follow-up branch)

The DSSE follow-up adds a transport adapter for the existing typed qualification predicate:

- DSSE protocol v1.0.2 PAE signs the exact serialized in-toto Statement bytes and the payload type; the test vector fixes the byte lengths, payload digest, PAE digest and Ed25519 signature.
- The payload uses the in-toto Statement v1 type URI and a versioned MACE Evidence Qualification predicate type. The single Statement subject binds the candidate-set SHA-256; the predicate carries the full MACE request binding and per-evidence disposition.
- The parser rejects duplicate JSON object keys, malformed/non-canonical Base64, wrong payload type, multiple signatures under the current MACE v1 profile, a mismatched subject, and predicate/receipt mismatch.
- It accepts standard and URL-safe Base64, padded and unpadded; outer DSSE keyid is optional and is only a key-selection hint. The authenticated MACE predicate carries the actual key ID and the pinned verifier checks it against the trust profile.
- Unrecognized DSSE envelope, signature-object and in-toto Statement extension fields are ignored; the MACE v1 predicate itself remains closed and exact.

**Profile limit:** DSSE allows multiple signatures and application-specific threshold policies. MACE v1 currently requires exactly one Ed25519 signature; a future multi-party qualifier policy must define distinct signer roles, threshold, independence/common-cause rules, and rotation semantics before relaxing that restriction.

The signed bytes are not canonicalized and are never reconstructed for verification: the verifier signs/verifies DSSE PAE over the exact payload bytes received, then passes the same parsed statement/predicate object to the typed MACE binding gate. The receiver must not re-read or reparse a different payload after verification.

### Executable consumer DecisionNeed sufficiency gate (follow-up branch)

The next runtime increment adds a separate DecisionSufficiency consumer verifier and a consumer-owned DecisionNeed manifest:

- DecisionNeed v1 is loaded from an authorized consumer policy registry, strictly decoded, validated and domain-separated-hashed. It fixes the decision/domain/scope/purpose, action-set and utility-profile commitments, risk limit, confidence floor, regret tolerance, calibration artifacts, maximum evidence age, context/tool/cost/latency budgets, and the exact sorted evidence requirements (evidence classes and independent-root counts).
- The signed DecisionSufficiency receipt is bound to the exact qualified-evidence-set digest and original qualification receipt, as well as the DecisionNeed, sufficiency policy and calibration profile.
- A sufficient status requires every declared requirement to be met, a calibrated risk upper bound within the consumer limit, a regret upper bound within the consumer tolerance, and measured cost/latency/context/tool/evidence-age values within the DecisionNeed budgets.
- The sufficiency evaluator has a distinct verifier role, trust-profile digest and signer key ID from the qualifier. The receipt is DSSE/in-toto signed; it is not a candidate-side status flag.
- At every use the consumer revalidates the original qualification with the current qualifier verifier and the sufficiency receipt with the current evaluator verifier, using the same trusted current time. A new sufficiency receipt cannot revive an expired/revoked qualification.

This is a **consumer verifier**, not the evaluator implementation. It verifies the evaluator's signed claim and declared measurements; it does not independently reproduce the evaluator's risk model/calibration, measure resource telemetry, or prove the regret bound from held-out outcomes. Those require a separately operated evaluator and measurement pipeline. Even a valid sufficient result is not action authorization.

### Decision sufficiency is a different question

Admissibility answers: "May this evidence be considered under this policy and use context?"

Decision sufficiency answers: "Is the admissible evidence enough to support this particular decision within its risk/regret tolerance?"

The qualified projection does not answer the second question. A non-empty qualified batch is not a sufficient-evidence verdict. The next consumer must be the separate DecisionNeed sufficiency evaluator; it must be able to return insufficient, indeterminate, acquire-more, defer, or abstain.

### Action authorization and outcomes remain separate

Even a sufficient evidence set is not an action permit. At consequential use, the executor must re-check live subject authorization, rights, revocation, lifecycle currentness and policy. Any approval/consent that needs at-most-once use requires its own consume-once state and durable witness.

An outcome receipt must bind the chosen action, exact evidence projection, pre-action prediction, observed result and any counterfactual comparison. Strategy promotion must depend on independently observed, paired, held-out results—not on a candidate's self-reported success.

## 2. Compatibility boundary

The acquisition V2 branch and the earlier qualification prototype had diverged at the signature-verifier interface. The integrated contract now supplies both `issued_at_unix` and trusted `now_unix` to the verifier. The verified token also retains the complete binding and requires it again at use.

The existing candidate-evidence representation is byte-identical across the two implementation lines; the integration preserves the V2 branch's additional modules and acquisition-receipt code rather than replacing its library root with the older branch's version.

## 3. What this increment does not claim

Not implemented by this integration:
- a separately operated qualification service or private-key custody;
- independent source-observation ingestion, source-authority registry, rights engine, or tenant-isolation service;
- remote trust-profile rotation/revocation distribution or a monotonic trust-profile witness;
- an independently operated qualifier service that produces the DSSE/in-toto envelope, or sidecar request/response consumption of that envelope; the follow-up branch adds only the library-level DSSE/in-toto verifier and schemas;
- V2 sidecar request/response integration that consumes an external signed qualification and withholds unqualified context;
- the decision-sufficiency evaluator/calibration service, executor authorization, consume-once action ledger, or outcome/promotion integration; the follow-up branch adds the consumer-side verifier and DecisionNeed contract only;
- authenticated journal checkpoints or T9/T10 rollback/erasure-resurrection protection.

Accordingly, the LongMemEval-V2 sidecar's current query result remains **unqualified benchmark evidence**. The presence of a V2 acquisition receipt or a locally verifiable signature must never be presented as production permission to use the evidence for a consequential action.

## 4. Integration acceptance criteria

A later end-to-end sidecar/runtime change is acceptable only when all of the following are true:

1. An external qualifier receives the canonical candidate batch and independently verifies source receipts, issuer authority, scope, rights, provenance closure and lifecycle state.
2. The qualifier issues a signed receipt bound to the exact candidate set, exact DecisionNeed and exact purpose/policy/lifecycle snapshot.
3. The consumer obtains the verifier profile pin and monotonic minimum version independently of the candidate store.
4. The consumer validates the signature at issuance and at every use using a trusted clock and a freshly provisioned verifier/trust-profile pin, then re-presents the complete binding at every consumption. Cached verifiers or trust-profile state inside the candidate-store rollback domain do not satisfy this requirement.
5. The projection returns only independently admissible records, removes descendants whose required parents are not admissible, and abstains on empty or inconsistent results.
6. The sufficiency evaluator separately decides whether the projected evidence is enough for the named decision; otherwise it requests more evidence or abstains.
7. The executor rechecks live authority and lifecycle and uses separate consume-once authorization state where required.
8. Tests include wrong signer, unknown/revoked/expired key, profile rollback, stale lifecycle, purpose/tenant/DecisionNeed/candidate rebinding, missing or rejected parents, cycle, empty qualification, sidecar restart, and concurrent consumption.

## 5. Evaluation programme

Keep external benchmark protocols separate. MACE should report, at minimum:

| Evaluation lane | Primary question | Required measurements |
|---|---|---|
| Official AML Cycle 2 | Does the Add/Search system compete with the strongest reproduced adapters? | Per-track score, exact frozen API/config, ingest/search latency, memory and token budgets |
| LongMemEval-V2 | Can memory retrieve and answer long-horizon questions accurately and efficiently? | Per-ability accuracy, query latency, context budget, online/offline protocol |
| AMA-Bench / WorldMemArena | Does memory improve long-horizon agent decisions and interaction with the world? | Paired task success, tool/action errors, cost, latency, trajectory-level outcomes |
| FAMA / TimlMemBench | Does memory respect corrections, expiration, scope, consent and activation rules? | Stale-memory use, scope violations, correction latency, clean-task preservation |
| Memory security lifecycle | Can poisoned, revoked, rolled-back or erased memory influence later behavior? | Persistence, propagation, retrieval exposure, causal action influence, repair precision, T9/T10 resurrection |
| MACE Decision Memory Benchmark | Does the system acquire the least costly evidence that keeps decision regret within a declared tolerance? | Regret, sufficiency calibration, abstention, EVSI, evidence independence, verification cost, p50/p95/p99 latency |

The external AML open-method targets recorded in issue #13284 are **targets, not results**: +5 points over the matched Cycle 2 open-method leaders in each separate track (textual 66.22/100, multimodal 48.26/100, coding 68.33% resolved). Do not pool those track scores.

For the internal decision benchmark, compare the Pareto frontier of decision regret against all-in cost. Include no-memory, full-context, exact-scope scan, recency, BM25, dense, hybrid, entity/date/literal, neighbor expansion, state-first, causal-graph, hierarchical compression, typed HDC, event-ID HDC, and EVSI-guided acquisition. Use random, uncertainty-first, missing-dependency-first and fixed top-k query policies as controls; keep an evaluator-only oracle as an upper bound, never as a candidate input.

Pre-register primary endpoints, task families, budgets, evaluation splits and stopping rules. Report paired intervals and worst-family regressions. A retrieval-score win is not a decision-quality win; a decision-quality win is not proof of safe authorization.

## 6. Relevant standards and research context

- [DSSE Protocol v1.0.2](https://github.com/dsse-lf/dsse/blob/master/protocol.md): signature is over PAE(UTF8(PAYLOAD_TYPE), SERIALIZED_BODY); payload type is authenticated, keyid is an optional unauthenticated hint, and verifiers must accept standard or URL-safe Base64.
- [DSSE JSON Envelope v1.0.2](https://github.com/dsse-lf/dsse/blob/master/envelope.md): payload, payloadType and signatures; unknown envelope extensions must be ignored; the same payload bytes that are verified must be the bytes delivered to the application.
- [in-toto Attestation Framework v1.2](https://github.com/in-toto/attestation/tree/main/spec/v1): Statement binds the subject and predicate type; the Statement _type remains https://in-toto.io/Statement/v1. MACE uses a private, versioned predicate type rather than claiming to be a standard in-toto predicate.
- [DSSE test and reference implementation](https://github.com/dsse-lf/dsse/tree/master/implementation): independent interoperability reference. The Rust fixture is authored but still needs execution against an independent DSSE implementation; the Python-generated Ed25519 vector is a deterministic golden fixture, not cross-implementation conformance evidence.



- [RFC 9334 — Remote ATtestation procedureS (RATS) Architecture](https://www.rfc-editor.org/rfc/rfc9334) is a useful separation analogy: an Attester produces evidence, a Verifier appraises it, and a Relying Party consumes an attestation result. It does not provide MACE's source-authority or decision-sufficiency semantics.
- [MACE issue #13455](https://github.com/Luminous-Dynamics/symthaea/issues/13455) owns the independent qualification integration.
- [MACE issue #13264](https://github.com/Luminous-Dynamics/symthaea/issues/13264) owns verified transitive evidence dependencies and lifecycle closure.
- [MACE issue #13284](https://github.com/Luminous-Dynamics/symthaea/issues/13284) owns decision-sufficiency frontiers and EVSI-guided evidence acquisition.
- [MACE issue #13319](https://github.com/Luminous-Dynamics/symthaea/issues/13319) owns adversarial memory-lifecycle and erasure-resurrection evaluation.
- The [Agent Memory Integrity Internet-Draft](https://datatracker.ietf.org/doc/draft-khandelwal-bpm-agent-memory-integrity/) is work in progress, not an IETF standard; its stale-backup erasure-resurrection case is a useful threat-model input, not an endorsement or compliance claim.
