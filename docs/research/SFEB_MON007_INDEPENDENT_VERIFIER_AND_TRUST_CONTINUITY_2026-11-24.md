# SFEB MON-007 — independent action-evidence verification and trust continuity

**Date:** 2026-11-24  
**Status:** research and qualification specification; no implementation or test result is claimed  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/13550  
**Related work:** MON-005 #13515 / draft PR #13518; MON-006 #13528 / corrected draft PR #13531

## Executive decision

SFEB should become an **independently verifiable action-assurance profile**, not another workflow engine, generic transparency log, or replacement for DBOS and Temporal.

MON-005 established a local reference progression from signed tenant-bound checkpoints to RFC 9162 consistency proofs and locally persisted witness state. MON-006 established a SCITT/COSE wire prototype and corrected its CBOR map ordering to RFC 8949 core deterministic ordering. The next bottleneck is no longer producing a signature or a Merkle root. It is proving that independent implementations can reconstruct the complete assurance decision from portable evidence, including the policy and trust state that applied when the evidence was registered.

The primary architectural recommendation is therefore:

1. Reuse SCITT (RFC 9943) and COSE Receipts (RFC 9942) as the transparency interchange.
2. Reuse an established transparency-log implementation and witness policy where it satisfies the threat model.
3. Make SFEB's differentiated layer the action semantics, causal evidence graph, trust continuity, sink-enforced authority/fencing contract, and independent verifier.
4. Keep action truth, source provenance, authority, external effect, readback, and transparency registration as separate predicates.
5. Require cross-implementation verification before claiming interoperability.

This avoids building an SFEB-only transparency ecosystem unless testing demonstrates a concrete assurance property that the existing ecosystem cannot supply.

## 1. Current evidence and explicit qualification boundaries

### MON-005

The open draft PR #13518 records:

- v1: 39/39 deterministic cases and 496 inclusion proofs.
- v2: 45/45 deterministic cases, 496 inclusion proofs and 2,016 consistency-proof pairs over the stated tree-size range.
- v3: 49/49 deterministic cases, retaining the Merkle results and adding four SQLite cases for handle close/reopen, recovery after a committed write with lost acknowledgement, rollback on pre-commit failure, and writer contention.

The critical limitation remains that all four SQLite witness databases were exercised on one host in one process. The tests do not establish independent process or host failure domains, physical power-loss durability, distributed atomic compare-and-set, independent operators, or production key custody. The quorum safety argument is conditional on the stated Byzantine bound and durable refusal to equivocate; the local harness does not prove those distributed assumptions.

### MON-006

The corrected open draft PR #13531 records 29/29 deterministic cases in four fresh raw-source runs and one compressed-stdin run. Its Signed Statement, two receipt fixtures and Transparent Statement are byte-identical across those five executions. The CBOR comparator correction uses RFC 8949 core deterministic bytewise ordering of encoded keys for the supported subset, not the earlier length-first ordering.

The prototype still uses two mock transparency services in one process with fixture keys. It is not a general RFC 8949 implementation and has not been differentially verified against an independent COSE/CBOR implementation. No live SCITT service, real mutation sink, sink-side fence, or independent postcondition reader is connected. Its local CPU microbenchmarks exclude networking, persistent storage, live registration and external actions; they are not industry comparisons.

### Rust preflight

The standalone rust-evidence-attestation crate contains strict JSON duplicate-key rejection, RFC 8785 canonicalization, DSSE v1 PAE and Ed25519 source-receipt verification logic. Its own README explicitly says it has not been compiled or executed because the Rust/Cargo toolchain was unavailable in that environment. It remains a preflight, not a production verifier or root of trust.

**No result in this document upgrades these boundaries.** The work below is a proposed protocol and qualification plan. Implementation, execution and CI evidence must be recorded separately at exact source heads.

## 2. Architectural pivot: reuse transparency infrastructure

### 2.1 SCITT is the interchange; SFEB is the action-assurance profile

RFC 9943 provides a useful architecture for signed statements registered with transparency services. RFC 9942 defines COSE Receipts carrying proofs for verifiable data structures. SFEB should define a domain-specific profile over these standards rather than inventing another signed-log protocol.

A transparency receipt means that a transparency service accepted and registered a statement under its registration process and that the relevant inclusion proof verifies. It does **not** establish that the statement's claims are factually true. A valid issuer signature likewise proves who signed bytes, not that a resource was changed or that authority was valid.

SFEB's verifier must therefore distinguish at least:

- issuer_authenticated
- source_observed
- authority_verified
- fence_enforced_at_sink
- sink_accepted
- postcondition_read_back
- registration_policy_verified
- transparency_registered
- checkpoint_current
- causal_relationship_verified

These are independent predicates, not stages that can be collapsed into one success boolean. A statement may be signed and transparently registered while its sink claim remains unverified or contradicted.

### 2.2 Do not duplicate generic witness/log infrastructure without evidence

MON-005's custom 3-of-4 checkpoint quorum is a valuable executable safety model. It should not automatically become a second production transparency-witness network.

Evaluate this decision explicitly:

- **SCITT Transparency Service:** signed-statement registration, policy enforcement, receipts and auditable registration collateral.
- **Tessera / C2SP witness policy:** established transparency-log and checkpoint-witness mechanisms to evaluate as infrastructure.
- **Independent monitors:** retain trusted checkpoints, verify consistency, gossip observations and detect split views or freezes.
- **SFEB verifier:** verifies the action claim graph, tenant authority, sink fencing, readback and the transparency evidence.

Tessera is a transparency-log implementation/library, not itself a SCITT Transparency Service. Treat it as a possible log backend or component, not as a drop-in SCITT API. Trillian's project documentation describes maintenance mode and directs new log operators toward Tessera; verify current project status and operational fit again when selecting a production dependency.

**Decision gate:** prefer an established transparency service/log with independently operated witnesses and monitors if it meets the threat model and audit requirements. Retain or extend custom MON-005 witnessing only if experiments identify a specific SFEB assurance property that the chosen SCITT/log/witness combination cannot provide—for example, cross-service action-level checkpoint linkage or an independently governed tenant/resource anti-equivocation rule. Do not preserve custom infrastructure merely because it already exists.

### 2.3 Registration policy is evidence, not configuration trivia

RFC 9943's registration-policy and trust-anchor requirements are especially important. A verifier must be able to establish which registration policy and trust anchors applied when the TS accepted the statement. The TS's current policy cannot silently be substituted for its historical policy.

The portable evidence should therefore identify and carry or reference:

- TS identity and key identity;
- TS registration-policy statement digest and version;
- TS trust-anchor statement digest and version;
- relevant collateral needed to replay registration checks;
- registration time/sequence as asserted by the TS;
- receipt and proof identifying the VDS entry and checkpoint;
- the policy/trust-root chain that establishes why those policy statements were trusted.

The verifier must independently validate the policy statement, trust-anchor history, registration proof and receipt signature. A receipt without the historical policy context is incomplete for an auditor attempting to reproduce the TS's registration decision.

### 2.4 Pin evolving API interfaces

The SCITT Reference APIs document is an Internet-Draft and must not be treated as a final published RFC. An adapter should pin the exact draft revision it implements, expose an explicit adapter/profile version, preserve wire fixtures, and tolerate future API changes through a versioned boundary. Do not let an unversioned draft endpoint become an implicit protocol dependency.

## 3. SFEB evidence model: typed claims and dependency graph

### 3.1 Claim graph, not a single envelope boolean

Represent assurance as a directed acyclic graph of claims and evidence references. Every claim has:

- a stable claim ID and schema/profile version;
- subject, tenant and resource scope;
- issuer and key identity;
- the exact signed payload bytes or content digest;
- evidence type and verification method;
- validity/revocation context;
- causal predecessor references;
- independent verifier result and reason code.

A claim can depend on other claims. For example:

- sink_accepted depends on an authenticated sink receipt and a matched stable operation ID.
- postcondition_read_back depends on a separately authorized readback observation and a resource revision at least as new as the sink operation's committed revision.
- authority_verified depends on a trusted policy/authority statement and proof that its revision was current at the sink's acceptance point.
- externally_anchored depends on a valid signed statement, TS registration receipt(s), valid VDS inclusion proof(s), trusted TS identity/policy and a checkpoint accepted by the verifier's monitor policy.

A failed dependency must not be hidden by a successful dependent signature. Missing or contradictory evidence yields indeterminate, rejected, or a narrower qualified state—not success.

### 3.2 Suggested result vocabulary

Use a structured result rather than a single success:

- verified: all required predicates for the requested assurance profile have independently verified.
- partially_verified: some predicates verify, but others are absent or outside the profile.
- indeterminate: evidence is missing, stale, unavailable, or cannot resolve the outcome.
- rejected: a required predicate is contradicted, invalid, unauthorized, replayed, or out of scope.
- revoked: the relevant key, authority, profile, or trust root is revoked under the applicable policy.

Return an explicit predicate map, reason codes, verifier/profile version, and the exact evidence digests considered. A caller must not infer effect_verified from externally_anchored.

### 3.3 Stable operation identity and tenant/resource binding

The action statement must bind at minimum:

- tenant ID;
- canonical resource identity and resource type;
- stable operation ID generated before the first attempt;
- action type and canonical request digest;
- source identity, source event ID and source sequence/high-watermark;
- authority subject, capability and policy revision;
- objective/fencing token and the sink-side precondition that enforces it;
- expected resource revision or conditional-write version;
- sink receipt/transaction ID and resulting revision;
- independent readback receipt, reader identity, observed revision and postcondition digest;
- predecessor operation/aggregate digest and per-resource sequence;
- statement profile and schema version.

The same operation ID must be reused after timeouts, retries and lost acknowledgements. An unknown result enters reconciliation for that same operation ID. A fresh ID must never be minted to make an ambiguous outcome appear to be a new action.

## 4. Causal ordering: timestamps are not causality

Wall clocks can be skewed, delayed or dishonest. They are metadata, not sufficient proof of order.

Use explicit causal edges and monotonic source/resource/sink sequences:

source_sequence -> authorized_operation -> sink_revision -> readback_revision -> transparency_checkpoint

The actual edge relation must be profile-defined and may contain parallel or incomparable observations. Do not force a total order when the evidence only establishes a partial order.

The verifier should check:

1. source event identity and source sequence are valid for the source;
2. the authority revision and fence token are the ones enforced by the sink;
3. the operation ID is stable across dispatch, retry, sink receipt and readback;
4. the sink's resulting resource revision satisfies the stated precondition and expected transition;
5. readback observes the same resource and a revision that includes the operation;
6. aggregate/resource sequence and predecessor digests do not skip, regress or fork;
7. transparency inclusion is not mistaken for action/cause order.

A clock-skewed or reordered transcript should still verify if causal evidence is sufficient. Otherwise it must produce indeterminate or rejected, not a guessed order.

## 5. Trust continuity, anti-rollback and witness diversity

### 5.1 Trust-root and witness-set rotation

Adapt the useful TUF trust-root discipline:

- monotonically increasing root and witness-set versions;
- persistence of the highest accepted version and digest outside the replaceable application database;
- old-root threshold and new-root threshold authorization for rotation;
- cross-signing and explicit activation time/sequence;
- revocation and expiry semantics;
- freeze detection when expected updates stop arriving;
- fail-closed behavior for missing intermediate versions or unknown transition paths.

The verifier must be able to reconstruct which trust root authorized a key at the time of signature/registration. A current root must not rewrite historical trust decisions. An old but otherwise valid signature must not silently become current after rollback of the trust store.

### 5.2 Independence is a failure-domain property

Distinct key IDs and signatures do not establish operational independence. A witness diversity policy must record and assess correlated dependencies such as:

- operator/administrator and credential control;
- cloud account, provider, region and network route;
- hardware/KMS/HSM and backup custody;
- source-control, build and deployment pipeline;
- software implementation and shared libraries;
- time source and recovery process.

Quorum qualification should consider both cryptographic intersection and failure-domain intersection. A 3-of-4 quorum only has the claimed Byzantine tolerance if the actual deployment satisfies its assumptions. Publish the assumed maximum Byzantine/correlated failures and refuse to label a deployment qualified when its topology violates them.

### 5.3 Split-view and freeze monitoring

Independent monitors should:

- pin a trusted checkpoint and its TS/log identity;
- fetch later checkpoints independently;
- verify signed checkpoints and RFC 9162 consistency proofs;
- compare checkpoint size/root pairs across independent monitors and clients;
- gossip signed checkpoint observations;
- alert on same-size/different-root, rollback, unexplained gaps and stale progress;
- retain enough state to detect a service that withholds updates (freeze);
- distinguish service unavailability from a valid but stale checkpoint.

A single monitor's view is not proof of global consistency. Monitor diversity and checkpoint gossip are part of the trust model.

## 6. Portable evidence package and verifier contract

A portable package should contain or reference:

1. Original issuer-signed COSE Signed Statement.
2. Source provenance receipt(s) and source trust metadata.
3. Authority/policy statement and revision history.
4. Objective/fence specification plus sink proof that it was enforced.
5. Sink receipt, operation identity and resulting resource revision.
6. Separately authorized postcondition readback receipt.
7. SCITT Transparent Statement and one or more TS receipts.
8. Historical TS registration policy and trust-anchor collateral.
9. VDS inclusion proof, checkpoint and consistency proof to a pinned checkpoint.
10. Trust-root/witness-set transition chain and relevant revocations.
11. Verifier policy/profile version, predicate results and reason codes.

The verifier must work offline when the package contains sufficient collateral. It must not query the original operator's mutable database as an implicit source of truth. Online refresh may be used to obtain newer revocation/checkpoint status, but the verifier must label the time and trust basis for that refresh.

Privacy is a first-class design constraint. Do not place raw tenant payloads, personal data or resource secrets in public transparency logs. Prefer content digests and privacy-reviewed claims in public statements, with encrypted/offline collateral and explicit disclosure policy. The proof must remain useful without making the transparency log a data leak.

## 7. Independent verifier and interoperability plan

### 7.1 Implementation diversity

Retain JavaScript as the existing reference prototype. Add independent Rust and Go verifiers using independently selected, maintained implementations for:

- deterministic CBOR / COSE;
- Ed25519 and trust-key validation;
- RFC 9942 receipt parsing and proof validation;
- RFC 9162 Merkle inclusion/consistency;
- SFEB profile and claim-graph validation.

Do not share parser, CBOR canonicalizer, Merkle implementation, signature-decoding or trust-selection code across the Rust and Go implementations. A common test corpus and schema are allowed; common critical verifier code is not.

Library selection must be a reviewed dependency decision based on standards coverage, maintenance status, security posture and interoperability vectors. Do not add uncompiled dependencies to the production workspace merely to make the architecture look complete.

### 7.2 Differential conformance corpus

Create versioned wire fixtures for valid and invalid cases. For each fixture, each verifier must emit the same normalized predicate map, status, reason-code set and evidence digest set.

Required vector families:

- RFC 8949 core deterministic encoding for supported values, malformed and unsupported CBOR forms;
- RFC 9052 COSE structures and signature verification;
- RFC 9942 receipt structures and detached Merkle proof reconstruction;
- RFC 9943 Signed Statement, registration and Transparent Statement rules;
- RFC 9162 inclusion and consistency proof vectors;
- RFC 8785 JCS vectors for JSON subdocuments, with duplicate-key rejection before map collapse;
- SFEB profile vectors for tenant/resource/action and claim-graph dependencies;
- root/witness-set rotation and revocation history.

Cross-language agreement alone is not enough: include official standards vectors and an independent implementation as the oracle. When implementations disagree, preserve the failing fixture and stop profile qualification until resolved.

## 8. Adversarial qualification matrix

Every test family must include an expected result and a counterexample that proves the test would fail if the gate were removed.

| Attack family | Required negative control | Required verifier behavior |
|---|---|---|
| Tenant/resource substitution | Valid signature with changed tenant/resource | Reject |
| False sink claim | Issuer signs a fabricated sink/readback result | Registration may pass; effect verification must not |
| Stale authority | Authority revision revoked/superseded before sink acceptance | Reject effect qualification |
| Fence race | Old fence token races with newer token at sink | Sink rejects old token; verifier rejects any unproven fence |
| Unknown outcome | Sink commits but ACK is lost; readback unavailable | Indeterminate; retry same operation ID |
| Duplicate/replay | Reused operation ID with changed payload or scope | Reject and preserve original outcome |
| Causal reorder | Sink/readback revisions contradict predecessor sequence | Reject or indeterminate |
| Clock skew | Timestamps disagree with signed monotonic sequences | Do not infer causality from time |
| CBOR ambiguity | Duplicate map key, non-shortest encoding, unsupported item | Reject under the profile |
| Signature confusion | Wrong issuer/key/algorithm or spoofed key hint | Reject |
| Receipt confusion | Same TS identity presented twice as two receipts | Reject quorum requirement |
| Wrong VDS/log | Receipt points to unexpected VDS, log or checkpoint | Reject |
| Historical-policy substitution | Current TS policy presented instead of policy at registration | Reject historical audit |
| Trust-root rollback | Older root/witness-set version restored from backup | Reject |
| Invalid rotation | Missing old/new threshold, skipped version or unauthorized key | Reject |
| Split view | Same tree size with different signed roots | Alarm and withhold current checkpoint qualification |
| Freeze | Valid but stale checkpoint after freshness bound | Stale/indeterminate; never current |
| Byzantine witness | One signer equivocates or is unavailable | Preserve safety under declared bound; report liveness loss |
| Correlated control | Nominally distinct witnesses share one admin/KMS domain | Mark deployment not diversity-qualified |

Also test crashes before/after sink commit, after evidence append, after receipt issuance, and after checkpoint publication; lost acknowledgements at each boundary; delayed/reordered network messages; network partitions; DB restore to an older snapshot; key revocation during a long-running operation; and readback authorization loss.

## 9. Benchmarks and industry comparison

### 9.1 Correct matched baselines

Compare the exact capability, not product names as if every runtime has one guarantee.

| Baseline | What it supplies | SFEB comparison question |
|---|---|---|
| PostgreSQL transactional outbox | Atomic application-state/outbox commit in one database | What evidence can an external auditor verify beyond the database boundary? |
| DBOS Go RunAsTransaction | Application writes and DBOS durability record in one datasource transaction | Does SFEB add portable source/authority/sink/readback proof without duplicating atomicity? |
| DBOS Java transactional step factory | Application database writes and step output commit in the same transaction | What does SFEB independently establish outside this transaction? |
| DBOS Rust regular step | At-least-once step execution; a crash can cause re-execution | Compare fairly, but don't use it as a proxy for Go/Java transaction features |
| Temporal | Durable event history and replay; activities still need correct idempotency/sink semantics | Can an independent verifier qualify the external action and its authority? |
| Tessera / C2SP witness policy | Transparency-log and checkpoint-witness infrastructure | Can established log/witness components replace custom general-purpose MON-005 machinery? |
| SCITT Transparency Service | Signed-statement registration, policy and receipts | Can a third party replay the historical registration-policy decision and validate SFEB action evidence? |

Run native and SFEB-augmented arms using the same external sink, operation identity, request payload, workload trace, durability settings and injected fault schedule. For every transactional native baseline, report both its native transaction feature and SFEB's incremental cost/value.

### 9.2 Safety gates (proposed, not achieved)

- Zero accepted unauthorized or cross-tenant effects over at least 1,000,000 adversarial attempts per release candidate, with counts broken out by attack family.
- Zero accepted rollback, same-size fork, conflicting accepted checkpoint, stale-authority effect, or rewritten witnessed prefix within the stated fault model.
- 100% of outputs labelled externally_anchored independently verify issuer signature, historical TS policy/trust state, receipt, VDS proof and checkpoint continuity.
- 100% of outputs labelled effect_verified have sink-side fence/conditional-write evidence and separately authorized authoritative readback.
- No successful outcome for unresolved sink state. Unknown remains unknown; retry/reconciliation preserves operation identity.
- Zero cross-implementation conformance divergences across the supported profile vector corpus.

Million-attempt testing is a release-candidate safety campaign, not a substitute for formal analysis, threat modelling, or independence review. Report numerator, denominator, attack family and test harness version. A test count does not establish production failure probability.

### 9.3 Performance and scale gates (proposed, not achieved)

Target at most 10% incremental p95/p99 overhead on matched native external-action workloads, subject to the actual latency budget. Do not hide asynchronous evidence work inside synchronous action latency.

Measure separately:

- dispatch and sink-commit latency;
- source/authority evidence acquisition;
- action statement construction and signing;
- independent verification CPU;
- TS registration and receipt latency;
- witness/checkpoint publication latency;
- time-to-independent-anchor;
- readback latency and resolution time for unknown outcomes.

Sweep batch size 1/10/100/1,000/10,000; producers 1/8/32/128/512; checkpoint intervals 250 ms/1 s/10 s; witness/TS availability and outage duration; evidence payload size; and varying proof-tree sizes.

Report p50/p95/p99, throughput, proof and statement bytes, signature/CBOR/Merkle CPU, memory, storage growth, network and witness egress, checkpoint lag, recovery time, stale-detection time, readback resolution time, availability, and cost per independently verified action.

Measure the safety/performance trade-off explicitly: synchronous publication gives a stronger immediate anchoring result but can place TS/witness availability in the action's critical path. Asynchronous publication can preserve action availability but must return anchor_pending, never an overstated anchored result.

### 9.4 Competitive strategy

Do not claim to outperform DBOS or Temporal at workflow durability. They are mature execution systems with native transactional/replay capabilities. SFEB wins only if it demonstrates measurable incremental assurance at the external-action boundary:

- third-party verifiability after the application DB/operator is untrusted;
- current tenant authority and sink-enforced fencing;
- stable operation identity across ambiguous retries;
- independently authorized readback;
- auditable historical TS registration policy;
- split-view/freeze detection across independent monitors;
- cross-language proof verification without shared critical implementation.

A fair outcome may be that SFEB uses DBOS or Temporal underneath, SCITT for transparency, and Tessera or another established log/witness component, while its own code remains a thin action-evidence contract and independent verifier. That is a stronger product position than competing with all of those systems at once.

## 10. Execution plan and stop/go gates

### Phase 0 — specification (this deliverable)

- Review MON-007 claim graph, state semantics, trust model, API/profile boundaries and threat assumptions.
- Decide which guarantees are synchronous and which may be pending.
- Freeze exact terminology: statement signed, source verified, authority verified, sink accepted, effect read back, TS registered, checkpoint current.
- No implementation claim.

### Phase 1 — wire profile and golden corpus

- Publish a versioned profile draft with examples and privacy rules.
- Freeze raw CBOR wire fixtures for Signed Statement, Receipt and Transparent Statement.
- Add vectors for TS policy/trust-anchor history and trust-root/witness-set rotation.
- Independently verify all standards vectors using at least two implementations before calling the profile interoperable.

### Phase 2 — verifier diversity

- Implement Rust and Go verifiers as separate binaries/crates with independent critical libraries.
- Run differential and negative-control tests against MON-006 fixtures.
- Compile, test, lint and record exact-head evidence locally. Do not report pending CI as pass.

### Phase 3 — real trust infrastructure

- Evaluate real SCITT TS APIs against the pinned draft interface.
- Evaluate Tessera/C2SP policy as a log/checkpoint backend, not as a substitute for SCITT semantics.
- Operate independent monitors in distinct failure domains; verify historical policies, receipts, consistency and gossip.
- Use production key custody only after key lifecycle and recovery policy review.

### Phase 4 — real external action

- Select one low-risk, reversible, idempotent/conditional-write sink.
- Implement a sink-enforced monotonic objective/fence token.
- Bind authorization revision to that sink-side precondition.
- Use a separate reader identity for authoritative postcondition readback.
- Exercise crashes and lost acknowledgements at each boundary.

### Phase 5 — matched benchmark and qualification

- Run the native/augmented baseline matrix on the same hardware and fault traces.
- Publish raw traces, workload generators, versions, source SHAs and complete environment details.
- Publish all failures and degraded modes, not only successful runs.
- Keep the <=10% p95/p99 overhead and million-attempt safety targets labelled as targets until independently reproduced.

## 11. Open design questions that must be resolved before production

1. What exact action profile is required for the first sink: one resource mutation, a multi-resource transaction, or a long-running saga?
2. Which fields may be public in a VDS leaf, and which evidence must remain encrypted/offline?
3. What is the acceptable maximum age of a checkpoint_current result for each risk class?
4. Is TS registration required before an action is reported complete, or may the action be effect_verified while anchor_pending?
5. Which authority revisions/fence tokens can the target sink actually enforce atomically?
6. What independent operators and key custodians can satisfy the fault-domain model?
7. Which published TS API/profile version will be pinned for the first integration?

These are protocol choices, not reasons to block the specification work.

## References

- RFC 9943, SCITT architecture: https://www.rfc-editor.org/rfc/rfc9943.html
- RFC 9942, COSE Receipts: https://www.rfc-editor.org/rfc/rfc9942.html
- RFC 9162, Certificate Transparency Version 2.0: https://www.rfc-editor.org/rfc/rfc9162.html
- RFC 8785, JSON Canonicalization Scheme: https://www.rfc-editor.org/rfc/rfc8785.html
- RFC 8949, Concise Binary Object Representation: https://www.rfc-editor.org/rfc/rfc8949.html
- RFC 9052, CBOR Object Signing and Encryption: https://www.rfc-editor.org/rfc/rfc9052.html
- SCITT Reference APIs, draft-ietf-scitt-scrapi-11 (Internet-Draft): https://datatracker.ietf.org/doc/draft-ietf-scitt-scrapi/
- TUF specification: https://theupdateframework.github.io/specification/latest/
- Tessera project: https://github.com/transparency-dev/trillian-tessera
- Trillian project status: https://github.com/google/trillian
- DBOS Go transactional steps: https://docs.dbos.dev/golang/tutorials/transaction-tutorial
- DBOS Java transactional step factories: https://docs.dbos.dev/java/tutorials/step-factory-tutorial
- DBOS Rust steps: https://docs.dbos.dev/rust/tutorials/step-tutorial
- Temporal event history: https://docs.temporal.io/encyclopedia/event-history
