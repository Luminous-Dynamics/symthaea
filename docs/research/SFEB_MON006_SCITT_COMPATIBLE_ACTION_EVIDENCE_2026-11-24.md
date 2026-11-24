# SFEB MON-006 — SCITT-compatible action statements and COSE receipt interoperability

**Date:** 2026-11-24  
**Local execution:** Node.js v22.16.0; 28/28 deterministic cases passed in four fresh raw-source runs and one compressed-stdin run.

## The paradigm shift

MON-005 proved a useful progression: Merkle inclusion and consistency proofs, a 3-of-4 witness model, durable local witness state and process-level crash recovery. But productionizing a proprietary transparency network adds unnecessary burden: independent witness operators, key governance, receipts, log monitors, split-view gossip and trust anchors.

The better default is to align with IETF SCITT (RFC 9943) and COSE Receipts (RFC 9942), retaining SFEB-specific semantics above that generic transparency layer.

RFC 9943 uses tagged COSE_Sign1 Signed Statements. The protected header binds algorithm/key identity and CWT issuer/subject claims (iss/sub); a content type identifies the payload. A Transparency Service verifies the issuer signature and protected headers, applies registration policy, registers the exact Signed Statement and then returns a receipt. Before registration, the unprotected header must be empty. The same exact Signed Statement can be independently registered with multiple services, and each receipt can later be attached in a Transparent Statement under unprotected COSE header label 394.

RFC 9942 defines COSE Receipts. For RFC9162_SHA256 the VDS identifier is 1; inclusion proof type -1 carries tree size, leaf index and inclusion path; VDP is in header label 396. The COSE receipt's payload is detached and contains the Merkle root. A verifier checks candidate-entry inclusion to derive the root, then verifies the transparency service's signature over that root.

**Critical boundary:** this proves issuer provenance and transparency-log registration, not that the issuer's action statement is factually correct. RFC 9943 explicitly discusses false statements and warns that VDS append order is not necessarily statement issuance/cause order. SFEB therefore includes an explicit aggregate/action sequence in its signed payload and independently checks source, tenant authority, external sink result and postcondition readback.

## Executed prototype

- 28/28 deterministic tests pass in four fresh invocations plus compressed-stdin execution.
- A deterministic-CBOR, tag-18 COSE_Sign1 Signed Statement is created with EdDSA (-8), protected content_type/kid/CWT iss/sub and tenant-scoped action payload.
- Payload commits to tenant/resource, stable operation ID, request digest, source receipt digest, authority revision, objective fence, execution/sink evidence digests, independent postcondition readback digest, aggregate sequence and issue time.
- Two differently keyed mock TSs register the same exact statement in separate local RFC9162 SHA-256 Merkle logs and return separate detached-root COSE inclusion receipts.
- Both receipts attach to the issuer's original Signed Statement under header label 394; original issuer signature remains valid.
- Negative controls reject payload/header/signature tampering, tenant/resource substitution, unknown issuer/kid, non-deterministic CBOR, non-empty pre-registration unprotected header, wrong TS identity/key, proof tampering, VDS substitution, duplicate TS identity and insufficient receipt count.
- A fabricated sink/readback claim signed by the issuer is deliberately registered by the mock TS. This is an intentional negative-control demonstration.

## Local microbenchmark

Four fresh runs, 600 create+verify actions and 600 detached receipt verifications per run:

- Signed action create+verify median **2,790 ops/s** (range **2,476–2,922**).
- Detached receipt verification median **5,712.5 ops/s** (range **4,856–5,982**).

This is same-process Node/CBOR/Ed25519/Merkle CPU only. No network, persistent service, live registration, real sink or postcondition reader. It is not a DBOS, Temporal, PostgreSQL or SOTA comparison. Wire fixtures are byte-identical across invocations; whole-run JSON differs because timing fields are variable.

## Recommended production architecture: SCITT-first, SFEB-semantic

1. Issuer constructs a tenant-scoped action-evidence statement containing source receipt, authority revision, objective fence, stable action key/request digest, execution receipt, sink receipt, readback digest and explicit aggregate sequence.
2. Issuer signs the exact statement using a protected COSE_Sign1 profile with registered media type and issuer/key identity.
3. Register the exact Signed Statement with one or more independently governed SCITT Transparency Services.
4. Preserve receipts in a Transparent Statement and verify issuer signature, TS identity/signature, inclusion/consistency proofs and trust/policy.
5. Separately validate domain truth: source evidence, current authorization, sink-side fence, idempotency identity and independently authorized readback. A receipt must never resolve an unknown sink outcome.
6. Use different TS operators/failure domains and monitors/gossip for split-view detection. Retain threshold witnesses only for threat models that need them; do not replace existing transparency services by default.

## Industry baselines and what SFEB must add

| Layer | Strong native baseline | SFEB incremental obligation |
|---|---|---|
| Workflow durability | Temporal history/replay | Integrate; do not rebuild |
| Same-database transaction | DBOS Go RunAsTransaction; DBOS Java transactional-step factory; PostgreSQL transaction/outbox | Matched native comparison |
| Retryable steps | DBOS Rust regular at-least-once steps | Stable action identity, unknown-outcome reconciliation |
| Transparency | RFC 9943 SCITT, RFC 9942 COSE Receipts, RFC 9162 Merkle proofs | SFEB domain-specific action-evidence semantics |
| External truth | Sink authorization/idempotency + independent readback | Verify actual source, authority, mutation and postcondition |

Proposed gates (not achieved): zero unauthorized/cross-tenant actions over >=1M independent adversarial attempts; 100% externally anchored claims independently verify issuer/signature/TS receipt/inclusion proof and SFEB source/authority/sink/readback; unknown outcomes remain unknown; <=10% p95/p99 overhead against matched native action workloads. Report synchronous action latency separately from asynchronous transparency registration/anchoring.

## Next steps

1. Complete an interoperable SFEB profile: media type, schema/versioning, CWT issuer/subject namespace, algorithms, key discovery, detached payload policy, privacy, expiry/revocation.
2. Differential-test CBOR/COSE/receipts against independent libraries and RFC examples in Rust, Go and JavaScript.
3. Connect one real SCITT TS and validate its registration API and receipt bytes.
4. Validate two independently administered TSs, outage, split-view and gossip behavior.
5. Integrate one idempotent/conditional-write sink with sink-side objective fencing and separately authorized postcondition readback.
6. Run paired PostgreSQL outbox, DBOS Go, DBOS Java, DBOS Rust and Temporal native/augmented benchmark arms using the same traces/sink/fault schedules.

## References

- RFC 9943 SCITT: https://www.rfc-editor.org/rfc/rfc9943.html
- RFC 9942 COSE Receipts: https://www.rfc-editor.org/rfc/rfc9942.html
- RFC 9162 Merkle Tree Hash: https://www.rfc-editor.org/rfc/rfc9162.html
- RFC 9052 COSE Structures: https://www.rfc-editor.org/rfc/rfc9052.html
