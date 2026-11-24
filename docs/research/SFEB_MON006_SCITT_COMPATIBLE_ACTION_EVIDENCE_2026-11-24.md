# SFEB MON-006 — SCITT action evidence and RFC 8949 CBOR correction

**Date:** 2026-11-24. **Local result:** 29/29 deterministic cases pass in four fresh raw-source invocations and one compressed-stdin invocation.

## Standards alignment

RFC 9943 defines COSE_Sign1 Signed Statements with protected CWT issuer and subject claims, signature/key identification and content type. A Transparency Service verifies the issuer statement, applies registration policy, registers it, and returns a receipt. The statement's unprotected header must be empty before registration. The same Signed Statement may be registered with multiple independent services, and receipts attach under unprotected COSE label 394 in a Transparent Statement.

RFC 9942 defines COSE Receipts for verifiable data structures. For RFC9162_SHA256, VDS identifier 1 identifies the Merkle Tree; inclusion proof type -1 carries tree size, leaf index and path; VDP is carried in label 396. The receipt payload is the detached Merkle root: verify inclusion against the candidate statement, derive the root, then verify the TS signature.

The local prototype exercises those wire structures using CBOR tag 18 COSE_Sign1, EdDSA (-8), protected content_type/kid/CWT iss/sub, two differently keyed mock TS logs and two detached-root inclusion receipts. These are deterministic local test objects, not live SCITT registrations.

## CBOR standards audit and correction

A pre-merge review found an earlier comparator that used legacy RFC 7049 length-first ordering for CBOR map keys. That is not RFC 8949 core deterministic encoding. The comparator was corrected to bytewise lexicographic order of each key's deterministic encoded bytes, shortest integer/length arguments and definite lengths for the supported data types. A mixed-length key vector (10, 100, -1, "z", "aa") now explicitly distinguishes RFC 8949 core ordering from length-first ordering.

The corrected source passes **29/29 tests** in four fresh raw runs and one compressed run. The Signed Statement, Receipt A/B and Transparent Statement bytes are identical across all five runs. This encoder remains a limited supported subset, not a general RFC 8949 implementation; independent library differential testing is still required.

## What the receipt does not prove

A TS receipt proves that the TS accepted/registered the issuer's signed statement and that the statement is included in its VDS. It does not prove that the signed action claim is true. The test suite deliberately signs a fabricated sink/readback result and shows that the mock TS accepts it. SFEB must separately verify source provenance, current tenant authority, objective fence at the external sink, stable idempotency identity, sink response and independent postcondition readback. An unknown outcome must stay unknown regardless of transparency receipts.

VDS append order also is not a causal-order guarantee. The SFEB action payload therefore includes an explicit aggregate/action sequence and stable operation identity.

## Executed benchmark (not SOTA)

Four same-process runs, 600 statement create+verify and 600 receipt verifications per run:
- Signed action create+verify: median **3,272.5 ops/s** (range **3,020–3,537**).
- Detached receipt verification: median **6,559.5 ops/s** (range **5,335–6,604**).

These exclude network, persistent TS storage, registration policy, key discovery, external sink and readback. They are not comparable to DBOS, Temporal, PostgreSQL or provider sizing numbers.

## Paradigm shift and next work

Use SCITT as the standardized transparency layer and SFEB as the domain-specific semantics and independent verifier. This avoids creating a competing transparency ecosystem by default. Keep threshold witnesses only where threat-model analysis justifies them.

Next: complete an interoperable SFEB media type/schema profile; cross-language RFC 8949/COSE/receipt vectors in Rust, Go and JavaScript; differential test with independent libraries; connect a real SCITT TS; verify receipts from two independently administered TSs; test outage/split-view/gossip; connect one real idempotent sink with sink-side objective fencing and independent readback; then benchmark paired PostgreSQL outbox, DBOS Go, DBOS Java, DBOS Rust and Temporal native/augmented arms on identical traces and faults.

Proposed gates (not achieved): zero unauthorized/cross-tenant effects over >=1M independently denominated attempts; 100% of externally anchored claims independently verify issuer/TS/SFEB evidence; no success on unresolved sink outcomes; <=10% p95/p99 overhead against matched native external-action workloads. Report effect latency separately from asynchronous registration/anchor latency.

## References

- RFC 9943 SCITT: https://www.rfc-editor.org/rfc/rfc9943.html
- RFC 9942 COSE Receipts: https://www.rfc-editor.org/rfc/rfc9942.html
- RFC 9162 Merkle Tree Hash: https://www.rfc-editor.org/rfc/rfc9162.html
- RFC 8949 Core Deterministic Encoding: https://www.rfc-editor.org/rfc/rfc8949.html
- RFC 9052 COSE structures: https://www.rfc-editor.org/rfc/rfc9052.html
