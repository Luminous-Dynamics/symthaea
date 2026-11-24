# SFEB MON-006 — SCITT-compatible action evidence v1 (corrected CBOR ordering)

## Local execution

Node.js v22.16.0; built-in modules only.

- **29/29 deterministic tests passed** in four fresh raw-source invocations and one compressed-stdin invocation.
- Exact CBOR wire fixtures (Signed Statement, Receipt A/B, Transparent Statement) are byte-identical across all five runs. Whole-run JSON differs because it includes variable timings.
- The issuer statement is tagged CBOR tag 18 COSE_Sign1 using EdDSA (-8), protected content_type, kid and CWT iss/sub.
- The tenant-bound action payload commits to resource, stable operation identity, request/source digests, authority revision, objective fence, execution/sink/readback digests and aggregate sequence.
- Two differently keyed mock Transparency Services register the exact same Signed Statement and issue separate RFC 9942-shaped RFC 9162 inclusion receipts. The receipts attach under COSE label 394.
- Negative controls cover tampering, tenant/resource substitution, issuer/TS identity, proof tampering, VDS substitution, duplicate TS identity, insufficient receipts and registration-header constraints.
- A fabricated sink/readback claim signed by the issuer is deliberately accepted by the mock TS: transparency establishes provenance and registration, not factual truth.

## CBOR standards-audit correction

The initial comparator used legacy length-first ordering. It is corrected to RFC 8949 core deterministic bytewise lexicographic order of encoded map keys. A mixed-length regression vector explicitly distinguishes core ordering from length-first ordering. Supported integer/length arguments use shortest encodings and indefinite lengths are rejected. This remains a narrow supported CBOR subset, not a general RFC 8949 library.

## Run

```sh
gzip -dc benchmarks/frontend/monitoring/mon006-scitt-action-evidence-v1.mjs.gz | node --input-type=module --check -
gzip -dc benchmarks/frontend/monitoring/mon006-scitt-action-evidence-v1.mjs.gz | node --input-type=module
```

## Local four-run microbenchmark

- Signed action create+verify: median **3,272.5 ops/s**, range **3,020–3,537**.
- Detached inclusion-receipt verification: median **6,559.5 ops/s**, range **5,335–6,604**.

Same-process CBOR/Ed25519/Merkle CPU only. No network, persistence, live registration, key discovery, real sink or readback. Not an industry or SOTA comparison.

## Limits

No live SCITT TS, independent TS operators, production keys, key rotation/revocation, full RFC 8949 implementation, independent COSE differential test, KMS/HSM, source/authority services, mutation sink or independent postcondition reader. The action media type is a prototype and needs a complete interoperable SFEB profile.
