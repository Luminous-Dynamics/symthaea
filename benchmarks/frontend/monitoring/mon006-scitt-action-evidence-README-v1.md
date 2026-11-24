# SFEB MON-006 — SCITT-compatible action evidence v1

## Local execution

Node.js v22.16.0; built-in modules only.

- **28/28 deterministic tests passed** in four fresh raw-source invocations and one compressed-stdin invocation.
- Raw and compressed runs emit byte-identical Signed Statement, Receipt A, Receipt B and Transparent Statement wire fixtures. Whole-run JSON differs because benchmark timings vary naturally.
- Tenant-bound action encoded as CBOR tag 18 COSE_Sign1 using EdDSA (-8), protected kid, content_type and CWT iss/sub.
- Two differently keyed mock Transparency Service identities register the exact same Signed Statement and issue separate RFC 9942-shaped inclusion receipts.
- Receipts attach via unprotected header label 394; original issuer signature remains verifiable.
- Negative controls cover tampered payload/header/signature, tenant/resource substitution, unknown issuer/kid, non-deterministic CBOR, non-empty pre-registration unprotected header, TS key/identity mismatch, proof tampering, VDS substitution, duplicate TS identity and insufficient receipts.
- A signed fabricated sink/readback claim is deliberately accepted by the mock TS. This demonstrates that transparency proves provenance and registration, not real-world truth.

## Run

```sh
gzip -dc benchmarks/frontend/monitoring/mon006-scitt-action-evidence-v1.mjs.gz | node --input-type=module --check -
gzip -dc benchmarks/frontend/monitoring/mon006-scitt-action-evidence-v1.mjs.gz | node --input-type=module
```

## Four-run same-process microbenchmark

Each invocation created and verified 600 signed action statements and verified 600 detached inclusion receipts.

- Signed action create+verify: median **2,790 ops/s**, range **2,476–2,922**.
- Detached inclusion-receipt verification: median **5,712.5 ops/s**, range **4,856–5,982**.

CPU/CBOR/Ed25519/Merkle only. No network, persistence, live registration, key discovery, external sink or independent readback. Not comparable to DBOS, Temporal or provider throughput figures.

## Wire and scope

mon006-scitt-wire-examples-v1.json contains exact deterministic CBOR hex for the issuer Signed Statement, two receipts and assembled Transparent Statement. These are test fixtures, not live TS receipts.

Limits: no live SCITT TS, no general CBOR library, no independent COSE/SCITT implementation differential test, deterministic test keys, no production key rotation, no independent TS operators, no source/auth/sink/readback integration. The application media type is a prototype requiring a complete interoperable SFEB profile.
