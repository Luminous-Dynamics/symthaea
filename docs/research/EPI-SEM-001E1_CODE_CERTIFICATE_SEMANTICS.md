# EPI-SEM-001E1 — Code-certificate semantics discovery

Parent: #5392 (`EPI-SEM-001E`)

Status: measurement only; no certificate semantics or production behavior changed.

## Purpose

Freeze the current `CodeCertificate` boundary before changing it.

The current implementation combines several useful kinds of metadata in one object:

- a BLAKE3 source commitment;
- backend/source-provenance descriptors;
- verification-layer metadata;
- internal epistemic-status metadata;
- a local issuance timestamp;
- optional topology/oracle/sheaf metadata.

The current documentation also describes the object as a "cryptographic receipt" and as proving how code was generated and verified.

This tranche measures what the current object and its production consumers actually establish.

## Core theorem

```text
content hash match
!= authenticated issuer provenance
```

```text
recorded verification-layer metadata
!= independent re-execution of those verification claims
```

```text
local wall-clock timestamp
!= trusted chronology
```

```text
internal epistemic status
!= code verification property
```

```text
compile/test/formal evidence, where separately established
!= generic semantic correctness
```

## Current measured architecture

The expected production reference set is intentionally small:

```text
src/language/code_certificate.rs
src/language/code_orchestrator.rs
```

`CodeOrchestrator` creates the record after accepted code generation, copies request epistemic status and verification-layer metadata into it, stores it in an in-memory vector, and exposes a cloning accessor.

The repository's current coding-improvement plan already notes that no production file-write/API/CLI consumer reads these certificates back; tests and benchmarks do inspect them.

This means E1 must not describe the current object as a deployed portable credential. The current product boundary is closer to:

```text
in-process post-acceptance audit metadata
+ source-byte commitment/tamper detection
```

than to:

```text
authenticated portable verification credential
```

## Measurement program

`scripts/discover_code_certificate_semantics.py` performs two kinds of checks.

### Dynamic conservative discovery

It scans production Rust paths for `CodeCertificate` symbol membership and emits:

- sorted reference paths;
- exact path-set digest;
- current `CodeCertificate` field inventory;
- any authentication-shaped fields newly appearing in the struct.

The expected production reference set and field list are ratchets. Any change requires review rather than silently preserving a stale semantic conclusion.

### Exact witnesses

The script requires source witnesses for:

- current cryptographic-receipt claim language;
- BLAKE3 source commitment and source-byte verification;
- JSON serialization;
- orchestrator construction/storage/accessor behavior;
- request epistemic-status copying;
- verification-layer copying;
- current post-acceptance-metadata wording;
- the repository's existing note that the certificate has no production persistence/API/CLI consumer.

## Authentication-field rule

The current `CodeCertificate` field set contains no dedicated issuer-signature/authentication field.

E1 therefore explicitly fails for review if fields with names such as these appear later:

```text
signature
issuer
signer
public_key
verification_key
key_id
attestation
authentication_receipt
```

That does **not** mean any future field with one of those names automatically establishes authentication. It only means the E1 conclusion has become stale and must be re-qualified under the new design.

Likewise, absence of those fields is evidence only about this exact structure, not a universal proof that no external envelope can exist. The production reference-set ratchet and later E2/E4 work own that stronger architecture.

## Qualification contract

An exact-head qualification may establish only:

```text
schema=epi-sem-001e1-code-certificate-discovery-v1
authority_scope=measurement-only-content-integrity-verification-metadata-and-authentication-boundary
production_reference_semantics=conservative-file-level-symbol-membership
result=PASS_DISCOVERY
```

A PASS means only that the frozen source still matches the measured boundary.

## Explicit nonclaims

E1 does not establish:

- authenticated certificate issuer identity;
- qualified verifier identity;
- trustworthy verification-layer contents;
- re-execution of compiler/tests/formal proof;
- trusted time or ordering;
- anti-replay;
- key lineage;
- source authorship;
- generic correctness;
- security;
- fitness for purpose;
- deployment approval;
- action authority.

## Intended next steps

After E1 qualifies:

1. E2 should replace overloaded certificate semantics with a versioned, canonical content-bound verification/audit record.
2. E3 should bind positive compile/test/formal properties to source-native verifier receipts rather than caller-populated metadata.
3. E4 may then add a purpose-separated authenticated envelope using qualified Xenia primitives.
4. E5 should run the compatibility and adversarial corpus from #5392.

The order matters:

```text
precise payload semantics
    -> source-native verification evidence
    -> authenticated envelope
```

rather than signing an ambiguous object and accidentally authenticating overbroad claims.
