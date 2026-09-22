# EPI-SEM-001E1-R2 — Code-certificate semantics discovery

Parent: #5392 (`EPI-SEM-001E`)

Supersedes the failed measurement subject #5393 / qualifier #5394.

Status: measurement only; no certificate semantics or production behavior changed.

## Why R2 exists

R1 used raw substring matching for exact witnesses. That made the measurement
incorrectly sensitive to source formatting.

The existing review document contains this prose across a line boundary:

```text
Certificates
are generated, held in an in-memory `Vec`, and discarded when the
```

while R1 required the equivalent prose as one contiguous single-space string.
Because R1 checked `needle in text`, line wrapping alone was sufficient to force
`REVIEW_REQUIRED`.

That is a measurement defect, not evidence that `CodeCertificate` semantics
changed.

R2 fixes only that defect. Witness matching now canonicalizes whitespace in both
the source text and the required witness before substring comparison.

```text
formatting-equivalent witness
!= semantic boundary change
```

The R2 matcher does **not** normalize or rewrite non-whitespace characters.

## Purpose

Freeze the current `CodeCertificate` boundary before changing it.

The current implementation combines several useful kinds of metadata in one
object:

- a BLAKE3 source commitment;
- backend/source-provenance descriptors;
- verification-layer metadata;
- internal epistemic-status metadata;
- a local issuance timestamp;
- optional topology/oracle/sheaf metadata.

The current documentation also describes the object as a "cryptographic receipt"
and as proving how code was generated and verified.

This tranche measures what the current object and its production consumers
actually establish.

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

The expected production reference set remains intentionally small:

```text
src/language/code_certificate.rs
src/language/code_orchestrator.rs
```

`CodeOrchestrator` creates the record after accepted code generation, copies
request epistemic status and verification-layer metadata into it, stores it in
an in-memory vector, and exposes a cloning accessor.

The repository's current coding-improvement plan notes that no production
file-write/API/CLI consumer reads these certificates back; tests and benchmarks
do inspect them.

Therefore E1 must not describe the current object as a deployed portable
credential. The current product boundary is closer to:

```text
in-process post-acceptance audit metadata
+ source-byte commitment/tamper detection
```

than to:

```text
authenticated portable verification credential
```

## Measurement program

`scripts/discover_code_certificate_semantics.py` performs three independent
classes of checks.

### 1. Conservative production reference discovery

It scans production Rust paths for `CodeCertificate` symbol membership and
emits:

- sorted reference paths;
- exact path-set digest;
- current `CodeCertificate` field inventory;
- any authentication-shaped fields newly appearing in the struct.

The expected production reference set and field list remain exact ratchets.
Any change requires review.

### 2. Whitespace-canonicalized witnesses

Required source witnesses cover:

- current cryptographic-receipt claim language;
- BLAKE3 source commitment and source-byte verification;
- JSON serialization;
- orchestrator construction/storage/accessor behavior;
- request epistemic-status copying;
- verification-layer copying;
- current post-acceptance-metadata wording;
- the repository's existing note that the certificate has no production
  persistence/API/CLI consumer.

For these witnesses only, R2 uses:

```text
canonicalize_whitespace(s) = " ".join(s.split())
```

before substring comparison.

This admits line wrapping and formatter-only whitespace changes. It does not
admit changed identifiers, punctuation, field names, function names, literals,
or claim language.

### 3. Authentication-field absence ratchet

The current `CodeCertificate` field set contains no dedicated
issuer-signature/authentication field.

E1 therefore fails for review if fields with names such as these appear later:

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

Absence of those fields does not establish a universal absence of external
authentication; it only freezes the current structure.

## Qualification contract

An exact-head qualification may establish only:

```text
schema=epi-sem-001e1-code-certificate-discovery-v2
authority_scope=measurement-only-content-integrity-verification-metadata-and-authentication-boundary
production_reference_semantics=conservative-file-level-symbol-membership
witness_matching=whitespace-canonicalized-substring
result=PASS_DISCOVERY
```

A PASS means only that the frozen source matches this measured boundary under
the explicitly declared R2 witness semantics.

## Explicit nonclaims

E1-R2 does not establish:

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

After E1-R2 qualifies:

1. E2 should replace overloaded certificate semantics with a versioned,
   canonical content-bound verification/audit record.
2. E3 should bind positive compile/test/formal properties to source-native
   verifier receipts rather than caller-populated metadata.
3. E4 may then add a purpose-separated authenticated envelope using qualified
   Xenia primitives.
4. E5 should run the compatibility and adversarial corpus from #5392.

The order remains:

```text
precise payload semantics
    -> source-native verification evidence
    -> authenticated envelope
```

rather than signing an ambiguous object and accidentally authenticating
overbroad claims.
