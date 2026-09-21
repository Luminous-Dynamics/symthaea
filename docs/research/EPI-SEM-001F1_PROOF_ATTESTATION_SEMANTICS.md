# EPI-SEM-001F1 — ProofAttestation trust-boundary discovery

Parent: #5395 (`EPI-SEM-001F`)

Status: measurement only; no signing, verification, proof, or key-handling behavior changed.

## Purpose

Freeze the exact current semantics of `ProofAttestation` and `SovereignAttestor` before hardening them.

The current mechanism is valuable cryptographic evidence, but it mixes several distinct properties that must remain separate:

```text
signature validity under one key
subject commitment
artifact commitment
signer identity / trust admission
proof-engine verdict
portable verification authority
```

E1 measures the current relationships without promoting any of them.

## Current record

The frozen field inventory is:

```text
label
smtlib2_hash
binary_hash?
verdict
signature
public_key
```

The record is serializable/deserializable data. That is not itself a verified positive typestate.

## Current signing path

The current signed bytes are constructed as:

```text
label bytes
|| SHA-256(SMT-LIB2 bytes)
|| optional SHA-256(binary bytes)
|| verdict bytes
```

and signed by the process `DilithiumKeypair`.

The current transcript has no explicit application-domain separator, schema version, proof-profile identity, or canonical length-delimited envelope.

E1 does not claim a practical collision. It records the current protocol shape so a later migration can be explicit.

## Current signer-key behavior

`from_env_or_generate()` attempts to load:

```text
SYMTHAEA_ATTESTOR_PUBLIC_KEY_HEX
SYMTHAEA_ATTESTOR_SECRET_KEY_HEX
```

When those values are absent or invalid, construction falls back to a freshly generated keypair.

Therefore:

```text
process attestor exists
!= stable provisioned signer identity
```

An explicitly ephemeral local signer can remain useful, but a future profile claiming stable issuer identity must not silently fall back to a new key.

## Current verification boundary

`SovereignAttestor::verify()` currently receives:

```text
ProofAttestation
actual_binary?
```

It does not receive actual SMT-LIB2 bytes for digest recomputation.

It reconstructs the signed message from the fields already present in the attestation and verifies the signature against the public key carried by the same record.

Therefore:

```text
valid signature
!= admitted signer identity
```

and:

```text
stored smtlib2_hash signed
!= supplied SMT subject independently rebound
```

### Optional binary check

When both `attestation.binary_hash` and `actual_binary` are present, the current verifier recomputes the binary SHA-256 and rejects mismatch.

When `binary_hash` is present but `actual_binary` is absent, the current function does not reject solely for that absence; signature verification continues.

Therefore a true return must not be interpreted as universally meaning:

```text
attested binary realization checked
```

## Current production call graph

The expected production `ProofAttestation` symbol-reference set is:

```text
src/language/sovereign_attestor.rs
src/language/verified_generation.rs
```

`verified_generation.rs` emits process-key attestations after its proof engine reaches a `Proven` result and stores them in `VerifiedCode.attestation`.

At E1 freeze time, the measurement expects no external production call site for:

```text
SovereignAttestor::verify(...)
```

If one appears later, E1 must fail for review rather than silently assuming the consumer grants no authority.

## Measurement program

`scripts/discover_proof_attestation_semantics.py` freezes:

- production symbol-reference paths and path-set digest;
- exact `ProofAttestation` field inventory;
- any production `SovereignAttestor::verify(...)` consumers outside the defining module;
- environment-key loading and fallback behavior;
- exact signed-message construction;
- binary-verification condition;
- embedded-key signature verification;
- the verified-generation producer relationship.

## Qualification contract

A future exact-head qualifier may establish only:

```text
schema=epi-sem-001f1-proof-attestation-discovery-v1
authority_scope=measurement-only-proof-subject-binding-signer-trust-and-verifier-consumer-boundary
production_reference_semantics=conservative-file-level-symbol-membership
result=PASS_DISCOVERY
```

## Nonclaims

E1 establishes none of the following:

- trusted signer identity;
- stable key lineage;
- application-context separation;
- exact SMT subject re-verification;
- mandatory binary realization checking;
- trusted chronology;
- replay resistance;
- proof completeness;
- correctness outside the formalized property;
- safe deployment;
- action authority.

## Next architecture

After F1 qualifies, the intended sequence is:

```text
F2 canonical purpose-separated formal-proof subject
    ↓
F3 source-bound verifier + opaque verified typestate
    ↓
F4 Xenia-qualified signer identity/key lineage
    ↓
F5 compatibility + hostile corpus
```

The key design rule is:

```text
cryptographic validity
+ exact subject binding
+ signer admission
```

must remain separate inputs, even when one higher-level policy requires all three.
