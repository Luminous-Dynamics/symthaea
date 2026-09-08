# Safety Profile Authority Root Contract v1

This document is normative for `SafetyProfileAuthorizationSubject` schema
`symthaea-safety-profile-authorization-v1`.

## Purpose

The authorization subject identifies the **already externally trusted** authority
that is permitted to authorize one exact safety profile for one exact node. The
subject is evidence to be authenticated; it is not a mechanism for establishing
or rotating its own trust root.

## V1 signature suite

The required authorization signature suite is:

`ml-dsa-65-fips204`

A verifier MUST reject an authorization that verifies only under a different
signature suite, even if the signature and supplied public key are otherwise
self-consistent.

## Authority-root fingerprint

For v1, `ProfileAuthorityRootDigest::Blake3_256(bytes)` has one exact meaning:

`bytes = BLAKE3-256(raw_verifier_public_key_bytes)`

where `raw_verifier_public_key_bytes` is the canonical 1952-byte FIPS 204
ML-DSA-65 verifying-key encoding consumed by the verifier backend.

The BLAKE3 input is **only** those raw 1952 key bytes. It MUST NOT contain:

- a schema string,
- an algorithm label,
- a length prefix,
- a domain separator,
- JSON/CBOR/bincode serialization,
- an Xenia key-binding structure,
- an Ed25519 key,
- a hybrid identity fingerprint,
- or any other metadata.

This definition is intentionally byte-identical to Xenia's
`compute_evidence_public_key_fingerprint(public_key)` and
`EvidencePublicKeyBinding::public_key_fingerprint` when `public_key` is the raw
ML-DSA-65 verifier key.

## Canonical subject encoding

The existing v1 canonical signing bytes remain unchanged. The authority-root
fingerprint is encoded as:

1. digest algorithm tag `0x01`, followed by
2. the 32-byte BLAKE3-256 fingerprint defined above.

Changing this semantic interpretation requires a new authorization schema
version; it MUST NOT silently change under v1.

## Trust-source invariant

The following values are verifier policy inputs and MUST originate from trusted
provisioning or a separately verified authority-transition mechanism:

- required signature suite,
- trusted authority-root fingerprint,
- subject node identity or permitted node scope.

They MUST NOT be inferred solely from the signed authorization artifact being
verified.

In particular, `authority_root_id` is an audit/administrative identifier. It is
not itself a trust anchor.

## No self-authorization

A `SafetyProfileAuthorizationSubject` MAY bind the fingerprint of the root that
is expected to authenticate it, but successful verification does not authorize a
new root.

Bootstrap and authority rotation require a separate transition whose authority
comes from an already trusted predecessor or an explicit out-of-band bootstrap
policy. A candidate new root MUST NOT become trusted merely because it signs an
authorization subject containing its own fingerprint.

Conceptually:

`externally trusted root -> verify subject -> profile authorization`

is valid, while:

`candidate root -> subject names candidate root -> candidate verifies itself`

is invalid.

## Cross-project verifier agreement

A conforming Xenia verifier for this contract must therefore use:

- `SignatureSuite::MlDsa65Fips204`,
- the externally provisioned 32-byte root fingerprint as
  `expected_public_key_fingerprint`,
- the exact `SafetyProfileAuthorizationSubject::canonical_signing_bytes()` as
  the detached message,
- and the ML-DSA-65 evidence-signature backend.

The resulting opaque proof is evidence that the exact subject bytes were
authenticated by the exact externally trusted root. Higher-level Symthaea policy
must still check node, generation, validity, profile identity, and monotone
admission before the authorization can affect executable safety configuration.
