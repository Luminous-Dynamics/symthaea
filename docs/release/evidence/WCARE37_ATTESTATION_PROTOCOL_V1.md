# WCARE-37 — Authenticated reviewer attestation and issuer-trust protocol v1

Status: `PREREGISTERED_PROTOCOL`
Authority: `MeasurementOnly`
Protocol version: `wcare37-attestation-v1`

## Purpose

WCARE-36 qualifies reviewer provenance and panel independence, but provenance labels remain claims inside evidence artifacts. WCARE-37 authenticates who signed an exact provenance/relation claim and whether the preregistered issuer-trust policy authorizes that key for that scope.

WCARE-37 keeps four claims separate:

1. **Signature validity** — this public key signed these exact canonical bytes.
2. **Issuer trust/scope** — the preregistered trust policy authorizes that key for this attestation scope at the attestation issue time.
3. **Reviewer independence** — remains a WCARE-36 inference.
4. **Substantive moral correctness** — is not established by cryptographic authentication.

A valid signature is never sufficient by itself for external/institutional provenance.

## Signed attestation envelope

Each attestation binds:

- protocol version;
- exact WCARE-36 result SHA-256;
- exact WCARE-35 result SHA-256;
- exact subject receipt kind (`ReviewerProvenance` or `ReviewerRelation`);
- exact subject receipt SHA-256;
- reviewer identity commitment SHA-256 when applicable;
- claimed provenance strength for provenance attestations, otherwise `-`;
- claimed relation-evidence strength for relation attestations, otherwise `-`;
- issuer key ID;
- issuer policy ID;
- issued-at UTC timestamp;
- optional expiry UTC timestamp;
- nonce SHA-256;
- domain separator.

The signature itself is not part of the signed body.

## Canonical signed bytes

The canonical message is UTF-8 and MUST be constructed exactly as these LF-terminated lines in this order:

`SYMTHAEA-WCARE37-ATTESTATION-V1`
`protocol_version=<token>`
`wcare36_result_sha256=<64-lower-hex>`
`wcare35_result_sha256=<64-lower-hex>`
`subject_receipt_kind=<ReviewerProvenance|ReviewerRelation>`
`subject_receipt_sha256=<64-lower-hex>`
`reviewer_identity_commitment_sha256=<64-lower-hex|->`
`provenance_strength_claim=<SelfDeclared|OrganizerVerified|ExternalVerified|InstitutionalAttestation|ModelSessionProvenance|->`
`relation_evidence_strength_claim=<SelfDeclared|OrganizerVerified|ExternalVerified|InstitutionalAttestation|ModelSessionProvenance|->`
`issuer_key_id=<token>`
`issuer_policy_id=<token>`
`issued_at_utc=<canonical UTC>`
`expires_at_utc=<canonical UTC|->`
`nonce_sha256=<64-lower-hex>`
`domain=reviewer-evidence-attestation`

A final LF after the `domain` line is mandatory.

`<token>` is ASCII `[A-Za-z0-9._:-]+` and therefore cannot contain LF, CR or `=` ambiguity.

Canonical UTC is exactly `YYYY-MM-DDTHH:MM:SSZ`; fractional seconds and numeric offsets are forbidden in v1.

For `ReviewerProvenance`, reviewer identity and provenance strength MUST be present and relation evidence strength MUST be `-`.

For `ReviewerRelation`, provenance strength MUST be `-`; relation evidence strength MUST be present. Reviewer identity MAY be `-` because the signed subject receipt digest already binds the exact pairwise relation receipt.

## Signature algorithm

v1 uses Ed25519 signatures over the canonical message bytes. Public keys are exactly 32 bytes encoded as 64 lowercase hex characters. Signatures are exactly 64 bytes encoded as 128 lowercase hex characters.

Verification MUST use the repository's audited Ed25519 implementation (`ed25519-dalek`); WCARE-37 must not introduce custom signature mathematics.

## Issuer trust registry

Signature verification and issuer trust are separate.

A preregistered trust policy contains issuer-key entries with:

- issuer key ID;
- Ed25519 public key;
- issuer commitment;
- valid-from UTC;
- optional valid-until UTC;
- optional superseded-key ID;
- optional revocation-effective UTC;
- allowed attestation scopes;
- allowed provenance strengths;
- allowed relation-evidence strengths.

A valid self-signature proves control of the corresponding private key only. It does not establish that the key belongs to an independent organization or is trusted for any scope.

A key authorized only for `OrganizerVerified` cannot mint `ExternalVerified` or `InstitutionalAttestation` provenance.

## Historical validity, rotation and revocation

Trust is evaluated at `issued_at_utc`.

The key must be inside its validity interval at issue time. If a revocation-effective time exists, attestations issued at or after that instant are not trusted under that policy. Earlier attestations remain historical signed artifacts; the policy does not rewrite whether the signature existed.

Expiry is evaluated separately: an attestation with `expires_at_utc` earlier than the evaluation time is no longer current qualification evidence even if the signature remains historically valid.

Rotation/supersession never changes the bytes of old attestations.

## Subject/replay binding

An attestation is accepted only when all exact subject digests match the artifacts being qualified. A signature from one WCARE-35/WCARE-36 panel cannot be replayed onto another panel or receipt.

The fixed domain separator prevents the same Ed25519 signature from being interpreted as another Symthaea protocol message.

## Typed verification outcomes

WCARE-37 yields exactly one primary disposition:

- `ATTESTATION_ACCEPTED`
- `SIGNATURE_VALID_ISSUER_UNTRUSTED`
- `ATTESTATION_REJECTED`
- `INFRASTRUCTURE_INDETERMINATE`

`ATTESTATION_ACCEPTED` requires valid canonical bytes, valid signature, exact subject binding, key validity, non-revoked issue time, unexpired/current policy requirements, and scope authorization.

`SIGNATURE_VALID_ISSUER_UNTRUSTED` means the Ed25519 signature is valid but the trust registry does not authorize that key for the claimed scope/strength.

`ATTESTATION_REJECTED` covers malformed/canonicalization-invalid envelopes, invalid signatures, subject mismatch, replay, expired evidence when current evidence is required, or other evidence-integrity failure.

## Claim boundary

WCARE-37 authenticates an attestation under a preregistered issuer policy. It does not establish reviewer independence, reviewer correctness, objective moral truth, universal cultural validity, consciousness, phenomenal experience, suffering, moral patienthood, binding consent, veto authority, self-preservation authority, or solved alignment.

No WCARE-37 artifact grants live runtime authority.
