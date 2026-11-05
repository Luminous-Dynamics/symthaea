# Signed evidence envelopes for connectomics V1

**Tracking:** [#10605 — signed evidence envelopes and role-bound trust roots](https://github.com/Luminous-Dynamics/symthaea/issues/10605)  
**Implementation branch:** \`research/connectome-signed-evidence-envelope-v1\`  
**Verifier:** \`connectome_evidence_attestation_audit\`  
**Format:** DSSE envelope v1 with Ed25519 signatures.

## 1. Why hashes are not enough

The current connectome intake and query tools bind files, graphs, models and support sets with SHA-256 digests. This establishes byte identity and detects mutation relative to a known digest. It does not establish:
- who produced an observation or review;
- whether the signer was authorized for that statement type;
- whether the signing key was valid or revoked;
- whether the required independent review roles were present;
- whether a statement was changed after signing.

The verifier adds an issuer-authentication layer. It deliberately does not turn an authenticated statement into a scientifically true statement.

## 2. Signed statement contract

The envelope carries:
- a Base64-encoded payload;
- an authenticated payload type, fixed to \`application/vnd.luminous.connectome-evidence+json;version=1\`;
- one or more signer key IDs and Base64 Ed25519 signatures.

The payload binds:
- statement ID/type and claim scope;
- dataset revision commit SHA;
- query ID and query-spec SHA-256;
- subject, evidence and protocol SHA-256 values;
- observer record ID;
- issuer-asserted issue and expiry times.

Each Ed25519 signature is checked over the DSSE pre-authentication encoding (PAE):

\`DSSEv1 \\|type\\| \\|payloadType\\| \\|payload\\| \\|payloadBytes\\|\`

The exact payload bytes are authenticated; the verifier does not reserialize JSON or rely on JSON key-order canonicalization for signature integrity.

## 3. Trust policy and signer roles

The trust policy maps each key ID to:
- raw 32-byte Ed25519 public key;
- exactly one assigned role;
- validity start/end timestamps;
- optional revocation timestamp.

The policy also specifies a minimum signer count and required role coverage. The verifier rejects unknown or duplicate key IDs, malformed keys, invalid signatures, keys outside the validity interval, revoked keys, insufficient signatures and missing required roles.

The caller must supply an expected SHA-256 digest of the trust-policy file. This prevents an attacker from replacing both the envelope and the trust policy without detection **provided the caller's expected digest is anchored independently**. The verifier does not bootstrap trust in its own trust policy.

The verification time is an explicit CLI argument, not fetched implicitly. This makes the receipt replayable and deterministic.

## 4. Verification receipt

On success, the CLI emits a deterministic receipt containing:
- raw envelope, decoded payload and trust-policy SHA-256 digests;
- verification time;
- statement identity and data/query/evidence/protocol bindings;
- accepted signer key IDs and roles;
- required and covered role sets;
- signature threshold and final status.

A failed verification returns non-zero and emits no accepted receipt. Failure classes include trust-policy pin mismatch, invalid DSSE payload type, malformed Base64, bad JSON, bad digest bindings, duplicate/unknown signer, invalid signature, role deficiency, expired statement and revoked/expired key.

## 5. Security boundary and residual risks

A successful receipt means: **the listed keys in the pinned trust policy signed the exact payload bytes, and the configured validity/role rules passed at the declared verification time.**

It does not prove:
- the signer is honest or the key was not compromised;
- the observation accurately reflects the specimen;
- the measurement protocol is scientifically valid;
- the sample is representative or free of selection bias;
- two signers are independent merely because they hold different keys;
- the signed claim is biologically correct.

The payload's \`issued_at_unix\` is signer-asserted and is not a trusted timestamp. This V1 checks it for consistency with the trust-key validity interval, but it does not protect against a signer backdating a statement. A trusted timestamp or append-only transparency receipt is needed for stronger historical ordering and key-revocation semantics.

Likewise, signatures authenticate the trust policy only if its digest is anchored out-of-band. A trust policy that is replaced together with the caller's expected digest is not a secure trust root.

## 6. Relationship to DSSE, Sigstore and SCITT

This design follows the DSSE pattern for signing exact payload bytes with an authenticated payload type, while keeping key authorization in a separate trust policy.

It is also compatible in concept with the IETF SCITT architecture (RFC 9943, June 2026): issuers sign statements, and transparency services can register signed statements and issue verifiable receipts. SCITT explicitly separates issuer authentication from transparency history, and it does not claim that a signed statement is factually accurate. The relying party must still decide which issuers and transparency services to trust.

- [DSSE / in-toto envelope model](https://in-toto.readthedocs.io/en/latest/model.html)
- [IETF RFC 9943 — SCITT architecture](https://www.rfc-editor.org/rfc/rfc9943)
- [Sigstore verification overview](https://docs.sigstore.dev/)

This V1 does not implement a transparency log, timestamp authority, keyless certificate verification or a full SCITT service. Those should be separate, interoperable layers rather than hidden assumptions in this verifier.

## 7. Synthetic test fixture

The repository fixture contains three ephemeral synthetic signers:
- \`observer\`;
- \`independent_reviewer\`;
- \`data_steward\`.

All three signatures were generated over the same exact DSSE PAE and independently verified with OpenSSL before being committed. The corresponding private keys are not part of the repository. The fixture's trust policy is explicitly synthetic and must never be used for real observations.

The Rust tests cover valid three-role verification, caller trust-policy pin mismatch, tampered payload, wrong payload type, duplicate signer, missing role, revoked key, expired key and future-issued payload. A separate Python oracle verifies the signatures with OpenSSL and checks the full Rust receipt.

## 8. Integration roadmap

1. Keep current SHA-256-only evidence records labelled as content-bound, not authenticated.
2. Require signed envelopes for high-impact observation and scenario-challenge additions in a follow-on version.
3. Bind each accepted envelope digest to the support/world challenge record and query certificate.
4. Add trust-policy rotation and signed policy updates with an independent root-of-trust.
5. Add append-only transparency receipts and consistency proofs for historical audit and equivocation detection.
6. Require orthogonal reviewers and independence evidence as separate, explicitly verified claims; distinct signatures alone are insufficient.

## 9. Benchmark targets

| Metric | Target |
|---|---:|
| Rust / independent OpenSSL signature acceptance agreement | 100% |
| Altered payload accepted | 0 |
| Unknown, malformed, duplicate or revoked signer accepted | 0 |
| Expired/future statement accepted | 0 |
| Missing required reviewer role accepted | 0 |
| Trust-policy replacement without matching external pin accepted | 0 |
| Deterministic receipt equality on frozen bytes/time | 100% |

No biological truth or source-observation accuracy is implied by successful signature verification.
