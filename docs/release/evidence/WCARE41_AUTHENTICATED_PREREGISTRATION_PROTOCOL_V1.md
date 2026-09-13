# WCARE-41 — Authenticated builder provenance and preregistration protocol v1

Status: `PREREGISTERED_PROTOCOL`
Authority: `MeasurementOnly`
Protocol version: `wcare41-authenticated-preregistration-v1`

## Purpose

WCARE-40 can qualify replication agreement and a conservative builder/fault-domain graph, but its provenance-strength labels and `plan_created_utc` remain claims inside evidence artifacts. WCARE-41 adds two orthogonal authentication questions:

1. **builder evidence authentication** — did a trusted issuer authenticate the exact builder provenance or pairwise relation receipt used by WCARE-40?
2. **temporal preregistration** — did an accepted external verifier establish that the exact WCARE-40 plan commitment existed before the qualifying replica executions began?

The governing distinctions are:

`valid signature != trusted issuer != independent builder != correct subject`

`self-declared timestamp != externally established preregistration`

`builder authentication != temporal preregistration`

Neither channel grants runtime authority.

## Frozen WCARE-40 subject

Every WCARE-41 evaluation binds the exact SHA-256 of:

- the WCARE-40 plan bytes;
- the WCARE-40 result bytes;
- the WCARE-40 front-door implementation;
- the WCARE-40 core verifier implementation.

WCARE-41 never rewrites WCARE-40 historical receipts. Authentication is an overlay that may preserve or reduce evidentiary weight.

## Builder attestation subjects

A builder attestation has exactly one subject kind:

- `BuilderProvenance` — authenticates one exact WCARE-40 builder provenance receipt;
- `BuilderRelation` — authenticates one exact WCARE-40 pairwise relationship receipt.

The attestation envelope binds:

- exact WCARE-40 plan SHA-256;
- optional exact WCARE-40 result SHA-256 (`-` when the attestation predates result production);
- subject kind;
- exact subject receipt SHA-256;
- exact strength claim copied from the subject receipt;
- issuer key ID and Ed25519 public key;
- exact issuer trust-policy SHA-256;
- issue/expiry timestamps;
- nonce commitment;
- domain separator.

A provenance attestation cannot authenticate a relation receipt and vice versa.

## Canonical builder attestation bytes

The canonical message is UTF-8, LF terminated, with these lines in exactly this order:

`SYMTHAEA-WCARE41-BUILDER-ATTESTATION-V1`
`protocol_version=wcare41-authenticated-preregistration-v1`
`wcare40_plan_sha256=<64-lower-hex>`
`wcare40_result_sha256=<64-lower-hex|->`
`subject_kind=<BuilderProvenance|BuilderRelation>`
`subject_receipt_sha256=<64-lower-hex>`
`provenance_strength_claim=<SelfDeclared|OrganizerVerified|ExternalVerified|InstitutionalAttestation|->`
`relation_evidence_strength_claim=<SelfDeclared|OrganizerAssessed|ExternalVerified|InstitutionalAttestation|->`
`issuer_key_id=<token>`
`issuer_public_key_ed25519_hex=<64-lower-hex>`
`issuer_policy_sha256=<64-lower-hex>`
`issued_at_utc=<YYYY-MM-DDTHH:MM:SSZ>`
`expires_at_utc=<YYYY-MM-DDTHH:MM:SSZ|->`
`nonce_sha256=<64-lower-hex>`
`domain=builder-evidence-attestation`

For `BuilderProvenance`, the provenance-strength claim is present and relation strength is `-`. For `BuilderRelation`, relation strength is present and provenance strength is `-`.

v1 uses Ed25519 and must reuse an audited implementation such as `ed25519-dalek`; no custom signature mathematics is permitted.

## Builder verifier boundary

WCARE-41 does not trust a precomputed `ATTESTATION_ACCEPTED` JSON value.

The WCARE-41 authentication plan binds an exact builder-verifier backend ID, implementation SHA-256, and trust-policy SHA-256. A qualifying WCARE-41 implementation must re-execute that exact verifier over the exact envelope, trust policy, WCARE-40 plan/result, and subject receipt.

Until such an exact verifier executes successfully, `builder_authentication_established` remains false.

Mock or synthetic backends may be used only by protocol self-tests and cannot produce a real authenticated-builder claim.

## Temporal preregistration proof

The exact WCARE-40 plan SHA-256 is the temporal commitment subject.

A temporal proof package binds:

- exact WCARE-40 plan SHA-256;
- verifier backend ID and implementation SHA-256;
- service/log identity commitment;
- proof artifact SHA-256;
- externally asserted commitment UTC;
- optional inclusion/index commitment;
- backend policy SHA-256;
- proof format/version.

The WCARE-41 plan binds the exact temporal verifier implementation and backend policy before evaluation.

A generic timestamp string, filesystem time, Git author/committer time, WCARE-40 `plan_created_utc`, or self-authored receipt is insufficient.

## Temporal verifier boundary

WCARE-41 does not trust a precomputed `ESTABLISHED` label.

A qualifying implementation must re-execute the exact plan-bound temporal verifier over the exact WCARE-40 plan and proof package. The verifier must independently establish the plan commitment and external time under its own backend policy.

The resulting externally evidenced commitment time must be strictly earlier than every qualifying WCARE-39 replica command start used by the WCARE-40 supported result.

If the backend cannot execute or the external time cannot be established, temporal status is indeterminate/not established rather than guessed.

## Authentication plan

The WCARE-41 plan binds:

- WCARE-40 plan/result hashes and verifier hashes;
- exact builder-verifier backend ID/hash/policy hash;
- exact temporal-verifier backend ID/hash/policy hash;
- evaluation UTC;
- whether complete builder attestation coverage is required;
- whether temporal preregistration is required for the target claim.

Changing a verifier implementation or policy creates a different WCARE-41 evaluation subject.

## Monotonicity

WCARE-41 may never increase WCARE-40 independence.

Let `I40` be WCARE-40 effective independent components and `I41` the count remaining after required builder receipts/relations are authenticated.

The invariant is:

`0 <= I41 <= I40`

Missing, untrusted, expired, invalid, or indeterminate attestations can collapse components or make authentication incomplete. They never create a new independent component.

Duplicate attestations cannot multiply weight.

## Orthogonal result dimensions

Builder authentication status is one of:

- `AUTHENTICATED`
- `PARTIAL`
- `UNAUTHENTICATED`
- `INDETERMINATE`
- `INVALID`

Temporal preregistration status is one of:

- `ESTABLISHED`
- `NOT_ESTABLISHED`
- `INDETERMINATE`
- `INVALID`

Either dimension may pass while the other fails.

`preregistration_temporal_precedence_established` is true only when the exact temporal verifier establishes an external plan commitment strictly preceding all qualifying replica starts.

`builder_authentication_established` is true only when the exact builder verifier authenticates the required builder evidence under the bound trust policy and the authenticated overlay still satisfies the plan's required coverage.

## Historical and replay semantics

Authentication never mutates WCARE-40 receipts.

Builder attestations are exact-subject artifacts. Replay across a different plan, result, provenance receipt, relation receipt, claim strength, issuer policy, or domain separator fails binding.

Key rotation/revocation preserves historical signature existence but cannot authorize new attestations after the applicable revocation boundary.

Temporal proofs are exact-plan artifacts. A proof for plan A cannot establish preregistration of plan B even if both plans describe similar campaigns.

## Claim boundary

WCARE-41 may support only bounded claims about authenticated builder evidence and externally evidenced plan precedence under explicit verifier/trust policies.

It does not establish reviewer independence, objective builder identity outside authenticated claims, subject correctness, network or sandbox isolation, consciousness, phenomenal experience, suffering, moral patienthood, objective moral truth, cultural universality, binding consent, veto/self-preservation authority, or solved alignment.

No WCARE-41 artifact grants live cognitive or action authority.
