# Symthaea ↔ Mycelix Neurosemantic Authority Bridge

Status: integration contract / fail-closed boundary

## Purpose

Symthaea owns the neurosemantic data-class, inference-class, handling-policy, and packet-integrity semantics.

Mycelix (or another explicitly designated policy authority) owns external identity, issuer trust, signed-consent resolution, revocation/status resolution, audit history, and policy applicability.

This document defines the minimum evidence that must exist before an externally attested handling policy is treated as authoritative.

## Authority verification contract

A Symthaea `NeurosemanticPolicyAuthorityAttestation` proves a cryptographic signature over:

- the authority reference;
- the authority key reference;
- the exact Symthaea handling-policy fingerprint;
- the exact Symthaea external-provenance digest;
- the attestation validity interval.

Symthaea verifies the signature and the exact policy/provenance bindings.

That signature is **not** sufficient to establish that the signer is an authorized Mycelix authority.

The external resolver MUST additionally establish:

1. **Authority identity** — `authority_ref` resolves to an authorized issuer/controller identity.
2. **Key binding** — `key_ref` resolves to the public key actually authorized for that authority and purpose.
3. **Signature validity** — the attestation signature verifies against that resolved key.
4. **Temporal validity** — the attestation is valid at the authorization time.
5. **Status validity** — the attestation/policy record is not revoked or suspended according to the authoritative status mechanism.
6. **Policy semantics** — the external record identified by `policy_provenance_ref` actually represents the policy semantics whose fingerprint was attested.
7. **Consent linkage** — where the external record represents consent rather than a reusable organizational policy, it is linked to the correct subject, peer, purpose, direction, and consent epoch outside the Symthaea transport layer.
8. **Resolution freshness** — cached identity/status information is fresh enough for the policy's declared risk class; stale or unavailable authoritative status MUST NOT silently become "active".
9. **Auditability** — the resolver can retain or reproduce the authority/status resolution evidence needed to explain why the policy was accepted.

## Fail-closed requirements

The bridge MUST reject, rather than downgrade, the following states:

- unknown authority;
- unknown key reference;
- unverifiable signature;
- expired or not-yet-valid attestation;
- revoked/suspended attestation or policy;
- missing policy semantics;
- stale status beyond the declared freshness bound;
- unavailable authoritative status when status is required;
- ambiguous subject/consent linkage;
- a policy record whose semantics do not match the attested Symthaea fingerprint.

A local cache hit is evidence of previously observed state, not proof of current authority.

A successful cryptographic signature check is evidence of signer control of a key, not proof that the key is authorized by the Mycelix governance layer.

## Concrete signed resolution contract

The communication crate now materializes the recommended resolution as `NeurosemanticAuthorityResolutionAttestation`.

The resolution is signed by the configured external resolver and binds, in one immutable snapshot:

- the external resolver and resolver key reference;
- the authority and authority-key references;
- the exact fingerprint of the authority attestation artifact being resolved;
- the exact Symthaea handling-policy fingerprint;
- the exact external policy-provenance reference and digest;
- the exact subject, peer, lease identifier, and consent epoch;
- the content hash of the complete consent lease;
- the communication purpose, channel, and direction;
- an explicit authority status;
- the status source reference;
- the status checked-at time and a bounded freshness expiry.

Only `Active` status can produce a `NeurosemanticPolicyProvenanceBinding`. `Suspended`, `Revoked`, `Unknown`, and `Unavailable` are explicit fail-closed states.

The resolution lifetime is bounded to 24 hours by the Symthaea protocol as a defensive upper bound. Deployments handling higher-risk neurosemantic data should use a materially shorter freshness window.

The resolver signature authenticates the exact snapshot to the configured resolver key. The snapshot also commits to the exact authority-attestation fingerprint, preventing an otherwise valid but different attestation from being substituted after resolution. The integration must still establish that the resolver key is trusted and authorized; signature verification is not itself a governance decision.

A resolution cannot outlive the authority attestation that it resolves. This prevents a freshness snapshot from extending an older issuer proof beyond its cryptographic validity window. The status-check timestamp must also not predate the attestation issuance time; a resolver cannot use a later-discovered authority proof to retroactively justify an earlier resolution.

The resulting capability retains the exact resolution fingerprint and its context. Because the resolution also commits to the complete consent-lease fingerprint, changing scopes, sensitivity ceilings, data/inference permissions, validity, revocation state, or other lease fields invalidates an older capability even when an implementation accidentally reuses the same lease ID and epoch. Subsequent handling therefore fails closed when the resolution is expired, non-active, or bound to a different subject, peer, lease, consent state, purpose, channel, or direction.

## Recommended resolution object

The integration layer should expose a machine-readable resolution result containing at minimum:

- authority reference;
- key reference;
- resolved identity/reference version;
- policy provenance reference;
- external record digest;
- Symthaea handling-policy fingerprint;
- attestation fingerprint;
- signature verification result;
- issuer/key trust result;
- status state and status-source reference;
- status checked-at timestamp;
- maximum acceptable status age;
- consent/subject linkage result;
- resolution execution revision or content-addressed evidence identity.

Symthaea can then construct its handling capability from the exact resolved state without importing Mycelix's identity ontology into the communication crate.

## Revocation/status integration

W3C Bitstring Status List v1.0 is a Recommendation for privacy-preserving status publication and explicitly covers credential suspension/revocation. The current W3C family also has a Bitstring Status List v1.1 working draft and VC Data Model v2.1 draft work.

The Symthaea boundary should therefore treat status as a first-class external assertion rather than encoding a local "verified" boolean into the neurosemantic protocol.

The status mechanism is deliberately left abstract here so Mycelix can use its Holochain/DHT governance and identity mechanisms without forcing Symthaea to depend on one status-list representation.

## No authority laundering

The following transformations are prohibited:

`packet_hash valid -> policy trusted`

`signature valid -> issuer trusted`

`issuer trusted -> consent granted`

`consent granted -> every inference authorized`

`transmit allowed -> persist allowed`

`policy valid -> legal compliance established`

Each implication requires its own evidence boundary.

## Relationship to Symthaea N0/N1

N0 can establish deterministic protocol properties such as packet integrity, consent-scope enforcement, handling-policy enforcement, provenance binding, replay behavior, and stale-capability rejection.

N0 must not be interpreted as evidence that an external Mycelix authority is legitimate.

N1 should bind the exact authority-resolution evidence identity into the experiment provenance when neurosemantic decoding claims eventually depend on real external credentials or consent records.

## Research alignment

W3C Verifiable Credentials 2.0 defines a cryptographically secure, machine-verifiable credential ecosystem with issuer, holder, and verifier roles. The W3C Data Integrity work likewise separates cryptographic proof from the surrounding trust model.

For neurotechnology, this separation is especially important because authorization must remain granular across raw signals, derived features, decoded content, inference classes, secondary uses, retention, and downstream jurisdiction.

The bridge is therefore deliberately a **trust composition contract**, not a claim that a cryptographic signature by itself establishes social, legal, clinical, or governance legitimacy.
