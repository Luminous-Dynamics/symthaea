# Replicator Safety Kernel — Xenia Witness Trust Context v0.1

**Status:** Reference trust-composition contract; not production-admitted  
**Change Class:** A  
**Production admission:** DENIED / NOT YET ELIGIBLE

## Purpose

Xenia may prove that an exact state commitment was signed by a configured set of
trusted cryptographic keys. RSK additionally requires governed signer identity,
key lifecycle, role, failure-domain independence, and exact trust-snapshot
binding before that witness evidence may contribute to the external monotonic
continuity predicate.

This contract defines that higher-level trust context. It does not grant or
restore replication authority.

## Governing distinction

The following are separate predicates:

```text
verified cryptographic key quorum
!=
verified signer-identity quorum
!=
verified failure-domain quorum
```

Multiple valid keys belonging to one signer count as multiple keys but one
signer. Multiple valid signers controlled by one failure domain count as multiple
signers but one failure domain.

## Inputs

The reference verifier consumes only:

1. exact key bindings that Xenia has already cryptographically verified;
2. an independently authenticated RSK trust snapshot mapping each key to:
   - `key_id`;
   - `signer_id`;
   - role;
   - failure domain;
   - signature profile;
   - lifecycle;
   - validity interval;
3. the exact RSK monotonic-anchor policy;
4. signer-lifecycle and failure-domain policy identities;
5. the exact trust-snapshot identity;
6. the RSK trusted-time interval.

No unsigned witness-bundle field may manufacture signer identity or failure-domain
identity.

## Xenia key identity

The admitted key binding uses Xenia's existing evidence-key identity:

```text
fingerprint_algorithm = "blake3-256"
public_key_fingerprint = BLAKE3(raw_public_key)
```

The signature suite/profile is carried separately.

RSK does not recompute BLAKE3 in the reference Python layer. The future production
adapter must consume an opaque Xenia verified-key type or equivalently verified
key-binding evidence.

## Canonical trust context

The v0.1 trust-context object binds:

- schema/version;
- exact monotonic-anchor-policy digest;
- admitted Xenia state-witness profile;
- Xenia key-fingerprint algorithm;
- minimum distinct-key quorum;
- minimum distinct-signer quorum;
- minimum independent-failure-domain quorum;
- allowed signature profiles;
- signer-lifecycle policy digest;
- failure-domain policy digest;
- trust-snapshot digest;
- complete sorted key→signer→role→failure-domain mappings, including lifecycle
  and validity intervals.

Its identity is:

```text
TrustContextDigest = SHA256(
    "symthaea.rsk.xenia-witness-trust-context.v1\0"
    || canonical_json(trust_context)
)
```

The digest is derived from verified policy state. A digest copied from a Xenia
bundle or remote service is not self-authenticating.

## Structural integrity

A trust context is invalid if any of the following occurs:

- one `key_id` maps to multiple signer identities;
- one raw-key fingerprint maps to multiple `key_id` values;
- one signer has conflicting role/failure-domain metadata;
- principal ordering is noncanonical;
- signature profile is not policy-allowed;
- lifecycle is unknown;
- validity interval is malformed;
- configured active principals cannot structurally satisfy all required quorums;
- trust-snapshot identity differs from the monotonic-anchor policy.

## Live witness evaluation

For each exact Xenia-verified key:

1. the fingerprint must appear exactly once in the authenticated trust context;
2. its signature profile must match the trusted mapping;
3. the principal must be active;
4. the principal must be valid for the full RSK trusted-time interval;
5. duplicate verified keys are rejected;
6. the verifier accumulates **distinct** key IDs, signer IDs, and failure domains.

Admission requires all three thresholds independently:

```text
unique(key_id)          >= minimum_key_quorum
unique(signer_id)       >= minimum_signer_identities
unique(failure_domain)  >= minimum_failure_domains
```

Failure or uncertainty freezes the external-continuity predicate. It does not
quarantine, revoke, destroy, or repair state by itself.

## Composition with Xenia state witness

The exact `TrustContextDigest` becomes the expected
`StateCommitment.trust_context_digest` for the Xenia commitment. Therefore a
valid state signature gathered under one trust configuration cannot be silently
reinterpreted under another signer set, failure-domain policy, lifecycle policy,
or trust snapshot.

Changing the trust context during one ordinary witness epoch is not transparent
continuity. It requires the governed trust/epoch-transition semantics defined by
RSK recovery policy.

## Non-amplification

A successfully verified witness trust context can prove only that the external
witness evidence satisfies the configured key/signer/failure-domain policy. It
cannot:

- mint a replication grant;
- satisfy an unrelated replication-approval quorum;
- clear quarantine or revocation;
- extend expiry;
- make stale evidence fresh;
- replace the local registry state;
- repair a fork;
- reset the monotonic epoch;
- substitute for trusted time or runtime admission.

## Reference vectors

`golden/RSK_XENIA_WITNESS_TRUST_CONTEXT_GOLDEN_V0_1.json` freezes an abstract
three-key / two-signer / two-failure-domain case. The fingerprints are synthetic
test identities; they are not production keys.

The corresponding reference implementation is
`reference/rsk_xenia_witness_trust.py` and its adversarial test matrix is
`reference/test_rsk_xenia_witness_trust.py`.

## Promotion blockers

Production remains blocked until at least:

- Xenia's verified-key evidence path has executed Rust test/Clippy evidence;
- RSK consumes an opaque verified Xenia-key type rather than freely constructible
  Python reference records;
- the authenticated trust-snapshot verifier and key lifecycle source are bound;
- failure-domain metadata is independently governed and auditable;
- trusted-time integration is real;
- the trust-context digest is integrated into exact Xenia commitment vectors;
- retained/CAS witness storage and crash recovery are qualified;
- exact-head Class A CI is green.
