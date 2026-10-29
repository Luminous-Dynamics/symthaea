# Aviation Evidence Bundle Protocol (AEBP-001)

**Status:** Initial implementation committed on a research branch; compile/test execution pending.  
**Implementation:** aviation_evidence_bundle.rs in symthaea-operations-research.  
**Tracking:** AVIATION-ENTERPRISE-013, issue #8840.  
**Claim ceiling:** evidence integrity and authenticity checks only. This protocol does not provide dispatch, maintenance-release, airworthiness, or aircraft-control authority.

## Trust model

The readiness function needs to know not only that specialist reports exist, but that the reports are the exact artifacts produced for this run by expected verifiers. A matching ID or internally consistent JSON is not an authenticity guarantee.

The caller supplies an independently provisioned trust store and an ExpectedSourceIdentity obtained from pinned build metadata outside the evidence bundle. The verifier requires the manifest commit and source-tree digest to match those expected values; copying them from the untrusted manifest defeats this check. Evidence cannot add its own trusted key. Each trust anchor pins a key ID, permitted role, verifier identity, verifier-binary SHA-256, public key, validity interval, maximum receipt lifetime, and revocation state. Production key provisioning, rotation, revocation, and incident response remain separate operational responsibilities.

## Manifest binding

The immutable version-1 manifest binds:
- schema version and run ID;
- creation and decision timestamps;
- source commit and source-tree digest;
- data snapshot and scenario digests;
- model and policy/configuration digests;
- candidate plan digest;
- calibration artifact and drift-monitor report digests;
- plan-verification, distributional-risk, and network-harm report digests.

SHA-256 digests are 64 lowercase hexadecimal characters. Git source commit IDs may use 40-character SHA-1 or 64-character SHA-256 object IDs. Invalid encodings block verification.

Canonical bytes are deliberately independent of JSON whitespace and object-key ordering. Version 1 uses fixed field order and the record form:

    field-name:UTF-8-byte-length:value\n

The manifest domain string is luminous-dynamics.aviation-evidence-manifest.v1. The receipt-signature domain string is luminous-dynamics.aviation-verifier-receipt.v1. Field lengths count UTF-8 bytes. Domain separation and canonical field ordering are part of the protocol contract and must change only with a protocol-version change.

## Required receipt roles

Exactly one receipt is required for each of these roles:
1. calibration;
2. drift_monitor;
3. plan_feasibility;
4. distributional_risk;
5. network_harm.

Each Ed25519 receipt signs its schema version, receipt ID, role, exact manifest digest, role-specific subject digest, verifier identity, verifier-binary digest, key ID, issue time, and expiry time. The signature field is not included in the signed payload.

Verification fails closed on unsupported schema, unknown role, missing or duplicate role, duplicate receipt ID, malformed digest/signature, manifest or subject mismatch, issue time before manifest creation, future issue time, expired receipt, excessive lifetime, stale manifest or decision snapshot, zero freshness limits, inconsistent manifest/decision chronology, unknown/duplicate/revoked/out-of-validity trust anchor, verifier-role/identity/binary mismatch, or invalid signature. Maximum manifest age and maximum decision age are supplied by explicit verification policy; they are never chosen by the untrusted manifest. Blockers are sorted deterministically. Input structures reject unknown JSON fields when deserialized through their strict serde types.

## What Verified means

AEBP-001 reports Verified only when the manifest and all five role receipts pass the protocol's binding, freshness, trust-identity, and signature checks. Otherwise it reports Blocked and a complete blocker list.

Verified is not the readiness gate's EligibleForHumanReview state. The evidence bundle proves only that required artifacts are cryptographically anchored to the supplied trust store. It does not execute the domain verifiers, parse their conclusions, or establish that a schedule is feasible or low-risk. Integration with the existing readiness gate must remain a separate step that binds the current candidate and calibration to this exact manifest and requires both the evidence check and all domain gates to pass.

## Test coverage defined

The module has unit tests for valid signing across all required roles, deterministic canonical bytes, manifest mutation, receipt-order independence, missing/duplicate roles, expired and future timestamps, pre-manifest receipt issuance, stale manifest/decision snapshots, zero freshness limits, invalid manifest/decision chronology, external source identity mismatch, changed verifier binary, unknown key, revoked key, modified signature, and subject-digest mismatch.

A separately compiled integration oracle independently reconstructs manifest and receipt bytes, computes SHA-256 using its own implementation path, creates deterministic test signatures, compares canonical bytes, checks order independence, mutates every manifest identity/artifact digest to ensure prior receipts fail, verifies externally pinned source identity, and rejects unknown manifest/receipt/trust-anchor fields.

These are test definitions, not passing evidence: they must execute against the exact commit in a Rust-equipped environment. Fixture keys are test-only and must never become production trust anchors.

## Known gaps and follow-up

- The verifier is not yet wired into the current readiness gate or the synthetic run manifest.
- No independent production verifier binary or production trust-store provisioning has been delivered.
- The ingress layer still needs explicit input-size and resource limits.
- Add mutation tests across every manifest, receipt, trust-anchor, and time-boundary field.
- Run formatting, locked package tests, example execution, and independent JSON parsing; record the exact tested SHA and artifact digests.
- Do not interpret this protocol's Verified result as operational permission or certification.
