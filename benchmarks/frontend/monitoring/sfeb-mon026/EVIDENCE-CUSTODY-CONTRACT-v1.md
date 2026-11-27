# MON-026 — Evidence Custody and Reproducibility Envelope Contract v1

**Profile:** `mon026-evidence-custody-v1`  
**Status:** deterministic manifest reference and offline verifier implemented; MON-025 adapter, key lifecycle and independent custody operations pending.

## 1. Objective

Make a collected evidence bundle portable and independently verifiable without trusting the collector's UI, current CI system, recovery orchestrator or online vendor service.

MON-026 verifies:
1. the declared artifact inventory is well-formed;
2. each artifact exists beneath the bundle root and is not reached through a symlink;
3. every artifact's recorded size and SHA-256 match the observed bytes;
4. the manifest root matches a deterministic encoding of its semantic fields;
5. an independently trusted Ed25519 public key verifies the signature;
6. the run's predecessor link, sequence and source boundary satisfy the verifier's policy;
7. capture freshness and clock uncertainty remain inside declared bounds.

## 2. Manifest fields

The v1 model includes:
- profile/version, run ID and monotonically increasing bundle sequence;
- previous manifest root (empty only for sequence 1);
- capture time, clock domain and clock uncertainty in nanoseconds;
- a source boundary: incident, source cluster, database, system identifier, timeline, snapshot, capture LSN, scope digest and inventory digest;
- artifacts: relative path, media type, evidence role, producer, command ID, byte size, SHA-256, collection time, clock domain and clock uncertainty;
- semantic root SHA-256;
- signature algorithm, signer key ID and base64 Ed25519 signature.

Secrets should not be placed in manifests. Redact credentials and record the redaction operation as its own artifact or metadata entry. Never silently overwrite original evidence; create a new bundle sequence with an explicit supersession link.

## 3. Deterministic root

The root is SHA-256 over a domain-separated, length-prefixed binary encoding:
- fixed field order for manifest metadata and source boundary;
- artifact entries sorted lexicographically by normalized relative path;
- fixed field order for each artifact.

JSON serialization bytes are not signed directly. This means whitespace, property ordering and equivalent JSON formatting do not affect the semantic root. Artifact path, size, digest, role, producer, timestamp and clock metadata all affect the root.

The Ed25519 message is domain-separated from other protocols and contains the root digest. The signature is not included in the root calculation, avoiding a cycle.

## 4. Trust model

The verifier receives the expected run ID, exact bundle sequence, expected predecessor root and trusted-key map from outside the bundle. The bundle's `key_id` selects one key from that configured map. An unknown key is indeterminate; an invalid signature is blocked. A missing signature is indeterminate and never qualifies.

The v1 verifier does not decide how keys are provisioned, rotated, revoked or protected. Production deployment should use:
- signer identities distinct from the system that generated source and restore observations;
- key custody in a controlled KMS/HSM or offline signing service;
- short key validity periods and explicit revocation policy;
- signed trust-root updates;
- an independent timestamp or transparency service where non-repudiation and backdating resistance are required.

## 5. Chain semantics

A first bundle uses sequence 1 and has no previous root. Later bundles must:
- increment the sequence by exactly one;
- include the prior manifest root;
- be checked against the verifier's separately supplied expected previous root.

A missing prior bundle or unavailable expected run ID, sequence or predecessor is indeterminate. A predecessor mismatch or sequence discontinuity is blocked. A signature alone does not prove the prior bundle was retained; the verifier or custody service must retain the expected chain head.

## 6. Artifact verification

Paths use normalized forward-slash relative form. Absolute paths, empty paths, `.`, `..`, backslashes, NUL bytes, duplicate paths and symlink components are rejected. Only regular files are accepted.

The verifier requires the bundle directory to contain exactly the declared regular files (no unlisted files or symlinks), then checks observed size and SHA-256 against the signed manifest. Missing files, size mismatch or digest mismatch block the bundle. File bytes remain outside the manifest, so copying the bundle preserves the raw artifacts and permits independent re-hashing.

The current reference rejects symlinks and validates all path components before opening. It assumes the bundle is quiescent during verification; a hostile concurrent filesystem mutation could require stronger platform-specific no-follow APIs or immutable storage.

## 7. Clock and freshness semantics

Manifest and artifact times identify their clock domain and uncertainty. Missing or mismatched domains and uncertainty above policy limits are indeterminate. A capture time outside the maximum age is indeterminate. Future timestamps beyond the declared uncertainty are indeterminate, not silently trusted.

These rules are not a trusted timestamp authority. The signature only authenticates the signer's assertion of time. A trusted time source or transparency log is a separate production dependency.

## 8. Verdicts

- `qualified`: structure, artifacts, root, trusted signature, predecessor and time policy all verify.
- `blocked`: malformed path/manifest, duplicate artifact, digest/size mismatch, invalid signature, root mismatch or known predecessor/sequence contradiction.
- `indeterminate`: missing signature, unknown key, missing predecessor expectation, stale/future/unbounded/incomparable time or insufficient trust configuration.

If both blocked and indeterminate findings exist, blocked takes precedence. Missing evidence is never converted into a pass.

## 9. Threats and limitations

Addressed: artifact substitution, accidental corruption, missing evidence files, path traversal, symlink indirection, manifest whitespace/key-order ambiguity, unsigned or unknown signer claims, replayed/out-of-order run chains and weak freshness declarations.

Not solved by this layer: truthful observation, colluding signers, compromised signing keys, incorrect source instrumentation, hidden artifacts omitted before manifest creation, hostile concurrent filesystem mutation, malicious storage administrators, trusted wall-clock creation, or complete source-state inventory. These require independent observers, key separation, transparency/retention services, MON-019–025 checks and external audits.

## 10. MON-025 integration

A future adapter should:
1. gather the run manifest, image digests, source-boundary report, CDC event/transaction reports, backup manifest, archive observations, source/restore state manifests, fault records and logs;
2. retain each raw file and record it in a signed custody manifest;
3. sign through a separate attestation identity;
4. transport the bundle to a verifier under a separate administrative boundary;
5. verify file bytes, signature, chain head and clock policy;
6. only then translate custody-qualified artifacts into MON-024 receipts.

MON-026 is a custody and integrity prerequisite, not a source of database truth. It cannot independently make a restore claim true.
