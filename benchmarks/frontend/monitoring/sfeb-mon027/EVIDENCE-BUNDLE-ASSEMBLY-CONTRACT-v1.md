# MON-027 — Evidence Bundle Assembly and External Signing Contract v1

**Profile:** `mon027-evidence-bundle-assembly-v1`

## 1. Objective

Construct MON-026 custody manifests from a deliberately allowlisted set of MON-025 output files, while keeping the private signing key outside the collection and assembly process.

## 2. Input index

The caller provides a strict JSON input index containing:
- run ID, sequence and predecessor manifest root;
- manifest capture time, clock domain and uncertainty;
- exact source boundary (incident, cluster, database, system ID, timeline, snapshot, capture LSN, scope and inventory digests);
- signer key ID;
- artifact paths and their declared media type, role, producer, command ID, collection time, clock domain and uncertainty.

The index is not itself source truth. It is metadata about which raw artifacts are being packaged. It must be built from a MON-025 run and its provenance.

## 3. Assembly

The assembler:
1. parses exactly one JSON input document and rejects unknown fields;
2. checks profile/version and required source boundary values;
3. validates relative paths, rejecting absolute paths, traversal, drive-qualified names, backslashes and NUL;
4. rejects symlink components and non-regular files;
5. requires the actual artifact directory to contain exactly the allowlisted files (no missing or unlisted files);
6. streams each file through SHA-256 and records observed byte size;
7. computes the MON-026 deterministic semantic root;
8. writes an unsigned manifest and a detached signing request outside the artifact directory.

It does not sign or certify the bundle.

## 4. Signing request

The signing request includes profile/version, deterministic request ID, run ID, sequence, root SHA-256, algorithm, key ID and the exact domain-separated message bytes in base64. The request ID is an idempotency key derived from run ID, sequence, root and key ID.

The external signer must:
- authenticate the requesting identity and its authorization to sign this specific run/sequence/root;
- check key status and allowed algorithm;
- sign only the supplied message, without reserializing or reconstructing it;
- return a detached envelope with the same request ID, run ID, sequence, root, algorithm/key ID and a base64 Ed25519 signature;
- record signer audit evidence separately.

A signing service should reject duplicate request IDs or use a durable idempotency record to avoid ambiguous repeated operations.

## 5. Signature attachment

The attach command checks that the detached signature envelope matches the manifest key ID and algorithm. It inserts the signature and then calls the MON-026 verifier with:
- out-of-band expected run ID and sequence;
- out-of-band expected previous root for sequence greater than one;
- out-of-band trusted public key;
- clock domain, maximum bundle age and clock uncertainty policy.

The signed manifest is written only after the complete artifact directory and cryptographic verification qualify. Unknown signer/trust state remains indeterminate; invalid signatures, artifact mutation and chain mismatches block.

## 6. Exact inventory

The artifact directory contains only raw evidence artifacts. The input index, unsigned manifest, signing request, signature response and signed manifest are kept outside that root. This avoids unlisted metadata files being confused with raw run artifacts and prevents silent omission.

## 7. Threat model and limits

Addressed: accidental/hostile artifact substitution before signature, unlisted files, path traversal, symlink escapes, stale run replay, sequence mismatch, previous-root splice, signature/key-ID mismatch and collector-held-key self-certification.

Not solved: false source observations, dishonest index authoring, compromised trusted signer, signer/collector collusion, malicious concurrent filesystem mutation, clock authority compromise, incomplete raw evidence before packaging, key revocation distribution or retention of the chain head. The verifier assumes a quiescent artifact directory; production should use immutable snapshots/object lock and a separately administered signer.

## 8. MON-024 composition

MON-027's output is still not a MON-024 production receipt. An independent custody verifier must validate the signed manifest; domain-specific observers must independently qualify the raw PostgreSQL, CDC, backup, restore and service facts. Only then should an adapter produce normalized MON-024 stage receipts.
