# Attested role-scoped memory access

Status: research implementation on `research/broca-attested-role-access-2026-10-30`. The implementation is not yet compiled or integrated into the production worker.

## Why this exists

The role-scoped evidence adapter previously accepted a structurally valid `RoleScopePolicy` created directly from an in-memory map. That proved the map was well-formed, not that an authorized governance principal issued it. The role gate and Ed25519 governance were therefore separate trust domains.

This change introduces a dedicated signed role-to-scope policy protocol. It is not encoded as a source-authority policy: agent roles are principals and evidence scopes are resources.

## Protocol

A `RoleScopePolicyDocument` contains:

- `policy_version`: strictly positive and monotonically increased by the policy issuer.
- `recorded_at`: policy signer's timestamp, included in the signature but not trusted as activation time.
- `valid_until`: mandatory exclusive expiry.
- `readable_scopes`: explicit role-to-scope grants. Empty roles/scopes, whitespace-ambiguous identifiers, and wildcard grants are rejected.

The signed envelope uses its own schema and domain-separation strings:

- `BROCA-ROLE-SCOPE-POLICY-V1`
- `LUMINOUS-DYNAMICS/BROCA/ROLE-SCOPE-POLICY/V1`
- `LUMINOUS-DYNAMICS/BROCA/ROLE-SCOPE-POLICY-INGESTION-RECEIPT/V1`

The policy is signed by a trusted key with the existing policy-signing role. A separate trusted ingress key signs a receipt binding the accepted time, policy version, signer identity, and policy signature digest. Existing trust-store configuration rejects duplicate key material and keys with multiple signing roles.

Activation begins at the receipt's `accepted_at`, not the source-controlled `recorded_at`. The signed `valid_until` is exclusive. The trust store checks both signers' key windows at the trusted acceptance time and rejects a policy that was already expired when accepted.

### Verification and use

1. Provision an `EvidenceTrustStore` from trusted configuration. Do not load signer keys from the policy being verified.
2. Sign a `RoleScopePolicyDocument` with a policy key and attach a receipt from a distinct ingress key.
3. Call `EvidenceTrustStore::verify_role_scope_policy(envelope, minimum_policy_version_exclusive)`.
4. Persist the highest accepted policy version in rollback-resistant state. Supply that value on every subsequent verification. A version less than or equal to it is rejected.
5. Convert the unforgeable result with `RoleScopePolicy::from_verified`.
6. Use `EvidenceIndex::search_bm25_for_role` or `resolve_for_role`. Both check the grant and policy activity window before retrieval/materialization.
7. Keep the `AuthorizedEvidenceContext` provenance outside the model prompt. Pass only its `context` field to the language model.

The authorization record returned with each item now includes the policy version, digest, trusted acceptance time, and expiry, in addition to the evidence source and content digest.

## Security properties tested in source

The new unit tests cover:

- valid policy signature plus independent ingress receipt;
- stale/rolled-back policy versions;
- policy expiry before trusted acceptance;
- policy payload tampering;
- ingress receipt timestamp tampering;
- missing ingress receipt;
- a valid event signer attempting to issue policy;
- wildcard and empty grants;
- a role policy being inactive before receipt time and at its exclusive expiry.

These are authored tests only. They have not been compiled or executed in this environment.

## Explicit limits and production requirements

- The `requesting_role` argument must be obtained from an authenticated runtime principal. The retrieval API does not authenticate caller-supplied role strings.
- The caller must persist the highest accepted policy version in rollback-resistant storage. The in-memory token cannot prevent a process from deliberately verifying an old policy with a reset minimum version.
- A cached, already verified policy does not receive instantaneous revocation when a signer key is later retired. Workers must reload the current policy, advance the version floor, and/or enforce a trusted revocation feed.
- The evidence index is process-local. Persistent lifecycle/tombstone state must commit atomically with any durable HDC/vector index.
- Existing low-level `search_bm25` and `resolve` methods remain available for compatibility. Multi-agent workers must use the role-authorized entry points and must not expose the low-level methods as their memory API.
- This is scope-level authorization. Per-record ACLs, delegation, policy transparency proofs, Mycelix identity binding, and worker integration remain separate work.
- Signatures authenticate who issued a particular policy payload; they do not establish that the grants are wise, fair, or factually justified.

## Next security gates

Before production use, require authenticated-principal binding, durable anti-rollback state, policy-reload/revocation semantics, persistence atomicity, and integration tests that prove the worker cannot call an unauthorized legacy retrieval path.
