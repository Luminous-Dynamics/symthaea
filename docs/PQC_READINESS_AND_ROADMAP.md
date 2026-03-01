# Mycelix PQC Readiness and Roadmap

**Last updated**: 2026-02-24  
**Scope**: Mycelix-wide cryptographic posture against post-quantum threats.

---

## Executive Summary

Mycelix already contains PQC-capable tooling and data structures (PQC key envelopes, DID support,
and hybrid signature handling), but the default Holochain stack still relies on Ed25519/X25519
for agent identity and transport. The near-term strategy is **hybridization**: add PQC keys to DIDs,
sign critical artifacts with Ed25519 + PQC, and use PQC KEM envelopes for payload encryption. This
delivers PQC integrity guarantees **without** breaking Holochain compatibility while LAIR and the
core conductor remain ECC-based.

---

## Inventory (What Exists Today)

| Area | Current Primitive | PQC Status | Evidence |
|------|-------------------|------------|----------|
| Holochain agent identity | Ed25519 | Not PQC (upstream) | `nix/modules/holochain-base.nix`, Holochain defaults |
| Holochain key agreement | X25519 | Not PQC (upstream) | Holochain defaults |
| DID registry keys | Multibase, algorithm-tagged | PQC-ready | `mycelix-identity/zomes/did_registry` |
| DID key rotation | Deprecated old key + new key | Implemented | `mycelix-identity/zomes/did_registry` |
| PQC signing library | ML-DSA, SPHINCS+, hybrid | Implemented (native feature) | `mycelix-identity/crates/mycelix-crypto` |
| PQC keygen + signing CLI | Ed25519, ML-DSA, hybrid | Implemented | `mycelix-identity/cli` |
| Zome signature verification | Structural accept (WASM) | Partial | `mycelix-identity/zomes/mfa`, `verifiable_credential` |
| Mail payload encryption | ML-KEM envelope | Implemented | `mycelix-mail/happ/backend-rs/src/routes/emails.rs` |
| Mail CLI subject encryption | X25519 + ChaCha20-Poly1305 | Not PQC | `mycelix-mail/happ/cli/src/commands/send.rs` |
| SDK PQC (TS) | Simulated PQC signatures | Implemented (sim) | `mycelix-workspace/sdk-ts/src/bridge/pqc-signatures.ts` |

---

## Gaps and Risks

1. **Core Holochain identity remains ECC-based**  
   Agent signatures and capability grants are Ed25519 in LAIR. This is the largest PQC gap
   and cannot be fixed purely at the app layer.

2. **WASM verification limitations**  
   PQC verification is currently performed off-chain. Zomes accept PQC signatures structurally,
   which is correct for compatibility but insufficient alone for high-assurance workflows.

3. **Partial PQC encryption**  
   Mail payload encryption uses ML-KEM envelopes, but several client-side encryption paths still
   rely on X25519. This creates mixed security domains.

---

## Roadmap (Hybrid-First)

### Phase 1: Systematic Hybridization (0-3 months)

- **Add PQC keys to every DID** using `add_verification_method` and `add_key_agreement`.
- **Dual-sign critical artifacts** (credentials, trust attestations, governance votes):
  Ed25519 for on-chain verification + ML-DSA for long-term integrity.
- **KEM envelope for payloads** beyond mail (attachments, large datasets, federation payloads).
- **Document and enforce PQC verification off-chain** in clients and services.

### Phase 2: PQC-First Defaults (3-9 months)

- **Rotate DIDs to PQC-first** using `rotate_key` / `rotate_key_agreement`.
- **Set policy**: require hybrid signatures for any long-lived record (credentials, treasury,
  governance decisions, health records).
- **Standardize PQC KEM envelope format** across apps (one canonical envelope structure).

### Phase 3: Upstream Alignment (9+ months)

- Track Holochain/Lair keystore PQC upgrades.
- Align key storage, signature verification, and transport once upstream supports PQC primitives.

---

## Rollout Policy (Recommended)

This policy provides a safe migration path from **observe → hybrid → PQC-first** while
minimizing disruptions.

### Stage A: Observe (default)

- Hybrid signatures supported but not required.
- Metrics only: track how many credentials are hybrid vs Ed25519-only.

### Stage B: Warn-on-Ed25519

- Services return warnings when verification uses Ed25519-only proofs.
- UI/SDK surfaces warnings to operators and power users.

### Stage C: Enforce Hybrid for Verification (critical apps)

- Gateways reject non-hybrid proofs for **credentials, governance, finance, health**.
- Enable enforcement flags:
  - Mail backend: `IDENTITY_REQUIRE_HYBRID_SIGNATURES=true`
  - SDK clients: set `requireHybridSignatures: true` in Identity client config.

### Stage D: PQC-first Keys

- New keys are ML-DSA/ML-KEM by default; Ed25519 retained only for legacy verification.
- Legacy keys are revoked after data re-signing or re-encryption completes.

---

## DPKI Key Roll Runbook

This runbook uses existing DID registry APIs to rotate both signing and KEM keys while retaining
verification continuity for historical data.

### 1) Generate PQC Keys

- Generate PQC signing keys (ML-DSA-65/87) and KEM keys (ML-KEM-768/1024) using:
  `mycelix-identity/cli` or `mycelix-identity/crates/mycelix-crypto`.
- JSON templates for verification methods:
  - `docs/did-verification-method-auth.json`
  - `docs/did-verification-method-kem.json`

### 2) Publish PQC Keys to DID

- Add PQC verification method:
  - `add_verification_method(VerificationMethod { ... })`
- Add PQC key agreement method:
  - `add_key_agreement(VerificationMethod { ... })`

### 3) Rotate Primary Signing Key

- `rotate_key { old_key_id, new_method }`
  - Old key is retained as `-deprecated-v{version}` for historical verification.

### 4) Rotate KEM Key Agreement

- `rotate_key_agreement { old_key_id, new_method }`
  - Old KEM key remains for decrypting legacy envelopes.

### 5) Re-issue or Re-sign Long-Lived Records

- Re-issue credentials with hybrid signatures.
- Update governance records and other durable attestations.

### 6) Announce Rotation (Optional but Recommended)

- Emit a DID update event or record a rotation entry for auditing.
- Notify dependent hApps to refresh cached DID documents.

### Scripted Rotation (SDK)

The TypeScript SDK includes a helper script:

```bash
npm run rotate:did-keys -- \
  --old-key-id "#keys-1" \
  --new-auth-key docs/did-verification-method-auth.json \
  --new-kem-key docs/did-verification-method-kem.json
```

---

## Implementation Notes

- **Hybrid signatures** are already supported in `mycelix-crypto` and accepted structurally in WASM.
  Off-chain verification should be mandatory for high-assurance records.
- **KEM envelopes** are the path forward for payload secrecy while Holochain remains ECC-based.
- **Deprecated keys should not be deleted** until all dependent data is re-signed or re-encrypted.
- **Gateway enforcement** can be enabled via identity clients (e.g. Mail backend) using
  `IDENTITY_REQUIRE_HYBRID_SIGNATURES=true`.

---

## Open Questions

1. Which hApps require mandatory hybrid signatures (e.g., governance, finance, health)?
2. Where should PQC verification be enforced: clients, gateways, or both?
3. Should legacy Ed25519-only records be re-signed automatically or on-demand?
