# GDPR Compliance Documentation

**Mycelix Federated Learning Platform**
**Document Version**: 1.0
**Last Updated**: 2026-01-18
**Classification**: Internal / Compliance

---

## Executive Summary

This document describes how the Mycelix federated learning platform complies with the General Data Protection Regulation (GDPR) (EU) 2016/679, with particular emphasis on:
- Article 17: Right to Erasure ("Right to be Forgotten")
- Article 25: Data Protection by Design and by Default
- Article 30: Records of Processing Activities
- Article 32: Security of Processing

---

## 1. Data Processing Inventory

### 1.1 Categories of Personal Data Processed

| Data Category | Description | Legal Basis | Retention | Storage Location |
|---------------|-------------|-------------|-----------|------------------|
| Agent Public Keys | Cryptographic identifiers for FL participants | Contract (Art. 6.1.b) | Until erasure request | Holochain DHT |
| Model Gradients | Differential updates from local training | Legitimate Interest (Art. 6.1.f) | 30 days rolling | Encrypted on-chain |
| Trust Scores | MATL reputation metrics (PoGQ, TCDM, Entropy) | Legitimate Interest (Art. 6.1.f) | Until erasure request | Holochain DHT |
| Contribution Metadata | Timestamps, round IDs, aggregation proofs | Contract (Art. 6.1.b) | 90 days | Ethereum anchor |
| ZK Proofs | Zero-knowledge proofs of contribution validity | Legitimate Interest (Art. 6.1.f) | Permanent (no PII) | Immutable chain |

### 1.2 Data Flows

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   Participant   │         │   FL Aggregator │         │  Global Model   │
│   (Agent Node)  │         │   (Coordinator) │         │   (Ethereum)    │
└────────┬────────┘         └────────┬────────┘         └────────┬────────┘
         │                           │                           │
         │ 1. Local Training         │                           │
         │ (data never leaves device)│                           │
         │                           │                           │
         │ 2. Gradient + ZK Proof    │                           │
         │──────────────────────────>│                           │
         │   (encrypted w/ user key) │                           │
         │                           │ 3. Aggregate              │
         │                           │ (privacy-preserved)       │
         │                           │                           │
         │                           │ 4. Anchor Proof           │
         │                           │──────────────────────────>│
         │                           │                           │
         │ 5. Updated Model          │                           │
         │<──────────────────────────│                           │
```

### 1.3 Third-Party Processors

| Processor | Purpose | Data Shared | DPA Status |
|-----------|---------|-------------|------------|
| Ethereum Mainnet | Proof anchoring | Hashed commitment only | N/A (public chain) |
| Holochain Network | DHT storage | Encrypted contributions | Decentralized (no single processor) |

---

## 2. Right to Erasure Implementation (Article 17)

### 2.1 Cryptographic Erasure Mechanism

Mycelix implements GDPR Article 17 through **cryptographic erasure**: each agent's data is encrypted with a unique key, and erasure is achieved by permanently deleting that key.

**Technical Implementation**: `libs/fl-aggregator/src/privacy/erasure.rs`

```rust
// Encrypt data for a specific agent
let manager = ErasureKeyManager::new();
let encrypted = manager.encrypt_for_agent("agent-123", &contribution_data)?;

// Store encrypted data (key held separately)
store_to_dht(encrypted);

// When erasure is requested (GDPR Art. 17)
let receipt = manager.request_erasure("agent-123")?;

// Data is now cryptographically unrecoverable
// Receipt provides audit trail for compliance
```

### 2.2 Erasure Request Procedure

**For Data Subjects**:

1. **Submit Request**: Send erasure request through Mycelix dashboard or API
2. **Identity Verification**: Prove ownership of agent public key
3. **Processing**: System deletes encryption key (typically < 24 hours)
4. **Confirmation**: Receive `ErasureReceipt` with cryptographic proof

**For Data Controllers**:

1. Receive request via `POST /api/v1/erasure/request`
2. Verify agent identity (signature verification)
3. Call `ErasureKeyManager::request_erasure(agent_id)`
4. Store receipt for audit trail
5. Return receipt to data subject

### 2.3 Erasure Receipt

Each erasure operation generates a verifiable receipt:

```rust
pub struct ErasureReceipt {
    pub receipt_id: String,       // Unique identifier
    pub agent_hash: String,       // SHA3-256 hash of agent ID
    pub key_hash: String,         // SHA3-256 hash of deleted key
    pub erased_at: u64,           // Unix timestamp
    pub items_affected: u64,      // Number of encrypted items
    pub proof_hash: String,       // Cryptographic proof of erasure
}
```

**Verification**: `receipt.verify()` returns `true` if proof is valid.

### 2.4 Data Covered by Erasure

| Data Type | Erasure Effect |
|-----------|----------------|
| Encrypted Gradients | Unrecoverable (key deleted) |
| Trust Scores | Removed from DHT |
| Contribution Metadata | Removed from local storage |
| ZK Proofs | Retained (contain no PII) |
| Ethereum Anchors | Immutable but unlinkable (hashed) |

### 2.5 Exceptions (Article 17.3)

Erasure does NOT apply to:
- Anonymized aggregate model weights (no PII)
- ZK proofs (mathematical proofs, no personal data)
- Ethereum anchors (cryptographic commitments only)
- Data required for legal claims

---

## 3. Privacy by Design (Article 25)

### 3.1 Data Minimization

- **Local Training**: Raw data never leaves participant devices
- **Gradient Compression**: Only model updates transmitted
- **Differential Privacy**: Noise added before aggregation

### 3.2 Encryption

| Data State | Encryption |
|------------|------------|
| At Rest (DHT) | ChaCha20-Poly1305 (per-user keys) |
| In Transit | TLS 1.3 + Holochain encryption |
| Gradients | Optional AES-GCM (privacy feature) |

### 3.3 Pseudonymization

- Agent public keys are pseudonymous identifiers
- Trust scores linked to pseudonyms, not real identities
- Cross-referencing prevented by design

---

## 4. Security Measures (Article 32)

### 4.1 Cryptographic Controls

| Control | Implementation |
|---------|----------------|
| Key Generation | `getrandom` (OS CSPRNG) |
| Symmetric Encryption | ChaCha20-Poly1305 (256-bit) |
| Hashing | SHA3-256, BLAKE3 |
| Signatures | Ed25519 |
| Post-Quantum Ready | Dilithium, Kyber (optional features) |

### 4.2 Access Controls

- Per-agent encryption keys
- Holochain capability tokens
- Byzantine fault tolerance (45% threshold)

### 4.3 Audit Logging

All erasure operations logged with:
- Timestamp
- Agent hash (not ID)
- Receipt ID
- Proof hash

---

## 5. Data Subject Rights

| Right | Implementation |
|-------|----------------|
| Access (Art. 15) | Export via dashboard/API |
| Rectification (Art. 16) | Re-submit corrected contributions |
| Erasure (Art. 17) | Cryptographic key deletion |
| Restriction (Art. 18) | Pause contribution processing |
| Portability (Art. 20) | JSON export of contributions |
| Object (Art. 21) | Opt-out of aggregation rounds |

---

## 6. Compliance Verification

### 6.1 Automated Tests

Run compliance test suite:
```bash
cd libs/fl-aggregator
cargo test --lib --features privacy privacy::erasure
```

Expected output: 7/7 tests pass

### 6.2 Manual Verification Checklist

- [ ] Erasure request processed within 30 days
- [ ] Receipt provided to data subject
- [ ] Data unrecoverable after erasure (test decryption fails)
- [ ] Audit log entry created
- [ ] No PII in ZK proofs or anchors

---

## 7. Contact Information

**Data Protection Officer**: [To be appointed]
**Privacy Inquiries**: privacy@luminous-dynamics.io
**Erasure Requests**: erasure@luminous-dynamics.io

---

## Appendix A: API Reference

### Erasure Endpoints

```
POST /api/v1/erasure/request
  Body: { "agent_id": "...", "signature": "..." }
  Response: { "receipt": ErasureReceipt }

GET /api/v1/erasure/status/{agent_hash}
  Response: { "is_erased": bool, "receipt": Option<ErasureReceipt> }

GET /api/v1/erasure/receipt/{receipt_id}
  Response: { "receipt": ErasureReceipt, "verified": bool }
```

### Rust API

```rust
use fl_aggregator::privacy::{ErasureKeyManager, ErasureReceipt};

// Initialize
let manager = ErasureKeyManager::new();

// Encrypt user data
let encrypted = manager.encrypt_for_agent(agent_id, data)?;

// Request erasure
let receipt: ErasureReceipt = manager.request_erasure(agent_id)?;

// Verify receipt
assert!(receipt.verify());

// Check status
assert!(manager.is_erased(agent_id));
```

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-18 | Privacy Team | Initial creation |
