# MFDI (Multi-Factor Delegated Identity) System Status

**Created**: 2026-02-06
**Last Updated**: 2026-02-06
**Status**: 3/9 Factors Fully Implemented (33%)

---

## Executive Summary

The Multi-Factor Decentralized Identity (MFDI) system provides graduated identity verification across 9 factor types in 5 categories. The system maps to 5 assurance levels (E0-E4) aligned with the Epistemic Charter v2.0, enabling capability-gated access to ecosystem features including Federated Learning participation and governance voting.

**Current State**: Core infrastructure is complete. 3 factors are fully implemented with tests, 3 are partially implemented (infrastructure exists but incomplete), and 3 are not yet started.

---

## The 9 Identity Factors

### Factor Overview by Status

| # | Factor | Category | Status | Implementation Location | Effort |
|---|--------|----------|--------|------------------------|--------|
| 1 | Primary Key (Ed25519) | Cryptographic | **IMPLEMENTED** | `Mycelix-Core/libs/fl-aggregator/src/identity/factors.rs` | Done |
| 2 | Hardware Key (WebAuthn) | Cryptographic | **PARTIAL** | `mycelix-workspace/sdk/src/identity/webauthn.rs` | M |
| 3 | Biometric Hash | Biometric | **IMPLEMENTED** | `Mycelix-Core/libs/fl-aggregator/src/identity/factors.rs` | Done |
| 4 | Social Recovery Guardians | Social Proof | **IMPLEMENTED** | `mycelix-identity/zomes/recovery/`, `Mycelix-Core/...factors.rs` | Done |
| 5 | Reputation Attestation | Social Proof | **PARTIAL** | Bridge exists, attestation logic incomplete | S |
| 6 | Gitcoin Passport | External Verification | **PARTIAL** | Types defined, API integration incomplete | M |
| 7 | Verifiable Credentials | External Verification | **PARTIAL** | `mycelix-identity/zomes/verifiable_credential/` | S |
| 8 | Recovery Phrase (BIP39) | Knowledge | **NOT STARTED** | - | S |
| 9 | Security Questions | Knowledge | **NOT STARTED** | - | S |

**Legend**: S = Small (1-2 days), M = Medium (3-5 days), L = Large (1-2 weeks)

---

## Detailed Factor Status

### Factor 1: Primary Key Pair (Ed25519) - IMPLEMENTED

**Location**: `/srv/luminous-dynamics/Mycelix-Core/libs/fl-aggregator/src/identity/factors.rs`

**Implementation Details**:
- `CryptoKeyFactor` struct with Ed25519 key generation via `ed25519-dalek`
- Full signing and verification with `sign()` and `verify()` methods
- DID derivation: `did:mycelix:{hash(public_key)[0:16]}`
- Contribution: 0.5 (50% towards assurance level when active)
- Tests: 2 tests passing (generation, signing)

**Code Sample**:
```rust
pub struct CryptoKeyFactor {
    pub factor_id: String,
    pub public_key: VerifyingKey,
    signing_key: Option<SigningKey>,
    pub status: FactorStatus,
    pub created_at: DateTime<Utc>,
    pub last_verified: Option<DateTime<Utc>>,
}
```

**Status**: Complete with tests.

---

### Factor 2: Hardware Key (WebAuthn/FIDO2) - PARTIAL

**Location**: `/srv/luminous-dynamics/mycelix-workspace/sdk/src/identity/webauthn.rs`

**What Exists** (1424 lines):
- Full `WebAuthnCredential` type with credential_id, public_key, sign_count
- `WebAuthnService` for registration and authentication challenge flows
- `HardwareKeyBridge` for connecting to Byzantine Identity Coordinator
- 21 tests covering challenge creation, credential validation, signing

**What's Missing**:
- CBOR parsing for attestation objects (returns stub error)
- COSE key signature verification (feature-gated stub)
- Actual WebAuthn protocol implementation requires browser integration

**Blockers**:
- Needs `ciborium` or similar CBOR library for attestation parsing
- Needs COSE signature verification (via `coset` + `ring`/`p256`)

**Contribution**: 0.3 when active

**Effort to Complete**: Medium (3-5 days)
- Add CBOR parsing: 1-2 days
- Implement COSE verification: 1-2 days
- Integration testing: 1 day

---

### Factor 3: Biometric Hash - IMPLEMENTED

**Location**: `/srv/luminous-dynamics/Mycelix-Core/libs/fl-aggregator/src/identity/factors.rs`

**Implementation Details**:
- `BiometricFactor` struct with template hash (never raw biometrics)
- `BiometricType` enum: Face, Fingerprint, Iris, Voice, Behavioral
- Liveness verification flag for enhanced security
- Contribution: 0.2 without liveness, 0.3 with liveness

**Code Sample**:
```rust
pub struct BiometricFactor {
    pub factor_id: String,
    pub biometric_type: BiometricType,
    pub template_hash: String,
    pub liveness_verified: bool,
    pub status: FactorStatus,
    pub created_at: DateTime<Utc>,
    pub last_verified: Option<DateTime<Utc>>,
}
```

**Status**: Complete. Client-side biometric capture is out of scope (handled by UI layer).

---

### Factor 4: Social Recovery Guardians - IMPLEMENTED

**Locations**:
- Holochain Zome: `/srv/luminous-dynamics/mycelix-identity/zomes/recovery/coordinator/src/lib.rs`
- SDK Factor: `/srv/luminous-dynamics/Mycelix-Core/libs/fl-aggregator/src/identity/factors.rs`

**Implementation Details**:
- `SocialRecoveryFactor` with guardian DIDs, threshold, Shamir secret sharing
- Holochain zome with `setup_recovery`, `initiate_recovery`, `cast_vote`, `complete_recovery`
- `RecoveryConfig`, `RecoveryRequest`, `RecoveryVote` entry types
- Time-lock mechanism for recovery delay (default 7 days)
- Contribution: 0.2 (3+ guardians), 0.3 (5+ guardians with threshold >= 3)
- Tests: Full zome tests in `recovery/coordinator/src/lib.rs`

**Status**: Complete with Holochain integration.

---

### Factor 5: Reputation Attestation - PARTIAL

**What Exists**:
- Bridge infrastructure in `sdk/src/bridge/byzantine_identity.rs`
- `AggregatedReputation` type combining hApp scores, K-Vector, MATL

**What's Missing**:
- `ReputationAttestationFactor` struct not defined
- Peer vouching workflow not implemented
- Link to MATL scoring for attestation weight

**Effort to Complete**: Small (1-2 days)
- Define `ReputationAttestationFactor` struct
- Wire to existing MATL bridge
- Add attestation creation/verification

---

### Factor 6: Gitcoin Passport - PARTIAL

**Location**: `/srv/luminous-dynamics/Mycelix-Core/libs/fl-aggregator/src/identity/factors.rs`

**What Exists**:
- `GitcoinPassportFactor` struct with address, score, stamps, expiry
- Score thresholds: >= 20 for Active, >= 50 for high contribution
- Contribution: 0.3 (score >= 20), 0.4 (score >= 50)
- Tests: 3 tests passing

**What's Missing**:
- `GitcoinPassportClient` for actual API integration (types defined, no HTTP calls)
- Stamp verification logic
- Passport score refresh mechanism

**Effort to Complete**: Medium (3-5 days)
- Implement HTTP client for Gitcoin API: 2 days
- Add stamp parsing and verification: 1 day
- Add periodic refresh/expiry handling: 1 day

---

### Factor 7: Verifiable Credentials - PARTIAL

**Location**: `/srv/luminous-dynamics/mycelix-identity/zomes/verifiable_credential/coordinator/src/lib.rs`

**What Exists**:
- W3C VC structure with issuer, subject, claims, proof
- Ed25519 signature creation and verification (fixed in PARALLEL_TRACK)
- ISO8601 expiration parsing
- Multiple VC types: VerifiedHumanity, KYC, Professional, etc.

**What's Missing**:
- Progressive VC accumulation for assurance levels
- `VerifiableCredentialFactor` wrapper for MFDI system
- Integration with factor contribution calculation

**Effort to Complete**: Small (1-2 days)
- Create factor wrapper
- Wire to assurance calculation

---

### Factor 8: Recovery Phrase (BIP39) - NOT STARTED

**Spec** (from factors.rs):
```rust
pub struct RecoveryPhraseFactor {
    pub factor_id: String,
    pub phrase_hash: String,  // Never store plaintext
    pub word_count: u8,       // 12, 18, or 24
    pub status: FactorStatus,
    pub created_at: DateTime<Utc>,
    pub last_verified: Option<DateTime<Utc>>,
}
```

**Implementation Plan**:
1. Add BIP39 wordlist validation
2. Implement phrase hashing with PBKDF2/Argon2
3. Create client-side phrase generation (12/24 words)
4. Add verification flow

**Contribution**: 0.25 (backup factor)
**Decay**: Never decays (user has it or not)

**Effort**: Small (1-2 days)

---

### Factor 9: Security Questions - NOT STARTED

**Spec** (from ECOSYSTEM_IMPROVEMENT_PLAN.md):
```rust
pub struct SecurityQuestionsFactor {
    pub factor_id: String,
    pub question_hashes: Vec<String>,  // Hash of question+answer
    pub question_count: u8,
    pub required_correct: u8,
    pub status: FactorStatus,
    pub created_at: DateTime<Utc>,
    pub last_verified: Option<DateTime<Utc>>,
}
```

**Implementation Plan**:
1. Define question set (or allow custom questions)
2. Hash questions+answers with salt
3. Implement verification requiring N-of-M correct
4. Add anti-brute-force measures

**Contribution**: 0.15-0.2
**Decay**: 180 days grace, 365 days half-life

**Effort**: Small (1-2 days)

---

## Factor Freshness Decay System - IMPLEMENTED

**Location**: `/srv/luminous-dynamics/Mycelix-Core/libs/fl-aggregator/src/identity/freshness.rs`

The freshness decay system is **fully implemented** with:

| Factor Type | Grace Period | Half-Life | Min Strength | Re-verify At |
|-------------|--------------|-----------|--------------|--------------|
| CryptoKey | 90 days | 365 days | 0.3 | 50% strength |
| HardwareKey | 180 days | 730 days | 0.4 | 50% strength |
| Biometric | 30 days | 90 days | 0.2 | 50% strength |
| SocialRecovery | 60 days | 180 days | 0.3 | 40% strength |
| GitcoinPassport | 30 days | 90 days | 0.2 | 50% strength |
| RecoveryPhrase | 365 days | Never | 0.5 | Proof required |

**Features**:
- `FactorFreshness` struct with exponential decay calculation
- `FreshnessManager` for tracking all factors
- `FreshnessStatus` enum: Fresh/Good/Stale/Warning/Expired
- `calculate_effective_strength()` applies decay to assurance calculation
- 9 tests covering decay scenarios

---

## Assurance Level Calculation - IMPLEMENTED

**Location**: `/srv/luminous-dynamics/Mycelix-Core/libs/fl-aggregator/src/identity/assurance.rs`

| Level | Score Range | Requirements | Capabilities |
|-------|-------------|--------------|--------------|
| E0 (Anonymous) | 0.0 | None | Read-only |
| E1 (Basic) | 0.3+ | 1 factor | Post, message |
| E2 (Verified) | 0.5+ | 3+ factors, 2+ categories | FL participation, proposals |
| E3 (Highly Assured) | 0.7+ | 5+ factors, 3+ categories | Governance voting |
| E4 (Constitutional) | 0.9+ | All critical factors | Constitutional amendments |

**Features**:
- Diversity bonus: +5% per unique category
- Capability mapping per level
- Recommendation engine for level advancement
- 6 tests passing

---

## Implementation Plan for Remaining 6 Factors

### Phase 1: Complete Partial Factors (Week 1-2)

| Week | Task | Owner | Deliverables |
|------|------|-------|--------------|
| 1.1 | Reputation Attestation Factor | Identity | `ReputationAttestationFactor`, MATL integration |
| 1.2 | VC Factor wrapper | Identity | `VerifiableCredentialFactor`, contribution calc |
| 1.3 | Gitcoin API client | Identity | HTTP integration, stamp verification |
| 2.1 | WebAuthn CBOR parsing | Identity | `ciborium` integration, attestation parsing |
| 2.2 | WebAuthn COSE verification | Identity | Signature verification with `coset` |

### Phase 2: Implement Missing Factors (Week 3)

| Day | Task | Deliverables |
|-----|------|--------------|
| 3.1 | Recovery Phrase Factor | BIP39 validation, hashing, verification |
| 3.2 | Security Questions Factor | Question set, hash storage, verification |
| 3.3 | Integration testing | All 9 factors in single identity flow |

### Phase 3: FL Participation Gating (Week 4)

| Task | Deliverables |
|------|--------------|
| FL admission gate | `check_fl_eligibility()` with all factor checks |
| Byzantine accountability | Link FL participation to identity for tracking |
| E2E tests | Full identity → FL participation flow |

---

## Dependencies

### Required Crates

| Crate | Purpose | Status |
|-------|---------|--------|
| `ed25519-dalek` | Primary key signing | Added |
| `sha2` | Hashing | Added |
| `chrono` | Timestamps | Added |
| `ciborium` | CBOR for WebAuthn | **NEEDED** |
| `coset` | COSE for WebAuthn | **NEEDED** |
| `bip39` or manual | Recovery phrase | **NEEDED** |
| `argon2` | Phrase/question hashing | **NEEDED** |

### External APIs

| API | Purpose | Status |
|-----|---------|--------|
| Gitcoin Passport API | Humanity verification | Types defined, client needed |

---

## Key Files Reference

### Core Implementation
```
Mycelix-Core/libs/fl-aggregator/src/identity/
├── mod.rs              # MycelixIdentity, AgentType, MFAState
├── factors.rs          # 6 factor types implemented
├── assurance.rs        # Assurance level calculation
├── freshness.rs        # Factor decay system
├── gitcoin_passport.rs # Passport types (client incomplete)
├── matl.rs            # MATL trust integration
└── fl_participation.rs # FL gating requirements
```

### Holochain Zomes
```
mycelix-identity/zomes/
├── recovery/           # Social recovery (complete)
├── verifiable_credential/ # W3C VCs (signatures fixed)
├── trust_credential/   # Trust attestations
└── bridge/            # Cross-hApp communication
```

### SDK
```
mycelix-workspace/sdk/src/identity/
├── mod.rs             # Module exports
├── webauthn.rs        # WebAuthn service (1424 lines)
└── bridge_integration.rs # Hardware key bridge
```

---

## Success Metrics

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Factors implemented | 3/9 | 9/9 | 33% |
| Freshness decay | Complete | Complete | Done |
| Assurance calculation | Complete | Complete | Done |
| FL participation gating | Partial | Complete | 60% |
| Tests passing | ~40 | 80+ | 50% |

---

## Blockers & Risks

| Blocker | Impact | Mitigation |
|---------|--------|------------|
| WebAuthn CBOR/COSE libs | Factor 2 incomplete | Add ciborium + coset |
| Gitcoin API rate limits | Factor 6 reliability | Implement caching |
| BIP39 wordlist size | Binary size | Use lazy loading |

---

## Recommendations

1. **Priority 1**: Complete Gitcoin Passport API integration - provides immediate Sybil resistance
2. **Priority 2**: Add Recovery Phrase factor - users expect this pattern
3. **Priority 3**: Complete WebAuthn - highest security factor, differentiator

---

*For the commons, by the commons, with the commons.*
