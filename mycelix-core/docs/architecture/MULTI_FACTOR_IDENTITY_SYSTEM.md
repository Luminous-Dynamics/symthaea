# 🔐 Multi-Factor Decentralized Identity (MFDI) System
## Design Specification v1.0

**Status**: Design Phase
**Target**: Phase 1 Core Production (Q4 2025)
**Constitutional Alignment**: Epistemic Charter v2.0, Constitution v0.24

---

## 🎯 Design Principles

### 1. **Agent Sovereignty**
Every agent (human, AI, collective) owns their identity completely. No central authority can revoke or modify identity without consent.

### 2. **Graduated Verifiability**
Identity verification follows the Epistemic Cube (E-Axis), allowing easy onboarding with progressive verification.

### 3. **Multi-Factor Security**
Multiple independent verification factors prevent single points of failure and enable graceful recovery.

### 4. **Privacy by Design**
Zero-knowledge proofs allow verification of identity properties without revealing identity data.

### 5. **Eternal Accessibility**
Recovery mechanisms ensure identity access even after loss of multiple factors.

---

## 🏗️ Architecture Overview

### Core Identity Structure

```rust
pub struct MycelixIdentity {
    // Core DID (W3C Standard)
    pub did: String,  // "did:mycelix:{agent_pub_key}"

    // Agent Type Classification
    pub agent_type: AgentType,

    // Multi-Factor Authentication State
    pub mfa_state: MFAState,

    // Verifiable Credentials (Progressive)
    pub credentials: Vec<VerifiableCredential>,

    // MATL Integration
    pub trust_metrics: TrustMetrics,

    // Recovery Configuration
    pub recovery_config: RecoveryConfig,

    // Epistemic Classification
    pub epistemic_tier_e: EpistemicTierE,  // Identity verification level
}

pub enum AgentType {
    HumanMember {
        biometric_hash: Option<Hash>,  // Never raw biometrics
        humanity_proofs: Vec<HumanityProof>,
    },
    InstrumentalActor {
        model_type: String,
        version: String,
        operator_did: String,  // Accountable human/DAO
    },
    DAOCollective {
        member_dids: Vec<String>,
        governance_contract: Hash,
    },
}

pub struct MFAState {
    // Primary Factors (Choose 3+ for E2+)
    pub factors: Vec<IdentityFactor>,

    // Active Recovery Guardians
    pub recovery_guardians: Vec<GuardianConfig>,

    // Verification History
    pub verification_history: Vec<VerificationEvent>,

    // Current Assurance Level
    pub assurance_level: AssuranceLevel,
}
```

### Hierarchical Identity (Individual, Delegated, Collective)

Identities exist at multiple scopes to support DAOs, delegated authority, and collective decision-making:

```rust
/// Identity scope supporting individuals, delegates, and collectives
pub enum IdentityScope {
    /// Standard individual identity
    Individual(MycelixIdentity),

    /// Delegated authority from one identity to another
    Delegated {
        /// The principal (delegating party) DID
        principal: String,
        /// The delegate DID
        delegate: String,
        /// Specific capabilities granted
        scope: Vec<Capability>,
        /// When the delegation expires
        expires_at: Option<Timestamp>,
        /// Proof of delegation (signed by principal)
        delegation_proof: Signature,
    },

    /// Collective identity (DAO/multisig)
    Collective {
        /// The DAO's DID
        dao_did: String,
        /// Member DIDs authorized to act
        signers: Vec<String>,
        /// Required signatures for action
        threshold: u8,
        /// Governance contract hash (defines rules)
        governance_contract: Hash,
    },
}

/// Capabilities that can be delegated
pub enum Capability {
    /// Submit gradients on behalf of principal
    FLParticipation { round_limit: Option<u64> },
    /// Vote on proposals
    GovernanceVoting { proposal_types: Vec<ProposalType> },
    /// Manage storage and data
    DataManagement { read: bool, write: bool, delete: bool },
    /// Issue attestations
    AttestationAuthority { credential_types: Vec<VCType> },
    /// Treasury operations
    TreasuryAccess { spend_limit: Option<u64> },
    /// Custom capability with arbitrary scope
    Custom { name: String, parameters: serde_json::Value },
}

/// Resolution rules for hierarchical identity actions
pub struct IdentityScopeResolver {
    /// Maximum delegation depth (prevents infinite chains)
    pub max_delegation_depth: u8,  // Default: 3
    /// Whether delegated actions inherit principal's reputation
    pub inherit_reputation: bool,
    /// Time buffer for delegation expiry checks
    pub expiry_buffer: Duration,
}
```

**Use Cases**:
- **DAO Treasury Multisig**: 3-of-5 signers for treasury operations
- **FL Participation Delegation**: Allow ML infrastructure to submit gradients on member's behalf
- **Governance Proxy Voting**: Delegate voting power to trusted representatives
- **Service Account Authorization**: Authorize automated services with scoped capabilities

---

## 🔑 Multi-Factor Framework

### Identity Factors (9 Types)

Each factor provides independent verification, increasing the epistemic tier (E-Axis):

#### **Category 1: Cryptographic Factors** (High Security)

**1. Primary Key Pair (Ed25519)**
- **What**: Agent's Holochain AgentPubKey
- **Strength**: E3 (Cryptographically Proven)
- **Recovery**: Requires 3+ other factors
- **Storage**: Source chain, never leaves device
```rust
pub struct CryptoFactor {
    pub public_key: AgentPubKey,
    pub key_creation_timestamp: Timestamp,
    pub device_fingerprint: Hash,  // Optional
    pub last_used: Timestamp,
}
```

**2. Hardware Security Key (YubiKey, Ledger, etc.)**
- **What**: Physical FIDO2/U2F device
- **Strength**: E3 (Hardware-backed proof)
- **Recovery**: Physical possession required
- **Backup**: User should have 2+ keys
```rust
pub struct HardwareKeyFactor {
    pub device_id: String,
    pub attestation_cert: Certificate,
    pub counter: u32,  // Prevents cloning
    pub added_date: Timestamp,
}
```

#### **Category 2: Biometric Factors** (Convenience + Security)

**3. Biometric Hash (Privacy-Preserving)**
- **What**: Hash of biometric template (face, fingerprint, iris)
- **Strength**: E1 (Testimonial) → E3 (with liveness proof)
- **Privacy**: Never stores raw biometrics
- **Recovery**: Can be re-enrolled with other factors
```rust
pub struct BiometricFactor {
    pub biometric_hash: Hash,  // Hash of template, not raw data
    pub biometric_type: BiometricType,  // Face, Fingerprint, Iris, Voice
    pub template_version: String,
    pub liveness_proof_required: bool,
    pub enrolled_date: Timestamp,
}

pub enum BiometricType {
    Face { liveness: bool },
    Fingerprint,
    Iris,
    Voice { passphrase_hash: Option<Hash> },
    Behavioral { typing_pattern: bool, gait: bool },
}
```

#### **Category 3: Social Proof Factors** (Distributed Trust)

**4. Social Recovery Guardians (Shamir Secret Sharing)**
- **What**: Trusted contacts who hold recovery shares
- **Strength**: E2 (Privately Verifiable) if 3+ guardians
- **Recovery**: Threshold (e.g., 3-of-5) can reconstitute key
- **Implementation**: Shamir Secret Sharing over master recovery seed
```rust
pub struct SocialRecoveryFactor {
    pub guardians: Vec<Guardian>,
    pub threshold: u8,  // e.g., 3 of 5
    pub total_shares: u8,
    pub recovery_key_hash: Hash,  // Hash of the reconstructable key
}

pub struct Guardian {
    pub did: String,
    pub share_hash: Hash,  // Hash of their shard
    pub trust_score: f64,  // From MATL
    pub relationship_type: RelationshipType,  // Family, Friend, Colleague
    pub added_date: Timestamp,
    pub can_initiate_recovery: bool,
}
```

**5. Reputation Attestations (Peer Vouching)**
- **What**: Other verified members attest to your identity
- **Strength**: E1 (Testimonial) → E2 (if high-trust attestors)
- **Recovery**: Can help verify identity during recovery
- **MATL Integration**: Attestors' trust scores weight the attestation
```rust
pub struct ReputationAttestationFactor {
    pub attestations: Vec<Attestation>,
    pub weighted_trust_score: f64,  // MATL-weighted aggregate
    pub attestation_threshold: u8,  // e.g., need 5 attestations
}

pub struct Attestation {
    pub attestor_did: String,
    pub attestor_trust_score: f64,  // From MATL
    pub attestation_claim: String,  // "I verify this person is who they claim"
    pub epistemic_tier: EpistemicTierE,
    pub timestamp: Timestamp,
    pub signature: Signature,
}
```

#### **Category 4: External Verification Factors** (Sybil Resistance)

**6. Gitcoin Passport (Sybil Scoring)**
- **What**: Aggregated proof of personhood across multiple platforms
- **Strength**: E2 (Privately Verifiable) via Gitcoin's scoring
- **Recovery**: Can be re-linked after verifying other factors
- **Requirement**: Score ≥20 for Phase 1
```rust
pub struct GitcoinPassportFactor {
    pub passport_id: String,
    pub score: f64,  // 0-100 scale
    pub stamps: Vec<PassportStamp>,
    pub last_verified: Timestamp,
    pub verification_cadence: Duration,  // Re-verify monthly
}

pub struct PassportStamp {
    pub platform: String,  // "github", "twitter", "ens", etc.
    pub verified: bool,
    pub weight: f64,  // Contribution to score
}
```

**7. Verifiable Credentials (Progressive)**
- **What**: Signed credentials from trusted issuers
- **Strength**: E3 (Cryptographically Proven)
- **Recovery**: Can be re-issued after identity verification
- **Types**: KYC, Professional licenses, Memberships, Age proofs
```rust
pub struct VerifiableCredentialFactor {
    pub credentials: Vec<VC>,
    pub required_credential_types: Vec<VCType>,  // For certain roles
}

pub struct VC {
    pub vc_id: String,
    pub issuer_did: String,
    pub subject_did: String,  // This identity
    pub credential_type: VCType,
    pub claims: Vec<Claim>,
    pub issuance_date: Timestamp,
    pub expiration_date: Option<Timestamp>,
    pub proof: VCProof,  // Cryptographic signature
    pub revocation_list_url: Option<String>,
}

pub enum VCType {
    VerifiedHumanity,  // Phase 1 baseline
    KYCBasic,
    KYCEnhanced,
    ProfessionalLicense,
    EducationDegree,
    MembershipProof,
    AgeOver18,
    ResidencyProof,
}
```

#### **Category 5: Knowledge Factors** (Backup)

**8. Encrypted Recovery Phrase (Mnemonic)**
- **What**: 24-word BIP39 mnemonic, encrypted with passphrase
- **Strength**: E3 (Cryptographically Proven) if passphrase strong
- **Recovery**: Can regenerate keys from seed
- **Storage**: User memorizes or stores securely (not on device)
```rust
pub struct RecoveryPhraseFactor {
    pub phrase_hash: Hash,  // Never stores plaintext
    pub encryption_method: EncryptionMethod,
    pub created_date: Timestamp,
    pub last_verified: Option<Timestamp>,  // User proves they have it
}
```

**9. Security Questions (Multi-Layer)**
- **What**: Personalized questions with hashed answers
- **Strength**: E1 (Testimonial) - Weak alone, strong with others
- **Recovery**: Can help verify identity during recovery
- **Implementation**: Minimum 5 questions, answers hashed + salted
```rust
pub struct SecurityQuestionsFactor {
    pub questions: Vec<SecurityQuestion>,
    pub required_correct: u8,  // e.g., 4 of 5
    pub max_attempts: u8,  // Rate limiting
    pub lockout_duration: Duration,
}

pub struct SecurityQuestion {
    pub question_hash: Hash,  // Even question is hashed
    pub answer_hash: Hash,  // Salted + hashed
    pub salt: Vec<u8>,
    pub created_date: Timestamp,
}
```

---

## ⏱️ Factor Freshness & Decay

Identity factors lose effective strength over time without re-verification. This prevents stale proofs from maintaining high assurance levels indefinitely.

### Freshness Decay Model

```rust
/// Tracks freshness state for each identity factor
pub struct FactorFreshness {
    /// When the factor was last verified/used
    pub last_verified: Timestamp,
    /// Factor-specific decay rate (0.0-1.0 per period)
    pub decay_rate: f32,
    /// Grace period before decay begins
    pub grace_period: Duration,
    /// When re-verification becomes mandatory
    pub reverification_required: Duration,
    /// Current effective strength (0.0-1.0)
    pub effective_strength: f32,
}

/// Decay configuration per factor type
pub struct FactorDecayConfig {
    pub factor_type: FactorType,
    pub base_strength: f32,        // Starting strength when fresh
    pub grace_period: Duration,    // No decay during this period
    pub half_life: Duration,       // Time for strength to halve
    pub minimum_strength: f32,     // Floor (factor never drops below)
    pub reverify_threshold: f32,   // Below this, must re-verify
}

impl FactorFreshness {
    /// Calculate current effective strength based on time elapsed
    pub fn current_strength(&self, now: Timestamp) -> f32 {
        let elapsed = now - self.last_verified;

        // Within grace period: full strength
        if elapsed <= self.grace_period {
            return 1.0;
        }

        // Exponential decay after grace period
        let decay_time = elapsed - self.grace_period;
        let decay_factor = (-self.decay_rate * decay_time.as_secs_f32()).exp();

        (decay_factor).max(0.0).min(1.0)
    }

    /// Check if factor requires re-verification
    pub fn needs_reverification(&self, now: Timestamp) -> bool {
        self.current_strength(now) < 0.5
            || (now - self.last_verified) > self.reverification_required
    }
}
```

### Default Decay Parameters by Factor Type

| Factor Type | Grace Period | Half-Life | Min Strength | Re-verify At |
|-------------|--------------|-----------|--------------|--------------|
| Primary Key Pair | 90 days | 365 days | 0.3 | 0.5 |
| Hardware Key | 180 days | 730 days | 0.4 | 0.5 |
| Biometric | 30 days | 90 days | 0.2 | 0.5 |
| Social Recovery | 60 days | 180 days | 0.3 | 0.4 |
| Reputation Attestation | 14 days | 60 days | 0.1 | 0.3 |
| Gitcoin Passport | 30 days | 90 days | 0.2 | 0.5 |
| Verifiable Credentials | Expiry-based | N/A | 0.0 | Expiry |
| Recovery Phrase | 365 days | Never | 0.5 | Proof required |
| Security Questions | 180 days | 365 days | 0.2 | 0.4 |

### Assurance Level Impact

Factor decay affects the effective assurance level:

```rust
pub fn calculate_effective_assurance(
    identity: &MycelixIdentity,
    now: Timestamp,
) -> (AssuranceLevel, f32) {
    let mut total_strength = 0.0;
    let mut category_count = HashSet::new();

    for factor in &identity.mfa_state.factors {
        let freshness = factor.freshness();
        let effective = freshness.current_strength(now);

        // Only count factors above minimum threshold
        if effective >= 0.3 {
            total_strength += effective * factor.base_weight();
            category_count.insert(factor.category());
        }
    }

    // Assurance level based on effective strength and diversity
    let level = match (total_strength, category_count.len()) {
        (s, c) if s >= 4.0 && c >= 4 => AssuranceLevel::E4,
        (s, c) if s >= 3.0 && c >= 3 => AssuranceLevel::E3,
        (s, c) if s >= 2.0 && c >= 2 => AssuranceLevel::E2,
        (s, _) if s >= 1.0 => AssuranceLevel::E1,
        _ => AssuranceLevel::E0,
    };

    (level, total_strength)
}
```

### Re-verification Prompts

The system proactively prompts users to re-verify decaying factors:

1. **Soft Reminder** (strength < 0.7): "Your biometric verification is getting stale. Re-verify to maintain full access."
2. **Warning** (strength < 0.5): "Verification required within 7 days to maintain E2 status."
3. **Critical** (strength < 0.3): "Factor excluded from assurance calculation. Re-verify now."
4. **Expired** (strength = 0): "Factor has expired and must be re-enrolled."

---

## 🎚️ Assurance Levels (Graduated Verification)

Users progress through assurance levels as they add verification factors:

### **Level 0: Anonymous (E0, N0, M0)**
- **Factors Required**: None
- **Capabilities**: Read-only, browse public content
- **Restrictions**: Cannot vote, cannot post, cannot earn reputation
- **Use Case**: Curious observers, pre-registration

### **Level 1: Basic (E1, N0, M1)**
- **Factors Required**: 1 factor (typically primary key pair)
- **Capabilities**: Post comments, participate in discussions
- **Restrictions**: Cannot vote on proposals, limited reputation earning
- **Use Case**: Community members, casual participation

### **Level 2: Verified (E2, N1, M2)**
- **Factors Required**: 3+ factors from 2+ categories
- **Example Combos**:
  - Primary Key + Gitcoin Passport (≥20) + Social Recovery (3 guardians)
  - Primary Key + Hardware Key + Biometric Hash
- **Capabilities**: Vote in Local/Sector DAOs, earn reputation, submit proposals
- **Restrictions**: Cannot vote on Global DAO or constitutional matters
- **Use Case**: Active community members, most users

### **Level 3: Highly Assured (E3, N2, M2)**
- **Factors Required**: 5+ factors from 3+ categories, including:
  - Gitcoin Passport ≥35 OR VerifiedHumanity VC
  - Hardware key OR biometric with liveness
  - Social recovery (5 guardians, 3-of-5 threshold)
- **Capabilities**: Vote on Global DAO proposals, serve as guardian, run for delegate
- **Restrictions**: None (full participation)
- **Use Case**: Governance participants, delegates, core contributors

### **Level 4: Constitutionally Critical (E3, N3, M3)**
- **Factors Required**: All of:
  - Hardware key (2+ registered)
  - Biometric with liveness proof
  - Social recovery (7 guardians, 4-of-7)
  - KYC Basic VC (for legal accountability)
  - Gitcoin Passport ≥50
- **Capabilities**: Vote on constitutional amendments, serve on councils, manage treasury multisigs
- **Use Case**: Network stewards, council members, validators

---

## 🔄 Recovery Mechanisms

### **Scenario 1: Single Factor Loss (e.g., Lost Phone)**
**Trigger**: User still has access to 2+ other factors
**Process**:
1. Authenticate with remaining factors (e.g., hardware key + biometric)
2. Issue new primary key pair
3. Migrate identity to new device
4. Revoke lost factor
5. Update DHT with new key binding

**Time**: Immediate (<5 minutes)
**Assurance Level**: Maintained

### **Scenario 2: Multi-Factor Loss (e.g., Lost Phone + Hardware Key)**
**Trigger**: User has <2 accessible factors, but has guardians
**Process**:
1. User contacts guardians out-of-band
2. Guardians receive recovery request on DHT
3. Threshold guardians approve (e.g., 3 of 5)
4. Guardians submit their Shamir shares
5. Recovery seed reconstructed
6. New primary key pair generated from seed
7. User re-enrolls lost factors

**Time**: 24-72 hours (guardian response time)
**Assurance Level**: Drops to E2 temporarily, recovers after re-enrollment

### **Scenario 3: Catastrophic Loss (All Factors Lost, No Guardians)**
**Trigger**: User has nothing left
**Process** (High-friction by design):
1. User creates new identity (starts at E0)
2. User requests "Identity Continuity Petition" on DHT
3. Provides old DID + maximum known information
4. Knowledge Council reviews:
   - Historical activity patterns (MATL behavioral analysis)
   - Reputation attestations from peers who knew old identity
   - Any recoverable encrypted backups (e.g., stored on IPFS)
5. If approved (⅔ vote), old reputation/assets transferred
6. If rejected, user starts fresh

**Time**: 2-4 weeks (review process)
**Assurance Level**: Starts at E0, must rebuild
**Success Rate**: ~30% (intentionally difficult to prevent abuse)

### **Scenario 4: Planned Migration (Device Upgrade)**
**Trigger**: User wants to move to new device proactively
**Process**:
1. Generate new key pair on new device
2. Sign "Key Rotation Certificate" with old key
3. Authenticate with all available factors
4. Gradual transition (both keys valid for 30 days)
5. Old key deprecated after transition period

**Time**: Immediate start, 30-day transition
**Assurance Level**: Maintained or improved

---

## 🧮 MATL Integration

### Identity Factors as Trust Inputs

The Multi-Factor Identity system feeds directly into MATL's composite trust scoring:

```rust
pub struct IdentityTrustScore {
    // Base score from assurance level
    pub assurance_level_score: f64,  // 0.0-1.0

    // Factor diversity bonus
    pub factor_diversity_score: f64,  // More categories = higher

    // Factor quality score
    pub factor_quality_score: f64,  // Hardware key > password

    // Recovery preparedness
    pub recovery_readiness_score: f64,  // Guardians configured = higher

    // Longevity score
    pub identity_age_score: f64,  // Older = more trusted

    // Verification freshness
    pub verification_recency_score: f64,  // Recent = better

    // Social proof score
    pub attestation_score: f64,  // From reputation attestations

    // Behavioral consistency
    pub behavioral_consistency_score: f64,  // MATL analyzes patterns
}

// Composite identity trust score (feeds into MATL's TCDM component)
pub fn calculate_identity_trust(identity: &MycelixIdentity) -> f64 {
    let weights = IdentityTrustWeights {
        assurance_level: 0.30,
        factor_diversity: 0.15,
        factor_quality: 0.15,
        recovery_readiness: 0.10,
        identity_age: 0.10,
        verification_recency: 0.10,
        attestation_score: 0.05,
        behavioral_consistency: 0.05,
    };

    // Weighted sum
    weights.assurance_level * identity.mfa_state.assurance_level_score()
        + weights.factor_diversity * calculate_factor_diversity(&identity)
        + weights.factor_quality * calculate_factor_quality(&identity)
        + weights.recovery_readiness * calculate_recovery_readiness(&identity)
        + weights.identity_age * calculate_identity_age(&identity)
        + weights.verification_recency * calculate_verification_recency(&identity)
        + weights.attestation_score * calculate_attestation_score(&identity)
        + weights.behavioral_consistency * calculate_behavioral_consistency(&identity)
}
```

### Sybil Attack Mitigation

MATL uses identity factors to detect Sybil attacks:

1. **Factor Correlation Analysis**: Cluster identities with similar factor patterns
2. **Guardian Graph Analysis**: Detect circular guardian relationships
3. **Temporal Anomaly Detection**: Flag mass account creation
4. **Gitcoin Passport Clustering**: Group low-score accounts
5. **Behavioral Fingerprinting**: Analyze interaction patterns

**Quarantine Protocol**: Identities flagged as potential Sybils:
- Assurance level capped at E1 (no voting)
- Trust score capped at 0.3
- Manual review required for elevation
- Reputation earning rate reduced 90%

---

## 🤖 Identity-Gated Federated Learning Participation

Identity verification is the gateway to FL participation. This prevents Sybil attacks on the learning process and ensures gradient quality.

### FL Participation Requirements

```rust
/// Requirements for participating in federated learning rounds
pub struct FLParticipationRequirements {
    /// Minimum assurance level required
    pub min_assurance_level: AssuranceLevel,

    /// Required factor categories (at least one from each)
    pub required_factor_categories: Vec<FactorCategory>,

    /// Minimum reputation score from MATL
    pub min_reputation_score: f32,

    /// Whether humanity proof is mandatory
    pub humanity_proof_required: bool,

    /// Minimum factor freshness (effective strength)
    pub min_factor_freshness: f32,

    /// Whether delegated participation is allowed
    pub allow_delegation: bool,

    /// Maximum rounds per identity per epoch
    pub max_rounds_per_epoch: Option<u64>,
}

/// Default requirements by FL task sensitivity
impl FLParticipationRequirements {
    /// Standard training tasks (majority of FL)
    pub fn standard() -> Self {
        Self {
            min_assurance_level: AssuranceLevel::E2,
            required_factor_categories: vec![
                FactorCategory::Cryptographic,
                FactorCategory::ExternalVerification,
            ],
            min_reputation_score: 0.4,
            humanity_proof_required: false,
            min_factor_freshness: 0.5,
            allow_delegation: true,
            max_rounds_per_epoch: None,
        }
    }

    /// Sensitive model training (medical, financial)
    pub fn sensitive() -> Self {
        Self {
            min_assurance_level: AssuranceLevel::E3,
            required_factor_categories: vec![
                FactorCategory::Cryptographic,
                FactorCategory::Biometric,
                FactorCategory::ExternalVerification,
            ],
            min_reputation_score: 0.7,
            humanity_proof_required: true,
            min_factor_freshness: 0.7,
            allow_delegation: false,
            max_rounds_per_epoch: Some(100),
        }
    }

    /// Critical infrastructure models
    pub fn critical() -> Self {
        Self {
            min_assurance_level: AssuranceLevel::E4,
            required_factor_categories: vec![
                FactorCategory::Cryptographic,
                FactorCategory::Biometric,
                FactorCategory::SocialProof,
                FactorCategory::ExternalVerification,
            ],
            min_reputation_score: 0.85,
            humanity_proof_required: true,
            min_factor_freshness: 0.9,
            allow_delegation: false,
            max_rounds_per_epoch: Some(50),
        }
    }
}
```

### FL Admission Gate

```rust
/// Result of FL participation eligibility check
pub struct FLAdmissionResult {
    pub eligible: bool,
    pub identity_did: String,
    pub effective_assurance: AssuranceLevel,
    pub trust_score: f32,
    pub denial_reasons: Vec<FLDenialReason>,
    pub capabilities: FLCapabilities,
}

pub enum FLDenialReason {
    InsufficientAssuranceLevel { required: AssuranceLevel, actual: AssuranceLevel },
    MissingFactorCategory(FactorCategory),
    LowReputationScore { required: f32, actual: f32 },
    StaleFactors { min_freshness: f32, actual: f32 },
    HumanityProofRequired,
    DelegationNotAllowed,
    RoundLimitExceeded { limit: u64, used: u64 },
    IdentityQuarantined,
    ByzantineFlagged { detection_score: f32 },
}

/// Capabilities granted upon admission
pub struct FLCapabilities {
    /// Can submit gradients
    pub can_submit_gradients: bool,
    /// Maximum gradient size (bytes)
    pub max_gradient_size: usize,
    /// Eligible model types
    pub eligible_models: Vec<ModelType>,
    /// Rate limit (submissions per minute)
    pub rate_limit: u32,
    /// Whether this participant can be an aggregator
    pub can_aggregate: bool,
    /// Whether this participant can validate others
    pub can_validate: bool,
}

/// Check if identity can participate in FL
pub async fn check_fl_eligibility(
    identity: &MycelixIdentity,
    requirements: &FLParticipationRequirements,
    matl_client: &MATLClient,
) -> FLAdmissionResult {
    let mut denial_reasons = Vec::new();

    // 1. Check assurance level (with freshness decay)
    let (effective_level, _) = calculate_effective_assurance(identity, Timestamp::now());
    if effective_level < requirements.min_assurance_level {
        denial_reasons.push(FLDenialReason::InsufficientAssuranceLevel {
            required: requirements.min_assurance_level,
            actual: effective_level,
        });
    }

    // 2. Check required factor categories
    let available_categories: HashSet<_> = identity
        .mfa_state
        .factors
        .iter()
        .filter(|f| f.freshness().current_strength(Timestamp::now()) >= requirements.min_factor_freshness)
        .map(|f| f.category())
        .collect();

    for required in &requirements.required_factor_categories {
        if !available_categories.contains(required) {
            denial_reasons.push(FLDenialReason::MissingFactorCategory(*required));
        }
    }

    // 3. Check MATL reputation score
    let trust_score = matl_client.get_trust_score(&identity.did).await;
    if trust_score < requirements.min_reputation_score {
        denial_reasons.push(FLDenialReason::LowReputationScore {
            required: requirements.min_reputation_score,
            actual: trust_score,
        });
    }

    // 4. Check humanity proof if required
    if requirements.humanity_proof_required && !identity.has_humanity_proof() {
        denial_reasons.push(FLDenialReason::HumanityProofRequired);
    }

    // 5. Check Byzantine flags from previous rounds
    if let Some(byzantine_score) = matl_client.get_byzantine_score(&identity.did).await {
        if byzantine_score > 0.3 {
            denial_reasons.push(FLDenialReason::ByzantineFlagged {
                detection_score: byzantine_score,
            });
        }
    }

    let eligible = denial_reasons.is_empty();

    FLAdmissionResult {
        eligible,
        identity_did: identity.did.clone(),
        effective_assurance: effective_level,
        trust_score,
        denial_reasons,
        capabilities: if eligible {
            calculate_capabilities(effective_level, trust_score)
        } else {
            FLCapabilities::none()
        },
    }
}
```

### Identity-Based Byzantine Detection Integration

FL participants' identities are tracked for Byzantine behavior:

```rust
/// Links FL participation to identity for accountability
pub struct FLParticipationRecord {
    pub identity_did: String,
    pub round_number: u64,
    pub gradient_hash: String,
    pub submission_time: Timestamp,
    pub pogq_score: f32,  // Post-Optimum Gradient Quality
    pub byzantine_classification: Option<ByzantineClassification>,
}

/// Byzantine behavior affects identity trust score
pub async fn report_byzantine_behavior(
    identity_did: &str,
    classification: ByzantineClassification,
    evidence: ByzantineEvidence,
    matl_client: &MATLClient,
) {
    // Record in MATL for trust score adjustment
    matl_client.report_negative_behavior(
        identity_did,
        BehaviorType::ByzantineFL,
        classification.severity(),
        evidence,
    ).await;

    // Severe violations trigger identity quarantine
    if classification.severity() >= Severity::High {
        matl_client.quarantine_identity(identity_did, Duration::days(7)).await;
    }
}
```

### FL Participation Incentives

Identity-verified participation earns reputation and credits:

| Action | Credit Earned | Reputation Impact |
|--------|---------------|-------------------|
| Submit valid gradient | 10 | +0.001 |
| High-quality gradient (POGQ > 0.9) | 25 | +0.005 |
| Detect Byzantine peer | 50 | +0.01 |
| Complete training round | 5 | +0.0005 |
| Serve as aggregator | 100 | +0.02 |

### Zero-Knowledge FL Participation Proofs

Participants can prove FL contributions without revealing identity:

```rust
/// ZK proof of FL participation
pub struct FLParticipationProof {
    /// Proves identity meets requirements without revealing DID
    pub eligibility_proof: SNARKProof,
    /// Proves gradient was submitted without revealing content
    pub submission_proof: SNARKProof,
    /// Proves contribution quality without revealing score
    pub quality_proof: RangeProof,  // "POGQ >= 0.8"
    /// Anonymity set size
    pub anonymity_set: u32,
}
```

---

## 🔒 Privacy Guarantees

### Zero-Knowledge Identity Proofs

Users can prove identity properties without revealing identity:

**Example 1: Prove Age ≥18 Without Revealing Age**
```rust
pub struct AgeProof {
    pub claim: "age >= 18",
    pub zk_proof: SNARKProof,  // Proves claim without revealing age
    pub verifier_contract: Hash,
    pub expiry: Timestamp,
}
```

**Example 2: Prove VerifiedHumanity Without Revealing DID**
```rust
pub struct HumanityProof {
    pub claim: "possesses VerifiedHumanity VC",
    pub zk_proof: BulletProof,  // Ring signature over all VerifiedHumanity holders
    pub anonymity_set_size: u32,  // Size of ring
}
```

**Example 3: Prove Reputation ≥0.7 Without Revealing Exact Score**
```rust
pub struct ReputationProof {
    pub claim: "reputation >= 0.7",
    pub zk_proof: RangeProof,  // Bulletproof range proof
    pub timestamp: Timestamp,
}
```

### Differential Privacy for Governance

When voting on high-stakes proposals, MATL applies Adaptive DP:
- Identity verification proved via ZK
- Vote encrypted until threshold aggregation
- Individual votes never revealed
- Only aggregate outcome published

---

## 📊 Epistemic Classification

Identity claims are classified using the Epistemic Cube (LEM v2.0):

| Claim | E-Tier | N-Tier | M-Tier | Example |
|-------|--------|--------|--------|---------|
| "I created this DID" | E3 | N0 | M3 | Cryptographically signed |
| "I am human" (self-attested) | E1 | N0 | M1 | Testimonial only |
| "I am human" (VerifiedHumanity VC) | E3 | N1 | M2 | VC signed by issuer |
| "I am ≥18 years old" (ZK proof) | E3 | N0 | M1 | Cryptographic proof |
| "I am a trusted member" (reputation) | E2 | N1 | M2 | Audit Guild attests via MATL |
| "I am authorized to act for DAO X" | E3 | N2 | M3 | VC from DAO multisig |

---

## 🗺️ Implementation Roadmap

### **Phase 1: Core Identity (Q4 2025)** ✅ Target
**Goal**: Minimum viable multi-factor identity for Phase 1 launch

**Deliverables**:
- ✅ W3C DID implementation (`did:mycelix:`)
- ✅ Primary key pair factor (Ed25519)
- ✅ Gitcoin Passport integration (≥20 score requirement)
- ✅ Basic VerifiedHumanity VC support
- ✅ Assurance Levels 0-2 functional
- ✅ Simple recovery (2-of-3 guardians)
- ✅ MATL identity trust scoring

**Complexity**: Medium
**Timeline**: 6-8 weeks
**Team**: 2 developers + 1 security auditor

### **Phase 2: Enhanced Verification (Q1 2026)**
**Goal**: Full multi-factor support and advanced recovery

**Deliverables**:
- Hardware security key support (FIDO2/U2F)
- Biometric hash enrollment (with liveness detection)
- Full Shamir Secret Sharing recovery (5-of-7)
- Reputation attestation system
- Assurance Levels 3-4 functional
- ZK identity proofs (age, membership)

**Complexity**: High
**Timeline**: 8-10 weeks

### **Phase 3: Privacy & Governance (Q2 2026)**
**Goal**: Constitutional-grade identity for governance

**Deliverables**:
- Full Verifiable Credentials framework
- ZK-proof voting integration
- Adaptive DP for governance votes
- Identity Continuity Petition system
- KYC VC integration (for legal roles)
- Behavioral consistency analysis (MATL)

**Complexity**: Very High
**Timeline**: 10-12 weeks

### **Phase 4: Advanced Features (Q3 2026+)**
**Goal**: Future-proof identity system

**Deliverables**:
- Post-quantum cryptography migration
- Multi-device identity synchronization
- Delegated authority framework (act on behalf of)
- Cross-chain identity portability
- AI agent identity verification
- Biometric behavioral patterns (typing, gait)

**Complexity**: Research-grade
**Timeline**: Ongoing

---

## 🧪 Testing Strategy

### Unit Tests (Target: 95% Coverage)
- Factor enrollment/removal
- Assurance level calculation
- Recovery mechanisms
- Trust score computation
- ZK proof generation/verification

### Integration Tests
- End-to-end identity creation
- Multi-factor authentication flows
- Guardian-based recovery
- MATL integration
- DHT persistence

### Security Tests
- Sybil attack simulation (1000+ fake identities)
- Factor compromise scenarios
- Guardian collusion attacks
- Replay attack prevention
- ZK proof soundness verification

### Performance Tests
- 10,000+ concurrent identity creations
- Factor verification latency (<100ms target)
- Recovery request processing
- Trust score calculation speed

### User Experience Tests
- Onboarding friction measurement
- Recovery success rates
- Time-to-verified metrics
- Mobile device compatibility

---

## 🛡️ Security Considerations

### Attack Vectors & Mitigations

**1. Sybil Attack**
- **Threat**: Mass fake identity creation
- **Mitigation**: Gitcoin Passport + MATL ML detection + Guardian graph analysis
- **Severity**: High → Medium (with mitigations)

**2. Guardian Collusion**
- **Threat**: Guardians cooperate to steal identity
- **Mitigation**:
  - Threshold cryptography (need k-of-n, k > 50%)
  - Guardian trust scores (MATL)
  - Time-delayed recovery (24hr minimum)
  - Original identity holder notification
- **Severity**: Medium

**3. Biometric Spoofing**
- **Threat**: Fake biometric to impersonate user
- **Mitigation**:
  - Liveness detection mandatory
  - Multiple biometric types
  - Biometric used only as 1 factor (never sole factor)
- **Severity**: Medium → Low

**4. Hardware Key Cloning**
- **Threat**: Physical theft + cloning of hardware key
- **Mitigation**:
  - Counter-based authentication (prevents cloning)
  - Device attestation certificates
  - Rate limiting (3 failed attempts = lockout)
- **Severity**: Low

**5. Social Engineering**
- **Threat**: Attacker tricks user into recovery
- **Mitigation**:
  - High-friction recovery process
  - Guardian verification required
  - Knowledge Council review for catastrophic recovery
  - Clear user notifications
- **Severity**: Medium

**6. Quantum Computer Attack**
- **Threat**: Future quantum computers break Ed25519
- **Mitigation**:
  - Post-quantum migration path planned (Phase 4)
  - Hybrid signatures (classical + PQ) available
  - Forward secrecy for all factors
- **Severity**: Future (10+ years)

---

## 📜 Compliance & Legal

### GDPR Compliance
- **Right to Erasure**: Users can request identity deletion
  - Source chain entries pruned
  - DHT references removed
  - Only hashes remain (irreversible)
- **Data Minimization**: No unnecessary PII stored
- **Purpose Limitation**: Identity used only for access control + reputation
- **Transparency**: Full audit trail of factor additions/removals

### KYC/AML (For High-Stakes Roles)
- Optional KYC VC for constitutional roles
- Issued by regulated third-party
- Only proves "KYC Complete", not identity details
- Stored as Verifiable Credential, revocable

### Legal Accountability
- Constitutional roles require E3 identity
- Dispute resolution can subpoena identity proof
- Member Redress Council can audit identity in fraud cases
- Still preserves privacy for most users

---

## 🎓 User Education

### Onboarding Flow (Progressive Disclosure)

**Step 1: Anonymous Browse** (E0)
```
Welcome to Mycelix! 👋

You're currently browsing anonymously.
You can explore public content without signing up.

Want to participate? Create an identity →
```

**Step 2: Basic Identity** (E1)
```
Create Your Identity

We'll generate a secure key pair for you.
This is stored only on your device. 🔐

[ Generate Key Pair ]

What's a key pair? (Learn More →)
```

**Step 3: Add Verification** (E1 → E2)
```
Increase Your Verification Level

Current: Basic (E1)
Capabilities: Read, Comment
Next: Verified (E2)
New Capabilities: Vote, Earn Reputation

Add 2 more verification factors:
☐ Gitcoin Passport (Recommended)
☐ Hardware Key
☐ Social Recovery Guardians
☐ Biometric

Why verify? (Learn More →)
```

**Step 4: Recovery Setup** (Critical)
```
Protect Your Identity 🛡️

If you lose your device, you'll need a way to recover.

Recommended: Add 3-5 recovery guardians
These are trusted friends who can help you recover.

[ Add Guardians ] [ Skip (Not Recommended) ]

How does recovery work? (Learn More →)
```

---

## 🔮 Future Enhancements

### 1. Behavioral Biometrics
- Typing rhythm analysis
- Mouse movement patterns
- Voice cadence recognition
- Continuous authentication (not just login)

### 2. Delegated Authority
- "I authorize Agent X to act on my behalf"
- Time-bounded delegation
- Scope-limited permissions
- Cryptographic proof of delegation

### 3. Cross-Chain Identity
- Bridge identities to Ethereum, Cosmos, Polkadot
- Unified DID across chains
- Portable reputation

### 4. AI Agent Verification
- Prove AI capabilities without revealing model
- Accountability chain to human operator
- Resource consumption tracking

### 5. Quantum-Resistant Upgrade
- Migrate to post-quantum signatures (Dilithium, Falcon)
- Quantum-safe key exchange
- Hybrid classical+PQ mode

---

## 📈 Success Metrics

### Phase 1 Goals (Q4 2025)
- ✅ 1000+ users onboarded
- ✅ 90% reach E2 (Verified) within 30 days
- ✅ 95% recovery success rate
- ✅ <5% false Sybil detection rate
- ✅ <100ms average factor verification latency
- ✅ 0 catastrophic security breaches

### Phase 2 Goals (Q1 2026)
- 10,000+ users
- 50% reach E3 (Highly Assured)
- 5+ hardware key types supported
- 98% recovery success rate
- <50ms average verification latency

### Phase 3 Goals (Q2 2026)
- 100,000+ users
- 5% reach E4 (Constitutionally Critical)
- Full ZK-proof voting deployed
- 99.5% recovery success rate
- 0.1% Sybil detection false positive rate

---

## 🤝 Dependencies

### External Systems
- **Holochain DHT**: Identity storage and retrieval
- **MATL**: Trust scoring and Sybil detection
- **Gitcoin Passport API**: Sybil resistance scoring
- **W3C DID Resolver**: Standard DID resolution
- **VC Issuers**: Third-party credential issuance

### Internal Systems
- **Layer 1 (DHT)**: Persistent identity storage
- **Layer 6 (MATL)**: Trust scoring input
- **Layer 7 (Governance)**: Voting weight calculation
- **Layer 2 (DKG)**: Epistemic claim storage

---

## 💰 Cost Estimate

### Development Costs (Phase 1)
- **Core Development**: 320 hours @ $150/hr = $48,000
- **Security Audit**: 80 hours @ $200/hr = $16,000
- **UX Design**: 40 hours @ $100/hr = $4,000
- **Testing**: 80 hours @ $100/hr = $8,000
- **Total Phase 1**: **$76,000**

### Operational Costs (Annual)
- **Gitcoin Passport API**: $5,000/year (10K users)
- **Infrastructure**: $12,000/year (DHT, monitoring)
- **Security Audits**: $20,000/year (quarterly reviews)
- **Legal Compliance**: $10,000/year (GDPR, etc.)
- **Total Annual**: **$47,000**

### Revenue Potential (MATL Licensing)
- **Enterprise MATL License**: $50K-$200K per customer
- **Target**: 5 customers by end 2026
- **Revenue**: $250K-$1M/year

**Net**: Revenue positive by Q4 2026

---

## 🎯 Next Steps

### Immediate Actions (Week 1-2)
1. ✅ Review this design document with team
2. ✅ Get constitutional alignment confirmation
3. ✅ Prototype W3C DID structure
4. ✅ Set up development environment
5. ✅ Create GitHub issues for each component

### Short-Term (Week 3-8)
6. Implement core DID creation/resolution
7. Integrate Gitcoin Passport verification
8. Build factor enrollment framework
9. Create assurance level calculation
10. Implement basic guardian recovery

### Medium-Term (Week 9-16)
11. Add hardware key support
12. Implement ZK identity proofs
13. Integrate with MATL trust scoring
14. Build catastrophic recovery system
15. Complete Phase 1 security audit

---

**Document Status**: Design Complete ✅
**Next Review**: Weekly during development
**Maintainer**: Identity Working Group
**Constitutional Alignment**: Verified ✅

**Version**: 1.1
**Published**: November 2025
**Updated**: January 2026

### Version 1.1 Changes
- Added **Hierarchical Identity** (Individual, Delegated, Collective scopes)
- Added **Factor Freshness & Decay** mechanism with time-based strength calculation
- Added **Identity-Gated FL Participation** section for FL system integration
- Added FL participation requirements, admission gate, and Byzantine accountability
- Added ZK proofs for anonymous FL contribution verification

**Related Documents**:
- [Epistemic Charter v2.0](./THE%20EPISTEMIC%20CHARTER%20(v2.0).md)
- [System Architecture v5.2](./Mycelix%20Protocol_%20Integrated%20System%20Architecture%20v5.2.md)
- [MATL Whitepaper](../0TML/docs/06-architecture/matl_whitepaper.md)

---

📍 **Navigation**: [← System Architecture](./Mycelix%20Protocol_%20Integrated%20System%20Architecture%20v5.2.md) | [↑ Architecture Index](../README.md) | [Implementation Plan →](#-implementation-roadmap)
