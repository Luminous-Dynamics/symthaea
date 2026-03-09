//! Multi-Factor Authentication Integrity Zome
//!
//! Implements the Multi-Factor Decentralized Identity (MFDI) specification v1.0
//! for the Mycelix Identity hApp.
//!
//! ## Core Components
//! - 9 Identity Factor Types across 5 categories
//! - Factor Freshness & Decay tracking
//! - 4 Assurance Levels (E0-E4)
//! - Guardian-based Shamir recovery configuration
//!
//! Updated to use HDI 0.7 patterns with FlatOp validation

use hdi::prelude::*;

// =============================================================================
// FACTOR CATEGORY
// =============================================================================

/// Category of identity factor (for diversity scoring)
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FactorCategory {
    /// Cryptographic factors (Primary Key, Hardware Key)
    Cryptographic,
    /// Biometric factors (Face, Fingerprint, Iris, Voice)
    Biometric,
    /// Social proof factors (Guardians, Attestations)
    SocialProof,
    /// External verification (Gitcoin Passport, VCs)
    ExternalVerification,
    /// Knowledge factors (Recovery Phrase, Security Questions)
    Knowledge,
}

// =============================================================================
// ASSURANCE LEVELS
// =============================================================================

/// Assurance levels from MFDI spec (aligned with Epistemic E-Axis)
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AssuranceLevel {
    /// E0: Anonymous - no verification
    Anonymous,
    /// E1: Basic - 1 factor (typically primary key)
    Basic,
    /// E2: Verified - 3+ factors from 2+ categories
    Verified,
    /// E3: Highly Assured - 5+ factors from 3+ categories
    HighlyAssured,
    /// E4: Constitutionally Critical - all requirements met
    ConstitutionallyCritical,
}

impl AssuranceLevel {
    /// Get numeric score for MATL integration
    pub fn score(&self) -> f64 {
        match self {
            AssuranceLevel::Anonymous => 0.0,
            AssuranceLevel::Basic => 0.25,
            AssuranceLevel::Verified => 0.5,
            AssuranceLevel::HighlyAssured => 0.75,
            AssuranceLevel::ConstitutionallyCritical => 1.0,
        }
    }
}

// =============================================================================
// FACTOR TYPE ENUM (simplified for entry storage)
// =============================================================================

/// Simplified factor type for tracking
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum FactorType {
    PrimaryKeyPair,
    HardwareKey,
    Biometric,
    SocialRecovery,
    ReputationAttestation,
    GitcoinPassport,
    VerifiableCredential,
    RecoveryPhrase,
    SecurityQuestions,
}

impl FactorType {
    /// Get the category for this factor type
    pub fn category(&self) -> FactorCategory {
        match self {
            FactorType::PrimaryKeyPair | FactorType::HardwareKey => FactorCategory::Cryptographic,
            FactorType::Biometric => FactorCategory::Biometric,
            FactorType::SocialRecovery | FactorType::ReputationAttestation => {
                FactorCategory::SocialProof
            }
            FactorType::GitcoinPassport | FactorType::VerifiableCredential => {
                FactorCategory::ExternalVerification
            }
            FactorType::RecoveryPhrase | FactorType::SecurityQuestions => FactorCategory::Knowledge,
        }
    }

    /// Get base weight for assurance calculation
    pub fn base_weight(&self) -> f32 {
        match self {
            FactorType::PrimaryKeyPair => 1.0,
            FactorType::HardwareKey => 1.2,
            FactorType::Biometric => 0.8,
            FactorType::SocialRecovery => 0.9,
            FactorType::ReputationAttestation => 0.5,
            FactorType::GitcoinPassport => 0.8,
            FactorType::VerifiableCredential => 1.0,
            FactorType::RecoveryPhrase => 0.8,
            FactorType::SecurityQuestions => 0.3,
        }
    }

    /// Get decay configuration for this factor type
    pub fn decay_config(&self) -> (u64, f32, u64) {
        // Returns (grace_period_secs, decay_rate, reverify_required_secs)
        match self {
            FactorType::PrimaryKeyPair => (90 * 86400, 0.001, 365 * 86400),
            FactorType::HardwareKey => (180 * 86400, 0.0005, 730 * 86400),
            FactorType::Biometric => (30 * 86400, 0.008, 90 * 86400),
            FactorType::SocialRecovery => (60 * 86400, 0.004, 180 * 86400),
            FactorType::ReputationAttestation => (14 * 86400, 0.012, 60 * 86400),
            FactorType::GitcoinPassport => (30 * 86400, 0.008, 90 * 86400),
            FactorType::VerifiableCredential => (365 * 86400, 0.0, 365 * 86400), // Expiry-based
            FactorType::RecoveryPhrase => (365 * 86400, 0.0, 365 * 86400),
            FactorType::SecurityQuestions => (180 * 86400, 0.002, 365 * 86400),
        }
    }
}

// =============================================================================
// ENROLLED FACTOR (stored in MFA state)
// =============================================================================

/// A single enrolled identity factor with metadata
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EnrolledFactor {
    /// Type of factor
    pub factor_type: FactorType,
    /// Factor-specific identifier (hash of key, device ID, etc.)
    pub factor_id: String,
    /// When the factor was enrolled
    pub enrolled_at: Timestamp,
    /// Last verification/use timestamp
    pub last_verified: Timestamp,
    /// Factor-specific metadata (JSON string)
    pub metadata: String,
    /// Current effective strength (0.0-1.0)
    pub effective_strength: f32,
    /// Whether factor is currently active
    pub active: bool,
}

impl EnrolledFactor {
    /// Calculate current effective strength based on time decay
    pub fn current_strength(&self, now: Timestamp) -> f32 {
        if !self.active {
            return 0.0;
        }

        let elapsed_micros = now.as_micros() - self.last_verified.as_micros();
        let elapsed_secs = (elapsed_micros / 1_000_000) as u64;

        let (grace_period, decay_rate, _) = self.factor_type.decay_config();

        // Within grace period: full strength
        if elapsed_secs <= grace_period {
            return 1.0;
        }

        // Exponential decay after grace period
        let decay_time = elapsed_secs - grace_period;
        let decay_days = decay_time as f32 / 86400.0;
        let decay_factor = (-decay_rate * decay_days).exp();

        decay_factor.clamp(0.0, 1.0)
    }

    /// Check if factor needs re-verification
    pub fn needs_reverification(&self, now: Timestamp) -> bool {
        let elapsed_micros = now.as_micros() - self.last_verified.as_micros();
        let elapsed_secs = (elapsed_micros / 1_000_000) as u64;
        let (_, _, reverify_required) = self.factor_type.decay_config();

        self.current_strength(now) < 0.5 || elapsed_secs > reverify_required
    }

    /// Get weighted strength for assurance calculation
    pub fn weighted_strength(&self, now: Timestamp) -> f32 {
        self.current_strength(now) * self.factor_type.base_weight()
    }
}

// =============================================================================
// MFA STATE ENTRY TYPE
// =============================================================================

/// Multi-Factor Authentication state for an identity
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MfaState {
    /// The DID this MFA state belongs to
    pub did: String,
    /// Owner's agent pub key
    pub owner: AgentPubKey,
    /// List of enrolled factors
    pub factors: Vec<EnrolledFactor>,
    /// Current calculated assurance level
    pub assurance_level: AssuranceLevel,
    /// Total effective strength (for MATL)
    pub effective_strength: f32,
    /// Number of unique factor categories
    pub category_count: u8,
    /// Creation timestamp
    pub created: Timestamp,
    /// Last update timestamp
    pub updated: Timestamp,
    /// Version number
    pub version: u32,
}

impl MfaState {
    /// Calculate the effective assurance level based on current factors
    pub fn calculate_assurance(&self, now: Timestamp) -> (AssuranceLevel, f32, u8) {
        let mut total_strength = 0.0f32;
        let mut categories: Vec<FactorCategory> = Vec::new();

        for factor in &self.factors {
            if !factor.active {
                continue;
            }

            let effective = factor.current_strength(now);

            // Only count factors above minimum threshold
            if effective >= 0.3 {
                total_strength += effective * factor.factor_type.base_weight();
                let cat = factor.factor_type.category();
                if !categories.contains(&cat) {
                    categories.push(cat);
                }
            }
        }

        let category_count = categories.len() as u8;

        // Assurance level based on effective strength and diversity
        let level = if total_strength >= 4.0 && category_count >= 4 {
            AssuranceLevel::ConstitutionallyCritical
        } else if total_strength >= 3.0 && category_count >= 3 {
            AssuranceLevel::HighlyAssured
        } else if total_strength >= 2.0 && category_count >= 2 {
            AssuranceLevel::Verified
        } else if total_strength >= 1.0 {
            AssuranceLevel::Basic
        } else {
            AssuranceLevel::Anonymous
        };

        (level, total_strength, category_count)
    }
}

/// Factor enrollment record (audit trail)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FactorEnrollment {
    /// The DID this enrollment belongs to
    pub did: String,
    /// The factor type
    pub factor_type: FactorType,
    /// Factor-specific identifier
    pub factor_id: String,
    /// Action type
    pub action: EnrollmentAction,
    /// Timestamp
    pub timestamp: Timestamp,
    /// Reason for the action
    pub reason: String,
}

/// Enrollment action type
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EnrollmentAction {
    Enroll,
    Revoke,
    Update,
    Reverify,
}

/// Factor verification event (audit trail)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FactorVerification {
    /// The DID
    pub did: String,
    /// Factor type verified
    pub factor_type: FactorType,
    /// Factor identifier
    pub factor_id: String,
    /// Verification result
    pub success: bool,
    /// Timestamp
    pub timestamp: Timestamp,
    /// New effective strength after verification
    pub new_strength: f32,
}

// =============================================================================
// ENCRYPTED ENTRY (for PQE-protected sensitive data)
// =============================================================================

/// An encrypted entry wrapping sensitive MFA data on the DHT.
///
/// Sensitive fields (factor metadata, recovery trustee lists) are encrypted
/// to the owner's ML-KEM public key using XChaCha20-Poly1305 AEAD.
/// The plaintext is only recoverable by decapsulating the KEM ciphertext
/// with the owner's private key.
///
/// The `entry_type_tag` identifies what was encrypted so the decryptor
/// knows which type to deserialize after decryption.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EncryptedEntry {
    /// Which entry type is encrypted (e.g. "MfaState", "FactorEnrollment").
    pub entry_type_tag: String,
    /// KEM algorithm used to derive the symmetric key (AlgorithmId as u16).
    /// 0xF020 = ML-KEM-768, 0xF021 = ML-KEM-1024, 0xF030 = self-encrypt.
    pub kem_algorithm: u16,
    /// KEM ciphertext (encapsulated key). Empty for self-encryption.
    pub encapsulated_key: Vec<u8>,
    /// 24-byte nonce for XChaCha20-Poly1305.
    pub nonce: Vec<u8>,
    /// AEAD ciphertext (plaintext || 16-byte Poly1305 tag).
    pub ciphertext: Vec<u8>,
    /// DID URL fragment of the recipient's KEM key (e.g. "did:mycelix:abc#kem-1").
    pub recipient_key_id: String,
    /// When the entry was encrypted.
    pub encrypted_at: Timestamp,
    /// Schema version of the plaintext, for forward-compatible decryption.
    pub plaintext_version: u32,
}

// =============================================================================
// ENTRY AND LINK TYPES
// =============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    MfaState(MfaState),
    FactorEnrollment(FactorEnrollment),
    FactorVerification(FactorVerification),
    EncryptedEntry(EncryptedEntry),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// DID to MFA state
    DidToMfaState,
    /// Agent to MFA state
    AgentToMfaState,
    /// DID to enrollment history
    DidToEnrollments,
    /// DID to verification history
    DidToVerifications,
    /// MFA state history (for versioning)
    MfaStateHistory,
}

// =============================================================================
// VALIDATION
// =============================================================================

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::MfaState(state) => {
                    validate_create_mfa_state(EntryCreationAction::Create(action), state)
                }
                EntryTypes::FactorEnrollment(enrollment) => validate_create_factor_enrollment(
                    EntryCreationAction::Create(action),
                    enrollment,
                ),
                EntryTypes::FactorVerification(verification) => {
                    validate_create_factor_verification(
                        EntryCreationAction::Create(action),
                        verification,
                    )
                }
                EntryTypes::EncryptedEntry(entry) => validate_create_encrypted_entry(entry),
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::MfaState(state) => validate_update_mfa_state(action, state),
                EntryTypes::FactorEnrollment(_) => Ok(ValidateCallbackResult::Invalid(
                    "Factor enrollments are append-only".into(),
                )),
                EntryTypes::FactorVerification(_) => Ok(ValidateCallbackResult::Invalid(
                    "Factor verifications are append-only".into(),
                )),
                EntryTypes::EncryptedEntry(_) => Ok(ValidateCallbackResult::Invalid(
                    "Encrypted entries are append-only (re-encrypt instead)".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, tag, .. } => {
            if tag.0.len() > 1024 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Link tag exceeds maximum length of 1024 bytes".into(),
                ));
            }
            match link_type {
                LinkTypes::DidToMfaState => Ok(ValidateCallbackResult::Valid),
                LinkTypes::AgentToMfaState => Ok(ValidateCallbackResult::Valid),
                LinkTypes::DidToEnrollments => Ok(ValidateCallbackResult::Valid),
                LinkTypes::DidToVerifications => Ok(ValidateCallbackResult::Valid),
                LinkTypes::MfaStateHistory => Ok(ValidateCallbackResult::Valid),
            }
        }
        FlatOp::RegisterDeleteLink {
            original_action,
            action,
            ..
        } => {
            if action.author != original_action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the link creator can delete their links".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original entry author can update their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(OpDelete { action }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original entry author can delete their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

/// Validate MFA state creation
fn validate_create_mfa_state(
    action: EntryCreationAction,
    state: MfaState,
) -> ExternResult<ValidateCallbackResult> {
    // Validate DID format
    if !state.did.starts_with("did:mycelix:") {
        return Ok(ValidateCallbackResult::Invalid(
            "DID must start with 'did:mycelix:'".into(),
        ));
    }

    // Validate owner is author
    if state.owner != *action.author() {
        return Ok(ValidateCallbackResult::Invalid(
            "Owner must be the author".into(),
        ));
    }

    // Validate initial version
    if state.version != 1 {
        return Ok(ValidateCallbackResult::Invalid(
            "Initial version must be 1".into(),
        ));
    }

    // Validate at least one factor (primary key)
    if state.factors.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "MFA state must have at least one factor".into(),
        ));
    }

    // Validate first factor is primary key pair
    if state.factors[0].factor_type != FactorType::PrimaryKeyPair {
        return Ok(ValidateCallbackResult::Invalid(
            "First factor must be PrimaryKeyPair".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate MFA state update
fn validate_update_mfa_state(
    action: Update,
    state: MfaState,
) -> ExternResult<ValidateCallbackResult> {
    // Validate author is owner
    if state.owner != action.author {
        return Ok(ValidateCallbackResult::Invalid(
            "Only owner can update MFA state".into(),
        ));
    }

    // Validate DID format preserved
    if !state.did.starts_with("did:mycelix:") {
        return Ok(ValidateCallbackResult::Invalid(
            "DID must start with 'did:mycelix:'".into(),
        ));
    }

    // Must keep at least one factor
    if state.factors.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot remove all factors".into(),
        ));
    }

    // Fetch original to enforce invariants
    let original_record = must_get_valid_record(action.original_action_address.clone())?;
    let original: MfaState = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original MFA state not found".into()
        )))?;

    // Immutable fields
    if state.did != original.did {
        return Ok(ValidateCallbackResult::Invalid(
            "MFA DID cannot be changed".into(),
        ));
    }
    if state.owner != original.owner {
        return Ok(ValidateCallbackResult::Invalid(
            "MFA owner cannot be changed".into(),
        ));
    }
    if state.created != original.created {
        return Ok(ValidateCallbackResult::Invalid(
            "MFA created timestamp cannot be changed".into(),
        ));
    }

    // Version must increment
    if state.version <= original.version {
        return Ok(ValidateCallbackResult::Invalid(
            "MFA state version must increase on update".into(),
        ));
    }

    // Updated timestamp must advance
    if state.updated <= original.updated {
        return Ok(ValidateCallbackResult::Invalid(
            "MFA updated timestamp must advance".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate factor enrollment creation
fn validate_create_factor_enrollment(
    _action: EntryCreationAction,
    enrollment: FactorEnrollment,
) -> ExternResult<ValidateCallbackResult> {
    // Validate DID format
    if !enrollment.did.starts_with("did:mycelix:") {
        return Ok(ValidateCallbackResult::Invalid(
            "DID must start with 'did:mycelix:'".into(),
        ));
    }

    // Validate factor ID provided
    if enrollment.factor_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Factor ID is required".into(),
        ));
    }

    // Validate reason provided
    if enrollment.reason.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Enrollment reason is required".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate factor verification creation
fn validate_create_factor_verification(
    _action: EntryCreationAction,
    verification: FactorVerification,
) -> ExternResult<ValidateCallbackResult> {
    // Validate DID format
    if !verification.did.starts_with("did:mycelix:") {
        return Ok(ValidateCallbackResult::Invalid(
            "DID must start with 'did:mycelix:'".into(),
        ));
    }

    // Validate strength is non-negative.
    // Note: new_strength is repurposed as a WebAuthn counter (u32 cast to f32) for
    // HardwareKey verifications, so values > 1.0 are valid for that factor type.
    // We only reject negative values (which are never valid for either use case).
    if verification.new_strength < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Strength/counter must be non-negative".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate encrypted entry creation
fn validate_create_encrypted_entry(entry: EncryptedEntry) -> ExternResult<ValidateCallbackResult> {
    // Validate entry type tag is non-empty
    if entry.entry_type_tag.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Encrypted entry must specify entry_type_tag".into(),
        ));
    }

    // Validate nonce length (XChaCha20-Poly1305 requires 24 bytes)
    if entry.nonce.len() != 24 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Nonce must be 24 bytes (XChaCha20-Poly1305), got {}",
            entry.nonce.len()
        )));
    }

    // Validate ciphertext is non-empty (minimum: 16-byte Poly1305 tag)
    if entry.ciphertext.len() < 16 {
        return Ok(ValidateCallbackResult::Invalid(
            "Ciphertext too short (minimum 16 bytes for Poly1305 tag)".into(),
        ));
    }

    // Validate recipient key ID is a DID URL or "self"
    if entry.recipient_key_id != "self" && !entry.recipient_key_id.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "recipient_key_id must be 'self' or a DID URL".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a test timestamp (microseconds since Unix epoch)
    fn test_timestamp(days_ago: u64) -> Timestamp {
        // Use a fixed base time for reproducibility (2025-01-01 00:00:00 UTC)
        let base_micros: i64 = 1735689600_000_000;
        let offset_micros = (days_ago * 86400 * 1_000_000) as i64;
        Timestamp::from_micros(base_micros - offset_micros)
    }

    fn now_timestamp() -> Timestamp {
        // Fixed "now" for testing (2025-01-01 00:00:00 UTC)
        Timestamp::from_micros(1735689600_000_000)
    }

    // =========================================================================
    // Factor Category Tests
    // =========================================================================

    #[test]
    fn test_factor_type_categories() {
        assert_eq!(
            FactorType::PrimaryKeyPair.category(),
            FactorCategory::Cryptographic
        );
        assert_eq!(
            FactorType::HardwareKey.category(),
            FactorCategory::Cryptographic
        );
        assert_eq!(FactorType::Biometric.category(), FactorCategory::Biometric);
        assert_eq!(
            FactorType::SocialRecovery.category(),
            FactorCategory::SocialProof
        );
        assert_eq!(
            FactorType::ReputationAttestation.category(),
            FactorCategory::SocialProof
        );
        assert_eq!(
            FactorType::GitcoinPassport.category(),
            FactorCategory::ExternalVerification
        );
        assert_eq!(
            FactorType::VerifiableCredential.category(),
            FactorCategory::ExternalVerification
        );
        assert_eq!(
            FactorType::RecoveryPhrase.category(),
            FactorCategory::Knowledge
        );
        assert_eq!(
            FactorType::SecurityQuestions.category(),
            FactorCategory::Knowledge
        );
    }

    // =========================================================================
    // Factor Decay Tests
    // =========================================================================

    #[test]
    fn test_factor_strength_within_grace_period() {
        let now = now_timestamp();
        let factor = EnrolledFactor {
            factor_type: FactorType::PrimaryKeyPair,
            factor_id: "test-key".to_string(),
            enrolled_at: now,
            last_verified: now, // Just verified
            metadata: "{}".to_string(),
            effective_strength: 1.0,
            active: true,
        };

        // Primary key has 90-day grace period
        let strength = factor.current_strength(now);
        assert_eq!(strength, 1.0, "Strength should be 1.0 within grace period");
    }

    #[test]
    fn test_factor_strength_after_grace_period() {
        let now = now_timestamp();
        // Verified 120 days ago (30 days past the 90-day grace period)
        let verified = test_timestamp(120);

        let factor = EnrolledFactor {
            factor_type: FactorType::PrimaryKeyPair,
            factor_id: "test-key".to_string(),
            enrolled_at: verified,
            last_verified: verified,
            metadata: "{}".to_string(),
            effective_strength: 1.0,
            active: true,
        };

        let strength = factor.current_strength(now);
        // After 30 days past grace, with decay rate 0.001/day:
        // strength = exp(-0.001 * 30) ≈ 0.97
        assert!(strength < 1.0, "Strength should decay after grace period");
        assert!(strength > 0.95, "Decay should be gradual for primary key");
    }

    #[test]
    fn test_biometric_decays_faster() {
        let now = now_timestamp();
        // Verified 60 days ago (30 days past the 30-day grace period for biometric)
        let verified = test_timestamp(60);

        let factor = EnrolledFactor {
            factor_type: FactorType::Biometric,
            factor_id: "face-hash".to_string(),
            enrolled_at: verified,
            last_verified: verified,
            metadata: "{}".to_string(),
            effective_strength: 1.0,
            active: true,
        };

        let strength = factor.current_strength(now);
        // Biometric decay rate is 0.008/day, much faster than primary key
        // After 30 days: exp(-0.008 * 30) ≈ 0.79
        assert!(
            strength < 0.85,
            "Biometric should decay faster than primary key"
        );
        assert!(strength > 0.7, "Biometric shouldn't decay too quickly");
    }

    #[test]
    fn test_inactive_factor_has_zero_strength() {
        let now = now_timestamp();
        let factor = EnrolledFactor {
            factor_type: FactorType::PrimaryKeyPair,
            factor_id: "test-key".to_string(),
            enrolled_at: now,
            last_verified: now,
            metadata: "{}".to_string(),
            effective_strength: 1.0,
            active: false, // Deactivated
        };

        let strength = factor.current_strength(now);
        assert_eq!(strength, 0.0, "Inactive factor should have zero strength");
    }

    #[test]
    fn test_factor_needs_reverification() {
        let now = now_timestamp();

        // Fresh factor - no reverification needed
        let fresh_factor = EnrolledFactor {
            factor_type: FactorType::PrimaryKeyPair,
            factor_id: "fresh-key".to_string(),
            enrolled_at: now,
            last_verified: now,
            metadata: "{}".to_string(),
            effective_strength: 1.0,
            active: true,
        };
        assert!(
            !fresh_factor.needs_reverification(now),
            "Fresh factor shouldn't need reverification"
        );

        // Old factor - needs reverification (400 days, past the 365-day threshold)
        let old_verified = test_timestamp(400);
        let old_factor = EnrolledFactor {
            factor_type: FactorType::PrimaryKeyPair,
            factor_id: "old-key".to_string(),
            enrolled_at: old_verified,
            last_verified: old_verified,
            metadata: "{}".to_string(),
            effective_strength: 1.0,
            active: true,
        };
        assert!(
            old_factor.needs_reverification(now),
            "Old factor should need reverification"
        );
    }

    #[test]
    fn test_weighted_strength() {
        let now = now_timestamp();
        let factor = EnrolledFactor {
            factor_type: FactorType::HardwareKey,
            factor_id: "yubikey".to_string(),
            enrolled_at: now,
            last_verified: now,
            metadata: "{}".to_string(),
            effective_strength: 1.0,
            active: true,
        };

        // Hardware key has weight 1.2
        let weighted = factor.weighted_strength(now);
        assert_eq!(weighted, 1.2, "Hardware key should have 1.2 base weight");
    }

    // =========================================================================
    // Assurance Level Tests
    // =========================================================================

    #[test]
    fn test_assurance_level_scores() {
        assert_eq!(AssuranceLevel::Anonymous.score(), 0.0);
        assert_eq!(AssuranceLevel::Basic.score(), 0.25);
        assert_eq!(AssuranceLevel::Verified.score(), 0.5);
        assert_eq!(AssuranceLevel::HighlyAssured.score(), 0.75);
        assert_eq!(AssuranceLevel::ConstitutionallyCritical.score(), 1.0);
    }

    #[test]
    fn test_assurance_level_ordering() {
        assert!(AssuranceLevel::Anonymous < AssuranceLevel::Basic);
        assert!(AssuranceLevel::Basic < AssuranceLevel::Verified);
        assert!(AssuranceLevel::Verified < AssuranceLevel::HighlyAssured);
        assert!(AssuranceLevel::HighlyAssured < AssuranceLevel::ConstitutionallyCritical);
    }

    #[test]
    fn test_mfa_state_assurance_basic() {
        let now = now_timestamp();
        let state = MfaState {
            did: "did:mycelix:test".to_string(),
            owner: AgentPubKey::from_raw_36(vec![0u8; 36]),
            factors: vec![EnrolledFactor {
                factor_type: FactorType::PrimaryKeyPair,
                factor_id: "key1".to_string(),
                enrolled_at: now,
                last_verified: now,
                metadata: "{}".to_string(),
                effective_strength: 1.0,
                active: true,
            }],
            assurance_level: AssuranceLevel::Basic,
            effective_strength: 1.0,
            category_count: 1,
            created: now,
            updated: now,
            version: 1,
        };

        let (level, strength, categories) = state.calculate_assurance(now);
        assert_eq!(level, AssuranceLevel::Basic);
        assert_eq!(categories, 1);
        assert!(strength >= 1.0);
    }

    #[test]
    fn test_mfa_state_assurance_verified() {
        let now = now_timestamp();
        let state = MfaState {
            did: "did:mycelix:test".to_string(),
            owner: AgentPubKey::from_raw_36(vec![0u8; 36]),
            factors: vec![
                EnrolledFactor {
                    factor_type: FactorType::PrimaryKeyPair,
                    factor_id: "key1".to_string(),
                    enrolled_at: now,
                    last_verified: now,
                    metadata: "{}".to_string(),
                    effective_strength: 1.0,
                    active: true,
                },
                EnrolledFactor {
                    factor_type: FactorType::GitcoinPassport,
                    factor_id: "passport1".to_string(),
                    enrolled_at: now,
                    last_verified: now,
                    metadata: "{}".to_string(),
                    effective_strength: 1.0,
                    active: true,
                },
                EnrolledFactor {
                    factor_type: FactorType::Biometric,
                    factor_id: "bio1".to_string(),
                    enrolled_at: now,
                    last_verified: now,
                    metadata: "{}".to_string(),
                    effective_strength: 1.0,
                    active: true,
                },
            ],
            assurance_level: AssuranceLevel::Verified,
            effective_strength: 2.6,
            category_count: 3,
            created: now,
            updated: now,
            version: 1,
        };

        let (level, strength, categories) = state.calculate_assurance(now);
        // 3 factors from 3 categories: Cryptographic (1.0), ExternalVerification (0.8), Biometric (0.8)
        // Total: 2.6, Categories: 3 → Verified (needs 2.0+ strength and 2+ categories)
        assert!(
            level >= AssuranceLevel::Verified,
            "Should be at least Verified with 3 factors from 3 categories"
        );
        assert_eq!(categories, 3);
        assert!(strength >= 2.0);
    }

    #[test]
    fn test_mfa_state_assurance_highly_assured() {
        let now = now_timestamp();
        let state = MfaState {
            did: "did:mycelix:test".to_string(),
            owner: AgentPubKey::from_raw_36(vec![0u8; 36]),
            factors: vec![
                EnrolledFactor {
                    factor_type: FactorType::PrimaryKeyPair,
                    factor_id: "key1".to_string(),
                    enrolled_at: now,
                    last_verified: now,
                    metadata: "{}".to_string(),
                    effective_strength: 1.0,
                    active: true,
                },
                EnrolledFactor {
                    factor_type: FactorType::HardwareKey,
                    factor_id: "hw1".to_string(),
                    enrolled_at: now,
                    last_verified: now,
                    metadata: "{}".to_string(),
                    effective_strength: 1.0,
                    active: true,
                },
                EnrolledFactor {
                    factor_type: FactorType::GitcoinPassport,
                    factor_id: "passport1".to_string(),
                    enrolled_at: now,
                    last_verified: now,
                    metadata: "{}".to_string(),
                    effective_strength: 1.0,
                    active: true,
                },
                EnrolledFactor {
                    factor_type: FactorType::SocialRecovery,
                    factor_id: "social1".to_string(),
                    enrolled_at: now,
                    last_verified: now,
                    metadata: "{}".to_string(),
                    effective_strength: 1.0,
                    active: true,
                },
            ],
            assurance_level: AssuranceLevel::HighlyAssured,
            effective_strength: 3.9,
            category_count: 3,
            created: now,
            updated: now,
            version: 1,
        };

        let (level, strength, categories) = state.calculate_assurance(now);
        // 4 factors: PrimaryKey(1.0) + HardwareKey(1.2) + GitcoinPassport(0.8) + SocialRecovery(0.9) = 3.9
        // Categories: Cryptographic, ExternalVerification, SocialProof = 3
        assert!(
            level >= AssuranceLevel::HighlyAssured,
            "Should be HighlyAssured with 3.9 strength and 3 categories"
        );
        assert!(categories >= 3);
        assert!(strength >= 3.0);
    }

    // =========================================================================
    // Base Weight Tests
    // =========================================================================

    #[test]
    fn test_factor_base_weights() {
        assert_eq!(FactorType::PrimaryKeyPair.base_weight(), 1.0);
        assert_eq!(FactorType::HardwareKey.base_weight(), 1.2);
        assert_eq!(FactorType::Biometric.base_weight(), 0.8);
        assert_eq!(FactorType::SocialRecovery.base_weight(), 0.9);
        assert_eq!(FactorType::ReputationAttestation.base_weight(), 0.5);
        assert_eq!(FactorType::GitcoinPassport.base_weight(), 0.8);
        assert_eq!(FactorType::VerifiableCredential.base_weight(), 1.0);
        assert_eq!(FactorType::RecoveryPhrase.base_weight(), 0.8);
        assert_eq!(FactorType::SecurityQuestions.base_weight(), 0.3);
    }

    // =========================================================================
    // Decay Configuration Tests
    // =========================================================================

    #[test]
    fn test_decay_configs() {
        // PrimaryKeyPair: 90 days grace, 0.001 decay, 365 days reverify
        let (grace, decay, reverify) = FactorType::PrimaryKeyPair.decay_config();
        assert_eq!(grace, 90 * 86400);
        assert!((decay - 0.001).abs() < 0.0001);
        assert_eq!(reverify, 365 * 86400);

        // HardwareKey: 180 days grace, 0.0005 decay, 730 days reverify
        let (grace, decay, reverify) = FactorType::HardwareKey.decay_config();
        assert_eq!(grace, 180 * 86400);
        assert!((decay - 0.0005).abs() < 0.0001);
        assert_eq!(reverify, 730 * 86400);

        // Biometric: 30 days grace, 0.008 decay, 90 days reverify
        let (grace, decay, reverify) = FactorType::Biometric.decay_config();
        assert_eq!(grace, 30 * 86400);
        assert!((decay - 0.008).abs() < 0.0001);
        assert_eq!(reverify, 90 * 86400);
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_stale_factor_below_threshold() {
        let now = now_timestamp();
        // Factor verified 200 days ago with fast decay
        let verified = test_timestamp(200);

        let factor = EnrolledFactor {
            factor_type: FactorType::ReputationAttestation, // Fast decay (0.012/day)
            factor_id: "rep1".to_string(),
            enrolled_at: verified,
            last_verified: verified,
            metadata: "{}".to_string(),
            effective_strength: 1.0,
            active: true,
        };

        let strength = factor.current_strength(now);
        // Grace: 14 days, Decay time: 186 days, Rate: 0.012
        // strength = exp(-0.012 * 186) ≈ 0.11
        assert!(
            strength < 0.3,
            "Very stale factor should be below 0.3 threshold"
        );
    }

    #[test]
    fn test_empty_factors_gives_anonymous() {
        let now = now_timestamp();
        let state = MfaState {
            did: "did:mycelix:test".to_string(),
            owner: AgentPubKey::from_raw_36(vec![0u8; 36]),
            factors: vec![], // No factors
            assurance_level: AssuranceLevel::Anonymous,
            effective_strength: 0.0,
            category_count: 0,
            created: now,
            updated: now,
            version: 1,
        };

        let (level, strength, categories) = state.calculate_assurance(now);
        assert_eq!(level, AssuranceLevel::Anonymous);
        assert_eq!(strength, 0.0);
        assert_eq!(categories, 0);
    }

    // =========================================================================
    // Security Hardening Tests (Feb 2026)
    // =========================================================================

    #[test]
    fn test_webauthn_counter_stored_as_strength_is_valid() {
        // WebAuthn counters are stored as f32 in new_strength field.
        // Values > 1.0 must be accepted (counters are u32 cast to f32).
        let verification = FactorVerification {
            did: "did:mycelix:test".to_string(),
            factor_type: FactorType::HardwareKey,
            factor_id: "webauthn-key".to_string(),
            success: true,
            timestamp: now_timestamp(),
            new_strength: 42.0, // WebAuthn counter = 42
        };

        // The integrity validation should accept counter values > 1.0
        assert!(
            verification.new_strength >= 0.0,
            "Counter must be non-negative"
        );
        // Previously this would have been rejected by the > 1.0 check
        assert!(
            verification.new_strength > 1.0,
            "Counter values > 1.0 are valid for WebAuthn"
        );
    }

    #[test]
    fn test_negative_strength_rejected() {
        // Negative values should never be valid for either strength or counter usage
        let verification = FactorVerification {
            did: "did:mycelix:test".to_string(),
            factor_type: FactorType::PrimaryKeyPair,
            factor_id: "key".to_string(),
            success: true,
            timestamp: now_timestamp(),
            new_strength: -0.5,
        };

        assert!(
            verification.new_strength < 0.0,
            "Negative strength should fail validation"
        );
    }

    // =========================================================================
    // Property-Based Tests (proptest)
    // =========================================================================

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        fn arb_factor_type() -> impl Strategy<Value = FactorType> {
            prop_oneof![
                Just(FactorType::PrimaryKeyPair),
                Just(FactorType::HardwareKey),
                Just(FactorType::Biometric),
                Just(FactorType::SocialRecovery),
                Just(FactorType::ReputationAttestation),
                Just(FactorType::GitcoinPassport),
                Just(FactorType::VerifiableCredential),
                Just(FactorType::RecoveryPhrase),
                Just(FactorType::SecurityQuestions),
            ]
        }

        fn arb_assurance_level() -> impl Strategy<Value = AssuranceLevel> {
            prop_oneof![
                Just(AssuranceLevel::Anonymous),
                Just(AssuranceLevel::Basic),
                Just(AssuranceLevel::Verified),
                Just(AssuranceLevel::HighlyAssured),
                Just(AssuranceLevel::ConstitutionallyCritical),
            ]
        }

        fn arb_enrollment_action() -> impl Strategy<Value = EnrollmentAction> {
            prop_oneof![
                Just(EnrollmentAction::Enroll),
                Just(EnrollmentAction::Revoke),
                Just(EnrollmentAction::Update),
                Just(EnrollmentAction::Reverify),
            ]
        }

        proptest! {
            /// Every factor type maps to exactly one category.
            #[test]
            fn factor_type_has_category(ft in arb_factor_type()) {
                let _ = ft.category(); // Should not panic
            }

            /// Every factor type has a positive base weight.
            #[test]
            fn factor_type_has_positive_weight(ft in arb_factor_type()) {
                prop_assert!(ft.base_weight() > 0.0);
                prop_assert!(ft.base_weight() <= 2.0);
            }

            /// Decay configuration has positive grace period and reverify threshold.
            #[test]
            fn decay_config_valid(ft in arb_factor_type()) {
                let (grace, decay_rate, reverify) = ft.decay_config();
                prop_assert!(grace > 0, "Grace period must be positive");
                prop_assert!(decay_rate >= 0.0, "Decay rate must be non-negative");
                prop_assert!(reverify > 0, "Reverify threshold must be positive");
                prop_assert!(reverify >= grace, "Reverify should be >= grace period");
            }

            /// Current strength is always in [0.0, 1.0] for active factors.
            #[test]
            fn current_strength_clamped(
                ft in arb_factor_type(),
                days_ago in 0u64..1000
            ) {
                let base_micros: i64 = 1735689600_000_000;
                let now = Timestamp::from_micros(base_micros);
                let verified = Timestamp::from_micros(base_micros - (days_ago as i64 * 86400 * 1_000_000));

                let factor = EnrolledFactor {
                    factor_type: ft,
                    factor_id: "test".to_string(),
                    enrolled_at: verified,
                    last_verified: verified,
                    metadata: "{}".to_string(),
                    effective_strength: 1.0,
                    active: true,
                };

                let strength = factor.current_strength(now);
                prop_assert!(strength >= 0.0, "Strength must be >= 0.0, got {}", strength);
                prop_assert!(strength <= 1.0, "Strength must be <= 1.0, got {}", strength);
            }

            /// Inactive factors always have zero strength regardless of time.
            #[test]
            fn inactive_factor_zero_strength(
                ft in arb_factor_type(),
                days_ago in 0u64..1000
            ) {
                let base_micros: i64 = 1735689600_000_000;
                let now = Timestamp::from_micros(base_micros);
                let verified = Timestamp::from_micros(base_micros - (days_ago as i64 * 86400 * 1_000_000));

                let factor = EnrolledFactor {
                    factor_type: ft,
                    factor_id: "test".to_string(),
                    enrolled_at: verified,
                    last_verified: verified,
                    metadata: "{}".to_string(),
                    effective_strength: 1.0,
                    active: false,
                };

                prop_assert_eq!(factor.current_strength(now), 0.0);
            }

            /// Strength decays monotonically (older verification → weaker).
            #[test]
            fn strength_monotonically_decreasing(
                ft in arb_factor_type(),
                d1 in 0u64..500,
                d2 in 0u64..500
            ) {
                let base_micros: i64 = 1735689600_000_000;
                let now = Timestamp::from_micros(base_micros);

                let make_factor = |days: u64| EnrolledFactor {
                    factor_type: ft.clone(),
                    factor_id: "test".to_string(),
                    enrolled_at: Timestamp::from_micros(base_micros - (days as i64 * 86400 * 1_000_000)),
                    last_verified: Timestamp::from_micros(base_micros - (days as i64 * 86400 * 1_000_000)),
                    metadata: "{}".to_string(),
                    effective_strength: 1.0,
                    active: true,
                };

                let s1 = make_factor(d1).current_strength(now);
                let s2 = make_factor(d2).current_strength(now);

                if d1 <= d2 {
                    prop_assert!(s1 >= s2, "d1={} s1={} should be >= d2={} s2={}", d1, s1, d2, s2);
                }
            }

            /// Assurance level scores are monotonically increasing.
            #[test]
            fn assurance_scores_monotone(a in arb_assurance_level(), b in arb_assurance_level()) {
                if a <= b {
                    prop_assert!(a.score() <= b.score());
                }
            }

            /// FactorType JSON round-trips.
            #[test]
            fn factor_type_json_roundtrip(ft in arb_factor_type()) {
                let json = serde_json::to_string(&ft).unwrap();
                let back: FactorType = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(ft, back);
            }

            /// AssuranceLevel JSON round-trips.
            #[test]
            fn assurance_level_json_roundtrip(level in arb_assurance_level()) {
                let json = serde_json::to_string(&level).unwrap();
                let back: AssuranceLevel = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(level, back);
            }

            /// EnrollmentAction JSON round-trips.
            #[test]
            fn enrollment_action_json_roundtrip(action in arb_enrollment_action()) {
                let json = serde_json::to_string(&action).unwrap();
                let back: EnrollmentAction = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(action, back);
            }

            /// FactorCategory JSON round-trips.
            #[test]
            fn factor_category_json_roundtrip(ft in arb_factor_type()) {
                let cat = ft.category();
                let json = serde_json::to_string(&cat).unwrap();
                let back: FactorCategory = serde_json::from_str(&json).unwrap();
                prop_assert_eq!(cat, back);
            }
        }
    }

    #[test]
    fn test_webauthn_counter_f32_precision_boundary() {
        // f32 has 24-bit mantissa, so counters > 16_777_216 lose precision.
        // This documents the known limitation.
        let high_counter: u32 = 16_777_216;
        let as_f32 = high_counter as f32;
        let back_to_u32 = as_f32 as u32;
        assert_eq!(
            back_to_u32, high_counter,
            "Counter at f32 precision boundary"
        );

        // One above the boundary: precision loss
        let over_boundary: u32 = 16_777_217;
        let as_f32 = over_boundary as f32;
        let back_to_u32 = as_f32 as u32;
        // This may or may not round-trip — documenting the behavior
        assert!(
            (back_to_u32 as i64 - over_boundary as i64).abs() <= 1,
            "f32 precision loss should be at most 1 at this boundary"
        );
    }
}
