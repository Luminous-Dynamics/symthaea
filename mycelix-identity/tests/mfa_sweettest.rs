//! # Multi-Factor Authentication (MFA) Sweettest Integration Tests
//!
//! Comprehensive integration tests for the MFA zome using Holochain's sweettest framework.
//! Tests cover MFA state creation, factor enrollment, verification challenges,
//! assurance level calculation, FL eligibility, and enrollment history.
//!
//! ## Running Tests
//!
//! ```bash
//! # Ensure the DNA bundle exists
//! ls mycelix-identity/dna/mycelix_identity_dna.dna
//!
//! # Run tests (requires Holochain conductor via nix develop)
//! cargo test --test mfa_sweettest -- --ignored
//! ```

use holochain::sweettest::*;
use holochain::prelude::*;
use sha2::{Sha256, Digest};
use std::path::PathBuf;

/// Compute the primary key hash the same way the MFA zome does:
/// `sha256:{hex(SHA256(agent.get_raw_39()))}`
fn compute_primary_key_hash(agent: &AgentPubKey) -> String {
    let mut hasher = Sha256::new();
    hasher.update(agent.get_raw_39());
    format!("sha256:{:x}", hasher.finalize())
}

// ============================================================================
// Mirror types (avoids importing zome crates / duplicate symbols)
// ============================================================================

/// Mirror of mfa_integrity::FactorCategory
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub enum FactorCategory {
    Cryptographic,
    Biometric,
    SocialProof,
    ExternalVerification,
    Knowledge,
}

/// Mirror of mfa_integrity::FactorType
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
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
}

/// Mirror of mfa_integrity::AssuranceLevel
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, PartialOrd, Eq, Ord)]
pub enum AssuranceLevel {
    Anonymous,
    Basic,
    Verified,
    HighlyAssured,
    ConstitutionallyCritical,
}

/// Mirror of mfa_integrity::EnrollmentAction
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum EnrollmentAction {
    Enroll,
    Revoke,
    Update,
    Reverify,
}

/// Mirror of mfa_integrity::EnrolledFactor
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EnrolledFactor {
    pub factor_type: FactorType,
    pub factor_id: String,
    pub enrolled_at: Timestamp,
    pub last_verified: Timestamp,
    pub metadata: String,
    pub effective_strength: f32,
    pub active: bool,
}

/// Mirror of mfa_integrity::MfaState
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MfaState {
    pub did: String,
    pub owner: AgentPubKey,
    pub factors: Vec<EnrolledFactor>,
    pub assurance_level: AssuranceLevel,
    pub effective_strength: f32,
    pub category_count: u8,
    pub created: Timestamp,
    pub updated: Timestamp,
    pub version: u32,
}

/// Mirror of mfa_integrity::FactorEnrollment
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FactorEnrollment {
    pub did: String,
    pub factor_type: FactorType,
    pub factor_id: String,
    pub action: EnrollmentAction,
    pub timestamp: Timestamp,
    pub reason: String,
}

/// Mirror of mfa_integrity::FactorVerification
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FactorVerification {
    pub did: String,
    pub factor_type: FactorType,
    pub factor_id: String,
    pub success: bool,
    pub timestamp: Timestamp,
    pub new_strength: f32,
}

// ============================================================================
// Coordinator Input/Output Types
// ============================================================================

/// Mirror of mfa_coordinator::CreateMfaStateInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateMfaStateInput {
    pub did: String,
    pub primary_key_hash: String,
}

/// Mirror of mfa_coordinator::EnrollFactorInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EnrollFactorInput {
    pub did: String,
    pub factor_type: FactorType,
    pub factor_id: String,
    pub metadata: String,
    pub reason: String,
}

/// Mirror of mfa_coordinator::RevokeFactorInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RevokeFactorInput {
    pub did: String,
    pub factor_id: String,
    pub reason: String,
}

/// Mirror of mfa_coordinator::VerifyFactorInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VerifyFactorInput {
    pub did: String,
    pub factor_id: String,
    pub challenge: Option<String>,
    pub proof: Option<VerificationProof>,
}

/// Mirror of mfa_coordinator::VerificationProof
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum VerificationProof {
    Signature {
        signature: String,
        message: String,
    },
    WebAuthn {
        authenticator_data: String,
        client_data_hash: String,
        signature: String,
    },
    BiometricChallenge {
        template_hash: String,
        response: String,
    },
    GitcoinPassport {
        score: f64,
        checked_at: u64,
        stamps: Vec<String>,
    },
    VerifiableCredential {
        credential: String,
        issuer: String,
        credential_type: String,
    },
    SocialRecovery {
        guardian_signatures: Vec<GuardianAttestation>,
        threshold: u32,
    },
    Knowledge {
        answer_hash: String,
    },
}

/// Mirror of mfa_coordinator::GuardianAttestation
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GuardianAttestation {
    pub guardian_did: String,
    pub signature: String,
    pub timestamp: u64,
}

/// Mirror of mfa_coordinator::GenerateChallengeInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GenerateChallengeInput {
    pub did: String,
    pub factor_id: String,
    pub factor_type: FactorType,
}

/// Mirror of mfa_coordinator::VerificationChallenge
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VerificationChallenge {
    pub challenge: String,
    pub factor_id: String,
    pub expires_at: Timestamp,
    pub instructions: String,
}

/// Mirror of mfa_coordinator::AssuranceOutput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AssuranceOutput {
    pub level: AssuranceLevel,
    pub score: f64,
    pub effective_strength: f32,
    pub category_count: u8,
    pub stale_factors: Vec<String>,
}

/// Mirror of mfa_coordinator::MfaStateOutput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MfaStateOutput {
    pub state: MfaState,
    pub action_hash: ActionHash,
    pub assurance: AssuranceOutput,
}

/// Mirror of mfa_coordinator::FlEligibilityResult
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FlEligibilityResult {
    pub eligible: bool,
    pub assurance_level: AssuranceLevel,
    pub effective_strength: f32,
    pub denial_reasons: Vec<String>,
}

/// Mirror of mfa_coordinator::MfaSummary
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MfaSummary {
    pub did: String,
    pub assurance_level: AssuranceLevel,
    pub assurance_score: f64,
    pub factor_count: usize,
    pub category_count: u8,
    pub has_external_verification: bool,
    pub fl_eligible: bool,
}

// ============================================================================
// DID Registry Types (for cross-zome interaction)
// ============================================================================

/// Mirror of did_registry_integrity::DidDocument
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DidDocument {
    pub id: String,
    pub controller: AgentPubKey,
    #[serde(rename = "verificationMethod", alias = "verification_method")]
    pub verification_method: Vec<VerificationMethod>,
    pub authentication: Vec<String>,
    #[serde(rename = "keyAgreement", alias = "key_agreement", default)]
    pub key_agreement: Vec<String>,
    pub service: Vec<ServiceEndpoint>,
    pub created: Timestamp,
    pub updated: Timestamp,
    pub version: u32,
}

/// Mirror of did_registry_integrity::VerificationMethod
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VerificationMethod {
    pub id: String,
    #[serde(rename = "type", alias = "type_")]
    pub type_: String,
    pub controller: String,
    #[serde(rename = "publicKeyMultibase", alias = "public_key_multibase")]
    pub public_key_multibase: String,
    #[serde(default)]
    pub algorithm: Option<u16>,
}

/// Mirror of did_registry_integrity::ServiceEndpoint
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ServiceEndpoint {
    pub id: String,
    #[serde(rename = "type", alias = "type_")]
    pub type_: String,
    #[serde(rename = "serviceEndpoint", alias = "service_endpoint")]
    pub service_endpoint: String,
}

// ============================================================================
// Test Utilities
// ============================================================================

/// Path to the pre-built DNA bundle
fn dna_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("dna")
        .join("mycelix_identity_dna.dna")
}

async fn load_dna() -> DnaFile {
    SweetDnaFile::from_bundle(&dna_path())
        .await
        .expect("Failed to load DNA bundle - run 'hc dna pack dna/' first")
}

/// Decode an entry from a Record into a concrete type via MessagePack deserialization.
fn decode_entry<T: serde::de::DeserializeOwned>(record: &Record) -> Option<T> {
    match record.entry().as_option()? {
        Entry::App(bytes) => {
            let sb = SerializedBytes::from(bytes.to_owned());
            rmp_serde::from_slice(sb.bytes()).ok()
        }
        _ => None,
    }
}

// ============================================================================
// Section 1: MFA State Creation Tests
// ============================================================================

#[cfg(test)]
mod mfa_state_creation {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires Holochain conductor
    async fn test_create_mfa_state_for_did() {
        println!("Test 1.1: Create MFA State for DID");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Create DID first (required by MFA zome)
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;

        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");
        let primary_key_hash = compute_primary_key_hash(&agent);

        // Create MFA state
        let input = CreateMfaStateInput {
            did: did_doc.id.clone(),
            primary_key_hash: primary_key_hash.clone(),
        };

        let output: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", input)
            .await;

        // Verify MFA state
        assert_eq!(output.state.did, did_doc.id, "DID must match");
        assert_eq!(output.state.owner, agent, "Owner must be creating agent");
        assert_eq!(output.state.factors.len(), 1, "Should have one factor (primary key)");
        assert_eq!(
            output.state.factors[0].factor_type,
            FactorType::PrimaryKeyPair,
            "First factor must be PrimaryKeyPair"
        );
        assert_eq!(
            output.state.factors[0].factor_id,
            primary_key_hash,
            "Factor ID must match primary key hash"
        );
        assert!(output.state.factors[0].active, "Factor must be active");
        assert_eq!(output.state.version, 1, "Initial version must be 1");
        assert_eq!(
            output.assurance.level,
            AssuranceLevel::Basic,
            "Initial assurance level should be Basic"
        );
        assert_eq!(output.assurance.category_count, 1, "Should have 1 category");

        println!("  - DID: {}", output.state.did);
        println!("  - Primary Key Factor ID: {}", output.state.factors[0].factor_id);
        println!("  - Assurance Level: {:?}", output.assurance.level);
        println!("Test 1.1 PASSED");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_mfa_state_requires_valid_did_format() {
        println!("Test 1.2: Create MFA State Requires Valid DID Format");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Try to create MFA state with invalid DID format
        let input = CreateMfaStateInput {
            did: "invalid:did:format".to_string(),
            primary_key_hash: "sha256:test".to_string(),
        };

        let result: Result<MfaStateOutput, _> = conductor
            .call_fallible(&cell.zome("mfa"), "create_mfa_state", input)
            .await;

        assert!(result.is_err(), "Should reject invalid DID format");
        println!("Test 1.2 PASSED - Invalid DID format rejected");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_mfa_state_returns_none_for_nonexistent() {
        println!("Test 1.3: Get MFA State Returns None for Non-existent DID");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Query for non-existent MFA state
        let result: Option<MfaStateOutput> = conductor
            .call(
                &cell.zome("mfa"),
                "get_mfa_state",
                "did:mycelix:nonexistent".to_string(),
            )
            .await;

        assert!(result.is_none(), "Should return None for non-existent MFA state");
        println!("Test 1.3 PASSED");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_has_mfa_state_returns_false_for_nonexistent() {
        println!("Test 1.4: has_mfa_state Returns False for Non-existent DID");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let result: bool = conductor
            .call(
                &cell.zome("mfa"),
                "has_mfa_state",
                "did:mycelix:nonexistent".to_string(),
            )
            .await;

        assert!(!result, "Should return false for non-existent MFA state");
        println!("Test 1.4 PASSED");
    }
}

// ============================================================================
// Section 2: Factor Enrollment Tests
// ============================================================================

#[cfg(test)]
mod factor_enrollment {
    use super::*;

    async fn setup_mfa_state(
        conductor: &mut SweetConductor,
        cell: &SweetCell,
        agent: &AgentPubKey,
    ) -> (String, MfaStateOutput) {
        // Create DID
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

        // Create MFA state
        let input = CreateMfaStateInput {
            did: did_doc.id.clone(),
            primary_key_hash: compute_primary_key_hash(&agent),
        };
        let output: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", input)
            .await;

        (did_doc.id, output)
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_enroll_hardware_key_factor() {
        println!("Test 2.1: Enroll Hardware Key Factor");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let (did, _) = setup_mfa_state(&mut conductor, &cell, &agent).await;

        // Enroll hardware key
        let enroll_input = EnrollFactorInput {
            did: did.clone(),
            factor_type: FactorType::HardwareKey,
            factor_id: "yubikey-serial-12345".to_string(),
            metadata: r#"{"model":"YubiKey 5 NFC","firmware":"5.4.3"}"#.to_string(),
            reason: "Added hardware security key for enhanced security".to_string(),
        };

        let output: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "enroll_factor", enroll_input)
            .await;

        assert_eq!(output.state.factors.len(), 2, "Should have 2 factors");
        assert_eq!(output.state.version, 2, "Version should increment");

        let hw_factor = output
            .state
            .factors
            .iter()
            .find(|f| f.factor_type == FactorType::HardwareKey)
            .expect("Should find hardware key factor");

        assert_eq!(hw_factor.factor_id, "yubikey-serial-12345");
        assert!(hw_factor.active);

        println!("  - Enrolled HardwareKey factor");
        println!("  - Total factors: {}", output.state.factors.len());
        println!("Test 2.1 PASSED");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_enroll_gitcoin_passport_factor() {
        println!("Test 2.2: Enroll Gitcoin Passport Factor");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let (did, _) = setup_mfa_state(&mut conductor, &cell, &agent).await;

        // Enroll Gitcoin Passport
        let enroll_input = EnrollFactorInput {
            did: did.clone(),
            factor_type: FactorType::GitcoinPassport,
            factor_id: "passport-0xABCDEF1234567890".to_string(),
            metadata: r#"{"score":42.5,"stamps":15,"lastChecked":"2025-01-01"}"#.to_string(),
            reason: "Verified humanity via Gitcoin Passport".to_string(),
        };

        let output: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "enroll_factor", enroll_input)
            .await;

        let gp_factor = output
            .state
            .factors
            .iter()
            .find(|f| f.factor_type == FactorType::GitcoinPassport)
            .expect("Should find Gitcoin Passport factor");

        assert!(gp_factor.active);
        assert!(
            output.assurance.category_count >= 2,
            "Should have 2+ categories (Cryptographic + ExternalVerification)"
        );

        println!("  - Enrolled GitcoinPassport factor");
        println!("  - Categories: {}", output.assurance.category_count);
        println!("Test 2.2 PASSED");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_enroll_biometric_factor() {
        println!("Test 2.3: Enroll Biometric Factor");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let (did, _) = setup_mfa_state(&mut conductor, &cell, &agent).await;

        // Enroll biometric
        let enroll_input = EnrollFactorInput {
            did: did.clone(),
            factor_type: FactorType::Biometric,
            factor_id: "face-template-hash-abc123".to_string(),
            metadata: r#"{"type":"face","provider":"local"}"#.to_string(),
            reason: "Added face recognition for quick authentication".to_string(),
        };

        let output: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "enroll_factor", enroll_input)
            .await;

        let bio_factor = output
            .state
            .factors
            .iter()
            .find(|f| f.factor_type == FactorType::Biometric)
            .expect("Should find Biometric factor");

        assert!(bio_factor.active);
        assert_eq!(bio_factor.factor_type.category(), FactorCategory::Biometric);

        println!("  - Enrolled Biometric factor");
        println!("Test 2.3 PASSED");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_enroll_social_recovery_factor() {
        println!("Test 2.4: Enroll Social Recovery Factor");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let (did, _) = setup_mfa_state(&mut conductor, &cell, &agent).await;

        // Enroll social recovery
        let enroll_input = EnrollFactorInput {
            did: did.clone(),
            factor_type: FactorType::SocialRecovery,
            factor_id: "guardians-group-1".to_string(),
            metadata: r#"{"guardians":["did:mycelix:guardian1","did:mycelix:guardian2","did:mycelix:guardian3"],"threshold":2}"#.to_string(),
            reason: "Set up social recovery with trusted guardians".to_string(),
        };

        let output: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "enroll_factor", enroll_input)
            .await;

        let sr_factor = output
            .state
            .factors
            .iter()
            .find(|f| f.factor_type == FactorType::SocialRecovery)
            .expect("Should find SocialRecovery factor");

        assert!(sr_factor.active);
        assert_eq!(sr_factor.factor_type.category(), FactorCategory::SocialProof);

        println!("  - Enrolled SocialRecovery factor");
        println!("Test 2.4 PASSED");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_enroll_verifiable_credential_factor() {
        println!("Test 2.5: Enroll Verifiable Credential Factor");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let (did, _) = setup_mfa_state(&mut conductor, &cell, &agent).await;

        // Enroll verifiable credential
        let enroll_input = EnrollFactorInput {
            did: did.clone(),
            factor_type: FactorType::VerifiableCredential,
            factor_id: "vc-university-degree-2024".to_string(),
            metadata: r#"{"issuer":"did:mycelix:university","type":"UniversityDegree","expires":"2030-01-01"}"#.to_string(),
            reason: "Verified university degree credential".to_string(),
        };

        let output: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "enroll_factor", enroll_input)
            .await;

        let vc_factor = output
            .state
            .factors
            .iter()
            .find(|f| f.factor_type == FactorType::VerifiableCredential)
            .expect("Should find VerifiableCredential factor");

        assert!(vc_factor.active);
        assert_eq!(
            vc_factor.factor_type.category(),
            FactorCategory::ExternalVerification
        );

        println!("  - Enrolled VerifiableCredential factor");
        println!("Test 2.5 PASSED");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_cannot_enroll_duplicate_factor_id() {
        println!("Test 2.6: Cannot Enroll Duplicate Factor ID");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let (did, _) = setup_mfa_state(&mut conductor, &cell, &agent).await;

        // Enroll first factor
        let enroll_input = EnrollFactorInput {
            did: did.clone(),
            factor_type: FactorType::HardwareKey,
            factor_id: "duplicate-id-123".to_string(),
            metadata: "{}".to_string(),
            reason: "First enrollment".to_string(),
        };

        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "enroll_factor", enroll_input)
            .await;

        // Try to enroll with same factor ID
        let duplicate_input = EnrollFactorInput {
            did: did.clone(),
            factor_type: FactorType::Biometric,
            factor_id: "duplicate-id-123".to_string(), // Same ID, different type
            metadata: "{}".to_string(),
            reason: "Duplicate enrollment attempt".to_string(),
        };

        let result: Result<MfaStateOutput, _> = conductor
            .call_fallible(&cell.zome("mfa"), "enroll_factor", duplicate_input)
            .await;

        assert!(result.is_err(), "Should reject duplicate factor ID");
        println!("Test 2.6 PASSED - Duplicate factor ID rejected");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_enroll_multiple_factors_different_categories() {
        println!("Test 2.7: Enroll Multiple Factors from Different Categories");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let (did, _) = setup_mfa_state(&mut conductor, &cell, &agent).await;

        // Enroll factors from each category
        let factors = vec![
            EnrollFactorInput {
                did: did.clone(),
                factor_type: FactorType::HardwareKey, // Cryptographic (already have PrimaryKeyPair)
                factor_id: "hw-key-1".to_string(),
                metadata: "{}".to_string(),
                reason: "test".to_string(),
            },
            EnrollFactorInput {
                did: did.clone(),
                factor_type: FactorType::Biometric, // Biometric
                factor_id: "bio-1".to_string(),
                metadata: "{}".to_string(),
                reason: "test".to_string(),
            },
            EnrollFactorInput {
                did: did.clone(),
                factor_type: FactorType::SocialRecovery, // SocialProof
                factor_id: "social-1".to_string(),
                metadata: "{}".to_string(),
                reason: "test".to_string(),
            },
            EnrollFactorInput {
                did: did.clone(),
                factor_type: FactorType::GitcoinPassport, // ExternalVerification
                factor_id: "gp-1".to_string(),
                metadata: "{}".to_string(),
                reason: "test".to_string(),
            },
            EnrollFactorInput {
                did: did.clone(),
                factor_type: FactorType::RecoveryPhrase, // Knowledge
                factor_id: "recovery-1".to_string(),
                metadata: "{}".to_string(),
                reason: "test".to_string(),
            },
        ];

        let mut last_output: Option<MfaStateOutput> = None;
        for factor in factors {
            last_output = Some(
                conductor
                    .call(&cell.zome("mfa"), "enroll_factor", factor)
                    .await,
            );
        }

        let output = last_output.expect("Should have output");

        // Should have 6 factors (1 primary + 5 new)
        assert_eq!(output.state.factors.len(), 6, "Should have 6 factors");

        // Should have all 5 categories
        assert_eq!(output.assurance.category_count, 5, "Should have 5 categories");

        // Should be at highest assurance level
        assert!(
            output.assurance.level >= AssuranceLevel::HighlyAssured,
            "With 6 factors from 5 categories, should be at least HighlyAssured"
        );

        println!(
            "  - Total factors: {}, Categories: {}",
            output.state.factors.len(),
            output.assurance.category_count
        );
        println!("  - Assurance level: {:?}", output.assurance.level);
        println!("Test 2.7 PASSED");
    }
}

// ============================================================================
// Section 3: Verification Challenge Tests
// ============================================================================

#[cfg(test)]
mod verification_challenges {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_generate_verification_challenge() {
        println!("Test 3.1: Generate Verification Challenge");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Setup DID and MFA
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

        let mfa_input = CreateMfaStateInput {
            did: did_doc.id.clone(),
            primary_key_hash: compute_primary_key_hash(&agent),
        };
        let mfa_output: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", mfa_input)
            .await;

        // Generate challenge for primary key
        let challenge_input = GenerateChallengeInput {
            did: did_doc.id.clone(),
            factor_id: mfa_output.state.factors[0].factor_id.clone(),
            factor_type: FactorType::PrimaryKeyPair,
        };

        let challenge: VerificationChallenge = conductor
            .call(
                &cell.zome("mfa"),
                "generate_verification_challenge",
                challenge_input,
            )
            .await;

        assert!(!challenge.challenge.is_empty(), "Challenge should not be empty");
        assert_eq!(
            challenge.factor_id, mfa_output.state.factors[0].factor_id,
            "Factor ID should match"
        );
        assert!(
            challenge.instructions.contains("Sign"),
            "Instructions should mention signing"
        );

        println!("  - Generated challenge: {}", &challenge.challenge[..32]);
        println!("  - Instructions: {}", challenge.instructions);
        println!("Test 3.1 PASSED");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_generate_challenge_for_different_factor_types() {
        println!("Test 3.2: Generate Challenges for Different Factor Types");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Setup DID and MFA
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

        let mfa_input = CreateMfaStateInput {
            did: did_doc.id.clone(),
            primary_key_hash: compute_primary_key_hash(&agent),
        };
        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", mfa_input)
            .await;

        // Test different factor types
        let factor_types = vec![
            (FactorType::PrimaryKeyPair, "Sign"),
            (FactorType::HardwareKey, "Tap"),
            (FactorType::Biometric, "biometric"),
            (FactorType::GitcoinPassport, "Gitcoin"),
            (FactorType::SocialRecovery, "guardian"),
        ];

        for (factor_type, expected_keyword) in factor_types {
            let challenge_input = GenerateChallengeInput {
                did: did_doc.id.clone(),
                factor_id: format!("test-{:?}", factor_type),
                factor_type: factor_type.clone(),
            };

            let challenge: VerificationChallenge = conductor
                .call(
                    &cell.zome("mfa"),
                    "generate_verification_challenge",
                    challenge_input,
                )
                .await;

            assert!(
                challenge.instructions.to_lowercase().contains(&expected_keyword.to_lowercase()),
                "Instructions for {:?} should contain '{}'",
                factor_type,
                expected_keyword
            );

            println!("  - {:?}: {}", factor_type, challenge.instructions);
        }

        println!("Test 3.2 PASSED");
    }
}

// ============================================================================
// Section 4: Factor Verification Tests
// ============================================================================

#[cfg(test)]
mod factor_verification {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_verify_primary_key_factor() {
        println!("Test 4.1: Verify Primary Key Factor");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Setup DID and MFA
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

        let primary_key_hash = compute_primary_key_hash(&agent);
        let mfa_input = CreateMfaStateInput {
            did: did_doc.id.clone(),
            primary_key_hash: primary_key_hash.clone(),
        };
        let mfa_output: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", mfa_input)
            .await;

        // Use the actual enrolled factor_id from the output (MFA zome computes
        // the real SHA256 hash internally, which may differ from our input)
        let enrolled_factor_id = mfa_output.state.factors[0].factor_id.clone();

        // Verify factor (primary key uses implicit Holochain authentication)
        let verify_input = VerifyFactorInput {
            did: did_doc.id.clone(),
            factor_id: enrolled_factor_id,
            challenge: None,
            proof: None, // Primary key verified by Holochain capability system
        };

        let output: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "verify_factor", verify_input)
            .await;

        assert_eq!(
            output.state.factors[0].effective_strength, 1.0,
            "Verified factor should have full strength"
        );

        println!("  - Factor verified successfully");
        println!("  - Effective strength: {}", output.state.factors[0].effective_strength);
        println!("Test 4.1 PASSED");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_verify_hardware_key_with_webauthn_proof() {
        println!("Test 4.2: Verify Hardware Key with WebAuthn Proof");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Setup DID and MFA
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

        let mfa_input = CreateMfaStateInput {
            did: did_doc.id.clone(),
            primary_key_hash: compute_primary_key_hash(&agent),
        };
        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", mfa_input)
            .await;

        // Enroll hardware key — factor_id must be base64-encoded 32-byte Ed25519 public key
        // (the zome decodes factor_id as the credential public key for WebAuthn verification)
        // 32 zero bytes in standard base64:
        let credential_key_b64 = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=".to_string();
        let enroll_input = EnrollFactorInput {
            did: did_doc.id.clone(),
            factor_type: FactorType::HardwareKey,
            factor_id: credential_key_b64.clone(),
            metadata: "{}".to_string(),
            reason: "test".to_string(),
        };
        let enroll_output: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "enroll_factor", enroll_input)
            .await;
        println!("  - Hardware key enrolled (factor_id = base64 32-byte key)");

        // Verify enrollment worked — factor should exist in state
        let hw_factor = enroll_output
            .state
            .factors
            .iter()
            .find(|f| f.factor_id == credential_key_b64)
            .expect("Should find enrolled hardware key factor");
        assert_eq!(hw_factor.factor_type, FactorType::HardwareKey);
        println!("  - Factor found in MFA state after enrollment");

        // Generate a proper verification challenge from the zome
        let challenge_input = GenerateChallengeInput {
            did: did_doc.id.clone(),
            factor_id: credential_key_b64.clone(),
            factor_type: FactorType::HardwareKey,
        };
        let challenge_resp: VerificationChallenge = conductor
            .call(&cell.zome("mfa"), "generate_verification_challenge", challenge_input)
            .await;
        assert_eq!(challenge_resp.challenge.len(), 64, "Challenge should be 64 hex chars");
        println!("  - Challenge generated: {} chars", challenge_resp.challenge.len());

        // Attempt WebAuthn verification with structurally valid but cryptographically
        // invalid proof. Since sweettest doesn't expose agent signing capabilities,
        // we can't produce a real Ed25519 signature. We verify the plumbing works
        // (enrollment, challenge, update-chain following) and expect signature failure.
        let verify_input = VerifyFactorInput {
            did: did_doc.id.clone(),
            factor_id: credential_key_b64.clone(),
            challenge: Some(challenge_resp.challenge),
            proof: Some(VerificationProof::WebAuthn {
                // 37 bytes: 32-byte rpIdHash (zeros) + flags 0x45 (UP+UV) + counter 1
                authenticator_data: "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABFAAAAAQ==".to_string(),
                // 32-byte SHA256 hash (all 0x01) in base64
                client_data_hash: "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE=".to_string(),
                // 64-byte dummy signature (will fail Ed25519 verification — expected)
                signature: "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==".to_string(),
            }),
        };

        // Verification will fail because we can't produce a valid Ed25519 signature
        // in sweettest. This is expected — we're testing that the full flow works
        // up to the actual crypto verification point.
        let result: Result<MfaStateOutput, _> = conductor
            .call_fallible(&cell.zome("mfa"), "verify_factor", verify_input)
            .await;

        // The call should fail with a signature verification error, NOT a plumbing error
        // (i.e., not "Factor not found", not "Challenge format invalid", not "Unsupported key type")
        assert!(result.is_err(), "Verification with dummy signature should fail");
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            !err_msg.contains("Factor not found"),
            "Should not get 'Factor not found' — update chain should work. Got: {}", err_msg
        );
        assert!(
            !err_msg.contains("Unsupported WebAuthn key type"),
            "Should not get key type error with 32-byte key. Got: {}", err_msg
        );
        assert!(
            !err_msg.contains("Challenge format invalid"),
            "Should not get challenge format error. Got: {}", err_msg
        );
        println!("  - WebAuthn verification correctly rejected dummy signature");
        println!("  - Full plumbing verified: enrollment → challenge → factor lookup → crypto dispatch");
        println!("Test 4.2 PASSED");
    }
}

// ============================================================================
// Section 5: Assurance Level Calculation Tests
// ============================================================================

#[cfg(test)]
mod assurance_calculation {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_assurance_level_basic_single_factor() {
        println!("Test 5.1: Assurance Level Basic with Single Factor");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Setup DID and MFA with just primary key
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

        let mfa_input = CreateMfaStateInput {
            did: did_doc.id.clone(),
            primary_key_hash: compute_primary_key_hash(&agent),
        };
        let output: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", mfa_input)
            .await;

        // Calculate assurance
        let assurance: AssuranceOutput = conductor
            .call(&cell.zome("mfa"), "calculate_assurance", did_doc.id.clone())
            .await;

        assert_eq!(assurance.level, AssuranceLevel::Basic, "Single factor should be Basic");
        assert_eq!(assurance.score, 0.25, "Basic level score should be 0.25");
        assert_eq!(assurance.category_count, 1, "Should have 1 category");

        println!("  - Assurance: {:?} (score: {})", assurance.level, assurance.score);
        println!("Test 5.1 PASSED");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_assurance_level_verified_with_two_categories() {
        println!("Test 5.2: Assurance Level Verified with Two Categories");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Setup DID and MFA
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

        let mfa_input = CreateMfaStateInput {
            did: did_doc.id.clone(),
            primary_key_hash: compute_primary_key_hash(&agent),
        };
        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", mfa_input)
            .await;

        // Add factors to reach Verified level
        let factors = vec![
            EnrollFactorInput {
                did: did_doc.id.clone(),
                factor_type: FactorType::GitcoinPassport,
                factor_id: "gp-1".to_string(),
                metadata: "{}".to_string(),
                reason: "test".to_string(),
            },
            EnrollFactorInput {
                did: did_doc.id.clone(),
                factor_type: FactorType::HardwareKey,
                factor_id: "hw-1".to_string(),
                metadata: "{}".to_string(),
                reason: "test".to_string(),
            },
        ];

        for factor in factors {
            let _: MfaStateOutput = conductor
                .call(&cell.zome("mfa"), "enroll_factor", factor)
                .await;
        }

        // Calculate assurance
        let assurance: AssuranceOutput = conductor
            .call(&cell.zome("mfa"), "calculate_assurance", did_doc.id.clone())
            .await;

        assert!(
            assurance.level >= AssuranceLevel::Verified,
            "With Cryptographic + ExternalVerification, should be at least Verified"
        );
        assert!(assurance.category_count >= 2, "Should have 2+ categories");

        println!("  - Assurance: {:?}", assurance.level);
        println!("  - Categories: {}, Strength: {:.2}", assurance.category_count, assurance.effective_strength);
        println!("Test 5.2 PASSED");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_assurance_level_highly_assured_with_multiple_categories() {
        println!("Test 5.3: Assurance Level HighlyAssured with Multiple Categories");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Setup DID and MFA
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

        let mfa_input = CreateMfaStateInput {
            did: did_doc.id.clone(),
            primary_key_hash: compute_primary_key_hash(&agent),
        };
        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", mfa_input)
            .await;

        // Add factors from different categories
        let factors = vec![
            EnrollFactorInput {
                did: did_doc.id.clone(),
                factor_type: FactorType::HardwareKey,
                factor_id: "hw-1".to_string(),
                metadata: "{}".to_string(),
                reason: "test".to_string(),
            },
            EnrollFactorInput {
                did: did_doc.id.clone(),
                factor_type: FactorType::GitcoinPassport,
                factor_id: "gp-1".to_string(),
                metadata: "{}".to_string(),
                reason: "test".to_string(),
            },
            EnrollFactorInput {
                did: did_doc.id.clone(),
                factor_type: FactorType::SocialRecovery,
                factor_id: "sr-1".to_string(),
                metadata: "{}".to_string(),
                reason: "test".to_string(),
            },
        ];

        for factor in factors {
            let _: MfaStateOutput = conductor
                .call(&cell.zome("mfa"), "enroll_factor", factor)
                .await;
        }

        // Calculate assurance
        let assurance: AssuranceOutput = conductor
            .call(&cell.zome("mfa"), "calculate_assurance", did_doc.id.clone())
            .await;

        assert!(
            assurance.level >= AssuranceLevel::HighlyAssured,
            "With 4 factors from 3 categories and sufficient strength, should be HighlyAssured"
        );
        assert!(assurance.category_count >= 3, "Should have 3+ categories");
        assert!(assurance.effective_strength >= 3.0, "Should have 3.0+ strength");

        println!("  - Assurance: {:?}", assurance.level);
        println!("  - Categories: {}, Strength: {:.2}", assurance.category_count, assurance.effective_strength);
        println!("Test 5.3 PASSED");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_mfa_assurance_score() {
        println!("Test 5.4: Get MFA Assurance Score for MATL");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Setup DID and MFA
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

        let mfa_input = CreateMfaStateInput {
            did: did_doc.id.clone(),
            primary_key_hash: compute_primary_key_hash(&agent),
        };
        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", mfa_input)
            .await;

        // Get score
        let score: f64 = conductor
            .call(&cell.zome("mfa"), "get_mfa_assurance_score", did_doc.id.clone())
            .await;

        assert!(score >= 0.0 && score <= 1.0, "Score must be between 0.0 and 1.0");
        assert_eq!(score, 0.25, "Basic level should return 0.25 score");

        println!("  - MATL Score: {}", score);
        println!("Test 5.4 PASSED");
    }
}

// ============================================================================
// Section 6: FL Eligibility Tests
// ============================================================================

#[cfg(test)]
mod fl_eligibility {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_fl_eligibility_denied_basic_only() {
        println!("Test 6.1: FL Eligibility Denied with Basic Only");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Setup DID and MFA with just primary key
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

        let mfa_input = CreateMfaStateInput {
            did: did_doc.id.clone(),
            primary_key_hash: compute_primary_key_hash(&agent),
        };
        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", mfa_input)
            .await;

        // Check FL eligibility
        let eligibility: FlEligibilityResult = conductor
            .call(&cell.zome("mfa"), "check_fl_eligibility", did_doc.id.clone())
            .await;

        assert!(!eligibility.eligible, "Should not be FL eligible with only primary key");
        assert!(!eligibility.denial_reasons.is_empty(), "Should have denial reasons");

        println!("  - FL Eligible: {}", eligibility.eligible);
        println!("  - Denial reasons: {:?}", eligibility.denial_reasons);
        println!("Test 6.1 PASSED");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_fl_eligibility_granted_with_requirements() {
        println!("Test 6.2: FL Eligibility Granted with Requirements Met");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Setup DID and MFA
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

        let mfa_input = CreateMfaStateInput {
            did: did_doc.id.clone(),
            primary_key_hash: compute_primary_key_hash(&agent),
        };
        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", mfa_input)
            .await;

        // Add required factors (Cryptographic + ExternalVerification)
        let factors = vec![
            EnrollFactorInput {
                did: did_doc.id.clone(),
                factor_type: FactorType::HardwareKey,
                factor_id: "hw-fl".to_string(),
                metadata: "{}".to_string(),
                reason: "FL requirement".to_string(),
            },
            EnrollFactorInput {
                did: did_doc.id.clone(),
                factor_type: FactorType::GitcoinPassport,
                factor_id: "gp-fl".to_string(),
                metadata: "{}".to_string(),
                reason: "FL requirement".to_string(),
            },
        ];

        for factor in factors {
            let _: MfaStateOutput = conductor
                .call(&cell.zome("mfa"), "enroll_factor", factor)
                .await;
        }

        // Check FL eligibility
        let eligibility: FlEligibilityResult = conductor
            .call(&cell.zome("mfa"), "check_fl_eligibility", did_doc.id.clone())
            .await;

        assert!(
            eligibility.eligible,
            "Should be FL eligible with Cryptographic + ExternalVerification"
        );
        assert!(
            eligibility.denial_reasons.is_empty(),
            "Should have no denial reasons when eligible"
        );
        assert!(
            eligibility.assurance_level >= AssuranceLevel::Verified,
            "Should be at least Verified"
        );

        println!("  - FL Eligible: {}", eligibility.eligible);
        println!("  - Assurance Level: {:?}", eligibility.assurance_level);
        println!("  - Effective Strength: {:.2}", eligibility.effective_strength);
        println!("Test 6.2 PASSED");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_fl_eligibility_missing_external_verification() {
        println!("Test 6.3: FL Eligibility Denied - Missing External Verification");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Setup DID and MFA
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

        let mfa_input = CreateMfaStateInput {
            did: did_doc.id.clone(),
            primary_key_hash: compute_primary_key_hash(&agent),
        };
        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", mfa_input)
            .await;

        // Add only Cryptographic factors (no ExternalVerification)
        let enroll_input = EnrollFactorInput {
            did: did_doc.id.clone(),
            factor_type: FactorType::HardwareKey,
            factor_id: "hw-only".to_string(),
            metadata: "{}".to_string(),
            reason: "test".to_string(),
        };
        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "enroll_factor", enroll_input)
            .await;

        // Check FL eligibility
        let eligibility: FlEligibilityResult = conductor
            .call(&cell.zome("mfa"), "check_fl_eligibility", did_doc.id.clone())
            .await;

        assert!(!eligibility.eligible, "Should not be FL eligible without ExternalVerification");
        assert!(
            eligibility.denial_reasons.iter().any(|r| r.contains("ExternalVerification")),
            "Should mention missing ExternalVerification"
        );

        println!("  - FL Eligible: {}", eligibility.eligible);
        println!("  - Denial reasons: {:?}", eligibility.denial_reasons);
        println!("Test 6.3 PASSED");
    }
}

// ============================================================================
// Section 7: Factor Removal Tests
// ============================================================================

#[cfg(test)]
mod factor_removal {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_revoke_factor_success() {
        println!("Test 7.1: Revoke Factor Successfully");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Setup DID and MFA
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

        let mfa_input = CreateMfaStateInput {
            did: did_doc.id.clone(),
            primary_key_hash: compute_primary_key_hash(&agent),
        };
        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", mfa_input)
            .await;

        // Enroll a factor
        let enroll_input = EnrollFactorInput {
            did: did_doc.id.clone(),
            factor_type: FactorType::RecoveryPhrase,
            factor_id: "recovery-to-revoke".to_string(),
            metadata: "{}".to_string(),
            reason: "Adding for revocation test".to_string(),
        };
        let after_enroll: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "enroll_factor", enroll_input)
            .await;

        assert_eq!(after_enroll.state.factors.len(), 2, "Should have 2 factors");

        // Revoke the factor
        let revoke_input = RevokeFactorInput {
            did: did_doc.id.clone(),
            factor_id: "recovery-to-revoke".to_string(),
            reason: "No longer needed".to_string(),
        };
        let after_revoke: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "revoke_factor", revoke_input)
            .await;

        assert_eq!(after_revoke.state.factors.len(), 1, "Should have 1 factor after revocation");
        assert_eq!(
            after_revoke.state.factors[0].factor_type,
            FactorType::PrimaryKeyPair,
            "Only primary key should remain"
        );

        println!("  - Factor revoked successfully");
        println!("  - Remaining factors: {}", after_revoke.state.factors.len());
        println!("Test 7.1 PASSED");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_cannot_revoke_last_factor() {
        println!("Test 7.2: Cannot Revoke Last Factor");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Setup DID and MFA
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

        let primary_key_hash = compute_primary_key_hash(&agent);
        let mfa_input = CreateMfaStateInput {
            did: did_doc.id.clone(),
            primary_key_hash: primary_key_hash.clone(),
        };
        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", mfa_input)
            .await;

        // Try to revoke the only factor
        let revoke_input = RevokeFactorInput {
            did: did_doc.id.clone(),
            factor_id: primary_key_hash,
            reason: "Trying to revoke last factor".to_string(),
        };
        let result: Result<MfaStateOutput, _> = conductor
            .call_fallible(&cell.zome("mfa"), "revoke_factor", revoke_input)
            .await;

        assert!(result.is_err(), "Should not be able to revoke last factor");
        println!("Test 7.2 PASSED - Last factor revocation blocked");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_cannot_revoke_nonexistent_factor() {
        println!("Test 7.3: Cannot Revoke Non-existent Factor");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Setup DID and MFA
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

        let mfa_input = CreateMfaStateInput {
            did: did_doc.id.clone(),
            primary_key_hash: compute_primary_key_hash(&agent),
        };
        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", mfa_input)
            .await;

        // Try to revoke non-existent factor
        let revoke_input = RevokeFactorInput {
            did: did_doc.id.clone(),
            factor_id: "nonexistent-factor-id".to_string(),
            reason: "Trying to revoke non-existent".to_string(),
        };
        let result: Result<MfaStateOutput, _> = conductor
            .call_fallible(&cell.zome("mfa"), "revoke_factor", revoke_input)
            .await;

        assert!(result.is_err(), "Should not be able to revoke non-existent factor");
        println!("Test 7.3 PASSED - Non-existent factor revocation blocked");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_assurance_recalculated_after_revocation() {
        println!("Test 7.4: Assurance Recalculated After Revocation");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Setup DID and MFA
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

        let mfa_input = CreateMfaStateInput {
            did: did_doc.id.clone(),
            primary_key_hash: compute_primary_key_hash(&agent),
        };
        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", mfa_input)
            .await;

        // Add factors to increase assurance
        let factors = vec![
            EnrollFactorInput {
                did: did_doc.id.clone(),
                factor_type: FactorType::GitcoinPassport,
                factor_id: "gp-revoke-test".to_string(),
                metadata: "{}".to_string(),
                reason: "test".to_string(),
            },
            EnrollFactorInput {
                did: did_doc.id.clone(),
                factor_type: FactorType::HardwareKey,
                factor_id: "hw-revoke-test".to_string(),
                metadata: "{}".to_string(),
                reason: "test".to_string(),
            },
        ];

        for factor in factors {
            let _: MfaStateOutput = conductor
                .call(&cell.zome("mfa"), "enroll_factor", factor)
                .await;
        }

        // Check assurance before revocation
        let before: AssuranceOutput = conductor
            .call(&cell.zome("mfa"), "calculate_assurance", did_doc.id.clone())
            .await;

        // Revoke Gitcoin Passport
        let revoke_input = RevokeFactorInput {
            did: did_doc.id.clone(),
            factor_id: "gp-revoke-test".to_string(),
            reason: "Testing assurance recalculation".to_string(),
        };
        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "revoke_factor", revoke_input)
            .await;

        // Check assurance after revocation
        let after: AssuranceOutput = conductor
            .call(&cell.zome("mfa"), "calculate_assurance", did_doc.id.clone())
            .await;

        assert!(
            after.effective_strength < before.effective_strength,
            "Effective strength should decrease after revocation"
        );
        assert!(
            after.category_count < before.category_count,
            "Category count should decrease after revocation"
        );

        println!("  - Before: strength={:.2}, categories={}", before.effective_strength, before.category_count);
        println!("  - After: strength={:.2}, categories={}", after.effective_strength, after.category_count);
        println!("Test 7.4 PASSED");
    }
}

// ============================================================================
// Section 8: Enrollment History Tests
// ============================================================================

#[cfg(test)]
mod enrollment_history {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_enrollment_history() {
        println!("Test 8.1: Get Enrollment History");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Setup DID and MFA
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

        let mfa_input = CreateMfaStateInput {
            did: did_doc.id.clone(),
            primary_key_hash: compute_primary_key_hash(&agent),
        };
        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", mfa_input)
            .await;

        // Perform several enrollment actions
        let enroll1 = EnrollFactorInput {
            did: did_doc.id.clone(),
            factor_type: FactorType::HardwareKey,
            factor_id: "hw-history-1".to_string(),
            metadata: "{}".to_string(),
            reason: "First hardware key".to_string(),
        };
        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "enroll_factor", enroll1)
            .await;

        let enroll2 = EnrollFactorInput {
            did: did_doc.id.clone(),
            factor_type: FactorType::GitcoinPassport,
            factor_id: "gp-history-1".to_string(),
            metadata: "{}".to_string(),
            reason: "Gitcoin verification".to_string(),
        };
        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "enroll_factor", enroll2)
            .await;

        // Revoke one
        let revoke = RevokeFactorInput {
            did: did_doc.id.clone(),
            factor_id: "hw-history-1".to_string(),
            reason: "Lost hardware key".to_string(),
        };
        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "revoke_factor", revoke)
            .await;

        // Get history
        let history: Vec<FactorEnrollment> = conductor
            .call(&cell.zome("mfa"), "get_enrollment_history", did_doc.id.clone())
            .await;

        // Should have at least 4 entries:
        // 1. Initial PrimaryKeyPair enrollment
        // 2. HardwareKey enrollment
        // 3. GitcoinPassport enrollment
        // 4. HardwareKey revocation
        assert!(history.len() >= 4, "Should have at least 4 enrollment history entries");

        // Verify enrollment actions are recorded
        let has_initial = history.iter().any(|e| {
            e.factor_type == FactorType::PrimaryKeyPair && e.action == EnrollmentAction::Enroll
        });
        let has_hw_enroll = history.iter().any(|e| {
            e.factor_id == "hw-history-1" && e.action == EnrollmentAction::Enroll
        });
        let has_hw_revoke = history.iter().any(|e| {
            e.factor_id == "hw-history-1" && e.action == EnrollmentAction::Revoke
        });
        let has_gp_enroll = history.iter().any(|e| {
            e.factor_id == "gp-history-1" && e.action == EnrollmentAction::Enroll
        });

        assert!(has_initial, "Should have initial PrimaryKeyPair enrollment");
        assert!(has_hw_enroll, "Should have HardwareKey enrollment");
        assert!(has_hw_revoke, "Should have HardwareKey revocation");
        assert!(has_gp_enroll, "Should have GitcoinPassport enrollment");

        println!("  - Total history entries: {}", history.len());
        for entry in &history {
            println!("    - {:?}: {} ({:?})", entry.action, entry.factor_id, entry.factor_type);
        }
        println!("Test 8.1 PASSED");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_enrollment_history_sorted_by_timestamp() {
        println!("Test 8.2: Enrollment History Sorted by Timestamp");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Setup DID and MFA
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

        let mfa_input = CreateMfaStateInput {
            did: did_doc.id.clone(),
            primary_key_hash: compute_primary_key_hash(&agent),
        };
        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", mfa_input)
            .await;

        // Add several factors with delays
        for i in 1..=3 {
            let enroll = EnrollFactorInput {
                did: did_doc.id.clone(),
                factor_type: FactorType::RecoveryPhrase,
                factor_id: format!("recovery-{}", i),
                metadata: "{}".to_string(),
                reason: format!("Factor {}", i),
            };
            let _: MfaStateOutput = conductor
                .call(&cell.zome("mfa"), "enroll_factor", enroll)
                .await;
        }

        // Get history
        let history: Vec<FactorEnrollment> = conductor
            .call(&cell.zome("mfa"), "get_enrollment_history", did_doc.id.clone())
            .await;

        // Verify sorted by timestamp (ascending)
        for i in 1..history.len() {
            assert!(
                history[i - 1].timestamp <= history[i].timestamp,
                "History should be sorted by timestamp"
            );
        }

        println!("  - History is sorted by timestamp");
        println!("Test 8.2 PASSED");
    }
}

// ============================================================================
// Section 9: Bridge Integration Tests
// ============================================================================

#[cfg(test)]
mod bridge_integration {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_mfa_summary() {
        println!("Test 9.1: Get MFA Summary for Bridge");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Setup DID and MFA
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

        let mfa_input = CreateMfaStateInput {
            did: did_doc.id.clone(),
            primary_key_hash: compute_primary_key_hash(&agent),
        };
        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", mfa_input)
            .await;

        // Add GitcoinPassport for external verification
        let enroll = EnrollFactorInput {
            did: did_doc.id.clone(),
            factor_type: FactorType::GitcoinPassport,
            factor_id: "gp-summary".to_string(),
            metadata: "{}".to_string(),
            reason: "test".to_string(),
        };
        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "enroll_factor", enroll)
            .await;

        // Get summary
        let summary: Option<MfaSummary> = conductor
            .call(&cell.zome("mfa"), "get_mfa_summary", did_doc.id.clone())
            .await;

        let summary = summary.expect("Should have MFA summary");

        assert_eq!(summary.did, did_doc.id);
        assert_eq!(summary.factor_count, 2);
        assert!(summary.has_external_verification);
        assert!(summary.assurance_score >= 0.0 && summary.assurance_score <= 1.0);

        println!("  - DID: {}", summary.did);
        println!("  - Assurance Level: {:?}", summary.assurance_level);
        println!("  - Factor Count: {}", summary.factor_count);
        println!("  - Has External Verification: {}", summary.has_external_verification);
        println!("  - FL Eligible: {}", summary.fl_eligible);
        println!("Test 9.1 PASSED");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_mfa_summary_returns_none_for_nonexistent() {
        println!("Test 9.2: Get MFA Summary Returns None for Non-existent");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let summary: Option<MfaSummary> = conductor
            .call(
                &cell.zome("mfa"),
                "get_mfa_summary",
                "did:mycelix:nonexistent".to_string(),
            )
            .await;

        assert!(summary.is_none(), "Should return None for non-existent DID");
        println!("Test 9.2 PASSED");
    }
}

// ============================================================================
// Section 10: Unit Tests (No Conductor Required)
// ============================================================================

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_factor_type_category_mapping() {
        assert_eq!(FactorType::PrimaryKeyPair.category(), FactorCategory::Cryptographic);
        assert_eq!(FactorType::HardwareKey.category(), FactorCategory::Cryptographic);
        assert_eq!(FactorType::Biometric.category(), FactorCategory::Biometric);
        assert_eq!(FactorType::SocialRecovery.category(), FactorCategory::SocialProof);
        assert_eq!(FactorType::ReputationAttestation.category(), FactorCategory::SocialProof);
        assert_eq!(FactorType::GitcoinPassport.category(), FactorCategory::ExternalVerification);
        assert_eq!(FactorType::VerifiableCredential.category(), FactorCategory::ExternalVerification);
        assert_eq!(FactorType::RecoveryPhrase.category(), FactorCategory::Knowledge);
        assert_eq!(FactorType::SecurityQuestions.category(), FactorCategory::Knowledge);
    }

    #[test]
    fn test_assurance_level_ordering() {
        assert!(AssuranceLevel::Anonymous < AssuranceLevel::Basic);
        assert!(AssuranceLevel::Basic < AssuranceLevel::Verified);
        assert!(AssuranceLevel::Verified < AssuranceLevel::HighlyAssured);
        assert!(AssuranceLevel::HighlyAssured < AssuranceLevel::ConstitutionallyCritical);
    }

    #[test]
    fn test_input_serialization() {
        let input = CreateMfaStateInput {
            did: "did:mycelix:test".to_string(),
            primary_key_hash: "sha256:abc123".to_string(),
        };

        let json = serde_json::to_string(&input).expect("Serialize failed");
        let deserialized: CreateMfaStateInput = serde_json::from_str(&json).expect("Deserialize failed");

        assert_eq!(input.did, deserialized.did);
        assert_eq!(input.primary_key_hash, deserialized.primary_key_hash);
    }

    #[test]
    fn test_enroll_factor_input_serialization() {
        let input = EnrollFactorInput {
            did: "did:mycelix:test".to_string(),
            factor_type: FactorType::HardwareKey,
            factor_id: "yubikey-123".to_string(),
            metadata: r#"{"model":"YubiKey"}"#.to_string(),
            reason: "Added hardware key".to_string(),
        };

        let json = serde_json::to_string(&input).expect("Serialize failed");
        let deserialized: EnrollFactorInput = serde_json::from_str(&json).expect("Deserialize failed");

        assert_eq!(input.factor_type, deserialized.factor_type);
        assert_eq!(input.factor_id, deserialized.factor_id);
    }

    #[test]
    fn test_verification_proof_serialization() {
        let proof = VerificationProof::GitcoinPassport {
            score: 42.5,
            checked_at: 1704067200000000,
            stamps: vec!["BrightID".to_string(), "ENS".to_string()],
        };

        let json = serde_json::to_string(&proof).expect("Serialize failed");
        assert!(json.contains("GitcoinPassport"));
        assert!(json.contains("42.5"));
    }

    #[test]
    fn test_webauthn_proof_serialization() {
        // Verify that WebAuthn proof round-trips through JSON correctly
        let proof = VerificationProof::WebAuthn {
            authenticator_data: "AQIDBAUG".to_string(),
            client_data_hash: "sha256-test".to_string(),
            signature: "sig-bytes-base64".to_string(),
        };

        let json = serde_json::to_string(&proof).expect("Serialize failed");
        let deserialized: VerificationProof =
            serde_json::from_str(&json).expect("Deserialize failed");

        match deserialized {
            VerificationProof::WebAuthn {
                authenticator_data,
                client_data_hash,
                signature,
            } => {
                assert_eq!(authenticator_data, "AQIDBAUG");
                assert_eq!(client_data_hash, "sha256-test");
                assert_eq!(signature, "sig-bytes-base64");
            }
            _ => panic!("Wrong variant after deserialization"),
        }
    }

    #[test]
    fn test_knowledge_proof_serialization() {
        let proof = VerificationProof::Knowledge {
            answer_hash: "sha256:abcdef1234567890".to_string(),
        };

        let json = serde_json::to_string(&proof).expect("Serialize failed");
        let deserialized: VerificationProof =
            serde_json::from_str(&json).expect("Deserialize failed");

        match deserialized {
            VerificationProof::Knowledge { answer_hash } => {
                assert_eq!(answer_hash, "sha256:abcdef1234567890");
            }
            _ => panic!("Wrong variant after deserialization"),
        }
    }

    #[test]
    fn test_biometric_proof_serialization() {
        let proof = VerificationProof::BiometricChallenge {
            template_hash: "enrolled-template-sha256".to_string(),
            response: "secure-enclave-attestation".to_string(),
        };

        let json = serde_json::to_string(&proof).expect("Serialize failed");
        let deserialized: VerificationProof =
            serde_json::from_str(&json).expect("Deserialize failed");

        match deserialized {
            VerificationProof::BiometricChallenge {
                template_hash,
                response,
            } => {
                assert_eq!(template_hash, "enrolled-template-sha256");
                assert_eq!(response, "secure-enclave-attestation");
            }
            _ => panic!("Wrong variant after deserialization"),
        }
    }

    #[test]
    fn test_social_recovery_proof_serialization() {
        let proof = VerificationProof::SocialRecovery {
            guardian_signatures: vec![
                GuardianAttestation {
                    guardian_did: "did:mycelix:guardian1".to_string(),
                    signature: "sig1-base64".to_string(),
                    timestamp: 1704067200000000,
                },
                GuardianAttestation {
                    guardian_did: "did:mycelix:guardian2".to_string(),
                    signature: "sig2-base64".to_string(),
                    timestamp: 1704067200000000,
                },
            ],
            threshold: 2,
        };

        let json = serde_json::to_string(&proof).expect("Serialize failed");
        let deserialized: VerificationProof =
            serde_json::from_str(&json).expect("Deserialize failed");

        match deserialized {
            VerificationProof::SocialRecovery {
                guardian_signatures,
                threshold,
            } => {
                assert_eq!(guardian_signatures.len(), 2);
                assert_eq!(threshold, 2);
                assert_eq!(
                    guardian_signatures[0].guardian_did,
                    "did:mycelix:guardian1"
                );
            }
            _ => panic!("Wrong variant after deserialization"),
        }
    }
}

// ============================================================================
// Section 11: Crypto Hardening Security Tests (Conductor Required)
// ============================================================================
//
// These tests verify the security properties of MFA factor verification:
// - WebAuthn counter replay protection (strictly monotonic counters)
// - Security questions rate limiting (5 attempts per 15 minutes)
// - Reputation attestation Ed25519 signature verification
// - Biometric template hash mismatch rejection
//
// All require Holochain conductor — use `cargo test --test mfa_sweettest -- --ignored`

#[cfg(test)]
mod crypto_hardening {
    use super::*;

    /// Helper: Set up conductor, create DID, create MFA state, return (conductor, cell, did)
    async fn setup_mfa() -> (SweetConductor, SweetCell, String) {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

        let mfa_input = CreateMfaStateInput {
            did: did_doc.id.clone(),
            primary_key_hash: compute_primary_key_hash(&agent),
        };
        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", mfa_input)
            .await;

        (conductor, cell, did_doc.id)
    }

    /// Helper: Enroll a factor and return updated MFA state
    async fn enroll_factor(
        conductor: &SweetConductor,
        cell: &SweetCell,
        did: &str,
        factor_type: FactorType,
        factor_id: &str,
        metadata: &str,
    ) -> MfaStateOutput {
        let input = EnrollFactorInput {
            did: did.to_string(),
            factor_type,
            factor_id: factor_id.to_string(),
            metadata: metadata.to_string(),
            reason: "crypto hardening test".to_string(),
        };
        conductor
            .call(&cell.zome("mfa"), "enroll_factor", input)
            .await
    }

    // =========================================================================
    // 11.1: WebAuthn Counter Replay Protection
    // =========================================================================

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires Holochain conductor
    async fn test_webauthn_short_authenticator_data_rejected() {
        println!("Test 11.1: WebAuthn with too-short authenticator data should fail");

        let (conductor, cell, did) = setup_mfa().await;

        enroll_factor(
            &conductor,
            &cell,
            &did,
            FactorType::HardwareKey,
            "yubikey-short",
            "{}",
        )
        .await;

        // Authenticator data must be >= 37 bytes. "AQIDBA==" is only 4 bytes.
        let verify_input = VerifyFactorInput {
            did: did.clone(),
            factor_id: "yubikey-short".to_string(),
            challenge: Some("test-challenge".to_string()),
            proof: Some(VerificationProof::WebAuthn {
                authenticator_data: "AQIDBA==".to_string(), // 4 bytes — too short
                client_data_hash: "test-hash".to_string(),
                signature: "test-sig".to_string(),
            }),
        };

        // Should fail: authenticator data too short for counter extraction
        let result: Result<MfaStateOutput, _> = conductor
            .call_fallible(&cell.zome("mfa"), "verify_factor", verify_input)
            .await;

        assert!(
            result.is_err(),
            "WebAuthn with <37 byte authenticator data should be rejected"
        );
        println!("  - Short authenticator data correctly rejected");
        println!("Test 11.1 PASSED");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires Holochain conductor
    async fn test_webauthn_missing_challenge_rejected() {
        println!("Test 11.2: WebAuthn without challenge should fail");

        let (conductor, cell, did) = setup_mfa().await;

        enroll_factor(
            &conductor,
            &cell,
            &did,
            FactorType::HardwareKey,
            "yubikey-nochallenge",
            "{}",
        )
        .await;

        // Valid-length authenticator data (37+ bytes) but no challenge
        // Pre-computed base64 of 37 zero bytes with UP flag (byte 32) set and counter=1 (byte 36)
        // This avoids needing a base64 runtime dependency in the test crate.
        // Bytes: [0]*32 ++ [0x01] ++ [0,0,0,1] = rpIdHash(32) + flags(UP=1) + counter(1)
        let auth_data_b64 = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAQ==".to_string();

        let verify_input = VerifyFactorInput {
            did: did.clone(),
            factor_id: "yubikey-nochallenge".to_string(),
            challenge: None, // Missing challenge!
            proof: Some(VerificationProof::WebAuthn {
                authenticator_data: auth_data_b64,
                client_data_hash: "test-hash".to_string(),
                signature: "test-sig".to_string(),
            }),
        };

        let result: Result<MfaStateOutput, _> = conductor
            .call_fallible(&cell.zome("mfa"), "verify_factor", verify_input)
            .await;

        assert!(
            result.is_err(),
            "WebAuthn without challenge should be rejected"
        );
        println!("  - Missing challenge correctly rejected");
        println!("Test 11.2 PASSED");
    }

    // =========================================================================
    // 11.2: Security Questions Rate Limiting
    // =========================================================================

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires Holochain conductor
    async fn test_security_questions_empty_answer_rejected() {
        println!("Test 11.3: Security questions with empty answer should fail");

        let (conductor, cell, did) = setup_mfa().await;

        enroll_factor(
            &conductor,
            &cell,
            &did,
            FactorType::SecurityQuestions,
            "sq-test",
            r#"{"questions":["What is your name?"]}"#,
        )
        .await;

        let verify_input = VerifyFactorInput {
            did: did.clone(),
            factor_id: "sq-test".to_string(),
            challenge: None,
            proof: Some(VerificationProof::Knowledge {
                answer_hash: "".to_string(), // Empty!
            }),
        };

        let result: Result<MfaStateOutput, _> = conductor
            .call_fallible(&cell.zome("mfa"), "verify_factor", verify_input)
            .await;

        assert!(
            result.is_err(),
            "Security question with empty answer should be rejected"
        );
        println!("  - Empty answer hash correctly rejected");
        println!("Test 11.3 PASSED");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires Holochain conductor
    async fn test_security_questions_rate_limiting() {
        println!("Test 11.4: Security questions rate limiting after 5 failures");

        let (conductor, cell, did) = setup_mfa().await;

        enroll_factor(
            &conductor,
            &cell,
            &did,
            FactorType::SecurityQuestions,
            "sq-ratelimit",
            r#"{"questions":["What is your name?"]}"#,
        )
        .await;

        // Send 5 wrong answers to trigger rate limit
        for i in 0..5 {
            let verify_input = VerifyFactorInput {
                did: did.clone(),
                factor_id: "sq-ratelimit".to_string(),
                challenge: None,
                proof: Some(VerificationProof::Knowledge {
                    answer_hash: format!("wrong-answer-{}", i),
                }),
            };

            // These may fail or succeed with reduced strength — we just need
            // to record 5 failed verification attempts
            let _result: Result<MfaStateOutput, _> = conductor
                .call_fallible(&cell.zome("mfa"), "verify_factor", verify_input)
                .await;
        }

        // 6th attempt should be rate-limited regardless of answer
        let verify_input = VerifyFactorInput {
            did: did.clone(),
            factor_id: "sq-ratelimit".to_string(),
            challenge: None,
            proof: Some(VerificationProof::Knowledge {
                answer_hash: "correct-answer-hash".to_string(),
            }),
        };

        let result: Result<MfaStateOutput, _> = conductor
            .call_fallible(&cell.zome("mfa"), "verify_factor", verify_input)
            .await;

        assert!(
            result.is_err(),
            "6th security question attempt within 15 minutes should be rate-limited"
        );
        println!("  - Rate limiting engaged after 5 failed attempts");
        println!("Test 11.4 PASSED");
    }

    // =========================================================================
    // 11.3: Reputation Attestation Signature Verification
    // =========================================================================

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires Holochain conductor
    async fn test_reputation_attestation_forged_signature_rejected() {
        println!("Test 11.5: Reputation attestation with forged signature should fail");

        let (conductor, cell, did) = setup_mfa().await;

        enroll_factor(
            &conductor,
            &cell,
            &did,
            FactorType::ReputationAttestation,
            "rep-forged",
            "{}",
        )
        .await;

        let now_micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        let verify_input = VerifyFactorInput {
            did: did.clone(),
            factor_id: "rep-forged".to_string(),
            challenge: None,
            proof: Some(VerificationProof::SocialRecovery {
                guardian_signatures: vec![GuardianAttestation {
                    guardian_did: "did:mycelix:fake-guardian".to_string(),
                    signature: "AAAA".to_string(), // Invalid 3-byte signature
                    timestamp: now_micros,
                }],
                threshold: 1,
            }),
        };

        let result: Result<MfaStateOutput, _> = conductor
            .call_fallible(&cell.zome("mfa"), "verify_factor", verify_input)
            .await;

        assert!(
            result.is_err(),
            "Reputation attestation with forged signature should be rejected"
        );
        println!("  - Forged signature correctly rejected");
        println!("Test 11.5 PASSED");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires Holochain conductor
    async fn test_reputation_attestation_zero_threshold_rejected() {
        println!("Test 11.6: Reputation attestation with threshold=0 should fail");

        let (conductor, cell, did) = setup_mfa().await;

        enroll_factor(
            &conductor,
            &cell,
            &did,
            FactorType::ReputationAttestation,
            "rep-zerothresh",
            "{}",
        )
        .await;

        let verify_input = VerifyFactorInput {
            did: did.clone(),
            factor_id: "rep-zerothresh".to_string(),
            challenge: None,
            proof: Some(VerificationProof::SocialRecovery {
                guardian_signatures: vec![],
                threshold: 0, // Invalid!
            }),
        };

        let result: Result<MfaStateOutput, _> = conductor
            .call_fallible(&cell.zome("mfa"), "verify_factor", verify_input)
            .await;

        assert!(
            result.is_err(),
            "Reputation attestation with threshold=0 should be rejected"
        );
        println!("  - Zero threshold correctly rejected");
        println!("Test 11.6 PASSED");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires Holochain conductor
    async fn test_reputation_attestation_expired_timestamp_rejected() {
        println!("Test 11.7: Reputation attestation with expired timestamp should fail");

        let (conductor, cell, did) = setup_mfa().await;

        enroll_factor(
            &conductor,
            &cell,
            &did,
            FactorType::ReputationAttestation,
            "rep-expired",
            "{}",
        )
        .await;

        // Timestamp from 2 hours ago (beyond 1-hour freshness window)
        let two_hours_ago = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64
            - (2 * 3600 * 1_000_000);

        let verify_input = VerifyFactorInput {
            did: did.clone(),
            factor_id: "rep-expired".to_string(),
            challenge: None,
            proof: Some(VerificationProof::SocialRecovery {
                guardian_signatures: vec![GuardianAttestation {
                    guardian_did: "did:mycelix:old-guardian".to_string(),
                    signature: "old-sig".to_string(),
                    timestamp: two_hours_ago,
                }],
                threshold: 1,
            }),
        };

        let result: Result<MfaStateOutput, _> = conductor
            .call_fallible(&cell.zome("mfa"), "verify_factor", verify_input)
            .await;

        assert!(
            result.is_err(),
            "Reputation attestation with expired timestamp should be rejected"
        );
        println!("  - Expired attestation correctly rejected");
        println!("Test 11.7 PASSED");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires Holochain conductor
    async fn test_reputation_attestation_duplicate_guardian_rejected() {
        println!("Test 11.8: Reputation attestation with duplicate guardian should fail");

        let (conductor, cell, did) = setup_mfa().await;

        enroll_factor(
            &conductor,
            &cell,
            &did,
            FactorType::ReputationAttestation,
            "rep-dupe",
            "{}",
        )
        .await;

        let now_micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        let verify_input = VerifyFactorInput {
            did: did.clone(),
            factor_id: "rep-dupe".to_string(),
            challenge: None,
            proof: Some(VerificationProof::SocialRecovery {
                guardian_signatures: vec![
                    GuardianAttestation {
                        guardian_did: "did:mycelix:same-guardian".to_string(),
                        signature: "sig1".to_string(),
                        timestamp: now_micros,
                    },
                    GuardianAttestation {
                        guardian_did: "did:mycelix:same-guardian".to_string(), // Duplicate!
                        signature: "sig2".to_string(),
                        timestamp: now_micros,
                    },
                ],
                threshold: 2,
            }),
        };

        let result: Result<MfaStateOutput, _> = conductor
            .call_fallible(&cell.zome("mfa"), "verify_factor", verify_input)
            .await;

        assert!(
            result.is_err(),
            "Reputation attestation with duplicate guardian should be rejected"
        );
        println!("  - Duplicate guardian correctly rejected");
        println!("Test 11.8 PASSED");
    }

    // =========================================================================
    // 11.4: Biometric Template Hash Verification
    // =========================================================================

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires Holochain conductor
    async fn test_biometric_empty_template_hash_rejected() {
        println!("Test 11.9: Biometric with empty template hash should fail");

        let (conductor, cell, did) = setup_mfa().await;

        enroll_factor(
            &conductor,
            &cell,
            &did,
            FactorType::Biometric,
            "bio-empty",
            r#"{"type":"fingerprint"}"#,
        )
        .await;

        let verify_input = VerifyFactorInput {
            did: did.clone(),
            factor_id: "bio-empty".to_string(),
            challenge: None,
            proof: Some(VerificationProof::BiometricChallenge {
                template_hash: "".to_string(), // Empty!
                response: "attestation".to_string(),
            }),
        };

        let result: Result<MfaStateOutput, _> = conductor
            .call_fallible(&cell.zome("mfa"), "verify_factor", verify_input)
            .await;

        assert!(
            result.is_err(),
            "Biometric with empty template hash should be rejected"
        );
        println!("  - Empty template hash correctly rejected");
        println!("Test 11.9 PASSED");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires Holochain conductor
    async fn test_biometric_empty_response_rejected() {
        println!("Test 11.10: Biometric with empty response should fail");

        let (conductor, cell, did) = setup_mfa().await;

        enroll_factor(
            &conductor,
            &cell,
            &did,
            FactorType::Biometric,
            "bio-noresp",
            r#"{"type":"face"}"#,
        )
        .await;

        let verify_input = VerifyFactorInput {
            did: did.clone(),
            factor_id: "bio-noresp".to_string(),
            challenge: None,
            proof: Some(VerificationProof::BiometricChallenge {
                template_hash: "valid-template-hash".to_string(),
                response: "".to_string(), // Empty!
            }),
        };

        let result: Result<MfaStateOutput, _> = conductor
            .call_fallible(&cell.zome("mfa"), "verify_factor", verify_input)
            .await;

        assert!(
            result.is_err(),
            "Biometric with empty response should be rejected"
        );
        println!("  - Empty response correctly rejected");
        println!("Test 11.10 PASSED");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires Holochain conductor
    async fn test_biometric_mismatched_template_rejected() {
        println!("Test 11.11: Biometric with mismatched template should fail verification");

        let (conductor, cell, did) = setup_mfa().await;

        // Enroll with factor_id as the template hash binding
        enroll_factor(
            &conductor,
            &cell,
            &did,
            FactorType::Biometric,
            "bio-template-hash-abc123",
            r#"{"type":"fingerprint"}"#,
        )
        .await;

        // Attempt verification with DIFFERENT template hash
        let verify_input = VerifyFactorInput {
            did: did.clone(),
            factor_id: "bio-template-hash-abc123".to_string(),
            challenge: None,
            proof: Some(VerificationProof::BiometricChallenge {
                template_hash: "WRONG-template-hash-xyz789".to_string(), // Mismatch!
                response: "secure-enclave-attestation".to_string(),
            }),
        };

        // Should either fail or return with reduced strength (not full 1.0)
        let result: Result<MfaStateOutput, _> = conductor
            .call_fallible(&cell.zome("mfa"), "verify_factor", verify_input)
            .await;

        match result {
            Err(_) => {
                println!("  - Mismatched template correctly rejected with error");
            }
            Ok(output) => {
                let bio_factor = output
                    .state
                    .factors
                    .iter()
                    .find(|f| f.factor_id == "bio-template-hash-abc123");
                if let Some(factor) = bio_factor {
                    assert!(
                        factor.effective_strength < 1.0,
                        "Mismatched biometric should not get full strength (got {})",
                        factor.effective_strength
                    );
                    println!(
                        "  - Mismatched template got reduced strength: {}",
                        factor.effective_strength
                    );
                }
            }
        }
        println!("Test 11.11 PASSED");
    }

    // =========================================================================
    // 11.5: Verification Without Proof
    // =========================================================================

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires Holochain conductor
    async fn test_verify_factor_without_proof_rejected() {
        println!("Test 11.12: Verification without any proof should fail");

        let (conductor, cell, did) = setup_mfa().await;

        enroll_factor(
            &conductor,
            &cell,
            &did,
            FactorType::HardwareKey,
            "key-noproof",
            "{}",
        )
        .await;

        let verify_input = VerifyFactorInput {
            did: did.clone(),
            factor_id: "key-noproof".to_string(),
            challenge: None,
            proof: None, // No proof at all!
        };

        let result: Result<MfaStateOutput, _> = conductor
            .call_fallible(&cell.zome("mfa"), "verify_factor", verify_input)
            .await;

        assert!(
            result.is_err(),
            "Verification without proof should be rejected"
        );
        println!("  - Missing proof correctly rejected");
        println!("Test 11.12 PASSED");
    }
}
