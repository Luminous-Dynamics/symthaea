// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Education Zome Integration Tests (Legacy Bridge)
//!
//! Sweettest-based integration tests for the education coordinator zome:
//! - Academic credential creation (W3C VC 2.0)
//! - Credential retrieval by action hash
//! - Query by institution DID
//! - Query by subject DID
//! - ZK commitment verification
//! - Revocation request workflow
//!
//! Uses pre-built .happ bundle instead of linking zome crates directly,
//! avoiding duplicate `__num_entry_types`/`__num_link_types` symbol collisions
//! from multiple integrity zomes.
//!
//! ## Running Tests
//!
//! ```bash
//! # Build the hApp first
//! cd mycelix-identity && hc app pack .
//!
//! # Run tests (require DNA bundle)
//! cargo test --test education_test -- --ignored
//! ```

use holochain::sweettest::*;
use holochain::prelude::*;
use std::path::PathBuf;

// ============================================================================
// DID Mirror Types (needed for dynamic DID creation)
// ============================================================================

/// Mirror of did_registry_integrity::DidDocument (minimal fields for education tests)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DidDocument {
    pub id: String,
    pub controller: AgentPubKey,
    #[serde(rename = "verificationMethod", alias = "verification_method")]
    pub verification_method: Vec<serde_json::Value>,
    pub authentication: Vec<String>,
    #[serde(rename = "keyAgreement", alias = "key_agreement", default)]
    pub key_agreement: Vec<String>,
    pub service: Vec<serde_json::Value>,
    pub created: Timestamp,
    pub updated: Timestamp,
    pub version: u32,
}

/// Decode an entry from a Record via MessagePack deserialization
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
// Mirror types for deserialization (avoids importing zome crates)
// ============================================================================

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DegreeType {
    HighSchool,
    Associate,
    Bachelor,
    Master,
    Doctorate,
    Professional,
    Certificate,
    Diploma,
    Microcredential,
    CourseCompletion,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DnssecStatus {
    Validated,
    Insecure,
    Invalid,
    Unsigned,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum RevocationReason {
    AcademicFraud,
    DegreeRescinded,
    IssuedInError,
    HolderRequest,
    CourtOrder,
    InstitutionChange,
    Other,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct InstitutionalIssuer {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub issuer_type: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub location: Option<InstitutionLocation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accreditation: Option<Vec<Accreditation>>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct InstitutionLocation {
    pub country: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Accreditation {
    pub accreditor: String,
    pub accreditation_type: String,
    pub valid_from: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub valid_until: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AcademicSubject {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name_hash: Option<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub birth_date: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub student_id: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AcademicProof {
    #[serde(rename = "type")]
    pub proof_type: String,
    pub created: String,
    #[serde(rename = "verificationMethod")]
    pub verification_method: String,
    #[serde(rename = "proofPurpose")]
    pub proof_purpose: String,
    #[serde(rename = "proofValue")]
    pub proof_value: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cryptosuite: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub challenge: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub algorithm: Option<u16>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AchievementMetadata {
    pub degree_type: DegreeType,
    pub degree_name: String,
    pub field_of_study: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minors: Option<Vec<String>>,
    pub conferral_date: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpa: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub honors: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cip_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub credits_earned: Option<u32>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DnsDid {
    pub domain: String,
    pub did: String,
    pub txt_record: String,
    pub dnssec: DnssecStatus,
    pub last_verified: Timestamp,
    pub verification_chain: Vec<DnsVerificationRecord>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DnsVerificationRecord {
    pub timestamp: Timestamp,
    pub resolver: String,
    pub dnssec_status: DnssecStatus,
    pub ttl_seconds: u32,
}

/// Mirror of AcademicCredential (returned from zome calls)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AcademicCredential {
    #[serde(rename = "@context")]
    pub context: Vec<String>,
    pub id: String,
    #[serde(rename = "type")]
    pub credential_type: Vec<String>,
    pub issuer: InstitutionalIssuer,
    #[serde(rename = "validFrom")]
    pub valid_from: String,
    #[serde(rename = "validUntil")]
    pub valid_until: Option<String>,
    #[serde(rename = "credentialSubject")]
    pub credential_subject: AcademicSubject,
    pub proof: AcademicProof,
    pub zk_commitment: Vec<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commitment_nonce: Option<Vec<u8>>,
    pub revocation_registry_id: String,
    pub revocation_index: u32,
    pub dns_did: DnsDid,
    pub achievement: AchievementMetadata,
    pub mycelix_schema_id: String,
    pub mycelix_created: Timestamp,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub legacy_import_ref: Option<String>,
}

// ============================================================================
// Coordinator input/output types (mirrors)
// ============================================================================

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct CreateAcademicCredentialInput {
    pub issuer: InstitutionalIssuer,
    pub subject: AcademicSubject,
    pub achievement: AchievementMetadata,
    pub dns_did: DnsDid,
    pub revocation_registry_id: String,
    pub valid_from: String,
    pub valid_until: Option<String>,
    pub proof: AcademicProof,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct CreateAcademicCredentialOutput {
    pub action_hash: ActionHash,
    pub credential_id: String,
    pub revocation_index: u32,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct VerifyCommitmentInput {
    pub credential_id: String,
    pub subject_id: String,
    pub nonce: Vec<u8>,
    pub expected_commitment: Vec<u8>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct RequestRevocationInput {
    pub credential_id: String,
    pub reason: RevocationReason,
    pub explanation: String,
    pub evidence: Option<Vec<String>>,
}

// ============================================================================
// Helpers
// ============================================================================

fn dna_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("dna")
        .join("mycelix_identity_dna.dna")
}

async fn load_dna() -> DnaFile {
    SweetDnaFile::from_bundle(&dna_path())
        .await
        .expect("Failed to load DNA bundle")
}

fn test_issuer(institution_did: &str) -> InstitutionalIssuer {
    InstitutionalIssuer {
        id: institution_did.to_string(),
        name: "Test University".to_string(),
        issuer_type: vec!["EducationalOrganization".to_string()],
        image: None,
        location: Some(InstitutionLocation {
            country: "US".to_string(),
            region: Some("TX".to_string()),
            city: Some("Richardson".to_string()),
        }),
        accreditation: None,
    }
}

fn test_subject(suffix: &str) -> AcademicSubject {
    AcademicSubject {
        id: format!("did:mycelix:student:{}", suffix),
        name: Some(format!("Student {}", suffix)),
        name_hash: None,
        birth_date: Some("1995-06-15".to_string()),
        student_id: Some(format!("STU-{}", suffix)),
    }
}

fn test_achievement() -> AchievementMetadata {
    AchievementMetadata {
        degree_type: DegreeType::Bachelor,
        degree_name: "Bachelor of Science in Computer Science".to_string(),
        field_of_study: "Computer Science".to_string(),
        minors: Some(vec!["Mathematics".to_string()]),
        conferral_date: "2024-05-15".to_string(),
        gpa: Some(3.85),
        honors: Some(vec!["Magna Cum Laude".to_string()]),
        cip_code: Some("11.0701".to_string()),
        credits_earned: Some(120),
    }
}

fn test_dns_did() -> DnsDid {
    DnsDid {
        domain: "registrar.testuniversity.edu".to_string(),
        did: "did:dns:registrar.testuniversity.edu".to_string(),
        txt_record: "_did.registrar.testuniversity.edu".to_string(),
        dnssec: DnssecStatus::Validated,
        last_verified: Timestamp::now(),
        verification_chain: vec![DnsVerificationRecord {
            timestamp: Timestamp::now(),
            resolver: "8.8.8.8".to_string(),
            dnssec_status: DnssecStatus::Validated,
            ttl_seconds: 3600,
        }],
    }
}

fn test_proof() -> AcademicProof {
    AcademicProof {
        proof_type: "DataIntegrityProof".to_string(),
        created: "2024-05-15T10:00:00Z".to_string(),
        verification_method: "did:dns:registrar.testuniversity.edu#key-1".to_string(),
        proof_purpose: "assertionMethod".to_string(),
        proof_value: "z3FXQqFwbSzT5bGbMSKG8KrZ1sLKPqM5xJ2wZvNmKz9nY".to_string(),
        cryptosuite: Some("eddsa-rdfc-2022".to_string()),
        domain: None,
        challenge: None,
        algorithm: None,
    }
}

fn test_create_input(institution_did: &str, subject_suffix: &str) -> CreateAcademicCredentialInput {
    CreateAcademicCredentialInput {
        issuer: test_issuer(institution_did),
        subject: test_subject(subject_suffix),
        achievement: test_achievement(),
        dns_did: test_dns_did(),
        revocation_registry_id: "rev-reg-test-001".to_string(),
        valid_from: "2024-05-15T00:00:00Z".to_string(),
        valid_until: None,
        proof: test_proof(),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod education_integration {
    use super::*;

    /// Setup conductor with DNA, create a DID for the agent (needed for institution auth check)
    async fn setup() -> (SweetConductor, SweetCell, String) {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("education-test", &[dna.clone()])
            .await
            .expect("Failed to install hApp");
        let cell = app.cells()[0].clone();

        // Create a DID for the agent — education zome checks caller DID matches issuer.id
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record)
            .expect("Failed to decode DID document");
        let institution_did = did_doc.id.clone();

        (conductor, cell, institution_did)
    }

    /// Test: Create an academic credential and retrieve it
    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires DNA bundle
    async fn test_create_and_get_credential() {
        let (conductor, cell, institution_did) = setup().await;

        let input = test_create_input(&institution_did, "alice");
        let output: CreateAcademicCredentialOutput = conductor
            .call(&cell.zome("education"), "create_academic_credential", input)
            .await;

        assert!(
            output.credential_id.starts_with("urn:uuid:"),
            "Credential ID should be a URN UUID"
        );

        // Retrieve by action hash
        let credential: Option<AcademicCredential> = conductor
            .call(
                &cell.zome("education"),
                "get_academic_credential",
                output.action_hash.clone(),
            )
            .await;

        let cred = credential.expect("Credential should exist");
        assert_eq!(cred.id, output.credential_id);
        assert_eq!(cred.issuer.name, "Test University");
        assert_eq!(cred.achievement.degree_type, DegreeType::Bachelor);
        assert_eq!(cred.credential_subject.id, "did:mycelix:student:alice");
        assert!(cred.context.contains(&"https://www.w3.org/ns/credentials/v2".to_string()));
        assert!(cred.credential_type.contains(&"AcademicCredential".to_string()));
        assert_eq!(cred.zk_commitment.len(), 32, "ZK commitment should be 32 bytes");
    }

    /// Test: Query credentials by institution DID
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_query_by_institution() {
        let (conductor, cell, institution_did) = setup().await;

        // Create two credentials from same institution
        let input1 = test_create_input(&institution_did, "bob");
        let input2 = test_create_input(&institution_did, "carol");

        let _: CreateAcademicCredentialOutput = conductor
            .call(&cell.zome("education"), "create_academic_credential", input1)
            .await;
        let _: CreateAcademicCredentialOutput = conductor
            .call(&cell.zome("education"), "create_academic_credential", input2)
            .await;

        let hashes: Vec<ActionHash> = conductor
            .call(
                &cell.zome("education"),
                "get_credentials_by_institution",
                institution_did.clone(),
            )
            .await;

        assert!(
            hashes.len() >= 2,
            "Should find at least 2 credentials for the institution"
        );
    }

    /// Test: Query credentials by subject DID
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_query_by_subject() {
        let (conductor, cell, institution_did) = setup().await;

        let input = test_create_input(&institution_did, "dave");
        let output: CreateAcademicCredentialOutput = conductor
            .call(&cell.zome("education"), "create_academic_credential", input)
            .await;

        let hashes: Vec<ActionHash> = conductor
            .call(
                &cell.zome("education"),
                "get_credentials_by_subject",
                "did:mycelix:student:dave".to_string(),
            )
            .await;

        assert_eq!(hashes.len(), 1, "Should find exactly 1 credential for dave");
        assert_eq!(hashes[0], output.action_hash);
    }

    /// Test: ZK commitment verification
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_zk_commitment_verification() {
        let (conductor, cell, institution_did) = setup().await;

        let input = test_create_input(&institution_did, "eve");
        let output: CreateAcademicCredentialOutput = conductor
            .call(&cell.zome("education"), "create_academic_credential", input)
            .await;

        // Get the credential to extract the nonce
        let credential: Option<AcademicCredential> = conductor
            .call(
                &cell.zome("education"),
                "get_academic_credential",
                output.action_hash.clone(),
            )
            .await;

        let cred = credential.expect("Credential should exist");
        let nonce = cred
            .commitment_nonce
            .expect("Commitment nonce should be present");

        // Verify using the zome function
        let verify_input = VerifyCommitmentInput {
            credential_id: cred.id.clone(),
            subject_id: cred.credential_subject.id.clone(),
            nonce,
            expected_commitment: cred.zk_commitment.clone(),
        };

        let is_valid: bool = conductor
            .call(
                &cell.zome("education"),
                "verify_zk_commitment",
                verify_input,
            )
            .await;

        assert!(is_valid, "ZK commitment should verify with correct nonce");
    }

    /// Test: Revocation request
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_revocation_request() {
        let (conductor, cell, institution_did) = setup().await;

        let input = test_create_input(&institution_did, "frank");
        let output: CreateAcademicCredentialOutput = conductor
            .call(&cell.zome("education"), "create_academic_credential", input)
            .await;

        let revoke_input = RequestRevocationInput {
            credential_id: output.credential_id,
            reason: RevocationReason::IssuedInError,
            explanation: "Credential was issued to wrong student".to_string(),
            evidence: Some(vec!["internal-ticket-12345".to_string()]),
        };

        let request_hash: ActionHash = conductor
            .call(
                &cell.zome("education"),
                "request_academic_revocation",
                revoke_input,
            )
            .await;

        assert_ne!(request_hash, ActionHash::from_raw_36(vec![0; 36]), "Revocation request should return a valid action hash");
    }
}
