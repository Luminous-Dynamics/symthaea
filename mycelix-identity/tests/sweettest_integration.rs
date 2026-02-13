//! # Mycelix Identity - Sweettest Integration Tests
//!
//! Comprehensive integration tests using Holochain's sweettest framework.
//! Tests cover DID creation, resolution, service management, credential schemas,
//! identity bridge operations, and cross-zome identity lifecycle.
//!
//! ## Running Tests
//!
//! ```bash
//! # Ensure the DNA bundle exists
//! ls mycelix-identity/dna/mycelix_identity_dna.dna
//!
//! # Run tests (requires Holochain conductor via nix develop)
//! cargo test --test sweettest_integration -- --ignored
//! ```

use holochain::sweettest::*;
use holochain::prelude::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types (avoids importing zome crates / duplicate symbols)
// ============================================================================

/// Mirror of did_registry_integrity::DidDocument
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DidDocument {
    pub id: String,
    pub controller: AgentPubKey,
    pub verification_method: Vec<VerificationMethod>,
    pub authentication: Vec<String>,
    pub service: Vec<ServiceEndpoint>,
    pub created: Timestamp,
    pub updated: Timestamp,
    pub version: u32,
}

/// Mirror of did_registry_integrity::VerificationMethod
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VerificationMethod {
    pub id: String,
    pub type_: String,
    pub controller: String,
    pub public_key_multibase: String,
}

/// Mirror of did_registry_integrity::ServiceEndpoint
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ServiceEndpoint {
    pub id: String,
    pub type_: String,
    pub service_endpoint: String,
}

/// Mirror of did_registry_integrity::DidDeactivation
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DidDeactivation {
    pub did: String,
    pub reason: String,
    pub deactivated_at: Timestamp,
}

/// Mirror of credential_schema_integrity::CredentialSchema
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CredentialSchema {
    pub id: String,
    pub name: String,
    pub description: String,
    pub version: String,
    pub author: String,
    pub schema: String,
    pub required_fields: Vec<String>,
    pub optional_fields: Vec<String>,
    pub credential_type: Vec<String>,
    pub default_expiration: u64,
    pub revocable: bool,
    pub active: bool,
    pub created: Timestamp,
    pub updated: Timestamp,
}

/// Mirror of bridge coordinator::RegisterHappInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegisterHappInput {
    pub happ_id: String,
    pub happ_name: String,
    pub capabilities: Vec<String>,
}

/// Mirror of bridge coordinator::QueryIdentityInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QueryIdentityInput {
    pub did: String,
    pub source_happ: String,
    pub requested_fields: Vec<String>,
}

/// Mirror of bridge coordinator::IdentityVerificationResult
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct IdentityVerificationResult {
    pub verification_hash: ActionHash,
    pub did: String,
    pub is_valid: bool,
    pub matl_score: f64,
    pub credential_count: u32,
}

/// Mirror of bridge coordinator::ReportReputationInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReportReputationInput {
    pub did: String,
    pub source_happ: String,
    pub score: f64,
    pub interactions: u64,
}

/// Mirror of bridge coordinator::AggregatedReputation
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AggregatedReputation {
    pub did: String,
    pub aggregate_score: f64,
    pub sources: Vec<ReputationSource>,
    pub total_interactions: u64,
}

/// Mirror of bridge coordinator::ReputationSource
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReputationSource {
    pub source_happ: String,
    pub score: f64,
    pub interactions: u64,
}

/// Mirror of verifiable_credential coordinator::IssueCredentialInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct IssueCredentialInput {
    pub subject_did: String,
    pub schema_id: String,
    pub claims: serde_json::Value,
    pub credential_types: Vec<String>,
    pub issuer_name: Option<String>,
    pub expiration_days: Option<u32>,
    pub enable_revocation: bool,
}

/// Mirror of verifiable_credential coordinator::VerificationResult (VC)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VcVerificationResult {
    pub credential_id: String,
    pub valid: bool,
    pub checks_passed: Vec<String>,
    pub errors: Vec<String>,
    pub verified_at: Timestamp,
}

/// Mirror of revocation coordinator::RevokeInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RevokeInput {
    pub credential_id: String,
    pub issuer_did: String,
    pub reason: String,
    pub effective_from: Option<Timestamp>,
}

/// Mirror of verifiable_credential_integrity::VerifiableCredential (partial)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VerifiableCredential {
    pub id: String,
    #[serde(rename = "type")]
    pub credential_type: Vec<String>,
}

/// Mirror of trust_credential coordinator::IssueTrustCredentialInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct IssueTrustCredentialInput {
    pub subject_did: String,
    pub issuer_did: String,
    pub kvector_commitment: Vec<u8>,
    pub range_proof: Vec<u8>,
    pub trust_score_lower: f32,
    pub trust_score_upper: f32,
    pub expires_at: Option<Timestamp>,
    pub supersedes: Option<String>,
}

/// Mirror of trust_credential_integrity::TrustCredential
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrustCredential {
    pub id: String,
    pub subject_did: String,
    pub issuer_did: String,
    pub kvector_commitment: Vec<u8>,
    pub range_proof: Vec<u8>,
    pub trust_score_range: TrustScoreRange,
    pub trust_tier: TrustTier,
    pub issued_at: Timestamp,
    pub expires_at: Option<Timestamp>,
    pub revoked: bool,
    pub revocation_reason: Option<String>,
    pub supersedes: Option<String>,
}

/// Mirror of trust_credential_integrity::TrustScoreRange
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrustScoreRange {
    pub lower: f32,
    pub upper: f32,
}

/// Mirror of trust_credential_integrity::TrustTier
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum TrustTier {
    Observer,
    Basic,
    Standard,
    Elevated,
    Guardian,
}

/// Mirror of trust_credential_integrity::KVectorComponent
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum KVectorComponent {
    Reputation,
    Activity,
    Integrity,
    Performance,
    Membership,
    Stake,
    History,
    Topology,
}

/// Mirror of trust_credential_integrity::AttestationStatus
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AttestationStatus {
    Pending,
    Fulfilled,
    Declined,
    Expired,
    Cancelled,
}

/// Mirror of trust_credential_integrity::AttestationRequest
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AttestationRequest {
    pub id: String,
    pub requester_did: String,
    pub subject_did: String,
    pub components: Vec<KVectorComponent>,
    pub min_trust_score: Option<f32>,
    pub min_tier: Option<TrustTier>,
    pub purpose: String,
    pub expires_at: Timestamp,
    pub status: AttestationStatus,
    pub created_at: Timestamp,
}

/// Mirror of trust_credential coordinator::RequestAttestationInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RequestAttestationInput {
    pub requester_did: String,
    pub subject_did: String,
    pub components: Vec<KVectorComponent>,
    pub min_trust_score: Option<f32>,
    pub min_tier: Option<TrustTier>,
    pub purpose: String,
    pub expires_at: Timestamp,
}

/// Mirror of trust_credential coordinator::FulfillAttestationInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FulfillAttestationInput {
    pub request_id: String,
    pub subject_did: String,
    pub kvector_commitment: Vec<u8>,
    pub range_proof: Vec<u8>,
    pub trust_score_lower: f32,
    pub trust_score_upper: f32,
    pub expires_at: Option<Timestamp>,
}

/// Mirror of trust_credential coordinator::FulfillAttestationResult
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FulfillAttestationResult {
    pub credential_record: Record,
    pub request_id: String,
}

/// Mirror of trust_credential coordinator::TrustVerificationResult
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrustVerificationResult {
    pub credential_id: String,
    pub commitment_valid: bool,
    pub tier_consistent: bool,
    pub not_revoked: bool,
    pub not_expired: bool,
    pub proof_format_valid: bool,
    pub message: String,
}

/// Mirror of recovery_integrity::RecoveryConfig
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RecoveryConfig {
    pub did: String,
    pub owner: AgentPubKey,
    pub trustees: Vec<String>,
    pub threshold: u32,
    pub time_lock: u64,
    pub active: bool,
    pub created: Timestamp,
    pub updated: Timestamp,
}

/// Mirror of recovery_integrity::RecoveryRequest
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RecoveryRequest {
    pub id: String,
    pub did: String,
    pub new_agent: AgentPubKey,
    pub initiated_by: String,
    pub reason: String,
    pub status: RecoveryStatus,
    pub created: Timestamp,
    pub time_lock_expires: Option<Timestamp>,
}

/// Mirror of recovery_integrity::RecoveryVote
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RecoveryVote {
    pub request_id: String,
    pub trustee: String,
    pub vote: VoteDecision,
    pub comment: Option<String>,
    pub voted_at: Timestamp,
}

/// Mirror of recovery_integrity::RecoveryStatus
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum RecoveryStatus {
    Pending,
    Approved,
    ReadyToExecute,
    Completed,
    Rejected,
    Cancelled,
}

/// Mirror of recovery_integrity::VoteDecision
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum VoteDecision {
    Approve,
    Reject,
    Abstain,
}

/// Mirror of recovery coordinator::SetupRecoveryInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SetupRecoveryInput {
    pub did: String,
    pub trustees: Vec<String>,
    pub threshold: u32,
    pub time_lock: Option<u64>,
}

/// Mirror of recovery coordinator::InitiateRecoveryInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct InitiateRecoveryInput {
    pub did: String,
    pub initiator_did: String,
    pub new_agent: AgentPubKey,
    pub reason: String,
}

/// Mirror of recovery coordinator::VoteOnRecoveryInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VoteOnRecoveryInput {
    pub request_id: String,
    pub trustee_did: String,
    pub vote: VoteDecision,
    pub comment: Option<String>,
}

/// Mirror of did_registry coordinator::RotateKeyInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RotateKeyInput {
    pub old_key_id: String,
    pub new_method: VerificationMethod,
}

/// Mirror of did_registry coordinator::UpdateDidInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UpdateDidInput {
    #[serde(rename = "verificationMethod", alias = "verification_method")]
    pub verification_method: Option<Vec<VerificationMethod>>,
    pub authentication: Option<Vec<String>>,
    #[serde(rename = "keyAgreement", alias = "key_agreement")]
    pub key_agreement: Option<Vec<String>>,
    pub service: Option<Vec<ServiceEndpoint>>,
}

/// Mirror of revocation coordinator::SuspendInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SuspendInput {
    pub credential_id: String,
    pub issuer_did: String,
    pub reason: String,
    pub suspension_end: Timestamp,
}

/// Mirror of revocation coordinator::ReinstateInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReinstateInput {
    pub credential_id: String,
    pub issuer_did: String,
    pub reason: String,
}

/// Mirror of revocation_integrity::RevocationStatus
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum RevocationStatus {
    Active,
    Suspended,
    Revoked,
}

/// Mirror of revocation coordinator::RevocationCheckResult
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RevocationCheckResult {
    pub credential_id: String,
    pub status: RevocationStatus,
    pub reason: Option<String>,
    pub checked_at: Timestamp,
}

/// Mirror of revocation coordinator::BatchRevokeInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BatchRevokeInput {
    pub credential_ids: Vec<String>,
    pub issuer_did: String,
    pub reason: String,
    pub effective_from: Option<Timestamp>,
    pub revocation_list_id: Option<String>,
}

/// Mirror of revocation coordinator::BatchRevokeResult
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BatchRevokeResult {
    pub revoked_count: u32,
    pub failed_count: u32,
    pub revoked: Vec<String>,
    pub failed: Vec<BatchItemError>,
}

/// Mirror of revocation coordinator::BatchItemError
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BatchItemError {
    pub credential_id: String,
    pub error: String,
}

/// Mirror of revocation coordinator::CreateRevocationListInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateRevocationListInput {
    pub id: String,
    pub issuer_did: String,
}

/// Mirror of revocation_integrity::RevocationList
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RevocationList {
    pub id: String,
    pub issuer: String,
    pub revoked: Vec<String>,
    pub updated: Timestamp,
    pub version: u32,
}

/// Mirror of verifiable_credential coordinator::CredentialStatusResponse
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CredentialStatusResponse {
    pub credential_id: String,
    pub is_valid: bool,
    pub status_type: String,
    pub reason: Option<String>,
    pub checked_at: Timestamp,
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
// DID Registry Tests
// ============================================================================

#[cfg(test)]
mod did_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires Holochain conductor
    async fn test_create_and_resolve_did() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Create DID
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;

        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

        assert!(did_doc.id.starts_with("did:mycelix:"), "DID must use mycelix method");
        assert_eq!(did_doc.controller, agent, "Controller must match creating agent");
        assert!(!did_doc.verification_method.is_empty(), "Must have verification method");
        assert_eq!(did_doc.version, 1, "Initial version must be 1");

        // Resolve by DID string
        let resolved: Option<Record> = conductor
            .call(&cell.zome("did_registry"), "resolve_did", did_doc.id.clone())
            .await;

        assert!(resolved.is_some(), "DID resolution should succeed");

        let resolved_doc: DidDocument =
            decode_entry(&resolved.unwrap()).expect("Failed to decode resolved DID");
        assert_eq!(resolved_doc.id, did_doc.id, "Resolved DID must match");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_did_by_agent() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Create DID
        let _: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;

        // Get DID by agent pub key
        let did_record: Option<Record> = conductor
            .call(&cell.zome("did_registry"), "get_did_document", agent.clone())
            .await;

        assert!(did_record.is_some(), "Should find DID for agent");

        let did_doc: DidDocument =
            decode_entry(&did_record.unwrap()).expect("Failed to decode");
        assert_eq!(did_doc.controller, agent, "Controller must match agent");

        // Test get_my_did convenience function
        let my_did: Option<Record> = conductor
            .call(&cell.zome("did_registry"), "get_my_did", ())
            .await;

        assert!(my_did.is_some(), "get_my_did should return DID");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_add_service_endpoint() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Create DID
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;

        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode");

        // Add service endpoint
        let service = ServiceEndpoint {
            id: format!("{}#messaging", did_doc.id),
            type_: "MessagingService".to_string(),
            service_endpoint: "https://messaging.mycelix.net/agent123".to_string(),
        };

        let updated_record: Record = conductor
            .call(&cell.zome("did_registry"), "add_service_endpoint", service.clone())
            .await;

        let updated_doc: DidDocument =
            decode_entry(&updated_record).expect("Failed to decode");

        assert_eq!(updated_doc.service.len(), 1, "Should have one service");
        assert_eq!(updated_doc.service[0].id, service.id, "Service ID must match");
        assert_eq!(updated_doc.version, 2, "Version should increment");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_deactivate_did() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Create DID
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;

        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode");

        // Check active before deactivation
        let is_active_before: bool = conductor
            .call(&cell.zome("did_registry"), "is_did_active", did_doc.id.clone())
            .await;

        assert!(is_active_before, "DID should be active initially");

        // Deactivate
        let reason = "Key rotation required".to_string();
        let deactivation_record: Record = conductor
            .call(&cell.zome("did_registry"), "deactivate_did", reason.clone())
            .await;

        let deactivation: DidDeactivation =
            decode_entry(&deactivation_record).expect("Failed to decode deactivation");

        assert_eq!(deactivation.did, did_doc.id, "Deactivation must reference DID");
        assert_eq!(deactivation.reason, reason, "Reason must match");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_unique_dids_per_agent() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;

        let mut dids = Vec::new();

        for i in 0..3 {
            let app = conductor
                .setup_app(&format!("test-app-{}", i), &[dna.clone()])
                .await
                .unwrap();
            let cell = app.cells()[0].clone();

            let did_record: Record = conductor
                .call(&cell.zome("did_registry"), "create_did", ())
                .await;

            let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode");
            dids.push(did_doc.id);
        }

        // Verify all DIDs are unique
        let unique_count = dids.iter().collect::<std::collections::HashSet<_>>().len();
        assert_eq!(unique_count, 3, "All DIDs must be unique");
    }
}

// ============================================================================
// Credential Schema Tests
// ============================================================================

#[cfg(test)]
mod schema_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_credential_schema() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Create DID first
        let _: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;

        let author_did = format!("did:mycelix:{}", agent);
        let now = Timestamp::now();

        let schema = CredentialSchema {
            id: "mycelix:schema:education:degree:v1".to_string(),
            name: "University Degree".to_string(),
            description: "Schema for university degree credentials".to_string(),
            version: "1.0.0".to_string(),
            author: author_did.clone(),
            schema: r#"{"type":"object","properties":{"degree":{"type":"string"}}}"#.to_string(),
            required_fields: vec!["degree".to_string(), "university".to_string()],
            optional_fields: vec!["honors".to_string()],
            credential_type: vec![
                "VerifiableCredential".to_string(),
                "EducationCredential".to_string(),
            ],
            default_expiration: 86400 * 365 * 4,
            revocable: true,
            active: true,
            created: now,
            updated: now,
        };

        let schema_record: Record = conductor
            .call(&cell.zome("credential_schema"), "create_schema", schema.clone())
            .await;

        let created_schema: CredentialSchema =
            decode_entry(&schema_record).expect("Failed to decode schema");

        assert_eq!(created_schema.id, schema.id, "Schema ID must match");
        assert_eq!(created_schema.author, author_did, "Author must match");
        assert!(created_schema.active, "Schema should be active");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_schemas_by_author() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Create DID
        let _: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;

        let author_did = format!("did:mycelix:{}", agent);
        let now = Timestamp::now();

        // Create multiple schemas
        for i in 0..3 {
            let schema = CredentialSchema {
                id: format!("mycelix:schema:test:schema{}:v1", i),
                name: format!("Test Schema {}", i),
                description: "Test schema".to_string(),
                version: "1.0.0".to_string(),
                author: author_did.clone(),
                schema: r#"{"type":"object"}"#.to_string(),
                required_fields: vec![],
                optional_fields: vec![],
                credential_type: vec!["VerifiableCredential".to_string()],
                default_expiration: 86400,
                revocable: true,
                active: true,
                created: now,
                updated: now,
            };

            let _: Record = conductor
                .call(&cell.zome("credential_schema"), "create_schema", schema)
                .await;
        }

        // Get schemas by author
        let schemas: Vec<Record> = conductor
            .call(
                &cell.zome("credential_schema"),
                "get_schemas_by_author",
                author_did.clone(),
            )
            .await;

        assert!(schemas.len() >= 3, "Should have at least 3 schemas");

        for record in &schemas {
            let schema: CredentialSchema =
                decode_entry(record).expect("Failed to decode schema");
            assert_eq!(schema.author, author_did, "All schemas should belong to author");
        }
    }
}

// ============================================================================
// Identity Bridge Tests
// ============================================================================

#[cfg(test)]
mod bridge_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_register_happ_and_query_identity() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Create DID
        let _: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;

        let did = format!("did:mycelix:{}", agent);

        // Register hApp
        let registration = RegisterHappInput {
            happ_id: "test-finance".to_string(),
            happ_name: "Test Finance Module".to_string(),
            capabilities: vec!["identity_query".to_string(), "reputation".to_string()],
        };

        let _: Record = conductor
            .call(&cell.zome("identity_bridge"), "register_happ", registration)
            .await;

        // Query identity
        let query = QueryIdentityInput {
            did: did.clone(),
            source_happ: "test-finance".to_string(),
            requested_fields: vec!["is_valid".to_string(), "matl_score".to_string()],
        };

        let verification: IdentityVerificationResult = conductor
            .call(&cell.zome("identity_bridge"), "query_identity", query)
            .await;

        assert_eq!(verification.did, did, "DID must match");
        assert!(verification.is_valid, "DID should be valid");
        assert!(
            verification.matl_score >= 0.0 && verification.matl_score <= 1.0,
            "MATL score must be 0-1"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_report_and_aggregate_reputation() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Create DID
        let _: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;

        let did = format!("did:mycelix:{}", agent);

        // Report reputation from multiple sources
        let reports = vec![
            ReportReputationInput {
                did: did.clone(),
                source_happ: "finance".to_string(),
                score: 0.9,
                interactions: 100,
            },
            ReportReputationInput {
                did: did.clone(),
                source_happ: "governance".to_string(),
                score: 0.8,
                interactions: 50,
            },
            ReportReputationInput {
                did: did.clone(),
                source_happ: "energy".to_string(),
                score: 0.85,
                interactions: 75,
            },
        ];

        for report in reports {
            let _: Record = conductor
                .call(&cell.zome("identity_bridge"), "report_reputation", report)
                .await;
        }

        // Get aggregated reputation
        let aggregated: AggregatedReputation = conductor
            .call(&cell.zome("identity_bridge"), "get_reputation", did.clone())
            .await;

        assert_eq!(aggregated.did, did, "DID must match");
        assert!(aggregated.sources.len() >= 3, "Should have 3+ sources");
        assert!(
            aggregated.aggregate_score >= 0.0 && aggregated.aggregate_score <= 1.0,
            "Score must be 0-1"
        );

        // Verify weighted average
        let expected = (0.9 * 100.0 + 0.8 * 50.0 + 0.85 * 75.0) / (100.0 + 50.0 + 75.0);
        let tolerance = 0.01;
        assert!(
            (aggregated.aggregate_score - expected).abs() < tolerance,
            "Aggregate should be weighted average: expected {:.4}, got {:.4}",
            expected,
            aggregated.aggregate_score
        );
    }
}

// ============================================================================
// Full Lifecycle Test
// ============================================================================

#[cfg(test)]
mod lifecycle_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_complete_identity_lifecycle() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let _agent = app.agent().clone();

        // 1. Create DID
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;

        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

        // 2. Add service endpoint
        let service = ServiceEndpoint {
            id: format!("{}#profile", did_doc.id),
            type_: "ProfileService".to_string(),
            service_endpoint: "https://mycelix.net/profile/user123".to_string(),
        };

        let _: Record = conductor
            .call(&cell.zome("did_registry"), "add_service_endpoint", service)
            .await;

        // 3. Create credential schema
        let now = Timestamp::now();
        let schema = CredentialSchema {
            id: "mycelix:schema:verification:v1".to_string(),
            name: "Identity Verification".to_string(),
            description: "Basic identity verification schema".to_string(),
            version: "1.0.0".to_string(),
            author: did_doc.id.clone(),
            schema: r#"{"type":"object","properties":{"verified":{"type":"boolean"}}}"#.to_string(),
            required_fields: vec!["verified".to_string()],
            optional_fields: vec![],
            credential_type: vec!["VerifiableCredential".to_string()],
            default_expiration: 86400 * 365,
            revocable: true,
            active: true,
            created: now,
            updated: now,
        };

        let _: Record = conductor
            .call(&cell.zome("credential_schema"), "create_schema", schema)
            .await;

        // 4. Report reputation
        let reputation = ReportReputationInput {
            did: did_doc.id.clone(),
            source_happ: "test-ecosystem".to_string(),
            score: 0.95,
            interactions: 200,
        };

        let _: Record = conductor
            .call(&cell.zome("identity_bridge"), "report_reputation", reputation)
            .await;

        // 5. Verify complete state
        let final_did: Option<Record> = conductor
            .call(&cell.zome("did_registry"), "get_my_did", ())
            .await;

        assert!(final_did.is_some(), "DID should exist");

        let matl_score: f64 = conductor
            .call(&cell.zome("identity_bridge"), "get_matl_score", did_doc.id.clone())
            .await;

        assert!(matl_score > 0.0, "MATL score should be positive after reputation report");
    }
}

// ============================================================================
// MFA (Multi-Factor Authentication) Tests
// ============================================================================

#[cfg(test)]
mod mfa_tests {
    use super::*;

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

    /// Mirror of mfa_integrity::AssuranceLevel
    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, PartialOrd)]
    pub enum AssuranceLevel {
        Anonymous,
        Basic,
        Verified,
        HighlyAssured,
        ConstitutionallyCritical,
    }

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

    /// Mirror of mfa_coordinator::FlEligibilityResult
    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    pub struct FlEligibilityResult {
        pub eligible: bool,
        pub assurance_level: AssuranceLevel,
        pub effective_strength: f32,
        pub denial_reasons: Vec<String>,
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore] // Requires Holochain conductor
    async fn test_create_mfa_state() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Create DID first
        let _: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;

        let did = format!("did:mycelix:{}", agent);
        // Use agent string representation as a simple hash for testing
        let primary_key_hash = format!("sha256:{}", agent);

        // Create MFA state
        let input = CreateMfaStateInput {
            did: did.clone(),
            primary_key_hash: primary_key_hash.clone(),
        };

        let output: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", input)
            .await;

        assert_eq!(output.state.did, did, "DID must match");
        assert_eq!(output.state.owner, agent, "Owner must be creating agent");
        assert_eq!(output.state.factors.len(), 1, "Should have one factor (primary key)");
        assert_eq!(
            output.state.factors[0].factor_type,
            FactorType::PrimaryKeyPair,
            "First factor must be PrimaryKeyPair"
        );
        assert_eq!(output.state.version, 1, "Initial version must be 1");
        assert_eq!(
            output.assurance.level,
            AssuranceLevel::Basic,
            "Initial level should be Basic"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_enroll_multiple_factors() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Create DID
        let _: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;

        let did = format!("did:mycelix:{}", agent);

        // Create MFA state
        let create_input = CreateMfaStateInput {
            did: did.clone(),
            primary_key_hash: "primary-key-hash".to_string(),
        };

        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", create_input)
            .await;

        // Enroll hardware key
        let hw_input = EnrollFactorInput {
            did: did.clone(),
            factor_type: FactorType::HardwareKey,
            factor_id: "yubikey-serial-12345".to_string(),
            metadata: r#"{"model":"YubiKey 5 NFC"}"#.to_string(),
            reason: "Added hardware security key".to_string(),
        };

        let after_hw: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "enroll_factor", hw_input)
            .await;

        assert_eq!(after_hw.state.factors.len(), 2, "Should have 2 factors");
        assert_eq!(after_hw.state.version, 2, "Version should increment");

        // Enroll Gitcoin Passport
        let gp_input = EnrollFactorInput {
            did: did.clone(),
            factor_type: FactorType::GitcoinPassport,
            factor_id: "passport-0x1234567890".to_string(),
            metadata: r#"{"score":42.5,"stamps":15}"#.to_string(),
            reason: "Verified via Gitcoin Passport".to_string(),
        };

        let after_gp: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "enroll_factor", gp_input)
            .await;

        assert_eq!(after_gp.state.factors.len(), 3, "Should have 3 factors");
        assert!(
            after_gp.assurance.level >= AssuranceLevel::Verified,
            "With 3 factors from different categories, should be at least Verified"
        );
        assert!(
            after_gp.assurance.category_count >= 2,
            "Should have factors from multiple categories"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_revoke_factor() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Create DID and MFA state
        let _: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;

        let did = format!("did:mycelix:{}", agent);

        let create_input = CreateMfaStateInput {
            did: did.clone(),
            primary_key_hash: "primary-key-hash".to_string(),
        };

        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", create_input)
            .await;

        // Enroll a factor
        let enroll_input = EnrollFactorInput {
            did: did.clone(),
            factor_type: FactorType::RecoveryPhrase,
            factor_id: "bip39-phrase-hash".to_string(),
            metadata: "{}".to_string(),
            reason: "Added recovery phrase".to_string(),
        };

        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "enroll_factor", enroll_input)
            .await;

        // Revoke it
        let revoke_input = RevokeFactorInput {
            did: did.clone(),
            factor_id: "bip39-phrase-hash".to_string(),
            reason: "No longer needed".to_string(),
        };

        let after_revoke: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "revoke_factor", revoke_input)
            .await;

        assert_eq!(
            after_revoke.state.factors.len(),
            1,
            "Should be back to 1 factor"
        );
        assert_eq!(
            after_revoke.state.factors[0].factor_type,
            FactorType::PrimaryKeyPair,
            "Only primary key should remain"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_cannot_revoke_last_factor() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Create DID and MFA state
        let _: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;

        let did = format!("did:mycelix:{}", agent);

        let create_input = CreateMfaStateInput {
            did: did.clone(),
            primary_key_hash: "primary-key-hash".to_string(),
        };

        let initial: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", create_input)
            .await;

        // Try to revoke the only factor - should fail
        let revoke_input = RevokeFactorInput {
            did: did.clone(),
            factor_id: initial.state.factors[0].factor_id.clone(),
            reason: "Trying to revoke primary key".to_string(),
        };

        let result: Result<MfaStateOutput, _> = conductor
            .call_fallible(&cell.zome("mfa"), "revoke_factor", revoke_input)
            .await;

        assert!(result.is_err(), "Should not be able to revoke last factor");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_fl_eligibility_requirements() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Create DID and MFA state
        let _: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;

        let did = format!("did:mycelix:{}", agent);

        let create_input = CreateMfaStateInput {
            did: did.clone(),
            primary_key_hash: "primary-key-hash".to_string(),
        };

        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", create_input)
            .await;

        // Check eligibility with only primary key - should fail
        let basic_eligibility: FlEligibilityResult = conductor
            .call(&cell.zome("mfa"), "check_fl_eligibility", did.clone())
            .await;

        assert!(
            !basic_eligibility.eligible,
            "Should not be FL eligible with only primary key"
        );
        assert!(
            !basic_eligibility.denial_reasons.is_empty(),
            "Should have denial reasons"
        );

        // Enroll ExternalVerification factor (Gitcoin Passport)
        let gp_input = EnrollFactorInput {
            did: did.clone(),
            factor_type: FactorType::GitcoinPassport,
            factor_id: "passport-verified".to_string(),
            metadata: r#"{"score":50}"#.to_string(),
            reason: "Gitcoin Passport verification".to_string(),
        };

        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "enroll_factor", gp_input)
            .await;

        // Add another factor for category diversity
        let hw_input = EnrollFactorInput {
            did: did.clone(),
            factor_type: FactorType::HardwareKey,
            factor_id: "yubikey-for-fl".to_string(),
            metadata: "{}".to_string(),
            reason: "Added for FL participation".to_string(),
        };

        let _: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "enroll_factor", hw_input)
            .await;

        // Now check eligibility - should be eligible
        let full_eligibility: FlEligibilityResult = conductor
            .call(&cell.zome("mfa"), "check_fl_eligibility", did.clone())
            .await;

        assert!(
            full_eligibility.eligible,
            "Should be FL eligible with Cryptographic + ExternalVerification factors"
        );
        assert!(
            full_eligibility.denial_reasons.is_empty(),
            "Should have no denial reasons when eligible"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_assurance_level_calculation() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-app", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();
        let agent = app.agent().clone();

        // Create DID and MFA state
        let _: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;

        let did = format!("did:mycelix:{}", agent);

        let create_input = CreateMfaStateInput {
            did: did.clone(),
            primary_key_hash: "primary-key-hash".to_string(),
        };

        // E1: Basic (1 factor)
        let e1: MfaStateOutput = conductor
            .call(&cell.zome("mfa"), "create_mfa_state", create_input)
            .await;
        assert_eq!(e1.assurance.level, AssuranceLevel::Basic);

        // Add factors from different categories
        let factors = vec![
            EnrollFactorInput {
                did: did.clone(),
                factor_type: FactorType::GitcoinPassport,
                factor_id: "gp-1".to_string(),
                metadata: "{}".to_string(),
                reason: "test".to_string(),
            },
            EnrollFactorInput {
                did: did.clone(),
                factor_type: FactorType::Biometric,
                factor_id: "bio-1".to_string(),
                metadata: "{}".to_string(),
                reason: "test".to_string(),
            },
            EnrollFactorInput {
                did: did.clone(),
                factor_type: FactorType::RecoveryPhrase,
                factor_id: "recovery-1".to_string(),
                metadata: "{}".to_string(),
                reason: "test".to_string(),
            },
            EnrollFactorInput {
                did: did.clone(),
                factor_type: FactorType::SocialRecovery,
                factor_id: "social-1".to_string(),
                metadata: "{}".to_string(),
                reason: "test".to_string(),
            },
        ];

        let mut last_output = e1;
        for factor in factors {
            last_output = conductor
                .call(&cell.zome("mfa"), "enroll_factor", factor)
                .await;
        }

        // With 5 factors from 5 categories, should be HighlyAssured or ConstitutionallyCritical
        assert!(
            last_output.assurance.level >= AssuranceLevel::HighlyAssured,
            "With 5 factors from 5 categories, should be at least HighlyAssured"
        );
        assert!(
            last_output.assurance.category_count >= 4,
            "Should have 4+ categories"
        );
    }
}

// ============================================================================
// Security edge case tests (SEC-005, SEC-017, FIND-001, FIND-003)
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
async fn test_reject_malformed_multibase_key_no_prefix() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor.setup_app("test", &[dna]).await.unwrap();
    let cell = app.cells()[0].clone();

    // Create DID first
    let _record: Record = conductor
        .call(&cell.zome("did_registry"), "create_did", ())
        .await;

    // Try to add a verification method with no 'z' prefix (invalid multibase)
    let update = UpdateDidInput {
        verification_method: Some(vec![VerificationMethod {
            id: "did:mycelix:test#key-2".to_string(),
            type_: "Ed25519VerificationKey2020".to_string(),
            controller: "did:mycelix:test".to_string(),
            public_key_multibase: "ABCDEF1234567890ABCDEF1234567890ABCDEF12".to_string(),
        }]),
        authentication: None,
        key_agreement: None,
        service: None,
    };

    let result: Result<Record, _> = conductor
        .call_fallible(&cell.zome("did_registry"), "update_did_document", update)
        .await;

    assert!(result.is_err(), "Should reject multibase key without 'z' prefix");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
async fn test_reject_multibase_key_with_invalid_base58_chars() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor.setup_app("test", &[dna]).await.unwrap();
    let cell = app.cells()[0].clone();

    let _record: Record = conductor
        .call(&cell.zome("did_registry"), "create_did", ())
        .await;

    // 'z' prefix but contains '0' and 'O' which are not in base58btc Bitcoin alphabet
    let update = UpdateDidInput {
        verification_method: Some(vec![VerificationMethod {
            id: "did:mycelix:test#key-2".to_string(),
            type_: "Ed25519VerificationKey2020".to_string(),
            controller: "did:mycelix:test".to_string(),
            public_key_multibase: "z0OIlInvalidBase58Characters!!".to_string(),
        }]),
        authentication: None,
        key_agreement: None,
        service: None,
    };

    let result: Result<Record, _> = conductor
        .call_fallible(&cell.zome("did_registry"), "update_did_document", update)
        .await;

    assert!(result.is_err(), "Should reject multibase key with invalid base58btc characters");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
async fn test_reject_multibase_key_too_short() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor.setup_app("test", &[dna]).await.unwrap();
    let cell = app.cells()[0].clone();

    let _record: Record = conductor
        .call(&cell.zome("did_registry"), "create_did", ())
        .await;

    // Valid prefix and chars, but way too short for an Ed25519 key
    let update = UpdateDidInput {
        verification_method: Some(vec![VerificationMethod {
            id: "did:mycelix:test#key-2".to_string(),
            type_: "Ed25519VerificationKey2020".to_string(),
            controller: "did:mycelix:test".to_string(),
            public_key_multibase: "zABC".to_string(),
        }]),
        authentication: None,
        key_agreement: None,
        service: None,
    };

    let result: Result<Record, _> = conductor
        .call_fallible(&cell.zome("did_registry"), "update_did_document", update)
        .await;

    assert!(result.is_err(), "Should reject multibase key that is too short for Ed25519");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
async fn test_accept_valid_non_ed25519_key_type() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor.setup_app("test", &[dna]).await.unwrap();
    let cell = app.cells()[0].clone();

    let _record: Record = conductor
        .call(&cell.zome("did_registry"), "create_did", ())
        .await;

    // Non-Ed25519 key types should not be validated by multibase Ed25519 rules
    let update = UpdateDidInput {
        verification_method: Some(vec![VerificationMethod {
            id: "did:mycelix:test#key-2".to_string(),
            type_: "X25519KeyAgreementKey2020".to_string(),
            controller: "did:mycelix:test".to_string(),
            public_key_multibase: "zSomeOpaqueKeyMaterial12345678901234567890".to_string(),
        }]),
        authentication: None,
        key_agreement: None,
        service: None,
    };

    // Should succeed - X25519 keys are not validated by Ed25519 multibase rules
    let result: Result<Record, _> = conductor
        .call_fallible(&cell.zome("did_registry"), "update_did_document", update)
        .await;

    assert!(result.is_ok(), "Non-Ed25519 key types should bypass multibase Ed25519 validation");
}

// ============================================================================
// Cross-Zome Bridge Notification Tests
// ============================================================================

/// Mirror of bridge coordinator::DidDeactivatedInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DidDeactivatedInput {
    pub did: String,
    pub reason: String,
    pub deactivated_at: String,
}

/// Mirror of bridge coordinator::MfaAssuranceChangedInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MfaAssuranceChangedInput {
    pub did: String,
    pub old_level: String,
    pub new_level: String,
    pub new_score: f64,
}

/// Mirror of bridge coordinator::GetEventsInput
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GetEventsInput {
    pub event_type: Option<String>,
    pub since: Option<u64>,
    pub limit: Option<u32>,
}

/// Mirror of bridge coordinator::BridgeEvent
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BridgeEvent {
    pub id: String,
    pub event_type: String,
    pub subject: String,
    pub payload: String,
    pub source_happ: String,
    pub timestamp: Timestamp,
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
async fn test_notify_did_deactivated_creates_event() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor.setup_app("test", &[dna]).await.unwrap();
    let cell = app.cells()[0].clone();

    // First create a DID so the bridge zome has context
    let did_record: Record = conductor
        .call(&cell.zome("did_registry"), "create_did", ())
        .await;
    let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

    // Call notify_did_deactivated on the bridge
    let input = DidDeactivatedInput {
        did: did_doc.id.clone(),
        reason: "Key compromised".to_string(),
        deactivated_at: "2026-02-12T00:00:00Z".to_string(),
    };

    let event_record: Record = conductor
        .call(&cell.zome("identity_bridge"), "notify_did_deactivated", input)
        .await;

    let event: BridgeEvent = decode_entry(&event_record).expect("Failed to decode event");
    assert!(event.id.starts_with("event:"), "Event ID should start with 'event:'");
    assert_eq!(event.subject, did_doc.id, "Event subject should be the DID");
    assert!(event.payload.contains("Key compromised"), "Payload should contain reason");

    // Verify event is retrievable via get_recent_events
    let query = GetEventsInput {
        event_type: None,
        since: None,
        limit: Some(10),
    };

    let events: Vec<Record> = conductor
        .call(&cell.zome("identity_bridge"), "get_recent_events", query)
        .await;

    assert!(!events.is_empty(), "Should have at least one recent event");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
async fn test_notify_mfa_assurance_changed_creates_event() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor.setup_app("test", &[dna]).await.unwrap();
    let cell = app.cells()[0].clone();

    // Create a DID first
    let did_record: Record = conductor
        .call(&cell.zome("did_registry"), "create_did", ())
        .await;
    let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

    // Notify MFA assurance change
    let input = MfaAssuranceChangedInput {
        did: did_doc.id.clone(),
        old_level: "Basic".to_string(),
        new_level: "Verified".to_string(),
        new_score: 0.65,
    };

    let event_record: Record = conductor
        .call(&cell.zome("identity_bridge"), "notify_mfa_assurance_changed", input)
        .await;

    let event: BridgeEvent = decode_entry(&event_record).expect("Failed to decode event");
    assert_eq!(event.subject, did_doc.id, "Event subject should be the DID");
    assert!(event.payload.contains("Verified"), "Payload should contain new level");
    assert!(event.payload.contains("0.65"), "Payload should contain new score");
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
async fn test_multiple_notifications_retrievable() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna = load_dna().await;
    let app = conductor.setup_app("test", &[dna]).await.unwrap();
    let cell = app.cells()[0].clone();

    // Create a DID
    let did_record: Record = conductor
        .call(&cell.zome("did_registry"), "create_did", ())
        .await;
    let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

    // Send DID deactivation notification
    let deactivation = DidDeactivatedInput {
        did: did_doc.id.clone(),
        reason: "Rotation".to_string(),
        deactivated_at: "2026-02-12T12:00:00Z".to_string(),
    };
    let _: Record = conductor
        .call(&cell.zome("identity_bridge"), "notify_did_deactivated", deactivation)
        .await;

    // Send MFA assurance change
    let mfa_change = MfaAssuranceChangedInput {
        did: did_doc.id.clone(),
        old_level: "Anonymous".to_string(),
        new_level: "Basic".to_string(),
        new_score: 0.25,
    };
    let _: Record = conductor
        .call(&cell.zome("identity_bridge"), "notify_mfa_assurance_changed", mfa_change)
        .await;

    // Query all recent events
    let query = GetEventsInput {
        event_type: None,
        since: None,
        limit: Some(50),
    };

    let events: Vec<Record> = conductor
        .call(&cell.zome("identity_bridge"), "get_recent_events", query)
        .await;

    assert!(events.len() >= 2, "Should have at least 2 events, got {}", events.len());
}

// ============================================================================
// Verifiable Credential Lifecycle Tests
// ============================================================================

#[cfg(test)]
mod vc_lifecycle_tests {
    use super::*;

    /// Full VC lifecycle: issue → verify (valid) → revoke → verify (invalid)
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
    async fn test_vc_issue_verify_revoke_verify() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-vc-lifecycle", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // 1. Create issuer DID
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");
        let issuer_did = did_doc.id.clone();

        // 2. Create a credential schema
        let now = Timestamp::now();
        let schema = CredentialSchema {
            id: "mycelix:schema:vc-lifecycle-test:v1".to_string(),
            name: "Lifecycle Test Credential".to_string(),
            description: "Schema for VC lifecycle integration test".to_string(),
            version: "1.0.0".to_string(),
            author: issuer_did.clone(),
            schema: r#"{"type":"object","properties":{"role":{"type":"string"}}}"#.to_string(),
            required_fields: vec!["role".to_string()],
            optional_fields: vec![],
            credential_type: vec![
                "VerifiableCredential".to_string(),
                "TestCredential".to_string(),
            ],
            default_expiration: 86400 * 365,
            revocable: true,
            active: true,
            created: now,
            updated: now,
        };

        let _: Record = conductor
            .call(&cell.zome("credential_schema"), "create_schema", schema)
            .await;

        // 3. Issue a credential
        let issue_input = IssueCredentialInput {
            subject_did: format!("did:mycelix:subject-{}", now.as_micros()),
            schema_id: "mycelix:schema:vc-lifecycle-test:v1".to_string(),
            claims: serde_json::json!({
                "role": "developer",
            }),
            credential_types: vec!["TestCredential".to_string()],
            issuer_name: Some("Lifecycle Test Issuer".to_string()),
            expiration_days: Some(365),
            enable_revocation: true,
        };

        let vc_record: Record = conductor
            .call(
                &cell.zome("verifiable_credential"),
                "issue_credential",
                issue_input,
            )
            .await;

        let vc: VerifiableCredential =
            decode_entry(&vc_record).expect("Failed to decode issued credential");
        let credential_id = vc.id.clone();
        assert!(
            vc.credential_type.contains(&"VerifiableCredential".to_string()),
            "Must include VerifiableCredential type"
        );

        // 4. Verify — should be valid
        let result: VcVerificationResult = conductor
            .call(
                &cell.zome("verifiable_credential"),
                "verify_credential",
                credential_id.clone(),
            )
            .await;

        assert!(
            result.valid,
            "Freshly issued credential should verify as valid, but got errors: {:?}",
            result.errors
        );

        // 5. Revoke the credential
        let revoke_input = RevokeInput {
            credential_id: credential_id.clone(),
            issuer_did: issuer_did.clone(),
            reason: "Lifecycle test revocation".to_string(),
            effective_from: None,
        };

        let _: Record = conductor
            .call(&cell.zome("revocation"), "revoke_credential", revoke_input)
            .await;

        // 6. Verify again — should now fail with revocation error
        let result_after: VcVerificationResult = conductor
            .call(
                &cell.zome("verifiable_credential"),
                "verify_credential",
                credential_id.clone(),
            )
            .await;

        assert!(
            !result_after.valid,
            "Credential should be invalid after revocation"
        );
        assert!(
            result_after.errors.iter().any(|e| e.to_lowercase().contains("revok")),
            "Errors should mention revocation, got: {:?}",
            result_after.errors
        );
    }
}

// ============================================================================
// Trust Credential Attestation Tests
// ============================================================================

#[cfg(test)]
mod trust_attestation_tests {
    use super::*;

    /// Issue a trust credential and verify it on-chain
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
    async fn test_trust_credential_issue_and_verify() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-trust", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Create two DIDs (issuer and subject)
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");
        let agent_did = did_doc.id.clone();

        // Issue a trust credential with a valid commitment and proof
        let commitment = vec![42u8; 32]; // 32-byte commitment
        let range_proof = vec![1u8, 2, 3, 4, 5]; // non-empty proof

        let issue_input = IssueTrustCredentialInput {
            subject_did: agent_did.clone(),
            issuer_did: agent_did.clone(), // self-attestation
            kvector_commitment: commitment,
            range_proof: range_proof,
            trust_score_lower: 0.6,
            trust_score_upper: 0.75,
            expires_at: None,
            supersedes: None,
        };

        let cred_record: Record = conductor
            .call(
                &cell.zome("trust_credential"),
                "issue_trust_credential",
                issue_input,
            )
            .await;

        let cred: TrustCredential =
            decode_entry(&cred_record).expect("Failed to decode trust credential");

        assert_eq!(cred.subject_did, agent_did, "Subject should match");
        assert_eq!(cred.trust_tier, TrustTier::Elevated, "Mid-score 0.675 => Elevated");
        assert!(!cred.revoked, "New credential should not be revoked");

        // Verify the credential on-chain
        let verify_result: TrustVerificationResult = conductor
            .call(
                &cell.zome("trust_credential"),
                "verify_credential",
                cred.id.clone(),
            )
            .await;

        assert!(
            verify_result.commitment_valid,
            "Commitment should be valid"
        );
        assert!(
            verify_result.tier_consistent,
            "Tier should be consistent with score range"
        );
        assert!(verify_result.not_revoked, "Should not be revoked");
        assert!(verify_result.not_expired, "Should not be expired");
        assert!(
            verify_result.proof_format_valid,
            "Proof format should be valid"
        );
    }

    /// Request attestation → fulfill → verify the resulting credential
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
    async fn test_attestation_request_fulfill_verify() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-attestation", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Create requester DID
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");
        let requester_did = did_doc.id.clone();

        // The subject DID (in a real scenario, a different agent)
        let now = Timestamp::now();
        let subject_did = format!("did:mycelix:subject-attestee-{}", now.as_micros());

        // Set expiration 1 hour from now
        let one_hour = Timestamp::from_micros(now.as_micros() + 3_600_000_000);

        // 1. Request attestation
        let request_input = RequestAttestationInput {
            requester_did: requester_did.clone(),
            subject_did: subject_did.clone(),
            components: vec![KVectorComponent::Reputation, KVectorComponent::Integrity],
            min_trust_score: Some(0.4),
            min_tier: None,
            purpose: "Integration test attestation".to_string(),
            expires_at: one_hour,
        };

        let request_record: Record = conductor
            .call(
                &cell.zome("trust_credential"),
                "request_attestation",
                request_input,
            )
            .await;

        let request: AttestationRequest =
            decode_entry(&request_record).expect("Failed to decode attestation request");

        assert_eq!(request.status, AttestationStatus::Pending, "Request should be Pending");
        assert_eq!(request.requester_did, requester_did);
        assert_eq!(request.subject_did, subject_did);

        // 2. Check pending requests for subject
        let pending: Vec<Record> = conductor
            .call(
                &cell.zome("trust_credential"),
                "get_pending_requests",
                subject_did.clone(),
            )
            .await;

        assert!(
            !pending.is_empty(),
            "Subject should have at least one pending request"
        );

        // 3. Fulfill the attestation
        let fulfill_input = FulfillAttestationInput {
            request_id: request.id.clone(),
            subject_did: subject_did.clone(),
            kvector_commitment: vec![99u8; 32], // 32-byte commitment
            range_proof: vec![10, 20, 30],       // non-empty proof
            trust_score_lower: 0.5,
            trust_score_upper: 0.7,
            expires_at: None,
        };

        let fulfill_result: FulfillAttestationResult = conductor
            .call(
                &cell.zome("trust_credential"),
                "fulfill_attestation",
                fulfill_input,
            )
            .await;

        assert_eq!(
            fulfill_result.request_id, request.id,
            "Fulfilled request ID should match"
        );

        // 4. Verify the resulting trust credential
        let cred: TrustCredential =
            decode_entry(&fulfill_result.credential_record)
                .expect("Failed to decode fulfilled credential");

        assert_eq!(cred.subject_did, subject_did);
        assert_eq!(cred.trust_tier, TrustTier::Elevated, "Mid-score 0.6 => Elevated");
        assert!(!cred.revoked, "Fulfilled credential should not be revoked");

        // 5. Verify on-chain
        let verify_result: TrustVerificationResult = conductor
            .call(
                &cell.zome("trust_credential"),
                "verify_credential",
                cred.id.clone(),
            )
            .await;

        assert!(
            verify_result.commitment_valid && verify_result.tier_consistent
                && verify_result.not_revoked && verify_result.not_expired
                && verify_result.proof_format_valid,
            "All verification checks should pass: {}",
            verify_result.message
        );

        // 6. Verify the request status was updated to Fulfilled
        // (get_pending_requests should no longer return it)
        let pending_after: Vec<Record> = conductor
            .call(
                &cell.zome("trust_credential"),
                "get_pending_requests",
                subject_did.clone(),
            )
            .await;

        // The fulfilled request should no longer be in the pending list
        let still_pending: Vec<_> = pending_after
            .iter()
            .filter_map(|r| decode_entry::<AttestationRequest>(r))
            .filter(|req| req.id == request.id)
            .collect();
        assert!(
            still_pending.is_empty(),
            "Fulfilled request should no longer be pending"
        );
    }
}

// ============================================================================
// Social Recovery E2E Tests
// ============================================================================

#[cfg(test)]
mod social_recovery_tests {
    use super::*;

    /// Full recovery lifecycle: setup → initiate → vote (reach threshold) → execute
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
    async fn test_recovery_setup_initiate_vote_execute() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-recovery", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // 1. Create owner DID
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");
        let owner_did = did_doc.id.clone();

        // 2. Setup recovery with 3 trustees, threshold 2
        let trustee1 = "did:mycelix:trustee-alice".to_string();
        let trustee2 = "did:mycelix:trustee-bob".to_string();
        let trustee3 = "did:mycelix:trustee-carol".to_string();

        let setup_input = SetupRecoveryInput {
            did: owner_did.clone(),
            trustees: vec![trustee1.clone(), trustee2.clone(), trustee3.clone()],
            threshold: 2,
            time_lock: Some(86400), // 1 day (minimum)
        };

        let config_record: Record = conductor
            .call(&cell.zome("recovery"), "setup_recovery", setup_input)
            .await;

        let config: RecoveryConfig =
            decode_entry(&config_record).expect("Failed to decode RecoveryConfig");

        assert_eq!(config.did, owner_did, "Config DID should match");
        assert_eq!(config.trustees.len(), 3, "Should have 3 trustees");
        assert_eq!(config.threshold, 2, "Threshold should be 2");
        assert!(config.active, "Config should be active");

        // 3. Verify config retrieval
        let retrieved: Option<Record> = conductor
            .call(&cell.zome("recovery"), "get_recovery_config", owner_did.clone())
            .await;
        assert!(retrieved.is_some(), "Recovery config should be retrievable");

        // 4. Initiate recovery (trustee1 starts it)
        let new_agent = app.agent().clone(); // reuse agent for test simplicity
        let initiate_input = InitiateRecoveryInput {
            did: owner_did.clone(),
            initiator_did: trustee1.clone(),
            new_agent: new_agent.clone(),
            reason: "Owner lost access to device".to_string(),
        };

        let request_record: Record = conductor
            .call(&cell.zome("recovery"), "initiate_recovery", initiate_input)
            .await;

        let request: RecoveryRequest =
            decode_entry(&request_record).expect("Failed to decode RecoveryRequest");

        assert_eq!(request.did, owner_did, "Request DID should match");
        assert_eq!(request.initiated_by, trustee1, "Initiator should match");
        assert!(
            request.status == RecoveryStatus::Pending || request.status == RecoveryStatus::Approved,
            "Status should be Pending or Approved (initiator auto-votes), got: {:?}",
            request.status
        );

        // 5. Second trustee votes to approve (should reach threshold of 2)
        let vote_input = VoteOnRecoveryInput {
            request_id: request.id.clone(),
            trustee_did: trustee2.clone(),
            vote: VoteDecision::Approve,
            comment: Some("I confirm identity via phone call".to_string()),
        };

        let _vote_record: Record = conductor
            .call(&cell.zome("recovery"), "vote_on_recovery", vote_input)
            .await;

        // 6. Check votes
        let votes: Vec<Record> = conductor
            .call(
                &cell.zome("recovery"),
                "get_recovery_votes",
                request.id.clone(),
            )
            .await;

        assert!(
            votes.len() >= 2,
            "Should have at least 2 votes (initiator auto-vote + trustee2), got {}",
            votes.len()
        );

        // 7. Get updated request - should be Approved (threshold reached)
        let updated_request: Option<Record> = conductor
            .call(
                &cell.zome("recovery"),
                "get_recovery_request",
                request.id.clone(),
            )
            .await;

        if let Some(req_record) = updated_request {
            let req: RecoveryRequest =
                decode_entry(&req_record).expect("Failed to decode updated request");
            assert!(
                req.status == RecoveryStatus::Approved
                    || req.status == RecoveryStatus::ReadyToExecute
                    || req.status == RecoveryStatus::Completed,
                "Request should be at least Approved after threshold, got: {:?}",
                req.status
            );
        }
    }

    /// Setup recovery and verify trustee responsibilities query
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
    async fn test_trustee_responsibilities_query() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-trustee-query", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Create DID
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

        let trustee_did = "did:mycelix:trustee-dan".to_string();

        // Setup recovery with the trustee
        let setup_input = SetupRecoveryInput {
            did: did_doc.id.clone(),
            trustees: vec![
                trustee_did.clone(),
                "did:mycelix:trustee-eve".to_string(),
                "did:mycelix:trustee-frank".to_string(),
            ],
            threshold: 2,
            time_lock: Some(86400),
        };

        let _: Record = conductor
            .call(&cell.zome("recovery"), "setup_recovery", setup_input)
            .await;

        // Query trustee responsibilities
        let responsibilities: Vec<Record> = conductor
            .call(
                &cell.zome("recovery"),
                "get_trustee_responsibilities",
                trustee_did.clone(),
            )
            .await;

        assert!(
            !responsibilities.is_empty(),
            "Trustee should have at least one responsibility"
        );
    }

    /// Owner cancels a pending recovery request before execution
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
    async fn test_cancel_recovery_by_owner() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("test-cancel-recovery", &[dna])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();

        // Create owner DID
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("decode DID");
        let owner_did = did_doc.id.clone();

        // Setup recovery with 3 trustees
        let setup_input = SetupRecoveryInput {
            did: owner_did.clone(),
            trustees: vec![
                "did:mycelix:trustee-cancel-a".to_string(),
                "did:mycelix:trustee-cancel-b".to_string(),
                "did:mycelix:trustee-cancel-c".to_string(),
            ],
            threshold: 2,
            time_lock: Some(86400),
        };

        let _: Record = conductor
            .call(&cell.zome("recovery"), "setup_recovery", setup_input)
            .await;

        // Trustee initiates recovery
        let initiate_input = InitiateRecoveryInput {
            did: owner_did.clone(),
            initiator_did: "did:mycelix:trustee-cancel-a".to_string(),
            new_agent: app.agent().clone(),
            reason: "Testing cancellation flow".to_string(),
        };

        let request_record: Record = conductor
            .call(
                &cell.zome("recovery"),
                "initiate_recovery",
                initiate_input,
            )
            .await;
        let request: RecoveryRequest =
            decode_entry(&request_record).expect("decode RecoveryRequest");

        // Owner cancels the recovery
        let cancelled_record: Record = conductor
            .call(
                &cell.zome("recovery"),
                "cancel_recovery",
                request.id.clone(),
            )
            .await;
        let cancelled: RecoveryRequest =
            decode_entry(&cancelled_record).expect("decode cancelled request");

        assert_eq!(
            cancelled.status,
            RecoveryStatus::Cancelled,
            "Recovery should be Cancelled after owner cancels, got: {:?}",
            cancelled.status
        );

        // Verify retrieval returns cancelled status
        let retrieved: Option<Record> = conductor
            .call(
                &cell.zome("recovery"),
                "get_recovery_request",
                request.id.clone(),
            )
            .await;
        assert!(
            retrieved.is_some(),
            "Cancelled request should still be retrievable"
        );
    }

    /// Retrieve a specific recovery request by ID
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
    async fn test_get_recovery_request_by_id() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("test-get-request", &[dna])
            .await
            .unwrap();
        let cell = app.cells()[0].clone();

        // Create DID
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("decode DID");

        // Setup recovery
        let setup_input = SetupRecoveryInput {
            did: did_doc.id.clone(),
            trustees: vec![
                "did:mycelix:trustee-get-a".to_string(),
                "did:mycelix:trustee-get-b".to_string(),
                "did:mycelix:trustee-get-c".to_string(),
            ],
            threshold: 2,
            time_lock: Some(86400),
        };

        let _: Record = conductor
            .call(&cell.zome("recovery"), "setup_recovery", setup_input)
            .await;

        // Initiate recovery to create a request
        let initiate_input = InitiateRecoveryInput {
            did: did_doc.id.clone(),
            initiator_did: "did:mycelix:trustee-get-a".to_string(),
            new_agent: app.agent().clone(),
            reason: "Testing request retrieval".to_string(),
        };

        let request_record: Record = conductor
            .call(
                &cell.zome("recovery"),
                "initiate_recovery",
                initiate_input,
            )
            .await;
        let request: RecoveryRequest =
            decode_entry(&request_record).expect("decode RecoveryRequest");

        // Retrieve by ID
        let retrieved: Option<Record> = conductor
            .call(
                &cell.zome("recovery"),
                "get_recovery_request",
                request.id.clone(),
            )
            .await;

        assert!(retrieved.is_some(), "Should retrieve request by ID");
        let retrieved_request: RecoveryRequest =
            decode_entry(&retrieved.unwrap()).expect("decode retrieved request");

        assert_eq!(
            retrieved_request.id, request.id,
            "Retrieved request ID should match"
        );
        assert_eq!(
            retrieved_request.did, did_doc.id,
            "Retrieved request DID should match"
        );
        assert_eq!(
            retrieved_request.reason, "Testing request retrieval",
            "Retrieved request reason should match"
        );

        // Verify non-existent ID returns None
        let missing: Option<Record> = conductor
            .call(
                &cell.zome("recovery"),
                "get_recovery_request",
                "non-existent-request-id-12345".to_string(),
            )
            .await;

        assert!(
            missing.is_none(),
            "Non-existent request ID should return None"
        );
    }
}

// ============================================================================
// Key Rotation E2E Tests
// ============================================================================

#[cfg(test)]
mod key_rotation_tests {
    use super::*;

    /// Create DID → update verification method → resolve → verify new state
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
    async fn test_did_key_rotation_via_update() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-key-rotation", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // 1. Create DID
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");
        let did_id = did_doc.id.clone();

        assert_eq!(did_doc.version, 1, "Initial version should be 1");
        let initial_vm_count = did_doc.verification_method.len();
        assert!(initial_vm_count > 0, "Should have initial verification method");

        let initial_key = did_doc.verification_method[0].public_key_multibase.clone();
        let initial_auth = did_doc.authentication.clone();

        // 2. Update DID with a new verification method (simulate key rotation via update_did_document)
        let new_method = VerificationMethod {
            id: "#keys-2".to_string(),
            type_: "Multikey".to_string(),
            controller: did_id.clone(),
            public_key_multibase: "z6Mktest123rotatedkey456".to_string(),
        };

        let mut updated_methods = did_doc.verification_method.clone();
        // Deprecate old key
        updated_methods[0].id = format!("{}-deprecated-v1", updated_methods[0].id);
        updated_methods.push(new_method.clone());

        let update_input = UpdateDidInput {
            verification_method: Some(updated_methods.clone()),
            authentication: Some(vec!["#keys-2".to_string()]),
            key_agreement: None,
            service: None,
        };

        let updated_record: Record = conductor
            .call(&cell.zome("did_registry"), "update_did_document", update_input)
            .await;

        let updated_doc: DidDocument =
            decode_entry(&updated_record).expect("Failed to decode updated DID");

        // 3. Verify the updated DID
        assert_eq!(updated_doc.version, 2, "Version should increment to 2");
        assert_eq!(
            updated_doc.verification_method.len(),
            initial_vm_count + 1,
            "Should have old (deprecated) + new method"
        );
        assert!(
            updated_doc.authentication.contains(&"#keys-2".to_string()),
            "Authentication should reference new key"
        );
        assert!(
            !updated_doc.authentication.contains(&initial_auth[0]),
            "Authentication should no longer reference old key"
        );

        // 4. Resolve the DID and verify it returns the updated state
        let resolved: Option<Record> = conductor
            .call(&cell.zome("did_registry"), "resolve_did", did_id.clone())
            .await;

        let resolved_doc: DidDocument =
            decode_entry(&resolved.expect("DID resolution should succeed"))
                .expect("Failed to decode resolved DID");

        assert_eq!(resolved_doc.version, 2, "Resolved version should be 2");

        // Verify deprecated key is still present (for old signature verification)
        let deprecated_keys: Vec<_> = resolved_doc
            .verification_method
            .iter()
            .filter(|m| m.id.contains("deprecated"))
            .collect();
        assert_eq!(deprecated_keys.len(), 1, "Should have exactly one deprecated key");
        assert_eq!(
            deprecated_keys[0].public_key_multibase, initial_key,
            "Deprecated key should retain original public key material"
        );

        // New key should be present
        let new_keys: Vec<_> = resolved_doc
            .verification_method
            .iter()
            .filter(|m| m.id == "#keys-2")
            .collect();
        assert_eq!(new_keys.len(), 1, "Should have new key");
    }

    /// Add a service endpoint, then update DID to remove it
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
    async fn test_did_service_update_lifecycle() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-service-update", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Create DID
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let _did_doc: DidDocument = decode_entry(&did_record).expect("Failed to decode DID");

        // Add service endpoint via update
        let svc = ServiceEndpoint {
            id: "#messaging".to_string(),
            type_: "MessagingService".to_string(),
            service_endpoint: "https://example.com/msg".to_string(),
        };

        let update_input = UpdateDidInput {
            verification_method: None, // preserve
            authentication: None,       // preserve
            key_agreement: None,        // preserve
            service: Some(vec![svc.clone()]),
        };

        let updated: Record = conductor
            .call(&cell.zome("did_registry"), "update_did_document", update_input)
            .await;

        let updated_doc: DidDocument =
            decode_entry(&updated).expect("Failed to decode updated DID");

        assert_eq!(updated_doc.service.len(), 1, "Should have one service");
        assert_eq!(updated_doc.service[0].id, "#messaging");
        assert_eq!(updated_doc.version, 2);

        // Remove the service by updating with empty array
        let remove_input = UpdateDidInput {
            verification_method: None,
            authentication: None,
            key_agreement: None,
            service: Some(vec![]),
        };

        let removed: Record = conductor
            .call(&cell.zome("did_registry"), "update_did_document", remove_input)
            .await;

        let removed_doc: DidDocument =
            decode_entry(&removed).expect("Failed to decode removed service DID");

        assert!(removed_doc.service.is_empty(), "Service list should be empty after removal");
        assert_eq!(removed_doc.version, 3, "Version should be 3 after two updates");
    }
}

// ============================================================================
// Revocation Lifecycle Tests
// ============================================================================

#[cfg(test)]
mod revocation_tests {
    use super::*;

    /// Helper: create DID + schema + issue a VC, return (issuer_did, credential_id)
    async fn setup_and_issue_vc(
        conductor: &mut SweetConductor,
        cell: &SweetCell,
        suffix: &str,
    ) -> (String, String) {
        // Create DID
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("decode DID");
        let issuer_did = did_doc.id.clone();

        // Create schema
        let now = Timestamp::now();
        let schema = CredentialSchema {
            id: format!("mycelix:schema:revocation-test-{}:v1", suffix),
            name: "Revocation Test Schema".to_string(),
            description: "Schema for revocation tests".to_string(),
            version: "1.0.0".to_string(),
            author: issuer_did.clone(),
            schema: r#"{"type":"object","properties":{"role":{"type":"string"}}}"#.to_string(),
            required_fields: vec!["role".to_string()],
            optional_fields: vec![],
            credential_type: vec![
                "VerifiableCredential".to_string(),
                "TestCredential".to_string(),
            ],
            default_expiration: 86400 * 365,
            revocable: true,
            active: true,
            created: now,
            updated: now,
        };
        let _: Record = conductor
            .call(&cell.zome("credential_schema"), "create_schema", schema.clone())
            .await;

        // Issue credential
        let issue_input = IssueCredentialInput {
            subject_did: format!("did:mycelix:subject-{}-{}", suffix, now.as_micros()),
            schema_id: schema.id.clone(),
            claims: serde_json::json!({"role": "tester"}),
            credential_types: vec!["TestCredential".to_string()],
            issuer_name: None,
            expiration_days: Some(365),
            enable_revocation: true,
        };

        let vc_record: Record = conductor
            .call(&cell.zome("verifiable_credential"), "issue_credential", issue_input)
            .await;

        let vc: VerifiableCredential = decode_entry(&vc_record).expect("decode VC");
        (issuer_did, vc.id)
    }

    /// Revoke a credential and verify status transitions to Revoked
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
    async fn test_revoke_and_check_status() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-revoke-status", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let (issuer_did, credential_id) =
            setup_and_issue_vc(&mut conductor, &cell, "revoke-check").await;

        // Check initial status — should be Active
        let initial: RevocationCheckResult = conductor
            .call(
                &cell.zome("revocation"),
                "check_revocation_status",
                credential_id.clone(),
            )
            .await;

        assert_eq!(initial.status, RevocationStatus::Active, "Should start Active");

        // Revoke
        let revoke = RevokeInput {
            credential_id: credential_id.clone(),
            issuer_did: issuer_did.clone(),
            reason: "Credential holder left organization".to_string(),
            effective_from: None,
        };

        let _: Record = conductor
            .call(&cell.zome("revocation"), "revoke_credential", revoke)
            .await;

        // Check status after revocation
        let after: RevocationCheckResult = conductor
            .call(
                &cell.zome("revocation"),
                "check_revocation_status",
                credential_id.clone(),
            )
            .await;

        assert_eq!(after.status, RevocationStatus::Revoked, "Should be Revoked");
        assert!(
            after.reason.as_deref().unwrap_or("").contains("left organization"),
            "Reason should be preserved"
        );

        // Also check via VC zome's credential status endpoint
        let vc_status: CredentialStatusResponse = conductor
            .call(
                &cell.zome("verifiable_credential"),
                "get_credential_status",
                credential_id.clone(),
            )
            .await;

        assert!(!vc_status.is_valid, "VC should report as invalid after revocation");
        assert!(
            vc_status.status_type.to_lowercase().contains("revok"),
            "Status type should mention revocation, got: {}",
            vc_status.status_type
        );
    }

    /// Suspend → check (Suspended) → reinstate → check (Active)
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
    async fn test_suspend_and_reinstate() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-suspend-reinstate", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        let (issuer_did, credential_id) =
            setup_and_issue_vc(&mut conductor, &cell, "suspend-reinstate").await;

        // Suspend for 1 week from now
        let now = Timestamp::now();
        let one_week = Timestamp::from_micros(now.as_micros() + 7 * 86400 * 1_000_000);

        let suspend = SuspendInput {
            credential_id: credential_id.clone(),
            issuer_did: issuer_did.clone(),
            reason: "Under investigation".to_string(),
            suspension_end: one_week,
        };

        let _: Record = conductor
            .call(&cell.zome("revocation"), "suspend_credential", suspend)
            .await;

        // Check — should be Suspended
        let after_suspend: RevocationCheckResult = conductor
            .call(
                &cell.zome("revocation"),
                "check_revocation_status",
                credential_id.clone(),
            )
            .await;

        assert_eq!(
            after_suspend.status,
            RevocationStatus::Suspended,
            "Should be Suspended"
        );

        // Reinstate
        let reinstate = ReinstateInput {
            credential_id: credential_id.clone(),
            issuer_did: issuer_did.clone(),
            reason: "Investigation cleared".to_string(),
        };

        let _: Record = conductor
            .call(&cell.zome("revocation"), "reinstate_credential", reinstate)
            .await;

        // Check — should be Active again
        let after_reinstate: RevocationCheckResult = conductor
            .call(
                &cell.zome("revocation"),
                "check_revocation_status",
                credential_id.clone(),
            )
            .await;

        assert_eq!(
            after_reinstate.status,
            RevocationStatus::Active,
            "Should be Active after reinstatement"
        );
    }

    /// Batch revoke multiple credentials at once
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
    async fn test_batch_revoke_credentials() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-batch-revoke", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Issue 3 credentials (same issuer, same schema)
        let (issuer_did, cred_id_1) =
            setup_and_issue_vc(&mut conductor, &cell, "batch-1").await;

        // Issue 2 more using the same issuer DID (since DID already exists)
        let now = Timestamp::now();

        // Issue cred 2
        let schema2 = CredentialSchema {
            id: "mycelix:schema:batch-2:v1".to_string(),
            name: "Batch Test 2".to_string(),
            description: "Schema for batch revocation test 2".to_string(),
            version: "1.0.0".to_string(),
            author: issuer_did.clone(),
            schema: r#"{"type":"object","properties":{"role":{"type":"string"}}}"#.to_string(),
            required_fields: vec!["role".to_string()],
            optional_fields: vec![],
            credential_type: vec!["VerifiableCredential".to_string()],
            default_expiration: 86400 * 365,
            revocable: true,
            active: true,
            created: now,
            updated: now,
        };
        let _: Record = conductor
            .call(&cell.zome("credential_schema"), "create_schema", schema2.clone())
            .await;

        let issue2 = IssueCredentialInput {
            subject_did: format!("did:mycelix:batch-subject-2-{}", now.as_micros()),
            schema_id: schema2.id.clone(),
            claims: serde_json::json!({"role": "reviewer"}),
            credential_types: vec![],
            issuer_name: None,
            expiration_days: Some(365),
            enable_revocation: true,
        };
        let vc2_record: Record = conductor
            .call(&cell.zome("verifiable_credential"), "issue_credential", issue2)
            .await;
        let vc2: VerifiableCredential = decode_entry(&vc2_record).expect("decode VC2");

        // Issue cred 3
        let schema3 = CredentialSchema {
            id: "mycelix:schema:batch-3:v1".to_string(),
            name: "Batch Test 3".to_string(),
            description: "Schema for batch revocation test 3".to_string(),
            version: "1.0.0".to_string(),
            author: issuer_did.clone(),
            schema: r#"{"type":"object","properties":{"role":{"type":"string"}}}"#.to_string(),
            required_fields: vec!["role".to_string()],
            optional_fields: vec![],
            credential_type: vec!["VerifiableCredential".to_string()],
            default_expiration: 86400 * 365,
            revocable: true,
            active: true,
            created: now,
            updated: now,
        };
        let _: Record = conductor
            .call(&cell.zome("credential_schema"), "create_schema", schema3.clone())
            .await;

        let issue3 = IssueCredentialInput {
            subject_did: format!("did:mycelix:batch-subject-3-{}", now.as_micros()),
            schema_id: schema3.id.clone(),
            claims: serde_json::json!({"role": "admin"}),
            credential_types: vec![],
            issuer_name: None,
            expiration_days: Some(365),
            enable_revocation: true,
        };
        let vc3_record: Record = conductor
            .call(&cell.zome("verifiable_credential"), "issue_credential", issue3)
            .await;
        let vc3: VerifiableCredential = decode_entry(&vc3_record).expect("decode VC3");

        // Batch revoke all 3
        let batch_input = BatchRevokeInput {
            credential_ids: vec![cred_id_1.clone(), vc2.id.clone(), vc3.id.clone()],
            issuer_did: issuer_did.clone(),
            reason: "Organization restructuring".to_string(),
            effective_from: None,
            revocation_list_id: None,
        };

        let batch_result: BatchRevokeResult = conductor
            .call(
                &cell.zome("revocation"),
                "batch_revoke_credentials",
                batch_input,
            )
            .await;

        assert_eq!(batch_result.revoked_count, 3, "Should revoke all 3");
        assert_eq!(batch_result.failed_count, 0, "No failures expected");
        assert_eq!(batch_result.revoked.len(), 3, "Should list 3 revoked IDs");

        // Verify each is revoked
        for cred_id in &[&cred_id_1, &vc2.id, &vc3.id] {
            let status: RevocationCheckResult = conductor
                .call(
                    &cell.zome("revocation"),
                    "check_revocation_status",
                    cred_id.to_string(),
                )
                .await;
            assert_eq!(
                status.status,
                RevocationStatus::Revoked,
                "Credential {} should be Revoked",
                cred_id
            );
        }
    }

    /// Create a revocation list and query it
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
    async fn test_revocation_list_management() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-rev-list", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Create DID
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("decode DID");
        let issuer_did = did_doc.id.clone();

        // Create a revocation list
        let create_list = CreateRevocationListInput {
            id: "rev-list-test-001".to_string(),
            issuer_did: issuer_did.clone(),
        };

        let list_record: Record = conductor
            .call(
                &cell.zome("revocation"),
                "create_revocation_list",
                create_list,
            )
            .await;

        let list: RevocationList = decode_entry(&list_record).expect("decode RevocationList");
        assert_eq!(list.id, "rev-list-test-001");
        assert_eq!(list.issuer, issuer_did);
        assert!(list.revoked.is_empty(), "New list should be empty");
        assert_eq!(list.version, 1, "Initial version should be 1");

        // Retrieve the list
        let retrieved: Option<Record> = conductor
            .call(
                &cell.zome("revocation"),
                "get_revocation_list",
                "rev-list-test-001".to_string(),
            )
            .await;

        assert!(retrieved.is_some(), "List should be retrievable");

        // Query lists by issuer
        let issuer_lists: Vec<Record> = conductor
            .call(
                &cell.zome("revocation"),
                "get_issuer_revocation_lists",
                issuer_did.clone(),
            )
            .await;

        assert!(
            !issuer_lists.is_empty(),
            "Issuer should have at least one revocation list"
        );
    }
}

// ============================================================================
// VC Signature Rejection Tests
// ============================================================================

#[cfg(test)]
mod vc_rejection_tests {
    use super::*;

    /// Issue a credential with immediate expiration → verify rejects it
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
    async fn test_vc_verify_rejects_expired_credential() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-vc-expired", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Create DID + schema
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("decode DID");

        let now = Timestamp::now();
        let schema = CredentialSchema {
            id: "mycelix:schema:expired-test:v1".to_string(),
            name: "Expiration Test".to_string(),
            description: "Schema for expiration tests".to_string(),
            version: "1.0.0".to_string(),
            author: did_doc.id.clone(),
            schema: r#"{"type":"object","properties":{"note":{"type":"string"}}}"#.to_string(),
            required_fields: vec!["note".to_string()],
            optional_fields: vec![],
            credential_type: vec!["VerifiableCredential".to_string()],
            default_expiration: 1, // 1 second — will expire almost immediately
            revocable: true,
            active: true,
            created: now,
            updated: now,
        };
        let _: Record = conductor
            .call(&cell.zome("credential_schema"), "create_schema", schema)
            .await;

        // Issue with 0 expiration days (should default to schema's 1 second)
        // Use expiration_days = Some(0) to try to force expired-on-creation behavior
        // The actual behavior depends on the coordinator: if 0 means "today", the credential
        // valid_until will be the creation timestamp, making it expired immediately.
        let issue_input = IssueCredentialInput {
            subject_did: format!("did:mycelix:expiry-test-{}", now.as_micros()),
            schema_id: "mycelix:schema:expired-test:v1".to_string(),
            claims: serde_json::json!({"note": "this will expire"}),
            credential_types: vec![],
            issuer_name: None,
            expiration_days: Some(0), // Expire immediately
            enable_revocation: false,
        };

        let vc_record: Record = conductor
            .call(
                &cell.zome("verifiable_credential"),
                "issue_credential",
                issue_input,
            )
            .await;

        let vc: VerifiableCredential = decode_entry(&vc_record).expect("decode VC");

        // Small delay to ensure expiration has passed
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        // Verify — should report invalid due to expiration
        let result: VcVerificationResult = conductor
            .call(
                &cell.zome("verifiable_credential"),
                "verify_credential",
                vc.id.clone(),
            )
            .await;

        assert!(
            !result.valid,
            "Credential with 0 expiration days should fail verification"
        );
        assert!(
            result.errors.iter().any(|e| {
                let lower = e.to_lowercase();
                lower.contains("expir") || lower.contains("valid_until")
            }),
            "Error should mention expiration, got: {:?}",
            result.errors
        );
    }

    /// Issue → revoke → check via VC zome's get_credential_status
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
    async fn test_vc_status_after_suspension() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-vc-suspension", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Create DID + schema + issue
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("decode DID");
        let issuer_did = did_doc.id.clone();

        let now = Timestamp::now();
        let schema = CredentialSchema {
            id: "mycelix:schema:suspend-status-test:v1".to_string(),
            name: "Suspension Status Test".to_string(),
            description: "Schema for suspension status tests".to_string(),
            version: "1.0.0".to_string(),
            author: issuer_did.clone(),
            schema: r#"{"type":"object","properties":{"role":{"type":"string"}}}"#.to_string(),
            required_fields: vec!["role".to_string()],
            optional_fields: vec![],
            credential_type: vec!["VerifiableCredential".to_string()],
            default_expiration: 86400 * 365,
            revocable: true,
            active: true,
            created: now,
            updated: now,
        };
        let _: Record = conductor
            .call(&cell.zome("credential_schema"), "create_schema", schema)
            .await;

        let issue_input = IssueCredentialInput {
            subject_did: format!("did:mycelix:suspend-test-{}", now.as_micros()),
            schema_id: "mycelix:schema:suspend-status-test:v1".to_string(),
            claims: serde_json::json!({"role": "engineer"}),
            credential_types: vec![],
            issuer_name: None,
            expiration_days: Some(365),
            enable_revocation: true,
        };

        let vc_record: Record = conductor
            .call(
                &cell.zome("verifiable_credential"),
                "issue_credential",
                issue_input,
            )
            .await;
        let vc: VerifiableCredential = decode_entry(&vc_record).expect("decode VC");

        // Suspend the credential
        let one_month = Timestamp::from_micros(now.as_micros() + 30 * 86400 * 1_000_000);
        let suspend = SuspendInput {
            credential_id: vc.id.clone(),
            issuer_did: issuer_did.clone(),
            reason: "Pending review".to_string(),
            suspension_end: one_month,
        };

        let _: Record = conductor
            .call(&cell.zome("revocation"), "suspend_credential", suspend)
            .await;

        // Check status via VC zome
        let status: CredentialStatusResponse = conductor
            .call(
                &cell.zome("verifiable_credential"),
                "get_credential_status",
                vc.id.clone(),
            )
            .await;

        assert!(
            !status.is_valid,
            "Suspended credential should not be valid"
        );
        assert!(
            status.status_type.to_lowercase().contains("suspend"),
            "Status type should mention suspension, got: {}",
            status.status_type
        );

        // Verify returns invalid too
        let result: VcVerificationResult = conductor
            .call(
                &cell.zome("verifiable_credential"),
                "verify_credential",
                vc.id.clone(),
            )
            .await;

        assert!(
            !result.valid,
            "Suspended credential should fail verification"
        );
    }
}

// ============================================================================
// Academic Education Credential Tests
// ============================================================================

#[cfg(test)]
mod education_tests {
    use super::*;

    /// Mirror of education_integrity::DegreeType
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

    /// Mirror of education_integrity::InstitutionLocation
    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    pub struct InstitutionLocation {
        pub country: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub region: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub city: Option<String>,
    }

    /// Mirror of education_integrity::Accreditation
    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    pub struct Accreditation {
        pub accreditor: String,
        pub accreditation_type: String,
        pub valid_from: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub valid_until: Option<String>,
    }

    /// Mirror of education_integrity::InstitutionalIssuer
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

    /// Mirror of education_integrity::AcademicSubject
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

    /// Mirror of education_integrity::AcademicProof
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

    /// Mirror of education_integrity::AchievementMetadata
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

    /// Mirror of education_integrity::DnssecStatus
    #[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    pub enum DnssecStatus {
        Validated,
        Insecure,
        Invalid,
        Unsigned,
        Unknown,
    }

    /// Mirror of education_integrity::DnsVerificationRecord
    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    pub struct DnsVerificationRecord {
        pub timestamp: Timestamp,
        pub resolver: String,
        pub dnssec_status: DnssecStatus,
        pub ttl_seconds: u32,
    }

    /// Mirror of education_integrity::DnsDid
    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    pub struct DnsDid {
        pub domain: String,
        pub did: String,
        pub txt_record: String,
        pub dnssec: DnssecStatus,
        pub last_verified: Timestamp,
        pub verification_chain: Vec<DnsVerificationRecord>,
    }

    /// Mirror of education coordinator::CreateAcademicCredentialInput
    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
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

    /// Mirror of education coordinator::CreateAcademicCredentialOutput
    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    pub struct CreateAcademicCredentialOutput {
        pub action_hash: ActionHash,
        pub credential_id: String,
        pub revocation_index: u32,
    }

    /// Build a test InstitutionalIssuer
    fn test_issuer(did: &str) -> InstitutionalIssuer {
        InstitutionalIssuer {
            id: did.to_string(),
            name: "Mycelix University".to_string(),
            issuer_type: vec!["EducationalOrganization".to_string()],
            image: None,
            location: Some(InstitutionLocation {
                country: "US".to_string(),
                region: Some("TX".to_string()),
                city: Some("Richardson".to_string()),
            }),
            accreditation: Some(vec![Accreditation {
                accreditor: "SACSCOC".to_string(),
                accreditation_type: "Regional".to_string(),
                valid_from: "2020-01-01".to_string(),
                valid_until: Some("2030-12-31".to_string()),
            }]),
        }
    }

    /// Build a test DnsDid
    fn test_dns_did(did: &str) -> DnsDid {
        let now = Timestamp::now();
        DnsDid {
            domain: "mycelix-university.edu".to_string(),
            did: did.to_string(),
            txt_record: format!("did={}", did),
            dnssec: DnssecStatus::Validated,
            last_verified: now,
            verification_chain: vec![DnsVerificationRecord {
                timestamp: now,
                resolver: "8.8.8.8".to_string(),
                dnssec_status: DnssecStatus::Validated,
                ttl_seconds: 3600,
            }],
        }
    }

    /// Build a test AcademicProof
    fn test_proof(did: &str) -> AcademicProof {
        AcademicProof {
            proof_type: "DataIntegrityProof".to_string(),
            created: "2026-02-12T00:00:00Z".to_string(),
            verification_method: format!("{}#key-1", did),
            proof_purpose: "assertionMethod".to_string(),
            proof_value: "z3FXQENoiDav5JJz5GWvg7yVmLkS5cpEafuEH8".to_string(),
            cryptosuite: Some("eddsa-jcs-2022".to_string()),
            domain: Some("mycelix-university.edu".to_string()),
            challenge: None,
            algorithm: None, // Ed25519 default
        }
    }

    /// Create an academic credential and retrieve it
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
    async fn test_create_and_get_academic_credential() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-edu-create", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Create DID for the institution
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("decode DID");
        let institution_did = did_doc.id.clone();

        let student_did = format!("did:mycelix:student-{}", Timestamp::now().as_micros());

        let input = CreateAcademicCredentialInput {
            issuer: test_issuer(&institution_did),
            subject: AcademicSubject {
                id: student_did.clone(),
                name: Some("Jane Doe".to_string()),
                name_hash: None,
                birth_date: Some("1998-05-15".to_string()),
                student_id: Some("STU-2026-001".to_string()),
            },
            achievement: AchievementMetadata {
                degree_type: DegreeType::Bachelor,
                degree_name: "Bachelor of Science".to_string(),
                field_of_study: "Computer Science".to_string(),
                minors: Some(vec!["Mathematics".to_string()]),
                conferral_date: "2026-05-15".to_string(),
                gpa: Some(3.85),
                honors: Some(vec!["Magna Cum Laude".to_string()]),
                cip_code: Some("11.0701".to_string()),
                credits_earned: Some(128),
            },
            dns_did: test_dns_did(&institution_did),
            revocation_registry_id: "rev-registry-edu-001".to_string(),
            valid_from: "2026-05-15T00:00:00Z".to_string(),
            valid_until: None,
            proof: test_proof(&institution_did),
        };

        let output: CreateAcademicCredentialOutput = conductor
            .call(
                &cell.zome("education"),
                "create_academic_credential",
                input,
            )
            .await;

        assert!(
            !output.credential_id.is_empty(),
            "Credential ID should be non-empty"
        );

        // Retrieve the credential
        let retrieved: Option<Record> = conductor
            .call(
                &cell.zome("education"),
                "get_academic_credential",
                output.action_hash.clone(),
            )
            .await;

        assert!(
            retrieved.is_some(),
            "Should be able to retrieve the academic credential"
        );
    }

    /// Create credentials from two institutions, query by institution
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
    async fn test_query_credentials_by_institution() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-edu-by-inst", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Create DID
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("decode DID");
        let institution_did = did_doc.id.clone();

        let now = Timestamp::now();

        // Issue two credentials from the same institution
        for i in 0..2 {
            let student_did = format!("did:mycelix:student-{}-{}", i, now.as_micros());
            let input = CreateAcademicCredentialInput {
                issuer: test_issuer(&institution_did),
                subject: AcademicSubject {
                    id: student_did,
                    name: Some(format!("Student {}", i)),
                    name_hash: None,
                    birth_date: None,
                    student_id: Some(format!("STU-{}", i)),
                },
                achievement: AchievementMetadata {
                    degree_type: DegreeType::Master,
                    degree_name: "Master of Science".to_string(),
                    field_of_study: "Data Science".to_string(),
                    minors: None,
                    conferral_date: "2026-05-15".to_string(),
                    gpa: None,
                    honors: None,
                    cip_code: None,
                    credits_earned: None,
                },
                dns_did: test_dns_did(&institution_did),
                revocation_registry_id: format!("rev-registry-{}", i),
                valid_from: "2026-05-15T00:00:00Z".to_string(),
                valid_until: None,
                proof: test_proof(&institution_did),
            };

            let _: CreateAcademicCredentialOutput = conductor
                .call(
                    &cell.zome("education"),
                    "create_academic_credential",
                    input,
                )
                .await;
        }

        // Query by institution
        let results: Vec<ActionHash> = conductor
            .call(
                &cell.zome("education"),
                "get_credentials_by_institution",
                institution_did.clone(),
            )
            .await;

        assert!(
            results.len() >= 2,
            "Should find at least 2 credentials for institution, got {}",
            results.len()
        );
    }

    /// Create a credential and query by subject DID
    #[tokio::test(flavor = "multi_thread")]
    #[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
    async fn test_query_credentials_by_subject() {
        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("test-edu-by-subj", &[dna]).await.unwrap();
        let cell = app.cells()[0].clone();

        // Create DID
        let did_record: Record = conductor
            .call(&cell.zome("did_registry"), "create_did", ())
            .await;
        let did_doc: DidDocument = decode_entry(&did_record).expect("decode DID");
        let institution_did = did_doc.id.clone();

        let student_did = format!("did:mycelix:student-subj-{}", Timestamp::now().as_micros());

        let input = CreateAcademicCredentialInput {
            issuer: test_issuer(&institution_did),
            subject: AcademicSubject {
                id: student_did.clone(),
                name: Some("Alice Wonderland".to_string()),
                name_hash: None,
                birth_date: None,
                student_id: Some("STU-ALICE-001".to_string()),
            },
            achievement: AchievementMetadata {
                degree_type: DegreeType::Doctorate,
                degree_name: "Doctor of Philosophy".to_string(),
                field_of_study: "Quantum Computing".to_string(),
                minors: None,
                conferral_date: "2026-06-01".to_string(),
                gpa: Some(3.95),
                honors: Some(vec!["With Distinction".to_string()]),
                cip_code: Some("11.0104".to_string()),
                credits_earned: Some(72),
            },
            dns_did: test_dns_did(&institution_did),
            revocation_registry_id: "rev-registry-subj-001".to_string(),
            valid_from: "2026-06-01T00:00:00Z".to_string(),
            valid_until: None,
            proof: test_proof(&institution_did),
        };

        let _: CreateAcademicCredentialOutput = conductor
            .call(
                &cell.zome("education"),
                "create_academic_credential",
                input,
            )
            .await;

        // Query by subject
        let results: Vec<ActionHash> = conductor
            .call(
                &cell.zome("education"),
                "get_credentials_by_subject",
                student_did.clone(),
            )
            .await;

        assert!(
            !results.is_empty(),
            "Should find at least one credential for subject"
        );
    }
}
