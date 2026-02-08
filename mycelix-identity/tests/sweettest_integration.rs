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

/// Mirror of UpdateDidInput for security tests
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UpdateDidInput {
    pub verification_method: Option<Vec<VerificationMethod>>,
    pub authentication: Option<Vec<String>>,
    pub service: Option<Vec<ServiceEndpoint>>,
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires holochain conductor - run with: cargo test --release -- --ignored"]
async fn test_reject_malformed_multibase_key_no_prefix() {
    let conductor = SweetConductor::from_standard_config().await;
    let dna_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("dna/mycelix_identity_dna.dna");
    let dna = SweetDnaFile::from_bundle(&dna_path).await.unwrap();
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
    let conductor = SweetConductor::from_standard_config().await;
    let dna_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("dna/mycelix_identity_dna.dna");
    let dna = SweetDnaFile::from_bundle(&dna_path).await.unwrap();
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
    let conductor = SweetConductor::from_standard_config().await;
    let dna_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("dna/mycelix_identity_dna.dna");
    let dna = SweetDnaFile::from_bundle(&dna_path).await.unwrap();
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
    let conductor = SweetConductor::from_standard_config().await;
    let dna_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("dna/mycelix_identity_dna.dna");
    let dna = SweetDnaFile::from_bundle(&dna_path).await.unwrap();
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
        service: None,
    };

    // Should succeed - X25519 keys are not validated by Ed25519 multibase rules
    let result: Result<Record, _> = conductor
        .call_fallible(&cell.zome("did_registry"), "update_did_document", update)
        .await;

    assert!(result.is_ok(), "Non-Ed25519 key types should bypass multibase Ed25519 validation");
}
