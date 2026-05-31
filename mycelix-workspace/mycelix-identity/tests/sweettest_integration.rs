// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! # Mycelix Identity - Sweettest Integration Tests
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

use holochain::prelude::*;
use holochain::sweettest::*;
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

        assert!(
            did_doc.id.starts_with("did:mycelix:"),
            "DID must use mycelix method"
        );
        assert_eq!(
            did_doc.controller, agent,
            "Controller must match creating agent"
        );
        assert!(
            !did_doc.verification_method.is_empty(),
            "Must have verification method"
        );
        assert_eq!(did_doc.version, 1, "Initial version must be 1");

        // Resolve by DID string
        let resolved: Option<Record> = conductor
            .call(
                &cell.zome("did_registry"),
                "resolve_did",
                did_doc.id.clone(),
            )
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
            .call(
                &cell.zome("did_registry"),
                "get_did_document",
                agent.clone(),
            )
            .await;

        assert!(did_record.is_some(), "Should find DID for agent");

        let did_doc: DidDocument = decode_entry(&did_record.unwrap()).expect("Failed to decode");
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
            .call(
                &cell.zome("did_registry"),
                "add_service_endpoint",
                service.clone(),
            )
            .await;

        let updated_doc: DidDocument = decode_entry(&updated_record).expect("Failed to decode");

        assert_eq!(updated_doc.service.len(), 1, "Should have one service");
        assert_eq!(
            updated_doc.service[0].id, service.id,
            "Service ID must match"
        );
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
            .call(
                &cell.zome("did_registry"),
                "is_did_active",
                did_doc.id.clone(),
            )
            .await;

        assert!(is_active_before, "DID should be active initially");

        // Deactivate
        let reason = "Key rotation required".to_string();
        let deactivation_record: Record = conductor
            .call(&cell.zome("did_registry"), "deactivate_did", reason.clone())
            .await;

        let deactivation: DidDeactivation =
            decode_entry(&deactivation_record).expect("Failed to decode deactivation");

        assert_eq!(
            deactivation.did, did_doc.id,
            "Deactivation must reference DID"
        );
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
            .call(
                &cell.zome("credential_schema"),
                "create_schema",
                schema.clone(),
            )
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
            let schema: CredentialSchema = decode_entry(record).expect("Failed to decode schema");
            assert_eq!(
                schema.author, author_did,
                "All schemas should belong to author"
            );
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
            .call(
                &cell.zome("identity_bridge"),
                "report_reputation",
                reputation,
            )
            .await;

        // 5. Verify complete state
        let final_did: Option<Record> = conductor
            .call(&cell.zome("did_registry"), "get_my_did", ())
            .await;

        assert!(final_did.is_some(), "DID should exist");

        let matl_score: f64 = conductor
            .call(
                &cell.zome("identity_bridge"),
                "get_matl_score",
                did_doc.id.clone(),
            )
            .await;

        assert!(
            matl_score > 0.0,
            "MATL score should be positive after reputation report"
        );
    }
}
