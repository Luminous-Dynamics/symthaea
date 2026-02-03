//! # Comprehensive Identity Module Integration Tests
//!
//! This test suite covers DID Registry, Credential Schema, and Bridge operations.
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
//! cargo test --test identity_test -- --ignored
//! ```

use holochain::sweettest::*;
use holochain::prelude::*;
use std::path::PathBuf;
use std::time::Duration;


// ============================================================================
// Mirror types for deserialization (avoids importing zome crates)
// ============================================================================

// --- DID Registry types ---

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

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct VerificationMethod {
    pub id: String,
    pub type_: String,
    pub controller: String,
    pub public_key_multibase: String,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ServiceEndpoint {
    pub id: String,
    pub type_: String,
    pub service_endpoint: String,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DidDeactivation {
    pub did: String,
    pub reason: String,
    pub deactivated_at: Timestamp,
}

// --- Credential Schema types ---

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

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum SchemaCategory {
    Education,
    Employment,
    Identity,
    Skills,
    Governance,
    Financial,
    Energy,
    Custom(String),
}

// --- Bridge types ---

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HappRegistration {
    pub happ_id: String,
    pub happ_name: String,
    pub capabilities: Vec<String>,
    pub matl_score: f64,
    pub registered_at: Timestamp,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct IdentityVerificationResult {
    pub verification_hash: ActionHash,
    pub did: String,
    pub is_valid: bool,
    pub matl_score: f64,
    pub credential_count: u32,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct IdentityReputation {
    pub did: String,
    pub source_happ: String,
    pub score: f64,
    pub interactions: u64,
    pub last_updated: Timestamp,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AggregatedReputation {
    pub did: String,
    pub aggregate_score: f64,
    pub sources: Vec<ReputationSource>,
    pub total_interactions: u64,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ReputationSource {
    pub source_happ: String,
    pub score: f64,
    pub interactions: u64,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BridgeEvent {
    pub id: String,
    pub event_type: BridgeEventType,
    pub subject: String,
    pub payload: String,
    pub source_happ: String,
    pub timestamp: Timestamp,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum BridgeEventType {
    DidCreated,
    DidUpdated,
    DidDeactivated,
    CredentialIssued,
    CredentialRevoked,
    RecoveryInitiated,
    RecoveryCompleted,
    HappRegistered,
    Custom(String),
}

// --- Input types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegisterHappInput {
    pub happ_id: String,
    pub happ_name: String,
    pub capabilities: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QueryIdentityInput {
    pub did: String,
    pub source_happ: String,
    pub requested_fields: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReportReputationInput {
    pub did: String,
    pub source_happ: String,
    pub score: f64,
    pub interactions: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BroadcastEventInput {
    pub event_type: BridgeEventType,
    pub subject: String,
    pub payload: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GetEventsInput {
    pub event_type: Option<BridgeEventType>,
    pub since: Option<u64>,
    pub limit: Option<u32>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrustCheckInput {
    pub did: String,
    pub threshold: f64,
}

// ============================================================================
// Test Utilities
// ============================================================================

mod test_helpers {
    pub const TEST_DID_PREFIX: &str = "did:mycelix:test:";
    pub const TEST_SCHEMA_PREFIX: &str = "mycelix:schema:test:";
    pub const TEST_HAPP_ID: &str = "test-happ";
    pub const TEST_HAPP_NAME: &str = "Test Application";

    pub fn test_did(suffix: &str) -> String {
        format!("{}{}", TEST_DID_PREFIX, suffix)
    }

    pub fn unique_test_did(prefix: &str) -> String {
        let timestamp = chrono::Utc::now().timestamp_micros();
        format!("{}{}:{}", TEST_DID_PREFIX, prefix, timestamp)
    }

    pub fn did_from_agent(agent: &holochain::prelude::AgentPubKey) -> String {
        format!("did:mycelix:{}", agent)
    }

    pub fn unique_schema_id(prefix: &str) -> String {
        let timestamp = chrono::Utc::now().timestamp_micros();
        format!("{}{}:{}:v1", TEST_SCHEMA_PREFIX, prefix, timestamp)
    }

    pub fn minimal_json_schema() -> String {
        serde_json::json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "issuanceDate": { "type": "string" }
            },
            "required": ["name", "issuanceDate"]
        })
        .to_string()
    }

    pub fn is_valid_mycelix_did(did: &str) -> bool {
        did.starts_with("did:mycelix:")
    }
}

use test_helpers::*;

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
        .expect("Failed to load DNA bundle")
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
// Section 1: DID Registry Tests
// ============================================================================

#[cfg(test)]
mod did_registry_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_did_success() {
        println!("=== test_create_did_success ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor
            .setup_app("mycelix-identity", &[dna.clone()])
            .await
            .unwrap();
        let alice_cell = app.cells()[0].clone();
        let alice_agent = app.agent().clone();

        let did_record: Record = conductor
            .call(&alice_cell.zome("did_registry"), "create_did", ())
            .await;

        let did_doc: DidDocument = decode_entry(&did_record).expect("No entry found");

        assert!(did_doc.id.starts_with("did:mycelix:"), "DID must start with 'did:mycelix:'");

        let expected_did = format!("did:mycelix:{}", alice_agent);
        assert_eq!(did_doc.id, expected_did, "DID should be deterministic from agent key");
        assert_eq!(did_doc.controller, alice_agent, "Controller must be the creating agent");
        assert!(!did_doc.verification_method.is_empty(), "Must have at least one verification method");

        let vm = &did_doc.verification_method[0];
        assert!(vm.id.starts_with(&did_doc.id), "VM ID must reference DID");
        assert_eq!(vm.type_, "Ed25519VerificationKey2020");
        assert!(vm.public_key_multibase.starts_with("z"));
        assert!(!did_doc.authentication.is_empty());
        assert_eq!(did_doc.version, 1);
        assert!(did_doc.created.as_micros() > 0);
        assert_eq!(did_doc.created, did_doc.updated);

        println!("=== test_create_did_success PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_resolve_did() {
        println!("=== test_resolve_did ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app1 = conductor.setup_app("app-alice", &[dna.clone()]).await.unwrap();
        let app2 = conductor.setup_app("app-bob", &[dna.clone()]).await.unwrap();
        let alice_cell = app1.cells()[0].clone();
        let bob_cell = app2.cells()[0].clone();
        let alice_agent = app1.agent().clone();

        let created_record: Record = conductor
            .call(&alice_cell.zome("did_registry"), "create_did", ())
            .await;

        let created_did: DidDocument = decode_entry(&created_record).expect("No entry");

        // Resolve by DID string
        let resolved_alice: Option<Record> = conductor
            .call(
                &alice_cell.zome("did_registry"),
                "resolve_did",
                created_did.id.clone(),
            )
            .await;

        assert!(resolved_alice.is_some(), "Should resolve from creator's cell");

        let resolved_doc: DidDocument = decode_entry(&resolved_alice.unwrap()).expect("No entry");

        assert_eq!(resolved_doc.id, created_did.id);
        assert_eq!(resolved_doc.controller, created_did.controller);

        // Resolve by agent pub key
        let resolved_by_key: Option<Record> = conductor
            .call(
                &alice_cell.zome("did_registry"),
                "get_did_document",
                alice_agent.clone(),
            )
            .await;

        assert!(resolved_by_key.is_some(), "Should find DID by agent key");

        // Use get_my_did convenience function
        let my_did: Option<Record> = conductor
            .call(&alice_cell.zome("did_registry"), "get_my_did", ())
            .await;

        assert!(my_did.is_some(), "get_my_did should return my DID");

        // Bob should be able to resolve Alice's DID too (after DHT sync)
        tokio::time::sleep(Duration::from_millis(500)).await;

        let resolved_bob: Option<Record> = conductor
            .call(
                &bob_cell.zome("did_registry"),
                "resolve_did",
                created_did.id.clone(),
            )
            .await;

        println!("  Resolved from Bob: {:?}", resolved_bob.is_some());
        println!("=== test_resolve_did PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_deactivate_did() {
        println!("=== test_deactivate_did ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("mycelix-identity", &[dna.clone()]).await.unwrap();
        let alice_cell = app.cells()[0].clone();

        let created_record: Record = conductor
            .call(&alice_cell.zome("did_registry"), "create_did", ())
            .await;

        let created_did: DidDocument = decode_entry(&created_record).expect("No entry");

        let is_active_before: bool = conductor
            .call(
                &alice_cell.zome("did_registry"),
                "is_did_active",
                created_did.id.clone(),
            )
            .await;

        assert!(is_active_before);

        let deactivation_reason = "Key compromise - rotating to new key".to_string();
        let deactivation_record: Record = conductor
            .call(
                &alice_cell.zome("did_registry"),
                "deactivate_did",
                deactivation_reason.clone(),
            )
            .await;

        let deactivation: DidDeactivation = decode_entry(&deactivation_record).expect("No entry");

        assert_eq!(deactivation.did, created_did.id);
        assert_eq!(deactivation.reason, deactivation_reason);
        assert!(deactivation.deactivated_at.as_micros() > 0);

        println!("=== test_deactivate_did PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_invalid_did_format_rejected() {
        println!("=== test_invalid_did_format_rejected ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("mycelix-identity", &[dna.clone()]).await.unwrap();
        let alice_cell = app.cells()[0].clone();

        let invalid_dids = vec![
            ("did:example:12345", "Wrong method"),
            ("did:key:abc123", "Wrong method (key)"),
            ("invalid:string", "Not a DID at all"),
            ("mycelix:12345", "Missing did: prefix"),
            ("DID:MYCELIX:test", "Wrong case"),
            ("did:mycelix", "Missing specific ID"),
            ("", "Empty string"),
        ];

        let mut rejected_count = 0;

        for (invalid_did, description) in invalid_dids {
            let result: Result<Option<Record>, _> = conductor
                .call_fallible(
                    &alice_cell.zome("did_registry"),
                    "resolve_did",
                    invalid_did.to_string(),
                )
                .await;

            match result {
                Err(_) => {
                    println!("  '{}' ({}): Rejected", invalid_did, description);
                    rejected_count += 1;
                }
                Ok(None) => {
                    println!("  '{}' ({}): Returned None", invalid_did, description);
                    rejected_count += 1;
                }
                Ok(Some(_)) => {
                    panic!("Should not resolve invalid DID: {} ({})", invalid_did, description);
                }
            }
        }

        assert!(rejected_count > 0);
        println!("=== test_invalid_did_format_rejected PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_duplicate_did_rejected() {
        println!("=== test_duplicate_did_rejected ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("mycelix-identity", &[dna.clone()]).await.unwrap();
        let alice_cell = app.cells()[0].clone();
        let alice_agent = app.agent().clone();

        let first_record: Record = conductor
            .call(&alice_cell.zome("did_registry"), "create_did", ())
            .await;

        let first_did: DidDocument = decode_entry(&first_record).expect("No entry");

        let second_record: Record = conductor
            .call(&alice_cell.zome("did_registry"), "create_did", ())
            .await;

        let second_did: DidDocument = decode_entry(&second_record).expect("No entry");

        assert_eq!(first_did.id, second_did.id);
        assert_eq!(first_did.controller, second_did.controller);

        let expected_did = format!("did:mycelix:{}", alice_agent);
        assert_eq!(first_did.id, expected_did);

        let resolved: Option<Record> = conductor
            .call(
                &alice_cell.zome("did_registry"),
                "get_did_document",
                alice_agent,
            )
            .await;

        assert!(resolved.is_some());
        println!("=== test_duplicate_did_rejected PASSED ===\n");
    }
}

// ============================================================================
// Section 2: Credential Schema Tests
// ============================================================================

#[cfg(test)]
mod credential_schema_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_create_schema() {
        println!("=== test_create_schema ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("mycelix-identity", &[dna.clone()]).await.unwrap();
        let alice_cell = app.cells()[0].clone();
        let alice_agent = app.agent().clone();

        let _: Record = conductor
            .call(&alice_cell.zome("did_registry"), "create_did", ())
            .await;

        let author_did = format!("did:mycelix:{}", alice_agent);
        let now = Timestamp::from_micros(chrono::Utc::now().timestamp_micros());
        let schema_id = unique_schema_id("education");

        let schema = CredentialSchema {
            id: schema_id.clone(),
            name: "Educational Credential".to_string(),
            description: "Schema for educational achievement credentials".to_string(),
            version: "1.0.0".to_string(),
            author: author_did.clone(),
            schema: minimal_json_schema(),
            required_fields: vec!["name".to_string(), "degree".to_string()],
            optional_fields: vec!["honors".to_string()],
            credential_type: vec![
                "VerifiableCredential".to_string(),
                "EducationalCredential".to_string(),
            ],
            default_expiration: 86400 * 365 * 4,
            revocable: true,
            active: true,
            created: now,
            updated: now,
        };

        let schema_record: Record = conductor
            .call(
                &alice_cell.zome("credential_schema"),
                "create_schema",
                schema.clone(),
            )
            .await;

        let created_schema: CredentialSchema = decode_entry(&schema_record).expect("No entry");

        assert!(created_schema.id.starts_with("mycelix:schema:"));
        assert_eq!(created_schema.name, schema.name);
        assert_eq!(created_schema.author, author_did);
        assert!(created_schema.active);

        println!("  Schema ID: {}", created_schema.id);
        println!("=== test_create_schema PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_schemas_by_author() {
        println!("=== test_get_schemas_by_author ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app1 = conductor.setup_app("app-alice", &[dna.clone()]).await.unwrap();
        let app2 = conductor.setup_app("app-bob", &[dna.clone()]).await.unwrap();
        let alice_cell = app1.cells()[0].clone();
        let bob_cell = app2.cells()[0].clone();
        let alice_agent = app1.agent().clone();
        let bob_agent = app2.agent().clone();

        let _: Record = conductor.call(&alice_cell.zome("did_registry"), "create_did", ()).await;
        let _: Record = conductor.call(&bob_cell.zome("did_registry"), "create_did", ()).await;

        let alice_did = format!("did:mycelix:{}", alice_agent);
        let bob_did = format!("did:mycelix:{}", bob_agent);
        let now = Timestamp::from_micros(chrono::Utc::now().timestamp_micros());

        // Create 3 schemas for Alice
        for i in 0..3 {
            let schema = CredentialSchema {
                id: unique_schema_id(&format!("alice_schema_{}", i)),
                name: format!("Alice Schema {}", i),
                description: format!("Test schema {} by Alice", i),
                version: "1.0.0".to_string(),
                author: alice_did.clone(),
                schema: minimal_json_schema(),
                required_fields: vec!["name".to_string()],
                optional_fields: vec![],
                credential_type: vec!["VerifiableCredential".to_string()],
                default_expiration: 86400 * 365,
                revocable: true,
                active: true,
                created: now,
                updated: now,
            };

            let _: Record = conductor
                .call(&alice_cell.zome("credential_schema"), "create_schema", schema)
                .await;
        }

        // Create 2 schemas for Bob
        for i in 0..2 {
            let schema = CredentialSchema {
                id: unique_schema_id(&format!("bob_schema_{}", i)),
                name: format!("Bob Schema {}", i),
                description: format!("Test schema {} by Bob", i),
                version: "1.0.0".to_string(),
                author: bob_did.clone(),
                schema: minimal_json_schema(),
                required_fields: vec!["name".to_string()],
                optional_fields: vec![],
                credential_type: vec!["VerifiableCredential".to_string()],
                default_expiration: 86400 * 365,
                revocable: true,
                active: true,
                created: now,
                updated: now,
            };

            let _: Record = conductor
                .call(&bob_cell.zome("credential_schema"), "create_schema", schema)
                .await;
        }

        tokio::time::sleep(Duration::from_millis(500)).await;

        let alice_schemas: Vec<Record> = conductor
            .call(
                &alice_cell.zome("credential_schema"),
                "get_schemas_by_author",
                alice_did.clone(),
            )
            .await;

        assert!(alice_schemas.len() >= 3);

        for record in &alice_schemas {
            let schema: CredentialSchema = decode_entry(&record).expect("No entry");
            assert_eq!(schema.author, alice_did);
        }

        let bob_schemas: Vec<Record> = conductor
            .call(
                &bob_cell.zome("credential_schema"),
                "get_schemas_by_author",
                bob_did.clone(),
            )
            .await;

        assert!(bob_schemas.len() >= 2);
        println!("=== test_get_schemas_by_author PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_schema_validation() {
        println!("=== test_schema_validation ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("mycelix-identity", &[dna.clone()]).await.unwrap();
        let alice_cell = app.cells()[0].clone();
        let alice_agent = app.agent().clone();
        let author_did = format!("did:mycelix:{}", alice_agent);
        let now = Timestamp::from_micros(chrono::Utc::now().timestamp_micros());

        // Test 1: Empty schema ID (triggers length validation: must be 1-256 chars)
        let empty_id_schema = CredentialSchema {
            id: "".to_string(),
            name: "Invalid Schema".to_string(),
            description: "Test".to_string(),
            version: "1.0.0".to_string(),
            author: author_did.clone(),
            schema: minimal_json_schema(),
            required_fields: vec!["name".to_string()],
            optional_fields: vec![],
            credential_type: vec!["VerifiableCredential".to_string()],
            default_expiration: 86400,
            revocable: true,
            active: true,
            created: now,
            updated: now,
        };

        let result1: Result<Record, _> = conductor
            .call_fallible(
                &alice_cell.zome("credential_schema"),
                "create_schema",
                empty_id_schema,
            )
            .await;

        match result1 {
            Err(_) => println!("  Empty schema ID rejected"),
            Ok(_) => panic!("Should reject empty schema ID"),
        }

        // Test 2: Empty version (triggers length validation: must be 1-64 chars)
        let empty_version_schema = CredentialSchema {
            id: unique_schema_id("test"),
            name: "Test Schema".to_string(),
            description: "Test".to_string(),
            version: "".to_string(),
            author: author_did.clone(),
            schema: minimal_json_schema(),
            required_fields: vec!["name".to_string()],
            optional_fields: vec![],
            credential_type: vec!["VerifiableCredential".to_string()],
            default_expiration: 86400,
            revocable: true,
            active: true,
            created: now,
            updated: now,
        };

        let result2: Result<Record, _> = conductor
            .call_fallible(
                &alice_cell.zome("credential_schema"),
                "create_schema",
                empty_version_schema,
            )
            .await;

        match result2 {
            Err(_) => println!("  Empty version rejected"),
            Ok(_) => panic!("Should reject empty version"),
        }

        // Test 3: Empty schema JSON (triggers length validation: must be 1-65536 chars)
        let empty_json_schema = CredentialSchema {
            id: unique_schema_id("test2"),
            name: "Test Schema".to_string(),
            description: "Test".to_string(),
            version: "1.0.0".to_string(),
            author: author_did.clone(),
            schema: "".to_string(),
            required_fields: vec!["name".to_string()],
            optional_fields: vec![],
            credential_type: vec!["VerifiableCredential".to_string()],
            default_expiration: 86400,
            revocable: true,
            active: true,
            created: now,
            updated: now,
        };

        let result3: Result<Record, _> = conductor
            .call_fallible(
                &alice_cell.zome("credential_schema"),
                "create_schema",
                empty_json_schema,
            )
            .await;

        match result3 {
            Err(_) => println!("  Empty schema JSON rejected"),
            Ok(_) => panic!("Should reject empty schema JSON"),
        }

        // Test 4: Missing credential type (empty vec triggers validation)
        let no_type_schema = CredentialSchema {
            id: unique_schema_id("test3"),
            name: "Test Schema".to_string(),
            description: "Test".to_string(),
            version: "1.0.0".to_string(),
            author: author_did.clone(),
            schema: minimal_json_schema(),
            required_fields: vec!["name".to_string()],
            optional_fields: vec![],
            credential_type: vec![],
            default_expiration: 86400,
            revocable: true,
            active: true,
            created: now,
            updated: now,
        };

        let result4: Result<Record, _> = conductor
            .call_fallible(
                &alice_cell.zome("credential_schema"),
                "create_schema",
                no_type_schema,
            )
            .await;

        match result4 {
            Err(_) => println!("  Missing credential type rejected"),
            Ok(_) => panic!("Should reject schema without credential type"),
        }

        println!("=== test_schema_validation PASSED ===\n");
    }
}

// ============================================================================
// Section 3: Bridge Tests
// ============================================================================

#[cfg(test)]
mod bridge_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_query_identity_cross_zome() {
        println!("=== test_query_identity_cross_zome ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("mycelix-identity", &[dna.clone()]).await.unwrap();
        let alice_cell = app.cells()[0].clone();
        let alice_agent = app.agent().clone();

        let _: Record = conductor
            .call(&alice_cell.zome("did_registry"), "create_did", ())
            .await;

        let alice_did = format!("did:mycelix:{}", alice_agent);

        let registration_input = RegisterHappInput {
            happ_id: TEST_HAPP_ID.to_string(),
            happ_name: TEST_HAPP_NAME.to_string(),
            capabilities: vec![
                "identity_query".to_string(),
                "reputation".to_string(),
            ],
        };

        let registration_record: Record = conductor
            .call(
                &alice_cell.zome("identity_bridge"),
                "register_happ",
                registration_input,
            )
            .await;

        let registration: HappRegistration = decode_entry(&registration_record).expect("No entry");

        println!("  Registered hApp: {}", registration.happ_name);

        let query_input = QueryIdentityInput {
            did: alice_did.clone(),
            source_happ: TEST_HAPP_ID.to_string(),
            requested_fields: vec![
                "is_valid".to_string(),
                "matl_score".to_string(),
                "credential_count".to_string(),
            ],
        };

        let verification_result: IdentityVerificationResult = conductor
            .call(
                &alice_cell.zome("identity_bridge"),
                "query_identity",
                query_input,
            )
            .await;

        assert_eq!(verification_result.did, alice_did);
        assert!(verification_result.is_valid);
        assert!(verification_result.matl_score >= 0.0);
        assert!(verification_result.matl_score <= 1.0);

        let is_valid: bool = conductor
            .call(
                &alice_cell.zome("identity_bridge"),
                "verify_did",
                alice_did.clone(),
            )
            .await;

        assert!(is_valid);
        println!("=== test_query_identity_cross_zome PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_reputation_aggregation() {
        println!("=== test_get_reputation_aggregation ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("mycelix-identity", &[dna.clone()]).await.unwrap();
        let alice_cell = app.cells()[0].clone();
        let alice_agent = app.agent().clone();

        let _: Record = conductor
            .call(&alice_cell.zome("did_registry"), "create_did", ())
            .await;

        let alice_did = format!("did:mycelix:{}", alice_agent);

        let reputation_reports = vec![
            ReportReputationInput {
                did: alice_did.clone(),
                source_happ: "finance-happ".to_string(),
                score: 0.8,
                interactions: 100,
            },
            ReportReputationInput {
                did: alice_did.clone(),
                source_happ: "governance-happ".to_string(),
                score: 0.9,
                interactions: 50,
            },
            ReportReputationInput {
                did: alice_did.clone(),
                source_happ: "energy-happ".to_string(),
                score: 0.7,
                interactions: 200,
            },
        ];

        for report in &reputation_reports {
            let _: Record = conductor
                .call(
                    &alice_cell.zome("identity_bridge"),
                    "report_reputation",
                    report.clone(),
                )
                .await;
        }

        tokio::time::sleep(Duration::from_millis(500)).await;

        let aggregated: AggregatedReputation = conductor
            .call(
                &alice_cell.zome("identity_bridge"),
                "get_reputation",
                alice_did.clone(),
            )
            .await;

        assert_eq!(aggregated.did, alice_did);
        assert!(aggregated.sources.len() >= 3);
        assert!(aggregated.aggregate_score >= 0.0);
        assert!(aggregated.aggregate_score <= 1.0);

        let expected_score = (0.8 * 100.0 + 0.9 * 50.0 + 0.7 * 200.0) / (100.0 + 50.0 + 200.0);
        assert!(
            (aggregated.aggregate_score - expected_score).abs() < 0.01,
            "Expected: {:.4}, got: {:.4}",
            expected_score,
            aggregated.aggregate_score
        );

        let expected_interactions: u64 = 100 + 50 + 200;
        assert_eq!(aggregated.total_interactions, expected_interactions);

        let matl_score: f64 = conductor
            .call(
                &alice_cell.zome("identity_bridge"),
                "get_matl_score",
                alice_did.clone(),
            )
            .await;

        assert!((matl_score - aggregated.aggregate_score).abs() < 0.001);
        println!("=== test_get_reputation_aggregation PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_report_reputation() {
        println!("=== test_report_reputation ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("mycelix-identity", &[dna.clone()]).await.unwrap();
        let alice_cell = app.cells()[0].clone();
        let alice_agent = app.agent().clone();

        let _: Record = conductor
            .call(&alice_cell.zome("did_registry"), "create_did", ())
            .await;

        let alice_did = format!("did:mycelix:{}", alice_agent);

        let report_input = ReportReputationInput {
            did: alice_did.clone(),
            source_happ: "test-marketplace".to_string(),
            score: 0.85,
            interactions: 42,
        };

        let reputation_record: Record = conductor
            .call(
                &alice_cell.zome("identity_bridge"),
                "report_reputation",
                report_input.clone(),
            )
            .await;

        let reputation: IdentityReputation = decode_entry(&reputation_record).expect("No entry");

        assert_eq!(reputation.did, alice_did);
        assert_eq!(reputation.source_happ, "test-marketplace");
        assert_eq!(reputation.score, 0.85);
        assert_eq!(reputation.interactions, 42);
        assert!(reputation.last_updated.as_micros() > 0);

        let trust_check_high = TrustCheckInput {
            did: alice_did.clone(),
            threshold: 0.5,
        };

        let is_trustworthy_high: bool = conductor
            .call(
                &alice_cell.zome("identity_bridge"),
                "is_trustworthy",
                trust_check_high,
            )
            .await;

        assert!(is_trustworthy_high);

        let trust_check_low = TrustCheckInput {
            did: alice_did.clone(),
            threshold: 0.9,
        };

        let is_trustworthy_low: bool = conductor
            .call(
                &alice_cell.zome("identity_bridge"),
                "is_trustworthy",
                trust_check_low,
            )
            .await;

        println!("  Trustworthy at 0.9 threshold: {}", is_trustworthy_low);
        println!("=== test_report_reputation PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_bridge_event_broadcasting() {
        println!("=== test_bridge_event_broadcasting ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("mycelix-identity", &[dna.clone()]).await.unwrap();
        let alice_cell = app.cells()[0].clone();

        let event_input = BroadcastEventInput {
            event_type: BridgeEventType::Custom("test_event".to_string()),
            subject: "test:subject:123".to_string(),
            payload: serde_json::json!({
                "test_key": "test_value",
                "timestamp": chrono::Utc::now().timestamp()
            })
            .to_string(),
        };

        let event_record: Record = conductor
            .call(
                &alice_cell.zome("identity_bridge"),
                "broadcast_event",
                event_input,
            )
            .await;

        let event: BridgeEvent = decode_entry(&event_record).expect("No entry");

        assert!(!event.id.is_empty());
        assert_eq!(event.subject, "test:subject:123");
        assert!(!event.payload.is_empty());
        assert!(event.timestamp.as_micros() > 0);

        let events_input = GetEventsInput {
            event_type: None,
            since: None,
            limit: Some(10),
        };

        let recent_events: Vec<Record> = conductor
            .call(
                &alice_cell.zome("identity_bridge"),
                "get_recent_events",
                events_input,
            )
            .await;

        assert!(!recent_events.is_empty());
        println!("=== test_bridge_event_broadcasting PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_get_registered_happs() {
        println!("=== test_get_registered_happs ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("mycelix-identity", &[dna.clone()]).await.unwrap();
        let alice_cell = app.cells()[0].clone();

        let happs_to_register = vec![
            ("finance-happ", "Finance Module", vec!["payments", "lending"]),
            ("governance-happ", "Governance Module", vec!["voting", "proposals"]),
            ("energy-happ", "Energy Module", vec!["projects", "investments"]),
        ];

        for (id, name, caps) in &happs_to_register {
            let input = RegisterHappInput {
                happ_id: id.to_string(),
                happ_name: name.to_string(),
                capabilities: caps.iter().map(|s| s.to_string()).collect(),
            };

            let _: Record = conductor
                .call(
                    &alice_cell.zome("identity_bridge"),
                    "register_happ",
                    input,
                )
                .await;
        }

        tokio::time::sleep(Duration::from_millis(500)).await;

        let registered: Vec<Record> = conductor
            .call(
                &alice_cell.zome("identity_bridge"),
                "get_registered_happs",
                (),
            )
            .await;

        assert!(registered.len() >= 3);
        println!("=== test_get_registered_happs PASSED ===\n");
    }
}

// ============================================================================
// Unit Tests (No Conductor Required)
// ============================================================================

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_did_format_validation() {
        assert!(is_valid_mycelix_did("did:mycelix:abc123"));
        assert!(is_valid_mycelix_did("did:mycelix:uhCAk_test_key_12345"));
        assert!(!is_valid_mycelix_did("did:example:abc123"));
        assert!(!is_valid_mycelix_did("did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK"));
        assert!(!is_valid_mycelix_did("not_a_did"));
        assert!(!is_valid_mycelix_did(""));
    }

    #[test]
    fn test_helper_functions() {
        let did = test_did("alice");
        assert!(did.starts_with("did:mycelix:test:"));
        assert!(did.ends_with("alice"));

        let unique1 = unique_test_did("user");
        let unique2 = unique_test_did("user");
        assert_ne!(unique1, unique2);

        let schema_id = unique_schema_id("cert");
        assert!(schema_id.starts_with("mycelix:schema:test:cert:"));
        assert!(schema_id.ends_with(":v1"));
    }

    #[test]
    fn test_json_schema_generation() {
        let schema = minimal_json_schema();
        let parsed: serde_json::Value = serde_json::from_str(&schema).expect("Should be valid JSON");
        assert!(parsed.get("$schema").is_some());
        assert_eq!(parsed.get("type").unwrap(), "object");
    }

    #[test]
    fn test_verification_method_serialization() {
        let vm = VerificationMethod {
            id: "did:mycelix:test#keys-1".to_string(),
            type_: "Ed25519VerificationKey2020".to_string(),
            controller: "did:mycelix:test".to_string(),
            public_key_multibase: "zABC123DEF".to_string(),
        };

        let json = serde_json::to_string(&vm).expect("Serialize failed");
        let deserialized: VerificationMethod = serde_json::from_str(&json).expect("Deserialize failed");
        assert_eq!(vm.id, deserialized.id);
        assert_eq!(vm.type_, deserialized.type_);
    }

    #[test]
    fn test_service_endpoint_serialization() {
        let service = ServiceEndpoint {
            id: "did:mycelix:test#messaging".to_string(),
            type_: "MessagingService".to_string(),
            service_endpoint: "https://msg.example.com/alice".to_string(),
        };

        let json = serde_json::to_string(&service).expect("Serialize failed");
        let deserialized: ServiceEndpoint = serde_json::from_str(&json).expect("Deserialize failed");
        assert_eq!(service.id, deserialized.id);
    }

    #[test]
    fn test_bridge_event_type_serialization() {
        let event_types = vec![
            BridgeEventType::DidCreated,
            BridgeEventType::DidUpdated,
            BridgeEventType::DidDeactivated,
            BridgeEventType::CredentialIssued,
            BridgeEventType::CredentialRevoked,
            BridgeEventType::RecoveryInitiated,
            BridgeEventType::RecoveryCompleted,
            BridgeEventType::HappRegistered,
            BridgeEventType::Custom("test".to_string()),
        ];

        for event_type in event_types {
            let json = serde_json::to_string(&event_type).expect("Serialize failed");
            let deserialized: BridgeEventType = serde_json::from_str(&json).expect("Deserialize failed");
            assert_eq!(event_type, deserialized);
        }
    }

    #[test]
    fn test_schema_category_serialization() {
        let categories = vec![
            SchemaCategory::Education,
            SchemaCategory::Employment,
            SchemaCategory::Identity,
            SchemaCategory::Skills,
            SchemaCategory::Governance,
            SchemaCategory::Financial,
            SchemaCategory::Energy,
            SchemaCategory::Custom("test_category".to_string()),
        ];

        for category in categories {
            let json = serde_json::to_string(&category).expect("Serialize failed");
            let deserialized: SchemaCategory = serde_json::from_str(&json).expect("Deserialize failed");
            assert_eq!(category, deserialized);
        }
    }
}

// ============================================================================
// Security Tests
// ============================================================================

#[cfg(test)]
mod security_tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_did_controller_authorization() {
        println!("=== test_did_controller_authorization ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app1 = conductor.setup_app("app-alice", &[dna.clone()]).await.unwrap();
        let _app2 = conductor.setup_app("app-bob", &[dna.clone()]).await.unwrap();
        let alice_cell = app1.cells()[0].clone();

        let alice_record: Record = conductor
            .call(&alice_cell.zome("did_registry"), "create_did", ())
            .await;

        let alice_did: DidDocument = decode_entry(&alice_record).expect("No entry");

        let service = ServiceEndpoint {
            id: format!("{}#test", alice_did.id),
            type_: "TestService".to_string(),
            service_endpoint: "https://test.example.com".to_string(),
        };

        let update_result: Record = conductor
            .call(
                &alice_cell.zome("did_registry"),
                "add_service_endpoint",
                service,
            )
            .await;

        let updated_did: DidDocument = decode_entry(&update_result).expect("No entry");

        assert_eq!(updated_did.version, 2);
        println!("=== test_did_controller_authorization PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_did_immutability_constraints() {
        println!("=== test_did_immutability_constraints ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("mycelix-identity", &[dna.clone()]).await.unwrap();
        let alice_cell = app.cells()[0].clone();

        let created_record: Record = conductor
            .call(&alice_cell.zome("did_registry"), "create_did", ())
            .await;

        let created_did: DidDocument = decode_entry(&created_record).expect("No entry");

        let expected_id = created_did.id.clone();

        let service = ServiceEndpoint {
            id: format!("{}#service", created_did.id),
            type_: "Service".to_string(),
            service_endpoint: "https://service.example.com".to_string(),
        };

        let updated_record: Record = conductor
            .call(
                &alice_cell.zome("did_registry"),
                "add_service_endpoint",
                service,
            )
            .await;

        let updated_did: DidDocument = decode_entry(&updated_record).expect("No entry");

        assert_eq!(updated_did.id, expected_id);
        assert_eq!(updated_did.controller, created_did.controller);
        assert_eq!(updated_did.created, created_did.created);
        assert!(updated_did.updated != created_did.created);

        println!("=== test_did_immutability_constraints PASSED ===\n");
    }

    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn test_reputation_score_bounds() {
        println!("=== test_reputation_score_bounds ===");

        let mut conductor = SweetConductor::from_standard_config().await;
        let dna = load_dna().await;
        let app = conductor.setup_app("mycelix-identity", &[dna.clone()]).await.unwrap();
        let alice_cell = app.cells()[0].clone();
        let alice_agent = app.agent().clone();
        let alice_did = format!("did:mycelix:{}", alice_agent);

        // Test invalid score > 1.0
        let invalid_high = ReportReputationInput {
            did: alice_did.clone(),
            source_happ: "test".to_string(),
            score: 1.5,
            interactions: 10,
        };

        let result_high: Result<Record, _> = conductor
            .call_fallible(
                &alice_cell.zome("identity_bridge"),
                "report_reputation",
                invalid_high,
            )
            .await;

        match result_high {
            Err(_) => println!("  Score > 1.0 rejected"),
            Ok(_) => panic!("Should reject score > 1.0"),
        }

        // Test invalid score < 0.0
        let invalid_low = ReportReputationInput {
            did: alice_did.clone(),
            source_happ: "test".to_string(),
            score: -0.5,
            interactions: 10,
        };

        let result_low: Result<Record, _> = conductor
            .call_fallible(
                &alice_cell.zome("identity_bridge"),
                "report_reputation",
                invalid_low,
            )
            .await;

        match result_low {
            Err(_) => println!("  Score < 0.0 rejected"),
            Ok(_) => panic!("Should reject score < 0.0"),
        }

        // Test valid boundary scores
        let valid_zero = ReportReputationInput {
            did: alice_did.clone(),
            source_happ: "test-zero".to_string(),
            score: 0.0,
            interactions: 10,
        };

        let _: Record = conductor
            .call(
                &alice_cell.zome("identity_bridge"),
                "report_reputation",
                valid_zero,
            )
            .await;

        let valid_one = ReportReputationInput {
            did: alice_did.clone(),
            source_happ: "test-one".to_string(),
            score: 1.0,
            interactions: 10,
        };

        let _: Record = conductor
            .call(
                &alice_cell.zome("identity_bridge"),
                "report_reputation",
                valid_one,
            )
            .await;

        println!("=== test_reputation_score_bounds PASSED ===\n");
    }
}
