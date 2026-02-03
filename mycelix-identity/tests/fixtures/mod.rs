//! Test Fixtures for Mycelix Identity
//!
//! Common test utilities and helpers. Uses local mirror types to avoid
//! importing zome crates (which cause duplicate symbol linking errors).

use holochain::sweettest::*;
use holochain::prelude::*;
use std::path::PathBuf;

// ============================================================================
// Constants
// ============================================================================

pub const TEST_DID_PREFIX: &str = "did:mycelix:test:";
pub const INVALID_DID_PREFIX: &str = "did:invalid:";
pub const TEST_SCHEMA_PREFIX: &str = "mycelix:schema:test:";
pub const TEST_CREDENTIAL_PREFIX: &str = "credential:test:";
pub const TEST_HAPP_ID: &str = "test-happ";

// ============================================================================
// DID Helpers
// ============================================================================

pub fn test_did(suffix: &str) -> String {
    format!("{}{}", TEST_DID_PREFIX, suffix)
}

pub fn unique_test_did(prefix: &str) -> String {
    let timestamp = chrono::Utc::now().timestamp_micros();
    format!("{}{}:{}", TEST_DID_PREFIX, prefix, timestamp)
}

pub fn invalid_did(suffix: &str) -> String {
    format!("{}{}", INVALID_DID_PREFIX, suffix)
}

pub fn did_from_agent_key(agent_pub_key: &AgentPubKey) -> String {
    format!("did:mycelix:{}", agent_pub_key)
}

// ============================================================================
// Schema Helpers
// ============================================================================

pub fn test_schema_id(name: &str) -> String {
    format!("{}{}:v1", TEST_SCHEMA_PREFIX, name)
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
            "name": { "type": "string" }
        },
        "required": ["name"]
    })
    .to_string()
}

// ============================================================================
// Credential Helpers
// ============================================================================

pub fn test_credential_id(suffix: &str) -> String {
    format!("{}{}", TEST_CREDENTIAL_PREFIX, suffix)
}

pub fn unique_credential_id() -> String {
    let uuid = uuid::Uuid::new_v4();
    format!("{}{}", TEST_CREDENTIAL_PREFIX, uuid)
}

// ============================================================================
// Recovery Helpers
// ============================================================================

pub fn test_trustees(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| test_did(&format!("trustee_{}", i)))
        .collect()
}

// ============================================================================
// Conductor Helpers
// ============================================================================

/// Path to the pre-built DNA bundle
pub fn dna_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("dna")
        .join("mycelix_identity_dna.dna")
}

pub async fn setup_conductor() -> Result<SweetConductor, Box<dyn std::error::Error + Send + Sync>> {
    let conductor = SweetConductor::from_standard_config().await;
    Ok(conductor)
}

// ============================================================================
// Validation Helpers
// ============================================================================

pub fn is_valid_mycelix_did(did: &str) -> bool {
    did.starts_with("did:mycelix:") && did.len() > "did:mycelix:".len()
}

pub fn is_valid_schema_id(schema_id: &str) -> bool {
    schema_id.starts_with("mycelix:schema:")
}

pub fn is_valid_reputation_score(score: f64) -> bool {
    score >= 0.0 && score <= 1.0
}

// ============================================================================
// Timestamp Helpers
// ============================================================================

pub fn timestamp() -> i64 {
    chrono::Utc::now().timestamp()
}

pub fn current_timestamp() -> Timestamp {
    Timestamp::from_micros(chrono::Utc::now().timestamp_micros())
}

pub fn future_timestamp(days: i64) -> Timestamp {
    let future = chrono::Utc::now() + chrono::Duration::days(days);
    Timestamp::from_micros(future.timestamp_micros())
}

// ============================================================================
// Unit Tests for Fixtures
// ============================================================================

#[cfg(test)]
mod fixture_tests {
    use super::*;

    #[test]
    fn test_did_generation() {
        let did = test_did("alice");
        assert!(did.starts_with(TEST_DID_PREFIX));
        assert!(did.ends_with("alice"));
    }

    #[test]
    fn test_unique_did_generation() {
        let did1 = unique_test_did("user");
        let did2 = unique_test_did("user");
        assert_ne!(did1, did2);
    }

    #[test]
    fn test_invalid_did_generation() {
        let did = invalid_did("attacker");
        assert!(!did.starts_with("did:mycelix:"));
        assert!(did.starts_with("did:invalid:"));
    }

    #[test]
    fn test_schema_id_generation() {
        let schema_id = test_schema_id("education");
        assert!(schema_id.starts_with("mycelix:schema:test:education"));
        assert!(schema_id.ends_with(":v1"));
    }

    #[test]
    fn test_unique_schema_id_generation() {
        let id1 = unique_schema_id("cert");
        let id2 = unique_schema_id("cert");
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_minimal_json_schema_valid() {
        let schema = minimal_json_schema();
        let parsed: serde_json::Value = serde_json::from_str(&schema).expect("Should be valid JSON");
        assert!(parsed.get("$schema").is_some());
        assert!(parsed.get("type").is_some());
    }

    #[test]
    fn test_trustees_generation() {
        let trustees = test_trustees(5);
        assert_eq!(trustees.len(), 5);
        for trustee in &trustees {
            assert!(trustee.starts_with("did:mycelix:test:trustee_"));
        }
    }

    #[test]
    fn test_did_validation() {
        assert!(is_valid_mycelix_did("did:mycelix:abc123"));
        assert!(!is_valid_mycelix_did("did:example:abc"));
        assert!(!is_valid_mycelix_did("did:mycelix:"));
        assert!(!is_valid_mycelix_did("not-a-did"));
    }

    #[test]
    fn test_reputation_score_validation() {
        assert!(is_valid_reputation_score(0.0));
        assert!(is_valid_reputation_score(1.0));
        assert!(!is_valid_reputation_score(-0.1));
        assert!(!is_valid_reputation_score(1.1));
    }
}
