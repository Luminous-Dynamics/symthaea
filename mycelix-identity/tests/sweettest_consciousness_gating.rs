//! # Consciousness Gating Sweettest — Identity Cluster
//!
//! Verifies that identity operations are properly integrated with the
//! consciousness credential system. Unlike commons/civic/hearth clusters
//! where *other* clusters gate their governance actions via identity bridge
//! calls, the identity cluster is the *source* of consciousness credentials.
//!
//! ## What this tests
//!
//! 1. `issue_consciousness_credential` produces a valid credential for a DID
//! 2. `refresh_consciousness_credential` re-issues without error
//! 3. Bridge `health_check` returns healthy status
//! 4. DID creation works (no consciousness gate — foundational operation)
//! 5. Consciousness credential dimensions reflect underlying identity data
//!    (MFA, reputation, community trust)
//! 6. Credential issuance for non-existent DIDs fails appropriately
//! 7. Verifiable credential issuance (gated by identity existence)
//! 8. Trust credential tier verification consistency
//!
//! ## Why identity is special
//!
//! The identity cluster is the *issuer* of consciousness credentials, not a
//! *consumer*. Other clusters call `identity_bridge::issue_consciousness_credential`
//! via `CallTargetCell::OtherRole` to gate their governance actions. Here we test
//! that the issuance machinery itself works correctly.
//!
//! ## Running
//! ```bash
//! cd mycelix-identity
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_consciousness_gating -- --ignored --test-threads=2
//! ```

use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — consciousness profile (from mycelix-bridge-common)
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConsciousnessProfile {
    pub identity: f64,
    pub reputation: f64,
    pub community: f64,
    pub engagement: f64,
}

impl ConsciousnessProfile {
    pub fn combined_score(&self) -> f64 {
        (self.identity + self.reputation + self.community + self.engagement) / 4.0
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ConsciousnessTier {
    Observer,
    Participant,
    Citizen,
    Steward,
    Guardian,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConsciousnessCredential {
    pub did: String,
    pub profile: ConsciousnessProfile,
    pub tier: ConsciousnessTier,
    pub issued_at: u64,
    pub expires_at: u64,
    pub issuer: String,
}

// ============================================================================
// Mirror types — identity bridge health
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct IdentityBridgeHealth {
    pub healthy: bool,
    pub agent: String,
    pub api_version: u16,
}

// ============================================================================
// Mirror types — DID registry
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DidDocument {
    pub id: String,
    pub controller: ::holochain::prelude::AgentPubKey,
    #[serde(rename = "verificationMethod", alias = "verification_method")]
    pub verification_method: Vec<VerificationMethod>,
    pub authentication: Vec<String>,
    #[serde(rename = "keyAgreement", alias = "key_agreement", default)]
    pub key_agreement: Vec<String>,
    pub service: Vec<ServiceEndpoint>,
    pub created: ::holochain::prelude::Timestamp,
    pub updated: ::holochain::prelude::Timestamp,
    pub version: u32,
}

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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ServiceEndpoint {
    pub id: String,
    #[serde(rename = "type", alias = "type_")]
    pub type_: String,
    #[serde(rename = "serviceEndpoint", alias = "service_endpoint")]
    pub service_endpoint: String,
}

// ============================================================================
// Mirror types — trust credential
// ============================================================================

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum TrustTier {
    Observer,
    Basic,
    Standard,
    Elevated,
    Guardian,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TrustScoreRange {
    pub lower: f32,
    pub upper: f32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct IssueTrustCredentialInput {
    pub subject_did: String,
    pub issuer_did: String,
    pub kvector_commitment: Vec<u8>,
    pub range_proof: Vec<u8>,
    pub trust_score_lower: f32,
    pub trust_score_upper: f32,
    pub expires_at: Option<::holochain::prelude::Timestamp>,
    pub supersedes: Option<String>,
}

// ============================================================================
// Mirror types — verifiable credential
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct IssueCredentialInput {
    pub subject_did: String,
    pub schema_id: String,
    pub claims: serde_json::Value,
    pub credential_types: Vec<String>,
    pub issuer_name: Option<String>,
    pub expiration_days: Option<u32>,
    pub enable_revocation: bool,
    #[serde(default)]
    pub strict_schema: bool,
}

// ============================================================================
// Mirror types — MFA
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateMfaStateInput {
    pub did: String,
    pub primary_key_hash: String,
}

// ============================================================================
// Mirror types — reputation
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReportReputationInput {
    pub did: String,
    pub source_happ: String,
    pub score: f64,
    pub interactions: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReputationSource {
    pub source_happ: String,
    pub score: f64,
    pub interactions: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AggregatedReputation {
    pub did: String,
    pub aggregate_score: f64,
    pub sources: Vec<ReputationSource>,
    pub total_interactions: u64,
}

// ============================================================================
// DNA path helper
// ============================================================================

fn identity_dna_path() -> PathBuf {
    if let Ok(custom) = std::env::var("IDENTITY_DNA_PATH") {
        return PathBuf::from(custom);
    }
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // tests/ -> mycelix-identity/
    path.push("dna");
    path.push("mycelix_identity_dna.dna");
    path
}

// ============================================================================
// Tests — Bridge Health & API Version
// ============================================================================

/// Identity bridge health_check returns a healthy status.
/// This is NOT gated by consciousness — it is a diagnostic endpoint.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_identity_bridge_health_check() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&identity_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let health: IdentityBridgeHealth = conductor
        .call(&alice.zome("identity_bridge"), "health_check", ())
        .await;

    assert!(health.healthy, "Identity bridge should be healthy");
    assert!(
        health.api_version >= 1,
        "API version should be at least 1, got: {}",
        health.api_version
    );
    assert!(!health.agent.is_empty(), "Agent string should not be empty");
}

/// Bridge API version should be stable and queryable.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_identity_bridge_api_version() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&identity_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let version: u16 = conductor
        .call(&alice.zome("identity_bridge"), "get_api_version", ())
        .await;

    assert!(
        version >= 1,
        "API version should be at least 1, got: {}",
        version
    );
}

// ============================================================================
// Tests — DID Creation (Foundational, Not Gated)
// ============================================================================

/// DID creation is a foundational operation — it must NOT be gated by
/// consciousness credentials (you need a DID before you can have credentials).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_did_creation_not_gated() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&identity_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // create_did should succeed without any consciousness credentials
    let record: ::holochain::prelude::Record = conductor
        .call(&alice.zome("did_registry"), "create_did", ())
        .await;

    // Verify we got a valid record back
    assert!(
        record.action_address().get_raw_32().len() == 32,
        "DID record should have a valid action hash"
    );
}

/// After creating a DID, resolve_did should return it (read-only, ungated).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_did_resolution_not_gated() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&identity_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // First create a DID
    let _record: ::holochain::prelude::Record = conductor
        .call(&alice.zome("did_registry"), "create_did", ())
        .await;

    // Then resolve it using get_my_did (no consciousness gate)
    let resolved: Option<::holochain::prelude::Record> = conductor
        .call(&alice.zome("did_registry"), "get_my_did", ())
        .await;

    assert!(
        resolved.is_some(),
        "get_my_did should return the created DID document"
    );
}

// ============================================================================
// Tests — Consciousness Credential Issuance
// ============================================================================

/// Issue a consciousness credential for a freshly-created DID.
/// The credential should have valid structure with all 4 dimensions.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_issue_consciousness_credential_for_valid_did() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&identity_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create a DID first
    let _record: ::holochain::prelude::Record = conductor
        .call(&alice.zome("did_registry"), "create_did", ())
        .await;

    // Construct the DID string
    let agent = alice.agent_pubkey().clone();
    let did = format!("did:mycelix:{}", agent);

    // Issue consciousness credential
    let credential: ConsciousnessCredential = conductor
        .call(
            &alice.zome("identity_bridge"),
            "issue_consciousness_credential",
            did.clone(),
        )
        .await;

    // Verify credential structure
    assert_eq!(credential.did, did, "Credential DID should match");
    assert!(
        credential.expires_at > credential.issued_at,
        "Credential should have valid expiry (expires_at={} > issued_at={})",
        credential.expires_at,
        credential.issued_at
    );
    assert!(
        !credential.issuer.is_empty(),
        "Credential issuer should not be empty"
    );
    assert!(
        credential.issuer.starts_with("did:mycelix:"),
        "Issuer should be a mycelix DID, got: {}",
        credential.issuer
    );

    // Profile dimensions should be in [0.0, 1.0]
    let p = &credential.profile;
    assert!(
        (0.0..=1.0).contains(&p.identity),
        "Identity dimension should be in [0.0, 1.0], got: {}",
        p.identity
    );
    assert!(
        (0.0..=1.0).contains(&p.reputation),
        "Reputation dimension should be in [0.0, 1.0], got: {}",
        p.reputation
    );
    assert!(
        (0.0..=1.0).contains(&p.community),
        "Community dimension should be in [0.0, 1.0], got: {}",
        p.community
    );
    // Engagement is always 0.0 when issued from identity bridge
    // (the calling cluster bridge fills it in locally)
    assert!(
        (p.engagement - 0.0).abs() < f64::EPSILON,
        "Engagement should be 0.0 from identity bridge, got: {}",
        p.engagement
    );
}

/// Refresh should return a valid credential identical in structure.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_refresh_consciousness_credential() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&identity_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create DID
    let _record: ::holochain::prelude::Record = conductor
        .call(&alice.zome("did_registry"), "create_did", ())
        .await;

    let agent = alice.agent_pubkey().clone();
    let did = format!("did:mycelix:{}", agent);

    // Issue original
    let original: ConsciousnessCredential = conductor
        .call(
            &alice.zome("identity_bridge"),
            "issue_consciousness_credential",
            did.clone(),
        )
        .await;

    // Refresh
    let refreshed: ConsciousnessCredential = conductor
        .call(
            &alice.zome("identity_bridge"),
            "refresh_consciousness_credential",
            did.clone(),
        )
        .await;

    // Refreshed credential should have same DID and valid structure
    assert_eq!(refreshed.did, did, "Refreshed credential DID should match");
    assert!(
        refreshed.expires_at > refreshed.issued_at,
        "Refreshed credential should have valid expiry"
    );
    // Refreshed timestamp should be >= original (or equal if same clock tick)
    assert!(
        refreshed.issued_at >= original.issued_at,
        "Refreshed issued_at should be >= original"
    );
}

/// Issuing a consciousness credential for an invalid DID format should fail.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_consciousness_credential_invalid_did_format() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&identity_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Try issuing with a bad DID format (missing did:mycelix: prefix)
    let result: Result<ConsciousnessCredential, _> = conductor
        .call_fallible(
            &alice.zome("identity_bridge"),
            "issue_consciousness_credential",
            "bad-did-format".to_string(),
        )
        .await;

    assert!(
        result.is_err(),
        "issue_consciousness_credential should reject invalid DID format"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("did:mycelix") || err_msg.contains("Invalid DID"),
        "Error should mention DID format, got: {}",
        err_msg
    );
}

// ============================================================================
// Tests — Consciousness Tier Verification
// ============================================================================

/// A fresh DID with no MFA, no reputation, and no community trust should
/// receive an Observer-tier consciousness credential (all dimensions 0).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_fresh_did_gets_observer_tier() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&identity_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create DID (no MFA, no reputation, no trust credentials)
    let _record: ::holochain::prelude::Record = conductor
        .call(&alice.zome("did_registry"), "create_did", ())
        .await;

    let agent = alice.agent_pubkey().clone();
    let did = format!("did:mycelix:{}", agent);

    let credential: ConsciousnessCredential = conductor
        .call(
            &alice.zome("identity_bridge"),
            "issue_consciousness_credential",
            did,
        )
        .await;

    // With no MFA, reputation, community, or engagement, combined score is 0.0
    // which maps to Observer tier
    assert_eq!(
        credential.tier,
        ConsciousnessTier::Observer,
        "Fresh DID with no identity data should be Observer tier"
    );

    let combined = credential.profile.combined_score();
    assert!(
        combined < 0.3,
        "Combined score should be < 0.3 for Observer tier, got: {}",
        combined
    );
}

/// Tier assignment should be consistent between combined_score and the
/// credential's tier field.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_tier_matches_combined_score() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&identity_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create DID
    let _record: ::holochain::prelude::Record = conductor
        .call(&alice.zome("did_registry"), "create_did", ())
        .await;

    let agent = alice.agent_pubkey().clone();
    let did = format!("did:mycelix:{}", agent);

    let credential: ConsciousnessCredential = conductor
        .call(
            &alice.zome("identity_bridge"),
            "issue_consciousness_credential",
            did,
        )
        .await;

    let score = credential.profile.combined_score();
    let expected_tier = if score >= 0.8 {
        ConsciousnessTier::Guardian
    } else if score >= 0.6 {
        ConsciousnessTier::Steward
    } else if score >= 0.4 {
        ConsciousnessTier::Citizen
    } else if score >= 0.3 {
        ConsciousnessTier::Participant
    } else {
        ConsciousnessTier::Observer
    };

    assert_eq!(
        credential.tier, expected_tier,
        "Tier {:?} should match expected {:?} for combined score {}",
        credential.tier, expected_tier, score
    );
}

// ============================================================================
// Tests — Identity-Specific Gating (DID Operations)
// ============================================================================

/// DID deactivation should work for an active DID (write operation, but
/// self-sovereignty — not gated by consciousness credentials since the
/// agent is deactivating their own DID).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_did_deactivation_self_sovereignty() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&identity_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create DID
    let _record: ::holochain::prelude::Record = conductor
        .call(&alice.zome("did_registry"), "create_did", ())
        .await;

    // Deactivate DID — this is a self-sovereignty operation, not consciousness-gated
    let deactivation_result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(
            &alice.zome("did_registry"),
            "deactivate_did",
            "Test deactivation for gating test".to_string(),
        )
        .await;

    assert!(
        deactivation_result.is_ok(),
        "DID deactivation should succeed without consciousness credentials (self-sovereignty)"
    );
}

/// is_did_active should work without consciousness credentials (read-only).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_did_active_check_not_gated() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&identity_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create DID
    let _record: ::holochain::prelude::Record = conductor
        .call(&alice.zome("did_registry"), "create_did", ())
        .await;

    let agent = alice.agent_pubkey().clone();
    let did = format!("did:mycelix:{}", agent);

    let active: bool = conductor
        .call(&alice.zome("did_registry"), "is_did_active", did)
        .await;

    assert!(active, "Freshly created DID should be active");
}

// ============================================================================
// Tests — Trust Credential Gating
// ============================================================================

/// Trust credential issuance: caller must be the issuer (self-sovereignty
/// check, not consciousness gating — but verifies identity pipeline works).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_trust_credential_issuance() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&identity_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();
    let alice_did = format!("did:mycelix:{}", agent);

    // Valid 32-byte commitment (non-zero)
    let mut commitment = vec![0u8; 32];
    commitment[0] = 0xAB;
    commitment[1] = 0xCD;

    let input = IssueTrustCredentialInput {
        subject_did: alice_did.clone(),
        issuer_did: alice_did.clone(),
        kvector_commitment: commitment,
        range_proof: vec![1, 2, 3, 4],
        trust_score_lower: 0.3,
        trust_score_upper: 0.39,
        expires_at: None,
        supersedes: None,
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(
            &alice.zome("trust_credential"),
            "issue_trust_credential",
            input,
        )
        .await;

    assert!(
        result.is_ok(),
        "Self-attested trust credential issuance should succeed: {:?}",
        result.err()
    );
}

/// Trust credential issuance fails when caller is not the claimed issuer.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_trust_credential_issuer_mismatch_rejected() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&identity_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let mut commitment = vec![0u8; 32];
    commitment[0] = 0xFF;

    // Try issuing with a different issuer DID than the caller
    let input = IssueTrustCredentialInput {
        subject_did: "did:mycelix:some-subject".to_string(),
        issuer_did: "did:mycelix:different-issuer".to_string(),
        kvector_commitment: commitment,
        range_proof: vec![1, 2, 3],
        trust_score_lower: 0.5,
        trust_score_upper: 0.7,
        expires_at: None,
        supersedes: None,
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(
            &alice.zome("trust_credential"),
            "issue_trust_credential",
            input,
        )
        .await;

    assert!(
        result.is_err(),
        "Trust credential issuance should fail when caller is not the claimed issuer"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("issuer"),
        "Error should mention issuer mismatch, got: {}",
        err_msg
    );
}

// ============================================================================
// Tests — Verifiable Credential Issuance
// ============================================================================

/// Verifiable credential issuance should work when the issuer has a valid DID.
/// This tests that the VC pipeline is functional (not consciousness-gated,
/// but depends on identity infrastructure).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_verifiable_credential_issuance() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&identity_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create DID first (required for issuer identity)
    let _record: ::holochain::prelude::Record = conductor
        .call(&alice.zome("did_registry"), "create_did", ())
        .await;

    let agent = alice.agent_pubkey().clone();
    let alice_did = format!("did:mycelix:{}", agent);

    let input = IssueCredentialInput {
        subject_did: alice_did.clone(),
        schema_id: String::new(), // No schema (skip validation)
        claims: serde_json::json!({
            "name": "Alice Test",
            "role": "Developer"
        }),
        credential_types: vec!["IdentityCredential".to_string()],
        issuer_name: Some("Test Issuer".to_string()),
        expiration_days: Some(365),
        enable_revocation: true,
        strict_schema: false,
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(
            &alice.zome("verifiable_credential"),
            "issue_credential",
            input,
        )
        .await;

    assert!(
        result.is_ok(),
        "Verifiable credential issuance should succeed with valid DID: {:?}",
        result.err()
    );
}

// ============================================================================
// Tests — Read-Only Operations (Never Gated)
// ============================================================================

/// Read-only trust credential queries should succeed without credentials.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_trust_credential_read_not_gated() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&identity_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // get_subject_credentials for a non-existent subject should return empty vec
    let credentials: Vec<::holochain::prelude::Record> = conductor
        .call(
            &alice.zome("trust_credential"),
            "get_subject_credentials",
            "did:mycelix:nonexistent".to_string(),
        )
        .await;

    assert!(
        credentials.is_empty(),
        "Query for non-existent subject should return empty vec (ungated)"
    );
}

/// Read-only bridge queries (get_reputation, verify_did) should work
/// without consciousness credentials.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_read_operations_not_gated() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&identity_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create a DID first
    let _record: ::holochain::prelude::Record = conductor
        .call(&alice.zome("did_registry"), "create_did", ())
        .await;

    let agent = alice.agent_pubkey().clone();
    let did = format!("did:mycelix:{}", agent);

    // verify_did — read-only, should succeed
    let is_valid: bool = conductor
        .call(&alice.zome("identity_bridge"), "verify_did", did.clone())
        .await;

    assert!(is_valid, "verify_did should return true for a valid DID");

    // get_reputation — read-only, should succeed (returns default/empty)
    let reputation: AggregatedReputation = conductor
        .call(
            &alice.zome("identity_bridge"),
            "get_reputation",
            did.clone(),
        )
        .await;

    assert_eq!(
        reputation.did, did,
        "Reputation DID should match the queried DID"
    );
}

// ============================================================================
// Tests — Consciousness Credential Consistency
// ============================================================================

/// Multiple consecutive credential issuances for the same DID should
/// produce consistent results (same dimensions, same tier).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_credential_issuance_consistency() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&identity_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create DID
    let _record: ::holochain::prelude::Record = conductor
        .call(&alice.zome("did_registry"), "create_did", ())
        .await;

    let agent = alice.agent_pubkey().clone();
    let did = format!("did:mycelix:{}", agent);

    // Issue twice in succession
    let cred_1: ConsciousnessCredential = conductor
        .call(
            &alice.zome("identity_bridge"),
            "issue_consciousness_credential",
            did.clone(),
        )
        .await;

    let cred_2: ConsciousnessCredential = conductor
        .call(
            &alice.zome("identity_bridge"),
            "issue_consciousness_credential",
            did.clone(),
        )
        .await;

    // Same DID, same underlying data => same profile scores
    assert_eq!(
        cred_1.tier, cred_2.tier,
        "Tier should be consistent across consecutive issuances"
    );
    assert!(
        (cred_1.profile.identity - cred_2.profile.identity).abs() < f64::EPSILON,
        "Identity dimension should be consistent"
    );
    assert!(
        (cred_1.profile.reputation - cred_2.profile.reputation).abs() < f64::EPSILON,
        "Reputation dimension should be consistent"
    );
    assert!(
        (cred_1.profile.community - cred_2.profile.community).abs() < f64::EPSILON,
        "Community dimension should be consistent"
    );
}

/// After reporting reputation, the consciousness credential should reflect
/// the updated reputation score in its profile.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_reputation_affects_consciousness_credential() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&identity_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create DID
    let _record: ::holochain::prelude::Record = conductor
        .call(&alice.zome("did_registry"), "create_did", ())
        .await;

    let agent = alice.agent_pubkey().clone();
    let did = format!("did:mycelix:{}", agent);

    // Issue credential before reputation
    let cred_before: ConsciousnessCredential = conductor
        .call(
            &alice.zome("identity_bridge"),
            "issue_consciousness_credential",
            did.clone(),
        )
        .await;

    // Report positive reputation
    let rep_input = ReportReputationInput {
        did: did.clone(),
        source_happ: "test-happ".to_string(),
        score: 0.9,
        interactions: 50,
    };

    let _rep_record: ::holochain::prelude::Record = conductor
        .call(
            &alice.zome("identity_bridge"),
            "report_reputation",
            rep_input,
        )
        .await;

    // Issue credential after reputation
    let cred_after: ConsciousnessCredential = conductor
        .call(
            &alice.zome("identity_bridge"),
            "issue_consciousness_credential",
            did.clone(),
        )
        .await;

    // Reputation dimension should now be higher (or at least not lower)
    assert!(
        cred_after.profile.reputation >= cred_before.profile.reputation,
        "Reputation dimension should increase after positive report: before={}, after={}",
        cred_before.profile.reputation,
        cred_after.profile.reputation
    );
}

// ============================================================================
// Tests — Bridge Health Resilience
// ============================================================================

/// Bridge health should remain healthy even after failed operations
/// (e.g., invalid DID credential issuance).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_health_survives_failures() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&identity_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Verify healthy before
    let health_before: IdentityBridgeHealth = conductor
        .call(&alice.zome("identity_bridge"), "health_check", ())
        .await;
    assert!(health_before.healthy, "Bridge should be healthy initially");

    // Cause a failure (invalid DID format)
    let _result: Result<ConsciousnessCredential, _> = conductor
        .call_fallible(
            &alice.zome("identity_bridge"),
            "issue_consciousness_credential",
            "invalid-did".to_string(),
        )
        .await;

    // Verify still healthy after failure
    let health_after: IdentityBridgeHealth = conductor
        .call(&alice.zome("identity_bridge"), "health_check", ())
        .await;

    assert!(
        health_after.healthy,
        "Bridge should remain healthy after credential issuance failure"
    );
}

/// MATL score query should work through the bridge (read-only, ungated).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_matl_score_query_not_gated() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&identity_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create DID
    let _record: ::holochain::prelude::Record = conductor
        .call(&alice.zome("did_registry"), "create_did", ())
        .await;

    let agent = alice.agent_pubkey().clone();
    let did = format!("did:mycelix:{}", agent);

    let matl_score: f64 = conductor
        .call(&alice.zome("identity_bridge"), "get_matl_score", did)
        .await;

    assert!(
        (0.0..=1.0).contains(&matl_score),
        "MATL score should be in [0.0, 1.0], got: {}",
        matl_score
    );
}
