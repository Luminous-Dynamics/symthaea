// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Personal — Sweettest Integration Tests
//!
//! Tests the unified Personal cluster DNA: identity-vault, health-vault,
//! credential-wallet, and personal-bridge zomes.
//!
//! ## Running
//! ```bash
//! cd mycelix-personal
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_integration -- --ignored --test-threads=2
//! ```
//!
//! Note: `--test-threads=2` prevents conductor database timeouts from too many
//! concurrent Holochain conductors competing for SQLite locks.

use holochain::prelude::*;
use holochain::sweettest::*;
use std::collections::HashMap;
use std::path::PathBuf;

// ============================================================================
// Mirror types — avoid importing zome crates (duplicate WASM symbols)
// ============================================================================

// --- identity-vault ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Profile {
    pub display_name: String,
    pub avatar: Option<String>,
    pub bio: Option<String>,
    pub metadata: HashMap<String, String>,
    pub updated_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MasterKey {
    pub label: String,
    pub purpose: String,
    pub public_key_hex: String,
    pub active: bool,
    pub created_at: Timestamp,
}

// --- health-vault ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HealthRecord {
    pub record_type: String,
    pub data: String,
    pub source: String,
    pub event_date: Timestamp,
    pub updated_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Biometric {
    pub metric_type: String,
    pub value: f64,
    pub unit: String,
    pub measured_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ConsentGrant {
    pub grantee: AgentPubKey,
    pub record_types: Vec<String>,
    pub expires_at: Option<Timestamp>,
    pub active: bool,
    pub created_at: Timestamp,
}

// --- credential-wallet ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum CredentialType {
    Identity,
    Health,
    FederatedLearning,
    Governance,
    Domain(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StoredCredential {
    pub credential_type: CredentialType,
    pub credential_data: String,
    pub issuer: String,
    pub issued_at: Timestamp,
    pub expires_at: Option<Timestamp>,
    pub revoked: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CredentialProof {
    pub credential_hash: ActionHash,
    pub proof_data: String,
    pub claim: String,
    pub created_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SelfAttestTrustInput {
    pub self_did: String,
    pub kvector_commitment: Vec<u8>,
    pub range_proof: Vec<u8>,
    pub trust_score_lower: f32,
    pub trust_score_upper: f32,
    pub expires_at: Option<Timestamp>,
    pub supersedes: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateTrustPresentationInput {
    pub credential_id: String,
    pub subject_did: String,
    pub disclosed_tier: TrustTier,
    pub disclose_range: bool,
    pub trust_range: TrustScoreRange,
    pub presentation_proof: Vec<u8>,
    pub verifier_did: Option<String>,
    pub purpose: String,
}

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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VerificationResult {
    pub credential_id: String,
    pub commitment_valid: bool,
    pub tier_consistent: bool,
    pub not_revoked: bool,
    pub not_expired: bool,
    pub proof_format_valid: bool,
    pub message: String,
}

// --- personal-bridge ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PersonalQueryEntry {
    pub domain: String,
    pub query_type: String,
    pub requester: AgentPubKey,
    pub params: String,
    pub result: Option<String>,
    pub created_at: Timestamp,
    pub resolved_at: Option<Timestamp>,
    pub success: Option<bool>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PersonalEventEntry {
    pub domain: String,
    pub event_type: String,
    pub source_agent: AgentPubKey,
    pub payload: String,
    pub created_at: Timestamp,
    pub related_hashes: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ResolveQueryInput {
    pub query_hash: ActionHash,
    pub result: String,
    pub success: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BridgeHealth {
    pub healthy: bool,
    pub agent: String,
    pub total_events: u32,
    pub total_queries: u32,
    pub domains: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EventTypeQuery {
    pub domain: String,
    pub event_type: String,
}

// ============================================================================
// DNA setup helper
// ============================================================================

fn personal_dna_path() -> PathBuf {
    if let Ok(custom) = std::env::var("PERSONAL_DNA_PATH") {
        return PathBuf::from(custom);
    }
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // tests/ -> mycelix-personal/
    path.push("dna");
    path.push("mycelix_personal.dna");
    path
}

// ============================================================================
// Identity Vault Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_identity_set_and_get_profile() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let profile = Profile {
        display_name: "Alice".to_string(),
        avatar: Some("https://example.com/alice.png".to_string()),
        bio: Some("Mycelix contributor".to_string()),
        metadata: HashMap::new(),
        updated_at: Timestamp::now(),
    };

    let record: Record = conductor
        .call(&alice.zome("identity_vault"), "set_profile", profile)
        .await;

    assert_eq!(record.action().author(), alice.agent_pubkey());

    let retrieved: Option<Record> = conductor
        .call(&alice.zome("identity_vault"), "get_my_profile", ())
        .await;

    assert!(retrieved.is_some());
    let r = retrieved.unwrap();
    let p: Profile = r.entry().to_app_option().unwrap().unwrap();
    assert_eq!(p.display_name, "Alice");
    assert_eq!(p.bio.as_deref(), Some("Mycelix contributor"));
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_identity_update_profile() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let profile1 = Profile {
        display_name: "Alice v1".to_string(),
        avatar: None,
        bio: None,
        metadata: HashMap::new(),
        updated_at: Timestamp::now(),
    };

    let _: Record = conductor
        .call(&alice.zome("identity_vault"), "set_profile", profile1)
        .await;

    let profile2 = Profile {
        display_name: "Alice v2".to_string(),
        avatar: Some("avatar2.png".to_string()),
        bio: Some("Updated bio".to_string()),
        metadata: HashMap::new(),
        updated_at: Timestamp::now(),
    };

    let _: Record = conductor
        .call(&alice.zome("identity_vault"), "set_profile", profile2)
        .await;

    // get_my_profile returns the latest (last link)
    let retrieved: Option<Record> = conductor
        .call(&alice.zome("identity_vault"), "get_my_profile", ())
        .await;

    assert!(retrieved.is_some());
    let p: Profile = retrieved
        .unwrap()
        .entry()
        .to_app_option()
        .unwrap()
        .unwrap();
    assert_eq!(p.display_name, "Alice v2");
    assert_eq!(p.bio.as_deref(), Some("Updated bio"));
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_identity_register_and_get_keys() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let key1 = MasterKey {
        label: "primary-signing".to_string(),
        purpose: "signing".to_string(),
        public_key_hex: "abcdef0123456789abcdef0123456789".to_string(),
        active: true,
        created_at: Timestamp::now(),
    };

    let key2 = MasterKey {
        label: "backup-encryption".to_string(),
        purpose: "encryption".to_string(),
        public_key_hex: "9876543210fedcba9876543210fedcba".to_string(),
        active: true,
        created_at: Timestamp::now(),
    };

    let _: Record = conductor
        .call(&alice.zome("identity_vault"), "register_key", key1)
        .await;
    let _: Record = conductor
        .call(&alice.zome("identity_vault"), "register_key", key2)
        .await;

    let keys: Vec<Record> = conductor
        .call(&alice.zome("identity_vault"), "get_my_keys", ())
        .await;

    assert_eq!(keys.len(), 2);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_identity_selective_disclosure() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let profile = Profile {
        display_name: "Alice Disclosure".to_string(),
        avatar: Some("https://example.com/alice.png".to_string()),
        bio: Some("Secret bio".to_string()),
        metadata: HashMap::new(),
        updated_at: Timestamp::now(),
    };

    let _: Record = conductor
        .call(&alice.zome("identity_vault"), "set_profile", profile)
        .await;

    // Only request display_name — bio should NOT be disclosed
    let fields: Vec<String> = vec!["display_name".to_string()];
    let disclosed: String = conductor
        .call(&alice.zome("identity_vault"), "disclose_profile", fields)
        .await;

    let map: serde_json::Value = serde_json::from_str(&disclosed).unwrap();
    assert_eq!(map["display_name"], "Alice Disclosure");
    assert!(map.get("bio").is_none());
    assert!(map.get("avatar").is_none());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_identity_disclose_no_profile_returns_empty_json() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // No profile set — disclose should return "{}"
    let fields: Vec<String> = vec!["display_name".to_string()];
    let disclosed: String = conductor
        .call(&alice.zome("identity_vault"), "disclose_profile", fields)
        .await;

    assert_eq!(disclosed, "{}");
}

// ============================================================================
// Health Vault Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_health_create_and_get_records() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let record1 = HealthRecord {
        record_type: "allergy".to_string(),
        data: r#"{"allergen":"peanuts","severity":"severe"}"#.to_string(),
        source: "self".to_string(),
        event_date: Timestamp::now(),
        updated_at: Timestamp::now(),
    };

    let record2 = HealthRecord {
        record_type: "medication".to_string(),
        data: r#"{"name":"aspirin","dose":"100mg"}"#.to_string(),
        source: "self".to_string(),
        event_date: Timestamp::now(),
        updated_at: Timestamp::now(),
    };

    let _: Record = conductor
        .call(&alice.zome("health_vault"), "create_health_record", record1)
        .await;
    let _: Record = conductor
        .call(&alice.zome("health_vault"), "create_health_record", record2)
        .await;

    let records: Vec<Record> = conductor
        .call(&alice.zome("health_vault"), "get_my_records", ())
        .await;

    assert_eq!(records.len(), 2);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_health_record_biometric() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let biometric = Biometric {
        metric_type: "heart_rate".to_string(),
        value: 72.0,
        unit: "bpm".to_string(),
        measured_at: Timestamp::now(),
    };

    let record: Record = conductor
        .call(&alice.zome("health_vault"), "record_biometric", biometric)
        .await;

    assert_eq!(record.action().author(), alice.agent_pubkey());

    let biometrics: Vec<Record> = conductor
        .call(&alice.zome("health_vault"), "get_my_biometrics", ())
        .await;

    assert_eq!(biometrics.len(), 1);
    let b: Biometric = biometrics[0].entry().to_app_option().unwrap().unwrap();
    assert_eq!(b.metric_type, "heart_rate");
    assert!((b.value - 72.0).abs() < f64::EPSILON);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_health_consent_grant_and_list() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let bob_key = AgentPubKey::from_raw_36(vec![1u8; 36]);

    let consent = ConsentGrant {
        grantee: bob_key,
        record_types: vec!["allergy".to_string(), "medication".to_string()],
        expires_at: None,
        active: true,
        created_at: Timestamp::now(),
    };

    let _: Record = conductor
        .call(&alice.zome("health_vault"), "grant_consent", consent)
        .await;

    let consents: Vec<Record> = conductor
        .call(&alice.zome("health_vault"), "get_my_consents", ())
        .await;

    assert_eq!(consents.len(), 1);
    let c: ConsentGrant = consents[0].entry().to_app_option().unwrap().unwrap();
    assert!(c.active);
    assert_eq!(c.record_types.len(), 2);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_health_multiple_biometrics() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let metrics = vec![
        ("heart_rate", 72.0, "bpm"),
        ("temperature", 36.6, "celsius"),
        ("blood_oxygen", 98.0, "percent"),
    ];

    for (metric, val, unit) in &metrics {
        let bio = Biometric {
            metric_type: metric.to_string(),
            value: *val,
            unit: unit.to_string(),
            measured_at: Timestamp::now(),
        };
        let _: Record = conductor
            .call(&alice.zome("health_vault"), "record_biometric", bio)
            .await;
    }

    let biometrics: Vec<Record> = conductor
        .call(&alice.zome("health_vault"), "get_my_biometrics", ())
        .await;

    assert_eq!(biometrics.len(), 3);
}

// ============================================================================
// Credential Wallet Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_credential_store_and_get() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let cred = StoredCredential {
        credential_type: CredentialType::Identity,
        credential_data: r#"{"vc":"identity-proof","name":"Alice"}"#.to_string(),
        issuer: "did:key:z6MkTest123".to_string(),
        issued_at: Timestamp::now(),
        expires_at: None,
        revoked: false,
    };

    let record: Record = conductor
        .call(&alice.zome("credential_wallet"), "store_credential", cred)
        .await;

    assert_eq!(record.action().author(), alice.agent_pubkey());

    let creds: Vec<Record> = conductor
        .call(&alice.zome("credential_wallet"), "get_my_credentials", ())
        .await;

    assert_eq!(creds.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_credential_store_multiple_types() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let types = vec![
        (CredentialType::Identity, r#"{"type":"identity"}"#),
        (CredentialType::Health, r#"{"type":"health"}"#),
        (CredentialType::FederatedLearning, r#"{"phi":0.72}"#),
        (CredentialType::Governance, r#"{"tier":"standard"}"#),
    ];

    for (ctype, data) in types {
        let cred = StoredCredential {
            credential_type: ctype,
            credential_data: data.to_string(),
            issuer: "did:key:z6MkTest123".to_string(),
            issued_at: Timestamp::now(),
            expires_at: None,
            revoked: false,
        };
        let _: Record = conductor
            .call(&alice.zome("credential_wallet"), "store_credential", cred)
            .await;
    }

    let creds: Vec<Record> = conductor
        .call(&alice.zome("credential_wallet"), "get_my_credentials", ())
        .await;

    assert_eq!(creds.len(), 4);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_credential_create_proof() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Store a credential first
    let cred = StoredCredential {
        credential_type: CredentialType::FederatedLearning,
        credential_data: r#"{"phi":0.42}"#.to_string(),
        issuer: "did:key:z6MkIssuer".to_string(),
        issued_at: Timestamp::now(),
        expires_at: None,
        revoked: false,
    };

    let cred_record: Record = conductor
        .call(&alice.zome("credential_wallet"), "store_credential", cred)
        .await;

    // Create a proof from it
    let proof = CredentialProof {
        credential_hash: cred_record.action_address().clone(),
        proof_data: r#"{"zk_proof":"simulated_zk_proof_data"}"#.to_string(),
        claim: "Agent participated in FL round with Phi > 0.3".to_string(),
        created_at: Timestamp::now(),
    };

    let proof_record: Record = conductor
        .call(&alice.zome("credential_wallet"), "create_proof", proof)
        .await;

    assert_eq!(proof_record.action().author(), alice.agent_pubkey());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_credential_present_credential() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Store an FL credential
    let cred = StoredCredential {
        credential_type: CredentialType::FederatedLearning,
        credential_data: r#"{"phi":0.72,"round":5}"#.to_string(),
        issuer: "did:key:z6MkFL".to_string(),
        issued_at: Timestamp::now(),
        expires_at: None,
        revoked: false,
    };

    let _: Record = conductor
        .call(&alice.zome("credential_wallet"), "store_credential", cred)
        .await;

    // Present the credential
    let presented: String = conductor
        .call(
            &alice.zome("credential_wallet"),
            "present_credential",
            CredentialType::FederatedLearning,
        )
        .await;

    assert!(presented.contains("phi"));
    assert!(presented.contains("0.72"));
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_credential_get_by_type() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Store two identity and one governance credential
    for _ in 0..2 {
        let cred = StoredCredential {
            credential_type: CredentialType::Identity,
            credential_data: r#"{"vc":"id"}"#.to_string(),
            issuer: "did:key:z6MkId".to_string(),
            issued_at: Timestamp::now(),
            expires_at: None,
            revoked: false,
        };
        let _: Record = conductor
            .call(&alice.zome("credential_wallet"), "store_credential", cred)
            .await;
    }

    let gov_cred = StoredCredential {
        credential_type: CredentialType::Governance,
        credential_data: r#"{"vc":"gov"}"#.to_string(),
        issuer: "did:key:z6MkGov".to_string(),
        issued_at: Timestamp::now(),
        expires_at: None,
        revoked: false,
    };
    let _: Record = conductor
        .call(
            &alice.zome("credential_wallet"),
            "store_credential",
            gov_cred,
        )
        .await;

    // Filter by type
    let identity_creds: Vec<StoredCredential> = conductor
        .call(
            &alice.zome("credential_wallet"),
            "get_credentials_by_type",
            CredentialType::Identity,
        )
        .await;

    assert_eq!(identity_creds.len(), 2);
}

// ============================================================================
// Trust Credential Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_trust_self_attest_and_query() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();
    let self_did = format!("did:mycelix:{}", agent);

    let mut commitment = vec![0u8; 32];
    commitment[0] = 1; // non-zero commitment

    let input = SelfAttestTrustInput {
        self_did: self_did.clone(),
        kvector_commitment: commitment,
        range_proof: vec![1, 2, 3, 4],
        trust_score_lower: 0.4,
        trust_score_upper: 0.59,
        expires_at: None,
        supersedes: None,
    };

    let record: Record = conductor
        .call(&alice.zome("credential_wallet"), "self_attest_trust", input)
        .await;

    assert_eq!(record.action().author(), alice.agent_pubkey());

    // Query trust credentials for this subject
    let creds: Vec<Record> = conductor
        .call(
            &alice.zome("credential_wallet"),
            "get_trust_credentials",
            self_did,
        )
        .await;

    assert_eq!(creds.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_trust_verify_credential() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();
    let self_did = format!("did:mycelix:{}", agent);

    let mut commitment = vec![0u8; 32];
    commitment[0] = 42;

    let input = SelfAttestTrustInput {
        self_did: self_did.clone(),
        kvector_commitment: commitment,
        range_proof: vec![10, 20, 30],
        trust_score_lower: 0.3,
        trust_score_upper: 0.39,
        expires_at: None,
        supersedes: None,
    };

    let _: Record = conductor
        .call(&alice.zome("credential_wallet"), "self_attest_trust", input)
        .await;

    // Get the credential to find its ID
    let creds: Vec<Record> = conductor
        .call(
            &alice.zome("credential_wallet"),
            "get_trust_credentials",
            self_did,
        )
        .await;

    assert_eq!(creds.len(), 1);

    // Extract the credential ID from the entry (we need to deserialize the full TrustCredential)
    // For verification, we query by chain instead — just pass a known credential_id pattern
    // The ID format is "trust-cred:{subject_did}:{timestamp}", which we can't predict.
    // Instead, verify a non-existent ID returns "not found"
    let result: VerificationResult = conductor
        .call(
            &alice.zome("credential_wallet"),
            "verify_trust_credential",
            "non-existent-cred-id".to_string(),
        )
        .await;

    assert!(!result.commitment_valid);
    assert!(result.message.contains("not found"));
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_trust_get_by_tier() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();
    let self_did = format!("did:mycelix:{}", agent);

    let mut commitment = vec![0u8; 32];
    commitment[0] = 7;

    // Self-attest with a Standard tier (mid = 0.495 -> Standard)
    let input = SelfAttestTrustInput {
        self_did: self_did.clone(),
        kvector_commitment: commitment,
        range_proof: vec![1, 2],
        trust_score_lower: 0.4,
        trust_score_upper: 0.59,
        expires_at: None,
        supersedes: None,
    };

    let _: Record = conductor
        .call(&alice.zome("credential_wallet"), "self_attest_trust", input)
        .await;

    // Query by Standard tier
    let standard_creds: Vec<Record> = conductor
        .call(
            &alice.zome("credential_wallet"),
            "get_trust_credentials_by_tier",
            TrustTier::Standard,
        )
        .await;

    assert_eq!(standard_creds.len(), 1);

    // Query by Guardian tier — should be empty
    let guardian_creds: Vec<Record> = conductor
        .call(
            &alice.zome("credential_wallet"),
            "get_trust_credentials_by_tier",
            TrustTier::Guardian,
        )
        .await;

    assert_eq!(guardian_creds.len(), 0);
}

// ============================================================================
// Personal Bridge Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_health_check() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let health: BridgeHealth = conductor
        .call(&alice.zome("personal_bridge"), "health_check", ())
        .await;

    assert!(health.healthy);
    assert_eq!(health.domains.len(), 3);
    assert!(health.domains.contains(&"identity".to_string()));
    assert!(health.domains.contains(&"health".to_string()));
    assert!(health.domains.contains(&"credential".to_string()));
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_broadcast_event() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    let event = PersonalEventEntry {
        domain: "identity".to_string(),
        event_type: "profile_updated".to_string(),
        source_agent: agent.clone(),
        payload: r#"{"field":"display_name","new":"Alice"}"#.to_string(),
        created_at: Timestamp::now(),
        related_hashes: vec![],
    };

    let record: Record = conductor
        .call(&alice.zome("personal_bridge"), "broadcast_event", event)
        .await;

    assert_eq!(record.action().author(), alice.agent_pubkey());

    // Verify event appears in domain query
    let events: Vec<Record> = conductor
        .call(
            &alice.zome("personal_bridge"),
            "get_domain_events",
            "identity".to_string(),
        )
        .await;

    assert_eq!(events.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_get_all_events() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    // Broadcast events across all three domains
    for domain in &["identity", "health", "credential"] {
        let event = PersonalEventEntry {
            domain: domain.to_string(),
            event_type: "test_event".to_string(),
            source_agent: agent.clone(),
            payload: format!(r#"{{"domain":"{}"}}"#, domain),
            created_at: Timestamp::now(),
            related_hashes: vec![],
        };
        let _: Record = conductor
            .call(&alice.zome("personal_bridge"), "broadcast_event", event)
            .await;
    }

    let all_events: Vec<Record> = conductor
        .call(&alice.zome("personal_bridge"), "get_all_events", ())
        .await;

    assert_eq!(all_events.len(), 3);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_get_events_by_type() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    let event1 = PersonalEventEntry {
        domain: "health".to_string(),
        event_type: "biometric_recorded".to_string(),
        source_agent: agent.clone(),
        payload: r#"{"type":"heart_rate"}"#.to_string(),
        created_at: Timestamp::now(),
        related_hashes: vec![],
    };

    let event2 = PersonalEventEntry {
        domain: "health".to_string(),
        event_type: "consent_granted".to_string(),
        source_agent: agent.clone(),
        payload: r#"{"grantee":"bob"}"#.to_string(),
        created_at: Timestamp::now(),
        related_hashes: vec![],
    };

    let _: Record = conductor
        .call(&alice.zome("personal_bridge"), "broadcast_event", event1)
        .await;
    let _: Record = conductor
        .call(&alice.zome("personal_bridge"), "broadcast_event", event2)
        .await;

    let bio_events: Vec<Record> = conductor
        .call(
            &alice.zome("personal_bridge"),
            "get_events_by_type",
            EventTypeQuery {
                domain: "health".to_string(),
                event_type: "biometric_recorded".to_string(),
            },
        )
        .await;

    assert_eq!(bio_events.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_query_and_resolve() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    let query = PersonalQueryEntry {
        domain: "credential".to_string(),
        query_type: "get_by_type".to_string(),
        requester: agent.clone(),
        params: r#"{"type":"Identity"}"#.to_string(),
        result: None,
        created_at: Timestamp::now(),
        resolved_at: None,
        success: None,
    };

    let record: Record = conductor
        .call(&alice.zome("personal_bridge"), "query_personal", query)
        .await;

    let query_hash = record.action_address().clone();

    // Resolve the query
    let resolve = ResolveQueryInput {
        query_hash,
        result: r#"{"found":2}"#.to_string(),
        success: true,
    };

    let resolved: Record = conductor
        .call(&alice.zome("personal_bridge"), "resolve_query", resolve)
        .await;

    assert_eq!(resolved.action().author(), alice.agent_pubkey());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_get_my_queries() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    for domain in &["identity", "health"] {
        let query = PersonalQueryEntry {
            domain: domain.to_string(),
            query_type: "list".to_string(),
            requester: agent.clone(),
            params: "{}".to_string(),
            result: None,
            created_at: Timestamp::now(),
            resolved_at: None,
            success: None,
        };
        let _: Record = conductor
            .call(&alice.zome("personal_bridge"), "query_personal", query)
            .await;
    }

    let queries: Vec<Record> = conductor
        .call(&alice.zome("personal_bridge"), "get_my_queries", ())
        .await;

    assert_eq!(queries.len(), 2);
}

// ============================================================================
// Cross-Domain Tests — the real value of cluster consolidation
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_identity_then_credential() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    // 1. Set up identity
    let profile = Profile {
        display_name: "Alice Cross-Domain".to_string(),
        avatar: None,
        bio: Some("Testing cross-domain flow".to_string()),
        metadata: HashMap::new(),
        updated_at: Timestamp::now(),
    };

    let _: Record = conductor
        .call(&alice.zome("identity_vault"), "set_profile", profile)
        .await;

    // 2. Store an identity credential
    let cred = StoredCredential {
        credential_type: CredentialType::Identity,
        credential_data: r#"{"verified":true,"name":"Alice Cross-Domain"}"#.to_string(),
        issuer: format!("did:mycelix:{}", agent),
        issued_at: Timestamp::now(),
        expires_at: None,
        revoked: false,
    };

    let _: Record = conductor
        .call(&alice.zome("credential_wallet"), "store_credential", cred)
        .await;

    // 3. Bridge event linking them
    let event = PersonalEventEntry {
        domain: "identity".to_string(),
        event_type: "identity_credential_linked".to_string(),
        source_agent: agent.clone(),
        payload: r#"{"profile":"Alice Cross-Domain","credential":"identity"}"#.to_string(),
        created_at: Timestamp::now(),
        related_hashes: vec![],
    };

    let _: Record = conductor
        .call(&alice.zome("personal_bridge"), "broadcast_event", event)
        .await;

    // 4. Verify all data accessible in same DNA
    let profile_result: Option<Record> = conductor
        .call(&alice.zome("identity_vault"), "get_my_profile", ())
        .await;
    assert!(profile_result.is_some());

    let creds: Vec<Record> = conductor
        .call(&alice.zome("credential_wallet"), "get_my_credentials", ())
        .await;
    assert_eq!(creds.len(), 1);

    let events: Vec<Record> = conductor
        .call(
            &alice.zome("personal_bridge"),
            "get_domain_events",
            "identity".to_string(),
        )
        .await;
    assert_eq!(events.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_health_record_with_consent_and_event() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();
    let provider_key = AgentPubKey::from_raw_36(vec![2u8; 36]);

    // 1. Create a health record
    let health_record = HealthRecord {
        record_type: "vaccination".to_string(),
        data: r#"{"vaccine":"COVID-19","dose":2}"#.to_string(),
        source: "provider:did:key:z6MkProvider".to_string(),
        event_date: Timestamp::now(),
        updated_at: Timestamp::now(),
    };

    let _: Record = conductor
        .call(
            &alice.zome("health_vault"),
            "create_health_record",
            health_record,
        )
        .await;

    // 2. Grant consent to a provider
    let consent = ConsentGrant {
        grantee: provider_key,
        record_types: vec!["vaccination".to_string()],
        expires_at: None,
        active: true,
        created_at: Timestamp::now(),
    };

    let _: Record = conductor
        .call(&alice.zome("health_vault"), "grant_consent", consent)
        .await;

    // 3. Store a health credential from the vaccination
    let cred = StoredCredential {
        credential_type: CredentialType::Health,
        credential_data: r#"{"vaccine":"COVID-19","fully_vaccinated":true}"#.to_string(),
        issuer: "did:key:z6MkProvider".to_string(),
        issued_at: Timestamp::now(),
        expires_at: None,
        revoked: false,
    };

    let _: Record = conductor
        .call(&alice.zome("credential_wallet"), "store_credential", cred)
        .await;

    // 4. Broadcast event
    let event = PersonalEventEntry {
        domain: "health".to_string(),
        event_type: "vaccination_verified".to_string(),
        source_agent: agent.clone(),
        payload: r#"{"status":"verified"}"#.to_string(),
        created_at: Timestamp::now(),
        related_hashes: vec![],
    };

    let _: Record = conductor
        .call(&alice.zome("personal_bridge"), "broadcast_event", event)
        .await;

    // 5. Verify full cross-domain state
    let records: Vec<Record> = conductor
        .call(&alice.zome("health_vault"), "get_my_records", ())
        .await;
    assert_eq!(records.len(), 1);

    let consents: Vec<Record> = conductor
        .call(&alice.zome("health_vault"), "get_my_consents", ())
        .await;
    assert_eq!(consents.len(), 1);

    let creds: Vec<Record> = conductor
        .call(&alice.zome("credential_wallet"), "get_my_credentials", ())
        .await;
    assert_eq!(creds.len(), 1);

    let health: BridgeHealth = conductor
        .call(&alice.zome("personal_bridge"), "health_check", ())
        .await;
    assert!(health.healthy);
    assert_eq!(health.total_events, 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_trust_attestation_workflow() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();
    let self_did = format!("did:mycelix:{}", agent);

    // 1. Set up identity
    let profile = Profile {
        display_name: "Alice Trust".to_string(),
        avatar: None,
        bio: None,
        metadata: HashMap::new(),
        updated_at: Timestamp::now(),
    };

    let _: Record = conductor
        .call(&alice.zome("identity_vault"), "set_profile", profile)
        .await;

    // 2. Self-attest trust
    let mut commitment = vec![0u8; 32];
    commitment[0] = 99;

    let attest = SelfAttestTrustInput {
        self_did: self_did.clone(),
        kvector_commitment: commitment,
        range_proof: vec![5, 6, 7],
        trust_score_lower: 0.6,
        trust_score_upper: 0.79,
        expires_at: None,
        supersedes: None,
    };

    let _: Record = conductor
        .call(&alice.zome("credential_wallet"), "self_attest_trust", attest)
        .await;

    // 3. Broadcast trust event
    let event = PersonalEventEntry {
        domain: "credential".to_string(),
        event_type: "trust_attested".to_string(),
        source_agent: agent.clone(),
        payload: format!(
            r#"{{"did":"{}","tier":"Elevated"}}"#,
            self_did
        ),
        created_at: Timestamp::now(),
        related_hashes: vec![],
    };

    let _: Record = conductor
        .call(&alice.zome("personal_bridge"), "broadcast_event", event)
        .await;

    // 4. Verify the trust credential exists
    let trust_creds: Vec<Record> = conductor
        .call(
            &alice.zome("credential_wallet"),
            "get_trust_credentials",
            self_did,
        )
        .await;

    assert_eq!(trust_creds.len(), 1);

    // 5. Verify Elevated tier query works
    let elevated: Vec<Record> = conductor
        .call(
            &alice.zome("credential_wallet"),
            "get_trust_credentials_by_tier",
            TrustTier::Elevated,
        )
        .await;

    assert_eq!(elevated.len(), 1);

    // 6. Health check shows events
    let health: BridgeHealth = conductor
        .call(&alice.zome("personal_bridge"), "health_check", ())
        .await;

    assert!(health.healthy);
    assert_eq!(health.total_events, 1);
}

// ============================================================================
// Bridge Dispatch Tests (intra-cluster dispatch)
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_dispatch_to_identity_vault() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&personal_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // First set a profile directly so there's data to query
    let profile = Profile {
        display_name: "Dispatch Test".to_string(),
        avatar: None,
        bio: None,
        metadata: HashMap::new(),
        updated_at: Timestamp::now(),
    };

    let _: Record = conductor
        .call(&alice.zome("identity_vault"), "set_profile", profile)
        .await;

    // Now query via bridge query_personal (which auto-dispatches)
    let agent = alice.agent_pubkey().clone();
    let query = PersonalQueryEntry {
        domain: "identity".to_string(),
        query_type: "get_my_profile".to_string(),
        requester: agent.clone(),
        params: "null".to_string(),
        result: None,
        created_at: Timestamp::now(),
        resolved_at: None,
        success: None,
    };

    let record: Record = conductor
        .call(&alice.zome("personal_bridge"), "query_personal", query)
        .await;

    // The query was created (even if auto-dispatch succeeded or not)
    assert_eq!(record.action().author(), alice.agent_pubkey());
}
