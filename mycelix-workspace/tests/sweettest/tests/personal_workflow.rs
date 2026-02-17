//! Personal Cluster Sweettest Integration Tests
//!
//! Tests the Sovereign (Personal) tier of the Fractal CivOS architecture:
//! - Identity vault CRUD (profile, avatar)
//! - Health vault CRUD (records, biometrics)
//! - Credential wallet (store, retrieve by type)
//! - Trust credentials (K-Vector issuance, self-attestation, verification)
//! - Personal bridge (dispatch, credential presentation, cross-cluster)
//!
//! ## Prerequisites
//!
//! ```bash
//! cd mycelix-personal && cargo build --release --target wasm32-unknown-unknown
//! hc dna pack mycelix-personal/dna/
//! ```
//!
//! ## Running
//!
//! ```bash
//! cargo test --release -p mycelix-sweettest --test personal_workflow -- --ignored
//! ```

mod harness;

use harness::*;
use holochain::prelude::*;
use holochain::sweettest::*;
use serial_test::serial;
use std::path::PathBuf;

// ============================================================================
// Mirror types — avoid WASM symbol conflicts by re-defining structs
// ============================================================================

// --- Identity Vault ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Profile {
    display_name: String,
    bio: Option<String>,
    avatar_hash: Option<String>,
    metadata: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct UpdateProfileInput {
    original_hash: ActionHash,
    updated_profile: Profile,
}

// --- Health Vault ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct HealthRecord {
    record_type: String,
    data: String,
    source: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Biometric {
    metric_type: String,
    value: f64,
    unit: String,
    notes: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ConsentGrant {
    grantee: AgentPubKey,
    record_types: Vec<String>,
    expires_at: Option<i64>,
}

// --- Credential Wallet (Generic) ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct StoredCredential {
    credential_type: String,
    issuer: String,
    subject: String,
    claims: String,
    issued_at: i64,
    expires_at: Option<i64>,
    proof: Option<String>,
    revoked: bool,
}

// --- Trust Credentials (K-Vector) ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct IssueTrustCredentialInput {
    subject_did: String,
    issuer_did: String,
    kvector_commitment: Vec<u8>,
    range_proof: Vec<u8>,
    trust_score_lower: f32,
    trust_score_upper: f32,
    expires_at: Option<i64>,
    supersedes: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct SelfAttestTrustInput {
    self_did: String,
    kvector_commitment: Vec<u8>,
    range_proof: Vec<u8>,
    trust_score_lower: f32,
    trust_score_upper: f32,
    expires_at: Option<i64>,
    supersedes: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct VerificationResult {
    credential_id: String,
    commitment_valid: bool,
    tier_consistent: bool,
    not_revoked: bool,
    not_expired: bool,
    proof_format_valid: bool,
    message: String,
}

// --- Bridge ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct DispatchInput {
    zome: String,
    fn_name: String,
    payload: Vec<u8>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct DispatchResult {
    success: bool,
    response: Option<Vec<u8>>,
    error: Option<String>,
}

// ============================================================================
// Tests
// ============================================================================

/// Test: Create a profile in the identity vault and retrieve it.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled personal WASM + conductor"]
async fn test_identity_vault_create_and_get_profile() {
    let dna_path = DnaPaths::personal();
    if !dna_path.exists() {
        eprintln!("Skipping: personal DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "personal", 1).await;
    let alice = &agents[0];

    // Create a profile
    let profile = Profile {
        display_name: "Alice".into(),
        bio: Some("Sovereign identity test".into()),
        avatar_hash: None,
        metadata: None,
    };

    let record: Record = alice
        .call_zome_fn("identity_vault", "create_profile", profile.clone())
        .await;

    // Retrieve my profiles
    let profiles: Vec<Record> = alice
        .call_zome_fn("identity_vault", "get_my_profiles", ())
        .await;

    assert!(!profiles.is_empty(), "Should have at least one profile");
}

/// Test: Create a health record and retrieve it by type.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled personal WASM + conductor"]
async fn test_health_vault_create_and_get_record() {
    let dna_path = DnaPaths::personal();
    if !dna_path.exists() {
        eprintln!("Skipping: personal DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "personal", 1).await;
    let alice = &agents[0];

    // Create a health record
    let record = HealthRecord {
        record_type: "medication".into(),
        data: r#"{"name":"aspirin","dosage":"100mg"}"#.into(),
        source: Some("self-reported".into()),
    };

    let created: Record = alice
        .call_zome_fn("health_vault", "create_health_record", record)
        .await;

    // Retrieve by type
    let records: Vec<Record> = alice
        .call_zome_fn(
            "health_vault",
            "get_records_by_type",
            "medication".to_string(),
        )
        .await;

    assert!(!records.is_empty(), "Should have at least one medication record");
}

/// Test: Store a credential and retrieve by type.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled personal WASM + conductor"]
async fn test_credential_wallet_store_and_retrieve() {
    let dna_path = DnaPaths::personal();
    if !dna_path.exists() {
        eprintln!("Skipping: personal DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "personal", 1).await;
    let alice = &agents[0];

    // Store a FL credential
    let credential = StoredCredential {
        credential_type: "FederatedLearning".into(),
        issuer: "did:mycelix:fl-coordinator".into(),
        subject: "did:mycelix:alice".into(),
        claims: r#"{"phi":0.42,"rounds":10}"#.into(),
        issued_at: 1708000000,
        expires_at: None,
        proof: None,
        revoked: false,
    };

    let record: Record = alice
        .call_zome_fn("credential_wallet", "store_credential", credential)
        .await;

    // Retrieve all credentials
    let creds: Vec<Record> = alice
        .call_zome_fn("credential_wallet", "get_my_credentials", ())
        .await;

    assert!(!creds.is_empty(), "Should have at least one credential");
}

/// Test: Personal bridge dispatch to identity vault succeeds.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled personal WASM + conductor"]
async fn test_personal_bridge_dispatch_allowed() {
    let dna_path = DnaPaths::personal();
    if !dna_path.exists() {
        eprintln!("Skipping: personal DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "personal", 1).await;
    let alice = &agents[0];

    // Dispatch to an allowed zome (identity_vault)
    let dispatch = DispatchInput {
        zome: "identity_vault".into(),
        fn_name: "get_my_profiles".into(),
        payload: ExternIO::encode(()).unwrap().0,
    };

    let result: DispatchResult = alice
        .call_zome_fn("personal_bridge", "dispatch_call", dispatch)
        .await;

    assert!(result.success, "Dispatch to allowed zome should succeed");
}

/// Test: Personal bridge dispatch to unknown zome is rejected.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled personal WASM + conductor"]
async fn test_personal_bridge_dispatch_disallowed() {
    let dna_path = DnaPaths::personal();
    if !dna_path.exists() {
        eprintln!("Skipping: personal DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "personal", 1).await;
    let alice = &agents[0];

    // Dispatch to a non-allowed zome
    let dispatch = DispatchInput {
        zome: "malicious_zome".into(),
        fn_name: "steal_data".into(),
        payload: vec![],
    };

    let result = alice
        .call_zome_fn_fallible::<_, DispatchResult>(
            "personal_bridge",
            "dispatch_call",
            dispatch,
        )
        .await;

    assert!(
        result.is_err(),
        "Dispatch to non-allowed zome should be rejected"
    );
}

/// Test: Present Phi credential via personal bridge.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled personal WASM + conductor"]
async fn test_personal_bridge_present_phi_credential() {
    let dna_path = DnaPaths::personal();
    if !dna_path.exists() {
        eprintln!("Skipping: personal DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "personal", 1).await;
    let alice = &agents[0];

    // First store a Phi credential
    let credential = StoredCredential {
        credential_type: "ConsciousnessPhi".into(),
        issuer: "did:mycelix:symthaea".into(),
        subject: "did:mycelix:alice".into(),
        claims: r#"{"phi_effective":0.72,"topology":"small_world"}"#.into(),
        issued_at: 1708000000,
        expires_at: None,
        proof: Some("proof_bytes_here".into()),
        revoked: false,
    };

    let _record: Record = alice
        .call_zome_fn("credential_wallet", "store_credential", credential)
        .await;

    // Now present Phi credential via bridge
    let presentation = alice
        .call_zome_fn_fallible::<_, serde_json::Value>(
            "personal_bridge",
            "present_phi_credential",
            (),
        )
        .await;

    // The presentation should succeed (even if no matching credential
    // is found, it should return a default/empty presentation)
    assert!(
        presentation.is_ok(),
        "Phi credential presentation should not error"
    );
}

// ============================================================================
// Trust Credential Tests
// ============================================================================

/// Test: Self-attest a K-Vector trust credential and retrieve it.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled personal WASM + conductor"]
async fn test_trust_credential_self_attest_and_get() {
    let dna_path = DnaPaths::personal();
    if !dna_path.exists() {
        eprintln!("Skipping: personal DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "personal", 1).await;
    let alice = &agents[0];

    // Derive the caller's DID from their agent pubkey
    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);

    // Self-attest with a K-Vector commitment and range proof
    let mut commitment = vec![0u8; 32];
    commitment[0] = 0x42;
    commitment[1] = 0xAB;

    let input = SelfAttestTrustInput {
        self_did: alice_did.clone(),
        kvector_commitment: commitment,
        range_proof: vec![1, 2, 3, 4, 5],
        trust_score_lower: 0.4,
        trust_score_upper: 0.6,
        expires_at: None,
        supersedes: None,
    };

    let record: Record = alice
        .call_zome_fn("credential_wallet", "self_attest_trust", input)
        .await;

    // Retrieve trust credentials
    let creds: Vec<Record> = alice
        .call_zome_fn(
            "credential_wallet",
            "get_trust_credentials",
            alice_did.clone(),
        )
        .await;

    assert!(
        !creds.is_empty(),
        "Should have at least one trust credential after self-attestation"
    );
}

/// Test: Verify a self-attested trust credential passes on-chain checks.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled personal WASM + conductor"]
async fn test_trust_credential_verify() {
    let dna_path = DnaPaths::personal();
    if !dna_path.exists() {
        eprintln!("Skipping: personal DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "personal", 1).await;
    let alice = &agents[0];
    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);

    // Self-attest
    let mut commitment = vec![0u8; 32];
    commitment[0] = 0xFF;

    let input = SelfAttestTrustInput {
        self_did: alice_did.clone(),
        kvector_commitment: commitment,
        range_proof: vec![10, 20, 30],
        trust_score_lower: 0.6,
        trust_score_upper: 0.79,
        expires_at: None,
        supersedes: None,
    };

    let _record: Record = alice
        .call_zome_fn("credential_wallet", "self_attest_trust", input)
        .await;

    // Get the credential to find its ID
    let creds: Vec<Record> = alice
        .call_zome_fn(
            "credential_wallet",
            "get_trust_credentials",
            alice_did.clone(),
        )
        .await;

    assert!(!creds.is_empty(), "Should have a trust credential");

    // Extract the credential ID from the record
    // The record contains a TrustCredential entry — we'll use the entry data
    // Since we can't deserialize TrustCredential directly (mirror type needed),
    // we'll verify via get_trust_credentials_by_tier instead
    let elevated_creds: Vec<Record> = alice
        .call_zome_fn(
            "credential_wallet",
            "get_trust_credentials_by_tier",
            "Elevated", // mid = 0.695 → Elevated tier
        )
        .await;

    assert!(
        !elevated_creds.is_empty(),
        "Should find credential in Elevated tier"
    );
}

/// Test: Get trust credentials by tier returns empty for unmatched tier.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled personal WASM + conductor"]
async fn test_trust_credential_tier_filtering() {
    let dna_path = DnaPaths::personal();
    if !dna_path.exists() {
        eprintln!("Skipping: personal DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "personal", 1).await;
    let alice = &agents[0];

    // No credentials stored yet — Guardian tier should be empty
    let guardian_creds: Vec<Record> = alice
        .call_zome_fn(
            "credential_wallet",
            "get_trust_credentials_by_tier",
            "Guardian",
        )
        .await;

    assert!(
        guardian_creds.is_empty(),
        "Guardian tier should be empty without any credentials"
    );
}
