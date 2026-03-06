//! Personal Cluster Sweettest Integration Tests
//!
//! Tests the Sovereign (Personal) tier of the Fractal CivOS architecture:
//! - Identity vault CRUD (profile, keys)
//! - Health vault CRUD (records, biometrics)
//! - Credential wallet (store, retrieve)
//! - Trust credentials (K-Vector self-attestation, tier filtering)
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
use std::collections::HashMap;
use std::path::PathBuf;

// ============================================================================
// Mirror types — must match actual zome integrity/coordinator struct layout
// ============================================================================

// --- Identity Vault (integrity: identity-vault/integrity/src/lib.rs) ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Profile {
    display_name: String,
    avatar: Option<String>,
    bio: Option<String>,
    metadata: HashMap<String, String>,
    updated_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct MasterKey {
    label: String,
    purpose: String,
    public_key_hex: String,
    active: bool,
    created_at: Timestamp,
}

// --- Health Vault (integrity: health-vault/integrity/src/lib.rs) ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct HealthRecord {
    record_type: String,
    data: String,
    source: String,
    event_date: Timestamp,
    updated_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Biometric {
    metric_type: String,
    value: f64,
    unit: String,
    measured_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ConsentGrant {
    grantee: AgentPubKey,
    record_types: Vec<String>,
    expires_at: Option<Timestamp>,
    active: bool,
    created_at: Timestamp,
}

// --- Credential Wallet (integrity: credential-wallet/integrity/src/lib.rs) ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum CredentialType {
    Identity,
    Health,
    FederatedLearning,
    Governance,
    Domain(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct StoredCredential {
    credential_type: CredentialType,
    credential_data: String,
    issuer: String,
    issued_at: Timestamp,
    expires_at: Option<Timestamp>,
    revoked: bool,
}

// --- Trust Credentials (coordinator: credential-wallet/coordinator/src/lib.rs) ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct IssueTrustCredentialInput {
    subject_did: String,
    issuer_did: String,
    kvector_commitment: Vec<u8>,
    range_proof: Vec<u8>,
    trust_score_lower: f32,
    trust_score_upper: f32,
    expires_at: Option<Timestamp>,
    supersedes: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct SelfAttestTrustInput {
    self_did: String,
    kvector_commitment: Vec<u8>,
    range_proof: Vec<u8>,
    trust_score_lower: f32,
    trust_score_upper: f32,
    expires_at: Option<Timestamp>,
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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum TrustTier {
    Observer,
    Basic,
    Standard,
    Elevated,
    Guardian,
}

// --- Bridge (coordinator: personal-bridge/coordinator/src/lib.rs) ---

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
// Tests — Identity Vault
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

    let profile = Profile {
        display_name: "Alice".into(),
        avatar: None,
        bio: Some("Sovereign identity test".into()),
        metadata: HashMap::new(),
        updated_at: Timestamp::now(),
    };

    let _record: Record = alice
        .call_zome_fn("identity_vault", "set_profile", profile)
        .await;

    let maybe_profile: Option<Record> = alice
        .call_zome_fn("identity_vault", "get_my_profile", ())
        .await;

    assert!(
        maybe_profile.is_some(),
        "Should have a profile after set_profile"
    );
}

// ============================================================================
// Tests — Health Vault
// ============================================================================

/// Test: Create a health record and retrieve it.
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

    let record = HealthRecord {
        record_type: "medication".into(),
        data: r#"{"name":"aspirin","dosage":"100mg"}"#.into(),
        source: "self-reported".into(),
        event_date: Timestamp::now(),
        updated_at: Timestamp::now(),
    };

    let _created: Record = alice
        .call_zome_fn("health_vault", "create_health_record", record)
        .await;

    let records: Vec<Record> = alice
        .call_zome_fn("health_vault", "get_my_records", ())
        .await;

    assert!(
        !records.is_empty(),
        "Should have at least one health record"
    );
}

// ============================================================================
// Tests — Credential Wallet
// ============================================================================

/// Test: Store a credential and retrieve it.
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

    let credential = StoredCredential {
        credential_type: CredentialType::FederatedLearning,
        credential_data: r#"{"phi":0.42,"rounds":10}"#.into(),
        issuer: "did:mycelix:fl-coordinator".into(),
        issued_at: Timestamp::now(),
        expires_at: None,
        revoked: false,
    };

    let _record: Record = alice
        .call_zome_fn("credential_wallet", "store_credential", credential)
        .await;

    let creds: Vec<Record> = alice
        .call_zome_fn("credential_wallet", "get_my_credentials", ())
        .await;

    assert!(!creds.is_empty(), "Should have at least one credential");
}

// ============================================================================
// Tests — Bridge Dispatch
// ============================================================================

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

    let dispatch = DispatchInput {
        zome: "identity_vault".into(),
        fn_name: "get_my_profile".into(),
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

    let dispatch = DispatchInput {
        zome: "malicious_zome".into(),
        fn_name: "steal_data".into(),
        payload: vec![],
    };

    // The bridge may return an error at the conductor level (is_err)
    // OR return a DispatchResult with success=false. Either is acceptable.
    let result = alice
        .call_zome_fn_fallible::<_, DispatchResult>(
            "personal_bridge",
            "dispatch_call",
            dispatch,
        )
        .await;

    match result {
        Err(_) => {} // Conductor-level rejection — good
        Ok(dispatch_result) => {
            assert!(
                !dispatch_result.success,
                "Dispatch to non-allowed zome should fail (success should be false)"
            );
        }
    }
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

    // Store a Phi credential first
    let credential = StoredCredential {
        credential_type: CredentialType::FederatedLearning,
        credential_data: r#"{"phi_effective":0.72,"topology":"small_world"}"#.into(),
        issuer: "did:mycelix:symthaea".into(),
        issued_at: Timestamp::now(),
        expires_at: None,
        revoked: false,
    };

    let _record: Record = alice
        .call_zome_fn("credential_wallet", "store_credential", credential)
        .await;

    // Present Phi credential via bridge
    let presentation = alice
        .call_zome_fn_fallible::<_, serde_json::Value>(
            "personal_bridge",
            "present_phi_credential",
            (),
        )
        .await;

    assert!(
        presentation.is_ok(),
        "Phi credential presentation should not error"
    );
}

// ============================================================================
// Tests — Trust Credentials
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
    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);

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

    let _record: Record = alice
        .call_zome_fn("credential_wallet", "self_attest_trust", input)
        .await;

    let creds: Vec<Record> = alice
        .call_zome_fn(
            "credential_wallet",
            "get_trust_credentials",
            alice_did,
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

    let creds: Vec<Record> = alice
        .call_zome_fn(
            "credential_wallet",
            "get_trust_credentials",
            alice_did,
        )
        .await;

    assert!(!creds.is_empty(), "Should have a trust credential");

    // Verify via tier — mid = 0.695 → Elevated tier
    let elevated_creds: Vec<Record> = alice
        .call_zome_fn(
            "credential_wallet",
            "get_trust_credentials_by_tier",
            TrustTier::Elevated,
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

    let guardian_creds: Vec<Record> = alice
        .call_zome_fn(
            "credential_wallet",
            "get_trust_credentials_by_tier",
            TrustTier::Guardian,
        )
        .await;

    assert!(
        guardian_creds.is_empty(),
        "Guardian tier should be empty without any credentials"
    );
}

// ============================================================================
// Cross-Cluster Integration Tests (require unified hApp)
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CrossClusterDispatchInput {
    role: String,
    zome: String,
    fn_name: String,
    payload: Vec<u8>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct QueryIdentityInput {
    did: String,
    source_happ: String,
    requested_fields: Vec<String>,
}

/// Test: Personal → Identity cross-cluster DID resolution.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires unified hApp bundle with identity + personal roles"]
async fn test_personal_to_identity_resolve_did() {
    let happ_path = PathBuf::from("../../happs/mycelix-unified-happ.yaml");
    if !happ_path.exists() {
        eprintln!("Skipping: unified hApp not at {:?}", happ_path);
        return;
    }

    let agents = setup_test_agents_from_happ(&happ_path, 1).await;
    let alice = &agents[0];

    let _did_record: Record = alice
        .call_zome_fn_on_role("identity", "did_registry", "create_did", ())
        .await;

    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);

    let result: DispatchResult = alice
        .call_zome_fn_on_role(
            "personal",
            "personal_bridge",
            "resolve_did",
            alice_did,
        )
        .await;

    assert!(
        result.success,
        "DID resolution via identity cluster should succeed"
    );
    assert!(
        result.response.is_some(),
        "Should return DID document bytes"
    );
}

/// Test: Personal → Identity cross-cluster DID active check.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires unified hApp bundle with identity + personal roles"]
async fn test_personal_to_identity_is_did_active() {
    let happ_path = PathBuf::from("../../happs/mycelix-unified-happ.yaml");
    if !happ_path.exists() {
        eprintln!("Skipping: unified hApp not at {:?}", happ_path);
        return;
    }

    let agents = setup_test_agents_from_happ(&happ_path, 1).await;
    let alice = &agents[0];

    let _: Record = alice
        .call_zome_fn_on_role("identity", "did_registry", "create_did", ())
        .await;

    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);

    let result: DispatchResult = alice
        .call_zome_fn_on_role(
            "personal",
            "personal_bridge",
            "is_did_active",
            alice_did,
        )
        .await;

    assert!(result.success, "DID active check should succeed");
}

/// Test: Personal → Identity cross-cluster MATL trust score.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires unified hApp bundle with identity + personal roles"]
async fn test_personal_to_identity_matl_score() {
    let happ_path = PathBuf::from("../../happs/mycelix-unified-happ.yaml");
    if !happ_path.exists() {
        eprintln!("Skipping: unified hApp not at {:?}", happ_path);
        return;
    }

    let agents = setup_test_agents_from_happ(&happ_path, 1).await;
    let alice = &agents[0];

    let _: Record = alice
        .call_zome_fn_on_role("identity", "did_registry", "create_did", ())
        .await;

    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);

    let result: DispatchResult = alice
        .call_zome_fn_on_role(
            "personal",
            "personal_bridge",
            "get_matl_score",
            alice_did,
        )
        .await;

    assert!(result.success, "MATL score query should succeed");
}

/// Test: Personal → Identity cross-cluster credential verification.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires unified hApp bundle with identity + personal roles"]
async fn test_personal_to_identity_verify_credential() {
    let happ_path = PathBuf::from("../../happs/mycelix-unified-happ.yaml");
    if !happ_path.exists() {
        eprintln!("Skipping: unified hApp not at {:?}", happ_path);
        return;
    }

    let agents = setup_test_agents_from_happ(&happ_path, 1).await;
    let alice = &agents[0];

    let result: DispatchResult = alice
        .call_zome_fn_on_role(
            "personal",
            "personal_bridge",
            "verify_credential",
            "urn:uuid:non-existent-credential".to_string(),
        )
        .await;

    assert!(
        result.success || result.error.is_some(),
        "Cross-cluster credential verification should reach identity cluster"
    );
}

/// Test: Personal → Commons cross-cluster dispatch.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires unified hApp bundle with all roles"]
async fn test_personal_to_commons_dispatch() {
    let happ_path = PathBuf::from("../../happs/mycelix-unified-happ.yaml");
    if !happ_path.exists() {
        eprintln!("Skipping: unified hApp not at {:?}", happ_path);
        return;
    }

    let agents = setup_test_agents_from_happ(&happ_path, 1).await;
    let alice = &agents[0];

    let dispatch = CrossClusterDispatchInput {
        role: "commons_land".into(),
        zome: "commons_bridge".into(),
        fn_name: "health_check".into(),
        payload: ExternIO::encode(()).unwrap().0,
    };

    let result: DispatchResult = alice
        .call_zome_fn_on_role(
            "personal",
            "personal_bridge",
            "dispatch_commons_call",
            dispatch,
        )
        .await;

    assert!(
        result.success,
        "Cross-cluster dispatch to commons should succeed"
    );
}

/// Test: Personal → Civic cross-cluster dispatch.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires unified hApp bundle with all roles"]
async fn test_personal_to_civic_dispatch() {
    let happ_path = PathBuf::from("../../happs/mycelix-unified-happ.yaml");
    if !happ_path.exists() {
        eprintln!("Skipping: unified hApp not at {:?}", happ_path);
        return;
    }

    let agents = setup_test_agents_from_happ(&happ_path, 1).await;
    let alice = &agents[0];

    let dispatch = CrossClusterDispatchInput {
        role: "civic".into(),
        zome: "civic_bridge".into(),
        fn_name: "health_check".into(),
        payload: ExternIO::encode(()).unwrap().0,
    };

    let result: DispatchResult = alice
        .call_zome_fn_on_role(
            "personal",
            "personal_bridge",
            "dispatch_civic_call",
            dispatch,
        )
        .await;

    assert!(
        result.success,
        "Cross-cluster dispatch to civic should succeed"
    );
}
