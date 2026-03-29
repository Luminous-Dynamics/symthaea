// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
        .call_zome_fn_fallible::<_, DispatchResult>("personal_bridge", "dispatch_call", dispatch)
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
        .call_zome_fn("credential_wallet", "get_trust_credentials", alice_did)
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
        .call_zome_fn("credential_wallet", "get_trust_credentials", alice_did)
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
// Tests — Identity Vault: Key Management & Selective Disclosure
// ============================================================================

/// Test: Register a master key and retrieve it.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled personal WASM + conductor"]
async fn test_identity_vault_register_and_get_keys() {
    let dna_path = DnaPaths::personal();
    if !dna_path.exists() {
        eprintln!("Skipping: personal DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "personal", 1).await;
    let alice = &agents[0];

    let key = MasterKey {
        label: "primary-signing".into(),
        purpose: "signing".into(),
        public_key_hex: "abcdef1234567890".into(),
        active: true,
        created_at: Timestamp::now(),
    };

    let _record: Record = alice
        .call_zome_fn("identity_vault", "register_key", key)
        .await;

    let keys: Vec<Record> = alice
        .call_zome_fn("identity_vault", "get_my_keys", ())
        .await;

    assert!(!keys.is_empty(), "Should have at least one registered key");
}

/// Test: Register multiple keys and verify count.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled personal WASM + conductor"]
async fn test_identity_vault_multiple_keys() {
    let dna_path = DnaPaths::personal();
    if !dna_path.exists() {
        eprintln!("Skipping: personal DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "personal", 1).await;
    let alice = &agents[0];

    for (label, purpose) in &[
        ("signing-key", "signing"),
        ("encryption-key", "encryption"),
        ("issuance-key", "credential_issuance"),
    ] {
        let key = MasterKey {
            label: label.to_string(),
            purpose: purpose.to_string(),
            public_key_hex: format!("deadbeef{}", label),
            active: true,
            created_at: Timestamp::now(),
        };
        let _: Record = alice
            .call_zome_fn("identity_vault", "register_key", key)
            .await;
    }

    let keys: Vec<Record> = alice
        .call_zome_fn("identity_vault", "get_my_keys", ())
        .await;

    assert_eq!(keys.len(), 3, "Should have exactly 3 registered keys");
}

/// Test: Selective disclosure returns only requested fields.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled personal WASM + conductor"]
async fn test_identity_vault_selective_disclosure() {
    let dna_path = DnaPaths::personal();
    if !dna_path.exists() {
        eprintln!("Skipping: personal DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "personal", 1).await;
    let alice = &agents[0];

    let profile = Profile {
        display_name: "Alice Sovereign".into(),
        avatar: Some("ipfs://Qm123".into()),
        bio: Some("Privacy-first identity".into()),
        metadata: HashMap::new(),
        updated_at: Timestamp::now(),
    };

    let _: Record = alice
        .call_zome_fn("identity_vault", "set_profile", profile)
        .await;

    // Request only display_name — bio and avatar should be excluded
    let disclosed: String = alice
        .call_zome_fn(
            "identity_vault",
            "disclose_profile",
            vec!["display_name".to_string()],
        )
        .await;

    let parsed: serde_json::Value = serde_json::from_str(&disclosed).unwrap();
    assert_eq!(parsed["display_name"], "Alice Sovereign");
    assert!(
        parsed.get("bio").is_none(),
        "Bio should not be disclosed when not requested"
    );
    assert!(
        parsed.get("avatar").is_none(),
        "Avatar should not be disclosed when not requested"
    );
}

/// Test: Selective disclosure with no profile returns empty JSON.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled personal WASM + conductor"]
async fn test_identity_vault_disclose_no_profile() {
    let dna_path = DnaPaths::personal();
    if !dna_path.exists() {
        eprintln!("Skipping: personal DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "personal", 1).await;
    let alice = &agents[0];

    let disclosed: String = alice
        .call_zome_fn(
            "identity_vault",
            "disclose_profile",
            vec!["display_name".to_string()],
        )
        .await;

    assert_eq!(disclosed, "{}", "No profile should return empty JSON");
}

// ============================================================================
// Tests — Health Vault: Biometrics & Consent
// ============================================================================

/// Test: Record a biometric measurement and retrieve it.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled personal WASM + conductor"]
async fn test_health_vault_record_and_get_biometric() {
    let dna_path = DnaPaths::personal();
    if !dna_path.exists() {
        eprintln!("Skipping: personal DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "personal", 1).await;
    let alice = &agents[0];

    let biometric = Biometric {
        metric_type: "heart_rate".into(),
        value: 72.0,
        unit: "bpm".into(),
        measured_at: Timestamp::now(),
    };

    let _: Record = alice
        .call_zome_fn("health_vault", "record_biometric", biometric)
        .await;

    let biometrics: Vec<Record> = alice
        .call_zome_fn("health_vault", "get_my_biometrics", ())
        .await;

    assert!(
        !biometrics.is_empty(),
        "Should have at least one biometric measurement"
    );
}

/// Test: Record multiple biometric types.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled personal WASM + conductor"]
async fn test_health_vault_multiple_biometrics() {
    let dna_path = DnaPaths::personal();
    if !dna_path.exists() {
        eprintln!("Skipping: personal DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "personal", 1).await;
    let alice = &agents[0];

    let measurements = vec![
        ("heart_rate", 72.0, "bpm"),
        ("blood_pressure", 120.0, "mmHg"),
        ("temperature", 36.7, "celsius"),
        ("blood_oxygen", 98.0, "percent"),
    ];

    for (metric, value, unit) in &measurements {
        let biometric = Biometric {
            metric_type: metric.to_string(),
            value: *value,
            unit: unit.to_string(),
            measured_at: Timestamp::now(),
        };
        let _: Record = alice
            .call_zome_fn("health_vault", "record_biometric", biometric)
            .await;
    }

    let biometrics: Vec<Record> = alice
        .call_zome_fn("health_vault", "get_my_biometrics", ())
        .await;

    assert_eq!(
        biometrics.len(),
        4,
        "Should have exactly 4 biometric measurements"
    );
}

/// Test: Grant consent and retrieve active consents.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled personal WASM + conductor"]
async fn test_health_vault_grant_and_get_consent() {
    let dna_path = DnaPaths::personal();
    if !dna_path.exists() {
        eprintln!("Skipping: personal DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "personal", 2).await;
    let alice = &agents[0];
    let bob = &agents[1];

    let consent = ConsentGrant {
        grantee: bob.agent_pubkey.clone(),
        record_types: vec!["medication".into(), "allergy".into()],
        expires_at: None,
        active: true,
        created_at: Timestamp::now(),
    };

    let _: Record = alice
        .call_zome_fn("health_vault", "grant_consent", consent)
        .await;

    let consents: Vec<Record> = alice
        .call_zome_fn("health_vault", "get_my_consents", ())
        .await;

    assert!(
        !consents.is_empty(),
        "Should have at least one consent grant"
    );
}

// ============================================================================
// Tests — Credential Wallet: Proofs, Type Filtering, Presentation
// ============================================================================

/// Test: Create a credential proof.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled personal WASM + conductor"]
async fn test_credential_wallet_create_proof() {
    let dna_path = DnaPaths::personal();
    if !dna_path.exists() {
        eprintln!("Skipping: personal DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "personal", 1).await;
    let alice = &agents[0];

    // First store a credential to reference
    let credential = StoredCredential {
        credential_type: CredentialType::Identity,
        credential_data: r#"{"verified":true}"#.into(),
        issuer: "did:mycelix:registrar".into(),
        issued_at: Timestamp::now(),
        expires_at: None,
        revoked: false,
    };

    let cred_record: Record = alice
        .call_zome_fn("credential_wallet", "store_credential", credential)
        .await;

    let proof = CredentialProof {
        credential_hash: cred_record.action_address().clone(),
        proof_data: vec![0xDE, 0xAD, 0xBE, 0xEF],
        claim: "identity-verified".into(),
        created_at: Timestamp::now(),
    };

    let _: Record = alice
        .call_zome_fn("credential_wallet", "create_proof", proof)
        .await;
}

/// Test: Filter credentials by type.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled personal WASM + conductor"]
async fn test_credential_wallet_get_by_type() {
    let dna_path = DnaPaths::personal();
    if !dna_path.exists() {
        eprintln!("Skipping: personal DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "personal", 1).await;
    let alice = &agents[0];

    // Store credentials of different types
    let identity_cred = StoredCredential {
        credential_type: CredentialType::Identity,
        credential_data: r#"{"verified":true}"#.into(),
        issuer: "did:mycelix:registrar".into(),
        issued_at: Timestamp::now(),
        expires_at: None,
        revoked: false,
    };

    let health_cred = StoredCredential {
        credential_type: CredentialType::Health,
        credential_data: r#"{"immunization":"covid-19"}"#.into(),
        issuer: "did:mycelix:health-authority".into(),
        issued_at: Timestamp::now(),
        expires_at: None,
        revoked: false,
    };

    let _: Record = alice
        .call_zome_fn("credential_wallet", "store_credential", identity_cred)
        .await;
    let _: Record = alice
        .call_zome_fn("credential_wallet", "store_credential", health_cred)
        .await;

    // Filter by Identity type only
    let identity_creds: Vec<serde_json::Value> = alice
        .call_zome_fn(
            "credential_wallet",
            "get_credentials_by_type",
            CredentialType::Identity,
        )
        .await;

    assert_eq!(
        identity_creds.len(),
        1,
        "Should have exactly 1 identity credential"
    );

    // Filter by Health type
    let health_creds: Vec<serde_json::Value> = alice
        .call_zome_fn(
            "credential_wallet",
            "get_credentials_by_type",
            CredentialType::Health,
        )
        .await;

    assert_eq!(
        health_creds.len(),
        1,
        "Should have exactly 1 health credential"
    );

    // No governance credentials should exist
    let gov_creds: Vec<serde_json::Value> = alice
        .call_zome_fn(
            "credential_wallet",
            "get_credentials_by_type",
            CredentialType::Governance,
        )
        .await;

    assert!(
        gov_creds.is_empty(),
        "Should have no governance credentials"
    );
}

/// Test: Present a credential returns the data.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled personal WASM + conductor"]
async fn test_credential_wallet_present_credential() {
    let dna_path = DnaPaths::personal();
    if !dna_path.exists() {
        eprintln!("Skipping: personal DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "personal", 1).await;
    let alice = &agents[0];

    let credential = StoredCredential {
        credential_type: CredentialType::Governance,
        credential_data: r#"{"council":"steward","votes":42}"#.into(),
        issuer: "did:mycelix:governance".into(),
        issued_at: Timestamp::now(),
        expires_at: None,
        revoked: false,
    };

    let _: Record = alice
        .call_zome_fn("credential_wallet", "store_credential", credential)
        .await;

    let presented: String = alice
        .call_zome_fn(
            "credential_wallet",
            "present_credential",
            CredentialType::Governance,
        )
        .await;

    assert!(
        presented.contains("steward"),
        "Presented credential should contain the data"
    );
}

// ============================================================================
// Tests — Trust Credential Revocation & Verification
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct RevokeTrustCredentialInput {
    credential_id: String,
    subject_did: String,
    reason: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CreateTrustPresentationInput {
    credential_id: String,
    subject_did: String,
    disclosed_tier: TrustTier,
    disclose_range: bool,
    trust_range: TrustScoreRange,
    presentation_proof: Vec<u8>,
    verifier_did: Option<String>,
    purpose: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct TrustScoreRange {
    lower: f32,
    upper: f32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CredentialProof {
    credential_hash: ActionHash,
    proof_data: Vec<u8>,
    claim: String,
    created_at: Timestamp,
}

/// Test: Verify a trust credential on-chain (commitment, tier, revocation, expiry).
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled personal WASM + conductor"]
async fn test_trust_credential_on_chain_verify() {
    let dna_path = DnaPaths::personal();
    if !dna_path.exists() {
        eprintln!("Skipping: personal DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "personal", 1).await;
    let alice = &agents[0];
    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);

    let mut commitment = vec![0u8; 32];
    commitment[0] = 0xCA;
    commitment[1] = 0xFE;

    let input = SelfAttestTrustInput {
        self_did: alice_did.clone(),
        kvector_commitment: commitment,
        range_proof: vec![1, 2, 3, 4],
        trust_score_lower: 0.5,
        trust_score_upper: 0.7,
        expires_at: None,
        supersedes: None,
    };

    let _: Record = alice
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

    assert!(!creds.is_empty());

    // Extract credential ID from the entry
    let cred_entry = creds[0].entry().as_option().unwrap();
    let cred_bytes = cred_entry.as_bytes();
    let cred_json: serde_json::Value = serde_json::from_slice(cred_bytes).unwrap();
    let cred_id = cred_json["id"].as_str().unwrap().to_string();

    let result: VerificationResult = alice
        .call_zome_fn("credential_wallet", "verify_trust_credential", cred_id)
        .await;

    assert!(result.commitment_valid, "Commitment should be valid");
    assert!(result.tier_consistent, "Tier should be consistent");
    assert!(result.not_revoked, "Should not be revoked");
    assert!(result.not_expired, "Should not be expired");
    assert!(result.proof_format_valid, "Proof format should be valid");
}

/// Test: Verify non-existent credential returns all-false result.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled personal WASM + conductor"]
async fn test_trust_credential_verify_not_found() {
    let dna_path = DnaPaths::personal();
    if !dna_path.exists() {
        eprintln!("Skipping: personal DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "personal", 1).await;
    let alice = &agents[0];

    let result: VerificationResult = alice
        .call_zome_fn(
            "credential_wallet",
            "verify_trust_credential",
            "nonexistent-cred-id".to_string(),
        )
        .await;

    assert!(
        !result.commitment_valid,
        "Non-existent credential should fail verification"
    );
    assert!(
        result.message.contains("not found"),
        "Message should indicate credential not found"
    );
}

// ============================================================================
// Tests — Personal Bridge: Health Check & Event Broadcasting
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct BridgeHealth {
    healthy: bool,
    agent: String,
    total_events: usize,
    total_queries: usize,
    domains: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct PersonalEventEntry {
    domain: String,
    event_type: String,
    source_agent: String,
    payload: String,
    related_hashes: Vec<String>,
}

/// Test: Bridge health check returns healthy status.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled personal WASM + conductor"]
async fn test_personal_bridge_health_check() {
    let dna_path = DnaPaths::personal();
    if !dna_path.exists() {
        eprintln!("Skipping: personal DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "personal", 1).await;
    let alice = &agents[0];

    let health: BridgeHealth = alice
        .call_zome_fn("personal_bridge", "health_check", ())
        .await;

    assert!(health.healthy, "Bridge should report healthy");
    assert!(
        health.domains.contains(&"identity".to_string()),
        "Should include identity domain"
    );
    assert!(
        health.domains.contains(&"health".to_string()),
        "Should include health domain"
    );
    assert!(
        health.domains.contains(&"credential".to_string()),
        "Should include credential domain"
    );
}

/// Test: Broadcast an event via personal bridge.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled personal WASM + conductor"]
async fn test_personal_bridge_broadcast_event() {
    let dna_path = DnaPaths::personal();
    if !dna_path.exists() {
        eprintln!("Skipping: personal DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "personal", 1).await;
    let alice = &agents[0];

    let event = PersonalEventEntry {
        domain: "identity".into(),
        event_type: "profile_updated".into(),
        source_agent: format!("did:mycelix:{}", alice.agent_pubkey),
        payload: r#"{"field":"bio"}"#.into(),
        related_hashes: vec![],
    };

    let _: Record = alice
        .call_zome_fn("personal_bridge", "broadcast_event", event)
        .await;

    // Verify event is retrievable
    let events: Vec<Record> = alice
        .call_zome_fn(
            "personal_bridge",
            "get_domain_events",
            "identity".to_string(),
        )
        .await;

    assert!(
        !events.is_empty(),
        "Should have at least one identity domain event"
    );
}

/// Test: Get all events returns broadcast events.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires compiled personal WASM + conductor"]
async fn test_personal_bridge_get_all_events() {
    let dna_path = DnaPaths::personal();
    if !dna_path.exists() {
        eprintln!("Skipping: personal DNA not built at {:?}", dna_path);
        return;
    }

    let agents = setup_test_agents(&dna_path, "personal", 1).await;
    let alice = &agents[0];

    // Broadcast events to different domains
    for domain in &["identity", "health", "credential"] {
        let event = PersonalEventEntry {
            domain: domain.to_string(),
            event_type: "test_event".into(),
            source_agent: format!("did:mycelix:{}", alice.agent_pubkey),
            payload: "{}".into(),
            related_hashes: vec![],
        };

        let _: Record = alice
            .call_zome_fn("personal_bridge", "broadcast_event", event)
            .await;
    }

    let all_events: Vec<Record> = alice
        .call_zome_fn("personal_bridge", "get_all_events", ())
        .await;

    assert!(
        all_events.len() >= 3,
        "Should have at least 3 events (one per domain)"
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
        .call_zome_fn_on_role("personal", "personal_bridge", "resolve_did", alice_did)
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
        .call_zome_fn_on_role("personal", "personal_bridge", "is_did_active", alice_did)
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
        .call_zome_fn_on_role("personal", "personal_bridge", "get_matl_score", alice_did)
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
