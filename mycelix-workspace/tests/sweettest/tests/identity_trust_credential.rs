// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Trust Credential zome sweettest integration tests.
//!
//! Tests K-Vector trust credential issuance, verification, selective
//! disclosure presentations, and attestation request workflows.
//!
//! Prerequisites:
//!   cd mycelix-identity && cargo build --release --target wasm32-unknown-unknown
//!   hc dna pack dna/ -o dna/mycelix_identity_dna.dna
//!
//! Run: cargo test -p mycelix-sweettest -- --ignored identity_trust_credential

extern crate holochain_serialized_bytes;

mod harness;

use harness::*;
use holochain::prelude::*;
use serial_test::serial;

// ---------------------------------------------------------------------------
// Mirror types for deserialization
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct IssueTrustCredentialInput {
    subject_did: String,
    issuer_did: String,
    kvector_commitment: Vec<u8>,
    range_proof: Vec<u8>,
    trust_score_lower: f64,
    trust_score_upper: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    expires_at: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    supersedes: Option<String>,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct SelfAttestTrustInput {
    subject_did: String,
    kvector_commitment: Vec<u8>,
    range_proof: Vec<u8>,
    trust_score_lower: f64,
    trust_score_upper: f64,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct RevokeCredentialInput {
    credential_id: String,
    subject_did: String,
    reason: String,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct CreatePresentationInput {
    credential_id: String,
    subject_did: String,
    disclosed_tier: bool,
    disclose_range: bool,
    trust_range: Option<(f64, f64)>,
    presentation_proof: Vec<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    verifier_did: Option<String>,
    purpose: String,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct RequestAttestationInput {
    requester_did: String,
    subject_did: String,
    components: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_trust_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min_tier: Option<String>,
    purpose: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    expires_at: Option<i64>,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct FulfillAttestationInput {
    request_id: String,
    subject_did: String,
    kvector_commitment: Vec<u8>,
    range_proof: Vec<u8>,
    trust_score_lower: f64,
    trust_score_upper: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    expires_at: Option<i64>,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct FulfillAttestationResult {
    credential_action_hash: ActionHash,
    request_updated: bool,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct DeclineAttestationInput {
    request_id: String,
    subject_did: String,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct VerificationResult {
    commitment_valid: bool,
    tier_consistent: bool,
    not_revoked: bool,
    not_expired: bool,
    proof_format_valid: bool,
    message: String,
}

/// Trust tiers matching the zome's TrustTier enum
#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
enum TrustTier {
    Observer,
    Basic,
    Standard,
    Elevated,
    Guardian,
}

/// Helper: generate a 32-byte BLAKE2b-like commitment (non-zero for validation)
fn mock_commitment() -> Vec<u8> {
    vec![0x42u8; 32]
}

/// Helper: generate a mock range proof (non-empty for validation)
fn mock_range_proof() -> Vec<u8> {
    vec![0x01, 0x02, 0x03, 0x04]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Test: Issue a trust credential and verify it.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_issue_and_verify_trust_credential() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 2).await;
    let alice = &agents[0];
    let bob = &agents[1];

    // Create DIDs
    for agent in &agents {
        let _: Record = agent.call_zome_fn("did_registry", "create_did", ()).await;
    }

    wait_for_dht_sync().await;

    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);
    let bob_did = format!("did:mycelix:{}", bob.agent_pubkey);

    // Alice issues a trust credential for Bob (Standard tier: 0.4-0.6)
    let issue_input = IssueTrustCredentialInput {
        subject_did: bob_did.clone(),
        issuer_did: alice_did.clone(),
        kvector_commitment: mock_commitment(),
        range_proof: mock_range_proof(),
        trust_score_lower: 0.45,
        trust_score_upper: 0.55,
        expires_at: None,
        supersedes: None,
    };
    let cred_record: Record = alice
        .call_zome_fn("trust_credential", "issue_trust_credential", issue_input)
        .await;

    let cred_id = cred_record.action_hashed().hash.to_string();

    // Verify the credential
    let verification: VerificationResult = alice
        .call_zome_fn("trust_credential", "verify_credential", cred_id)
        .await;

    assert!(verification.commitment_valid, "Commitment should be valid (32 bytes, non-zero)");
    assert!(verification.tier_consistent, "Trust tier should be consistent with score range");
    assert!(verification.not_revoked, "Credential should not be revoked");
}

/// Test: Self-attest trust and retrieve subject credentials.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_self_attest_and_list_credentials() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 1).await;
    let alice = &agents[0];

    let _: Record = alice.call_zome_fn("did_registry", "create_did", ()).await;
    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);

    // Alice self-attests her K-Vector (Basic tier: 0.3-0.4)
    let self_attest_input = SelfAttestTrustInput {
        subject_did: alice_did.clone(),
        kvector_commitment: mock_commitment(),
        range_proof: mock_range_proof(),
        trust_score_lower: 0.3,
        trust_score_upper: 0.4,
    };
    let _: Record = alice
        .call_zome_fn("trust_credential", "self_attest_trust", self_attest_input)
        .await;

    // List Alice's credentials
    let credentials: Vec<Record> = alice
        .call_zome_fn("trust_credential", "get_subject_credentials", alice_did.clone())
        .await;

    assert!(
        !credentials.is_empty(),
        "Alice should have at least one trust credential (self-attestation)"
    );
}

/// Test: Revoke a trust credential and verify status.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_revoke_trust_credential() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 2).await;
    let alice = &agents[0];
    let bob = &agents[1];

    for agent in &agents {
        let _: Record = agent.call_zome_fn("did_registry", "create_did", ()).await;
    }

    wait_for_dht_sync().await;

    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);
    let bob_did = format!("did:mycelix:{}", bob.agent_pubkey);

    // Issue credential
    let issue_input = IssueTrustCredentialInput {
        subject_did: bob_did.clone(),
        issuer_did: alice_did.clone(),
        kvector_commitment: mock_commitment(),
        range_proof: mock_range_proof(),
        trust_score_lower: 0.6,
        trust_score_upper: 0.8,
        expires_at: None,
        supersedes: None,
    };
    let cred_record: Record = alice
        .call_zome_fn("trust_credential", "issue_trust_credential", issue_input)
        .await;

    let cred_id = cred_record.action_hashed().hash.to_string();

    // Revoke it
    let revoke_input = RevokeCredentialInput {
        credential_id: cred_id.clone(),
        subject_did: bob_did.clone(),
        reason: "Trust level downgraded".into(),
    };
    let _: Record = alice
        .call_zome_fn("trust_credential", "revoke_credential", revoke_input)
        .await;

    // Verify shows not_revoked = false
    let verification: VerificationResult = alice
        .call_zome_fn("trust_credential", "verify_credential", cred_id)
        .await;

    assert!(!verification.not_revoked, "Revoked credential should show not_revoked=false");
}

/// Test: Attestation request and fulfillment flow.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_attestation_request_and_fulfill() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 2).await;
    let alice = &agents[0];
    let bob = &agents[1];

    for agent in &agents {
        let _: Record = agent.call_zome_fn("did_registry", "create_did", ()).await;
    }

    wait_for_dht_sync().await;

    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);
    let bob_did = format!("did:mycelix:{}", bob.agent_pubkey);

    // Alice requests attestation from Bob
    let request_input = RequestAttestationInput {
        requester_did: alice_did.clone(),
        subject_did: bob_did.clone(),
        components: vec!["identity".into(), "competence".into()],
        min_trust_score: Some(0.4),
        min_tier: None,
        purpose: "Employment verification".into(),
        expires_at: None,
    };
    let request_record: Record = alice
        .call_zome_fn("trust_credential", "request_attestation", request_input)
        .await;

    let request_id = request_record.action_hashed().hash.to_string();

    wait_for_dht_sync().await;

    // Bob checks pending requests
    let pending: Vec<Record> = bob
        .call_zome_fn("trust_credential", "get_pending_requests", bob_did.clone())
        .await;

    assert!(
        !pending.is_empty(),
        "Bob should see Alice's attestation request"
    );

    // Bob fulfills the attestation
    let fulfill_input = FulfillAttestationInput {
        request_id: request_id.clone(),
        subject_did: bob_did.clone(),
        kvector_commitment: mock_commitment(),
        range_proof: mock_range_proof(),
        trust_score_lower: 0.5,
        trust_score_upper: 0.7,
        expires_at: None,
    };
    let result: FulfillAttestationResult = bob
        .call_zome_fn("trust_credential", "fulfill_attestation", fulfill_input)
        .await;

    assert!(result.request_updated, "Request should be marked as fulfilled");
}

/// Test: Decline an attestation request.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_decline_attestation_request() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 2).await;
    let alice = &agents[0];
    let bob = &agents[1];

    for agent in &agents {
        let _: Record = agent.call_zome_fn("did_registry", "create_did", ()).await;
    }

    wait_for_dht_sync().await;

    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);
    let bob_did = format!("did:mycelix:{}", bob.agent_pubkey);

    // Alice requests attestation from Bob
    let request_input = RequestAttestationInput {
        requester_did: alice_did.clone(),
        subject_did: bob_did.clone(),
        components: vec!["identity".into()],
        min_trust_score: None,
        min_tier: None,
        purpose: "Test decline flow".into(),
        expires_at: None,
    };
    let request_record: Record = alice
        .call_zome_fn("trust_credential", "request_attestation", request_input)
        .await;

    let request_id = request_record.action_hashed().hash.to_string();

    wait_for_dht_sync().await;

    // Bob declines the request
    let decline_input = DeclineAttestationInput {
        request_id: request_id.clone(),
        subject_did: bob_did.clone(),
    };
    let _: Record = bob
        .call_zome_fn("trust_credential", "decline_attestation", decline_input)
        .await;

    // Check pending requests — should be empty now
    let pending: Vec<Record> = bob
        .call_zome_fn("trust_credential", "get_pending_requests", bob_did.clone())
        .await;

    assert!(
        pending.is_empty(),
        "After declining, no pending requests should remain for Bob"
    );
}

/// Test: API version check.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_trust_credential_api_version() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 1).await;
    let alice = &agents[0];

    let version: u16 = alice
        .call_zome_fn("trust_credential", "get_api_version", ())
        .await;

    assert_eq!(version, 1, "Trust credential API version should be 1");
}
