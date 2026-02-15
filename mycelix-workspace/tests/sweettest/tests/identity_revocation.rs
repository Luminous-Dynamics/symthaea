//! Revocation zome sweettest integration tests.
//!
//! Tests credential revocation, suspension, reinstatement, and batch operations
//! using the Holochain sweettest framework with real conductors.
//!
//! Prerequisites:
//!   cd mycelix-identity && cargo build --release --target wasm32-unknown-unknown
//!   hc dna pack dna/ -o dna/mycelix_identity_dna.dna
//!
//! Run: cargo test -p mycelix-sweettest -- --ignored identity_revocation

extern crate holochain_serialized_bytes;

mod harness;

use harness::*;
use holochain::prelude::*;
use serial_test::serial;

// ---------------------------------------------------------------------------
// Mirror types for deserialization
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct RevokeInput {
    credential_id: String,
    issuer_did: String,
    reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    effective_from: Option<i64>,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct SuspendInput {
    credential_id: String,
    issuer_did: String,
    reason: String,
    suspension_end: i64,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct ReinstateInput {
    credential_id: String,
    issuer_did: String,
    reason: String,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct RevocationCheckResult {
    credential_id: String,
    status: String,
    reason: Option<String>,
    checked_at: i64,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct CreateRevocationListInput {
    id: String,
    issuer_did: String,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct BatchRevokeInput {
    credential_ids: Vec<String>,
    issuer_did: String,
    reason: String,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct BatchSuspendInput {
    credential_ids: Vec<String>,
    issuer_did: String,
    reason: String,
    suspension_end: i64,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct BatchRevokeResult {
    revoked_count: u32,
    failed_count: u32,
    revoked: Vec<String>,
    failed: Vec<String>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Test: Revoke a credential and verify status changes.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_revoke_credential_lifecycle() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 1).await;
    let alice = &agents[0];

    // First create a DID for Alice (required for issuer_did)
    let _: Record = alice.call_zome_fn("did_registry", "create_did", ()).await;
    let issuer_did = format!("did:mycelix:{}", alice.agent_pubkey);

    // Create a revocation list
    let list_input = CreateRevocationListInput {
        id: "alice-revocation-list-1".into(),
        issuer_did: issuer_did.clone(),
    };
    let _: Record = alice
        .call_zome_fn("revocation", "create_revocation_list", list_input)
        .await;

    // Revoke a credential
    let credential_id = "cred-001".to_string();
    let revoke_input = RevokeInput {
        credential_id: credential_id.clone(),
        issuer_did: issuer_did.clone(),
        reason: "Key compromise detected".into(),
        effective_from: None,
    };
    let _: Record = alice
        .call_zome_fn("revocation", "revoke_credential", revoke_input)
        .await;

    // Check revocation status
    let status: RevocationCheckResult = alice
        .call_zome_fn("revocation", "check_revocation_status", credential_id.clone())
        .await;

    assert_eq!(status.credential_id, credential_id);
    assert_eq!(status.status, "Revoked", "Credential should be revoked");
    assert!(status.reason.is_some(), "Revocation should include reason");
}

/// Test: Suspend and reinstate a credential.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_suspend_and_reinstate_credential() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 1).await;
    let alice = &agents[0];

    let _: Record = alice.call_zome_fn("did_registry", "create_did", ()).await;
    let issuer_did = format!("did:mycelix:{}", alice.agent_pubkey);

    // Suspend a credential (1 hour from now)
    let credential_id = "cred-suspend-001".to_string();
    let suspend_input = SuspendInput {
        credential_id: credential_id.clone(),
        issuer_did: issuer_did.clone(),
        reason: "Investigation pending".into(),
        suspension_end: (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64)
            + 3600,
    };
    let _: Record = alice
        .call_zome_fn("revocation", "suspend_credential", suspend_input)
        .await;

    // Check status is suspended
    let status: RevocationCheckResult = alice
        .call_zome_fn("revocation", "check_revocation_status", credential_id.clone())
        .await;
    assert_eq!(status.status, "Suspended", "Credential should be suspended");

    // Reinstate the credential
    let reinstate_input = ReinstateInput {
        credential_id: credential_id.clone(),
        issuer_did: issuer_did.clone(),
        reason: "Investigation cleared".into(),
    };
    let _: Record = alice
        .call_zome_fn("revocation", "reinstate_credential", reinstate_input)
        .await;

    // Check status is active again
    let status: RevocationCheckResult = alice
        .call_zome_fn("revocation", "check_revocation_status", credential_id.clone())
        .await;
    assert_eq!(status.status, "Active", "Credential should be active after reinstatement");
}

/// Test: Revocation status propagates across agents via DHT.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_revocation_cross_agent_propagation() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 2).await;
    let alice = &agents[0];
    let bob = &agents[1];

    let _: Record = alice.call_zome_fn("did_registry", "create_did", ()).await;
    let issuer_did = format!("did:mycelix:{}", alice.agent_pubkey);

    // Alice revokes a credential
    let credential_id = "cred-cross-001".to_string();
    let revoke_input = RevokeInput {
        credential_id: credential_id.clone(),
        issuer_did: issuer_did.clone(),
        reason: "Expired".into(),
        effective_from: None,
    };
    let _: Record = alice
        .call_zome_fn("revocation", "revoke_credential", revoke_input)
        .await;

    wait_for_dht_sync().await;

    // Bob checks the revocation status
    let status: RevocationCheckResult = bob
        .call_zome_fn("revocation", "check_revocation_status", credential_id.clone())
        .await;

    assert_eq!(status.status, "Revoked", "Bob should see Alice's revocation via DHT");
}

/// Test: Batch revocation of multiple credentials.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_batch_revoke_credentials() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 1).await;
    let alice = &agents[0];

    let _: Record = alice.call_zome_fn("did_registry", "create_did", ()).await;
    let issuer_did = format!("did:mycelix:{}", alice.agent_pubkey);

    let credential_ids: Vec<String> = (1..=5).map(|i| format!("batch-cred-{:03}", i)).collect();

    let batch_input = BatchRevokeInput {
        credential_ids: credential_ids.clone(),
        issuer_did: issuer_did.clone(),
        reason: "Batch key rotation".into(),
    };
    let result: BatchRevokeResult = alice
        .call_zome_fn("revocation", "batch_revoke_credentials", batch_input)
        .await;

    assert_eq!(result.revoked_count, 5, "All 5 credentials should be revoked");
    assert_eq!(result.failed_count, 0, "No failures expected");

    // Verify each is revoked
    let statuses: Vec<RevocationCheckResult> = alice
        .call_zome_fn("revocation", "batch_check_revocation", credential_ids)
        .await;

    for status in &statuses {
        assert_eq!(status.status, "Revoked", "Each credential should be revoked");
    }
}

/// Test: Issuer can list their revocations.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_get_revocations_by_issuer() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 1).await;
    let alice = &agents[0];

    let _: Record = alice.call_zome_fn("did_registry", "create_did", ()).await;
    let issuer_did = format!("did:mycelix:{}", alice.agent_pubkey);

    // Revoke two credentials
    for id in ["issuer-cred-001", "issuer-cred-002"] {
        let input = RevokeInput {
            credential_id: id.into(),
            issuer_did: issuer_did.clone(),
            reason: "Routine rotation".into(),
            effective_from: None,
        };
        let _: Record = alice
            .call_zome_fn("revocation", "revoke_credential", input)
            .await;
    }

    // List revocations by issuer
    let revocations: Vec<Record> = alice
        .call_zome_fn("revocation", "get_revocations_by_issuer", issuer_did.clone())
        .await;

    assert!(
        revocations.len() >= 2,
        "Should find at least 2 revocations for this issuer, got {}",
        revocations.len()
    );
}
