// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Verifiable Credential zome sweettest integration tests.
//!
//! Tests W3C verifiable credential issuance, verification, presentations,
//! derived credentials with selective disclosure, and request workflows.
//!
//! Prerequisites:
//!   cd mycelix-identity && cargo build --release --target wasm32-unknown-unknown
//!   hc dna pack dna/ -o dna/mycelix_identity_dna.dna
//!
//! Run: cargo test -p mycelix-sweettest -- --ignored identity_verifiable_credential

extern crate holochain_serialized_bytes;

mod harness;

use harness::*;
use holochain::prelude::*;
use serial_test::serial;

// ---------------------------------------------------------------------------
// Mirror types for deserialization
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct IssueCredentialInput {
    credential: CredentialData,
    issuer_did: String,
    credential_subject: String,
    valid_from: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    valid_until: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mycelix_schema_id: Option<String>,
    claims: serde_json::Value,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct CredentialData {
    #[serde(rename = "@context")]
    context: Vec<String>,
    #[serde(rename = "type")]
    type_: Vec<String>,
    issuer: String,
    #[serde(rename = "credentialSubject")]
    credential_subject: serde_json::Value,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct VerificationResult {
    valid: bool,
    schema_status: String,
    checks: Vec<String>,
    message: String,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct CreatePresentationInput {
    holder_did: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    verifier_did: Option<String>,
    credentials: Vec<ActionHash>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presentation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    expires_at: Option<i64>,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct VerifyPresentationInput {
    presentation_hash: ActionHash,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct PresentationVerificationResult {
    valid: bool,
    holder_verified: bool,
    credentials_valid: bool,
    message: String,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct CreateDerivedInput {
    original_credential_id: ActionHash,
    subject_did: String,
    disclosed_claims: Vec<String>,
    derivation_purpose: String,
    holder_did: String,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct DerivedVerificationResult {
    valid: bool,
    original_exists: bool,
    original_not_revoked: bool,
    claims_subset: bool,
    message: String,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct RequestCredentialInput {
    issuer_did: String,
    subject_did: String,
    credential_types: Vec<String>,
    purpose: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    expires_at: Option<i64>,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct UpdateRequestStatusInput {
    request_id: ActionHash,
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    issued_credential_id: Option<ActionHash>,
}

#[derive(serde::Serialize, serde::Deserialize, SerializedBytes, Debug, Clone)]
struct CredentialStatusResponse {
    credential_id: String,
    is_revoked: bool,
    revocation_reason: Option<String>,
}

/// Helper: create W3C credential data
fn w3c_credential(issuer_did: &str, subject_did: &str) -> CredentialData {
    CredentialData {
        context: vec![
            "https://www.w3.org/ns/credentials/v2".into(),
            "https://www.w3.org/ns/credentials/examples/v2".into(),
        ],
        type_: vec![
            "VerifiableCredential".into(),
            "EmploymentCredential".into(),
        ],
        issuer: issuer_did.to_string(),
        credential_subject: serde_json::json!({
            "id": subject_did,
            "employmentStatus": "Active",
            "position": "Software Engineer",
            "department": "Engineering",
        }),
    }
}

/// Helper: get current unix timestamp
fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Test: Issue a W3C verifiable credential and retrieve it.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_issue_and_get_credential() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 1).await;
    let alice = &agents[0];

    let _: Record = alice.call_zome_fn("did_registry", "create_did", ()).await;
    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);
    let subject_did = "did:mycelix:subject-001";

    let input = IssueCredentialInput {
        credential: w3c_credential(&alice_did, subject_did),
        issuer_did: alice_did.clone(),
        credential_subject: subject_did.into(),
        valid_from: now_secs(),
        valid_until: Some(now_secs() + 86400 * 365), // 1 year
        mycelix_schema_id: None,
        claims: serde_json::json!({
            "employmentStatus": "Active",
            "position": "Software Engineer",
        }),
    };
    let cred_record: Record = alice
        .call_zome_fn("verifiable_credential", "issue_credential", input)
        .await;

    let cred_hash = cred_record.action_hashed().hash.clone();

    // Retrieve credential
    let retrieved: Option<Record> = alice
        .call_zome_fn("verifiable_credential", "get_credential", cred_hash.to_string())
        .await;

    assert!(retrieved.is_some(), "Should retrieve the issued credential");
}

/// Test: Verify a credential passes validation.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_verify_credential() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 1).await;
    let alice = &agents[0];

    let _: Record = alice.call_zome_fn("did_registry", "create_did", ()).await;
    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);
    let subject_did = "did:mycelix:subject-verify";

    let input = IssueCredentialInput {
        credential: w3c_credential(&alice_did, subject_did),
        issuer_did: alice_did.clone(),
        credential_subject: subject_did.into(),
        valid_from: now_secs(),
        valid_until: None,
        mycelix_schema_id: None,
        claims: serde_json::json!({"role": "tester"}),
    };
    let cred_record: Record = alice
        .call_zome_fn("verifiable_credential", "issue_credential", input)
        .await;

    let cred_id = cred_record.action_hashed().hash.to_string();

    let result: VerificationResult = alice
        .call_zome_fn("verifiable_credential", "verify_credential", cred_id)
        .await;

    assert!(result.valid, "Freshly issued credential should be valid: {}", result.message);
}

/// Test: Cross-agent credential visibility via DHT.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_credential_cross_agent_visibility() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 2).await;
    let alice = &agents[0];
    let bob = &agents[1];

    for agent in &agents {
        let _: Record = agent.call_zome_fn("did_registry", "create_did", ()).await;
    }

    wait_for_dht_sync().await;

    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);
    let bob_did = format!("did:mycelix:{}", bob.agent_pubkey);

    // Alice issues credential for Bob
    let input = IssueCredentialInput {
        credential: w3c_credential(&alice_did, &bob_did),
        issuer_did: alice_did.clone(),
        credential_subject: bob_did.clone(),
        valid_from: now_secs(),
        valid_until: None,
        mycelix_schema_id: None,
        claims: serde_json::json!({"verified": true}),
    };
    let _: Record = alice
        .call_zome_fn("verifiable_credential", "issue_credential", input)
        .await;

    wait_for_dht_sync().await;

    // Bob should see credentials for himself
    let creds: Vec<Record> = bob
        .call_zome_fn("verifiable_credential", "get_credentials_for_subject", bob_did.clone())
        .await;

    assert!(
        !creds.is_empty(),
        "Bob should see the credential Alice issued for him via DHT"
    );
}

/// Test: Create a verifiable presentation from credentials.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_create_and_verify_presentation() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 1).await;
    let alice = &agents[0];

    let _: Record = alice.call_zome_fn("did_registry", "create_did", ()).await;
    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);

    // Issue a credential
    let issue_input = IssueCredentialInput {
        credential: w3c_credential(&alice_did, &alice_did),
        issuer_did: alice_did.clone(),
        credential_subject: alice_did.clone(),
        valid_from: now_secs(),
        valid_until: None,
        mycelix_schema_id: None,
        claims: serde_json::json!({"status": "active"}),
    };
    let cred_record: Record = alice
        .call_zome_fn("verifiable_credential", "issue_credential", issue_input)
        .await;

    let cred_hash = cred_record.action_hashed().hash.clone();

    // Create presentation
    let pres_input = CreatePresentationInput {
        holder_did: alice_did.clone(),
        verifier_did: Some("did:mycelix:verifier-001".into()),
        credentials: vec![cred_hash],
        presentation_id: Some("pres-001".into()),
        expires_at: Some(now_secs() + 3600),
    };
    let pres_record: Record = alice
        .call_zome_fn("verifiable_credential", "create_presentation", pres_input)
        .await;

    let pres_hash = pres_record.action_hashed().hash.clone();

    // Verify presentation
    let verify_input = VerifyPresentationInput {
        presentation_hash: pres_hash,
    };
    let result: PresentationVerificationResult = alice
        .call_zome_fn("verifiable_credential", "verify_presentation", verify_input)
        .await;

    assert!(result.valid, "Presentation should be valid: {}", result.message);
    assert!(result.holder_verified, "Holder should be verified");
}

/// Test: Derived credential with selective disclosure.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_derived_credential_selective_disclosure() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 1).await;
    let alice = &agents[0];

    let _: Record = alice.call_zome_fn("did_registry", "create_did", ()).await;
    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);

    // Issue full credential with multiple claims
    let issue_input = IssueCredentialInput {
        credential: w3c_credential(&alice_did, &alice_did),
        issuer_did: alice_did.clone(),
        credential_subject: alice_did.clone(),
        valid_from: now_secs(),
        valid_until: None,
        mycelix_schema_id: None,
        claims: serde_json::json!({
            "employmentStatus": "Active",
            "position": "Engineer",
            "salary": "confidential",
            "department": "Engineering",
        }),
    };
    let cred_record: Record = alice
        .call_zome_fn("verifiable_credential", "issue_credential", issue_input)
        .await;

    let cred_hash = cred_record.action_hashed().hash.clone();

    // Create derived credential disclosing only position and department
    let derived_input = CreateDerivedInput {
        original_credential_id: cred_hash.clone(),
        subject_did: alice_did.clone(),
        disclosed_claims: vec!["position".into(), "department".into()],
        derivation_purpose: "Job reference".into(),
        holder_did: alice_did.clone(),
    };
    let derived_record: Record = alice
        .call_zome_fn("verifiable_credential", "create_derived_credential", derived_input)
        .await;

    let derived_hash = derived_record.action_hashed().hash.clone();

    // Verify derived credential
    let result: DerivedVerificationResult = alice
        .call_zome_fn("verifiable_credential", "verify_derived_credential", derived_hash)
        .await;

    assert!(result.valid, "Derived credential should be valid: {}", result.message);
    assert!(result.original_exists, "Original credential should exist");
    assert!(result.original_not_revoked, "Original should not be revoked");
    assert!(result.claims_subset, "Disclosed claims should be subset of original");
}

/// Test: Credential request workflow.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_credential_request_workflow() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 2).await;
    let alice = &agents[0]; // Issuer
    let bob = &agents[1]; // Requester

    for agent in &agents {
        let _: Record = agent.call_zome_fn("did_registry", "create_did", ()).await;
    }

    wait_for_dht_sync().await;

    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);
    let bob_did = format!("did:mycelix:{}", bob.agent_pubkey);

    // Bob requests a credential from Alice
    let request_input = RequestCredentialInput {
        issuer_did: alice_did.clone(),
        subject_did: bob_did.clone(),
        credential_types: vec!["EmploymentCredential".into()],
        purpose: "Employment verification for bank".into(),
        expires_at: None,
    };
    let request_record: Record = bob
        .call_zome_fn("verifiable_credential", "request_credential", request_input)
        .await;

    let request_hash = request_record.action_hashed().hash.clone();

    wait_for_dht_sync().await;

    // Alice checks pending requests
    let pending: Vec<Record> = alice
        .call_zome_fn("verifiable_credential", "get_pending_requests", alice_did.clone())
        .await;

    assert!(
        !pending.is_empty(),
        "Alice should see Bob's credential request"
    );

    // Alice approves the request
    let update_input = UpdateRequestStatusInput {
        request_id: request_hash.clone(),
        status: "Approved".into(),
        issued_credential_id: None,
    };
    let _: Record = alice
        .call_zome_fn("verifiable_credential", "update_request_status", update_input)
        .await;
}

/// Test: My issued/received credentials.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_my_credentials_views() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 1).await;
    let alice = &agents[0];

    let _: Record = alice.call_zome_fn("did_registry", "create_did", ()).await;
    let alice_did = format!("did:mycelix:{}", alice.agent_pubkey);

    // Issue credential where Alice is both issuer and subject
    let input = IssueCredentialInput {
        credential: w3c_credential(&alice_did, &alice_did),
        issuer_did: alice_did.clone(),
        credential_subject: alice_did.clone(),
        valid_from: now_secs(),
        valid_until: None,
        mycelix_schema_id: None,
        claims: serde_json::json!({"test": true}),
    };
    let _: Record = alice
        .call_zome_fn("verifiable_credential", "issue_credential", input)
        .await;

    // Check my issued credentials
    let issued: Vec<Record> = alice
        .call_zome_fn("verifiable_credential", "get_my_issued_credentials", ())
        .await;

    assert!(
        !issued.is_empty(),
        "Alice should see her issued credential"
    );

    // Check my credentials (as subject)
    let mine: Vec<Record> = alice
        .call_zome_fn("verifiable_credential", "get_my_credentials", ())
        .await;

    assert!(
        !mine.is_empty(),
        "Alice should see credentials where she is subject"
    );
}

/// Test: API version check.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_verifiable_credential_api_version() {
    let agents = setup_test_agents(&DnaPaths::identity(), "mycelix-identity", 1).await;
    let alice = &agents[0];

    let version: u16 = alice
        .call_zome_fn("verifiable_credential", "get_api_version", ())
        .await;

    assert_eq!(version, 1, "Verifiable credential API version should be 1");
}
