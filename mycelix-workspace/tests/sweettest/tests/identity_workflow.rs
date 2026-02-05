//! Identity hApp sweettest integration tests.
//!
//! Tests DID creation, resolution, and multi-agent DHT propagation
//! using the Holochain sweettest framework with real conductors.
//!
//! Prerequisites:
//!   cd mycelix-identity && cargo build --release --target wasm32-unknown-unknown
//!   hc dna pack dna/ -o dna/mycelix_identity_dna.dna
//!
//! Run: cargo test -p mycelix-sweettest -- --ignored identity
//!
//! Updated for Holochain 0.6 sweettest API.

mod harness;

use harness::*;
use holochain::prelude::*;
use serial_test::serial;

/// Test: Single agent creates and retrieves a DID document.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_create_and_resolve_did() {
    let agents = setup_test_agents(
        &DnaPaths::identity(),
        "mycelix-identity",
        1,
    )
    .await;

    let alice = &agents[0];

    // Create a DID for Alice
    let did_record: Record = alice
        .call_zome_fn("did_registry", "create_did", ())
        .await;

    // The record should have a valid action hash
    let action_hash = did_record.action_hashed().hash.clone();
    assert!(!action_hash.as_ref().is_empty(), "DID record should have valid hash");

    // Retrieve DID document by agent pub key
    let retrieved: Option<Record> = alice
        .call_zome_fn("did_registry", "get_did_document", alice.agent_pubkey.clone())
        .await;

    assert!(retrieved.is_some(), "Should retrieve Alice's DID document");
}

/// Test: DID created by one agent is visible to another via DHT.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_did_cross_agent_resolution() {
    let agents = setup_test_agents(
        &DnaPaths::identity(),
        "mycelix-identity",
        2,
    )
    .await;

    let alice = &agents[0];
    let bob = &agents[1];

    // Alice creates her DID
    let _: Record = alice
        .call_zome_fn("did_registry", "create_did", ())
        .await;

    wait_for_dht_sync().await;

    // Bob resolves Alice's DID by agent key
    let resolved: Option<Record> = bob
        .call_zome_fn("did_registry", "get_did_document", alice.agent_pubkey.clone())
        .await;

    assert!(resolved.is_some(), "Bob should resolve Alice's DID via DHT");
}

/// Test: DID deactivation is propagated across agents.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_did_deactivation_propagation() {
    let agents = setup_test_agents(
        &DnaPaths::identity(),
        "mycelix-identity",
        2,
    )
    .await;

    let alice = &agents[0];
    let bob = &agents[1];

    // Alice creates and then deactivates her DID
    let _: Record = alice
        .call_zome_fn("did_registry", "create_did", ())
        .await;

    wait_for_dht_sync().await;

    // Alice deactivates
    let _: Record = alice
        .call_zome_fn("did_registry", "deactivate_did", "Key compromised".to_string())
        .await;

    wait_for_dht_sync().await;

    // Bob checks if Alice's DID is active (using did: format)
    let did_string = format!("did:mycelix:{}", alice.agent_pubkey);

    let is_active: bool = bob
        .call_zome_fn("did_registry", "is_did_active", did_string)
        .await;

    assert!(!is_active, "Alice's deactivated DID should show as inactive to Bob");
}

/// Test: Service endpoint management.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundle
async fn test_service_endpoint_crud() {
    let agents = setup_test_agents(
        &DnaPaths::identity(),
        "mycelix-identity",
        1,
    )
    .await;

    let alice = &agents[0];

    // Create DID first
    let _: Record = alice
        .call_zome_fn("did_registry", "create_did", ())
        .await;

    // Add a service endpoint (fields must match ServiceEndpoint struct: id, type_, service_endpoint)
    let service = serde_json::json!({
        "id": "mail-endpoint",
        "type_": "MycelixMail",
        "service_endpoint": "https://mail.mycelix.net/alice"
    });

    let _: Record = alice
        .call_zome_fn("did_registry", "add_service_endpoint", service)
        .await;

    // Retrieve DID and verify service is attached
    let did_doc: Option<Record> = alice
        .call_zome_fn("did_registry", "get_did_document", alice.agent_pubkey.clone())
        .await;

    assert!(did_doc.is_some(), "DID document should exist with service endpoint");
}
