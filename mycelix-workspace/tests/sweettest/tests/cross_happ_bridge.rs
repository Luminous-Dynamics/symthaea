// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-hApp Bridge Protocol sweettest integration tests.
//!
//! Tests the bridge protocol that enables communication between
//! different Mycelix hApps running on the same conductor.
//!
//! Prerequisites:
//!   Build and package identity + governance DNAs.
//!
//! Run: cargo test -p mycelix-sweettest -- --ignored bridge
//!
//! Updated for Holochain 0.6 sweettest API.

mod harness;

use harness::*;
use holochain::prelude::*;
use holochain::sweettest::*;
use serial_test::serial;

/// Multi-hApp agent with cells for both identity and governance.
struct MultiHappAgent {
    conductor: SweetConductor,
    identity_cell: SweetCell,
    governance_cell: SweetCell,
    agent_pubkey: AgentPubKey,
}

impl MultiHappAgent {
    /// Make a zome call to the identity hApp.
    async fn call_identity<I, O>(&self, fn_name: &str, input: I) -> O
    where
        I: serde::Serialize + std::fmt::Debug,
        O: serde::de::DeserializeOwned + std::fmt::Debug,
    {
        let zome = self.identity_cell.zome("did_registry");
        self.conductor.call(&zome, fn_name, input).await
    }

    /// Make a zome call to the governance hApp.
    async fn call_governance<I, O>(&self, zome_name: &str, fn_name: &str, input: I) -> O
    where
        I: serde::Serialize + std::fmt::Debug,
        O: serde::de::DeserializeOwned + std::fmt::Debug,
    {
        let zome = self.governance_cell.zome(zome_name);
        self.conductor.call(&zome, fn_name, input).await
    }
}

/// Set up a conductor with multiple hApps installed (identity + governance).
async fn setup_multi_happ_conductor() -> MultiHappAgent {
    let identity_dna = SweetDnaFile::from_bundle(&DnaPaths::identity())
        .await
        .expect("Identity DNA should exist");

    let governance_dna = SweetDnaFile::from_bundle(&DnaPaths::governance())
        .await
        .expect("Governance DNA should exist");

    let mut conductor = SweetConductor::from_standard_config().await;

    // Install both hApps for the same agent
    let identity_app = conductor
        .setup_app("mycelix-identity", &[identity_dna])
        .await
        .unwrap();

    let governance_app = conductor
        .setup_app("mycelix-governance", &[governance_dna])
        .await
        .unwrap();

    let identity_cell = identity_app.cells()[0].clone();
    let governance_cell = governance_app.cells()[0].clone();
    let agent_pubkey = identity_cell.agent_pubkey().clone();

    MultiHappAgent {
        conductor,
        identity_cell,
        governance_cell,
        agent_pubkey,
    }
}

/// Test: Agent creates DID in identity hApp, then uses it in governance.
///
/// This verifies that a single agent can operate across multiple hApps
/// on the same conductor, which is the foundation of bridge protocol.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundles
async fn test_identity_to_governance_flow() {
    let agent = setup_multi_happ_conductor().await;

    // Step 1: Create DID in identity hApp
    let did_record: Record = agent.call_identity("create_did", ()).await;

    assert!(
        !did_record.action_hashed().hash.as_ref().is_empty(),
        "DID should be created in identity hApp"
    );

    // Step 2: Create a governance proposal (requires an identity)
    let now = Timestamp::now();
    let voting_ends = Timestamp::from_micros(now.as_micros() + 24 * 60 * 60 * 1_000_000);

    let proposal_input = serde_json::json!({
        "id": "MIP-BRIDGE-001",
        "title": "Test cross-hApp integration",
        "description": "Verifying identity-governance bridge",
        "proposal_type": "Standard",
        "author": format!("did:mycelix:{}", agent.agent_pubkey),
        "status": "Draft",
        "actions": "{}",
        "discussion_url": null,
        "voting_starts": now,
        "voting_ends": voting_ends,
        "created": now,
        "updated": now,
        "version": 1
    });

    let proposal_record: Record = agent
        .call_governance("proposals", "create_proposal", proposal_input)
        .await;

    assert!(
        !proposal_record.action_hashed().hash.as_ref().is_empty(),
        "Proposal should be created in governance hApp"
    );
}

/// Test: Two agents with identity + governance see each other's data.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore] // Requires DNA bundles
async fn test_multi_agent_multi_happ_sync() {
    let identity_dna = SweetDnaFile::from_bundle(&DnaPaths::identity())
        .await
        .expect("Identity DNA should exist");

    let governance_dna = SweetDnaFile::from_bundle(&DnaPaths::governance())
        .await
        .expect("Governance DNA should exist");

    // Set up 2 conductors, each with both hApps
    let mut alice_conductor = SweetConductor::from_standard_config().await;
    let mut bob_conductor = SweetConductor::from_standard_config().await;

    let alice_id_app = alice_conductor
        .setup_app("mycelix-identity", &[identity_dna.clone()])
        .await
        .unwrap();
    let _alice_gov_app = alice_conductor
        .setup_app("mycelix-governance", &[governance_dna.clone()])
        .await
        .unwrap();

    let bob_id_app = bob_conductor
        .setup_app("mycelix-identity", &[identity_dna])
        .await
        .unwrap();
    let _bob_gov_app = bob_conductor
        .setup_app("mycelix-governance", &[governance_dna])
        .await
        .unwrap();

    let alice_identity_cell = alice_id_app.cells()[0].clone();
    let bob_identity_cell = bob_id_app.cells()[0].clone();
    let alice_agent = alice_identity_cell.agent_pubkey().clone();

    // Exchange peers for DHT sync
    SweetConductor::exchange_peer_info([&alice_conductor, &bob_conductor]).await;

    // Alice creates her DID
    let alice_zome = alice_identity_cell.zome("did_registry");
    let _: Record = alice_conductor.call(&alice_zome, "create_did", ()).await;

    // Wait for DHT propagation with retry (gossip can take variable time)
    let bob_zome = bob_identity_cell.zome("did_registry");
    let mut alice_did: Option<Record> = None;
    for _ in 0..10 {
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        alice_did = bob_conductor
            .call(&bob_zome, "get_did_document", alice_agent.clone())
            .await;
        if alice_did.is_some() {
            break;
        }
    }

    assert!(
        alice_did.is_some(),
        "Bob should see Alice's DID via DHT in multi-hApp setup"
    );
}
