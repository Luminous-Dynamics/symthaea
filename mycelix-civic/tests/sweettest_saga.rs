// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Civic — Saga Sweettest
//!
//! End-to-end tests for the emergency response saga workflow via the
//! civic-bridge coordinator zome.
//!
//! ## Running
//! ```bash
//! cd mycelix-civic
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_saga -- --ignored --test-threads=1
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — avoid importing zome crates (duplicate WASM symbols)
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EmergencySagaInput {
    pub incident_id: String,
    pub latitude: f64,
    pub longitude: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SagaEntry {
    pub schema_version: u8,
    pub saga_json: String,
    pub initiator: String,
    pub created_at: Timestamp,
}

// ============================================================================
// DNA setup helper
// ============================================================================

fn civic_dna_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // tests/ → mycelix-civic/
    path.push("dna");
    path.push("mycelix_civic.dna");
    path
}

// ============================================================================
// Emergency Response Saga Tests
// ============================================================================

/// Tests that `start_emergency_saga` creates a saga entry on the DHT and
/// that `get_saga_status` can retrieve it.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor + unified hApp with civic+commons+finance roles"]
async fn test_emergency_response_saga_creates_entry() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();

    let apps = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap();
    let (alice,) = apps.into_tuple();

    // Start the emergency response saga
    let input = EmergencySagaInput {
        incident_id: "INC-2026-0001".to_string(),
        latitude: 32.9483,
        longitude: -96.7299,
    };

    let saga_record: Record = conductor
        .call(&alice.zome("civic_bridge"), "start_emergency_saga", input)
        .await;

    // Verify the record was authored by alice
    assert_eq!(saga_record.action().author(), alice.agent_pubkey());

    // Retrieve the saga by its action hash
    let saga_hash = saga_record.action_address().clone();
    let fetched: Record = conductor
        .call(&alice.zome("civic_bridge"), "get_saga_status", saga_hash)
        .await;

    // Verify the fetched record matches what was created
    assert_eq!(
        fetched.action_address(),
        saga_record.action_address(),
        "get_saga_status should return the same saga record"
    );

    // Deserialize the entry and check saga JSON contents
    let entry: SagaEntry = fetched
        .entry()
        .to_app_option()
        .expect("deserialize SagaEntry")
        .expect("entry should be present");

    assert_eq!(entry.schema_version, 1);

    // Verify the saga JSON contains expected emergency-response steps
    assert!(
        entry.saga_json.contains("emergency-response"),
        "saga_json should contain the saga name"
    );
    assert!(
        entry.saga_json.contains("INC-2026-0001"),
        "saga_json should contain the incident ID"
    );
    assert!(
        entry.saga_json.contains("declare_disaster"),
        "saga_json should contain the declare_disaster step"
    );
    assert!(
        entry.saga_json.contains("emergency_allocate"),
        "saga_json should contain the emergency_allocate step"
    );
    assert!(
        entry.saga_json.contains("emergency_fund_release"),
        "saga_json should contain the emergency_fund_release step"
    );
}

/// Tests that `get_my_sagas` returns sagas initiated by the calling agent.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor + unified hApp with civic+commons+finance roles"]
async fn test_get_my_sagas_returns_initiated_sagas() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();

    let apps = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap();
    let (alice,) = apps.into_tuple();

    // Initially no sagas
    let empty: Vec<Record> = conductor
        .call(&alice.zome("civic_bridge"), "get_my_sagas", ())
        .await;
    assert!(empty.is_empty(), "no sagas should exist initially");

    // Start a saga
    let input = EmergencySagaInput {
        incident_id: "INC-2026-0002".to_string(),
        latitude: 33.0198,
        longitude: -96.6989,
    };

    let created: Record = conductor
        .call(&alice.zome("civic_bridge"), "start_emergency_saga", input)
        .await;

    // get_my_sagas should now return exactly one saga
    let sagas: Vec<Record> = conductor
        .call(&alice.zome("civic_bridge"), "get_my_sagas", ())
        .await;

    assert_eq!(sagas.len(), 1, "should have exactly one saga");
    assert_eq!(
        sagas[0].action_address(),
        created.action_address(),
        "returned saga should match the one we created"
    );
}

/// Tests that a second agent does not see sagas created by the first agent
/// via `get_my_sagas` (agent-scoped indexing).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor + unified hApp with civic+commons+finance roles"]
async fn test_saga_agent_isolation() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();

    let (alice,) = conductor
        .setup_app("alice-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let (bob,) = conductor
        .setup_app("bob-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Alice starts a saga
    let input = EmergencySagaInput {
        incident_id: "INC-2026-0003".to_string(),
        latitude: 32.7767,
        longitude: -96.7970,
    };

    let _: Record = conductor
        .call(&alice.zome("civic_bridge"), "start_emergency_saga", input)
        .await;

    // Alice should see 1 saga
    let alice_sagas: Vec<Record> = conductor
        .call(&alice.zome("civic_bridge"), "get_my_sagas", ())
        .await;
    assert_eq!(alice_sagas.len(), 1, "alice should have one saga");

    // Bob should see 0 sagas
    let bob_sagas: Vec<Record> = conductor
        .call(&bob.zome("civic_bridge"), "get_my_sagas", ())
        .await;
    assert!(bob_sagas.is_empty(), "bob should have no sagas");
}

/// Tests that the saga JSON contains the expected three-step structure
/// with correct cluster/zome/function assignments.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor + unified hApp with civic+commons+finance roles"]
async fn test_emergency_saga_step_structure() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();

    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = EmergencySagaInput {
        incident_id: "INC-2026-0004".to_string(),
        latitude: 32.8,
        longitude: -96.8,
    };

    let record: Record = conductor
        .call(&alice.zome("civic_bridge"), "start_emergency_saga", input)
        .await;

    let entry: SagaEntry = record
        .entry()
        .to_app_option()
        .expect("deserialize SagaEntry")
        .expect("entry should be present");

    // Parse the saga JSON to verify structure
    let saga: serde_json::Value =
        serde_json::from_str(&entry.saga_json).expect("saga_json should be valid JSON");

    // Verify saga name
    assert_eq!(saga["name"], "emergency-response");

    // Verify three steps exist
    let steps = saga["steps"].as_array().expect("steps should be an array");
    assert_eq!(steps.len(), 3, "emergency saga should have 3 steps");

    // Step 1: declare disaster (civic / emergency_incidents)
    assert_eq!(steps[0]["cluster"], "civic");
    assert_eq!(steps[0]["zome"], "emergency_incidents");
    assert_eq!(steps[0]["execute_fn"], "declare_disaster");
    assert_eq!(steps[0]["compensate_fn"], "end_disaster");

    // Step 2: resource allocation (commons / resource_mesh)
    assert_eq!(steps[1]["cluster"], "commons");
    assert_eq!(steps[1]["zome"], "resource_mesh");
    assert_eq!(steps[1]["execute_fn"], "emergency_allocate");
    assert_eq!(steps[1]["compensate_fn"], "emergency_deallocate");

    // Step 3: fund release (finance / treasury)
    assert_eq!(steps[2]["cluster"], "finance");
    assert_eq!(steps[2]["zome"], "treasury");
    assert_eq!(steps[2]["execute_fn"], "emergency_fund_release");
    assert_eq!(steps[2]["compensate_fn"], "emergency_fund_reclaim");
}
