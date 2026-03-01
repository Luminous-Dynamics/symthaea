//! # Fabrication — Bridge & Repair Workflow Sweettest
//!
//! Integration tests for repair predictions, workflow lifecycle,
//! event emission, rate limiting, audit trail, and authorization.
//!
//! ## Running
//! ```bash
//! cd mycelix-workspace/happs/fabrication/tests/sweettest
//! cargo test --release --test sweettest_bridge -- --ignored --test-threads=1
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

use fabrication_sweettest::common::*;

// ============================================================================
// Mirror types — bridge coordinator
// ============================================================================

/// Mirror of bridge coordinator's `CreateRepairPredictionInput`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateRepairPredictionInput {
    pub property_asset_hash: ActionHash,
    pub asset_model: String,
    pub predicted_failure_component: String,
    pub failure_probability: f32,
    pub estimated_failure_date: Timestamp,
    pub confidence_interval_days: u32,
    pub sensor_data_summary: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UpdateWorkflowInput {
    pub workflow_hash: ActionHash,
    pub status: String,
    pub design_hash: Option<ActionHash>,
    pub printer_hash: Option<ActionHash>,
    pub hearth_funding_hash: Option<ActionHash>,
    pub print_job_hash: Option<ActionHash>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EmitEventInput {
    pub event_type: String,
    pub design_id: Option<ActionHash>,
    pub payload: String,
}

/// Mirror of bridge coordinator's `GetRecentEventsInput`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GetRecentEventsInput {
    pub since: Option<Timestamp>,
    pub pagination: Option<PaginationInput>,
}

/// Mirror of bridge coordinator's `GetActiveWorkflowsInput`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GetActiveWorkflowsInput {
    pub pagination: Option<PaginationInput>,
}

/// Mirror of fabrication_common's `AuditTrailFilter`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AuditTrailFilter {
    pub domain: Option<String>,
    pub agent: Option<AgentPubKey>,
    pub after: Option<Timestamp>,
    pub before: Option<Timestamp>,
    pub limit: Option<u32>,
    pub pagination: Option<PaginationInput>,
}

/// Mirror of bridge coordinator's `ListDesignInput`.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ListOnMarketplaceInput {
    pub design_hash: ActionHash,
    pub price: Option<u64>,
    pub listing_type: String,
}

// ============================================================================
// Repair Prediction Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_repair_prediction_and_workflow() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create a repair prediction
    let prediction_input = CreateRepairPredictionInput {
        property_asset_hash: ActionHash::from_raw_39(vec![132u8; 39]),
        asset_model: "Bosch GSR 18V-60".to_string(),
        predicted_failure_component: "Battery clip".to_string(),
        failure_probability: 0.85,
        estimated_failure_date: Timestamp::from_micros(1740000000000000), // Far future
        confidence_interval_days: 7,
        sensor_data_summary: r#"{"vibration_increase": 0.15}"#.to_string(),
    };

    let prediction_record: Record = conductor
        .call(
            &alice.zome("bridge_coordinator"),
            "create_repair_prediction",
            prediction_input,
        )
        .await;

    let prediction_hash = prediction_record.action_address().clone();

    // Create workflow from prediction
    let workflow_record: Record = conductor
        .call(
            &alice.zome("bridge_coordinator"),
            "create_repair_workflow",
            prediction_hash.clone(),
        )
        .await;

    assert_eq!(workflow_record.action().author(), alice.agent_pubkey());

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_workflow_update_requires_creator() {
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let mut alice_conductor = SweetConductor::from_standard_config().await;
    let mut bob_conductor = SweetConductor::from_standard_config().await;

    let (alice,) = alice_conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();
    let (bob,) = bob_conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    SweetConductor::exchange_peer_info([&alice_conductor, &bob_conductor]).await;

    // Alice creates prediction + workflow
    let prediction_input = CreateRepairPredictionInput {
        property_asset_hash: ActionHash::from_raw_39(vec![132u8; 39]),
        asset_model: "Auth Test Tool".to_string(),
        predicted_failure_component: "motor".to_string(),
        failure_probability: 0.75,
        estimated_failure_date: Timestamp::from_micros(1740000000000000),
        confidence_interval_days: 10,
        sensor_data_summary: "{}".to_string(),
    };

    let prediction_record: Record = alice_conductor
        .call(
            &alice.zome("bridge_coordinator"),
            "create_repair_prediction",
            prediction_input,
        )
        .await;

    let prediction_hash = prediction_record.action_address().clone();

    let workflow_record: Record = alice_conductor
        .call(
            &alice.zome("bridge_coordinator"),
            "create_repair_workflow",
            prediction_hash,
        )
        .await;

    let workflow_hash = workflow_record.action_address().clone();

    // Wait for gossip propagation so Bob can see Alice's entries
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    // Bob tries to update Alice's workflow — should fail
    let update_input = UpdateWorkflowInput {
        workflow_hash,
        status: "Cancelled".to_string(),
        design_hash: None,
        printer_hash: None,
        hearth_funding_hash: None,
        print_job_hash: None,
    };

    let bob_result: Result<Record, _> = bob_conductor
        .call_fallible(
            &bob.zome("bridge_coordinator"),
            "update_repair_workflow",
            update_input,
        )
        .await;

    assert!(
        bob_result.is_err(),
        "Bob should not be able to update Alice's workflow"
    );

    drop(alice_conductor);
    drop(bob_conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_event_emission_and_audit_trail() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Emit an event
    let event_input = EmitEventInput {
        event_type: "DesignPublished".to_string(),
        design_id: None,
        payload: r#"{"title":"Test Design"}"#.to_string(),
    };

    let event_record: Record = conductor
        .call(
            &alice.zome("bridge_coordinator"),
            "emit_fabrication_event",
            event_input,
        )
        .await;

    assert!(event_record.action().author() == alice.agent_pubkey());

    // Query recent events
    let events: serde_json::Value = conductor
        .call(
            &alice.zome("bridge_coordinator"),
            "get_recent_events",
            GetRecentEventsInput {
                since: None,
                pagination: Some(PaginationInput {
                    offset: 0,
                    limit: 100,
                }),
            },
        )
        .await;

    // Should have items
    let items = events.get("items").and_then(|v| v.as_array());
    assert!(
        items.map(|a| a.len()).unwrap_or(0) >= 1,
        "Should have at least one event"
    );

    // Query audit trail — prediction creation should be logged
    let prediction_input = CreateRepairPredictionInput {
        property_asset_hash: ActionHash::from_raw_39(vec![132u8; 39]),
        asset_model: "Audit Test".to_string(),
        predicted_failure_component: "test".to_string(),
        failure_probability: 0.5,
        estimated_failure_date: Timestamp::from_micros(1740000000000000),
        confidence_interval_days: 7,
        sensor_data_summary: "{}".to_string(),
    };

    let _: Record = conductor
        .call(
            &alice.zome("bridge_coordinator"),
            "create_repair_prediction",
            prediction_input,
        )
        .await;

    let audit_result: serde_json::Value = conductor
        .call(
            &alice.zome("bridge_coordinator"),
            "get_audit_trail",
            AuditTrailFilter {
                domain: Some("Bridge".to_string()),
                agent: None,
                after: None,
                before: None,
                limit: Some(50),
                pagination: None,
            },
        )
        .await;

    let audit_items = audit_result.get("items").and_then(|v| v.as_array());
    assert!(
        audit_items.map(|a| !a.is_empty()).unwrap_or(false),
        "Audit trail should have entries from prediction creation"
    );

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_marketplace_listing() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&fabrication_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Create a design first
    let design_input = CreateDesignInput {
        title: "Marketplace Widget".to_string(),
        description: "For marketplace listing test".to_string(),
        category: "Parts".to_string(),
        intent_vector: None,
        parametric_schema: None,
        constraint_graph: None,
        material_compatibility: vec![],
        circularity_score: 0.5,
        embodied_energy_kwh: 1.0,
        repair_manifest: None,
        license: serde_json::json!("Proprietary"),
        safety_class: "Class1Functional".to_string(),
    };

    let design_record: Record = conductor
        .call(
            &alice.zome("designs_coordinator"),
            "create_design",
            design_input,
        )
        .await;

    let design_hash = design_record.action_address().clone();

    // List on marketplace
    let listing_input = ListOnMarketplaceInput {
        design_hash,
        price: Some(500),
        listing_type: "DesignSale".to_string(),
    };

    let listing_record: Record = conductor
        .call(
            &alice.zome("bridge_coordinator"),
            "list_design_on_marketplace",
            listing_input,
        )
        .await;

    assert!(listing_record.action().author() == alice.agent_pubkey());

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}
