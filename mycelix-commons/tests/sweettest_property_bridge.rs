// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Commons — Property & Bridge Sweettest
//!
//! Batch 1 of sweettest integration tests: property-registry CRUD and
//! commons-bridge basics (query, resolve, broadcast, health).
//!
//! Split from sweettest_integration.rs to reduce per-process conductor
//! memory pressure (~1-2 GB per conductor). Each [[test]] binary runs
//! as a separate OS process, so memory is fully reclaimed between batches.
//!
//! ## Running
//! ```bash
//! cd mycelix-commons/tests
//! cargo test --release --test sweettest_property_bridge -- --ignored --test-threads=1
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — property-registry
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegisterPropertyInput {
    pub property_type: PropertyType,
    pub title: String,
    pub description: String,
    pub owner_did: String,
    pub co_owners: Vec<CoOwner>,
    pub geolocation: Option<GeoLocation>,
    pub address: Option<Address>,
    pub metadata: PropertyMetadata,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum PropertyType {
    Land,
    Building,
    Unit,
    Equipment,
    Intellectual,
    Digital,
    Other(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CoOwner {
    pub did: String,
    pub share_basis_points: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GeoLocation {
    pub latitude: f64,
    pub longitude: f64,
    pub boundaries: Option<Vec<(f64, f64)>>,
    pub area_sqm: Option<f64>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Address {
    pub street: String,
    pub city: String,
    pub region: String,
    pub country: String,
    pub postal_code: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PropertyMetadata {
    pub appraised_value: Option<f64>,
    pub currency: Option<String>,
    pub legal_description: Option<String>,
    pub parcel_number: Option<String>,
    pub attachments: Vec<String>,
}

// ============================================================================
// Mirror types — commons-bridge
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CommonsQueryInput {
    pub domain: String,
    pub query_type: String,
    pub requester: AgentPubKey,
    pub params: String,
    pub created_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CommonsEventInput {
    pub domain: String,
    pub event_type: String,
    pub source_agent: AgentPubKey,
    pub payload: String,
    pub created_at: Timestamp,
    pub related_hashes: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ResolveQueryInput {
    pub query_hash: ActionHash,
    pub result: String,
    pub success: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BridgeHealth {
    pub healthy: bool,
    pub agent: String,
    pub total_events: u32,
    pub total_queries: u32,
    pub domains: Vec<String>,
}

// ============================================================================
// DNA setup helper
// ============================================================================

fn commons_dna_path() -> PathBuf {
    if let Ok(custom) = std::env::var("COMMONS_DNA_PATH") {
        return PathBuf::from(custom);
    }
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // tests/ -> mycelix-commons/
    path.push("dna");
    path.push("mycelix_commons.dna");
    path
}

// ============================================================================
// Property Registry Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_property_register_and_get() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    let input = RegisterPropertyInput {
        property_type: PropertyType::Building,
        title: "Test Property".to_string(),
        description: "A test property".to_string(),
        owner_did: format!("did:key:{}", agent),
        co_owners: vec![],
        geolocation: Some(GeoLocation {
            latitude: 32.9483,
            longitude: -96.7299,
            boundaries: None,
            area_sqm: Some(150.0),
        }),
        address: Some(Address {
            street: "123 Main St".to_string(),
            city: "Richardson".to_string(),
            region: "TX".to_string(),
            country: "US".to_string(),
            postal_code: Some("75080".to_string()),
        }),
        metadata: PropertyMetadata {
            appraised_value: Some(250_000.0),
            currency: Some("USD".to_string()),
            legal_description: None,
            parcel_number: None,
            attachments: vec![],
        },
    };

    let record: Record = conductor
        .call(&alice.zome("property_registry"), "register_property", input)
        .await;

    assert!(record.action().author() == alice.agent_pubkey());

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

// ============================================================================
// Commons Bridge Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_query_and_resolve() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    let query = CommonsQueryInput {
        domain: "property".to_string(),
        query_type: "ownership_check".to_string(),
        requester: agent.clone(),
        params: r#"{"property_id":"test-1"}"#.to_string(),
        created_at: Timestamp::now(),
    };

    let record: Record = conductor
        .call(&alice.zome("commons_bridge"), "query_commons", query)
        .await;

    let query_hash = record.action_address().clone();

    let resolve = ResolveQueryInput {
        query_hash,
        result: r#"{"owner":"did:key:abc"}"#.to_string(),
        success: true,
    };

    let resolved: Record = conductor
        .call(&alice.zome("commons_bridge"), "resolve_query", resolve)
        .await;

    assert!(resolved.action().author() == alice.agent_pubkey());

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_broadcast_event() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    let event = CommonsEventInput {
        domain: "housing".to_string(),
        event_type: "unit_created".to_string(),
        source_agent: agent.clone(),
        payload: r#"{"unit_id":"unit-1","name":"Apt 101"}"#.to_string(),
        created_at: Timestamp::now(),
        related_hashes: vec![],
    };

    let record: Record = conductor
        .call(&alice.zome("commons_bridge"), "broadcast_event", event)
        .await;

    assert!(record.action().author() == alice.agent_pubkey());

    let events: Vec<Record> = conductor
        .call(
            &alice.zome("commons_bridge"),
            "get_domain_events",
            "housing".to_string(),
        )
        .await;

    assert_eq!(events.len(), 1);

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_health_check() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let health: BridgeHealth = conductor
        .call(&alice.zome("commons_bridge"), "health_check", ())
        .await;

    assert!(health.healthy);
    assert_eq!(health.domains.len(), 9);
    assert!(health.domains.contains(&"property".to_string()));
    assert!(health.domains.contains(&"housing".to_string()));
    assert!(health.domains.contains(&"care".to_string()));
    assert!(health.domains.contains(&"mutualaid".to_string()));
    assert!(health.domains.contains(&"water".to_string()));
    assert!(health.domains.contains(&"food".to_string()));
    assert!(health.domains.contains(&"transport".to_string()));
    assert!(health.domains.contains(&"support".to_string()));
    assert!(health.domains.contains(&"space".to_string()));

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

// ============================================================================
// Cross-Domain Tests — property/housing/care/mutualaid
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_housing_queries_property() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    let prop_input = RegisterPropertyInput {
        property_type: PropertyType::Building,
        title: "CLT Community Housing".to_string(),
        description: "Community land trust property".to_string(),
        owner_did: format!("did:key:{}", agent),
        co_owners: vec![],
        geolocation: None,
        address: None,
        metadata: PropertyMetadata {
            appraised_value: Some(500_000.0),
            currency: None,
            legal_description: None,
            parcel_number: None,
            attachments: vec![],
        },
    };

    let prop_record: Record = conductor
        .call(
            &alice.zome("property_registry"),
            "register_property",
            prop_input,
        )
        .await;

    let event = CommonsEventInput {
        domain: "housing".to_string(),
        event_type: "clt_property_linked".to_string(),
        source_agent: agent.clone(),
        payload: serde_json::to_string(&serde_json::json!({
            "property_hash": prop_record.action_address().to_string(),
            "clt_name": "Richardson Community Trust"
        }))
        .unwrap(),
        created_at: Timestamp::now(),
        related_hashes: vec![prop_record.action_address().to_string()],
    };

    let event_record: Record = conductor
        .call(&alice.zome("commons_bridge"), "broadcast_event", event)
        .await;

    assert!(event_record.action().author() == alice.agent_pubkey());

    let housing_events: Vec<Record> = conductor
        .call(
            &alice.zome("commons_bridge"),
            "get_domain_events",
            "housing".to_string(),
        )
        .await;

    assert!(!housing_events.is_empty());

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_care_checks_mutualaid_resources() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&commons_dna_path())
        .await
        .unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    let query = CommonsQueryInput {
        domain: "mutualaid".to_string(),
        query_type: "available_resources".to_string(),
        requester: agent.clone(),
        params: r#"{"resource_type":"MeetingRoom","location":"Remote"}"#.to_string(),
        created_at: Timestamp::now(),
    };

    let _record: Record = conductor
        .call(&alice.zome("commons_bridge"), "query_commons", query)
        .await;

    let my_queries: Vec<Record> = conductor
        .call(&alice.zome("commons_bridge"), "get_my_queries", ())
        .await;

    assert_eq!(my_queries.len(), 1);

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}
