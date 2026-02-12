//! # Mycelix Commons — Sweettest Integration Tests
//!
//! Tests the unified Commons cluster DNA: property, housing, care,
//! mutualaid, water domain zomes + commons-bridge.
//!
//! ## Running
//! ```bash
//! cd mycelix-commons
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cargo test -p commons-tests --test sweettest_integration -- --ignored
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — avoid importing zome crates (duplicate WASM symbols)
// ============================================================================

// --- property-registry ---

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
    Residential,
    Commercial,
    Agricultural,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CoOwner {
    pub did: String,
    pub share_percentage: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GeoLocation {
    pub latitude: f64,
    pub longitude: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Address {
    pub street: String,
    pub city: String,
    pub state: String,
    pub country: String,
    pub postal_code: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PropertyMetadata {
    pub area_sqm: Option<f64>,
    pub year_built: Option<u32>,
    pub zoning: Option<String>,
    pub assessed_value: Option<u64>,
    pub currency: Option<String>,
}

// --- commons-bridge ---

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
pub struct CommonsHealthStatus {
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
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // tests/ → mycelix-commons/
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
    let conductor = SweetConductor::from_standard_config().await;
    let (alice,) = conductor
        .setup_app("test-app", &[DnaSource::Path(commons_dna_path())])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    let input = RegisterPropertyInput {
        property_type: PropertyType::Residential,
        title: "Test Property".to_string(),
        description: "A test property".to_string(),
        owner_did: format!("did:key:{}", agent),
        co_owners: vec![],
        geolocation: Some(GeoLocation {
            latitude: 32.9483,
            longitude: -96.7299,
        }),
        address: Some(Address {
            street: "123 Main St".to_string(),
            city: "Richardson".to_string(),
            state: "TX".to_string(),
            country: "US".to_string(),
            postal_code: "75080".to_string(),
        }),
        metadata: PropertyMetadata {
            area_sqm: Some(150.0),
            year_built: Some(2020),
            zoning: Some("R-1".to_string()),
            assessed_value: Some(250000),
            currency: Some("USD".to_string()),
        },
    };

    let record: Record = conductor
        .call(&alice.zome("property_registry"), "register_property", input)
        .await;

    assert!(record.action().author() == alice.agent_pubkey());
}

// ============================================================================
// Commons Bridge Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_query_and_resolve() {
    let conductor = SweetConductor::from_standard_config().await;
    let (alice,) = conductor
        .setup_app("test-app", &[DnaSource::Path(commons_dna_path())])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    // Submit a query via the bridge
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

    // Resolve the query
    let resolve = ResolveQueryInput {
        query_hash,
        result: r#"{"owner":"did:key:abc"}"#.to_string(),
        success: true,
    };

    let resolved: Record = conductor
        .call(&alice.zome("commons_bridge"), "resolve_query", resolve)
        .await;

    assert!(resolved.action().author() == alice.agent_pubkey());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_broadcast_event() {
    let conductor = SweetConductor::from_standard_config().await;
    let (alice,) = conductor
        .setup_app("test-app", &[DnaSource::Path(commons_dna_path())])
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

    // Verify event appears in domain query
    let events: Vec<Record> = conductor
        .call(
            &alice.zome("commons_bridge"),
            "get_domain_events",
            "housing".to_string(),
        )
        .await;

    assert_eq!(events.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_health_check() {
    let conductor = SweetConductor::from_standard_config().await;
    let (alice,) = conductor
        .setup_app("test-app", &[DnaSource::Path(commons_dna_path())])
        .await
        .unwrap()
        .into_tuple();

    let health: CommonsHealthStatus = conductor
        .call(&alice.zome("commons_bridge"), "health_check", ())
        .await;

    assert!(health.healthy);
    assert_eq!(health.domains.len(), 5);
    assert!(health.domains.contains(&"property".to_string()));
    assert!(health.domains.contains(&"housing".to_string()));
    assert!(health.domains.contains(&"care".to_string()));
    assert!(health.domains.contains(&"mutualaid".to_string()));
    assert!(health.domains.contains(&"water".to_string()));
}

// ============================================================================
// Cross-Domain Tests — the real value of cluster consolidation
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_housing_queries_property() {
    let conductor = SweetConductor::from_standard_config().await;
    let (alice,) = conductor
        .setup_app("test-app", &[DnaSource::Path(commons_dna_path())])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    // 1. Register a property
    let prop_input = RegisterPropertyInput {
        property_type: PropertyType::Residential,
        title: "CLT Community Housing".to_string(),
        description: "Community land trust property".to_string(),
        owner_did: format!("did:key:{}", agent),
        co_owners: vec![],
        geolocation: None,
        address: None,
        metadata: PropertyMetadata {
            area_sqm: Some(500.0),
            year_built: Some(2024),
            zoning: None,
            assessed_value: None,
            currency: None,
        },
    };

    let prop_record: Record = conductor
        .call(&alice.zome("property_registry"), "register_property", prop_input)
        .await;

    // 2. Bridge event: housing references the property
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

    // 3. Verify cross-domain event is retrievable by housing domain
    let housing_events: Vec<Record> = conductor
        .call(
            &alice.zome("commons_bridge"),
            "get_domain_events",
            "housing".to_string(),
        )
        .await;

    assert!(!housing_events.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_care_checks_mutualaid_resources() {
    let conductor = SweetConductor::from_standard_config().await;
    let (alice,) = conductor
        .setup_app("test-app", &[DnaSource::Path(commons_dna_path())])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    // Care zome queries mutualaid resources via bridge
    let query = CommonsQueryInput {
        domain: "mutualaid".to_string(),
        query_type: "available_resources".to_string(),
        requester: agent.clone(),
        params: r#"{"resource_type":"MeetingRoom","location":"Remote"}"#.to_string(),
        created_at: Timestamp::now(),
    };

    let record: Record = conductor
        .call(&alice.zome("commons_bridge"), "query_commons", query)
        .await;

    // The query is stored — in production the bridge would dispatch to
    // mutualaid_resources zome. For now, verify it's recorded.
    let my_queries: Vec<Record> = conductor
        .call(&alice.zome("commons_bridge"), "get_my_queries", ())
        .await;

    assert_eq!(my_queries.len(), 1);
}
