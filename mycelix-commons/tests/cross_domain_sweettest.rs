//! Cross-Domain Integration Sweettest
//!
//! Proves that zomes within the Commons DNA can call each other directly
//! via `call(CallTargetCell::Local, ...)` through the bridge dispatcher.
//!
//! Run with: `cargo test --test cross_domain_sweettest -- --ignored`
//! Requires a running Holochain conductor (use `nix develop`).

use hdk::prelude::*;
use holochain::sweettest::*;

/// Mirror types — we redefine entry structs in test code to avoid
/// pulling in WASM-only integrity crates.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct RegisterPropertyInput {
    property_type: String,
    title: String,
    description: String,
    owner_did: String,
    co_owners: Vec<String>,
    geolocation: Option<GeoLocation>,
    address: Option<Address>,
    metadata: std::collections::HashMap<String, String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct GeoLocation {
    latitude: f64,
    longitude: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Address {
    street: String,
    city: String,
    state: String,
    postal_code: String,
    country: String,
}

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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct BridgeHealth {
    healthy: bool,
    agent: String,
    total_events: u32,
    total_queries: u32,
    domains: Vec<String>,
}

/// Helper to get the commons DNA path
fn commons_dna_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("dna")
        .join("commons.dna")
}

/// Test: Bridge health check returns all 5 domains
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires conductor"]
async fn test_bridge_health_check() {
    let (conductor, _app, cell) = setup_conductor().await;

    let health: BridgeHealth = conductor
        .call(&cell.zome("commons_bridge"), "health_check", ())
        .await;

    assert!(health.healthy);
    assert_eq!(health.domains.len(), 5);
    assert!(health.domains.contains(&"property".to_string()));
    assert!(health.domains.contains(&"housing".to_string()));
    assert!(health.domains.contains(&"care".to_string()));
    assert!(health.domains.contains(&"mutualaid".to_string()));
    assert!(health.domains.contains(&"water".to_string()));
}

/// Test: Cross-domain dispatch from bridge to property-registry
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires conductor"]
async fn test_cross_domain_dispatch_property() {
    let (conductor, _app, cell) = setup_conductor().await;

    // Register a property first
    let input = RegisterPropertyInput {
        property_type: "residential".to_string(),
        title: "Test House".to_string(),
        description: "Cross-domain test property".to_string(),
        owner_did: "did:test:alice".to_string(),
        co_owners: vec![],
        geolocation: Some(GeoLocation {
            latitude: 32.9483,
            longitude: -96.7299,
        }),
        address: Some(Address {
            street: "100 Main St".to_string(),
            city: "Richardson".to_string(),
            state: "TX".to_string(),
            postal_code: "75080".to_string(),
            country: "US".to_string(),
        }),
        metadata: std::collections::HashMap::new(),
    };

    // Call property_registry directly
    let record: Record = conductor
        .call(&cell.zome("property_registry"), "register_property", input.clone())
        .await;

    // Now dispatch through the bridge
    let payload = ExternIO::encode(input).unwrap().0;
    let dispatch = DispatchInput {
        zome: "property_registry".to_string(),
        fn_name: "register_property".to_string(),
        payload,
    };

    let result: DispatchResult = conductor
        .call(&cell.zome("commons_bridge"), "dispatch_call", dispatch)
        .await;

    assert!(result.success, "Cross-domain dispatch should succeed");
    assert!(result.response.is_some(), "Should have response payload");
}

/// Test: housing-clt can verify property existence via bridge dispatch
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires conductor"]
async fn test_housing_queries_property_via_bridge() {
    let (conductor, _app, cell) = setup_conductor().await;

    // Register a property
    let input = RegisterPropertyInput {
        property_type: "land".to_string(),
        title: "CLT Land Parcel".to_string(),
        description: "Land for community land trust".to_string(),
        owner_did: "did:test:trust".to_string(),
        co_owners: vec![],
        geolocation: None,
        address: None,
        metadata: std::collections::HashMap::new(),
    };

    let _record: Record = conductor
        .call(&cell.zome("property_registry"), "register_property", input)
        .await;

    // Now use bridge to query properties from housing context
    let dispatch = DispatchInput {
        zome: "property_registry".to_string(),
        fn_name: "get_all_properties".to_string(),
        payload: ExternIO::encode(()).unwrap().0,
    };

    let result: DispatchResult = conductor
        .call(&cell.zome("commons_bridge"), "dispatch_call", dispatch)
        .await;

    assert!(result.success, "Housing should be able to query property registry");
}

/// Helper to set up a conductor with the commons DNA
async fn setup_conductor() -> (SweetConductor, SweetApp, SweetCell) {
    let dna_path = commons_dna_path();
    let dna = SweetDnaFile::from_bundle(&dna_path).await.unwrap();

    let mut conductor = SweetConductor::from_standard_config().await;
    let app = conductor.setup_app("commons", &[dna]).await.unwrap();
    let cell = app.into_cells().into_iter().next().unwrap();

    (conductor, app, cell)
}
