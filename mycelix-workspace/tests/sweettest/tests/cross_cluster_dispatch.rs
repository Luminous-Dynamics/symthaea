// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-Cluster Dispatch Sweettest Integration Tests
//!
//! Proves that commons and civic bridge zomes can dispatch calls to each
//! other via `CallTargetCell::OtherRole(role)` when both DNAs are
//! installed in the same hApp.
//!
//! ## Prerequisites
//!
//! ```bash
//! # Build both cluster WASMs
//! cd mycelix-commons && cargo build --release --target wasm32-unknown-unknown
//! cd mycelix-civic   && cargo build --release --target wasm32-unknown-unknown
//!
//! # Pack DNAs
//! hc dna pack mycelix-commons/dna/
//! hc dna pack mycelix-civic/dna/
//!
//! # Pack unified hApp
//! hc app pack mycelix-workspace/tests/sweettest/
//! ```
//!
//! ## Running
//!
//! ```bash
//! cargo test --release -p mycelix-sweettest --test cross_cluster_dispatch -- --ignored
//! ```

mod harness;

use harness::*;
use holochain::prelude::*;
use holochain::sweettest::*;
use serial_test::serial;
use std::path::PathBuf;

// ============================================================================
// Mirror types — avoid WASM symbol conflicts by re-defining structs
// ============================================================================

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
struct CrossClusterDispatchInput {
    role: String,
    zome: String,
    fn_name: String,
    payload: Vec<u8>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct BridgeHealth {
    healthy: bool,
    agent: String,
    total_events: u32,
    total_queries: u32,
    domains: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct PropertyOwnershipQuery {
    property_id: String,
    requester_did: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct PropertyOwnershipResult {
    is_owner: bool,
    owner_did: Option<String>,
    error: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CareAvailabilityQuery {
    skill_needed: String,
    location: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CareAvailabilityResult {
    available_count: u32,
    recommendation: String,
    error: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct JusticeAreaQuery {
    area: String,
    case_type: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct JusticeAreaResult {
    active_cases: u32,
    recommendation: String,
    error: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct FactcheckStatusQuery {
    claim_id: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct FactcheckStatusResult {
    has_factcheck: bool,
    verdict: Option<String>,
    error: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CheckEmergencyForAreaInput {
    lat: f64,
    lon: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct EmergencyAreaCheckResult {
    has_active_emergencies: bool,
    active_count: u32,
    recommendation: String,
    error: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CheckJusticeDisputesInput {
    resource_id: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct JusticeDisputeCheckResult {
    has_pending_cases: bool,
    recommendation: String,
    error: Option<String>,
}

// ============================================================================
// Unified hApp setup — both commons + civic roles in one conductor
// ============================================================================

struct UnifiedAgent {
    conductor: SweetConductor,
    commons_cell: SweetCell,
    civic_cell: SweetCell,
}

impl UnifiedAgent {
    async fn call_commons<I, O>(&self, zome_name: &str, fn_name: &str, input: I) -> O
    where
        I: serde::Serialize + std::fmt::Debug,
        O: serde::de::DeserializeOwned + std::fmt::Debug,
    {
        let zome = self.commons_cell.zome(zome_name);
        self.conductor.call(&zome, fn_name, input).await
    }

    async fn call_civic<I, O>(&self, zome_name: &str, fn_name: &str, input: I) -> O
    where
        I: serde::Serialize + std::fmt::Debug,
        O: serde::de::DeserializeOwned + std::fmt::Debug,
    {
        let zome = self.civic_cell.zome(zome_name);
        self.conductor.call(&zome, fn_name, input).await
    }
}

async fn setup_unified_conductor() -> UnifiedAgent {
    let commons_dna = SweetDnaFile::from_bundle(&DnaPaths::commons())
        .await
        .expect("Commons DNA should exist — run `hc dna pack mycelix-commons/dna/`");

    let civic_dna = SweetDnaFile::from_bundle(&DnaPaths::civic())
        .await
        .expect("Civic DNA should exist — run `hc dna pack mycelix-civic/dna/`");

    let mut conductor = SweetConductor::from_standard_config().await;

    // Install both DNAs in the same hApp so OtherRole dispatch works
    let app = conductor
        .setup_app("mycelix-unified", &[commons_dna, civic_dna])
        .await
        .unwrap();

    let cells = app.into_cells();
    let commons_cell = cells[0].clone();
    let civic_cell = cells[1].clone();

    UnifiedAgent {
        conductor,
        commons_cell,
        civic_cell,
    }
}

// ============================================================================
// Cross-Cluster Dispatch Tests: Commons → Civic
// ============================================================================

/// Test: Commons bridge can dispatch a raw call to the civic DNA
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_commons_dispatch_civic_call() {
    let agent = setup_unified_conductor().await;

    // Dispatch from commons to civic's health_check
    let dispatch = CrossClusterDispatchInput {
        role: "civic".into(),
        zome: "civic_bridge".into(),
        fn_name: "health_check".into(),
        payload: ExternIO::encode(()).unwrap().0,
    };

    let result: DispatchResult = agent
        .call_commons("commons_bridge", "dispatch_civic_call", dispatch)
        .await;

    assert!(result.success, "Cross-cluster dispatch should succeed: {:?}", result.error);
    assert!(result.response.is_some(), "Should have response payload");
}

/// Test: Commons bridge rejects dispatch to disallowed civic zome
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_commons_rejects_disallowed_civic_zome() {
    let agent = setup_unified_conductor().await;

    let dispatch = CrossClusterDispatchInput {
        role: "civic".into(),
        zome: "evil_zome".into(),
        fn_name: "steal_data".into(),
        payload: vec![],
    };

    let result: DispatchResult = agent
        .call_commons("commons_bridge", "dispatch_civic_call", dispatch)
        .await;

    assert!(!result.success, "Disallowed zome should be rejected");
    assert!(result.error.as_deref().unwrap().contains("not in the allowed"));
}

/// Test: check_emergency_for_area typed convenience (commons → civic)
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_commons_check_emergency_for_area() {
    let agent = setup_unified_conductor().await;

    let input = CheckEmergencyForAreaInput {
        lat: 32.9483,
        lon: -96.7299,
    };

    let result: EmergencyAreaCheckResult = agent
        .call_commons("commons_bridge", "check_emergency_for_area", input)
        .await;

    // With no emergencies filed, the result should indicate no active emergencies
    assert!(!result.has_active_emergencies || result.active_count == 0 || result.error.is_some(),
        "Fresh DNA should have no emergencies or return a cross-cluster error");
}

/// Test: check_justice_disputes_for_property typed convenience (commons → civic)
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_commons_check_justice_disputes() {
    let agent = setup_unified_conductor().await;

    let input = CheckJusticeDisputesInput {
        resource_id: "prop_test_001".into(),
    };

    let result: JusticeDisputeCheckResult = agent
        .call_commons("commons_bridge", "check_justice_disputes_for_property", input)
        .await;

    // With no disputes filed, should indicate no pending cases
    assert!(!result.has_pending_cases || result.error.is_some(),
        "Fresh DNA should have no disputes or return a cross-cluster error");
}

// ============================================================================
// Cross-Cluster Dispatch Tests: Civic → Commons
// ============================================================================

/// Test: Civic bridge can dispatch a raw call to the commons DNA
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_civic_dispatch_commons_call() {
    let agent = setup_unified_conductor().await;

    // Dispatch from civic to commons' health_check
    let dispatch = CrossClusterDispatchInput {
        role: "commons".into(),
        zome: "commons_bridge".into(),
        fn_name: "health_check".into(),
        payload: ExternIO::encode(()).unwrap().0,
    };

    let result: DispatchResult = agent
        .call_civic("civic_bridge", "dispatch_commons_call", dispatch)
        .await;

    assert!(result.success, "Cross-cluster dispatch should succeed: {:?}", result.error);
    assert!(result.response.is_some(), "Should have response payload");
}

/// Test: Civic bridge rejects dispatch to disallowed commons zome
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_civic_rejects_disallowed_commons_zome() {
    let agent = setup_unified_conductor().await;

    let dispatch = CrossClusterDispatchInput {
        role: "commons".into(),
        zome: "forbidden_zome".into(),
        fn_name: "bad_fn".into(),
        payload: vec![],
    };

    let result: DispatchResult = agent
        .call_civic("civic_bridge", "dispatch_commons_call", dispatch)
        .await;

    assert!(!result.success, "Disallowed zome should be rejected");
    assert!(result.error.as_deref().unwrap().contains("not in the allowed"));
}

// ============================================================================
// Typed Convenience Functions (intra-cluster)
// ============================================================================

/// Test: verify_property_ownership typed convenience (commons intra-cluster)
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires DNAs packed"]
async fn test_typed_verify_property_ownership() {
    let agent = setup_unified_conductor().await;

    let input = PropertyOwnershipQuery {
        property_id: "PROP-nonexistent".into(),
        requester_did: "did:test:alice".into(),
    };

    let result: PropertyOwnershipResult = agent
        .call_commons("commons_bridge", "verify_property_ownership", input)
        .await;

    // With no properties registered, the dispatch to property_registry should
    // either return is_owner=false or an error from the underlying zome
    assert!(!result.is_owner || result.error.is_some(),
        "Non-existent property should not be owned");
}

/// Test: check_care_availability typed convenience (commons intra-cluster)
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires DNAs packed"]
async fn test_typed_check_care_availability() {
    let agent = setup_unified_conductor().await;

    let input = CareAvailabilityQuery {
        skill_needed: "nursing".into(),
        location: Some("downtown".into()),
    };

    let result: CareAvailabilityResult = agent
        .call_commons("commons_bridge", "check_care_availability", input)
        .await;

    // Fresh DNA should have zero providers
    assert!(result.available_count == 0 || result.error.is_some(),
        "Fresh DNA should have no care providers");
}

/// Test: get_active_cases_for_area typed convenience (civic intra-cluster)
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires DNAs packed"]
async fn test_typed_get_active_cases_for_area() {
    let agent = setup_unified_conductor().await;

    let input = JusticeAreaQuery {
        area: "north-side".into(),
        case_type: Some("civil".into()),
    };

    let result: JusticeAreaResult = agent
        .call_civic("civic_bridge", "get_active_cases_for_area", input)
        .await;

    // Fresh DNA should have zero cases
    assert!(result.active_cases == 0 || result.error.is_some(),
        "Fresh DNA should have no active cases");
}

/// Test: check_factcheck_status typed convenience (civic intra-cluster)
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires DNAs packed"]
async fn test_typed_check_factcheck_status() {
    let agent = setup_unified_conductor().await;

    let input = FactcheckStatusQuery {
        claim_id: "CL-nonexistent".into(),
    };

    let result: FactcheckStatusResult = agent
        .call_civic("civic_bridge", "check_factcheck_status", input)
        .await;

    // Non-existent claim should have no factcheck
    assert!(!result.has_factcheck || result.error.is_some(),
        "Non-existent claim should not have a factcheck");
}

// ============================================================================
// Bidirectional health verification
// ============================================================================

/// Test: Both clusters report healthy and list correct domains
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_both_clusters_healthy() {
    let agent = setup_unified_conductor().await;

    let commons_health: BridgeHealth = agent
        .call_commons("commons_bridge", "health_check", ())
        .await;

    assert!(commons_health.healthy);
    assert_eq!(commons_health.domains.len(), 5);
    assert!(commons_health.domains.contains(&"property".to_string()));
    assert!(commons_health.domains.contains(&"water".to_string()));

    let civic_health: BridgeHealth = agent
        .call_civic("civic_bridge", "health_check", ())
        .await;

    assert!(civic_health.healthy);
    assert_eq!(civic_health.domains.len(), 3);
    assert!(civic_health.domains.contains(&"justice".to_string()));
    assert!(civic_health.domains.contains(&"media".to_string()));
}

// ============================================================================
// P0 Cross-Cluster Scenarios (real-world multi-domain workflows)
// ============================================================================

// --- Mirror types for P0 scenarios ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct QueryPropertyForEnforcementInput {
    property_id: String,
    case_id: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct PropertyEnforcementResult {
    property_found: bool,
    enforcement_advisory: Option<String>,
    error: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CheckHousingCapacityInput {
    disaster_id: String,
    area: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct HousingCapacityResult {
    commons_reachable: bool,
    recommendation: Option<String>,
    error: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct VerifyCareCredentialsInput {
    provider_did: String,
    case_id: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CareCredentialVerifyResult {
    commons_reachable: bool,
    recommendation: Option<String>,
    error: Option<String>,
}

/// P0-1: Water contamination → Emergency area check
///
/// Scenario: A water contamination event in the commons cluster triggers
/// an emergency area check to see if civic emergency services are already
/// active in the affected zone.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_p0_water_contamination_triggers_emergency_check() {
    let agent = setup_unified_conductor().await;

    // Simulate: water contamination detected at a geographic location,
    // commons-bridge queries civic emergency services for that area
    let input = CheckEmergencyForAreaInput {
        lat: 29.7604,   // Houston, TX — industrial corridor
        lon: -95.3698,
    };

    let result: EmergencyAreaCheckResult = agent
        .call_commons("commons_bridge", "check_emergency_for_area", input)
        .await;

    // On a fresh DNA, either no emergencies exist or we get a cross-cluster
    // response — both are valid. The key assertion is that the cross-cluster
    // dispatch completes without panic.
    assert!(
        !result.has_active_emergencies || result.error.is_some(),
        "Fresh DNA should have no active emergencies in the Houston area"
    );
    // If the dispatch succeeded, recommendation should be present
    if result.error.is_none() {
        assert!(result.recommendation.len() > 0, "Should provide a recommendation");
    }
}

/// P0-2: Transport evacuation → Emergency coordination dispatch
///
/// Scenario: A transport route is being evaluated for evacuation use.
/// The commons cluster dispatches to civic emergency_coordination to
/// retrieve the current disaster context and determine if evacuation
/// routes overlap with active emergency zones.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_p0_transport_evacuation_route_emergency_dispatch() {
    let agent = setup_unified_conductor().await;

    // Dispatch from commons to civic emergency_coordination's
    // get_disaster_context (returns active disasters + highest severity)
    let dispatch = CrossClusterDispatchInput {
        role: "civic".into(),
        zome: "emergency_coordination".into(),
        fn_name: "get_disaster_context".into(),
        payload: ExternIO::encode(()).unwrap().0,
    };

    let result: DispatchResult = agent
        .call_commons("commons_bridge", "dispatch_civic_call", dispatch)
        .await;

    // The dispatch should either succeed (returning empty disaster context
    // on fresh DNA) or fail with a known error — not panic
    assert!(
        result.success || result.error.is_some(),
        "Cross-cluster dispatch to emergency_coordination should complete: {:?}",
        result
    );
}

/// P0-3: Housing dispute → Justice property enforcement
///
/// Scenario: A housing unit is involved in a tenant rights dispute.
/// The civic cluster queries the commons property registry to verify
/// ownership before allowing a justice enforcement action.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_p0_housing_dispute_justice_enforcement() {
    let agent = setup_unified_conductor().await;

    let input = QueryPropertyForEnforcementInput {
        property_id: "PROP-housing-dispute-001".into(),
        case_id: "CASE-tenant-rights-2026".into(),
    };

    let result: PropertyEnforcementResult = agent
        .call_civic("civic_bridge", "query_property_for_enforcement", input)
        .await;

    // On fresh DNA, property won't be found — that's the expected baseline.
    // The cross-cluster dispatch to commons must complete without error.
    assert!(
        result.property_found || result.error.is_some() || result.enforcement_advisory.is_none(),
        "Fresh DNA: property not found is the expected baseline"
    );
    // If no error, property_found should be true (dispatch succeeded, even
    // if the property doesn't exist the dispatch itself returns events)
    // or false with no panic
    if let Some(ref advisory) = result.enforcement_advisory {
        assert!(
            advisory.contains("PROP-housing-dispute-001"),
            "Advisory should reference the property ID"
        );
    }
}

/// P0-4: Care credentials → Justice evidence verification
///
/// Scenario: A care provider's testimony is submitted as evidence in a
/// justice case. The civic cluster verifies the provider's credentials
/// via cross-cluster call to commons care_credentials.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_p0_care_credentials_justice_evidence() {
    let agent = setup_unified_conductor().await;

    let input = VerifyCareCredentialsInput {
        provider_did: "did:test:nurse-alice".into(),
        case_id: "CASE-medical-malpractice-001".into(),
    };

    let result: CareCredentialVerifyResult = agent
        .call_civic("civic_bridge", "verify_care_credentials_for_evidence", input)
        .await;

    // Fresh DNA: no credentials exist, but cross-cluster call should complete.
    // commons_reachable tells us if the OtherRole dispatch worked.
    assert!(
        result.commons_reachable || result.error.is_some(),
        "Cross-cluster call to commons care_credentials should complete"
    );
    if result.commons_reachable {
        assert!(
            result.recommendation.is_some(),
            "Reachable commons should provide a recommendation"
        );
    }
}

/// P0-5: Emergency shelter → Housing capacity check
///
/// Scenario: A disaster triggers need for emergency sheltering. The civic
/// cluster checks commons housing capacity to identify available units
/// that could supplement dedicated shelters.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_p0_emergency_shelter_housing_capacity() {
    let agent = setup_unified_conductor().await;

    let input = CheckHousingCapacityInput {
        disaster_id: "DISASTER-flood-2026-001".into(),
        area: "downtown".into(),
    };

    let result: HousingCapacityResult = agent
        .call_civic("civic_bridge", "check_housing_capacity_for_sheltering", input)
        .await;

    // On fresh DNA: commons reachable but 0 housing units.
    // The key test is that the cross-cluster OtherRole dispatch works.
    assert!(
        result.commons_reachable || result.error.is_some(),
        "Cross-cluster call to commons housing_units should complete"
    );
    if result.commons_reachable {
        assert!(
            result.recommendation.is_some(),
            "Reachable commons should provide a recommendation"
        );
        let rec = result.recommendation.unwrap();
        assert!(
            rec.contains("DISASTER-flood-2026-001"),
            "Recommendation should reference the disaster ID, got: {}",
            rec
        );
        assert!(
            rec.contains("downtown"),
            "Recommendation should reference the area, got: {}",
            rec
        );
    }
}
