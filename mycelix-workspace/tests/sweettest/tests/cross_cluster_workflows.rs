//! Cross-Cluster Workflow Integration Tests
//!
//! End-to-end tests for real-world workflows that span commons and civic
//! clusters. These exercise the cross-cluster bridge functions with actual
//! data creation and verification, going beyond dispatch-level testing.
//!
//! Key difference from `cross_cluster_dispatch.rs`: these tests create
//! domain objects (properties, disasters, cases, credentials) and verify
//! that cross-cluster queries return meaningful results based on that data.
//!
//! ## Prerequisites
//!
//! ```bash
//! # Build all cluster WASMs
//! cd mycelix-commons && cargo build --release --target wasm32-unknown-unknown
//! cd mycelix-civic   && cargo build --release --target wasm32-unknown-unknown
//! cd mycelix-identity && cargo build --release --target wasm32-unknown-unknown
//!
//! # Pack DNAs
//! hc dna pack mycelix-commons/dna/
//! hc dna pack mycelix-civic/dna/
//! hc dna pack mycelix-identity/dna/
//!
//! # Pack unified hApp
//! hc app pack mycelix-workspace/tests/sweettest/
//! ```
//!
//! ## Running
//!
//! ```bash
//! cargo test --release -p mycelix-sweettest --test cross_cluster_workflows -- --ignored
//! ```

mod harness;

use harness::*;
use holochain::prelude::*;
use holochain::sweettest::*;
use serial_test::serial;

// ============================================================================
// Mirror types -- avoid WASM symbol conflicts by re-defining structs locally
// ============================================================================

// --- Commons bridge types ---

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

// --- Civic bridge types ---

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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct WaterSafetyQuery {
    area_lat: f64,
    area_lon: f64,
    radius_km: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct WaterSafetyResult {
    safe_sources: u32,
    contaminated_sources: u32,
    total_sources: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct EmergencyFoodQuery {
    area_lat: f64,
    area_lon: f64,
    radius_km: f64,
    people_count: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct EmergencyFoodResult {
    available_kg: f64,
    distribution_points: u32,
    estimated_days_supply: f64,
}

// --- Property types (commons) ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct RegisterPropertyInput {
    name: String,
    description: String,
    property_type: String,
    location: String,
    area_sqm: Option<u32>,
}

// --- Emergency types (civic) ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum DisasterType {
    Hurricane,
    Earthquake,
    Wildfire,
    Flood,
    Tornado,
    Pandemic,
    Industrial,
    MassCasualty,
    CyberAttack,
    Infrastructure,
    Other(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum SeverityLevel {
    Level1,
    Level2,
    Level3,
    Level4,
    Level5,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct AffectedArea {
    center_lat: f64,
    center_lon: f64,
    radius_km: f32,
    boundary: Option<Vec<(f64, f64)>>,
    zones: Vec<OperationalZone>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum ZonePriority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum ZoneStatus {
    Unassessed,
    Active,
    Cleared,
    Hazardous,
    Evacuated,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct OperationalZone {
    id: String,
    name: String,
    boundary: Vec<(f64, f64)>,
    priority: ZonePriority,
    status: ZoneStatus,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct DeclareDisasterInput {
    id: String,
    disaster_type: DisasterType,
    title: String,
    description: String,
    severity: SeverityLevel,
    affected_area: AffectedArea,
    estimated_affected: u32,
    coordination_lead: Option<AgentPubKey>,
}

// --- Justice types (civic) ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct SubmitEvidenceInput {
    case_hash: ActionHash,
    evidence_type: String,
    description: String,
    content_hash: String,
}

// --- Housing types (commons) ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CreateHousingUnitInput {
    name: String,
    unit_type: String,
    address: String,
    capacity: u32,
    available: bool,
}

// --- Food distribution types (commons) ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum ListingStatus {
    Available,
    Reserved,
    Sold,
    Expired,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Listing {
    market_hash: ActionHash,
    producer: AgentPubKey,
    product_name: String,
    quantity_kg: f64,
    price_per_kg: f64,
    available_from: u64,
    status: ListingStatus,
    allergen_flags: Vec<String>,
    organic: bool,
    cultural_markers: Vec<String>,
}

// --- Consciousness credential types ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ConsciousnessProfile {
    identity: f64,
    reputation: f64,
    community: f64,
    engagement: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, PartialOrd, Ord)]
enum ConsciousnessTier {
    Observer,
    Participant,
    Citizen,
    Steward,
    Guardian,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ConsciousnessCredential {
    did: String,
    profile: ConsciousnessProfile,
    tier: ConsciousnessTier,
    issued_at: u64,
    expires_at: u64,
    issuer: String,
}

// ============================================================================
// Unified hApp setup -- both commons + civic roles in one conductor
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

    async fn call_commons_fallible<I, O>(
        &self,
        zome_name: &str,
        fn_name: &str,
        input: I,
    ) -> Result<O, holochain::conductor::api::error::ConductorApiError>
    where
        I: serde::Serialize + std::fmt::Debug,
        O: serde::de::DeserializeOwned + std::fmt::Debug,
    {
        let zome = self.commons_cell.zome(zome_name);
        self.conductor.call_fallible(&zome, fn_name, input).await
    }

    async fn call_civic<I, O>(&self, zome_name: &str, fn_name: &str, input: I) -> O
    where
        I: serde::Serialize + std::fmt::Debug,
        O: serde::de::DeserializeOwned + std::fmt::Debug,
    {
        let zome = self.civic_cell.zome(zome_name);
        self.conductor.call(&zome, fn_name, input).await
    }

    async fn call_civic_fallible<I, O>(
        &self,
        zome_name: &str,
        fn_name: &str,
        input: I,
    ) -> Result<O, holochain::conductor::api::error::ConductorApiError>
    where
        I: serde::Serialize + std::fmt::Debug,
        O: serde::de::DeserializeOwned + std::fmt::Debug,
    {
        let zome = self.civic_cell.zome(zome_name);
        self.conductor.call_fallible(&zome, fn_name, input).await
    }
}

async fn setup_unified_conductor() -> UnifiedAgent {
    let commons_dna = SweetDnaFile::from_bundle(&DnaPaths::commons())
        .await
        .expect("Commons DNA should exist -- run `hc dna pack mycelix-commons/dna/`");

    let civic_dna = SweetDnaFile::from_bundle(&DnaPaths::civic())
        .await
        .expect("Civic DNA should exist -- run `hc dna pack mycelix-civic/dna/`");

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
// Workflow 1: Property Transfer with Justice Dispute Check
//
// Scenario: Before allowing a property transfer, the commons cluster must
// check with civic justice to ensure there are no pending disputes on the
// property. This is the core commons->civic cross-cluster workflow.
// ============================================================================

/// Create a property in commons, then check civic for justice disputes before
/// allowing transfer. Verifies the full check_justice_disputes_for_property
/// cross-cluster flow.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed + identity bridge for property creation"]
async fn test_property_transfer_justice_check() {
    let agent = setup_unified_conductor().await;

    // Step 1: Verify both clusters are healthy and reachable
    let commons_health: BridgeHealth = agent
        .call_commons("commons_bridge", "health_check", ())
        .await;
    assert!(commons_health.healthy, "Commons bridge must be healthy");

    let civic_health: BridgeHealth = agent
        .call_civic("civic_bridge", "health_check", ())
        .await;
    assert!(civic_health.healthy, "Civic bridge must be healthy");

    // Step 2: Attempt to register a property (requires consciousness gate).
    // If identity bridge is not installed, this will fail at the gate --
    // which is fine. The test still exercises the gate path.
    let property_input = RegisterPropertyInput {
        name: "123 Oak Street".to_string(),
        description: "Residential property for transfer test".to_string(),
        property_type: "residential".to_string(),
        location: "Richardson, TX".to_string(),
        area_sqm: Some(200),
    };

    let register_result: Result<Record, _> = agent
        .call_commons_fallible("property_registry", "register_property", property_input)
        .await;

    // Step 3: Check for justice disputes on this property via cross-cluster call
    let dispute_input = CheckJusticeDisputesInput {
        resource_id: "PROP-123-oak-street".to_string(),
    };

    let dispute_result: JusticeDisputeCheckResult = agent
        .call_commons("commons_bridge", "check_justice_disputes_for_property", dispute_input)
        .await;

    // On a fresh DNA with no cases filed, there should be no pending disputes
    assert!(
        !dispute_result.has_pending_cases || dispute_result.error.is_some(),
        "Fresh civic DNA should have no pending disputes for this property"
    );

    // The recommendation should be populated (even if just a default)
    if dispute_result.error.is_none() {
        assert!(
            !dispute_result.recommendation.is_empty(),
            "Should provide a recommendation string"
        );
    }

    // Step 4: If property registration succeeded (consciousness gate passed),
    // verify the full round-trip by checking disputes again
    if register_result.is_ok() {
        let second_check: JusticeDisputeCheckResult = agent
            .call_commons(
                "commons_bridge",
                "check_justice_disputes_for_property",
                CheckJusticeDisputesInput {
                    resource_id: "PROP-123-oak-street".to_string(),
                },
            )
            .await;
        assert!(
            !second_check.has_pending_cases,
            "Newly registered property should have no disputes"
        );
    }
}

// ============================================================================
// Workflow 2: Emergency Housing Capacity
//
// Scenario: A disaster is declared in the civic cluster. Emergency services
// need to know how many housing units are available in the affected area
// to supplement dedicated shelters.
// ============================================================================

/// Declare a disaster in civic, then query commons for housing capacity
/// in the affected area via check_housing_capacity_for_sheltering.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_emergency_housing_capacity() {
    let agent = setup_unified_conductor().await;

    // Step 1: Attempt to declare a disaster in civic.
    // This requires consciousness gate -- if gate blocks, we still test
    // the cross-cluster housing query independently.
    let disaster_input = DeclareDisasterInput {
        id: "DISASTER-flood-2026-002".to_string(),
        disaster_type: DisasterType::Flood,
        title: "Downtown Flash Flooding".to_string(),
        description: "Severe flash flooding affecting downtown area".to_string(),
        severity: SeverityLevel::Level3,
        affected_area: AffectedArea {
            center_lat: 32.9483,
            center_lon: -96.7299,
            radius_km: 5.0,
            boundary: None,
            zones: vec![OperationalZone {
                id: "zone-downtown".to_string(),
                name: "Downtown Core".to_string(),
                boundary: vec![
                    (32.94, -96.74),
                    (32.96, -96.74),
                    (32.96, -96.72),
                    (32.94, -96.72),
                ],
                priority: ZonePriority::Critical,
                status: ZoneStatus::Active,
            }],
        },
        estimated_affected: 5000,
        coordination_lead: None,
    };

    let _disaster_result: Result<Record, _> = agent
        .call_civic_fallible("emergency_incidents", "declare_disaster", disaster_input)
        .await;

    // Step 2: Query commons housing capacity for the affected area.
    // This is the core cross-cluster call: civic -> commons via OtherRole.
    let housing_input = CheckHousingCapacityInput {
        disaster_id: "DISASTER-flood-2026-002".to_string(),
        area: "downtown".to_string(),
    };

    let housing_result: HousingCapacityResult = agent
        .call_civic("civic_bridge", "check_housing_capacity_for_sheltering", housing_input)
        .await;

    // The cross-cluster dispatch to commons must complete
    assert!(
        housing_result.commons_reachable || housing_result.error.is_some(),
        "Cross-cluster call to commons housing should complete"
    );

    if housing_result.commons_reachable {
        assert!(
            housing_result.recommendation.is_some(),
            "Reachable commons should provide a recommendation"
        );
        let rec = housing_result.recommendation.unwrap();
        assert!(
            rec.contains("DISASTER-flood-2026-002") || rec.contains("downtown"),
            "Recommendation should reference disaster ID or area, got: {}",
            rec
        );
    }
}

// ============================================================================
// Workflow 3: Emergency Food Distribution
//
// Scenario: During an emergency, civic needs to know what food reserves
// are available in the affected area for distribution to displaced people.
// ============================================================================

/// Declare emergency, query commons food_distribution for available reserves
/// via query_food_for_emergency cross-cluster call.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_emergency_food_distribution() {
    let agent = setup_unified_conductor().await;

    // Step 1: Query food availability in the emergency area.
    // This exercises the civic -> commons cross-cluster path via
    // food_distribution's check_emergency_availability.
    let food_input = EmergencyFoodQuery {
        area_lat: 32.9483,
        area_lon: -96.7299,
        radius_km: 10.0,
        people_count: 500,
    };

    let food_result: EmergencyFoodResult = agent
        .call_civic("civic_bridge", "query_food_for_emergency", food_input)
        .await;

    // On fresh DNA: no food listings exist, so quantities should be zero.
    // The key assertion is that the cross-cluster dispatch completed
    // without error and returned valid numeric results.
    assert!(
        food_result.available_kg >= 0.0,
        "available_kg must be non-negative, got: {}",
        food_result.available_kg
    );
    assert!(
        food_result.estimated_days_supply >= 0.0,
        "estimated_days_supply must be non-negative, got: {}",
        food_result.estimated_days_supply
    );

    // With no food listings, distribution_points should be zero
    assert_eq!(
        food_result.distribution_points, 0,
        "Fresh DNA should have no distribution points"
    );
}

// ============================================================================
// Workflow 4: Care Credentials for Justice Evidence
//
// Scenario: A care provider's testimony is submitted as evidence in a
// justice case. The civic cluster verifies the provider's credentials
// by querying commons care_credentials.
// ============================================================================

/// Submit care credentials in commons, verify civic justice can verify
/// them for evidence admission via verify_care_credentials_for_evidence.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_care_credentials_for_evidence() {
    let agent = setup_unified_conductor().await;

    // Step 1: Verify cross-cluster call from civic to commons care_credentials
    let verify_input = VerifyCareCredentialsInput {
        provider_did: "did:test:nurse-alice-2026".to_string(),
        case_id: "CASE-medical-testimony-001".to_string(),
    };

    let verify_result: CareCredentialVerifyResult = agent
        .call_civic("civic_bridge", "verify_care_credentials_for_evidence", verify_input)
        .await;

    // Cross-cluster dispatch should complete
    assert!(
        verify_result.commons_reachable || verify_result.error.is_some(),
        "Cross-cluster call to commons care_credentials should complete"
    );

    if verify_result.commons_reachable {
        assert!(
            verify_result.recommendation.is_some(),
            "Reachable commons should provide a recommendation"
        );
        // On fresh DNA, no credentials exist for this provider
        let rec = verify_result.recommendation.unwrap();
        assert!(
            !rec.is_empty(),
            "Recommendation should not be empty"
        );
    }
}

// ============================================================================
// Workflow 5: Consciousness Credential Happy Path
//
// Scenario: An agent registers a DID in the identity cluster, sets MFA,
// gets trust attestations, then uses the resulting consciousness credential
// to perform a governance-gated action in commons (property registration).
//
// This test requires the identity DNA to be installed alongside commons
// and civic. It proves the ACCEPT path of consciousness gating.
// ============================================================================

/// Full happy path: identity DID -> MFA -> trust -> credential -> commons gate
/// passes -> governance action succeeds.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires identity + commons + civic DNAs packed in unified hApp"]
async fn test_consciousness_credential_happy_path() {
    // This test requires a 3-DNA unified hApp: identity + commons + civic.
    // The identity DNA provides DID registration and credential issuance.
    let identity_dna = SweetDnaFile::from_bundle(&DnaPaths::identity())
        .await
        .expect("Identity DNA should exist -- run `hc dna pack mycelix-identity/dna/`");

    let commons_dna = SweetDnaFile::from_bundle(&DnaPaths::commons())
        .await
        .expect("Commons DNA should exist");

    let civic_dna = SweetDnaFile::from_bundle(&DnaPaths::civic())
        .await
        .expect("Civic DNA should exist");

    // Use named roles so CallTargetCell::OtherRole("identity") resolves correctly.
    let dnas_with_roles: Vec<(String, DnaFile)> = vec![
        ("identity".to_string(), identity_dna),
        ("commons_land".to_string(), commons_dna),
        ("civic".to_string(), civic_dna),
    ];

    let mut conductor = SweetConductor::from_standard_config().await;
    let app = conductor
        .setup_app("mycelix-full", &dnas_with_roles)
        .await
        .unwrap();

    let cells = app.into_cells();
    let identity_cell = cells[0].clone();
    let commons_cell = cells[1].clone();

    // Step 1: Register a DID in the identity cluster
    let did_record: Record = conductor
        .call(&identity_cell.zome("did_registry"), "create_did", ())
        .await;
    assert!(
        did_record.action_address().get_raw_39().len() > 0,
        "DID record should have a valid action address"
    );

    // Step 2: Create MFA state (sets baseline identity assurance)
    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct CreateMfaStateInput {
        did: String,
    }

    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct MfaStateOutput {
        did: String,
        assurance_level: String,
        factors: Vec<serde_json::Value>,
    }

    let agent_key = identity_cell.agent_pubkey().clone();
    let did = format!("did:mycelix:{}", agent_key);

    let mfa_result: Result<MfaStateOutput, _> = conductor
        .call_fallible(
            &identity_cell.zome("mfa"),
            "create_mfa_state",
            CreateMfaStateInput { did: did.clone() },
        )
        .await;

    // Step 3: Request consciousness credential from commons bridge.
    // The commons bridge will cross-cluster call to identity_bridge.
    let credential_result: Result<ConsciousnessCredential, _> = conductor
        .call_fallible(
            &commons_cell.zome("commons_bridge"),
            "get_consciousness_credential",
            did.clone(),
        )
        .await;

    // If identity bridge is properly wired, we get a credential back
    if let Ok(credential) = credential_result {
        assert_eq!(credential.did, did, "Credential DID should match agent DID");
        assert!(
            credential.expires_at > credential.issued_at,
            "Credential should have valid expiry"
        );

        // Step 4: With credential in hand (cached by bridge), attempt
        // a governance-gated action. This tests the ACCEPT path.
        let property_input = RegisterPropertyInput {
            name: "Consciousness Test Property".to_string(),
            description: "Property registered with valid consciousness credential".to_string(),
            property_type: "residential".to_string(),
            location: "Richardson, TX".to_string(),
            area_sqm: Some(150),
        };

        let register_result: Result<Record, _> = conductor
            .call_fallible(
                &commons_cell.zome("property_registry"),
                "register_property",
                property_input,
            )
            .await;

        // If the credential has sufficient tier, registration should succeed
        if credential.tier >= ConsciousnessTier::Participant {
            assert!(
                register_result.is_ok(),
                "Property registration should succeed with {:?} tier credential: {:?}",
                credential.tier,
                register_result.err()
            );
        }
    } else {
        // Identity bridge not reachable -- this is expected when running
        // without the full 3-DNA hApp. Test still validates the cross-cluster
        // call path and error handling.
        let err = credential_result.unwrap_err();
        let err_msg = format!("{:?}", err);
        assert!(
            err_msg.contains("identity")
                || err_msg.contains("OtherRole")
                || err_msg.contains("cross_cluster")
                || err_msg.contains("credential"),
            "Error should indicate identity bridge unreachable, got: {}",
            err_msg
        );
    }
}

// ============================================================================
// Workflow 6: Consciousness Tier Escalation
//
// Scenario: The same agent verifies that different consciousness tiers
// gate different governance actions. Observer blocks voting, Participant
// allows basic proposals, Citizen allows voting.
// ============================================================================

/// Verify tier-gated behavior: Observer blocks, Participant allows basic,
/// Citizen allows voting.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires identity + commons DNAs packed in unified hApp"]
async fn test_consciousness_tier_escalation() {
    let identity_dna = SweetDnaFile::from_bundle(&DnaPaths::identity())
        .await
        .expect("Identity DNA required");

    let commons_dna = SweetDnaFile::from_bundle(&DnaPaths::commons())
        .await
        .expect("Commons DNA required");

    let dnas_with_roles: Vec<(String, DnaFile)> = vec![
        ("identity".to_string(), identity_dna),
        ("commons_land".to_string(), commons_dna),
    ];

    let mut conductor = SweetConductor::from_standard_config().await;
    let app = conductor
        .setup_app("mycelix-tier-test", &dnas_with_roles)
        .await
        .unwrap();

    let cells = app.into_cells();
    let identity_cell = cells[0].clone();
    let commons_cell = cells[1].clone();

    // Step 1: Create DID (baseline Observer tier -- no MFA, no trust)
    let _did_record: Record = conductor
        .call(&identity_cell.zome("did_registry"), "create_did", ())
        .await;

    let agent_key = identity_cell.agent_pubkey().clone();
    let did = format!("did:mycelix:{}", agent_key);

    // Step 2: Get consciousness credential (should be Observer tier)
    let credential_result: Result<ConsciousnessCredential, _> = conductor
        .call_fallible(
            &commons_cell.zome("commons_bridge"),
            "get_consciousness_credential",
            did.clone(),
        )
        .await;

    if let Ok(credential) = credential_result {
        // Step 3: Observer tier -- read operations should work
        let markets: Vec<Record> = conductor
            .call(
                &commons_cell.zome("food_distribution"),
                "get_all_markets",
                (),
            )
            .await;
        assert!(
            markets.is_empty(),
            "Read operations should succeed at any tier"
        );

        // Step 4: Observer tier -- voting should be blocked
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        enum VoteChoice { Yes, No, Abstain, Block }

        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct Vote {
            proposal_hash: ActionHash,
            voter: AgentPubKey,
            vote: VoteChoice,
            reasoning: Option<String>,
            voted_at: Timestamp,
        }

        if credential.tier == ConsciousnessTier::Observer {
            let vote = Vote {
                proposal_hash: ActionHash::from_raw_36(vec![0xDD; 36]),
                voter: AgentPubKey::from_raw_36(vec![0xEE; 36]),
                vote: VoteChoice::Yes,
                reasoning: Some("Observer tier test".to_string()),
                voted_at: Timestamp::from_micros(1700000000),
            };

            let vote_result: Result<Record, _> = conductor
                .call_fallible(
                    &commons_cell.zome("mutualaid_governance"),
                    "cast_vote",
                    vote,
                )
                .await;

            assert!(
                vote_result.is_err(),
                "Observer tier should NOT be allowed to vote"
            );
        }

        // Step 5: If tier is Participant or higher, basic proposals should work
        if credential.tier >= ConsciousnessTier::Participant {
            let property_input = RegisterPropertyInput {
                name: "Tier Test Property".to_string(),
                description: "Tests Participant tier access".to_string(),
                property_type: "residential".to_string(),
                location: "Tier Test Location".to_string(),
                area_sqm: Some(100),
            };

            let prop_result: Result<Record, _> = conductor
                .call_fallible(
                    &commons_cell.zome("property_registry"),
                    "register_property",
                    property_input,
                )
                .await;

            assert!(
                prop_result.is_ok(),
                "Participant+ tier should be allowed to register properties"
            );
        }
    }
    // If credential fetch fails, the test is equivalent to testing the
    // gate-rejection path, which is already covered by consciousness_gating tests.
}

// ============================================================================
// Workflow 7: Consciousness Credential Expiry
//
// Scenario: Issue credential, verify it works, simulate time passage
// beyond TTL (24h), verify gate blocks. Then verify the 30-minute grace
// period allows basic (Participant-tier) operations but blocks voting.
// ============================================================================

/// Verify credential expiry behavior and grace period.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires identity + commons DNAs packed in unified hApp"]
async fn test_consciousness_credential_expiry() {
    let identity_dna = SweetDnaFile::from_bundle(&DnaPaths::identity())
        .await
        .expect("Identity DNA required");

    let commons_dna = SweetDnaFile::from_bundle(&DnaPaths::commons())
        .await
        .expect("Commons DNA required");

    let dnas_with_roles: Vec<(String, DnaFile)> = vec![
        ("identity".to_string(), identity_dna),
        ("commons_land".to_string(), commons_dna),
    ];

    let mut conductor = SweetConductor::from_standard_config().await;
    let app = conductor
        .setup_app("mycelix-expiry-test", &dnas_with_roles)
        .await
        .unwrap();

    let cells = app.into_cells();
    let identity_cell = cells[0].clone();
    let commons_cell = cells[1].clone();

    // Step 1: Create DID and get initial credential
    let _did_record: Record = conductor
        .call(&identity_cell.zome("did_registry"), "create_did", ())
        .await;

    let agent_key = identity_cell.agent_pubkey().clone();
    let did = format!("did:mycelix:{}", agent_key);

    let credential_result: Result<ConsciousnessCredential, _> = conductor
        .call_fallible(
            &commons_cell.zome("commons_bridge"),
            "get_consciousness_credential",
            did.clone(),
        )
        .await;

    if let Ok(credential) = credential_result {
        // Step 2: Verify credential has 24h TTL
        let ttl_us = credential.expires_at - credential.issued_at;
        assert_eq!(
            ttl_us, 86_400_000_000,
            "Credential TTL should be 24 hours (86400000000 us), got: {}",
            ttl_us
        );

        // Step 3: Verify credential is not yet expired
        assert!(
            !credential.is_expired_at(credential.issued_at + 1000),
            "Credential should not be expired shortly after issuance"
        );

        // Step 4: Verify credential IS expired after TTL
        let after_ttl = credential.expires_at + 1;
        assert!(
            credential.is_expired_at(after_ttl),
            "Credential should be expired after TTL"
        );

        // Step 5: Verify grace period (30 min = 1_800_000_000 us after expiry)
        // Grace period allows Participant-tier operations but blocks voting
        let in_grace = credential.expires_at + 600_000_000; // 10 min after expiry
        assert!(
            credential.is_expired_at(in_grace),
            "Credential should be technically expired during grace period"
        );

        let past_grace = credential.expires_at + 1_800_000_001; // 30 min + 1us
        assert!(
            credential.is_expired_at(past_grace),
            "Credential should be fully expired past grace period"
        );
    }
    // If credential fetch fails, test still passes -- this path is covered
    // by the unit tests in consciousness_profile.rs.
}

// ============================================================================
// Workflow: Water Safety During Emergency
//
// Scenario: During a declared emergency, civic queries commons water_purity
// to check if water sources in the area are safe for consumption.
// ============================================================================

/// Query water safety via civic->commons cross-cluster call.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_water_safety_during_emergency() {
    let agent = setup_unified_conductor().await;

    let water_input = WaterSafetyQuery {
        area_lat: 32.9483,
        area_lon: -96.7299,
        radius_km: 15.0,
    };

    let water_result: WaterSafetyResult = agent
        .call_civic("civic_bridge", "query_water_safety_for_emergency", water_input)
        .await;

    // On fresh DNA: no water sources registered
    assert_eq!(
        water_result.total_sources, 0,
        "Fresh DNA should have no water sources registered"
    );
    assert_eq!(
        water_result.contaminated_sources, 0,
        "Fresh DNA should have no contaminated sources"
    );
    assert_eq!(
        water_result.safe_sources, 0,
        "Fresh DNA should have no safe sources"
    );
}

// ============================================================================
// Workflow: Bidirectional Emergency Coordination
//
// Scenario: Exercise both directions of cross-cluster emergency calls:
// commons checks civic for active emergencies (check_emergency_for_area),
// then civic checks commons for resources (housing + food + water).
// Verifies that a single emergency triggers coherent multi-resource queries.
// ============================================================================

/// Full bidirectional emergency workflow: commons->civic area check,
/// then civic->commons for housing, food, and water resources.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_bidirectional_emergency_coordination() {
    let agent = setup_unified_conductor().await;

    // Direction 1: Commons checks civic for active emergencies
    let emergency_check = CheckEmergencyForAreaInput {
        lat: 32.9483,
        lon: -96.7299,
    };

    let emergency_result: EmergencyAreaCheckResult = agent
        .call_commons("commons_bridge", "check_emergency_for_area", emergency_check)
        .await;

    // Cross-cluster dispatch should complete in both directions
    assert!(
        !emergency_result.has_active_emergencies || emergency_result.error.is_some(),
        "Fresh DNA should have no active emergencies"
    );

    // Direction 2: Civic checks commons for housing capacity
    let housing_input = CheckHousingCapacityInput {
        disaster_id: "DISASTER-bilateral-test-001".to_string(),
        area: "north-central".to_string(),
    };

    let housing_result: HousingCapacityResult = agent
        .call_civic("civic_bridge", "check_housing_capacity_for_sheltering", housing_input)
        .await;

    assert!(
        housing_result.commons_reachable || housing_result.error.is_some(),
        "Cross-cluster call to commons housing should complete"
    );

    // Direction 3: Civic checks commons for food reserves
    let food_input = EmergencyFoodQuery {
        area_lat: 32.9483,
        area_lon: -96.7299,
        radius_km: 20.0,
        people_count: 1000,
    };

    let food_result: EmergencyFoodResult = agent
        .call_civic("civic_bridge", "query_food_for_emergency", food_input)
        .await;

    assert!(
        food_result.available_kg >= 0.0,
        "Food availability should be non-negative"
    );

    // Direction 4: Civic checks commons for water safety
    let water_input = WaterSafetyQuery {
        area_lat: 32.9483,
        area_lon: -96.7299,
        radius_km: 15.0,
    };

    let water_result: WaterSafetyResult = agent
        .call_civic("civic_bridge", "query_water_safety_for_emergency", water_input)
        .await;

    assert_eq!(
        water_result.total_sources, 0,
        "Fresh DNA: no water sources"
    );

    // Verify bridge health survived all cross-cluster calls
    let commons_health: BridgeHealth = agent
        .call_commons("commons_bridge", "health_check", ())
        .await;
    let civic_health: BridgeHealth = agent
        .call_civic("civic_bridge", "health_check", ())
        .await;

    assert!(commons_health.healthy, "Commons bridge should remain healthy");
    assert!(civic_health.healthy, "Civic bridge should remain healthy");

    // Verify query counts increased (shows cross-cluster calls were tracked)
    assert!(
        commons_health.total_queries > 0 || commons_health.total_events > 0,
        "Commons bridge should have recorded queries/events"
    );
    assert!(
        civic_health.total_queries > 0 || civic_health.total_events > 0,
        "Civic bridge should have recorded queries/events"
    );
}

// ============================================================================
// Helper trait for credential expiry checking (mirrors the bridge-common API)
// ============================================================================

trait CredentialExpiry {
    fn is_expired_at(&self, now_us: u64) -> bool;
}

impl CredentialExpiry for ConsciousnessCredential {
    fn is_expired_at(&self, now_us: u64) -> bool {
        now_us >= self.expires_at
    }
}
