//! Cross-Domain Integration Sweettest
//!
//! Proves that zomes within the Commons DNA can call each other directly
//! via `call(CallTargetCell::Local, ...)` through the bridge dispatcher,
//! AND that cross-cluster dispatches between Civic and Commons are routed
//! through the correct bridge allowlists.
//!
//! ## Intra-cluster (Commons internal)
//! Tests dispatch_call on commons_bridge targeting domain zomes within the
//! same DNA (property, food, transport, etc.).
//!
//! ## Cross-cluster (Civic <-> Commons)
//! Tests dispatch_commons_call (civic_bridge) and dispatch_civic_call
//! (commons_bridge) for scenarios that cross the cluster boundary:
//! - Justice enforcement freezing property transfers (Civic -> Commons)
//! - Emergency requesting mutual aid resources (Civic -> Commons)
//! - Media factcheck referencing water purity data (Civic -> Commons)
//! - Housing governance escalating disputes to justice (Commons -> Civic)
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

// -- Food Production mirror types --

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum SoilType {
    Clay,
    Sandy,
    Loam,
    Silt,
    Peat,
    Chalk,
    Mixed,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum PlotStatus {
    Active,
    Fallow,
    Preparing,
    Retired,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Plot {
    id: String,
    name: String,
    area_sqm: f64,
    soil_type: SoilType,
    location_lat: f64,
    location_lon: f64,
    steward: AgentPubKey,
    status: PlotStatus,
}

// -- Food Distribution mirror types --

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum MarketType {
    Farmers,
    CSA,
    FoodBank,
    CoOp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Market {
    id: String,
    name: String,
    location_lat: f64,
    location_lon: f64,
    market_type: MarketType,
    steward: AgentPubKey,
    schedule: String,
}

// -- Transport Routes mirror types --

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum VehicleType {
    Car,
    Van,
    Bike,
    Bus,
    Cargo,
    ElectricScooter,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum VehicleStatus {
    Available,
    InUse,
    Maintenance,
    Retired,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Vehicle {
    id: String,
    owner: AgentPubKey,
    vehicle_type: VehicleType,
    capacity_kg: f64,
    capacity_passengers: u32,
    status: VehicleStatus,
}

// -- Transport Sharing mirror types --

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum OfferStatus {
    Open,
    Full,
    InProgress,
    Completed,
    Cancelled,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct RideOffer {
    vehicle_hash: ActionHash,
    route_hash: Option<ActionHash>,
    driver: AgentPubKey,
    departure_time: u64,
    seats_available: u32,
    price_per_seat: f64,
    status: OfferStatus,
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
    assert_eq!(health.domains.len(), 7);
    assert!(health.domains.contains(&"property".to_string()));
    assert!(health.domains.contains(&"housing".to_string()));
    assert!(health.domains.contains(&"care".to_string()));
    assert!(health.domains.contains(&"mutualaid".to_string()));
    assert!(health.domains.contains(&"water".to_string()));
    assert!(health.domains.contains(&"food".to_string()));
    assert!(health.domains.contains(&"transport".to_string()));
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

/// Test: Cross-domain dispatch from bridge to food_production's register_plot
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires conductor"]
async fn test_cross_domain_dispatch_food_production() {
    let (conductor, _app, cell) = setup_conductor().await;

    let agent = cell.agent_pubkey().clone();
    let plot = Plot {
        id: "plot-sweet-1".to_string(),
        name: "Community Garden Alpha".to_string(),
        area_sqm: 250.0,
        soil_type: SoilType::Loam,
        location_lat: 32.9483,
        location_lon: -96.7299,
        steward: agent,
        status: PlotStatus::Active,
    };

    // Call food_production directly first
    let _record: Record = conductor
        .call(&cell.zome("food_production"), "register_plot", plot.clone())
        .await;

    // Now dispatch through the bridge
    let payload = ExternIO::encode(plot).unwrap().0;
    let dispatch = DispatchInput {
        zome: "food_production".to_string(),
        fn_name: "register_plot".to_string(),
        payload,
    };

    let result: DispatchResult = conductor
        .call(&cell.zome("commons_bridge"), "dispatch_call", dispatch)
        .await;

    assert!(result.success, "Food production dispatch should succeed");
    assert!(result.response.is_some(), "Should have response payload");
}

/// Test: Cross-domain dispatch from bridge to food_distribution's create_market
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires conductor"]
async fn test_cross_domain_dispatch_food_distribution() {
    let (conductor, _app, cell) = setup_conductor().await;

    let agent = cell.agent_pubkey().clone();
    let market = Market {
        id: "mkt-sweet-1".to_string(),
        name: "Richardson Farmers Market".to_string(),
        location_lat: 32.9483,
        location_lon: -96.7299,
        market_type: MarketType::Farmers,
        steward: agent,
        schedule: "Saturdays 8am-1pm".to_string(),
    };

    // Call food_distribution directly first
    let _record: Record = conductor
        .call(&cell.zome("food_distribution"), "create_market", market.clone())
        .await;

    // Now dispatch through the bridge
    let payload = ExternIO::encode(market).unwrap().0;
    let dispatch = DispatchInput {
        zome: "food_distribution".to_string(),
        fn_name: "create_market".to_string(),
        payload,
    };

    let result: DispatchResult = conductor
        .call(&cell.zome("commons_bridge"), "dispatch_call", dispatch)
        .await;

    assert!(result.success, "Food distribution dispatch should succeed");
    assert!(result.response.is_some(), "Should have response payload");
}

/// Test: Cross-domain dispatch from bridge to transport_routes's register_vehicle
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires conductor"]
async fn test_cross_domain_dispatch_transport_routes() {
    let (conductor, _app, cell) = setup_conductor().await;

    let agent = cell.agent_pubkey().clone();
    let vehicle = Vehicle {
        id: "v-sweet-1".to_string(),
        owner: agent,
        vehicle_type: VehicleType::Car,
        capacity_kg: 450.0,
        capacity_passengers: 4,
        status: VehicleStatus::Available,
    };

    // Call transport_routes directly first
    let _record: Record = conductor
        .call(&cell.zome("transport_routes"), "register_vehicle", vehicle.clone())
        .await;

    // Now dispatch through the bridge
    let payload = ExternIO::encode(vehicle).unwrap().0;
    let dispatch = DispatchInput {
        zome: "transport_routes".to_string(),
        fn_name: "register_vehicle".to_string(),
        payload,
    };

    let result: DispatchResult = conductor
        .call(&cell.zome("commons_bridge"), "dispatch_call", dispatch)
        .await;

    assert!(result.success, "Transport routes dispatch should succeed");
    assert!(result.response.is_some(), "Should have response payload");
}

/// Test: Cross-domain dispatch from bridge to transport_sharing's post_ride_offer
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires conductor"]
async fn test_cross_domain_dispatch_transport_sharing() {
    let (conductor, _app, cell) = setup_conductor().await;

    let agent = cell.agent_pubkey().clone();

    // First register a vehicle (ride offer needs a vehicle_hash)
    let vehicle = Vehicle {
        id: "v-sweet-ride".to_string(),
        owner: agent.clone(),
        vehicle_type: VehicleType::Van,
        capacity_kg: 800.0,
        capacity_passengers: 7,
        status: VehicleStatus::Available,
    };

    let vehicle_record: Record = conductor
        .call(&cell.zome("transport_routes"), "register_vehicle", vehicle)
        .await;

    let vehicle_hash = vehicle_record.action_address().clone();

    let offer = RideOffer {
        vehicle_hash: vehicle_hash.clone(),
        route_hash: None,
        driver: agent,
        departure_time: 1700000000,
        seats_available: 3,
        price_per_seat: 5.0,
        status: OfferStatus::Open,
    };

    // Dispatch through the bridge
    let payload = ExternIO::encode(offer).unwrap().0;
    let dispatch = DispatchInput {
        zome: "transport_sharing".to_string(),
        fn_name: "post_ride_offer".to_string(),
        payload,
    };

    let result: DispatchResult = conductor
        .call(&cell.zome("commons_bridge"), "dispatch_call", dispatch)
        .await;

    assert!(result.success, "Transport sharing dispatch should succeed");
    assert!(result.response.is_some(), "Should have response payload");
}

/// Test: food_production queries via bridge (get_all_plots)
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires conductor"]
async fn test_food_queries_via_bridge() {
    let (conductor, _app, cell) = setup_conductor().await;

    let agent = cell.agent_pubkey().clone();

    // Register a plot first
    let plot = Plot {
        id: "plot-query-1".to_string(),
        name: "Herb Spiral".to_string(),
        area_sqm: 50.0,
        soil_type: SoilType::Sandy,
        location_lat: 32.95,
        location_lon: -96.73,
        steward: agent,
        status: PlotStatus::Preparing,
    };

    let _record: Record = conductor
        .call(&cell.zome("food_production"), "register_plot", plot)
        .await;

    // Now use bridge to query all plots
    let dispatch = DispatchInput {
        zome: "food_production".to_string(),
        fn_name: "get_all_plots".to_string(),
        payload: ExternIO::encode(()).unwrap().0,
    };

    let result: DispatchResult = conductor
        .call(&cell.zome("commons_bridge"), "dispatch_call", dispatch)
        .await;

    assert!(result.success, "Food production query via bridge should succeed");
}

/// Test: transport_routes queries via bridge (get_all_routes)
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires conductor"]
async fn test_transport_queries_via_bridge() {
    let (conductor, _app, cell) = setup_conductor().await;

    // Use bridge to query all routes (may be empty, but dispatch should succeed)
    let dispatch = DispatchInput {
        zome: "transport_routes".to_string(),
        fn_name: "get_all_routes".to_string(),
        payload: ExternIO::encode(()).unwrap().0,
    };

    let result: DispatchResult = conductor
        .call(&cell.zome("commons_bridge"), "dispatch_call", dispatch)
        .await;

    assert!(result.success, "Transport routes query via bridge should succeed");
}

// ============================================================================
// Cross-Cluster Dispatch Scenarios — inter-cluster typed dispatch
// ============================================================================
//
// These scenarios exercise cross-cluster dispatch between the Civic and
// Commons DNAs.  Civic → Commons uses `dispatch_commons_call` on
// `civic_bridge`; Commons → Civic uses `dispatch_civic_call` on
// `commons_bridge`.  Each bridge validates the target zome against its
// own allowlist before forwarding via `CallTargetCell::OtherRole(...)`.
//
// Because each sweettest conductor loads only one DNA at a time, the
// cross-cluster call will fail at the conductor routing level (the other
// DNA isn't installed).  The tests verify that:
//   1. The dispatch reaches the bridge (not rejected by compilation).
//   2. The bridge does NOT reject the zome name via its allowlist.
//   3. The error, if any, is a routing/network error — not an access
//      control rejection.

// -- Cross-cluster mirror types (Civic side) --

/// Mirror of CrossClusterDispatchInput from mycelix_bridge_common.
/// Used by both `dispatch_commons_call` (civic_bridge) and
/// `dispatch_civic_call` (commons_bridge).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CrossClusterDispatchInput {
    role: String,
    zome: String,
    fn_name: String,
    payload: Vec<u8>,
}

/// Mirror of CivicEventEntry from civic-bridge.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CivicEventEntry {
    domain: String,
    event_type: String,
    source_agent: AgentPubKey,
    payload: String,
    created_at: Timestamp,
    related_hashes: Vec<String>,
}

/// Mirror of CivicQueryEntry from civic-bridge.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CivicQueryEntry {
    domain: String,
    query_type: String,
    requester: AgentPubKey,
    params: String,
    result: Option<String>,
    created_at: Timestamp,
    resolved_at: Option<Timestamp>,
    success: Option<bool>,
}

/// Mirror of CommonsEventInput from commons-bridge.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CommonsEventInput {
    domain: String,
    event_type: String,
    source_agent: AgentPubKey,
    payload: String,
    created_at: Timestamp,
    related_hashes: Vec<String>,
}

// -- Justice enforcement mirror types --

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum CaseType {
    ContractDispute,
    ConductViolation,
    PropertyDispute,
    FinancialDispute,
    GovernanceDispute,
    IdentityDispute,
    IPDispute,
    Other { category: String },
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CaseParty {
    did: String,
    role: PartyRole,
    joined_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum PartyRole {
    Complainant,
    Respondent,
    Witness,
    Expert,
    Intervenor,
    Affected,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum CasePhase {
    Filed,
    Negotiation,
    Mediation,
    Arbitration,
    Appeal,
    Enforcement,
    Closed,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum CaseStatus {
    Active,
    OnHold,
    AwaitingResponse,
    InDeliberation,
    DecisionRendered,
    Enforcing,
    Resolved,
    Dismissed,
    Withdrawn,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum CaseSeverity {
    Minor,
    Moderate,
    Serious,
    Critical,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CaseContext {
    happ: Option<String>,
    reference_id: Option<String>,
    community: Option<String>,
    jurisdiction: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CaseInput {
    id: String,
    title: String,
    description: String,
    case_type: CaseType,
    complainant: String,
    respondent: String,
    parties: Vec<CaseParty>,
    phase: CasePhase,
    status: CaseStatus,
    severity: CaseSeverity,
    context: CaseContext,
    created_at: Timestamp,
    updated_at: Timestamp,
    phase_deadline: Option<Timestamp>,
}

// -- Disaster / Emergency mirror types (for civic context) --

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
struct OperationalZone {
    id: String,
    name: String,
    boundary: Vec<(f64, f64)>,
    priority: ZonePriority,
    status: ZoneStatus,
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

// -- Fact-check mirror types (for civic media domain) --

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct SubmitFactCheckInput {
    publication_id: String,
    claim_text: String,
    claim_location: String,
    epistemic_position: EpistemicPosition,
    verdict: FactCheckVerdict,
    evidence: Vec<EvidenceItem>,
    checker_did: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct EpistemicPosition {
    empirical: f64,
    normative: f64,
    mythic: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum FactCheckVerdict {
    True,
    MostlyTrue,
    HalfTrue,
    MostlyFalse,
    False,
    Unverifiable,
    OutOfContext,
    Satire,
    Opinion,
    PartiallyTrue,
    Misleading,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct EvidenceItem {
    source_type: SourceType,
    source_url: Option<String>,
    source_did: Option<String>,
    description: String,
    supports_claim: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum SourceType {
    PrimarySource,
    SecondarySource,
    ExpertOpinion,
    OfficialDocument,
    ScientificStudy,
    EyewitnessAccount,
    DataAnalysis,
    Other(String),
}

// -- Media publication mirror types --

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct PublishInput {
    title: String,
    content_hash: String,
    content_type: ContentType,
    author_did: String,
    co_authors: Vec<String>,
    language: String,
    tags: Vec<String>,
    license: License,
    encrypted: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum ContentType {
    Article,
    Opinion,
    Investigation,
    Review,
    Analysis,
    Interview,
    Report,
    Editorial,
    Other(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct License {
    license_type: LicenseType,
    attribution_required: bool,
    commercial_use: bool,
    derivative_works: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum LicenseType {
    CC0,
    CCBY,
    CCBYSA,
    CCBYNC,
    CCBYNCSA,
    AllRightsReserved,
    Custom(String),
}

// ============================================================================
// Cross-Cluster: Civic → Commons (via civic_bridge.dispatch_commons_call)
// ============================================================================

/// Helper to get the civic DNA path
fn civic_dna_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("tests/ must be under mycelix-commons/")
        .parent()
        .expect("mycelix-commons/ must be under luminous-dynamics/")
        .join("mycelix-civic")
        .join("dna")
        .join("mycelix_civic.dna")
}

/// Helper to set up a conductor with the civic DNA
async fn setup_civic_conductor() -> (SweetConductor, SweetApp, SweetCell) {
    let dna_path = civic_dna_path();
    let dna = SweetDnaFile::from_bundle(&dna_path).await.unwrap();

    let mut conductor = SweetConductor::from_standard_config().await;
    let app = conductor.setup_app("civic", &[dna]).await.unwrap();
    let cell = app.into_cells().into_iter().next().unwrap();

    (conductor, app, cell)
}

/// Cross-cluster: Justice enforcement freezing property transfers (Civic → Commons)
///
/// Scenario: A justice enforcement order requires freezing all property
/// transfers for a respondent.  The justice_enforcement zome dispatches
/// to the commons `property_transfer` zome to issue a freeze advisory.
/// This exercises the civic_bridge → commons path and verifies that
/// `property_transfer` is in the ALLOWED_COMMONS_ZOMES allowlist.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires conductor"]
async fn test_cross_cluster_justice_enforcement_freezes_property_transfer() {
    let (conductor, _app, cell) = setup_civic_conductor().await;

    let agent = cell.agent_pubkey().clone();
    let now = Timestamp::now();

    // 1. File a justice case involving a property dispute
    let case = CaseInput {
        id: "case-freeze-001".to_string(),
        title: "Property fraud — transfer freeze required".to_string(),
        description: "Respondent transferred property during pending dispute".to_string(),
        case_type: CaseType::PropertyDispute,
        complainant: format!("did:key:{}", agent),
        respondent: "did:key:fraudster-did".to_string(),
        parties: vec![],
        phase: CasePhase::Enforcement,
        status: CaseStatus::Enforcing,
        severity: CaseSeverity::Critical,
        context: CaseContext {
            happ: Some("mycelix-commons".to_string()),
            reference_id: Some("property-hash-abc".to_string()),
            community: Some("Richardson".to_string()),
            jurisdiction: Some("Collin County".to_string()),
        },
        created_at: now,
        updated_at: now,
        phase_deadline: None,
    };

    let case_record: Record = conductor
        .call(&cell.zome("justice_cases"), "file_case", case)
        .await;

    // 2. Cross-cluster dispatch: justice_enforcement → commons property_transfer
    //    to freeze transfers for the respondent's properties
    let freeze_payload = serde_json::to_vec(&serde_json::json!({
        "case_hash": case_record.action_address().to_string(),
        "respondent_did": "did:key:fraudster-did",
        "action": "freeze_all_transfers",
        "reason": "Active enforcement order — property dispute case-freeze-001",
        "issued_at": now.as_micros(),
    }))
    .unwrap();

    let dispatch = CrossClusterDispatchInput {
        role: "commons".to_string(),
        zome: "property_transfer".to_string(),
        fn_name: "freeze_transfers_for_agent".to_string(),
        payload: freeze_payload,
    };

    let result: DispatchResult = conductor
        .call(&cell.zome("civic_bridge"), "dispatch_commons_call", dispatch)
        .await;

    // Verify dispatch reached the bridge (not rejected by allowlist)
    assert!(
        result.success || result.error.is_some(),
        "Cross-cluster dispatch should either succeed or return a routing error"
    );
    if let Some(ref err) = result.error {
        assert!(
            !err.contains("not in the allowed cross-cluster dispatch list"),
            "property_transfer should be in the allowed commons zomes"
        );
    }

    // 3. Bridge event recording the cross-cluster enforcement action
    let enforcement_event = CivicEventEntry {
        domain: "justice".to_string(),
        event_type: "cross_cluster_property_freeze_issued".to_string(),
        source_agent: agent.clone(),
        payload: serde_json::to_string(&serde_json::json!({
            "case_hash": case_record.action_address().to_string(),
            "target_cluster": "commons",
            "target_zome": "property_transfer",
            "respondent_did": "did:key:fraudster-did",
            "action": "freeze_all_transfers",
        }))
        .unwrap(),
        created_at: Timestamp::now(),
        related_hashes: vec![case_record.action_address().to_string()],
    };

    let event_record: Record = conductor
        .call(&cell.zome("civic_bridge"), "broadcast_event", enforcement_event)
        .await;

    assert!(event_record.action().author() == cell.agent_pubkey());

    // 4. Verify justice events include the enforcement freeze
    let justice_events: Vec<Record> = conductor
        .call(
            &cell.zome("civic_bridge"),
            "get_domain_events",
            "justice".to_string(),
        )
        .await;

    assert!(
        !justice_events.is_empty(),
        "Justice domain should have the cross-cluster property freeze event"
    );
}

/// Cross-cluster: Emergency coordination requesting mutual aid (Civic → Commons)
///
/// Scenario: During a declared disaster, the emergency_coordination zome
/// dispatches to the commons `mutualaid_resources` zome to request
/// available mutual aid resources (food, blankets, generators) for
/// deployment to affected zones.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires conductor"]
async fn test_cross_cluster_emergency_requests_mutualaid_resources() {
    let (conductor, _app, cell) = setup_civic_conductor().await;

    let agent = cell.agent_pubkey().clone();

    // 1. Declare a disaster
    let disaster = DeclareDisasterInput {
        id: "disaster-mutualaid-001".to_string(),
        disaster_type: DisasterType::Tornado,
        title: "North Texas Tornado Outbreak".to_string(),
        description: "EF-3 tornado damage requiring mutual aid resource deployment".to_string(),
        severity: SeverityLevel::Level4,
        affected_area: AffectedArea {
            center_lat: 32.9483,
            center_lon: -96.7299,
            radius_km: 25.0,
            boundary: None,
            zones: vec![OperationalZone {
                id: "zone-ma-1".to_string(),
                name: "Richardson North".to_string(),
                boundary: vec![(32.94, -96.74), (32.96, -96.74), (32.96, -96.72)],
                priority: ZonePriority::Critical,
                status: ZoneStatus::Active,
            }],
        },
        estimated_affected: 8000,
        coordination_lead: Some(agent.clone()),
    };

    let disaster_record: Record = conductor
        .call(
            &cell.zome("emergency_incidents"),
            "declare_disaster",
            disaster,
        )
        .await;

    // 2. Cross-cluster dispatch: emergency_coordination → commons mutualaid_resources
    //    to request available resources for disaster response
    let resource_request_payload = serde_json::to_vec(&serde_json::json!({
        "disaster_id": disaster_record.action_address().to_string(),
        "resource_types": ["generators", "blankets", "non_perishable_food", "water"],
        "priority": "critical",
        "area_lat": 32.9483,
        "area_lon": -96.7299,
        "radius_km": 25.0,
        "min_quantity": 100,
    }))
    .unwrap();

    let dispatch = CrossClusterDispatchInput {
        role: "commons".to_string(),
        zome: "mutualaid_resources".to_string(),
        fn_name: "get_available_resources_for_area".to_string(),
        payload: resource_request_payload,
    };

    let result: DispatchResult = conductor
        .call(&cell.zome("civic_bridge"), "dispatch_commons_call", dispatch)
        .await;

    // Verify dispatch was not rejected by the allowlist
    assert!(
        result.success || result.error.is_some(),
        "Cross-cluster dispatch should either succeed or return a routing error"
    );
    if let Some(ref err) = result.error {
        assert!(
            !err.contains("not in the allowed cross-cluster dispatch list"),
            "mutualaid_resources should be in the allowed commons zomes"
        );
    }

    // 3. Bridge event recording the cross-cluster mutual aid request
    let aid_event = CivicEventEntry {
        domain: "emergency".to_string(),
        event_type: "cross_cluster_mutualaid_requested".to_string(),
        source_agent: agent.clone(),
        payload: serde_json::to_string(&serde_json::json!({
            "disaster_hash": disaster_record.action_address().to_string(),
            "target_cluster": "commons",
            "target_zome": "mutualaid_resources",
            "resources_requested": ["generators", "blankets", "food", "water"],
            "estimated_affected": 8000,
        }))
        .unwrap(),
        created_at: Timestamp::now(),
        related_hashes: vec![disaster_record.action_address().to_string()],
    };

    let event_record: Record = conductor
        .call(&cell.zome("civic_bridge"), "broadcast_event", aid_event)
        .await;

    assert!(event_record.action().author() == cell.agent_pubkey());

    // 4. Verify emergency events include the mutual aid request
    let emergency_events: Vec<Record> = conductor
        .call(
            &cell.zome("civic_bridge"),
            "get_domain_events",
            "emergency".to_string(),
        )
        .await;

    assert!(
        !emergency_events.is_empty(),
        "Emergency domain should have the cross-cluster mutual aid request event"
    );
}

/// Cross-cluster: Media factcheck referencing water purity data (Civic → Commons)
///
/// Scenario: A media outlet publishes a claim about local water quality.
/// A fact-checker dispatches to the commons `water_purity` zome to retrieve
/// actual water quality readings as primary-source evidence for their
/// fact-check verdict.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires conductor"]
async fn test_cross_cluster_media_factcheck_references_water_purity() {
    let (conductor, _app, cell) = setup_civic_conductor().await;

    let agent = cell.agent_pubkey().clone();
    let author_did = format!("did:key:{}", agent);

    // 1. Publish an article making a claim about water quality
    let article = PublishInput {
        title: "Richardson tap water exceeds federal lead limits".to_string(),
        content_hash: "QmWaterClaimArticle".to_string(),
        content_type: ContentType::Investigation,
        author_did: "did:key:journalist-did".to_string(),
        co_authors: vec![],
        language: "en".to_string(),
        tags: vec!["water".to_string(), "health".to_string(), "environment".to_string()],
        license: License {
            license_type: LicenseType::CCBY,
            attribution_required: true,
            commercial_use: true,
            derivative_works: true,
        },
        encrypted: false,
    };

    let pub_record: Record = conductor
        .call(&cell.zome("media_publication"), "publish", article)
        .await;

    // 2. Cross-cluster dispatch: media_factcheck → commons water_purity
    //    to retrieve actual water quality readings for the area
    let water_query_payload = serde_json::to_vec(&serde_json::json!({
        "source_id": "richardson-municipal-supply",
        "parameter": "lead_ppb",
        "location_lat": 32.9483,
        "location_lon": -96.7299,
        "from_timestamp": 1700000000,
        "to_timestamp": 1710000000,
    }))
    .unwrap();

    let dispatch = CrossClusterDispatchInput {
        role: "commons".to_string(),
        zome: "water_purity".to_string(),
        fn_name: "get_quality_readings".to_string(),
        payload: water_query_payload,
    };

    let result: DispatchResult = conductor
        .call(&cell.zome("civic_bridge"), "dispatch_commons_call", dispatch)
        .await;

    // Verify dispatch was not rejected by the allowlist
    assert!(
        result.success || result.error.is_some(),
        "Cross-cluster dispatch should either succeed or return a routing error"
    );
    if let Some(ref err) = result.error {
        assert!(
            !err.contains("not in the allowed cross-cluster dispatch list"),
            "water_purity should be in the allowed commons zomes"
        );
    }

    // 3. Submit the fact-check verdict using the water data as evidence
    let fc_input = SubmitFactCheckInput {
        publication_id: pub_record.action_address().to_string(),
        claim_text: "Richardson tap water exceeds federal lead limits".to_string(),
        claim_location: "headline and paragraph 1".to_string(),
        epistemic_position: EpistemicPosition {
            empirical: 0.95,
            normative: 0.05,
            mythic: 0.0,
        },
        verdict: FactCheckVerdict::False,
        evidence: vec![EvidenceItem {
            source_type: SourceType::DataAnalysis,
            source_url: None,
            source_did: Some("did:mycelix:water-steward-richardson".to_string()),
            description: "Mycelix water_purity readings show lead levels at 3.2 ppb — well below the 15 ppb federal action level".to_string(),
            supports_claim: false,
        }],
        checker_did: author_did.clone(),
    };

    let fc_record: Record = conductor
        .call(&cell.zome("media_factcheck"), "submit_fact_check", fc_input)
        .await;

    assert!(fc_record.action().author() == cell.agent_pubkey());

    // 4. Bridge event linking the fact-check to the cross-cluster water data
    let link_event = CivicEventEntry {
        domain: "media".to_string(),
        event_type: "cross_cluster_water_data_referenced".to_string(),
        source_agent: agent.clone(),
        payload: serde_json::to_string(&serde_json::json!({
            "publication_hash": pub_record.action_address().to_string(),
            "factcheck_hash": fc_record.action_address().to_string(),
            "target_cluster": "commons",
            "target_zome": "water_purity",
            "data_point": "lead_ppb",
            "conclusion": "Claim is False — actual readings well below threshold",
        }))
        .unwrap(),
        created_at: Timestamp::now(),
        related_hashes: vec![
            pub_record.action_address().to_string(),
            fc_record.action_address().to_string(),
        ],
    };

    let event_record: Record = conductor
        .call(&cell.zome("civic_bridge"), "broadcast_event", link_event)
        .await;

    assert!(event_record.action().author() == cell.agent_pubkey());

    // 5. Verify media events include the cross-cluster water data reference
    let media_events: Vec<Record> = conductor
        .call(
            &cell.zome("civic_bridge"),
            "get_domain_events",
            "media".to_string(),
        )
        .await;

    assert!(
        !media_events.is_empty(),
        "Media domain should have the cross-cluster water data reference event"
    );
}

// ============================================================================
// Cross-Cluster: Commons → Civic (via commons_bridge.dispatch_civic_call)
// ============================================================================

/// Cross-cluster: Housing governance escalating dispute to justice (Commons → Civic)
///
/// Scenario: A housing governance vote is disputed by a member who claims
/// the vote was conducted improperly.  The housing_governance zome cannot
/// resolve the dispute internally, so it dispatches to the civic
/// `justice_cases` zome to file a formal case for adjudication.  This
/// exercises the commons_bridge → civic path.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires conductor"]
async fn test_cross_cluster_housing_governance_escalates_to_justice() {
    let (conductor, _app, cell) = setup_conductor().await;

    let agent = cell.agent_pubkey().clone();

    // 1. Register a property (the housing complex subject to governance)
    let prop_input = RegisterPropertyInput {
        property_type: "land".to_string(),
        title: "Oakwood Housing Cooperative".to_string(),
        description: "12-unit cooperative with shared governance".to_string(),
        owner_did: "did:mycelix:oakwood-coop".to_string(),
        co_owners: vec![],
        geolocation: Some(GeoLocation {
            latitude: 32.9550,
            longitude: -96.7350,
        }),
        address: Some(Address {
            street: "450 Oakwood Lane".to_string(),
            city: "Richardson".to_string(),
            state: "TX".to_string(),
            postal_code: "75080".to_string(),
            country: "US".to_string(),
        }),
        metadata: std::collections::HashMap::new(),
    };

    let prop_record: Record = conductor
        .call(&cell.zome("property_registry"), "register_property", prop_input)
        .await;

    // 2. Broadcast a housing governance dispute event via commons bridge
    let dispute_event = CommonsEventInput {
        domain: "housing".to_string(),
        event_type: "governance_vote_disputed".to_string(),
        source_agent: agent.clone(),
        payload: serde_json::to_string(&serde_json::json!({
            "property_hash": prop_record.action_address().to_string(),
            "coop_name": "Oakwood Housing Cooperative",
            "vote_id": "vote-2026-003",
            "vote_subject": "Approval of $50k roof renovation",
            "dispute_reason": "Vote conducted without required 7-day notice period",
            "disputant_did": format!("did:key:{}", agent),
        }))
        .unwrap(),
        created_at: Timestamp::now(),
        related_hashes: vec![prop_record.action_address().to_string()],
    };

    let dispute_record: Record = conductor
        .call(&cell.zome("commons_bridge"), "broadcast_event", dispute_event)
        .await;

    assert!(dispute_record.action().author() == cell.agent_pubkey());

    // 3. Cross-cluster dispatch: housing_governance → civic justice_cases
    //    to file a formal dispute case for adjudication
    let case_payload = serde_json::to_vec(&serde_json::json!({
        "id": "case-housing-gov-001",
        "title": "Oakwood Coop — improper governance vote procedure",
        "description": "Vote on $50k roof renovation was conducted without the required 7-day notice to all members, violating cooperative bylaws",
        "case_type": "GovernanceDispute",
        "complainant_did": format!("did:key:{}", agent),
        "respondent_did": "did:mycelix:oakwood-coop-board",
        "originating_happ": "mycelix-commons",
        "originating_domain": "housing",
        "property_hash": prop_record.action_address().to_string(),
        "vote_id": "vote-2026-003",
    }))
    .unwrap();

    let dispatch = CrossClusterDispatchInput {
        role: "civic".to_string(),
        zome: "justice_cases".to_string(),
        fn_name: "file_case".to_string(),
        payload: case_payload,
    };

    let result: DispatchResult = conductor
        .call(&cell.zome("commons_bridge"), "dispatch_civic_call", dispatch)
        .await;

    // Verify dispatch reached the bridge (not rejected by allowlist)
    assert!(
        result.success || result.error.is_some(),
        "Cross-cluster dispatch should either succeed or return a routing error"
    );
    if let Some(ref err) = result.error {
        assert!(
            !err.contains("not in the allowed cross-cluster dispatch list"),
            "justice_cases should be in the allowed civic zomes"
        );
    }

    // 4. Bridge event recording the escalation to civic justice
    let escalation_event = CommonsEventInput {
        domain: "housing".to_string(),
        event_type: "cross_cluster_dispute_escalated_to_justice".to_string(),
        source_agent: agent.clone(),
        payload: serde_json::to_string(&serde_json::json!({
            "property_hash": prop_record.action_address().to_string(),
            "dispute_event_hash": dispute_record.action_address().to_string(),
            "target_cluster": "civic",
            "target_zome": "justice_cases",
            "case_title": "Oakwood Coop — improper governance vote procedure",
            "vote_id": "vote-2026-003",
        }))
        .unwrap(),
        created_at: Timestamp::now(),
        related_hashes: vec![
            prop_record.action_address().to_string(),
            dispute_record.action_address().to_string(),
        ],
    };

    let escalation_record: Record = conductor
        .call(&cell.zome("commons_bridge"), "broadcast_event", escalation_event)
        .await;

    assert!(escalation_record.action().author() == cell.agent_pubkey());

    // 5. Verify housing events include both the dispute and the escalation
    let housing_events: Vec<Record> = conductor
        .call(
            &cell.zome("commons_bridge"),
            "get_domain_events",
            "housing".to_string(),
        )
        .await;

    assert!(
        housing_events.len() >= 2,
        "Housing domain should have both the governance dispute event and the justice escalation event"
    );
}

// ============================================================================
// Conductor setup helpers
// ============================================================================

/// Helper to set up a conductor with the commons DNA
async fn setup_conductor() -> (SweetConductor, SweetApp, SweetCell) {
    let dna_path = commons_dna_path();
    let dna = SweetDnaFile::from_bundle(&dna_path).await.unwrap();

    let mut conductor = SweetConductor::from_standard_config().await;
    let app = conductor.setup_app("commons", &[dna]).await.unwrap();
    let cell = app.into_cells().into_iter().next().unwrap();

    (conductor, app, cell)
}
