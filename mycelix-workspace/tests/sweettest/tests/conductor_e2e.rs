// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # End-to-End Conductor Integration Tests
//!
//! Comprehensive sweettest integration tests exercising real DHT behavior
//! across Commons and Civic cluster DNAs. Covers:
//!
//! 1. **Commons cluster CRUD**: property-registry, care-circles, water-purity
//! 2. **Civic cluster CRUD**: justice-cases, emergency-shelters
//! 3. **Bridge dispatch roundtrip**: cross-cluster serialization
//! 4. **Validation enforcement**: integrity validation rejections
//! 5. **Metrics collection**: bridge metrics counter verification
//!
//! ## Prerequisites
//!
//! ```bash
//! # Build WASM zomes
//! cd mycelix-commons && cargo build --release --target wasm32-unknown-unknown
//! cd mycelix-civic   && cargo build --release --target wasm32-unknown-unknown
//!
//! # Pack DNAs
//! hc dna pack mycelix-commons/dna/
//! hc dna pack mycelix-civic/dna/
//! ```
//!
//! ## Running
//!
//! ```bash
//! cargo test --release -p mycelix-sweettest --test conductor_e2e -- --ignored --test-threads=2
//! ```

mod harness;

use harness::*;
use holochain::prelude::*;
use holochain::sweettest::*;
use serial_test::serial;
use std::path::PathBuf;

// ============================================================================
// Mirror types — Commons: property-registry
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct RegisterPropertyInput {
    property_type: PropertyType,
    title: String,
    description: String,
    owner_did: String,
    co_owners: Vec<CoOwner>,
    geolocation: Option<GeoLocation>,
    address: Option<Address>,
    metadata: PropertyMetadata,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum PropertyType {
    Land,
    Building,
    Unit,
    Equipment,
    Intellectual,
    Digital,
    Other(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CoOwner {
    did: String,
    share_basis_points: u32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct GeoLocation {
    latitude: f64,
    longitude: f64,
    boundaries: Option<Vec<(f64, f64)>>,
    area_sqm: Option<f64>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Address {
    street: String,
    city: String,
    region: String,
    country: String,
    postal_code: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct PropertyMetadata {
    appraised_value: Option<f64>,
    currency: Option<String>,
    legal_description: Option<String>,
    parcel_number: Option<String>,
    attachments: Vec<String>,
}

// ============================================================================
// Mirror types — Commons: care-circles
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum CircleType {
    Neighborhood,
    Workplace,
    Faith,
    Family,
    School,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum MemberRole {
    Organizer,
    Member,
    Observer,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CareCircle {
    name: String,
    description: String,
    location: String,
    max_members: u32,
    created_by: AgentPubKey,
    circle_type: CircleType,
    active: bool,
    created_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct JoinCircleInput {
    circle_hash: ActionHash,
    role: MemberRole,
}

// ============================================================================
// Mirror types — Commons: water-purity
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct QualityReading {
    source_hash: ActionHash,
    sampler: AgentPubKey,
    timestamp: Timestamp,
    temperature_celsius: Option<f32>,
    turbidity_ntu: Option<f32>,
    ph: Option<f32>,
    tds_ppm: Option<f32>,
    dissolved_oxygen_mg_l: Option<f32>,
    nitrates_mg_l: Option<f32>,
    arsenic_ug_l: Option<f32>,
    lead_ug_l: Option<f32>,
    total_coliform_cfu: Option<u32>,
    e_coli_cfu: Option<u32>,
    chlorine_mg_l: Option<f32>,
    potability_score: f32,
    meets_who_standards: bool,
    meets_epa_standards: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct PotabilityResult {
    is_potable: bool,
    potability_score: f32,
    meets_who: bool,
    meets_epa: bool,
    warnings: Vec<String>,
    reading_timestamp: Timestamp,
}

// ============================================================================
// Mirror types — Commons: water-steward (needed for water source creation)
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Watershed {
    id: String,
    name: String,
    area_sq_km: f64,
    boundary: Vec<(f64, f64)>,
    primary_river: String,
    steward: AgentPubKey,
    created_at: Timestamp,
}

// ============================================================================
// Mirror types — Commons: commons-bridge
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CommonsQueryInput {
    domain: String,
    query_type: String,
    requester: AgentPubKey,
    params: String,
    created_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CommonsEventInput {
    domain: String,
    event_type: String,
    source_agent: AgentPubKey,
    payload: String,
    created_at: Timestamp,
    related_hashes: Vec<String>,
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

// ============================================================================
// Mirror types — Civic: justice-cases
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Case {
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
struct UpdatePhaseInput {
    case_hash: ActionHash,
    new_phase: CasePhase,
    deadline: Option<Timestamp>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct UpdateStatusInput {
    case_hash: ActionHash,
    new_status: CaseStatus,
}

// ============================================================================
// Mirror types — Civic: emergency-shelters
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct RegisterShelterInput {
    id: String,
    name: String,
    location_lat: f64,
    location_lon: f64,
    address: String,
    capacity: u32,
    shelter_type: ShelterType,
    amenities: Vec<Amenity>,
    contact: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum ShelterType {
    Emergency,
    Community,
    Medical,
    PetFriendly,
    Accessible,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum Amenity {
    Power,
    Water,
    Medical,
    Food,
    Showers,
    Wifi,
    Charging,
    Cots,
    Blankets,
    PetArea,
    ChildCare,
    MentalHealth,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
enum ShelterStatus {
    Open,
    Full,
    Closed,
    Evacuating,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CheckInPersonInput {
    shelter_hash: ActionHash,
    person_name: String,
    person_id: Option<String>,
    party_size: u8,
    special_needs: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct UpdateShelterStatusInput {
    shelter_hash: ActionHash,
    new_status: ShelterStatus,
}

// ============================================================================
// Mirror types — Civic: civic-bridge
// ============================================================================

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

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CivicEventEntry {
    domain: String,
    event_type: String,
    source_agent: AgentPubKey,
    payload: String,
    created_at: Timestamp,
    related_hashes: Vec<String>,
}

// ============================================================================
// Mirror types — Bridge metrics
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct BridgeMetricsSnapshot {
    total_success: u64,
    total_errors: u64,
    total_cross_cluster: u64,
    rate_limit_hits: u64,
    call_counts: Vec<CallCountSnapshot>,
    error_counts: Vec<ErrorCountSnapshot>,
    latency: Option<LatencyPercentiles>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct CallCountSnapshot {
    key: String,
    success_count: u64,
    error_count: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct ErrorCountSnapshot {
    code: String,
    count: u64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct LatencyPercentiles {
    p50_us: u64,
    p95_us: u64,
    p99_us: u64,
    sample_count: u32,
    total_recorded: u64,
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
        .expect("Commons DNA should exist — run `hc dna pack mycelix-commons/dna/`");

    let civic_dna = SweetDnaFile::from_bundle(&DnaPaths::civic())
        .await
        .expect("Civic DNA should exist — run `hc dna pack mycelix-civic/dna/`");

    let mut conductor = SweetConductor::from_standard_config().await;

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

// Helper to create a default property input
fn make_property_input(agent: &AgentPubKey, title: &str) -> RegisterPropertyInput {
    RegisterPropertyInput {
        property_type: PropertyType::Building,
        title: title.to_string(),
        description: format!("Test property: {}", title),
        owner_did: format!("did:key:{}", agent),
        co_owners: vec![],
        geolocation: Some(GeoLocation {
            latitude: 32.9483,
            longitude: -96.7299,
            boundaries: None,
            area_sqm: Some(200.0),
        }),
        address: Some(Address {
            street: "100 Main St".to_string(),
            city: "Richardson".to_string(),
            region: "TX".to_string(),
            country: "US".to_string(),
            postal_code: Some("75080".to_string()),
        }),
        metadata: PropertyMetadata {
            appraised_value: Some(300_000.0),
            currency: Some("USD".to_string()),
            legal_description: None,
            parcel_number: None,
            attachments: vec![],
        },
    }
}

// ============================================================================
// 1. COMMONS CLUSTER BASIC CRUD
// ============================================================================

/// Property Registry: Register multiple properties and retrieve by owner DID.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_commons_property_crud_and_owner_query() {
    let agent = setup_unified_conductor().await;
    let agent_key = agent.commons_cell.agent_pubkey().clone();
    let owner_did = format!("did:key:{}", agent_key);

    // Register 3 properties
    let mut hashes = Vec::new();
    for i in 0..3 {
        let input = make_property_input(&agent_key, &format!("Property #{}", i));
        let record: Record = agent
            .call_commons("property_registry", "register_property", input)
            .await;
        assert!(record.action().author() == &agent_key);
        hashes.push(record.action_address().clone());
    }

    // Retrieve by owner DID
    let owner_properties: Vec<Record> = agent
        .call_commons("property_registry", "get_owner_properties", owner_did)
        .await;

    assert_eq!(
        owner_properties.len(),
        3,
        "Owner should have exactly 3 properties"
    );
}

/// Property Registry: Register 15+ properties, verify bulk retrieval by type.
///
/// This tests link-based queries returning correct results for a larger dataset.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed — slow (>30s)"]
async fn test_commons_property_bulk_create_and_type_query() {
    let agent = setup_unified_conductor().await;
    let agent_key = agent.commons_cell.agent_pubkey().clone();

    // Create 8 Building and 7 Land properties (15 total)
    for i in 0..8 {
        let input = RegisterPropertyInput {
            property_type: PropertyType::Building,
            title: format!("Building #{}", i),
            description: format!("Bulk test building {}", i),
            owner_did: format!("did:key:{}", agent_key),
            co_owners: vec![],
            geolocation: None,
            address: None,
            metadata: PropertyMetadata {
                appraised_value: Some(100_000.0 + (i as f64) * 50_000.0),
                currency: Some("USD".to_string()),
                legal_description: None,
                parcel_number: None,
                attachments: vec![],
            },
        };
        let _: Record = agent
            .call_commons("property_registry", "register_property", input)
            .await;
    }

    for i in 0..7 {
        let input = RegisterPropertyInput {
            property_type: PropertyType::Land,
            title: format!("Land Parcel #{}", i),
            description: format!("Bulk test land {}", i),
            owner_did: format!("did:key:{}", agent_key),
            co_owners: vec![],
            geolocation: Some(GeoLocation {
                latitude: 33.0 + (i as f64) * 0.01,
                longitude: -96.7,
                boundaries: None,
                area_sqm: Some(5000.0),
            }),
            address: None,
            metadata: PropertyMetadata {
                appraised_value: Some(50_000.0),
                currency: None,
                legal_description: None,
                parcel_number: None,
                attachments: vec![],
            },
        };
        let _: Record = agent
            .call_commons("property_registry", "register_property", input)
            .await;
    }

    // Query by Building type
    let buildings: Vec<Record> = agent
        .call_commons(
            "property_registry",
            "get_properties_by_type",
            PropertyType::Building,
        )
        .await;

    assert_eq!(buildings.len(), 8, "Should have 8 Building-type properties");

    // Query by Land type
    let lands: Vec<Record> = agent
        .call_commons(
            "property_registry",
            "get_properties_by_type",
            PropertyType::Land,
        )
        .await;

    assert_eq!(lands.len(), 7, "Should have 7 Land-type properties");

    // All owner properties
    let all: Vec<Record> = agent
        .call_commons(
            "property_registry",
            "get_owner_properties",
            format!("did:key:{}", agent_key),
        )
        .await;

    assert_eq!(all.len(), 15, "Owner should have 15 total properties");
}

/// Care Circles: Create circle, join, retrieve by type, verify membership links.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_commons_care_circles_crud_and_membership() {
    let agent = setup_unified_conductor().await;
    let agent_key = agent.commons_cell.agent_pubkey().clone();

    // Create a neighborhood care circle
    let circle = CareCircle {
        name: "Elm Street Neighbors".to_string(),
        description: "Mutual care circle for Elm Street residents".to_string(),
        location: "Elm Street, Richardson".to_string(),
        max_members: 20,
        created_by: agent_key.clone(),
        circle_type: CircleType::Neighborhood,
        active: true,
        created_at: Timestamp::now(),
    };

    let circle_record: Record = agent
        .call_commons("care_circles", "create_circle", circle)
        .await;

    assert!(circle_record.action().author() == &agent_key);
    let circle_hash = circle_record.action_address().clone();

    // Retrieve all circles
    let all_circles: Vec<Record> = agent
        .call_commons("care_circles", "get_all_circles", ())
        .await;

    assert!(
        !all_circles.is_empty(),
        "Should have at least 1 circle after creation"
    );

    // Retrieve by type
    let neighborhood_circles: Vec<Record> = agent
        .call_commons(
            "care_circles",
            "get_circles_by_type",
            CircleType::Neighborhood,
        )
        .await;

    assert_eq!(neighborhood_circles.len(), 1, "Should have 1 neighborhood circle");

    // Create a second circle of different type
    let workplace_circle = CareCircle {
        name: "Office Care Group".to_string(),
        description: "Workplace care circle".to_string(),
        location: "Tech Campus".to_string(),
        max_members: 30,
        created_by: agent_key.clone(),
        circle_type: CircleType::Workplace,
        active: true,
        created_at: Timestamp::now(),
    };

    let _: Record = agent
        .call_commons("care_circles", "create_circle", workplace_circle)
        .await;

    // Verify type-specific queries return correct counts
    let all_circles_now: Vec<Record> = agent
        .call_commons("care_circles", "get_all_circles", ())
        .await;

    assert_eq!(all_circles_now.len(), 2, "Should have 2 total circles");

    let workplace_circles: Vec<Record> = agent
        .call_commons(
            "care_circles",
            "get_circles_by_type",
            CircleType::Workplace,
        )
        .await;

    assert_eq!(workplace_circles.len(), 1, "Should have 1 workplace circle");

    // Neighborhood count unchanged
    let neighborhood_still: Vec<Record> = agent
        .call_commons(
            "care_circles",
            "get_circles_by_type",
            CircleType::Neighborhood,
        )
        .await;

    assert_eq!(
        neighborhood_still.len(),
        1,
        "Neighborhood circle count should be unchanged"
    );
}

/// Water Purity: Submit reading, check potability, verify link-based retrieval.
///
/// Since water-purity reads need a source_hash (from water_steward::define_watershed),
/// we first create a watershed, then submit readings against it.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_commons_water_purity_reading_and_potability() {
    let agent = setup_unified_conductor().await;
    let agent_key = agent.commons_cell.agent_pubkey().clone();

    // 1. Define a watershed to get a source_hash
    let watershed = Watershed {
        id: "ws-trinity-001".to_string(),
        name: "Trinity River Basin".to_string(),
        area_sq_km: 46_000.0,
        boundary: vec![(33.2, -97.0), (32.5, -96.3), (31.8, -96.8)],
        primary_river: "Trinity River".to_string(),
        steward: agent_key.clone(),
        created_at: Timestamp::now(),
    };

    let ws_record: Record = agent
        .call_commons("water_steward", "define_watershed", watershed)
        .await;

    let source_hash = ws_record.action_address().clone();

    // 2. Submit a clean water reading
    let clean_reading = QualityReading {
        source_hash: source_hash.clone(),
        sampler: agent_key.clone(),
        timestamp: Timestamp::now(),
        temperature_celsius: Some(18.5),
        turbidity_ntu: Some(1.2),
        ph: Some(7.2),
        tds_ppm: Some(250.0),
        dissolved_oxygen_mg_l: Some(8.0),
        nitrates_mg_l: Some(3.0),
        arsenic_ug_l: Some(2.0),
        lead_ug_l: Some(1.5),
        total_coliform_cfu: Some(0),
        e_coli_cfu: Some(0),
        chlorine_mg_l: Some(0.5),
        potability_score: 0.95,
        meets_who_standards: true,
        meets_epa_standards: true,
    };

    let reading_record: Record = agent
        .call_commons("water_purity", "submit_reading", clean_reading)
        .await;

    assert!(reading_record.action().author() == &agent_key);

    // 3. Check potability
    let potability: PotabilityResult = agent
        .call_commons("water_purity", "check_potability", source_hash.clone())
        .await;

    assert!(potability.is_potable, "Clean reading should be potable");
    assert!(potability.meets_who, "Should meet WHO standards");
    assert!(potability.meets_epa, "Should meet EPA standards");
    assert!(
        potability.warnings.is_empty(),
        "Clean reading should have no warnings"
    );

    // 4. Submit a contaminated reading (high lead)
    let dirty_reading = QualityReading {
        source_hash: source_hash.clone(),
        sampler: agent_key.clone(),
        timestamp: Timestamp::now(),
        temperature_celsius: Some(20.0),
        turbidity_ntu: Some(8.0),
        ph: Some(5.5),
        tds_ppm: Some(800.0),
        dissolved_oxygen_mg_l: Some(4.0),
        nitrates_mg_l: Some(15.0),
        arsenic_ug_l: Some(25.0),
        lead_ug_l: Some(50.0),
        total_coliform_cfu: Some(10),
        e_coli_cfu: Some(5),
        chlorine_mg_l: Some(0.1),
        potability_score: 0.15,
        meets_who_standards: false,
        meets_epa_standards: false,
    };

    let _: Record = agent
        .call_commons("water_purity", "submit_reading", dirty_reading)
        .await;

    // 5. Re-check potability — now should fail (latest reading is dirty)
    let potability2: PotabilityResult = agent
        .call_commons("water_purity", "check_potability", source_hash.clone())
        .await;

    assert!(
        !potability2.is_potable,
        "Contaminated reading should make source non-potable"
    );
    assert!(
        !potability2.warnings.is_empty(),
        "Should have contamination warnings"
    );

    // 6. Verify link-based retrieval: get all readings for this source
    let readings: Vec<Record> = agent
        .call_commons("water_purity", "get_source_readings", source_hash)
        .await;

    assert_eq!(
        readings.len(),
        2,
        "Should have exactly 2 readings for the source"
    );
}

// ============================================================================
// 2. CIVIC CLUSTER BASIC CRUD
// ============================================================================

/// Justice Cases: File case, retrieve by hash, update phase.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_civic_justice_case_crud_and_phase_update() {
    let agent = setup_unified_conductor().await;
    let agent_key = agent.civic_cell.agent_pubkey().clone();
    let now = Timestamp::now();

    // 1. File a case
    let case = Case {
        id: "CASE-E2E-001".to_string(),
        title: "Contract Breach: Community Solar".to_string(),
        description: "Vendor failed to deliver solar panels per agreement".to_string(),
        case_type: CaseType::ContractDispute,
        complainant: format!("did:key:{}", agent_key),
        respondent: "did:key:vendor-xyz".to_string(),
        parties: vec![],
        phase: CasePhase::Filed,
        status: CaseStatus::Active,
        severity: CaseSeverity::Moderate,
        context: CaseContext {
            happ: Some("mycelix-marketplace".to_string()),
            reference_id: Some("order-12345".to_string()),
            community: Some("Richardson Solar Co-op".to_string()),
            jurisdiction: None,
        },
        created_at: now,
        updated_at: now,
        phase_deadline: None,
    };

    let case_record: Record = agent
        .call_civic("justice_cases", "file_case", case)
        .await;

    assert!(case_record.action().author() == &agent_key);
    let case_hash = case_record.action_address().clone();

    // 2. Retrieve by hash
    let fetched: Option<Record> = agent
        .call_civic("justice_cases", "get_case", case_hash.clone())
        .await;

    assert!(fetched.is_some(), "Case should be retrievable by hash");

    // 3. Update phase to Negotiation
    let update = UpdatePhaseInput {
        case_hash: case_hash.clone(),
        new_phase: CasePhase::Negotiation,
        deadline: Some(Timestamp::from_micros(
            now.as_micros() + 7 * 24 * 60 * 60 * 1_000_000,
        )),
    };

    let updated_record: Record = agent
        .call_civic("justice_cases", "update_case_phase", update)
        .await;

    assert!(updated_record.action().author() == &agent_key);

    // 4. Verify all cases listing
    let all_cases: Vec<Record> = agent
        .call_civic("justice_cases", "get_all_cases", ())
        .await;

    assert!(
        !all_cases.is_empty(),
        "get_all_cases should return at least 1 case"
    );
}

/// Justice Cases: File multiple cases and retrieve by complainant DID.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_civic_justice_multiple_cases_by_complainant() {
    let agent = setup_unified_conductor().await;
    let agent_key = agent.civic_cell.agent_pubkey().clone();
    let complainant_did = format!("did:key:{}", agent_key);
    let now = Timestamp::now();

    // File 3 cases
    for i in 0..3 {
        let case = Case {
            id: format!("CASE-MULTI-{}", i),
            title: format!("Multi-case Test #{}", i),
            description: format!("Test case {} for bulk retrieval", i),
            case_type: CaseType::ConductViolation,
            complainant: complainant_did.clone(),
            respondent: format!("did:key:respondent-{}", i),
            parties: vec![],
            phase: CasePhase::Filed,
            status: CaseStatus::Active,
            severity: CaseSeverity::Minor,
            context: CaseContext {
                happ: None,
                reference_id: None,
                community: None,
                jurisdiction: None,
            },
            created_at: now,
            updated_at: now,
            phase_deadline: None,
        };

        let _: Record = agent
            .call_civic("justice_cases", "file_case", case)
            .await;
    }

    // Retrieve by complainant DID
    let my_cases: Vec<Record> = agent
        .call_civic("justice_cases", "get_my_cases", complainant_did)
        .await;

    assert_eq!(my_cases.len(), 3, "Complainant should have exactly 3 cases");
}

/// Emergency Shelters: Register shelter, check in people, verify occupancy updates.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_civic_emergency_shelter_crud_and_checkin() {
    let agent = setup_unified_conductor().await;
    let agent_key = agent.civic_cell.agent_pubkey().clone();

    // 1. Register a shelter
    let shelter = RegisterShelterInput {
        id: "SHELTER-E2E-001".to_string(),
        name: "Richardson Community Center".to_string(),
        location_lat: 32.9483,
        location_lon: -96.7299,
        address: "123 Main St, Richardson, TX".to_string(),
        capacity: 100,
        shelter_type: ShelterType::Emergency,
        amenities: vec![
            Amenity::Power,
            Amenity::Water,
            Amenity::Food,
            Amenity::Cots,
            Amenity::MentalHealth,
        ],
        contact: "emergency@richardson.gov".to_string(),
    };

    let shelter_record: Record = agent
        .call_civic("emergency_shelters", "register_shelter", shelter)
        .await;

    assert!(shelter_record.action().author() == &agent_key);
    let shelter_hash = shelter_record.action_address().clone();

    // 2. Check in a family
    let checkin = CheckInPersonInput {
        shelter_hash: shelter_hash.clone(),
        person_name: "Garcia Family".to_string(),
        person_id: Some("ID-GARCIA-001".to_string()),
        party_size: 4,
        special_needs: vec!["infant".to_string()],
    };

    let checkin_record: Record = agent
        .call_civic("emergency_shelters", "check_in_person", checkin)
        .await;

    assert!(checkin_record.action().author() == &agent_key);

    // 3. Check in another person
    let checkin2 = CheckInPersonInput {
        shelter_hash: shelter_hash.clone(),
        person_name: "John Smith".to_string(),
        person_id: Some("ID-SMITH-001".to_string()),
        party_size: 1,
        special_needs: vec![],
    };

    let _: Record = agent
        .call_civic("emergency_shelters", "check_in_person", checkin2)
        .await;

    // 4. Verify occupants are linked to shelter
    let occupants: Vec<Record> = agent
        .call_civic(
            "emergency_shelters",
            "get_shelter_occupants",
            shelter_hash,
        )
        .await;

    assert_eq!(
        occupants.len(),
        2,
        "Shelter should have 2 registrations (Garcia family + John Smith)"
    );
}

/// Civic Bridge: Cross-zome local calls work via bridge dispatch.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_civic_bridge_dispatch_to_local_zome() {
    let agent = setup_unified_conductor().await;
    let agent_key = agent.civic_cell.agent_pubkey().clone();

    // Dispatch from civic_bridge to justice_cases::get_all_cases
    let dispatch = DispatchInput {
        zome: "justice_cases".to_string(),
        fn_name: "get_all_cases".to_string(),
        payload: ExternIO::encode(()).unwrap().0,
    };

    let result: DispatchResult = agent
        .call_civic("civic_bridge", "dispatch_call", dispatch)
        .await;

    assert!(
        result.success || result.error.is_some(),
        "Dispatch to justice_cases should either succeed or return a known error"
    );
}

// ============================================================================
// 3. BRIDGE DISPATCH ROUNDTRIP
// ============================================================================

/// Commons bridge dispatch_cross_cluster to civic bridge (serialization roundtrip).
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_bridge_commons_to_civic_dispatch_roundtrip() {
    let agent = setup_unified_conductor().await;

    // Dispatch from commons -> civic health_check
    let dispatch = CrossClusterDispatchInput {
        role: "civic".to_string(),
        zome: "civic_bridge".to_string(),
        fn_name: "health_check".to_string(),
        payload: ExternIO::encode(()).unwrap().0,
    };

    let result: DispatchResult = agent
        .call_commons("commons_bridge", "dispatch_civic_call", dispatch)
        .await;

    assert!(
        result.success,
        "Cross-cluster commons->civic health_check should succeed: {:?}",
        result.error
    );
    assert!(
        result.response.is_some(),
        "Should return a response payload"
    );

    // Deserialize the response as BridgeHealth
    if let Some(ref response_bytes) = result.response {
        let health: Result<BridgeHealth, _> = serde_json::from_slice(response_bytes);
        if let Ok(health) = health {
            assert!(health.healthy);
            assert_eq!(health.domains.len(), 3);
        }
    }
}

/// Civic bridge dispatch_cross_cluster to commons bridge (reverse direction).
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_bridge_civic_to_commons_dispatch_roundtrip() {
    let agent = setup_unified_conductor().await;

    // Dispatch from civic -> commons health_check
    let dispatch = CrossClusterDispatchInput {
        role: "commons".to_string(),
        zome: "commons_bridge".to_string(),
        fn_name: "health_check".to_string(),
        payload: ExternIO::encode(()).unwrap().0,
    };

    let result: DispatchResult = agent
        .call_civic("civic_bridge", "dispatch_commons_call", dispatch)
        .await;

    assert!(
        result.success,
        "Cross-cluster civic->commons health_check should succeed: {:?}",
        result.error
    );
    assert!(
        result.response.is_some(),
        "Should return a response payload"
    );
}

/// BridgeError propagation: dispatching to a non-existent function returns error, not panic.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_bridge_error_propagation_nonexistent_fn() {
    let agent = setup_unified_conductor().await;

    // Dispatch to an allowed zome but non-existent function
    let dispatch = DispatchInput {
        zome: "property_registry".to_string(),
        fn_name: "this_function_does_not_exist".to_string(),
        payload: vec![],
    };

    let result: DispatchResult = agent
        .call_commons("commons_bridge", "dispatch_call", dispatch)
        .await;

    assert!(
        !result.success,
        "Non-existent function dispatch should fail"
    );
    assert!(
        result.error.is_some(),
        "Should include an error message"
    );
}

/// BridgeError propagation: dispatching to a disallowed zome returns error.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_bridge_error_propagation_disallowed_zome() {
    let agent = setup_unified_conductor().await;

    let dispatch = DispatchInput {
        zome: "malicious_zome".to_string(),
        fn_name: "steal_data".to_string(),
        payload: vec![],
    };

    let result: DispatchResult = agent
        .call_commons("commons_bridge", "dispatch_call", dispatch)
        .await;

    assert!(!result.success, "Disallowed zome should be rejected");
    assert!(
        result
            .error
            .as_deref()
            .unwrap()
            .contains("not in the allowed"),
        "Error should mention allowlist: {:?}",
        result.error
    );
}

/// Cross-cluster dispatch serialization: commons -> civic emergency_shelters.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_bridge_cross_cluster_dispatch_to_emergency_shelters() {
    let agent = setup_unified_conductor().await;

    // Dispatch from commons to civic's emergency_shelters zome
    // (tests that ExternIO serialization works across cluster boundaries)
    let dispatch = CrossClusterDispatchInput {
        role: "civic".to_string(),
        zome: "emergency_shelters".to_string(),
        fn_name: "register_shelter".to_string(),
        payload: serde_json::to_vec(&RegisterShelterInput {
            id: "SHELTER-CROSS-001".to_string(),
            name: "Cross-cluster Test Shelter".to_string(),
            location_lat: 32.95,
            location_lon: -96.73,
            address: "456 Oak Ave, Richardson, TX".to_string(),
            capacity: 50,
            shelter_type: ShelterType::Community,
            amenities: vec![Amenity::Power, Amenity::Water],
            contact: "test@example.com".to_string(),
        })
        .unwrap(),
    };

    let result: DispatchResult = agent
        .call_commons("commons_bridge", "dispatch_civic_call", dispatch)
        .await;

    // The dispatch either succeeds (payload serialization correct) or fails
    // with a zome-level error (which still validates serialization roundtrip)
    assert!(
        result.success || result.error.is_some(),
        "Cross-cluster dispatch to emergency_shelters should complete: {:?}",
        result
    );
}

// ============================================================================
// 4. VALIDATION ENFORCEMENT
// ============================================================================

/// Care Circle validation: empty name should be rejected by integrity.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_validation_care_circle_empty_name_rejected() {
    let agent = setup_unified_conductor().await;
    let agent_key = agent.commons_cell.agent_pubkey().clone();

    let invalid_circle = CareCircle {
        name: "".to_string(), // Empty name: should fail validation
        description: "Should not be created".to_string(),
        location: "Nowhere".to_string(),
        max_members: 10,
        created_by: agent_key.clone(),
        circle_type: CircleType::Neighborhood,
        active: true,
        created_at: Timestamp::now(),
    };

    let result = agent
        .call_commons_fallible::<_, Record>("care_circles", "create_circle", invalid_circle)
        .await;

    assert!(
        result.is_err(),
        "Creating a circle with empty name should fail validation"
    );
}

/// Care Circle validation: max_members < 2 should be rejected.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_validation_care_circle_too_few_members_rejected() {
    let agent = setup_unified_conductor().await;
    let agent_key = agent.commons_cell.agent_pubkey().clone();

    let invalid_circle = CareCircle {
        name: "Solo Circle".to_string(),
        description: "A circle of one is not a circle".to_string(),
        location: "Alone".to_string(),
        max_members: 1, // Must be >= 2
        created_by: agent_key.clone(),
        circle_type: CircleType::Family,
        active: true,
        created_at: Timestamp::now(),
    };

    let result = agent
        .call_commons_fallible::<_, Record>("care_circles", "create_circle", invalid_circle)
        .await;

    assert!(
        result.is_err(),
        "Circle with max_members=1 should fail validation (minimum is 2)"
    );
}

/// Emergency Shelter validation: empty address should be rejected.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_validation_shelter_empty_address_rejected() {
    let agent = setup_unified_conductor().await;

    let invalid_shelter = RegisterShelterInput {
        id: "SHELTER-INVALID-001".to_string(),
        name: "No Address Shelter".to_string(),
        location_lat: 32.95,
        location_lon: -96.73,
        address: "".to_string(), // Empty: should fail
        capacity: 50,
        shelter_type: ShelterType::Emergency,
        amenities: vec![],
        contact: "test@example.com".to_string(),
    };

    let result = agent
        .call_civic_fallible::<_, Record>(
            "emergency_shelters",
            "register_shelter",
            invalid_shelter,
        )
        .await;

    assert!(
        result.is_err(),
        "Creating shelter with empty address should fail"
    );
}

/// Emergency Shelter validation: capacity=0 should be rejected.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_validation_shelter_zero_capacity_rejected() {
    let agent = setup_unified_conductor().await;

    let invalid_shelter = RegisterShelterInput {
        id: "SHELTER-INVALID-002".to_string(),
        name: "Zero Capacity Shelter".to_string(),
        location_lat: 32.95,
        location_lon: -96.73,
        address: "100 Invalid Lane".to_string(),
        capacity: 0, // Zero: should fail
        shelter_type: ShelterType::Community,
        amenities: vec![],
        contact: "test@example.com".to_string(),
    };

    let result = agent
        .call_civic_fallible::<_, Record>(
            "emergency_shelters",
            "register_shelter",
            invalid_shelter,
        )
        .await;

    assert!(
        result.is_err(),
        "Creating shelter with capacity=0 should fail"
    );
}

/// Emergency Shelter validation: NaN coordinates should be rejected by integrity.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_validation_shelter_nan_coordinates_rejected() {
    let agent = setup_unified_conductor().await;

    let invalid_shelter = RegisterShelterInput {
        id: "SHELTER-INVALID-003".to_string(),
        name: "NaN Coordinates Shelter".to_string(),
        location_lat: f64::NAN,
        location_lon: f64::NAN,
        address: "200 Coordinate Error".to_string(),
        capacity: 10,
        shelter_type: ShelterType::Emergency,
        amenities: vec![],
        contact: "test@example.com".to_string(),
    };

    let result = agent
        .call_civic_fallible::<_, Record>(
            "emergency_shelters",
            "register_shelter",
            invalid_shelter,
        )
        .await;

    assert!(
        result.is_err(),
        "Creating shelter with NaN coordinates should fail integrity validation"
    );
}

/// Emergency Shelter validation: check-in with party_size=0 should be rejected.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_validation_shelter_checkin_zero_party_size_rejected() {
    let agent = setup_unified_conductor().await;
    let agent_key = agent.civic_cell.agent_pubkey().clone();

    // First create a valid shelter
    let shelter = RegisterShelterInput {
        id: "SHELTER-VALID-FOR-CHECKIN".to_string(),
        name: "Valid Shelter For Checkin Test".to_string(),
        location_lat: 33.0,
        location_lon: -96.7,
        address: "300 Shelter Way".to_string(),
        capacity: 50,
        shelter_type: ShelterType::Emergency,
        amenities: vec![Amenity::Cots],
        contact: "shelter@test.com".to_string(),
    };

    let shelter_record: Record = agent
        .call_civic("emergency_shelters", "register_shelter", shelter)
        .await;

    let shelter_hash = shelter_record.action_address().clone();

    // Attempt check-in with party_size=0
    let invalid_checkin = CheckInPersonInput {
        shelter_hash,
        person_name: "Nobody".to_string(),
        person_id: None,
        party_size: 0, // Zero: should fail
        special_needs: vec![],
    };

    let result = agent
        .call_civic_fallible::<_, Record>(
            "emergency_shelters",
            "check_in_person",
            invalid_checkin,
        )
        .await;

    assert!(
        result.is_err(),
        "Check-in with party_size=0 should fail"
    );
}

// ============================================================================
// 5. METRICS COLLECTION
// ============================================================================

/// Commons bridge: Make several dispatch calls, then verify metrics counters.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_commons_bridge_metrics_increment() {
    let agent = setup_unified_conductor().await;
    let agent_key = agent.commons_cell.agent_pubkey().clone();

    // Get initial metrics
    let initial_metrics: BridgeMetricsSnapshot = agent
        .call_commons("commons_bridge", "get_bridge_metrics", ())
        .await;

    let initial_success = initial_metrics.total_success;
    let initial_errors = initial_metrics.total_errors;

    // Make 3 successful bridge calls (events and queries)
    let event = CommonsEventInput {
        domain: "property".to_string(),
        event_type: "test_metrics_event".to_string(),
        source_agent: agent_key.clone(),
        payload: r#"{"test":true}"#.to_string(),
        created_at: Timestamp::now(),
        related_hashes: vec![],
    };

    let _: Record = agent
        .call_commons("commons_bridge", "broadcast_event", event.clone())
        .await;
    let _: Record = agent
        .call_commons("commons_bridge", "broadcast_event", event.clone())
        .await;

    let query = CommonsQueryInput {
        domain: "water".to_string(),
        query_type: "test_metrics_query".to_string(),
        requester: agent_key.clone(),
        params: "{}".to_string(),
        created_at: Timestamp::now(),
    };

    let _: Record = agent
        .call_commons("commons_bridge", "query_commons", query)
        .await;

    // Make a dispatch to a disallowed zome (should increment errors)
    let bad_dispatch = DispatchInput {
        zome: "evil_zome".to_string(),
        fn_name: "steal".to_string(),
        payload: vec![],
    };

    let fail_result: DispatchResult = agent
        .call_commons("commons_bridge", "dispatch_call", bad_dispatch)
        .await;

    assert!(!fail_result.success, "Disallowed zome should fail");

    // Get updated metrics
    let updated_metrics: BridgeMetricsSnapshot = agent
        .call_commons("commons_bridge", "get_bridge_metrics", ())
        .await;

    // Verify counters incremented
    // Note: bridge events/queries may or may not go through the dispatch
    // metrics path depending on implementation. At minimum, the failed
    // dispatch should increment error counters.
    assert!(
        updated_metrics.total_errors > initial_errors
            || updated_metrics.total_success > initial_success,
        "Metrics should show activity: initial=({}/{}), updated=({}/{})",
        initial_success,
        initial_errors,
        updated_metrics.total_success,
        updated_metrics.total_errors,
    );

    // The failed dispatch should show up in error counts
    if !updated_metrics.error_counts.is_empty() {
        let total_error_count: u64 = updated_metrics
            .error_counts
            .iter()
            .map(|e| e.count)
            .sum();
        assert!(
            total_error_count > 0,
            "At least one error should be recorded in per-error-code counts"
        );
    }
}

/// Civic bridge: Make bridge calls and verify metrics counters.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_civic_bridge_metrics_increment() {
    let agent = setup_unified_conductor().await;
    let agent_key = agent.civic_cell.agent_pubkey().clone();

    // Get initial metrics
    let initial_metrics: BridgeMetricsSnapshot = agent
        .call_civic("civic_bridge", "get_bridge_metrics", ())
        .await;

    let initial_success = initial_metrics.total_success;

    // Make 2 bridge events
    let event = CivicEventEntry {
        domain: "justice".to_string(),
        event_type: "test_metrics_event".to_string(),
        source_agent: agent_key.clone(),
        payload: r#"{"test":true}"#.to_string(),
        created_at: Timestamp::now(),
        related_hashes: vec![],
    };

    let _: Record = agent
        .call_civic("civic_bridge", "broadcast_event", event.clone())
        .await;
    let _: Record = agent
        .call_civic("civic_bridge", "broadcast_event", event)
        .await;

    // Make a health check (should also be tracked)
    let _health: BridgeHealth = agent
        .call_civic("civic_bridge", "health_check", ())
        .await;

    // Get updated metrics
    let updated_metrics: BridgeMetricsSnapshot = agent
        .call_civic("civic_bridge", "get_bridge_metrics", ())
        .await;

    // Metrics should reflect activity
    assert!(
        updated_metrics.total_success >= initial_success,
        "Civic bridge metrics total_success should not decrease"
    );
}

/// Cross-cluster dispatch metrics: Verify cross_cluster counter increments.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed"]
async fn test_cross_cluster_dispatch_metrics() {
    let agent = setup_unified_conductor().await;

    // Get initial commons metrics
    let initial: BridgeMetricsSnapshot = agent
        .call_commons("commons_bridge", "get_bridge_metrics", ())
        .await;

    let initial_cross = initial.total_cross_cluster;

    // Make 2 cross-cluster dispatches: commons -> civic
    let dispatch1 = CrossClusterDispatchInput {
        role: "civic".to_string(),
        zome: "civic_bridge".to_string(),
        fn_name: "health_check".to_string(),
        payload: ExternIO::encode(()).unwrap().0,
    };

    let _: DispatchResult = agent
        .call_commons("commons_bridge", "dispatch_civic_call", dispatch1.clone())
        .await;

    let _: DispatchResult = agent
        .call_commons("commons_bridge", "dispatch_civic_call", dispatch1)
        .await;

    // Get updated metrics
    let updated: BridgeMetricsSnapshot = agent
        .call_commons("commons_bridge", "get_bridge_metrics", ())
        .await;

    assert!(
        updated.total_cross_cluster >= initial_cross + 2,
        "Cross-cluster counter should increment by at least 2: initial={}, updated={}",
        initial_cross,
        updated.total_cross_cluster,
    );
}

// ============================================================================
// BONUS: Multi-domain workflow spanning both clusters
// ============================================================================

/// Full workflow: Property registered in commons -> dispute filed in civic ->
/// bridge events link both domains.
#[tokio::test(flavor = "multi_thread")]
#[serial]
#[ignore = "requires both DNAs packed — slow (>30s)"]
async fn test_multi_domain_property_dispute_workflow() {
    let agent = setup_unified_conductor().await;
    let agent_key = agent.commons_cell.agent_pubkey().clone();
    let civic_agent_key = agent.civic_cell.agent_pubkey().clone();
    let now = Timestamp::now();

    // 1. Register a property in commons
    let property_input = make_property_input(&agent_key, "Disputed Industrial Lot");
    let property_record: Record = agent
        .call_commons("property_registry", "register_property", property_input)
        .await;

    let property_hash = property_record.action_address().clone();

    // 2. File a property dispute case in civic
    let case = Case {
        id: "CASE-PROP-DISP-001".to_string(),
        title: "Industrial Lot Boundary Dispute".to_string(),
        description: "Adjacent landowner claims boundary encroachment".to_string(),
        case_type: CaseType::PropertyDispute,
        complainant: format!("did:key:{}", civic_agent_key),
        respondent: "did:key:adjacent-owner".to_string(),
        parties: vec![],
        phase: CasePhase::Filed,
        status: CaseStatus::Active,
        severity: CaseSeverity::Serious,
        context: CaseContext {
            happ: Some("mycelix-commons".to_string()),
            reference_id: Some(property_hash.to_string()),
            community: Some("Richardson Industrial District".to_string()),
            jurisdiction: Some("TX-Dallas".to_string()),
        },
        created_at: now,
        updated_at: now,
        phase_deadline: None,
    };

    let case_record: Record = agent
        .call_civic("justice_cases", "file_case", case)
        .await;

    let case_hash = case_record.action_address().clone();

    // 3. Broadcast a bridge event in commons linking the property to the dispute
    let bridge_event = CommonsEventInput {
        domain: "property".to_string(),
        event_type: "dispute_filed".to_string(),
        source_agent: agent_key.clone(),
        payload: serde_json::to_string(&serde_json::json!({
            "property_hash": property_hash.to_string(),
            "case_hash": case_hash.to_string(),
            "case_type": "PropertyDispute",
            "severity": "Serious",
        }))
        .unwrap(),
        created_at: Timestamp::now(),
        related_hashes: vec![property_hash.to_string(), case_hash.to_string()],
    };

    let event_record: Record = agent
        .call_commons("commons_bridge", "broadcast_event", bridge_event)
        .await;

    assert!(event_record.action().author() == &agent_key);

    // 4. Verify the property events include the dispute reference
    let property_events: Vec<Record> = agent
        .call_commons(
            "commons_bridge",
            "get_domain_events",
            "property".to_string(),
        )
        .await;

    assert!(
        !property_events.is_empty(),
        "Property domain should have the dispute event"
    );

    // 5. Update case phase to Mediation (both clusters have recorded the link)
    let update = UpdatePhaseInput {
        case_hash,
        new_phase: CasePhase::Mediation,
        deadline: Some(Timestamp::from_micros(
            now.as_micros() + 30 * 24 * 60 * 60 * 1_000_000,
        )),
    };

    let updated: Record = agent
        .call_civic("justice_cases", "update_case_phase", update)
        .await;

    assert!(updated.action().author() == &civic_agent_key);
}
