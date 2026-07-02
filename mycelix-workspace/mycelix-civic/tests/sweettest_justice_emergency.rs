// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Civic — Justice & Emergency Sweettest
//!
//! Batch 2 of civic sweettest integration tests: justice domain (file case,
//! phase escalation, case with media evidence), emergency domain (declare
//! disaster, register shelter/resources, full workflow), cross-cluster
//! dispatch (emergency->food, justice->transport, emergency->housing),
//! and bridge audit trail.
//!
//! Split from sweettest_integration.rs to reduce per-process conductor
//! memory pressure (~1-2 GB per conductor). Each [[test]] binary runs
//! as a separate OS process, so memory is fully reclaimed between batches.
//!
//! ## Running
//! ```bash
//! cd mycelix-civic/tests
//! cargo test --release --test sweettest_justice_emergency -- --ignored --test-threads=1
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — civic-bridge
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CivicEventEntry {
    pub domain: String,
    pub event_type: String,
    pub source_agent: AgentPubKey,
    pub payload: String,
    pub created_at: Timestamp,
    pub related_hashes: Vec<String>,
}

// ============================================================================
// Mirror types — media (needed for justice+media test)
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PublishInput {
    pub title: String,
    pub content_hash: String,
    pub content_type: ContentType,
    pub author_did: String,
    pub co_authors: Vec<String>,
    pub language: String,
    pub tags: Vec<String>,
    pub license: License,
    pub encrypted: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ContentType {
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
pub struct License {
    pub license_type: LicenseType,
    pub attribution_required: bool,
    pub commercial_use: bool,
    pub derivative_works: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum LicenseType {
    CC0,
    CCBY,
    CCBYSA,
    CCBYNC,
    CCBYNCSA,
    AllRightsReserved,
    Custom(String),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SubmitFactCheckInput {
    pub publication_id: String,
    pub claim_text: String,
    pub claim_location: String,
    pub epistemic_position: EpistemicPosition,
    pub verdict: FactCheckVerdict,
    pub evidence: Vec<EvidenceItem>,
    pub checker_did: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EpistemicPosition {
    pub empirical: f64,
    pub normative: f64,
    pub mythic: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum FactCheckVerdict {
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
pub struct EvidenceItem {
    pub source_type: SourceType,
    pub source_url: Option<String>,
    pub source_did: Option<String>,
    pub description: String,
    pub supports_claim: bool,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum SourceType {
    PrimarySource,
    SecondarySource,
    ExpertOpinion,
    OfficialDocument,
    ScientificStudy,
    EyewitnessAccount,
    DataAnalysis,
    Other(String),
}

// ============================================================================
// Mirror types — justice
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CaseInput {
    pub id: String,
    pub title: String,
    pub description: String,
    pub case_type: CaseType,
    pub complainant: String,
    pub respondent: String,
    pub parties: Vec<CaseParty>,
    pub phase: CasePhase,
    pub status: CaseStatus,
    pub severity: CaseSeverity,
    pub context: CaseContext,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
    pub phase_deadline: Option<Timestamp>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum CaseType {
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
pub struct CaseParty {
    pub did: String,
    pub role: PartyRole,
    pub joined_at: Timestamp,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum PartyRole {
    Complainant,
    Respondent,
    Witness,
    Expert,
    Intervenor,
    Affected,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum CasePhase {
    Filed,
    Negotiation,
    Mediation,
    Arbitration,
    Appeal,
    Enforcement,
    Closed,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum CaseStatus {
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
pub enum CaseSeverity {
    Minor,
    Moderate,
    Serious,
    Critical,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CaseContext {
    pub happ: Option<String>,
    pub reference_id: Option<String>,
    pub community: Option<String>,
    pub jurisdiction: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UpdatePhaseInput {
    pub case_hash: ActionHash,
    pub new_phase: CasePhase,
    pub deadline: Option<Timestamp>,
}

// ============================================================================
// Mirror types — emergency
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DeclareDisasterInput {
    pub id: String,
    pub disaster_type: DisasterType,
    pub title: String,
    pub description: String,
    pub severity: SeverityLevel,
    pub affected_area: AffectedArea,
    pub estimated_affected: u32,
    pub coordination_lead: Option<AgentPubKey>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum DisasterType {
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
pub enum SeverityLevel {
    Level1,
    Level2,
    Level3,
    Level4,
    Level5,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AffectedArea {
    pub center_lat: f64,
    pub center_lon: f64,
    pub radius_km: f32,
    pub boundary: Option<Vec<(f64, f64)>>,
    pub zones: Vec<OperationalZone>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct OperationalZone {
    pub id: String,
    pub name: String,
    pub boundary: Vec<(f64, f64)>,
    pub priority: ZonePriority,
    pub status: ZoneStatus,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ZonePriority {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ZoneStatus {
    Unassessed,
    Active,
    Cleared,
    Hazardous,
    Evacuated,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegisterShelterInput {
    pub id: String,
    pub name: String,
    pub location_lat: f64,
    pub location_lon: f64,
    pub address: String,
    pub capacity: u32,
    pub shelter_type: ShelterType,
    pub amenities: Vec<Amenity>,
    pub contact: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ShelterType {
    Emergency,
    Community,
    Medical,
    PetFriendly,
    Accessible,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Amenity {
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
pub struct RegisterResourceInput {
    pub id: String,
    pub resource_type: EmergencyResourceType,
    pub name: String,
    pub quantity: u32,
    pub unit: String,
    pub location: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum EmergencyResourceType {
    Medical,
    Personnel,
    Equipment,
    Shelter,
    Transport,
    Communication,
    Food,
    Water,
    Power,
    Fuel,
}

// ============================================================================
// Mirror types — cross-cluster dispatch
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CrossClusterDispatchInput {
    pub role: String,
    pub zome: String,
    pub fn_name: String,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DispatchResult {
    pub success: bool,
    pub response: Option<Vec<u8>>,
    pub error: Option<String>,
}

// ============================================================================
// DNA setup helper
// ============================================================================

fn civic_dna_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // tests/ -> mycelix-civic/
    path.push("dna");
    path.push("mycelix_civic.dna");
    path
}

// ============================================================================
// Justice Domain Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_justice_file_case() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();
    let now = Timestamp::now();

    let case = CaseInput {
        id: "case-001".to_string(),
        title: "Property boundary dispute".to_string(),
        description: "Disagreement over boundary markers between lots".to_string(),
        case_type: CaseType::PropertyDispute,
        complainant: format!("did:key:{}", agent),
        respondent: "did:key:other-party".to_string(),
        parties: vec![],
        phase: CasePhase::Filed,
        status: CaseStatus::Active,
        severity: CaseSeverity::Moderate,
        context: CaseContext {
            happ: Some("mycelix-commons".to_string()),
            reference_id: Some("property-123".to_string()),
            community: Some("Oakwood".to_string()),
            jurisdiction: None,
        },
        created_at: now,
        updated_at: now,
        phase_deadline: None,
    };

    let record: Record = conductor
        .call(&alice.zome("justice_cases"), "file_case", case)
        .await;

    assert!(record.action().author() == alice.agent_pubkey());

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_justice_case_phase_escalation() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();
    let now = Timestamp::now();

    let case = CaseInput {
        id: "case-002".to_string(),
        title: "Contract breach".to_string(),
        description: "Service not delivered as agreed".to_string(),
        case_type: CaseType::ContractDispute,
        complainant: format!("did:key:{}", agent),
        respondent: "did:key:contractor-did".to_string(),
        parties: vec![],
        phase: CasePhase::Filed,
        status: CaseStatus::Active,
        severity: CaseSeverity::Serious,
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

    let filed_record: Record = conductor
        .call(&alice.zome("justice_cases"), "file_case", case)
        .await;

    let case_hash = filed_record.action_address().clone();

    let escalate = UpdatePhaseInput {
        case_hash: case_hash.clone(),
        new_phase: CasePhase::Mediation,
        deadline: None,
    };

    let updated: Record = conductor
        .call(&alice.zome("justice_cases"), "update_case_phase", escalate)
        .await;

    assert!(updated.action().author() == alice.agent_pubkey());

    let escalate2 = UpdatePhaseInput {
        case_hash: updated.action_address().clone(),
        new_phase: CasePhase::Arbitration,
        deadline: None,
    };

    let arbitration: Record = conductor
        .call(&alice.zome("justice_cases"), "update_case_phase", escalate2)
        .await;

    assert!(arbitration.action().author() == alice.agent_pubkey());

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

// ============================================================================
// Emergency Domain Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_emergency_declare_disaster() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = DeclareDisasterInput {
        id: "disaster-001".to_string(),
        disaster_type: DisasterType::Flood,
        title: "Gulf Coast Flood".to_string(),
        description: "Major flooding in coastal areas".to_string(),
        severity: SeverityLevel::Level4,
        affected_area: AffectedArea {
            center_lat: 29.7604,
            center_lon: -95.3698,
            radius_km: 50.0,
            boundary: None,
            zones: vec![OperationalZone {
                id: "zone-a".to_string(),
                name: "Downtown".to_string(),
                boundary: vec![(29.76, -95.37), (29.77, -95.37), (29.77, -95.36)],
                priority: ZonePriority::Critical,
                status: ZoneStatus::Active,
            }],
        },
        estimated_affected: 50000,
        coordination_lead: None,
    };

    let record: Record = conductor
        .call(
            &alice.zome("emergency_incidents"),
            "declare_disaster",
            input,
        )
        .await;

    assert!(record.action().author() == alice.agent_pubkey());

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_emergency_register_shelter() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = RegisterShelterInput {
        id: "shelter-001".to_string(),
        name: "Community Center Shelter".to_string(),
        location_lat: 29.7604,
        location_lon: -95.3698,
        address: "123 Main St, Houston, TX".to_string(),
        capacity: 200,
        shelter_type: ShelterType::Community,
        amenities: vec![Amenity::Food, Amenity::Medical, Amenity::Wifi],
        contact: "shelter-ops@example.com".to_string(),
    };

    let record: Record = conductor
        .call(&alice.zome("emergency_shelters"), "register_shelter", input)
        .await;

    assert!(record.action().author() == alice.agent_pubkey());

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_emergency_register_resource() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = RegisterResourceInput {
        id: "resource-001".to_string(),
        resource_type: EmergencyResourceType::Water,
        name: "Bottled Water Supply".to_string(),
        quantity: 5000,
        unit: "gallons".to_string(),
        location: "Warehouse District, Houston".to_string(),
    };

    let record: Record = conductor
        .call(
            &alice.zome("emergency_resources"),
            "register_resource",
            input,
        )
        .await;

    assert!(record.action().author() == alice.agent_pubkey());

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_emergency_full_workflow() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    let disaster = DeclareDisasterInput {
        id: "disaster-wf-001".to_string(),
        disaster_type: DisasterType::Earthquake,
        title: "Coastal earthquake".to_string(),
        description: "6.2 magnitude earthquake affecting coastal region".to_string(),
        severity: SeverityLevel::Level3,
        affected_area: AffectedArea {
            center_lat: 37.7749,
            center_lon: -122.4194,
            radius_km: 30.0,
            boundary: None,
            zones: vec![],
        },
        estimated_affected: 10000,
        coordination_lead: Some(agent.clone()),
    };

    let disaster_record: Record = conductor
        .call(
            &alice.zome("emergency_incidents"),
            "declare_disaster",
            disaster,
        )
        .await;

    let shelter = RegisterShelterInput {
        id: "shelter-wf-001".to_string(),
        name: "Bay Area Evacuation Center".to_string(),
        location_lat: 37.78,
        location_lon: -122.42,
        address: "456 Shelter Way, San Francisco".to_string(),
        capacity: 500,
        shelter_type: ShelterType::Emergency,
        amenities: vec![Amenity::Food, Amenity::Medical, Amenity::Power],
        contact: "ops@bayarea-emergency.org".to_string(),
    };

    let shelter_record: Record = conductor
        .call(
            &alice.zome("emergency_shelters"),
            "register_shelter",
            shelter,
        )
        .await;

    let supplies = RegisterResourceInput {
        id: "resource-wf-001".to_string(),
        resource_type: EmergencyResourceType::Medical,
        name: "First aid kits".to_string(),
        quantity: 200,
        unit: "kits".to_string(),
        location: "San Francisco warehouse".to_string(),
    };

    let resource_record: Record = conductor
        .call(
            &alice.zome("emergency_resources"),
            "register_resource",
            supplies,
        )
        .await;

    let coordination_event = CivicEventEntry {
        domain: "emergency".to_string(),
        event_type: "disaster_response_coordinated".to_string(),
        source_agent: agent.clone(),
        payload: serde_json::to_string(&serde_json::json!({
            "disaster_hash": disaster_record.action_address().to_string(),
            "shelter_hash": shelter_record.action_address().to_string(),
            "resource_hash": resource_record.action_address().to_string(),
            "action": "resources_deployed_to_shelter",
        }))
        .unwrap(),
        created_at: Timestamp::now(),
        related_hashes: vec![
            disaster_record.action_address().to_string(),
            shelter_record.action_address().to_string(),
            resource_record.action_address().to_string(),
        ],
    };

    let event_record: Record = conductor
        .call(
            &alice.zome("civic_bridge"),
            "broadcast_event",
            coordination_event,
        )
        .await;

    assert!(event_record.action().author() == alice.agent_pubkey());

    let events: Vec<Record> = conductor
        .call(
            &alice.zome("civic_bridge"),
            "get_domain_events",
            "emergency".to_string(),
        )
        .await;
    assert!(
        !events.is_empty(),
        "Emergency domain should have coordination events"
    );

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

// ============================================================================
// Justice + Media Integration
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_justice_case_with_media_evidence() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();
    let now = Timestamp::now();

    let case = CaseInput {
        id: "case-media-001".to_string(),
        title: "Defamation in published article".to_string(),
        description: "A published article contains false claims".to_string(),
        case_type: CaseType::ConductViolation,
        complainant: format!("did:key:{}", agent),
        respondent: "did:key:publisher-did".to_string(),
        parties: vec![],
        phase: CasePhase::Filed,
        status: CaseStatus::Active,
        severity: CaseSeverity::Serious,
        context: CaseContext {
            happ: Some("mycelix-civic".to_string()),
            reference_id: None,
            community: None,
            jurisdiction: None,
        },
        created_at: now,
        updated_at: now,
        phase_deadline: None,
    };

    let case_record: Record = conductor
        .call(&alice.zome("justice_cases"), "file_case", case)
        .await;

    let article = PublishInput {
        title: "Disputed Article About Governance".to_string(),
        content_hash: "QmDisputedContent".to_string(),
        content_type: ContentType::Article,
        author_did: "did:key:publisher-did".to_string(),
        co_authors: vec![],
        language: "en".to_string(),
        tags: vec!["governance".to_string(), "dispute".to_string()],
        license: License {
            license_type: LicenseType::AllRightsReserved,
            attribution_required: false,
            commercial_use: false,
            derivative_works: false,
        },
        encrypted: false,
    };

    let pub_record: Record = conductor
        .call(&alice.zome("media_publication"), "publish", article)
        .await;

    let fc = SubmitFactCheckInput {
        publication_id: pub_record.action_address().to_string(),
        claim_text: "The governance decision was unanimous".to_string(),
        claim_location: "paragraph 2".to_string(),
        epistemic_position: EpistemicPosition {
            empirical: 0.9,
            normative: 0.1,
            mythic: 0.0,
        },
        verdict: FactCheckVerdict::False,
        evidence: vec![EvidenceItem {
            source_type: SourceType::OfficialDocument,
            source_url: Some("https://governance.example.com/minutes".to_string()),
            source_did: None,
            description: "Official meeting minutes show 7-3 vote".to_string(),
            supports_claim: false,
        }],
        checker_did: format!("did:key:{}", agent),
    };

    let fc_record: Record = conductor
        .call(&alice.zome("media_factcheck"), "submit_fact_check", fc)
        .await;

    let link_event = CivicEventEntry {
        domain: "justice".to_string(),
        event_type: "evidence_from_media".to_string(),
        source_agent: agent.clone(),
        payload: serde_json::to_string(&serde_json::json!({
            "case_hash": case_record.action_address().to_string(),
            "publication_hash": pub_record.action_address().to_string(),
            "factcheck_hash": fc_record.action_address().to_string(),
            "verdict": "False",
        }))
        .unwrap(),
        created_at: Timestamp::now(),
        related_hashes: vec![
            case_record.action_address().to_string(),
            fc_record.action_address().to_string(),
        ],
    };

    let event: Record = conductor
        .call(&alice.zome("civic_bridge"), "broadcast_event", link_event)
        .await;

    assert!(event.action().author() == alice.agent_pubkey());

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

// ============================================================================
// Cross-Cluster Dispatch
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_cluster_emergency_queries_food_stock() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    let disaster = DeclareDisasterInput {
        id: "disaster-food-query".to_string(),
        disaster_type: DisasterType::Hurricane,
        title: "Hurricane Maria Response".to_string(),
        description: "Category 4 hurricane requiring food supply assessment".to_string(),
        severity: SeverityLevel::Level4,
        affected_area: AffectedArea {
            center_lat: 29.7604,
            center_lon: -95.3698,
            radius_km: 80.0,
            boundary: None,
            zones: vec![],
        },
        estimated_affected: 75000,
        coordination_lead: Some(agent.clone()),
    };

    let disaster_record: Record = conductor
        .call(
            &alice.zome("emergency_incidents"),
            "declare_disaster",
            disaster,
        )
        .await;

    let food_query_payload = serde_json::to_vec(&serde_json::json!({
        "disaster_id": disaster_record.action_address().to_string(),
        "area_lat": 29.7604, "area_lon": -95.3698, "radius_km": 80.0,
        "food_types": ["non_perishable", "water", "ready_to_eat"],
    }))
    .unwrap();

    let dispatch = CrossClusterDispatchInput {
        role: "commons".to_string(),
        zome: "food_distribution".to_string(),
        fn_name: "query_available_stock".to_string(),
        payload: food_query_payload,
    };

    let result: DispatchResult = conductor
        .call(
            &alice.zome("civic_bridge"),
            "dispatch_commons_call",
            dispatch,
        )
        .await;

    assert!(result.success || result.error.is_some());
    if let Some(ref err) = result.error {
        assert!(!err.contains("not in the allowed cross-cluster dispatch list"));
    }

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_cluster_justice_queries_transport_carbon_credits() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();
    let now = Timestamp::now();

    let case = CaseInput {
        id: "case-carbon-fraud".to_string(),
        title: "Suspected carbon credit fraud".to_string(),
        description: "Agent reported inflated trip distances to earn excess carbon credits"
            .to_string(),
        case_type: CaseType::FinancialDispute,
        complainant: format!("did:key:{}", agent),
        respondent: "did:key:suspect-agent-did".to_string(),
        parties: vec![],
        phase: CasePhase::Filed,
        status: CaseStatus::Active,
        severity: CaseSeverity::Serious,
        context: CaseContext {
            happ: Some("mycelix-commons".to_string()),
            reference_id: Some("transport-credits-audit".to_string()),
            community: Some("Richardson".to_string()),
            jurisdiction: None,
        },
        created_at: now,
        updated_at: now,
        phase_deadline: None,
    };

    let _case_record: Record = conductor
        .call(&alice.zome("justice_cases"), "file_case", case)
        .await;

    let credit_query_payload = serde_json::to_vec(&serde_json::json!({
        "agent_did": "did:key:suspect-agent-did",
        "from_timestamp": 1700000000, "to_timestamp": 1710000000,
    }))
    .unwrap();

    let dispatch = CrossClusterDispatchInput {
        role: "commons".to_string(),
        zome: "transport_impact".to_string(),
        fn_name: "get_agent_carbon_credits".to_string(),
        payload: credit_query_payload,
    };

    let result: DispatchResult = conductor
        .call(
            &alice.zome("civic_bridge"),
            "dispatch_commons_call",
            dispatch,
        )
        .await;

    assert!(result.success || result.error.is_some());
    if let Some(ref err) = result.error {
        assert!(!err.contains("not in the allowed cross-cluster dispatch list"));
    }

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_cluster_emergency_checks_housing_capacity() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    let disaster = DeclareDisasterInput {
        id: "disaster-shelter-overflow".to_string(),
        disaster_type: DisasterType::Tornado,
        title: "North Texas Tornado Complex".to_string(),
        description: "Multiple tornadoes requiring overflow housing capacity assessment"
            .to_string(),
        severity: SeverityLevel::Level5,
        affected_area: AffectedArea {
            center_lat: 32.9483,
            center_lon: -96.7299,
            radius_km: 40.0,
            boundary: None,
            zones: vec![OperationalZone {
                id: "zone-overflow".to_string(),
                name: "Richardson Central".to_string(),
                boundary: vec![(32.94, -96.73), (32.96, -96.73), (32.96, -96.71)],
                priority: ZonePriority::Critical,
                status: ZoneStatus::Active,
            }],
        },
        estimated_affected: 25000,
        coordination_lead: Some(agent.clone()),
    };

    let disaster_record: Record = conductor
        .call(
            &alice.zome("emergency_incidents"),
            "declare_disaster",
            disaster,
        )
        .await;

    let _shelter_record: Record = conductor
        .call(
            &alice.zome("emergency_shelters"),
            "register_shelter",
            RegisterShelterInput {
                id: "shelter-full-001".to_string(),
                name: "Richardson Community Center".to_string(),
                location_lat: 32.9483,
                location_lon: -96.7299,
                address: "100 Civic Center Dr, Richardson, TX".to_string(),
                capacity: 300,
                shelter_type: ShelterType::Emergency,
                amenities: vec![
                    Amenity::Food,
                    Amenity::Medical,
                    Amenity::Power,
                    Amenity::Cots,
                ],
                contact: "emergency-ops@richardson.gov".to_string(),
            },
        )
        .await;

    let housing_query_payload = serde_json::to_vec(&serde_json::json!({
        "area": "Richardson, TX",
        "disaster_id": disaster_record.action_address().to_string(),
        "min_capacity": 10, "unit_status": "available",
    }))
    .unwrap();

    let dispatch = CrossClusterDispatchInput {
        role: "commons".to_string(),
        zome: "housing_units".to_string(),
        fn_name: "get_available_units_for_area".to_string(),
        payload: housing_query_payload,
    };

    let result: DispatchResult = conductor
        .call(
            &alice.zome("civic_bridge"),
            "dispatch_commons_call",
            dispatch,
        )
        .await;

    assert!(result.success || result.error.is_some());
    if let Some(ref err) = result.error {
        assert!(!err.contains("not in the allowed cross-cluster dispatch list"));
    }

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}

// ============================================================================
// Bridge Audit Trail
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_audit_trail() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    for (domain, event_type) in [
        ("justice", "case_filed"),
        ("emergency", "disaster_declared"),
        ("media", "article_published"),
        ("justice", "case_escalated"),
    ] {
        let event = CivicEventEntry {
            domain: domain.to_string(),
            event_type: event_type.to_string(),
            source_agent: agent.clone(),
            payload: "{}".to_string(),
            created_at: Timestamp::now(),
            related_hashes: vec![],
        };

        let _: Record = conductor
            .call(&alice.zome("civic_bridge"), "broadcast_event", event)
            .await;
    }

    let all_events: Vec<Record> = conductor
        .call(&alice.zome("civic_bridge"), "get_my_events", ())
        .await;

    assert!(all_events.len() >= 4, "Should have at least 4 events total");

    let justice_events: Vec<Record> = conductor
        .call(
            &alice.zome("civic_bridge"),
            "get_domain_events",
            "justice".to_string(),
        )
        .await;

    assert!(
        justice_events.len() >= 2,
        "Should have at least 2 justice events"
    );

    let emergency_events: Vec<Record> = conductor
        .call(
            &alice.zome("civic_bridge"),
            "get_domain_events",
            "emergency".to_string(),
        )
        .await;

    assert!(!emergency_events.is_empty(), "Should have emergency events");

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}
