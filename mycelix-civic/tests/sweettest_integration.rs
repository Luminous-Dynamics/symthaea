//! # Mycelix Civic — Sweettest Integration Tests
//!
//! Tests the unified Civic cluster DNA: justice, emergency, media
//! domain zomes + civic-bridge.
//!
//! ## Running
//! ```bash
//! cd mycelix-civic
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_integration -- --ignored --test-threads=2
//! ```
//!
//! Note: `--test-threads=2` prevents conductor database timeouts from too many
//! concurrent Holochain conductors competing for SQLite locks.

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — avoid importing zome crates (duplicate WASM symbols)
// ============================================================================

// --- civic-bridge ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CivicQueryEntry {
    pub domain: String,
    pub query_type: String,
    pub requester: AgentPubKey,
    pub params: String,
    pub result: Option<String>,
    pub created_at: Timestamp,
    pub resolved_at: Option<Timestamp>,
    pub success: Option<bool>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CivicEventEntry {
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

// --- media-publication ---

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

// --- media-factcheck ---

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

// --- emergency-incidents ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ReportIncidentInput {
    pub incident_type: String,
    pub severity: String,
    pub description: String,
    pub location: String,
    pub reporter_did: String,
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
// Civic Bridge Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_civic_bridge_query() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    let query = CivicQueryEntry {
        domain: "justice".to_string(),
        query_type: "case_lookup".to_string(),
        requester: agent.clone(),
        params: r#"{"case_id":"case-001"}"#.to_string(),
        result: None,
        created_at: Timestamp::now(),
        resolved_at: None,
        success: None,
    };

    let record: Record = conductor
        .call(&alice.zome("civic_bridge"), "query_civic", query)
        .await;

    assert!(record.action().author() == alice.agent_pubkey());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_civic_bridge_broadcast_event() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    let event = CivicEventEntry {
        domain: "emergency".to_string(),
        event_type: "incident_reported".to_string(),
        source_agent: agent.clone(),
        payload: r#"{"severity":"high","type":"flood"}"#.to_string(),
        created_at: Timestamp::now(),
        related_hashes: vec![],
    };

    let record: Record = conductor
        .call(&alice.zome("civic_bridge"), "broadcast_event", event)
        .await;

    assert!(record.action().author() == alice.agent_pubkey());

    let events: Vec<Record> = conductor
        .call(
            &alice.zome("civic_bridge"),
            "get_domain_events",
            "emergency".to_string(),
        )
        .await;

    assert_eq!(events.len(), 1);
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_civic_bridge_health() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let health: BridgeHealth = conductor
        .call(&alice.zome("civic_bridge"), "health_check", ())
        .await;

    assert!(health.healthy);
    assert_eq!(health.domains.len(), 3);
    assert!(health.domains.contains(&"justice".to_string()));
    assert!(health.domains.contains(&"emergency".to_string()));
    assert!(health.domains.contains(&"media".to_string()));
}

// ============================================================================
// Media Domain Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_media_publish_and_factcheck() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();
    let author_did = format!("did:key:{}", agent);

    // 1. Publish an article
    let pub_input = PublishInput {
        title: "Test Article on Water Rights".to_string(),
        content_hash: "QmTest123".to_string(),
        content_type: ContentType::Investigation,
        author_did: author_did.clone(),
        co_authors: vec![],
        language: "en".to_string(),
        tags: vec!["water".to_string(), "rights".to_string()],
        license: License {
            license_type: LicenseType::CCBY,
            attribution_required: true,
            commercial_use: true,
            derivative_works: true,
        },
        encrypted: false,
    };

    let pub_record: Record = conductor
        .call(&alice.zome("media_publication"), "publish", pub_input)
        .await;

    // 2. Fact-check a claim in the article
    let fc_input = SubmitFactCheckInput {
        publication_id: pub_record.action_address().to_string(),
        claim_text: "Water rights are universally protected".to_string(),
        claim_location: "paragraph 3".to_string(),
        epistemic_position: EpistemicPosition {
            empirical: 0.4,
            normative: 0.6,
            mythic: 0.0,
        },
        verdict: FactCheckVerdict::PartiallyTrue,
        evidence: vec![EvidenceItem {
            source_type: SourceType::OfficialDocument,
            source_url: Some("https://example.com/water-law".to_string()),
            source_did: None,
            description: "UN Water Rights resolution".to_string(),
            supports_claim: true,
        }],
        checker_did: author_did.clone(),
    };

    let fc_record: Record = conductor
        .call(
            &alice.zome("media_factcheck"),
            "submit_fact_check",
            fc_input,
        )
        .await;

    assert!(fc_record.action().author() == alice.agent_pubkey());
}

// ============================================================================
// Cross-Domain Tests
// ============================================================================

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_media_references_emergency() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    // 1. Emergency broadcasts an incident event via bridge
    let incident_event = CivicEventEntry {
        domain: "emergency".to_string(),
        event_type: "disaster_declared".to_string(),
        source_agent: agent.clone(),
        payload: r#"{"type":"flood","region":"Gulf Coast","severity":"critical"}"#.to_string(),
        created_at: Timestamp::now(),
        related_hashes: vec![],
    };

    let incident_record: Record = conductor
        .call(
            &alice.zome("civic_bridge"),
            "broadcast_event",
            incident_event,
        )
        .await;

    // 2. Media publishes a report referencing the emergency
    let pub_input = PublishInput {
        title: "Gulf Coast Flood Emergency Report".to_string(),
        content_hash: "QmFloodReport".to_string(),
        content_type: ContentType::Report,
        author_did: format!("did:key:{}", agent),
        co_authors: vec![],
        language: "en".to_string(),
        tags: vec!["emergency".to_string(), "flood".to_string()],
        license: License {
            license_type: LicenseType::CC0,
            attribution_required: false,
            commercial_use: true,
            derivative_works: true,
        },
        encrypted: false,
    };

    let pub_record: Record = conductor
        .call(&alice.zome("media_publication"), "publish", pub_input)
        .await;

    // 3. Bridge event links them cross-domain
    let link_event = CivicEventEntry {
        domain: "media".to_string(),
        event_type: "report_linked_to_emergency".to_string(),
        source_agent: agent.clone(),
        payload: serde_json::to_string(&serde_json::json!({
            "publication_hash": pub_record.action_address().to_string(),
            "incident_hash": incident_record.action_address().to_string(),
        }))
        .unwrap(),
        created_at: Timestamp::now(),
        related_hashes: vec![
            incident_record.action_address().to_string(),
            pub_record.action_address().to_string(),
        ],
    };

    let link_record: Record = conductor
        .call(&alice.zome("civic_bridge"), "broadcast_event", link_event)
        .await;

    assert!(link_record.action().author() == alice.agent_pubkey());

    // 4. Verify both domains have events
    let emergency_events: Vec<Record> = conductor
        .call(
            &alice.zome("civic_bridge"),
            "get_domain_events",
            "emergency".to_string(),
        )
        .await;
    assert!(!emergency_events.is_empty());

    let media_events: Vec<Record> = conductor
        .call(
            &alice.zome("civic_bridge"),
            "get_domain_events",
            "media".to_string(),
        )
        .await;
    assert!(!media_events.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_domain_justice_queries_media() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let agent = alice.agent_pubkey().clone();

    // Justice system queries media for evidence related to a case
    let query = CivicQueryEntry {
        domain: "media".to_string(),
        query_type: "publication_search".to_string(),
        requester: agent.clone(),
        params: r#"{"tags":["evidence"],"content_type":"RawData"}"#.to_string(),
        result: None,
        created_at: Timestamp::now(),
        resolved_at: None,
        success: None,
    };

    let record: Record = conductor
        .call(&alice.zome("civic_bridge"), "query_civic", query)
        .await;

    let query_hash = record.action_address().clone();

    // Resolve with media search results
    let resolve = ResolveQueryInput {
        query_hash,
        result: r#"{"publications":[],"count":0}"#.to_string(),
        success: true,
    };

    let resolved: Record = conductor
        .call(&alice.zome("civic_bridge"), "resolve_query", resolve)
        .await;

    assert!(resolved.action().author() == alice.agent_pubkey());
}

// ============================================================================
// Justice Domain Tests
// ============================================================================

// --- Mirror types for justice-cases ---

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

// --- Mirror types for emergency-incidents ---

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

// --- Mirror types for emergency-shelters ---

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

// --- Mirror types for emergency-resources ---

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

// --- Mirror types for media-curation ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CreateCollectionInput {
    pub name: String,
    pub description: String,
    pub curator_did: String,
    pub visibility: Visibility,
    pub publication_ids: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Visibility {
    Public,
    Private,
    Unlisted,
}

// --- Mirror types for bridge audit trail ---

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AuditTrailInput {
    pub domain: Option<String>,
    pub limit: Option<u32>,
}

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

    // 1. File a case
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

    // 2. Escalate from Filed to Mediation
    let escalate = UpdatePhaseInput {
        case_hash: case_hash.clone(),
        new_phase: CasePhase::Mediation,
        deadline: None,
    };

    let updated: Record = conductor
        .call(&alice.zome("justice_cases"), "update_case_phase", escalate)
        .await;

    assert!(updated.action().author() == alice.agent_pubkey());

    // 3. Escalate to Arbitration
    let escalate2 = UpdatePhaseInput {
        case_hash: updated.action_address().clone(),
        new_phase: CasePhase::Arbitration,
        deadline: None,
    };

    let arbitration: Record = conductor
        .call(
            &alice.zome("justice_cases"),
            "update_case_phase",
            escalate2,
        )
        .await;

    assert!(arbitration.action().author() == alice.agent_pubkey());
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
        amenities: vec![
            Amenity::Food,
            Amenity::Medical,
            Amenity::Wifi,
        ],
        contact: "shelter-ops@example.com".to_string(),
    };

    let record: Record = conductor
        .call(
            &alice.zome("emergency_shelters"),
            "register_shelter",
            input,
        )
        .await;

    assert!(record.action().author() == alice.agent_pubkey());
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
}

// ============================================================================
// Cross-Domain: Emergency Multi-Zome Workflow
// ============================================================================

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

    // 1. Declare disaster
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

    // 2. Register shelter for evacuees
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

    // 3. Register emergency supplies
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

    // 4. Bridge event linking disaster to shelter and resources
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

    // 5. Verify emergency domain has coordination events
    let events: Vec<Record> = conductor
        .call(
            &alice.zome("civic_bridge"),
            "get_domain_events",
            "emergency".to_string(),
        )
        .await;
    assert!(!events.is_empty(), "Emergency domain should have coordination events");
}

// ============================================================================
// Cross-Domain: Justice + Media Integration
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

    // 1. File a case
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

    // 2. Publish the disputed article via media-publication
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

    // 3. Fact-check the disputed claims
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

    // 4. Bridge event linking justice case to media evidence
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
}

// ============================================================================
// Cross-Cluster Dispatch Scenarios — Civic -> Commons
// ============================================================================
//
// These scenarios exercise the `dispatch_commons_call` extern on the
// civic_bridge zome, which calls `dispatch_call_cross_cluster()` against
// the ALLOWED_COMMONS_ZOMES allowlist. Each test constructs a
// CrossClusterDispatchInput targeting a zome in the Commons DNA. In a
// real conductor with the unified hApp, the call would be routed via
// `CallTargetCell::OtherRole("commons")`.

// Mirror type for CrossClusterDispatchInput (from mycelix_bridge_common)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CrossClusterDispatchInput {
    pub role: String,
    pub zome: String,
    pub fn_name: String,
    pub payload: Vec<u8>,
}

// Mirror type for DispatchResult (from mycelix_bridge_common)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DispatchResult {
    pub success: bool,
    pub response: Option<Vec<u8>>,
    pub error: Option<String>,
}

/// Cross-cluster: Emergency -> Food (Civic -> Commons)
///
/// Scenario: During a disaster response, the emergency_incidents zome
/// needs to query food stock levels from the commons food_distribution
/// zome to plan rationing and supply logistics. The dispatch crosses
/// from the Civic DNA to the Commons DNA via OtherRole("commons").
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

    // 1. Declare a disaster first (the triggering incident)
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

    // 2. Cross-cluster dispatch: emergency -> commons food_distribution
    //    to query available food stocks during the disaster
    let food_query_payload = serde_json::to_vec(&serde_json::json!({
        "disaster_id": disaster_record.action_address().to_string(),
        "area_lat": 29.7604,
        "area_lon": -95.3698,
        "radius_km": 80.0,
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
        .call(&alice.zome("civic_bridge"), "dispatch_commons_call", dispatch)
        .await;

    // In a real unified hApp with both DNAs, this would reach the commons
    // food_distribution zome. Here, the dispatch validates the zome name
    // against ALLOWED_COMMONS_ZOMES. Without the commons DNA installed,
    // we expect a network/routing error rather than an allowlist rejection.
    assert!(
        result.success || result.error.is_some(),
        "Cross-cluster dispatch should either succeed or return a routing error"
    );
    // Verify it was NOT rejected by the allowlist (food_distribution is allowed)
    if let Some(ref err) = result.error {
        assert!(
            !err.contains("not in the allowed cross-cluster dispatch list"),
            "food_distribution should be in the allowed commons zomes"
        );
    }

    // 3. Bridge event recording the cross-cluster food stock query
    let query_event = CivicEventEntry {
        domain: "emergency".to_string(),
        event_type: "cross_cluster_food_stock_queried".to_string(),
        source_agent: agent.clone(),
        payload: serde_json::to_string(&serde_json::json!({
            "disaster_hash": disaster_record.action_address().to_string(),
            "target_cluster": "commons",
            "target_zome": "food_distribution",
            "query": "available food stocks for disaster response",
        }))
        .unwrap(),
        created_at: Timestamp::now(),
        related_hashes: vec![disaster_record.action_address().to_string()],
    };

    let event_record: Record = conductor
        .call(&alice.zome("civic_bridge"), "broadcast_event", query_event)
        .await;

    assert!(event_record.action().author() == alice.agent_pubkey());
}

/// Cross-cluster: Justice -> Transport (Civic -> Commons)
///
/// Scenario: A justice case involves suspected carbon credit fraud. The
/// justice_cases zome dispatches to the commons transport_impact zome to
/// query the suspect's carbon credit history for forensic analysis.
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

    // 1. File a justice case for suspected carbon credit fraud
    let case = CaseInput {
        id: "case-carbon-fraud".to_string(),
        title: "Suspected carbon credit fraud".to_string(),
        description: "Agent reported inflated trip distances to earn excess carbon credits".to_string(),
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

    let case_record: Record = conductor
        .call(&alice.zome("justice_cases"), "file_case", case)
        .await;

    // 2. Cross-cluster dispatch: justice -> commons transport_impact
    //    to query carbon credit history for the suspect
    let credit_query_payload = serde_json::to_vec(&serde_json::json!({
        "agent_did": "did:key:suspect-agent-did",
        "from_timestamp": 1700000000,
        "to_timestamp": 1710000000,
    }))
    .unwrap();

    let dispatch = CrossClusterDispatchInput {
        role: "commons".to_string(),
        zome: "transport_impact".to_string(),
        fn_name: "get_agent_carbon_credits".to_string(),
        payload: credit_query_payload,
    };

    let result: DispatchResult = conductor
        .call(&alice.zome("civic_bridge"), "dispatch_commons_call", dispatch)
        .await;

    assert!(
        result.success || result.error.is_some(),
        "Cross-cluster dispatch should either succeed or return a routing error"
    );
    // Verify it was NOT rejected by the allowlist
    if let Some(ref err) = result.error {
        assert!(
            !err.contains("not in the allowed cross-cluster dispatch list"),
            "transport_impact should be in the allowed commons zomes"
        );
    }

    // 3. Bridge event linking the case to the cross-cluster evidence query
    let evidence_event = CivicEventEntry {
        domain: "justice".to_string(),
        event_type: "cross_cluster_evidence_queried".to_string(),
        source_agent: agent.clone(),
        payload: serde_json::to_string(&serde_json::json!({
            "case_hash": case_record.action_address().to_string(),
            "target_cluster": "commons",
            "target_zome": "transport_impact",
            "evidence_type": "carbon_credit_history",
            "suspect_did": "did:key:suspect-agent-did",
        }))
        .unwrap(),
        created_at: Timestamp::now(),
        related_hashes: vec![case_record.action_address().to_string()],
    };

    let event_record: Record = conductor
        .call(&alice.zome("civic_bridge"), "broadcast_event", evidence_event)
        .await;

    assert!(event_record.action().author() == alice.agent_pubkey());

    // 4. Verify justice events include the cross-cluster query
    let justice_events: Vec<Record> = conductor
        .call(
            &alice.zome("civic_bridge"),
            "get_domain_events",
            "justice".to_string(),
        )
        .await;

    assert!(
        !justice_events.is_empty(),
        "Justice domain should have the cross-cluster evidence query event"
    );
}

/// Cross-cluster: Emergency -> Housing (Civic -> Commons)
///
/// Scenario: During a disaster, the emergency_coordination zome needs to
/// check available housing capacity in the commons cluster to supplement
/// dedicated shelters. This dispatches to commons housing_units.
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

    // 1. Declare a large-scale disaster exceeding shelter capacity
    let disaster = DeclareDisasterInput {
        id: "disaster-shelter-overflow".to_string(),
        disaster_type: DisasterType::Tornado,
        title: "North Texas Tornado Complex".to_string(),
        description: "Multiple tornadoes requiring overflow housing capacity assessment".to_string(),
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

    // 2. Register a shelter (at capacity)
    let shelter = RegisterShelterInput {
        id: "shelter-full-001".to_string(),
        name: "Richardson Community Center".to_string(),
        location_lat: 32.9483,
        location_lon: -96.7299,
        address: "100 Civic Center Dr, Richardson, TX".to_string(),
        capacity: 300,
        shelter_type: ShelterType::Emergency,
        amenities: vec![Amenity::Food, Amenity::Medical, Amenity::Power, Amenity::Cots],
        contact: "emergency-ops@richardson.gov".to_string(),
    };

    let _shelter_record: Record = conductor
        .call(
            &alice.zome("emergency_shelters"),
            "register_shelter",
            shelter,
        )
        .await;

    // 3. Cross-cluster dispatch: emergency -> commons housing_units
    //    to check available overflow housing capacity
    let housing_query_payload = serde_json::to_vec(&serde_json::json!({
        "area": "Richardson, TX",
        "disaster_id": disaster_record.action_address().to_string(),
        "min_capacity": 10,
        "unit_status": "available",
    }))
    .unwrap();

    let dispatch = CrossClusterDispatchInput {
        role: "commons".to_string(),
        zome: "housing_units".to_string(),
        fn_name: "get_available_units_for_area".to_string(),
        payload: housing_query_payload,
    };

    let result: DispatchResult = conductor
        .call(&alice.zome("civic_bridge"), "dispatch_commons_call", dispatch)
        .await;

    assert!(
        result.success || result.error.is_some(),
        "Cross-cluster dispatch should either succeed or return a routing error"
    );
    // Verify it was NOT rejected by the allowlist
    if let Some(ref err) = result.error {
        assert!(
            !err.contains("not in the allowed cross-cluster dispatch list"),
            "housing_units should be in the allowed commons zomes"
        );
    }

    // 4. Bridge event recording the cross-cluster shelter capacity check
    let capacity_event = CivicEventEntry {
        domain: "emergency".to_string(),
        event_type: "cross_cluster_housing_capacity_checked".to_string(),
        source_agent: agent.clone(),
        payload: serde_json::to_string(&serde_json::json!({
            "disaster_hash": disaster_record.action_address().to_string(),
            "target_cluster": "commons",
            "target_zome": "housing_units",
            "reason": "shelter capacity exceeded, checking overflow housing",
            "area": "Richardson, TX",
        }))
        .unwrap(),
        created_at: Timestamp::now(),
        related_hashes: vec![disaster_record.action_address().to_string()],
    };

    let event_record: Record = conductor
        .call(&alice.zome("civic_bridge"), "broadcast_event", capacity_event)
        .await;

    assert!(event_record.action().author() == alice.agent_pubkey());

    // 5. Verify emergency events include the housing capacity check
    let emergency_events: Vec<Record> = conductor
        .call(
            &alice.zome("civic_bridge"),
            "get_domain_events",
            "emergency".to_string(),
        )
        .await;

    assert!(
        emergency_events.len() >= 1,
        "Emergency domain should have the cross-cluster housing capacity event"
    );
}

// ============================================================================
// Bridge Audit Trail Test
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

    // 1. Create multiple events across domains
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

    // 2. Query all events (no domain filter)
    let all_events: Vec<Record> = conductor
        .call(
            &alice.zome("civic_bridge"),
            "get_my_events",
            (),
        )
        .await;

    assert!(all_events.len() >= 4, "Should have at least 4 events total");

    // 3. Query justice-specific events
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

    // 4. Query emergency-specific events
    let emergency_events: Vec<Record> = conductor
        .call(
            &alice.zome("civic_bridge"),
            "get_domain_events",
            "emergency".to_string(),
        )
        .await;

    assert!(
        !emergency_events.is_empty(),
        "Should have emergency events"
    );
}
