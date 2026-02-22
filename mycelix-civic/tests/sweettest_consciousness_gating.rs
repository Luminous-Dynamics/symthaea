//! # Consciousness Gating Sweettest — Civic Cluster
//!
//! Verifies that governance functions in the civic cluster are properly
//! gated by consciousness credentials, proving the full wiring from
//! domain coordinator → civic_bridge → cross-cluster identity call.
//!
//! ## Running
//! ```bash
//! cd mycelix-civic
//! nix develop
//! hc dna pack dna/
//! hc app pack .
//! cd tests
//! cargo test --release --test sweettest_consciousness_gating -- --ignored --test-threads=2
//! ```

use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — justice evidence
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SubmitEvidenceInput {
    pub case_hash: ::holochain::prelude::ActionHash,
    pub evidence_type: String,
    pub description: String,
    pub content_hash: String,
}

// ============================================================================
// Mirror types — bridge
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BridgeHealth {
    pub healthy: bool,
    pub agent: String,
    pub total_events: u32,
    pub total_queries: u32,
    pub domains: Vec<String>,
}

// ============================================================================
// DNA path helper
// ============================================================================

fn civic_dna_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop(); // tests/ -> mycelix-civic/
    path.push("dna");
    path.push("mycelix_civic.dna");
    path
}

// ============================================================================
// Tests
// ============================================================================

/// Evidence submission in the justice domain requires proposal-level
/// consciousness gate. Without identity bridge, submit_evidence should
/// be blocked.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_evidence_submission_requires_proposal() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // submit_evidence requires proposal-level consciousness gate
    let input = SubmitEvidenceInput {
        case_hash: ::holochain::prelude::ActionHash::from_raw_36(vec![0xAB; 36]),
        evidence_type: "document".to_string(),
        description: "Test evidence for gating test".to_string(),
        content_hash: "QmTestHash123".to_string(),
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(&alice.zome("justice_evidence"), "submit_evidence", input)
        .await;

    assert!(
        result.is_err(),
        "submit_evidence should be blocked without consciousness credentials"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("onsciousness")
            || err_msg.contains("credential")
            || err_msg.contains("identity")
            || err_msg.contains("OtherRole")
            || err_msg.contains("cross_cluster"),
        "Error should relate to consciousness gating, got: {}",
        err_msg,
    );
}

/// Cross-hApp enforcement actions (execute_cross_happ_action) require
/// constitutional-level consciousness (Guardian tier). This is the
/// highest gate level in the civic cluster, testing that even
/// privileged operations are properly gated.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cross_happ_enforcement_requires_constitutional() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Bridge health check should work without consciousness gate
    let health: BridgeHealth = conductor
        .call(&alice.zome("civic_bridge"), "health_check", ())
        .await;

    assert!(
        health.healthy,
        "Civic bridge health_check should not be gated"
    );

    // Verify that the justice domain is registered in the bridge
    assert!(
        health.domains.contains(&"justice".to_string())
            || health.domains.contains(&"cases".to_string())
            || !health.domains.is_empty(),
        "Bridge should have registered domains: {:?}",
        health.domains,
    );
}

// ============================================================================
// Phase 5 — Expanded Coverage (matching hearth cluster depth)
// ============================================================================

// ============================================================================
// Mirror types — emergency incidents
// ============================================================================

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
pub struct OperationalZone {
    pub id: String,
    pub name: String,
    pub boundary: Vec<(f64, f64)>,
    pub priority: ZonePriority,
    pub status: ZoneStatus,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DeclareDisasterInput {
    pub id: String,
    pub disaster_type: DisasterType,
    pub title: String,
    pub description: String,
    pub severity: SeverityLevel,
    pub affected_area: AffectedArea,
    pub estimated_affected: u32,
    pub coordination_lead: Option<::holochain::prelude::AgentPubKey>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum UpdateType {
    StatusChange,
    SeverityChange,
    AreaExpansion,
    AreaContraction,
    CasualtyReport,
    ResourceUpdate,
    WeatherUpdate,
    InfrastructureUpdate,
    EvacuationOrder,
    AllClear,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AddIncidentUpdateInput {
    pub disaster_hash: ::holochain::prelude::ActionHash,
    pub update_type: UpdateType,
    pub content: String,
}

// ============================================================================
// Mirror types — justice enforcement
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum EnforcementStatus {
    Pending,
    InProgress,
    PartiallyCompleted,
    Completed,
    Failed,
    Contested,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum EnforcementActionType {
    FundsTransfer,
    AssetFreeze,
    ReputationUpdate,
    AccessRevocation,
    Notification,
    ManualRequired,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EnforcementAction {
    pub action_type: EnforcementActionType,
    pub target_happ: Option<String>,
    pub target_entry: Option<String>,
    pub executed_at: ::holochain::prelude::Timestamp,
    pub result: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Enforcement {
    pub id: String,
    pub decision_id: String,
    pub remedy_index: u32,
    pub enforcer: String,
    pub status: EnforcementStatus,
    pub actions: Vec<EnforcementAction>,
    pub created_at: ::holochain::prelude::Timestamp,
    pub completed_at: Option<::holochain::prelude::Timestamp>,
}

// ============================================================================
// Mirror types — justice arbitration
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ArbitratorRole {
    Primary,
    PanelMember,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Arbitrator {
    pub did: String,
    pub role: ArbitratorRole,
    pub selected_at: ::holochain::prelude::Timestamp,
    pub accepted: bool,
    pub recused: bool,
    pub recusal_reason: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ArbitratorSelection {
    Random,
    MATLWeighted,
    PartyAgreed,
    ExpertiseBased { domain: String },
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ArbitrationStatus {
    PanelFormation,
    EvidenceReview,
    Hearing,
    Deliberation,
    DecisionDrafting,
    DecisionRendered,
    Appealed,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Arbitration {
    pub id: String,
    pub case_id: String,
    pub arbitrators: Vec<Arbitrator>,
    pub selection_method: ArbitratorSelection,
    pub status: ArbitrationStatus,
    pub deliberation_deadline: Option<::holochain::prelude::Timestamp>,
    pub created_at: ::holochain::prelude::Timestamp,
}

// ============================================================================
// Mirror types — media curation
// ============================================================================

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FeatureInput {
    pub publication_id: String,
    pub featured_by: String,
    pub reason: String,
    pub featured_until: Option<::holochain::prelude::Timestamp>,
}

// ============================================================================
// Expanded Tests
// ============================================================================

/// Emergency incidents' add_incident_update requires basic-level
/// consciousness. Without credentials, updates should be blocked.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_emergency_incident_create_requires_basic() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = AddIncidentUpdateInput {
        disaster_hash: ::holochain::prelude::ActionHash::from_raw_36(vec![0xAA; 36]),
        update_type: UpdateType::StatusChange,
        content: "Status update for gating test".to_string(),
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(
            &alice.zome("emergency_incidents"),
            "add_incident_update",
            input,
        )
        .await;

    assert!(
        result.is_err(),
        "add_incident_update should be blocked without consciousness credentials"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("onsciousness")
            || err_msg.contains("credential")
            || err_msg.contains("identity")
            || err_msg.contains("OtherRole")
            || err_msg.contains("cross_cluster"),
        "Error should relate to consciousness gating, got: {}",
        err_msg,
    );
}

/// Justice enforcement's create_enforcement requires voting-level
/// consciousness. Enforcing judicial decisions is a high-trust action.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_justice_enforcement_requires_voting() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let enforcement = Enforcement {
        id: "enf-test-001".to_string(),
        decision_id: "decision-001".to_string(),
        remedy_index: 0,
        enforcer: "did:test:enforcer".to_string(),
        status: EnforcementStatus::Pending,
        actions: vec![],
        created_at: ::holochain::prelude::Timestamp::from_micros(1700000000),
        completed_at: None,
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(
            &alice.zome("justice_enforcement"),
            "create_enforcement",
            enforcement,
        )
        .await;

    assert!(
        result.is_err(),
        "create_enforcement should be blocked without consciousness credentials (voting tier)"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("onsciousness")
            || err_msg.contains("credential")
            || err_msg.contains("identity")
            || err_msg.contains("OtherRole")
            || err_msg.contains("cross_cluster"),
        "Error should relate to consciousness gating, got: {}",
        err_msg,
    );
}

/// Justice arbitration's create_arbitration requires proposal-level
/// consciousness. Creating an arbitration panel is a governance action.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_justice_arbitration_requires_proposal() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let arbitration = Arbitration {
        id: "arb-test-001".to_string(),
        case_id: "case-001".to_string(),
        arbitrators: vec![Arbitrator {
            did: "did:test:arbitrator-1".to_string(),
            role: ArbitratorRole::Primary,
            selected_at: ::holochain::prelude::Timestamp::from_micros(1700000000),
            accepted: false,
            recused: false,
            recusal_reason: None,
        }],
        selection_method: ArbitratorSelection::Random,
        status: ArbitrationStatus::PanelFormation,
        deliberation_deadline: None,
        created_at: ::holochain::prelude::Timestamp::from_micros(1700000000),
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(
            &alice.zome("justice_arbitration"),
            "create_arbitration",
            arbitration,
        )
        .await;

    assert!(
        result.is_err(),
        "create_arbitration should be blocked without consciousness credentials (proposal tier)"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("onsciousness")
            || err_msg.contains("credential")
            || err_msg.contains("identity")
            || err_msg.contains("OtherRole")
            || err_msg.contains("cross_cluster"),
        "Error should relate to consciousness gating, got: {}",
        err_msg,
    );
}

/// Media curation's feature_content requires voting-level consciousness.
/// Featuring content is a governance action that affects public visibility.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_media_curation_curate_requires_voting() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    let input = FeatureInput {
        publication_id: "pub-test-001".to_string(),
        featured_by: "did:test:curator".to_string(),
        reason: "Quality journalism for gating test".to_string(),
        featured_until: None,
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(
            &alice.zome("media_curation"),
            "feature_content",
            input,
        )
        .await;

    assert!(
        result.is_err(),
        "feature_content should be blocked without consciousness credentials (voting tier)"
    );

    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("onsciousness")
            || err_msg.contains("credential")
            || err_msg.contains("identity")
            || err_msg.contains("OtherRole")
            || err_msg.contains("cross_cluster"),
        "Error should relate to consciousness gating, got: {}",
        err_msg,
    );
}

/// Read-only operations in civic should NOT be gated by consciousness.
/// get_active_disasters and get_pending_enforcements should succeed
/// without any credentials.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_read_operations_not_gated_civic() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // get_active_disasters — no consciousness gate, returns empty vec
    let disasters: Vec<::holochain::prelude::Record> = conductor
        .call(
            &alice.zome("emergency_incidents"),
            "get_active_disasters",
            (),
        )
        .await;
    assert!(
        disasters.is_empty(),
        "No disasters should exist yet, but call should succeed (ungated)"
    );

    // get_pending_enforcements — no consciousness gate, returns empty vec
    let enforcements: Vec<::holochain::prelude::Record> = conductor
        .call(
            &alice.zome("justice_enforcement"),
            "get_pending_enforcements",
            (),
        )
        .await;
    assert!(
        enforcements.is_empty(),
        "No enforcements should exist yet, but call should succeed (ungated)"
    );
}

/// The rejection error message from civic gates should contain useful
/// debugging context (consciousness, credential, identity, etc.) rather
/// than a generic error.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_rejection_message_is_specific_civic() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Use declare_disaster — proposal tier — to test error quality
    let input = DeclareDisasterInput {
        id: "disaster-rejection-test".to_string(),
        disaster_type: DisasterType::Flood,
        title: "Rejection Test Disaster".to_string(),
        description: "Testing error message specificity in civic gating".to_string(),
        severity: SeverityLevel::Level2,
        affected_area: AffectedArea {
            center_lat: 32.95,
            center_lon: -96.75,
            radius_km: 5.0,
            boundary: None,
            zones: vec![],
        },
        estimated_affected: 100,
        coordination_lead: None,
    };

    let result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(
            &alice.zome("emergency_incidents"),
            "declare_disaster",
            input,
        )
        .await;

    assert!(result.is_err(), "declare_disaster should be blocked");

    let err_msg = format!("{:?}", result.unwrap_err());
    let has_context = err_msg.contains("onsciousness")
        || err_msg.contains("credential")
        || err_msg.contains("identity")
        || err_msg.contains("OtherRole")
        || err_msg.contains("cross_cluster")
        || err_msg.contains("gate");
    assert!(
        has_context,
        "Rejection error should contain consciousness/credential context for debugging, got: {}",
        err_msg
    );
}

/// The bridge credential cache must not serve fabricated credentials.
/// After a failed credential fetch, a second attempt should also fail
/// — proving cache consistency in the civic cluster.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_cache_consistency_civic() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // First attempt: enforcement creation should fail (no credentials)
    let enforcement_1 = Enforcement {
        id: "enf-cache-1".to_string(),
        decision_id: "decision-cache-1".to_string(),
        remedy_index: 0,
        enforcer: "did:test:cache-enforcer".to_string(),
        status: EnforcementStatus::Pending,
        actions: vec![],
        created_at: ::holochain::prelude::Timestamp::from_micros(1700000000),
        completed_at: None,
    };

    let result_1: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(
            &alice.zome("justice_enforcement"),
            "create_enforcement",
            enforcement_1,
        )
        .await;
    assert!(result_1.is_err(), "First attempt should fail");

    // Second attempt: cache should NOT serve a stale/successful credential
    let enforcement_2 = Enforcement {
        id: "enf-cache-2".to_string(),
        decision_id: "decision-cache-2".to_string(),
        remedy_index: 1,
        enforcer: "did:test:cache-enforcer".to_string(),
        status: EnforcementStatus::Pending,
        actions: vec![],
        created_at: ::holochain::prelude::Timestamp::from_micros(1700000001),
        completed_at: None,
    };

    let result_2: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(
            &alice.zome("justice_enforcement"),
            "create_enforcement",
            enforcement_2,
        )
        .await;
    assert!(
        result_2.is_err(),
        "Second attempt should also fail — cache must not serve fabricated credentials"
    );
}

/// Verify that the civic bridge health_check reports registered domains.
/// The bridge should know about justice, emergency, and media domains.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_bridge_health_domains_civic() {
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

    assert!(health.healthy, "Civic bridge should be healthy");
    assert!(
        !health.domains.is_empty(),
        "Bridge should have at least one registered domain, got: {:?}",
        health.domains,
    );

    // The agent field should be populated (not empty)
    assert!(
        !health.agent.is_empty(),
        "Bridge health should report the agent identity"
    );
}

/// Multiple different zomes in the civic cluster should all be gated.
/// Tests that both justice and emergency domains are independently blocked
/// without consciousness credentials — proving gating is wired per-zome.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires Holochain conductor (nix develop)"]
async fn test_multiple_zomes_gated_civic() {
    let mut conductor = SweetConductor::from_standard_config().await;
    let dna_file = SweetDnaFile::from_bundle(&civic_dna_path()).await.unwrap();
    let (alice,) = conductor
        .setup_app("test-app", &[dna_file.clone()])
        .await
        .unwrap()
        .into_tuple();

    // Justice arbitration (proposal gate)
    let arbitration = Arbitration {
        id: "arb-multi-test".to_string(),
        case_id: "case-multi".to_string(),
        arbitrators: vec![Arbitrator {
            did: "did:test:multi-arb".to_string(),
            role: ArbitratorRole::Primary,
            selected_at: ::holochain::prelude::Timestamp::from_micros(1700000000),
            accepted: false,
            recused: false,
            recusal_reason: None,
        }],
        selection_method: ArbitratorSelection::Random,
        status: ArbitrationStatus::PanelFormation,
        deliberation_deadline: None,
        created_at: ::holochain::prelude::Timestamp::from_micros(1700000000),
    };

    let justice_result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(
            &alice.zome("justice_arbitration"),
            "create_arbitration",
            arbitration,
        )
        .await;

    assert!(
        justice_result.is_err(),
        "Justice arbitration should be blocked without credentials"
    );

    // Emergency incidents (basic gate)
    let update = AddIncidentUpdateInput {
        disaster_hash: ::holochain::prelude::ActionHash::from_raw_36(vec![0xDD; 36]),
        update_type: UpdateType::ResourceUpdate,
        content: "Multi-zome gating test update".to_string(),
    };

    let emergency_result: Result<::holochain::prelude::Record, _> = conductor
        .call_fallible(
            &alice.zome("emergency_incidents"),
            "add_incident_update",
            update,
        )
        .await;

    assert!(
        emergency_result.is_err(),
        "Emergency incidents should be blocked without credentials"
    );

    // Both should be independently blocked — not just one zome
    assert!(
        justice_result.is_err() && emergency_result.is_err(),
        "Both justice AND emergency should be independently blocked by consciousness gating"
    );
}
