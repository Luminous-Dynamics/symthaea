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
//! cargo test -p civic-tests --test sweettest_integration -- --ignored
//! ```

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
pub struct CivicBridgeHealth {
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
    Report,
    Opinion,
    Investigation,
    FactCheck,
    Editorial,
    RawData,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum License {
    CreativeCommons(String),
    PublicDomain,
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
pub enum EpistemicPosition {
    Established,
    Emerging,
    Contested,
    Speculative,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub enum FactCheckVerdict {
    True,
    False,
    PartiallyTrue,
    Misleading,
    Unverifiable,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EvidenceItem {
    pub source_url: String,
    pub description: String,
    pub evidence_type: String,
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
    let conductor = SweetConductor::from_standard_config().await;
    let (alice,) = conductor
        .setup_app("test-app", &[DnaSource::Path(civic_dna_path())])
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
    let conductor = SweetConductor::from_standard_config().await;
    let (alice,) = conductor
        .setup_app("test-app", &[DnaSource::Path(civic_dna_path())])
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
    let conductor = SweetConductor::from_standard_config().await;
    let (alice,) = conductor
        .setup_app("test-app", &[DnaSource::Path(civic_dna_path())])
        .await
        .unwrap()
        .into_tuple();

    let health: CivicBridgeHealth = conductor
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
    let conductor = SweetConductor::from_standard_config().await;
    let (alice,) = conductor
        .setup_app("test-app", &[DnaSource::Path(civic_dna_path())])
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
        license: License::CreativeCommons("CC-BY-4.0".to_string()),
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
        epistemic_position: EpistemicPosition::Contested,
        verdict: FactCheckVerdict::PartiallyTrue,
        evidence: vec![EvidenceItem {
            source_url: "https://example.com/water-law".to_string(),
            description: "UN Water Rights resolution".to_string(),
            evidence_type: "legal_document".to_string(),
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
    let conductor = SweetConductor::from_standard_config().await;
    let (alice,) = conductor
        .setup_app("test-app", &[DnaSource::Path(civic_dna_path())])
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
        license: License::PublicDomain,
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
    let conductor = SweetConductor::from_standard_config().await;
    let (alice,) = conductor
        .setup_app("test-app", &[DnaSource::Path(civic_dna_path())])
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
