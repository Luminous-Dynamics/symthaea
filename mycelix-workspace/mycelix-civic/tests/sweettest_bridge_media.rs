// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Civic — Bridge & Media Sweettest
//!
//! Batch 1 of civic sweettest integration tests: civic bridge basics
//! (query, broadcast, health), media publication + factcheck, and
//! cross-domain media/emergency/justice bridge scenarios.
//!
//! Split from sweettest_integration.rs to reduce per-process conductor
//! memory pressure (~1-2 GB per conductor). Each [[test]] binary runs
//! as a separate OS process, so memory is fully reclaimed between batches.
//!
//! ## Running
//! ```bash
//! cd mycelix-civic/tests
//! cargo test --release --test sweettest_bridge_media -- --ignored --test-threads=1
//! ```

use holochain::prelude::*;
use holochain::sweettest::*;
use std::path::PathBuf;

// ============================================================================
// Mirror types — civic-bridge
// ============================================================================

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

// ============================================================================
// Mirror types — media
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

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
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

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
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

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
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

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
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

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
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

    let resolve = ResolveQueryInput {
        query_hash,
        result: r#"{"publications":[],"count":0}"#.to_string(),
        success: true,
    };

    let resolved: Record = conductor
        .call(&alice.zome("civic_bridge"), "resolve_query", resolve)
        .await;

    assert!(resolved.action().author() == alice.agent_pubkey());

    drop(conductor);
    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
}
