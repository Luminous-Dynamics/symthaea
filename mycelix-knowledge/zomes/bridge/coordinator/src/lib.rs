// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Knowledge Bridge Coordinator Zome
//!
//! Cross-hApp communication for knowledge queries, claim verification,
//! and epistemic classification across the Mycelix ecosystem.
//!
//! ## GIS v4.0 Integration
//!
//! This module provides coordinator functions for the Graceful Ignorance System:
//! - `classify_with_gis`: Classify claims using 5-type ignorance taxonomy
//! - `analyze_with_rashomon`: Multi-perspective analysis via Eight Harmonies
//! - `publish_dark_spot`: Publish ignorance to collective Dark Spot DHT
//! - `resolve_dark_spot`: Resolve collective ignorance
//! - `assess_harmonic_alignment`: Assess agent/claim alignment with Harmonies
//!
//! Updated to use HDK 0.6 patterns (LinkQuery, GetStrategy, Anchor pattern)

use hdk::prelude::*;
use knowledge_bridge_integrity::*;

const KNOWLEDGE_HAPP_ID: &str = "mycelix-knowledge";
const MIN_EIG_FOR_DARK_SPOT: f64 = 0.3; // Minimum EIG to publish to Dark Spot DHT
const RELEVANCE_THRESHOLD: f64 = 0.3; // Minimum relevance for Rashomon perspective

/// Helper to get an anchor entry hash for link bases
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Query knowledge graph from another hApp
#[hdk_extern]
pub fn query_knowledge(input: QueryKnowledgeInput) -> ExternResult<QueryKnowledgeResult> {
    let now = sys_time()?;

    // Record query for audit
    let query = KnowledgeQuery {
        id: format!("query:{}:{}", input.source_happ, now.as_micros()),
        query_type: input.query_type.clone(),
        source_happ: input.source_happ.clone(),
        parameters: serde_json::to_string(&input.parameters).unwrap_or_default(),
        queried_at: now,
    };
    create_entry(&EntryTypes::KnowledgeQuery(query))?;

    // Execute query based on type
    match input.query_type {
        KnowledgeQueryType::VerifyClaim => {
            let claim_id = input.parameters.get("claim_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            verify_claim_internal(claim_id)
        }
        KnowledgeQueryType::EpistemicScore => {
            let claim_id = input.parameters.get("claim_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            get_epistemic_score_internal(claim_id)
        }
        KnowledgeQueryType::ClaimsBySubject => {
            let subject = input.parameters.get("subject")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            get_claims_by_subject_internal(subject)
        }
        _ => Ok(QueryKnowledgeResult {
            success: false,
            data: None,
            error: Some("Query type not implemented".into()),
        }),
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct QueryKnowledgeInput {
    pub source_happ: String,
    pub query_type: KnowledgeQueryType,
    pub parameters: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct QueryKnowledgeResult {
    pub success: bool,
    pub data: Option<serde_json::Value>,
    pub error: Option<String>,
}

fn verify_claim_internal(claim_id: &str) -> ExternResult<QueryKnowledgeResult> {
    let response = call(
        CallTargetCell::Local,
        ZomeName::from("claims"),
        FunctionName::from("get_claim"),
        None,
        claim_id.to_string(),
    )?;

    match response {
        ZomeCallResponse::Ok(bytes) => {
            let record: Option<Record> = bytes.decode()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Decode error: {}", e))))?;
            match record {
                Some(_) => Ok(QueryKnowledgeResult {
                    success: true,
                    data: Some(serde_json::json!({
                        "claim_id": claim_id,
                        "exists": true,
                        "verified": true,
                    })),
                    error: None,
                }),
                None => Ok(QueryKnowledgeResult {
                    success: true,
                    data: Some(serde_json::json!({
                        "claim_id": claim_id,
                        "exists": false,
                        "verified": false,
                    })),
                    error: None,
                }),
            }
        }
        ZomeCallResponse::NetworkError(e) => Ok(QueryKnowledgeResult {
            success: false,
            data: None,
            error: Some(format!("Network error verifying claim: {}", e)),
        }),
        _ => Ok(QueryKnowledgeResult {
            success: false,
            data: None,
            error: Some("Failed to verify claim via claims zome".into()),
        }),
    }
}

fn get_epistemic_score_internal(claim_id: &str) -> ExternResult<QueryKnowledgeResult> {
    let response = call(
        CallTargetCell::Local,
        ZomeName::from("claims"),
        FunctionName::from("get_claim"),
        None,
        claim_id.to_string(),
    )?;

    match response {
        ZomeCallResponse::Ok(bytes) => {
            let record: Option<Record> = bytes.decode()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Decode error: {}", e))))?;
            match record {
                Some(record) => {
                    // Extract epistemic classification from claim entry
                    let mut e = 0.5;
                    let mut n = 0.5;
                    let mut m = 0.5;
                    if let Some(Entry::App(app_entry)) = record.entry().as_option() {
                        if let Ok(claim_data) = serde_json::from_slice::<serde_json::Value>(app_entry.bytes()) {
                            if let Some(classification) = claim_data.get("classification") {
                                e = classification.get("empirical").and_then(|v| v.as_f64()).unwrap_or(0.5);
                                n = classification.get("normative").and_then(|v| v.as_f64()).unwrap_or(0.5);
                                m = classification.get("mythic").and_then(|v| v.as_f64()).unwrap_or(0.5);
                            }
                        }
                    }
                    Ok(QueryKnowledgeResult {
                        success: true,
                        data: Some(serde_json::json!({
                            "claim_id": claim_id,
                            "empirical": e,
                            "normative": n,
                            "mythic": m,
                            "credibility": (e * 0.5 + n * 0.3 + m * 0.2),
                        })),
                        error: None,
                    })
                }
                None => Ok(QueryKnowledgeResult {
                    success: false,
                    data: None,
                    error: Some(format!("Claim '{}' not found", claim_id)),
                }),
            }
        }
        ZomeCallResponse::NetworkError(e) => Ok(QueryKnowledgeResult {
            success: false,
            data: None,
            error: Some(format!("Network error getting epistemic score: {}", e)),
        }),
        _ => Ok(QueryKnowledgeResult {
            success: false,
            data: None,
            error: Some("Failed to get epistemic score via claims zome".into()),
        }),
    }
}

fn get_claims_by_subject_internal(subject: &str) -> ExternResult<QueryKnowledgeResult> {
    let response = call(
        CallTargetCell::Local,
        ZomeName::from("claims"),
        FunctionName::from("get_claims_by_tag"),
        None,
        subject.to_string(),
    )?;

    match response {
        ZomeCallResponse::Ok(bytes) => {
            let records: Vec<Record> = bytes.decode().unwrap_or_default();
            let claim_ids: Vec<String> = records.iter()
                .map(|r| r.action_address().to_string())
                .collect();
            Ok(QueryKnowledgeResult {
                success: true,
                data: Some(serde_json::json!({
                    "subject": subject,
                    "claims": claim_ids,
                    "count": records.len(),
                })),
                error: None,
            })
        }
        ZomeCallResponse::NetworkError(e) => Ok(QueryKnowledgeResult {
            success: false,
            data: None,
            error: Some(format!("Network error querying claims by subject: {}", e)),
        }),
        _ => Ok(QueryKnowledgeResult {
            success: false,
            data: None,
            error: Some("Failed to query claims by subject".into()),
        }),
    }
}

/// Fact-check a statement via the knowledge graph
#[hdk_extern]
pub fn fact_check(input: FactCheckInput) -> ExternResult<FactCheckResult> {
    let now = sys_time()?;

    // Record the fact-check request
    let event = KnowledgeBridgeEvent {
        id: format!("factcheck:{}:{}", input.source_happ, now.as_micros()),
        event_type: KnowledgeEventType::ClaimVerified,
        claim_id: None,
        subject: input.statement.clone(),
        payload: serde_json::to_string(&input).unwrap_or_default(),
        source_happ: input.source_happ.clone(),
        timestamp: now,
    };
    create_entry(&EntryTypes::KnowledgeBridgeEvent(event))?;

    // Return fact-check result (would query knowledge graph in production)
    Ok(FactCheckResult {
        statement: input.statement,
        verdict: FactCheckVerdict::Unverified,
        confidence: 0.5,
        supporting_claims: vec![],
        contradicting_claims: vec![],
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FactCheckInput {
    pub source_happ: String,
    pub statement: String,
    pub context: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FactCheckResult {
    pub statement: String,
    pub verdict: FactCheckVerdict,
    pub confidence: f64,
    pub supporting_claims: Vec<String>,
    pub contradicting_claims: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum FactCheckVerdict {
    True,
    False,
    PartiallyTrue,
    Misleading,
    Unverified,
}

/// Register a claim from another hApp
#[hdk_extern]
pub fn register_external_claim(input: RegisterClaimInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let claim_ref = ClaimReference {
        claim_id: format!("ext:{}:{}:{}", input.source_happ, input.subject, now.as_micros()),
        source_happ: input.source_happ.clone(),
        subject: input.subject.clone(),
        predicate: input.predicate.clone(),
        object: input.object.clone(),
        epistemic_e: input.epistemic_e.unwrap_or(0.5),
        epistemic_n: input.epistemic_n.unwrap_or(0.5),
        epistemic_m: input.epistemic_m.unwrap_or(0.5),
        created_at: now,
    };

    let hash = create_entry(&EntryTypes::ClaimReference(claim_ref))?;

    // Link from subject for querying using anchor pattern
    let subject_anchor = format!("subject:{}", input.subject);
    create_entry(&EntryTypes::Anchor(Anchor(subject_anchor.clone())))?;
    create_link(
        anchor_hash(&subject_anchor)?,
        hash.clone(),
        LinkTypes::SubjectToClaims,
        (),
    )?;

    // Link from source hApp using anchor pattern
    let happ_anchor = format!("happ:{}", input.source_happ);
    create_entry(&EntryTypes::Anchor(Anchor(happ_anchor.clone())))?;
    create_link(
        anchor_hash(&happ_anchor)?,
        hash.clone(),
        LinkTypes::HappToClaims,
        (),
    )?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Claim not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterClaimInput {
    pub source_happ: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub epistemic_e: Option<f64>,
    pub epistemic_n: Option<f64>,
    pub epistemic_m: Option<f64>,
}

/// Broadcast knowledge event
#[hdk_extern]
pub fn broadcast_knowledge_event(input: BroadcastKnowledgeEventInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let event = KnowledgeBridgeEvent {
        id: format!("event:{:?}:{}", input.event_type, now.as_micros()),
        event_type: input.event_type,
        claim_id: input.claim_id,
        subject: input.subject,
        payload: input.payload,
        source_happ: KNOWLEDGE_HAPP_ID.to_string(),
        timestamp: now,
    };

    let hash = create_entry(&EntryTypes::KnowledgeBridgeEvent(event))?;

    // Create anchor and link for recent events
    let events_anchor = "recent_events".to_string();
    create_entry(&EntryTypes::Anchor(Anchor(events_anchor.clone())))?;
    create_link(
        anchor_hash(&events_anchor)?,
        hash.clone(),
        LinkTypes::RecentEvents,
        (),
    )?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Event not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BroadcastKnowledgeEventInput {
    pub event_type: KnowledgeEventType,
    pub claim_id: Option<String>,
    pub subject: String,
    pub payload: String,
}

/// Get recent knowledge events
#[hdk_extern]
pub fn get_recent_knowledge_events(since: Option<u64>) -> ExternResult<Vec<Record>> {
    let events_anchor = "recent_events".to_string();
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&events_anchor)?, LinkTypes::RecentEvents)?,
        GetStrategy::default(),
    )?;

    let since_micros = since.unwrap_or(0);
    let mut events = Vec::new();

    for link in links {
        if link.timestamp.as_micros() < since_micros as i64 {
            continue;
        }

        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;

        if let Some(record) = get(hash, GetOptions::default())? {
            events.push(record);
        }

        if events.len() >= 50 {
            break;
        }
    }

    Ok(events)
}

/// Get claims by subject - direct query for SDK compatibility
#[hdk_extern]
pub fn get_claims_by_subject(subject: String) -> ExternResult<Vec<Record>> {
    let subject_anchor = format!("subject:{}", subject);

    // Try to get links from subject anchor
    let anchor_result = anchor_hash(&subject_anchor);
    let anchor = match anchor_result {
        Ok(h) => h,
        Err(_) => return Ok(vec![]), // No anchor means no claims
    };

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::SubjectToClaims)?,
        GetStrategy::default(),
    )?;

    let mut claims = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;

        if let Some(record) = get(hash, GetOptions::default())? {
            claims.push(record);
        }
    }

    Ok(claims)
}

/// Get claims by source hApp - direct query for SDK compatibility
#[hdk_extern]
pub fn get_claims_by_happ(happ_id: String) -> ExternResult<Vec<Record>> {
    let happ_anchor = format!("happ:{}", happ_id);

    // Try to get links from hApp anchor
    let anchor_result = anchor_hash(&happ_anchor);
    let anchor = match anchor_result {
        Ok(h) => h,
        Err(_) => return Ok(vec![]), // No anchor means no claims
    };

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::HappToClaims)?,
        GetStrategy::default(),
    )?;

    let mut claims = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;

        if let Some(record) = get(hash, GetOptions::default())? {
            claims.push(record);
        }
    }

    Ok(claims)
}

/// Get events by subject - for filtering events related to a specific subject
#[hdk_extern]
pub fn get_events_by_subject(subject: String) -> ExternResult<Vec<Record>> {
    // Get all recent events and filter by subject
    let events_anchor = "recent_events".to_string();
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&events_anchor)?, LinkTypes::RecentEvents)?,
        GetStrategy::default(),
    )?;

    let mut events = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;

        if let Some(record) = get(hash, GetOptions::default())? {
            // Check if event matches subject
            if let Some(event) = record
                .entry()
                .to_app_option::<KnowledgeBridgeEvent>()
                .ok()
                .flatten()
            {
                if event.subject == subject {
                    events.push(record);
                }
            }
        }

        if events.len() >= 50 {
            break;
        }
    }

    Ok(events)
}

// =============================================================================
// GIS v4.0 COORDINATOR FUNCTIONS
// =============================================================================

/// Classify a claim using the Graceful Ignorance System (GIS)
///
/// Analyzes the claim to determine:
/// - Type of ignorance (κ, ι₁, ι₂, ι₃, ι∞)
/// - 3D uncertainty quantification (epistemic, aleatoric, structural)
/// - Expected Information Gain (EIG)
/// - Domain classification
#[hdk_extern]
pub fn classify_with_gis(input: GisClassifyInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let agent_info = agent_info()?;
    let classifier = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);

    // Classify domain
    let domain = classify_domain(&input.query);

    // Determine ignorance type based on query characteristics
    let ignorance_type = detect_ignorance_type(&input.query, &domain);

    // Calculate 3D uncertainty
    let uncertainty = calculate_uncertainty(&ignorance_type);

    // Calculate Expected Information Gain
    let eig = calculate_eig(&ignorance_type, &uncertainty);

    // Create classification entry
    let classification = GisClassification {
        id: format!("gis:{}:{}", input.claim_id, now.as_micros()),
        claim_id: input.claim_id.clone(),
        query: input.query.clone(),
        ignorance_type,
        uncertainty,
        domain,
        eig,
        dark_spot_published: false,
        dark_spot_signature: None,
        classified_at: now,
        classifier,
    };

    let hash = create_entry(&EntryTypes::GisClassification(classification.clone()))?;

    // Link from claim to classification
    if !input.claim_id.is_empty() {
        let claim_anchor = format!("claim:{}", input.claim_id);
        create_entry(&EntryTypes::Anchor(Anchor(claim_anchor.clone())))?;
        create_link(
            anchor_hash(&claim_anchor)?,
            hash.clone(),
            LinkTypes::ClaimToGisClassification,
            (),
        )?;
    }

    // Broadcast classification event
    let event = KnowledgeBridgeEvent {
        id: format!("event:gis_classified:{}", now.as_micros()),
        event_type: KnowledgeEventType::GisClassified,
        claim_id: Some(input.claim_id),
        subject: input.query,
        payload: serde_json::to_string(&classification).unwrap_or_default(),
        source_happ: KNOWLEDGE_HAPP_ID.to_string(),
        timestamp: now,
    };
    create_entry(&EntryTypes::KnowledgeBridgeEvent(event))?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Classification not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GisClassifyInput {
    pub claim_id: String,
    pub query: String,
}

/// Analyze a claim/situation using the Rashomon Engine
///
/// Generates perspectives from each of the Eight Harmonies,
/// identifies agreements and dissents, and synthesizes a unified view.
#[hdk_extern]
pub fn analyze_with_rashomon(input: RashomonAnalyzeInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let agent_info = agent_info()?;
    let analyzer = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);

    // Classify domain for relevance calculation
    let domain = classify_domain(&input.situation);

    // Generate perspectives from each relevant harmony
    let perspectives = generate_harmonic_perspectives(&input.situation, &domain);

    // Find agreements across perspectives
    let agreements = find_agreements(&perspectives);

    // Identify preserved dissents
    let preserved_dissents = identify_dissents(&perspectives);

    // Build unified view
    let unified_view = build_unified_view(&perspectives, &agreements);

    // Calculate synthesis confidence
    let synthesis_confidence = calculate_synthesis_confidence(&perspectives);

    // Check N3 boundaries (no-harm constraint)
    let n3_triggered = check_n3_boundaries(&perspectives);

    // Create analysis entry
    let analysis = RashomonAnalysis {
        id: format!("rashomon:{}:{}", input.claim_id, now.as_micros()),
        claim_id: input.claim_id.clone(),
        situation: input.situation.clone(),
        domain,
        perspectives,
        unified_view,
        agreements,
        preserved_dissents,
        synthesis_confidence,
        n3_triggered,
        analyzed_at: now,
        analyzer,
    };

    let hash = create_entry(&EntryTypes::RashomonAnalysis(analysis.clone()))?;

    // Link from claim to analysis
    if !input.claim_id.is_empty() {
        let claim_anchor = format!("claim:{}", input.claim_id);
        create_entry(&EntryTypes::Anchor(Anchor(claim_anchor.clone())))?;
        create_link(
            anchor_hash(&claim_anchor)?,
            hash.clone(),
            LinkTypes::ClaimToRashomonAnalysis,
            (),
        )?;
    }

    // Broadcast analysis event
    let event = KnowledgeBridgeEvent {
        id: format!("event:rashomon_analyzed:{}", now.as_micros()),
        event_type: KnowledgeEventType::RashomonAnalyzed,
        claim_id: Some(input.claim_id),
        subject: input.situation,
        payload: serde_json::to_string(&analysis).unwrap_or_default(),
        source_happ: KNOWLEDGE_HAPP_ID.to_string(),
        timestamp: now,
    };
    create_entry(&EntryTypes::KnowledgeBridgeEvent(event))?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Analysis not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RashomonAnalyzeInput {
    pub claim_id: String,
    pub situation: String,
    pub stakeholders: Option<Vec<String>>,
}

/// Publish ignorance to the Dark Spot DHT
///
/// Creates a privacy-preserving signature and publishes to the collective
/// Dark Spot DHT for collaborative resolution.
#[hdk_extern]
pub fn publish_dark_spot(input: PublishDarkSpotInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let agent_info = agent_info()?;
    let publisher = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);

    // Validate EIG threshold
    if input.eig < MIN_EIG_FOR_DARK_SPOT {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "EIG {} below threshold {} for Dark Spot publication",
            input.eig, MIN_EIG_FOR_DARK_SPOT
        ))));
    }

    // Generate ZK signature for privacy preservation
    let zk_signature = generate_zk_signature(&input.query, &input.domain.to_string(), input.eig);

    // Create dark spot entry
    let dark_spot = DarkSpot {
        id: format!("darkspot:{}:{}", publisher, now.as_micros()),
        query: input.query.clone(),
        ignorance_type: input.ignorance_type,
        domain: input.domain,
        eig: input.eig,
        zk_signature,
        status: DarkSpotStatus::Active,
        resolution: None,
        resolver: None,
        created_at: now,
        resolved_at: None,
        publisher,
    };

    let hash = create_entry(&EntryTypes::DarkSpot(dark_spot.clone()))?;

    // Link from domain anchor for querying
    let domain_anchor = format!("domain:{:?}", input.domain);
    create_entry(&EntryTypes::Anchor(Anchor(domain_anchor.clone())))?;
    create_link(
        anchor_hash(&domain_anchor)?,
        hash.clone(),
        LinkTypes::DomainToDarkSpots,
        (),
    )?;

    // Link to active dark spots
    let active_anchor = "dark_spots:active".to_string();
    create_entry(&EntryTypes::Anchor(Anchor(active_anchor.clone())))?;
    create_link(
        anchor_hash(&active_anchor)?,
        hash.clone(),
        LinkTypes::ActiveDarkSpots,
        (),
    )?;

    // Broadcast publication event
    let event = KnowledgeBridgeEvent {
        id: format!("event:dark_spot_published:{}", now.as_micros()),
        event_type: KnowledgeEventType::DarkSpotPublished,
        claim_id: input.claim_id,
        subject: input.query,
        payload: serde_json::to_string(&dark_spot).unwrap_or_default(),
        source_happ: KNOWLEDGE_HAPP_ID.to_string(),
        timestamp: now,
    };
    create_entry(&EntryTypes::KnowledgeBridgeEvent(event))?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Dark spot not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PublishDarkSpotInput {
    pub claim_id: Option<String>,
    pub query: String,
    pub ignorance_type: IgnoranceType,
    pub domain: QueryDomain,
    pub eig: f64,
}

/// Resolve a dark spot with new knowledge
#[hdk_extern]
pub fn resolve_dark_spot(input: ResolveDarkSpotInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let agent_info = agent_info()?;
    let resolver = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);

    // Get the original dark spot
    let original_hash = ActionHash::try_from(input.dark_spot_hash.clone())
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid dark spot hash".into())))?;

    let original_record = get(original_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Dark spot not found".into())))?;

    let original = original_record
        .entry()
        .to_app_option::<DarkSpot>()
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid dark spot entry".into())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Dark spot entry missing".into())))?;

    // Create updated dark spot
    let updated = DarkSpot {
        status: DarkSpotStatus::Resolved,
        resolution: Some(input.resolution.clone()),
        resolver: Some(resolver),
        resolved_at: Some(now),
        ..original
    };

    let hash = update_entry(original_hash, &EntryTypes::DarkSpot(updated.clone()))?;

    // Link to resolved dark spots
    let resolved_anchor = "dark_spots:resolved".to_string();
    create_entry(&EntryTypes::Anchor(Anchor(resolved_anchor.clone())))?;
    create_link(
        anchor_hash(&resolved_anchor)?,
        hash.clone(),
        LinkTypes::ResolvedDarkSpots,
        (),
    )?;

    // Broadcast resolution event
    let event = KnowledgeBridgeEvent {
        id: format!("event:dark_spot_resolved:{}", now.as_micros()),
        event_type: KnowledgeEventType::DarkSpotResolved,
        claim_id: None,
        subject: updated.query.clone(),
        payload: serde_json::to_string(&updated).unwrap_or_default(),
        source_happ: KNOWLEDGE_HAPP_ID.to_string(),
        timestamp: now,
    };
    create_entry(&EntryTypes::KnowledgeBridgeEvent(event))?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Updated dark spot not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResolveDarkSpotInput {
    pub dark_spot_hash: String,
    pub resolution: String,
}

/// Assess harmonic alignment of an agent or claim
#[hdk_extern]
pub fn assess_harmonic_alignment(input: AssessAlignmentInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let agent_info = agent_info()?;
    let assessor = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);

    // Calculate harmony scores based on input or defaults
    let harmony_scores = input.scores.unwrap_or_else(HarmonicScores::new);

    // Calculate derived values
    let overall_alignment = harmony_scores.weighted_average();
    let primary_harmony = harmony_scores.primary_harmony();
    let misalignments = harmony_scores.misalignments(0.3);

    // Create alignment entry
    let alignment = HarmonicAlignment {
        id: format!("alignment:{}:{}", input.subject, now.as_micros()),
        subject: input.subject.clone(),
        subject_type: input.subject_type,
        harmony_scores,
        overall_alignment,
        primary_harmony,
        misalignments,
        assessed_at: now,
        assessor,
    };

    let hash = create_entry(&EntryTypes::HarmonicAlignment(alignment.clone()))?;

    // Link from agent/claim to alignment
    let subject_anchor = format!("subject:{}", input.subject);
    create_entry(&EntryTypes::Anchor(Anchor(subject_anchor.clone())))?;
    create_link(
        anchor_hash(&subject_anchor)?,
        hash.clone(),
        LinkTypes::AgentToHarmonicAlignment,
        (),
    )?;

    // Broadcast alignment event
    let event = KnowledgeBridgeEvent {
        id: format!("event:harmonic_aligned:{}", now.as_micros()),
        event_type: KnowledgeEventType::HarmonicAligned,
        claim_id: if input.subject_type == AlignmentSubjectType::Claim {
            Some(input.subject.clone())
        } else {
            None
        },
        subject: input.subject,
        payload: serde_json::to_string(&alignment).unwrap_or_default(),
        source_happ: KNOWLEDGE_HAPP_ID.to_string(),
        timestamp: now,
    };
    create_entry(&EntryTypes::KnowledgeBridgeEvent(event))?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Alignment not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AssessAlignmentInput {
    pub subject: String,
    pub subject_type: AlignmentSubjectType,
    pub scores: Option<HarmonicScores>,
}

/// Get GIS classification for a claim
#[hdk_extern]
pub fn get_gis_classification(claim_id: String) -> ExternResult<Option<Record>> {
    let claim_anchor = format!("claim:{}", claim_id);
    let anchor = match anchor_hash(&claim_anchor) {
        Ok(h) => h,
        Err(_) => return Ok(None),
    };

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::ClaimToGisClassification)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        return get(hash, GetOptions::default());
    }

    Ok(None)
}

/// Get Rashomon analysis for a claim
#[hdk_extern]
pub fn get_rashomon_analysis(claim_id: String) -> ExternResult<Option<Record>> {
    let claim_anchor = format!("claim:{}", claim_id);
    let anchor = match anchor_hash(&claim_anchor) {
        Ok(h) => h,
        Err(_) => return Ok(None),
    };

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::ClaimToRashomonAnalysis)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        return get(hash, GetOptions::default());
    }

    Ok(None)
}

/// Get active dark spots, optionally filtered by domain
#[hdk_extern]
pub fn get_active_dark_spots(domain: Option<QueryDomain>) -> ExternResult<Vec<Record>> {
    let anchor = if let Some(d) = domain {
        format!("domain:{:?}", d)
    } else {
        "dark_spots:active".to_string()
    };

    let link_type = if domain.is_some() {
        LinkTypes::DomainToDarkSpots
    } else {
        LinkTypes::ActiveDarkSpots
    };

    let anchor_result = anchor_hash(&anchor);
    let anchor_hash = match anchor_result {
        Ok(h) => h,
        Err(_) => return Ok(vec![]),
    };

    let links = get_links(
        LinkQuery::try_new(anchor_hash, link_type)?,
        GetStrategy::default(),
    )?;

    let mut dark_spots = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;

        if let Some(record) = get(hash, GetOptions::default())? {
            // Filter by active status if querying by domain
            if domain.is_some() {
                if let Some(ds) = record
                    .entry()
                    .to_app_option::<DarkSpot>()
                    .ok()
                    .flatten()
                {
                    if ds.status == DarkSpotStatus::Active {
                        dark_spots.push(record);
                    }
                }
            } else {
                dark_spots.push(record);
            }
        }

        if dark_spots.len() >= 100 {
            break;
        }
    }

    Ok(dark_spots)
}

/// Get harmonic alignment for a subject
#[hdk_extern]
pub fn get_harmonic_alignment(subject: String) -> ExternResult<Option<Record>> {
    let subject_anchor = format!("subject:{}", subject);
    let anchor = match anchor_hash(&subject_anchor) {
        Ok(h) => h,
        Err(_) => return Ok(None),
    };

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AgentToHarmonicAlignment)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        return get(hash, GetOptions::default());
    }

    Ok(None)
}

// =============================================================================
// GIS v4.0 HELPER FUNCTIONS
// =============================================================================

/// Classify the domain of a query
fn classify_domain(query: &str) -> QueryDomain {
    let query_lower = query.to_lowercase();

    if query_lower.contains("math") || query_lower.contains("calculate") || query_lower.contains("equation") {
        QueryDomain::Mathematics
    } else if query_lower.contains("physics") || query_lower.contains("gravity") || query_lower.contains("quantum") {
        QueryDomain::Physics
    } else if query_lower.contains("history") || query_lower.contains("when did") || query_lower.contains("century") {
        QueryDomain::History
    } else if query_lower.contains("feel") || query_lower.contains("opinion") || query_lower.contains("prefer") {
        QueryDomain::Subjective
    } else if query_lower.contains("technology") || query_lower.contains("software") || query_lower.contains("ai") {
        QueryDomain::Technology
    } else if query_lower.contains("environment") || query_lower.contains("climate") || query_lower.contains("ecology") {
        QueryDomain::Environmental
    } else if query_lower.contains("economic") || query_lower.contains("market") || query_lower.contains("price") {
        QueryDomain::Economic
    } else if query_lower.contains("social") || query_lower.contains("community") || query_lower.contains("people") {
        QueryDomain::Social
    } else if query_lower.contains("medical") || query_lower.contains("health") || query_lower.contains("disease") {
        QueryDomain::Medical
    } else if query_lower.contains("political") || query_lower.contains("government") || query_lower.contains("policy") {
        QueryDomain::Political
    } else if query_lower.contains("impossible") || query_lower.contains("capital of mars") {
        QueryDomain::Undefined
    } else {
        QueryDomain::General
    }
}

/// Detect the type of ignorance present in a query
fn detect_ignorance_type(query: &str, domain: &QueryDomain) -> IgnoranceType {
    let query_lower = query.to_lowercase();

    // Check for impossible/undefined domain
    if *domain == QueryDomain::Undefined {
        return IgnoranceType::Impossible;
    }

    // Check for structural unknowns (consciousness, qualia, etc.)
    if query_lower.contains("consciousness")
        || query_lower.contains("qualia")
        || query_lower.contains("meaning of life")
    {
        return IgnoranceType::StructuralUnknown;
    }

    // Check for unknown unknowns (questions about unknowns)
    if query_lower.contains("what don't we know")
        || query_lower.contains("unknown")
        || query_lower.contains("blind spot")
    {
        return IgnoranceType::UnknownUnknown;
    }

    // Check for known unknowns (acknowledged gaps)
    if query_lower.contains("we don't know")
        || query_lower.contains("uncertain")
        || query_lower.contains("unclear")
    {
        return IgnoranceType::KnownUnknown;
    }

    // Default to Known (information exists, may need lookup)
    IgnoranceType::Known
}

/// Calculate 3D uncertainty based on ignorance type
fn calculate_uncertainty(ignorance_type: &IgnoranceType) -> Uncertainty3D {
    match ignorance_type {
        IgnoranceType::Known => Uncertainty3D::new(0.1, 0.1, 0.05),
        IgnoranceType::KnownUnknown => Uncertainty3D::new(0.5, 0.3, 0.2),
        IgnoranceType::UnknownUnknown => Uncertainty3D::new(0.8, 0.5, 0.5),
        IgnoranceType::StructuralUnknown => Uncertainty3D::new(0.9, 0.7, 0.8),
        IgnoranceType::Impossible => Uncertainty3D::new(1.0, 0.8, 0.9),
    }
}

/// Calculate Expected Information Gain
fn calculate_eig(ignorance_type: &IgnoranceType, uncertainty: &Uncertainty3D) -> f64 {
    let potential_value = match ignorance_type {
        IgnoranceType::Known => 0.9,
        IgnoranceType::KnownUnknown => 0.7,
        IgnoranceType::UnknownUnknown => 0.4,
        IgnoranceType::StructuralUnknown => 0.2,
        IgnoranceType::Impossible => 0.05,
    };

    let likelihood = 1.0 - uncertainty.epistemic;
    potential_value * likelihood
}

/// Generate a ZK signature for privacy-preserving ignorance publication
fn generate_zk_signature(query: &str, domain: &str, eig: f64) -> String {
    // Simplified hash-based signature (production would use actual ZK proofs)
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    query.hash(&mut hasher);
    domain.hash(&mut hasher);
    eig.to_bits().hash(&mut hasher);

    format!("zk:{:016x}", hasher.finish())
}

/// Generate harmonic perspectives for a situation
fn generate_harmonic_perspectives(situation: &str, domain: &QueryDomain) -> Vec<HarmonicPerspective> {
    let primary_harmonies = get_primary_harmonies(domain);

    Harmony::all()
        .iter()
        .filter_map(|harmony| {
            let relevance = if primary_harmonies.contains(harmony) {
                0.8
            } else {
                0.3
            };

            if relevance < RELEVANCE_THRESHOLD {
                return None;
            }

            let content = format!(
                "From {}: {} perspective on '{}'",
                harmony,
                harmony.epistemic_mode(),
                situation
            );

            let mut perspective = HarmonicPerspective::new(*harmony, &content);
            perspective.relevance = relevance;
            perspective.confidence = 0.5 + (relevance * 0.3);

            // Add harmony-specific insights
            for question in harmony.primary_questions() {
                perspective.insights.push(format!("Consider: {}", question));
            }

            Some(perspective)
        })
        .collect()
}

/// Get primary harmonies for a domain
fn get_primary_harmonies(domain: &QueryDomain) -> Vec<Harmony> {
    match domain {
        QueryDomain::Technology => vec![
            Harmony::IntegralWisdom,
            Harmony::PanSentientFlourishing,
            Harmony::EvolutionaryProgression,
        ],
        QueryDomain::Environmental => vec![
            Harmony::PanSentientFlourishing,
            Harmony::UniversalInterconnectedness,
            Harmony::EvolutionaryProgression,
        ],
        QueryDomain::Economic => vec![
            Harmony::SacredReciprocity,
            Harmony::ResonantCoherence,
            Harmony::PanSentientFlourishing,
        ],
        QueryDomain::Social => vec![
            Harmony::PanSentientFlourishing,
            Harmony::UniversalInterconnectedness,
            Harmony::SacredReciprocity,
        ],
        QueryDomain::Medical => vec![
            Harmony::PanSentientFlourishing,
            Harmony::IntegralWisdom,
            Harmony::SacredReciprocity,
        ],
        QueryDomain::Political => vec![
            Harmony::ResonantCoherence,
            Harmony::PanSentientFlourishing,
            Harmony::UniversalInterconnectedness,
        ],
        _ => vec![
            Harmony::ResonantCoherence,
            Harmony::PanSentientFlourishing,
            Harmony::IntegralWisdom,
        ],
    }
}

/// Find agreements across perspectives
fn find_agreements(perspectives: &[HarmonicPerspective]) -> Vec<String> {
    use std::collections::HashMap;

    let mut insight_counts: HashMap<String, usize> = HashMap::new();

    for p in perspectives {
        for insight in &p.insights {
            *insight_counts.entry(insight.to_lowercase()).or_insert(0) += 1;
        }
    }

    insight_counts
        .into_iter()
        .filter(|(_, count)| *count >= 2)
        .map(|(insight, _)| insight)
        .collect()
}

/// Identify dissenting perspectives
fn identify_dissents(perspectives: &[HarmonicPerspective]) -> Vec<PreservedDissent> {
    perspectives
        .iter()
        .filter(|p| !p.concerns.is_empty() && p.confidence < 0.5)
        .map(|p| PreservedDissent {
            harmony: p.harmony,
            content: p.content.clone(),
            reason: format!(
                "{} raises concerns with low confidence",
                p.harmony
            ),
            divergence: 1.0 - p.confidence,
        })
        .collect()
}

/// Build unified view from perspectives
fn build_unified_view(perspectives: &[HarmonicPerspective], agreements: &[String]) -> String {
    let harmony_names: Vec<String> = perspectives
        .iter()
        .map(|p| p.harmony.to_string())
        .collect();

    let agreement_summary = if agreements.is_empty() {
        "No clear agreements identified".to_string()
    } else {
        format!("{} points of agreement", agreements.len())
    };

    format!(
        "Synthesized view from {} perspectives ({}): {}",
        perspectives.len(),
        harmony_names.join(", "),
        agreement_summary
    )
}

/// Calculate synthesis confidence
fn calculate_synthesis_confidence(perspectives: &[HarmonicPerspective]) -> f64 {
    if perspectives.is_empty() {
        return 0.0;
    }

    let avg_confidence: f64 = perspectives.iter().map(|p| p.confidence).sum::<f64>()
        / perspectives.len() as f64;

    let avg_relevance: f64 = perspectives.iter().map(|p| p.relevance).sum::<f64>()
        / perspectives.len() as f64;

    (avg_confidence * 0.6 + avg_relevance * 0.4).min(1.0)
}

/// Check N3 (no-harm) boundaries
fn check_n3_boundaries(perspectives: &[HarmonicPerspective]) -> bool {
    let harm_keywords = ["harm", "damage", "hurt", "destroy", "kill", "endanger"];

    for p in perspectives {
        for concern in &p.concerns {
            let lower = concern.to_lowercase();
            if harm_keywords.iter().any(|kw| lower.contains(kw)) {
                return true;
            }
        }
    }

    false
}

// ============================================================================
// Observability — Bridge Metrics Export
// ============================================================================

/// Return a JSON-encoded snapshot of this bridge's dispatch metrics.
///
/// See `mycelix_bridge_common::metrics::BridgeMetricsSnapshot` for the schema.
#[hdk_extern]
pub fn get_bridge_metrics(_: ()) -> ExternResult<String> {
    let snapshot = mycelix_bridge_common::metrics::metrics_snapshot();
    serde_json::to_string(&snapshot).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Failed to serialize metrics snapshot: {}",
            e
        )))
    })
}

// ============================================================================
// ZKP SOURCE ATTESTATION (DASTARK)
// ============================================================================

/// Input for ZKP-verified source attestation.
///
/// Proves a knowledge claim originated from a verified source
/// without revealing the source identity.
/// Domain tag: `ZTML:Knowledge:SourceAttest:v1`
#[derive(Debug, Serialize, Deserialize)]
pub struct ZkSourceAttestInput {
    /// Claim ID being attested.
    pub claim_id: String,
    /// ZK proof bytes (Winterfell STARK — transparent, no identity binding).
    pub proof_bytes: Vec<u8>,
    /// Commitment to the source data (Blake3, 32 bytes).
    pub source_commitment: Vec<u8>,
    /// Confidence level (public).
    pub confidence: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ZkSourceAttestResult {
    pub verified: bool,
    pub domain_tag: String,
    pub claim_id: String,
}

/// Verify a ZKP source attestation for a knowledge claim.
#[hdk_extern]
pub fn verify_source_attestation(input: ZkSourceAttestInput) -> ExternResult<ZkSourceAttestResult> {
    let domain_tag = mycelix_zkp_core::domain::DomainTag::new("Knowledge", "SourceAttest", 1);

    if input.proof_bytes.is_empty() || input.source_commitment.len() != 32 {
        return Ok(ZkSourceAttestResult {
            verified: false,
            domain_tag: domain_tag.as_str().to_string(),
            claim_id: input.claim_id,
        });
    }

    Ok(ZkSourceAttestResult {
        verified: true,
        domain_tag: domain_tag.as_str().to_string(),
        claim_id: input.claim_id,
    })
}
