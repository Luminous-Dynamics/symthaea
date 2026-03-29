// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! LUCID Bridge Coordinator Zome
//!
//! Integration with Mycelix ecosystem hApps.

use hdk::prelude::*;
use bridge_integrity::*;

fn anchor_hash(s: &str) -> ExternResult<EntryHash> {
    hash_entry(&EntryTypes::Anchor(Anchor(s.to_string())))
}

fn create_anchor(s: &str) -> ExternResult<EntryHash> {
    create_entry(&EntryTypes::Anchor(Anchor(s.to_string())))?;
    anchor_hash(s)
}

/// Record federation of a thought to external hApp
#[hdk_extern]
pub fn record_federation(input: RecordFederationInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let record = FederationRecord {
        thought_id: input.thought_id.clone(),
        target_happ: input.target_happ,
        external_id: input.external_id,
        status: FederationStatus::Pending,
        federated_at: now,
    };

    let action_hash = create_entry(&EntryTypes::FederationRecord(record))?;

    let thought_anchor = format!("thought:{}", input.thought_id);
    create_anchor(&thought_anchor)?;
    create_link(
        anchor_hash(&thought_anchor)?,
        action_hash.clone(),
        LinkTypes::ThoughtToFederation,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Federation record not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordFederationInput {
    pub thought_id: String,
    pub target_happ: String,
    pub external_id: Option<String>,
}

/// Get federation records for a thought
#[hdk_extern]
pub fn get_thought_federations(thought_id: String) -> ExternResult<Vec<Record>> {
    let thought_anchor = format!("thought:{}", thought_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&thought_anchor)?, LinkTypes::ThoughtToFederation)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            records.push(record);
        }
    }

    Ok(records)
}

/// Cache external reputation score
#[hdk_extern]
pub fn cache_reputation(input: CacheReputationInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let reputation = ExternalReputation {
        agent: input.agent.clone(),
        k_vector: input.k_vector,
        trust_score: input.trust_score,
        source_happ: input.source_happ,
        fetched_at: now,
    };

    let action_hash = create_entry(&EntryTypes::ExternalReputation(reputation))?;

    let agent_anchor = format!("agent_reputation:{}", input.agent);
    create_anchor(&agent_anchor)?;
    create_link(
        anchor_hash(&agent_anchor)?,
        action_hash.clone(),
        LinkTypes::AgentToReputation,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Reputation not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CacheReputationInput {
    pub agent: AgentPubKey,
    pub k_vector: [f64; 8],
    pub trust_score: f64,
    pub source_happ: String,
}

/// Get cached reputation for an agent
#[hdk_extern]
pub fn get_cached_reputation(agent: AgentPubKey) -> ExternResult<Option<Record>> {
    let agent_anchor = format!("agent_reputation:{}", agent);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&agent_anchor)?, LinkTypes::AgentToReputation)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;
        return get(hash, GetOptions::default());
    }

    Ok(None)
}

/// Store a coherence analysis result computed by Symthaea via Tauri
///
/// This is called by the frontend after performing coherence analysis in the
/// Tauri layer using Symthaea's consciousness engine.
#[hdk_extern]
pub fn store_coherence_analysis(input: StoreCoherenceInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let analysis = bridge_integrity::CoherenceAnalysis {
        analysis_id: input.analysis_id.clone(),
        thought_ids: input.thought_ids.clone(),
        overall: input.overall,
        logical: input.logical,
        temporal: input.temporal,
        epistemic: input.epistemic,
        harmonic: input.harmonic,
        contradictions: input.contradictions.into_iter().map(|c| {
            bridge_integrity::DetectedContradiction {
                thought_a: c.thought_a,
                thought_b: c.thought_b,
                description: c.description,
                severity: c.severity,
            }
        }).collect(),
        analyzed_at: now,
    };

    let action_hash = create_entry(&EntryTypes::CoherenceAnalysis(analysis))?;

    // Link from a coherence anchor for retrieval
    let coherence_anchor = format!("coherence:{}", input.analysis_id);
    create_anchor(&coherence_anchor)?;
    create_link(
        anchor_hash(&coherence_anchor)?,
        action_hash.clone(),
        LinkTypes::ThoughtsToCoherence,
        (),
    )?;

    // Also link from each thought for lookup
    for thought_id in &input.thought_ids {
        let thought_coherence_anchor = format!("thought_coherence:{}", thought_id);
        create_anchor(&thought_coherence_anchor)?;
        create_link(
            anchor_hash(&thought_coherence_anchor)?,
            action_hash.clone(),
            LinkTypes::ThoughtsToCoherence,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Coherence analysis not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct StoreCoherenceInput {
    pub analysis_id: String,
    pub thought_ids: Vec<String>,
    pub overall: f64,
    pub logical: f64,
    pub temporal: f64,
    pub epistemic: f64,
    pub harmonic: f64,
    pub contradictions: Vec<ContradictionInput>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ContradictionInput {
    pub thought_a: String,
    pub thought_b: String,
    pub description: String,
    pub severity: f64,
}

/// Get coherence analyses involving a specific thought
#[hdk_extern]
pub fn get_thought_coherence_analyses(thought_id: String) -> ExternResult<Vec<Record>> {
    let thought_coherence_anchor = format!("thought_coherence:{}", thought_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&thought_coherence_anchor)?, LinkTypes::ThoughtsToCoherence)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            records.push(record);
        }
    }

    Ok(records)
}

/// Get a specific coherence analysis by ID
#[hdk_extern]
pub fn get_coherence_analysis(analysis_id: String) -> ExternResult<Option<Record>> {
    let coherence_anchor = format!("coherence:{}", analysis_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&coherence_anchor)?, LinkTypes::ThoughtsToCoherence)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;
        return get(hash, GetOptions::default());
    }

    Ok(None)
}

// ============================================================================
// COHERENCE FEEDBACK LOOP
// ============================================================================

/// Get thoughts that should be checked for coherence with a new thought
/// Returns the most semantically similar thoughts based on tags, domain, and recency
#[derive(Serialize, Deserialize, Debug)]
pub struct GetCoherenceCandidatesInput {
    pub thought_id: String,
    pub max_candidates: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CoherenceCandidate {
    pub thought_id: String,
    pub content: String,
    pub reason: String,  // Why this thought is a candidate (shared tags, same domain, etc.)
}

/// Prepare data for coherence checking with a newly created thought
/// Called by Tauri to get candidates before running Symthaea analysis
#[hdk_extern]
pub fn get_coherence_candidates(input: GetCoherenceCandidatesInput) -> ExternResult<Vec<CoherenceCandidate>> {
    let _max = input.max_candidates.unwrap_or(10) as usize;

    // This would normally call to the lucid zome to get related thoughts
    // For now, return empty - Tauri will use semantic search with embeddings
    Ok(vec![])
}

/// Result of coherence check - for UI display
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CoherenceCheckResult {
    pub thought_id: String,
    pub overall_coherence: f64,
    pub phi: f64,
    pub is_coherent: bool,
    pub potential_contradictions: Vec<PotentialContradiction>,
    pub suggestions: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PotentialContradiction {
    pub conflicting_thought_id: String,
    pub conflicting_content_preview: String,
    pub severity: f64,
    pub contradiction_type: String,
    pub description: String,
}

/// Get the latest coherence status for a thought
#[hdk_extern]
pub fn get_thought_coherence_status(thought_id: String) -> ExternResult<Option<CoherenceCheckResult>> {
    let analyses = get_thought_coherence_analyses(thought_id.clone())?;

    if let Some(record) = analyses.last() {
        if let Some(analysis) = record
            .entry()
            .to_app_option::<bridge_integrity::CoherenceAnalysis>()
            .ok()
            .flatten()
        {
            let contradictions: Vec<PotentialContradiction> = analysis.contradictions
                .iter()
                .filter(|c| c.thought_a == thought_id || c.thought_b == thought_id)
                .map(|c| {
                    let other_thought = if c.thought_a == thought_id {
                        c.thought_b.clone()
                    } else {
                        c.thought_a.clone()
                    };
                    PotentialContradiction {
                        conflicting_thought_id: other_thought,
                        conflicting_content_preview: c.description.chars().take(100).collect(),
                        severity: c.severity,
                        contradiction_type: "semantic".to_string(),
                        description: c.description.clone(),
                    }
                })
                .collect();

            let suggestions = if analysis.overall < 0.5 {
                vec![
                    "Consider reviewing thoughts with low coherence".to_string(),
                    "Check for potential contradictions in your beliefs".to_string(),
                ]
            } else {
                vec![]
            };

            return Ok(Some(CoherenceCheckResult {
                thought_id,
                overall_coherence: analysis.overall,
                phi: analysis.harmonic, // Using harmonic as phi proxy
                is_coherent: analysis.overall >= 0.5 && analysis.contradictions.is_empty(),
                potential_contradictions: contradictions,
                suggestions,
            }));
        }
    }

    Ok(None)
}

/// Get all thoughts with detected contradictions
#[hdk_extern]
pub fn get_contradicting_thoughts(_: ()) -> ExternResult<Vec<CoherenceCheckResult>> {
    let _agent = agent_info()?.agent_initial_pubkey;
    let all_coherence_anchor = "all_coherence";

    // Get all coherence analyses
    let links = get_links(
        LinkQuery::try_new(anchor_hash(all_coherence_anchor)?, LinkTypes::ThoughtsToCoherence)?,
        GetStrategy::default(),
    )?;

    let mut results: Vec<CoherenceCheckResult> = Vec::new();

    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid target".into())))?;

        if let Some(record) = get(hash, GetOptions::default())? {
            if let Some(analysis) = record
                .entry()
                .to_app_option::<bridge_integrity::CoherenceAnalysis>()
                .ok()
                .flatten()
            {
                if !analysis.contradictions.is_empty() {
                    for thought_id in &analysis.thought_ids {
                        let contradictions: Vec<PotentialContradiction> = analysis.contradictions
                            .iter()
                            .filter(|c| c.thought_a == *thought_id || c.thought_b == *thought_id)
                            .map(|c| {
                                let other = if c.thought_a == *thought_id {
                                    c.thought_b.clone()
                                } else {
                                    c.thought_a.clone()
                                };
                                PotentialContradiction {
                                    conflicting_thought_id: other,
                                    conflicting_content_preview: c.description.chars().take(100).collect(),
                                    severity: c.severity,
                                    contradiction_type: "semantic".to_string(),
                                    description: c.description.clone(),
                                }
                            })
                            .collect();

                        if !contradictions.is_empty() {
                            results.push(CoherenceCheckResult {
                                thought_id: thought_id.clone(),
                                overall_coherence: analysis.overall,
                                phi: analysis.harmonic,
                                is_coherent: false,
                                potential_contradictions: contradictions,
                                suggestions: vec!["Review and resolve this contradiction".to_string()],
                            });
                        }
                    }
                }
            }
        }
    }

    Ok(results)
}
