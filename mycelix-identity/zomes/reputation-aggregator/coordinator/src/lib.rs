// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-cluster reputation aggregator coordinator zome.
//!
//! Aggregates reputation signals from multiple Mycelix clusters into a
//! unified composite score. Feeds the consciousness profile's `reputation`
//! dimension, closing the loop for data-driven governance weights.
//!
//! ## Architecture
//!
//! 1. **Local trust**: Calls `web-of-trust` zome (same DNA) for BFS trust chains
//! 2. **Cross-cluster scores**: Other clusters push `DomainScoreReport` entries
//! 3. **Composite**: Weighted average of all available scores
//! 4. **Enrichment**: Feeds composite into `ConsciousnessProfile.reputation`
//!
//! ## Weights
//!
//! | Source | Weight | Rationale |
//! |--------|--------|-----------|
//! | Web-of-Trust | 0.35 | Peer attestation, high signal |
//! | MYCEL Score | 0.25 | Ecosystem participation |
//! | Domain Average | 0.40 | Cross-cluster activity |

use hdk::prelude::*;
use reputation_aggregator_integrity::*;
use serde::{Deserialize, Serialize};

mod reputation_logic;
use reputation_logic::*;

use mycelix_bridge_common::consciousness_profile::ConsciousnessProfile;

// ============================================================================
// Constants
// ============================================================================

/// Weight for web-of-trust score in composite
const WEIGHT_WOT: f64 = 0.35;
/// Weight for MYCEL score in composite
const WEIGHT_MYCEL: f64 = 0.25;
/// Weight for domain score average in composite
const WEIGHT_DOMAIN: f64 = 0.40;
/// Staleness threshold: 24 hours in microseconds
const STALENESS_THRESHOLD_US: i64 = 86_400_000_000;

// ============================================================================
// Input/Output Types
// ============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DomainScoreInput {
    pub agent_pubkey_b64: String,
    pub cluster: String,
    pub score: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PaginatedAgentInput {
    pub agent_pubkey_b64: String,
    pub limit: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AggregatorMetrics {
    pub total_reputations: u32,
    pub total_domain_reports: u32,
    pub stale_count: u32,
}

// ============================================================================
// Helper: Anchor
// ============================================================================

fn ensure_anchor(name: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(name.to_string());
    create_entry(&EntryTypes::Anchor(anchor.clone()))?;
    hash_entry(&anchor)
}

// ============================================================================
// Extern Functions
// ============================================================================

/// Get the composite reputation for an agent, computing if not cached.
/// Get the composite reputation for an agent, computing if not cached.
#[hdk_extern]
pub fn get_composite_reputation(agent_b64: String) -> ExternResult<AggregatedReputation> {
    // 1. Calculate recursive score
    let raw_score = calculate_recursive_reputation(agent_b64.clone())?;
    
    // 2. Normalize using Sigmoid
    let _normalized_score = apply_sigmoid_normalization(raw_score);
    
    // Existing static implementation (for transition):
    let agent_anchor = ensure_anchor(&format!("agent_rep:{}", agent_b64))?;
    let links = get_links(
        LinkQuery::try_new(agent_anchor.clone(), LinkTypes::AgentToReputation)?,
        GetStrategy::Local,
    )?;

    if let Some(link) = links.last() {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            let rep: AggregatedReputation = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or(wasm_error!(WasmErrorInner::Guest("No entry".into())))?;
            return Ok(rep);
        }
    }
    
    // No cached reputation — compute fresh
    refresh_reputation(agent_b64)
}

/// Force-refresh the composite reputation for an agent.
#[hdk_extern]
pub fn refresh_reputation(agent_b64: String) -> ExternResult<AggregatedReputation> {
    let now = sys_time()?;

    // 1. Get web-of-trust score (local call)
    let wot_score = get_local_trust_score(&agent_b64).unwrap_or(0.0);

    // 2. Get MYCEL score (cross-cluster to finance, best-effort)
    let mycel_score = get_cross_cluster_mycel(&agent_b64).unwrap_or(0.0);

    // 3. Get domain scores from reports
    let domain_scores = get_domain_scores(&agent_b64)?;
    let domain_avg = if domain_scores.is_empty() {
        0.0
    } else {
        let sum: f64 = domain_scores.iter().map(|(_, s)| s).sum();
        sum / domain_scores.len() as f64
    };

    // 4. Compute composite
    let composite =
        (WEIGHT_WOT * wot_score + WEIGHT_MYCEL * mycel_score + WEIGHT_DOMAIN * domain_avg)
            .clamp(0.0, 1.0);

    // 5. Check staleness
    let staleness_warning = domain_scores.iter().any(|(_, _)| {
        // Domain scores older than 24h are stale — for now, no timestamp check
        // since we'd need to store when each was last updated
        false
    });

    let rep = AggregatedReputation {
        agent_pubkey_b64: agent_b64.clone(),
        web_of_trust_score: wot_score,
        mycel_score,
        domain_scores,
        composite,
        computed_at: now,
        staleness_warning,
    };

    // Store
    let action_hash = create_entry(&EntryTypes::AggregatedReputation(rep.clone()))?;
    let agent_anchor = ensure_anchor(&format!("agent_rep:{}", agent_b64))?;
    create_link(
        agent_anchor,
        action_hash.clone(),
        LinkTypes::AgentToReputation,
        (),
    )?;

    let all_anchor = ensure_anchor("all_reputations")?;
    create_link(all_anchor, action_hash, LinkTypes::AllReputations, ())?;

    Ok(rep)
}

/// Get reputation history for an agent (paginated).
#[hdk_extern]
pub fn get_reputation_history(input: PaginatedAgentInput) -> ExternResult<Vec<Record>> {
    let agent_anchor = ensure_anchor(&format!("agent_rep:{}", input.agent_pubkey_b64))?;
    let links = get_links(
        LinkQuery::try_new(agent_anchor, LinkTypes::AgentToReputation)?,
        GetStrategy::Local,
    )?;

    let limit = input.limit.unwrap_or(20).min(100);
    let mut records = Vec::new();
    for link in links.iter().rev().take(limit) {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

/// Report a domain-specific score from another cluster.
///
/// Called by other cluster bridges (via cross-cluster dispatch) to push
/// their domain-specific reputation assessment for an agent.
#[hdk_extern]
pub fn report_domain_score(input: DomainScoreInput) -> ExternResult<ActionHash> {
    let score = input.score.clamp(0.0, 1.0);
    if !score.is_finite() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Score must be finite".into()
        )));
    }

    let caller = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let report = DomainScoreReport {
        agent_pubkey_b64: input.agent_pubkey_b64.clone(),
        cluster: input.cluster.clone(),
        score,
        source_timestamp: now,
        reporter_pubkey_b64: format!("{:?}", caller),
    };

    let action_hash = create_entry(&EntryTypes::DomainScoreReport(report))?;

    // Link agent → domain score
    let agent_anchor = ensure_anchor(&format!("domain_scores:{}", input.agent_pubkey_b64))?;
    create_link(
        agent_anchor,
        action_hash.clone(),
        LinkTypes::AgentToDomainScore,
        (),
    )?;

    // Link cluster → domain score
    let cluster_anchor = ensure_anchor(&format!("cluster_scores:{}", input.cluster))?;
    create_link(
        cluster_anchor,
        action_hash.clone(),
        LinkTypes::ClusterToDomainScore,
        (),
    )?;

    Ok(action_hash)
}

/// Get an enriched consciousness profile with aggregated reputation.
///
/// Takes the existing consciousness profile dimensions and replaces
/// the `reputation` field with the cross-cluster composite score.
#[hdk_extern]
pub fn get_consciousness_profile_enriched(agent_b64: String) -> ExternResult<ConsciousnessProfile> {
    let rep = get_composite_reputation(agent_b64)?;

    // Build enriched profile using the composite as the reputation dimension
    // The other dimensions (identity, community, engagement) should be provided
    // by the caller or fetched from the credential system
    Ok(ConsciousnessProfile {
        identity: 0.5, // Default — caller should override with actual MFA level
        reputation: rep.composite,
        community: rep.web_of_trust_score.clamp(0.0, 1.0),
        engagement: rep.domain_scores.len().min(10) as f64 / 10.0, // More domains = more engaged
    })
}

/// Get aggregator metrics.
#[hdk_extern]
pub fn get_aggregator_metrics(_: ()) -> ExternResult<AggregatorMetrics> {
    let all_anchor = ensure_anchor("all_reputations")?;
    let rep_links = get_links(
        LinkQuery::try_new(all_anchor, LinkTypes::AllReputations)?,
        GetStrategy::Local,
    )?;

    Ok(AggregatorMetrics {
        total_reputations: rep_links.len() as u32,
        total_domain_reports: 0, // Would need a global counter
        stale_count: 0,
    })
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Get trust score from local web-of-trust zome.
fn get_local_trust_score(agent_b64: &str) -> Result<f64, ()> {
    // Call web-of-trust zome locally
    match call(
        CallTargetCell::Local,
        ZomeName::from("web_of_trust"),
        FunctionName::from("get_trust_score"),
        None,
        agent_b64.to_string(),
    ) {
        Ok(ZomeCallResponse::Ok(io)) => {
            io.decode::<f64>().ok().map(|s| s.clamp(0.0, 1.0)).ok_or(())
        }
        _ => Err(()),
    }
}

/// Get MYCEL score from finance/recognition (cross-cluster, best-effort).
fn get_cross_cluster_mycel(agent_b64: &str) -> Result<f64, ()> {
    match call(
        CallTargetCell::OtherRole("finance".into()),
        ZomeName::from("recognition"),
        FunctionName::from("get_mycel_score"),
        None,
        agent_b64.to_string(),
    ) {
        Ok(ZomeCallResponse::Ok(io)) => {
            io.decode::<f64>().ok().map(|s| s.clamp(0.0, 1.0)).ok_or(())
        }
        _ => Err(()),
    }
}

/// Get all domain score reports for an agent.
fn get_domain_scores(agent_b64: &str) -> ExternResult<Vec<(String, f64)>> {
    let agent_anchor = ensure_anchor(&format!("domain_scores:{}", agent_b64))?;
    let links = get_links(
        LinkQuery::try_new(agent_anchor, LinkTypes::AgentToDomainScore)?,
        GetStrategy::Local,
    )?;

    let mut scores: std::collections::HashMap<String, f64> = std::collections::HashMap::new();

    // Take the most recent score per cluster
    for link in links.iter().rev() {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            if let Ok(Some(report)) = record.entry().to_app_option::<DomainScoreReport>() {
                scores.entry(report.cluster).or_insert(report.score);
            }
        }
    }

    Ok(scores.into_iter().collect())
}
