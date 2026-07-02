// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Civitas Reputation Coordinator Zome
//!
//! Business logic for the Civitas reputation system on Holochain 0.6.
//! This zome provides functions to update and query agent reputation scores.

use hdk::prelude::*;
use civitas_reputation_integrity::{CivitasReputationScore, EntryTypes, LinkTypes};

/// Helper function to get the link type filter
fn reputation_link_filter() -> LinkTypeFilter {
    LinkTypeFilter::single_type(0.into(), (LinkTypes::AgentToReputationScore as u8).into())
}

/// Helper function to ensure a path exists and return its entry hash
fn ensure_path(path: Path, link_type: LinkTypes) -> ExternResult<EntryHash> {
    let typed = path.typed(link_type)?;
    typed.ensure()?;
    typed.path_entry_hash()
}

/// Input for the update_causal_reputation function.
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateCausalReputationInput {
    pub agent: AgentPubKey,
    pub causal_contribution_score: f64,
}

/// Updates an agent's causal reputation score.
/// This function is called by the causal_contribution zome when a new
/// contribution is recorded.
#[hdk_extern]
pub fn update_causal_reputation(input: UpdateCausalReputationInput) -> ExternResult<ActionHash> {
    let now = sys_time()?;

    // Get the current reputation score, or create a new one
    let agent_path = Path::from(format!("agents/{}", input.agent));
    let agent_entry_hash = ensure_path(agent_path, LinkTypes::AgentToReputationScore)?;

    let links = get_links(
        LinkQuery::new(agent_entry_hash.clone(), reputation_link_filter()),
        GetStrategy::default(),
    )?;

    let (mut current_score, maybe_action_hash) = if links.is_empty() {
        // Initialize new reputation
        (CivitasReputationScore {
            agent: input.agent.clone(),
            reputation: 0.5, // Initial neutral reputation
            rounds_participated: 0,
            last_updated: now,
        }, None)
    } else {
        // Get the latest reputation entry
        let latest_link = links.into_iter()
            .max_by_key(|l| l.timestamp)
            .ok_or(wasm_error!(WasmErrorInner::Guest("No links found".into())))?;

        let target_hash = ActionHash::try_from(latest_link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid action hash".into())))?;

        let record = get(target_hash.clone(), GetOptions::default())?
            .ok_or(wasm_error!(WasmErrorInner::Guest("Record not found".into())))?;

        let score: CivitasReputationScore = record
            .entry()
            .to_app_option()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid entry".into())))?;

        (score, Some(target_hash))
    };

    // Normalize the causal contribution score to [0, 1] using sigmoid
    let normalized_score = 1.0 / (1.0 + (-input.causal_contribution_score).exp());

    // Apply exponential moving average update
    let alpha = 0.1;
    current_score.reputation = (1.0 - alpha) * current_score.reputation + alpha * normalized_score;
    current_score.rounds_participated += 1;
    current_score.last_updated = now;

    // Create or update the reputation entry
    let action_hash = if let Some(prev_hash) = maybe_action_hash {
        update_entry(prev_hash, &current_score)?
    } else {
        let hash = create_entry(&EntryTypes::CivitasReputationScore(current_score.clone()))?;
        // Create link from agent path to entry
        create_link(
            agent_entry_hash,
            hash.clone(),
            LinkTypes::AgentToReputationScore,
            (),
        )?;
        hash
    };

    Ok(action_hash)
}

/// Gets an agent's causal reputation score.
#[hdk_extern]
pub fn get_causal_reputation(agent: AgentPubKey) -> ExternResult<Option<CivitasReputationScore>> {
    let agent_path = Path::from(format!("agents/{}", agent));
    let agent_entry_hash = ensure_path(agent_path, LinkTypes::AgentToReputationScore)?;

    let links = get_links(
        LinkQuery::new(agent_entry_hash, reputation_link_filter()),
        GetStrategy::default(),
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    let latest_link = links.into_iter()
        .max_by_key(|l| l.timestamp)
        .ok_or(wasm_error!(WasmErrorInner::Guest("No links found".into())))?;

    let target_hash = ActionHash::try_from(latest_link.target)
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid action hash".into())))?;

    let record = get(target_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Record not found".into())))?;

    let score: CivitasReputationScore = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid entry".into())))?;

    Ok(Some(score))
}

/// Gets reputation scores for multiple agents (batch query)
#[hdk_extern]
pub fn get_reputations_batch(agents: Vec<AgentPubKey>) -> ExternResult<Vec<(AgentPubKey, Option<CivitasReputationScore>)>> {
    let mut results = Vec::with_capacity(agents.len());

    for agent in agents {
        let score = get_causal_reputation(agent.clone())?;
        results.push((agent, score));
    }

    Ok(results)
}

/// Calculates the trust threshold for an agent based on their reputation
#[hdk_extern]
pub fn get_trust_threshold(agent: AgentPubKey) -> ExternResult<f64> {
    match get_causal_reputation(agent)? {
        Some(score) => {
            // High reputation = high trust threshold
            // Low reputation = low trust threshold (more scrutiny)
            let base_threshold = 0.5;
            let reputation_factor = score.reputation * 0.4; // Max 0.4 bonus
            let rounds_factor = (score.rounds_participated as f64 / 100.0).min(0.1); // Max 0.1 bonus

            Ok(base_threshold + reputation_factor + rounds_factor)
        }
        None => Ok(0.5), // Default threshold for new agents
    }
}
