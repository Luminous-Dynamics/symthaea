// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reputation Tracker Zome for ZeroTrustML
//!
//! Tracks peer reputation scores over time for Byzantine resistance

use hdk::prelude::*;
use serde::{Deserialize, Serialize};
use hdk::hash_path::anchor::anchor;

/// Reputation entry for a node
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ReputationEntry {
    /// Node ID being tracked
    pub node_id: u32,

    /// Current reputation score (0.0 - 1.0)
    pub reputation_score: f32,

    /// Number of gradients contributed
    pub gradients_contributed: u32,

    /// Number of gradients that passed validation
    pub gradients_validated: u32,

    /// Number of gradients rejected
    pub gradients_rejected: u32,

    /// Whether node is blacklisted
    pub is_blacklisted: bool,

    /// Reason for blacklist (if applicable)
    pub blacklist_reason: Option<String>,

    /// Timestamp of last update
    pub last_updated: Timestamp,
}

/// Input for updating reputation
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateReputationInput {
    pub node_id: u32,
    pub reputation_score: f32,
    pub gradient_validated: bool,
    pub blacklist: Option<String>,  // Some(reason) to blacklist, None to unblacklist
}

/// Entry types for the zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    ReputationEntry(ReputationEntry),
}

/// Link types for the zome
#[hdk_link_types]
pub enum LinkTypes {
    NodeToReputation,
    BlacklistedNodes,
}

/// Initialize the zome
#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    Ok(InitCallbackResult::Pass)
}

/// Update or create reputation for a node
#[hdk_extern]
pub fn update_reputation(input: UpdateReputationInput) -> ExternResult<ActionHash> {
    // Get existing reputation if it exists
    let existing_reputation = get_reputation_for_node(input.node_id)?;

    let reputation_entry = match existing_reputation {
        Some(mut existing) => {
            // Update existing reputation
            existing.reputation_score = input.reputation_score;
            existing.gradients_contributed += 1;

            if input.gradient_validated {
                existing.gradients_validated += 1;
            } else {
                existing.gradients_rejected += 1;
            }

            // Handle blacklisting
            if let Some(reason) = input.blacklist {
                existing.is_blacklisted = true;
                existing.blacklist_reason = Some(reason);
            }

            existing.last_updated = sys_time()?;
            existing
        }
        None => {
            // Create new reputation entry
            ReputationEntry {
                node_id: input.node_id,
                reputation_score: input.reputation_score,
                gradients_contributed: 1,
                gradients_validated: if input.gradient_validated { 1 } else { 0 },
                gradients_rejected: if input.gradient_validated { 0 } else { 1 },
                is_blacklisted: input.blacklist.is_some(),
                blacklist_reason: input.blacklist,
                last_updated: sys_time()?,
            }
        }
    };

    // Store in DHT
    let action_hash = create_entry(EntryTypes::ReputationEntry(reputation_entry.clone()))?;
    let entry_hash = hash_entry(&reputation_entry)?;

    // Create link from node ID
    let node_anchor = create_or_get_anchor(format!("node_{}", input.node_id))?;
    create_link(
        node_anchor,
        entry_hash.clone(),
        LinkTypes::NodeToReputation,
        (),
    )?;

    // If blacklisted, add to blacklist
    if reputation_entry.is_blacklisted {
        let blacklist_anchor = create_or_get_anchor("blacklisted".to_string())?;
        create_link(
            blacklist_anchor,
            entry_hash,
            LinkTypes::BlacklistedNodes,
            (),
        )?;
    }

    Ok(action_hash)
}

/// Get reputation for a specific node
#[hdk_extern]
pub fn get_reputation(node_id: u32) -> ExternResult<Option<ReputationEntry>> {
    get_reputation_for_node(node_id)
}

/// Get all blacklisted nodes
#[hdk_extern]
pub fn get_blacklisted_nodes(_: ()) -> ExternResult<Vec<ReputationEntry>> {
    let blacklist_anchor = create_or_get_anchor("blacklisted".to_string())?;
    let links = get_links(GetLinksInputBuilder::try_new(blacklist_anchor, LinkTypes::BlacklistedNodes)?.build())?;

    let mut blacklisted = Vec::new();
    for link in links {
        if let Some(entry_hash) = link.target.into_entry_hash() {
            if let Some(record) = get(entry_hash, GetOptions::default())? {
                if let Some(reputation) = record.entry().to_app_option()
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))? {
                    blacklisted.push(reputation);
                }
            }
        }
    }

    Ok(blacklisted)
}

/// Get reputation statistics
#[hdk_extern]
pub fn get_reputation_statistics(_: ()) -> ExternResult<ReputationStatistics> {
    let blacklisted = get_blacklisted_nodes(())?;

    // In a real implementation, you'd iterate through all reputation entries
    // For now, return basic stats
    Ok(ReputationStatistics {
        total_nodes_tracked: 0,  // Would need to count all nodes
        blacklisted_nodes: blacklisted.len(),
        average_reputation: 0.7,  // Would need to calculate average
    })
}

/// Reputation statistics
#[derive(Serialize, Deserialize, Debug)]
pub struct ReputationStatistics {
    pub total_nodes_tracked: usize,
    pub blacklisted_nodes: usize,
    pub average_reputation: f32,
}

/// Helper to get reputation for a node
fn get_reputation_for_node(node_id: u32) -> ExternResult<Option<ReputationEntry>> {
    let node_anchor = create_or_get_anchor(format!("node_{}", node_id))?;
    let links = get_links(GetLinksInputBuilder::try_new(node_anchor, LinkTypes::NodeToReputation)?.build())?;

    // Get most recent reputation entry
    for link in links.into_iter().rev() {
        if let Some(entry_hash) = link.target.into_entry_hash() {
            if let Some(record) = get(entry_hash, GetOptions::default())? {
                if let Some(reputation) = record.entry().to_app_option()
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))? {
                    return Ok(Some(reputation));
                }
            }
        }
    }

    Ok(None)
}

/// Helper to create or get an anchor
fn create_or_get_anchor(anchor_text: String) -> ExternResult<EntryHash> {
    // Use anchor API to create anchor (HDK 0.4+)
    Ok(anchor(LinkTypes::NodeToReputation, "node".to_string(), anchor_text)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reputation_entry_creation() {
        let reputation = ReputationEntry {
            node_id: 1,
            reputation_score: 0.8,
            gradients_contributed: 10,
            gradients_validated: 9,
            gradients_rejected: 1,
            is_blacklisted: false,
            blacklist_reason: None,
            last_updated: Timestamp::now(),
        };

        assert_eq!(reputation.node_id, 1);
        assert!(!reputation.is_blacklisted);
        assert_eq!(reputation.gradients_contributed, 10);
    }
}