// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Civitas Reputation Integrity Zome
//!
//! Defines entry types and validation rules for the Civitas reputation system
//! on Holochain 0.6. This zome tracks agent reputation scores based on their
//! causal contributions to the federated learning network.

use hdi::prelude::*;

/// The causal reputation score for an agent in the Civitas system.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CivitasReputationScore {
    /// The agent whose reputation is being tracked.
    pub agent: AgentPubKey,
    /// The agent's causal reputation score, in the range [0, 1].
    pub reputation: f64,
    /// The number of rounds the agent has participated in.
    pub rounds_participated: u64,
    /// The last time the score was updated.
    pub last_updated: Timestamp,
}

/// Entry types for the Civitas reputation system
#[hdk_entry_types]
#[unit_enum(EntryTypesUnit)]
pub enum EntryTypes {
    CivitasReputationScore(CivitasReputationScore),
}

/// Link types for the Civitas reputation system
#[hdk_link_types]
pub enum LinkTypes {
    /// Links from an agent to their reputation score entry
    AgentToReputationScore,
}

// === Validation Functions ===

/// Validation function for CivitasReputationScore entries
pub fn validate_reputation_score(score: CivitasReputationScore) -> ExternResult<ValidateCallbackResult> {
    // Reputation must be in [0, 1]
    if !(0.0..=1.0).contains(&score.reputation) {
        return Ok(ValidateCallbackResult::Invalid(
            "Reputation score must be between 0.0 and 1.0".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Main validation dispatcher
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op {
        Op::StoreEntry(store_entry) => match store_entry.action.hashed.content.entry_type() {
            EntryType::App(app_entry_def) => {
                let entry = store_entry.entry;
                match EntryTypes::deserialize_from_type(
                    app_entry_def.zome_index,
                    app_entry_def.entry_index,
                    &entry,
                )? {
                    Some(EntryTypes::CivitasReputationScore(score)) => validate_reputation_score(score),
                    None => Ok(ValidateCallbackResult::Valid),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
