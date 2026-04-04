#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Data Sharing Preferences Coordinator — CRUD for per-user flow controls.
//!
//! The bridge dispatch layer queries these preferences before routing
//! cross-cluster calls. If a user has blocked a flow, the bridge returns
//! BRG-011 (UserBlocked) instead of executing the call.

use hdk::prelude::*;
use data_preferences_integrity::*;

/// Set a data sharing preference for a cluster pair.
///
/// Creates or updates the preference. Logs the change for audit.
#[hdk_extern]
pub fn set_preference(pref: DataSharingPreference) -> ExternResult<ActionHash> {
    let agent = agent_info()?.agent_latest_pubkey;

    // Check if a preference already exists for this pair
    let existing = get_preference_for_pair(&pref.source_cluster, &pref.target_cluster)?;

    let action_hash = create_entry(EntryTypes::DataSharingPreference(pref.clone()))?;

    // Link from agent
    create_link(
        agent.clone(),
        action_hash.clone(),
        LinkTypes::AgentToPreferences,
        LinkTag::new(format!("{}→{}", pref.source_cluster, pref.target_cluster)),
    )?;

    // Log the change
    let was_allowed = existing.map(|e| e.allowed).unwrap_or(true);
    if was_allowed != pref.allowed {
        let log = PreferenceChangeLog {
            source_cluster: pref.source_cluster,
            target_cluster: pref.target_cluster,
            was_allowed,
            now_allowed: pref.allowed,
            changed_at: pref.updated_at,
        };
        let log_hash = create_entry(EntryTypes::PreferenceChangeLog(log))?;
        create_link(agent, log_hash, LinkTypes::AgentToChangeLog, ())?;
    }

    Ok(action_hash)
}

/// Get all data sharing preferences for the current agent.
#[hdk_extern]
pub fn get_my_preferences(_: ()) -> ExternResult<Vec<DataSharingPreference>> {
    let agent = agent_info()?.agent_latest_pubkey;
    let links = get_links(
        GetLinksInputBuilder::try_new(agent, LinkTypes::AgentToPreferences)?.build(),
    )?;

    let mut prefs = Vec::new();
    for link in links {
        let hash: ActionHash = link.target.into_action_hash()
            .ok_or(wasm_error!("Invalid link target"))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            if let Some(p) = record.entry().to_app_option::<DataSharingPreference>()
                .map_err(|e| wasm_error!("Deserialize: {}", e))? {
                prefs.push(p);
            }
        }
    }

    Ok(prefs)
}

/// Check if a specific flow is allowed for the current agent.
///
/// Returns true if no preference exists (default: allow all).
/// This is the function the bridge dispatch layer calls.
#[hdk_extern]
pub fn is_flow_allowed(input: FlowCheckInput) -> ExternResult<bool> {
    let pref = get_preference_for_pair(&input.source_cluster, &input.target_cluster)?;
    match pref {
        Some(p) => {
            if !p.allowed {
                return Ok(false);
            }
            // Check per-zome blocks
            if !p.blocked_zomes.is_empty() && p.blocked_zomes.contains(&input.zome_name) {
                return Ok(false);
            }
            Ok(true)
        }
        None => Ok(true), // No preference = allow
    }
}

/// Get the preference change log for the current agent.
#[hdk_extern]
pub fn get_change_log(_: ()) -> ExternResult<Vec<PreferenceChangeLog>> {
    let agent = agent_info()?.agent_latest_pubkey;
    let links = get_links(
        GetLinksInputBuilder::try_new(agent, LinkTypes::AgentToChangeLog)?.build(),
    )?;

    let mut logs = Vec::new();
    for link in links {
        let hash: ActionHash = link.target.into_action_hash()
            .ok_or(wasm_error!("Invalid link target"))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            if let Some(log) = record.entry().to_app_option::<PreferenceChangeLog>()
                .map_err(|e| wasm_error!("Deserialize: {}", e))? {
                logs.push(log);
            }
        }
    }

    Ok(logs)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FlowCheckInput {
    pub source_cluster: String,
    pub target_cluster: String,
    pub zome_name: String,
}

/// Internal: find existing preference for a cluster pair.
fn get_preference_for_pair(source: &str, target: &str) -> ExternResult<Option<DataSharingPreference>> {
    let agent = agent_info()?.agent_latest_pubkey;
    let tag = LinkTag::new(format!("{source}→{target}"));
    let links = get_links(
        GetLinksInputBuilder::try_new(agent, LinkTypes::AgentToPreferences)?
            .tag_prefix(tag)
            .build(),
    )?;

    // Return the most recent preference (last link)
    if let Some(link) = links.last() {
        let hash: ActionHash = link.target.clone().into_action_hash()
            .ok_or(wasm_error!("Invalid link target"))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            return record.entry().to_app_option::<DataSharingPreference>()
                .map_err(|e| wasm_error!("Deserialize: {}", e));
        }
    }

    Ok(None)
}
