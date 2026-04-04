#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Data Sharing Preferences — per-user control over cross-cluster data flows.
//!
//! Users configure which cluster pairs are allowed to exchange data.
//! The bridge dispatch layer (`dispatch_call_cross_cluster`) checks these
//! preferences before routing, returning BRG-011 when a user has blocked a flow.

use hdi::prelude::*;
use serde::{Deserialize, Serialize};

/// A user's preference for a specific cross-cluster data flow.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct DataSharingPreference {
    /// Source cluster ID (e.g., "health").
    pub source_cluster: String,
    /// Target cluster ID (e.g., "identity").
    pub target_cluster: String,
    /// Whether this flow is allowed.
    pub allowed: bool,
    /// Specific zomes blocked within this flow (empty = block all if !allowed).
    pub blocked_zomes: Vec<String>,
    /// Human-readable reason for the preference.
    pub reason: String,
    /// When this preference was last updated.
    pub updated_at: Timestamp,
}

/// A log entry recording when a preference was changed.
#[hdk_entry_helper]
#[derive(Clone)]
pub struct PreferenceChangeLog {
    /// The preference that was changed.
    pub source_cluster: String,
    pub target_cluster: String,
    /// Previous state.
    pub was_allowed: bool,
    /// New state.
    pub now_allowed: bool,
    /// When the change occurred.
    pub changed_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type]
    DataSharingPreference(DataSharingPreference),
    #[entry_type]
    PreferenceChangeLog(PreferenceChangeLog),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Agent → their preferences (for quick lookup)
    AgentToPreferences,
    /// Agent → preference change log
    AgentToChangeLog,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } => match app_entry {
                EntryTypes::DataSharingPreference(pref) => validate_preference(&pref),
                EntryTypes::PreferenceChangeLog(log) => validate_change_log(&log),
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::DataSharingPreference(pref) => validate_preference(&pref),
                EntryTypes::PreferenceChangeLog(log) => validate_change_log(&log),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { tag, .. } => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDeleteLink {
            original_action,
            action,
            ..
        } => {
            if action.author != original_action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original author can delete this link".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original entry author can update their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original entry author can delete their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_preference(pref: &DataSharingPreference) -> ExternResult<ValidateCallbackResult> {
    if pref.source_cluster.trim().is_empty() || pref.target_cluster.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "source_cluster and target_cluster are required".into(),
        ));
    }
    if pref.source_cluster.len() > 64 || pref.target_cluster.len() > 64 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cluster IDs must be 64 characters or fewer".into(),
        ));
    }
    if pref.blocked_zomes.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many blocked zomes (max 128)".into(),
        ));
    }
    for z in &pref.blocked_zomes {
        if z.len() > 128 {
            return Ok(ValidateCallbackResult::Invalid(
                "Blocked zome name too long (max 128 chars)".into(),
            ));
        }
    }
    if pref.reason.len() > 1024 {
        return Ok(ValidateCallbackResult::Invalid(
            "Reason too long (max 1024 chars)".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_change_log(log: &PreferenceChangeLog) -> ExternResult<ValidateCallbackResult> {
    if log.source_cluster.trim().is_empty() || log.target_cluster.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "source_cluster and target_cluster are required".into(),
        ));
    }
    if log.source_cluster.len() > 64 || log.target_cluster.len() > 64 {
        return Ok(ValidateCallbackResult::Invalid(
            "Cluster IDs must be 64 characters or fewer".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
