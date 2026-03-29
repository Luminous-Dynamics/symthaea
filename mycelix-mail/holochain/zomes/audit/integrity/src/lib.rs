// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Audit Integrity Zome
//!
//! Tamper-evident audit log entries.

use hdi::prelude::*;

/// Audit log entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AuditEntry {
    pub id: String,
    pub timestamp: u64,
    pub category: AuditCategory,
    pub action: String,
    pub severity: AuditSeverity,
    pub actor: AuditActor,
    pub target: Option<AuditTarget>,
    pub details: AuditDetails,
    pub metadata: AuditMetadata,
    pub signature: Option<Vec<u8>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum AuditCategory {
    Authentication,
    Email,
    Contact,
    Trust,
    Sync,
    Settings,
    Security,
    Admin,
    System,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum AuditSeverity {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AuditActor {
    pub agent_pub_key: Option<AgentPubKey>,
    pub email: Option<String>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub session_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AuditTarget {
    pub target_type: String,
    pub id: Option<String>,
    pub hash: Option<ActionHash>,
    pub description: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AuditDetails {
    pub message: String,
    pub before: Option<String>,  // JSON string
    pub after: Option<String>,   // JSON string
    pub error: Option<String>,
    pub stack_trace: Option<String>,
    pub custom: Option<String>,  // JSON string
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AuditMetadata {
    pub source: AuditSource,
    pub version: String,
    pub correlation_id: Option<String>,
    pub parent_id: Option<String>,
    pub tags: Option<Vec<String>>,
    pub retention: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum AuditSource {
    Client,
    Server,
    System,
    External,
}

/// Audit summary for reporting
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AuditSummary {
    pub period_start: u64,
    pub period_end: u64,
    pub total_entries: u32,
    pub by_category: Vec<(String, u32)>,
    pub by_severity: Vec<(String, u32)>,
    pub unique_actors: u32,
    pub generated_at: u64,
    pub hash: String,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    AuditEntry(AuditEntry),
    AuditSummary(AuditSummary),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllAuditEntries,
    EntriesByCategory,
    EntriesBySeverity,
    EntriesByActor,
    EntriesByCorrelation,
    AuditSummaries,
}

/// Validate audit entry - entries are append-only
fn validate_create_audit_entry(
    _action: Create,
    entry: AuditEntry,
) -> ExternResult<ValidateCallbackResult> {
    // Validate required fields
    if entry.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Audit entry ID cannot be empty".to_string(),
        ));
    }

    if entry.details.message.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Audit message cannot be empty".to_string(),
        ));
    }

    if entry.action.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Audit action cannot be empty".to_string(),
        ));
    }

    // Note: Time-based validation moved to coordinator zome since sys_time
    // is not available in integrity zomes

    Ok(ValidateCallbackResult::Valid)
}


/// Main validation dispatcher
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::AuditEntry(entry) => validate_create_audit_entry(action, entry),
                EntryTypes::AuditSummary(_) => Ok(ValidateCallbackResult::Valid),
            },
            OpEntry::UpdateEntry { app_entry, .. } => match app_entry {
                EntryTypes::AuditEntry(_) => Ok(ValidateCallbackResult::Invalid(
                    "Audit entries cannot be updated".to_string(),
                )),
                EntryTypes::AuditSummary(_) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}
