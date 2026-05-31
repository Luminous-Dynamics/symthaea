// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Justice Bridge Integrity Zome
//!
//! Entry types for cross-hApp dispute resolution and enforcement.

use hdi::prelude::*;

/// Dispute submitted from another hApp
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CrossHappDispute {
    pub id: String,
    pub source_happ: String,
    pub complainant_did: String,
    pub respondent_did: String,
    pub dispute_type: DisputeType,
    pub description: String,
    pub evidence_refs: Vec<String>,
    pub status: DisputeStatus,
    pub created_at: Timestamp,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DisputeType {
    ContractBreach,
    Fraud,
    PropertyDispute,
    Defamation,
    ServiceFailure,
    PaymentDispute,
    AccessViolation,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DisputeStatus {
    Filed,
    Mediation,
    Arbitration,
    Appeal,
    Resolved,
    Dismissed,
}

/// Enforcement action to be executed on another hApp
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EnforcementAction {
    pub id: String,
    pub dispute_id: String,
    pub target_happ: String,
    pub target_did: String,
    pub action_type: EnforcementType,
    pub parameters: String,
    pub status: EnforcementStatus,
    pub ordered_at: Timestamp,
    pub executed_at: Option<Timestamp>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EnforcementType {
    ReputationPenalty(f64),
    TemporarySuspension { days: u32 },
    PermanentBan,
    AssetFreeze,
    PaymentOrder { amount: u64, currency: String },
    AccessRestriction,
    PublicNotice,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EnforcementStatus {
    Ordered,
    Pending,
    InProgress,
    Executed,
    Failed,
    Appealed,
    Reversed,
}

/// Justice event for broadcasting
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct JusticeBridgeEvent {
    pub id: String,
    pub event_type: JusticeEventType,
    pub dispute_id: Option<String>,
    pub subject_did: String,
    pub payload: String,
    pub source_happ: String,
    pub timestamp: Timestamp,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JusticeEventType {
    DisputeFiled,
    MediationStarted,
    ArbitrationStarted,
    JudgmentIssued,
    EnforcementOrdered,
    EnforcementExecuted,
    AppealFiled,
    DisputeResolved,
}

/// Query about a DID's dispute history
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct DisputeHistoryQuery {
    pub id: String,
    pub did: String,
    pub source_happ: String,
    pub queried_at: Timestamp,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    CrossHappDispute(CrossHappDispute),
    EnforcementAction(EnforcementAction),
    JusticeBridgeEvent(JusticeBridgeEvent),
    DisputeHistoryQuery(DisputeHistoryQuery),
}

#[hdk_link_types]
pub enum LinkTypes {
    ActiveDisputes,
    DidToDisputes,
    HappToEnforcements,
    RecentEvents,
}

/// Validation callback for entry operations
/// Op::StoreEntry contains both the action and entry
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry {
                app_entry,
                action: _,
            } => validate_create_entry(app_entry),
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => validate_create_entry(app_entry),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(store_record) => match store_record {
            OpRecord::CreateEntry {
                app_entry,
                action: _,
            } => validate_create_entry(app_entry),
            OpRecord::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => validate_create_entry(app_entry),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity { .. } => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_entry(app_entry: EntryTypes) -> ExternResult<ValidateCallbackResult> {
    match app_entry {
        EntryTypes::CrossHappDispute(dispute) => {
            if !dispute.complainant_did.starts_with("did:mycelix:") {
                return Ok(ValidateCallbackResult::Invalid(
                    "Complainant must have valid DID".into(),
                ));
            }
            if !dispute.respondent_did.starts_with("did:mycelix:") {
                return Ok(ValidateCallbackResult::Invalid(
                    "Respondent must have valid DID".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::EnforcementAction(action) => {
            if action.target_happ.is_empty() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Target hApp required".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::JusticeBridgeEvent(event) => {
            if event.source_happ.is_empty() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Source hApp required".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        EntryTypes::DisputeHistoryQuery(query) => {
            if !query.did.starts_with("did:mycelix:") {
                return Ok(ValidateCallbackResult::Invalid("DID must be valid".into()));
            }
            Ok(ValidateCallbackResult::Valid)
        }
    }
}
