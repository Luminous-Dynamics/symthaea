// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Triage Integrity Zome
//! Mass casualty triage records using START protocol

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// A triage assessment record
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TriageRecord {
    pub disaster_hash: ActionHash,
    pub patient_id: String,
    pub patient_hash: Option<ActionHash>,
    pub category: TriageCategory,
    pub injuries: String,
    pub location: String,
    pub timestamp: Timestamp,
    pub triaged_by: AgentPubKey,
    pub transport_priority: TransportPriority,
    pub notes: String,
}

/// START triage categories
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TriageCategory {
    /// Immediate - life-threatening, salvageable
    Immediate,
    /// Delayed - serious but can wait
    Delayed,
    /// Minor - walking wounded
    Minor,
    /// Expectant - unlikely to survive
    Expectant,
    /// Deceased
    Dead,
}

/// Transport priority
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TransportPriority {
    Urgent,
    Priority,
    Routine,
    None,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    TriageRecord(TriageRecord),
}

#[hdk_link_types]
pub enum LinkTypes {
    DisasterToTriage,
    CategoryToTriage,
    PatientToTriage,
    AgentToTriage,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::TriageRecord(record) => validate_create_triage(action, record),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::TriageRecord(record) => validate_update_triage(action, record),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::DisasterToTriage => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CategoryToTriage => Ok(ValidateCallbackResult::Valid),
            LinkTypes::PatientToTriage => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToTriage => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::CategoryToTriage => Ok(ValidateCallbackResult::Valid),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_triage(
    _action: Create,
    record: TriageRecord,
) -> ExternResult<ValidateCallbackResult> {
    if record.patient_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Patient ID cannot be empty".into(),
        ));
    }
    if record.location.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Location cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_triage(
    _action: Update,
    record: TriageRecord,
) -> ExternResult<ValidateCallbackResult> {
    if record.patient_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Patient ID cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
