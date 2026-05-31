// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Triage Coordinator Zome
//! Mass casualty triage operations using START protocol

use hdk::prelude::*;
use triage_integrity::*;

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Triage a patient
#[hdk_extern]
pub fn triage_patient(input: TriagePatientInput) -> ExternResult<Record> {
    if input.patient_id.is_empty() || input.patient_id.len() > 128 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Patient ID must be 1-128 characters".into()
        )));
    }
    if input.location.is_empty() || input.location.len() > 512 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Location must be 1-512 characters".into()
        )));
    }

    let agent_info = agent_info()?;
    let now = sys_time()?;

    let record = TriageRecord {
        disaster_hash: input.disaster_hash.clone(),
        patient_id: input.patient_id.clone(),
        patient_hash: input.patient_hash,
        category: input.category.clone(),
        injuries: input.injuries,
        location: input.location,
        timestamp: now,
        triaged_by: agent_info.agent_initial_pubkey.clone(),
        transport_priority: input.transport_priority,
        notes: input.notes,
    };

    let action_hash = create_entry(&EntryTypes::TriageRecord(record))?;

    // Link disaster to triage record
    let disaster_anchor = format!("disaster_triage:{}", input.disaster_hash);
    create_entry(&EntryTypes::Anchor(Anchor(disaster_anchor.clone())))?;
    create_link(
        anchor_hash(&disaster_anchor)?,
        action_hash.clone(),
        LinkTypes::DisasterToTriage,
        (),
    )?;

    // Link by triage category
    let category_anchor = format!(
        "triage_category:{}:{:?}",
        input.disaster_hash, input.category
    );
    create_entry(&EntryTypes::Anchor(Anchor(category_anchor.clone())))?;
    create_link(
        anchor_hash(&category_anchor)?,
        action_hash.clone(),
        LinkTypes::CategoryToTriage,
        (),
    )?;

    // Link patient to triage history
    let patient_anchor = format!("patient_triage:{}", input.patient_id);
    create_entry(&EntryTypes::Anchor(Anchor(patient_anchor.clone())))?;
    create_link(
        anchor_hash(&patient_anchor)?,
        action_hash.clone(),
        LinkTypes::PatientToTriage,
        (),
    )?;

    // Link agent to their triage records
    create_link(
        agent_info.agent_initial_pubkey,
        action_hash.clone(),
        LinkTypes::AgentToTriage,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created triage record".into()
    )))
}

/// Input for triaging a patient
#[derive(Serialize, Deserialize, Debug)]
pub struct TriagePatientInput {
    pub disaster_hash: ActionHash,
    pub patient_id: String,
    pub patient_hash: Option<ActionHash>,
    pub category: TriageCategory,
    pub injuries: String,
    pub location: String,
    pub transport_priority: TransportPriority,
    pub notes: String,
}

/// Update a triage assessment (re-triage)
#[hdk_extern]
pub fn update_triage(input: UpdateTriageInput) -> ExternResult<Record> {
    let current_record = get(input.original_triage_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Triage record not found".into())),
    )?;

    let current_triage: TriageRecord = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid triage entry".into()
        )))?;

    let agent_info = agent_info()?;
    let now = sys_time()?;

    let updated_triage = TriageRecord {
        category: input.new_category.clone(),
        injuries: input.injuries.unwrap_or(current_triage.injuries),
        transport_priority: input
            .transport_priority
            .unwrap_or(current_triage.transport_priority),
        notes: input.notes.unwrap_or(current_triage.notes),
        timestamp: now,
        triaged_by: agent_info.agent_initial_pubkey,
        ..current_triage
    };

    let new_action_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::TriageRecord(updated_triage),
    )?;

    // Update category link: remove old, add new
    let old_category_anchor = format!(
        "triage_category:{}:{:?}",
        input.original_disaster_hash, input.old_category
    );
    let old_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&old_category_anchor)?,
            LinkTypes::CategoryToTriage,
        )?,
        GetStrategy::default(),
    )?;
    for link in old_links {
        let target = ActionHash::try_from(link.target.clone());
        if let Ok(target_hash) = target {
            if target_hash == input.original_triage_hash {
                delete_link(link.create_link_hash, GetOptions::default())?;
            }
        }
    }

    let new_category_anchor = format!(
        "triage_category:{}:{:?}",
        input.original_disaster_hash, input.new_category
    );
    create_entry(&EntryTypes::Anchor(Anchor(new_category_anchor.clone())))?;
    create_link(
        anchor_hash(&new_category_anchor)?,
        new_action_hash.clone(),
        LinkTypes::CategoryToTriage,
        (),
    )?;

    get(new_action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated triage record".into()
    )))
}

/// Input for updating a triage record
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateTriageInput {
    pub original_triage_hash: ActionHash,
    pub original_disaster_hash: ActionHash,
    pub old_category: TriageCategory,
    pub new_category: TriageCategory,
    pub injuries: Option<String>,
    pub transport_priority: Option<TransportPriority>,
    pub notes: Option<String>,
}

/// Get all triage records for a disaster
#[hdk_extern]
pub fn get_disaster_triage(disaster_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let disaster_anchor = format!("disaster_triage:{}", disaster_hash);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&disaster_anchor)?, LinkTypes::DisasterToTriage)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }

    records.sort_by(|a, b| a.action().timestamp().cmp(&b.action().timestamp()));
    Ok(records)
}

/// Get triage records by category for a disaster
#[hdk_extern]
pub fn get_triage_by_category(input: TriageByCategoryInput) -> ExternResult<Vec<Record>> {
    let category_anchor = format!(
        "triage_category:{}:{:?}",
        input.disaster_hash, input.category
    );
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&category_anchor)?, LinkTypes::CategoryToTriage)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }

    Ok(records)
}

/// Input for querying triage by category
#[derive(Serialize, Deserialize, Debug)]
pub struct TriageByCategoryInput {
    pub disaster_hash: ActionHash,
    pub category: TriageCategory,
}

/// Get triage history for a specific patient
#[hdk_extern]
pub fn get_patient_triage_history(patient_id: String) -> ExternResult<Vec<Record>> {
    let patient_anchor = format!("patient_triage:{}", patient_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&patient_anchor)?, LinkTypes::PatientToTriage)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }

    records.sort_by(|a, b| a.action().timestamp().cmp(&b.action().timestamp()));
    Ok(records)
}
