// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Incidents Coordinator Zome
//! Business logic for disaster declaration and lifecycle management

use hdk::prelude::*;
use incidents_integrity::*;

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Declare a new disaster
#[hdk_extern]
pub fn declare_disaster(input: DeclareDisasterInput) -> ExternResult<Record> {
    if input.title.is_empty() || input.title.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Title must be 1-256 characters".into()
        )));
    }
    if input.description.is_empty() || input.description.len() > 8192 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Description must be 1-8192 characters".into()
        )));
    }
    if input.affected_area.radius_km <= 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Affected area radius must be positive".into()
        )));
    }

    let agent_info = agent_info()?;
    let now = sys_time()?;

    let disaster = Disaster {
        id: input.id.clone(),
        disaster_type: input.disaster_type.clone(),
        title: input.title,
        description: input.description,
        severity: input.severity,
        declared_by: agent_info.agent_initial_pubkey.clone(),
        declared_at: now,
        affected_area: input.affected_area,
        status: DisasterStatus::Declared,
        estimated_affected: input.estimated_affected,
        coordination_lead: input.coordination_lead,
    };

    let action_hash = create_entry(&EntryTypes::Disaster(disaster))?;

    // Link to all disasters anchor
    create_entry(&EntryTypes::Anchor(Anchor("all_disasters".to_string())))?;
    create_link(
        anchor_hash("all_disasters")?,
        action_hash.clone(),
        LinkTypes::AllDisasters,
        (),
    )?;

    // Link to active disasters anchor
    create_entry(&EntryTypes::Anchor(Anchor("active_disasters".to_string())))?;
    create_link(
        anchor_hash("active_disasters")?,
        action_hash.clone(),
        LinkTypes::ActiveDisasters,
        (),
    )?;

    // Link by disaster type
    let type_anchor = format!("disaster_type:{:?}", input.disaster_type);
    create_entry(&EntryTypes::Anchor(Anchor(type_anchor.clone())))?;
    create_link(
        anchor_hash(&type_anchor)?,
        action_hash.clone(),
        LinkTypes::DisasterByType,
        (),
    )?;

    // Link agent to disaster
    create_link(
        agent_info.agent_initial_pubkey,
        action_hash.clone(),
        LinkTypes::AgentToDisaster,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created disaster".into()
    )))
}

/// Input for declaring a disaster
#[derive(Serialize, Deserialize, Debug)]
pub struct DeclareDisasterInput {
    pub id: String,
    pub disaster_type: DisasterType,
    pub title: String,
    pub description: String,
    pub severity: SeverityLevel,
    pub affected_area: AffectedArea,
    pub estimated_affected: u32,
    pub coordination_lead: Option<AgentPubKey>,
}

/// Get all active disasters
#[hdk_extern]
pub fn get_active_disasters(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("active_disasters")?, LinkTypes::ActiveDisasters)?,
        GetStrategy::default(),
    )?;

    let mut disasters = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            disasters.push(record);
        }
    }

    Ok(disasters)
}

/// Update a disaster's status
#[hdk_extern]
pub fn update_disaster_status(input: UpdateDisasterStatusInput) -> ExternResult<Record> {
    let current_record = get(input.disaster_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Disaster not found".into())),
    )?;

    let current_disaster: Disaster = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid disaster entry".into()
        )))?;

    let updated_disaster = Disaster {
        status: input.new_status.clone(),
        ..current_disaster
    };

    let new_action_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::Disaster(updated_disaster),
    )?;

    // If closing or in recovery, remove from active disasters
    if matches!(
        input.new_status,
        DisasterStatus::Closed | DisasterStatus::Recovery
    ) {
        let links = get_links(
            LinkQuery::try_new(anchor_hash("active_disasters")?, LinkTypes::ActiveDisasters)?,
            GetStrategy::default(),
        )?;
        for link in links {
            let target = ActionHash::try_from(link.target.clone());
            if let Ok(target_hash) = target {
                if target_hash == input.disaster_hash {
                    delete_link(link.create_link_hash, GetOptions::default())?;
                }
            }
        }
    }

    get(new_action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated disaster".into()
    )))
}

/// Input for updating disaster status
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateDisasterStatusInput {
    pub disaster_hash: ActionHash,
    pub new_status: DisasterStatus,
}

/// Add an incident update to a disaster
#[hdk_extern]
pub fn add_incident_update(input: AddIncidentUpdateInput) -> ExternResult<Record> {
    if input.content.is_empty() || input.content.len() > 8192 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Content must be 1-8192 characters".into()
        )));
    }

    let agent_info = agent_info()?;
    let now = sys_time()?;

    let update = IncidentUpdate {
        disaster_hash: input.disaster_hash.clone(),
        author: agent_info.agent_initial_pubkey,
        timestamp: now,
        update_type: input.update_type,
        content: input.content,
    };

    let action_hash = create_entry(&EntryTypes::IncidentUpdate(update))?;

    // Link disaster to update
    create_link(
        input.disaster_hash,
        action_hash.clone(),
        LinkTypes::DisasterToUpdate,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created update".into()
    )))
}

/// Input for adding an incident update
#[derive(Serialize, Deserialize, Debug)]
pub struct AddIncidentUpdateInput {
    pub disaster_hash: ActionHash,
    pub update_type: UpdateType,
    pub content: String,
}

/// End a disaster (set status to Closed)
#[hdk_extern]
pub fn end_disaster(disaster_hash: ActionHash) -> ExternResult<Record> {
    update_disaster_status(UpdateDisasterStatusInput {
        disaster_hash,
        new_status: DisasterStatus::Closed,
    })
}

/// Get the timeline of updates for a disaster
#[hdk_extern]
pub fn get_disaster_timeline(disaster_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(disaster_hash, LinkTypes::DisasterToUpdate)?,
        GetStrategy::default(),
    )?;

    let mut updates = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            updates.push(record);
        }
    }

    updates.sort_by(|a, b| {
        let ts_a = a.action().timestamp();
        let ts_b = b.action().timestamp();
        ts_a.cmp(&ts_b)
    });

    Ok(updates)
}
