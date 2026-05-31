// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Capture Coordinator Zome
//! Business logic for water harvesting, storage, and aquifer recharge

use capture_integrity::*;
use hdk::prelude::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
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

// ============================================================================
// HARVEST SYSTEMS
// ============================================================================

/// Register a new water harvesting system
#[hdk_extern]
pub fn register_harvest_system(system: HarvestSystem) -> ExternResult<Record> {
    if system.id.is_empty() || system.id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "System ID must be 1-256 characters".into()
        )));
    }
    if system.name.is_empty() || system.name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "System name must be 1-256 characters".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::HarvestSystem(system.clone()))?;

    // Link to all systems
    create_entry(&EntryTypes::Anchor(Anchor("all_systems".to_string())))?;
    create_link(
        anchor_hash("all_systems")?,
        action_hash.clone(),
        LinkTypes::AllSystems,
        (),
    )?;

    // Link owner to system
    create_link(
        system.owner.clone(),
        action_hash.clone(),
        LinkTypes::OwnerToSystem,
        (),
    )?;

    // Link system type to system
    let type_anchor = format!("harvest_type:{:?}", system.system_type);
    create_entry(&EntryTypes::Anchor(Anchor(type_anchor.clone())))?;
    create_link(
        anchor_hash(&type_anchor)?,
        action_hash.clone(),
        LinkTypes::SystemTypeToSystem,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created harvest system".into()
    )))
}

/// Get all harvest systems owned by the calling agent
#[hdk_extern]
pub fn get_my_systems(_: ()) -> ExternResult<Vec<Record>> {
    let agent_info = agent_info()?;
    let links = get_links(
        LinkQuery::try_new(agent_info.agent_initial_pubkey, LinkTypes::OwnerToSystem)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get all registered harvest systems
#[hdk_extern]
pub fn get_all_systems(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_systems")?, LinkTypes::AllSystems)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// STORAGE TANKS
// ============================================================================

/// Register a new storage tank
#[hdk_extern]
pub fn register_tank(tank: StorageTank) -> ExternResult<Record> {
    if tank.id.is_empty() || tank.id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Tank ID must be 1-256 characters".into()
        )));
    }
    if tank.name.is_empty() || tank.name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Tank name must be 1-256 characters".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::StorageTank(tank.clone()))?;

    // Link to all tanks
    create_entry(&EntryTypes::Anchor(Anchor("all_tanks".to_string())))?;
    create_link(
        anchor_hash("all_tanks")?,
        action_hash.clone(),
        LinkTypes::AllTanks,
        (),
    )?;

    // Link owner to tank
    create_link(
        tank.owner.clone(),
        action_hash.clone(),
        LinkTypes::OwnerToTank,
        (),
    )?;

    // If connected to a system, create that link
    if let Some(system_hash) = tank.connected_system.clone() {
        create_link(
            system_hash,
            action_hash.clone(),
            LinkTypes::SystemToTank,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created tank".into()
    )))
}

/// Update the current water level in a tank
#[hdk_extern]
pub fn update_tank_level(input: UpdateTankLevelInput) -> ExternResult<Record> {
    let agent_info = agent_info()?;
    let record = get(input.tank_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Tank not found".into())))?;
    let mut tank: StorageTank = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid tank entry".into()
        )))?;

    if tank.owner != agent_info.agent_initial_pubkey {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the tank owner can update the level".into()
        )));
    }

    if input.new_level_liters > tank.capacity_liters {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "New level cannot exceed tank capacity".into()
        )));
    }

    tank.current_level_liters = input.new_level_liters;

    let new_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::StorageTank(tank),
    )?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated tank".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateTankLevelInput {
    pub tank_hash: ActionHash,
    pub new_level_liters: u64,
}

// ============================================================================
// HARVEST RECORDS
// ============================================================================

/// Record a water harvest from a system
#[hdk_extern]
pub fn record_harvest(harvest: HarvestRecord) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::HarvestRecord(harvest.clone()))?;

    // Link system to harvest record
    create_link(
        harvest.system_hash.clone(),
        action_hash.clone(),
        LinkTypes::SystemToHarvestRecord,
        (),
    )?;

    // Link credited agent to harvest record
    create_link(
        harvest.credited_to.clone(),
        action_hash.clone(),
        LinkTypes::AgentToHarvestRecord,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created harvest record".into()
    )))
}

/// Get harvest history for a system
#[hdk_extern]
pub fn get_harvest_history(system_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(system_hash, LinkTypes::SystemToHarvestRecord)?,
        GetStrategy::default(),
    )?;
    let mut records = records_from_links(links)?;
    records.sort_by(|a, b| a.action().timestamp().cmp(&b.action().timestamp()));
    Ok(records)
}

// ============================================================================
// AQUIFER RECHARGE
// ============================================================================

/// Register a new aquifer recharge project
#[hdk_extern]
pub fn register_recharge_project(project: RechargeProject) -> ExternResult<Record> {
    if project.id.is_empty() || project.id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Project ID must be 1-256 characters".into()
        )));
    }
    if project.name.is_empty() || project.name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Project name must be 1-256 characters".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::RechargeProject(project))?;

    // Link to all recharge projects
    create_entry(&EntryTypes::Anchor(Anchor(
        "all_recharge_projects".to_string(),
    )))?;
    create_link(
        anchor_hash("all_recharge_projects")?,
        action_hash.clone(),
        LinkTypes::AllRechargeProjects,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created recharge project".into()
    )))
}

/// Get all recharge projects
#[hdk_extern]
pub fn get_all_recharge_projects(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash("all_recharge_projects")?,
            LinkTypes::AllRechargeProjects,
        )?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}
