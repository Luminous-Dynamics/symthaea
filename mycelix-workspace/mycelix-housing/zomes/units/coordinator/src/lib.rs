// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Units Coordinator Zome
//! Business logic for buildings and housing units.

use hdk::prelude::*;
use units_integrity::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Register a new building in the cooperative
#[hdk_extern]
pub fn register_building(building: Building) -> ExternResult<Record> {
    if building.name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Building name must be at most 256 characters".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::Building(building.clone()))?;

    // Link to all buildings anchor
    create_entry(&EntryTypes::Anchor(Anchor("all_buildings".to_string())))?;
    create_link(
        anchor_hash("all_buildings")?,
        action_hash.clone(),
        LinkTypes::AllBuildings,
        (),
    )?;

    // Link by building type
    let type_anchor = format!("building_type:{:?}", building.building_type);
    create_entry(&EntryTypes::Anchor(Anchor(type_anchor.clone())))?;
    create_link(
        anchor_hash(&type_anchor)?,
        action_hash.clone(),
        LinkTypes::BuildingTypeToBuilding,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created building".into()
    )))
}

/// Register a new unit within a building
#[hdk_extern]
pub fn register_unit(unit: Unit) -> ExternResult<Record> {
    if unit.unit_number.len() > 64 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Unit number must be at most 64 characters".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::HousingUnit(unit.clone()))?;

    // Link building to unit
    create_link(
        unit.building_hash.clone(),
        action_hash.clone(),
        LinkTypes::BuildingToUnit,
        (),
    )?;

    // If available, add to available units index
    if unit.status == UnitStatus::Available {
        create_entry(&EntryTypes::Anchor(Anchor("available_units".to_string())))?;
        create_link(
            anchor_hash("available_units")?,
            action_hash.clone(),
            LinkTypes::AvailableUnits,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created unit".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateUnitStatusInput {
    pub unit_action_hash: ActionHash,
    pub new_status: UnitStatus,
}

/// Update the status of a unit
#[hdk_extern]
pub fn update_unit_status(input: UpdateUnitStatusInput) -> ExternResult<Record> {
    let record = get(input.unit_action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Unit not found".into())))?;

    let mut unit: Unit = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid unit entry".into()
        )))?;

    let was_available = unit.status == UnitStatus::Available;
    unit.status = input.new_status.clone();
    let is_available = unit.status == UnitStatus::Available;

    let new_hash = update_entry(
        input.unit_action_hash.clone(),
        &EntryTypes::HousingUnit(unit),
    )?;

    // Manage available units index
    if was_available && !is_available {
        // Remove from available index - find and delete the link
        let links = get_links(
            LinkQuery::try_new(anchor_hash("available_units")?, LinkTypes::AvailableUnits)?,
            GetStrategy::default(),
        )?;
        for link in links {
            let target = ActionHash::try_from(link.target.clone());
            if let Ok(target_hash) = target {
                if target_hash == input.unit_action_hash {
                    delete_link(link.create_link_hash, GetOptions::default())?;
                }
            }
        }
    } else if !was_available && is_available {
        create_entry(&EntryTypes::Anchor(Anchor("available_units".to_string())))?;
        create_link(
            anchor_hash("available_units")?,
            new_hash.clone(),
            LinkTypes::AvailableUnits,
            (),
        )?;
    }

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated unit".into()
    )))
}

/// Get all units in a building
#[hdk_extern]
pub fn get_building_units(building_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(building_hash, LinkTypes::BuildingToUnit)?,
        GetStrategy::default(),
    )?;

    let mut units = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            units.push(record);
        }
    }

    Ok(units)
}

/// Get all available units across all buildings
#[hdk_extern]
pub fn get_available_units(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("available_units")?, LinkTypes::AvailableUnits)?,
        GetStrategy::default(),
    )?;

    let mut units = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            units.push(record);
        }
    }

    Ok(units)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AssignOccupantInput {
    pub unit_action_hash: ActionHash,
    pub occupant: AgentPubKey,
}

/// Assign an occupant to a unit
#[hdk_extern]
pub fn assign_occupant(input: AssignOccupantInput) -> ExternResult<Record> {
    let record = get(input.unit_action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Unit not found".into())))?;

    let mut unit: Unit = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid unit entry".into()
        )))?;

    if unit.status == UnitStatus::Occupied {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Unit is already occupied".into()
        )));
    }

    unit.current_occupant = Some(input.occupant.clone());
    unit.status = UnitStatus::Occupied;

    let new_hash = update_entry(
        input.unit_action_hash.clone(),
        &EntryTypes::HousingUnit(unit),
    )?;

    // Link occupant to unit
    create_link(
        input.occupant,
        new_hash.clone(),
        LinkTypes::OccupantToUnit,
        (),
    )?;

    // Remove from available index
    let links = get_links(
        LinkQuery::try_new(anchor_hash("available_units")?, LinkTypes::AvailableUnits)?,
        GetStrategy::default(),
    )?;
    for link in links {
        let target = ActionHash::try_from(link.target.clone());
        if let Ok(target_hash) = target {
            if target_hash == input.unit_action_hash {
                delete_link(link.create_link_hash, GetOptions::default())?;
            }
        }
    }

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated unit".into()
    )))
}

/// Vacate a unit, removing the occupant
#[hdk_extern]
pub fn vacate_unit(unit_action_hash: ActionHash) -> ExternResult<Record> {
    let record = get(unit_action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Unit not found".into())))?;

    let mut unit: Unit = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid unit entry".into()
        )))?;

    if unit.current_occupant.is_none() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Unit has no occupant to vacate".into()
        )));
    }

    let old_occupant = unit.current_occupant.clone();
    unit.current_occupant = None;
    unit.status = UnitStatus::Available;

    let new_hash = update_entry(unit_action_hash.clone(), &EntryTypes::HousingUnit(unit))?;

    // Remove occupant link
    if let Some(occupant) = old_occupant {
        let links = get_links(
            LinkQuery::try_new(occupant, LinkTypes::OccupantToUnit)?,
            GetStrategy::default(),
        )?;
        for link in links {
            let target = ActionHash::try_from(link.target.clone());
            if let Ok(target_hash) = target {
                if target_hash == unit_action_hash {
                    delete_link(link.create_link_hash, GetOptions::default())?;
                }
            }
        }
    }

    // Add back to available index
    create_entry(&EntryTypes::Anchor(Anchor("available_units".to_string())))?;
    create_link(
        anchor_hash("available_units")?,
        new_hash.clone(),
        LinkTypes::AvailableUnits,
        (),
    )?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated unit".into()
    )))
}
