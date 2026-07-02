// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Machines Coordinator Zome
//!
//! Machine registry and status management for the manufacturing floor.

use hdk::prelude::*;
use machines_integrity::*;
use manufacturing_common::{MachineStatus, MachineType};

// ============================================================================
// Input types
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterMachineInput {
    pub name: String,
    pub machine_type: MachineType,
    pub capabilities: Vec<String>,
    pub location: String,
    pub max_throughput_per_hour: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateMachineStatusInput {
    pub machine_hash: ActionHash,
    pub new_status: MachineStatus,
    pub work_order_hash: Option<ActionHash>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetMachinesByTypeInput {
    pub machine_type: MachineType,
}

// ============================================================================
// Extern functions
// ============================================================================

/// Register a new machine in the manufacturing cell.
#[hdk_extern]
pub fn register_machine(input: RegisterMachineInput) -> ExternResult<ActionHash> {
    if input.name.is_empty() || input.name.len() > 200 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "machine name must be 1-200 characters".to_string()
        )));
    }
    if input.location.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "location is required".to_string()
        )));
    }
    if input.max_throughput_per_hour == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "max_throughput_per_hour must be > 0".to_string()
        )));
    }

    let now = sys_time()?;
    let type_tag = machine_type_tag(&input.machine_type);
    let entry = MachineEntry {
        name: input.name,
        machine_type: input.machine_type,
        capabilities: input.capabilities,
        location: input.location.clone(),
        max_throughput_per_hour: input.max_throughput_per_hour,
        status: MachineStatus::Available,
        current_work_order: None,
        registered_at: now,
    };

    let action_hash = create_entry(EntryTypes::Machine(entry))?;

    // Link from "all_machines" anchor
    let all_path = Path::from("all_machines").typed(LinkTypes::AllMachines)?;
    all_path.ensure()?;
    create_link(
        all_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::AllMachines,
        (),
    )?;

    // Link from machine type
    let type_path = Path::from(format!("machine_type/{type_tag}")).typed(LinkTypes::TypeToMachines)?;
    type_path.ensure()?;
    create_link(
        type_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::TypeToMachines,
        (),
    )?;

    // Link from location
    let loc_path =
        Path::from(format!("location/{}", input.location)).typed(LinkTypes::LocationToMachines)?;
    loc_path.ensure()?;
    create_link(
        loc_path.path_entry_hash()?,
        action_hash.clone(),
        LinkTypes::LocationToMachines,
        (),
    )?;

    Ok(action_hash)
}

/// Get a machine by its action hash.
#[hdk_extern]
pub fn get_machine(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Update machine status (e.g., Available -> Running).
#[hdk_extern]
pub fn update_machine_status(input: UpdateMachineStatusInput) -> ExternResult<ActionHash> {
    let record = get(input.machine_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Machine not found".to_string())),
    )?;

    let machine: MachineEntry = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not deserialize machine".to_string()
        )))?;

    if !machine.status.can_transition_to(&input.new_status) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Invalid machine transition: {:?} -> {:?}",
            machine.status, input.new_status
        ))));
    }

    let now = sys_time()?;

    // Log the status change
    let log = MachineStatusLog {
        machine_hash: input.machine_hash.clone(),
        previous_status: machine.status.clone(),
        new_status: input.new_status.clone(),
        work_order_hash: input.work_order_hash.clone(),
        changed_at: now,
    };
    let log_hash = create_entry(EntryTypes::StatusLog(log))?;
    create_link(
        input.machine_hash.clone(),
        log_hash,
        LinkTypes::MachineToStatusLog,
        (),
    )?;

    // Update the machine entry
    let updated = MachineEntry {
        status: input.new_status,
        current_work_order: input.work_order_hash,
        ..machine
    };
    update_entry(input.machine_hash, EntryTypes::Machine(updated))
}

/// Get all machines that are currently Available.
#[hdk_extern]
pub fn get_available_machines(_: ()) -> ExternResult<Vec<Record>> {
    let all_path = Path::from("all_machines").typed(LinkTypes::AllMachines)?;
    let links = get_links(
        GetLinksInputBuilder::try_new(all_path.path_entry_hash()?, LinkTypes::AllMachines)?.build(),
    )?;

    let mut available = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(machine) = record
                    .entry()
                    .to_app_option::<MachineEntry>()
                    .ok()
                    .flatten()
                {
                    if machine.status == MachineStatus::Available {
                        available.push(record);
                    }
                }
            }
        }
    }
    Ok(available)
}

/// Get machines by type via type anchor links.
#[hdk_extern]
pub fn get_machines_by_type(input: GetMachinesByTypeInput) -> ExternResult<Vec<Link>> {
    let tag = machine_type_tag(&input.machine_type);
    let type_path = Path::from(format!("machine_type/{tag}")).typed(LinkTypes::TypeToMachines)?;
    get_links(
        GetLinksInputBuilder::try_new(type_path.path_entry_hash()?, LinkTypes::TypeToMachines)?
            .build(),
    )
}

/// Produce a stable string tag for each machine type variant.
fn machine_type_tag(mt: &MachineType) -> String {
    match mt {
        MachineType::FDM => "fdm".to_string(),
        MachineType::SLA => "sla".to_string(),
        MachineType::SLS => "sls".to_string(),
        MachineType::CNC3Axis => "cnc3".to_string(),
        MachineType::CNC5Axis => "cnc5".to_string(),
        MachineType::LaserCutter => "laser".to_string(),
        MachineType::Lathe => "lathe".to_string(),
        MachineType::Assembly => "assembly".to_string(),
        MachineType::Custom(s) => format!("custom_{s}"),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_input_serde() {
        let input = RegisterMachineInput {
            name: "CNC Mill #3".to_string(),
            machine_type: MachineType::CNC3Axis,
            capabilities: vec!["milling".to_string(), "drilling".to_string()],
            location: "Bay 2".to_string(),
            max_throughput_per_hour: 12,
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: RegisterMachineInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "CNC Mill #3");
        assert_eq!(back.machine_type, MachineType::CNC3Axis);
    }

    #[test]
    fn test_machine_type_tag_variants() {
        assert_eq!(machine_type_tag(&MachineType::FDM), "fdm");
        assert_eq!(machine_type_tag(&MachineType::CNC5Axis), "cnc5");
        assert_eq!(machine_type_tag(&MachineType::LaserCutter), "laser");
        assert_eq!(
            machine_type_tag(&MachineType::Custom("Waterjet".to_string())),
            "custom_Waterjet"
        );
    }

    #[test]
    fn test_update_status_input_serde() {
        let input = UpdateMachineStatusInput {
            machine_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            new_status: MachineStatus::Running,
            work_order_hash: Some(ActionHash::from_raw_36(vec![1u8; 36])),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: UpdateMachineStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.new_status, MachineStatus::Running);
        assert!(back.work_order_hash.is_some());
    }

    #[test]
    fn test_machine_status_all_variants_serde() {
        for s in [
            MachineStatus::Available,
            MachineStatus::Running,
            MachineStatus::Maintenance,
            MachineStatus::Offline,
        ] {
            let json = serde_json::to_string(&s).unwrap();
            let back: MachineStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, s);
        }
    }

    #[test]
    fn test_machine_entry_serde() {
        let entry = MachineEntry {
            name: "Lathe #1".to_string(),
            machine_type: MachineType::Lathe,
            capabilities: vec!["turning".to_string()],
            location: "Bay 3".to_string(),
            max_throughput_per_hour: 8,
            status: MachineStatus::Available,
            current_work_order: None,
            registered_at: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("Lathe #1"));
        assert!(json.contains("\"status\":\"Available\""));
    }
}
