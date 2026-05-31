// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Capture Integrity Zome
//! Water harvesting systems, storage tanks, harvest records, and aquifer recharge

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// HARVEST SYSTEMS
// ============================================================================

/// Type of water harvesting system
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum HarvestType {
    RoofRainwater,
    GroundCatchment,
    FogCollection,
    DewCollection,
    Snowmelt,
}

/// A water harvesting system installation
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct HarvestSystem {
    /// Unique identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Type of harvesting system
    pub system_type: HarvestType,
    /// Maximum collection capacity in liters
    pub capacity_liters: u64,
    /// GPS latitude
    pub location_lat: f64,
    /// GPS longitude
    pub location_lon: f64,
    /// Owner/operator of the system
    pub owner: AgentPubKey,
    /// When the system was installed
    pub installed_at: Timestamp,
    /// Catchment area in square meters (for rainwater systems)
    pub catchment_area_sqm: Option<u32>,
    /// Collection efficiency as a percentage (0-100)
    pub efficiency_percent: u8,
}

// ============================================================================
// STORAGE TANKS
// ============================================================================

/// Type of water storage tank
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TankType {
    Underground,
    Aboveground,
    Bladder,
    Cistern,
    Dam,
}

/// A water storage tank
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct StorageTank {
    /// Unique identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Maximum capacity in liters
    pub capacity_liters: u64,
    /// Current water level in liters
    pub current_level_liters: u64,
    /// Type of tank
    pub tank_type: TankType,
    /// Optional link to connected harvest system
    pub connected_system: Option<ActionHash>,
    /// GPS latitude
    pub location_lat: f64,
    /// GPS longitude
    pub location_lon: f64,
    /// Owner of the tank
    pub owner: AgentPubKey,
}

// ============================================================================
// HARVEST RECORDS
// ============================================================================

/// A record of water collected by a harvest system
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct HarvestRecord {
    /// The harvest system that collected the water
    pub system_hash: ActionHash,
    /// Volume collected in liters
    pub liters_collected: u64,
    /// Start of collection period
    pub collection_period_start: Timestamp,
    /// End of collection period
    pub collection_period_end: Timestamp,
    /// Weather conditions during collection
    pub weather_conditions: Option<String>,
    /// Agent credited for this harvest
    pub credited_to: AgentPubKey,
}

// ============================================================================
// AQUIFER RECHARGE
// ============================================================================

/// Method of aquifer recharge
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RechargeMethod {
    /// Surface spreading basins
    Basin,
    /// Direct injection wells
    Injection,
    /// Controlled spreading across land
    Spreading,
    /// Enhanced natural infiltration
    Infiltration,
}

/// Status of a recharge project
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RechargeStatus {
    Proposed,
    Active,
    Paused,
    Completed,
    Abandoned,
}

/// An aquifer recharge project
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RechargeProject {
    /// Unique identifier
    pub id: String,
    /// Project name
    pub name: String,
    /// Target aquifer identifier
    pub aquifer_id: String,
    /// Recharge method used
    pub method: RechargeMethod,
    /// Daily recharge capacity in liters
    pub capacity_liters_per_day: u64,
    /// GPS latitude
    pub location_lat: f64,
    /// GPS longitude
    pub location_lon: f64,
    /// Current project status
    pub status: RechargeStatus,
}

// ============================================================================
// ENTRY & LINK TYPE REGISTRATION
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    HarvestSystem(HarvestSystem),
    StorageTank(StorageTank),
    HarvestRecord(HarvestRecord),
    RechargeProject(RechargeProject),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Anchor to all harvest systems
    AllSystems,
    /// Owner to their harvest systems
    OwnerToSystem,
    /// System type to systems
    SystemTypeToSystem,
    /// Anchor to all tanks
    AllTanks,
    /// Owner to their tanks
    OwnerToTank,
    /// System to its harvest records
    SystemToHarvestRecord,
    /// Agent to harvest records credited to them
    AgentToHarvestRecord,
    /// System to connected tank
    SystemToTank,
    /// Anchor to all recharge projects
    AllRechargeProjects,
}

// ============================================================================
// VALIDATION
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::HarvestSystem(system) => validate_create_harvest_system(action, system),
                EntryTypes::StorageTank(tank) => validate_create_storage_tank(action, tank),
                EntryTypes::HarvestRecord(record) => validate_create_harvest_record(action, record),
                EntryTypes::RechargeProject(project) => {
                    validate_create_recharge_project(action, project)
                }
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::StorageTank(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::RechargeProject(_) => Ok(ValidateCallbackResult::Valid),
                _ => Ok(ValidateCallbackResult::Valid),
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
            LinkTypes::AllSystems => Ok(ValidateCallbackResult::Valid),
            LinkTypes::OwnerToSystem => Ok(ValidateCallbackResult::Valid),
            LinkTypes::SystemTypeToSystem => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AllTanks => Ok(ValidateCallbackResult::Valid),
            LinkTypes::OwnerToTank => Ok(ValidateCallbackResult::Valid),
            LinkTypes::SystemToHarvestRecord => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToHarvestRecord => Ok(ValidateCallbackResult::Valid),
            LinkTypes::SystemToTank => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AllRechargeProjects => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type: _,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_harvest_system(
    _action: Create,
    system: HarvestSystem,
) -> ExternResult<ValidateCallbackResult> {
    if system.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Harvest system ID cannot be empty".into(),
        ));
    }
    if system.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Harvest system name cannot be empty".into(),
        ));
    }
    if system.capacity_liters == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Capacity must be greater than zero".into(),
        ));
    }
    if system.efficiency_percent > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Efficiency percent cannot exceed 100".into(),
        ));
    }
    if system.location_lat < -90.0 || system.location_lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be between -90 and 90".into(),
        ));
    }
    if system.location_lon < -180.0 || system.location_lon > 180.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be between -180 and 180".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_storage_tank(
    _action: Create,
    tank: StorageTank,
) -> ExternResult<ValidateCallbackResult> {
    if tank.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Tank ID cannot be empty".into(),
        ));
    }
    if tank.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Tank name cannot be empty".into(),
        ));
    }
    if tank.capacity_liters == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Tank capacity must be greater than zero".into(),
        ));
    }
    if tank.current_level_liters > tank.capacity_liters {
        return Ok(ValidateCallbackResult::Invalid(
            "Current level cannot exceed capacity".into(),
        ));
    }
    if tank.location_lat < -90.0 || tank.location_lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be between -90 and 90".into(),
        ));
    }
    if tank.location_lon < -180.0 || tank.location_lon > 180.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be between -180 and 180".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_harvest_record(
    _action: Create,
    record: HarvestRecord,
) -> ExternResult<ValidateCallbackResult> {
    if record.liters_collected == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Harvest liters must be greater than zero".into(),
        ));
    }
    if record.collection_period_end <= record.collection_period_start {
        return Ok(ValidateCallbackResult::Invalid(
            "Collection period end must be after start".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_recharge_project(
    _action: Create,
    project: RechargeProject,
) -> ExternResult<ValidateCallbackResult> {
    if project.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Recharge project ID cannot be empty".into(),
        ));
    }
    if project.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Recharge project name cannot be empty".into(),
        ));
    }
    if project.aquifer_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Aquifer ID cannot be empty".into(),
        ));
    }
    if project.capacity_liters_per_day == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Recharge capacity must be greater than zero".into(),
        ));
    }
    if project.location_lat < -90.0 || project.location_lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be between -90 and 90".into(),
        ));
    }
    if project.location_lon < -180.0 || project.location_lon > 180.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be between -180 and 180".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
