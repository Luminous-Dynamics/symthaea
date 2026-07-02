// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Capture Integrity Zome
//! Water harvesting systems, storage tanks, harvest records, and aquifer recharge

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

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
    /// Geohash spatial index
    GeoIndex,
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
                EntryTypes::HarvestSystem(system) => validate_update_harvest_system(system),
                EntryTypes::StorageTank(tank) => validate_update_storage_tank(tank),
                EntryTypes::HarvestRecord(record) => validate_update_harvest_record(record),
                EntryTypes::RechargeProject(project) => validate_update_recharge_project(project),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => match link_type {
            LinkTypes::AllSystems => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllSystems link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::OwnerToSystem => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "OwnerToSystem link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::SystemTypeToSystem => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "SystemTypeToSystem link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllTanks => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllTanks link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::OwnerToTank => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "OwnerToTank link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::SystemToHarvestRecord => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "SystemToHarvestRecord link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToHarvestRecord => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToHarvestRecord link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::SystemToTank => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "SystemToTank link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllRechargeProjects => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllRechargeProjects link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::GeoIndex => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            Ok(check_link_author_match(
                original_action.action().author(),
                &action.author,
            ))
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "update",
            ))
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "delete",
            ))
        }
    }
}

fn validate_create_harvest_system(
    _action: Create,
    system: HarvestSystem,
) -> ExternResult<ValidateCallbackResult> {
    if system.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Harvest system ID cannot be empty".into(),
        ));
    }
    if system.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Harvest system ID must be 256 characters or fewer".into(),
        ));
    }
    if system.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Harvest system name cannot be empty".into(),
        ));
    }
    if system.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Harvest system name must be 256 characters or fewer".into(),
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
    if !system.location_lat.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be a finite number".into(),
        ));
    }
    if system.location_lat < -90.0 || system.location_lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be between -90 and 90".into(),
        ));
    }
    if !system.location_lon.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be a finite number".into(),
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
    if tank.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Tank ID cannot be empty".into(),
        ));
    }
    if tank.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Tank ID must be 256 characters or fewer".into(),
        ));
    }
    if tank.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Tank name cannot be empty".into(),
        ));
    }
    if tank.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Tank name must be 256 characters or fewer".into(),
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
    if !tank.location_lat.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be a finite number".into(),
        ));
    }
    if tank.location_lat < -90.0 || tank.location_lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be between -90 and 90".into(),
        ));
    }
    if !tank.location_lon.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be a finite number".into(),
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
    if let Some(ref wc) = record.weather_conditions {
        if wc.len() > 4096 {
            return Ok(ValidateCallbackResult::Invalid(
                "Weather conditions must be 4096 characters or fewer".into(),
            ));
        }
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
    if project.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Recharge project ID cannot be empty".into(),
        ));
    }
    if project.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Recharge project ID must be 256 characters or fewer".into(),
        ));
    }
    if project.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Recharge project name cannot be empty".into(),
        ));
    }
    if project.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Recharge project name must be 256 characters or fewer".into(),
        ));
    }
    if project.aquifer_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Aquifer ID cannot be empty".into(),
        ));
    }
    if project.aquifer_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Aquifer ID must be 256 characters or fewer".into(),
        ));
    }
    if project.capacity_liters_per_day == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Recharge capacity must be greater than zero".into(),
        ));
    }
    if !project.location_lat.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be a finite number".into(),
        ));
    }
    if project.location_lat < -90.0 || project.location_lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be between -90 and 90".into(),
        ));
    }
    if !project.location_lon.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be a finite number".into(),
        ));
    }
    if project.location_lon < -180.0 || project.location_lon > 180.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be between -180 and 180".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_harvest_system(system: HarvestSystem) -> ExternResult<ValidateCallbackResult> {
    if system.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Harvest system ID must be 256 characters or fewer".into(),
        ));
    }
    if system.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Harvest system name must be 256 characters or fewer".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_storage_tank(tank: StorageTank) -> ExternResult<ValidateCallbackResult> {
    if tank.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Tank ID must be 256 characters or fewer".into(),
        ));
    }
    if tank.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Tank name must be 256 characters or fewer".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_harvest_record(record: HarvestRecord) -> ExternResult<ValidateCallbackResult> {
    if let Some(ref wc) = record.weather_conditions {
        if wc.len() > 4096 {
            return Ok(ValidateCallbackResult::Invalid(
                "Weather conditions must be 4096 characters or fewer".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_recharge_project(
    project: RechargeProject,
) -> ExternResult<ValidateCallbackResult> {
    if project.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Recharge project ID must be 256 characters or fewer".into(),
        ));
    }
    if project.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Recharge project name must be 256 characters or fewer".into(),
        ));
    }
    if project.aquifer_id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Aquifer ID must be 256 characters or fewer".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // HELPERS
    // ========================================================================

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_entry_hash() -> EntryHash {
        EntryHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_create() -> Create {
        Create {
            author: fake_agent(),
            timestamp: Timestamp::from_micros(0),
            action_seq: 0,
            prev_action: fake_action_hash(),
            entry_type: EntryType::App(AppEntryDef::new(
                EntryDefIndex(0),
                ZomeIndex(0),
                EntryVisibility::Public,
            )),
            entry_hash: fake_entry_hash(),
            weight: EntryRateWeight::default(),
        }
    }

    fn is_valid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Valid))
    }

    fn is_invalid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Invalid(_)))
    }

    fn invalid_msg(result: &ExternResult<ValidateCallbackResult>) -> String {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => msg.clone(),
            _ => panic!("Expected Invalid, got {:?}", result),
        }
    }

    fn make_harvest_system() -> HarvestSystem {
        HarvestSystem {
            id: "hs-001".into(),
            name: "Roof Catchment A".into(),
            system_type: HarvestType::RoofRainwater,
            capacity_liters: 5_000,
            location_lat: 35.0,
            location_lon: -95.5,
            owner: fake_agent(),
            installed_at: Timestamp::from_micros(0),
            catchment_area_sqm: Some(100),
            efficiency_percent: 85,
        }
    }

    fn make_storage_tank() -> StorageTank {
        StorageTank {
            id: "tank-001".into(),
            name: "Underground Cistern 1".into(),
            capacity_liters: 10_000,
            current_level_liters: 7_500,
            tank_type: TankType::Underground,
            connected_system: Some(fake_action_hash()),
            location_lat: 35.0,
            location_lon: -95.5,
            owner: fake_agent(),
        }
    }

    fn make_harvest_record() -> HarvestRecord {
        HarvestRecord {
            system_hash: fake_action_hash(),
            liters_collected: 150,
            collection_period_start: Timestamp::from_micros(0),
            collection_period_end: Timestamp::from_micros(1_000_000),
            weather_conditions: Some("Light rain".into()),
            credited_to: fake_agent(),
        }
    }

    fn make_recharge_project() -> RechargeProject {
        RechargeProject {
            id: "rp-001".into(),
            name: "Basin Recharge Alpha".into(),
            aquifer_id: "aquifer-12".into(),
            method: RechargeMethod::Basin,
            capacity_liters_per_day: 50_000,
            location_lat: 40.0,
            location_lon: -110.0,
            status: RechargeStatus::Active,
        }
    }

    // ========================================================================
    // SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn serde_roundtrip_harvest_type() {
        for variant in [
            HarvestType::RoofRainwater,
            HarvestType::GroundCatchment,
            HarvestType::FogCollection,
            HarvestType::DewCollection,
            HarvestType::Snowmelt,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: HarvestType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn serde_roundtrip_tank_type() {
        for variant in [
            TankType::Underground,
            TankType::Aboveground,
            TankType::Bladder,
            TankType::Cistern,
            TankType::Dam,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: TankType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn serde_roundtrip_recharge_method() {
        for variant in [
            RechargeMethod::Basin,
            RechargeMethod::Injection,
            RechargeMethod::Spreading,
            RechargeMethod::Infiltration,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: RechargeMethod = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn serde_roundtrip_recharge_status() {
        for variant in [
            RechargeStatus::Proposed,
            RechargeStatus::Active,
            RechargeStatus::Paused,
            RechargeStatus::Completed,
            RechargeStatus::Abandoned,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: RechargeStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    // ========================================================================
    // HARVEST SYSTEM VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_harvest_system_passes() {
        let result = validate_create_harvest_system(fake_create(), make_harvest_system());
        assert!(is_valid(&result));
    }

    #[test]
    fn harvest_system_empty_id_rejected() {
        let mut sys = make_harvest_system();
        sys.id = "".into();
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Harvest system ID cannot be empty");
    }

    #[test]
    fn harvest_system_empty_name_rejected() {
        let mut sys = make_harvest_system();
        sys.name = "".into();
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Harvest system name cannot be empty");
    }

    #[test]
    fn harvest_system_zero_capacity_rejected() {
        let mut sys = make_harvest_system();
        sys.capacity_liters = 0;
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Capacity must be greater than zero");
    }

    #[test]
    fn harvest_system_one_liter_capacity_accepted() {
        let mut sys = make_harvest_system();
        sys.capacity_liters = 1;
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_valid(&result));
    }

    #[test]
    fn harvest_system_u64_max_capacity_accepted() {
        let mut sys = make_harvest_system();
        sys.capacity_liters = u64::MAX;
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_valid(&result));
    }

    #[test]
    fn harvest_system_efficiency_100_accepted() {
        let mut sys = make_harvest_system();
        sys.efficiency_percent = 100;
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_valid(&result));
    }

    #[test]
    fn harvest_system_efficiency_over_100_rejected() {
        let mut sys = make_harvest_system();
        sys.efficiency_percent = 101;
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Efficiency percent cannot exceed 100");
    }

    #[test]
    fn harvest_system_efficiency_255_rejected() {
        let mut sys = make_harvest_system();
        sys.efficiency_percent = 255;
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Efficiency percent cannot exceed 100");
    }

    #[test]
    fn harvest_system_efficiency_zero_accepted() {
        let mut sys = make_harvest_system();
        sys.efficiency_percent = 0;
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_valid(&result));
    }

    #[test]
    fn harvest_system_lat_over_90_rejected() {
        let mut sys = make_harvest_system();
        sys.location_lat = 90.001;
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Latitude must be between -90 and 90");
    }

    #[test]
    fn harvest_system_lat_under_neg90_rejected() {
        let mut sys = make_harvest_system();
        sys.location_lat = -90.001;
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Latitude must be between -90 and 90");
    }

    #[test]
    fn harvest_system_lat_exactly_90_accepted() {
        let mut sys = make_harvest_system();
        sys.location_lat = 90.0;
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_valid(&result));
    }

    #[test]
    fn harvest_system_lat_exactly_neg90_accepted() {
        let mut sys = make_harvest_system();
        sys.location_lat = -90.0;
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_valid(&result));
    }

    #[test]
    fn harvest_system_lat_zero_accepted() {
        let mut sys = make_harvest_system();
        sys.location_lat = 0.0;
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_valid(&result));
    }

    #[test]
    fn harvest_system_lon_over_180_rejected() {
        let mut sys = make_harvest_system();
        sys.location_lon = 180.001;
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Longitude must be between -180 and 180"
        );
    }

    #[test]
    fn harvest_system_lon_under_neg180_rejected() {
        let mut sys = make_harvest_system();
        sys.location_lon = -180.001;
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Longitude must be between -180 and 180"
        );
    }

    #[test]
    fn harvest_system_lon_exactly_180_accepted() {
        let mut sys = make_harvest_system();
        sys.location_lon = 180.0;
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_valid(&result));
    }

    #[test]
    fn harvest_system_lon_exactly_neg180_accepted() {
        let mut sys = make_harvest_system();
        sys.location_lon = -180.0;
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_valid(&result));
    }

    #[test]
    fn harvest_system_lon_zero_accepted() {
        let mut sys = make_harvest_system();
        sys.location_lon = 0.0;
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_valid(&result));
    }

    #[test]
    fn harvest_system_all_types_accepted() {
        for system_type in [
            HarvestType::RoofRainwater,
            HarvestType::GroundCatchment,
            HarvestType::FogCollection,
            HarvestType::DewCollection,
            HarvestType::Snowmelt,
        ] {
            let mut sys = make_harvest_system();
            sys.system_type = system_type;
            let result = validate_create_harvest_system(fake_create(), sys);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn harvest_system_with_catchment_area_accepted() {
        let mut sys = make_harvest_system();
        sys.catchment_area_sqm = Some(250);
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_valid(&result));
    }

    #[test]
    fn harvest_system_without_catchment_area_accepted() {
        let mut sys = make_harvest_system();
        sys.catchment_area_sqm = None;
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_valid(&result));
    }

    #[test]
    fn harvest_system_zero_catchment_area_accepted() {
        let mut sys = make_harvest_system();
        sys.catchment_area_sqm = Some(0);
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_valid(&result));
    }

    #[test]
    fn harvest_system_u32_max_catchment_area_accepted() {
        let mut sys = make_harvest_system();
        sys.catchment_area_sqm = Some(u32::MAX);
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // STORAGE TANK VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_storage_tank_passes() {
        let result = validate_create_storage_tank(fake_create(), make_storage_tank());
        assert!(is_valid(&result));
    }

    #[test]
    fn storage_tank_empty_id_rejected() {
        let mut tank = make_storage_tank();
        tank.id = "".into();
        let result = validate_create_storage_tank(fake_create(), tank);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Tank ID cannot be empty");
    }

    #[test]
    fn storage_tank_empty_name_rejected() {
        let mut tank = make_storage_tank();
        tank.name = "".into();
        let result = validate_create_storage_tank(fake_create(), tank);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Tank name cannot be empty");
    }

    #[test]
    fn storage_tank_zero_capacity_rejected() {
        let mut tank = make_storage_tank();
        tank.capacity_liters = 0;
        let result = validate_create_storage_tank(fake_create(), tank);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Tank capacity must be greater than zero"
        );
    }

    #[test]
    fn storage_tank_one_liter_capacity_accepted() {
        let mut tank = make_storage_tank();
        tank.capacity_liters = 1;
        tank.current_level_liters = 0;
        let result = validate_create_storage_tank(fake_create(), tank);
        assert!(is_valid(&result));
    }

    #[test]
    fn storage_tank_u64_max_capacity_accepted() {
        let mut tank = make_storage_tank();
        tank.capacity_liters = u64::MAX;
        tank.current_level_liters = 0;
        let result = validate_create_storage_tank(fake_create(), tank);
        assert!(is_valid(&result));
    }

    #[test]
    fn storage_tank_current_level_exceeds_capacity_rejected() {
        let mut tank = make_storage_tank();
        tank.capacity_liters = 1_000;
        tank.current_level_liters = 1_001;
        let result = validate_create_storage_tank(fake_create(), tank);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Current level cannot exceed capacity");
    }

    #[test]
    fn storage_tank_current_level_equals_capacity_accepted() {
        let mut tank = make_storage_tank();
        tank.capacity_liters = 1_000;
        tank.current_level_liters = 1_000;
        let result = validate_create_storage_tank(fake_create(), tank);
        assert!(is_valid(&result));
    }

    #[test]
    fn storage_tank_current_level_zero_accepted() {
        let mut tank = make_storage_tank();
        tank.current_level_liters = 0;
        let result = validate_create_storage_tank(fake_create(), tank);
        assert!(is_valid(&result));
    }

    #[test]
    fn storage_tank_lat_over_90_rejected() {
        let mut tank = make_storage_tank();
        tank.location_lat = 90.001;
        let result = validate_create_storage_tank(fake_create(), tank);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Latitude must be between -90 and 90");
    }

    #[test]
    fn storage_tank_lat_under_neg90_rejected() {
        let mut tank = make_storage_tank();
        tank.location_lat = -90.001;
        let result = validate_create_storage_tank(fake_create(), tank);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Latitude must be between -90 and 90");
    }

    #[test]
    fn storage_tank_lat_exactly_90_accepted() {
        let mut tank = make_storage_tank();
        tank.location_lat = 90.0;
        let result = validate_create_storage_tank(fake_create(), tank);
        assert!(is_valid(&result));
    }

    #[test]
    fn storage_tank_lat_exactly_neg90_accepted() {
        let mut tank = make_storage_tank();
        tank.location_lat = -90.0;
        let result = validate_create_storage_tank(fake_create(), tank);
        assert!(is_valid(&result));
    }

    #[test]
    fn storage_tank_lat_zero_accepted() {
        let mut tank = make_storage_tank();
        tank.location_lat = 0.0;
        let result = validate_create_storage_tank(fake_create(), tank);
        assert!(is_valid(&result));
    }

    #[test]
    fn storage_tank_lon_over_180_rejected() {
        let mut tank = make_storage_tank();
        tank.location_lon = 180.001;
        let result = validate_create_storage_tank(fake_create(), tank);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Longitude must be between -180 and 180"
        );
    }

    #[test]
    fn storage_tank_lon_under_neg180_rejected() {
        let mut tank = make_storage_tank();
        tank.location_lon = -180.001;
        let result = validate_create_storage_tank(fake_create(), tank);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Longitude must be between -180 and 180"
        );
    }

    #[test]
    fn storage_tank_lon_exactly_180_accepted() {
        let mut tank = make_storage_tank();
        tank.location_lon = 180.0;
        let result = validate_create_storage_tank(fake_create(), tank);
        assert!(is_valid(&result));
    }

    #[test]
    fn storage_tank_lon_exactly_neg180_accepted() {
        let mut tank = make_storage_tank();
        tank.location_lon = -180.0;
        let result = validate_create_storage_tank(fake_create(), tank);
        assert!(is_valid(&result));
    }

    #[test]
    fn storage_tank_lon_zero_accepted() {
        let mut tank = make_storage_tank();
        tank.location_lon = 0.0;
        let result = validate_create_storage_tank(fake_create(), tank);
        assert!(is_valid(&result));
    }

    #[test]
    fn storage_tank_all_types_accepted() {
        for tank_type in [
            TankType::Underground,
            TankType::Aboveground,
            TankType::Bladder,
            TankType::Cistern,
            TankType::Dam,
        ] {
            let mut tank = make_storage_tank();
            tank.tank_type = tank_type;
            let result = validate_create_storage_tank(fake_create(), tank);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn storage_tank_with_connected_system_accepted() {
        let mut tank = make_storage_tank();
        tank.connected_system = Some(fake_action_hash());
        let result = validate_create_storage_tank(fake_create(), tank);
        assert!(is_valid(&result));
    }

    #[test]
    fn storage_tank_without_connected_system_accepted() {
        let mut tank = make_storage_tank();
        tank.connected_system = None;
        let result = validate_create_storage_tank(fake_create(), tank);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // HARVEST RECORD VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_harvest_record_passes() {
        let result = validate_create_harvest_record(fake_create(), make_harvest_record());
        assert!(is_valid(&result));
    }

    #[test]
    fn harvest_record_zero_liters_rejected() {
        let mut rec = make_harvest_record();
        rec.liters_collected = 0;
        let result = validate_create_harvest_record(fake_create(), rec);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Harvest liters must be greater than zero"
        );
    }

    #[test]
    fn harvest_record_one_liter_accepted() {
        let mut rec = make_harvest_record();
        rec.liters_collected = 1;
        let result = validate_create_harvest_record(fake_create(), rec);
        assert!(is_valid(&result));
    }

    #[test]
    fn harvest_record_u64_max_liters_accepted() {
        let mut rec = make_harvest_record();
        rec.liters_collected = u64::MAX;
        let result = validate_create_harvest_record(fake_create(), rec);
        assert!(is_valid(&result));
    }

    #[test]
    fn harvest_record_end_before_start_rejected() {
        let mut rec = make_harvest_record();
        rec.collection_period_start = Timestamp::from_micros(1_000_000);
        rec.collection_period_end = Timestamp::from_micros(999_999);
        let result = validate_create_harvest_record(fake_create(), rec);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Collection period end must be after start"
        );
    }

    #[test]
    fn harvest_record_end_equals_start_rejected() {
        let mut rec = make_harvest_record();
        rec.collection_period_start = Timestamp::from_micros(1_000_000);
        rec.collection_period_end = Timestamp::from_micros(1_000_000);
        let result = validate_create_harvest_record(fake_create(), rec);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Collection period end must be after start"
        );
    }

    #[test]
    fn harvest_record_end_one_microsecond_after_start_accepted() {
        let mut rec = make_harvest_record();
        rec.collection_period_start = Timestamp::from_micros(1_000_000);
        rec.collection_period_end = Timestamp::from_micros(1_000_001);
        let result = validate_create_harvest_record(fake_create(), rec);
        assert!(is_valid(&result));
    }

    #[test]
    fn harvest_record_with_weather_conditions_accepted() {
        let mut rec = make_harvest_record();
        rec.weather_conditions = Some("Heavy rain, thunderstorms".into());
        let result = validate_create_harvest_record(fake_create(), rec);
        assert!(is_valid(&result));
    }

    #[test]
    fn harvest_record_without_weather_conditions_accepted() {
        let mut rec = make_harvest_record();
        rec.weather_conditions = None;
        let result = validate_create_harvest_record(fake_create(), rec);
        assert!(is_valid(&result));
    }

    #[test]
    fn harvest_record_empty_weather_string_accepted() {
        let mut rec = make_harvest_record();
        rec.weather_conditions = Some("".into());
        let result = validate_create_harvest_record(fake_create(), rec);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // RECHARGE PROJECT VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_recharge_project_passes() {
        let result = validate_create_recharge_project(fake_create(), make_recharge_project());
        assert!(is_valid(&result));
    }

    #[test]
    fn recharge_project_empty_id_rejected() {
        let mut proj = make_recharge_project();
        proj.id = "".into();
        let result = validate_create_recharge_project(fake_create(), proj);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Recharge project ID cannot be empty");
    }

    #[test]
    fn recharge_project_empty_name_rejected() {
        let mut proj = make_recharge_project();
        proj.name = "".into();
        let result = validate_create_recharge_project(fake_create(), proj);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Recharge project name cannot be empty"
        );
    }

    #[test]
    fn recharge_project_empty_aquifer_id_rejected() {
        let mut proj = make_recharge_project();
        proj.aquifer_id = "".into();
        let result = validate_create_recharge_project(fake_create(), proj);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Aquifer ID cannot be empty");
    }

    #[test]
    fn recharge_project_zero_capacity_rejected() {
        let mut proj = make_recharge_project();
        proj.capacity_liters_per_day = 0;
        let result = validate_create_recharge_project(fake_create(), proj);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Recharge capacity must be greater than zero"
        );
    }

    #[test]
    fn recharge_project_one_liter_capacity_accepted() {
        let mut proj = make_recharge_project();
        proj.capacity_liters_per_day = 1;
        let result = validate_create_recharge_project(fake_create(), proj);
        assert!(is_valid(&result));
    }

    #[test]
    fn recharge_project_u64_max_capacity_accepted() {
        let mut proj = make_recharge_project();
        proj.capacity_liters_per_day = u64::MAX;
        let result = validate_create_recharge_project(fake_create(), proj);
        assert!(is_valid(&result));
    }

    #[test]
    fn recharge_project_lat_over_90_rejected() {
        let mut proj = make_recharge_project();
        proj.location_lat = 90.001;
        let result = validate_create_recharge_project(fake_create(), proj);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Latitude must be between -90 and 90");
    }

    #[test]
    fn recharge_project_lat_under_neg90_rejected() {
        let mut proj = make_recharge_project();
        proj.location_lat = -90.001;
        let result = validate_create_recharge_project(fake_create(), proj);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Latitude must be between -90 and 90");
    }

    #[test]
    fn recharge_project_lat_exactly_90_accepted() {
        let mut proj = make_recharge_project();
        proj.location_lat = 90.0;
        let result = validate_create_recharge_project(fake_create(), proj);
        assert!(is_valid(&result));
    }

    #[test]
    fn recharge_project_lat_exactly_neg90_accepted() {
        let mut proj = make_recharge_project();
        proj.location_lat = -90.0;
        let result = validate_create_recharge_project(fake_create(), proj);
        assert!(is_valid(&result));
    }

    #[test]
    fn recharge_project_lat_zero_accepted() {
        let mut proj = make_recharge_project();
        proj.location_lat = 0.0;
        let result = validate_create_recharge_project(fake_create(), proj);
        assert!(is_valid(&result));
    }

    #[test]
    fn recharge_project_lon_over_180_rejected() {
        let mut proj = make_recharge_project();
        proj.location_lon = 180.001;
        let result = validate_create_recharge_project(fake_create(), proj);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Longitude must be between -180 and 180"
        );
    }

    #[test]
    fn recharge_project_lon_under_neg180_rejected() {
        let mut proj = make_recharge_project();
        proj.location_lon = -180.001;
        let result = validate_create_recharge_project(fake_create(), proj);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Longitude must be between -180 and 180"
        );
    }

    #[test]
    fn recharge_project_lon_exactly_180_accepted() {
        let mut proj = make_recharge_project();
        proj.location_lon = 180.0;
        let result = validate_create_recharge_project(fake_create(), proj);
        assert!(is_valid(&result));
    }

    #[test]
    fn recharge_project_lon_exactly_neg180_accepted() {
        let mut proj = make_recharge_project();
        proj.location_lon = -180.0;
        let result = validate_create_recharge_project(fake_create(), proj);
        assert!(is_valid(&result));
    }

    #[test]
    fn recharge_project_lon_zero_accepted() {
        let mut proj = make_recharge_project();
        proj.location_lon = 0.0;
        let result = validate_create_recharge_project(fake_create(), proj);
        assert!(is_valid(&result));
    }

    #[test]
    fn recharge_project_all_methods_accepted() {
        for method in [
            RechargeMethod::Basin,
            RechargeMethod::Injection,
            RechargeMethod::Spreading,
            RechargeMethod::Infiltration,
        ] {
            let mut proj = make_recharge_project();
            proj.method = method;
            let result = validate_create_recharge_project(fake_create(), proj);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn recharge_project_all_statuses_accepted() {
        for status in [
            RechargeStatus::Proposed,
            RechargeStatus::Active,
            RechargeStatus::Paused,
            RechargeStatus::Completed,
            RechargeStatus::Abandoned,
        ] {
            let mut proj = make_recharge_project();
            proj.status = status;
            let result = validate_create_recharge_project(fake_create(), proj);
            assert!(is_valid(&result));
        }
    }

    // ========================================================================
    // STRING LENGTH LIMIT TESTS - HARVEST SYSTEM
    // ========================================================================

    #[test]
    fn harvest_system_id_exactly_64_accepted() {
        let mut sys = make_harvest_system();
        sys.id = "x".repeat(64);
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_valid(&result));
    }

    #[test]
    fn harvest_system_id_65_rejected() {
        let mut sys = make_harvest_system();
        sys.id = "x".repeat(257);
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Harvest system ID must be 256 characters or fewer"
        );
    }

    #[test]
    fn harvest_system_name_exactly_256_accepted() {
        let mut sys = make_harvest_system();
        sys.name = "x".repeat(256);
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_valid(&result));
    }

    #[test]
    fn harvest_system_name_257_rejected() {
        let mut sys = make_harvest_system();
        sys.name = "x".repeat(257);
        let result = validate_create_harvest_system(fake_create(), sys);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Harvest system name must be 256 characters or fewer"
        );
    }

    // ========================================================================
    // STRING LENGTH LIMIT TESTS - STORAGE TANK
    // ========================================================================

    #[test]
    fn storage_tank_id_exactly_64_accepted() {
        let mut tank = make_storage_tank();
        tank.id = "x".repeat(64);
        let result = validate_create_storage_tank(fake_create(), tank);
        assert!(is_valid(&result));
    }

    #[test]
    fn storage_tank_id_65_rejected() {
        let mut tank = make_storage_tank();
        tank.id = "x".repeat(257);
        let result = validate_create_storage_tank(fake_create(), tank);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Tank ID must be 256 characters or fewer"
        );
    }

    #[test]
    fn storage_tank_name_exactly_256_accepted() {
        let mut tank = make_storage_tank();
        tank.name = "x".repeat(256);
        let result = validate_create_storage_tank(fake_create(), tank);
        assert!(is_valid(&result));
    }

    #[test]
    fn storage_tank_name_257_rejected() {
        let mut tank = make_storage_tank();
        tank.name = "x".repeat(257);
        let result = validate_create_storage_tank(fake_create(), tank);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Tank name must be 256 characters or fewer"
        );
    }

    // ========================================================================
    // STRING LENGTH LIMIT TESTS - HARVEST RECORD
    // ========================================================================

    #[test]
    fn harvest_record_weather_exactly_4096_accepted() {
        let mut rec = make_harvest_record();
        rec.weather_conditions = Some("x".repeat(4096));
        let result = validate_create_harvest_record(fake_create(), rec);
        assert!(is_valid(&result));
    }

    #[test]
    fn harvest_record_weather_4097_rejected() {
        let mut rec = make_harvest_record();
        rec.weather_conditions = Some("x".repeat(4097));
        let result = validate_create_harvest_record(fake_create(), rec);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Weather conditions must be 4096 characters or fewer"
        );
    }

    // ========================================================================
    // STRING LENGTH LIMIT TESTS - RECHARGE PROJECT
    // ========================================================================

    #[test]
    fn recharge_project_id_exactly_64_accepted() {
        let mut proj = make_recharge_project();
        proj.id = "x".repeat(64);
        let result = validate_create_recharge_project(fake_create(), proj);
        assert!(is_valid(&result));
    }

    #[test]
    fn recharge_project_id_65_rejected() {
        let mut proj = make_recharge_project();
        proj.id = "x".repeat(257);
        let result = validate_create_recharge_project(fake_create(), proj);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Recharge project ID must be 256 characters or fewer"
        );
    }

    #[test]
    fn recharge_project_name_exactly_256_accepted() {
        let mut proj = make_recharge_project();
        proj.name = "x".repeat(256);
        let result = validate_create_recharge_project(fake_create(), proj);
        assert!(is_valid(&result));
    }

    #[test]
    fn recharge_project_name_257_rejected() {
        let mut proj = make_recharge_project();
        proj.name = "x".repeat(257);
        let result = validate_create_recharge_project(fake_create(), proj);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Recharge project name must be 256 characters or fewer"
        );
    }

    #[test]
    fn recharge_project_aquifer_id_exactly_64_accepted() {
        let mut proj = make_recharge_project();
        proj.aquifer_id = "x".repeat(64);
        let result = validate_create_recharge_project(fake_create(), proj);
        assert!(is_valid(&result));
    }

    #[test]
    fn recharge_project_aquifer_id_65_rejected() {
        let mut proj = make_recharge_project();
        proj.aquifer_id = "x".repeat(257);
        let result = validate_create_recharge_project(fake_create(), proj);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Aquifer ID must be 256 characters or fewer"
        );
    }

    // ========================================================================
    // LINK TAG LENGTH VALIDATION TESTS
    // ========================================================================

    fn validate_link_tag_for(
        link_type: LinkTypes,
        tag_bytes: Vec<u8>,
    ) -> ExternResult<ValidateCallbackResult> {
        let tag = LinkTag(tag_bytes);
        match link_type {
            LinkTypes::AllSystems
            | LinkTypes::OwnerToSystem
            | LinkTypes::AllTanks
            | LinkTypes::OwnerToTank
            | LinkTypes::SystemToHarvestRecord
            | LinkTypes::AgentToHarvestRecord
            | LinkTypes::SystemToTank
            | LinkTypes::AllRechargeProjects
            | LinkTypes::GeoIndex => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(format!(
                        "{:?} link tag too long (max 256 bytes)",
                        link_type
                    )));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::SystemTypeToSystem => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(format!(
                        "{:?} link tag too long (max 512 bytes)",
                        link_type
                    )));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        }
    }

    #[test]
    fn link_all_systems_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::AllSystems, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_all_systems_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::AllSystems, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    #[test]
    fn link_owner_to_system_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::OwnerToSystem, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_owner_to_system_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::OwnerToSystem, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    #[test]
    fn link_system_type_to_system_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::SystemTypeToSystem, vec![0u8; 512]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_system_type_to_system_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::SystemTypeToSystem, vec![0u8; 513]);
        assert!(is_invalid(&result));
    }

    #[test]
    fn link_all_tanks_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::AllTanks, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_all_tanks_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::AllTanks, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    #[test]
    fn link_owner_to_tank_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::OwnerToTank, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_owner_to_tank_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::OwnerToTank, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    #[test]
    fn link_system_to_harvest_record_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::SystemToHarvestRecord, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_system_to_harvest_record_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::SystemToHarvestRecord, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    #[test]
    fn link_agent_to_harvest_record_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::AgentToHarvestRecord, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_agent_to_harvest_record_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::AgentToHarvestRecord, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    #[test]
    fn link_system_to_tank_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::SystemToTank, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_system_to_tank_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::SystemToTank, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    #[test]
    fn link_all_recharge_projects_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::AllRechargeProjects, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_all_recharge_projects_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::AllRechargeProjects, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }
}
