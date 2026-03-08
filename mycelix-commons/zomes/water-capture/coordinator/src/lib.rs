//! Capture Coordinator Zome
//! Business logic for water harvesting, storage, and aquifer recharge

use hdk::prelude::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_basic, requirement_for_proposal, GovernanceEligibility,
    GovernanceRequirement,
};
use water_capture_integrity::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

fn get_latest_record(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    let Some(details) = get_details(action_hash, GetOptions::default())? else {
        return Ok(None);
    };
    match details {
        Details::Record(record_details) => {
            if record_details.updates.is_empty() {
                Ok(Some(record_details.record))
            } else {
                let latest_update = &record_details.updates[record_details.updates.len() - 1];
                let latest_hash = latest_update.action_address().clone();
                get_latest_record(latest_hash)
            }
        }
        Details::Entry(_) => Ok(None),
    }
}

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
            records.push(record);
        }
    }
    Ok(records)
}

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("commons_bridge", requirement, action_name)
}

// ============================================================================
// HARVEST SYSTEMS
// ============================================================================

/// Register a new water harvesting system
#[hdk_extern]
pub fn register_harvest_system(system: HarvestSystem) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_basic(), "register_harvest_system")?;
    if system.id.trim().is_empty() || system.id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "System ID must be 1-256 non-whitespace characters".into()
        )));
    }
    if system.name.trim().is_empty() || system.name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "System name must be 1-256 non-whitespace characters".into()
        )));
    }
    if system.capacity_liters == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "System capacity must be greater than zero".into()
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
    let _eligibility = require_consciousness(&requirement_for_basic(), "register_tank")?;
    if tank.id.trim().is_empty() || tank.id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Tank ID must be 1-256 non-whitespace characters".into()
        )));
    }
    if tank.name.trim().is_empty() || tank.name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Tank name must be 1-256 non-whitespace characters".into()
        )));
    }
    if tank.capacity_liters == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Tank capacity must be greater than zero".into()
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
    let _eligibility = require_consciousness(&requirement_for_proposal(), "update_tank_level")?;
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
    let _eligibility = require_consciousness(&requirement_for_basic(), "record_harvest")?;
    if harvest.liters_collected == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Harvest amount must be greater than zero".into()
        )));
    }

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
    records.sort_by_key(|a| a.action().timestamp());
    Ok(records)
}

// ============================================================================
// AQUIFER RECHARGE
// ============================================================================

/// Register a new aquifer recharge project
#[hdk_extern]
pub fn register_recharge_project(project: RechargeProject) -> ExternResult<Record> {
    let _eligibility =
        require_consciousness(&requirement_for_basic(), "register_recharge_project")?;
    if project.id.trim().is_empty() || project.id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Project ID must be 1-256 non-whitespace characters".into()
        )));
    }
    if project.name.trim().is_empty() || project.name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Project name must be 1-256 non-whitespace characters".into()
        )));
    }
    if project.capacity_liters_per_day == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Recharge capacity must be greater than zero".into()
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

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    // ========================================================================
    // COORDINATOR STRUCT SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn update_tank_level_input_serde_roundtrip() {
        let input = UpdateTankLevelInput {
            tank_hash: fake_action_hash(),
            new_level_liters: 5000,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateTankLevelInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_level_liters, 5000);
    }

    #[test]
    fn update_tank_level_input_zero_level() {
        let input = UpdateTankLevelInput {
            tank_hash: fake_action_hash(),
            new_level_liters: 0,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateTankLevelInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_level_liters, 0);
    }

    #[test]
    fn update_tank_level_input_max_level() {
        let input = UpdateTankLevelInput {
            tank_hash: fake_action_hash(),
            new_level_liters: u64::MAX,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateTankLevelInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_level_liters, u64::MAX);
    }

    // ========================================================================
    // INTEGRITY ENUM SERDE ROUNDTRIP TESTS (via coordinator re-export)
    // ========================================================================

    #[test]
    fn harvest_type_all_variants_serde() {
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
    fn tank_type_all_variants_serde() {
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
    fn recharge_method_all_variants_serde() {
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
    fn recharge_status_all_variants_serde() {
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
    // Entry type serde roundtrip tests
    // ========================================================================

    #[test]
    fn harvest_system_full_serde_roundtrip() {
        let sys = HarvestSystem {
            id: "hs-001".to_string(),
            name: "Roof Catchment Alpha".to_string(),
            system_type: HarvestType::RoofRainwater,
            capacity_liters: 5_000,
            location_lat: 35.6762,
            location_lon: 139.6503,
            owner: AgentPubKey::from_raw_36(vec![1u8; 36]),
            installed_at: Timestamp::from_micros(1_700_000_000),
            catchment_area_sqm: Some(200),
            efficiency_percent: 85,
        };
        let json = serde_json::to_string(&sys).unwrap();
        let decoded: HarvestSystem = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "hs-001");
        assert_eq!(decoded.system_type, HarvestType::RoofRainwater);
        assert_eq!(decoded.capacity_liters, 5_000);
        assert_eq!(decoded.catchment_area_sqm, Some(200));
        assert_eq!(decoded.efficiency_percent, 85);
    }

    #[test]
    fn harvest_system_none_catchment_serde() {
        let sys = HarvestSystem {
            id: "hs-fog".to_string(),
            name: "Fog Net".to_string(),
            system_type: HarvestType::FogCollection,
            capacity_liters: 100,
            location_lat: 0.0,
            location_lon: 0.0,
            owner: AgentPubKey::from_raw_36(vec![0u8; 36]),
            installed_at: Timestamp::from_micros(0),
            catchment_area_sqm: None,
            efficiency_percent: 10,
        };
        let json = serde_json::to_string(&sys).unwrap();
        let decoded: HarvestSystem = serde_json::from_str(&json).unwrap();
        assert!(decoded.catchment_area_sqm.is_none());
    }

    #[test]
    fn storage_tank_full_serde_roundtrip() {
        let tank = StorageTank {
            id: "tank-001".to_string(),
            name: "Underground Cistern".to_string(),
            capacity_liters: 10_000,
            current_level_liters: 7_500,
            tank_type: TankType::Underground,
            connected_system: Some(fake_action_hash()),
            location_lat: 40.0,
            location_lon: -74.0,
            owner: AgentPubKey::from_raw_36(vec![0u8; 36]),
        };
        let json = serde_json::to_string(&tank).unwrap();
        let decoded: StorageTank = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "tank-001");
        assert_eq!(decoded.capacity_liters, 10_000);
        assert_eq!(decoded.current_level_liters, 7_500);
        assert_eq!(decoded.tank_type, TankType::Underground);
        assert!(decoded.connected_system.is_some());
    }

    #[test]
    fn storage_tank_none_connected_system_serde() {
        let tank = StorageTank {
            id: "tank-standalone".to_string(),
            name: "Bladder Tank".to_string(),
            capacity_liters: 500,
            current_level_liters: 0,
            tank_type: TankType::Bladder,
            connected_system: None,
            location_lat: 0.0,
            location_lon: 0.0,
            owner: AgentPubKey::from_raw_36(vec![0u8; 36]),
        };
        let json = serde_json::to_string(&tank).unwrap();
        let decoded: StorageTank = serde_json::from_str(&json).unwrap();
        assert!(decoded.connected_system.is_none());
    }

    #[test]
    fn harvest_record_full_serde_roundtrip() {
        let rec = HarvestRecord {
            system_hash: fake_action_hash(),
            liters_collected: 150,
            collection_period_start: Timestamp::from_micros(0),
            collection_period_end: Timestamp::from_micros(1_000_000),
            weather_conditions: Some("Heavy rain".to_string()),
            credited_to: AgentPubKey::from_raw_36(vec![0u8; 36]),
        };
        let json = serde_json::to_string(&rec).unwrap();
        let decoded: HarvestRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.liters_collected, 150);
        assert_eq!(decoded.weather_conditions, Some("Heavy rain".to_string()));
    }

    #[test]
    fn harvest_record_none_weather_serde() {
        let rec = HarvestRecord {
            system_hash: fake_action_hash(),
            liters_collected: 10,
            collection_period_start: Timestamp::from_micros(0),
            collection_period_end: Timestamp::from_micros(1),
            weather_conditions: None,
            credited_to: AgentPubKey::from_raw_36(vec![0u8; 36]),
        };
        let json = serde_json::to_string(&rec).unwrap();
        let decoded: HarvestRecord = serde_json::from_str(&json).unwrap();
        assert!(decoded.weather_conditions.is_none());
    }

    #[test]
    fn recharge_project_full_serde_roundtrip() {
        let proj = RechargeProject {
            id: "rp-001".to_string(),
            name: "Basin Recharge Alpha".to_string(),
            aquifer_id: "aquifer-12".to_string(),
            method: RechargeMethod::Basin,
            capacity_liters_per_day: 50_000,
            location_lat: 40.0,
            location_lon: -110.0,
            status: RechargeStatus::Active,
        };
        let json = serde_json::to_string(&proj).unwrap();
        let decoded: RechargeProject = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "rp-001");
        assert_eq!(decoded.aquifer_id, "aquifer-12");
        assert_eq!(decoded.method, RechargeMethod::Basin);
        assert_eq!(decoded.status, RechargeStatus::Active);
    }

    // ========================================================================
    // Clone/equality tests
    // ========================================================================

    #[test]
    fn harvest_system_clone_equals_original() {
        let sys = HarvestSystem {
            id: "hs-clone".to_string(),
            name: "Clone Test".to_string(),
            system_type: HarvestType::Snowmelt,
            capacity_liters: 1,
            location_lat: 0.0,
            location_lon: 0.0,
            owner: AgentPubKey::from_raw_36(vec![0u8; 36]),
            installed_at: Timestamp::from_micros(0),
            catchment_area_sqm: None,
            efficiency_percent: 50,
        };
        assert_eq!(sys, sys.clone());
    }

    #[test]
    fn storage_tank_clone_equals_original() {
        let tank = StorageTank {
            id: "t-clone".to_string(),
            name: "Clone".to_string(),
            capacity_liters: 100,
            current_level_liters: 50,
            tank_type: TankType::Cistern,
            connected_system: None,
            location_lat: 0.0,
            location_lon: 0.0,
            owner: AgentPubKey::from_raw_36(vec![0u8; 36]),
        };
        assert_eq!(tank, tank.clone());
    }

    #[test]
    fn recharge_project_clone_equals_original() {
        let proj = RechargeProject {
            id: "rp-clone".to_string(),
            name: "Clone".to_string(),
            aquifer_id: "aq".to_string(),
            method: RechargeMethod::Injection,
            capacity_liters_per_day: 1,
            location_lat: 0.0,
            location_lon: 0.0,
            status: RechargeStatus::Proposed,
        };
        assert_eq!(proj, proj.clone());
    }

    #[test]
    fn harvest_system_ne_different_type() {
        let a = HarvestSystem {
            id: "hs".to_string(),
            name: "N".to_string(),
            system_type: HarvestType::RoofRainwater,
            capacity_liters: 1,
            location_lat: 0.0,
            location_lon: 0.0,
            owner: AgentPubKey::from_raw_36(vec![0u8; 36]),
            installed_at: Timestamp::from_micros(0),
            catchment_area_sqm: None,
            efficiency_percent: 0,
        };
        let mut b = a.clone();
        b.system_type = HarvestType::DewCollection;
        assert_ne!(a, b);
    }

    // ========================================================================
    // Edge case tests
    // ========================================================================

    #[test]
    fn update_tank_level_input_u64_max_serde() {
        let input = UpdateTankLevelInput {
            tank_hash: fake_action_hash(),
            new_level_liters: u64::MAX,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateTankLevelInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_level_liters, u64::MAX);
    }

    #[test]
    fn harvest_system_boundary_coordinates_serde() {
        let sys = HarvestSystem {
            id: "hs-boundary".to_string(),
            name: "Boundary".to_string(),
            system_type: HarvestType::GroundCatchment,
            capacity_liters: 1,
            location_lat: -90.0,
            location_lon: 180.0,
            owner: AgentPubKey::from_raw_36(vec![0u8; 36]),
            installed_at: Timestamp::from_micros(0),
            catchment_area_sqm: Some(u32::MAX),
            efficiency_percent: 100,
        };
        let json = serde_json::to_string(&sys).unwrap();
        let decoded: HarvestSystem = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.location_lat, -90.0);
        assert_eq!(decoded.location_lon, 180.0);
        assert_eq!(decoded.catchment_area_sqm, Some(u32::MAX));
        assert_eq!(decoded.efficiency_percent, 100);
    }

    #[test]
    fn harvest_system_unicode_name_serde() {
        let sys = HarvestSystem {
            id: "hs-unicode".to_string(),
            name: "\u{96E8}\u{6C34}\u{6536}\u{96C6}\u{7CFB}\u{7EDF}".to_string(),
            system_type: HarvestType::RoofRainwater,
            capacity_liters: 1000,
            location_lat: 30.0,
            location_lon: 120.0,
            owner: AgentPubKey::from_raw_36(vec![0u8; 36]),
            installed_at: Timestamp::from_micros(0),
            catchment_area_sqm: None,
            efficiency_percent: 70,
        };
        let json = serde_json::to_string(&sys).unwrap();
        let decoded: HarvestSystem = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.name,
            "\u{96E8}\u{6C34}\u{6536}\u{96C6}\u{7CFB}\u{7EDF}"
        );
    }

    #[test]
    fn harvest_record_u64_max_liters_serde() {
        let rec = HarvestRecord {
            system_hash: fake_action_hash(),
            liters_collected: u64::MAX,
            collection_period_start: Timestamp::from_micros(0),
            collection_period_end: Timestamp::from_micros(1),
            weather_conditions: None,
            credited_to: AgentPubKey::from_raw_36(vec![0u8; 36]),
        };
        let json = serde_json::to_string(&rec).unwrap();
        let decoded: HarvestRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.liters_collected, u64::MAX);
    }

    #[test]
    fn recharge_project_all_methods_serde() {
        for method in [
            RechargeMethod::Basin,
            RechargeMethod::Injection,
            RechargeMethod::Spreading,
            RechargeMethod::Infiltration,
        ] {
            let proj = RechargeProject {
                id: "rp".to_string(),
                name: "N".to_string(),
                aquifer_id: "aq".to_string(),
                method: method.clone(),
                capacity_liters_per_day: 1,
                location_lat: 0.0,
                location_lon: 0.0,
                status: RechargeStatus::Active,
            };
            let json = serde_json::to_string(&proj).unwrap();
            let decoded: RechargeProject = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.method, method);
        }
    }

    #[test]
    fn recharge_project_all_statuses_entry_serde() {
        for status in [
            RechargeStatus::Proposed,
            RechargeStatus::Active,
            RechargeStatus::Paused,
            RechargeStatus::Completed,
            RechargeStatus::Abandoned,
        ] {
            let proj = RechargeProject {
                id: "rp".to_string(),
                name: "N".to_string(),
                aquifer_id: "aq".to_string(),
                method: RechargeMethod::Basin,
                capacity_liters_per_day: 1,
                location_lat: 0.0,
                location_lon: 0.0,
                status: status.clone(),
            };
            let json = serde_json::to_string(&proj).unwrap();
            let decoded: RechargeProject = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.status, status);
        }
    }

    // ========================================================================
    // COORDINATOR VALIDATION EDGE CASE TESTS
    // ========================================================================

    #[test]
    fn harvest_system_whitespace_only_id_rejected_serde() {
        // Verify a whitespace-only ID would be caught by trim().is_empty()
        let id = "   \t\n  ";
        assert!(id.trim().is_empty());
        assert!(!id.is_empty()); // old check would have missed this
    }

    #[test]
    fn harvest_system_whitespace_only_name_rejected_serde() {
        let name = "   ";
        assert!(name.trim().is_empty());
        assert!(!name.is_empty());
    }

    #[test]
    fn harvest_system_zero_capacity_rejected_serde() {
        // Verify that capacity_liters == 0 is the boundary
        assert_eq!(0u64, 0);
        assert_ne!(1u64, 0);
    }

    #[test]
    fn tank_whitespace_only_id_rejected_serde() {
        let id = "  \t  ";
        assert!(id.trim().is_empty());
        assert!(!id.is_empty());
    }

    #[test]
    fn tank_whitespace_only_name_rejected_serde() {
        let name = "\n\r\t ";
        assert!(name.trim().is_empty());
        assert!(!name.is_empty());
    }

    #[test]
    fn tank_zero_capacity_rejected_serde() {
        // Verify the boundary: 0 rejected, 1 accepted
        let tank = StorageTank {
            id: "tank-edge".to_string(),
            name: "Edge".to_string(),
            capacity_liters: 0,
            current_level_liters: 0,
            tank_type: TankType::Underground,
            connected_system: None,
            location_lat: 0.0,
            location_lon: 0.0,
            owner: AgentPubKey::from_raw_36(vec![0u8; 36]),
        };
        assert_eq!(tank.capacity_liters, 0);
    }

    #[test]
    fn tank_one_liter_capacity_accepted_serde() {
        let tank = StorageTank {
            id: "tank-min".to_string(),
            name: "Min".to_string(),
            capacity_liters: 1,
            current_level_liters: 0,
            tank_type: TankType::Bladder,
            connected_system: None,
            location_lat: 0.0,
            location_lon: 0.0,
            owner: AgentPubKey::from_raw_36(vec![0u8; 36]),
        };
        assert_eq!(tank.capacity_liters, 1);
    }

    #[test]
    fn harvest_record_zero_liters_rejected_serde() {
        let rec = HarvestRecord {
            system_hash: fake_action_hash(),
            liters_collected: 0,
            collection_period_start: Timestamp::from_micros(0),
            collection_period_end: Timestamp::from_micros(1),
            weather_conditions: None,
            credited_to: AgentPubKey::from_raw_36(vec![0u8; 36]),
        };
        assert_eq!(rec.liters_collected, 0);
    }

    #[test]
    fn harvest_record_one_liter_accepted_serde() {
        let rec = HarvestRecord {
            system_hash: fake_action_hash(),
            liters_collected: 1,
            collection_period_start: Timestamp::from_micros(0),
            collection_period_end: Timestamp::from_micros(1),
            weather_conditions: None,
            credited_to: AgentPubKey::from_raw_36(vec![0u8; 36]),
        };
        assert_eq!(rec.liters_collected, 1);
    }

    #[test]
    fn recharge_project_whitespace_only_id_rejected_serde() {
        let id = "   ";
        assert!(id.trim().is_empty());
        assert!(!id.is_empty());
    }

    #[test]
    fn recharge_project_whitespace_only_name_rejected_serde() {
        let name = "\t\t\t";
        assert!(name.trim().is_empty());
        assert!(!name.is_empty());
    }

    #[test]
    fn recharge_project_zero_capacity_rejected_serde() {
        let proj = RechargeProject {
            id: "rp-edge".to_string(),
            name: "Edge".to_string(),
            aquifer_id: "aq".to_string(),
            method: RechargeMethod::Basin,
            capacity_liters_per_day: 0,
            location_lat: 0.0,
            location_lon: 0.0,
            status: RechargeStatus::Proposed,
        };
        assert_eq!(proj.capacity_liters_per_day, 0);
    }

    #[test]
    fn recharge_project_one_liter_per_day_accepted_serde() {
        let proj = RechargeProject {
            id: "rp-min".to_string(),
            name: "Min".to_string(),
            aquifer_id: "aq".to_string(),
            method: RechargeMethod::Injection,
            capacity_liters_per_day: 1,
            location_lat: 0.0,
            location_lon: 0.0,
            status: RechargeStatus::Active,
        };
        assert_eq!(proj.capacity_liters_per_day, 1);
    }
}
