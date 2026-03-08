//! Units Coordinator Zome
//! Business logic for buildings and housing units.

use hdk::prelude::*;
use housing_units_integrity::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_proposal, requirement_for_voting, GovernanceEligibility,
    GovernanceRequirement,
};

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("commons_bridge", requirement, action_name)
}

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Register a new building in the cooperative

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

#[hdk_extern]
pub fn register_building(building: Building) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_proposal(), "register_building")?;
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

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created building".into()
    )))
}

/// Register a new unit within a building
#[hdk_extern]
pub fn register_unit(unit: Unit) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_proposal(), "register_unit")?;
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

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
    let _eligibility = require_consciousness(&requirement_for_proposal(), "update_unit_status")?;
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

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
        if let Some(record) = get_latest_record(action_hash)? {
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
        if let Some(record) = get_latest_record(action_hash)? {
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
    let _eligibility = require_consciousness(&requirement_for_voting(), "assign_occupant")?;
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

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated unit".into()
    )))
}

/// Vacate a unit, removing the occupant
#[hdk_extern]
pub fn vacate_unit(unit_action_hash: ActionHash) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_proposal(), "vacate_unit")?;
    let record = get_latest_record(unit_action_hash.clone())?
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

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated unit".into()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    fn make_building() -> Building {
        Building {
            id: "bldg-001".to_string(),
            name: "Oak Terrace".to_string(),
            address: "123 Main St".to_string(),
            location_lat: 45.5,
            location_lon: -122.6,
            total_units: 24,
            year_built: Some(1985),
            building_type: BuildingType::Apartment,
            cooperative_hash: None,
        }
    }

    fn make_unit() -> Unit {
        Unit {
            building_hash: fake_action_hash(),
            unit_number: "4B".to_string(),
            unit_type: UnitType::TwoBedroom,
            square_meters: 75,
            floor: 4,
            bedrooms: 2,
            bathrooms: 1,
            accessibility_features: vec![AccessFeature::Elevator, AccessFeature::WideDoorways],
            current_occupant: Some(fake_agent()),
            status: UnitStatus::Occupied,
        }
    }

    // ── Coordinator struct serde roundtrips ────────────────────────────

    #[test]
    fn update_unit_status_input_serde_roundtrip() {
        let input = UpdateUnitStatusInput {
            unit_action_hash: fake_action_hash(),
            new_status: UnitStatus::UnderMaintenance,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateUnitStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_status, UnitStatus::UnderMaintenance);
    }

    #[test]
    fn assign_occupant_input_serde_roundtrip() {
        let input = AssignOccupantInput {
            unit_action_hash: fake_action_hash(),
            occupant: fake_agent(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AssignOccupantInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.occupant, fake_agent());
    }

    // ── Integrity enum serde roundtrips ────────────────────────────────

    #[test]
    fn building_type_all_variants_serde() {
        let variants = vec![
            BuildingType::Apartment,
            BuildingType::Townhouse,
            BuildingType::SingleFamily,
            BuildingType::Duplex,
            BuildingType::CoHousing,
            BuildingType::MixedUse,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let decoded: BuildingType = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, v);
        }
    }

    #[test]
    fn unit_type_all_variants_serde() {
        let variants = vec![
            UnitType::Studio,
            UnitType::OneBedroom,
            UnitType::TwoBedroom,
            UnitType::ThreeBedroom,
            UnitType::FourPlus,
            UnitType::Accessible,
            UnitType::Family,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let decoded: UnitType = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, v);
        }
    }

    #[test]
    fn access_feature_all_variants_serde() {
        let variants = vec![
            AccessFeature::WheelchairAccessible,
            AccessFeature::Elevator,
            AccessFeature::GrabBars,
            AccessFeature::WideDoorways,
            AccessFeature::LowCounters,
            AccessFeature::VisualAlerts,
            AccessFeature::HearingLoop,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let decoded: AccessFeature = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, v);
        }
    }

    #[test]
    fn unit_status_all_variants_serde() {
        let variants = vec![
            UnitStatus::Available,
            UnitStatus::Occupied,
            UnitStatus::UnderMaintenance,
            UnitStatus::Reserved,
            UnitStatus::Renovation,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let decoded: UnitStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, v);
        }
    }

    // ── Integrity entry struct serde roundtrips ────────────────────────

    #[test]
    fn building_serde_roundtrip() {
        let building = make_building();
        let json = serde_json::to_string(&building).unwrap();
        let decoded: Building = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, building);
    }

    #[test]
    fn building_no_year_built_no_cooperative() {
        let building = Building {
            id: "bldg-002".to_string(),
            name: "New Build".to_string(),
            address: "456 Elm St".to_string(),
            location_lat: 0.0,
            location_lon: 0.0,
            total_units: 1,
            year_built: None,
            building_type: BuildingType::SingleFamily,
            cooperative_hash: None,
        };
        let json = serde_json::to_string(&building).unwrap();
        let decoded: Building = serde_json::from_str(&json).unwrap();
        assert!(decoded.year_built.is_none());
        assert!(decoded.cooperative_hash.is_none());
    }

    #[test]
    fn building_with_cooperative_hash() {
        let mut building = make_building();
        building.cooperative_hash = Some(fake_action_hash());
        let json = serde_json::to_string(&building).unwrap();
        let decoded: Building = serde_json::from_str(&json).unwrap();
        assert!(decoded.cooperative_hash.is_some());
    }

    #[test]
    fn building_boundary_lat_lon() {
        let building = Building {
            id: "edge".to_string(),
            name: "North Pole".to_string(),
            address: "Arctic".to_string(),
            location_lat: 90.0,
            location_lon: 180.0,
            total_units: 1,
            year_built: None,
            building_type: BuildingType::CoHousing,
            cooperative_hash: None,
        };
        let json = serde_json::to_string(&building).unwrap();
        let decoded: Building = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.location_lat, 90.0);
        assert_eq!(decoded.location_lon, 180.0);
    }

    #[test]
    fn building_negative_lat_lon() {
        let building = Building {
            id: "south".to_string(),
            name: "South Pole".to_string(),
            address: "Antarctic".to_string(),
            location_lat: -90.0,
            location_lon: -180.0,
            total_units: 1,
            year_built: None,
            building_type: BuildingType::Duplex,
            cooperative_hash: None,
        };
        let json = serde_json::to_string(&building).unwrap();
        let decoded: Building = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.location_lat, -90.0);
        assert_eq!(decoded.location_lon, -180.0);
    }

    #[test]
    fn building_unicode_name_address() {
        let building = Building {
            id: "uni".to_string(),
            name: "\u{5927}\u{5EC8}".to_string(),
            address: "\u{00C9}dimbourg, \u{00C9}cosse".to_string(),
            location_lat: 55.95,
            location_lon: -3.19,
            total_units: 10,
            year_built: Some(1800),
            building_type: BuildingType::Townhouse,
            cooperative_hash: None,
        };
        let json = serde_json::to_string(&building).unwrap();
        let decoded: Building = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, "\u{5927}\u{5EC8}");
    }

    #[test]
    fn building_u16_max_total_units() {
        let mut building = make_building();
        building.total_units = u16::MAX;
        building.year_built = Some(u16::MAX);
        let json = serde_json::to_string(&building).unwrap();
        let decoded: Building = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.total_units, u16::MAX);
        assert_eq!(decoded.year_built, Some(u16::MAX));
    }

    #[test]
    fn unit_serde_roundtrip() {
        let unit = make_unit();
        let json = serde_json::to_string(&unit).unwrap();
        let decoded: Unit = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, unit);
    }

    #[test]
    fn unit_no_occupant_serde_roundtrip() {
        let unit = Unit {
            building_hash: fake_action_hash(),
            unit_number: "1A".to_string(),
            unit_type: UnitType::Studio,
            square_meters: 30,
            floor: 1,
            bedrooms: 0,
            bathrooms: 1,
            accessibility_features: vec![],
            current_occupant: None,
            status: UnitStatus::Available,
        };
        let json = serde_json::to_string(&unit).unwrap();
        let decoded: Unit = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, unit);
    }

    #[test]
    fn unit_all_accessibility_features() {
        let unit = Unit {
            building_hash: fake_action_hash(),
            unit_number: "ADA-1".to_string(),
            unit_type: UnitType::Accessible,
            square_meters: 80,
            floor: 1,
            bedrooms: 1,
            bathrooms: 1,
            accessibility_features: vec![
                AccessFeature::WheelchairAccessible,
                AccessFeature::Elevator,
                AccessFeature::GrabBars,
                AccessFeature::WideDoorways,
                AccessFeature::LowCounters,
                AccessFeature::VisualAlerts,
                AccessFeature::HearingLoop,
            ],
            current_occupant: None,
            status: UnitStatus::Available,
        };
        let json = serde_json::to_string(&unit).unwrap();
        let decoded: Unit = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.accessibility_features.len(), 7);
    }

    #[test]
    fn unit_u32_max_square_meters() {
        let mut unit = make_unit();
        unit.square_meters = u32::MAX;
        unit.floor = u8::MAX;
        unit.bedrooms = u8::MAX;
        unit.bathrooms = u8::MAX;
        let json = serde_json::to_string(&unit).unwrap();
        let decoded: Unit = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.square_meters, u32::MAX);
        assert_eq!(decoded.floor, u8::MAX);
        assert_eq!(decoded.bedrooms, u8::MAX);
        assert_eq!(decoded.bathrooms, u8::MAX);
    }

    #[test]
    fn unit_unicode_unit_number() {
        let mut unit = make_unit();
        unit.unit_number = "\u{00C9}tage 3, Apt \u{00B0}7".to_string();
        let json = serde_json::to_string(&unit).unwrap();
        let decoded: Unit = serde_json::from_str(&json).unwrap();
        assert!(decoded.unit_number.contains('\u{00C9}'));
    }

    // ── UpdateUnitStatusInput with all statuses ───────────────────────

    #[test]
    fn update_unit_status_all_statuses_serde() {
        for status in [
            UnitStatus::Available,
            UnitStatus::Occupied,
            UnitStatus::UnderMaintenance,
            UnitStatus::Reserved,
            UnitStatus::Renovation,
        ] {
            let input = UpdateUnitStatusInput {
                unit_action_hash: fake_action_hash(),
                new_status: status.clone(),
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: UpdateUnitStatusInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.new_status, status);
        }
    }

    // ── Clone / equality tests ────────────────────────────────────────

    #[test]
    fn building_clone_equals() {
        let building = make_building();
        let cloned = building.clone();
        assert_eq!(building, cloned);
    }

    #[test]
    fn unit_clone_equals() {
        let unit = make_unit();
        let cloned = unit.clone();
        assert_eq!(unit, cloned);
    }

    #[test]
    fn unit_status_clone_eq() {
        let s = UnitStatus::Renovation;
        assert_eq!(s.clone(), UnitStatus::Renovation);
    }

    #[test]
    fn building_type_clone_eq() {
        let bt = BuildingType::MixedUse;
        assert_eq!(bt.clone(), BuildingType::MixedUse);
    }

    // ── Edge case: empty unit number ──────────────────────────────────

    #[test]
    fn unit_empty_unit_number() {
        let unit = Unit {
            building_hash: fake_action_hash(),
            unit_number: String::new(),
            unit_type: UnitType::Studio,
            square_meters: 0,
            floor: 0,
            bedrooms: 0,
            bathrooms: 0,
            accessibility_features: vec![],
            current_occupant: None,
            status: UnitStatus::Available,
        };
        let json = serde_json::to_string(&unit).unwrap();
        let decoded: Unit = serde_json::from_str(&json).unwrap();
        assert!(decoded.unit_number.is_empty());
        assert_eq!(decoded.square_meters, 0);
    }
}
