// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Units Integrity Zome
//! Defines entry types and validation for buildings and housing units.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Type of building in the cooperative
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum BuildingType {
    Apartment,
    Townhouse,
    SingleFamily,
    Duplex,
    CoHousing,
    MixedUse,
}

/// A building managed by the housing cooperative
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Building {
    pub id: String,
    pub name: String,
    pub address: String,
    pub location_lat: f64,
    pub location_lon: f64,
    pub total_units: u16,
    pub year_built: Option<u16>,
    pub building_type: BuildingType,
    pub cooperative_hash: Option<ActionHash>,
}

/// Type of housing unit
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum UnitType {
    Studio,
    OneBedroom,
    TwoBedroom,
    ThreeBedroom,
    FourPlus,
    Accessible,
    Family,
}

/// Accessibility features available in a unit
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AccessFeature {
    WheelchairAccessible,
    Elevator,
    GrabBars,
    WideDoorways,
    LowCounters,
    VisualAlerts,
    HearingLoop,
}

/// Current status of a housing unit
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum UnitStatus {
    Available,
    Occupied,
    UnderMaintenance,
    Reserved,
    Renovation,
}

/// A housing unit within a building
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Unit {
    pub building_hash: ActionHash,
    pub unit_number: String,
    pub unit_type: UnitType,
    pub square_meters: u32,
    pub floor: u8,
    pub bedrooms: u8,
    pub bathrooms: u8,
    pub accessibility_features: Vec<AccessFeature>,
    pub current_occupant: Option<AgentPubKey>,
    pub status: UnitStatus,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Building(Building),
    HousingUnit(Unit),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// All buildings anchor
    AllBuildings,
    /// Building to its units
    BuildingToUnit,
    /// Available units anchor
    AvailableUnits,
    /// Occupant to their unit
    OccupantToUnit,
    /// Building type index
    BuildingTypeToBuilding,
    /// Geohash spatial index
    GeoIndex,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Building(building) => validate_create_building(action, building),
                EntryTypes::HousingUnit(unit) => validate_create_unit(action, unit),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Building(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::HousingUnit(unit) => validate_update_unit(unit),
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
            LinkTypes::AllBuildings => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllBuildings link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::BuildingToUnit => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "BuildingToUnit link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AvailableUnits => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AvailableUnits link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::OccupantToUnit => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "OccupantToUnit link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::BuildingTypeToBuilding => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "BuildingTypeToBuilding link tag too long (max 512 bytes)".into(),
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

fn validate_create_building(
    _action: Create,
    building: Building,
) -> ExternResult<ValidateCallbackResult> {
    if building.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Building ID cannot be empty".into(),
        ));
    }
    if building.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Building ID must be at most 256 characters".into(),
        ));
    }
    if building.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Building name cannot be empty".into(),
        ));
    }
    if building.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Building name must be at most 256 characters".into(),
        ));
    }
    if building.address.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Building address cannot be empty".into(),
        ));
    }
    if building.address.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Building address must be at most 256 characters".into(),
        ));
    }
    if !building.location_lat.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be a finite number".into(),
        ));
    }
    if building.location_lat < -90.0 || building.location_lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be between -90 and 90".into(),
        ));
    }
    if !building.location_lon.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be a finite number".into(),
        ));
    }
    if building.location_lon < -180.0 || building.location_lon > 180.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be between -180 and 180".into(),
        ));
    }
    if building.total_units == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Building must have at least one unit".into(),
        ));
    }
    if let Some(year) = building.year_built {
        if !(1800..=2100).contains(&year) {
            return Ok(ValidateCallbackResult::Invalid(
                "Year built must be between 1800 and 2100".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_unit(_action: Create, unit: Unit) -> ExternResult<ValidateCallbackResult> {
    if unit.unit_number.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Unit number cannot be empty".into(),
        ));
    }
    if unit.unit_number.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid(
            "Unit number must be at most 128 characters".into(),
        ));
    }
    if unit.square_meters == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Square meters must be greater than 0".into(),
        ));
    }
    if unit.bathrooms == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Unit must have at least one bathroom".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_unit(unit: Unit) -> ExternResult<ValidateCallbackResult> {
    if unit.unit_number.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Unit number cannot be empty".into(),
        ));
    }
    if unit.unit_number.len() > 128 {
        return Ok(ValidateCallbackResult::Invalid(
            "Unit number must be at most 128 characters".into(),
        ));
    }
    if unit.square_meters == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Square meters must be greater than 0".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Factory Functions ==========

    fn valid_building() -> Building {
        Building {
            id: "building-001".into(),
            name: "Community Commons".into(),
            address: "123 Cooperative Way".into(),
            location_lat: 42.3601,
            location_lon: -71.0589,
            total_units: 24,
            year_built: Some(2020),
            building_type: BuildingType::Apartment,
            cooperative_hash: None,
        }
    }

    fn valid_unit() -> Unit {
        Unit {
            building_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            unit_number: "101".into(),
            unit_type: UnitType::OneBedroom,
            square_meters: 65,
            floor: 1,
            bedrooms: 1,
            bathrooms: 1,
            accessibility_features: vec![],
            current_occupant: None,
            status: UnitStatus::Available,
        }
    }

    fn mock_create_action() -> Create {
        Create {
            author: AgentPubKey::from_raw_36(vec![0u8; 36]),
            timestamp: Timestamp::from_micros(1_000_000),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0u8; 36]),
            entry_type: EntryType::App(AppEntryDef {
                entry_index: 0.into(),
                zome_index: 0.into(),
                visibility: EntryVisibility::Public,
            }),
            entry_hash: EntryHash::from_raw_36(vec![0u8; 36]),
            weight: Default::default(),
        }
    }

    // ========== Building Create Validation Tests ==========

    #[test]
    fn test_valid_building() {
        let building = valid_building();
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_building_empty_id() {
        let mut building = valid_building();
        building.id = "".into();
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Building ID cannot be empty");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_building_empty_name() {
        let mut building = valid_building();
        building.name = "".into();
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Building name cannot be empty");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_building_empty_address() {
        let mut building = valid_building();
        building.address = "".into();
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Building address cannot be empty");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_building_lat_min_valid() {
        let mut building = valid_building();
        building.location_lat = -90.0;
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_building_lat_max_valid() {
        let mut building = valid_building();
        building.location_lat = 90.0;
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_building_lat_below_min() {
        let mut building = valid_building();
        building.location_lat = -90.001;
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Latitude must be between -90 and 90");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_building_lat_above_max() {
        let mut building = valid_building();
        building.location_lat = 90.001;
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Latitude must be between -90 and 90");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_building_lon_min_valid() {
        let mut building = valid_building();
        building.location_lon = -180.0;
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_building_lon_max_valid() {
        let mut building = valid_building();
        building.location_lon = 180.0;
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_building_lon_below_min() {
        let mut building = valid_building();
        building.location_lon = -180.001;
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Longitude must be between -180 and 180");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_building_lon_above_max() {
        let mut building = valid_building();
        building.location_lon = 180.001;
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Longitude must be between -180 and 180");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_building_zero_units() {
        let mut building = valid_building();
        building.total_units = 0;
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Building must have at least one unit");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_building_one_unit_valid() {
        let mut building = valid_building();
        building.total_units = 1;
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_building_year_built_none() {
        let mut building = valid_building();
        building.year_built = None;
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_building_year_built_min_valid() {
        let mut building = valid_building();
        building.year_built = Some(1800);
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_building_year_built_max_valid() {
        let mut building = valid_building();
        building.year_built = Some(2100);
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_building_year_built_below_min() {
        let mut building = valid_building();
        building.year_built = Some(1799);
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Year built must be between 1800 and 2100");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_building_year_built_above_max() {
        let mut building = valid_building();
        building.year_built = Some(2101);
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Year built must be between 1800 and 2100");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    // ========== Building Length Limit Tests ==========

    #[test]
    fn test_building_id_at_max_length() {
        let mut building = valid_building();
        building.id = "a".repeat(64);
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_building_id_over_max_length() {
        let mut building = valid_building();
        building.id = "a".repeat(257);
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Building ID must be at most 256 characters");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_building_name_at_max_length() {
        let mut building = valid_building();
        building.name = "a".repeat(256);
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_building_name_over_max_length() {
        let mut building = valid_building();
        building.name = "a".repeat(257);
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Building name must be at most 256 characters");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_building_address_at_max_length() {
        let mut building = valid_building();
        building.address = "a".repeat(256);
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_building_address_over_max_length() {
        let mut building = valid_building();
        building.address = "a".repeat(257);
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Building address must be at most 256 characters");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    // ========== Unit Create Validation Tests ==========

    #[test]
    fn test_valid_unit() {
        let unit = valid_unit();
        let result = validate_create_unit(mock_create_action(), unit);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_unit_empty_number() {
        let mut unit = valid_unit();
        unit.unit_number = "".into();
        let result = validate_create_unit(mock_create_action(), unit);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Unit number cannot be empty");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_unit_number_at_max_length() {
        let mut unit = valid_unit();
        unit.unit_number = "a".repeat(128);
        let result = validate_create_unit(mock_create_action(), unit);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_unit_number_over_max_length() {
        let mut unit = valid_unit();
        unit.unit_number = "a".repeat(129);
        let result = validate_create_unit(mock_create_action(), unit);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Unit number must be at most 128 characters");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_update_unit_number_at_max_length() {
        let mut unit = valid_unit();
        unit.unit_number = "a".repeat(128);
        let result = validate_update_unit(unit);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_unit_number_over_max_length() {
        let mut unit = valid_unit();
        unit.unit_number = "a".repeat(129);
        let result = validate_update_unit(unit);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Unit number must be at most 128 characters");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_unit_zero_square_meters() {
        let mut unit = valid_unit();
        unit.square_meters = 0;
        let result = validate_create_unit(mock_create_action(), unit);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Square meters must be greater than 0");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_unit_one_square_meter_valid() {
        let mut unit = valid_unit();
        unit.square_meters = 1;
        let result = validate_create_unit(mock_create_action(), unit);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_unit_zero_bathrooms() {
        let mut unit = valid_unit();
        unit.bathrooms = 0;
        let result = validate_create_unit(mock_create_action(), unit);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Unit must have at least one bathroom");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_unit_one_bathroom_valid() {
        let mut unit = valid_unit();
        unit.bathrooms = 1;
        let result = validate_create_unit(mock_create_action(), unit);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_unit_zero_bedrooms_valid() {
        let mut unit = valid_unit();
        unit.bedrooms = 0;
        let result = validate_create_unit(mock_create_action(), unit);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_unit_with_accessibility_features() {
        let mut unit = valid_unit();
        unit.accessibility_features =
            vec![AccessFeature::WheelchairAccessible, AccessFeature::GrabBars];
        let result = validate_create_unit(mock_create_action(), unit);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_unit_with_occupant() {
        let mut unit = valid_unit();
        unit.current_occupant = Some(AgentPubKey::from_raw_36(vec![1u8; 36]));
        let result = validate_create_unit(mock_create_action(), unit);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    // ========== Unit Update Validation Tests ==========

    #[test]
    fn test_update_unit_valid() {
        let unit = valid_unit();
        let result = validate_update_unit(unit);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_update_unit_empty_number() {
        let mut unit = valid_unit();
        unit.unit_number = "".into();
        let result = validate_update_unit(unit);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Unit number cannot be empty");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_update_unit_zero_square_meters() {
        let mut unit = valid_unit();
        unit.square_meters = 0;
        let result = validate_update_unit(unit);
        assert!(result.is_ok());
        match result.unwrap() {
            ValidateCallbackResult::Invalid(msg) => {
                assert_eq!(msg, "Square meters must be greater than 0");
            }
            _ => panic!("Expected Invalid result"),
        }
    }

    #[test]
    fn test_update_unit_zero_bathrooms_allowed() {
        let mut unit = valid_unit();
        unit.bathrooms = 0;
        let result = validate_update_unit(unit);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    // ========== Enum Serde Roundtrip Tests ==========

    #[test]
    fn test_building_type_serde_apartment() {
        let bt = BuildingType::Apartment;
        let json = serde_json::to_string(&bt).unwrap();
        let deserialized: BuildingType = serde_json::from_str(&json).unwrap();
        assert_eq!(bt, deserialized);
    }

    #[test]
    fn test_building_type_serde_townhouse() {
        let bt = BuildingType::Townhouse;
        let json = serde_json::to_string(&bt).unwrap();
        let deserialized: BuildingType = serde_json::from_str(&json).unwrap();
        assert_eq!(bt, deserialized);
    }

    #[test]
    fn test_building_type_serde_single_family() {
        let bt = BuildingType::SingleFamily;
        let json = serde_json::to_string(&bt).unwrap();
        let deserialized: BuildingType = serde_json::from_str(&json).unwrap();
        assert_eq!(bt, deserialized);
    }

    #[test]
    fn test_building_type_serde_duplex() {
        let bt = BuildingType::Duplex;
        let json = serde_json::to_string(&bt).unwrap();
        let deserialized: BuildingType = serde_json::from_str(&json).unwrap();
        assert_eq!(bt, deserialized);
    }

    #[test]
    fn test_building_type_serde_co_housing() {
        let bt = BuildingType::CoHousing;
        let json = serde_json::to_string(&bt).unwrap();
        let deserialized: BuildingType = serde_json::from_str(&json).unwrap();
        assert_eq!(bt, deserialized);
    }

    #[test]
    fn test_building_type_serde_mixed_use() {
        let bt = BuildingType::MixedUse;
        let json = serde_json::to_string(&bt).unwrap();
        let deserialized: BuildingType = serde_json::from_str(&json).unwrap();
        assert_eq!(bt, deserialized);
    }

    #[test]
    fn test_unit_type_serde_studio() {
        let ut = UnitType::Studio;
        let json = serde_json::to_string(&ut).unwrap();
        let deserialized: UnitType = serde_json::from_str(&json).unwrap();
        assert_eq!(ut, deserialized);
    }

    #[test]
    fn test_unit_type_serde_one_bedroom() {
        let ut = UnitType::OneBedroom;
        let json = serde_json::to_string(&ut).unwrap();
        let deserialized: UnitType = serde_json::from_str(&json).unwrap();
        assert_eq!(ut, deserialized);
    }

    #[test]
    fn test_unit_type_serde_two_bedroom() {
        let ut = UnitType::TwoBedroom;
        let json = serde_json::to_string(&ut).unwrap();
        let deserialized: UnitType = serde_json::from_str(&json).unwrap();
        assert_eq!(ut, deserialized);
    }

    #[test]
    fn test_unit_type_serde_three_bedroom() {
        let ut = UnitType::ThreeBedroom;
        let json = serde_json::to_string(&ut).unwrap();
        let deserialized: UnitType = serde_json::from_str(&json).unwrap();
        assert_eq!(ut, deserialized);
    }

    #[test]
    fn test_unit_type_serde_four_plus() {
        let ut = UnitType::FourPlus;
        let json = serde_json::to_string(&ut).unwrap();
        let deserialized: UnitType = serde_json::from_str(&json).unwrap();
        assert_eq!(ut, deserialized);
    }

    #[test]
    fn test_unit_type_serde_accessible() {
        let ut = UnitType::Accessible;
        let json = serde_json::to_string(&ut).unwrap();
        let deserialized: UnitType = serde_json::from_str(&json).unwrap();
        assert_eq!(ut, deserialized);
    }

    #[test]
    fn test_unit_type_serde_family() {
        let ut = UnitType::Family;
        let json = serde_json::to_string(&ut).unwrap();
        let deserialized: UnitType = serde_json::from_str(&json).unwrap();
        assert_eq!(ut, deserialized);
    }

    #[test]
    fn test_access_feature_serde_wheelchair_accessible() {
        let af = AccessFeature::WheelchairAccessible;
        let json = serde_json::to_string(&af).unwrap();
        let deserialized: AccessFeature = serde_json::from_str(&json).unwrap();
        assert_eq!(af, deserialized);
    }

    #[test]
    fn test_access_feature_serde_elevator() {
        let af = AccessFeature::Elevator;
        let json = serde_json::to_string(&af).unwrap();
        let deserialized: AccessFeature = serde_json::from_str(&json).unwrap();
        assert_eq!(af, deserialized);
    }

    #[test]
    fn test_access_feature_serde_grab_bars() {
        let af = AccessFeature::GrabBars;
        let json = serde_json::to_string(&af).unwrap();
        let deserialized: AccessFeature = serde_json::from_str(&json).unwrap();
        assert_eq!(af, deserialized);
    }

    #[test]
    fn test_access_feature_serde_wide_doorways() {
        let af = AccessFeature::WideDoorways;
        let json = serde_json::to_string(&af).unwrap();
        let deserialized: AccessFeature = serde_json::from_str(&json).unwrap();
        assert_eq!(af, deserialized);
    }

    #[test]
    fn test_access_feature_serde_low_counters() {
        let af = AccessFeature::LowCounters;
        let json = serde_json::to_string(&af).unwrap();
        let deserialized: AccessFeature = serde_json::from_str(&json).unwrap();
        assert_eq!(af, deserialized);
    }

    #[test]
    fn test_access_feature_serde_visual_alerts() {
        let af = AccessFeature::VisualAlerts;
        let json = serde_json::to_string(&af).unwrap();
        let deserialized: AccessFeature = serde_json::from_str(&json).unwrap();
        assert_eq!(af, deserialized);
    }

    #[test]
    fn test_access_feature_serde_hearing_loop() {
        let af = AccessFeature::HearingLoop;
        let json = serde_json::to_string(&af).unwrap();
        let deserialized: AccessFeature = serde_json::from_str(&json).unwrap();
        assert_eq!(af, deserialized);
    }

    #[test]
    fn test_unit_status_serde_available() {
        let us = UnitStatus::Available;
        let json = serde_json::to_string(&us).unwrap();
        let deserialized: UnitStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(us, deserialized);
    }

    #[test]
    fn test_unit_status_serde_occupied() {
        let us = UnitStatus::Occupied;
        let json = serde_json::to_string(&us).unwrap();
        let deserialized: UnitStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(us, deserialized);
    }

    #[test]
    fn test_unit_status_serde_under_maintenance() {
        let us = UnitStatus::UnderMaintenance;
        let json = serde_json::to_string(&us).unwrap();
        let deserialized: UnitStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(us, deserialized);
    }

    #[test]
    fn test_unit_status_serde_reserved() {
        let us = UnitStatus::Reserved;
        let json = serde_json::to_string(&us).unwrap();
        let deserialized: UnitStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(us, deserialized);
    }

    #[test]
    fn test_unit_status_serde_renovation() {
        let us = UnitStatus::Renovation;
        let json = serde_json::to_string(&us).unwrap();
        let deserialized: UnitStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(us, deserialized);
    }

    // ========== Additional Edge Case Tests ==========

    #[test]
    fn test_building_all_types() {
        let types = vec![
            BuildingType::Apartment,
            BuildingType::Townhouse,
            BuildingType::SingleFamily,
            BuildingType::Duplex,
            BuildingType::CoHousing,
            BuildingType::MixedUse,
        ];
        for bt in types {
            let mut building = valid_building();
            building.building_type = bt;
            let result = validate_create_building(mock_create_action(), building);
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
        }
    }

    #[test]
    fn test_unit_all_types() {
        let types = vec![
            UnitType::Studio,
            UnitType::OneBedroom,
            UnitType::TwoBedroom,
            UnitType::ThreeBedroom,
            UnitType::FourPlus,
            UnitType::Accessible,
            UnitType::Family,
        ];
        for ut in types {
            let mut unit = valid_unit();
            unit.unit_type = ut;
            let result = validate_create_unit(mock_create_action(), unit);
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
        }
    }

    #[test]
    fn test_unit_all_statuses() {
        let statuses = vec![
            UnitStatus::Available,
            UnitStatus::Occupied,
            UnitStatus::UnderMaintenance,
            UnitStatus::Reserved,
            UnitStatus::Renovation,
        ];
        for status in statuses {
            let mut unit = valid_unit();
            unit.status = status;
            let result = validate_create_unit(mock_create_action(), unit);
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
        }
    }

    #[test]
    fn test_building_with_cooperative_hash() {
        let mut building = valid_building();
        building.cooperative_hash = Some(ActionHash::from_raw_36(vec![42u8; 36]));
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_unit_max_floor() {
        let mut unit = valid_unit();
        unit.floor = u8::MAX;
        let result = validate_create_unit(mock_create_action(), unit);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_unit_max_square_meters() {
        let mut unit = valid_unit();
        unit.square_meters = u32::MAX;
        let result = validate_create_unit(mock_create_action(), unit);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_building_max_units() {
        let mut building = valid_building();
        building.total_units = u16::MAX;
        let result = validate_create_building(mock_create_action(), building);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_unit_all_accessibility_features() {
        let mut unit = valid_unit();
        unit.accessibility_features = vec![
            AccessFeature::WheelchairAccessible,
            AccessFeature::Elevator,
            AccessFeature::GrabBars,
            AccessFeature::WideDoorways,
            AccessFeature::LowCounters,
            AccessFeature::VisualAlerts,
            AccessFeature::HearingLoop,
        ];
        let result = validate_create_unit(mock_create_action(), unit);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), ValidateCallbackResult::Valid);
    }

    // ============================================================================
    // Link Tag Validation Tests
    // ============================================================================

    fn validate_create_link_tag(
        link_type: LinkTypes,
        tag_bytes: Vec<u8>,
    ) -> ExternResult<ValidateCallbackResult> {
        let tag = LinkTag(tag_bytes);
        match link_type {
            LinkTypes::AllBuildings
            | LinkTypes::BuildingToUnit
            | LinkTypes::AvailableUnits
            | LinkTypes::OccupantToUnit
            | LinkTypes::GeoIndex => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(format!(
                        "{:?} link tag too long (max 256 bytes)",
                        link_type
                    )));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::BuildingTypeToBuilding => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "BuildingTypeToBuilding link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        }
    }

    #[test]
    fn test_link_tag_all_buildings_at_limit() {
        let result = validate_create_link_tag(LinkTypes::AllBuildings, vec![0u8; 256]).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_tag_all_buildings_over_limit() {
        let result = validate_create_link_tag(LinkTypes::AllBuildings, vec![0u8; 257]).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_tag_building_type_to_building_at_limit() {
        let result =
            validate_create_link_tag(LinkTypes::BuildingTypeToBuilding, vec![0u8; 512]).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_tag_building_type_to_building_over_limit() {
        let result =
            validate_create_link_tag(LinkTypes::BuildingTypeToBuilding, vec![0u8; 513]).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_tag_occupant_to_unit_at_limit() {
        let result = validate_create_link_tag(LinkTypes::OccupantToUnit, vec![0u8; 256]).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_link_tag_occupant_to_unit_over_limit() {
        let result = validate_create_link_tag(LinkTypes::OccupantToUnit, vec![0u8; 257]).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_link_tag_empty_tag_valid() {
        let result = validate_create_link_tag(LinkTypes::AllBuildings, vec![]).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }
}
