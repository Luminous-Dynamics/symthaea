// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Units Integrity Zome
//! Defines entry types and validation for buildings and housing units.

use hdi::prelude::*;

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
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::AllBuildings => Ok(ValidateCallbackResult::Valid),
            LinkTypes::BuildingToUnit => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AvailableUnits => Ok(ValidateCallbackResult::Valid),
            LinkTypes::OccupantToUnit => Ok(ValidateCallbackResult::Valid),
            LinkTypes::BuildingTypeToBuilding => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::AvailableUnits => Ok(ValidateCallbackResult::Valid),
            LinkTypes::OccupantToUnit => Ok(ValidateCallbackResult::Valid),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_building(
    _action: Create,
    building: Building,
) -> ExternResult<ValidateCallbackResult> {
    if building.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Building ID cannot be empty".into(),
        ));
    }
    if building.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Building name cannot be empty".into(),
        ));
    }
    if building.address.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Building address cannot be empty".into(),
        ));
    }
    if building.location_lat < -90.0 || building.location_lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be between -90 and 90".into(),
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
        if year < 1800 || year > 2100 {
            return Ok(ValidateCallbackResult::Invalid(
                "Year built must be between 1800 and 2100".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_unit(_action: Create, unit: Unit) -> ExternResult<ValidateCallbackResult> {
    if unit.unit_number.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Unit number cannot be empty".into(),
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
    if unit.unit_number.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Unit number cannot be empty".into(),
        ));
    }
    if unit.square_meters == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Square meters must be greater than 0".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
