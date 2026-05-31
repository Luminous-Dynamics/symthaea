// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Shelters Integrity Zome
//! Emergency shelter registration and occupancy tracking

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// An emergency shelter
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Shelter {
    pub id: String,
    pub name: String,
    pub location_lat: f64,
    pub location_lon: f64,
    pub address: String,
    pub capacity: u32,
    pub current_occupancy: u32,
    pub shelter_type: ShelterType,
    pub amenities: Vec<Amenity>,
    pub status: ShelterStatus,
    pub contact: String,
}

/// Types of shelters
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ShelterType {
    Emergency,
    Community,
    Medical,
    PetFriendly,
    Accessible,
}

/// Amenities available at a shelter
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Amenity {
    Power,
    Water,
    Medical,
    Food,
    Showers,
    Wifi,
    Charging,
    Cots,
    Blankets,
    PetArea,
    ChildCare,
    MentalHealth,
}

/// Shelter operational status
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ShelterStatus {
    Open,
    Full,
    Closed,
    Evacuating,
}

/// A person registered at a shelter
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ShelterRegistration {
    pub shelter_hash: ActionHash,
    pub person_name: String,
    pub person_id: Option<String>,
    pub party_size: u8,
    pub special_needs: Vec<String>,
    pub registered_at: Timestamp,
    pub checked_out_at: Option<Timestamp>,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Shelter(Shelter),
    ShelterRegistration(ShelterRegistration),
}

#[hdk_link_types]
pub enum LinkTypes {
    AllShelters,
    OpenShelters,
    ShelterToRegistration,
    ShelterByType,
    PersonToRegistration,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Shelter(shelter) => validate_create_shelter(action, shelter),
                EntryTypes::ShelterRegistration(reg) => validate_create_registration(action, reg),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Shelter(shelter) => validate_update_shelter(shelter),
                EntryTypes::ShelterRegistration(_) => Ok(ValidateCallbackResult::Valid),
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
            LinkTypes::AllShelters => Ok(ValidateCallbackResult::Valid),
            LinkTypes::OpenShelters => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ShelterToRegistration => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ShelterByType => Ok(ValidateCallbackResult::Valid),
            LinkTypes::PersonToRegistration => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::OpenShelters => Ok(ValidateCallbackResult::Valid),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_shelter(
    _action: Create,
    shelter: Shelter,
) -> ExternResult<ValidateCallbackResult> {
    if shelter.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Shelter ID cannot be empty".into(),
        ));
    }
    if shelter.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Shelter name cannot be empty".into(),
        ));
    }
    if shelter.address.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Shelter address cannot be empty".into(),
        ));
    }
    if shelter.capacity == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Shelter capacity must be greater than 0".into(),
        ));
    }
    if shelter.current_occupancy > shelter.capacity {
        return Ok(ValidateCallbackResult::Invalid(
            "Occupancy cannot exceed capacity".into(),
        ));
    }
    if shelter.location_lat < -90.0 || shelter.location_lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be between -90 and 90".into(),
        ));
    }
    if shelter.location_lon < -180.0 || shelter.location_lon > 180.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be between -180 and 180".into(),
        ));
    }
    if shelter.contact.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Contact information cannot be empty".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_shelter(shelter: Shelter) -> ExternResult<ValidateCallbackResult> {
    if shelter.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Shelter ID cannot be empty".into(),
        ));
    }
    if shelter.current_occupancy > shelter.capacity {
        return Ok(ValidateCallbackResult::Invalid(
            "Occupancy cannot exceed capacity".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_registration(
    _action: Create,
    reg: ShelterRegistration,
) -> ExternResult<ValidateCallbackResult> {
    if reg.person_name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Person name cannot be empty".into(),
        ));
    }
    if reg.party_size == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Party size must be at least 1".into(),
        ));
    }
    if reg.checked_out_at.is_some() {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot create registration already checked out".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
