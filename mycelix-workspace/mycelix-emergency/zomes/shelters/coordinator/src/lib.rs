// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Shelters Coordinator Zome
//! Shelter registration, occupancy tracking, and person check-in/out

use hdk::prelude::*;
use shelters_integrity::*;

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Register a new emergency shelter
#[hdk_extern]
pub fn register_shelter(input: RegisterShelterInput) -> ExternResult<Record> {
    if input.name.is_empty() || input.name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Name must be 1-256 characters".into()
        )));
    }
    if input.address.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Address cannot be empty".into()
        )));
    }
    if input.capacity == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Capacity must be greater than 0".into()
        )));
    }
    if input.contact.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Contact cannot be empty".into()
        )));
    }

    let shelter = Shelter {
        id: input.id.clone(),
        name: input.name,
        location_lat: input.location_lat,
        location_lon: input.location_lon,
        address: input.address,
        capacity: input.capacity,
        current_occupancy: 0,
        shelter_type: input.shelter_type.clone(),
        amenities: input.amenities,
        status: ShelterStatus::Open,
        contact: input.contact,
    };

    let action_hash = create_entry(&EntryTypes::Shelter(shelter))?;

    // Link to all shelters
    create_entry(&EntryTypes::Anchor(Anchor("all_shelters".to_string())))?;
    create_link(
        anchor_hash("all_shelters")?,
        action_hash.clone(),
        LinkTypes::AllShelters,
        (),
    )?;

    // Link to open shelters
    create_entry(&EntryTypes::Anchor(Anchor("open_shelters".to_string())))?;
    create_link(
        anchor_hash("open_shelters")?,
        action_hash.clone(),
        LinkTypes::OpenShelters,
        (),
    )?;

    // Link by shelter type
    let type_anchor = format!("shelter_type:{:?}", input.shelter_type);
    create_entry(&EntryTypes::Anchor(Anchor(type_anchor.clone())))?;
    create_link(
        anchor_hash(&type_anchor)?,
        action_hash.clone(),
        LinkTypes::ShelterByType,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created shelter".into()
    )))
}

/// Input for registering a shelter
#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterShelterInput {
    pub id: String,
    pub name: String,
    pub location_lat: f64,
    pub location_lon: f64,
    pub address: String,
    pub capacity: u32,
    pub shelter_type: ShelterType,
    pub amenities: Vec<Amenity>,
    pub contact: String,
}

/// Update shelter status
#[hdk_extern]
pub fn update_shelter_status(input: UpdateShelterStatusInput) -> ExternResult<Record> {
    let current_record = get(input.shelter_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Shelter not found".into())),
    )?;

    let current_shelter: Shelter = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid shelter entry".into()
        )))?;

    let updated_shelter = Shelter {
        status: input.new_status.clone(),
        ..current_shelter
    };

    let new_action_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::Shelter(updated_shelter),
    )?;

    // Remove from open shelters if closing or full
    if matches!(
        input.new_status,
        ShelterStatus::Closed | ShelterStatus::Full | ShelterStatus::Evacuating
    ) {
        let links = get_links(
            LinkQuery::try_new(anchor_hash("open_shelters")?, LinkTypes::OpenShelters)?,
            GetStrategy::default(),
        )?;
        for link in links {
            let target = ActionHash::try_from(link.target.clone());
            if let Ok(target_hash) = target {
                if target_hash == input.shelter_hash {
                    delete_link(link.create_link_hash, GetOptions::default())?;
                }
            }
        }
    }

    get(new_action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated shelter".into()
    )))
}

/// Input for updating shelter status
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateShelterStatusInput {
    pub shelter_hash: ActionHash,
    pub new_status: ShelterStatus,
}

/// Check in a person or party to a shelter
#[hdk_extern]
pub fn check_in_person(input: CheckInPersonInput) -> ExternResult<Record> {
    if input.person_name.is_empty() || input.person_name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Person name must be 1-256 characters".into()
        )));
    }
    if input.party_size == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Party size must be at least 1".into()
        )));
    }

    // Get current shelter to update occupancy
    let shelter_record = get(input.shelter_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Shelter not found".into())),
    )?;

    let current_shelter: Shelter = shelter_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid shelter entry".into()
        )))?;

    let new_occupancy = current_shelter.current_occupancy + input.party_size as u32;
    if new_occupancy > current_shelter.capacity {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Shelter does not have enough capacity".into()
        )));
    }

    let now = sys_time()?;

    let registration = ShelterRegistration {
        shelter_hash: input.shelter_hash.clone(),
        person_name: input.person_name.clone(),
        person_id: input.person_id.clone(),
        party_size: input.party_size,
        special_needs: input.special_needs,
        registered_at: now,
        checked_out_at: None,
    };

    let action_hash = create_entry(&EntryTypes::ShelterRegistration(registration))?;

    // Link shelter to registration
    create_link(
        input.shelter_hash.clone(),
        action_hash.clone(),
        LinkTypes::ShelterToRegistration,
        (),
    )?;

    // Link person to registration (by name or ID)
    if let Some(ref person_id) = input.person_id {
        let person_anchor = format!("person:{}", person_id);
        create_entry(&EntryTypes::Anchor(Anchor(person_anchor.clone())))?;
        create_link(
            anchor_hash(&person_anchor)?,
            action_hash.clone(),
            LinkTypes::PersonToRegistration,
            (),
        )?;
    }

    // Update shelter occupancy
    let new_status = if new_occupancy >= current_shelter.capacity {
        ShelterStatus::Full
    } else {
        current_shelter.status.clone()
    };

    let updated_shelter = Shelter {
        current_occupancy: new_occupancy,
        status: new_status.clone(),
        ..current_shelter
    };

    update_entry(
        shelter_record.action_address().clone(),
        &EntryTypes::Shelter(updated_shelter),
    )?;

    // If shelter became full, remove from open list
    if matches!(new_status, ShelterStatus::Full) {
        let links = get_links(
            LinkQuery::try_new(anchor_hash("open_shelters")?, LinkTypes::OpenShelters)?,
            GetStrategy::default(),
        )?;
        for link in links {
            let target = ActionHash::try_from(link.target.clone());
            if let Ok(target_hash) = target {
                if target_hash == input.shelter_hash {
                    delete_link(link.create_link_hash, GetOptions::default())?;
                }
            }
        }
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created registration".into()
    )))
}

/// Input for checking in a person
#[derive(Serialize, Deserialize, Debug)]
pub struct CheckInPersonInput {
    pub shelter_hash: ActionHash,
    pub person_name: String,
    pub person_id: Option<String>,
    pub party_size: u8,
    pub special_needs: Vec<String>,
}

/// Check out a person from a shelter
#[hdk_extern]
pub fn check_out_person(input: CheckOutPersonInput) -> ExternResult<Record> {
    let current_record = get(input.registration_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Registration not found".into())),
    )?;

    let current_reg: ShelterRegistration = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid registration entry".into()
        )))?;

    if current_reg.checked_out_at.is_some() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Person already checked out".into()
        )));
    }

    let now = sys_time()?;

    let updated_reg = ShelterRegistration {
        checked_out_at: Some(now),
        ..current_reg.clone()
    };

    let new_action_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::ShelterRegistration(updated_reg),
    )?;

    // Update shelter occupancy
    let shelter_record = get(current_reg.shelter_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Shelter not found".into())),
    )?;

    let current_shelter: Shelter = shelter_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid shelter entry".into()
        )))?;

    let new_occupancy = current_shelter
        .current_occupancy
        .saturating_sub(current_reg.party_size as u32);

    let was_full = current_shelter.status == ShelterStatus::Full;

    let updated_shelter = Shelter {
        current_occupancy: new_occupancy,
        status: if was_full && new_occupancy < current_shelter.capacity {
            ShelterStatus::Open
        } else {
            current_shelter.status
        },
        ..current_shelter
    };

    update_entry(
        shelter_record.action_address().clone(),
        &EntryTypes::Shelter(updated_shelter),
    )?;

    // If shelter was full and now has space, re-add to open list
    if was_full && new_occupancy < current_shelter.capacity {
        create_link(
            anchor_hash("open_shelters")?,
            current_reg.shelter_hash,
            LinkTypes::OpenShelters,
            (),
        )?;
    }

    get(new_action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated registration".into()
    )))
}

/// Input for checking out a person
#[derive(Serialize, Deserialize, Debug)]
pub struct CheckOutPersonInput {
    pub registration_hash: ActionHash,
}

/// Get all occupants of a shelter
#[hdk_extern]
pub fn get_shelter_occupants(shelter_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(shelter_hash, LinkTypes::ShelterToRegistration)?,
        GetStrategy::default(),
    )?;

    let mut occupants = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            // Only include those not yet checked out
            if let Some(reg) = record
                .entry()
                .to_app_option::<ShelterRegistration>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if reg.checked_out_at.is_none() {
                    occupants.push(record);
                }
            }
        }
    }

    Ok(occupants)
}

/// Find nearby shelters with capacity (simplified: returns all open shelters)
#[hdk_extern]
pub fn find_nearby_shelters(input: FindNearbySheltersInput) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("open_shelters")?, LinkTypes::OpenShelters)?,
        GetStrategy::default(),
    )?;

    let mut shelters = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(shelter) = record
                .entry()
                .to_app_option::<Shelter>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                // Simple distance filter using Haversine approximation
                let dlat = (shelter.location_lat - input.lat).to_radians();
                let dlon = (shelter.location_lon - input.lon).to_radians();
                let a = (dlat / 2.0).sin().powi(2)
                    + input.lat.to_radians().cos()
                        * shelter.location_lat.to_radians().cos()
                        * (dlon / 2.0).sin().powi(2);
                let c = 2.0 * a.sqrt().asin();
                let distance_km = 6371.0 * c;

                if distance_km <= input.radius_km as f64 {
                    shelters.push(record);
                }
            }
        }
    }

    Ok(shelters)
}

/// Input for finding nearby shelters
#[derive(Serialize, Deserialize, Debug)]
pub struct FindNearbySheltersInput {
    pub lat: f64,
    pub lon: f64,
    pub radius_km: f32,
}
