// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shelters Coordinator Zome
//! Shelter registration, occupancy tracking, and person check-in/out

use emergency_shelters_integrity::*;
use hdk::prelude::*;
use mycelix_bridge_common::{
    civic_requirement_basic, civic_requirement_proposal, GovernanceEligibility,
};
use mycelix_zome_helpers::{get_latest_record};

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Register a new emergency shelter

#[hdk_extern]
pub fn register_shelter(input: RegisterShelterInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_basic(), "register_shelter")?;
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

    // Geo-spatial index for shelter location
    let geo_hash = commons_types::geo::geohash_encode(input.location_lat, input.location_lon, 6);
    let geo_anchor_str = format!("geo:{}", geo_hash);
    create_entry(&EntryTypes::Anchor(Anchor(geo_anchor_str.clone())))?;
    create_link(
        anchor_hash(&geo_anchor_str)?,
        action_hash.clone(),
        LinkTypes::GeoIndex,
        geo_hash.as_bytes().to_vec(),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "update_shelter_status")?;
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

    get_latest_record(new_action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated shelter".into()
    )))
}

/// Input for updating shelter status
#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateShelterStatusInput {
    pub shelter_hash: ActionHash,
    pub new_status: ShelterStatus,
}

/// Update a shelter entry (general update)
#[hdk_extern]
pub fn update_shelter(input: UpdateShelterInput) -> ExternResult<ActionHash> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_proposal(), "update_shelter")?;
    update_entry(
        input.original_action_hash,
        &EntryTypes::Shelter(input.updated_entry),
    )
}

/// Input for updating a shelter
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateShelterInput {
    pub original_action_hash: ActionHash,
    pub updated_entry: Shelter,
}

/// Check in a person or party to a shelter
#[hdk_extern]
pub fn check_in_person(input: CheckInPersonInput) -> ExternResult<Record> {
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_basic(), "check_in_person")?;
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

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
    mycelix_zome_helpers::require_civic("civic_bridge", &civic_requirement_basic(), "check_out_person")?;
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

    get_latest_record(new_action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
        if let Some(record) = get_latest_record(action_hash)? {
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
        if let Some(record) = get_latest_record(action_hash)? {
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

/// Haversine distance between two (lat, lon) points in kilometres.
///
/// Uses the standard formula with Earth radius = 6371 km.
/// Extracted as a pure function for testability.
pub fn haversine_distance_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let dlat = (lat2 - lat1).to_radians();
    let dlon = (lon2 - lon1).to_radians();
    let a = (dlat / 2.0).sin().powi(2)
        + lat1.to_radians().cos() * lat2.to_radians().cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    6371.0 * c
}

/// Get shelters near a given location using geohash-based proximity search.
#[hdk_extern]
pub fn get_nearby_shelters(input: commons_types::geo::NearbyQuery) -> ExternResult<Vec<Record>> {
    let center_hash = commons_types::geo::geohash_encode(input.latitude, input.longitude, 6);
    let mut all_cells = vec![center_hash.clone()];
    all_cells.extend(commons_types::geo::geohash_neighbors(&center_hash));

    let mut records = Vec::new();
    for cell in &all_cells {
        let anchor_str = format!("geo:{}", cell);
        let anchor_entry = Anchor(anchor_str);
        let anchor_hash = hash_entry(&anchor_entry)?;
        if let Ok(links) = get_links(
            LinkQuery::try_new(anchor_hash, LinkTypes::GeoIndex)?,
            GetStrategy::Local,
        ) {
            for link in links {
                if let Ok(action_hash) = ActionHash::try_from(link.target) {
                    if let Some(record) = get(action_hash, GetOptions::default())? {
                        records.push(record);
                    }
                }
            }
        }
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // HAVERSINE DISTANCE TESTS
    // ========================================================================

    #[test]
    fn haversine_same_point_is_zero() {
        let d = haversine_distance_km(40.7128, -74.0060, 40.7128, -74.0060);
        assert!(d.abs() < 1e-10, "Same point should be 0, got {}", d);
    }

    #[test]
    fn haversine_nyc_to_la() {
        // NYC (40.7128, -74.0060) to LA (34.0522, -118.2437)
        // Expected: ~3,944 km
        let d = haversine_distance_km(40.7128, -74.0060, 34.0522, -118.2437);
        assert!(
            (d - 3944.0).abs() < 50.0,
            "NYC to LA expected ~3944 km, got {} km",
            d
        );
    }

    #[test]
    fn haversine_london_to_tokyo() {
        // London (51.5074, -0.1278) to Tokyo (35.6762, 139.6503)
        // Expected: ~9,560 km
        let d = haversine_distance_km(51.5074, -0.1278, 35.6762, 139.6503);
        assert!(
            (d - 9560.0).abs() < 50.0,
            "London to Tokyo expected ~9560 km, got {} km",
            d
        );
    }

    #[test]
    fn haversine_antipodal_points() {
        // North pole to south pole: exactly 20015.09 km (half circumference)
        let d = haversine_distance_km(90.0, 0.0, -90.0, 0.0);
        assert!(
            (d - 20015.0).abs() < 100.0,
            "Pole-to-pole expected ~20015 km, got {} km",
            d
        );
    }

    #[test]
    fn haversine_equator_opposite_sides() {
        // (0, 0) to (0, 180): half equatorial circumference ~20015 km
        let d = haversine_distance_km(0.0, 0.0, 0.0, 180.0);
        assert!(
            (d - 20015.0).abs() < 100.0,
            "Equator 0-180 expected ~20015 km, got {} km",
            d
        );
    }

    #[test]
    fn haversine_symmetry() {
        let d1 = haversine_distance_km(40.7128, -74.0060, 34.0522, -118.2437);
        let d2 = haversine_distance_km(34.0522, -118.2437, 40.7128, -74.0060);
        assert!(
            (d1 - d2).abs() < 1e-10,
            "Distance should be symmetric: {} vs {}",
            d1,
            d2
        );
    }

    #[test]
    fn haversine_short_distance() {
        // Two points about 1 km apart in Richardson, TX
        // (32.9483, -96.7299) to (32.9573, -96.7299) -- ~1 km north
        let d = haversine_distance_km(32.9483, -96.7299, 32.9573, -96.7299);
        assert!((d - 1.0).abs() < 0.1, "Expected ~1 km, got {} km", d);
    }

    // ========================================================================
    // INPUT STRUCT SERDE TESTS
    // ========================================================================

    #[test]
    fn register_shelter_input_serde_roundtrip() {
        let input = RegisterShelterInput {
            id: "shelter-1".into(),
            name: "Community Center".into(),
            location_lat: 32.9483,
            location_lon: -96.7299,
            address: "123 Main St".into(),
            capacity: 200,
            shelter_type: ShelterType::Community,
            amenities: vec![Amenity::Power, Amenity::Water],
            contact: "555-0100".into(),
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: RegisterShelterInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.id, "shelter-1");
        assert_eq!(parsed.name, "Community Center");
        assert!((parsed.location_lat - 32.9483).abs() < 1e-10);
        assert_eq!(parsed.capacity, 200);
        assert_eq!(parsed.amenities.len(), 2);
    }

    #[test]
    fn find_nearby_input_serde_roundtrip() {
        let input = FindNearbySheltersInput {
            lat: 32.9483,
            lon: -96.7299,
            radius_km: 10.0,
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: FindNearbySheltersInput = serde_json::from_str(&json).expect("deserialize");
        assert!((parsed.lat - 32.9483).abs() < 1e-10);
        assert!((parsed.lon - (-96.7299)).abs() < 1e-10);
        assert!((parsed.radius_km - 10.0).abs() < 1e-6);
    }

    #[test]
    fn update_shelter_status_input_serde_roundtrip() {
        // Can't construct ActionHash easily in tests, but we can test
        // the ShelterStatus enum serde via the integrity crate
        let statuses = vec![
            ShelterStatus::Open,
            ShelterStatus::Full,
            ShelterStatus::Closed,
            ShelterStatus::Evacuating,
        ];
        for status in statuses {
            let json = serde_json::to_string(&status).expect("serialize");
            let parsed: ShelterStatus = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed, status);
        }
    }

    #[test]
    fn check_in_person_input_serde_roundtrip() {
        // Test the serializable parts of CheckInPersonInput
        // (ActionHash field requires special handling, test the rest)
        let special_needs: Vec<String> = vec!["wheelchair".into(), "oxygen".into()];
        let json = serde_json::to_string(&special_needs).expect("serialize");
        let parsed: Vec<String> = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, vec!["wheelchair", "oxygen"]);
    }

    // ========================================================================
    // FULL INPUT STRUCT SERDE TESTS
    // ========================================================================

    #[test]
    fn check_in_person_input_full_serde_roundtrip() {
        let input = CheckInPersonInput {
            shelter_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            person_name: "Alice Smith".into(),
            person_id: Some("TX-99999".into()),
            party_size: 4,
            special_needs: vec!["diabetic".into(), "service animal".into()],
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: CheckInPersonInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.person_name, "Alice Smith");
        assert_eq!(parsed.person_id, Some("TX-99999".into()));
        assert_eq!(parsed.party_size, 4);
        assert_eq!(parsed.special_needs.len(), 2);
    }

    #[test]
    fn check_in_person_input_no_id_serde() {
        let input = CheckInPersonInput {
            shelter_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            person_name: "Bob Jones".into(),
            person_id: None,
            party_size: 1,
            special_needs: vec![],
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: CheckInPersonInput = serde_json::from_str(&json).expect("deserialize");
        assert!(parsed.person_id.is_none());
        assert!(parsed.special_needs.is_empty());
        assert_eq!(parsed.party_size, 1);
    }

    #[test]
    fn check_out_person_input_serde_roundtrip() {
        let input = CheckOutPersonInput {
            registration_hash: ActionHash::from_raw_36(vec![5u8; 36]),
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let _parsed: CheckOutPersonInput = serde_json::from_str(&json).expect("deserialize");
    }

    #[test]
    fn update_shelter_status_input_full_serde_roundtrip() {
        let input = UpdateShelterStatusInput {
            shelter_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            new_status: ShelterStatus::Evacuating,
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: UpdateShelterStatusInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.new_status, ShelterStatus::Evacuating);
    }

    // ========================================================================
    // SHELTER TYPE ALL VARIANTS SERDE
    // ========================================================================

    #[test]
    fn shelter_type_all_variants_serde() {
        let variants = vec![
            ShelterType::Emergency,
            ShelterType::Community,
            ShelterType::Medical,
            ShelterType::PetFriendly,
            ShelterType::Accessible,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).expect("serialize");
            let parsed: ShelterType = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed, variant);
        }
    }

    // ========================================================================
    // AMENITY ALL VARIANTS SERDE
    // ========================================================================

    #[test]
    fn amenity_all_variants_serde() {
        let variants = vec![
            Amenity::Power,
            Amenity::Water,
            Amenity::Medical,
            Amenity::Food,
            Amenity::Showers,
            Amenity::Wifi,
            Amenity::Charging,
            Amenity::Cots,
            Amenity::Blankets,
            Amenity::PetArea,
            Amenity::ChildCare,
            Amenity::MentalHealth,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).expect("serialize");
            let parsed: Amenity = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed, variant);
        }
    }

    // ========================================================================
    // SHELTER STRUCT SERDE TESTS
    // ========================================================================

    #[test]
    fn shelter_full_struct_serde_roundtrip() {
        let shelter = Shelter {
            id: "shelter-100".into(),
            name: "High School Gym".into(),
            location_lat: 32.95,
            location_lon: -96.73,
            address: "456 Oak Ave, Richardson TX".into(),
            capacity: 500,
            current_occupancy: 250,
            shelter_type: ShelterType::Emergency,
            amenities: vec![
                Amenity::Power,
                Amenity::Water,
                Amenity::Food,
                Amenity::Cots,
                Amenity::Blankets,
                Amenity::MentalHealth,
            ],
            status: ShelterStatus::Open,
            contact: "911-SHELTER".into(),
        };
        let json = serde_json::to_string(&shelter).expect("serialize");
        let parsed: Shelter = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, shelter);
    }

    #[test]
    fn shelter_zero_occupancy_serde() {
        let shelter = Shelter {
            id: "s-new".into(),
            name: "New Shelter".into(),
            location_lat: 0.0,
            location_lon: 0.0,
            address: "1 Street".into(),
            capacity: 100,
            current_occupancy: 0,
            shelter_type: ShelterType::Community,
            amenities: vec![],
            status: ShelterStatus::Open,
            contact: "555-0000".into(),
        };
        let json = serde_json::to_string(&shelter).expect("serialize");
        let parsed: Shelter = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.current_occupancy, 0);
        assert!(parsed.amenities.is_empty());
    }

    #[test]
    fn shelter_full_capacity_status_serde() {
        let shelter = Shelter {
            id: "s-full".into(),
            name: "Full Shelter".into(),
            location_lat: 33.0,
            location_lon: -97.0,
            address: "789 Elm St".into(),
            capacity: 50,
            current_occupancy: 50,
            shelter_type: ShelterType::PetFriendly,
            amenities: vec![Amenity::PetArea],
            status: ShelterStatus::Full,
            contact: "555-1111".into(),
        };
        let json = serde_json::to_string(&shelter).expect("serialize");
        let parsed: Shelter = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.status, ShelterStatus::Full);
        assert_eq!(parsed.current_occupancy, parsed.capacity);
    }

    // ========================================================================
    // SHELTER REGISTRATION STRUCT SERDE TESTS
    // ========================================================================

    #[test]
    fn shelter_registration_full_serde_roundtrip() {
        let reg = ShelterRegistration {
            shelter_hash: ActionHash::from_raw_36(vec![1u8; 36]),
            person_name: "Jane Doe".into(),
            person_id: Some("DL-123456".into()),
            party_size: 3,
            special_needs: vec!["infant".into(), "medication refrigeration".into()],
            registered_at: Timestamp::from_micros(5000000),
            checked_out_at: None,
        };
        let json = serde_json::to_string(&reg).expect("serialize");
        let parsed: ShelterRegistration = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.person_name, "Jane Doe");
        assert_eq!(parsed.party_size, 3);
        assert!(parsed.checked_out_at.is_none());
        assert_eq!(parsed.special_needs.len(), 2);
    }

    #[test]
    fn shelter_registration_checked_out_serde() {
        let reg = ShelterRegistration {
            shelter_hash: ActionHash::from_raw_36(vec![2u8; 36]),
            person_name: "John Smith".into(),
            person_id: None,
            party_size: 1,
            special_needs: vec![],
            registered_at: Timestamp::from_micros(1000),
            checked_out_at: Some(Timestamp::from_micros(9000)),
        };
        let json = serde_json::to_string(&reg).expect("serialize");
        let parsed: ShelterRegistration = serde_json::from_str(&json).expect("deserialize");
        assert!(parsed.checked_out_at.is_some());
        assert!(parsed.person_id.is_none());
    }

    // ========================================================================
    // BOUNDARY CONDITION TESTS
    // ========================================================================

    #[test]
    fn check_in_person_input_max_party_size() {
        let input = CheckInPersonInput {
            shelter_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            person_name: "Large Family".into(),
            person_id: None,
            party_size: u8::MAX, // 255
            special_needs: vec![],
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: CheckInPersonInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.party_size, 255);
    }

    #[test]
    fn register_shelter_input_max_capacity() {
        let input = RegisterShelterInput {
            id: "s-big".into(),
            name: "Convention Center".into(),
            location_lat: 40.0,
            location_lon: -74.0,
            address: "1 Convention Way".into(),
            capacity: u32::MAX,
            shelter_type: ShelterType::Emergency,
            amenities: vec![],
            contact: "emergency@city.gov".into(),
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: RegisterShelterInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.capacity, u32::MAX);
    }

    #[test]
    fn register_shelter_input_all_amenities() {
        let input = RegisterShelterInput {
            id: "s-deluxe".into(),
            name: "Deluxe Shelter".into(),
            location_lat: 32.0,
            location_lon: -96.0,
            address: "1 Luxury Way".into(),
            capacity: 1000,
            shelter_type: ShelterType::Accessible,
            amenities: vec![
                Amenity::Power,
                Amenity::Water,
                Amenity::Medical,
                Amenity::Food,
                Amenity::Showers,
                Amenity::Wifi,
                Amenity::Charging,
                Amenity::Cots,
                Amenity::Blankets,
                Amenity::PetArea,
                Amenity::ChildCare,
                Amenity::MentalHealth,
            ],
            contact: "555-ALL-AMENITIES".into(),
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: RegisterShelterInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.amenities.len(), 12);
    }

    // ========================================================================
    // UPDATE SHELTER INPUT TESTS
    // ========================================================================

    #[test]
    fn update_shelter_input_serde_roundtrip() {
        let input = UpdateShelterInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            updated_entry: Shelter {
                id: "shelter-1".into(),
                name: "Updated Community Center".into(),
                location_lat: 32.9483,
                location_lon: -96.7299,
                address: "123 Main St".into(),
                capacity: 300,
                current_occupancy: 50,
                shelter_type: ShelterType::Community,
                amenities: vec![Amenity::Power, Amenity::Water, Amenity::Medical],
                status: ShelterStatus::Open,
                contact: "555-0200".into(),
            },
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: UpdateShelterInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(
            parsed.original_action_hash,
            ActionHash::from_raw_36(vec![0xdb; 36])
        );
        assert_eq!(parsed.updated_entry.name, "Updated Community Center");
        assert_eq!(parsed.updated_entry.capacity, 300);
        assert_eq!(parsed.updated_entry.amenities.len(), 3);
    }

    #[test]
    fn update_shelter_input_clone() {
        let input = UpdateShelterInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xab; 36]),
            updated_entry: Shelter {
                id: "s-clone".into(),
                name: "Clone Test".into(),
                location_lat: 0.0,
                location_lon: 0.0,
                address: "1 St".into(),
                capacity: 10,
                current_occupancy: 0,
                shelter_type: ShelterType::Emergency,
                amenities: vec![],
                status: ShelterStatus::Open,
                contact: "555-0000".into(),
            },
        };
        let cloned = input.clone();
        assert_eq!(cloned.original_action_hash, input.original_action_hash);
        assert_eq!(cloned.updated_entry.name, "Clone Test");
    }

    #[test]
    fn update_shelter_input_full_status_serde() {
        let input = UpdateShelterInput {
            original_action_hash: ActionHash::from_raw_36(vec![0u8; 36]),
            updated_entry: Shelter {
                id: "s-full".into(),
                name: "Full Shelter".into(),
                location_lat: 33.0,
                location_lon: -97.0,
                address: "789 Elm St".into(),
                capacity: 50,
                current_occupancy: 50,
                shelter_type: ShelterType::PetFriendly,
                amenities: vec![Amenity::PetArea],
                status: ShelterStatus::Full,
                contact: "555-1111".into(),
            },
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: UpdateShelterInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.updated_entry.status, ShelterStatus::Full);
        assert_eq!(
            parsed.updated_entry.current_occupancy,
            parsed.updated_entry.capacity
        );
    }

    #[test]
    fn find_nearby_input_zero_radius() {
        let input = FindNearbySheltersInput {
            lat: 0.0,
            lon: 0.0,
            radius_km: 0.0,
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: FindNearbySheltersInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.radius_km, 0.0);
    }

    #[test]
    fn find_nearby_input_extreme_coordinates() {
        let input = FindNearbySheltersInput {
            lat: -90.0,
            lon: 180.0,
            radius_km: 20015.0,
        };
        let json = serde_json::to_string(&input).expect("serialize");
        let parsed: FindNearbySheltersInput = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.lat, -90.0);
        assert_eq!(parsed.lon, 180.0);
    }

    // ========================================================================
    // ANCHOR STRUCT SERDE
    // ========================================================================

    // ========================================================================
    // REGISTER SHELTER INPUT EACH SHELTER TYPE
    // ========================================================================

    #[test]
    fn register_shelter_input_each_shelter_type() {
        let types = vec![
            ShelterType::Emergency,
            ShelterType::Community,
            ShelterType::Medical,
            ShelterType::PetFriendly,
            ShelterType::Accessible,
        ];
        for st in types {
            let input = RegisterShelterInput {
                id: "s-t".into(),
                name: "Type Test".into(),
                location_lat: 0.0,
                location_lon: 0.0,
                address: "123 St".into(),
                capacity: 10,
                shelter_type: st.clone(),
                amenities: vec![],
                contact: "555-0000".into(),
            };
            let json = serde_json::to_string(&input).expect("serialize");
            let parsed: RegisterShelterInput = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed.shelter_type, st);
        }
    }
}
