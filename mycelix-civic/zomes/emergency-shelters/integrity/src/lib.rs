//! Shelters Integrity Zome
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
            tag,
            action: _,
        } => {
            let tag_len = tag.0.len();
            match link_type {
                LinkTypes::AllShelters => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::OpenShelters => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::ShelterToRegistration => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::ShelterByType => {
                    // Type links may store serialized type metadata
                    if tag_len > 512 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 512 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::PersonToRegistration => {
                    if tag_len > 256 {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link tag too long (max 256 bytes)".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
            }
        }
        FlatOp::RegisterDeleteLink {
            link_type: _,
            original_action: _,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => {
            if tag.0.len() > 256 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Delete link tag too long (max 256 bytes)".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
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
    if shelter.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Shelter ID cannot be empty".into(),
        ));
    }
    if shelter.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Shelter ID too long (max 256 chars)".into(),
        ));
    }
    if shelter.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Shelter name cannot be empty".into(),
        ));
    }
    if shelter.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Shelter name too long (max 256 chars)".into(),
        ));
    }
    if shelter.address.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Shelter address cannot be empty".into(),
        ));
    }
    if shelter.address.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Shelter address too long (max 256 chars)".into(),
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
    if !shelter.location_lat.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "location_lat must be a finite number".into(),
        ));
    }
    if shelter.location_lat < -90.0 || shelter.location_lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be between -90 and 90".into(),
        ));
    }
    if !shelter.location_lon.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "location_lon must be a finite number".into(),
        ));
    }
    if shelter.location_lon < -180.0 || shelter.location_lon > 180.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be between -180 and 180".into(),
        ));
    }
    if shelter.contact.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Contact information cannot be empty".into(),
        ));
    }
    if shelter.contact.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Contact too long (max 256 chars)".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_shelter(shelter: Shelter) -> ExternResult<ValidateCallbackResult> {
    if shelter.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Shelter ID cannot be empty".into(),
        ));
    }
    if shelter.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Shelter ID too long (max 256 chars)".into(),
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
    if reg.person_name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Person name cannot be empty".into(),
        ));
    }
    if reg.person_name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Person name too long (max 256 chars)".into(),
        ));
    }
    if let Some(ref pid) = reg.person_id {
        if pid.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Person ID too long (max 256 chars)".into(),
            ));
        }
    }
    if reg.special_needs.len() > 100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Too many special needs entries (max 100)".into(),
        ));
    }
    for need in &reg.special_needs {
        if need.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Special need entry too long (max 256 chars)".into(),
            ));
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // RESULT HELPERS
    // ========================================================================

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

    // ========================================================================
    // CONSTRUCTION HELPERS
    // ========================================================================

    fn fake_create() -> Create {
        Create {
            author: AgentPubKey::from_raw_36(vec![0u8; 36]),
            timestamp: Timestamp::from_micros(0),
            action_seq: 0,
            prev_action: ActionHash::from_raw_36(vec![0u8; 36]),
            entry_type: EntryType::App(AppEntryDef::new(
                EntryDefIndex(0),
                ZomeIndex(0),
                EntryVisibility::Public,
            )),
            entry_hash: EntryHash::from_raw_36(vec![0u8; 36]),
            weight: EntryRateWeight::default(),
        }
    }

    fn ah() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn ts() -> Timestamp {
        Timestamp::from_micros(0)
    }

    fn make_shelter() -> Shelter {
        Shelter {
            id: "shelter-1".into(),
            name: "Downtown Community Center".into(),
            location_lat: 32.9483,
            location_lon: -96.7299,
            address: "123 Main St, Richardson, TX".into(),
            capacity: 200,
            current_occupancy: 45,
            shelter_type: ShelterType::Community,
            amenities: vec![Amenity::Power, Amenity::Water, Amenity::Food],
            status: ShelterStatus::Open,
            contact: "555-0100".into(),
        }
    }

    fn make_registration() -> ShelterRegistration {
        ShelterRegistration {
            shelter_hash: ah(),
            person_name: "Alice Johnson".into(),
            person_id: Some("TX-12345".into()),
            party_size: 3,
            special_needs: vec!["wheelchair accessible".into()],
            registered_at: ts(),
            checked_out_at: None,
        }
    }

    // ========================================================================
    // SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn shelter_serde_roundtrip() {
        let shelter = make_shelter();
        let json = serde_json::to_string(&shelter).expect("serialize");
        let parsed: Shelter = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, shelter);
    }

    #[test]
    fn shelter_type_all_variants_serde_roundtrip() {
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

    #[test]
    fn amenity_all_variants_serde_roundtrip() {
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

    #[test]
    fn shelter_status_all_variants_serde_roundtrip() {
        let variants = vec![
            ShelterStatus::Open,
            ShelterStatus::Full,
            ShelterStatus::Closed,
            ShelterStatus::Evacuating,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).expect("serialize");
            let parsed: ShelterStatus = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed, variant);
        }
    }

    // ========================================================================
    // validate_create_shelter TESTS
    // ========================================================================

    #[test]
    fn create_shelter_valid_passes() {
        let result = validate_create_shelter(fake_create(), make_shelter());
        assert!(is_valid(&result));
    }

    #[test]
    fn create_shelter_empty_id_rejected() {
        let mut s = make_shelter();
        s.id = "".into();
        let result = validate_create_shelter(fake_create(), s);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Shelter ID cannot be empty");
    }

    #[test]
    fn create_shelter_empty_name_rejected() {
        let mut s = make_shelter();
        s.name = "".into();
        let result = validate_create_shelter(fake_create(), s);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Shelter name cannot be empty");
    }

    #[test]
    fn create_shelter_empty_address_rejected() {
        let mut s = make_shelter();
        s.address = "".into();
        let result = validate_create_shelter(fake_create(), s);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Shelter address cannot be empty");
    }

    #[test]
    fn create_shelter_zero_capacity_rejected() {
        let mut s = make_shelter();
        s.capacity = 0;
        s.current_occupancy = 0;
        let result = validate_create_shelter(fake_create(), s);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Shelter capacity must be greater than 0"
        );
    }

    #[test]
    fn create_shelter_occupancy_exceeds_capacity_rejected() {
        let mut s = make_shelter();
        s.capacity = 100;
        s.current_occupancy = 101;
        let result = validate_create_shelter(fake_create(), s);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Occupancy cannot exceed capacity");
    }

    #[test]
    fn create_shelter_occupancy_equals_capacity_passes() {
        let mut s = make_shelter();
        s.capacity = 50;
        s.current_occupancy = 50;
        let result = validate_create_shelter(fake_create(), s);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_shelter_lat_below_negative_90_rejected() {
        let mut s = make_shelter();
        s.location_lat = -90.1;
        let result = validate_create_shelter(fake_create(), s);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Latitude must be between -90 and 90");
    }

    #[test]
    fn create_shelter_lat_above_90_rejected() {
        let mut s = make_shelter();
        s.location_lat = 90.1;
        let result = validate_create_shelter(fake_create(), s);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Latitude must be between -90 and 90");
    }

    #[test]
    fn create_shelter_lat_boundary_values_pass() {
        for lat in [-90.0, 0.0, 90.0] {
            let mut s = make_shelter();
            s.location_lat = lat;
            let result = validate_create_shelter(fake_create(), s);
            assert!(is_valid(&result), "lat={} should pass", lat);
        }
    }

    #[test]
    fn create_shelter_lon_below_negative_180_rejected() {
        let mut s = make_shelter();
        s.location_lon = -180.1;
        let result = validate_create_shelter(fake_create(), s);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Longitude must be between -180 and 180"
        );
    }

    #[test]
    fn create_shelter_lon_above_180_rejected() {
        let mut s = make_shelter();
        s.location_lon = 180.1;
        let result = validate_create_shelter(fake_create(), s);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Longitude must be between -180 and 180"
        );
    }

    #[test]
    fn create_shelter_lon_boundary_values_pass() {
        for lon in [-180.0, 0.0, 180.0] {
            let mut s = make_shelter();
            s.location_lon = lon;
            let result = validate_create_shelter(fake_create(), s);
            assert!(is_valid(&result), "lon={} should pass", lon);
        }
    }

    #[test]
    fn create_shelter_empty_contact_rejected() {
        let mut s = make_shelter();
        s.contact = "".into();
        let result = validate_create_shelter(fake_create(), s);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Contact information cannot be empty");
    }

    // ========================================================================
    // validate_update_shelter TESTS
    // ========================================================================

    #[test]
    fn update_shelter_valid_passes() {
        let result = validate_update_shelter(make_shelter());
        assert!(is_valid(&result));
    }

    #[test]
    fn update_shelter_empty_id_rejected() {
        let mut s = make_shelter();
        s.id = "".into();
        let result = validate_update_shelter(s);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Shelter ID cannot be empty");
    }

    #[test]
    fn update_shelter_occupancy_exceeds_capacity_rejected() {
        let mut s = make_shelter();
        s.capacity = 100;
        s.current_occupancy = 101;
        let result = validate_update_shelter(s);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Occupancy cannot exceed capacity");
    }

    #[test]
    fn update_shelter_occupancy_equals_capacity_passes() {
        let mut s = make_shelter();
        s.capacity = 200;
        s.current_occupancy = 200;
        let result = validate_update_shelter(s);
        assert!(is_valid(&result));
    }

    #[test]
    fn update_shelter_zero_capacity_with_zero_occupancy_passes() {
        // validate_update_shelter does NOT check capacity > 0
        let mut s = make_shelter();
        s.capacity = 0;
        s.current_occupancy = 0;
        let result = validate_update_shelter(s);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // validate_create_registration TESTS
    // ========================================================================

    #[test]
    fn create_registration_valid_passes() {
        let result = validate_create_registration(fake_create(), make_registration());
        assert!(is_valid(&result));
    }

    #[test]
    fn create_registration_empty_person_name_rejected() {
        let mut reg = make_registration();
        reg.person_name = "".into();
        let result = validate_create_registration(fake_create(), reg);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Person name cannot be empty");
    }

    #[test]
    fn create_registration_zero_party_size_rejected() {
        let mut reg = make_registration();
        reg.party_size = 0;
        let result = validate_create_registration(fake_create(), reg);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Party size must be at least 1");
    }

    #[test]
    fn create_registration_party_size_one_passes() {
        let mut reg = make_registration();
        reg.party_size = 1;
        let result = validate_create_registration(fake_create(), reg);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_registration_already_checked_out_rejected() {
        let mut reg = make_registration();
        reg.checked_out_at = Some(Timestamp::from_micros(1_000_000));
        let result = validate_create_registration(fake_create(), reg);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Cannot create registration already checked out"
        );
    }

    #[test]
    fn create_registration_no_person_id_passes() {
        let mut reg = make_registration();
        reg.person_id = None;
        let result = validate_create_registration(fake_create(), reg);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_registration_empty_special_needs_passes() {
        let mut reg = make_registration();
        reg.special_needs = vec![];
        let result = validate_create_registration(fake_create(), reg);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_registration_max_party_size_passes() {
        let mut reg = make_registration();
        reg.party_size = u8::MAX; // 255
        let result = validate_create_registration(fake_create(), reg);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // STRING LENGTH LIMIT TESTS
    // ========================================================================

    #[test]
    fn create_shelter_id_at_limit_passes() {
        let mut s = make_shelter();
        s.id = "i".repeat(256);
        let result = validate_create_shelter(fake_create(), s);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_shelter_id_over_limit_rejected() {
        let mut s = make_shelter();
        s.id = "i".repeat(257);
        let result = validate_create_shelter(fake_create(), s);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Shelter ID too long (max 256 chars)");
    }

    #[test]
    fn create_shelter_name_at_limit_passes() {
        let mut s = make_shelter();
        s.name = "n".repeat(256);
        let result = validate_create_shelter(fake_create(), s);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_shelter_name_over_limit_rejected() {
        let mut s = make_shelter();
        s.name = "n".repeat(257);
        let result = validate_create_shelter(fake_create(), s);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Shelter name too long (max 256 chars)"
        );
    }

    #[test]
    fn create_shelter_address_at_limit_passes() {
        let mut s = make_shelter();
        s.address = "a".repeat(256);
        let result = validate_create_shelter(fake_create(), s);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_shelter_address_over_limit_rejected() {
        let mut s = make_shelter();
        s.address = "a".repeat(257);
        let result = validate_create_shelter(fake_create(), s);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Shelter address too long (max 256 chars)"
        );
    }

    #[test]
    fn create_shelter_contact_at_limit_passes() {
        let mut s = make_shelter();
        s.contact = "c".repeat(256);
        let result = validate_create_shelter(fake_create(), s);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_shelter_contact_over_limit_rejected() {
        let mut s = make_shelter();
        s.contact = "c".repeat(257);
        let result = validate_create_shelter(fake_create(), s);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Contact too long (max 256 chars)");
    }

    #[test]
    fn create_registration_person_name_at_limit_passes() {
        let mut reg = make_registration();
        reg.person_name = "p".repeat(256);
        let result = validate_create_registration(fake_create(), reg);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_registration_person_name_over_limit_rejected() {
        let mut reg = make_registration();
        reg.person_name = "p".repeat(257);
        let result = validate_create_registration(fake_create(), reg);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Person name too long (max 256 chars)");
    }

    #[test]
    fn create_registration_person_id_at_limit_passes() {
        let mut reg = make_registration();
        reg.person_id = Some("x".repeat(256));
        let result = validate_create_registration(fake_create(), reg);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_registration_person_id_over_limit_rejected() {
        let mut reg = make_registration();
        reg.person_id = Some("x".repeat(257));
        let result = validate_create_registration(fake_create(), reg);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Person ID too long (max 256 chars)");
    }

    #[test]
    fn create_registration_special_needs_item_at_limit_passes() {
        let mut reg = make_registration();
        reg.special_needs = vec!["n".repeat(256)];
        let result = validate_create_registration(fake_create(), reg);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_registration_special_needs_item_over_limit_rejected() {
        let mut reg = make_registration();
        reg.special_needs = vec!["n".repeat(257)];
        let result = validate_create_registration(fake_create(), reg);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Special need entry too long (max 256 chars)"
        );
    }

    #[test]
    fn create_registration_special_needs_count_at_limit_passes() {
        let mut reg = make_registration();
        reg.special_needs = (0..100).map(|i| format!("need-{}", i)).collect();
        let result = validate_create_registration(fake_create(), reg);
        assert!(is_valid(&result));
    }

    #[test]
    fn create_registration_special_needs_count_over_limit_rejected() {
        let mut reg = make_registration();
        reg.special_needs = (0..101).map(|i| format!("need-{}", i)).collect();
        let result = validate_create_registration(fake_create(), reg);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Too many special needs entries (max 100)"
        );
    }

    #[test]
    fn update_shelter_id_over_limit_rejected() {
        let mut s = make_shelter();
        s.id = "i".repeat(257);
        let result = validate_update_shelter(s);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Shelter ID too long (max 256 chars)");
    }

    // ========================================================================
    // LINK TAG VALIDATION TESTS
    // ========================================================================

    fn validate_create_link_tag(link_type: &LinkTypes, tag: &LinkTag) -> ValidateCallbackResult {
        let tag_len = tag.0.len();
        match link_type {
            LinkTypes::AllShelters
            | LinkTypes::OpenShelters
            | LinkTypes::ShelterToRegistration
            | LinkTypes::PersonToRegistration => {
                if tag_len > 256 {
                    ValidateCallbackResult::Invalid("Link tag too long (max 256 bytes)".into())
                } else {
                    ValidateCallbackResult::Valid
                }
            }
            LinkTypes::ShelterByType => {
                if tag_len > 512 {
                    ValidateCallbackResult::Invalid("Link tag too long (max 512 bytes)".into())
                } else {
                    ValidateCallbackResult::Valid
                }
            }
        }
    }

    fn validate_delete_link_tag(tag: &LinkTag) -> ValidateCallbackResult {
        if tag.0.len() > 256 {
            ValidateCallbackResult::Invalid("Delete link tag too long (max 256 bytes)".into())
        } else {
            ValidateCallbackResult::Valid
        }
    }

    // -- AllShelters (256-byte limit) boundary tests --

    #[test]
    fn link_tag_all_shelters_empty_valid() {
        let tag = LinkTag::new(vec![]);
        let result = validate_create_link_tag(&LinkTypes::AllShelters, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_all_shelters_at_limit_valid() {
        let tag = LinkTag::new(vec![0u8; 256]);
        let result = validate_create_link_tag(&LinkTypes::AllShelters, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_all_shelters_over_limit_invalid() {
        let tag = LinkTag::new(vec![0u8; 257]);
        let result = validate_create_link_tag(&LinkTypes::AllShelters, &tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- ShelterByType (512-byte limit) boundary tests --

    #[test]
    fn link_tag_shelter_by_type_empty_valid() {
        let tag = LinkTag::new(vec![]);
        let result = validate_create_link_tag(&LinkTypes::ShelterByType, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_shelter_by_type_at_limit_valid() {
        let tag = LinkTag::new(vec![0u8; 512]);
        let result = validate_create_link_tag(&LinkTypes::ShelterByType, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_shelter_by_type_over_limit_invalid() {
        let tag = LinkTag::new(vec![0u8; 513]);
        let result = validate_create_link_tag(&LinkTypes::ShelterByType, &tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- PersonToRegistration (256-byte limit) boundary tests --

    #[test]
    fn link_tag_person_to_registration_at_limit_valid() {
        let tag = LinkTag::new(vec![0xBB; 256]);
        let result = validate_create_link_tag(&LinkTypes::PersonToRegistration, &tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn link_tag_person_to_registration_over_limit_invalid() {
        let tag = LinkTag::new(vec![0xBB; 257]);
        let result = validate_create_link_tag(&LinkTypes::PersonToRegistration, &tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    // -- DoS prevention: massive tags rejected for all link types --

    #[test]
    fn link_tag_dos_prevention_all_types() {
        let massive_tag = LinkTag::new(vec![0xFF; 10_000]);
        let all_types = [
            LinkTypes::AllShelters,
            LinkTypes::OpenShelters,
            LinkTypes::ShelterToRegistration,
            LinkTypes::ShelterByType,
            LinkTypes::PersonToRegistration,
        ];
        for lt in &all_types {
            let result = validate_create_link_tag(lt, &massive_tag);
            assert!(
                matches!(result, ValidateCallbackResult::Invalid(_)),
                "Massive tag should be rejected for {:?}",
                lt
            );
        }
    }

    // -- Delete link tag tests --

    #[test]
    fn delete_link_tag_empty_valid() {
        let tag = LinkTag::new(vec![]);
        let result = validate_delete_link_tag(&tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn delete_link_tag_at_limit_valid() {
        let tag = LinkTag::new(vec![0u8; 256]);
        let result = validate_delete_link_tag(&tag);
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn delete_link_tag_over_limit_invalid() {
        let tag = LinkTag::new(vec![0u8; 257]);
        let result = validate_delete_link_tag(&tag);
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }
}
