//! Property Registry Integrity Zome
use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Property {
    pub id: String,
    pub property_type: PropertyType,
    pub title: String,
    pub description: String,
    pub owner_did: String,
    pub co_owners: Vec<CoOwner>,
    pub geolocation: Option<GeoLocation>,
    pub address: Option<Address>,
    pub metadata: PropertyMetadata,
    pub registered: Timestamp,
    pub last_transfer: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PropertyType {
    Land,
    Building,
    Unit,
    Equipment,
    Intellectual,
    Digital,
    Other(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct CoOwner {
    pub did: String,
    pub share_percentage: f64,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GeoLocation {
    pub latitude: f64,
    pub longitude: f64,
    pub boundaries: Option<Vec<(f64, f64)>>,
    pub area_sqm: Option<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Address {
    pub street: String,
    pub city: String,
    pub region: String,
    pub country: String,
    pub postal_code: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PropertyMetadata {
    pub appraised_value: Option<f64>,
    pub currency: Option<String>,
    pub legal_description: Option<String>,
    pub parcel_number: Option<String>,
    pub attachments: Vec<String>,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TitleDeed {
    pub id: String,
    pub property_id: String,
    pub owner_did: String,
    pub deed_type: DeedType,
    pub issued: Timestamp,
    pub previous_deed_id: Option<String>,
    pub encumbrances: Vec<Encumbrance>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DeedType {
    Original,
    Transfer,
    Inheritance,
    CourtOrder,
    Fractional,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Encumbrance {
    pub encumbrance_type: EncumbranceType,
    pub holder_did: String,
    pub amount: Option<f64>,
    pub registered: Timestamp,
    pub expires: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EncumbranceType {
    Mortgage,
    Lien,
    Easement,
    Restriction,
    Lease,
}

/// Anchor entry for deterministic link bases from strings
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Property(Property),
    TitleDeed(TitleDeed),
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
}

#[hdk_link_types]
pub enum LinkTypes {
    OwnerToProperties,
    PropertyToDeeds,
    LocationToProperty,
    PropertyToEncumbrances,
}

/// Genesis self-check
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => {
                match app_entry {
                    EntryTypes::Property(property) => {
                        validate_create_property(EntryCreationAction::Create(action), property)
                    }
                    EntryTypes::TitleDeed(deed) => {
                        validate_create_title_deed(EntryCreationAction::Create(action), deed)
                    }
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                }
            }
            OpEntry::UpdateEntry { app_entry, action, .. } => {
                match app_entry {
                    EntryTypes::Property(property) => {
                        validate_update_property(action, property)
                    }
                    EntryTypes::TitleDeed(_) => {
                        Ok(ValidateCallbackResult::Invalid(
                            "Title deeds cannot be updated".into(),
                        ))
                    }
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => {
            match link_type {
                LinkTypes::OwnerToProperties => Ok(ValidateCallbackResult::Valid),
                LinkTypes::PropertyToDeeds => Ok(ValidateCallbackResult::Valid),
                LinkTypes::LocationToProperty => Ok(ValidateCallbackResult::Valid),
                LinkTypes::PropertyToEncumbrances => Ok(ValidateCallbackResult::Valid),
            }
        }
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_property(
    _action: EntryCreationAction,
    property: Property,
) -> ExternResult<ValidateCallbackResult> {
    if property.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Property ID cannot be empty".into()));
    }
    if property.title.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Property title cannot be empty".into()));
    }
    if !property.owner_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Owner must be a valid DID".into()));
    }
    if let Some(ref geo) = property.geolocation {
        if geo.latitude < -90.0 || geo.latitude > 90.0 {
            return Ok(ValidateCallbackResult::Invalid("Latitude must be between -90 and 90".into()));
        }
        if geo.longitude < -180.0 || geo.longitude > 180.0 {
            return Ok(ValidateCallbackResult::Invalid("Longitude must be between -180 and 180".into()));
        }
        if let Some(area) = geo.area_sqm {
            if area <= 0.0 {
                return Ok(ValidateCallbackResult::Invalid("Area must be positive".into()));
            }
        }
    }
    if let Some(ref meta) = property.metadata.appraised_value {
        if *meta < 0.0 {
            return Ok(ValidateCallbackResult::Invalid("Appraised value cannot be negative".into()));
        }
    }
    let mut total_share = 100.0;
    for co_owner in &property.co_owners {
        if !co_owner.did.starts_with("did:") {
            return Ok(ValidateCallbackResult::Invalid("Co-owner must be a valid DID".into()));
        }
        if co_owner.share_percentage <= 0.0 || co_owner.share_percentage > 100.0 {
            return Ok(ValidateCallbackResult::Invalid("Share must be between 0 and 100".into()));
        }
        total_share -= co_owner.share_percentage;
    }
    if total_share < 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Total shares exceed 100%".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_property(
    _action: Update,
    _property: Property,
) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_title_deed(
    _action: EntryCreationAction,
    deed: TitleDeed,
) -> ExternResult<ValidateCallbackResult> {
    if deed.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Deed ID cannot be empty".into()));
    }
    if deed.property_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Deed property_id cannot be empty".into()));
    }
    if !deed.owner_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid("Owner must be a valid DID".into()));
    }
    for enc in &deed.encumbrances {
        if !enc.holder_did.starts_with("did:") {
            return Ok(ValidateCallbackResult::Invalid("Encumbrance holder must be a valid DID".into()));
        }
        if let Some(amount) = enc.amount {
            if amount < 0.0 {
                return Ok(ValidateCallbackResult::Invalid("Encumbrance amount cannot be negative".into()));
            }
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

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

    fn fake_entry_creation_action() -> EntryCreationAction {
        EntryCreationAction::Create(fake_create())
    }

    fn is_valid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Valid))
    }

    fn is_invalid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Invalid(_)))
    }

    fn make_property() -> Property {
        Property {
            id: "prop-001".into(),
            property_type: PropertyType::Building,
            title: "Community Center".into(),
            description: "A shared community building".into(),
            owner_did: "did:key:z6Mk001".into(),
            co_owners: vec![],
            geolocation: Some(GeoLocation {
                latitude: 32.9483,
                longitude: -96.7299,
                boundaries: None,
                area_sqm: Some(500.0),
            }),
            address: Some(Address {
                street: "123 Main St".into(),
                city: "Richardson".into(),
                region: "Texas".into(),
                country: "US".into(),
                postal_code: Some("75080".into()),
            }),
            metadata: PropertyMetadata {
                appraised_value: Some(250_000.0),
                currency: Some("USD".into()),
                legal_description: Some("Lot 1, Block A".into()),
                parcel_number: Some("R000001".into()),
                attachments: vec![],
            },
            registered: Timestamp::from_micros(1_000_000),
            last_transfer: None,
        }
    }

    fn make_title_deed() -> TitleDeed {
        TitleDeed {
            id: "deed-001".into(),
            property_id: "prop-001".into(),
            owner_did: "did:key:z6Mk001".into(),
            deed_type: DeedType::Original,
            issued: Timestamp::from_micros(1_000_000),
            previous_deed_id: None,
            encumbrances: vec![],
        }
    }

    // ========================================================================
    // PROPERTY SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn property_serde_roundtrip() {
        let property = make_property();
        let json = serde_json::to_string(&property).unwrap();
        let deserialized: Property = serde_json::from_str(&json).unwrap();
        assert_eq!(property, deserialized);
    }

    #[test]
    fn property_type_all_variants_serde() {
        let variants = vec![
            PropertyType::Land,
            PropertyType::Building,
            PropertyType::Unit,
            PropertyType::Equipment,
            PropertyType::Intellectual,
            PropertyType::Digital,
            PropertyType::Other("Custom".into()),
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let deserialized: PropertyType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, deserialized);
        }
    }

    #[test]
    fn co_owner_serde_roundtrip() {
        let co_owner = CoOwner {
            did: "did:key:z6Mk002".into(),
            share_percentage: 25.0,
        };
        let json = serde_json::to_string(&co_owner).unwrap();
        let deserialized: CoOwner = serde_json::from_str(&json).unwrap();
        assert_eq!(co_owner, deserialized);
    }

    #[test]
    fn geolocation_serde_roundtrip() {
        let geo = GeoLocation {
            latitude: 45.5231,
            longitude: -122.6765,
            boundaries: Some(vec![(45.5, -122.7), (45.6, -122.6), (45.5, -122.5)]),
            area_sqm: Some(10_000.0),
        };
        let json = serde_json::to_string(&geo).unwrap();
        let deserialized: GeoLocation = serde_json::from_str(&json).unwrap();
        assert_eq!(geo, deserialized);
    }

    #[test]
    fn address_serde_roundtrip() {
        let addr = Address {
            street: "456 Oak Ave".into(),
            city: "Portland".into(),
            region: "Oregon".into(),
            country: "US".into(),
            postal_code: Some("97201".into()),
        };
        let json = serde_json::to_string(&addr).unwrap();
        let deserialized: Address = serde_json::from_str(&json).unwrap();
        assert_eq!(addr, deserialized);
    }

    #[test]
    fn property_metadata_serde_roundtrip() {
        let meta = PropertyMetadata {
            appraised_value: Some(500_000.0),
            currency: Some("EUR".into()),
            legal_description: Some("Section 12, Township 1N".into()),
            parcel_number: None,
            attachments: vec!["doc1.pdf".into(), "photo.jpg".into()],
        };
        let json = serde_json::to_string(&meta).unwrap();
        let deserialized: PropertyMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(meta, deserialized);
    }

    // ========================================================================
    // TITLE DEED SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn title_deed_serde_roundtrip() {
        let deed = make_title_deed();
        let json = serde_json::to_string(&deed).unwrap();
        let deserialized: TitleDeed = serde_json::from_str(&json).unwrap();
        assert_eq!(deed, deserialized);
    }

    #[test]
    fn deed_type_all_variants_serde() {
        let variants = vec![
            DeedType::Original,
            DeedType::Transfer,
            DeedType::Inheritance,
            DeedType::CourtOrder,
            DeedType::Fractional,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let deserialized: DeedType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, deserialized);
        }
    }

    #[test]
    fn encumbrance_serde_roundtrip() {
        let enc = Encumbrance {
            encumbrance_type: EncumbranceType::Mortgage,
            holder_did: "did:key:z6MkBank".into(),
            amount: Some(150_000.0),
            registered: Timestamp::from_micros(500_000),
            expires: Some(Timestamp::from_micros(9_000_000)),
        };
        let json = serde_json::to_string(&enc).unwrap();
        let deserialized: Encumbrance = serde_json::from_str(&json).unwrap();
        assert_eq!(enc, deserialized);
    }

    #[test]
    fn encumbrance_type_all_variants_serde() {
        let variants = vec![
            EncumbranceType::Mortgage,
            EncumbranceType::Lien,
            EncumbranceType::Easement,
            EncumbranceType::Restriction,
            EncumbranceType::Lease,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).unwrap();
            let deserialized: EncumbranceType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, deserialized);
        }
    }

    // ========================================================================
    // PROPERTY VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_property_passes() {
        let result = validate_create_property(fake_entry_creation_action(), make_property());
        assert!(is_valid(&result));
    }

    #[test]
    fn property_owner_without_did_prefix_rejected() {
        let mut property = make_property();
        property.owner_did = "not-a-did".into();
        let result = validate_create_property(fake_entry_creation_action(), property);
        assert!(is_invalid(&result));
    }

    #[test]
    fn property_owner_empty_string_rejected() {
        let mut property = make_property();
        property.owner_did = "".into();
        let result = validate_create_property(fake_entry_creation_action(), property);
        assert!(is_invalid(&result));
    }

    #[test]
    fn property_co_owner_without_did_prefix_rejected() {
        let mut property = make_property();
        property.co_owners = vec![CoOwner {
            did: "invalid-co-owner".into(),
            share_percentage: 25.0,
        }];
        let result = validate_create_property(fake_entry_creation_action(), property);
        assert!(is_invalid(&result));
    }

    #[test]
    fn property_co_owner_zero_share_rejected() {
        let mut property = make_property();
        property.co_owners = vec![CoOwner {
            did: "did:key:z6Mk002".into(),
            share_percentage: 0.0,
        }];
        let result = validate_create_property(fake_entry_creation_action(), property);
        assert!(is_invalid(&result));
    }

    #[test]
    fn property_co_owner_negative_share_rejected() {
        let mut property = make_property();
        property.co_owners = vec![CoOwner {
            did: "did:key:z6Mk002".into(),
            share_percentage: -5.0,
        }];
        let result = validate_create_property(fake_entry_creation_action(), property);
        assert!(is_invalid(&result));
    }

    #[test]
    fn property_co_owner_share_over_100_rejected() {
        let mut property = make_property();
        property.co_owners = vec![CoOwner {
            did: "did:key:z6Mk002".into(),
            share_percentage: 100.1,
        }];
        let result = validate_create_property(fake_entry_creation_action(), property);
        assert!(is_invalid(&result));
    }

    #[test]
    fn property_co_owner_share_exactly_100_accepted() {
        let mut property = make_property();
        property.co_owners = vec![CoOwner {
            did: "did:key:z6Mk002".into(),
            share_percentage: 100.0,
        }];
        let result = validate_create_property(fake_entry_creation_action(), property);
        assert!(is_valid(&result));
    }

    #[test]
    fn property_co_owner_small_positive_share_accepted() {
        let mut property = make_property();
        property.co_owners = vec![CoOwner {
            did: "did:key:z6Mk002".into(),
            share_percentage: 0.01,
        }];
        let result = validate_create_property(fake_entry_creation_action(), property);
        assert!(is_valid(&result));
    }

    #[test]
    fn property_total_shares_exceeding_100_rejected() {
        let mut property = make_property();
        property.co_owners = vec![
            CoOwner {
                did: "did:key:z6Mk002".into(),
                share_percentage: 60.0,
            },
            CoOwner {
                did: "did:key:z6Mk003".into(),
                share_percentage: 50.0,
            },
        ];
        let result = validate_create_property(fake_entry_creation_action(), property);
        assert!(is_invalid(&result));
    }

    #[test]
    fn property_total_shares_exactly_100_accepted() {
        let mut property = make_property();
        property.co_owners = vec![
            CoOwner {
                did: "did:key:z6Mk002".into(),
                share_percentage: 50.0,
            },
            CoOwner {
                did: "did:key:z6Mk003".into(),
                share_percentage: 50.0,
            },
        ];
        let result = validate_create_property(fake_entry_creation_action(), property);
        assert!(is_valid(&result));
    }

    #[test]
    fn property_no_co_owners_accepted() {
        let property = make_property(); // Default has empty co_owners
        let result = validate_create_property(fake_entry_creation_action(), property);
        assert!(is_valid(&result));
    }

    #[test]
    fn property_multiple_valid_co_owners_accepted() {
        let mut property = make_property();
        property.co_owners = vec![
            CoOwner {
                did: "did:key:z6Mk002".into(),
                share_percentage: 25.0,
            },
            CoOwner {
                did: "did:key:z6Mk003".into(),
                share_percentage: 25.0,
            },
            CoOwner {
                did: "did:key:z6Mk004".into(),
                share_percentage: 25.0,
            },
        ];
        let result = validate_create_property(fake_entry_creation_action(), property);
        assert!(is_valid(&result));
    }

    #[test]
    fn property_second_co_owner_invalid_did_rejected() {
        let mut property = make_property();
        property.co_owners = vec![
            CoOwner {
                did: "did:key:z6Mk002".into(),
                share_percentage: 25.0,
            },
            CoOwner {
                did: "not-a-did".into(),
                share_percentage: 25.0,
            },
        ];
        let result = validate_create_property(fake_entry_creation_action(), property);
        assert!(is_invalid(&result));
    }

    // ========================================================================
    // TITLE DEED VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_title_deed_passes() {
        let result = validate_create_title_deed(fake_entry_creation_action(), make_title_deed());
        assert!(is_valid(&result));
    }

    #[test]
    fn title_deed_owner_without_did_prefix_rejected() {
        let mut deed = make_title_deed();
        deed.owner_did = "not-a-did".into();
        let result = validate_create_title_deed(fake_entry_creation_action(), deed);
        assert!(is_invalid(&result));
    }

    #[test]
    fn title_deed_owner_empty_string_rejected() {
        let mut deed = make_title_deed();
        deed.owner_did = "".into();
        let result = validate_create_title_deed(fake_entry_creation_action(), deed);
        assert!(is_invalid(&result));
    }

    #[test]
    fn title_deed_with_encumbrances_passes() {
        let mut deed = make_title_deed();
        deed.encumbrances = vec![Encumbrance {
            encumbrance_type: EncumbranceType::Mortgage,
            holder_did: "did:key:z6MkBank".into(),
            amount: Some(200_000.0),
            registered: Timestamp::from_micros(1_000_000),
            expires: None,
        }];
        let result = validate_create_title_deed(fake_entry_creation_action(), deed);
        assert!(is_valid(&result));
    }

    #[test]
    fn title_deed_transfer_type_passes() {
        let mut deed = make_title_deed();
        deed.deed_type = DeedType::Transfer;
        deed.previous_deed_id = Some("deed-000".into());
        let result = validate_create_title_deed(fake_entry_creation_action(), deed);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // NEW PROPERTY VALIDATION TESTS (field-level)
    // ========================================================================

    #[test]
    fn property_empty_id_rejected() {
        let mut p = make_property();
        p.id = "".into();
        let result = validate_create_property(fake_entry_creation_action(), p);
        assert!(is_invalid(&result));
    }

    #[test]
    fn property_whitespace_id_rejected() {
        let mut p = make_property();
        p.id = "   ".into();
        let result = validate_create_property(fake_entry_creation_action(), p);
        assert!(is_invalid(&result));
    }

    #[test]
    fn property_empty_title_rejected() {
        let mut p = make_property();
        p.title = "".into();
        let result = validate_create_property(fake_entry_creation_action(), p);
        assert!(is_invalid(&result));
    }

    #[test]
    fn property_geolocation_invalid_lat_rejected() {
        let mut p = make_property();
        p.geolocation = Some(GeoLocation {
            latitude: 91.0,
            longitude: -96.7,
            boundaries: None,
            area_sqm: Some(500.0),
        });
        let result = validate_create_property(fake_entry_creation_action(), p);
        assert!(is_invalid(&result));
    }

    #[test]
    fn property_geolocation_invalid_lon_rejected() {
        let mut p = make_property();
        p.geolocation = Some(GeoLocation {
            latitude: 32.9,
            longitude: -181.0,
            boundaries: None,
            area_sqm: Some(500.0),
        });
        let result = validate_create_property(fake_entry_creation_action(), p);
        assert!(is_invalid(&result));
    }

    #[test]
    fn property_geolocation_zero_area_rejected() {
        let mut p = make_property();
        p.geolocation = Some(GeoLocation {
            latitude: 32.9,
            longitude: -96.7,
            boundaries: None,
            area_sqm: Some(0.0),
        });
        let result = validate_create_property(fake_entry_creation_action(), p);
        assert!(is_invalid(&result));
    }

    #[test]
    fn property_geolocation_negative_area_rejected() {
        let mut p = make_property();
        p.geolocation = Some(GeoLocation {
            latitude: 32.9,
            longitude: -96.7,
            boundaries: None,
            area_sqm: Some(-100.0),
        });
        let result = validate_create_property(fake_entry_creation_action(), p);
        assert!(is_invalid(&result));
    }

    #[test]
    fn property_negative_appraised_value_rejected() {
        let mut p = make_property();
        p.metadata.appraised_value = Some(-1.0);
        let result = validate_create_property(fake_entry_creation_action(), p);
        assert!(is_invalid(&result));
    }

    #[test]
    fn property_no_geolocation_passes() {
        let mut p = make_property();
        p.geolocation = None;
        let result = validate_create_property(fake_entry_creation_action(), p);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // NEW TITLE DEED VALIDATION TESTS (field-level)
    // ========================================================================

    #[test]
    fn deed_empty_id_rejected() {
        let mut deed = make_title_deed();
        deed.id = "".into();
        let result = validate_create_title_deed(fake_entry_creation_action(), deed);
        assert!(is_invalid(&result));
    }

    #[test]
    fn deed_empty_property_id_rejected() {
        let mut deed = make_title_deed();
        deed.property_id = "".into();
        let result = validate_create_title_deed(fake_entry_creation_action(), deed);
        assert!(is_invalid(&result));
    }

    #[test]
    fn deed_encumbrance_invalid_did_rejected() {
        let mut deed = make_title_deed();
        deed.encumbrances = vec![Encumbrance {
            encumbrance_type: EncumbranceType::Mortgage,
            holder_did: "not-a-did".into(),
            amount: Some(100_000.0),
            registered: Timestamp::from_micros(1_000_000),
            expires: None,
        }];
        let result = validate_create_title_deed(fake_entry_creation_action(), deed);
        assert!(is_invalid(&result));
    }

    #[test]
    fn deed_encumbrance_negative_amount_rejected() {
        let mut deed = make_title_deed();
        deed.encumbrances = vec![Encumbrance {
            encumbrance_type: EncumbranceType::Lien,
            holder_did: "did:key:z6MkBank".into(),
            amount: Some(-500.0),
            registered: Timestamp::from_micros(1_000_000),
            expires: None,
        }];
        let result = validate_create_title_deed(fake_entry_creation_action(), deed);
        assert!(is_invalid(&result));
    }

    #[test]
    fn deed_encumbrance_valid_passes() {
        let mut deed = make_title_deed();
        deed.encumbrances = vec![Encumbrance {
            encumbrance_type: EncumbranceType::Easement,
            holder_did: "did:key:z6MkNeighbor".into(),
            amount: None,
            registered: Timestamp::from_micros(1_000_000),
            expires: None,
        }];
        let result = validate_create_title_deed(fake_entry_creation_action(), deed);
        assert!(is_valid(&result));
    }
}
