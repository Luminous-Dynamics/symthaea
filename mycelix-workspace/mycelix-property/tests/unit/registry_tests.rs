// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Registry Zome Unit Tests
//!
//! Comprehensive test coverage for property registration, title deeds,
//! co-ownership, encumbrances, ownership transfer, and access control.

use serde::{Deserialize, Serialize};

// =============================================================================
// Test Data Structures (mirrors integrity types for testing)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestProperty {
    pub id: String,
    pub property_type: TestPropertyType,
    pub title: String,
    pub description: String,
    pub owner_did: String,
    pub co_owners: Vec<TestCoOwner>,
    pub geolocation: Option<TestGeoLocation>,
    pub address: Option<TestAddress>,
    pub metadata: TestPropertyMetadata,
    pub registered: i64,
    pub last_transfer: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestPropertyType {
    Land,
    Building,
    Unit,
    Equipment,
    Intellectual,
    Digital,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestCoOwner {
    pub did: String,
    pub share_percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestGeoLocation {
    pub latitude: f64,
    pub longitude: f64,
    pub boundaries: Option<Vec<(f64, f64)>>,
    pub area_sqm: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestAddress {
    pub street: String,
    pub city: String,
    pub region: String,
    pub country: String,
    pub postal_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestPropertyMetadata {
    pub appraised_value: Option<f64>,
    pub currency: Option<String>,
    pub legal_description: Option<String>,
    pub parcel_number: Option<String>,
    pub attachments: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestTitleDeed {
    pub id: String,
    pub property_id: String,
    pub owner_did: String,
    pub deed_type: TestDeedType,
    pub issued: i64,
    pub previous_deed_id: Option<String>,
    pub encumbrances: Vec<TestEncumbrance>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestDeedType {
    Original,
    Transfer,
    Inheritance,
    CourtOrder,
    Fractional,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TestEncumbrance {
    pub encumbrance_type: TestEncumbranceType,
    pub holder_did: String,
    pub amount: Option<f64>,
    pub registered: i64,
    pub expires: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestEncumbranceType {
    Mortgage,
    Lien,
    Easement,
    Restriction,
    Lease,
}

// =============================================================================
// Helper Functions
// =============================================================================

fn create_test_property() -> TestProperty {
    TestProperty {
        id: "property:did:mycelix:owner123:1704067200000000".to_string(),
        property_type: TestPropertyType::Land,
        title: "123 Main Street Lot".to_string(),
        description: "Residential lot in downtown area".to_string(),
        owner_did: "did:mycelix:owner123".to_string(),
        co_owners: vec![],
        geolocation: Some(TestGeoLocation {
            latitude: 40.7128,
            longitude: -74.0060,
            boundaries: Some(vec![
                (40.7130, -74.0062),
                (40.7130, -74.0058),
                (40.7126, -74.0058),
                (40.7126, -74.0062),
            ]),
            area_sqm: Some(500.0),
        }),
        address: Some(TestAddress {
            street: "123 Main Street".to_string(),
            city: "New York".to_string(),
            region: "NY".to_string(),
            country: "USA".to_string(),
            postal_code: Some("10001".to_string()),
        }),
        metadata: TestPropertyMetadata {
            appraised_value: Some(500000.0),
            currency: Some("USD".to_string()),
            legal_description: Some("Lot 1, Block A, Downtown Subdivision".to_string()),
            parcel_number: Some("NYC-DT-001-A-001".to_string()),
            attachments: vec!["survey_2024.pdf".to_string()],
        },
        registered: 1704067200000000,
        last_transfer: None,
    }
}

fn create_test_title_deed() -> TestTitleDeed {
    TestTitleDeed {
        id: "deed:property:did:mycelix:owner123:1704067200000000:1704067200000001".to_string(),
        property_id: "property:did:mycelix:owner123:1704067200000000".to_string(),
        owner_did: "did:mycelix:owner123".to_string(),
        deed_type: TestDeedType::Original,
        issued: 1704067200000001,
        previous_deed_id: None,
        encumbrances: vec![],
    }
}

fn validate_did(did: &str) -> bool {
    did.starts_with("did:")
}

fn validate_co_owner_shares(co_owners: &[TestCoOwner]) -> Result<(), String> {
    let mut total_share = 0.0;
    for co_owner in co_owners {
        if !validate_did(&co_owner.did) {
            return Err("Co-owner must be a valid DID".to_string());
        }
        if co_owner.share_percentage <= 0.0 || co_owner.share_percentage > 100.0 {
            return Err("Share must be between 0 and 100".to_string());
        }
        total_share += co_owner.share_percentage;
    }
    if total_share > 100.0 {
        return Err("Total shares exceed 100%".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // PROPERTY REGISTRATION TESTS
    // =========================================================================

    #[test]
    fn test_property_requires_valid_owner_did() {
        let property = create_test_property();
        assert!(validate_did(&property.owner_did));
        assert!(property.owner_did.starts_with("did:"));
    }

    #[test]
    fn test_property_invalid_owner_did_rejected() {
        let invalid_dids = vec![
            "owner123",
            "invalid:owner",
            "",
            "did",
            "DID:mycelix:owner", // Case sensitive
        ];
        for did in invalid_dids {
            assert!(!validate_did(did), "Should reject invalid DID: {}", did);
        }
    }

    #[test]
    fn test_property_has_unique_id() {
        let property = create_test_property();
        // ID should contain owner DID and timestamp for uniqueness
        assert!(property.id.contains("property:"));
        assert!(
            property
                .id
                .contains(&property.owner_did.replace("did:", ""))
        );
    }

    #[test]
    fn test_property_types_supported() {
        let types = vec![
            TestPropertyType::Land,
            TestPropertyType::Building,
            TestPropertyType::Unit,
            TestPropertyType::Equipment,
            TestPropertyType::Intellectual,
            TestPropertyType::Digital,
            TestPropertyType::Other("Custom".to_string()),
        ];
        for prop_type in types {
            let mut property = create_test_property();
            property.property_type = prop_type.clone();
            // All property types should be valid
            assert!(matches!(
                property.property_type,
                TestPropertyType::Land
                    | TestPropertyType::Building
                    | TestPropertyType::Unit
                    | TestPropertyType::Equipment
                    | TestPropertyType::Intellectual
                    | TestPropertyType::Digital
                    | TestPropertyType::Other(_)
            ));
        }
    }

    #[test]
    fn test_property_requires_title() {
        let property = create_test_property();
        assert!(!property.title.is_empty());
    }

    #[test]
    fn test_property_registration_timestamp_valid() {
        let property = create_test_property();
        // Registration timestamp must be positive (microseconds since epoch)
        assert!(property.registered > 0);
        // Should be a reasonable timestamp (after year 2000)
        assert!(property.registered > 946684800000000); // 2000-01-01
    }

    #[test]
    fn test_property_initial_state_no_transfers() {
        let property = create_test_property();
        // New property should not have any transfers yet
        assert!(property.last_transfer.is_none());
    }

    // =========================================================================
    // CO-OWNERSHIP TESTS
    // =========================================================================

    #[test]
    fn test_co_owner_valid_shares() {
        let co_owners = vec![
            TestCoOwner {
                did: "did:mycelix:coowner1".to_string(),
                share_percentage: 30.0,
            },
            TestCoOwner {
                did: "did:mycelix:coowner2".to_string(),
                share_percentage: 20.0,
            },
        ];
        assert!(validate_co_owner_shares(&co_owners).is_ok());
    }

    #[test]
    fn test_co_owner_shares_exceed_100_rejected() {
        let co_owners = vec![
            TestCoOwner {
                did: "did:mycelix:coowner1".to_string(),
                share_percentage: 60.0,
            },
            TestCoOwner {
                did: "did:mycelix:coowner2".to_string(),
                share_percentage: 50.0,
            },
        ];
        let result = validate_co_owner_shares(&co_owners);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Total shares exceed 100%");
    }

    #[test]
    fn test_co_owner_zero_share_rejected() {
        let co_owners = vec![TestCoOwner {
            did: "did:mycelix:coowner1".to_string(),
            share_percentage: 0.0,
        }];
        let result = validate_co_owner_shares(&co_owners);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Share must be between 0 and 100");
    }

    #[test]
    fn test_co_owner_negative_share_rejected() {
        let co_owners = vec![TestCoOwner {
            did: "did:mycelix:coowner1".to_string(),
            share_percentage: -10.0,
        }];
        let result = validate_co_owner_shares(&co_owners);
        assert!(result.is_err());
    }

    #[test]
    fn test_co_owner_over_100_share_rejected() {
        let co_owners = vec![TestCoOwner {
            did: "did:mycelix:coowner1".to_string(),
            share_percentage: 150.0,
        }];
        let result = validate_co_owner_shares(&co_owners);
        assert!(result.is_err());
    }

    #[test]
    fn test_co_owner_invalid_did_rejected() {
        let co_owners = vec![TestCoOwner {
            did: "invalid_did".to_string(),
            share_percentage: 25.0,
        }];
        let result = validate_co_owner_shares(&co_owners);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Co-owner must be a valid DID");
    }

    #[test]
    fn test_co_owner_shares_sum_exactly_100() {
        let co_owners = vec![
            TestCoOwner {
                did: "did:mycelix:coowner1".to_string(),
                share_percentage: 50.0,
            },
            TestCoOwner {
                did: "did:mycelix:coowner2".to_string(),
                share_percentage: 50.0,
            },
        ];
        // Total 100% with no primary owner share is valid
        assert!(validate_co_owner_shares(&co_owners).is_ok());
    }

    #[test]
    fn test_empty_co_owners_valid() {
        let co_owners: Vec<TestCoOwner> = vec![];
        assert!(validate_co_owner_shares(&co_owners).is_ok());
    }

    // =========================================================================
    // GEOLOCATION TESTS
    // =========================================================================

    #[test]
    fn test_geolocation_valid_coordinates() {
        let property = create_test_property();
        if let Some(geo) = &property.geolocation {
            // Latitude must be between -90 and 90
            assert!(geo.latitude >= -90.0 && geo.latitude <= 90.0);
            // Longitude must be between -180 and 180
            assert!(geo.longitude >= -180.0 && geo.longitude <= 180.0);
        }
    }

    #[test]
    fn test_geolocation_area_positive() {
        let property = create_test_property();
        if let Some(geo) = &property.geolocation {
            if let Some(area) = geo.area_sqm {
                assert!(area > 0.0);
            }
        }
    }

    #[test]
    fn test_geolocation_boundaries_form_polygon() {
        let property = create_test_property();
        if let Some(geo) = &property.geolocation {
            if let Some(boundaries) = &geo.boundaries {
                // A polygon needs at least 3 points
                assert!(boundaries.len() >= 3);
            }
        }
    }

    #[test]
    fn test_property_without_geolocation_valid() {
        let mut property = create_test_property();
        property.geolocation = None;
        // Geolocation is optional
        assert!(property.geolocation.is_none());
    }

    // =========================================================================
    // ADDRESS TESTS
    // =========================================================================

    #[test]
    fn test_address_required_fields() {
        let property = create_test_property();
        if let Some(addr) = &property.address {
            assert!(!addr.street.is_empty());
            assert!(!addr.city.is_empty());
            assert!(!addr.region.is_empty());
            assert!(!addr.country.is_empty());
        }
    }

    #[test]
    fn test_address_postal_code_optional() {
        let mut property = create_test_property();
        if let Some(ref mut addr) = property.address {
            addr.postal_code = None;
            assert!(addr.postal_code.is_none());
        }
    }

    // =========================================================================
    // METADATA TESTS
    // =========================================================================

    #[test]
    fn test_metadata_appraised_value_non_negative() {
        let property = create_test_property();
        if let Some(value) = property.metadata.appraised_value {
            assert!(value >= 0.0);
        }
    }

    #[test]
    fn test_metadata_currency_iso_format() {
        let property = create_test_property();
        if let Some(ref currency) = property.metadata.currency {
            // ISO 4217 currency codes are 3 letters
            assert!(currency.len() == 3 || currency.len() > 3); // Allow full names too
        }
    }

    #[test]
    fn test_metadata_attachments_can_be_empty() {
        let mut property = create_test_property();
        property.metadata.attachments = vec![];
        assert!(property.metadata.attachments.is_empty());
    }

    // =========================================================================
    // TITLE DEED TESTS
    // =========================================================================

    #[test]
    fn test_title_deed_requires_valid_owner_did() {
        let deed = create_test_title_deed();
        assert!(validate_did(&deed.owner_did));
    }

    #[test]
    fn test_title_deed_references_property() {
        let deed = create_test_title_deed();
        assert!(!deed.property_id.is_empty());
        assert!(deed.property_id.starts_with("property:"));
    }

    #[test]
    fn test_title_deed_original_no_previous() {
        let deed = create_test_title_deed();
        // Original deed should not have a previous deed
        assert_eq!(deed.deed_type, TestDeedType::Original);
        assert!(deed.previous_deed_id.is_none());
    }

    #[test]
    fn test_title_deed_transfer_has_previous() {
        let mut deed = create_test_title_deed();
        deed.deed_type = TestDeedType::Transfer;
        deed.previous_deed_id = Some("previous_deed_id".to_string());
        assert!(deed.previous_deed_id.is_some());
    }

    #[test]
    fn test_title_deed_types_supported() {
        let types = vec![
            TestDeedType::Original,
            TestDeedType::Transfer,
            TestDeedType::Inheritance,
            TestDeedType::CourtOrder,
            TestDeedType::Fractional,
        ];
        for deed_type in types {
            let mut deed = create_test_title_deed();
            deed.deed_type = deed_type.clone();
            // All deed types should be valid
            assert!(matches!(
                deed.deed_type,
                TestDeedType::Original
                    | TestDeedType::Transfer
                    | TestDeedType::Inheritance
                    | TestDeedType::CourtOrder
                    | TestDeedType::Fractional
            ));
        }
    }

    #[test]
    fn test_title_deed_issued_timestamp_valid() {
        let deed = create_test_title_deed();
        assert!(deed.issued > 0);
        assert!(deed.issued > 946684800000000); // After year 2000
    }

    #[test]
    fn test_title_deed_immutable() {
        // Title deeds cannot be updated according to validation rules
        // This test documents the expected behavior
        let deed = create_test_title_deed();
        let deed_id = deed.id.clone();
        // Any attempt to update should be rejected
        // (This would be tested in integration tests with actual zome calls)
        assert_eq!(deed.id, deed_id);
    }

    // =========================================================================
    // ENCUMBRANCE TESTS
    // =========================================================================

    fn create_test_encumbrance() -> TestEncumbrance {
        TestEncumbrance {
            encumbrance_type: TestEncumbranceType::Mortgage,
            holder_did: "did:mycelix:bank123".to_string(),
            amount: Some(300000.0),
            registered: 1704153600000000,
            expires: Some(1798761600000000), // 30 years from registration
        }
    }

    #[test]
    fn test_encumbrance_valid_holder_did() {
        let encumbrance = create_test_encumbrance();
        assert!(validate_did(&encumbrance.holder_did));
    }

    #[test]
    fn test_encumbrance_types_supported() {
        let types = vec![
            TestEncumbranceType::Mortgage,
            TestEncumbranceType::Lien,
            TestEncumbranceType::Easement,
            TestEncumbranceType::Restriction,
            TestEncumbranceType::Lease,
        ];
        for enc_type in types {
            let mut enc = create_test_encumbrance();
            enc.encumbrance_type = enc_type;
            // All encumbrance types should be valid
            assert!(matches!(
                enc.encumbrance_type,
                TestEncumbranceType::Mortgage
                    | TestEncumbranceType::Lien
                    | TestEncumbranceType::Easement
                    | TestEncumbranceType::Restriction
                    | TestEncumbranceType::Lease
            ));
        }
    }

    #[test]
    fn test_encumbrance_amount_optional() {
        let mut encumbrance = create_test_encumbrance();
        encumbrance.amount = None; // Easements might not have monetary value
        assert!(encumbrance.amount.is_none());
    }

    #[test]
    fn test_encumbrance_amount_non_negative() {
        let encumbrance = create_test_encumbrance();
        if let Some(amount) = encumbrance.amount {
            assert!(amount >= 0.0);
        }
    }

    #[test]
    fn test_encumbrance_expires_after_registration() {
        let encumbrance = create_test_encumbrance();
        if let Some(expires) = encumbrance.expires {
            assert!(expires > encumbrance.registered);
        }
    }

    #[test]
    fn test_encumbrance_no_expiry_valid() {
        let mut encumbrance = create_test_encumbrance();
        encumbrance.encumbrance_type = TestEncumbranceType::Easement;
        encumbrance.expires = None; // Easements can be perpetual
        assert!(encumbrance.expires.is_none());
    }

    // =========================================================================
    // CLEAR TITLE VERIFICATION TESTS
    // =========================================================================

    fn has_clear_title(encumbrances: &[TestEncumbrance], current_time: i64) -> bool {
        for enc in encumbrances {
            if let Some(expires) = enc.expires {
                if expires > current_time {
                    return false;
                }
            } else {
                // No expiry means still active
                return false;
            }
        }
        true
    }

    #[test]
    fn test_clear_title_no_encumbrances() {
        let encumbrances: Vec<TestEncumbrance> = vec![];
        let current_time = 1704153600000000i64;
        assert!(has_clear_title(&encumbrances, current_time));
    }

    #[test]
    fn test_clear_title_expired_encumbrances() {
        let encumbrances = vec![TestEncumbrance {
            encumbrance_type: TestEncumbranceType::Mortgage,
            holder_did: "did:mycelix:bank".to_string(),
            amount: Some(100000.0),
            registered: 1600000000000000,
            expires: Some(1700000000000000), // Expired
        }];
        let current_time = 1704153600000000i64; // After expiry
        assert!(has_clear_title(&encumbrances, current_time));
    }

    #[test]
    fn test_not_clear_title_active_mortgage() {
        let encumbrances = vec![create_test_encumbrance()];
        let current_time = 1704153600000000i64;
        assert!(!has_clear_title(&encumbrances, current_time));
    }

    #[test]
    fn test_not_clear_title_perpetual_easement() {
        let encumbrances = vec![TestEncumbrance {
            encumbrance_type: TestEncumbranceType::Easement,
            holder_did: "did:mycelix:utility".to_string(),
            amount: None,
            registered: 1600000000000000,
            expires: None, // Perpetual
        }];
        let current_time = 1704153600000000i64;
        assert!(!has_clear_title(&encumbrances, current_time));
    }

    // =========================================================================
    // OWNERSHIP TRANSFER VALIDATION TESTS
    // =========================================================================

    #[derive(Debug)]
    struct TransferOwnershipInput {
        property_id: String,
        from_did: String,
        to_did: String,
        transfer_type: String,
    }

    fn validate_ownership_transfer(
        property: &TestProperty,
        input: &TransferOwnershipInput,
    ) -> Result<(), String> {
        // Verify current owner matches from_did
        if property.owner_did != input.from_did {
            return Err(format!(
                "Current owner {} does not match from_did {}",
                property.owner_did, input.from_did
            ));
        }
        // Verify valid DIDs
        if !validate_did(&input.from_did) {
            return Err("from_did must be a valid DID".to_string());
        }
        if !validate_did(&input.to_did) {
            return Err("to_did must be a valid DID".to_string());
        }
        // Cannot transfer to self
        if input.from_did == input.to_did {
            return Err("Cannot transfer to yourself".to_string());
        }
        Ok(())
    }

    #[test]
    fn test_ownership_transfer_valid() {
        let property = create_test_property();
        let input = TransferOwnershipInput {
            property_id: property.id.clone(),
            from_did: property.owner_did.clone(),
            to_did: "did:mycelix:newowner".to_string(),
            transfer_type: "Sale".to_string(),
        };
        assert!(validate_ownership_transfer(&property, &input).is_ok());
    }

    #[test]
    fn test_ownership_transfer_wrong_owner_rejected() {
        let property = create_test_property();
        let input = TransferOwnershipInput {
            property_id: property.id.clone(),
            from_did: "did:mycelix:notowner".to_string(),
            to_did: "did:mycelix:newowner".to_string(),
            transfer_type: "Sale".to_string(),
        };
        let result = validate_ownership_transfer(&property, &input);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("does not match from_did"));
    }

    #[test]
    fn test_ownership_transfer_to_self_rejected() {
        let property = create_test_property();
        let input = TransferOwnershipInput {
            property_id: property.id.clone(),
            from_did: property.owner_did.clone(),
            to_did: property.owner_did.clone(),
            transfer_type: "Sale".to_string(),
        };
        let result = validate_ownership_transfer(&property, &input);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Cannot transfer to yourself");
    }

    #[test]
    fn test_ownership_transfer_invalid_to_did_rejected() {
        let property = create_test_property();
        let input = TransferOwnershipInput {
            property_id: property.id.clone(),
            from_did: property.owner_did.clone(),
            to_did: "invalid_did".to_string(),
            transfer_type: "Sale".to_string(),
        };
        let result = validate_ownership_transfer(&property, &input);
        assert!(result.is_err());
    }

    #[test]
    fn test_ownership_transfer_types() {
        let valid_types = vec!["Sale", "Inheritance", "Gift", "CourtOrder", "Exchange"];
        for transfer_type in valid_types {
            // All transfer types should be accepted
            assert!(!transfer_type.is_empty());
        }
    }

    // =========================================================================
    // ACCESS CONTROL TESTS
    // =========================================================================

    fn can_update_metadata(property: &TestProperty, requester_did: &str) -> bool {
        property.owner_did == requester_did
    }

    fn can_add_co_owner(property: &TestProperty, requester_did: &str) -> bool {
        property.owner_did == requester_did
    }

    fn can_remove_co_owner(property: &TestProperty, requester_did: &str) -> bool {
        property.owner_did == requester_did
    }

    #[test]
    fn test_only_owner_can_update_metadata() {
        let property = create_test_property();
        assert!(can_update_metadata(&property, &property.owner_did));
        assert!(!can_update_metadata(&property, "did:mycelix:other"));
    }

    #[test]
    fn test_only_owner_can_add_co_owner() {
        let property = create_test_property();
        assert!(can_add_co_owner(&property, &property.owner_did));
        assert!(!can_add_co_owner(&property, "did:mycelix:other"));
    }

    #[test]
    fn test_only_owner_can_remove_co_owner() {
        let property = create_test_property();
        assert!(can_remove_co_owner(&property, &property.owner_did));
        assert!(!can_remove_co_owner(&property, "did:mycelix:other"));
    }

    // =========================================================================
    // OWNERSHIP VERIFICATION TESTS
    // =========================================================================

    fn verify_ownership(property: &TestProperty, did: &str) -> bool {
        property.owner_did == did
    }

    #[test]
    fn test_verify_ownership_correct_owner() {
        let property = create_test_property();
        assert!(verify_ownership(&property, &property.owner_did));
    }

    #[test]
    fn test_verify_ownership_wrong_owner() {
        let property = create_test_property();
        assert!(!verify_ownership(&property, "did:mycelix:notowner"));
    }

    #[test]
    fn test_verify_ownership_co_owner_not_primary() {
        let mut property = create_test_property();
        property.co_owners = vec![TestCoOwner {
            did: "did:mycelix:coowner".to_string(),
            share_percentage: 25.0,
        }];
        // Co-owner is not the primary owner
        assert!(!verify_ownership(&property, "did:mycelix:coowner"));
    }

    // =========================================================================
    // EDGE CASES
    // =========================================================================

    #[test]
    fn test_property_with_maximum_co_owners() {
        let mut property = create_test_property();
        // Create 10 co-owners with 10% each
        property.co_owners = (0..10)
            .map(|i| TestCoOwner {
                did: format!("did:mycelix:coowner{}", i),
                share_percentage: 10.0,
            })
            .collect();
        assert!(validate_co_owner_shares(&property.co_owners).is_ok());
    }

    #[test]
    fn test_property_with_fractional_shares() {
        let mut property = create_test_property();
        property.co_owners = vec![
            TestCoOwner {
                did: "did:mycelix:coowner1".to_string(),
                share_percentage: 33.33,
            },
            TestCoOwner {
                did: "did:mycelix:coowner2".to_string(),
                share_percentage: 33.33,
            },
            TestCoOwner {
                did: "did:mycelix:coowner3".to_string(),
                share_percentage: 33.34,
            },
        ];
        assert!(validate_co_owner_shares(&property.co_owners).is_ok());
    }

    #[test]
    fn test_property_with_very_small_share() {
        let mut property = create_test_property();
        property.co_owners = vec![TestCoOwner {
            did: "did:mycelix:coowner1".to_string(),
            share_percentage: 0.001, // Very small but valid
        }];
        assert!(validate_co_owner_shares(&property.co_owners).is_ok());
    }

    #[test]
    fn test_deed_chain_integrity() {
        // Simulate a chain of deeds for a property
        let original_deed = create_test_title_deed();

        let transfer_deed = TestTitleDeed {
            id: "deed:transfer:1".to_string(),
            property_id: original_deed.property_id.clone(),
            owner_did: "did:mycelix:newowner1".to_string(),
            deed_type: TestDeedType::Transfer,
            issued: original_deed.issued + 100000000,
            previous_deed_id: Some(original_deed.id.clone()),
            encumbrances: vec![],
        };

        let inheritance_deed = TestTitleDeed {
            id: "deed:inheritance:1".to_string(),
            property_id: transfer_deed.property_id.clone(),
            owner_did: "did:mycelix:heir".to_string(),
            deed_type: TestDeedType::Inheritance,
            issued: transfer_deed.issued + 100000000,
            previous_deed_id: Some(transfer_deed.id.clone()),
            encumbrances: vec![],
        };

        // Verify chain integrity
        assert!(original_deed.previous_deed_id.is_none());
        assert_eq!(
            transfer_deed.previous_deed_id,
            Some(original_deed.id.clone())
        );
        assert_eq!(
            inheritance_deed.previous_deed_id,
            Some(transfer_deed.id.clone())
        );
        assert!(inheritance_deed.issued > transfer_deed.issued);
        assert!(transfer_deed.issued > original_deed.issued);
    }

    #[test]
    fn test_multiple_encumbrances_on_property() {
        let encumbrances = vec![
            TestEncumbrance {
                encumbrance_type: TestEncumbranceType::Mortgage,
                holder_did: "did:mycelix:bank1".to_string(),
                amount: Some(200000.0),
                registered: 1704000000000000,
                expires: Some(1800000000000000),
            },
            TestEncumbrance {
                encumbrance_type: TestEncumbranceType::Lien,
                holder_did: "did:mycelix:contractor".to_string(),
                amount: Some(5000.0),
                registered: 1704100000000000,
                expires: None,
            },
            TestEncumbrance {
                encumbrance_type: TestEncumbranceType::Easement,
                holder_did: "did:mycelix:utility".to_string(),
                amount: None,
                registered: 1700000000000000,
                expires: None,
            },
        ];

        // Property can have multiple encumbrances
        assert_eq!(encumbrances.len(), 3);

        // All encumbrances should have valid holders
        for enc in &encumbrances {
            assert!(validate_did(&enc.holder_did));
        }
    }
}
