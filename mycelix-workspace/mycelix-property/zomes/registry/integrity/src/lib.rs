// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Property Registry Integrity Zome
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
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Property(property) => {
                    validate_create_property(EntryCreationAction::Create(action), property)
                }
                EntryTypes::TitleDeed(deed) => {
                    validate_create_title_deed(EntryCreationAction::Create(action), deed)
                }
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::Property(property) => validate_update_property(action, property),
                EntryTypes::TitleDeed(_) => Ok(ValidateCallbackResult::Invalid(
                    "Title deeds cannot be updated".into(),
                )),
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::OwnerToProperties => Ok(ValidateCallbackResult::Valid),
            LinkTypes::PropertyToDeeds => Ok(ValidateCallbackResult::Valid),
            LinkTypes::LocationToProperty => Ok(ValidateCallbackResult::Valid),
            LinkTypes::PropertyToEncumbrances => Ok(ValidateCallbackResult::Valid),
        },
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
    if !property.owner_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Owner must be a valid DID".into(),
        ));
    }
    let mut total_share = 100.0;
    for co_owner in &property.co_owners {
        if !co_owner.did.starts_with("did:") {
            return Ok(ValidateCallbackResult::Invalid(
                "Co-owner must be a valid DID".into(),
            ));
        }
        if co_owner.share_percentage <= 0.0 || co_owner.share_percentage > 100.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Share must be between 0 and 100".into(),
            ));
        }
        total_share -= co_owner.share_percentage;
    }
    if total_share < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Total shares exceed 100%".into(),
        ));
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
    if !deed.owner_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Owner must be a valid DID".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
