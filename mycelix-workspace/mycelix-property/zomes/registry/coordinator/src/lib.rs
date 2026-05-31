// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Property Registry Coordinator Zome
use hdk::prelude::*;
use mycelix_property_shared::batch::links_to_records;
use registry_integrity::*;

/// Get or create an anchor entry and return its EntryHash for use as link base
fn anchor_hash(anchor_string: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_string.to_string());
    let _ = create_entry(&EntryTypes::Anchor(anchor.clone()));
    hash_entry(&anchor)
}

#[hdk_extern]
pub fn register_property(input: RegisterPropertyInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let property = Property {
        id: format!("property:{}:{}", input.owner_did, now.as_micros()),
        property_type: input.property_type,
        title: input.title,
        description: input.description,
        owner_did: input.owner_did.clone(),
        co_owners: input.co_owners,
        geolocation: input.geolocation.clone(),
        address: input.address,
        metadata: input.metadata,
        registered: now,
        last_transfer: None,
    };

    let action_hash = create_entry(&EntryTypes::Property(property.clone()))?;
    create_link(
        anchor_hash(&input.owner_did)?,
        action_hash.clone(),
        LinkTypes::OwnerToProperties,
        (),
    )?;

    // Link by location if available
    if let Some(geo) = input.geolocation {
        let geo_key = format!(
            "geo:{}:{}",
            (geo.latitude * 1000.0) as i64,
            (geo.longitude * 1000.0) as i64
        );
        create_link(
            anchor_hash(&geo_key)?,
            action_hash.clone(),
            LinkTypes::LocationToProperty,
            (),
        )?;
    }

    // Create initial title deed
    let deed = TitleDeed {
        id: format!("deed:{}:{}", property.id, now.as_micros()),
        property_id: property.id,
        owner_did: input.owner_did,
        deed_type: DeedType::Original,
        issued: now,
        previous_deed_id: None,
        encumbrances: Vec::new(),
    };
    let deed_hash = create_entry(&EntryTypes::TitleDeed(deed))?;
    create_link(
        action_hash.clone(),
        deed_hash,
        LinkTypes::PropertyToDeeds,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterPropertyInput {
    pub property_type: PropertyType,
    pub title: String,
    pub description: String,
    pub owner_did: String,
    pub co_owners: Vec<CoOwner>,
    pub geolocation: Option<GeoLocation>,
    pub address: Option<Address>,
    pub metadata: PropertyMetadata,
}

#[hdk_extern]
pub fn get_property(property_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Property,
        )?))
        .include_entries(true);
    for record in query(filter)? {
        if let Some(property) = record.entry().to_app_option::<Property>().ok().flatten() {
            if property.id == property_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

/// Get all properties owned by a DID
///
/// OPTIMIZED: Uses batch query to avoid N+1 pattern
#[hdk_extern]
pub fn get_owner_properties(did: String) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&did)?, LinkTypes::OwnerToProperties)?,
        GetStrategy::default(),
    )?;
    // FIXED N+1: Use batch fetch instead of individual get() calls
    links_to_records(links)
}

#[hdk_extern]
pub fn add_encumbrance(input: AddEncumbranceInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::TitleDeed,
        )?))
        .include_entries(true);
    for record in query(filter)? {
        if let Some(deed) = record.entry().to_app_option::<TitleDeed>().ok().flatten() {
            if deed.property_id == input.property_id {
                let now = sys_time()?;
                let new_encumbrance = Encumbrance {
                    encumbrance_type: input.encumbrance_type,
                    holder_did: input.holder_did,
                    amount: input.amount,
                    registered: now,
                    expires: input.expires,
                };
                let mut encumbrances = deed.encumbrances.clone();
                encumbrances.push(new_encumbrance);
                let updated = TitleDeed {
                    encumbrances,
                    ..deed
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::TitleDeed(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Property not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddEncumbranceInput {
    pub property_id: String,
    pub encumbrance_type: EncumbranceType,
    pub holder_did: String,
    pub amount: Option<f64>,
    pub expires: Option<Timestamp>,
}

/// Search for properties by location
///
/// OPTIMIZED: Uses batch query to avoid N+1 pattern
#[hdk_extern]
pub fn search_by_location(input: LocationSearchInput) -> ExternResult<Vec<Record>> {
    let geo_key = format!(
        "geo:{}:{}",
        (input.latitude * 1000.0) as i64,
        (input.longitude * 1000.0) as i64
    );
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&geo_key)?, LinkTypes::LocationToProperty)?,
        GetStrategy::default(),
    )?;
    // FIXED N+1: Use batch fetch instead of individual get() calls
    links_to_records(links)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LocationSearchInput {
    pub latitude: f64,
    pub longitude: f64,
    pub radius_km: f64,
}

/// Get title deed for a property
#[hdk_extern]
pub fn get_title_deed(property_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::TitleDeed,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(deed) = record.entry().to_app_option::<TitleDeed>().ok().flatten() {
            if deed.property_id == property_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

/// Get all title deeds for a property (history)
///
/// OPTIMIZED: Uses batch query to avoid N+1 pattern
#[hdk_extern]
pub fn get_property_deeds(property_id: String) -> ExternResult<Vec<Record>> {
    let property = get_property(property_id)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Property not found".into()
    )))?;

    let links = get_links(
        LinkQuery::try_new(
            property.action_address().clone(),
            LinkTypes::PropertyToDeeds,
        )?,
        GetStrategy::default(),
    )?;
    // FIXED N+1: Use batch fetch instead of individual get() calls
    links_to_records(links)
}

/// Update property metadata
#[hdk_extern]
pub fn update_property_metadata(input: UpdateMetadataInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Property,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(property) = record.entry().to_app_option::<Property>().ok().flatten() {
            if property.id == input.property_id {
                // Only owner can update
                if property.owner_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only owner can update metadata".into()
                    )));
                }

                let updated = Property {
                    metadata: input.metadata,
                    ..property
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Property(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Property not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateMetadataInput {
    pub property_id: String,
    pub requester_did: String,
    pub metadata: PropertyMetadata,
}

/// Remove an encumbrance (when paid off)
#[hdk_extern]
pub fn remove_encumbrance(input: RemoveEncumbranceInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::TitleDeed,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(deed) = record.entry().to_app_option::<TitleDeed>().ok().flatten() {
            if deed.property_id == input.property_id {
                let encumbrances: Vec<Encumbrance> = deed
                    .encumbrances
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != input.encumbrance_index)
                    .map(|(_, e)| e.clone())
                    .collect();

                let updated = TitleDeed {
                    encumbrances,
                    ..deed
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::TitleDeed(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Property not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RemoveEncumbranceInput {
    pub property_id: String,
    pub encumbrance_index: usize,
}

/// Get properties by type
#[hdk_extern]
pub fn get_properties_by_type(property_type: PropertyType) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Property,
        )?))
        .include_entries(true);

    let mut results = Vec::new();
    for record in query(filter)? {
        if let Some(property) = record.entry().to_app_option::<Property>().ok().flatten() {
            if property.property_type == property_type {
                results.push(record);
            }
        }
    }
    Ok(results)
}

/// Add a co-owner to property
#[hdk_extern]
pub fn add_co_owner(input: AddCoOwnerInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Property,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(property) = record.entry().to_app_option::<Property>().ok().flatten() {
            if property.id == input.property_id {
                // Only owner can add co-owners
                if property.owner_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only owner can add co-owners".into()
                    )));
                }

                // Check total shares don't exceed 100%
                let current_shares: f64 =
                    property.co_owners.iter().map(|c| c.share_percentage).sum();
                if current_shares + input.co_owner.share_percentage > 100.0 {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Total shares would exceed 100%".into()
                    )));
                }

                let mut co_owners = property.co_owners.clone();
                co_owners.push(input.co_owner);

                let updated = Property {
                    co_owners,
                    ..property
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Property(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Property not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddCoOwnerInput {
    pub property_id: String,
    pub requester_did: String,
    pub co_owner: CoOwner,
}

/// Remove a co-owner from property
#[hdk_extern]
pub fn remove_co_owner(input: RemoveCoOwnerInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Property,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(property) = record.entry().to_app_option::<Property>().ok().flatten() {
            if property.id == input.property_id {
                // Only owner can remove co-owners
                if property.owner_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only owner can remove co-owners".into()
                    )));
                }

                let co_owners: Vec<CoOwner> = property
                    .co_owners
                    .iter()
                    .filter(|c| c.did != input.co_owner_did)
                    .cloned()
                    .collect();

                let updated = Property {
                    co_owners,
                    ..property
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Property(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Property not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RemoveCoOwnerInput {
    pub property_id: String,
    pub requester_did: String,
    pub co_owner_did: String,
}

/// Get all encumbrances for a property
#[hdk_extern]
pub fn get_encumbrances(property_id: String) -> ExternResult<Vec<Encumbrance>> {
    let deed = get_title_deed(property_id)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Title deed not found".into()
    )))?;

    let deed_data = deed
        .entry()
        .to_app_option::<TitleDeed>()
        .ok()
        .flatten()
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid deed data".into()
        )))?;

    Ok(deed_data.encumbrances)
}

/// Check if property has any active encumbrances
#[hdk_extern]
pub fn has_clear_title(property_id: String) -> ExternResult<bool> {
    let encumbrances = get_encumbrances(property_id)?;
    let now = sys_time()?;

    // Check for any active (non-expired) encumbrances
    for enc in encumbrances {
        if let Some(expires) = enc.expires {
            if expires.as_micros() > now.as_micros() {
                return Ok(false);
            }
        } else {
            // No expiry means still active
            return Ok(false);
        }
    }
    Ok(true)
}

// =============================================================================
// Ownership Transfer Functions
// =============================================================================

/// Transfer ownership of a property to a new owner.
///
/// This function:
/// 1. Updates the Property entry with new owner and transfer timestamp
/// 2. Creates a new TitleDeed with the new owner
/// 3. Links the new deed to the previous deed (deed chain)
/// 4. Carries forward any encumbrances from the previous deed
/// 5. Updates owner links
#[hdk_extern]
pub fn transfer_ownership(input: TransferOwnershipInput) -> ExternResult<TransferOwnershipResult> {
    let now = sys_time()?;

    // 1. Get the current property
    let property_record = get_property(input.property_id.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Property not found".into())
    ))?;

    let property = property_record
        .entry()
        .to_app_option::<Property>()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid property data".into()
        )))?;

    // 2. Verify the current owner matches from_did
    if property.owner_did != input.from_did {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Current owner {} does not match from_did {}",
            property.owner_did, input.from_did
        ))));
    }

    // 3. Get the current title deed
    let deed_record = get_title_deed(input.property_id.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Title deed not found".into())
    ))?;

    let old_deed = deed_record
        .entry()
        .to_app_option::<TitleDeed>()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid deed data".into()
        )))?;

    // 4. Update the Property entry with new owner
    let updated_property = Property {
        owner_did: input.to_did.clone(),
        last_transfer: Some(now),
        ..property.clone()
    };

    let property_action_hash = update_entry(
        property_record.action_address().clone(),
        &EntryTypes::Property(updated_property),
    )?;

    // 5. Remove old owner link and create new owner link
    // (Links are immutable, so we just create a new one - cleanup handled by app)
    create_link(
        anchor_hash(&input.to_did)?,
        property_action_hash.clone(),
        LinkTypes::OwnerToProperties,
        (),
    )?;

    // 6. Create new TitleDeed with new owner
    let deed_type = match input.transfer_type.as_str() {
        "Sale" => DeedType::Transfer,
        "Inheritance" => DeedType::Inheritance,
        "Gift" => DeedType::Transfer, // Could add DeedType::Gift if needed
        "CourtOrder" => DeedType::CourtOrder,
        _ => DeedType::Transfer,
    };

    let new_deed = TitleDeed {
        id: format!("deed:{}:{}", input.property_id, now.as_micros()),
        property_id: input.property_id.clone(),
        owner_did: input.to_did.clone(),
        deed_type,
        issued: now,
        previous_deed_id: Some(old_deed.id.clone()),
        encumbrances: old_deed.encumbrances.clone(), // Carry forward encumbrances
    };

    let deed_action_hash = create_entry(&EntryTypes::TitleDeed(new_deed.clone()))?;

    // 7. Link new deed to property
    create_link(
        property_action_hash.clone(),
        deed_action_hash.clone(),
        LinkTypes::PropertyToDeeds,
        (),
    )?;

    Ok(TransferOwnershipResult {
        property_action_hash,
        new_deed_id: new_deed.id,
        deed_action_hash,
        previous_deed_id: old_deed.id,
        encumbrances_carried: old_deed.encumbrances.len() as u32,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TransferOwnershipInput {
    pub property_id: String,
    pub from_did: String,
    pub to_did: String,
    /// Transfer type (Sale, Inheritance, Gift, CourtOrder)
    pub transfer_type: String,
    /// Optional reference to the transfer record in transfer zome
    pub transfer_id: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TransferOwnershipResult {
    pub property_action_hash: ActionHash,
    pub new_deed_id: String,
    pub deed_action_hash: ActionHash,
    pub previous_deed_id: String,
    pub encumbrances_carried: u32,
}

/// Get ownership history for a property
#[hdk_extern]
pub fn get_ownership_history(property_id: String) -> ExternResult<Vec<OwnershipRecord>> {
    let deeds = get_property_deeds(property_id)?;

    let mut history: Vec<OwnershipRecord> = Vec::new();

    for record in deeds {
        if let Some(deed) = record.entry().to_app_option::<TitleDeed>().ok().flatten() {
            history.push(OwnershipRecord {
                deed_id: deed.id,
                owner_did: deed.owner_did,
                deed_type: deed.deed_type,
                issued: deed.issued,
                previous_deed_id: deed.previous_deed_id,
            });
        }
    }

    // Sort by issue date (oldest first)
    history.sort_by_key(|r| r.issued.as_micros() as i64);

    Ok(history)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OwnershipRecord {
    pub deed_id: String,
    pub owner_did: String,
    pub deed_type: DeedType,
    pub issued: Timestamp,
    pub previous_deed_id: Option<String>,
}

/// Verify that a DID owns a property
#[hdk_extern]
pub fn verify_ownership(input: VerifyOwnershipInput) -> ExternResult<bool> {
    let property_opt = get_property(input.property_id)?;

    if let Some(record) = property_opt {
        if let Some(property) = record.entry().to_app_option::<Property>().ok().flatten() {
            return Ok(property.owner_did == input.did);
        }
    }

    Ok(false)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyOwnershipInput {
    pub property_id: String,
    pub did: String,
}
