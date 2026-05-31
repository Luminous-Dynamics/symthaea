// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Attribution Coordinator Zome
use attribution_integrity::*;
use hdk::prelude::*;

/// Helper function to create an anchor entry and return its hash
fn anchor_hash(anchor_string: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_string.to_string());
    let _ = create_entry(&EntryTypes::Anchor(anchor.clone()));
    hash_entry(&anchor)
}

#[hdk_extern]
pub fn add_attribution(input: AddAttributionInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let attribution = Attribution {
        id: format!(
            "attr:{}:{}:{}",
            input.publication_id,
            input.contributor_did,
            now.as_micros()
        ),
        publication_id: input.publication_id.clone(),
        contributor_did: input.contributor_did.clone(),
        role: input.role,
        share_percentage: input.share_percentage,
        verified: false,
        created: now,
    };

    let action_hash = create_entry(&EntryTypes::Attribution(attribution))?;
    create_link(
        anchor_hash(&input.publication_id)?,
        action_hash.clone(),
        LinkTypes::PublicationToAttributions,
        (),
    )?;
    create_link(
        anchor_hash(&input.contributor_did)?,
        action_hash.clone(),
        LinkTypes::ContributorToAttributions,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddAttributionInput {
    pub publication_id: String,
    pub contributor_did: String,
    pub role: ContributorRole,
    pub share_percentage: f64,
}

#[hdk_extern]
pub fn set_royalty_rule(input: SetRoyaltyInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let rule = RoyaltyRule {
        id: format!(
            "royalty:{}:{:?}:{}",
            input.publication_id,
            input.rule_type,
            now.as_micros()
        ),
        publication_id: input.publication_id.clone(),
        rule_type: input.rule_type,
        percentage: input.percentage,
        minimum_amount: input.minimum_amount,
        currency: input.currency,
        active: true,
    };

    let action_hash = create_entry(&EntryTypes::RoyaltyRule(rule))?;
    create_link(
        anchor_hash(&input.publication_id)?,
        action_hash.clone(),
        LinkTypes::PublicationToRoyalties,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SetRoyaltyInput {
    pub publication_id: String,
    pub rule_type: RoyaltyType,
    pub percentage: f64,
    pub minimum_amount: Option<f64>,
    pub currency: String,
}

#[hdk_extern]
pub fn record_usage(input: RecordUsageInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let usage = UsageRecord {
        id: format!("usage:{}:{}", input.publication_id, now.as_micros()),
        publication_id: input.publication_id.clone(),
        usage_type: input.usage_type,
        user_did: input.user_did,
        timestamp: now,
        royalty_paid: input.royalty_paid,
    };

    let action_hash = create_entry(&EntryTypes::UsageRecord(usage))?;
    create_link(
        anchor_hash(&input.publication_id)?,
        action_hash.clone(),
        LinkTypes::PublicationToUsage,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordUsageInput {
    pub publication_id: String,
    pub usage_type: UsageType,
    pub user_did: Option<String>,
    pub royalty_paid: Option<f64>,
}

#[hdk_extern]
pub fn get_publication_attributions(publication_id: String) -> ExternResult<Vec<Record>> {
    let mut attributions = Vec::new();
    let query = LinkQuery::new(
        anchor_hash(&publication_id)?,
        LinkTypeFilter::single_type(
            0.into(),
            (LinkTypes::PublicationToAttributions as u8).into(),
        ),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            attributions.push(record);
        }
    }
    Ok(attributions)
}

#[hdk_extern]
pub fn get_contributor_works(did: String) -> ExternResult<Vec<Record>> {
    let mut works = Vec::new();
    let query = LinkQuery::new(
        anchor_hash(&did)?,
        LinkTypeFilter::single_type(
            0.into(),
            (LinkTypes::ContributorToAttributions as u8).into(),
        ),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            works.push(record);
        }
    }
    Ok(works)
}

/// Get a specific attribution by ID
#[hdk_extern]
pub fn get_attribution(attribution_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Attribution,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(attr) = record.entry().to_app_option::<Attribution>().ok().flatten() {
            if attr.id == attribution_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

/// Verify an attribution
#[hdk_extern]
pub fn verify_attribution(input: VerifyAttributionInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Attribution,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(attr) = record.entry().to_app_option::<Attribution>().ok().flatten() {
            if attr.id == input.attribution_id {
                // Only contributor can verify
                if attr.contributor_did != input.requester_did {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only contributor can verify attribution".into()
                    )));
                }

                let verified = Attribution {
                    verified: true,
                    ..attr
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Attribution(verified),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Attribution not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyAttributionInput {
    pub attribution_id: String,
    pub requester_did: String,
}

/// Update share percentage
#[hdk_extern]
pub fn update_share_percentage(input: UpdateShareInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Attribution,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(attr) = record.entry().to_app_option::<Attribution>().ok().flatten() {
            if attr.id == input.attribution_id {
                let updated = Attribution {
                    share_percentage: input.new_share_percentage,
                    ..attr
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Attribution(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Attribution not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateShareInput {
    pub attribution_id: String,
    pub new_share_percentage: f64,
}

/// Get publication royalty rules
#[hdk_extern]
pub fn get_publication_royalties(publication_id: String) -> ExternResult<Vec<Record>> {
    let mut royalties = Vec::new();
    let query = LinkQuery::new(
        anchor_hash(&publication_id)?,
        LinkTypeFilter::single_type(0.into(), (LinkTypes::PublicationToRoyalties as u8).into()),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            royalties.push(record);
        }
    }
    Ok(royalties)
}

/// Get publication usage records
#[hdk_extern]
pub fn get_publication_usage(publication_id: String) -> ExternResult<Vec<Record>> {
    let mut usage = Vec::new();
    let query = LinkQuery::new(
        anchor_hash(&publication_id)?,
        LinkTypeFilter::single_type(0.into(), (LinkTypes::PublicationToUsage as u8).into()),
    );
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            usage.push(record);
        }
    }
    Ok(usage)
}

/// Deactivate a royalty rule
#[hdk_extern]
pub fn deactivate_royalty_rule(input: DeactivateRoyaltyInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::RoyaltyRule,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(rule) = record.entry().to_app_option::<RoyaltyRule>().ok().flatten() {
            if rule.id == input.rule_id {
                let updated = RoyaltyRule {
                    active: false,
                    ..rule
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::RoyaltyRule(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Royalty rule not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DeactivateRoyaltyInput {
    pub rule_id: String,
}

/// Get contributor earnings summary
#[hdk_extern]
pub fn get_contributor_earnings(did: String) -> ExternResult<ContributorEarnings> {
    let attribution_filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Attribution,
        )?))
        .include_entries(true);

    let mut publication_ids = Vec::new();
    let mut total_share = 0.0;

    for record in query(attribution_filter)? {
        if let Some(attr) = record.entry().to_app_option::<Attribution>().ok().flatten() {
            if attr.contributor_did == did {
                publication_ids.push(attr.publication_id);
                total_share += attr.share_percentage;
            }
        }
    }

    // Calculate total royalties earned
    let usage_filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::UsageRecord,
        )?))
        .include_entries(true);

    let mut total_royalties = 0.0;
    for record in query(usage_filter)? {
        if let Some(usage) = record.entry().to_app_option::<UsageRecord>().ok().flatten() {
            if publication_ids.contains(&usage.publication_id) {
                if let Some(paid) = usage.royalty_paid {
                    total_royalties += paid;
                }
            }
        }
    }

    Ok(ContributorEarnings {
        contributor_did: did,
        publication_count: publication_ids.len() as u32,
        average_share_percentage: if publication_ids.is_empty() {
            0.0
        } else {
            total_share / publication_ids.len() as f64
        },
        total_royalties_earned: total_royalties,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ContributorEarnings {
    pub contributor_did: String,
    pub publication_count: u32,
    pub average_share_percentage: f64,
    pub total_royalties_earned: f64,
}

/// Get attributions by role
#[hdk_extern]
pub fn get_attributions_by_role(role: ContributorRole) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Attribution,
        )?))
        .include_entries(true);

    let mut results = Vec::new();
    for record in query(filter)? {
        if let Some(attr) = record.entry().to_app_option::<Attribution>().ok().flatten() {
            if attr.role == role {
                results.push(record);
            }
        }
    }
    Ok(results)
}

/// Calculate total shares for a publication
#[hdk_extern]
pub fn get_publication_shares(publication_id: String) -> ExternResult<PublicationShares> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Attribution,
        )?))
        .include_entries(true);

    let mut total_percentage = 0.0;
    let mut contributor_count = 0;
    let mut verified_count = 0;

    for record in query(filter)? {
        if let Some(attr) = record.entry().to_app_option::<Attribution>().ok().flatten() {
            if attr.publication_id == publication_id {
                total_percentage += attr.share_percentage;
                contributor_count += 1;
                if attr.verified {
                    verified_count += 1;
                }
            }
        }
    }

    Ok(PublicationShares {
        publication_id,
        total_share_percentage: total_percentage,
        contributor_count,
        verified_count,
        remaining_percentage: 100.0 - total_percentage,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PublicationShares {
    pub publication_id: String,
    pub total_share_percentage: f64,
    pub contributor_count: u32,
    pub verified_count: u32,
    pub remaining_percentage: f64,
}

/// Remove an attribution (only by original author/creator)
#[hdk_extern]
pub fn remove_attribution(input: RemoveAttributionInput) -> ExternResult<()> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Attribution,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(attr) = record.entry().to_app_option::<Attribution>().ok().flatten() {
            if attr.id == input.attribution_id {
                delete_entry(record.action_address().clone())?;
                return Ok(());
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Attribution not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RemoveAttributionInput {
    pub attribution_id: String,
    pub requester_did: String,
}
