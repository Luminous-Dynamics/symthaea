// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Commons Management Coordinator Zome
use commons_integrity::*;
use hdk::prelude::*;

/// Get or create an anchor entry and return its EntryHash for use as link base
fn anchor_hash(anchor_string: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_string.to_string());
    let _ = create_entry(&EntryTypes::Anchor(anchor.clone()));
    hash_entry(&anchor)
}

#[hdk_extern]
pub fn create_common_resource(input: CreateResourceInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let resource = CommonResource {
        id: format!(
            "commons:{}:{}",
            input.name.replace(' ', "_"),
            now.as_micros()
        ),
        name: input.name,
        description: input.description,
        resource_type: input.resource_type,
        property_id: input.property_id,
        stewards: input.stewards.clone(),
        governance_rules: input.governance_rules,
        created: now,
    };

    let action_hash = create_entry(&EntryTypes::CommonResource(resource))?;
    for steward in input.stewards {
        create_link(
            anchor_hash(&steward)?,
            action_hash.clone(),
            LinkTypes::StewardToResource,
            (),
        )?;
    }
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateResourceInput {
    pub name: String,
    pub description: String,
    pub resource_type: ResourceType,
    pub property_id: Option<String>,
    pub stewards: Vec<String>,
    pub governance_rules: GovernanceRules,
}

#[hdk_extern]
pub fn grant_usage_right(input: GrantRightInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let right = UsageRight {
        id: format!(
            "right:{}:{}:{}",
            input.resource_id,
            input.holder_did,
            now.as_micros()
        ),
        resource_id: input.resource_id.clone(),
        holder_did: input.holder_did.clone(),
        right_type: input.right_type,
        quota: input.quota,
        granted: now,
        expires: input.expires,
        active: true,
    };

    let action_hash = create_entry(&EntryTypes::UsageRight(right))?;
    create_link(
        anchor_hash(&input.resource_id)?,
        action_hash.clone(),
        LinkTypes::ResourceToRights,
        (),
    )?;
    create_link(
        anchor_hash(&input.holder_did)?,
        action_hash.clone(),
        LinkTypes::HolderToRights,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GrantRightInput {
    pub resource_id: String,
    pub holder_did: String,
    pub right_type: RightType,
    pub quota: Option<f64>,
    pub expires: Option<Timestamp>,
}

#[hdk_extern]
pub fn log_usage(input: LogUsageInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let log = UsageLog {
        id: format!(
            "usage:{}:{}:{}",
            input.resource_id,
            input.user_did,
            now.as_micros()
        ),
        resource_id: input.resource_id.clone(),
        user_did: input.user_did,
        usage_type: input.usage_type,
        quantity: input.quantity,
        unit: input.unit,
        timestamp: now,
    };

    let action_hash = create_entry(&EntryTypes::UsageLog(log))?;
    create_link(
        anchor_hash(&input.resource_id)?,
        action_hash.clone(),
        LinkTypes::ResourceToLogs,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LogUsageInput {
    pub resource_id: String,
    pub user_did: String,
    pub usage_type: String,
    pub quantity: f64,
    pub unit: String,
}

#[hdk_extern]
pub fn get_resource_usage(resource_id: String) -> ExternResult<Vec<Record>> {
    let mut logs = Vec::new();
    for link in get_links(
        LinkQuery::try_new(anchor_hash(&resource_id)?, LinkTypes::ResourceToLogs)?,
        GetStrategy::default(),
    )? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            logs.push(record);
        }
    }
    Ok(logs)
}

#[hdk_extern]
pub fn check_usage_quota(input: CheckQuotaInput) -> ExternResult<bool> {
    // Get user's rights for this resource
    let rights_links = get_links(
        LinkQuery::try_new(anchor_hash(&input.user_did)?, LinkTypes::HolderToRights)?,
        GetStrategy::default(),
    )?;

    for link in rights_links {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            if let Some(right) = record.entry().to_app_option::<UsageRight>().ok().flatten() {
                if right.resource_id == input.resource_id && right.active {
                    if let Some(quota) = right.quota {
                        // Sum current usage (simplified - would need time-based filtering)
                        let usage_logs = get_resource_usage(input.resource_id.clone())?;
                        let mut total_usage = 0.0;
                        for log_record in usage_logs {
                            if let Some(log) = log_record
                                .entry()
                                .to_app_option::<UsageLog>()
                                .ok()
                                .flatten()
                            {
                                if log.user_did == input.user_did {
                                    total_usage += log.quantity;
                                }
                            }
                        }
                        return Ok(total_usage + input.requested_amount <= quota);
                    }
                    return Ok(true); // No quota limit
                }
            }
        }
    }
    Ok(false) // No valid right found
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CheckQuotaInput {
    pub resource_id: String,
    pub user_did: String,
    pub requested_amount: f64,
}

/// Get a specific common resource by ID
#[hdk_extern]
pub fn get_resource(resource_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::CommonResource,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(resource) = record
            .entry()
            .to_app_option::<CommonResource>()
            .ok()
            .flatten()
        {
            if resource.id == resource_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

/// Get all resources stewarded by a DID
#[hdk_extern]
pub fn get_steward_resources(steward_did: String) -> ExternResult<Vec<Record>> {
    let mut resources = Vec::new();
    for link in get_links(
        LinkQuery::try_new(anchor_hash(&steward_did)?, LinkTypes::StewardToResource)?,
        GetStrategy::default(),
    )? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            resources.push(record);
        }
    }
    Ok(resources)
}

/// Get all usage rights for a resource
#[hdk_extern]
pub fn get_resource_rights(resource_id: String) -> ExternResult<Vec<Record>> {
    let mut rights = Vec::new();
    for link in get_links(
        LinkQuery::try_new(anchor_hash(&resource_id)?, LinkTypes::ResourceToRights)?,
        GetStrategy::default(),
    )? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            rights.push(record);
        }
    }
    Ok(rights)
}

/// Revoke a usage right
#[hdk_extern]
pub fn revoke_usage_right(input: RevokeRightInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::UsageRight,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(right) = record.entry().to_app_option::<UsageRight>().ok().flatten() {
            if right.id == input.right_id {
                // Verify revoker is a steward
                let resource = get_resource(right.resource_id.clone())?.ok_or(wasm_error!(
                    WasmErrorInner::Guest("Resource not found".into())
                ))?;
                let resource_data = resource
                    .entry()
                    .to_app_option::<CommonResource>()
                    .ok()
                    .flatten()
                    .ok_or(wasm_error!(WasmErrorInner::Guest(
                        "Invalid resource data".into()
                    )))?;

                if !resource_data.stewards.contains(&input.revoker_did) {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only stewards can revoke rights".into()
                    )));
                }

                let revoked = UsageRight {
                    active: false,
                    ..right
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::UsageRight(revoked),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Right not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RevokeRightInput {
    pub right_id: String,
    pub revoker_did: String,
}

/// Add a steward to a resource
#[hdk_extern]
pub fn add_steward(input: AddStewardInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::CommonResource,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(resource) = record
            .entry()
            .to_app_option::<CommonResource>()
            .ok()
            .flatten()
        {
            if resource.id == input.resource_id {
                // Verify caller is a steward
                if !resource.stewards.contains(&input.added_by_did) {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only stewards can add new stewards".into()
                    )));
                }

                if resource.stewards.contains(&input.new_steward_did) {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Already a steward".into()
                    )));
                }

                let mut stewards = resource.stewards.clone();
                stewards.push(input.new_steward_did.clone());

                let updated = CommonResource {
                    stewards,
                    ..resource
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::CommonResource(updated),
                )?;
                create_link(
                    anchor_hash(&input.new_steward_did)?,
                    action_hash.clone(),
                    LinkTypes::StewardToResource,
                    (),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Resource not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AddStewardInput {
    pub resource_id: String,
    pub new_steward_did: String,
    pub added_by_did: String,
}

/// Remove a steward from a resource (cannot remove last steward)
#[hdk_extern]
pub fn remove_steward(input: RemoveStewardInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::CommonResource,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(resource) = record
            .entry()
            .to_app_option::<CommonResource>()
            .ok()
            .flatten()
        {
            if resource.id == input.resource_id {
                // Verify caller is a steward
                if !resource.stewards.contains(&input.removed_by_did) {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only stewards can remove stewards".into()
                    )));
                }

                if resource.stewards.len() <= 1 {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Cannot remove last steward".into()
                    )));
                }

                if !resource.stewards.contains(&input.steward_did) {
                    return Err(wasm_error!(WasmErrorInner::Guest("Not a steward".into())));
                }

                let stewards: Vec<String> = resource
                    .stewards
                    .iter()
                    .filter(|s| *s != &input.steward_did)
                    .cloned()
                    .collect();

                let updated = CommonResource {
                    stewards,
                    ..resource
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::CommonResource(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Resource not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RemoveStewardInput {
    pub resource_id: String,
    pub steward_did: String,
    pub removed_by_did: String,
}

/// Get all rights held by a DID
#[hdk_extern]
pub fn get_holder_rights(holder_did: String) -> ExternResult<Vec<Record>> {
    let mut rights = Vec::new();
    for link in get_links(
        LinkQuery::try_new(anchor_hash(&holder_did)?, LinkTypes::HolderToRights)?,
        GetStrategy::default(),
    )? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            rights.push(record);
        }
    }
    Ok(rights)
}

/// Update governance rules for a resource
#[hdk_extern]
pub fn update_governance_rules(input: UpdateGovernanceInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::CommonResource,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(resource) = record
            .entry()
            .to_app_option::<CommonResource>()
            .ok()
            .flatten()
        {
            if resource.id == input.resource_id {
                // Verify caller is a steward
                if !resource.stewards.contains(&input.steward_did) {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only stewards can update governance".into()
                    )));
                }

                let updated = CommonResource {
                    governance_rules: input.new_rules,
                    ..resource
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::CommonResource(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Resource not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateGovernanceInput {
    pub resource_id: String,
    pub steward_did: String,
    pub new_rules: GovernanceRules,
}

/// Get resources by type
#[hdk_extern]
pub fn get_resources_by_type(resource_type: ResourceType) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::CommonResource,
        )?))
        .include_entries(true);

    let mut results = Vec::new();
    for record in query(filter)? {
        if let Some(resource) = record
            .entry()
            .to_app_option::<CommonResource>()
            .ok()
            .flatten()
        {
            if resource.resource_type == resource_type {
                results.push(record);
            }
        }
    }
    Ok(results)
}

/// Get a specific usage right by ID
#[hdk_extern]
pub fn get_usage_right(right_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::UsageRight,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(right) = record.entry().to_app_option::<UsageRight>().ok().flatten() {
            if right.id == right_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

/// Get total usage for a user in a resource
#[hdk_extern]
pub fn get_user_usage(input: UserUsageInput) -> ExternResult<f64> {
    let logs = get_resource_usage(input.resource_id)?;
    let mut total = 0.0;
    for log_record in logs {
        if let Some(log) = log_record
            .entry()
            .to_app_option::<UsageLog>()
            .ok()
            .flatten()
        {
            if log.user_did == input.user_did {
                total += log.quantity;
            }
        }
    }
    Ok(total)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UserUsageInput {
    pub resource_id: String,
    pub user_did: String,
}

/// Update usage right quota
#[hdk_extern]
pub fn update_right_quota(input: UpdateQuotaInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::UsageRight,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(right) = record.entry().to_app_option::<UsageRight>().ok().flatten() {
            if right.id == input.right_id {
                // Verify updater is a steward of the resource
                let resource = get_resource(right.resource_id.clone())?.ok_or(wasm_error!(
                    WasmErrorInner::Guest("Resource not found".into())
                ))?;
                let resource_data = resource
                    .entry()
                    .to_app_option::<CommonResource>()
                    .ok()
                    .flatten()
                    .ok_or(wasm_error!(WasmErrorInner::Guest(
                        "Invalid resource data".into()
                    )))?;

                if !resource_data.stewards.contains(&input.steward_did) {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only stewards can update quotas".into()
                    )));
                }

                let updated = UsageRight {
                    quota: input.new_quota,
                    ..right
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::UsageRight(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Right not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateQuotaInput {
    pub right_id: String,
    pub steward_did: String,
    pub new_quota: Option<f64>,
}
