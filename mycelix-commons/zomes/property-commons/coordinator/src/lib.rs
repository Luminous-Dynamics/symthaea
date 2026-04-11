// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Commons Management Coordinator Zome
use hdk::prelude::*;
use mycelix_bridge_common::{civic_requirement_basic, civic_requirement_proposal};
use mycelix_zome_helpers::get_latest_record;
use property_commons_integrity::*;


/// Get or create an anchor entry and return its EntryHash for use as link base
fn anchor_hash(anchor_string: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_string.to_string());
    if let Err(e) = create_entry(&EntryTypes::Anchor(anchor.clone())) { debug!("Anchor creation warning: {:?}", e); }
    hash_entry(&anchor)
}

#[hdk_extern]
pub fn create_common_resource(input: CreateResourceInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "create_common_resource")?;
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
    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "grant_usage_right")?;
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
    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "log_usage")?;
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
    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "revoke_usage_right")?;
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
                return get_latest_record(action_hash)?
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
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "add_steward")?;
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
                return get_latest_record(action_hash)?
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
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "remove_steward")?;
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
                return get_latest_record(action_hash)?
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
    let _eligibility =
        mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "update_governance_rules")?;
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
                return get_latest_record(action_hash)?
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
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "update_right_quota")?;
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
                return get_latest_record(action_hash)?
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_resource_input_serde_roundtrip() {
        let input = CreateResourceInput {
            name: "Community Garden".to_string(),
            description: "Shared urban garden".to_string(),
            resource_type: ResourceType::Land,
            property_id: Some("prop-001".to_string()),
            stewards: vec!["did:key:z6Mk001".to_string(), "did:key:z6Mk002".to_string()],
            governance_rules: GovernanceRules {
                access_rules: vec!["Members only".to_string()],
                usage_limits: vec![UsageLimit {
                    limit_type: "water_liters".to_string(),
                    max_per_period: 500.0,
                    period_days: 7,
                }],
                maintenance_rotation: true,
                decision_method: DecisionMethod::Consensus,
                penalty_for_violation: Some("Warning".to_string()),
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateResourceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, "Community Garden");
        assert_eq!(decoded.stewards.len(), 2);
        assert!(decoded.governance_rules.maintenance_rotation);
    }

    #[test]
    fn grant_right_input_serde_roundtrip() {
        let input = GrantRightInput {
            resource_id: "commons-001".to_string(),
            holder_did: "did:key:z6Mk003".to_string(),
            right_type: RightType::Access,
            quota: Some(100.0),
            expires: Some(Timestamp::from_micros(9_000_000)),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: GrantRightInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.resource_id, "commons-001");
        assert_eq!(decoded.quota, Some(100.0));
    }

    #[test]
    fn log_usage_input_serde_roundtrip() {
        let input = LogUsageInput {
            resource_id: "commons-001".to_string(),
            user_did: "did:key:z6Mk003".to_string(),
            usage_type: "irrigation".to_string(),
            quantity: 50.0,
            unit: "liters".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: LogUsageInput = serde_json::from_str(&json).unwrap();
        assert!((decoded.quantity - 50.0).abs() < f64::EPSILON);
        assert_eq!(decoded.unit, "liters");
    }

    #[test]
    fn check_quota_input_serde_roundtrip() {
        let input = CheckQuotaInput {
            resource_id: "commons-001".to_string(),
            user_did: "did:key:z6Mk003".to_string(),
            requested_amount: 25.0,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CheckQuotaInput = serde_json::from_str(&json).unwrap();
        assert!((decoded.requested_amount - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn revoke_right_input_serde_roundtrip() {
        let input = RevokeRightInput {
            right_id: "right-001".to_string(),
            revoker_did: "did:key:z6Mk001".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RevokeRightInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.right_id, "right-001");
    }

    #[test]
    fn add_steward_input_serde_roundtrip() {
        let input = AddStewardInput {
            resource_id: "commons-001".to_string(),
            new_steward_did: "did:key:z6Mk004".to_string(),
            added_by_did: "did:key:z6Mk001".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AddStewardInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_steward_did, "did:key:z6Mk004");
    }

    #[test]
    fn remove_steward_input_serde_roundtrip() {
        let input = RemoveStewardInput {
            resource_id: "commons-001".to_string(),
            steward_did: "did:key:z6Mk002".to_string(),
            removed_by_did: "did:key:z6Mk001".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RemoveStewardInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.steward_did, "did:key:z6Mk002");
    }

    #[test]
    fn update_governance_input_serde_roundtrip() {
        let input = UpdateGovernanceInput {
            resource_id: "commons-001".to_string(),
            steward_did: "did:key:z6Mk001".to_string(),
            new_rules: GovernanceRules {
                access_rules: vec!["Open access".to_string()],
                usage_limits: vec![],
                maintenance_rotation: false,
                decision_method: DecisionMethod::Majority,
                penalty_for_violation: None,
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateGovernanceInput = serde_json::from_str(&json).unwrap();
        assert!(!decoded.new_rules.maintenance_rotation);
    }

    #[test]
    fn update_quota_input_serde_roundtrip() {
        let input = UpdateQuotaInput {
            right_id: "right-001".to_string(),
            steward_did: "did:key:z6Mk001".to_string(),
            new_quota: Some(200.0),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateQuotaInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_quota, Some(200.0));
    }

    #[test]
    fn user_usage_input_serde_roundtrip() {
        let input = UserUsageInput {
            resource_id: "commons-001".to_string(),
            user_did: "did:key:z6Mk003".to_string(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UserUsageInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.user_did, "did:key:z6Mk003");
    }

    #[test]
    fn resource_type_all_variants_serialize() {
        let types = vec![
            ResourceType::Land,
            ResourceType::Water,
            ResourceType::Forest,
            ResourceType::Fishery,
            ResourceType::Pasture,
            ResourceType::Infrastructure,
            ResourceType::Digital,
            ResourceType::Other("Custom".to_string()),
        ];
        for rt in types {
            let json = serde_json::to_string(&rt).unwrap();
            let decoded: ResourceType = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, rt);
        }
    }

    #[test]
    fn decision_method_all_variants_serialize() {
        let methods = vec![
            DecisionMethod::Consensus,
            DecisionMethod::Majority,
            DecisionMethod::SuperMajority,
            DecisionMethod::Stewards,
        ];
        for method in methods {
            let json = serde_json::to_string(&method).unwrap();
            let decoded: DecisionMethod = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, method);
        }
    }

    #[test]
    fn right_type_all_variants_serialize() {
        let types = vec![
            RightType::Access,
            RightType::Extraction,
            RightType::Management,
            RightType::Exclusion,
            RightType::Alienation,
        ];
        for rt in types {
            let json = serde_json::to_string(&rt).unwrap();
            let decoded: RightType = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, rt);
        }
    }

    #[test]
    fn create_resource_without_optional_fields() {
        let input = CreateResourceInput {
            name: "Digital Commons".to_string(),
            description: "Shared knowledge base".to_string(),
            resource_type: ResourceType::Digital,
            property_id: None,
            stewards: vec!["did:key:z6Mk001".to_string()],
            governance_rules: GovernanceRules {
                access_rules: vec![],
                usage_limits: vec![],
                maintenance_rotation: false,
                decision_method: DecisionMethod::Stewards,
                penalty_for_violation: None,
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateResourceInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.property_id.is_none());
    }

    #[test]
    fn grant_right_without_quota_or_expiry() {
        let input = GrantRightInput {
            resource_id: "commons-001".to_string(),
            holder_did: "did:key:z6Mk003".to_string(),
            right_type: RightType::Access,
            quota: None,
            expires: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: GrantRightInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.quota.is_none());
        assert!(decoded.expires.is_none());
    }

    // ── Governance rule serde with all decision methods ──────────────

    #[test]
    fn governance_rules_consensus_serde_roundtrip() {
        let rules = GovernanceRules {
            access_rules: vec![
                "Open to all members".to_string(),
                "Must register".to_string(),
            ],
            usage_limits: vec![
                UsageLimit {
                    limit_type: "water_liters".to_string(),
                    max_per_period: 500.0,
                    period_days: 7,
                },
                UsageLimit {
                    limit_type: "harvest_kg".to_string(),
                    max_per_period: 20.0,
                    period_days: 30,
                },
            ],
            maintenance_rotation: true,
            decision_method: DecisionMethod::Consensus,
            penalty_for_violation: Some("Written warning, then suspension".to_string()),
        };
        let json = serde_json::to_string(&rules).unwrap();
        let decoded: GovernanceRules = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.access_rules.len(), 2);
        assert_eq!(decoded.usage_limits.len(), 2);
        assert!(decoded.maintenance_rotation);
        assert_eq!(decoded.decision_method, DecisionMethod::Consensus);
        assert!(decoded
            .penalty_for_violation
            .as_ref()
            .unwrap()
            .contains("warning"));
    }

    #[test]
    fn governance_rules_supermajority_no_limits_serde() {
        let rules = GovernanceRules {
            access_rules: vec![],
            usage_limits: vec![],
            maintenance_rotation: false,
            decision_method: DecisionMethod::SuperMajority,
            penalty_for_violation: None,
        };
        let json = serde_json::to_string(&rules).unwrap();
        let decoded: GovernanceRules = serde_json::from_str(&json).unwrap();
        assert!(decoded.access_rules.is_empty());
        assert!(decoded.usage_limits.is_empty());
        assert!(!decoded.maintenance_rotation);
        assert_eq!(decoded.decision_method, DecisionMethod::SuperMajority);
        assert!(decoded.penalty_for_violation.is_none());
    }

    // ── Usage limit boundary conditions ─────────────────────────────

    #[test]
    fn usage_limit_zero_max_serde_roundtrip() {
        let limit = UsageLimit {
            limit_type: "extraction_liters".to_string(),
            max_per_period: 0.0,
            period_days: 1,
        };
        let json = serde_json::to_string(&limit).unwrap();
        let decoded: UsageLimit = serde_json::from_str(&json).unwrap();
        assert!((decoded.max_per_period - 0.0).abs() < f64::EPSILON);
        assert_eq!(decoded.period_days, 1);
    }

    #[test]
    fn usage_limit_very_large_values_serde() {
        let limit = UsageLimit {
            limit_type: "megawatt_hours".to_string(),
            max_per_period: f64::MAX,
            period_days: u32::MAX,
        };
        let json = serde_json::to_string(&limit).unwrap();
        let decoded: UsageLimit = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.max_per_period, f64::MAX);
        assert_eq!(decoded.period_days, u32::MAX);
    }

    // ── Integrity entry struct serde roundtrips from coordinator ─────

    #[test]
    fn common_resource_entry_serde_roundtrip() {
        let resource = CommonResource {
            id: "commons:test:123".to_string(),
            name: "Village Forest".to_string(),
            description: "Ancient managed woodland".to_string(),
            resource_type: ResourceType::Forest,
            property_id: Some("prop-forest-01".to_string()),
            stewards: vec!["did:key:z6Mk001".to_string()],
            governance_rules: GovernanceRules {
                access_rules: vec!["Daylight hours only".to_string()],
                usage_limits: vec![],
                maintenance_rotation: false,
                decision_method: DecisionMethod::Stewards,
                penalty_for_violation: None,
            },
            created: Timestamp::from_micros(1_000_000),
        };
        let json = serde_json::to_string(&resource).unwrap();
        let decoded: CommonResource = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, resource.id);
        assert_eq!(decoded.name, "Village Forest");
        assert_eq!(decoded.resource_type, ResourceType::Forest);
        assert_eq!(decoded.stewards.len(), 1);
    }

    #[test]
    fn usage_right_entry_serde_roundtrip() {
        let right = UsageRight {
            id: "right:res1:holder1:999".to_string(),
            resource_id: "res1".to_string(),
            holder_did: "did:key:z6MkHolder".to_string(),
            right_type: RightType::Management,
            quota: Some(999.99),
            granted: Timestamp::from_micros(100),
            expires: Some(Timestamp::from_micros(999_999)),
            active: true,
        };
        let json = serde_json::to_string(&right).unwrap();
        let decoded: UsageRight = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, right.id);
        assert_eq!(decoded.right_type, RightType::Management);
        assert!(decoded.active);
        assert!((decoded.quota.unwrap() - 999.99).abs() < f64::EPSILON);
    }

    #[test]
    fn usage_log_entry_serde_roundtrip() {
        let log = UsageLog {
            id: "usage:res1:user1:500".to_string(),
            resource_id: "res1".to_string(),
            user_did: "did:web:example.org".to_string(),
            usage_type: "grazing".to_string(),
            quantity: 12.5,
            unit: "hectare_hours".to_string(),
            timestamp: Timestamp::from_micros(500_000),
        };
        let json = serde_json::to_string(&log).unwrap();
        let decoded: UsageLog = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, log.id);
        assert_eq!(decoded.user_did, "did:web:example.org");
        assert!((decoded.quantity - 12.5).abs() < f64::EPSILON);
        assert_eq!(decoded.unit, "hectare_hours");
    }

    // ── Edge cases: quota boundary values ───────────────────────────

    #[test]
    fn check_quota_input_zero_requested_amount() {
        let input = CheckQuotaInput {
            resource_id: "commons-001".to_string(),
            user_did: "did:key:z6Mk003".to_string(),
            requested_amount: 0.0,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CheckQuotaInput = serde_json::from_str(&json).unwrap();
        assert!((decoded.requested_amount - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn update_quota_input_none_removes_limit() {
        let input = UpdateQuotaInput {
            right_id: "right-001".to_string(),
            steward_did: "did:key:z6Mk001".to_string(),
            new_quota: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateQuotaInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.new_quota.is_none());
    }

    // ── Resource type Other with edge-case strings ──────────────────

    #[test]
    fn resource_type_other_empty_string_serde() {
        let rt = ResourceType::Other("".to_string());
        let json = serde_json::to_string(&rt).unwrap();
        let decoded: ResourceType = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, rt);
    }

    #[test]
    fn resource_type_other_unicode_string_serde() {
        let rt = ResourceType::Other("\u{6C34}\u{8CC7}\u{6E90}".to_string()); // Chinese: water resource
        let json = serde_json::to_string(&rt).unwrap();
        let decoded: ResourceType = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, rt);
    }

    // ── Combined input with many stewards ───────────────────────────

    #[test]
    fn create_resource_input_many_stewards_serde() {
        let stewards: Vec<String> = (0..50).map(|i| format!("did:key:z6Mk{:04}", i)).collect();
        let input = CreateResourceInput {
            name: "Large Commons".to_string(),
            description: "A resource with 50 stewards".to_string(),
            resource_type: ResourceType::Pasture,
            property_id: None,
            stewards: stewards.clone(),
            governance_rules: GovernanceRules {
                access_rules: vec![],
                usage_limits: vec![],
                maintenance_rotation: true,
                decision_method: DecisionMethod::Majority,
                penalty_for_violation: None,
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateResourceInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.stewards.len(), 50);
        assert_eq!(decoded.stewards[0], "did:key:z6Mk0000");
        assert_eq!(decoded.stewards[49], "did:key:z6Mk0049");
    }

    // ── Grant right with all right type variants ────────────────────

    #[test]
    fn grant_right_all_types_serde() {
        let right_types = vec![
            RightType::Access,
            RightType::Extraction,
            RightType::Management,
            RightType::Exclusion,
            RightType::Alienation,
        ];
        for rt in right_types {
            let input = GrantRightInput {
                resource_id: "commons-001".to_string(),
                holder_did: "did:key:z6MkHolder".to_string(),
                right_type: rt.clone(),
                quota: Some(1.0),
                expires: None,
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: GrantRightInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.right_type, rt);
        }
    }
}
