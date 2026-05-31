// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Commons Management Integrity Zome
use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CommonResource {
    pub id: String,
    pub name: String,
    pub description: String,
    pub resource_type: ResourceType,
    pub property_id: Option<String>,
    pub stewards: Vec<String>,
    pub governance_rules: GovernanceRules,
    pub created: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ResourceType {
    Land,
    Water,
    Forest,
    Fishery,
    Pasture,
    Infrastructure,
    Digital,
    Other(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct GovernanceRules {
    pub access_rules: Vec<String>,
    pub usage_limits: Vec<UsageLimit>,
    pub maintenance_rotation: bool,
    pub decision_method: DecisionMethod,
    pub penalty_for_violation: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct UsageLimit {
    pub limit_type: String,
    pub max_per_period: f64,
    pub period_days: u32,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum DecisionMethod {
    Consensus,
    Majority,
    SuperMajority,
    Stewards,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct UsageRight {
    pub id: String,
    pub resource_id: String,
    pub holder_did: String,
    pub right_type: RightType,
    pub quota: Option<f64>,
    pub granted: Timestamp,
    pub expires: Option<Timestamp>,
    pub active: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RightType {
    Access,
    Extraction,
    Management,
    Exclusion,
    Alienation,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct UsageLog {
    pub id: String,
    pub resource_id: String,
    pub user_did: String,
    pub usage_type: String,
    pub quantity: f64,
    pub unit: String,
    pub timestamp: Timestamp,
}

/// Anchor entry for deterministic link bases from strings
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    CommonResource(CommonResource),
    UsageRight(UsageRight),
    UsageLog(UsageLog),
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
}

#[hdk_link_types]
pub enum LinkTypes {
    StewardToResource,
    ResourceToRights,
    HolderToRights,
    ResourceToLogs,
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
                EntryTypes::CommonResource(resource) => {
                    validate_create_common_resource(EntryCreationAction::Create(action), resource)
                }
                EntryTypes::UsageRight(right) => {
                    validate_create_usage_right(EntryCreationAction::Create(action), right)
                }
                EntryTypes::UsageLog(log) => {
                    validate_create_usage_log(EntryCreationAction::Create(action), log)
                }
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::CommonResource(resource) => {
                    validate_update_common_resource(action, resource)
                }
                EntryTypes::UsageRight(right) => validate_update_usage_right(action, right),
                EntryTypes::UsageLog(_) => Ok(ValidateCallbackResult::Invalid(
                    "Usage logs cannot be updated".into(),
                )),
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::StewardToResource => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ResourceToRights => Ok(ValidateCallbackResult::Valid),
            LinkTypes::HolderToRights => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ResourceToLogs => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_common_resource(
    _action: EntryCreationAction,
    resource: CommonResource,
) -> ExternResult<ValidateCallbackResult> {
    if resource.stewards.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Resource must have at least one steward".into(),
        ));
    }
    for steward in &resource.stewards {
        if !steward.starts_with("did:") {
            return Ok(ValidateCallbackResult::Invalid(
                "Stewards must be valid DIDs".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_common_resource(
    _action: Update,
    _resource: CommonResource,
) -> ExternResult<ValidateCallbackResult> {
    // Governance rules and stewards can be updated
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_usage_right(
    _action: EntryCreationAction,
    right: UsageRight,
) -> ExternResult<ValidateCallbackResult> {
    if !right.holder_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Holder must be a valid DID".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_usage_right(
    _action: Update,
    _right: UsageRight,
) -> ExternResult<ValidateCallbackResult> {
    // Status and quota can be updated
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_usage_log(
    _action: EntryCreationAction,
    log: UsageLog,
) -> ExternResult<ValidateCallbackResult> {
    if !log.user_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "User must be a valid DID".into(),
        ));
    }
    if log.quantity < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Quantity cannot be negative".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
