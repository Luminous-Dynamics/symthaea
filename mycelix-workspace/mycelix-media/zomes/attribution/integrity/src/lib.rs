// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Attribution Integrity Zome
//! Updated to use HDI 0.7 patterns with FlatOp validation
use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Attribution {
    pub id: String,
    pub publication_id: String,
    pub contributor_did: String,
    pub role: ContributorRole,
    pub share_percentage: f64,
    pub verified: bool,
    pub created: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ContributorRole {
    Author,
    CoAuthor,
    Editor,
    Researcher,
    Photographer,
    Illustrator,
    Translator,
    Source,
    Other(String),
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RoyaltyRule {
    pub id: String,
    pub publication_id: String,
    pub rule_type: RoyaltyType,
    pub percentage: f64,
    pub minimum_amount: Option<f64>,
    pub currency: String,
    pub active: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RoyaltyType {
    PerView,
    PerShare,
    PerDownload,
    PerDerivative,
    Subscription,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct UsageRecord {
    pub id: String,
    pub publication_id: String,
    pub usage_type: UsageType,
    pub user_did: Option<String>,
    pub timestamp: Timestamp,
    pub royalty_paid: Option<f64>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum UsageType {
    View,
    Share,
    Download,
    Derivative,
    Citation,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
    Attribution(Attribution),
    RoyaltyRule(RoyaltyRule),
    UsageRecord(UsageRecord),
}

#[hdk_link_types]
pub enum LinkTypes {
    PublicationToAttributions,
    ContributorToAttributions,
    PublicationToRoyalties,
    PublicationToUsage,
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
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Attribution(attribution) => {
                    validate_create_attribution(EntryCreationAction::Create(action), attribution)
                }
                EntryTypes::RoyaltyRule(rule) => {
                    validate_create_royalty_rule(EntryCreationAction::Create(action), rule)
                }
                EntryTypes::UsageRecord(record) => {
                    validate_create_usage_record(EntryCreationAction::Create(action), record)
                }
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Attribution(attribution) => {
                    validate_update_attribution(action, attribution)
                }
                EntryTypes::RoyaltyRule(rule) => validate_update_royalty_rule(action, rule),
                EntryTypes::UsageRecord(_) => Ok(ValidateCallbackResult::Invalid(
                    "Usage records cannot be updated".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::PublicationToAttributions => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ContributorToAttributions => Ok(ValidateCallbackResult::Valid),
            LinkTypes::PublicationToRoyalties => Ok(ValidateCallbackResult::Valid),
            LinkTypes::PublicationToUsage => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_attribution(
    _action: EntryCreationAction,
    attribution: Attribution,
) -> ExternResult<ValidateCallbackResult> {
    if !attribution.contributor_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Contributor must be a valid DID".into(),
        ));
    }
    if attribution.share_percentage < 0.0 || attribution.share_percentage > 100.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Share must be 0-100%".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_attribution(
    _action: Update,
    attribution: Attribution,
) -> ExternResult<ValidateCallbackResult> {
    if attribution.share_percentage < 0.0 || attribution.share_percentage > 100.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Share must be 0-100%".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_royalty_rule(
    _action: EntryCreationAction,
    rule: RoyaltyRule,
) -> ExternResult<ValidateCallbackResult> {
    if rule.percentage < 0.0 || rule.percentage > 100.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Percentage must be 0-100".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_royalty_rule(
    _action: Update,
    rule: RoyaltyRule,
) -> ExternResult<ValidateCallbackResult> {
    if rule.percentage < 0.0 || rule.percentage > 100.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Percentage must be 0-100".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_usage_record(
    _action: EntryCreationAction,
    _record: UsageRecord,
) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}
