// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Credit Scoring Integrity Zome
//! Updated to use HDI 0.7 patterns with FlatOp validation
use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CreditProfile {
    pub did: String,
    pub matl_score: f64,
    pub payment_history_score: f64,
    pub collateral_ratio: f64,
    pub account_age_days: u64,
    pub activity_score: f64,
    pub computed_score: f64,
    pub last_updated: Timestamp,
    pub version: u32,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PaymentRecord {
    pub id: String,
    pub profile_did: String,
    pub loan_id: String,
    pub amount: f64,
    pub currency: String,
    pub due_date: Timestamp,
    pub paid_date: Option<Timestamp>,
    pub status: PaymentStatus,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PaymentStatus {
    Pending,
    OnTime,
    Late,
    Missed,
    Forgiven,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CollateralRecord {
    pub id: String,
    pub owner_did: String,
    pub asset_type: CollateralType,
    pub asset_id: String,
    pub appraised_value: f64,
    pub currency: String,
    pub locked_for_loan: Option<String>,
    pub registered: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CollateralType {
    Property,
    EnergyAsset,
    Credential,
    Token,
    Other(String),
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    CreditProfile(CreditProfile),
    PaymentRecord(PaymentRecord),
    CollateralRecord(CollateralRecord),
}

#[hdk_link_types]
pub enum LinkTypes {
    DidToProfile,
    ProfileToPayments,
    ProfileToCollateral,
    LoanToPayments,
}

/// Genesis self-check - called when app is installed
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback using FlatOp pattern matching
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::CreditProfile(profile) => {
                    validate_create_credit_profile(EntryCreationAction::Create(action), profile)
                }
                EntryTypes::PaymentRecord(record) => {
                    validate_create_payment_record(EntryCreationAction::Create(action), record)
                }
                EntryTypes::CollateralRecord(record) => {
                    validate_create_collateral_record(EntryCreationAction::Create(action), record)
                }
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::CreditProfile(profile) => {
                    validate_update_credit_profile(action, profile)
                }
                EntryTypes::PaymentRecord(record) => validate_update_payment_record(action, record),
                EntryTypes::CollateralRecord(_) => Ok(ValidateCallbackResult::Invalid(
                    "Collateral records cannot be updated, only new ones created".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::DidToProfile => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ProfileToPayments => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ProfileToCollateral => Ok(ValidateCallbackResult::Valid),
            LinkTypes::LoanToPayments => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate credit profile creation
fn validate_create_credit_profile(
    _action: EntryCreationAction,
    profile: CreditProfile,
) -> ExternResult<ValidateCallbackResult> {
    if !profile.did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "DID must be a valid DID format".into(),
        ));
    }
    if profile.computed_score < 0.0 || profile.computed_score > 1000.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Credit score must be between 0 and 1000".into(),
        ));
    }
    if profile.matl_score < 0.0 || profile.matl_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "MATL score must be between 0 and 1".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate credit profile update
fn validate_update_credit_profile(
    _action: Update,
    profile: CreditProfile,
) -> ExternResult<ValidateCallbackResult> {
    // Apply same rules as creation
    if profile.computed_score < 0.0 || profile.computed_score > 1000.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Credit score must be between 0 and 1000".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate payment record creation
fn validate_create_payment_record(
    _action: EntryCreationAction,
    record: PaymentRecord,
) -> ExternResult<ValidateCallbackResult> {
    if record.amount <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Payment amount must be positive".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate payment record update
fn validate_update_payment_record(
    _action: Update,
    record: PaymentRecord,
) -> ExternResult<ValidateCallbackResult> {
    if record.amount <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Payment amount must be positive".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate collateral record creation
fn validate_create_collateral_record(
    _action: EntryCreationAction,
    record: CollateralRecord,
) -> ExternResult<ValidateCallbackResult> {
    if !record.owner_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Owner must be a valid DID".into(),
        ));
    }
    if record.appraised_value <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Appraised value must be positive".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
