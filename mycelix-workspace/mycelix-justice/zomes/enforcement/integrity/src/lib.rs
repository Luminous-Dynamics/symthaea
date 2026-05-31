// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Enforcement Integrity Zome
use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EnforcementOrder {
    pub id: String,
    pub judgment_id: String,
    pub target: String,
    pub enforcement_type: EnforcementType,
    pub status: EnforcementStatus,
    pub issued: Timestamp,
    pub executed: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EnforcementType {
    FinancialPenalty(f64),
    TrustReduction(f64),
    AccessRestriction(String),
    CredentialRevocation(String),
    Other(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EnforcementStatus {
    Pending,
    InProgress,
    Executed,
    Failed,
    Appealed,
    Suspended,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct EnforcementAppeal {
    pub id: String,
    pub order_id: String,
    pub appellant: String,
    pub reason: String,
    pub filed: Timestamp,
    pub status: AppealStatus,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AppealStatus {
    Pending,
    UnderReview,
    Granted,
    Denied,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    EnforcementOrder(EnforcementOrder),
    EnforcementAppeal(EnforcementAppeal),
}

#[hdk_link_types]
pub enum LinkTypes {
    JudgmentToEnforcement,
    TargetToEnforcement,
    OrderToAppeal,
}

#[hdk_extern]
pub fn validate_create_entry_enforcement_order(
    _action: EntryCreationAction,
    order: EnforcementOrder,
) -> ExternResult<ValidateCallbackResult> {
    if !order.target.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Target must be a valid DID".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
