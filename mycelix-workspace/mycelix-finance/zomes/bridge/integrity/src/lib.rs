// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Finance Bridge Integrity Zome
//! Updated to use HDI 0.7 patterns with FlatOp validation
//!
//! Entry types for cross-hApp credit queries, payment processing, and collateral management.

use hdi::prelude::*;

/// Credit query from another hApp
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CreditQuery {
    pub id: String,
    pub did: String,
    pub source_happ: String,
    pub purpose: CreditPurpose,
    pub queried_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CreditPurpose {
    LoanApplication,
    TrustVerification,
    MarketplaceTransaction,
    PropertyPurchase,
    EnergyInvestment,
}

/// Credit score result
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CreditResult {
    pub id: String,
    pub did: String,
    pub matl_score: f64,
    pub credit_score: f64,
    pub payment_history_score: f64,
    pub collateral_ratio: f64,
    pub active_loans: u32,
    pub total_repaid: u64,
    pub calculated_at: Timestamp,
}

/// Cross-hApp payment request
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CrossHappPayment {
    pub id: String,
    pub source_happ: String,
    pub from_did: String,
    pub to_did: String,
    pub amount: u64,
    pub currency: String,
    pub reference: String,
    pub status: PaymentStatus,
    pub created_at: Timestamp,
    pub completed_at: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PaymentStatus {
    Pending,
    Processing,
    Completed,
    Failed,
    Refunded,
    Disputed,
}

/// Collateral registration from property hApp
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CollateralRegistration {
    pub id: String,
    pub owner_did: String,
    pub asset_type: AssetType,
    pub asset_id: String,
    pub source_happ: String,
    pub value_estimate: u64,
    pub currency: String,
    pub status: CollateralStatus,
    pub registered_at: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AssetType {
    RealEstate,
    Vehicle,
    Cryptocurrency,
    EnergyAsset,
    Equipment,
    Other(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CollateralStatus {
    Available,
    Pledged,
    Frozen,
    Released,
}

/// Finance event for broadcasting
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FinanceBridgeEvent {
    pub id: String,
    pub event_type: FinanceEventType,
    pub subject_did: String,
    pub amount: Option<u64>,
    pub payload: String,
    pub source_happ: String,
    pub timestamp: Timestamp,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum FinanceEventType {
    CreditScoreUpdated,
    LoanApproved,
    LoanRepaid,
    PaymentCompleted,
    CollateralPledged,
    CollateralReleased,
    DefaultOccurred,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    CreditQuery(CreditQuery),
    CreditResult(CreditResult),
    CrossHappPayment(CrossHappPayment),
    CollateralRegistration(CollateralRegistration),
    FinanceBridgeEvent(FinanceBridgeEvent),
}

#[hdk_link_types]
pub enum LinkTypes {
    DidToCredit,
    DidToPayments,
    HappToPayments,
    CollateralRegistry,
    RecentEvents,
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
                EntryTypes::CreditQuery(query) => {
                    validate_create_credit_query(EntryCreationAction::Create(action), query)
                }
                EntryTypes::CreditResult(result) => {
                    validate_create_credit_result(EntryCreationAction::Create(action), result)
                }
                EntryTypes::CrossHappPayment(payment) => {
                    validate_create_cross_happ_payment(EntryCreationAction::Create(action), payment)
                }
                EntryTypes::CollateralRegistration(collateral) => {
                    validate_create_collateral_registration(
                        EntryCreationAction::Create(action),
                        collateral,
                    )
                }
                EntryTypes::FinanceBridgeEvent(event) => {
                    validate_create_finance_bridge_event(EntryCreationAction::Create(action), event)
                }
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::CreditQuery(_) => Ok(ValidateCallbackResult::Invalid(
                    "Credit queries cannot be updated".into(),
                )),
                EntryTypes::CreditResult(result) => validate_update_credit_result(action, result),
                EntryTypes::CrossHappPayment(payment) => {
                    validate_update_cross_happ_payment(action, payment)
                }
                EntryTypes::CollateralRegistration(collateral) => {
                    validate_update_collateral_registration(action, collateral)
                }
                EntryTypes::FinanceBridgeEvent(_) => Ok(ValidateCallbackResult::Invalid(
                    "Finance events cannot be updated".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::DidToCredit => Ok(ValidateCallbackResult::Valid),
            LinkTypes::DidToPayments => Ok(ValidateCallbackResult::Valid),
            LinkTypes::HappToPayments => Ok(ValidateCallbackResult::Valid),
            LinkTypes::CollateralRegistry => Ok(ValidateCallbackResult::Valid),
            LinkTypes::RecentEvents => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_credit_query(
    _action: EntryCreationAction,
    query: CreditQuery,
) -> ExternResult<ValidateCallbackResult> {
    if !query.did.starts_with("did:mycelix:") {
        return Ok(ValidateCallbackResult::Invalid("DID must be valid".into()));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_credit_result(
    _action: EntryCreationAction,
    result: CreditResult,
) -> ExternResult<ValidateCallbackResult> {
    if result.credit_score < 0.0 || result.credit_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Credit score must be 0.0-1.0".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_credit_result(
    _action: Update,
    result: CreditResult,
) -> ExternResult<ValidateCallbackResult> {
    if result.credit_score < 0.0 || result.credit_score > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Credit score must be 0.0-1.0".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_cross_happ_payment(
    _action: EntryCreationAction,
    payment: CrossHappPayment,
) -> ExternResult<ValidateCallbackResult> {
    if payment.amount == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Amount must be positive".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_cross_happ_payment(
    _action: Update,
    _payment: CrossHappPayment,
) -> ExternResult<ValidateCallbackResult> {
    // Status can be updated
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_collateral_registration(
    _action: EntryCreationAction,
    _collateral: CollateralRegistration,
) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_collateral_registration(
    _action: Update,
    _collateral: CollateralRegistration,
) -> ExternResult<ValidateCallbackResult> {
    // Status can be updated
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_finance_bridge_event(
    _action: EntryCreationAction,
    event: FinanceBridgeEvent,
) -> ExternResult<ValidateCallbackResult> {
    if event.source_happ.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Source hApp required".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
