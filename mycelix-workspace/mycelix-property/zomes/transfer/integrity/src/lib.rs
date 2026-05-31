// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Property Transfer Integrity Zome
use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Transfer {
    pub id: String,
    pub property_id: String,
    pub from_did: String,
    pub to_did: String,
    pub transfer_type: TransferType,
    pub price: Option<f64>,
    pub currency: Option<String>,
    pub conditions: Vec<TransferCondition>,
    pub status: TransferStatus,
    pub initiated: Timestamp,
    pub completed: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TransferType {
    Sale,
    Gift,
    Inheritance,
    CourtOrder,
    Exchange,
    Other,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TransferCondition {
    pub condition_type: ConditionType,
    pub description: String,
    pub satisfied: bool,
    pub verified_by: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ConditionType {
    PaymentReceived,
    InspectionComplete,
    TitleClear,
    DocumentsSigned,
    TaxesPaid,
    Custom(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TransferStatus {
    Initiated,
    AwaitingAcceptance,
    InEscrow,
    ConditionsPending,
    Completed,
    Cancelled,
    Disputed,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Escrow {
    pub id: String,
    pub transfer_id: String,
    pub escrow_agent_did: Option<String>,
    pub amount: f64,
    pub currency: String,
    pub funded: bool,
    pub release_conditions: Vec<String>,
    pub created: Timestamp,
    pub released: Option<Timestamp>,
}

/// Anchor entry for deterministic link bases from strings
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Transfer(Transfer),
    Escrow(Escrow),
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
}

#[hdk_link_types]
pub enum LinkTypes {
    PropertyToTransfers,
    SellerToTransfers,
    BuyerToTransfers,
    TransferToEscrow,
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
                EntryTypes::Transfer(transfer) => {
                    validate_create_transfer(EntryCreationAction::Create(action), transfer)
                }
                EntryTypes::Escrow(escrow) => {
                    validate_create_escrow(EntryCreationAction::Create(action), escrow)
                }
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::Transfer(transfer) => validate_update_transfer(action, transfer),
                EntryTypes::Escrow(escrow) => validate_update_escrow(action, escrow),
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::PropertyToTransfers => Ok(ValidateCallbackResult::Valid),
            LinkTypes::SellerToTransfers => Ok(ValidateCallbackResult::Valid),
            LinkTypes::BuyerToTransfers => Ok(ValidateCallbackResult::Valid),
            LinkTypes::TransferToEscrow => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { .. } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_transfer(
    _action: EntryCreationAction,
    transfer: Transfer,
) -> ExternResult<ValidateCallbackResult> {
    if !transfer.from_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Seller must be a valid DID".into(),
        ));
    }
    if !transfer.to_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Buyer must be a valid DID".into(),
        ));
    }
    if transfer.from_did == transfer.to_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot transfer to yourself".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_transfer(
    _action: Update,
    _transfer: Transfer,
) -> ExternResult<ValidateCallbackResult> {
    // Status can be updated
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_escrow(
    _action: EntryCreationAction,
    escrow: Escrow,
) -> ExternResult<ValidateCallbackResult> {
    if escrow.amount <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Escrow amount must be positive".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_escrow(_action: Update, escrow: Escrow) -> ExternResult<ValidateCallbackResult> {
    if escrow.amount <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Escrow amount must be positive".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
