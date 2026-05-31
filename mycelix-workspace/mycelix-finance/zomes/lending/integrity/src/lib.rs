// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Lending Integrity Zome
//! Updated to use HDI 0.7 patterns with FlatOp validation
use hdi::prelude::*;

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Loan {
    pub id: String,
    pub borrower_did: String,
    pub lender_did: String,
    pub principal: f64,
    pub currency: String,
    pub interest_rate: f64,
    pub term_days: u32,
    pub collateral_ids: Vec<String>,
    pub status: LoanStatus,
    pub created: Timestamp,
    pub funded: Option<Timestamp>,
    pub maturity: Option<Timestamp>,
    pub repaid: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum LoanStatus {
    Requested,
    Offered,
    Funded,
    Active,
    Repaid,
    Defaulted,
    Cancelled,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct LoanOffer {
    pub id: String,
    pub lender_did: String,
    pub max_amount: f64,
    pub min_amount: f64,
    pub currency: String,
    pub base_interest_rate: f64,
    pub min_credit_score: f64,
    pub max_term_days: u32,
    pub collateral_required: bool,
    pub active: bool,
    pub created: Timestamp,
}

#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct PaymentSchedule {
    pub loan_id: String,
    pub payments: Vec<ScheduledPayment>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ScheduledPayment {
    pub payment_number: u32,
    pub due_date: Timestamp,
    pub principal_amount: f64,
    pub interest_amount: f64,
    pub total_amount: f64,
    pub paid: bool,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Loan(Loan),
    LoanOffer(LoanOffer),
    PaymentSchedule(PaymentSchedule),
}

#[hdk_link_types]
pub enum LinkTypes {
    BorrowerToLoans,
    LenderToLoans,
    LenderToOffers,
    LoanToSchedule,
    CollateralToLoan,
    ActiveOffers,
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
                EntryTypes::Loan(loan) => {
                    validate_create_loan(EntryCreationAction::Create(action), loan)
                }
                EntryTypes::LoanOffer(offer) => {
                    validate_create_loan_offer(EntryCreationAction::Create(action), offer)
                }
                EntryTypes::PaymentSchedule(schedule) => {
                    validate_create_payment_schedule(EntryCreationAction::Create(action), schedule)
                }
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::Loan(loan) => validate_update_loan(action, loan),
                EntryTypes::LoanOffer(offer) => validate_update_loan_offer(action, offer),
                EntryTypes::PaymentSchedule(_) => Ok(ValidateCallbackResult::Invalid(
                    "Payment schedules cannot be updated".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address,
            target_address,
            ..
        } => {
            // Validate hash lengths (39 bytes for Holochain hashes)
            let base_valid = base_address.as_ref().len() == 39;
            let target_valid = target_address.as_ref().len() == 39;

            match link_type {
                LinkTypes::BorrowerToLoans
                | LinkTypes::LenderToLoans
                | LinkTypes::LenderToOffers => {
                    // Agent to entry links
                    if !base_valid || !target_valid {
                        return Ok(ValidateCallbackResult::Invalid(
                            "Link must connect valid agent and entry hashes".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::LoanToSchedule => {
                    // Entry to entry link
                    if !base_valid || !target_valid {
                        return Ok(ValidateCallbackResult::Invalid(
                            "LoanToSchedule must connect valid entry hashes".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::CollateralToLoan => {
                    // Entry to entry link - collateral to loan
                    if !base_valid || !target_valid {
                        return Ok(ValidateCallbackResult::Invalid(
                            "CollateralToLoan must connect valid entry hashes".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
                LinkTypes::ActiveOffers => {
                    // Anchor to offer entries
                    if !target_valid {
                        return Ok(ValidateCallbackResult::Invalid(
                            "ActiveOffers target must be a valid entry hash".into(),
                        ));
                    }
                    Ok(ValidateCallbackResult::Valid)
                }
            }
        }
        FlatOp::RegisterDeleteLink { link_type, .. } => {
            match link_type {
                // Loan to schedule links are immutable once created
                LinkTypes::LoanToSchedule => Ok(ValidateCallbackResult::Invalid(
                    "LoanToSchedule links cannot be deleted - payment schedules are immutable"
                        .into(),
                )),
                // Collateral links cannot be deleted while loan is active
                LinkTypes::CollateralToLoan => Ok(ValidateCallbackResult::Invalid(
                    "CollateralToLoan links cannot be deleted - collateral is locked".into(),
                )),
                _ => Ok(ValidateCallbackResult::Valid),
            }
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_loan(
    _action: EntryCreationAction,
    loan: Loan,
) -> ExternResult<ValidateCallbackResult> {
    if !loan.borrower_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Borrower must be a valid DID".into(),
        ));
    }
    if !loan.lender_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Lender must be a valid DID".into(),
        ));
    }
    if loan.principal <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Principal must be positive".into(),
        ));
    }
    if loan.interest_rate < 0.0 || loan.interest_rate > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Interest rate must be between 0 and 100%".into(),
        ));
    }
    if loan.borrower_did == loan.lender_did {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot lend to yourself".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_loan(_action: Update, loan: Loan) -> ExternResult<ValidateCallbackResult> {
    // Principal and parties cannot change, but status can
    if loan.principal <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Principal must be positive".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_loan_offer(
    _action: EntryCreationAction,
    offer: LoanOffer,
) -> ExternResult<ValidateCallbackResult> {
    if !offer.lender_did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Lender must be a valid DID".into(),
        ));
    }
    if offer.min_amount > offer.max_amount {
        return Ok(ValidateCallbackResult::Invalid(
            "Min amount cannot exceed max amount".into(),
        ));
    }
    if offer.base_interest_rate < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Interest rate cannot be negative".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_loan_offer(
    _action: Update,
    offer: LoanOffer,
) -> ExternResult<ValidateCallbackResult> {
    if offer.min_amount > offer.max_amount {
        return Ok(ValidateCallbackResult::Invalid(
            "Min amount cannot exceed max amount".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_payment_schedule(
    _action: EntryCreationAction,
    schedule: PaymentSchedule,
) -> ExternResult<ValidateCallbackResult> {
    if schedule.payments.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Schedule must have at least one payment".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
