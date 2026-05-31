// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Finances Integrity Zome
//! Entry types and validation for charges, payments, reserves, and budgets.

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// A monthly charge for a member
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct MonthlyCharge {
    pub member: AgentPubKey,
    pub unit_hash: ActionHash,
    pub period_year: u16,
    pub period_month: u8,
    pub base_rent_cents: u64,
    pub maintenance_fee_cents: u64,
    pub utilities_cents: u64,
    pub reserve_contribution_cents: u64,
    pub total_cents: u64,
}

/// Method of payment
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum PaymentMethod {
    BankTransfer,
    MutualCredit,
    Cash,
    Check,
    TimeBankCredit,
}

/// A payment record
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Payment {
    pub member: AgentPubKey,
    pub charge_hash: Option<ActionHash>,
    pub amount_cents: u64,
    pub payment_method: PaymentMethod,
    pub paid_at: Timestamp,
    pub reference: Option<String>,
}

/// Type of reserve fund
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum FundType {
    CapitalReserve,
    OperatingReserve,
    EmergencyFund,
    ImprovementFund,
}

/// A reserve fund
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ReserveFund {
    pub name: String,
    pub fund_type: FundType,
    pub balance_cents: u64,
    pub target_cents: u64,
    pub description: String,
}

/// A budget category line item
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BudgetCategory {
    pub name: String,
    pub allocated_cents: u64,
    pub spent_cents: u64,
}

/// An annual budget
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Budget {
    pub fiscal_year: u16,
    pub income_projected_cents: u64,
    pub expenses_projected_cents: u64,
    pub categories: Vec<BudgetCategory>,
    pub approved: bool,
    pub approved_at: Option<Timestamp>,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    MonthlyCharge(MonthlyCharge),
    Payment(Payment),
    ReserveFund(ReserveFund),
    Budget(Budget),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Member to their charges
    MemberToCharge,
    /// Charge to payments
    ChargeToPayment,
    /// Member to their payments
    MemberToPayment,
    /// All reserve funds
    AllReserveFunds,
    /// Fiscal year to budget
    YearToBudget,
    /// Period anchor to charges
    PeriodToCharge,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::MonthlyCharge(charge) => validate_create_charge(action, charge),
                EntryTypes::Payment(payment) => validate_create_payment(action, payment),
                EntryTypes::ReserveFund(fund) => validate_create_fund(action, fund),
                EntryTypes::Budget(budget) => validate_create_budget(action, budget),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::MonthlyCharge(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Payment(_) => Ok(ValidateCallbackResult::Invalid(
                    "Payments cannot be modified after creation".into(),
                )),
                EntryTypes::ReserveFund(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Budget(_) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::MemberToCharge => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ChargeToPayment => Ok(ValidateCallbackResult::Valid),
            LinkTypes::MemberToPayment => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AllReserveFunds => Ok(ValidateCallbackResult::Valid),
            LinkTypes::YearToBudget => Ok(ValidateCallbackResult::Valid),
            LinkTypes::PeriodToCharge => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type: _,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_charge(
    _action: Create,
    charge: MonthlyCharge,
) -> ExternResult<ValidateCallbackResult> {
    if charge.period_month < 1 || charge.period_month > 12 {
        return Ok(ValidateCallbackResult::Invalid(
            "Month must be between 1 and 12".into(),
        ));
    }
    if charge.period_year < 2020 || charge.period_year > 2100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Year must be between 2020 and 2100".into(),
        ));
    }
    let computed_total = charge.base_rent_cents
        + charge.maintenance_fee_cents
        + charge.utilities_cents
        + charge.reserve_contribution_cents;
    if charge.total_cents != computed_total {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Total ({}) must equal sum of components ({})",
            charge.total_cents, computed_total
        )));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_payment(
    _action: Create,
    payment: Payment,
) -> ExternResult<ValidateCallbackResult> {
    if payment.amount_cents == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Payment amount must be greater than 0".into(),
        ));
    }
    if let Some(ref reference) = payment.reference {
        if reference.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Payment reference must be at most 256 characters".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_fund(
    _action: Create,
    fund: ReserveFund,
) -> ExternResult<ValidateCallbackResult> {
    if fund.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Fund name cannot be empty".into(),
        ));
    }
    if fund.target_cents == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Fund target must be greater than 0".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_budget(_action: Create, budget: Budget) -> ExternResult<ValidateCallbackResult> {
    if budget.fiscal_year < 2020 || budget.fiscal_year > 2100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Fiscal year must be between 2020 and 2100".into(),
        ));
    }
    if budget.categories.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Budget must have at least one category".into(),
        ));
    }
    for cat in &budget.categories {
        if cat.name.is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Budget category name cannot be empty".into(),
            ));
        }
    }
    // Verify category allocations sum to projected expenses
    let total_allocated: u64 = budget.categories.iter().map(|c| c.allocated_cents).sum();
    if total_allocated != budget.expenses_projected_cents {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Category allocations ({}) must equal projected expenses ({})",
            total_allocated, budget.expenses_projected_cents
        )));
    }
    Ok(ValidateCallbackResult::Valid)
}
