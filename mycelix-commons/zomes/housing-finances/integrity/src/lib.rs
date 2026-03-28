// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Finances Integrity Zome
//! Entry types and validation for charges, payments, reserves, and budgets.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

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
                EntryTypes::MonthlyCharge(charge) => validate_update_charge(charge),
                EntryTypes::Payment(_) => Ok(ValidateCallbackResult::Invalid(
                    "Payments cannot be modified after creation".into(),
                )),
                EntryTypes::ReserveFund(fund) => validate_update_fund(fund),
                EntryTypes::Budget(_) => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag,
            action: _,
        } => match link_type {
            LinkTypes::MemberToCharge => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "MemberToCharge link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::ChargeToPayment => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "ChargeToPayment link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::MemberToPayment => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "MemberToPayment link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AllReserveFunds => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllReserveFunds link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::YearToBudget => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "YearToBudget link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::PeriodToCharge => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "PeriodToCharge link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        FlatOp::RegisterDeleteLink { action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            Ok(check_link_author_match(
                original_action.action().author(),
                &action.author,
            ))
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "update",
            ))
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "delete",
            ))
        }
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
    if fund.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Fund name cannot be empty".into(),
        ));
    }
    if fund.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Fund name too long (max 256 chars)".into(),
        ));
    }
    if fund.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Fund description too long (max 4096 chars)".into(),
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
        if cat.name.trim().is_empty() {
            return Ok(ValidateCallbackResult::Invalid(
                "Budget category name cannot be empty".into(),
            ));
        }
        if cat.name.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Budget category name too long (max 256 chars)".into(),
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

fn validate_update_charge(charge: MonthlyCharge) -> ExternResult<ValidateCallbackResult> {
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

fn validate_update_fund(fund: ReserveFund) -> ExternResult<ValidateCallbackResult> {
    if fund.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Fund name cannot be empty".into(),
        ));
    }
    if fund.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Fund name too long (max 256 chars)".into(),
        ));
    }
    if fund.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Fund description too long (max 4096 chars)".into(),
        ));
    }
    if fund.target_cents == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Fund target must be greater than 0".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // HELPERS
    // ========================================================================

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_entry_hash() -> EntryHash {
        EntryHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_create() -> Create {
        Create {
            author: fake_agent(),
            timestamp: Timestamp::from_micros(0),
            action_seq: 0,
            prev_action: fake_action_hash(),
            entry_type: EntryType::App(AppEntryDef::new(
                EntryDefIndex(0),
                ZomeIndex(0),
                EntryVisibility::Public,
            )),
            entry_hash: fake_entry_hash(),
            weight: EntryRateWeight::default(),
        }
    }

    fn is_valid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Valid))
    }

    fn is_invalid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Invalid(_)))
    }

    fn make_charge() -> MonthlyCharge {
        MonthlyCharge {
            member: fake_agent(),
            unit_hash: fake_action_hash(),
            period_year: 2025,
            period_month: 6,
            base_rent_cents: 100_000,
            maintenance_fee_cents: 20_000,
            utilities_cents: 15_000,
            reserve_contribution_cents: 5_000,
            total_cents: 140_000,
        }
    }

    fn make_payment() -> Payment {
        Payment {
            member: fake_agent(),
            charge_hash: Some(fake_action_hash()),
            amount_cents: 140_000,
            payment_method: PaymentMethod::BankTransfer,
            paid_at: Timestamp::from_micros(0),
            reference: None,
        }
    }

    fn make_fund() -> ReserveFund {
        ReserveFund {
            name: "Capital Reserve".into(),
            fund_type: FundType::CapitalReserve,
            balance_cents: 50_000,
            target_cents: 500_000,
            description: "Long-term capital improvements".into(),
        }
    }

    fn make_budget() -> Budget {
        Budget {
            fiscal_year: 2025,
            income_projected_cents: 1_000_000,
            expenses_projected_cents: 900_000,
            categories: vec![
                BudgetCategory {
                    name: "Maintenance".into(),
                    allocated_cents: 400_000,
                    spent_cents: 0,
                },
                BudgetCategory {
                    name: "Utilities".into(),
                    allocated_cents: 300_000,
                    spent_cents: 0,
                },
                BudgetCategory {
                    name: "Insurance".into(),
                    allocated_cents: 200_000,
                    spent_cents: 0,
                },
            ],
            approved: false,
            approved_at: None,
        }
    }

    // ========================================================================
    // CHARGE VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_charge_passes() {
        let result = validate_create_charge(fake_create(), make_charge());
        assert!(is_valid(&result));
    }

    #[test]
    fn charge_month_zero_rejected() {
        let mut charge = make_charge();
        charge.period_month = 0;
        let result = validate_create_charge(fake_create(), charge);
        assert!(is_invalid(&result));
    }

    #[test]
    fn charge_month_13_rejected() {
        let mut charge = make_charge();
        charge.period_month = 13;
        let result = validate_create_charge(fake_create(), charge);
        assert!(is_invalid(&result));
    }

    #[test]
    fn charge_month_1_accepted() {
        let mut charge = make_charge();
        charge.period_month = 1;
        let result = validate_create_charge(fake_create(), charge);
        assert!(is_valid(&result));
    }

    #[test]
    fn charge_month_12_accepted() {
        let mut charge = make_charge();
        charge.period_month = 12;
        let result = validate_create_charge(fake_create(), charge);
        assert!(is_valid(&result));
    }

    #[test]
    fn charge_year_2019_rejected() {
        let mut charge = make_charge();
        charge.period_year = 2019;
        let result = validate_create_charge(fake_create(), charge);
        assert!(is_invalid(&result));
    }

    #[test]
    fn charge_year_2101_rejected() {
        let mut charge = make_charge();
        charge.period_year = 2101;
        let result = validate_create_charge(fake_create(), charge);
        assert!(is_invalid(&result));
    }

    #[test]
    fn charge_year_2020_accepted() {
        let mut charge = make_charge();
        charge.period_year = 2020;
        let result = validate_create_charge(fake_create(), charge);
        assert!(is_valid(&result));
    }

    #[test]
    fn charge_year_2100_accepted() {
        let mut charge = make_charge();
        charge.period_year = 2100;
        let result = validate_create_charge(fake_create(), charge);
        assert!(is_valid(&result));
    }

    #[test]
    fn charge_total_mismatch_rejected() {
        let mut charge = make_charge();
        charge.total_cents = 999_999; // Wrong total
        let result = validate_create_charge(fake_create(), charge);
        assert!(is_invalid(&result));
    }

    #[test]
    fn charge_total_off_by_one_rejected() {
        let mut charge = make_charge();
        charge.total_cents = 140_001; // Off by 1
        let result = validate_create_charge(fake_create(), charge);
        assert!(is_invalid(&result));
    }

    #[test]
    fn charge_all_zeros_valid() {
        let mut charge = make_charge();
        charge.base_rent_cents = 0;
        charge.maintenance_fee_cents = 0;
        charge.utilities_cents = 0;
        charge.reserve_contribution_cents = 0;
        charge.total_cents = 0;
        let result = validate_create_charge(fake_create(), charge);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // PAYMENT VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_payment_passes() {
        let result = validate_create_payment(fake_create(), make_payment());
        assert!(is_valid(&result));
    }

    #[test]
    fn payment_zero_amount_rejected() {
        let mut payment = make_payment();
        payment.amount_cents = 0;
        let result = validate_create_payment(fake_create(), payment);
        assert!(is_invalid(&result));
    }

    #[test]
    fn payment_one_cent_accepted() {
        let mut payment = make_payment();
        payment.amount_cents = 1;
        let result = validate_create_payment(fake_create(), payment);
        assert!(is_valid(&result));
    }

    #[test]
    fn payment_no_reference_accepted() {
        let payment = make_payment(); // reference is None by default
        let result = validate_create_payment(fake_create(), payment);
        assert!(is_valid(&result));
    }

    #[test]
    fn payment_short_reference_accepted() {
        let mut payment = make_payment();
        payment.reference = Some("TXN-12345".into());
        let result = validate_create_payment(fake_create(), payment);
        assert!(is_valid(&result));
    }

    #[test]
    fn payment_reference_at_limit_accepted() {
        let mut payment = make_payment();
        payment.reference = Some("x".repeat(256));
        let result = validate_create_payment(fake_create(), payment);
        assert!(is_valid(&result));
    }

    #[test]
    fn payment_reference_over_limit_rejected() {
        let mut payment = make_payment();
        payment.reference = Some("x".repeat(257));
        let result = validate_create_payment(fake_create(), payment);
        assert!(is_invalid(&result));
    }

    #[test]
    fn payment_no_charge_hash_accepted() {
        let mut payment = make_payment();
        payment.charge_hash = None;
        let result = validate_create_payment(fake_create(), payment);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // FUND VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_fund_passes() {
        let result = validate_create_fund(fake_create(), make_fund());
        assert!(is_valid(&result));
    }

    #[test]
    fn fund_empty_name_rejected() {
        let mut fund = make_fund();
        fund.name = "".into();
        let result = validate_create_fund(fake_create(), fund);
        assert!(is_invalid(&result));
    }

    #[test]
    fn fund_zero_target_rejected() {
        let mut fund = make_fund();
        fund.target_cents = 0;
        let result = validate_create_fund(fake_create(), fund);
        assert!(is_invalid(&result));
    }

    #[test]
    fn fund_one_cent_target_accepted() {
        let mut fund = make_fund();
        fund.target_cents = 1;
        let result = validate_create_fund(fake_create(), fund);
        assert!(is_valid(&result));
    }

    #[test]
    fn fund_zero_balance_accepted() {
        let mut fund = make_fund();
        fund.balance_cents = 0;
        let result = validate_create_fund(fake_create(), fund);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // BUDGET VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_budget_passes() {
        let result = validate_create_budget(fake_create(), make_budget());
        assert!(is_valid(&result));
    }

    #[test]
    fn budget_year_2019_rejected() {
        let mut budget = make_budget();
        budget.fiscal_year = 2019;
        let result = validate_create_budget(fake_create(), budget);
        assert!(is_invalid(&result));
    }

    #[test]
    fn budget_year_2101_rejected() {
        let mut budget = make_budget();
        budget.fiscal_year = 2101;
        let result = validate_create_budget(fake_create(), budget);
        assert!(is_invalid(&result));
    }

    #[test]
    fn budget_year_2020_accepted() {
        let mut budget = make_budget();
        budget.fiscal_year = 2020;
        let result = validate_create_budget(fake_create(), budget);
        assert!(is_valid(&result));
    }

    #[test]
    fn budget_year_2100_accepted() {
        let mut budget = make_budget();
        budget.fiscal_year = 2100;
        let result = validate_create_budget(fake_create(), budget);
        assert!(is_valid(&result));
    }

    #[test]
    fn budget_no_categories_rejected() {
        let mut budget = make_budget();
        budget.categories = vec![];
        let result = validate_create_budget(fake_create(), budget);
        assert!(is_invalid(&result));
    }

    #[test]
    fn budget_empty_category_name_rejected() {
        let mut budget = make_budget();
        budget.categories[1].name = "".into();
        let result = validate_create_budget(fake_create(), budget);
        assert!(is_invalid(&result));
    }

    #[test]
    fn budget_allocation_mismatch_rejected() {
        let mut budget = make_budget();
        budget.expenses_projected_cents = 1_000_000; // Doesn't match 900k allocated
        let result = validate_create_budget(fake_create(), budget);
        assert!(is_invalid(&result));
    }

    #[test]
    fn budget_single_category_matching_accepted() {
        let budget = Budget {
            fiscal_year: 2025,
            income_projected_cents: 500_000,
            expenses_projected_cents: 200_000,
            categories: vec![BudgetCategory {
                name: "Operations".into(),
                allocated_cents: 200_000,
                spent_cents: 0,
            }],
            approved: true,
            approved_at: Some(Timestamp::from_micros(1_000_000)),
        };
        let result = validate_create_budget(fake_create(), budget);
        assert!(is_valid(&result));
    }

    #[test]
    fn budget_approved_with_timestamp_accepted() {
        let mut budget = make_budget();
        budget.approved = true;
        budget.approved_at = Some(Timestamp::from_micros(1_000_000));
        let result = validate_create_budget(fake_create(), budget);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn serde_roundtrip_payment_method() {
        for variant in [
            PaymentMethod::BankTransfer,
            PaymentMethod::MutualCredit,
            PaymentMethod::Cash,
            PaymentMethod::Check,
            PaymentMethod::TimeBankCredit,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: PaymentMethod = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn serde_roundtrip_fund_type() {
        for variant in [
            FundType::CapitalReserve,
            FundType::OperatingReserve,
            FundType::EmergencyFund,
            FundType::ImprovementFund,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: FundType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn serde_roundtrip_budget_category() {
        let cat = BudgetCategory {
            name: "Maintenance".into(),
            allocated_cents: 400_000,
            spent_cents: 123_456,
        };
        let json = serde_json::to_string(&cat).unwrap();
        let back: BudgetCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(cat, back);
    }

    #[test]
    fn serde_roundtrip_anchor() {
        let anchor = Anchor("charges_2025_06".into());
        let json = serde_json::to_string(&anchor).unwrap();
        let back: Anchor = serde_json::from_str(&json).unwrap();
        assert_eq!(anchor, back);
    }

    #[test]
    fn serde_roundtrip_monthly_charge() {
        let charge = make_charge();
        let json = serde_json::to_string(&charge).unwrap();
        let back: MonthlyCharge = serde_json::from_str(&json).unwrap();
        assert_eq!(charge, back);
    }

    #[test]
    fn serde_roundtrip_payment() {
        let payment = make_payment();
        let json = serde_json::to_string(&payment).unwrap();
        let back: Payment = serde_json::from_str(&json).unwrap();
        assert_eq!(payment, back);
    }

    #[test]
    fn serde_roundtrip_payment_with_reference() {
        let mut payment = make_payment();
        payment.reference = Some("REF-2025-06-001".into());
        let json = serde_json::to_string(&payment).unwrap();
        let back: Payment = serde_json::from_str(&json).unwrap();
        assert_eq!(payment, back);
    }

    #[test]
    fn serde_roundtrip_payment_no_charge_hash() {
        let mut payment = make_payment();
        payment.charge_hash = None;
        let json = serde_json::to_string(&payment).unwrap();
        let back: Payment = serde_json::from_str(&json).unwrap();
        assert_eq!(payment, back);
    }

    #[test]
    fn serde_roundtrip_reserve_fund() {
        let fund = make_fund();
        let json = serde_json::to_string(&fund).unwrap();
        let back: ReserveFund = serde_json::from_str(&json).unwrap();
        assert_eq!(fund, back);
    }

    #[test]
    fn serde_roundtrip_budget() {
        let budget = make_budget();
        let json = serde_json::to_string(&budget).unwrap();
        let back: Budget = serde_json::from_str(&json).unwrap();
        assert_eq!(budget, back);
    }

    #[test]
    fn serde_roundtrip_budget_approved() {
        let mut budget = make_budget();
        budget.approved = true;
        budget.approved_at = Some(Timestamp::from_micros(1_700_000_000_000_000));
        let json = serde_json::to_string(&budget).unwrap();
        let back: Budget = serde_json::from_str(&json).unwrap();
        assert_eq!(budget, back);
    }

    // ========================================================================
    // CHARGE EDGE CASES
    // ========================================================================

    #[test]
    fn charge_max_u64_components_with_correct_total() {
        // Use large values that don't overflow u64 when summed
        let mut charge = make_charge();
        charge.base_rent_cents = u64::MAX / 4;
        charge.maintenance_fee_cents = u64::MAX / 4;
        charge.utilities_cents = u64::MAX / 4;
        charge.reserve_contribution_cents = u64::MAX / 4;
        let total = charge.base_rent_cents
            + charge.maintenance_fee_cents
            + charge.utilities_cents
            + charge.reserve_contribution_cents;
        charge.total_cents = total;
        let result = validate_create_charge(fake_create(), charge);
        assert!(is_valid(&result));
    }

    #[test]
    fn charge_single_component_nonzero() {
        let mut charge = make_charge();
        charge.base_rent_cents = 100;
        charge.maintenance_fee_cents = 0;
        charge.utilities_cents = 0;
        charge.reserve_contribution_cents = 0;
        charge.total_cents = 100;
        let result = validate_create_charge(fake_create(), charge);
        assert!(is_valid(&result));
    }

    #[test]
    fn charge_month_255_rejected() {
        let mut charge = make_charge();
        charge.period_month = 255; // u8 max
        let result = validate_create_charge(fake_create(), charge);
        assert!(is_invalid(&result));
    }

    #[test]
    fn charge_year_0_rejected() {
        let mut charge = make_charge();
        charge.period_year = 0;
        let result = validate_create_charge(fake_create(), charge);
        assert!(is_invalid(&result));
    }

    #[test]
    fn charge_year_u16_max_rejected() {
        let mut charge = make_charge();
        charge.period_year = u16::MAX;
        let result = validate_create_charge(fake_create(), charge);
        assert!(is_invalid(&result));
    }

    #[test]
    fn charge_total_zero_when_components_nonzero_rejected() {
        let mut charge = make_charge();
        // Components sum to 140_000 but total says 0
        charge.total_cents = 0;
        let result = validate_create_charge(fake_create(), charge);
        assert!(is_invalid(&result));
    }

    #[test]
    fn charge_every_boundary_month() {
        // Test months 1 through 12 are all valid
        for m in 1u8..=12 {
            let mut charge = make_charge();
            charge.period_month = m;
            let result = validate_create_charge(fake_create(), charge);
            assert!(is_valid(&result), "month {} should be valid", m);
        }
    }

    // ========================================================================
    // PAYMENT EDGE CASES
    // ========================================================================

    #[test]
    fn payment_max_u64_amount_accepted() {
        let mut payment = make_payment();
        payment.amount_cents = u64::MAX;
        let result = validate_create_payment(fake_create(), payment);
        assert!(is_valid(&result));
    }

    #[test]
    fn payment_all_methods_accepted() {
        for method in [
            PaymentMethod::BankTransfer,
            PaymentMethod::MutualCredit,
            PaymentMethod::Cash,
            PaymentMethod::Check,
            PaymentMethod::TimeBankCredit,
        ] {
            let mut payment = make_payment();
            payment.payment_method = method.clone();
            let result = validate_create_payment(fake_create(), payment);
            assert!(
                is_valid(&result),
                "payment method {:?} should be accepted",
                method
            );
        }
    }

    #[test]
    fn payment_reference_unicode_accepted() {
        let mut payment = make_payment();
        payment.reference = Some("Zahlung-\u{00FC}berweisung-\u{20AC}100".into());
        let result = validate_create_payment(fake_create(), payment);
        assert!(is_valid(&result));
    }

    #[test]
    fn payment_reference_unicode_over_256_chars_rejected() {
        let mut payment = make_payment();
        // Each emoji is 4 bytes but 1 char; we need >256 chars
        let emoji_ref: String = "\u{1F600}".repeat(257);
        payment.reference = Some(emoji_ref);
        let result = validate_create_payment(fake_create(), payment);
        // Note: .len() counts bytes, not chars. 257 emojis = 1028 bytes > 256
        assert!(is_invalid(&result));
    }

    #[test]
    fn payment_reference_exactly_256_bytes_multibyte_accepted() {
        let mut payment = make_payment();
        // 256 ASCII chars = 256 bytes = 256 len
        payment.reference = Some("A".repeat(256));
        let result = validate_create_payment(fake_create(), payment);
        assert!(is_valid(&result));
    }

    #[test]
    fn payment_empty_reference_string_accepted() {
        let mut payment = make_payment();
        payment.reference = Some(String::new());
        let result = validate_create_payment(fake_create(), payment);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // FUND EDGE CASES
    // ========================================================================

    #[test]
    fn fund_all_types_accepted() {
        for fund_type in [
            FundType::CapitalReserve,
            FundType::OperatingReserve,
            FundType::EmergencyFund,
            FundType::ImprovementFund,
        ] {
            let mut fund = make_fund();
            fund.fund_type = fund_type.clone();
            let result = validate_create_fund(fake_create(), fund);
            assert!(
                is_valid(&result),
                "fund type {:?} should be accepted",
                fund_type
            );
        }
    }

    #[test]
    fn fund_unicode_name_accepted() {
        let mut fund = make_fund();
        fund.name = "\u{00C9}mergence R\u{00E9}serve \u{2014} \u{1F30D}".into();
        let result = validate_create_fund(fake_create(), fund);
        assert!(is_valid(&result));
    }

    #[test]
    fn fund_whitespace_only_name_rejected() {
        let mut fund = make_fund();
        fund.name = "   ".into();
        let result = validate_create_fund(fake_create(), fund);
        assert!(is_invalid(&result));
    }

    #[test]
    fn fund_max_target_accepted() {
        let mut fund = make_fund();
        fund.target_cents = u64::MAX;
        let result = validate_create_fund(fake_create(), fund);
        assert!(is_valid(&result));
    }

    #[test]
    fn fund_balance_exceeds_target_accepted() {
        // No validation prevents balance > target
        let mut fund = make_fund();
        fund.balance_cents = 1_000_000;
        fund.target_cents = 500_000;
        let result = validate_create_fund(fake_create(), fund);
        assert!(is_valid(&result));
    }

    #[test]
    fn fund_empty_description_accepted() {
        let mut fund = make_fund();
        fund.description = String::new();
        let result = validate_create_fund(fake_create(), fund);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // BUDGET EDGE CASES
    // ========================================================================

    #[test]
    fn budget_many_categories_accepted() {
        let cats: Vec<BudgetCategory> = (0..100)
            .map(|i| BudgetCategory {
                name: format!("Category {}", i),
                allocated_cents: 1_000,
                spent_cents: 0,
            })
            .collect();
        let budget = Budget {
            fiscal_year: 2025,
            income_projected_cents: 200_000,
            expenses_projected_cents: 100_000, // 100 * 1000
            categories: cats,
            approved: false,
            approved_at: None,
        };
        let result = validate_create_budget(fake_create(), budget);
        assert!(is_valid(&result));
    }

    #[test]
    fn budget_unicode_category_names_accepted() {
        let budget = Budget {
            fiscal_year: 2025,
            income_projected_cents: 500_000,
            expenses_projected_cents: 300_000,
            categories: vec![
                BudgetCategory {
                    name: "\u{0420}\u{0435}\u{043C}\u{043E}\u{043D}\u{0442}".into(), // Russian: Remont
                    allocated_cents: 150_000,
                    spent_cents: 0,
                },
                BudgetCategory {
                    name: "\u{4FDD}\u{9669}".into(), // Chinese: Insurance
                    allocated_cents: 150_000,
                    spent_cents: 0,
                },
            ],
            approved: false,
            approved_at: None,
        };
        let result = validate_create_budget(fake_create(), budget);
        assert!(is_valid(&result));
    }

    #[test]
    fn budget_zero_projections_with_zero_allocations_accepted() {
        let budget = Budget {
            fiscal_year: 2025,
            income_projected_cents: 0,
            expenses_projected_cents: 0,
            categories: vec![BudgetCategory {
                name: "Placeholder".into(),
                allocated_cents: 0,
                spent_cents: 0,
            }],
            approved: false,
            approved_at: None,
        };
        let result = validate_create_budget(fake_create(), budget);
        assert!(is_valid(&result));
    }

    #[test]
    fn budget_income_less_than_expenses_accepted() {
        // Deficit budget should pass validation (no such check)
        let budget = Budget {
            fiscal_year: 2025,
            income_projected_cents: 100_000,
            expenses_projected_cents: 500_000,
            categories: vec![BudgetCategory {
                name: "Operations".into(),
                allocated_cents: 500_000,
                spent_cents: 0,
            }],
            approved: false,
            approved_at: None,
        };
        let result = validate_create_budget(fake_create(), budget);
        assert!(is_valid(&result));
    }

    #[test]
    fn budget_unapproved_with_timestamp_accepted() {
        // No validation preventing approved=false + approved_at=Some(...)
        let mut budget = make_budget();
        budget.approved = false;
        budget.approved_at = Some(Timestamp::from_micros(1_000_000));
        let result = validate_create_budget(fake_create(), budget);
        assert!(is_valid(&result));
    }

    #[test]
    fn budget_all_categories_empty_name_rejected() {
        let budget = Budget {
            fiscal_year: 2025,
            income_projected_cents: 500_000,
            expenses_projected_cents: 0,
            categories: vec![BudgetCategory {
                name: "".into(),
                allocated_cents: 0,
                spent_cents: 0,
            }],
            approved: false,
            approved_at: None,
        };
        let result = validate_create_budget(fake_create(), budget);
        assert!(is_invalid(&result));
    }

    #[test]
    fn budget_second_category_empty_name_rejected() {
        let budget = Budget {
            fiscal_year: 2025,
            income_projected_cents: 500_000,
            expenses_projected_cents: 300_000,
            categories: vec![
                BudgetCategory {
                    name: "Valid".into(),
                    allocated_cents: 300_000,
                    spent_cents: 0,
                },
                BudgetCategory {
                    name: "".into(),
                    allocated_cents: 0,
                    spent_cents: 0,
                },
            ],
            approved: false,
            approved_at: None,
        };
        let result = validate_create_budget(fake_create(), budget);
        assert!(is_invalid(&result));
    }

    #[test]
    fn budget_allocation_off_by_one_rejected() {
        let mut budget = make_budget();
        // categories sum to 900_000, set projected to 900_001
        budget.expenses_projected_cents = 900_001;
        let result = validate_create_budget(fake_create(), budget);
        assert!(is_invalid(&result));
    }

    #[test]
    fn budget_year_boundary_2020_accepted() {
        let mut budget = make_budget();
        budget.fiscal_year = 2020;
        let result = validate_create_budget(fake_create(), budget);
        assert!(is_valid(&result));
    }

    #[test]
    fn budget_year_boundary_2100_accepted() {
        let mut budget = make_budget();
        budget.fiscal_year = 2100;
        let result = validate_create_budget(fake_create(), budget);
        assert!(is_valid(&result));
    }

    #[test]
    fn budget_categories_with_nonzero_spent_accepted() {
        let budget = Budget {
            fiscal_year: 2025,
            income_projected_cents: 500_000,
            expenses_projected_cents: 300_000,
            categories: vec![BudgetCategory {
                name: "Operations".into(),
                allocated_cents: 300_000,
                spent_cents: 250_000,
            }],
            approved: true,
            approved_at: Some(Timestamp::from_micros(1_000_000)),
        };
        let result = validate_create_budget(fake_create(), budget);
        assert!(is_valid(&result));
    }

    #[test]
    fn budget_spent_exceeds_allocated_accepted() {
        // No validation prevents overspending
        let budget = Budget {
            fiscal_year: 2025,
            income_projected_cents: 500_000,
            expenses_projected_cents: 300_000,
            categories: vec![BudgetCategory {
                name: "Overbudget".into(),
                allocated_cents: 300_000,
                spent_cents: 999_999,
            }],
            approved: false,
            approved_at: None,
        };
        let result = validate_create_budget(fake_create(), budget);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // VALIDATION ERROR MESSAGE TESTS
    // ========================================================================

    #[test]
    fn charge_month_error_message_correct() {
        let mut charge = make_charge();
        charge.period_month = 0;
        let result = validate_create_charge(fake_create(), charge);
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert_eq!(msg, "Month must be between 1 and 12");
            }
            other => panic!("Expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn charge_year_error_message_correct() {
        let mut charge = make_charge();
        charge.period_year = 2019;
        let result = validate_create_charge(fake_create(), charge);
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert_eq!(msg, "Year must be between 2020 and 2100");
            }
            other => panic!("Expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn charge_total_error_message_contains_values() {
        let mut charge = make_charge();
        charge.total_cents = 999;
        let result = validate_create_charge(fake_create(), charge);
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(msg.contains("999"), "message should contain actual total");
                assert!(
                    msg.contains("140000"),
                    "message should contain computed total"
                );
            }
            other => panic!("Expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn payment_zero_error_message_correct() {
        let mut payment = make_payment();
        payment.amount_cents = 0;
        let result = validate_create_payment(fake_create(), payment);
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert_eq!(msg, "Payment amount must be greater than 0");
            }
            other => panic!("Expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn payment_reference_error_message_correct() {
        let mut payment = make_payment();
        payment.reference = Some("x".repeat(257));
        let result = validate_create_payment(fake_create(), payment);
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert_eq!(msg, "Payment reference must be at most 256 characters");
            }
            other => panic!("Expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn fund_name_error_message_correct() {
        let mut fund = make_fund();
        fund.name = String::new();
        let result = validate_create_fund(fake_create(), fund);
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert_eq!(msg, "Fund name cannot be empty");
            }
            other => panic!("Expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn fund_target_error_message_correct() {
        let mut fund = make_fund();
        fund.target_cents = 0;
        let result = validate_create_fund(fake_create(), fund);
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert_eq!(msg, "Fund target must be greater than 0");
            }
            other => panic!("Expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn budget_year_error_message_correct() {
        let mut budget = make_budget();
        budget.fiscal_year = 0;
        let result = validate_create_budget(fake_create(), budget);
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert_eq!(msg, "Fiscal year must be between 2020 and 2100");
            }
            other => panic!("Expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn budget_no_categories_error_message_correct() {
        let mut budget = make_budget();
        budget.categories = vec![];
        let result = validate_create_budget(fake_create(), budget);
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert_eq!(msg, "Budget must have at least one category");
            }
            other => panic!("Expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn budget_empty_category_name_error_message_correct() {
        let budget = Budget {
            fiscal_year: 2025,
            income_projected_cents: 500_000,
            expenses_projected_cents: 0,
            categories: vec![BudgetCategory {
                name: "".into(),
                allocated_cents: 0,
                spent_cents: 0,
            }],
            approved: false,
            approved_at: None,
        };
        let result = validate_create_budget(fake_create(), budget);
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert_eq!(msg, "Budget category name cannot be empty");
            }
            other => panic!("Expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn budget_allocation_error_message_contains_values() {
        let mut budget = make_budget();
        budget.expenses_projected_cents = 1_000_000;
        let result = validate_create_budget(fake_create(), budget);
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(
                    msg.contains("900000"),
                    "message should contain allocated total"
                );
                assert!(
                    msg.contains("1000000"),
                    "message should contain projected expenses"
                );
            }
            other => panic!("Expected Invalid, got {:?}", other),
        }
    }

    // ========================================================================
    // VALIDATION PRIORITY / ORDER TESTS
    // ========================================================================

    #[test]
    fn charge_month_checked_before_total() {
        let mut charge = make_charge();
        charge.period_month = 0; // Invalid month
        charge.total_cents = 999; // Also wrong total
        let result = validate_create_charge(fake_create(), charge);
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(
                    msg.contains("Month"),
                    "month check should come before total check"
                );
            }
            other => panic!("Expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn charge_year_checked_before_total() {
        let mut charge = make_charge();
        charge.period_year = 9999; // Invalid year
        charge.total_cents = 999; // Also wrong total
        let result = validate_create_charge(fake_create(), charge);
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(
                    msg.contains("Year"),
                    "year check should come before total check"
                );
            }
            other => panic!("Expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn budget_year_checked_before_categories() {
        let mut budget = make_budget();
        budget.fiscal_year = 0; // Invalid year
        budget.categories = vec![]; // Also empty categories
        let result = validate_create_budget(fake_create(), budget);
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(
                    msg.contains("Fiscal year"),
                    "year check should come before categories check"
                );
            }
            other => panic!("Expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn budget_empty_categories_checked_before_names() {
        let mut budget = make_budget();
        budget.categories = vec![];
        let result = validate_create_budget(fake_create(), budget);
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(
                    msg.contains("at least one"),
                    "empty categories check should come before name check"
                );
            }
            other => panic!("Expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn budget_category_name_checked_before_allocation_sum() {
        let budget = Budget {
            fiscal_year: 2025,
            income_projected_cents: 500_000,
            expenses_projected_cents: 999_999, // Mismatched
            categories: vec![BudgetCategory {
                name: "".into(), // Empty name
                allocated_cents: 300_000,
                spent_cents: 0,
            }],
            approved: false,
            approved_at: None,
        };
        let result = validate_create_budget(fake_create(), budget);
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(
                    msg.contains("name cannot be empty"),
                    "name check should come before allocation sum check"
                );
            }
            other => panic!("Expected Invalid, got {:?}", other),
        }
    }

    #[test]
    fn payment_amount_checked_before_reference() {
        let mut payment = make_payment();
        payment.amount_cents = 0; // Invalid amount
        payment.reference = Some("x".repeat(999)); // Also too long
        let result = validate_create_payment(fake_create(), payment);
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(
                    msg.contains("amount"),
                    "amount check should come before reference check"
                );
            }
            other => panic!("Expected Invalid, got {:?}", other),
        }
    }

    // ========================================================================
    // CLONE & EQUALITY TESTS
    // ========================================================================

    #[test]
    fn monthly_charge_clone_eq() {
        let charge = make_charge();
        let cloned = charge.clone();
        assert_eq!(charge, cloned);
    }

    #[test]
    fn payment_clone_eq() {
        let payment = make_payment();
        let cloned = payment.clone();
        assert_eq!(payment, cloned);
    }

    #[test]
    fn reserve_fund_clone_eq() {
        let fund = make_fund();
        let cloned = fund.clone();
        assert_eq!(fund, cloned);
    }

    #[test]
    fn budget_clone_eq() {
        let budget = make_budget();
        let cloned = budget.clone();
        assert_eq!(budget, cloned);
    }

    #[test]
    fn anchor_clone_eq() {
        let anchor = Anchor("test_anchor".into());
        let cloned = anchor.clone();
        assert_eq!(anchor, cloned);
    }

    #[test]
    fn payment_method_clone_eq() {
        let method = PaymentMethod::TimeBankCredit;
        let cloned = method.clone();
        assert_eq!(method, cloned);
    }

    #[test]
    fn fund_type_clone_eq() {
        let ft = FundType::EmergencyFund;
        let cloned = ft.clone();
        assert_eq!(ft, cloned);
    }

    #[test]
    fn budget_category_clone_eq() {
        let cat = BudgetCategory {
            name: "Test".into(),
            allocated_cents: 100,
            spent_cents: 50,
        };
        let cloned = cat.clone();
        assert_eq!(cat, cloned);
    }

    // ========================================================================
    // UPDATE CHARGE VALIDATION TESTS
    // ========================================================================

    #[test]
    fn update_charge_valid() {
        let result = validate_update_charge(make_charge());
        assert!(is_valid(&result));
    }

    #[test]
    fn update_charge_month_zero_rejected() {
        let mut charge = make_charge();
        charge.period_month = 0;
        let result = validate_update_charge(charge);
        assert!(is_invalid(&result));
    }

    #[test]
    fn update_charge_month_13_rejected() {
        let mut charge = make_charge();
        charge.period_month = 13;
        let result = validate_update_charge(charge);
        assert!(is_invalid(&result));
    }

    #[test]
    fn update_charge_year_2019_rejected() {
        let mut charge = make_charge();
        charge.period_year = 2019;
        let result = validate_update_charge(charge);
        assert!(is_invalid(&result));
    }

    #[test]
    fn update_charge_year_2101_rejected() {
        let mut charge = make_charge();
        charge.period_year = 2101;
        let result = validate_update_charge(charge);
        assert!(is_invalid(&result));
    }

    #[test]
    fn update_charge_total_mismatch_rejected() {
        let mut charge = make_charge();
        charge.total_cents = 999_999;
        let result = validate_update_charge(charge);
        assert!(is_invalid(&result));
    }

    #[test]
    fn update_charge_all_zeros_valid() {
        let mut charge = make_charge();
        charge.base_rent_cents = 0;
        charge.maintenance_fee_cents = 0;
        charge.utilities_cents = 0;
        charge.reserve_contribution_cents = 0;
        charge.total_cents = 0;
        let result = validate_update_charge(charge);
        assert!(is_valid(&result));
    }

    #[test]
    fn update_charge_month_checked_before_total() {
        let mut charge = make_charge();
        charge.period_month = 0;
        charge.total_cents = 999;
        let result = validate_update_charge(charge);
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => {
                assert!(msg.contains("Month"));
            }
            other => panic!("Expected Invalid, got {:?}", other),
        }
    }

    // ========================================================================
    // UPDATE FUND VALIDATION TESTS
    // ========================================================================

    #[test]
    fn update_fund_valid() {
        let result = validate_update_fund(make_fund());
        assert!(is_valid(&result));
    }

    #[test]
    fn update_fund_empty_name_rejected() {
        let mut fund = make_fund();
        fund.name = "".into();
        let result = validate_update_fund(fund);
        assert!(is_invalid(&result));
    }

    #[test]
    fn update_fund_whitespace_name_rejected() {
        let mut fund = make_fund();
        fund.name = "   ".into();
        let result = validate_update_fund(fund);
        assert!(is_invalid(&result));
    }

    #[test]
    fn update_fund_zero_target_rejected() {
        let mut fund = make_fund();
        fund.target_cents = 0;
        let result = validate_update_fund(fund);
        assert!(is_invalid(&result));
    }

    #[test]
    fn update_fund_one_cent_target_accepted() {
        let mut fund = make_fund();
        fund.target_cents = 1;
        let result = validate_update_fund(fund);
        assert!(is_valid(&result));
    }

    #[test]
    fn update_fund_zero_balance_accepted() {
        let mut fund = make_fund();
        fund.balance_cents = 0;
        let result = validate_update_fund(fund);
        assert!(is_valid(&result));
    }

    // ============================================================================
    // Link Tag Validation Tests
    // ============================================================================

    fn validate_create_link_tag(
        link_type: LinkTypes,
        tag_bytes: Vec<u8>,
    ) -> ExternResult<ValidateCallbackResult> {
        let tag = LinkTag(tag_bytes);
        match link_type {
            LinkTypes::MemberToCharge
            | LinkTypes::ChargeToPayment
            | LinkTypes::MemberToPayment
            | LinkTypes::AllReserveFunds
            | LinkTypes::YearToBudget => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(format!(
                        "{:?} link tag too long (max 256 bytes)",
                        link_type
                    )));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::PeriodToCharge => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "PeriodToCharge link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        }
    }

    #[test]
    fn link_tag_member_to_charge_at_limit() {
        let result = validate_create_link_tag(LinkTypes::MemberToCharge, vec![0u8; 256]).unwrap();
        assert!(is_valid(&Ok(result)));
    }

    #[test]
    fn link_tag_member_to_charge_over_limit() {
        let result = validate_create_link_tag(LinkTypes::MemberToCharge, vec![0u8; 257]).unwrap();
        assert!(is_invalid(&Ok(result)));
    }

    #[test]
    fn link_tag_period_to_charge_at_limit() {
        let result = validate_create_link_tag(LinkTypes::PeriodToCharge, vec![0u8; 512]).unwrap();
        assert!(is_valid(&Ok(result)));
    }

    #[test]
    fn link_tag_period_to_charge_over_limit() {
        let result = validate_create_link_tag(LinkTypes::PeriodToCharge, vec![0u8; 513]).unwrap();
        assert!(is_invalid(&Ok(result)));
    }

    #[test]
    fn link_tag_year_to_budget_at_limit() {
        let result = validate_create_link_tag(LinkTypes::YearToBudget, vec![0u8; 256]).unwrap();
        assert!(is_valid(&Ok(result)));
    }

    #[test]
    fn link_tag_year_to_budget_over_limit() {
        let result = validate_create_link_tag(LinkTypes::YearToBudget, vec![0u8; 257]).unwrap();
        assert!(is_invalid(&Ok(result)));
    }

    #[test]
    fn link_tag_empty_tag_valid() {
        let result = validate_create_link_tag(LinkTypes::MemberToCharge, vec![]).unwrap();
        assert!(is_valid(&Ok(result)));
    }

    // ========================================================================
    // STRING LENGTH BOUNDARY TESTS
    // ========================================================================

    #[test]
    fn fund_name_at_limit_accepted() {
        let mut fund = make_fund();
        fund.name = "x".repeat(256);
        let result = validate_create_fund(fake_create(), fund);
        assert!(is_valid(&result));
    }

    #[test]
    fn fund_name_over_limit_rejected() {
        let mut fund = make_fund();
        fund.name = "x".repeat(257);
        let result = validate_create_fund(fake_create(), fund);
        assert!(is_invalid(&result));
    }

    #[test]
    fn fund_description_at_limit_accepted() {
        let mut fund = make_fund();
        fund.description = "x".repeat(4096);
        let result = validate_create_fund(fake_create(), fund);
        assert!(is_valid(&result));
    }

    #[test]
    fn fund_description_over_limit_rejected() {
        let mut fund = make_fund();
        fund.description = "x".repeat(4097);
        let result = validate_create_fund(fake_create(), fund);
        assert!(is_invalid(&result));
    }

    #[test]
    fn update_fund_name_at_limit_accepted() {
        let mut fund = make_fund();
        fund.name = "x".repeat(256);
        let result = validate_update_fund(fund);
        assert!(is_valid(&result));
    }

    #[test]
    fn update_fund_name_over_limit_rejected() {
        let mut fund = make_fund();
        fund.name = "x".repeat(257);
        let result = validate_update_fund(fund);
        assert!(is_invalid(&result));
    }

    #[test]
    fn update_fund_description_at_limit_accepted() {
        let mut fund = make_fund();
        fund.description = "x".repeat(4096);
        let result = validate_update_fund(fund);
        assert!(is_valid(&result));
    }

    #[test]
    fn update_fund_description_over_limit_rejected() {
        let mut fund = make_fund();
        fund.description = "x".repeat(4097);
        let result = validate_update_fund(fund);
        assert!(is_invalid(&result));
    }

    #[test]
    fn budget_category_name_at_limit_accepted() {
        let budget = Budget {
            fiscal_year: 2025,
            income_projected_cents: 100_000,
            expenses_projected_cents: 100_000,
            categories: vec![BudgetCategory {
                name: "x".repeat(256),
                allocated_cents: 100_000,
                spent_cents: 0,
            }],
            approved: false,
            approved_at: None,
        };
        let result = validate_create_budget(fake_create(), budget);
        assert!(is_valid(&result));
    }

    #[test]
    fn budget_category_name_over_limit_rejected() {
        let budget = Budget {
            fiscal_year: 2025,
            income_projected_cents: 100_000,
            expenses_projected_cents: 100_000,
            categories: vec![BudgetCategory {
                name: "x".repeat(257),
                allocated_cents: 100_000,
                spent_cents: 0,
            }],
            approved: false,
            approved_at: None,
        };
        let result = validate_create_budget(fake_create(), budget);
        assert!(is_invalid(&result));
    }
}
