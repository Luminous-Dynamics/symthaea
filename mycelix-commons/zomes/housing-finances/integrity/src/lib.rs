//! Finances Integrity Zome
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
}
