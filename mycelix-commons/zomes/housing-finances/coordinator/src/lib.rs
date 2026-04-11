// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Finances Coordinator Zome
//! Business logic for charges, payments, reserves, and budgets.

use hdk::prelude::*;
use housing_finances_integrity::*;
use mycelix_bridge_common::{
    civic_requirement_basic,
    civic_requirement_constitutional,
    civic_requirement_proposal,
    civic_requirement_voting,
};
use mycelix_zome_helpers::get_latest_record;


fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GenerateChargesInput {
    pub members: Vec<MemberChargeInfo>,
    pub period_year: u16,
    pub period_month: u8,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MemberChargeInfo {
    pub member: AgentPubKey,
    pub unit_hash: ActionHash,
    pub base_rent_cents: u64,
    pub maintenance_fee_cents: u64,
    pub utilities_cents: u64,
    pub reserve_contribution_cents: u64,
}

/// Generate monthly charges for all specified members

#[hdk_extern]
pub fn generate_monthly_charges(input: GenerateChargesInput) -> ExternResult<Vec<Record>> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", 
        &civic_requirement_constitutional(),
        "generate_monthly_charges",
    )?;
    let period_anchor = format!("period:{}:{:02}", input.period_year, input.period_month);
    create_entry(&EntryTypes::Anchor(Anchor(period_anchor.clone())))?;

    let mut records = Vec::new();

    for info in input.members {
        let total_cents = info.base_rent_cents
            + info.maintenance_fee_cents
            + info.utilities_cents
            + info.reserve_contribution_cents;

        let charge = MonthlyCharge {
            member: info.member.clone(),
            unit_hash: info.unit_hash,
            period_year: input.period_year,
            period_month: input.period_month,
            base_rent_cents: info.base_rent_cents,
            maintenance_fee_cents: info.maintenance_fee_cents,
            utilities_cents: info.utilities_cents,
            reserve_contribution_cents: info.reserve_contribution_cents,
            total_cents,
        };

        let action_hash = create_entry(&EntryTypes::MonthlyCharge(charge))?;

        // Link member to charge
        create_link(
            info.member,
            action_hash.clone(),
            LinkTypes::MemberToCharge,
            (),
        )?;

        // Link period to charge
        create_link(
            anchor_hash(&period_anchor)?,
            action_hash.clone(),
            LinkTypes::PeriodToCharge,
            (),
        )?;

        if let Some(record) = get_latest_record(action_hash)? {
            records.push(record);
        }
    }

    Ok(records)
}

/// Record a payment
#[hdk_extern]
pub fn record_payment(payment: Payment) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_basic(), "record_payment")?;
    let action_hash = create_entry(&EntryTypes::Payment(payment.clone()))?;

    // Link member to payment
    create_link(
        payment.member,
        action_hash.clone(),
        LinkTypes::MemberToPayment,
        (),
    )?;

    // Link charge to payment if applicable
    if let Some(charge_hash) = payment.charge_hash {
        create_link(
            charge_hash,
            action_hash.clone(),
            LinkTypes::ChargeToPayment,
            (),
        )?;
    }

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created payment".into()
    )))
}

/// Get all payments for a member
#[hdk_extern]
pub fn get_member_payments(member: AgentPubKey) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(member, LinkTypes::MemberToPayment)?,
        GetStrategy::default(),
    )?;

    let mut payments = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
            payments.push(record);
        }
    }

    Ok(payments)
}

/// Create a reserve fund
#[hdk_extern]
pub fn create_reserve_fund(fund: ReserveFund) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_voting(), "create_reserve_fund")?;
    if fund.name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Fund name must be at most 256 characters".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::ReserveFund(fund))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_reserve_funds".to_string())))?;
    create_link(
        anchor_hash("all_reserve_funds")?,
        action_hash.clone(),
        LinkTypes::AllReserveFunds,
        (),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created fund".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DepositToReserveInput {
    pub fund_hash: ActionHash,
    pub amount_cents: u64,
}

/// Deposit funds into a reserve
#[hdk_extern]
pub fn deposit_to_reserve(input: DepositToReserveInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_proposal(), "deposit_to_reserve")?;
    if input.amount_cents == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Deposit amount must be greater than 0".into()
        )));
    }

    let record = get(input.fund_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Reserve fund not found".into())
    ))?;

    let mut fund: ReserveFund = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid fund entry".into()
        )))?;

    fund.balance_cents += input.amount_cents;

    let new_hash = update_entry(input.fund_hash, &EntryTypes::ReserveFund(fund))?;

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated fund".into()
    )))
}

/// Create an annual budget
#[hdk_extern]
pub fn create_budget(budget: Budget) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_voting(), "create_budget")?;
    let action_hash = create_entry(&EntryTypes::Budget(budget.clone()))?;

    let year_anchor = format!("fiscal_year:{}", budget.fiscal_year);
    create_entry(&EntryTypes::Anchor(Anchor(year_anchor.clone())))?;
    create_link(
        anchor_hash(&year_anchor)?,
        action_hash.clone(),
        LinkTypes::YearToBudget,
        (),
    )?;

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created budget".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ApproveBudgetInput {
    pub budget_hash: ActionHash,
}

/// Approve a budget
#[hdk_extern]
pub fn approve_budget(input: ApproveBudgetInput) -> ExternResult<Record> {
    let _eligibility = mycelix_zome_helpers::require_civic("commons_bridge", &civic_requirement_constitutional(), "approve_budget")?;
    let record = get(input.budget_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Budget not found".into())
    ))?;

    let mut budget: Budget = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid budget entry".into()
        )))?;

    if budget.approved {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Budget is already approved".into()
        )));
    }

    let now = sys_time()?;
    budget.approved = true;
    budget.approved_at = Some(now);

    let new_hash = update_entry(input.budget_hash, &EntryTypes::Budget(budget))?;

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated budget".into()
    )))
}

/// Financial summary response
#[derive(Serialize, Deserialize, Debug)]
pub struct FinancialSummary {
    pub total_charges_cents: u64,
    pub total_payments_cents: u64,
    pub outstanding_cents: u64,
    pub reserve_funds: Vec<ReserveFundSummary>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ReserveFundSummary {
    pub name: String,
    pub fund_type: FundType,
    pub balance_cents: u64,
    pub target_cents: u64,
    pub percent_funded: f32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FinancialSummaryInput {
    pub period_year: u16,
    pub period_month: u8,
}

/// Get a financial summary for a given period
#[hdk_extern]
pub fn get_financial_summary(input: FinancialSummaryInput) -> ExternResult<FinancialSummary> {
    let period_anchor = format!("period:{}:{:02}", input.period_year, input.period_month);

    // Get all charges for the period
    let charge_links = get_links(
        LinkQuery::try_new(anchor_hash(&period_anchor)?, LinkTypes::PeriodToCharge)?,
        GetStrategy::default(),
    )?;

    let mut total_charges_cents: u64 = 0;
    let mut total_payments_cents: u64 = 0;

    for link in charge_links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get_latest_record(action_hash.clone())? {
            if let Some(charge) = record
                .entry()
                .to_app_option::<MonthlyCharge>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                total_charges_cents += charge.total_cents;
            }

            // Get payments for this charge
            let payment_links = get_links(
                LinkQuery::try_new(action_hash, LinkTypes::ChargeToPayment)?,
                GetStrategy::default(),
            )?;

            for plink in payment_links {
                let p_hash = ActionHash::try_from(plink.target).map_err(|_| {
                    wasm_error!(WasmErrorInner::Guest("Invalid link target".into()))
                })?;
                if let Some(precord) = get_latest_record(p_hash)? {
                    if let Some(payment) = precord
                        .entry()
                        .to_app_option::<Payment>()
                        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                    {
                        total_payments_cents += payment.amount_cents;
                    }
                }
            }
        }
    }

    // Get reserve funds
    let fund_links = get_links(
        LinkQuery::try_new(
            anchor_hash("all_reserve_funds")?,
            LinkTypes::AllReserveFunds,
        )?,
        GetStrategy::default(),
    )?;

    let mut reserve_funds = Vec::new();
    for link in fund_links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
            if let Some(fund) = record
                .entry()
                .to_app_option::<ReserveFund>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                let percent_funded = if fund.target_cents > 0 {
                    (fund.balance_cents as f32 / fund.target_cents as f32) * 100.0
                } else {
                    0.0
                };
                reserve_funds.push(ReserveFundSummary {
                    name: fund.name,
                    fund_type: fund.fund_type,
                    balance_cents: fund.balance_cents,
                    target_cents: fund.target_cents,
                    percent_funded,
                });
            }
        }
    }

    let outstanding_cents = total_charges_cents.saturating_sub(total_payments_cents);

    Ok(FinancialSummary {
        total_charges_cents,
        total_payments_cents,
        outstanding_cents,
        reserve_funds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    // ── Coordinator struct serde roundtrips ────────────────────────────

    #[test]
    fn member_charge_info_serde_roundtrip() {
        let info = MemberChargeInfo {
            member: fake_agent(),
            unit_hash: fake_action_hash(),
            base_rent_cents: 100000,
            maintenance_fee_cents: 20000,
            utilities_cents: 15000,
            reserve_contribution_cents: 5000,
        };
        let json = serde_json::to_string(&info).unwrap();
        let decoded: MemberChargeInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.base_rent_cents, 100000);
        assert_eq!(decoded.maintenance_fee_cents, 20000);
        assert_eq!(decoded.utilities_cents, 15000);
        assert_eq!(decoded.reserve_contribution_cents, 5000);
    }

    #[test]
    fn generate_charges_input_serde_roundtrip() {
        let input = GenerateChargesInput {
            members: vec![MemberChargeInfo {
                member: fake_agent(),
                unit_hash: fake_action_hash(),
                base_rent_cents: 100000,
                maintenance_fee_cents: 20000,
                utilities_cents: 15000,
                reserve_contribution_cents: 5000,
            }],
            period_year: 2025,
            period_month: 6,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: GenerateChargesInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.period_year, 2025);
        assert_eq!(decoded.period_month, 6);
        assert_eq!(decoded.members.len(), 1);
    }

    #[test]
    fn generate_charges_input_empty_members_serde() {
        let input = GenerateChargesInput {
            members: vec![],
            period_year: 2026,
            period_month: 1,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: GenerateChargesInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.members.is_empty());
    }

    #[test]
    fn deposit_to_reserve_input_serde_roundtrip() {
        let input = DepositToReserveInput {
            fund_hash: fake_action_hash(),
            amount_cents: 50000,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: DepositToReserveInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.amount_cents, 50000);
    }

    #[test]
    fn approve_budget_input_serde_roundtrip() {
        let input = ApproveBudgetInput {
            budget_hash: fake_action_hash(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: ApproveBudgetInput = serde_json::from_str(&json).unwrap();
        // Verify it deserialized without error (ActionHash equality via roundtrip)
        let json2 = serde_json::to_string(&decoded).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn financial_summary_input_serde_roundtrip() {
        let input = FinancialSummaryInput {
            period_year: 2025,
            period_month: 12,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: FinancialSummaryInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.period_year, 2025);
        assert_eq!(decoded.period_month, 12);
    }

    #[test]
    fn financial_summary_serde_roundtrip() {
        let summary = FinancialSummary {
            total_charges_cents: 500000,
            total_payments_cents: 350000,
            outstanding_cents: 150000,
            reserve_funds: vec![
                ReserveFundSummary {
                    name: "Capital Reserve".to_string(),
                    fund_type: FundType::CapitalReserve,
                    balance_cents: 200000,
                    target_cents: 500000,
                    percent_funded: 40.0,
                },
                ReserveFundSummary {
                    name: "Emergency Fund".to_string(),
                    fund_type: FundType::EmergencyFund,
                    balance_cents: 100000,
                    target_cents: 100000,
                    percent_funded: 100.0,
                },
            ],
        };
        let json = serde_json::to_string(&summary).unwrap();
        let decoded: FinancialSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.total_charges_cents, 500000);
        assert_eq!(decoded.total_payments_cents, 350000);
        assert_eq!(decoded.outstanding_cents, 150000);
        assert_eq!(decoded.reserve_funds.len(), 2);
    }

    #[test]
    fn financial_summary_empty_reserves_serde() {
        let summary = FinancialSummary {
            total_charges_cents: 0,
            total_payments_cents: 0,
            outstanding_cents: 0,
            reserve_funds: vec![],
        };
        let json = serde_json::to_string(&summary).unwrap();
        let decoded: FinancialSummary = serde_json::from_str(&json).unwrap();
        assert!(decoded.reserve_funds.is_empty());
    }

    #[test]
    fn reserve_fund_summary_serde_roundtrip() {
        let summary = ReserveFundSummary {
            name: "Improvement Fund".to_string(),
            fund_type: FundType::ImprovementFund,
            balance_cents: 75000,
            target_cents: 300000,
            percent_funded: 25.0,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let decoded: ReserveFundSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, "Improvement Fund");
        assert_eq!(decoded.fund_type, FundType::ImprovementFund);
        assert_eq!(decoded.balance_cents, 75000);
        assert!((decoded.percent_funded - 25.0).abs() < f32::EPSILON);
    }

    // ── Integrity enum serde roundtrips ────────────────────────────────

    #[test]
    fn payment_method_all_variants_serde() {
        let variants = vec![
            PaymentMethod::BankTransfer,
            PaymentMethod::MutualCredit,
            PaymentMethod::Cash,
            PaymentMethod::Check,
            PaymentMethod::TimeBankCredit,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let decoded: PaymentMethod = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, v);
        }
    }

    #[test]
    fn fund_type_all_variants_serde() {
        let variants = vec![
            FundType::CapitalReserve,
            FundType::OperatingReserve,
            FundType::EmergencyFund,
            FundType::ImprovementFund,
        ];
        for v in &variants {
            let json = serde_json::to_string(v).unwrap();
            let decoded: FundType = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, v);
        }
    }

    // ── Integrity entry struct serde roundtrips ────────────────────────

    #[test]
    fn monthly_charge_serde_roundtrip() {
        let charge = MonthlyCharge {
            member: fake_agent(),
            unit_hash: fake_action_hash(),
            period_year: 2025,
            period_month: 6,
            base_rent_cents: 100000,
            maintenance_fee_cents: 20000,
            utilities_cents: 15000,
            reserve_contribution_cents: 5000,
            total_cents: 140000,
        };
        let json = serde_json::to_string(&charge).unwrap();
        let decoded: MonthlyCharge = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, charge);
    }

    #[test]
    fn payment_serde_roundtrip() {
        let payment = Payment {
            member: fake_agent(),
            charge_hash: Some(fake_action_hash()),
            amount_cents: 140000,
            payment_method: PaymentMethod::BankTransfer,
            paid_at: Timestamp::from_micros(1000),
            reference: Some("TXN-001".to_string()),
        };
        let json = serde_json::to_string(&payment).unwrap();
        let decoded: Payment = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, payment);
    }

    #[test]
    fn reserve_fund_serde_roundtrip() {
        let fund = ReserveFund {
            name: "Capital Reserve".to_string(),
            fund_type: FundType::CapitalReserve,
            balance_cents: 200000,
            target_cents: 500000,
            description: "Long-term improvements".to_string(),
        };
        let json = serde_json::to_string(&fund).unwrap();
        let decoded: ReserveFund = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, fund);
    }

    #[test]
    fn budget_serde_roundtrip() {
        let budget = Budget {
            fiscal_year: 2025,
            income_projected_cents: 1000000,
            expenses_projected_cents: 900000,
            categories: vec![
                BudgetCategory {
                    name: "Maintenance".to_string(),
                    allocated_cents: 400000,
                    spent_cents: 150000,
                },
                BudgetCategory {
                    name: "Utilities".to_string(),
                    allocated_cents: 500000,
                    spent_cents: 300000,
                },
            ],
            approved: true,
            approved_at: Some(Timestamp::from_micros(2000)),
        };
        let json = serde_json::to_string(&budget).unwrap();
        let decoded: Budget = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, budget);
    }

    #[test]
    fn budget_category_serde_roundtrip() {
        let cat = BudgetCategory {
            name: "Insurance".to_string(),
            allocated_cents: 200000,
            spent_cents: 0,
        };
        let json = serde_json::to_string(&cat).unwrap();
        let decoded: BudgetCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, cat);
    }

    // ── Additional edge-case and boundary tests ─────────────────────────

    #[test]
    fn member_charge_info_all_zeros_roundtrip() {
        let info = MemberChargeInfo {
            member: fake_agent(),
            unit_hash: fake_action_hash(),
            base_rent_cents: 0,
            maintenance_fee_cents: 0,
            utilities_cents: 0,
            reserve_contribution_cents: 0,
        };
        let json = serde_json::to_string(&info).unwrap();
        let decoded: MemberChargeInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.base_rent_cents, 0);
        assert_eq!(decoded.maintenance_fee_cents, 0);
        assert_eq!(decoded.utilities_cents, 0);
        assert_eq!(decoded.reserve_contribution_cents, 0);
    }

    #[test]
    fn deposit_to_reserve_input_max_amount_roundtrip() {
        let input = DepositToReserveInput {
            fund_hash: fake_action_hash(),
            amount_cents: u64::MAX,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: DepositToReserveInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.amount_cents, u64::MAX);
    }

    #[test]
    fn financial_summary_input_boundary_month_values() {
        for month in [1u8, 6, 12] {
            let input = FinancialSummaryInput {
                period_year: 2026,
                period_month: month,
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: FinancialSummaryInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.period_month, month);
        }
    }

    #[test]
    fn payment_no_charge_no_reference_roundtrip() {
        let payment = Payment {
            member: fake_agent(),
            charge_hash: None,
            amount_cents: 1,
            payment_method: PaymentMethod::Cash,
            paid_at: Timestamp::from_micros(0),
            reference: None,
        };
        let json = serde_json::to_string(&payment).unwrap();
        let decoded: Payment = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.charge_hash, None);
        assert_eq!(decoded.reference, None);
        assert_eq!(decoded.amount_cents, 1);
    }

    #[test]
    fn budget_unapproved_no_timestamp_roundtrip() {
        let budget = Budget {
            fiscal_year: 2026,
            income_projected_cents: 0,
            expenses_projected_cents: 0,
            categories: vec![BudgetCategory {
                name: "Placeholder".to_string(),
                allocated_cents: 0,
                spent_cents: 0,
            }],
            approved: false,
            approved_at: None,
        };
        let json = serde_json::to_string(&budget).unwrap();
        let decoded: Budget = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.approved, false);
        assert_eq!(decoded.approved_at, None);
        assert_eq!(decoded.fiscal_year, 2026);
    }

    #[test]
    fn reserve_fund_summary_zero_target_zero_percent() {
        let summary = ReserveFundSummary {
            name: "Empty Fund".to_string(),
            fund_type: FundType::OperatingReserve,
            balance_cents: 0,
            target_cents: 0,
            percent_funded: 0.0,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let decoded: ReserveFundSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.balance_cents, 0);
        assert_eq!(decoded.target_cents, 0);
        assert!((decoded.percent_funded - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn monthly_charge_max_u64_total_roundtrip() {
        let charge = MonthlyCharge {
            member: fake_agent(),
            unit_hash: fake_action_hash(),
            period_year: 2025,
            period_month: 1,
            base_rent_cents: u64::MAX,
            maintenance_fee_cents: 0,
            utilities_cents: 0,
            reserve_contribution_cents: 0,
            total_cents: u64::MAX,
        };
        let json = serde_json::to_string(&charge).unwrap();
        let decoded: MonthlyCharge = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.base_rent_cents, u64::MAX);
        assert_eq!(decoded.total_cents, u64::MAX);
    }

    #[test]
    fn generate_charges_input_many_members_roundtrip() {
        let members: Vec<MemberChargeInfo> = (0..10)
            .map(|i| MemberChargeInfo {
                member: AgentPubKey::from_raw_36(vec![i as u8; 36]),
                unit_hash: fake_action_hash(),
                base_rent_cents: 100_000,
                maintenance_fee_cents: 20_000,
                utilities_cents: 15_000,
                reserve_contribution_cents: 5_000,
            })
            .collect();
        let input = GenerateChargesInput {
            members,
            period_year: 2026,
            period_month: 2,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: GenerateChargesInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.members.len(), 10);
    }
}
