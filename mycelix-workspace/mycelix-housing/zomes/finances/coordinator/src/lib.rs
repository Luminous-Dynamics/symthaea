// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Finances Coordinator Zome
//! Business logic for charges, payments, reserves, and budgets.

use finances_integrity::*;
use hdk::prelude::*;

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

        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }

    Ok(records)
}

/// Record a payment
#[hdk_extern]
pub fn record_payment(payment: Payment) -> ExternResult<Record> {
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
        if let Some(record) = get(action_hash, GetOptions::default())? {
            payments.push(record);
        }
    }

    Ok(payments)
}

/// Create a reserve fund
#[hdk_extern]
pub fn create_reserve_fund(fund: ReserveFund) -> ExternResult<Record> {
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
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

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated fund".into()
    )))
}

/// Create an annual budget
#[hdk_extern]
pub fn create_budget(budget: Budget) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::Budget(budget.clone()))?;

    let year_anchor = format!("fiscal_year:{}", budget.fiscal_year);
    create_entry(&EntryTypes::Anchor(Anchor(year_anchor.clone())))?;
    create_link(
        anchor_hash(&year_anchor)?,
        action_hash.clone(),
        LinkTypes::YearToBudget,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
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

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
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

        if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
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
                if let Some(precord) = get(p_hash, GetOptions::default())? {
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
        if let Some(record) = get(action_hash, GetOptions::default())? {
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

    let outstanding_cents = if total_charges_cents > total_payments_cents {
        total_charges_cents - total_payments_cents
    } else {
        0
    };

    Ok(FinancialSummary {
        total_charges_cents,
        total_payments_cents,
        outstanding_cents,
        reserve_funds,
    })
}
