// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Community Land Trust Coordinator Zome
//! Business logic for land trusts, ground leases, resale calculations,
//! and affordability reporting.

use clt_integrity::*;
use hdk::prelude::*;

fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Create a new community land trust
#[hdk_extern]
pub fn create_land_trust(trust: LandTrust) -> ExternResult<Record> {
    if trust.name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Trust name must be at most 256 characters".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::LandTrust(trust))?;

    create_entry(&EntryTypes::Anchor(Anchor("all_trusts".to_string())))?;
    create_link(
        anchor_hash("all_trusts")?,
        action_hash.clone(),
        LinkTypes::AllTrusts,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created trust".into()
    )))
}

/// Issue a ground lease for a unit under the trust
#[hdk_extern]
pub fn issue_ground_lease(lease: GroundLease) -> ExternResult<Record> {
    let action_hash = create_entry(&EntryTypes::GroundLease(lease.clone()))?;

    // Link trust to lease
    create_link(
        lease.trust_hash,
        action_hash.clone(),
        LinkTypes::TrustToLease,
        (),
    )?;

    // Link leaseholder to lease
    create_link(
        lease.leaseholder,
        action_hash.clone(),
        LinkTypes::LeaseholderToLease,
        (),
    )?;

    // Link unit to lease
    create_link(
        lease.unit_hash,
        action_hash.clone(),
        LinkTypes::UnitToLease,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created lease".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CalculateResaleInput {
    pub lease_hash: ActionHash,
    pub original_price_cents: u64,
    pub years_held: u32,
    pub improvements_value_cents: u64,
    pub ami_at_purchase: Option<u64>,
    pub current_ami: Option<u64>,
}

/// Calculate the maximum resale price under the ground lease formula.
///
/// Formula types:
/// - AppreciationCap: original * (1 + rate)^years + improvement_credit
/// - AreaMedianIncome: min(appreciated_value, ami_cap_percent * current_ami / 12 * affordability_factor)
/// - ConsumerPriceIndex: original * (1 + 0.03)^years + improvement_credit (3% assumed CPI)
/// - Hybrid: min(appreciation_cap_result, ami_result)
#[hdk_extern]
pub fn calculate_max_resale_price(input: CalculateResaleInput) -> ExternResult<Record> {
    let lease_record = get(input.lease_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Lease not found".into())))?;

    let lease: GroundLease = lease_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid lease entry".into()
        )))?;

    let formula = &lease.resale_formula;
    let improvement_credit_pct = formula.improvement_credit_percent.unwrap_or(0) as u64;
    let improvement_credit = input.improvements_value_cents * improvement_credit_pct / 100;

    let calculated_max_price_cents = match formula.formula_type {
        FormulaType::AppreciationCap => {
            let annual_rate = formula.max_appreciation_percent_annual.unwrap_or(2) as f64 / 100.0;
            let appreciated = input.original_price_cents as f64
                * (1.0 + annual_rate).powi(input.years_held as i32);
            appreciated as u64 + improvement_credit
        }
        FormulaType::AreaMedianIncome => {
            let ami_cap_pct = formula.ami_cap_percent.unwrap_or(80) as f64 / 100.0;
            let current_ami = input.current_ami.unwrap_or(60000_00) as f64;
            // Maximum price = AMI cap % * annual AMI * affordability factor (assume 3x annual income)
            let ami_based_max = (ami_cap_pct * current_ami * 3.0) as u64;

            // Also calculate simple appreciation as a floor
            let appreciated =
                input.original_price_cents as f64 * (1.02_f64).powi(input.years_held as i32);
            let appreciated_with_credit = appreciated as u64 + improvement_credit;

            // Take the lesser of AMI-based and appreciated value
            ami_based_max.min(appreciated_with_credit)
        }
        FormulaType::ConsumerPriceIndex => {
            // Use 3% as assumed CPI rate
            let appreciated =
                input.original_price_cents as f64 * (1.03_f64).powi(input.years_held as i32);
            appreciated as u64 + improvement_credit
        }
        FormulaType::Hybrid => {
            // Appreciation cap calculation
            let annual_rate = formula.max_appreciation_percent_annual.unwrap_or(2) as f64 / 100.0;
            let appreciation_cap = input.original_price_cents as f64
                * (1.0 + annual_rate).powi(input.years_held as i32);
            let appreciation_result = appreciation_cap as u64 + improvement_credit;

            // AMI-based calculation
            let ami_cap_pct = formula.ami_cap_percent.unwrap_or(80) as f64 / 100.0;
            let current_ami = input.current_ami.unwrap_or(60000_00) as f64;
            let ami_result = (ami_cap_pct * current_ami * 3.0) as u64;

            // Hybrid takes the minimum of both to ensure maximum affordability
            appreciation_result.min(ami_result)
        }
    };

    let calc = ResaleCalculation {
        lease_hash: input.lease_hash.clone(),
        original_price_cents: input.original_price_cents,
        years_held: input.years_held,
        improvements_value_cents: input.improvements_value_cents,
        calculated_max_price_cents,
        ami_at_purchase: input.ami_at_purchase,
        current_ami: input.current_ami,
    };

    let action_hash = create_entry(&EntryTypes::ResaleCalculation(calc))?;

    // Link lease to calculation
    create_link(
        input.lease_hash,
        action_hash.clone(),
        LinkTypes::LeaseToResaleCalc,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created resale calculation".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TransferLeaseInput {
    pub lease_hash: ActionHash,
    pub new_leaseholder: AgentPubKey,
}

/// Transfer a ground lease to a new leaseholder
#[hdk_extern]
pub fn transfer_lease(input: TransferLeaseInput) -> ExternResult<Record> {
    let record = get(input.lease_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Lease not found".into())))?;

    let mut lease: GroundLease = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid lease entry".into()
        )))?;

    let old_leaseholder = lease.leaseholder.clone();
    lease.leaseholder = input.new_leaseholder.clone();

    let new_hash = update_entry(input.lease_hash.clone(), &EntryTypes::GroundLease(lease))?;

    // Remove old leaseholder link
    let links = get_links(
        LinkQuery::try_new(old_leaseholder, LinkTypes::LeaseholderToLease)?,
        GetStrategy::default(),
    )?;
    for link in links {
        let target = ActionHash::try_from(link.target.clone());
        if let Ok(target_hash) = target {
            if target_hash == input.lease_hash {
                delete_link(link.create_link_hash, GetOptions::default())?;
            }
        }
    }

    // Add new leaseholder link
    create_link(
        input.new_leaseholder,
        new_hash.clone(),
        LinkTypes::LeaseholderToLease,
        (),
    )?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated lease".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GenerateAffordabilityInput {
    pub trust_hash: ActionHash,
    pub total_units: u32,
    pub affordable_units: u32,
    pub average_monthly_cost_cents: u64,
    pub median_area_income_cents: u64,
}

/// Generate an affordability report for a trust
#[hdk_extern]
pub fn generate_affordability_report(input: GenerateAffordabilityInput) -> ExternResult<Record> {
    let now = sys_time()?;

    // Affordability ratio = (average monthly cost * 12) / median annual income
    // A ratio <= 0.30 is considered affordable (30% rule)
    let annual_cost = input.average_monthly_cost_cents as f64 * 12.0;
    let affordability_ratio = if input.median_area_income_cents > 0 {
        (annual_cost / input.median_area_income_cents as f64) as f32
    } else {
        1.0
    };

    let report = AffordabilityReport {
        trust_hash: input.trust_hash.clone(),
        report_date: now,
        total_units: input.total_units,
        affordable_units: input.affordable_units,
        average_monthly_cost_cents: input.average_monthly_cost_cents,
        median_area_income_cents: input.median_area_income_cents,
        affordability_ratio,
    };

    let action_hash = create_entry(&EntryTypes::AffordabilityReport(report))?;

    create_link(
        input.trust_hash,
        action_hash.clone(),
        LinkTypes::TrustToReport,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created report".into()
    )))
}

/// Get all ground leases for a trust
#[hdk_extern]
pub fn get_trust_leases(trust_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(trust_hash, LinkTypes::TrustToLease)?,
        GetStrategy::default(),
    )?;

    let mut leases = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            leases.push(record);
        }
    }

    Ok(leases)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateTrustBoardInput {
    pub trust_hash: ActionHash,
    pub new_board: Vec<AgentPubKey>,
}

/// Update the stewardship board of a land trust
#[hdk_extern]
pub fn update_trust_board(input: UpdateTrustBoardInput) -> ExternResult<Record> {
    if input.new_board.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Board must have at least one member".into()
        )));
    }

    let record = get(input.trust_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Trust not found".into())))?;

    let mut trust: LandTrust = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid trust entry".into()
        )))?;

    trust.stewardship_board = input.new_board;

    let new_hash = update_entry(input.trust_hash, &EntryTypes::LandTrust(trust))?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated trust".into()
    )))
}
