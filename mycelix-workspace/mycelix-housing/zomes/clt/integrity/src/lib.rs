// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Community Land Trust Integrity Zome
//! Entry types and validation for land trusts, ground leases, resale formulas,
//! and affordability reporting.

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// A Community Land Trust
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct LandTrust {
    pub id: String,
    pub name: String,
    pub mission: String,
    pub boundary: Vec<(f64, f64)>,
    pub charter_hash: Option<ActionHash>,
    pub stewardship_board: Vec<AgentPubKey>,
    pub affordability_target_ami_percent: u8,
    pub created_at: Timestamp,
}

/// The type of resale formula
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum FormulaType {
    AppreciationCap,
    AreaMedianIncome,
    ConsumerPriceIndex,
    Hybrid,
}

/// Resale formula configuration for a ground lease
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ResaleFormula {
    pub formula_type: FormulaType,
    pub max_appreciation_percent_annual: Option<u8>,
    pub ami_cap_percent: Option<u8>,
    pub improvement_credit_percent: Option<u8>,
}

/// A ground lease linking a unit to the land trust
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GroundLease {
    pub trust_hash: ActionHash,
    pub unit_hash: ActionHash,
    pub leaseholder: AgentPubKey,
    pub lease_term_years: u32,
    pub ground_rent_monthly_cents: u64,
    pub resale_formula: ResaleFormula,
    pub started_at: Timestamp,
    pub expires_at: Timestamp,
}

/// A resale price calculation
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ResaleCalculation {
    pub lease_hash: ActionHash,
    pub original_price_cents: u64,
    pub years_held: u32,
    pub improvements_value_cents: u64,
    pub calculated_max_price_cents: u64,
    pub ami_at_purchase: Option<u64>,
    pub current_ami: Option<u64>,
}

/// An affordability report for a trust
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AffordabilityReport {
    pub trust_hash: ActionHash,
    pub report_date: Timestamp,
    pub total_units: u32,
    pub affordable_units: u32,
    pub average_monthly_cost_cents: u64,
    pub median_area_income_cents: u64,
    pub affordability_ratio: f32,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    LandTrust(LandTrust),
    GroundLease(GroundLease),
    ResaleCalculation(ResaleCalculation),
    AffordabilityReport(AffordabilityReport),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// All trusts anchor
    AllTrusts,
    /// Trust to its ground leases
    TrustToLease,
    /// Leaseholder to their leases
    LeaseholderToLease,
    /// Lease to resale calculations
    LeaseToResaleCalc,
    /// Trust to affordability reports
    TrustToReport,
    /// Unit to lease
    UnitToLease,
}

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::LandTrust(trust) => validate_create_trust(action, trust),
                EntryTypes::GroundLease(lease) => validate_create_lease(action, lease),
                EntryTypes::ResaleCalculation(calc) => validate_resale_calc(action, calc),
                EntryTypes::AffordabilityReport(report) => {
                    validate_affordability_report(action, report)
                }
            },
            OpEntry::UpdateEntry {
                app_entry,
                action: _,
                original_action_hash: _,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::LandTrust(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::GroundLease(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::ResaleCalculation(_) => Ok(ValidateCallbackResult::Invalid(
                    "Resale calculations are immutable records".into(),
                )),
                EntryTypes::AffordabilityReport(_) => Ok(ValidateCallbackResult::Invalid(
                    "Affordability reports are immutable records".into(),
                )),
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
            LinkTypes::AllTrusts => Ok(ValidateCallbackResult::Valid),
            LinkTypes::TrustToLease => Ok(ValidateCallbackResult::Valid),
            LinkTypes::LeaseholderToLease => Ok(ValidateCallbackResult::Valid),
            LinkTypes::LeaseToResaleCalc => Ok(ValidateCallbackResult::Valid),
            LinkTypes::TrustToReport => Ok(ValidateCallbackResult::Valid),
            LinkTypes::UnitToLease => Ok(ValidateCallbackResult::Valid),
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

fn validate_create_trust(
    _action: Create,
    trust: LandTrust,
) -> ExternResult<ValidateCallbackResult> {
    if trust.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust ID cannot be empty".into(),
        ));
    }
    if trust.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust name cannot be empty".into(),
        ));
    }
    if trust.mission.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust mission cannot be empty".into(),
        ));
    }
    if trust.boundary.len() < 3 {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust boundary must have at least 3 coordinate points".into(),
        ));
    }
    for (lat, lon) in &trust.boundary {
        if *lat < -90.0 || *lat > 90.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Boundary latitude must be between -90 and 90".into(),
            ));
        }
        if *lon < -180.0 || *lon > 180.0 {
            return Ok(ValidateCallbackResult::Invalid(
                "Boundary longitude must be between -180 and 180".into(),
            ));
        }
    }
    if trust.stewardship_board.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Trust must have at least one stewardship board member".into(),
        ));
    }
    if trust.affordability_target_ami_percent == 0 || trust.affordability_target_ami_percent > 200 {
        return Ok(ValidateCallbackResult::Invalid(
            "Affordability target must be between 1 and 200 percent of AMI".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_lease(
    _action: Create,
    lease: GroundLease,
) -> ExternResult<ValidateCallbackResult> {
    if lease.lease_term_years == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Lease term must be at least 1 year".into(),
        ));
    }
    if lease.lease_term_years > 199 {
        return Ok(ValidateCallbackResult::Invalid(
            "Lease term cannot exceed 199 years".into(),
        ));
    }
    if lease.expires_at <= lease.started_at {
        return Ok(ValidateCallbackResult::Invalid(
            "Lease expiration must be after start date".into(),
        ));
    }
    // Validate resale formula consistency
    match lease.resale_formula.formula_type {
        FormulaType::AppreciationCap => {
            if lease
                .resale_formula
                .max_appreciation_percent_annual
                .is_none()
            {
                return Ok(ValidateCallbackResult::Invalid(
                    "AppreciationCap formula requires max_appreciation_percent_annual".into(),
                ));
            }
        }
        FormulaType::AreaMedianIncome => {
            if lease.resale_formula.ami_cap_percent.is_none() {
                return Ok(ValidateCallbackResult::Invalid(
                    "AreaMedianIncome formula requires ami_cap_percent".into(),
                ));
            }
        }
        FormulaType::ConsumerPriceIndex => {}
        FormulaType::Hybrid => {
            if lease
                .resale_formula
                .max_appreciation_percent_annual
                .is_none()
                && lease.resale_formula.ami_cap_percent.is_none()
            {
                return Ok(ValidateCallbackResult::Invalid(
                    "Hybrid formula requires at least one cap parameter".into(),
                ));
            }
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_resale_calc(
    _action: Create,
    calc: ResaleCalculation,
) -> ExternResult<ValidateCallbackResult> {
    if calc.original_price_cents == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Original price must be greater than 0".into(),
        ));
    }
    if calc.calculated_max_price_cents == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Calculated max price must be greater than 0".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_affordability_report(
    _action: Create,
    report: AffordabilityReport,
) -> ExternResult<ValidateCallbackResult> {
    if report.total_units == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Total units must be greater than 0".into(),
        ));
    }
    if report.affordable_units > report.total_units {
        return Ok(ValidateCallbackResult::Invalid(
            "Affordable units cannot exceed total units".into(),
        ));
    }
    if report.affordability_ratio < 0.0 || report.affordability_ratio > 1.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Affordability ratio must be between 0 and 1".into(),
        ));
    }
    if report.median_area_income_cents == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Median area income must be greater than 0".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
