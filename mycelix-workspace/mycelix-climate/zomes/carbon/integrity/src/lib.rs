// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Carbon Integrity Zome
//!
//! Defines entry types and validation for carbon footprint tracking and carbon credits.
//! Uses HDI 0.7.0-dev.1 with FlatOp validation pattern.

use hdi::prelude::*;

/// Anchor entry for creating deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq, Eq)]
pub struct Anchor(pub String);

/// Status of a carbon credit
#[hdk_entry_helper]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CreditStatus {
    /// Credit is active and can be transferred
    Active,
    /// Credit has been transferred to another owner
    Transferred,
    /// Credit has been retired (used for offsetting)
    Retired,
}

/// Carbon footprint measurement for an entity over a time period
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CarbonFootprint {
    /// DID of the entity being measured
    pub entity_did: String,
    /// Start of measurement period (Unix timestamp)
    pub period_start: i64,
    /// End of measurement period (Unix timestamp)
    pub period_end: i64,
    /// Scope 1 emissions in tonnes CO2e (direct emissions)
    pub scope1: f64,
    /// Scope 2 emissions in tonnes CO2e (indirect from purchased energy)
    pub scope2: f64,
    /// Scope 3 emissions in tonnes CO2e (other indirect emissions)
    pub scope3: f64,
    /// Methodology used for measurement (e.g., "GHG Protocol", "ISO 14064")
    pub methodology: String,
    /// DID of verifier (if verified)
    pub verified_by: Option<String>,
}

/// A tradeable carbon credit representing verified emission reductions
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CarbonCredit {
    /// Unique identifier for this credit
    pub id: String,
    /// ID of the climate project that generated this credit
    pub project_id: String,
    /// Year the emission reduction occurred
    pub vintage_year: u32,
    /// Amount of CO2 equivalent in tonnes
    pub tonnes_co2e: f64,
    /// Current status of the credit
    pub status: CreditStatus,
    /// DID of current owner
    pub owner_did: String,
    /// Timestamp when credit was retired (if retired)
    pub retired_at: Option<i64>,
}

/// Entry types for the carbon zome
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    #[entry_type(visibility = "public")]
    Anchor(Anchor),
    #[entry_type(visibility = "public")]
    CarbonFootprint(CarbonFootprint),
    #[entry_type(visibility = "public")]
    CarbonCredit(CarbonCredit),
}

/// Link types for the carbon zome
#[hdk_link_types]
pub enum LinkTypes {
    /// Anchor to footprints for an entity
    AnchorToFootprints,
    /// Anchor to credits by owner
    AnchorToCredits,
    /// Anchor to credits by project
    ProjectToCredits,
    /// Footprint updates chain
    FootprintUpdates,
    /// Credit transfer history
    CreditTransfers,
}

/// Validate DIDs have proper format
fn validate_did(did: &str) -> ExternResult<ValidateCallbackResult> {
    if did.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("DID cannot be empty".to_string()));
    }
    if !did.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "DID must start with 'did:' prefix".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate that emissions values are non-negative
fn validate_emissions(scope1: f64, scope2: f64, scope3: f64) -> ExternResult<ValidateCallbackResult> {
    if scope1 < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Scope 1 emissions cannot be negative".to_string(),
        ));
    }
    if scope2 < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Scope 2 emissions cannot be negative".to_string(),
        ));
    }
    if scope3 < 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Scope 3 emissions cannot be negative".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate a CarbonFootprint entry
fn validate_carbon_footprint(footprint: &CarbonFootprint) -> ExternResult<ValidateCallbackResult> {
    // Validate entity DID
    let did_result = validate_did(&footprint.entity_did)?;
    if let ValidateCallbackResult::Invalid(_) = did_result {
        return Ok(did_result);
    }

    // Validate verifier DID if present
    if let Some(ref verifier) = footprint.verified_by {
        let verifier_result = validate_did(verifier)?;
        if let ValidateCallbackResult::Invalid(_) = verifier_result {
            return Ok(verifier_result);
        }
    }

    // Validate emissions are non-negative
    let emissions_result = validate_emissions(footprint.scope1, footprint.scope2, footprint.scope3)?;
    if let ValidateCallbackResult::Invalid(_) = emissions_result {
        return Ok(emissions_result);
    }

    // Validate time period
    if footprint.period_start >= footprint.period_end {
        return Ok(ValidateCallbackResult::Invalid(
            "Period start must be before period end".to_string(),
        ));
    }

    // Validate methodology
    if footprint.methodology.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Methodology cannot be empty".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a CarbonCredit entry
fn validate_carbon_credit(credit: &CarbonCredit) -> ExternResult<ValidateCallbackResult> {
    // Validate credit ID
    if credit.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Credit ID cannot be empty".to_string(),
        ));
    }

    // Validate project ID
    if credit.project_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Project ID cannot be empty".to_string(),
        ));
    }

    // Validate owner DID
    let did_result = validate_did(&credit.owner_did)?;
    if let ValidateCallbackResult::Invalid(_) = did_result {
        return Ok(did_result);
    }

    // Validate tonnes are positive
    if credit.tonnes_co2e <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Credit tonnes must be positive".to_string(),
        ));
    }

    // Validate vintage year is reasonable (1990-2100)
    if credit.vintage_year < 1990 || credit.vintage_year > 2100 {
        return Ok(ValidateCallbackResult::Invalid(
            "Vintage year must be between 1990 and 2100".to_string(),
        ));
    }

    // Validate retired_at matches status
    match credit.status {
        CreditStatus::Retired => {
            if credit.retired_at.is_none() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Retired credits must have retired_at timestamp".to_string(),
                ));
            }
        }
        _ => {
            if credit.retired_at.is_some() {
                return Ok(ValidateCallbackResult::Invalid(
                    "Non-retired credits cannot have retired_at timestamp".to_string(),
                ));
            }
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, .. } | OpEntry::UpdateEntry { app_entry, .. } => {
                match app_entry {
                    EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                    EntryTypes::CarbonFootprint(footprint) => validate_carbon_footprint(&footprint),
                    EntryTypes::CarbonCredit(credit) => validate_carbon_credit(&credit),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, .. } => match link_type {
            LinkTypes::AnchorToFootprints
            | LinkTypes::AnchorToCredits
            | LinkTypes::ProjectToCredits
            | LinkTypes::FootprintUpdates
            | LinkTypes::CreditTransfers => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink { link_type, .. } => match link_type {
            LinkTypes::CreditTransfers => Ok(ValidateCallbackResult::Invalid(
                "Credit transfer links cannot be deleted".to_string(),
            )),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_)
        | FlatOp::RegisterAgentActivity(_)
        | FlatOp::RegisterUpdate(_)
        | FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}
