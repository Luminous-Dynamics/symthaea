// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Carbon Coordinator Zome
//!
//! Provides functions for carbon footprint tracking and carbon credit management.
//! Uses HDK 0.6.0-dev.1 with LinkQuery::try_new() pattern.

use hdk::prelude::*;
use carbon_integrity::*;

/// Get or create an anchor for the given string
fn get_or_create_anchor(anchor_text: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_text.to_string());
    let entry_hash = hash_entry(&anchor)?;

    // Check if anchor exists
    if get(entry_hash.clone(), GetOptions::default())?.is_none() {
        create_entry(&EntryTypes::Anchor(anchor))?;
    }

    Ok(entry_hash)
}

// ============================================================================
// Carbon Footprint Management
// ============================================================================

/// Input for creating a carbon footprint measurement
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateFootprintInput {
    pub entity_did: String,
    pub period_start: i64,
    pub period_end: i64,
    pub scope1: f64,
    pub scope2: f64,
    pub scope3: f64,
    pub methodology: String,
    pub verified_by: Option<String>,
}

/// Create a new carbon footprint measurement
#[hdk_extern]
pub fn create_carbon_footprint(input: CreateFootprintInput) -> ExternResult<Record> {
    let footprint = CarbonFootprint {
        entity_did: input.entity_did.clone(),
        period_start: input.period_start,
        period_end: input.period_end,
        scope1: input.scope1,
        scope2: input.scope2,
        scope3: input.scope3,
        methodology: input.methodology,
        verified_by: input.verified_by,
    };

    let action_hash = create_entry(&EntryTypes::CarbonFootprint(footprint))?;

    // Link from entity anchor to footprint
    let anchor_hash = get_or_create_anchor(&format!("footprints:{}", input.entity_did))?;
    create_link(
        anchor_hash,
        action_hash.clone(),
        LinkTypes::AnchorToFootprints,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created footprint".into())))
}

/// Get all footprints for an entity
#[hdk_extern]
pub fn get_footprints_by_entity(entity_did: String) -> ExternResult<Vec<Record>> {
    let anchor_hash = get_or_create_anchor(&format!("footprints:{}", entity_did))?;
    let query = LinkQuery::try_new(anchor_hash, LinkTypes::AnchorToFootprints)?;
    let links = get_links(query, GetStrategy::default())?;

    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid action hash".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }

    Ok(records)
}

/// Get a specific footprint by action hash
#[hdk_extern]
pub fn get_carbon_footprint(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

/// Input for updating a footprint with verification
#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyFootprintInput {
    pub original_action_hash: ActionHash,
    pub verifier_did: String,
}

/// Mark a footprint as verified
#[hdk_extern]
pub fn verify_carbon_footprint(input: VerifyFootprintInput) -> ExternResult<Record> {
    let record = get(input.original_action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Footprint not found".into())))?;

    let footprint: CarbonFootprint = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid footprint entry".into())))?;

    let updated = CarbonFootprint {
        verified_by: Some(input.verifier_did),
        ..footprint
    };

    let action_hash = update_entry(input.original_action_hash.clone(), &EntryTypes::CarbonFootprint(updated))?;

    // Create update link
    create_link(
        input.original_action_hash,
        action_hash.clone(),
        LinkTypes::FootprintUpdates,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get updated footprint".into())))
}

/// Calculate total emissions for an entity
#[hdk_extern]
pub fn get_total_emissions(entity_did: String) -> ExternResult<f64> {
    let records = get_footprints_by_entity(entity_did)?;
    let mut total = 0.0;

    for record in records {
        if let Some(footprint) = record
            .entry()
            .to_app_option::<CarbonFootprint>()
            .ok()
            .flatten()
        {
            total += footprint.scope1 + footprint.scope2 + footprint.scope3;
        }
    }

    Ok(total)
}

// ============================================================================
// Carbon Credit Management
// ============================================================================

/// Input for creating a carbon credit
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateCreditInput {
    pub id: String,
    pub project_id: String,
    pub vintage_year: u32,
    pub tonnes_co2e: f64,
    pub owner_did: String,
}

/// Create a new carbon credit
#[hdk_extern]
pub fn create_carbon_credit(input: CreateCreditInput) -> ExternResult<Record> {
    let credit = CarbonCredit {
        id: input.id.clone(),
        project_id: input.project_id.clone(),
        vintage_year: input.vintage_year,
        tonnes_co2e: input.tonnes_co2e,
        status: CreditStatus::Active,
        owner_did: input.owner_did.clone(),
        retired_at: None,
    };

    let action_hash = create_entry(&EntryTypes::CarbonCredit(credit))?;

    // Link from owner anchor to credit
    let owner_anchor = get_or_create_anchor(&format!("credits:{}", input.owner_did))?;
    create_link(
        owner_anchor,
        action_hash.clone(),
        LinkTypes::AnchorToCredits,
        (),
    )?;

    // Link from project anchor to credit
    let project_anchor = get_or_create_anchor(&format!("project_credits:{}", input.project_id))?;
    create_link(
        project_anchor,
        action_hash.clone(),
        LinkTypes::ProjectToCredits,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get created credit".into())))
}

/// Get a specific credit by action hash
#[hdk_extern]
pub fn get_carbon_credit(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

/// Get all credits owned by a DID
#[hdk_extern]
pub fn get_credits_by_owner(owner_did: String) -> ExternResult<Vec<Record>> {
    let anchor_hash = get_or_create_anchor(&format!("credits:{}", owner_did))?;
    let query = LinkQuery::try_new(anchor_hash, LinkTypes::AnchorToCredits)?;
    let links = get_links(query, GetStrategy::default())?;

    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid action hash".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }

    Ok(records)
}

/// Get all credits for a project
#[hdk_extern]
pub fn get_credits_by_project(project_id: String) -> ExternResult<Vec<Record>> {
    let anchor_hash = get_or_create_anchor(&format!("project_credits:{}", project_id))?;
    let query = LinkQuery::try_new(anchor_hash, LinkTypes::ProjectToCredits)?;
    let links = get_links(query, GetStrategy::default())?;

    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid action hash".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }

    Ok(records)
}

/// Input for transferring a credit
#[derive(Serialize, Deserialize, Debug)]
pub struct TransferCreditInput {
    pub credit_action_hash: ActionHash,
    pub new_owner_did: String,
}

/// Transfer a credit to a new owner
#[hdk_extern]
pub fn transfer_credit(input: TransferCreditInput) -> ExternResult<Record> {
    let record = get(input.credit_action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Credit not found".into())))?;

    let credit: CarbonCredit = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid credit entry".into())))?;

    // Verify credit can be transferred
    if credit.status != CreditStatus::Active {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only active credits can be transferred".into()
        )));
    }

    let old_owner = credit.owner_did.clone();

    let updated = CarbonCredit {
        owner_did: input.new_owner_did.clone(),
        status: CreditStatus::Transferred,
        ..credit
    };

    // Re-activate as the new owner
    let final_credit = CarbonCredit {
        status: CreditStatus::Active,
        ..updated
    };

    let action_hash = update_entry(input.credit_action_hash.clone(), &EntryTypes::CarbonCredit(final_credit))?;

    // Create transfer link for history
    create_link(
        input.credit_action_hash,
        action_hash.clone(),
        LinkTypes::CreditTransfers,
        (),
    )?;

    // Link to new owner
    let new_owner_anchor = get_or_create_anchor(&format!("credits:{}", input.new_owner_did))?;
    create_link(
        new_owner_anchor,
        action_hash.clone(),
        LinkTypes::AnchorToCredits,
        (),
    )?;

    // Note: Old owner's link remains for historical purposes
    // In production, you might want to delete the old link
    let _ = old_owner; // Acknowledge we're not deleting old link

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get transferred credit".into())))
}

/// Input for retiring a credit
#[derive(Serialize, Deserialize, Debug)]
pub struct RetireCreditInput {
    pub credit_action_hash: ActionHash,
}

/// Retire a credit (used for offsetting, cannot be undone)
#[hdk_extern]
pub fn retire_credit(input: RetireCreditInput) -> ExternResult<Record> {
    let record = get(input.credit_action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Credit not found".into())))?;

    let credit: CarbonCredit = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid credit entry".into())))?;

    // Verify credit can be retired
    if credit.status != CreditStatus::Active {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only active credits can be retired".into()
        )));
    }

    let now = sys_time()?;
    let retired_at = now.as_micros() / 1_000_000; // Convert to seconds

    let updated = CarbonCredit {
        status: CreditStatus::Retired,
        retired_at: Some(retired_at),
        ..credit
    };

    let action_hash = update_entry(input.credit_action_hash, &EntryTypes::CarbonCredit(updated))?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to get retired credit".into())))
}

/// Get the transfer history for a credit
#[hdk_extern]
pub fn get_credit_transfer_history(original_action_hash: ActionHash) -> ExternResult<Vec<ActionHash>> {
    let mut history = vec![original_action_hash.clone()];
    let mut current = original_action_hash;

    loop {
        let query = LinkQuery::try_new(current.clone(), LinkTypes::CreditTransfers)?;
        let links = get_links(query, GetStrategy::default())?;

        if let Some(link) = links.first() {
            let next = ActionHash::try_from(link.target.clone())
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid action hash".into())))?;
            history.push(next.clone());
            current = next;
        } else {
            break;
        }
    }

    Ok(history)
}

// ============================================================================
// Statistics
// ============================================================================

/// Summary of credits for an owner
#[derive(Serialize, Deserialize, Debug)]
pub struct CreditSummary {
    pub total_credits: u64,
    pub total_tonnes: f64,
    pub active_tonnes: f64,
    pub retired_tonnes: f64,
    pub transferred_count: u64,
}

/// Get credit summary for an owner
#[hdk_extern]
pub fn get_credit_summary(owner_did: String) -> ExternResult<CreditSummary> {
    let records = get_credits_by_owner(owner_did)?;

    let mut summary = CreditSummary {
        total_credits: 0,
        total_tonnes: 0.0,
        active_tonnes: 0.0,
        retired_tonnes: 0.0,
        transferred_count: 0,
    };

    for record in records {
        if let Some(credit) = record
            .entry()
            .to_app_option::<CarbonCredit>()
            .ok()
            .flatten()
        {
            summary.total_credits += 1;
            summary.total_tonnes += credit.tonnes_co2e;

            match credit.status {
                CreditStatus::Active => summary.active_tonnes += credit.tonnes_co2e,
                CreditStatus::Retired => summary.retired_tonnes += credit.tonnes_co2e,
                CreditStatus::Transferred => summary.transferred_count += 1,
            }
        }
    }

    Ok(summary)
}
