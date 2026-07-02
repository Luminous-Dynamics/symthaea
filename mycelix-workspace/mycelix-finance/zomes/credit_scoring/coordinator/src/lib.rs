// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Credit Scoring Coordinator Zome
//!
//! # DEPRECATION NOTICE
//!
//! This module is DEPRECATED and scheduled for removal.
//!
//! The credit scoring approach conflicts with the Mycelix Constitution's vision
//! of "plural forms of economic coordination, including gift economies, mutual
//! credit, time exchange, stewardship recognition, and market mechanisms."
//! (Constitution Article VIII, Section 1)
//!
//! This module has been replaced by:
//! - **CGC** (Civic Gifting Credits) - for recognition, not scoring
//! - **TEND** (Time Exchange) - for mutual credit time banking
//! - **HEARTH** (Commons Pool) - for collective resource governance
//!
//! See: Commons Charter v1.0, Article II
//!
//! Migration path:
//! 1. Stop using credit_score-based lending decisions
//! 2. Implement CGC for gratitude/recognition needs
//! 3. Implement TEND for service exchange needs
//! 4. Implement HEARTH for community resource pooling
//!
//! This module will be removed in v6.0.
//!
#![deprecated(
    since = "5.4.0",
    note = "Credit scoring conflicts with Commons Charter. Use CGC, TEND, or HEARTH modules instead."
)]

use credit_scoring_integrity::*;
use hdk::prelude::*;
use mycelix_finance_shared::anchors::anchor_hash;
use mycelix_finance_shared::batch::links_to_records;

// anchor_hash is now imported from mycelix_finance_shared::anchors

#[hdk_extern]
pub fn create_credit_profile(input: CreateProfileInput) -> ExternResult<Record> {
    if input.did.is_empty() || input.did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DID must be 1-256 characters".into()
        )));
    }
    if input.matl_score < 0.0 || input.matl_score > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "MATL score must be between 0.0 and 1.0".into()
        )));
    }
    if input.collateral_ratio < 0.0 || input.collateral_ratio > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Collateral ratio must be between 0.0 and 1.0".into()
        )));
    }
    let now = sys_time()?;
    let computed_score = compute_credit_score(
        input.matl_score,
        1.0, // initial payment history (no history = perfect)
        input.collateral_ratio,
        0,   // new account
        0.5, // initial activity
    );

    let profile = CreditProfile {
        did: input.did.clone(),
        matl_score: input.matl_score,
        payment_history_score: 1.0,
        collateral_ratio: input.collateral_ratio,
        account_age_days: 0,
        activity_score: 0.5,
        computed_score,
        last_updated: now,
        version: 1,
    };

    let action_hash = create_entry(&EntryTypes::CreditProfile(profile))?;
    create_link(
        anchor_hash(&input.did)?,
        action_hash.clone(),
        LinkTypes::DidToProfile,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateProfileInput {
    pub did: String,
    pub matl_score: f64,
    pub collateral_ratio: f64,
}

fn compute_credit_score(
    matl: f64,
    payment_history: f64,
    collateral: f64,
    age_days: u64,
    activity: f64,
) -> f64 {
    let age_factor = (age_days as f64 / 365.0).min(1.0);
    let raw = (matl * 0.40)
        + (payment_history * 0.30)
        + (collateral * 0.15)
        + (age_factor * 0.10)
        + (activity * 0.05);
    (raw * 1000.0).round()
}

#[hdk_extern]
pub fn get_credit_profile(did: String) -> ExternResult<Option<Record>> {
    if did.is_empty() || did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DID must be 1-256 characters".into()
        )));
    }
    let query = LinkQuery::try_new(anchor_hash(&did)?, LinkTypes::DidToProfile)?;
    let links = get_links(query, GetStrategy::default())?;
    if let Some(link) = links.first() {
        let action_hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?;
        return get(action_hash, GetOptions::default());
    }
    Ok(None)
}

#[hdk_extern]
pub fn record_payment(input: RecordPaymentInput) -> ExternResult<Record> {
    if input.profile_did.is_empty() || input.profile_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Profile DID must be 1-256 characters".into()
        )));
    }
    if input.loan_id.is_empty() || input.loan_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Loan ID must be 1-256 characters".into()
        )));
    }
    if input.amount <= 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Amount must be greater than 0".into()
        )));
    }
    if input.currency.is_empty() || input.currency.len() > 10 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Currency must be 1-10 characters".into()
        )));
    }
    let now = sys_time()?;
    let status = if let Some(paid) = input.paid_date {
        if paid.as_micros() <= input.due_date.as_micros() {
            PaymentStatus::OnTime
        } else {
            PaymentStatus::Late
        }
    } else if now.as_micros() > input.due_date.as_micros() {
        PaymentStatus::Missed
    } else {
        PaymentStatus::Pending
    };

    let record = PaymentRecord {
        id: format!("payment:{}:{}", input.loan_id, now.as_micros()),
        profile_did: input.profile_did.clone(),
        loan_id: input.loan_id.clone(),
        amount: input.amount,
        currency: input.currency,
        due_date: input.due_date,
        paid_date: input.paid_date,
        status,
    };

    let action_hash = create_entry(&EntryTypes::PaymentRecord(record))?;
    create_link(
        anchor_hash(&input.profile_did)?,
        action_hash.clone(),
        LinkTypes::ProfileToPayments,
        (),
    )?;
    create_link(
        anchor_hash(&input.loan_id)?,
        action_hash.clone(),
        LinkTypes::LoanToPayments,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordPaymentInput {
    pub profile_did: String,
    pub loan_id: String,
    pub amount: f64,
    pub currency: String,
    pub due_date: Timestamp,
    pub paid_date: Option<Timestamp>,
}

#[hdk_extern]
pub fn register_collateral(input: RegisterCollateralInput) -> ExternResult<Record> {
    if input.owner_did.is_empty() || input.owner_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Owner DID must be 1-256 characters".into()
        )));
    }
    if input.asset_id.is_empty() || input.asset_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Asset ID must be 1-256 characters".into()
        )));
    }
    if input.appraised_value <= 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Appraised value must be greater than 0".into()
        )));
    }
    if input.currency.is_empty() || input.currency.len() > 10 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Currency must be 1-10 characters".into()
        )));
    }
    let now = sys_time()?;
    let record = CollateralRecord {
        id: format!("collateral:{}:{}", input.owner_did, now.as_micros()),
        owner_did: input.owner_did.clone(),
        asset_type: input.asset_type,
        asset_id: input.asset_id,
        appraised_value: input.appraised_value,
        currency: input.currency,
        locked_for_loan: None,
        registered: now,
    };

    let action_hash = create_entry(&EntryTypes::CollateralRecord(record))?;
    create_link(
        anchor_hash(&input.owner_did)?,
        action_hash.clone(),
        LinkTypes::ProfileToCollateral,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterCollateralInput {
    pub owner_did: String,
    pub asset_type: CollateralType,
    pub asset_id: String,
    pub appraised_value: f64,
    pub currency: String,
}

#[hdk_extern]
pub fn update_credit_score(did: String) -> ExternResult<Option<Record>> {
    if did.is_empty() || did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DID must be 1-256 characters".into()
        )));
    }
    // Get all payment records for this profile
    let payment_records = get_payment_records(did.clone())?;

    // Calculate payment history score
    let (on_time, late, missed) = payment_records.iter().fold((0, 0, 0), |acc, r| {
        if let Some(record) = r.entry().to_app_option::<PaymentRecord>().ok().flatten() {
            match record.status {
                PaymentStatus::OnTime => (acc.0 + 1, acc.1, acc.2),
                PaymentStatus::Late => (acc.0, acc.1 + 1, acc.2),
                PaymentStatus::Missed => (acc.0, acc.1, acc.2 + 1),
                _ => acc,
            }
        } else {
            acc
        }
    });

    let total = on_time + late + missed;
    let payment_history = if total > 0 {
        (on_time as f64 * 1.0 + late as f64 * 0.5) / total as f64
    } else {
        1.0 // No history = perfect score
    };

    // Get collateral value
    let collateral_records = get_collateral(did.clone())?;
    let total_collateral: f64 = collateral_records
        .iter()
        .filter_map(|r| {
            r.entry()
                .to_app_option::<CollateralRecord>()
                .ok()
                .flatten()
                .map(|c| c.appraised_value)
        })
        .sum();

    // Update the profile
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::CreditProfile,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(profile) = record
            .entry()
            .to_app_option::<CreditProfile>()
            .ok()
            .flatten()
        {
            if profile.did == did {
                let now = sys_time()?;
                let new_collateral_ratio = total_collateral / 10000.0; // Normalize
                let new_score = compute_credit_score(
                    profile.matl_score,
                    payment_history,
                    new_collateral_ratio.min(1.0),
                    profile.account_age_days,
                    profile.activity_score,
                );

                let updated = CreditProfile {
                    payment_history_score: payment_history,
                    collateral_ratio: new_collateral_ratio.min(1.0),
                    computed_score: new_score,
                    last_updated: now,
                    version: profile.version + 1,
                    ..profile
                };

                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::CreditProfile(updated),
                )?;
                return get(action_hash, GetOptions::default());
            }
        }
    }
    Ok(None)
}

/// Get all payment records for a profile
///
/// OPTIMIZED: Uses batch query to avoid N+1 pattern
#[hdk_extern]
pub fn get_payment_records(did: String) -> ExternResult<Vec<Record>> {
    if did.is_empty() || did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DID must be 1-256 characters".into()
        )));
    }
    let query = LinkQuery::try_new(anchor_hash(&did)?, LinkTypes::ProfileToPayments)?;
    let links = get_links(query, GetStrategy::default())?;
    // FIXED N+1: Use batch fetch instead of individual get() calls
    links_to_records(links)
}

/// Get all collateral for a profile
///
/// OPTIMIZED: Uses batch query to avoid N+1 pattern
#[hdk_extern]
pub fn get_collateral(did: String) -> ExternResult<Vec<Record>> {
    if did.is_empty() || did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DID must be 1-256 characters".into()
        )));
    }
    let query = LinkQuery::try_new(anchor_hash(&did)?, LinkTypes::ProfileToCollateral)?;
    let links = get_links(query, GetStrategy::default())?;
    // FIXED N+1: Use batch fetch instead of individual get() calls
    links_to_records(links)
}

/// Lock collateral for a loan
#[hdk_extern]
pub fn lock_collateral(input: LockCollateralInput) -> ExternResult<Record> {
    if input.collateral_id.is_empty() || input.collateral_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Collateral ID must be 1-256 characters".into()
        )));
    }
    if input.loan_id.is_empty() || input.loan_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Loan ID must be 1-256 characters".into()
        )));
    }
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::CollateralRecord,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(collateral) = record
            .entry()
            .to_app_option::<CollateralRecord>()
            .ok()
            .flatten()
        {
            if collateral.id == input.collateral_id {
                if collateral.locked_for_loan.is_some() {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Collateral already locked".into()
                    )));
                }
                let updated = CollateralRecord {
                    locked_for_loan: Some(input.loan_id),
                    ..collateral
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::CollateralRecord(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Collateral not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LockCollateralInput {
    pub collateral_id: String,
    pub loan_id: String,
}

/// Release collateral after loan is repaid
#[hdk_extern]
pub fn release_collateral(collateral_id: String) -> ExternResult<Record> {
    if collateral_id.is_empty() || collateral_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Collateral ID must be 1-256 characters".into()
        )));
    }
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::CollateralRecord,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(collateral) = record
            .entry()
            .to_app_option::<CollateralRecord>()
            .ok()
            .flatten()
        {
            if collateral.id == collateral_id {
                let updated = CollateralRecord {
                    locked_for_loan: None,
                    ..collateral
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::CollateralRecord(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Collateral not found".into()
    )))
}

/// Update MATL score (called from bridge when trust changes)
#[hdk_extern]
pub fn update_matl_score(input: UpdateMatlInput) -> ExternResult<Option<Record>> {
    if input.did.is_empty() || input.did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DID must be 1-256 characters".into()
        )));
    }
    if input.matl_score < 0.0 || input.matl_score > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "MATL score must be between 0.0 and 1.0".into()
        )));
    }
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::CreditProfile,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(profile) = record
            .entry()
            .to_app_option::<CreditProfile>()
            .ok()
            .flatten()
        {
            if profile.did == input.did {
                let now = sys_time()?;
                let new_score = compute_credit_score(
                    input.matl_score,
                    profile.payment_history_score,
                    profile.collateral_ratio,
                    profile.account_age_days,
                    profile.activity_score,
                );

                let updated = CreditProfile {
                    matl_score: input.matl_score,
                    computed_score: new_score,
                    last_updated: now,
                    version: profile.version + 1,
                    ..profile
                };

                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::CreditProfile(updated),
                )?;
                return get(action_hash, GetOptions::default());
            }
        }
    }
    Ok(None)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateMatlInput {
    pub did: String,
    pub matl_score: f64,
}

/// Get credit profiles with scores in a range (for offer matching)
#[hdk_extern]
pub fn get_profiles_by_score_range(input: ScoreRangeInput) -> ExternResult<Vec<Record>> {
    if input.min_score > input.max_score {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "min_score must be <= max_score".into()
        )));
    }
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::CreditProfile,
        )?))
        .include_entries(true);

    let mut results = Vec::new();
    for record in query(filter)? {
        if let Some(profile) = record
            .entry()
            .to_app_option::<CreditProfile>()
            .ok()
            .flatten()
        {
            if profile.computed_score >= input.min_score
                && profile.computed_score <= input.max_score
            {
                results.push(record);
            }
        }
    }
    Ok(results)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ScoreRangeInput {
    pub min_score: f64,
    pub max_score: f64,
}
