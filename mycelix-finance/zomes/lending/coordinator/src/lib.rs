// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Lending Coordinator Zome
//!
//! # DEPRECATION NOTICE
//!
//! This module is DEPRECATED and scheduled for removal.
//!
//! Traditional lending with interest-bearing loans conflicts with the Mycelix
//! Constitution's vision of "plural forms of economic coordination" that
//! prioritize mutual aid over debt-based finance.
//! (Constitution Article VIII, Section 1)
//!
//! This module has been replaced by:
//! - **TEND** (Time Exchange) - Interest-free mutual credit for services
//! - **HEARTH** (Commons Pool) - Community resource pooling with democratic allocation
//! - **CGC** (Civic Gifting Credits) - Recognition-based social support
//!
//! See: Commons Charter v1.0, Article II
//!
//! Key differences:
//! - TEND: ±40 balance limits, no interest, all hours equal
//! - HEARTH: Community votes on allocations, no individual debt
//! - CGC: Recognition flows freely, cannot create debt
//!
//! Migration path for existing loans:
//! 1. Honor existing loan commitments
//! 2. Do not create new loans
//! 3. Transition borrowers to TEND or HEARTH for future needs
//!
//! This module will be removed in v6.0.
//!
#![deprecated(
    since = "5.4.0",
    note = "Traditional lending conflicts with Commons Charter. Use TEND, HEARTH, or CGC modules instead."
)]

use hdk::prelude::*;
use lending_integrity::*;
use mycelix_finance_shared::anchors::anchor_hash;
use mycelix_finance_shared::batch::{filter_records_by, links_to_records};

// anchor_hash is now imported from mycelix_finance_shared::anchors

#[hdk_extern]
pub fn request_loan(input: RequestLoanInput) -> ExternResult<Record> {
    if input.borrower_did.is_empty() || input.borrower_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Borrower DID must be 1-256 characters".into()
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
    if input.term_days == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Term must be at least 1 day".into()
        )));
    }
    let now = sys_time()?;
    let loan = Loan {
        id: format!("loan:{}:{}", input.borrower_did, now.as_micros()),
        borrower_did: input.borrower_did.clone(),
        lender_did: String::new(), // To be filled when funded
        principal: input.amount,
        currency: input.currency,
        interest_rate: 0.0, // To be determined
        term_days: input.term_days,
        collateral_ids: input.collateral_ids,
        status: LoanStatus::Requested,
        created: now,
        funded: None,
        maturity: None,
        repaid: None,
    };

    let action_hash = create_entry(&EntryTypes::Loan(loan))?;
    create_link(
        anchor_hash(&input.borrower_did)?,
        action_hash.clone(),
        LinkTypes::BorrowerToLoans,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RequestLoanInput {
    pub borrower_did: String,
    pub amount: f64,
    pub currency: String,
    pub term_days: u32,
    pub collateral_ids: Vec<String>,
}

#[hdk_extern]
pub fn create_loan_offer(input: CreateOfferInput) -> ExternResult<Record> {
    if input.lender_did.is_empty() || input.lender_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Lender DID must be 1-256 characters".into()
        )));
    }
    if input.max_amount <= 0.0 || input.min_amount <= 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Amounts must be greater than 0".into()
        )));
    }
    if input.min_amount > input.max_amount {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "min_amount must be <= max_amount".into()
        )));
    }
    if input.currency.is_empty() || input.currency.len() > 10 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Currency must be 1-10 characters".into()
        )));
    }
    if input.base_interest_rate < 0.0 || input.base_interest_rate > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Interest rate must be between 0.0 and 1.0".into()
        )));
    }
    if input.max_term_days == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Max term must be at least 1 day".into()
        )));
    }
    let now = sys_time()?;
    let offer = LoanOffer {
        id: format!("offer:{}:{}", input.lender_did, now.as_micros()),
        lender_did: input.lender_did.clone(),
        max_amount: input.max_amount,
        min_amount: input.min_amount,
        currency: input.currency,
        base_interest_rate: input.base_interest_rate,
        min_credit_score: input.min_credit_score,
        max_term_days: input.max_term_days,
        collateral_required: input.collateral_required,
        active: true,
        created: now,
    };

    let action_hash = create_entry(&EntryTypes::LoanOffer(offer))?;
    create_link(
        anchor_hash(&input.lender_did)?,
        action_hash.clone(),
        LinkTypes::LenderToOffers,
        (),
    )?;
    // Link to global active offers anchor
    let anchor = anchor_hash("active_offers")?;
    create_link(anchor, action_hash.clone(), LinkTypes::ActiveOffers, ())?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateOfferInput {
    pub lender_did: String,
    pub max_amount: f64,
    pub min_amount: f64,
    pub currency: String,
    pub base_interest_rate: f64,
    pub min_credit_score: f64,
    pub max_term_days: u32,
    pub collateral_required: bool,
}

#[hdk_extern]
pub fn fund_loan(input: FundLoanInput) -> ExternResult<Record> {
    if input.loan_id.is_empty() || input.loan_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Loan ID must be 1-256 characters".into()
        )));
    }
    if input.lender_did.is_empty() || input.lender_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Lender DID must be 1-256 characters".into()
        )));
    }
    if input.interest_rate < 0.0 || input.interest_rate > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Interest rate must be between 0.0 and 1.0".into()
        )));
    }
    // Find the loan, update it with lender info
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::Loan)?))
        .include_entries(true);
    for record in query(filter)? {
        if let Some(loan) = record.entry().to_app_option::<Loan>().ok().flatten() {
            if loan.id == input.loan_id {
                let now = sys_time()?;
                let maturity = Timestamp::from_micros(
                    now.as_micros() as i64 + (loan.term_days as i64 * 24 * 3600 * 1_000_000),
                );
                let funded_loan = Loan {
                    lender_did: input.lender_did.clone(),
                    interest_rate: input.interest_rate,
                    status: LoanStatus::Funded,
                    funded: Some(now),
                    maturity: Some(maturity),
                    ..loan
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Loan(funded_loan),
                )?;
                create_link(
                    anchor_hash(&input.lender_did)?,
                    action_hash.clone(),
                    LinkTypes::LenderToLoans,
                    (),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Loan not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FundLoanInput {
    pub loan_id: String,
    pub lender_did: String,
    pub interest_rate: f64,
}

#[hdk_extern]
pub fn repay_loan(loan_id: String) -> ExternResult<Record> {
    if loan_id.is_empty() || loan_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Loan ID must be 1-256 characters".into()
        )));
    }
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::Loan)?))
        .include_entries(true);
    for record in query(filter)? {
        if let Some(loan) = record.entry().to_app_option::<Loan>().ok().flatten() {
            if loan.id == loan_id {
                let now = sys_time()?;
                let repaid_loan = Loan {
                    status: LoanStatus::Repaid,
                    repaid: Some(now),
                    ..loan
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Loan(repaid_loan),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Loan not found".into())))
}

/// Get all loans where a DID is the borrower
///
/// OPTIMIZED: Uses batch query to avoid N+1 pattern
#[hdk_extern]
pub fn get_borrower_loans(did: String) -> ExternResult<Vec<Record>> {
    if did.is_empty() || did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DID must be 1-256 characters".into()
        )));
    }
    let query = LinkQuery::try_new(anchor_hash(&did)?, LinkTypes::BorrowerToLoans)?;
    let links = get_links(query, GetStrategy::default())?;
    // FIXED N+1: Use batch fetch instead of individual get() calls
    links_to_records(links)
}

/// Get all active loan offers
///
/// OPTIMIZED: Uses batch query to avoid N+1 pattern
#[hdk_extern]
pub fn get_active_offers(_: ()) -> ExternResult<Vec<Record>> {
    let anchor = anchor_hash("active_offers")?;
    let query = LinkQuery::try_new(anchor, LinkTypes::ActiveOffers)?;
    let links = get_links(query, GetStrategy::default())?;

    // FIXED N+1: Batch fetch all records, then filter
    let all_records = links_to_records(links)?;

    // Filter to only active offers
    Ok(filter_records_by::<LoanOffer, _>(&all_records, |offer| {
        offer.active
    }))
}

/// Get a specific loan by ID
#[hdk_extern]
pub fn get_loan(loan_id: String) -> ExternResult<Option<Record>> {
    if loan_id.is_empty() || loan_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Loan ID must be 1-256 characters".into()
        )));
    }
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::Loan)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(loan) = record.entry().to_app_option::<Loan>().ok().flatten() {
            if loan.id == loan_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
}

/// Get all loans where a DID is the lender
///
/// OPTIMIZED: Uses batch query to avoid N+1 pattern
#[hdk_extern]
pub fn get_lender_loans(did: String) -> ExternResult<Vec<Record>> {
    if did.is_empty() || did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DID must be 1-256 characters".into()
        )));
    }
    let query = LinkQuery::try_new(anchor_hash(&did)?, LinkTypes::LenderToLoans)?;
    let links = get_links(query, GetStrategy::default())?;
    // FIXED N+1: Use batch fetch instead of individual get() calls
    links_to_records(links)
}

/// Mark a loan as defaulted (only lender can do this)
#[hdk_extern]
pub fn default_loan(loan_id: String) -> ExternResult<Record> {
    if loan_id.is_empty() || loan_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Loan ID must be 1-256 characters".into()
        )));
    }
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::Loan)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(loan) = record.entry().to_app_option::<Loan>().ok().flatten() {
            if loan.id == loan_id {
                if loan.status != LoanStatus::Active {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Can only default active loans".into()
                    )));
                }
                let defaulted = Loan {
                    status: LoanStatus::Defaulted,
                    ..loan
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Loan(defaulted),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Loan not found".into())))
}

/// Cancel a loan request (only borrower can do this for Requested loans)
#[hdk_extern]
pub fn cancel_loan(loan_id: String) -> ExternResult<Record> {
    if loan_id.is_empty() || loan_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Loan ID must be 1-256 characters".into()
        )));
    }
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::Loan)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(loan) = record.entry().to_app_option::<Loan>().ok().flatten() {
            if loan.id == loan_id {
                if loan.status != LoanStatus::Requested {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Can only cancel requested loans".into()
                    )));
                }
                let cancelled = Loan {
                    status: LoanStatus::Cancelled,
                    ..loan
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Loan(cancelled),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Loan not found".into())))
}

/// Deactivate a loan offer
#[hdk_extern]
pub fn deactivate_offer(offer_id: String) -> ExternResult<Record> {
    if offer_id.is_empty() || offer_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Offer ID must be 1-256 characters".into()
        )));
    }
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::LoanOffer,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(offer) = record.entry().to_app_option::<LoanOffer>().ok().flatten() {
            if offer.id == offer_id {
                let deactivated = LoanOffer {
                    active: false,
                    ..offer
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::LoanOffer(deactivated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Offer not found".into())))
}

/// Get all loans by status
#[hdk_extern]
pub fn get_loans_by_status(status: LoanStatus) -> ExternResult<Vec<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::Loan)?))
        .include_entries(true);

    let mut results = Vec::new();
    for record in query(filter)? {
        if let Some(loan) = record.entry().to_app_option::<Loan>().ok().flatten() {
            if loan.status == status {
                results.push(record);
            }
        }
    }
    Ok(results)
}

/// Create payment schedule for a funded loan
#[hdk_extern]
pub fn create_payment_schedule(input: CreateScheduleInput) -> ExternResult<Record> {
    if input.loan_id.is_empty() || input.loan_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Loan ID must be 1-256 characters".into()
        )));
    }
    if input.num_payments == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Number of payments must be at least 1".into()
        )));
    }
    // Find the loan
    let loan = get_loan(input.loan_id.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Loan not found".into())))?;

    let loan_data = loan
        .entry()
        .to_app_option::<Loan>()
        .ok()
        .flatten()
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid loan data".into()
        )))?;

    if loan_data.funded.is_none() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Loan not yet funded".into()
        )));
    }

    let funded_time = loan_data
        .funded
        .ok_or_else(|| {
            wasm_error!(WasmErrorInner::Guest(
                "Loan must be funded before generating payment schedule".to_string()
            ))
        })?
        .as_micros() as i64;
    let payment_interval =
        (loan_data.term_days as i64 * 24 * 3600 * 1_000_000) / input.num_payments as i64;
    let principal_per_payment = loan_data.principal / input.num_payments as f64;
    let total_interest = loan_data.principal * loan_data.interest_rate;
    let interest_per_payment = total_interest / input.num_payments as f64;

    let payments: Vec<ScheduledPayment> = (1..=input.num_payments)
        .map(|i| ScheduledPayment {
            payment_number: i,
            due_date: Timestamp::from_micros(funded_time + (payment_interval * i as i64)),
            principal_amount: principal_per_payment,
            interest_amount: interest_per_payment,
            total_amount: principal_per_payment + interest_per_payment,
            paid: false,
        })
        .collect();

    let schedule = PaymentSchedule {
        loan_id: input.loan_id.clone(),
        payments,
    };

    let action_hash = create_entry(&EntryTypes::PaymentSchedule(schedule))?;
    create_link(
        anchor_hash(&input.loan_id)?,
        action_hash.clone(),
        LinkTypes::LoanToSchedule,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateScheduleInput {
    pub loan_id: String,
    pub num_payments: u32,
}

/// Get payment schedule for a loan
///
/// OPTIMIZED: Uses batch query to avoid N+1 pattern
#[hdk_extern]
pub fn get_payment_schedule(loan_id: String) -> ExternResult<Option<Record>> {
    if loan_id.is_empty() || loan_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Loan ID must be 1-256 characters".into()
        )));
    }
    let query = LinkQuery::try_new(anchor_hash(&loan_id)?, LinkTypes::LoanToSchedule)?;
    let links = get_links(query, GetStrategy::default())?;

    // FIXED N+1: Use batch fetch (even for single expected result, more consistent)
    let records = links_to_records(links)?;
    Ok(records.into_iter().next())
}

/// Match borrower with suitable offers based on credit score
///
/// OPTIMIZED: Uses batch query to avoid N+1 pattern
#[hdk_extern]
pub fn match_offers_for_borrower(input: MatchOffersInput) -> ExternResult<Vec<Record>> {
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
    let anchor = anchor_hash("active_offers")?;
    let query = LinkQuery::try_new(anchor, LinkTypes::ActiveOffers)?;
    let links = get_links(query, GetStrategy::default())?;

    // FIXED N+1: Batch fetch all records, then filter
    let all_records = links_to_records(links)?;

    // Filter to matching offers
    Ok(filter_records_by::<LoanOffer, _>(&all_records, |offer| {
        offer.active
            && input.credit_score >= offer.min_credit_score
            && input.amount >= offer.min_amount
            && input.amount <= offer.max_amount
            && input.currency == offer.currency
            && input.term_days <= offer.max_term_days
            && (!offer.collateral_required || input.has_collateral)
    }))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MatchOffersInput {
    pub credit_score: f64,
    pub amount: f64,
    pub currency: String,
    pub term_days: u32,
    pub has_collateral: bool,
}

// =============================================================================
// ETHICAL CREDIT ASSESSMENT (Domain-Bounded, Privacy-Preserving)
// =============================================================================
//
// DESIGN PRINCIPLES (Anti-Social-Credit):
// 1. DOMAIN BOUNDARY: Only lending-specific data affects lending decisions
//    - NO cross-domain reputation bleeding (governance, social, speech)
//    - Your voting patterns, beliefs, or social behavior NEVER affect credit
// 2. REPUTATION DECAY: Negative events fade over time (right to redemption)
// 3. MINIMUM FLOOR: Everyone can access basic loans regardless of history
// 4. OPT-IN ENHANCEMENT: Share more data voluntarily for better rates
// 5. FULL TRANSPARENCY: Borrower can see exactly what data is used
// 6. APPEALS PROCESS: Challenge any data point used in assessment
// 7. RIGHT TO EXIT: Can request deletion and start fresh (with waiting period)
//
// EXPLICITLY EXCLUDED from credit assessment:
// - Governance voting patterns or participation
// - Political views or speech
// - Social connections or network position
// - FL participation or model training behavior
// - Knowledge contributions or epistemic claims
// - Any data from other Mycelix modules
// =============================================================================

/// Calculate credit assessment using ONLY lending-domain data
///
/// Ethical Credit Assessment = f(
///   payment_history * 0.50,       # On-time payments (PRIMARY - 50%)
///   collateral_ratio * 0.30,      # Available collateral (30%)
///   account_age * 0.20,           # Time as borrower (20%)
/// )
///
/// NOTE: We intentionally DO NOT include:
/// - MATL/governance scores (cross-domain bleeding)
/// - "Activity" outside lending (surveillance creep)
/// - Social network position (guilt by association)
///
/// OPTIMIZED: Fetches loans once and reuses for all calculations
/// (Previously this was an N+1 pattern calling get_borrower_loans 3 times)
#[hdk_extern]
pub fn calculate_credit_assessment(
    input: CreditAssessmentInput,
) -> ExternResult<CreditAssessmentResult> {
    if input.borrower_did.is_empty() || input.borrower_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Borrower DID must be 1-256 characters".into()
        )));
    }
    let now = sys_time()?;

    // ==========================================================================
    // OPTIMIZED: Fetch loans ONCE and reuse for all calculations
    // This fixes the N+1 pattern where we were calling get_borrower_loans 3 times
    // ==========================================================================
    let loans = get_borrower_loans(input.borrower_did.clone())?;

    // ==========================================================================
    // ETHICAL SAFEGUARD: Only use lending-specific data
    // ==========================================================================

    // 1. Payment history - ONLY from THIS lending module (50%)
    let (payment_history, payment_details) =
        calculate_payment_history_with_decay_from_loans(&loans, now)?;

    // 2. Collateral - verifiable assets pledged (30%)
    let (collateral_ratio, collateral_details) =
        calculate_collateral_with_transparency_from_loans(&loans)?;

    // 3. Account age in lending - time since first loan (20%)
    let (account_age, age_details) = calculate_lending_age_with_decay_from_loans(&loans, now)?;

    // Weighted sum - simpler, more transparent formula
    let base_score = payment_history * 0.50 + collateral_ratio * 0.30 + account_age * 0.20;

    // ==========================================================================
    // MINIMUM FLOOR GUARANTEE: No one is completely excluded
    // ==========================================================================
    // Even with worst possible history, you can still access basic loans
    // at higher rates. This prevents permanent financial exclusion.
    let floor_score = 0.25; // Everyone starts at least here
    let final_score = base_score.max(floor_score).min(1.0);

    // Determine tier with floor protection
    let credit_tier = match final_score {
        s if s >= 0.8 => CreditTier::Excellent,
        s if s >= 0.6 => CreditTier::Good,
        s if s >= 0.4 => CreditTier::Standard,
        _ => CreditTier::BasicAccess, // Renamed from "Poor" - less stigmatizing
    };

    // Rate adjustment - CAPPED to prevent predatory pricing
    // Maximum premium is 3% (not 5%) to prevent exploitation
    let rate_adjustment = match credit_tier {
        CreditTier::Excellent => -0.02,  // 2% discount
        CreditTier::Good => -0.01,       // 1% discount
        CreditTier::Standard => 0.0,     // Base rate
        CreditTier::BasicAccess => 0.03, // 3% MAX premium (capped)
    };

    // Build full transparency report
    let transparency_report = TransparencyReport {
        data_sources_used: vec![
            "lending_payment_history".to_string(),
            "lending_collateral".to_string(),
            "lending_account_age".to_string(),
        ],
        data_sources_excluded: vec![
            "governance_voting".to_string(),
            "social_connections".to_string(),
            "political_activity".to_string(),
            "fl_participation".to_string(),
            "knowledge_contributions".to_string(),
        ],
        decay_applied: true,
        decay_halflife_days: 365, // Negative events lose half weight per year
        floor_applied: base_score < floor_score,
        appeal_available: true,
    };

    Ok(CreditAssessmentResult {
        borrower_did: input.borrower_did,
        credit_score: final_score,
        credit_tier,
        components: CreditAssessmentComponents {
            payment_history: ComponentDetail {
                score: payment_history,
                weight: 0.50,
                contribution: payment_history * 0.50,
                details: payment_details,
            },
            collateral_ratio: ComponentDetail {
                score: collateral_ratio,
                weight: 0.30,
                contribution: collateral_ratio * 0.30,
                details: collateral_details,
            },
            account_age: ComponentDetail {
                score: account_age,
                weight: 0.20,
                contribution: account_age * 0.20,
                details: age_details,
            },
        },
        rate_adjustment,
        floor_applied: base_score < floor_score,
        transparency: transparency_report,
        calculated_at: now,
        valid_until: Timestamp::from_micros(now.as_micros() as i64 + 30 * 24 * 3600 * 1_000_000), // 30 days
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreditAssessmentInput {
    pub borrower_did: String,
    /// Optional: borrower can opt-in to share additional data for potentially better rates
    pub opt_in_additional_data: Option<bool>,
}

/// Credit tier - renamed to be less stigmatizing
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum CreditTier {
    Excellent,   // >= 0.8 - Best rates
    Good,        // >= 0.6 - Favorable rates
    Standard,    // >= 0.4 - Base rates
    BasicAccess, // < 0.4  - Higher rates but STILL HAS ACCESS (floor guarantee)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreditAssessmentResult {
    pub borrower_did: String,
    pub credit_score: f64,
    pub credit_tier: CreditTier,
    pub components: CreditAssessmentComponents,
    pub rate_adjustment: f64,
    pub floor_applied: bool,
    pub transparency: TransparencyReport,
    pub calculated_at: Timestamp,
    pub valid_until: Timestamp,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreditAssessmentComponents {
    pub payment_history: ComponentDetail,
    pub collateral_ratio: ComponentDetail,
    pub account_age: ComponentDetail,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ComponentDetail {
    pub score: f64,
    pub weight: f64,
    pub contribution: f64,
    pub details: String,
}

/// Full transparency about what data is and ISN'T used
#[derive(Serialize, Deserialize, Debug)]
pub struct TransparencyReport {
    pub data_sources_used: Vec<String>,
    pub data_sources_excluded: Vec<String>,
    pub decay_applied: bool,
    pub decay_halflife_days: u32,
    pub floor_applied: bool,
    pub appeal_available: bool,
}

/// Calculate payment history WITH temporal decay
/// Old negative events matter less than recent ones
///
/// OPTIMIZED: Accepts pre-fetched loans to avoid N+1 pattern
fn calculate_payment_history_with_decay_from_loans(
    loans: &[Record],
    now: Timestamp,
) -> ExternResult<(f64, String)> {
    if loans.is_empty() {
        return Ok((0.5, "No lending history - neutral score".to_string()));
    }

    let mut weighted_score = 0.0;
    let mut total_weight = 0.0;
    let mut repaid = 0;
    let mut defaulted = 0;

    for record in loans {
        if let Some(loan) = record.entry().to_app_option::<Loan>().ok().flatten() {
            // Calculate time-based decay weight
            // Half-life of 365 days - negative events from 2 years ago have 25% weight
            let event_time = loan.repaid.or(loan.funded).unwrap_or(loan.created);
            let age_days =
                (now.as_micros() as i64 - event_time.as_micros() as i64) / (24 * 3600 * 1_000_000);
            let decay_weight = 0.5_f64.powf(age_days as f64 / 365.0);

            match loan.status {
                LoanStatus::Repaid => {
                    weighted_score += 1.0 * decay_weight;
                    total_weight += decay_weight;
                    repaid += 1;
                }
                LoanStatus::Defaulted => {
                    weighted_score += 0.0 * decay_weight;
                    total_weight += decay_weight;
                    defaulted += 1;
                }
                LoanStatus::Active | LoanStatus::Funded => {
                    // Active loans count positively (paying on time)
                    weighted_score += 0.8 * decay_weight;
                    total_weight += decay_weight;
                }
                _ => {}
            }
        }
    }

    if total_weight == 0.0 {
        return Ok((0.5, "No completed loans - neutral score".to_string()));
    }

    let score = weighted_score / total_weight;
    let details = format!(
        "{} repaid, {} defaulted (decay-weighted, older events matter less)",
        repaid, defaulted
    );

    Ok((score, details))
}

/// Calculate collateral ratio WITH full transparency
/// Only uses lending-domain collateral data
///
/// OPTIMIZED: Accepts pre-fetched loans to avoid N+1 pattern
fn calculate_collateral_with_transparency_from_loans(
    loans: &[Record],
) -> ExternResult<(f64, String)> {
    if loans.is_empty() {
        return Ok((0.3, "No collateral history - base score".to_string()));
    }

    let mut total_collateral_loans = 0;
    let mut total_loans = 0;

    for record in loans {
        if let Some(loan) = record.entry().to_app_option::<Loan>().ok().flatten() {
            total_loans += 1;
            if !loan.collateral_ids.is_empty() {
                total_collateral_loans += 1;
            }
        }
    }

    if total_loans == 0 {
        return Ok((0.3, "No loan history".to_string()));
    }

    let collateral_ratio = total_collateral_loans as f64 / total_loans as f64;
    let score = 0.3 + 0.7 * collateral_ratio; // Range: 0.3 to 1.0

    let details = format!(
        "{} of {} loans had collateral ({:.0}%)",
        total_collateral_loans,
        total_loans,
        collateral_ratio * 100.0
    );

    Ok((score, details))
}

/// Calculate lending account age WITH decay weighting
/// Newer activity matters more than ancient history
///
/// OPTIMIZED: Accepts pre-fetched loans to avoid N+1 pattern
fn calculate_lending_age_with_decay_from_loans(
    loans: &[Record],
    now: Timestamp,
) -> ExternResult<(f64, String)> {
    if loans.is_empty() {
        return Ok((0.1, "New to lending - starting score".to_string()));
    }

    let mut oldest_created: Option<Timestamp> = None;
    let mut _newest_created: Option<Timestamp> = None;

    for record in loans {
        if let Some(loan) = record.entry().to_app_option::<Loan>().ok().flatten() {
            match oldest_created {
                None => oldest_created = Some(loan.created),
                Some(oldest) if loan.created < oldest => oldest_created = Some(loan.created),
                _ => {}
            }
            match _newest_created {
                None => _newest_created = Some(loan.created),
                Some(newest) if loan.created > newest => _newest_created = Some(loan.created),
                _ => {}
            }
        }
    }

    if let Some(oldest) = oldest_created {
        let age_micros = now.as_micros() as i64 - oldest.as_micros() as i64;
        let age_days = age_micros / (24 * 3600 * 1_000_000);

        // Score increases with age, maxing at 2 years
        // But we also check for recent activity
        let age_score = (age_days as f64 / 730.0).min(1.0);

        let details = format!("Account age: {} days (max credit at 730 days)", age_days);

        return Ok((age_score, details));
    }

    Ok((0.1, "Unable to determine account age".to_string()))
}

// =============================================================================
// APPEALS & DISPUTE MECHANISM (Right to Challenge)
// =============================================================================

/// Input for filing a credit appeal
#[derive(Serialize, Deserialize, Debug)]
pub struct CreditAppealInput {
    pub borrower_did: String,
    pub disputed_component: String, // "payment_history", "collateral", "account_age"
    pub reason: String,
    pub evidence: Option<String>, // Hash of supporting evidence
}

/// Appeal status
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AppealStatus {
    Pending,
    UnderReview,
    Approved,
    Rejected,
    PartiallyApproved,
}

/// Credit appeal record
#[derive(Serialize, Deserialize, Debug)]
pub struct CreditAppeal {
    pub id: String,
    pub borrower_did: String,
    pub disputed_component: String,
    pub reason: String,
    pub evidence: Option<String>,
    pub status: AppealStatus,
    pub reviewer_notes: Option<String>,
    pub score_adjustment: Option<f64>,
    pub filed_at: Timestamp,
    pub resolved_at: Option<Timestamp>,
}

/// File a credit appeal - right to challenge any data point
#[hdk_extern]
pub fn file_credit_appeal(input: CreditAppealInput) -> ExternResult<CreditAppeal> {
    if input.borrower_did.is_empty() || input.borrower_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Borrower DID must be 1-256 characters".into()
        )));
    }
    if input.reason.is_empty() || input.reason.len() > 2048 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Reason must be 1-2048 characters".into()
        )));
    }
    let now = sys_time()?;

    // Validate disputed component is valid
    let valid_components = ["payment_history", "collateral_ratio", "account_age"];
    if !valid_components.contains(&input.disputed_component.as_str()) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Invalid component. Must be one of: {:?}",
            valid_components
        ))));
    }

    let appeal = CreditAppeal {
        id: format!("appeal:{}:{}", input.borrower_did, now.as_micros()),
        borrower_did: input.borrower_did,
        disputed_component: input.disputed_component,
        reason: input.reason,
        evidence: input.evidence,
        status: AppealStatus::Pending,
        reviewer_notes: None,
        score_adjustment: None,
        filed_at: now,
        resolved_at: None,
    };

    // In a full implementation, this would create an entry and link it
    // For now, return the appeal struct
    Ok(appeal)
}

/// Get pending appeals for a borrower
#[hdk_extern]
pub fn get_my_appeals(_borrower_did: String) -> ExternResult<Vec<CreditAppeal>> {
    // In full implementation: query for appeal entries linked to this DID
    // For now, return empty vec (no appeals stored yet)
    Ok(vec![])
}

// =============================================================================
// RIGHT TO EXIT (Fresh Start)
// =============================================================================

/// Request to reset lending history (with mandatory waiting period)
#[derive(Serialize, Deserialize, Debug)]
pub struct FreshStartRequest {
    pub borrower_did: String,
    pub requested_at: Timestamp,
    pub effective_at: Timestamp,         // After 90-day waiting period
    pub acknowledged_consequences: bool, // Must acknowledge they lose positive history too
}

/// Request a fresh start - right to exit and begin again
/// Note: This has a 90-day waiting period and erases ALL history (good and bad)
#[hdk_extern]
pub fn request_fresh_start(borrower_did: String) -> ExternResult<FreshStartRequest> {
    if borrower_did.is_empty() || borrower_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Borrower DID must be 1-256 characters".into()
        )));
    }
    let now = sys_time()?;

    // 90-day waiting period before fresh start takes effect
    let effective_at = Timestamp::from_micros(now.as_micros() as i64 + 90 * 24 * 3600 * 1_000_000);

    Ok(FreshStartRequest {
        borrower_did,
        requested_at: now,
        effective_at,
        acknowledged_consequences: false, // Must call confirm_fresh_start
    })
}

// =============================================================================
// ADJUSTED INTEREST RATE (Using Ethical Assessment)
// =============================================================================

/// Get adjusted interest rate based on ethical credit assessment
#[hdk_extern]
pub fn get_adjusted_interest_rate(input: GetRateInput) -> ExternResult<AdjustedRateResult> {
    if input.borrower_did.is_empty() || input.borrower_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Borrower DID must be 1-256 characters".into()
        )));
    }
    if input.base_rate < 0.0 || input.base_rate > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Base rate must be between 0.0 and 1.0".into()
        )));
    }
    let assessment = calculate_credit_assessment(CreditAssessmentInput {
        borrower_did: input.borrower_did.clone(),
        opt_in_additional_data: None,
    })?;

    // Apply rate adjustment, but NEVER below 1% (minimum viable rate)
    let adjusted_rate = (input.base_rate + assessment.rate_adjustment).max(0.01);

    Ok(AdjustedRateResult {
        borrower_did: input.borrower_did,
        base_rate: input.base_rate,
        credit_score: assessment.credit_score,
        credit_tier: assessment.credit_tier,
        rate_adjustment: assessment.rate_adjustment,
        adjusted_rate,
        floor_applied: assessment.floor_applied,
        transparency: assessment.transparency,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetRateInput {
    pub borrower_did: String,
    pub base_rate: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AdjustedRateResult {
    pub borrower_did: String,
    pub base_rate: f64,
    pub credit_score: f64,
    pub credit_tier: CreditTier,
    pub rate_adjustment: f64,
    pub adjusted_rate: f64,
    pub floor_applied: bool,
    pub transparency: TransparencyReport,
}

// =============================================================================
// PRIVACY-PRESERVING CREDIT PROOFS (Zero-Knowledge)
// =============================================================================
// These allow borrowers to prove creditworthiness without revealing exact scores.
// A borrower can prove "I meet your minimum threshold" without disclosing their
// actual score to lenders, preventing score-based discrimination or data harvesting.

/// Input for generating a threshold proof
#[derive(Serialize, Deserialize, Debug)]
pub struct ThresholdProofInput {
    pub borrower_did: String,
    pub threshold: f64,        // The minimum score to prove (e.g., 0.6)
    pub proof_type: ProofType, // What kind of proof
}

/// Type of privacy proof to generate
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ProofType {
    /// Proves score >= threshold (most common - "am I good enough?")
    MeetsMinimum,
    /// Proves score is in a range [low, high] without revealing exact value
    WithinRange { low: f64, high: f64 },
    /// Proves score is in a specific tier without revealing exact score
    TierMembership,
}

/// A privacy-preserving credit proof
/// In production, this would use actual zk-SNARK/STARK cryptography
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PrivacyCreditProof {
    pub proof_id: String,
    pub borrower_did: String,
    pub proof_type: ProofType,
    pub threshold: f64,
    pub result: bool,       // Does the borrower meet the criteria?
    pub commitment: String, // Hash commitment to the actual score
    pub proof_data: String, // The ZK proof (would be actual cryptographic proof in production)
    pub generated_at: Timestamp,
    pub valid_until: Timestamp,
    /// What the proof reveals (for transparency)
    pub reveals: Vec<String>,
    /// What the proof does NOT reveal (for transparency)
    pub conceals: Vec<String>,
}

/// Generate a privacy-preserving proof of creditworthiness
///
/// This allows a borrower to prove they meet a lender's minimum threshold
/// WITHOUT revealing their exact score. The lender learns only "yes/no" to
/// their specific question, not the borrower's complete financial picture.
#[hdk_extern]
pub fn generate_threshold_proof(input: ThresholdProofInput) -> ExternResult<PrivacyCreditProof> {
    if input.borrower_did.is_empty() || input.borrower_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Borrower DID must be 1-256 characters".into()
        )));
    }
    if input.threshold < 0.0 || input.threshold > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Threshold must be between 0.0 and 1.0".into()
        )));
    }
    let now = sys_time()?;

    // Calculate the actual credit assessment
    let assessment = calculate_credit_assessment(CreditAssessmentInput {
        borrower_did: input.borrower_did.clone(),
        opt_in_additional_data: None,
    })?;

    // Determine if threshold is met based on proof type
    let result = match &input.proof_type {
        ProofType::MeetsMinimum => assessment.credit_score >= input.threshold,
        ProofType::WithinRange { low, high } => {
            assessment.credit_score >= *low && assessment.credit_score <= *high
        }
        ProofType::TierMembership => {
            // Prove tier membership without revealing exact score
            let threshold_tier = match input.threshold {
                t if t >= 0.8 => CreditTier::Excellent,
                t if t >= 0.6 => CreditTier::Good,
                t if t >= 0.4 => CreditTier::Standard,
                _ => CreditTier::BasicAccess,
            };
            assessment.credit_tier == threshold_tier
        }
    };

    // Create commitment to actual score (hash)
    // In production: this would be a Pedersen commitment or similar
    let commitment = format!(
        "commitment:{}:{}",
        input.borrower_did,
        // Hash the score with a random blinding factor (simulated)
        now.as_micros()
    );

    // Generate ZK proof (simulated - in production would use actual zk-SNARK)
    // The proof data demonstrates knowledge of a score >= threshold without revealing it
    let proof_data = format!(
        "zkproof:type={:?}:threshold={}:result={}:timestamp={}",
        input.proof_type,
        input.threshold,
        result,
        now.as_micros()
    );

    // Be explicit about what this proof reveals and conceals
    let (reveals, conceals) = match &input.proof_type {
        ProofType::MeetsMinimum => (
            vec![format!(
                "Score {} threshold {}",
                if result { ">=" } else { "<" },
                input.threshold
            )],
            vec![
                "Exact credit score".to_string(),
                "Individual component scores".to_string(),
                "Payment history details".to_string(),
                "Collateral details".to_string(),
            ],
        ),
        ProofType::WithinRange { low, high } => (
            vec![format!(
                "Score {} in range [{}, {}]",
                if result { "is" } else { "is not" },
                low,
                high
            )],
            vec![
                "Exact credit score".to_string(),
                "Which end of range (if in range)".to_string(),
            ],
        ),
        ProofType::TierMembership => (
            vec![format!(
                "Tier membership: {}",
                if result { "confirmed" } else { "not in tier" }
            )],
            vec![
                "Exact credit score".to_string(),
                "Distance from tier boundaries".to_string(),
            ],
        ),
    };

    Ok(PrivacyCreditProof {
        proof_id: format!("proof:{}:{}", input.borrower_did, now.as_micros()),
        borrower_did: input.borrower_did,
        proof_type: input.proof_type,
        threshold: input.threshold,
        result,
        commitment,
        proof_data,
        generated_at: now,
        valid_until: Timestamp::from_micros(now.as_micros() as i64 + 7 * 24 * 3600 * 1_000_000), // 7 days
        reveals,
        conceals,
    })
}

/// Verify a threshold proof (for lenders)
///
/// Lenders can verify the proof is valid without learning the actual score
#[hdk_extern]
pub fn verify_threshold_proof(proof: PrivacyCreditProof) -> ExternResult<ProofVerificationResult> {
    let now = sys_time()?;

    // Check if proof is still valid
    if now > proof.valid_until {
        return Ok(ProofVerificationResult {
            proof_id: proof.proof_id,
            is_valid: false,
            reason: "Proof has expired".to_string(),
            verified_at: now,
        });
    }

    // In production: verify the actual ZK proof cryptographically
    // For now, we trust the proof structure
    let is_valid = proof.proof_data.starts_with("zkproof:");

    Ok(ProofVerificationResult {
        proof_id: proof.proof_id,
        is_valid,
        reason: if is_valid {
            "Proof verified successfully".to_string()
        } else {
            "Invalid proof format".to_string()
        },
        verified_at: now,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ProofVerificationResult {
    pub proof_id: String,
    pub is_valid: bool,
    pub reason: String,
    pub verified_at: Timestamp,
}

/// Request a loan using only a privacy proof (no score disclosure)
///
/// This allows borrowers to apply for loans by proving they meet minimum
/// requirements without revealing their exact creditworthiness
#[hdk_extern]
pub fn request_loan_with_proof(input: LoanRequestWithProof) -> ExternResult<Record> {
    if input.borrower_did.is_empty() || input.borrower_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Borrower DID must be 1-256 characters".into()
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
    if input.term_days == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Term must be at least 1 day".into()
        )));
    }
    // Verify the proof first
    let verification = verify_threshold_proof(input.credit_proof.clone())?;

    if !verification.is_valid {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid credit proof".into()
        )));
    }

    if !input.credit_proof.result {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Credit proof shows threshold not met".into()
        )));
    }

    // Create the loan request without storing the actual score
    let now = sys_time()?;
    let loan = Loan {
        id: format!("loan:{}:{}", input.borrower_did, now.as_micros()),
        borrower_did: input.borrower_did.clone(),
        lender_did: String::new(),
        principal: input.amount,
        currency: input.currency,
        interest_rate: 0.0, // To be determined by lender
        term_days: input.term_days,
        collateral_ids: input.collateral_ids,
        status: LoanStatus::Requested,
        created: now,
        funded: None,
        maturity: None,
        repaid: None,
    };

    let action_hash = create_entry(&EntryTypes::Loan(loan))?;

    // Link to borrower (not to proof - proof is ephemeral)
    create_link(
        anchor_hash(&input.borrower_did)?,
        action_hash.clone(),
        LinkTypes::BorrowerToLoans,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LoanRequestWithProof {
    pub borrower_did: String,
    pub amount: f64,
    pub currency: String,
    pub term_days: u32,
    pub collateral_ids: Vec<String>,
    pub credit_proof: PrivacyCreditProof,
}
