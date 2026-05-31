// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Finance Bridge Coordinator Zome
//!
//! Cross-hApp communication for credit queries, payment processing,
//! and collateral management across the Mycelix ecosystem.

use finance_bridge_integrity::*;
use hdk::prelude::*;

const FINANCE_HAPP_ID: &str = "mycelix-finance";

/// Helper to create anchor hash from string
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    anchor_str.hash(&mut hasher);
    let h1 = hasher.finish();
    hasher.write_u64(h1);
    let h2 = hasher.finish();
    hasher.write_u64(h2);
    let h3 = hasher.finish();
    hasher.write_u64(h3);
    let h4 = hasher.finish();

    let mut result = [0u8; 32];
    result[0..8].copy_from_slice(&h1.to_le_bytes());
    result[8..16].copy_from_slice(&h2.to_le_bytes());
    result[16..24].copy_from_slice(&h3.to_le_bytes());
    result[24..32].copy_from_slice(&h4.to_le_bytes());

    Ok(EntryHash::from_raw_32(result.to_vec()))
}

/// Query credit score for a DID
#[hdk_extern]
pub fn query_credit(input: QueryCreditInput) -> ExternResult<CreditResult> {
    let now = sys_time()?;

    let query = CreditQuery {
        id: format!(
            "credit:{}:{}:{}",
            input.source_happ,
            input.did,
            now.as_micros()
        ),
        did: input.did.clone(),
        source_happ: input.source_happ.clone(),
        purpose: input.purpose,
        queried_at: now,
    };
    create_entry(&EntryTypes::CreditQuery(query))?;

    // Calculate credit score (would integrate with MATL in production)
    let result = CreditResult {
        id: format!("result:{}:{}", input.did, now.as_micros()),
        did: input.did.clone(),
        matl_score: 0.5, // Would query identity hApp
        credit_score: 0.6,
        payment_history_score: 0.7,
        collateral_ratio: 0.0,
        active_loans: 0,
        total_repaid: 0,
        calculated_at: now,
    };

    let hash = create_entry(&EntryTypes::CreditResult(result.clone()))?;
    create_link(anchor_hash(&input.did)?, hash, LinkTypes::DidToCredit, ())?;

    Ok(result)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct QueryCreditInput {
    pub did: String,
    pub source_happ: String,
    pub purpose: CreditPurpose,
}

/// Process a cross-hApp payment
#[hdk_extern]
pub fn process_payment(input: ProcessPaymentInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let payment = CrossHappPayment {
        id: format!(
            "payment:{}:{}:{}",
            input.source_happ,
            input.from_did,
            now.as_micros()
        ),
        source_happ: input.source_happ.clone(),
        from_did: input.from_did.clone(),
        to_did: input.to_did.clone(),
        amount: input.amount,
        currency: input.currency.clone(),
        reference: input.reference.clone(),
        status: PaymentStatus::Processing,
        created_at: now,
        completed_at: None,
    };

    let hash = create_entry(&EntryTypes::CrossHappPayment(payment))?;

    create_link(
        anchor_hash(&input.from_did)?,
        hash.clone(),
        LinkTypes::DidToPayments,
        (),
    )?;
    create_link(
        anchor_hash(&input.source_happ)?,
        hash.clone(),
        LinkTypes::HappToPayments,
        (),
    )?;

    broadcast_finance_event(BroadcastFinanceEventInput {
        event_type: FinanceEventType::PaymentCompleted,
        subject_did: input.from_did,
        amount: Some(input.amount),
        payload: serde_json::json!({
            "to": input.to_did,
            "currency": input.currency,
            "reference": input.reference,
        })
        .to_string(),
    })?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Payment not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ProcessPaymentInput {
    pub source_happ: String,
    pub from_did: String,
    pub to_did: String,
    pub amount: u64,
    pub currency: String,
    pub reference: String,
}

/// Register collateral from another hApp
#[hdk_extern]
pub fn register_collateral(input: RegisterCollateralInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let collateral = CollateralRegistration {
        id: format!(
            "collateral:{}:{}:{}",
            input.owner_did,
            input.asset_id,
            now.as_micros()
        ),
        owner_did: input.owner_did.clone(),
        asset_type: input.asset_type,
        asset_id: input.asset_id,
        source_happ: input.source_happ,
        value_estimate: input.value_estimate,
        currency: input.currency,
        status: CollateralStatus::Available,
        registered_at: now,
    };

    let hash = create_entry(&EntryTypes::CollateralRegistration(collateral))?;

    create_link(
        anchor_hash("collateral_registry")?,
        hash.clone(),
        LinkTypes::CollateralRegistry,
        (),
    )?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Collateral not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterCollateralInput {
    pub owner_did: String,
    pub source_happ: String,
    pub asset_type: AssetType,
    pub asset_id: String,
    pub value_estimate: u64,
    pub currency: String,
}

/// Broadcast finance event
#[hdk_extern]
pub fn broadcast_finance_event(input: BroadcastFinanceEventInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let event = FinanceBridgeEvent {
        id: format!("event:{:?}:{}", input.event_type, now.as_micros()),
        event_type: input.event_type,
        subject_did: input.subject_did,
        amount: input.amount,
        payload: input.payload,
        source_happ: FINANCE_HAPP_ID.to_string(),
        timestamp: now,
    };

    let hash = create_entry(&EntryTypes::FinanceBridgeEvent(event))?;

    create_link(
        anchor_hash("recent_events")?,
        hash.clone(),
        LinkTypes::RecentEvents,
        (),
    )?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Event not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BroadcastFinanceEventInput {
    pub event_type: FinanceEventType,
    pub subject_did: String,
    pub amount: Option<u64>,
    pub payload: String,
}

/// Get payment history for a DID
#[hdk_extern]
pub fn get_payment_history(did: String) -> ExternResult<Vec<Record>> {
    let query = LinkQuery::try_new(anchor_hash(&did)?, LinkTypes::DidToPayments)?;
    let links = get_links(query, GetStrategy::default())?;

    let mut payments = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            payments.push(record);
        }
    }

    Ok(payments)
}
