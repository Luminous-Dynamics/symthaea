//! Finance Bridge Coordinator Zome
//!
//! Cross-hApp communication for payment processing, collateral management,
//! and collateral bridge deposits across the Mycelix ecosystem.

use hdk::prelude::*;
use finance_bridge_integrity::*;
use mycelix_finance_shared::{anchor_hash, verify_caller_is_did};

const FINANCE_HAPP_ID: &str = "mycelix-finance";

/// Maximum percentage of vault that any single member can deposit/redeem per day
const DAILY_RATE_LIMIT_PCT: f64 = 0.05; // 5%

/// 24 hours in microseconds
const DAY_MICROS: i64 = 24 * 60 * 60 * 1_000_000;

/// Process a cross-hApp payment
#[hdk_extern]
pub fn process_payment(input: ProcessPaymentInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let payment = CrossHappPayment {
        id: format!("payment:{}:{}:{}", input.source_happ, input.from_did, now.as_micros()),
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
        }).to_string(),
    })?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Payment not found".into())))
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
        id: format!("collateral:{}:{}:{}", input.owner_did, input.asset_id, now.as_micros()),
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

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Collateral not found".into())))
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

// ---------------------------------------------------------------------------
// Collateral Bridge Deposit functions
// ---------------------------------------------------------------------------

/// Deposit collateral to mint SAP.
/// Creates a CollateralBridgeDeposit entry recording the collateral-to-SAP conversion.
///
/// Rate-limited: max 5% of total vault value per day per member.
#[hdk_extern]
pub fn deposit_collateral(input: DepositCollateralInput) -> ExternResult<Record> {
    verify_caller_is_did(&input.depositor_did)?;
    let now = sys_time()?;

    let sap_minted = (input.collateral_amount as f64 * input.oracle_rate) as u64;

    // Enforce rate limit: max 5% of vault per day per member
    enforce_rate_limit(&input.depositor_did, sap_minted, now)?;

    let deposit = CollateralBridgeDeposit {
        id: format!("deposit:{}:{}:{}", input.depositor_did, input.collateral_type, now.as_micros()),
        depositor_did: input.depositor_did.clone(),
        collateral_type: input.collateral_type.clone(),
        collateral_amount: input.collateral_amount,
        sap_minted,
        oracle_rate: input.oracle_rate,
        status: BridgeDepositStatus::Pending,
        created_at: now,
        completed_at: None,
    };

    let hash = create_entry(&EntryTypes::CollateralBridgeDeposit(deposit))?;

    create_link(
        anchor_hash(&input.depositor_did)?,
        hash.clone(),
        LinkTypes::DidToDeposits,
        (),
    )?;

    // Broadcast the deposit event
    broadcast_finance_event(BroadcastFinanceEventInput {
        event_type: FinanceEventType::CollateralDeposited,
        subject_did: input.depositor_did,
        amount: Some(sap_minted),
        payload: serde_json::json!({
            "collateral_type": input.collateral_type,
            "collateral_amount": input.collateral_amount,
            "oracle_rate": input.oracle_rate,
            "sap_minted": sap_minted,
        }).to_string(),
    })?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Deposit not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DepositCollateralInput {
    pub depositor_did: String,
    pub collateral_type: String,  // "ETH" or "USDC"
    pub collateral_amount: u64,
    pub oracle_rate: f64,
}

/// Redeem collateral by marking a deposit as redeemed (SAP returned, collateral released).
///
/// Rate-limited: max 5% of total vault value per day per member.
#[hdk_extern]
pub fn redeem_collateral(deposit_id: String) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::CollateralBridgeDeposit)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(deposit) = record.entry().to_app_option::<CollateralBridgeDeposit>().ok().flatten() {
            if deposit.id == deposit_id {
                if deposit.status != BridgeDepositStatus::Confirmed {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only confirmed deposits can be redeemed".into()
                    )));
                }

                // Enforce rate limit on redemption
                let now = sys_time()?;
                enforce_rate_limit(&deposit.depositor_did, deposit.sap_minted, now)?;

                let now = sys_time()?;
                let redeemed = CollateralBridgeDeposit {
                    status: BridgeDepositStatus::Redeemed,
                    completed_at: Some(now),
                    ..deposit.clone()
                };

                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::CollateralBridgeDeposit(redeemed),
                )?;

                // Broadcast the redemption event
                broadcast_finance_event(BroadcastFinanceEventInput {
                    event_type: FinanceEventType::CollateralRedeemed,
                    subject_did: deposit.depositor_did,
                    amount: Some(deposit.sap_minted),
                    payload: serde_json::json!({
                        "collateral_type": deposit.collateral_type,
                        "collateral_amount": deposit.collateral_amount,
                        "sap_returned": deposit.sap_minted,
                    }).to_string(),
                })?;

                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Deposit not found".into())))
}

// ---------------------------------------------------------------------------
// Rate Limiting Helpers
// ---------------------------------------------------------------------------

/// Enforce the daily rate limit: no member may deposit/redeem more than
/// 5% of total vault value in any rolling 24-hour period.
///
/// Vault value = sum of `sap_minted` for all Confirmed deposits.
/// Daily activity = sum of `sap_minted` for this member's deposits/redemptions
/// created within the last 24 hours.
fn enforce_rate_limit(member_did: &str, new_amount: u64, now: Timestamp) -> ExternResult<()> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::CollateralBridgeDeposit)?))
        .include_entries(true);

    let deposits: Vec<CollateralBridgeDeposit> = query(filter)?
        .into_iter()
        .filter_map(|r| r.entry().to_app_option::<CollateralBridgeDeposit>().ok().flatten())
        .collect();

    // Total vault = sum of all confirmed deposit SAP
    let vault_total: u64 = deposits.iter()
        .filter(|d| d.status == BridgeDepositStatus::Confirmed)
        .map(|d| d.sap_minted)
        .sum();

    // If vault is empty, allow the first deposit (bootstrap case)
    if vault_total == 0 {
        return Ok(());
    }

    let daily_limit = (vault_total as f64 * DAILY_RATE_LIMIT_PCT) as u64;

    // Sum this member's activity in the last 24 hours
    let cutoff = now.as_micros() - DAY_MICROS;
    let daily_activity: u64 = deposits.iter()
        .filter(|d| d.depositor_did == member_did && d.created_at.as_micros() > cutoff)
        .map(|d| d.sap_minted)
        .sum();

    if daily_activity + new_amount > daily_limit {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Rate limit exceeded: max {} SAP/day (5% of vault {}). Already used {} today, requesting {}.",
            daily_limit, vault_total, daily_activity, new_amount
        ))));
    }

    Ok(())
}
