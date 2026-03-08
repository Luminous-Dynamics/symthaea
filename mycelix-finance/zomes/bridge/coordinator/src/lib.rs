//! Finance Bridge Coordinator Zome
//!
//! Cross-hApp communication for payment processing, collateral management,
//! and collateral bridge deposits across the Mycelix ecosystem.

use finance_bridge_integrity::*;
use hdk::prelude::*;
use mycelix_finance_shared::{anchor_hash, verify_caller_is_did};
use mycelix_finance_types::{FeeTier, TendLimitTier};

const FINANCE_HAPP_ID: &str = "mycelix-finance";

/// Maximum percentage of vault that any single member can deposit/redeem per day
const DAILY_RATE_LIMIT_PCT: f64 = 0.05; // 5%

/// 24 hours in microseconds
const DAY_MICROS: i64 = 24 * 60 * 60 * 1_000_000;

/// Process a cross-hApp payment
#[hdk_extern]
pub fn process_payment(input: ProcessPaymentInput) -> ExternResult<Record> {
    verify_caller_is_did(&input.from_did)?;
    if input.currency != "SAP" {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cross-hApp payments must use SAP currency".into()
        )));
    }
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
    verify_caller_is_did(&input.owner_did)?;
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
    verify_caller_is_did(&input.subject_did)?;
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

/// Input for paginated payment history queries
#[derive(Serialize, Deserialize, Debug)]
pub struct GetPaymentHistoryInput {
    pub did: String,
    pub limit: Option<usize>,
}

/// Get payment history for a DID (paginated, default limit 100)
#[hdk_extern]
pub fn get_payment_history(input: GetPaymentHistoryInput) -> ExternResult<Vec<Record>> {
    let limit = input.limit.unwrap_or(100);
    let query = LinkQuery::try_new(anchor_hash(&input.did)?, LinkTypes::DidToPayments)?;
    let links = get_links(query, GetStrategy::default())?;

    let mut payments = Vec::new();
    for link in links.into_iter().take(limit) {
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

    if input.oracle_rate.is_nan() || input.oracle_rate.is_infinite() || input.oracle_rate <= 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Oracle rate must be a finite positive number".into()
        )));
    }
    if input.collateral_amount == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Collateral amount must be greater than zero".into()
        )));
    }
    if input.collateral_type != "ETH" && input.collateral_type != "USDC" {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Collateral type must be \"ETH\" or \"USDC\"".into()
        )));
    }

    let sap_minted = (input.collateral_amount as f64 * input.oracle_rate) as u64;

    // Enforce rate limit: max 5% of vault per day per member
    enforce_rate_limit(&input.depositor_did, sap_minted, now)?;

    let deposit_id = format!(
        "deposit:{}:{}:{}",
        input.depositor_did,
        input.collateral_type,
        now.as_micros()
    );
    let deposit = CollateralBridgeDeposit {
        id: deposit_id.clone(),
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

    // Index by deposit ID for O(1) lookup
    create_link(
        anchor_hash(&deposit_id)?,
        hash.clone(),
        LinkTypes::DepositIdToDeposit,
        (),
    )?;

    // Credit minted SAP to depositor's balance via payments zome
    #[derive(Serialize, Debug)]
    struct CreditSapPayload {
        member_did: String,
        amount: u64,
        reason: String,
    }
    match call(
        CallTargetCell::Local,
        ZomeName::from("payments"),
        FunctionName::from("credit_sap"),
        None,
        CreditSapPayload {
            member_did: input.depositor_did.clone(),
            amount: sap_minted,
            reason: format!(
                "Collateral bridge deposit: {} {}",
                input.collateral_amount, input.collateral_type
            ),
        },
    ) {
        Ok(ZomeCallResponse::Ok(_)) => {}
        Ok(other) => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to credit SAP: unexpected response {:?}",
                other
            ))));
        }
        Err(e) => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Failed to credit SAP for deposit: {:?}",
                e
            ))));
        }
    }

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
        })
        .to_string(),
    })?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Deposit not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DepositCollateralInput {
    pub depositor_did: String,
    pub collateral_type: String, // "ETH" or "USDC"
    pub collateral_amount: u64,
    pub oracle_rate: f64,
}

/// Confirm a pending deposit after collateral has been verified.
///
/// Only the original depositor can confirm. Transitions Pending -> Confirmed.
/// Uses link-based indexing for O(1) lookup instead of chain scan.
#[hdk_extern]
pub fn confirm_deposit(deposit_id: String) -> ExternResult<Record> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&deposit_id)?, LinkTypes::DepositIdToDeposit)?,
        GetStrategy::default(),
    )?;
    let link = links.first().ok_or(wasm_error!(WasmErrorInner::Guest(
        "Deposit not found".into()
    )))?;
    let hash = ActionHash::try_from(link.target.clone())
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
    let record = get(hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Deposit not found".into())
    ))?;
    let deposit = record
        .entry()
        .to_app_option::<CollateralBridgeDeposit>()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Deserialization error: {:?}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Deposit entry missing".into()
        )))?;

    verify_caller_is_did(&deposit.depositor_did)?;

    if deposit.status != BridgeDepositStatus::Pending {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Deposit is {:?}, only Pending deposits can be confirmed",
            deposit.status
        ))));
    }

    let now = sys_time()?;
    let confirmed = CollateralBridgeDeposit {
        status: BridgeDepositStatus::Confirmed,
        completed_at: Some(now),
        ..deposit
    };

    let action_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::CollateralBridgeDeposit(confirmed),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

/// Redeem collateral by marking a deposit as redeemed (SAP returned, collateral released).
///
/// Rate-limited: max 5% of total vault value per day per member.
/// Uses link-based indexing for O(1) lookup instead of chain scan.
#[hdk_extern]
pub fn redeem_collateral(deposit_id: String) -> ExternResult<Record> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&deposit_id)?, LinkTypes::DepositIdToDeposit)?,
        GetStrategy::default(),
    )?;
    let link = links.first().ok_or(wasm_error!(WasmErrorInner::Guest(
        "Deposit not found".into()
    )))?;
    let hash = ActionHash::try_from(link.target.clone())
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
    let record = get(hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Deposit not found".into())
    ))?;
    let deposit = record
        .entry()
        .to_app_option::<CollateralBridgeDeposit>()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Deserialization error: {:?}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Deposit entry missing".into()
        )))?;

    verify_caller_is_did(&deposit.depositor_did)?;

    if deposit.status != BridgeDepositStatus::Confirmed {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only confirmed deposits can be redeemed".into()
        )));
    }

    // Enforce rate limit on redemption
    let now = sys_time()?;
    enforce_rate_limit(&deposit.depositor_did, deposit.sap_minted, now)?;

    let redeemed = CollateralBridgeDeposit {
        status: BridgeDepositStatus::Redeemed,
        completed_at: Some(now),
        ..deposit.clone()
    };

    let action_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::CollateralBridgeDeposit(redeemed),
    )?;

    // Debit SAP from depositor's balance via payments zome
    #[derive(Serialize, Debug)]
    struct DebitSapPayload {
        member_did: String,
        amount: u64,
        reason: String,
    }
    if let Err(e) = call(
        CallTargetCell::Local,
        ZomeName::from("payments"),
        FunctionName::from("debit_sap"),
        None,
        DebitSapPayload {
            member_did: deposit.depositor_did.clone(),
            amount: deposit.sap_minted,
            reason: format!("Collateral bridge redemption: {}", deposit.collateral_type),
        },
    ) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot redeem: failed to debit SAP — {:?}",
            e
        ))));
    }

    // Broadcast the redemption event
    broadcast_finance_event(BroadcastFinanceEventInput {
        event_type: FinanceEventType::CollateralRedeemed,
        subject_did: deposit.depositor_did,
        amount: Some(deposit.sap_minted),
        payload: serde_json::json!({
            "collateral_type": deposit.collateral_type,
            "collateral_amount": deposit.collateral_amount,
            "sap_returned": deposit.sap_minted,
        })
        .to_string(),
    })?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

// ---------------------------------------------------------------------------
// Cross-Cluster Wiring: MYCEL → Fee Tier → TEND Limits
// ---------------------------------------------------------------------------

/// Get a member's fee tier based on their MYCEL score.
///
/// This is the cross-cluster bridge function that other hApps call to determine
/// what fee rate a member should pay. Fetches MYCEL from recognition zome.
#[hdk_extern]
pub fn get_member_fee_tier(member_did: String) -> ExternResult<FeeTierResponse> {
    let mycel_score = fetch_mycel_score(&member_did);
    let tier = FeeTier::from_mycel(mycel_score);
    Ok(FeeTierResponse {
        member_did,
        mycel_score,
        tier_name: format!("{:?}", tier),
        base_fee_rate: tier.base_fee_rate(),
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FeeTierResponse {
    pub member_did: String,
    pub mycel_score: f64,
    pub tier_name: String,
    pub base_fee_rate: f64,
}

/// Get a member's effective TEND limit based on current network vitality.
///
/// Cross-cluster bridge function: reads oracle state from tend zome,
/// computes the TEND limit tier, and returns the effective balance limit.
#[hdk_extern]
pub fn get_member_tend_limit(member_did: String) -> ExternResult<TendLimitResponse> {
    // Fetch current vitality from tend oracle
    let vitality = fetch_oracle_vitality();
    let tier = TendLimitTier::from_vitality(vitality);
    let limit = tier.limit();

    Ok(TendLimitResponse {
        member_did,
        vitality,
        tier_name: format!("{:?}", tier),
        effective_limit: limit,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TendLimitResponse {
    pub member_did: String,
    pub vitality: u32,
    pub tier_name: String,
    pub effective_limit: i32,
}

/// Fetch MYCEL score via cross-zome call to recognition.
/// Falls back to 0.0 (Newcomer tier) if unavailable.
fn fetch_mycel_score(member_did: &str) -> f64 {
    match call(
        CallTargetCell::Local,
        ZomeName::from("recognition"),
        FunctionName::from("get_mycel_score"),
        None,
        member_did.to_string(),
    ) {
        Ok(ZomeCallResponse::Ok(result)) => {
            #[derive(Debug, Deserialize)]
            struct MycelState {
                mycel_score: f64,
            }
            match result.decode::<MycelState>() {
                Ok(state) if state.mycel_score.is_finite() => state.mycel_score,
                _ => 0.0,
            }
        }
        _ => 0.0,
    }
}

/// Fetch current oracle vitality via cross-zome call to tend.
/// Falls back to 50 (Normal tier) if unavailable.
fn fetch_oracle_vitality() -> u32 {
    match call(
        CallTargetCell::Local,
        ZomeName::from("tend"),
        FunctionName::from("get_oracle_state"),
        None,
        (),
    ) {
        Ok(ZomeCallResponse::Ok(result)) => {
            #[derive(Debug, Deserialize)]
            struct OracleResp {
                vitality: u32,
            }
            match result.decode::<OracleResp>() {
                Ok(state) => state.vitality,
                _ => 50,
            }
        }
        _ => 50,
    }
}

// ---------------------------------------------------------------------------
// Community Membership Queries
// ---------------------------------------------------------------------------

/// Get the number of members in a community/DAO.
///
/// Used by the currency-mint governance gate to determine whether a community
/// needs a governance proposal to create a currency (>10 members).
/// Queries the identity/governance cluster via cross-role call.
#[hdk_extern]
pub fn get_community_member_count(dao_did: String) -> ExternResult<u32> {
    // Try cross-cluster call to governance for membership roster
    match call(
        CallTargetCell::OtherRole("governance".into()),
        ZomeName::from("governance_bridge"),
        FunctionName::from("get_dao_member_count"),
        None,
        dao_did.clone(),
    ) {
        Ok(ZomeCallResponse::Ok(result)) => Ok(result.decode::<u32>().unwrap_or(0)),
        _ => {
            // Governance cluster unreachable — return 0 (permissive)
            Ok(0)
        }
    }
}

// ---------------------------------------------------------------------------
// Cross-Cluster Dispatch Handlers (incoming calls from other clusters)
// ---------------------------------------------------------------------------

/// Query a member's SAP balance.
///
/// Called by other clusters (commons, civic, personal) via
/// `call(CallTargetCell::OtherRole("finance"), "finance_bridge", "query_sap_balance", ..)`
#[hdk_extern]
pub fn query_sap_balance(member_did: String) -> ExternResult<BalanceResponse> {
    match call(
        CallTargetCell::Local,
        ZomeName::from("payments"),
        FunctionName::from("get_sap_balance"),
        None,
        member_did.clone(),
    ) {
        Ok(ZomeCallResponse::Ok(result)) => {
            #[derive(Debug, Deserialize)]
            struct SapBalance {
                balance: u64,
            }
            let balance = result
                .decode::<SapBalance>()
                .map(|b| b.balance)
                .unwrap_or(0);
            Ok(BalanceResponse {
                member_did,
                currency: "SAP".into(),
                balance,
                available: true,
            })
        }
        _ => Ok(BalanceResponse {
            member_did,
            currency: "SAP".into(),
            balance: 0,
            available: false,
        }),
    }
}

/// Query a member's TEND balance.
///
/// Called by other clusters to check mutual credit standing.
#[hdk_extern]
pub fn query_tend_balance(member_did: String) -> ExternResult<TendBalanceResponse> {
    match call(
        CallTargetCell::Local,
        ZomeName::from("tend"),
        FunctionName::from("get_balance"),
        None,
        member_did.clone(),
    ) {
        Ok(ZomeCallResponse::Ok(result)) => {
            #[derive(Debug, Deserialize)]
            struct TendBalance {
                balance: i32,
            }
            let balance = result
                .decode::<TendBalance>()
                .map(|b| b.balance)
                .unwrap_or(0);
            let tier = fetch_mycel_score(&member_did);
            Ok(TendBalanceResponse {
                member_did,
                balance,
                mycel_score: tier,
                available: true,
            })
        }
        _ => Ok(TendBalanceResponse {
            member_did,
            balance: 0,
            mycel_score: 0.0,
            available: false,
        }),
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BalanceResponse {
    pub member_did: String,
    pub currency: String,
    pub balance: u64,
    pub available: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TendBalanceResponse {
    pub member_did: String,
    pub balance: i32,
    pub mycel_score: f64,
    pub available: bool,
}

/// Get a unified financial summary for a member across all currencies.
///
/// Called by personal bridge, governance, or other clusters to get
/// a complete financial picture for consciousness gating or governance decisions.
#[hdk_extern]
pub fn get_finance_summary(member_did: String) -> ExternResult<FinanceSummaryResponse> {
    let sap = query_sap_balance(member_did.clone())?;
    let tend = query_tend_balance(member_did.clone())?;
    let fee_tier = get_member_fee_tier(member_did.clone())?;
    let tend_limit = get_member_tend_limit(member_did.clone())?;

    Ok(FinanceSummaryResponse {
        member_did,
        sap_balance: sap.balance,
        tend_balance: tend.balance,
        mycel_score: tend.mycel_score,
        fee_tier: fee_tier.tier_name,
        fee_rate: fee_tier.base_fee_rate,
        tend_limit: tend_limit.effective_limit,
        tend_tier: tend_limit.tier_name,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FinanceSummaryResponse {
    pub member_did: String,
    pub sap_balance: u64,
    pub tend_balance: i32,
    pub mycel_score: f64,
    pub fee_tier: String,
    pub fee_rate: f64,
    pub tend_limit: i32,
    pub tend_tier: String,
}

/// Health check for the finance bridge.
///
/// Standard bridge health endpoint for cross-cluster monitoring.
#[hdk_extern]
pub fn health_check(_: ()) -> ExternResult<FinanceBridgeHealth> {
    let agent = agent_info()?.agent_initial_pubkey;
    Ok(FinanceBridgeHealth {
        healthy: true,
        agent: format!("did:mycelix:{}", agent),
        zomes: vec![
            "payments".into(),
            "treasury".into(),
            "tend".into(),
            "staking".into(),
            "recognition".into(),
            "currency_mint".into(),
        ],
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FinanceBridgeHealth {
    pub healthy: bool,
    pub agent: String,
    pub zomes: Vec<String>,
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
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::CollateralBridgeDeposit,
        )?))
        .include_entries(true);

    let deposits: Vec<CollateralBridgeDeposit> = query(filter)?
        .into_iter()
        .filter_map(|r| {
            r.entry()
                .to_app_option::<CollateralBridgeDeposit>()
                .ok()
                .flatten()
        })
        .collect();

    // Total vault = sum of all confirmed deposit SAP
    let vault_total: u64 = deposits
        .iter()
        .filter(|d| d.status == BridgeDepositStatus::Confirmed)
        .map(|d| d.sap_minted)
        .fold(0u64, |acc, v| acc.saturating_add(v));

    // If vault is empty, allow the first deposit (bootstrap case)
    if vault_total == 0 {
        return Ok(());
    }

    let daily_limit = (vault_total as f64 * DAILY_RATE_LIMIT_PCT) as u64;

    // Sum this member's activity in the last 24 hours
    let cutoff = now.as_micros() - DAY_MICROS;
    let daily_activity: u64 = deposits
        .iter()
        .filter(|d| d.depositor_did == member_did && d.created_at.as_micros() > cutoff)
        .map(|d| d.sap_minted)
        .fold(0u64, |acc, v| acc.saturating_add(v));

    if daily_activity.saturating_add(new_amount) > daily_limit {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Rate limit exceeded: max {} SAP/day (5% of vault {}). Already used {} today, requesting {}.",
            daily_limit, vault_total, daily_activity, new_amount
        ))));
    }

    Ok(())
}
