#![deny(unsafe_code)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Finance Bridge Coordinator Zome
//!
//! Cross-hApp communication for payment processing, collateral management,
//! and collateral bridge deposits across the Mycelix ecosystem.

use finance_bridge_integrity::*;
use finance_wire_types::{
    BalanceResponse, DepositCollateralInput, FeeTierResponse, FinanceBridgeHealth,
    FinanceRuntimeDiscovery, GetPaymentHistoryInput, ProcessPaymentInput, RegisterCollateralInput,
    UpdateCollateralHealthInput,
};
use hdk::prelude::*;
use mycelix_finance_shared::{
    anchor_hash, follow_update_chain, verify_caller_is_did, verify_citizen_tier,
    verify_participant_tier,
};
use mycelix_finance_types::{FeeTier, TendLimitTier};

const FINANCE_HAPP_ID: &str = "mycelix-finance";

/// When true, cross-cluster bridge calls that fail to reach the governance
/// cluster will return errors instead of permissive defaults.
///
/// **Security tradeoff**:
/// - `false` (default): Operations proceed when governance is unreachable
///   (bootstrap, standalone, network partition). This prioritizes availability
///   — the integrity zome still enforces zero-sum and constitutional limits.
/// - `true`: All governance-dependent operations fail-closed when the governance
///   cluster is unreachable. This is stricter but can block legitimate operations
///   during network partitions or bootstrapping.
///
/// Set to `true` for high-security deployments where governance availability
/// is guaranteed and any gap in oversight is unacceptable.
///
/// SECURITY: Changed from `false` to `true` — financial operations must not
/// proceed without governance verification. The permissive default allowed
/// currency creation and proposal verification to bypass governance during
/// network partitions, which is unacceptable for production deployments.
const STRICT_GOVERNANCE_MODE: bool = true;

/// 24 hours in microseconds
const DAY_MICROS: i64 = 24 * 60 * 60 * 1_000_000;

/// Process a cross-hApp payment
#[hdk_extern]
pub fn process_payment(input: ProcessPaymentInput) -> ExternResult<Record> {
    verify_participant_tier()?;
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

/// Register collateral from another hApp
#[hdk_extern]
pub fn register_collateral(input: RegisterCollateralInput) -> ExternResult<Record> {
    verify_participant_tier()?;
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
        asset_type: map_wire_asset_type(input.asset_type),
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
pub struct UpdateCollateralStatusInput {
    pub collateral_id: String,
    pub owner_did: String,
    pub new_status: CollateralStatus,
}

/// Update collateral status (e.g., Available → Pledged, Pledged → Frozen).
///
/// Covenant enforcement: transitioning to `Pledged` or `Frozen` is blocked if
/// active covenants exist that would conflict with the status change.
/// Transitioning to `Released` is also blocked by active covenants
/// (use `redeem_collateral` for the full release flow).
#[hdk_extern]
pub fn update_collateral_status(input: UpdateCollateralStatusInput) -> ExternResult<Record> {
    verify_participant_tier()?;
    verify_caller_is_did(&input.owner_did)?;

    // Find the collateral registration
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash("collateral_registry")?,
            LinkTypes::CollateralRegistry,
        )?,
        GetStrategy::default(),
    )?;

    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        let record = follow_update_chain(hash)?;
        if let Ok(Some(collateral)) = record.entry().to_app_option::<CollateralRegistration>() {
            if collateral.id == input.collateral_id {
                // Verify ownership
                if collateral.owner_did != input.owner_did {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only the collateral owner can change its status".into()
                    )));
                }

                // Covenant enforcement: block Pledged, Frozen, and Released transitions
                // when active covenants exist on this collateral
                match input.new_status {
                    CollateralStatus::Pledged
                    | CollateralStatus::Frozen
                    | CollateralStatus::Released => {
                        enforce_no_active_covenants(&collateral.id)?;
                    }
                    CollateralStatus::Available => {
                        // Returning to Available is always permitted
                    }
                }

                let updated = CollateralRegistration {
                    status: input.new_status,
                    ..collateral
                };

                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::CollateralRegistration(updated),
                )?;

                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest(format!(
        "Collateral {} not found",
        input.collateral_id
    ))))
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
    verify_participant_tier()?;
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

    // Phase 1b: Verify oracle rate against consensus (oracle rate attestation)
    verify_oracle_rate_against_consensus(&input.collateral_type, input.oracle_rate)?;

    let sap_minted = (input.collateral_amount as f64 * input.oracle_rate) as u64;

    // Tier-scaled daily rate limit: higher consciousness tiers get larger limits
    let mycel_score = fetch_mycel_score(&input.depositor_did);
    let tier = FeeTier::from_mycel(mycel_score);
    let daily_limit_pct = match tier {
        FeeTier::Newcomer => 1, // 1% for newcomers (shouldn't reach here due to tier gate, but defense in depth)
        FeeTier::Member => 5,   // 5% for members
        FeeTier::Steward => 10, // 10% for stewards
    };

    // Enforce rate limit: max daily_limit_pct% of vault per day per member
    enforce_rate_limit(&input.depositor_did, sap_minted, now, daily_limit_pct)?;

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
    let record = follow_update_chain(hash)?;
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
    verify_participant_tier()?;
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&deposit_id)?, LinkTypes::DepositIdToDeposit)?,
        GetStrategy::default(),
    )?;
    let link = links.first().ok_or(wasm_error!(WasmErrorInner::Guest(
        "Deposit not found".into()
    )))?;
    let hash = ActionHash::try_from(link.target.clone())
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
    let record = follow_update_chain(hash)?;
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

    // Covenant enforcement: block redemption if active covenants exist on this deposit
    enforce_no_active_covenants(&deposit.id)?;

    // Enforce tier-scaled rate limit on redemption
    let now = sys_time()?;
    let mycel_score = fetch_mycel_score(&deposit.depositor_did);
    let redeem_tier = FeeTier::from_mycel(mycel_score);
    let redeem_daily_limit_pct = match redeem_tier {
        FeeTier::Newcomer => 1,
        FeeTier::Member => 5,
        FeeTier::Steward => 10,
    };
    enforce_rate_limit(
        &deposit.depositor_did,
        deposit.sap_minted,
        now,
        redeem_daily_limit_pct,
    )?;

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
    match call(
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
        Ok(ZomeCallResponse::Ok(_)) => {}
        Ok(other) => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Cannot redeem: failed to debit SAP — unexpected response {:?}",
                other
            ))));
        }
        Err(e) => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Cannot redeem: failed to debit SAP — {:?}",
                e
            ))));
        }
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
                Ok(state) => {
                    debug!(
                        "fetch_mycel_score: non-finite MYCEL score {:?} for {}, defaulting to 0.0",
                        state.mycel_score, member_did
                    );
                    0.0
                }
                Err(e) => {
                    debug!(
                        "fetch_mycel_score: decode error for {}: {:?}, defaulting to 0.0",
                        member_did, e
                    );
                    0.0
                }
            }
        }
        Ok(other) => {
            debug!(
                "fetch_mycel_score: recognition zome returned {:?} for {}, defaulting to 0.0",
                other, member_did
            );
            0.0
        }
        Err(e) => {
            debug!(
                "fetch_mycel_score: recognition zome unreachable for {}: {:?}, defaulting to 0.0",
                member_did, e
            );
            0.0
        }
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
                Err(e) => {
                    debug!(
                        "fetch_oracle_vitality: decode error: {:?}, defaulting to 50 (Normal tier)",
                        e
                    );
                    50
                }
            }
        }
        Ok(other) => {
            debug!(
                "fetch_oracle_vitality: tend zome returned {:?}, defaulting to 50 (Normal tier)",
                other
            );
            50
        }
        Err(e) => {
            debug!("fetch_oracle_vitality: tend zome unreachable: {:?}, defaulting to 50 (Normal tier)", e);
            50
        }
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
        Ok(other) => {
            // SECURITY NOTE: Returning 0 members is PERMISSIVE — it means the governance
            // proposal requirement (>10 members) will be skipped. This is deliberate:
            // when the governance cluster is unreachable (bootstrap, standalone, or network
            // partition), we allow small-community operations to proceed rather than blocking
            // all currency creation/amendment. The integrity zome still enforces zero-sum
            // and constitutional limits regardless of governance gate.
            //
            // When STRICT_GOVERNANCE_MODE is true, this returns an error instead,
            // blocking the operation until governance is reachable.
            if STRICT_GOVERNANCE_MODE {
                return Err(wasm_error!(WasmErrorInner::Guest(format!(
                    "Circuit breaker: governance cluster unavailable for {}, operation suspended: {:?}",
                    dao_did, other
                ))));
            }
            debug!("get_community_member_count: governance returned {:?} for {}, defaulting to 0 (permissive)", other, dao_did);
            Ok(0)
        }
        Err(e) => {
            // SECURITY NOTE: Same permissive default as above — see comment.
            // When STRICT_GOVERNANCE_MODE is true, fail-closed instead.
            if STRICT_GOVERNANCE_MODE {
                return Err(wasm_error!(WasmErrorInner::Guest(format!(
                    "Circuit breaker: governance cluster unreachable for {}, operation suspended: {:?}",
                    dao_did, e
                ))));
            }
            debug!("get_community_member_count: governance unreachable for {}: {:?}, defaulting to 0 (permissive)", dao_did, e);
            Ok(0)
        }
    }
}

/// Verify that a governance proposal exists and is in Approved/Executed state.
///
/// Used by currency-mint to validate that governance_proposal_id references a real
/// proposal before creating/amending currencies. Returns true if the proposal is
/// valid, false if not found or not approved.
///
/// **Fallback behavior** depends on `STRICT_GOVERNANCE_MODE`:
/// - `false` (default): Returns `true` when governance is unreachable (permissive).
/// - `true`: Returns `false` when governance is unreachable (fail-closed),
///   blocking any operation that requires a governance proposal until the
///   governance cluster is available.
#[hdk_extern]
pub fn verify_governance_proposal(proposal_id: String) -> ExternResult<bool> {
    match call(
        CallTargetCell::OtherRole("governance".into()),
        ZomeName::from("governance_bridge"),
        FunctionName::from("get_proposal_status"),
        None,
        proposal_id.clone(),
    ) {
        Ok(ZomeCallResponse::Ok(result)) => {
            // Expect a string status like "Approved", "Executed", "Pending", "Rejected"
            let status = result.decode::<String>().unwrap_or_default();
            Ok(status == "Approved" || status == "Executed")
        }
        Ok(_other) => {
            // Governance returned non-Ok — proposal likely doesn't exist
            Ok(false)
        }
        Err(e) => {
            // Circuit breaker: When governance cluster is unreachable, behavior depends
            // on STRICT_GOVERNANCE_MODE. In strict mode we fail-closed (return error),
            // blocking operations that need governance approval. In permissive mode
            // we return true, relying on local verify_governance_agent as a fallback.
            if STRICT_GOVERNANCE_MODE {
                return Err(wasm_error!(WasmErrorInner::Guest(format!(
                    "Circuit breaker: governance cluster unreachable for proposal {}, operation suspended: {:?}",
                    proposal_id, e
                ))));
            }
            debug!(
                "verify_governance_proposal: governance unreachable for {}, defaulting to true (permissive): {:?}",
                proposal_id, e
            );
            Ok(true)
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
        Ok(other) => {
            debug!(
                "query_sap_balance: payments zome returned {:?} for {}, reporting unavailable",
                other, member_did
            );
            Ok(BalanceResponse {
                member_did,
                currency: "SAP".into(),
                balance: 0,
                available: false,
            })
        }
        Err(e) => {
            debug!(
                "query_sap_balance: payments zome unreachable for {}: {:?}, reporting unavailable",
                member_did, e
            );
            Ok(BalanceResponse {
                member_did,
                currency: "SAP".into(),
                balance: 0,
                available: false,
            })
        }
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
        Ok(other) => {
            debug!(
                "query_tend_balance: tend zome returned {:?} for {}, reporting unavailable",
                other, member_did
            );
            Ok(TendBalanceResponse {
                member_did,
                balance: 0,
                mycel_score: 0.0,
                available: false,
            })
        }
        Err(e) => {
            debug!(
                "query_tend_balance: tend zome unreachable for {}: {:?}, reporting unavailable",
                member_did, e
            );
            Ok(TendBalanceResponse {
                member_did,
                balance: 0,
                mycel_score: 0.0,
                available: false,
            })
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TendBalanceResponse {
    pub member_did: String,
    pub balance: i32,
    pub mycel_score: f64,
    pub available: bool,
}

/// Discover the local finance runtime context for the connected agent.
///
/// This gives the frontend a best-effort default DAO/treasury/commons context
/// without relying on browser globals. Explicit frontend config may still
/// override these values.
#[hdk_extern]
pub fn discover_runtime_context(_: ()) -> ExternResult<FinanceRuntimeDiscovery> {
    let member_did = format!("did:mycelix:{}", agent_info()?.agent_initial_pubkey);

    let dao_dids = match call(
        CallTargetCell::Local,
        ZomeName::from("tend"),
        FunctionName::from("get_member_dao_contexts"),
        None,
        member_did.clone(),
    ) {
        Ok(ZomeCallResponse::Ok(result)) => result.decode::<Vec<String>>().unwrap_or_default(),
        Ok(other) => {
            debug!(
                "discover_runtime_context: tend returned {:?} for {}, defaulting to empty DAO list",
                other, member_did
            );
            Vec::new()
        }
        Err(err) => {
            debug!(
                "discover_runtime_context: tend unreachable for {}: {:?}",
                member_did, err
            );
            Vec::new()
        }
    };

    let default_dao_did = runtime_default_dao_did(&dao_dids);
    let commons_pool_id = default_dao_did
        .as_ref()
        .and_then(|dao_did| discover_commons_pool_id(dao_did));
    let treasury_id = discover_managed_treasury_id(&member_did);

    Ok(build_runtime_discovery(
        member_did,
        dao_dids,
        commons_pool_id,
        treasury_id,
    ))
}

fn runtime_default_dao_did(dao_dids: &[String]) -> Option<String> {
    dao_dids.first().cloned()
}

fn build_runtime_discovery(
    member_did: String,
    dao_dids: Vec<String>,
    commons_pool_id: Option<String>,
    treasury_id: Option<String>,
) -> FinanceRuntimeDiscovery {
    FinanceRuntimeDiscovery {
        member_did,
        default_dao_did: runtime_default_dao_did(&dao_dids),
        dao_dids,
        commons_pool_id,
        treasury_id,
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct TreasuryListInput {
    id: String,
    limit: Option<usize>,
}

#[derive(Deserialize)]
struct MinimalTreasuryRecord {
    id: String,
}

#[derive(Deserialize)]
struct MinimalCommonsPoolRecord {
    id: String,
}

fn discover_managed_treasury_id(member_did: &str) -> Option<String> {
    match call(
        CallTargetCell::Local,
        ZomeName::from("treasury"),
        FunctionName::from("get_manager_treasuries"),
        None,
        TreasuryListInput {
            id: member_did.to_string(),
            limit: Some(1),
        },
    ) {
        Ok(ZomeCallResponse::Ok(result)) => {
            result.decode::<Vec<Record>>().ok().and_then(|records| {
                records.into_iter().find_map(|record| {
                    decode_record_entry::<MinimalTreasuryRecord>(&record)
                        .map(|treasury| treasury.id)
                })
            })
        }
        Ok(other) => {
            debug!(
                "discover_managed_treasury_id: treasury returned {:?} for {}",
                other, member_did
            );
            None
        }
        Err(err) => {
            debug!(
                "discover_managed_treasury_id: treasury unreachable for {}: {:?}",
                member_did, err
            );
            None
        }
    }
}

fn discover_commons_pool_id(dao_did: &str) -> Option<String> {
    match call(
        CallTargetCell::Local,
        ZomeName::from("treasury"),
        FunctionName::from("get_dao_commons_pool"),
        None,
        dao_did.to_string(),
    ) {
        Ok(ZomeCallResponse::Ok(result)) => result
            .decode::<Option<Record>>()
            .ok()
            .flatten()
            .and_then(|record| {
                decode_record_entry::<MinimalCommonsPoolRecord>(&record).map(|pool| pool.id)
            }),
        Ok(other) => {
            debug!(
                "discover_commons_pool_id: treasury returned {:?} for {}",
                other, dao_did
            );
            None
        }
        Err(err) => {
            debug!(
                "discover_commons_pool_id: treasury unreachable for {}: {:?}",
                dao_did, err
            );
            None
        }
    }
}

fn decode_record_entry<T: serde::de::DeserializeOwned>(record: &Record) -> Option<T> {
    match record.entry().as_option()? {
        Entry::App(bytes) => {
            let sb = SerializedBytes::from(bytes.to_owned());
            rmp_serde::from_slice::<T>(sb.bytes())
                .ok()
                .or_else(|| serde_json::from_slice::<T>(sb.bytes()).ok())
        }
        _ => None,
    }
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

// ---------------------------------------------------------------------------
// Phase 1b: Oracle Rate Attestation
// ---------------------------------------------------------------------------

/// Verify that the claimed oracle rate is within tolerance of consensus.
///
/// Fetches consensus from the price-oracle zome. If the oracle is unreachable
/// (bootstrap/standalone), accepts the claimed rate with a warning.
fn verify_oracle_rate_against_consensus(
    collateral_type: &str,
    claimed_rate: f64,
) -> ExternResult<()> {
    use mycelix_finance_types::ORACLE_RATE_TOLERANCE;

    #[derive(Serialize, Debug)]
    struct GetConsensusInput {
        item: String,
    }
    #[derive(Debug, Deserialize)]
    struct ConsensusResult {
        median_price: f64,
    }

    let item = format!("{}_SAP", collateral_type); // e.g., "ETH_SAP", "USDC_SAP"

    match call(
        CallTargetCell::Local,
        ZomeName::from("price_oracle"),
        FunctionName::from("get_consensus_price"),
        None,
        GetConsensusInput { item },
    ) {
        Ok(ZomeCallResponse::Ok(result)) => match result.decode::<ConsensusResult>() {
            Ok(consensus) if consensus.median_price.is_finite() && consensus.median_price > 0.0 => {
                let deviation =
                    (claimed_rate - consensus.median_price).abs() / consensus.median_price;
                if deviation > ORACLE_RATE_TOLERANCE {
                    return Err(wasm_error!(WasmErrorInner::Guest(format!(
                        "Oracle rate {:.6} deviates {:.1}% from consensus {:.6} (max {:.0}%). \
                             Use get_consensus_price to fetch current rate before depositing.",
                        claimed_rate,
                        deviation * 100.0,
                        consensus.median_price,
                        ORACLE_RATE_TOLERANCE * 100.0
                    ))));
                }
                Ok(())
            }
            _ => {
                debug!(
                    "verify_oracle_rate: consensus invalid, accepting claimed rate {:.6}",
                    claimed_rate
                );
                Ok(())
            }
        },
        _ => {
            debug!(
                "verify_oracle_rate: price oracle unreachable, accepting claimed rate {:.6}",
                claimed_rate
            );
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 2a: Covenant Registry
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateCovenantInput {
    pub collateral_id: String,
    pub restriction: String,
    pub beneficiary_did: String,
    pub expires_at: Option<Timestamp>,
}

/// Create a covenant (restriction) on registered collateral.
///
/// Only Citizen+ tier can create covenants. The covenant prevents the
/// collateral from being transferred or otherwise disposed of while active.
#[hdk_extern]
pub fn create_covenant(input: CreateCovenantInput) -> ExternResult<Record> {
    verify_citizen_tier()?;
    verify_caller_is_did(&input.beneficiary_did)?;
    let now = sys_time()?;

    let covenant_id = format!(
        "cov:{}:{}:{}",
        input.collateral_id,
        input.beneficiary_did,
        now.as_micros()
    );

    let covenant = Covenant {
        id: covenant_id.clone(),
        collateral_id: input.collateral_id.clone(),
        restriction: input.restriction,
        beneficiary_did: input.beneficiary_did.clone(),
        created_at: now,
        expires_at: input.expires_at,
        released: false,
        released_by: None,
        released_at: None,
    };

    let hash = create_entry(&EntryTypes::Covenant(covenant))?;

    create_link(
        anchor_hash(&input.collateral_id)?,
        hash.clone(),
        LinkTypes::CollateralToCovenants,
        (),
    )?;

    create_link(
        anchor_hash(&covenant_id)?,
        hash.clone(),
        LinkTypes::CovenantIdToCovenant,
        (),
    )?;

    broadcast_finance_event(BroadcastFinanceEventInput {
        event_type: FinanceEventType::CovenantCreated,
        subject_did: input.beneficiary_did,
        amount: None,
        payload: serde_json::json!({
            "covenant_id": covenant_id,
            "collateral_id": input.collateral_id,
        })
        .to_string(),
    })?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Covenant not found".into()
    )))
}

/// Release a covenant. Only the beneficiary can release.
#[hdk_extern]
pub fn release_covenant(covenant_id: String) -> ExternResult<Record> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&covenant_id)?, LinkTypes::CovenantIdToCovenant)?,
        GetStrategy::default(),
    )?;
    let link = links.first().ok_or(wasm_error!(WasmErrorInner::Guest(
        "Covenant not found".into()
    )))?;
    let hash = ActionHash::try_from(link.target.clone())
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
    let record = follow_update_chain(hash)?;
    let covenant = record
        .entry()
        .to_app_option::<Covenant>()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Deserialization error: {:?}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Covenant entry missing".into()
        )))?;

    verify_caller_is_did(&covenant.beneficiary_did)?;

    if covenant.released {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Covenant is already released".into()
        )));
    }

    let now = sys_time()?;
    let released = Covenant {
        released: true,
        released_by: Some(covenant.beneficiary_did.clone()),
        released_at: Some(now),
        ..covenant.clone()
    };

    let action_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::Covenant(released),
    )?;

    broadcast_finance_event(BroadcastFinanceEventInput {
        event_type: FinanceEventType::CovenantReleased,
        subject_did: covenant.beneficiary_did,
        amount: None,
        payload: serde_json::json!({
            "covenant_id": covenant_id,
            "collateral_id": covenant.collateral_id,
        })
        .to_string(),
    })?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

/// Query active (unreleased) covenants for a collateral ID.
#[hdk_extern]
pub fn check_covenants(collateral_id: String) -> ExternResult<Vec<Record>> {
    get_active_covenants(&collateral_id)
}

/// Internal helper: query active (unreleased) covenants for a collateral ID.
/// Used by both the extern `check_covenants` and internal covenant enforcement.
fn get_active_covenants(collateral_id: &str) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(collateral_id)?,
            LinkTypes::CollateralToCovenants,
        )?,
        GetStrategy::default(),
    )?;

    let mut active = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        let record = follow_update_chain(hash)?;
        if let Ok(Some(covenant)) = record.entry().to_app_option::<Covenant>() {
            if !covenant.released {
                active.push(record);
            }
        }
    }

    Ok(active)
}

/// Enforce that no active covenants block a collateral status change.
///
/// Returns an error if any unreleased covenants exist for the given collateral ID.
/// Called before redemption, release, or any status change that would dispose of collateral.
fn enforce_no_active_covenants(collateral_id: &str) -> ExternResult<()> {
    let active = get_active_covenants(collateral_id)?;
    if !active.is_empty() {
        let covenant_ids: Vec<String> = active
            .iter()
            .filter_map(|r| {
                r.entry()
                    .to_app_option::<Covenant>()
                    .ok()
                    .flatten()
                    .map(|c| c.id)
            })
            .collect();
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot change collateral status: {} active covenant(s) blocking: [{}]",
            active.len(),
            covenant_ids.join(", ")
        ))));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Phase 2b: LTV Monitoring
// ---------------------------------------------------------------------------

/// Compute and store the LTV health status for a collateral position.
///
/// Fetches current value from the price oracle, computes the LTV ratio,
/// and stores a CollateralHealth snapshot linked to the collateral.
#[hdk_extern]
pub fn update_collateral_health(input: UpdateCollateralHealthInput) -> ExternResult<Record> {
    verify_participant_tier()?;
    let now = sys_time()?;

    // Fetch current collateral value from oracle
    let current_value = fetch_collateral_value(&input.collateral_id);

    let ltv_ratio = if current_value > 0 {
        input.obligation_amount as f64 / current_value as f64
    } else {
        f64::INFINITY
    };

    // Thresholds match canonical CollateralHealthStatus::from_ltv() in mycelix_finance_types
    let status = if !ltv_ratio.is_finite() || ltv_ratio > 0.95 {
        "Liquidation"
    } else if ltv_ratio > 0.90 {
        "MarginCall"
    } else if ltv_ratio > 0.80 {
        "Warning"
    } else {
        "Healthy"
    };

    let health = CollateralHealth {
        collateral_id: input.collateral_id.clone(),
        current_value,
        obligation_amount: input.obligation_amount,
        ltv_ratio: if ltv_ratio.is_finite() {
            ltv_ratio
        } else {
            999.0
        },
        status: status.to_string(),
        computed_at: now,
    };

    let hash = create_entry(&EntryTypes::CollateralHealth(health))?;

    create_link(
        anchor_hash(&input.collateral_id)?,
        hash.clone(),
        LinkTypes::CollateralToHealth,
        (),
    )?;

    broadcast_finance_event(BroadcastFinanceEventInput {
        event_type: FinanceEventType::CollateralHealthUpdated,
        subject_did: input.collateral_id,
        amount: Some(current_value),
        payload: serde_json::json!({
            "ltv_ratio": ltv_ratio,
            "status": status,
            "obligation": input.obligation_amount,
        })
        .to_string(),
    })?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Health entry not found".into()
    )))
}

fn map_wire_asset_type(asset_type: finance_wire_types::AssetType) -> AssetType {
    match asset_type {
        finance_wire_types::AssetType::RealEstate => AssetType::RealEstate,
        finance_wire_types::AssetType::Vehicle => AssetType::Vehicle,
        finance_wire_types::AssetType::Cryptocurrency => AssetType::Cryptocurrency,
        finance_wire_types::AssetType::EnergyAsset => AssetType::EnergyAsset,
        finance_wire_types::AssetType::Equipment => AssetType::Equipment,
        finance_wire_types::AssetType::Other(value) => AssetType::Other(value),
    }
}

/// Fetch current value for a collateral position from the price oracle.
/// Falls back to 0 if oracle is unreachable.
fn fetch_collateral_value(collateral_id: &str) -> u64 {
    #[derive(Serialize, Debug)]
    struct GetValueInput {
        collateral_id: String,
    }
    #[derive(Debug, Deserialize)]
    struct ValueResult {
        value: u64,
    }

    match call(
        CallTargetCell::Local,
        ZomeName::from("price_oracle"),
        FunctionName::from("get_collateral_value"),
        None,
        GetValueInput {
            collateral_id: collateral_id.to_string(),
        },
    ) {
        Ok(ZomeCallResponse::Ok(result)) => {
            result.decode::<ValueResult>().map(|v| v.value).unwrap_or(0)
        }
        _ => {
            debug!(
                "fetch_collateral_value: oracle unreachable for {}, defaulting to 0",
                collateral_id
            );
            0
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 2c: Energy Certificate Registry
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterEnergyCertificateInput {
    pub project_id: String,
    pub source: String,
    pub kwh_produced: f64,
    pub period_start: Timestamp,
    pub period_end: Timestamp,
    pub location_lat: f64,
    pub location_lon: f64,
    pub producer_did: String,
    pub terra_atlas_id: Option<String>,
}

/// Register a new energy production certificate. Requires Participant+ tier.
#[hdk_extern]
pub fn register_energy_certificate(input: RegisterEnergyCertificateInput) -> ExternResult<Record> {
    verify_participant_tier()?;
    verify_caller_is_did(&input.producer_did)?;
    let now = sys_time()?;

    let cert_id = format!(
        "cert:{}:{}:{}",
        input.project_id,
        input.producer_did,
        now.as_micros()
    );

    let cert = EnergyCertificate {
        id: cert_id.clone(),
        project_id: input.project_id,
        source: input.source,
        kwh_produced: input.kwh_produced,
        period_start: input.period_start,
        period_end: input.period_end,
        location_lat: input.location_lat,
        location_lon: input.location_lon,
        producer_did: input.producer_did.clone(),
        verifier_did: None,
        status: "Pending".into(),
        terra_atlas_id: input.terra_atlas_id,
        sap_value: None,
        created_at: now,
    };

    let hash = create_entry(&EntryTypes::EnergyCertificate(cert))?;

    create_link(
        anchor_hash("energy_certificates")?,
        hash.clone(),
        LinkTypes::EnergyCertificateRegistry,
        (),
    )?;

    broadcast_finance_event(BroadcastFinanceEventInput {
        event_type: FinanceEventType::EnergyCertificateCreated,
        subject_did: input.producer_did,
        amount: None,
        payload: serde_json::json!({
            "cert_id": cert_id,
            "kwh": input.kwh_produced,
        })
        .to_string(),
    })?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Certificate not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyEnergyCertificateInput {
    pub cert_id: String,
    pub verifier_did: String,
    pub sap_value: u64,
}

/// Verify an energy certificate and assign SAP value. Requires Citizen+ tier.
#[hdk_extern]
pub fn verify_energy_certificate(input: VerifyEnergyCertificateInput) -> ExternResult<Record> {
    verify_citizen_tier()?;
    verify_caller_is_did(&input.verifier_did)?;

    // Find the certificate by querying the registry
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash("energy_certificates")?,
            LinkTypes::EnergyCertificateRegistry,
        )?,
        GetStrategy::default(),
    )?;

    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        let record = follow_update_chain(hash)?;
        if let Ok(Some(cert)) = record.entry().to_app_option::<EnergyCertificate>() {
            if cert.id == input.cert_id {
                if cert.status != "Pending" {
                    return Err(wasm_error!(WasmErrorInner::Guest(format!(
                        "Certificate is {}, only Pending certificates can be verified",
                        cert.status
                    ))));
                }

                // Verify project exists in energy cluster (cross-cluster call)
                // Circuit breaker: if energy cluster is unreachable, log and proceed
                // (non-fatal — accept verification in offline mode)
                match call(
                    CallTargetCell::OtherRole("energy".into()),
                    ZomeName::from("energy_bridge"),
                    FunctionName::from("get_project"),
                    None,
                    cert.project_id.clone(),
                ) {
                    Ok(ZomeCallResponse::Ok(_)) => {} // Project exists
                    Ok(ZomeCallResponse::NetworkError(e)) => {
                        debug!(
                            "Circuit breaker: energy cluster unavailable (network error), proceeding without project verification: {}",
                            e
                        );
                    }
                    Ok(other) => {
                        debug!(
                            "Circuit breaker: energy cluster returned unexpected response, proceeding without project verification: {:?}",
                            other
                        );
                    }
                    Err(e) => {
                        debug!(
                            "Circuit breaker: energy cluster unreachable (transport error), proceeding without project verification: {:?}",
                            e
                        );
                    }
                }

                let verified = EnergyCertificate {
                    status: "Verified".into(),
                    verifier_did: Some(input.verifier_did),
                    sap_value: Some(input.sap_value),
                    ..cert
                };

                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::EnergyCertificate(verified),
                )?;

                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest(format!(
        "Energy certificate {} not found",
        input.cert_id
    ))))
}

// ---------------------------------------------------------------------------
// Phase 2d: Agricultural Asset Registry
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug)]
pub struct RegisterAgriculturalAssetInput {
    pub asset_type: String,
    pub quantity_kg: f64,
    pub location_lat: f64,
    pub location_lon: f64,
    pub production_date: Timestamp,
    pub viability_duration_micros: Option<i64>,
    pub producer_did: String,
}

/// Register a new agricultural asset. Requires Participant+ tier.
#[hdk_extern]
pub fn register_agricultural_asset(input: RegisterAgriculturalAssetInput) -> ExternResult<Record> {
    verify_participant_tier()?;
    verify_caller_is_did(&input.producer_did)?;
    let now = sys_time()?;

    let asset_id = format!(
        "agri:{}:{}:{}",
        input.asset_type,
        input.producer_did,
        now.as_micros()
    );

    let asset = AgriculturalAsset {
        id: asset_id.clone(),
        asset_type: input.asset_type,
        quantity_kg: input.quantity_kg,
        location_lat: input.location_lat,
        location_lon: input.location_lon,
        production_date: input.production_date,
        viability_duration_micros: input.viability_duration_micros,
        producer_did: input.producer_did.clone(),
        verifier_did: None,
        status: "Pending".into(),
        sap_value: None,
        created_at: now,
    };

    let hash = create_entry(&EntryTypes::AgriculturalAsset(asset))?;

    create_link(
        anchor_hash("agri_assets")?,
        hash.clone(),
        LinkTypes::AgriAssetRegistry,
        (),
    )?;

    broadcast_finance_event(BroadcastFinanceEventInput {
        event_type: FinanceEventType::AgriAssetRegistered,
        subject_did: input.producer_did,
        amount: None,
        payload: serde_json::json!({
            "asset_id": asset_id,
            "quantity_kg": input.quantity_kg,
        })
        .to_string(),
    })?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Asset not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyAgriculturalAssetInput {
    pub asset_id: String,
    pub verifier_did: String,
    pub sap_value: u64,
}

/// Verify an agricultural asset and assign SAP value. Requires Citizen+ tier.
#[hdk_extern]
pub fn verify_agricultural_asset(input: VerifyAgriculturalAssetInput) -> ExternResult<Record> {
    verify_citizen_tier()?;
    verify_caller_is_did(&input.verifier_did)?;

    let links = get_links(
        LinkQuery::try_new(anchor_hash("agri_assets")?, LinkTypes::AgriAssetRegistry)?,
        GetStrategy::default(),
    )?;

    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        let record = follow_update_chain(hash)?;
        if let Ok(Some(asset)) = record.entry().to_app_option::<AgriculturalAsset>() {
            if asset.id == input.asset_id {
                if asset.status != "Pending" {
                    return Err(wasm_error!(WasmErrorInner::Guest(format!(
                        "Asset is {}, only Pending assets can be verified",
                        asset.status
                    ))));
                }

                let verified = AgriculturalAsset {
                    status: "Verified".into(),
                    verifier_did: Some(input.verifier_did),
                    sap_value: Some(input.sap_value),
                    ..asset
                };

                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::AgriculturalAsset(verified),
                )?;

                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest(format!(
        "Agricultural asset {} not found",
        input.asset_id
    ))))
}

// ---------------------------------------------------------------------------
// Phase 2e: Multi-Collateral Positions
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateMultiCollateralPositionInput {
    pub holder_did: String,
    pub components_json: String,
    pub aggregate_value: u64,
    pub aggregate_obligation: u64,
}

/// Create a multi-collateral position (basket of assets).
///
/// Computes a diversification bonus based on the number of distinct asset types
/// in the basket. The bonus reduces effective LTV by up to 5%.
#[hdk_extern]
pub fn create_multi_collateral_position(
    input: CreateMultiCollateralPositionInput,
) -> ExternResult<Record> {
    verify_participant_tier()?;
    verify_caller_is_did(&input.holder_did)?;
    let now = sys_time()?;

    // Parse components to count distinct types for diversification bonus
    #[derive(Deserialize)]
    struct Component {
        #[serde(rename = "type")]
        _type: String,
    }
    let components: Vec<Component> = serde_json::from_str(&input.components_json).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Invalid components JSON: {:?}",
            e
        )))
    })?;

    if components.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Multi-collateral position must have at least one component".into()
        )));
    }

    // Diversification bonus: 1% per distinct asset type, capped at 5%
    let distinct_types: std::collections::HashSet<&str> =
        components.iter().map(|c| c._type.as_str()).collect();
    // Bonus starts from 2nd distinct class: 5% per class, capped at 20%
    // Matches canonical compute_diversification_bonus() in mycelix_finance_types
    let bonus_classes = distinct_types.len().saturating_sub(1);
    let diversification_bonus = (bonus_classes as f64 * 0.05).min(0.20);

    let base_ltv = if input.aggregate_value > 0 {
        input.aggregate_obligation as f64 / input.aggregate_value as f64
    } else {
        999.0
    };
    let effective_ltv = (base_ltv - diversification_bonus).max(0.0);

    // Thresholds match canonical CollateralHealthStatus::from_ltv() in mycelix_finance_types
    let status = if !effective_ltv.is_finite() || effective_ltv > 0.95 {
        "Liquidation"
    } else if effective_ltv > 0.90 {
        "MarginCall"
    } else if effective_ltv > 0.80 {
        "Warning"
    } else {
        "Healthy"
    };

    let position_id = format!("mcp:{}:{}", input.holder_did, now.as_micros());

    let position = MultiCollateralPosition {
        id: position_id.clone(),
        holder_did: input.holder_did.clone(),
        components_json: input.components_json,
        aggregate_value: input.aggregate_value,
        aggregate_obligation: input.aggregate_obligation,
        diversification_bonus,
        effective_ltv,
        status: status.to_string(),
        created_at: now,
        last_revalued_at: now,
    };

    let hash = create_entry(&EntryTypes::MultiCollateralPosition(position))?;

    create_link(
        anchor_hash("multi_collateral")?,
        hash.clone(),
        LinkTypes::MultiCollateralRegistry,
        (),
    )?;

    create_link(
        anchor_hash(&input.holder_did)?,
        hash.clone(),
        LinkTypes::HolderToPositions,
        (),
    )?;

    broadcast_finance_event(BroadcastFinanceEventInput {
        event_type: FinanceEventType::MultiCollateralCreated,
        subject_did: input.holder_did,
        amount: Some(input.aggregate_value),
        payload: serde_json::json!({
            "position_id": position_id,
            "effective_ltv": effective_ltv,
            "diversification_bonus": diversification_bonus,
            "status": status,
        })
        .to_string(),
    })?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Position not found".into()
    )))
}

// ---------------------------------------------------------------------------
// Phase 2f: Fiat Bridge Deposits
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug)]
pub struct DepositFiatInput {
    pub depositor_did: String,
    pub fiat_currency: String,
    pub fiat_amount: u64,
    pub exchange_rate: f64,
    pub verifier_did: String,
    pub external_reference: String,
}

/// Create a fiat bridge deposit (inbound: fiat -> SAP). Requires Citizen+ tier verifier.
#[hdk_extern]
pub fn deposit_fiat(input: DepositFiatInput) -> ExternResult<Record> {
    verify_citizen_tier()?;
    verify_caller_is_did(&input.verifier_did)?;
    let now = sys_time()?;

    if input.fiat_amount == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Fiat amount must be positive".into()
        )));
    }
    if !input.exchange_rate.is_finite() || input.exchange_rate <= 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Exchange rate must be a finite positive number".into()
        )));
    }

    let sap_minted = (input.fiat_amount as f64 * input.exchange_rate) as u64;

    let deposit_id = format!(
        "fiat:{}:{}:{}",
        input.depositor_did,
        input.fiat_currency,
        now.as_micros()
    );

    let deposit = FiatBridgeDeposit {
        id: deposit_id.clone(),
        depositor_did: input.depositor_did.clone(),
        fiat_currency: input.fiat_currency,
        fiat_amount: input.fiat_amount,
        sap_minted,
        exchange_rate: input.exchange_rate,
        verifier_did: input.verifier_did,
        status: "Pending".into(),
        external_reference: input.external_reference,
        created_at: now,
        verified_at: None,
    };

    let hash = create_entry(&EntryTypes::FiatBridgeDeposit(deposit))?;

    create_link(
        anchor_hash("fiat_deposits")?,
        hash.clone(),
        LinkTypes::FiatDepositRegistry,
        (),
    )?;

    broadcast_finance_event(BroadcastFinanceEventInput {
        event_type: FinanceEventType::FiatDeposited,
        subject_did: input.depositor_did,
        amount: Some(sap_minted),
        payload: serde_json::json!({
            "deposit_id": deposit_id,
            "fiat_amount": input.fiat_amount,
        })
        .to_string(),
    })?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Fiat deposit not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyFiatDepositInput {
    pub deposit_id: String,
    pub verifier_did: String,
}

/// Verify a pending fiat deposit and mint SAP. Requires Citizen+ tier.
#[hdk_extern]
pub fn verify_fiat_deposit(input: VerifyFiatDepositInput) -> ExternResult<Record> {
    verify_citizen_tier()?;
    verify_caller_is_did(&input.verifier_did)?;

    let links = get_links(
        LinkQuery::try_new(
            anchor_hash("fiat_deposits")?,
            LinkTypes::FiatDepositRegistry,
        )?,
        GetStrategy::default(),
    )?;

    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        let record = follow_update_chain(hash)?;
        if let Ok(Some(deposit)) = record.entry().to_app_option::<FiatBridgeDeposit>() {
            if deposit.id == input.deposit_id {
                if deposit.status != "Pending" {
                    return Err(wasm_error!(WasmErrorInner::Guest(format!(
                        "Deposit is {}, only Pending deposits can be verified",
                        deposit.status
                    ))));
                }

                let now = sys_time()?;
                let verified = FiatBridgeDeposit {
                    status: "Verified".into(),
                    verified_at: Some(now),
                    ..deposit.clone()
                };

                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::FiatBridgeDeposit(verified),
                )?;

                // Credit SAP to depositor
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
                        member_did: deposit.depositor_did.clone(),
                        amount: deposit.sap_minted,
                        reason: format!(
                            "Fiat bridge deposit: {} {}",
                            deposit.fiat_amount, deposit.fiat_currency
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
                            "Failed to credit SAP for fiat deposit: {:?}",
                            e
                        ))));
                    }
                }

                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest(format!(
        "Fiat deposit {} not found",
        input.deposit_id
    ))))
}

// ---------------------------------------------------------------------------
// Rate Limiting Helpers
// ---------------------------------------------------------------------------

/// Enforce the daily rate limit: no member may deposit/redeem more than
/// `daily_limit_pct`% of total vault value in any rolling 24-hour period.
///
/// The limit percentage is tier-scaled based on MYCEL score:
/// - Newcomer: 1%, Member: 5%, Steward: 10%
///
/// Vault value = sum of `sap_minted` for all Confirmed deposits.
/// Daily activity = sum of `sap_minted` for this member's deposits/redemptions
/// created within the last 24 hours.
fn enforce_rate_limit(
    member_did: &str,
    new_amount: u64,
    now: Timestamp,
    daily_limit_pct: u32,
) -> ExternResult<()> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::CollateralBridgeDeposit,
        )?))
        .include_entries(true);

    let deposits: Vec<CollateralBridgeDeposit> = query(filter)?
        .into_iter()
        .filter_map(
            |r| match r.entry().to_app_option::<CollateralBridgeDeposit>() {
                Ok(opt) => opt,
                Err(e) => {
                    debug!("enforce_rate_limit: deposit deserialization error: {:?}", e);
                    None
                }
            },
        )
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

    let daily_limit = (vault_total as f64 * (daily_limit_pct as f64 / 100.0)) as u64;

    // Sum this member's activity in the last 24 hours
    let cutoff = now.as_micros() - DAY_MICROS;
    let daily_activity: u64 = deposits
        .iter()
        .filter(|d| d.depositor_did == member_did && d.created_at.as_micros() > cutoff)
        .map(|d| d.sap_minted)
        .fold(0u64, |acc, v| acc.saturating_add(v));

    if daily_activity.saturating_add(new_amount) > daily_limit {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Rate limit exceeded: max {} SAP/day ({}% of vault {}). Already used {} today, requesting {}.",
            daily_limit, daily_limit_pct, vault_total, daily_activity, new_amount
        ))));
    }

    Ok(())
}

// ============================================================================
// Observability — Bridge Metrics Export
// ============================================================================

/// Return a JSON-encoded snapshot of this bridge's dispatch metrics.
///
/// See `mycelix_bridge_common::metrics::BridgeMetricsSnapshot` for the schema.
#[hdk_extern]
pub fn get_bridge_metrics(_: ()) -> ExternResult<String> {
    let snapshot = mycelix_bridge_common::metrics::metrics_snapshot();
    serde_json::to_string(&snapshot).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Failed to serialize metrics snapshot: {}",
            e
        )))
    })
}

// ============================================================================
// Notification Service
// ============================================================================

/// Receive a cross-cluster notification and store it locally.
#[hdk_extern]
pub fn receive_notification(
    notification: mycelix_bridge_entry_types::CrossClusterNotification,
) -> ExternResult<ActionHash> {
    let action_hash = create_entry(&EntryTypes::Notification(notification.clone()))?;
    let agent = agent_info()?.agent_initial_pubkey;
    let inbox_anchor = anchor_hash(&format!("notifications:{:?}", agent))?;
    create_link(
        inbox_anchor,
        action_hash.clone(),
        LinkTypes::AgentToNotification,
        (),
    )?;
    let all_anchor = anchor_hash("all_notifications")?;
    create_link(
        all_anchor,
        action_hash.clone(),
        LinkTypes::AllNotifications,
        (),
    )?;
    let signal = mycelix_bridge_common::notifications::NotificationSignal {
        signal_type: "cross_cluster_notification".into(),
        source_cluster: notification.source_cluster,
        event_type: notification.event_type,
        payload: notification.payload,
        priority: notification.priority,
    };
    emit_signal(&signal)?;
    Ok(action_hash)
}

/// Get notifications for the calling agent.
#[hdk_extern]
pub fn get_my_notifications(
    input: mycelix_bridge_common::notifications::NotificationQueryInput,
) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let inbox_anchor = anchor_hash(&format!("notifications:{:?}", agent))?;
    let links = get_links(
        LinkQuery::try_new(inbox_anchor, LinkTypes::AgentToNotification)?,
        GetStrategy::default(),
    )?;
    let limit = input
        .limit
        .unwrap_or(mycelix_bridge_common::notifications::DEFAULT_NOTIFICATION_LIMIT)
        .min(mycelix_bridge_common::notifications::MAX_NOTIFICATIONS_PER_AGENT);
    let mut records = Vec::new();
    for link in links.iter().rev().take(limit) {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

/// Get unread notification count for the calling agent.
#[hdk_extern]
pub fn get_unread_count(_: ()) -> ExternResult<u32> {
    let agent = agent_info()?.agent_initial_pubkey;
    let inbox_anchor = anchor_hash(&format!("notifications:{:?}", agent))?;
    let links = get_links(
        LinkQuery::try_new(inbox_anchor, LinkTypes::AgentToNotification)?,
        GetStrategy::default(),
    )?;
    Ok(links.len() as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_default_dao_did_uses_first_known_context() {
        let dao_dids = vec![
            "did:mycelix:dao-alpha".to_string(),
            "did:mycelix:dao-beta".to_string(),
        ];
        assert_eq!(
            runtime_default_dao_did(&dao_dids).as_deref(),
            Some("did:mycelix:dao-alpha")
        );
    }

    #[test]
    fn build_runtime_discovery_keeps_empty_state_explicit() {
        let discovery = build_runtime_discovery("did:mycelix:alice".into(), Vec::new(), None, None);

        assert_eq!(discovery.member_did, "did:mycelix:alice");
        assert!(discovery.dao_dids.is_empty());
        assert!(discovery.default_dao_did.is_none());
        assert!(discovery.commons_pool_id.is_none());
        assert!(discovery.treasury_id.is_none());
    }

    #[test]
    fn build_runtime_discovery_preserves_discovered_context() {
        let discovery = build_runtime_discovery(
            "did:mycelix:alice".into(),
            vec![
                "did:mycelix:dao-alpha".into(),
                "did:mycelix:dao-beta".into(),
            ],
            Some("commons-alpha".into()),
            Some("treasury-alpha".into()),
        );

        assert_eq!(
            discovery.default_dao_did.as_deref(),
            Some("did:mycelix:dao-alpha")
        );
        assert_eq!(discovery.commons_pool_id.as_deref(), Some("commons-alpha"));
        assert_eq!(discovery.treasury_id.as_deref(), Some("treasury-alpha"));
        assert_eq!(discovery.dao_dids.len(), 2);
    }

    #[test]
    fn map_wire_asset_type_preserves_variants() {
        assert_eq!(
            map_wire_asset_type(finance_wire_types::AssetType::RealEstate),
            AssetType::RealEstate
        );
        assert_eq!(
            map_wire_asset_type(finance_wire_types::AssetType::Vehicle),
            AssetType::Vehicle
        );
        assert_eq!(
            map_wire_asset_type(finance_wire_types::AssetType::Cryptocurrency),
            AssetType::Cryptocurrency
        );
        assert_eq!(
            map_wire_asset_type(finance_wire_types::AssetType::EnergyAsset),
            AssetType::EnergyAsset
        );
        assert_eq!(
            map_wire_asset_type(finance_wire_types::AssetType::Equipment),
            AssetType::Equipment
        );
        assert_eq!(
            map_wire_asset_type(finance_wire_types::AssetType::Other("bamboo".into())),
            AssetType::Other("bamboo".into())
        );
    }
}
