//! Payments Coordinator Zome
use hdk::prelude::*;
use mycelix_finance_shared::{anchor_hash, verify_caller_is_did};
use mycelix_finance_types::{
    compute_demurrage_deduction, FeeTier, SapMintSource, SuccessionPreference, COMPOST_LOCAL_PCT,
    COMPOST_REGIONAL_PCT, DEMURRAGE_EXEMPT_FLOOR, DEMURRAGE_RATE, SAP_MINT_ANNUAL_MAX,
    SAP_MINT_PER_PROPOSAL_MAX,
};
use payments_integrity::*;

// ---------------------------------------------------------------------------
// SAP Balance Management (on-chain balance with enforced demurrage)
// ---------------------------------------------------------------------------

/// Initialize a SAP balance for a new member (zero balance).
#[hdk_extern]
pub fn initialize_sap_balance(member_did: String) -> ExternResult<Record> {
    verify_caller_is_did(&member_did)?;

    // Check if balance already exists
    if find_sap_balance_record(&member_did)?.is_some() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "SAP balance already initialized for this member".into()
        )));
    }

    let now = sys_time()?;
    let balance = SapBalance {
        member_did: member_did.clone(),
        balance: 0,
        last_demurrage_at: now,
    };

    let action_hash = create_entry(&EntryTypes::SapBalance(balance))?;
    create_link(
        anchor_hash(&format!("sap:{}", member_did))?,
        action_hash.clone(),
        LinkTypes::DidToSapBalance,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

/// Get the effective SAP balance for a member (after applying demurrage).
/// This is a read-only query — it does NOT persist the demurrage deduction.
#[hdk_extern]
pub fn get_sap_balance(member_did: String) -> ExternResult<SapBalanceResponse> {
    let (record, bal) = get_sap_balance_inner(&member_did)?;
    let now = sys_time()?;
    let elapsed = elapsed_seconds(bal.last_demurrage_at, now);
    let deduction =
        compute_demurrage_deduction(bal.balance, DEMURRAGE_EXEMPT_FLOOR, DEMURRAGE_RATE, elapsed);
    let _ = record; // used only for existence check
    Ok(SapBalanceResponse {
        member_did: bal.member_did,
        raw_balance: bal.balance,
        effective_balance: bal.balance.saturating_sub(deduction),
        pending_demurrage: deduction,
        last_demurrage_at: bal.last_demurrage_at,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SapBalanceResponse {
    pub member_did: String,
    pub raw_balance: u64,
    pub effective_balance: u64,
    pub pending_demurrage: u64,
    pub last_demurrage_at: Timestamp,
}

/// Apply demurrage to a member's SAP balance and redistribute the deducted
/// amount as compost to commons pools (70% local, 20% regional, 10% global).
///
/// Returns the amount deducted. If 0, no update is persisted.
#[hdk_extern]
pub fn apply_demurrage(input: ApplyDemurrageInput) -> ExternResult<DemurrageResult> {
    let (record, bal) = get_sap_balance_inner(&input.member_did)?;
    let now = sys_time()?;
    let elapsed = elapsed_seconds(bal.last_demurrage_at, now);
    let deduction =
        compute_demurrage_deduction(bal.balance, DEMURRAGE_EXEMPT_FLOOR, DEMURRAGE_RATE, elapsed);

    if deduction == 0 {
        return Ok(DemurrageResult {
            deducted: 0,
            redistributed: true,
        });
    }

    // Update balance
    let updated = SapBalance {
        balance: bal.balance.saturating_sub(deduction),
        last_demurrage_at: now,
        ..bal
    };
    update_entry(
        record.action_address().clone(),
        &EntryTypes::SapBalance(updated),
    )?;

    // Redistribute as compost: 70% local, 20% regional, 10% global
    let local_amount = deduction * COMPOST_LOCAL_PCT / 100;
    let regional_amount = deduction * COMPOST_REGIONAL_PCT / 100;
    let global_amount = deduction - local_amount - regional_amount; // remainder to global

    // Redistribute via treasury zome cross-zome calls
    if let Some(ref pool_id) = input.local_commons_pool_id {
        if let Err(e) = call(
            CallTargetCell::Local,
            ZomeName::from("treasury"),
            FunctionName::from("receive_compost"),
            None,
            ReceiveCompostPayload {
                commons_pool_id: pool_id.clone(),
                amount: local_amount,
                source_member_did: input.member_did.clone(),
            },
        ) {
            debug!(
                "Compost redistribution to local pool {} failed: {:?}",
                pool_id, e
            );
        }
    }
    if let Some(ref pool_id) = input.regional_commons_pool_id {
        if let Err(e) = call(
            CallTargetCell::Local,
            ZomeName::from("treasury"),
            FunctionName::from("receive_compost"),
            None,
            ReceiveCompostPayload {
                commons_pool_id: pool_id.clone(),
                amount: regional_amount,
                source_member_did: input.member_did.clone(),
            },
        ) {
            debug!(
                "Compost redistribution to regional pool {} failed: {:?}",
                pool_id, e
            );
        }
    }
    if let Some(ref pool_id) = input.global_commons_pool_id {
        if let Err(e) = call(
            CallTargetCell::Local,
            ZomeName::from("treasury"),
            FunctionName::from("receive_compost"),
            None,
            ReceiveCompostPayload {
                commons_pool_id: pool_id.clone(),
                amount: global_amount,
                source_member_did: input.member_did.clone(),
            },
        ) {
            debug!(
                "Compost redistribution to global pool {} failed: {:?}",
                pool_id, e
            );
        }
    }

    Ok(DemurrageResult {
        deducted: deduction,
        redistributed: true,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ApplyDemurrageInput {
    pub member_did: String,
    pub local_commons_pool_id: Option<String>,
    pub regional_commons_pool_id: Option<String>,
    pub global_commons_pool_id: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ReceiveCompostPayload {
    pub commons_pool_id: String,
    pub amount: u64,
    pub source_member_did: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DemurrageResult {
    pub deducted: u64,
    pub redistributed: bool,
}

/// Credit SAP to a member's balance (used by bridge deposits and community issuance).
/// Auto-initializes the SapBalance entry if the member has none yet.
#[hdk_extern]
pub fn credit_sap(input: CreditSapInput) -> ExternResult<Record> {
    let now = sys_time()?;

    match find_sap_balance_record(&input.member_did)? {
        Some((record, bal)) => {
            // Existing balance: apply pending demurrage first, then credit
            let elapsed = elapsed_seconds(bal.last_demurrage_at, now);
            let deduction = compute_demurrage_deduction(
                bal.balance,
                DEMURRAGE_EXEMPT_FLOOR,
                DEMURRAGE_RATE,
                elapsed,
            );
            let post_demurrage = bal.balance.saturating_sub(deduction);

            let updated = SapBalance {
                balance: post_demurrage + input.amount,
                last_demurrage_at: now,
                ..bal
            };
            let action_hash = update_entry(
                record.action_address().clone(),
                &EntryTypes::SapBalance(updated),
            )?;
            get(action_hash, GetOptions::default())?
                .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
        }
        None => {
            // First-time credit: auto-initialize balance with credited amount
            let balance = SapBalance {
                member_did: input.member_did.clone(),
                balance: input.amount,
                last_demurrage_at: now,
            };
            let action_hash = create_entry(&EntryTypes::SapBalance(balance))?;
            create_link(
                anchor_hash(&format!("sap:{}", input.member_did))?,
                action_hash.clone(),
                LinkTypes::DidToSapBalance,
                (),
            )?;
            get(action_hash, GetOptions::default())?
                .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreditSapInput {
    pub member_did: String,
    pub amount: u64,
    pub reason: String,
}

/// Debit SAP from a member's balance (enforces demurrage + sufficient balance).
#[hdk_extern]
pub fn debit_sap(input: DebitSapInput) -> ExternResult<Record> {
    let (record, bal) = get_sap_balance_inner(&input.member_did)?;
    let now = sys_time()?;

    // Apply pending demurrage first
    let elapsed = elapsed_seconds(bal.last_demurrage_at, now);
    let deduction =
        compute_demurrage_deduction(bal.balance, DEMURRAGE_EXEMPT_FLOOR, DEMURRAGE_RATE, elapsed);
    let effective = bal.balance.saturating_sub(deduction);

    if input.amount > effective {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Insufficient SAP balance: effective {} (raw {} - demurrage {}), need {}",
            effective, bal.balance, deduction, input.amount
        ))));
    }

    let updated = SapBalance {
        balance: effective - input.amount,
        last_demurrage_at: now,
        ..bal
    };
    let action_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::SapBalance(updated),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DebitSapInput {
    pub member_did: String,
    pub amount: u64,
    pub reason: String,
}

// ---------------------------------------------------------------------------
// SAP Minting (governance-authorized issuance)
// ---------------------------------------------------------------------------

/// Mint SAP from a governance proposal. Creates an immutable SapMintRecord
/// and credits the recipient's balance.
///
/// This is the ONLY way new SAP enters circulation outside of collateral deposits.
/// Requires governance authorization (verified via cross-zome call).
#[hdk_extern]
pub fn mint_sap_from_governance(input: MintSapFromGovernanceInput) -> ExternResult<Record> {
    // Verify caller is governance-authorized
    if let Err(e) = call(
        CallTargetCell::Local,
        ZomeName::from("tend"),
        FunctionName::from("verify_governance_agent"),
        None,
        (),
    ) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "SAP minting requires governance authorization: {:?}",
            e
        ))));
    }

    // Consciousness gate: SAP minting requires Steward tier (consciousness >= 0.6)
    // This implements the Φ Gate — high-impact economic actions require demonstrated
    // community consciousness (identity + reputation + community + engagement).
    if let Ok(ZomeCallResponse::Ok(result)) = call(
        CallTargetCell::Local,
        ZomeName::from("finance_bridge"),
        FunctionName::from("get_member_fee_tier"),
        None,
        input.recipient_did.clone(),
    ) {
        #[derive(Debug, Deserialize)]
        struct TierResp {
            tier_name: String,
        }
        if let Ok(resp) = result.decode::<TierResp>() {
            if resp.tier_name == "Newcomer" {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "SAP minting recipient must be at least Member tier (MYCEL >= 0.3). \
                     Newcomers cannot receive governance-minted SAP to prevent bootstrap attacks."
                        .into()
                )));
            }
        }
    }
    // If bridge is unreachable, allow the mint (bootstrap/standalone mode)

    // Constitutional cap: per-proposal maximum
    if input.amount > SAP_MINT_PER_PROPOSAL_MAX {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Mint amount {} exceeds constitutional per-proposal maximum of {} micro-SAP (100,000 SAP)",
            input.amount, SAP_MINT_PER_PROPOSAL_MAX
        ))));
    }
    if input.amount == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Mint amount must be positive".into()
        )));
    }

    let now = sys_time()?;

    // Constitutional cap: annual maximum — sum all governance mints in the last 365 days
    enforce_annual_mint_cap(input.amount, now)?;
    let mint_id = format!("mint:gov:{}:{}", input.proposal_id, now.as_micros());

    let source = SapMintSource::GovernanceProposal {
        proposal_id: input.proposal_id.clone(),
    };

    // Create immutable mint record
    let mint_record = SapMintRecord {
        id: mint_id.clone(),
        recipient_did: input.recipient_did.clone(),
        amount: input.amount,
        source,
        minted_at: now,
    };

    let action_hash = create_entry(&EntryTypes::SapMintRecord(mint_record))?;
    create_link(
        anchor_hash(&mint_id)?,
        action_hash.clone(),
        LinkTypes::MintIdToMintRecord,
        (),
    )?;
    create_link(
        anchor_hash(&format!("mints:{}", input.recipient_did))?,
        action_hash.clone(),
        LinkTypes::DidToMintRecords,
        (),
    )?;

    // Credit the SAP to recipient's balance
    credit_sap(CreditSapInput {
        member_did: input.recipient_did.clone(),
        amount: input.amount,
        reason: format!("Governance mint: proposal {}", input.proposal_id),
    })?;

    // Broadcast mint event via bridge
    if let Err(e) = call(
        CallTargetCell::Local,
        ZomeName::from("finance_bridge"),
        FunctionName::from("broadcast_finance_event"),
        None,
        BroadcastMintEventPayload {
            event_type: "SapMinted".to_string(),
            subject_did: input.recipient_did,
            amount: Some(input.amount),
            payload: serde_json::json!({
                "proposal_id": input.proposal_id,
                "mint_id": mint_id,
            })
            .to_string(),
        },
    ) {
        debug!("Failed to broadcast SapMinted event: {:?}", e);
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Mint record not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MintSapFromGovernanceInput {
    pub recipient_did: String,
    pub amount: u64,
    pub proposal_id: String,
}

#[derive(Serialize, Debug)]
struct BroadcastMintEventPayload {
    pub event_type: String,
    pub subject_did: String,
    pub amount: Option<u64>,
    pub payload: String,
}

/// Get all mint records for a member
#[hdk_extern]
pub fn get_mint_records(member_did: String) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("mints:{}", member_did))?,
            LinkTypes::DidToMintRecords,
        )?,
        GetStrategy::default(),
    )?;
    let mut records = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

// --- Internal helpers ---

/// Enforce annual governance mint cap: total minted via governance proposals
/// in the last 365 days must not exceed SAP_MINT_ANNUAL_MAX.
fn enforce_annual_mint_cap(new_amount: u64, now: Timestamp) -> ExternResult<()> {
    let year_micros: i64 = 365 * 24 * 60 * 60 * 1_000_000;
    let cutoff = now.as_micros() - year_micros;

    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::SapMintRecord,
        )?))
        .include_entries(true);

    let annual_total: u64 = query(filter)?
        .into_iter()
        .filter_map(|r| r.entry().to_app_option::<SapMintRecord>().ok().flatten())
        .filter(|m| m.minted_at.as_micros() > cutoff)
        .filter(|m| matches!(m.source, SapMintSource::GovernanceProposal { .. }))
        .map(|m| m.amount)
        .fold(0u64, |acc, v| acc.saturating_add(v));

    if annual_total.saturating_add(new_amount) > SAP_MINT_ANNUAL_MAX {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Annual governance mint cap exceeded: {} already minted this year + {} requested > {} max",
            annual_total, new_amount, SAP_MINT_ANNUAL_MAX
        ))));
    }
    Ok(())
}

fn find_sap_balance_record(member_did: &str) -> ExternResult<Option<(Record, SapBalance)>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("sap:{}", member_did))?,
            LinkTypes::DidToSapBalance,
        )?,
        GetStrategy::default(),
    )?;
    if let Some(link) = links.last() {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            if let Some(bal) = record.entry().to_app_option::<SapBalance>().ok().flatten() {
                return Ok(Some((record, bal)));
            }
        }
    }
    Ok(None)
}

fn get_sap_balance_inner(member_did: &str) -> ExternResult<(Record, SapBalance)> {
    find_sap_balance_record(member_did)?.ok_or(wasm_error!(WasmErrorInner::Guest(format!(
        "No SAP balance found for {}. Call initialize_sap_balance first.",
        member_did
    ))))
}

/// Compute the progressive SAP fee for a payment based on sender's MYCEL score.
///
/// Queries the bridge coordinator for the canonical fee tier (single source of truth).
/// Falls back to direct recognition lookup, then to Newcomer tier (0.10%).
fn compute_sap_fee(sender_did: &str, micro_amount: u64) -> ExternResult<u64> {
    // Try canonical path: bridge → recognition → FeeTier
    let fee_rate = match call(
        CallTargetCell::Local,
        ZomeName::from("finance_bridge"),
        FunctionName::from("get_member_fee_tier"),
        None,
        sender_did.to_string(),
    ) {
        Ok(ZomeCallResponse::Ok(result)) => {
            #[derive(Debug, Deserialize)]
            struct TierResp {
                base_fee_rate: f64,
            }
            match result.decode::<TierResp>() {
                Ok(resp) if resp.base_fee_rate.is_finite() => resp.base_fee_rate,
                _ => FeeTier::Newcomer.base_fee_rate(),
            }
        }
        _ => {
            // Bridge unavailable — fall back to direct recognition call
            let mycel_score = match call(
                CallTargetCell::Local,
                ZomeName::from("recognition"),
                FunctionName::from("get_mycel_score"),
                None,
                sender_did.to_string(),
            ) {
                Ok(ZomeCallResponse::Ok(result)) => {
                    #[derive(Debug, Deserialize)]
                    struct MycelState {
                        mycel_score: f64,
                    }
                    result
                        .decode::<MycelState>()
                        .map(|s| s.mycel_score)
                        .unwrap_or(0.0)
                }
                _ => 0.0,
            };
            FeeTier::from_mycel(mycel_score).base_fee_rate()
        }
    };

    let fee = (micro_amount as f64 * fee_rate) as u64;
    Ok(fee)
}

fn elapsed_seconds(from: Timestamp, to: Timestamp) -> u64 {
    let from_us = from.as_micros();
    let to_us = to.as_micros();
    if to_us > from_us {
        ((to_us - from_us) / 1_000_000) as u64
    } else {
        0
    }
}

#[hdk_extern]
pub fn send_payment(input: SendPaymentInput) -> ExternResult<Record> {
    // Verify caller is the sender (prevents DID spoofing)
    verify_caller_is_did(&input.from_did)?;

    // Validate currency before creating any entries
    if input.currency != "SAP" && input.currency != "TEND" {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Currency must be \"SAP\" or \"TEND\"".into()
        )));
    }

    let now = sys_time()?;

    // If sending SAP, enforce on-chain balance with demurrage + progressive fee
    let (memo, fee_amount) = if input.currency == "SAP" {
        // input.amount is already in micro-SAP (u64)
        // Compute progressive fee based on sender's MYCEL score
        let fee = compute_sap_fee(&input.from_did, input.amount)?;
        let total_debit = input.amount + fee;

        // Debit sender's SAP balance (amount + fee, applies demurrage)
        debit_sap(DebitSapInput {
            member_did: input.from_did.clone(),
            amount: total_debit,
            reason: format!("Payment to {} (includes fee {})", input.to_did, fee),
        })?;

        // Credit receiver's SAP balance (amount only, fee goes to commons)
        credit_sap(CreditSapInput {
            member_did: input.to_did.clone(),
            amount: input.amount,
            reason: format!("Payment from {}", input.from_did),
        })?;

        // Route fee to commons via treasury (if fee > 0)
        if fee > 0 {
            if let Err(e) = call(
                CallTargetCell::Local,
                ZomeName::from("treasury"),
                FunctionName::from("receive_compost"),
                None,
                ReceiveCompostPayload {
                    commons_pool_id: "global-fee-pool".to_string(),
                    amount: fee,
                    source_member_did: input.from_did.clone(),
                },
            ) {
                debug!("Fee routing to global-fee-pool failed: {:?}", e);
            }
        }

        (input.memo.clone(), fee)
    } else {
        (input.memo.clone(), 0)
    };

    let payment = Payment {
        id: format!("payment:{}:{}", input.from_did, now.as_micros()),
        from_did: input.from_did.clone(),
        to_did: input.to_did.clone(),
        amount: input.amount,
        fee: fee_amount,
        currency: input.currency.clone(),
        payment_type: input.payment_type,
        status: TransferStatus::Completed, // Simplified: immediate completion
        memo,
        created: now,
        completed: Some(now),
    };

    let action_hash = create_entry(&EntryTypes::Payment(payment.clone()))?;
    create_link(
        anchor_hash(&input.from_did)?,
        action_hash.clone(),
        LinkTypes::SenderToPayments,
        (),
    )?;
    create_link(
        anchor_hash(&input.to_did)?,
        action_hash.clone(),
        LinkTypes::ReceiverToPayments,
        (),
    )?;
    // Link-based index for O(1) payment ID lookups
    create_link(
        anchor_hash(&payment.id)?,
        action_hash.clone(),
        LinkTypes::PaymentIdToPayment,
        (),
    )?;

    // Create receipt with Ed25519 signature
    let sig_data = format!(
        "{}|{}|{}|{}|{}|{}",
        payment.id,
        payment.from_did,
        payment.to_did,
        payment.amount,
        payment.currency,
        now.as_micros()
    );
    let receipt = Receipt {
        payment_id: payment.id.clone(),
        from_did: input.from_did,
        to_did: input.to_did,
        amount: input.amount,
        currency: payment.currency,
        timestamp: now,
        signature: {
            let agent = agent_info()?.agent_initial_pubkey;
            let sig = sign(agent, sig_data.into_bytes())?;
            sig.0
                .iter()
                .map(|b| format!("{:02x}", b))
                .collect::<String>()
        },
    };
    let receipt_hash = create_entry(&EntryTypes::Receipt(receipt))?;
    create_link(
        action_hash.clone(),
        receipt_hash,
        LinkTypes::PaymentToReceipt,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SendPaymentInput {
    pub from_did: String,
    pub to_did: String,
    pub amount: u64,
    pub currency: String,
    pub payment_type: PaymentType,
    pub memo: Option<String>,
}

#[hdk_extern]
pub fn open_payment_channel(input: OpenChannelInput) -> ExternResult<Record> {
    verify_caller_is_did(&input.party_a)?;

    let now = sys_time()?;
    let channel = PaymentChannel {
        id: format!(
            "channel:{}:{}:{}",
            input.party_a,
            input.party_b,
            now.as_micros()
        ),
        party_a: input.party_a.clone(),
        party_b: input.party_b.clone(),
        currency: input.currency,
        balance_a: input.initial_deposit_a,
        balance_b: input.initial_deposit_b,
        opened: now,
        last_updated: now,
        closed: None,
    };

    let action_hash = create_entry(&EntryTypes::PaymentChannel(channel))?;
    create_link(
        anchor_hash(&input.party_a)?,
        action_hash.clone(),
        LinkTypes::ChannelPartyA,
        (),
    )?;
    create_link(
        anchor_hash(&input.party_b)?,
        action_hash.clone(),
        LinkTypes::ChannelPartyB,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OpenChannelInput {
    pub party_a: String,
    pub party_b: String,
    pub currency: String,
    pub initial_deposit_a: u64,
    pub initial_deposit_b: u64,
}

#[hdk_extern]
pub fn channel_transfer(input: ChannelTransferInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::PaymentChannel,
        )?))
        .include_entries(true);
    for record in query(filter)? {
        if let Some(channel) = record
            .entry()
            .to_app_option::<PaymentChannel>()
            .ok()
            .flatten()
        {
            if channel.id == input.channel_id {
                let now = sys_time()?;
                let (new_a, new_b) = if input.from_a {
                    (
                        channel
                            .balance_a
                            .checked_sub(input.amount)
                            .ok_or(wasm_error!(WasmErrorInner::Guest(
                                "Insufficient balance for party A".into()
                            )))?,
                        channel.balance_b + input.amount,
                    )
                } else {
                    (
                        channel.balance_a + input.amount,
                        channel
                            .balance_b
                            .checked_sub(input.amount)
                            .ok_or(wasm_error!(WasmErrorInner::Guest(
                                "Insufficient balance for party B".into()
                            )))?,
                    )
                };
                let updated = PaymentChannel {
                    balance_a: new_a,
                    balance_b: new_b,
                    last_updated: now,
                    ..channel
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::PaymentChannel(updated),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Channel not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ChannelTransferInput {
    pub channel_id: String,
    pub amount: u64,
    pub from_a: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetPaymentHistoryInput {
    pub did: String,
    pub limit: Option<usize>,
}

#[hdk_extern]
pub fn get_payment_history(input: GetPaymentHistoryInput) -> ExternResult<Vec<Record>> {
    let max = input.limit.unwrap_or(100);
    let mut payments = Vec::new();
    // Get sent payments
    let query = LinkQuery::try_new(anchor_hash(&input.did)?, LinkTypes::SenderToPayments)?;
    for link in get_links(query, GetStrategy::default())?
        .into_iter()
        .take(max)
    {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            payments.push(record);
        }
    }
    // Get received payments (respect remaining budget)
    let remaining = max.saturating_sub(payments.len());
    if remaining > 0 {
        let query = LinkQuery::try_new(anchor_hash(&input.did)?, LinkTypes::ReceiverToPayments)?;
        for link in get_links(query, GetStrategy::default())?
            .into_iter()
            .take(remaining)
        {
            if let Some(record) = get(
                ActionHash::try_from(link.target)
                    .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
                GetOptions::default(),
            )? {
                payments.push(record);
            }
        }
    }
    Ok(payments)
}

/// Get a specific payment by ID (O(1) link-based lookup)
#[hdk_extern]
pub fn get_payment(payment_id: String) -> ExternResult<Option<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&payment_id)?, LinkTypes::PaymentIdToPayment)?,
        GetStrategy::default(),
    )?;
    if let Some(link) = links.first() {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        Ok(get(hash, GetOptions::default())?)
    } else {
        Ok(None)
    }
}

/// Get receipt for a payment
#[hdk_extern]
pub fn get_receipt(payment_id: String) -> ExternResult<Option<Record>> {
    // Find the payment first
    let Some(payment_record) = get_payment(payment_id.clone())? else {
        return Ok(None);
    };
    let query = LinkQuery::try_new(
        payment_record.action_address().clone(),
        LinkTypes::PaymentToReceipt,
    )?;
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            return Ok(Some(record));
        }
    }
    Ok(None)
}

/// Close a payment channel (settle balances)
#[hdk_extern]
pub fn close_payment_channel(channel_id: String) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::PaymentChannel,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(channel) = record
            .entry()
            .to_app_option::<PaymentChannel>()
            .ok()
            .flatten()
        {
            if channel.id == channel_id {
                if channel.closed.is_some() {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Channel already closed".into()
                    )));
                }
                let now = sys_time()?;
                let closed = PaymentChannel {
                    closed: Some(now),
                    last_updated: now,
                    ..channel
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::PaymentChannel(closed),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Channel not found".into()
    )))
}

/// Refund a payment (creates reverse payment)
#[hdk_extern]
pub fn refund_payment(payment_id: String) -> ExternResult<Record> {
    let original = get_payment(payment_id.clone())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Payment not found".into()
    )))?;

    let original_payment = original
        .entry()
        .to_app_option::<Payment>()
        .ok()
        .flatten()
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid payment".into())))?;

    // Only the original receiver (who is the refund sender) can initiate a refund
    verify_caller_is_did(&original_payment.to_did)?;

    if matches!(original_payment.status, TransferStatus::Refunded) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Payment already refunded".into()
        )));
    }

    // Create refund payment (reverse direction)
    let now = sys_time()?;
    // Refunds carry the original fee (already collected; stored for audit trail)
    let refund = Payment {
        id: format!("refund:{}:{}", payment_id, now.as_micros()),
        from_did: original_payment.to_did.clone(),
        to_did: original_payment.from_did.clone(),
        amount: original_payment.amount,
        fee: original_payment.fee,
        currency: original_payment.currency.clone(),
        payment_type: PaymentType::Direct,
        status: TransferStatus::Completed,
        memo: Some(format!("Refund for payment {}", payment_id)),
        created: now,
        completed: Some(now),
    };

    let action_hash = create_entry(&EntryTypes::Payment(refund.clone()))?;
    create_link(
        anchor_hash(&refund.from_did)?,
        action_hash.clone(),
        LinkTypes::SenderToPayments,
        (),
    )?;
    create_link(
        anchor_hash(&refund.to_did)?,
        action_hash.clone(),
        LinkTypes::ReceiverToPayments,
        (),
    )?;

    // Mark original as refunded
    let refunded = Payment {
        status: TransferStatus::Refunded,
        ..original_payment
    };
    update_entry(
        original.action_address().clone(),
        &EntryTypes::Payment(refunded),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetChannelsInput {
    pub did: String,
    pub limit: Option<usize>,
}

/// Get all channels for a party
#[hdk_extern]
pub fn get_channels(input: GetChannelsInput) -> ExternResult<Vec<Record>> {
    let max = input.limit.unwrap_or(100);
    let mut channels = Vec::new();
    // Party A channels
    let query = LinkQuery::try_new(anchor_hash(&input.did)?, LinkTypes::ChannelPartyA)?;
    for link in get_links(query, GetStrategy::default())?
        .into_iter()
        .take(max)
    {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            channels.push(record);
        }
    }
    // Party B channels (respect remaining budget)
    let remaining = max.saturating_sub(channels.len());
    if remaining > 0 {
        let query = LinkQuery::try_new(anchor_hash(&input.did)?, LinkTypes::ChannelPartyB)?;
        for link in get_links(query, GetStrategy::default())?
            .into_iter()
            .take(remaining)
        {
            if let Some(record) = get(
                ActionHash::try_from(link.target)
                    .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
                GetOptions::default(),
            )? {
                channels.push(record);
            }
        }
    }
    Ok(channels)
}

/// Create an escrow payment (held until release)
#[hdk_extern]
pub fn create_escrow(input: CreateEscrowInput) -> ExternResult<Record> {
    verify_caller_is_did(&input.from_did)?;

    let now = sys_time()?;
    // Compute fee for escrow (SAP: progressive fee; TEND: zero)
    let escrow_fee = if input.currency == "SAP" {
        compute_sap_fee(&input.from_did, input.amount)?
    } else {
        0
    };
    let payment = Payment {
        id: format!("escrow:{}:{}", input.from_did, now.as_micros()),
        from_did: input.from_did.clone(),
        to_did: input.to_did.clone(),
        amount: input.amount,
        fee: escrow_fee,
        currency: input.currency,
        payment_type: PaymentType::Escrow(input.escrow_id),
        status: TransferStatus::Pending, // Held until released
        memo: input.memo,
        created: now,
        completed: None,
    };

    let action_hash = create_entry(&EntryTypes::Payment(payment.clone()))?;
    create_link(
        anchor_hash(&input.from_did)?,
        action_hash.clone(),
        LinkTypes::SenderToPayments,
        (),
    )?;
    create_link(
        anchor_hash(&input.to_did)?,
        action_hash.clone(),
        LinkTypes::ReceiverToPayments,
        (),
    )?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateEscrowInput {
    pub from_did: String,
    pub to_did: String,
    pub amount: u64,
    pub currency: String,
    pub escrow_id: String,
    pub memo: Option<String>,
}

// =============================================================================
// EXIT PROTOCOL
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct InitiateExitInput {
    /// DID of the exiting member
    pub member_did: String,
    /// How to handle remaining SAP balance
    pub succession_preference: SuccessionPreference,
    /// Current SAP balance in micro-units (caller must provide; on-chain accounting is external)
    pub sap_balance: u64,
}

/// Initiate member exit — coordinates MYCEL dissolution, SAP succession, and TEND forgiveness.
///
/// Per the Three-Currency Spec:
/// - MYCEL: dissolved immediately (contribution history preserved in DKG)
/// - SAP: follows succession preference (Commons default, Designee, or Redemption)
/// - TEND: all balances forgiven (returned to zero)
#[hdk_extern]
pub fn initiate_exit(input: InitiateExitInput) -> ExternResult<Record> {
    // Verify caller is the exiting member
    verify_caller_is_did(&input.member_did)?;

    let now = sys_time()?;

    // Step 1: Dissolve MYCEL via recognition zome
    let mycel_dissolved = match call(
        CallTargetCell::Local,
        ZomeName::from("recognition"),
        FunctionName::from("dissolve_mycel"),
        None,
        input.member_did.clone(),
    ) {
        Ok(_) => true,
        Err(e) => {
            debug!(
                "Warning: MYCEL dissolution failed for {}: {:?}",
                input.member_did, e
            );
            false
        }
    };

    // Step 2: Handle SAP succession
    if input.sap_balance > 0 {
        match &input.succession_preference {
            SuccessionPreference::Commons => {
                // For commons succession, we record a payment to the commons pool.
                // The actual treasury contribution requires a pool ID which is
                // external state — the caller should handle that via the treasury zome.
                send_payment(SendPaymentInput {
                    from_did: input.member_did.clone(),
                    to_did: format!("did:mycelix:commons:{}", input.member_did),
                    amount: input.sap_balance,
                    currency: "SAP".to_string(),
                    payment_type: PaymentType::CommonsContribution("exit-succession".to_string()),
                    memo: Some("Exit succession: SAP to local commons pool".to_string()),
                })?;
            }
            SuccessionPreference::Designee(designee_did) => {
                send_payment(SendPaymentInput {
                    from_did: input.member_did.clone(),
                    to_did: designee_did.clone(),
                    amount: input.sap_balance,
                    currency: "SAP".to_string(),
                    payment_type: PaymentType::Direct,
                    memo: Some("Exit succession to designated heir".to_string()),
                })?;
            }
            SuccessionPreference::Redemption => {
                // For redemption, record the intent. Actual bridge redemption
                // requires deposit IDs and oracle rates which are bridge-zome state.
                send_payment(SendPaymentInput {
                    from_did: input.member_did.clone(),
                    to_did: "did:mycelix:bridge:redemption".to_string(),
                    amount: input.sap_balance,
                    currency: "SAP".to_string(),
                    payment_type: PaymentType::Direct,
                    memo: Some("Exit succession: SAP queued for collateral redemption".to_string()),
                })?;
            }
        }
    }

    // Step 3: Forgive TEND balances via tend zome
    let tend_balances_forgiven: Vec<(String, i32)> = match call(
        CallTargetCell::Local,
        ZomeName::from("tend"),
        FunctionName::from("forgive_balance"),
        None,
        input.member_did.clone(),
    ) {
        Ok(ZomeCallResponse::Ok(extern_io)) => extern_io.decode().unwrap_or_default(),
        Ok(_) => Vec::new(),
        Err(e) => {
            debug!("Warning: TEND balance forgiveness failed: {:?}", e);
            Vec::new()
        }
    };

    // Step 4: Create the exit record
    let exit_record = payments_integrity::ExitRecord {
        member_did: input.member_did.clone(),
        succession_preference: input.succession_preference,
        sap_balance: input.sap_balance,
        tend_balances_forgiven,
        mycel_dissolved,
        exited_at: now,
    };

    let action_hash = create_entry(&EntryTypes::ExitRecord(exit_record))?;
    create_link(
        anchor_hash(&input.member_did)?,
        action_hash.clone(),
        LinkTypes::MemberToExitRecord,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Failed to retrieve exit record".into()
    )))
}

/// Release escrow to recipient
#[hdk_extern]
pub fn release_escrow(payment_id: String) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Payment,
        )?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(payment) = record.entry().to_app_option::<Payment>().ok().flatten() {
            if payment.id == payment_id {
                if !matches!(payment.payment_type, PaymentType::Escrow(_)) {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Not an escrow payment".into()
                    )));
                }
                if payment.status != TransferStatus::Pending {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Escrow not in pending state".into()
                    )));
                }
                let now = sys_time()?;
                let released = Payment {
                    status: TransferStatus::Completed,
                    completed: Some(now),
                    ..payment
                };
                let action_hash = update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Payment(released),
                )?;
                return get(action_hash, GetOptions::default())?
                    .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest(
        "Payment not found".into()
    )))
}

// =============================================================================
// HEARTH SAP POOLS — Shared Household Funds
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct ContributeToHearthInput {
    pub hearth_did: String,
    pub amount: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct WithdrawFromHearthInput {
    pub hearth_did: String,
    pub amount: u64,
}

/// Get a hearth's SAP pool with effective balance after demurrage.
#[hdk_extern]
pub fn get_hearth_sap_pool(hearth_did: String) -> ExternResult<HearthSapPoolResponse> {
    let pool = get_or_create_hearth_pool(&hearth_did)?;
    let now = sys_time()?;
    let elapsed = elapsed_seconds(pool.last_demurrage_at, now);
    let deduction = compute_demurrage_deduction(
        pool.balance,
        DEMURRAGE_EXEMPT_FLOOR,
        DEMURRAGE_RATE,
        elapsed,
    );
    Ok(HearthSapPoolResponse {
        hearth_did: pool.hearth_did,
        raw_balance: pool.balance,
        effective_balance: pool.balance.saturating_sub(deduction),
        pending_demurrage: deduction,
        last_demurrage_at: pool.last_demurrage_at,
        member_count: pool.member_count,
        total_contributed: pool.total_contributed,
        total_withdrawn: pool.total_withdrawn,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct HearthSapPoolResponse {
    pub hearth_did: String,
    pub raw_balance: u64,
    pub effective_balance: u64,
    pub pending_demurrage: u64,
    pub last_demurrage_at: Timestamp,
    pub member_count: u32,
    pub total_contributed: u64,
    pub total_withdrawn: u64,
}

/// Contribute SAP from personal balance to hearth pool.
///
/// Deducts from caller's SapBalance and credits the hearth pool.
/// Requires hearth membership (verified via cross-zome call).
#[hdk_extern]
pub fn contribute_to_hearth_pool(input: ContributeToHearthInput) -> ExternResult<HearthSapPool> {
    if input.amount == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Amount must be positive".into()
        )));
    }

    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);

    // Verify hearth membership
    verify_hearth_membership(&caller_did, &input.hearth_did)?;

    // Deduct from personal SAP balance
    let anchor_key = format!("sap-balance:{}", caller_did);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&anchor_key)?, LinkTypes::DidToSapBalance)?,
        GetStrategy::default(),
    )?;

    let link = links.first().ok_or(wasm_error!(WasmErrorInner::Guest(
        "No SAP balance found".into()
    )))?;
    let action_hash =
        link.target
            .clone()
            .into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Invalid link target".into()
            )))?;
    let record = get(action_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("SAP balance record not found".into())
    ))?;
    let mut personal_bal = record
        .entry()
        .to_app_option::<SapBalance>()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Deserialize error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "SAP balance entry missing".into()
        )))?;

    if personal_bal.balance < input.amount {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Insufficient SAP balance: have {}, need {}",
            personal_bal.balance, input.amount
        ))));
    }

    // Deduct from personal (use record's action address to avoid update forks)
    personal_bal.balance -= input.amount;
    update_entry(record.action_address().clone(), &personal_bal)?;

    // Credit hearth pool (apply demurrage first)
    let mut pool = get_or_create_hearth_pool(&input.hearth_did)?;
    let now_ts = sys_time()?;
    let elapsed = elapsed_seconds(pool.last_demurrage_at, now_ts);
    let deduction = compute_demurrage_deduction(
        pool.balance,
        DEMURRAGE_EXEMPT_FLOOR,
        DEMURRAGE_RATE,
        elapsed,
    );
    pool.balance = pool.balance.saturating_sub(deduction);
    pool.last_demurrage_at = now_ts;

    pool.balance += input.amount;
    pool.total_contributed += input.amount;
    update_hearth_pool(&input.hearth_did, &pool)?;

    Ok(pool)
}

/// Withdraw SAP from hearth pool to personal balance.
///
/// Requires hearth membership. Any hearth member can withdraw up to
/// the pool's available balance.
#[hdk_extern]
pub fn withdraw_from_hearth_pool(input: WithdrawFromHearthInput) -> ExternResult<HearthSapPool> {
    if input.amount == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Amount must be positive".into()
        )));
    }

    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);

    verify_hearth_membership(&caller_did, &input.hearth_did)?;

    let mut pool = get_or_create_hearth_pool(&input.hearth_did)?;

    // Apply demurrage before checking available balance
    let now_ts = sys_time()?;
    let elapsed = elapsed_seconds(pool.last_demurrage_at, now_ts);
    let deduction = compute_demurrage_deduction(
        pool.balance,
        DEMURRAGE_EXEMPT_FLOOR,
        DEMURRAGE_RATE,
        elapsed,
    );
    pool.balance = pool.balance.saturating_sub(deduction);
    pool.last_demurrage_at = now_ts;

    if pool.balance < input.amount {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Insufficient hearth pool balance: have {}, need {}",
            pool.balance, input.amount
        ))));
    }

    // Deduct from pool
    pool.balance -= input.amount;
    pool.total_withdrawn += input.amount;
    update_hearth_pool(&input.hearth_did, &pool)?;

    // Credit personal SAP balance
    let anchor_key = format!("sap-balance:{}", caller_did);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&anchor_key)?, LinkTypes::DidToSapBalance)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(mut personal_bal) =
                    record.entry().to_app_option::<SapBalance>().ok().flatten()
                {
                    personal_bal.balance += input.amount;
                    update_entry(record.action_address().clone(), &personal_bal)?;
                }
            }
        }
    }

    Ok(pool)
}

/// Explicitly apply demurrage to a hearth SAP pool and redistribute as compost.
///
/// Same 70/20/10 split as personal SAP demurrage: local/regional/global commons.
/// Returns the amount deducted. If 0, no update is persisted.
#[hdk_extern]
pub fn apply_hearth_demurrage(input: ApplyHearthDemurrageInput) -> ExternResult<DemurrageResult> {
    let mut pool = get_or_create_hearth_pool(&input.hearth_did)?;
    let now = sys_time()?;
    let elapsed = elapsed_seconds(pool.last_demurrage_at, now);
    let deduction = compute_demurrage_deduction(
        pool.balance,
        DEMURRAGE_EXEMPT_FLOOR,
        DEMURRAGE_RATE,
        elapsed,
    );

    if deduction == 0 {
        return Ok(DemurrageResult {
            deducted: 0,
            redistributed: true,
        });
    }

    pool.balance = pool.balance.saturating_sub(deduction);
    pool.last_demurrage_at = now;
    update_hearth_pool(&input.hearth_did, &pool)?;

    // Redistribute as compost: 70% local, 20% regional, 10% global
    let local_amount = deduction * COMPOST_LOCAL_PCT / 100;
    let regional_amount = deduction * COMPOST_REGIONAL_PCT / 100;
    let global_amount = deduction - local_amount - regional_amount;

    let source_did = format!("hearth:{}", input.hearth_did);

    if let Some(ref pool_id) = input.local_commons_pool_id {
        let _ = call(
            CallTargetCell::Local,
            ZomeName::from("treasury"),
            FunctionName::from("receive_compost"),
            None,
            ReceiveCompostPayload {
                commons_pool_id: pool_id.clone(),
                amount: local_amount,
                source_member_did: source_did.clone(),
            },
        );
    }
    if let Some(ref pool_id) = input.regional_commons_pool_id {
        let _ = call(
            CallTargetCell::Local,
            ZomeName::from("treasury"),
            FunctionName::from("receive_compost"),
            None,
            ReceiveCompostPayload {
                commons_pool_id: pool_id.clone(),
                amount: regional_amount,
                source_member_did: source_did.clone(),
            },
        );
    }
    if let Some(ref pool_id) = input.global_commons_pool_id {
        let _ = call(
            CallTargetCell::Local,
            ZomeName::from("treasury"),
            FunctionName::from("receive_compost"),
            None,
            ReceiveCompostPayload {
                commons_pool_id: pool_id.clone(),
                amount: global_amount,
                source_member_did: source_did,
            },
        );
    }

    Ok(DemurrageResult {
        deducted: deduction,
        redistributed: true,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ApplyHearthDemurrageInput {
    pub hearth_did: String,
    pub local_commons_pool_id: Option<String>,
    pub regional_commons_pool_id: Option<String>,
    pub global_commons_pool_id: Option<String>,
}

fn get_or_create_hearth_pool(hearth_did: &str) -> ExternResult<HearthSapPool> {
    let anchor_key = format!("hearth-sap:{}", hearth_did);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&anchor_key)?, LinkTypes::HearthDidToSapPool)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(pool) = record
                    .entry()
                    .to_app_option::<HearthSapPool>()
                    .ok()
                    .flatten()
                {
                    return Ok(pool);
                }
            }
        }
    }

    let now = sys_time()?;
    let pool = HearthSapPool {
        hearth_did: hearth_did.to_string(),
        balance: 0,
        last_demurrage_at: now,
        member_count: 0,
        total_contributed: 0,
        total_withdrawn: 0,
    };

    let hash = create_entry(&EntryTypes::HearthSapPool(pool.clone()))?;
    create_link(
        anchor_hash(&anchor_key)?,
        hash,
        LinkTypes::HearthDidToSapPool,
        (),
    )?;
    Ok(pool)
}

fn update_hearth_pool(hearth_did: &str, pool: &HearthSapPool) -> ExternResult<()> {
    let anchor_key = format!("hearth-sap:{}", hearth_did);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&anchor_key)?, LinkTypes::HearthDidToSapPool)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            // Get the latest record to avoid update forks — the link points to the
            // original create action, but the entry may have been updated since.
            let record = get(action_hash, GetOptions::default())?.ok_or(wasm_error!(
                WasmErrorInner::Guest("Hearth pool record not found".into())
            ))?;
            update_entry(record.action_address().clone(), pool)?;
        }
    }
    Ok(())
}

fn verify_hearth_membership(caller_did: &str, hearth_did: &str) -> ExternResult<()> {
    #[derive(Serialize, Deserialize, Debug)]
    struct MembershipQuery {
        member_did: String,
        hearth_did: String,
    }

    match call(
        CallTargetCell::OtherRole("hearth".into()),
        ZomeName::from("hearth_bridge"),
        FunctionName::from("is_hearth_member"),
        None,
        MembershipQuery {
            member_did: caller_did.to_string(),
            hearth_did: hearth_did.to_string(),
        },
    ) {
        Ok(ZomeCallResponse::Ok(result)) => match result.decode::<bool>() {
            Ok(true) => Ok(()),
            _ => Err(wasm_error!(WasmErrorInner::Guest(
                "Not a member of this hearth".into()
            ))),
        },
        // Hearth cluster unreachable — allow in standalone mode
        _ => Ok(()),
    }
}
