//! Payments Coordinator Zome
use hdk::prelude::*;
use payments_integrity::*;
use mycelix_finance_shared::{anchor_hash, verify_caller_is_did};
use mycelix_finance_types::{
    compute_demurrage_deduction, SuccessionPreference, FeeTier,
    DEMURRAGE_RATE, DEMURRAGE_EXEMPT_FLOOR,
    COMPOST_LOCAL_PCT, COMPOST_REGIONAL_PCT,
};

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
    let deduction = compute_demurrage_deduction(
        bal.balance,
        DEMURRAGE_EXEMPT_FLOOR,
        DEMURRAGE_RATE,
        elapsed,
    );
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
    let deduction = compute_demurrage_deduction(
        bal.balance,
        DEMURRAGE_EXEMPT_FLOOR,
        DEMURRAGE_RATE,
        elapsed,
    );

    if deduction == 0 {
        return Ok(DemurrageResult { deducted: 0, redistributed: true });
    }

    // Update balance
    let updated = SapBalance {
        balance: bal.balance.saturating_sub(deduction),
        last_demurrage_at: now,
        ..bal
    };
    update_entry(record.action_address().clone(), &EntryTypes::SapBalance(updated))?;

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
            debug!("Compost redistribution to local pool {} failed: {:?}", pool_id, e);
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
            debug!("Compost redistribution to regional pool {} failed: {:?}", pool_id, e);
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
            debug!("Compost redistribution to global pool {} failed: {:?}", pool_id, e);
        }
    }

    Ok(DemurrageResult { deducted: deduction, redistributed: true })
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
            let action_hash = update_entry(record.action_address().clone(), &EntryTypes::SapBalance(updated))?;
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
    let deduction = compute_demurrage_deduction(
        bal.balance,
        DEMURRAGE_EXEMPT_FLOOR,
        DEMURRAGE_RATE,
        elapsed,
    );
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
    let action_hash = update_entry(record.action_address().clone(), &EntryTypes::SapBalance(updated))?;
    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DebitSapInput {
    pub member_did: String,
    pub amount: u64,
    pub reason: String,
}

// --- Internal helpers ---

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
    find_sap_balance_record(member_did)?
        .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
            "No SAP balance found for {}. Call initialize_sap_balance first.",
            member_did
        ))))
}

/// Compute the progressive SAP fee for a payment based on sender's MYCEL score.
///
/// Fetches the sender's MYCEL score via cross-zome call to recognition,
/// determines their FeeTier, and computes the fee in micro-SAP.
/// Falls back to Newcomer tier (0.10%) if MYCEL lookup fails.
fn compute_sap_fee(sender_did: &str, micro_amount: u64) -> ExternResult<u64> {
    let mycel_score = match call(
        CallTargetCell::Local,
        ZomeName::from("recognition"),
        FunctionName::from("get_mycel_score"),
        None,
        sender_did.to_string(),
    ) {
        Ok(ZomeCallResponse::Ok(result)) => {
            #[derive(Debug, Deserialize)]
            struct MycelState { mycel_score: f64 }
            match result.decode::<MycelState>() {
                Ok(state) => state.mycel_score,
                Err(_) => 0.0, // Fallback: Newcomer tier
            }
        }
        _ => 0.0, // Recognition zome unreachable → Newcomer tier
    };

    let tier = FeeTier::from_mycel(mycel_score);
    let fee = (micro_amount as f64 * tier.base_fee_rate()) as u64;
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
    create_link(anchor_hash(&input.from_did)?, action_hash.clone(), LinkTypes::SenderToPayments, ())?;
    create_link(anchor_hash(&input.to_did)?, action_hash.clone(), LinkTypes::ReceiverToPayments, ())?;
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
        payment.id, payment.from_did, payment.to_did,
        payment.amount, payment.currency, now.as_micros()
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
            sig.0.iter().map(|b| format!("{:02x}", b)).collect::<String>()
        },
    };
    let receipt_hash = create_entry(&EntryTypes::Receipt(receipt))?;
    create_link(action_hash.clone(), receipt_hash, LinkTypes::PaymentToReceipt, ())?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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
        id: format!("channel:{}:{}:{}", input.party_a, input.party_b, now.as_micros()),
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
    create_link(anchor_hash(&input.party_a)?, action_hash.clone(), LinkTypes::ChannelPartyA, ())?;
    create_link(anchor_hash(&input.party_b)?, action_hash.clone(), LinkTypes::ChannelPartyB, ())?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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
    let filter = ChainQueryFilter::new().entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::PaymentChannel)?)).include_entries(true);
    for record in query(filter)? {
        if let Some(channel) = record.entry().to_app_option::<PaymentChannel>().ok().flatten() {
            if channel.id == input.channel_id {
                let now = sys_time()?;
                let (new_a, new_b) = if input.from_a {
                    (
                        channel.balance_a.checked_sub(input.amount)
                            .ok_or(wasm_error!(WasmErrorInner::Guest("Insufficient balance for party A".into())))?,
                        channel.balance_b + input.amount,
                    )
                } else {
                    (
                        channel.balance_a + input.amount,
                        channel.balance_b.checked_sub(input.amount)
                            .ok_or(wasm_error!(WasmErrorInner::Guest("Insufficient balance for party B".into())))?,
                    )
                };
                let updated = PaymentChannel { balance_a: new_a, balance_b: new_b, last_updated: now, ..channel };
                let action_hash = update_entry(record.action_address().clone(), &EntryTypes::PaymentChannel(updated))?;
                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Channel not found".into())))
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
    for link in get_links(query, GetStrategy::default())?.into_iter().take(max) {
        if let Some(record) = get(ActionHash::try_from(link.target).map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?, GetOptions::default())? {
            payments.push(record);
        }
    }
    // Get received payments (respect remaining budget)
    let remaining = max.saturating_sub(payments.len());
    if remaining > 0 {
        let query = LinkQuery::try_new(anchor_hash(&input.did)?, LinkTypes::ReceiverToPayments)?;
        for link in get_links(query, GetStrategy::default())?.into_iter().take(remaining) {
            if let Some(record) = get(ActionHash::try_from(link.target).map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?, GetOptions::default())? {
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
    let query = LinkQuery::try_new(payment_record.action_address().clone(), LinkTypes::PaymentToReceipt)?;
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(ActionHash::try_from(link.target).map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?, GetOptions::default())? {
            return Ok(Some(record));
        }
    }
    Ok(None)
}

/// Close a payment channel (settle balances)
#[hdk_extern]
pub fn close_payment_channel(channel_id: String) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::PaymentChannel)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(channel) = record.entry().to_app_option::<PaymentChannel>().ok().flatten() {
            if channel.id == channel_id {
                if channel.closed.is_some() {
                    return Err(wasm_error!(WasmErrorInner::Guest("Channel already closed".into())));
                }
                let now = sys_time()?;
                let closed = PaymentChannel {
                    closed: Some(now),
                    last_updated: now,
                    ..channel
                };
                let action_hash = update_entry(record.action_address().clone(), &EntryTypes::PaymentChannel(closed))?;
                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Channel not found".into())))
}

/// Refund a payment (creates reverse payment)
#[hdk_extern]
pub fn refund_payment(payment_id: String) -> ExternResult<Record> {
    let original = get_payment(payment_id.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Payment not found".into())))?;

    let original_payment = original.entry().to_app_option::<Payment>().ok().flatten()
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid payment".into())))?;

    // Only the original receiver (who is the refund sender) can initiate a refund
    verify_caller_is_did(&original_payment.to_did)?;

    if matches!(original_payment.status, TransferStatus::Refunded) {
        return Err(wasm_error!(WasmErrorInner::Guest("Payment already refunded".into())));
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
    create_link(anchor_hash(&refund.from_did)?, action_hash.clone(), LinkTypes::SenderToPayments, ())?;
    create_link(anchor_hash(&refund.to_did)?, action_hash.clone(), LinkTypes::ReceiverToPayments, ())?;

    // Mark original as refunded
    let refunded = Payment {
        status: TransferStatus::Refunded,
        ..original_payment
    };
    update_entry(original.action_address().clone(), &EntryTypes::Payment(refunded))?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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
    for link in get_links(query, GetStrategy::default())?.into_iter().take(max) {
        if let Some(record) = get(ActionHash::try_from(link.target).map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?, GetOptions::default())? {
            channels.push(record);
        }
    }
    // Party B channels (respect remaining budget)
    let remaining = max.saturating_sub(channels.len());
    if remaining > 0 {
        let query = LinkQuery::try_new(anchor_hash(&input.did)?, LinkTypes::ChannelPartyB)?;
        for link in get_links(query, GetStrategy::default())?.into_iter().take(remaining) {
            if let Some(record) = get(ActionHash::try_from(link.target).map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?, GetOptions::default())? {
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
    create_link(anchor_hash(&input.from_did)?, action_hash.clone(), LinkTypes::SenderToPayments, ())?;
    create_link(anchor_hash(&input.to_did)?, action_hash.clone(), LinkTypes::ReceiverToPayments, ())?;
    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
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
            debug!("Warning: MYCEL dissolution failed for {}: {:?}", input.member_did, e);
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
                    to_did: format!("did:mycelix:bridge:redemption"),
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
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            extern_io.decode().unwrap_or_default()
        }
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

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Failed to retrieve exit record".into())))
}

/// Release escrow to recipient
#[hdk_extern]
pub fn release_escrow(payment_id: String) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::Payment)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(payment) = record.entry().to_app_option::<Payment>().ok().flatten() {
            if payment.id == payment_id {
                if !matches!(payment.payment_type, PaymentType::Escrow(_)) {
                    return Err(wasm_error!(WasmErrorInner::Guest("Not an escrow payment".into())));
                }
                if payment.status != TransferStatus::Pending {
                    return Err(wasm_error!(WasmErrorInner::Guest("Escrow not in pending state".into())));
                }
                let now = sys_time()?;
                let released = Payment {
                    status: TransferStatus::Completed,
                    completed: Some(now),
                    ..payment
                };
                let action_hash = update_entry(record.action_address().clone(), &EntryTypes::Payment(released))?;
                return get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())));
            }
        }
    }
    Err(wasm_error!(WasmErrorInner::Guest("Payment not found".into())))
}
