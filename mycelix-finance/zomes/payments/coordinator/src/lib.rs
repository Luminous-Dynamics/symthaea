//! Payments Coordinator Zome
use hdk::prelude::*;
use payments_integrity::*;

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

/// Compute demurrage deduction on SAP balances.
///
/// Implements: eligible * (1 - e^(-rate * years))
/// where eligible = max(balance - exempt_floor, 0) and years = seconds_elapsed / 31_536_000.
///
/// Returns the amount to deduct (in the same integer unit as balance).
pub fn compute_demurrage_deduction(balance: u64, exempt_floor: u64, rate: f64, seconds_elapsed: u64) -> u64 {
    if balance <= exempt_floor || seconds_elapsed == 0 {
        return 0;
    }
    let eligible = (balance - exempt_floor) as f64;
    let years = seconds_elapsed as f64 / 31_536_000.0;
    let decay = 1.0 - (-rate * years).exp();
    let deduction = eligible * decay;
    // Clamp to eligible amount and floor at zero
    if deduction < 0.0 {
        0
    } else if deduction > eligible {
        eligible as u64
    } else {
        deduction as u64
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DemurrageOnSendInput {
    pub balance: u64,
    pub exempt_floor: u64,
    pub rate: f64,
    pub last_demurrage_at: u64, // epoch seconds of last demurrage computation
    pub now: u64,               // current epoch seconds
}

#[hdk_extern]
pub fn send_payment(input: SendPaymentInput) -> ExternResult<Record> {
    // Validate currency before creating any entries
    if input.currency != "SAP" && input.currency != "TEND" {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Currency must be \"SAP\" or \"TEND\"".into()
        )));
    }

    let now = sys_time()?;

    // If sending SAP, compute informational demurrage note for memo
    let memo = if input.currency == "SAP" {
        if let Some(ref demurrage) = input.demurrage {
            let seconds_elapsed = if demurrage.now > demurrage.last_demurrage_at {
                demurrage.now - demurrage.last_demurrage_at
            } else {
                0
            };
            let deduction = compute_demurrage_deduction(
                demurrage.balance,
                demurrage.exempt_floor,
                demurrage.rate,
                seconds_elapsed,
            );
            if deduction > 0 {
                let base_memo = input.memo.clone().unwrap_or_default();
                Some(format!(
                    "{}{}[demurrage: {} SAP pending deduction over {}s]",
                    base_memo,
                    if base_memo.is_empty() { "" } else { " " },
                    deduction,
                    seconds_elapsed
                ))
            } else {
                input.memo.clone()
            }
        } else {
            input.memo.clone()
        }
    } else {
        input.memo.clone()
    };

    let payment = Payment {
        id: format!("payment:{}:{}", input.from_did, now.as_micros()),
        from_did: input.from_did.clone(),
        to_did: input.to_did.clone(),
        amount: input.amount,
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

    // Create receipt
    let receipt = Receipt {
        payment_id: payment.id.clone(),
        from_did: input.from_did,
        to_did: input.to_did,
        amount: input.amount,
        currency: payment.currency,
        timestamp: now,
        signature: format!("sig:{}", now.as_micros()), // Simplified
    };
    let receipt_hash = create_entry(&EntryTypes::Receipt(receipt))?;
    create_link(action_hash.clone(), receipt_hash, LinkTypes::PaymentToReceipt, ())?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest("Not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SendPaymentInput {
    pub from_did: String,
    pub to_did: String,
    pub amount: f64,
    pub currency: String,
    pub payment_type: PaymentType,
    pub memo: Option<String>,
    pub demurrage: Option<DemurrageOnSendInput>,
}

#[hdk_extern]
pub fn open_payment_channel(input: OpenChannelInput) -> ExternResult<Record> {
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
    pub initial_deposit_a: f64,
    pub initial_deposit_b: f64,
}

#[hdk_extern]
pub fn channel_transfer(input: ChannelTransferInput) -> ExternResult<Record> {
    let filter = ChainQueryFilter::new().entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::PaymentChannel)?)).include_entries(true);
    for record in query(filter)? {
        if let Some(channel) = record.entry().to_app_option::<PaymentChannel>().ok().flatten() {
            if channel.id == input.channel_id {
                let now = sys_time()?;
                let (new_a, new_b) = if input.from_a {
                    (channel.balance_a - input.amount, channel.balance_b + input.amount)
                } else {
                    (channel.balance_a + input.amount, channel.balance_b - input.amount)
                };
                if new_a < 0.0 || new_b < 0.0 {
                    return Err(wasm_error!(WasmErrorInner::Guest("Insufficient balance".into())));
                }
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
    pub amount: f64,
    pub from_a: bool,
}

#[hdk_extern]
pub fn get_payment_history(did: String) -> ExternResult<Vec<Record>> {
    let mut payments = Vec::new();
    // Get sent payments
    let query = LinkQuery::try_new(anchor_hash(&did)?, LinkTypes::SenderToPayments)?;
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(ActionHash::try_from(link.target).map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?, GetOptions::default())? {
            payments.push(record);
        }
    }
    // Get received payments
    let query = LinkQuery::try_new(anchor_hash(&did)?, LinkTypes::ReceiverToPayments)?;
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(ActionHash::try_from(link.target).map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?, GetOptions::default())? {
            payments.push(record);
        }
    }
    Ok(payments)
}

/// Get a specific payment by ID
#[hdk_extern]
pub fn get_payment(payment_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(UnitEntryTypes::Payment)?))
        .include_entries(true);

    for record in query(filter)? {
        if let Some(payment) = record.entry().to_app_option::<Payment>().ok().flatten() {
            if payment.id == payment_id {
                return Ok(Some(record));
            }
        }
    }
    Ok(None)
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

    if matches!(original_payment.status, TransferStatus::Refunded) {
        return Err(wasm_error!(WasmErrorInner::Guest("Payment already refunded".into())));
    }

    // Create refund payment (reverse direction)
    let now = sys_time()?;
    let refund = Payment {
        id: format!("refund:{}:{}", payment_id, now.as_micros()),
        from_did: original_payment.to_did.clone(),
        to_did: original_payment.from_did.clone(),
        amount: original_payment.amount,
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

/// Get all channels for a party
#[hdk_extern]
pub fn get_channels(did: String) -> ExternResult<Vec<Record>> {
    let mut channels = Vec::new();
    // Party A channels
    let query = LinkQuery::try_new(anchor_hash(&did)?, LinkTypes::ChannelPartyA)?;
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(ActionHash::try_from(link.target).map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?, GetOptions::default())? {
            channels.push(record);
        }
    }
    // Party B channels
    let query = LinkQuery::try_new(anchor_hash(&did)?, LinkTypes::ChannelPartyB)?;
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(ActionHash::try_from(link.target).map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?, GetOptions::default())? {
            channels.push(record);
        }
    }
    Ok(channels)
}

/// Create an escrow payment (held until release)
#[hdk_extern]
pub fn create_escrow(input: CreateEscrowInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let payment = Payment {
        id: format!("escrow:{}:{}", input.from_did, now.as_micros()),
        from_did: input.from_did.clone(),
        to_did: input.to_did.clone(),
        amount: input.amount,
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
    pub amount: f64,
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
    pub succession_preference: payments_integrity::SuccessionPreference,
    /// Current SAP balance (caller must provide; on-chain accounting is external)
    pub sap_balance: f64,
}

/// Initiate member exit — coordinates MYCEL dissolution, SAP succession, and TEND forgiveness.
///
/// Per the Three-Currency Spec:
/// - MYCEL: dissolved immediately (contribution history preserved in DKG)
/// - SAP: follows succession preference (Commons default, Designee, or Redemption)
/// - TEND: all balances forgiven (returned to zero)
#[hdk_extern]
pub fn initiate_exit(input: InitiateExitInput) -> ExternResult<Record> {
    if !input.member_did.starts_with("did:") {
        return Err(wasm_error!(WasmErrorInner::Guest("Member must be a valid DID".into())));
    }

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
    if input.sap_balance > 0.0 {
        match &input.succession_preference {
            payments_integrity::SuccessionPreference::Commons => {
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
                    demurrage: None,
                })?;
            }
            payments_integrity::SuccessionPreference::Designee(designee_did) => {
                send_payment(SendPaymentInput {
                    from_did: input.member_did.clone(),
                    to_did: designee_did.clone(),
                    amount: input.sap_balance,
                    currency: "SAP".to_string(),
                    payment_type: PaymentType::Direct,
                    memo: Some("Exit succession to designated heir".to_string()),
                    demurrage: None,
                })?;
            }
            payments_integrity::SuccessionPreference::Redemption => {
                // For redemption, record the intent. Actual bridge redemption
                // requires deposit IDs and oracle rates which are bridge-zome state.
                send_payment(SendPaymentInput {
                    from_did: input.member_did.clone(),
                    to_did: format!("did:mycelix:bridge:redemption"),
                    amount: input.sap_balance,
                    currency: "SAP".to_string(),
                    payment_type: PaymentType::Direct,
                    memo: Some("Exit succession: SAP queued for collateral redemption".to_string()),
                    demurrage: None,
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
        LinkTypes::SenderToPayments,
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
