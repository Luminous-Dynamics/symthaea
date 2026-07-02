// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Payments Coordinator Zome
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

#[hdk_extern]
pub fn send_payment(input: SendPaymentInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let payment = Payment {
        id: format!("payment:{}:{}", input.from_did, now.as_micros()),
        from_did: input.from_did.clone(),
        to_did: input.to_did.clone(),
        amount: input.amount,
        currency: input.currency,
        payment_type: input.payment_type,
        status: TransferStatus::Completed, // Simplified: immediate completion
        memo: input.memo,
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
    pub amount: f64,
    pub currency: String,
    pub payment_type: PaymentType,
    pub memo: Option<String>,
}

#[hdk_extern]
pub fn open_payment_channel(input: OpenChannelInput) -> ExternResult<Record> {
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
    pub initial_deposit_a: f64,
    pub initial_deposit_b: f64,
}

#[hdk_extern]
pub fn channel_transfer(input: ChannelTransferInput) -> ExternResult<Record> {
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);

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
                // Caller must be a participant (party_a or party_b) of this channel.
                if caller_did != channel.party_a && caller_did != channel.party_b {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only channel participants can transfer within this channel".into()
                    )));
                }
                let now = sys_time()?;
                let (new_a, new_b) = if input.from_a {
                    (
                        channel.balance_a - input.amount,
                        channel.balance_b + input.amount,
                    )
                } else {
                    (
                        channel.balance_a + input.amount,
                        channel.balance_b - input.amount,
                    )
                };
                if new_a < 0.0 || new_b < 0.0 {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Insufficient balance".into()
                    )));
                }
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
    pub amount: f64,
    pub from_a: bool,
}

#[hdk_extern]
pub fn get_payment_history(did: String) -> ExternResult<Vec<Record>> {
    let mut payments = Vec::new();
    // Get sent payments
    let query = LinkQuery::try_new(anchor_hash(&did)?, LinkTypes::SenderToPayments)?;
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            payments.push(record);
        }
    }
    // Get received payments
    let query = LinkQuery::try_new(anchor_hash(&did)?, LinkTypes::ReceiverToPayments)?;
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            payments.push(record);
        }
    }
    Ok(payments)
}

/// Get a specific payment by ID
#[hdk_extern]
pub fn get_payment(payment_id: String) -> ExternResult<Option<Record>> {
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Payment,
        )?))
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

    if matches!(original_payment.status, TransferStatus::Refunded) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Payment already refunded".into()
        )));
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

/// Get all channels for a party
#[hdk_extern]
pub fn get_channels(did: String) -> ExternResult<Vec<Record>> {
    let mut channels = Vec::new();
    // Party A channels
    let query = LinkQuery::try_new(anchor_hash(&did)?, LinkTypes::ChannelPartyA)?;
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
            channels.push(record);
        }
    }
    // Party B channels
    let query = LinkQuery::try_new(anchor_hash(&did)?, LinkTypes::ChannelPartyB)?;
    for link in get_links(query, GetStrategy::default())? {
        if let Some(record) = get(
            ActionHash::try_from(link.target)
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid".into())))?,
            GetOptions::default(),
        )? {
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
    pub amount: f64,
    pub currency: String,
    pub escrow_id: String,
    pub memo: Option<String>,
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
