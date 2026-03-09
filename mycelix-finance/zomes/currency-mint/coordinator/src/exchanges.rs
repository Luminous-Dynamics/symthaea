//! Exchange recording, confirmation, cancellation, and queries.

use currency_mint_integrity::*;
use hdk::prelude::*;
use mycelix_finance_shared::anchor_hash;
use mycelix_finance_types::CurrencyStatus;

use crate::helpers::*;
use crate::{GetMemberExchangesInput, PaginatedCurrencyInput, RecordMintedExchangeInput};

/// Record an exchange in a community-minted currency.
///
/// Enforces the currency's specific credit limit, max service hours, and
/// min service minutes. The zero-sum invariant is maintained: provider gains
/// exactly what receiver loses.
#[hdk_extern]
pub fn record_minted_exchange(input: RecordMintedExchangeInput) -> ExternResult<MintedExchange> {
    // Load currency definition
    let (_, def) = get_currency_inner(&input.currency_id)?;

    // Currency must be Active
    if def.status != CurrencyStatus::Active {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Currency {} is {:?}, exchanges only allowed when Active",
            def.params.name, def.status
        ))));
    }

    // Guard against NaN/Inf — must come before any arithmetic
    if !input.hours.is_finite() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Hours must be a finite number".into()
        )));
    }

    // Validate service duration against currency's parameters
    let minutes = (input.hours * 60.0) as u32;
    if minutes < def.params.min_service_minutes {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Minimum service duration for {} is {} minutes",
            def.params.name, def.params.min_service_minutes
        ))));
    }
    if input.hours > def.params.max_service_hours as f32 {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Maximum service duration for {} is {} hours",
            def.params.name, def.params.max_service_hours
        ))));
    }
    if input.hours <= 0.0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Hours must be positive".into()
        )));
    }
    if input.service_description.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Service description required".into()
        )));
    }

    let caller = agent_info()?.agent_initial_pubkey;
    let provider_did = format!("did:mycelix:{}", caller);
    let now = sys_time()?;

    if provider_did == input.receiver_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot exchange with yourself".into()
        )));
    }

    // Check balance limits against THIS currency's credit_limit
    let provider_bal =
        get_or_create_minted_balance(provider_did.clone(), input.currency_id.clone())?;
    let new_provider = provider_bal.balance + (input.hours.round() as i32);
    if new_provider > def.params.credit_limit {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Would exceed {} credit limit of +{}. Current: {}",
            def.params.name, def.params.credit_limit, provider_bal.balance
        ))));
    }

    let receiver_bal =
        get_or_create_minted_balance(input.receiver_did.clone(), input.currency_id.clone())?;
    let new_receiver = receiver_bal.balance - (input.hours.round() as i32);
    if new_receiver < -def.params.credit_limit {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Receiver would exceed {} debt limit of -{}. Current: {}",
            def.params.name, def.params.credit_limit, receiver_bal.balance
        ))));
    }

    // Rate limit: check daily exchange count for this provider
    if def.params.max_exchanges_per_day > 0 {
        let day_ago =
            Timestamp::from_micros(now.as_micros().saturating_sub(24 * 3_600 * 1_000_000));
        let all_exchanges = collect_linked_entries::<MintedExchange>(
            &format!("mex:{}", input.currency_id),
            LinkTypes::CurrencyToExchanges,
        )?;

        let today_count: u8 = all_exchanges
            .iter()
            .filter(|(ex, _)| {
                ex.provider_did == provider_did && ex.timestamp.as_micros() >= day_ago.as_micros()
            })
            .count()
            .min(u8::MAX as usize) as u8;

        if today_count >= def.params.max_exchanges_per_day {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Daily exchange limit reached ({}/{})",
                today_count, def.params.max_exchanges_per_day
            ))));
        }
    }

    // Create exchange entry
    let exchange_id = format!(
        "mex:{}:{}:{}",
        input.currency_id,
        provider_did,
        now.as_micros()
    );

    let confirmed = !def.params.requires_confirmation;

    let exchange = MintedExchange {
        id: exchange_id.clone(),
        currency_id: input.currency_id.clone(),
        provider_did: provider_did.clone(),
        receiver_did: input.receiver_did.clone(),
        hours: input.hours,
        service_description: input.service_description,
        timestamp: now,
        confirmed,
    };

    let hash = create_entry(&EntryTypes::MintedExchange(exchange.clone()))?;
    create_link(
        anchor_hash(&format!("mex:{}", input.currency_id))?,
        hash.clone(),
        LinkTypes::CurrencyToExchanges,
        (),
    )?;
    // Index by exchange ID for lookup by confirm_minted_exchange
    create_link(
        anchor_hash(&format!("mex-id:{}", exchange_id))?,
        hash.clone(),
        LinkTypes::CurrencyToExchanges,
        (),
    )?;

    // Update balances only if confirmed (instant confirmation currencies)
    if confirmed {
        update_minted_balance(&provider_did, &input.currency_id, input.hours, true)?;
        update_minted_balance(&input.receiver_did, &input.currency_id, input.hours, false)?;
    } else {
        // Index by receiver DID for pending exchange queries
        create_link(
            anchor_hash(&format!("receiver-pending:{}", input.receiver_did))?,
            hash,
            LinkTypes::CurrencyToExchanges,
            (),
        )?;
    }

    Ok(exchange)
}

/// Confirm a pending exchange as the receiver.
///
/// Only the receiver can confirm. Creates a confirmation receipt and
/// triggers the balance updates (zero-sum: provider +hours, receiver -hours).
/// No-ops if the exchange is already confirmed.
#[hdk_extern]
pub fn confirm_minted_exchange(exchange_id: String) -> ExternResult<MintedExchange> {
    // Find the exchange
    let (exchange, _) = find_minted_exchange(&exchange_id)?;

    if exchange.confirmed {
        return Ok(exchange);
    }

    // Currency must still be Active to confirm
    let (_, def) = get_currency_inner(&exchange.currency_id)?;
    if def.status != CurrencyStatus::Active {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot confirm exchange — currency is {:?}",
            def.status
        ))));
    }

    // Only the receiver can confirm
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);
    if caller_did != exchange.receiver_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the receiver can confirm an exchange".into()
        )));
    }

    // Check no existing confirmation
    let confirm_anchor = format!("confirm:{}", exchange_id);
    let existing = get_links(
        LinkQuery::try_new(
            anchor_hash(&confirm_anchor)?,
            LinkTypes::ExchangeToConfirmation,
        )?,
        GetStrategy::default(),
    )?;
    if !existing.is_empty() {
        return Ok(exchange);
    }

    // Create confirmation receipt
    let now = sys_time()?;
    let confirmation = MintedExchangeConfirmation {
        exchange_id: exchange_id.clone(),
        confirmer_did: caller_did,
        timestamp: now,
    };
    let conf_hash = create_entry(&EntryTypes::MintedExchangeConfirmation(confirmation))?;
    create_link(
        anchor_hash(&confirm_anchor)?,
        conf_hash,
        LinkTypes::ExchangeToConfirmation,
        (),
    )?;

    // Now update balances (zero-sum)
    update_minted_balance(
        &exchange.provider_did,
        &exchange.currency_id,
        exchange.hours,
        true,
    )?;
    update_minted_balance(
        &exchange.receiver_did,
        &exchange.currency_id,
        exchange.hours,
        false,
    )?;

    // Return the exchange with confirmed status
    Ok(MintedExchange {
        confirmed: true,
        ..exchange
    })
}

/// List pending (unconfirmed) exchanges for a currency.
#[hdk_extern]
pub fn list_pending_exchanges(currency_id: String) -> ExternResult<Vec<MintedExchange>> {
    let entries = collect_linked_entries::<MintedExchange>(
        &format!("mex:{}", currency_id),
        LinkTypes::CurrencyToExchanges,
    )?;
    Ok(entries
        .into_iter()
        .map(|(ex, _)| ex)
        .filter(|ex| !ex.confirmed)
        .collect())
}

/// List pending exchanges awaiting confirmation by a specific receiver.
#[hdk_extern]
pub fn list_pending_for_receiver(receiver_did: String) -> ExternResult<Vec<MintedExchange>> {
    let entries = collect_linked_entries::<MintedExchange>(
        &format!("receiver-pending:{}", receiver_did),
        LinkTypes::CurrencyToExchanges,
    )?;
    Ok(entries
        .into_iter()
        .map(|(ex, _)| ex)
        .filter(|ex| !ex.confirmed && ex.receiver_did == receiver_did)
        .collect())
}

/// Cancel an expired unconfirmed exchange.
///
/// Checks the currency's `confirmation_timeout_hours`. If the exchange is
/// unconfirmed and older than the timeout, it is cancelled (no balance changes).
/// Only the provider or receiver can cancel.
#[hdk_extern]
pub fn cancel_expired_exchange(exchange_id: String) -> ExternResult<bool> {
    let (exchange, _) = find_minted_exchange(&exchange_id)?;

    if exchange.confirmed {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Exchange is already confirmed — cannot cancel".into()
        )));
    }

    // Verify caller is a participant
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);
    if caller_did != exchange.provider_did && caller_did != exchange.receiver_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only exchange participants can cancel".into()
        )));
    }

    // Check timeout
    let (_, def) = get_currency_inner(&exchange.currency_id)?;
    let timeout_hours = def.params.confirmation_timeout_hours;

    if timeout_hours == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "This currency has no confirmation timeout — exchanges never expire".into()
        )));
    }

    let now = sys_time()?;
    let elapsed_hours = (now
        .as_micros()
        .saturating_sub(exchange.timestamp.as_micros()))
        / 1_000_000
        / 3_600;

    if (elapsed_hours as u32) < timeout_hours {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Exchange has not expired yet ({} of {} hours elapsed)",
            elapsed_hours, timeout_hours
        ))));
    }

    // Exchange is expired — no balance changes needed since balances were
    // never updated for unconfirmed exchanges.
    // Remove from receiver's pending index so it stops appearing in
    // list_pending_for_receiver and can't be confirmed after cancellation.
    let pending_anchor = format!("receiver-pending:{}", exchange.receiver_did);
    let pending_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&pending_anchor)?,
            LinkTypes::CurrencyToExchanges,
        )?,
        GetStrategy::default(),
    )?;
    for link in pending_links {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Ok(record) = follow_update_chain(action_hash) {
                if let Some(ex) = record
                    .entry()
                    .to_app_option::<MintedExchange>()
                    .ok()
                    .flatten()
                {
                    if ex.id == exchange_id {
                        delete_link(link.create_link_hash, GetOptions::default())?;
                    }
                }
            }
        }
    }

    Ok(true)
}

/// Get a single exchange by ID.
#[hdk_extern]
pub fn get_exchange(exchange_id: String) -> ExternResult<Option<MintedExchange>> {
    match find_minted_exchange(&exchange_id) {
        Ok((ex, _)) => Ok(Some(ex)),
        Err(_) => Ok(None),
    }
}

/// Get recent exchanges for a community-minted currency.
///
/// Supports cursor-based pagination via `after_timestamp` — only returns
/// exchanges with timestamp > the cursor. Results sorted newest-first.
#[hdk_extern]
pub fn get_currency_exchanges(input: PaginatedCurrencyInput) -> ExternResult<Vec<MintedExchange>> {
    let limit = input.limit.unwrap_or(100);
    let entries = collect_linked_entries::<MintedExchange>(
        &format!("mex:{}", input.currency_id),
        LinkTypes::CurrencyToExchanges,
    )?;

    let mut exchanges: Vec<MintedExchange> = entries
        .into_iter()
        .map(|(ex, _)| ex)
        .filter(|ex| {
            if let Some(cursor) = &input.after_timestamp {
                ex.timestamp.as_micros() > cursor.as_micros()
            } else {
                true
            }
        })
        .collect();

    // Sort newest-first for consistent pagination
    exchanges.sort_by(|a, b| b.timestamp.as_micros().cmp(&a.timestamp.as_micros()));
    exchanges.truncate(limit);
    Ok(exchanges)
}

/// Get all exchanges for a specific member in a currency (as provider or receiver).
///
/// Supports cursor-based pagination via `after_timestamp`. Results sorted newest-first.
#[hdk_extern]
pub fn get_member_exchanges(input: GetMemberExchangesInput) -> ExternResult<Vec<MintedExchange>> {
    let limit = input.limit.unwrap_or(100);
    let entries = collect_linked_entries::<MintedExchange>(
        &format!("mex:{}", input.currency_id),
        LinkTypes::CurrencyToExchanges,
    )?;

    let mut exchanges: Vec<MintedExchange> = entries
        .into_iter()
        .map(|(ex, _)| ex)
        .filter(|ex| {
            (ex.provider_did == input.member_did || ex.receiver_did == input.member_did)
                && input
                    .after_timestamp
                    .as_ref()
                    .map_or(true, |cursor| ex.timestamp.as_micros() > cursor.as_micros())
        })
        .collect();

    exchanges.sort_by(|a, b| b.timestamp.as_micros().cmp(&a.timestamp.as_micros()));
    exchanges.truncate(limit);
    Ok(exchanges)
}
