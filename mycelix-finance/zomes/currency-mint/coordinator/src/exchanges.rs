//! Exchange recording, confirmation, cancellation, and queries.

use currency_mint_integrity::*;
use hdk::prelude::*;
use mycelix_finance_shared::{anchor_hash, rate_limit_anchor_key};
use mycelix_finance_types::CurrencyStatus;

/// Per-agent rate limit for minted exchange recording (operations per minute).
/// Tighter than the global default because exchange recording is the most
/// spammable operation — it creates entries AND triggers balance changes.
const EXCHANGE_RATE_LIMIT_PER_MINUTE: usize = 60;

use crate::helpers::*;
use crate::{
    GetMemberExchangesInput, PaginatedCurrencyInput, PaginatedReceiverInput,
    RecordMintedExchangeInput,
};

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

    // Per-agent rate limit: reject if this agent has exceeded the exchange
    // recording limit within the current 60-second window.
    {
        let agent = agent_info()?.agent_initial_pubkey;
        let now_micros = sys_time()?.as_micros();
        let key = rate_limit_anchor_key("exchange", &agent, now_micros);
        let anchor = anchor_hash(&key)?;
        let recent_links = get_links(
            LinkQuery::try_new(anchor.clone(), LinkTypes::CurrencyToExchanges)?,
            GetStrategy::default(),
        )?;
        if recent_links.len() >= EXCHANGE_RATE_LIMIT_PER_MINUTE {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Rate limit exceeded: max {} exchange recordings per minute",
                EXCHANGE_RATE_LIMIT_PER_MINUTE
            ))));
        }
        // Record this operation in the rate-limit bucket
        create_link(
            anchor,
            AnyLinkableHash::from(agent.clone()),
            LinkTypes::CurrencyToExchanges,
            (),
        )?;
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

    // RC-19 fix: Create confirmation link FIRST, then verify we won the race.
    // This prevents duplicate balance updates from concurrent calls — both
    // callers create their link, but only the one whose link is "first" by
    // ActionHash ordering proceeds with balance updates.
    let confirm_anchor = format!("confirm:{}", exchange_id);
    let confirm_anchor_hash = anchor_hash(&confirm_anchor)?;

    // Quick pre-check: if confirmation already exists, return idempotently.
    let pre_existing = get_links(
        LinkQuery::try_new(
            confirm_anchor_hash.clone(),
            LinkTypes::ExchangeToConfirmation,
        )?,
        GetStrategy::default(),
    )?;
    if !pre_existing.is_empty() {
        return Ok(exchange);
    }

    // Create confirmation receipt and link BEFORE balance updates
    let now = sys_time()?;
    let confirmation = MintedExchangeConfirmation {
        exchange_id: exchange_id.clone(),
        confirmer_did: caller_did,
        timestamp: now,
    };
    let conf_hash = create_entry(&EntryTypes::MintedExchangeConfirmation(confirmation))?;
    let our_link_hash = create_link(
        confirm_anchor_hash.clone(),
        conf_hash,
        LinkTypes::ExchangeToConfirmation,
        (),
    )?;

    // Re-read links to detect concurrent confirmations (create-then-verify)
    let all_confirm_links = get_links(
        LinkQuery::try_new(confirm_anchor_hash, LinkTypes::ExchangeToConfirmation)?,
        GetStrategy::default(),
    )?;

    if all_confirm_links.len() > 1 {
        // Race detected — determine winner by lowest ActionHash (deterministic).
        let winner = pick_race_winner(&all_confirm_links)?;

        if winner.create_link_hash != our_link_hash {
            // We lost the race — clean up our link and return idempotently.
            delete_link(our_link_hash, GetOptions::default())?;
            return Ok(exchange);
        }
        // We won — clean up the loser's duplicate links for tidiness.
        for link in &all_confirm_links {
            if link.create_link_hash != our_link_hash {
                // Best-effort cleanup; ignore errors (other agent may own it)
                let _ = delete_link(link.create_link_hash.clone(), GetOptions::default());
            }
        }
    }

    // Only the race winner reaches here.

    // Create a PendingMintedAdjustment BEFORE balance updates for crash recovery.
    // If a crash occurs between the two updates, recover_incomplete_minted_confirmations
    // can complete the interrupted operation and restore the zero-sum invariant.
    let pending_adj = PendingMintedAdjustment {
        exchange_id: exchange.id.clone(),
        provider_did: exchange.provider_did.clone(),
        receiver_did: exchange.receiver_did.clone(),
        hours: exchange.hours as f64,
        currency_id: exchange.currency_id.clone(),
        provider_completed: false,
        receiver_completed: false,
        created_at: now,
    };
    let pending_adj_hash = create_entry(&EntryTypes::PendingMintedAdjustment(pending_adj))?;

    // Link from a well-known anchor so recover_incomplete_minted_confirmations can find them
    let pending_adj_anchor = anchor_hash("pending-minted-adjustments")?;
    create_link(
        pending_adj_anchor,
        pending_adj_hash.clone(),
        LinkTypes::PendingMintedAdjustmentToExchange,
        (),
    )?;

    // Update balances — provider first (zero-sum)
    update_minted_balance(
        &exchange.provider_did,
        &exchange.currency_id,
        exchange.hours,
        true,
    )?;

    // Mark provider side as completed
    let pending_adj_provider_done = PendingMintedAdjustment {
        exchange_id: exchange.id.clone(),
        provider_did: exchange.provider_did.clone(),
        receiver_did: exchange.receiver_did.clone(),
        hours: exchange.hours as f64,
        currency_id: exchange.currency_id.clone(),
        provider_completed: true,
        receiver_completed: false,
        created_at: now,
    };
    update_entry(
        pending_adj_hash.clone(),
        &EntryTypes::PendingMintedAdjustment(pending_adj_provider_done),
    )?;

    // Update balances — receiver second
    update_minted_balance(
        &exchange.receiver_did,
        &exchange.currency_id,
        exchange.hours,
        false,
    )?;

    // Mark both sides as completed
    let pending_adj_all_done = PendingMintedAdjustment {
        exchange_id: exchange.id.clone(),
        provider_did: exchange.provider_did.clone(),
        receiver_did: exchange.receiver_did.clone(),
        hours: exchange.hours as f64,
        currency_id: exchange.currency_id.clone(),
        provider_completed: true,
        receiver_completed: true,
        created_at: now,
    };
    update_entry(
        pending_adj_hash,
        &EntryTypes::PendingMintedAdjustment(pending_adj_all_done),
    )?;

    // Remove from receiver's pending index so it no longer appears in
    // list_pending_for_receiver after confirmation.
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
                match record.entry().to_app_option::<MintedExchange>() {
                    Ok(Some(ex)) if ex.id == exchange_id => {
                        delete_link(link.create_link_hash, GetOptions::default())?;
                    }
                    Err(e) => {
                        debug!("confirm_exchange: deserialization error: {:?}", e);
                    }
                    _ => {}
                }
            }
        }
    }

    // Return the exchange with confirmed status
    Ok(MintedExchange {
        confirmed: true,
        ..exchange
    })
}

/// List pending (unconfirmed) exchanges for a currency (paginated, default limit 100).
#[hdk_extern]
pub fn list_pending_exchanges(input: PaginatedCurrencyInput) -> ExternResult<Vec<MintedExchange>> {
    validate_id(&input.currency_id, "currency_id")?;
    let limit = input.limit.unwrap_or(100);
    let entries = collect_linked_entries::<MintedExchange>(
        &format!("mex:{}", input.currency_id),
        LinkTypes::CurrencyToExchanges,
    )?;
    let mut results: Vec<MintedExchange> = entries
        .into_iter()
        .map(|(ex, _)| ex)
        .filter(|ex| !ex.confirmed)
        .collect();
    results.truncate(limit);
    Ok(results)
}

/// List pending exchanges awaiting confirmation by a specific receiver (paginated, default limit 100).
#[hdk_extern]
pub fn list_pending_for_receiver(
    input: PaginatedReceiverInput,
) -> ExternResult<Vec<MintedExchange>> {
    let limit = input.limit.unwrap_or(100);
    let entries = collect_linked_entries::<MintedExchange>(
        &format!("receiver-pending:{}", input.receiver_did),
        LinkTypes::CurrencyToExchanges,
    )?;
    let mut results: Vec<MintedExchange> = entries
        .into_iter()
        .map(|(ex, _)| ex)
        .filter(|ex| !ex.confirmed && ex.receiver_did == input.receiver_did)
        .collect();
    results.truncate(limit);
    Ok(results)
}

/// Cancel an expired unconfirmed exchange.
///
/// Checks the currency's `confirmation_timeout_hours`. If the exchange is
/// unconfirmed and older than the timeout, it is cancelled (no balance changes).
/// Only the provider or receiver can cancel.
#[hdk_extern]
pub fn cancel_expired_exchange(exchange_id: String) -> ExternResult<bool> {
    validate_id(&exchange_id, "exchange_id")?;
    let (exchange, original_action_hash) = find_minted_exchange(&exchange_id)?;

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
                match record.entry().to_app_option::<MintedExchange>() {
                    Ok(Some(ex)) if ex.id == exchange_id => {
                        delete_link(link.create_link_hash, GetOptions::default())?;
                    }
                    Err(e) => {
                        debug!("resolve_dispute: deserialization error: {:?}", e);
                    }
                    _ => {}
                }
            }
        }
    }

    // Clean up global exchange index so cancelled exchanges don't appear
    // in list_pending_exchanges or currency exchange queries.
    let global_anchor = anchor_hash(&format!("mex:{}", exchange.currency_id))?;
    let global_links = get_links(
        LinkQuery::try_new(global_anchor, LinkTypes::CurrencyToExchanges)?,
        GetStrategy::default(),
    )?;
    for link in global_links {
        if let Some(hash) = link.target.into_action_hash() {
            if hash == original_action_hash {
                delete_link(link.create_link_hash, GetOptions::default())?;
                break;
            }
        }
    }

    Ok(true)
}

/// Get a single exchange by ID.
#[hdk_extern]
pub fn get_exchange(exchange_id: String) -> ExternResult<Option<MintedExchange>> {
    validate_id(&exchange_id, "exchange_id")?;
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
                    .is_none_or(|cursor| ex.timestamp.as_micros() > cursor.as_micros())
        })
        .collect();

    exchanges.sort_by(|a, b| b.timestamp.as_micros().cmp(&a.timestamp.as_micros()));
    exchanges.truncate(limit);
    Ok(exchanges)
}

// =============================================================================
// PENDING MINTED ADJUSTMENT RECOVERY
// =============================================================================

/// Recover incomplete balance adjustments from interrupted confirm_minted_exchange calls.
///
/// When confirm_minted_exchange crashes between the provider and receiver balance updates,
/// the zero-sum invariant is broken. This function finds all PendingMintedAdjustment
/// entries that are not fully completed and retries the missing balance updates.
///
/// Only callable by a governance agent (or during bootstrap when no agents exist).
///
/// Returns the number of adjustments that were recovered.
#[hdk_extern]
pub fn recover_incomplete_minted_confirmations(currency_id: String) -> ExternResult<u32> {
    if currency_id.is_empty() || currency_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Currency ID must be 1-256 characters".into()
        )));
    }

    // Governance gate: communities above threshold require authorization
    let (_, def) = get_currency_inner(&currency_id)?;
    let community_size = fetch_community_size(&def.creator_dao_did);
    if community_size > COMMUNITY_GOVERNANCE_THRESHOLD {
        match call(
            CallTargetCell::Local,
            ZomeName::from("tend"),
            FunctionName::from("verify_governance_agent"),
            None,
            (),
        ) {
            Ok(ZomeCallResponse::Ok(_)) => {} // Authorized
            _ => {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Recovery of pending minted adjustments requires governance authorization for communities >10 members".into()
                )));
            }
        }
    }

    let pending_anchor = anchor_hash("pending-minted-adjustments")?;
    let links = get_links(
        LinkQuery::try_new(pending_anchor, LinkTypes::PendingMintedAdjustmentToExchange)?,
        GetStrategy::default(),
    )?;

    let mut recovered: u32 = 0;

    for link in links {
        let Some(target_hash) = link.target.into_action_hash() else {
            continue;
        };

        // Follow the update chain to get the latest version of this entry
        let record = match follow_update_chain(target_hash) {
            Ok(r) => r,
            Err(_) => continue,
        };

        let Some(adj) = record
            .entry()
            .to_app_option::<PendingMintedAdjustment>()
            .map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "PendingMintedAdjustment deserialization error: {:?}",
                    e
                )))
            })?
        else {
            continue;
        };

        // Skip entries not matching the requested currency
        if adj.currency_id != currency_id {
            continue;
        }

        // Skip fully completed adjustments
        if adj.provider_completed && adj.receiver_completed {
            continue;
        }

        let hours = adj.hours as f32;
        let mut current_action = record.action_address().clone();

        // Retry missing balance updates
        if !adj.provider_completed {
            // Neither side completed — retry provider first
            update_minted_balance(
                &adj.provider_did,
                &adj.currency_id,
                hours,
                true, // provider gains
            )?;

            // Mark provider done
            let updated = PendingMintedAdjustment {
                provider_completed: true,
                ..adj.clone()
            };
            current_action = update_entry(
                current_action,
                &EntryTypes::PendingMintedAdjustment(updated),
            )?;
        }

        if !adj.receiver_completed {
            update_minted_balance(
                &adj.receiver_did,
                &adj.currency_id,
                hours,
                false, // receiver pays
            )?;

            // Mark both sides done
            let updated = PendingMintedAdjustment {
                provider_completed: true,
                receiver_completed: true,
                ..adj.clone()
            };
            update_entry(
                current_action,
                &EntryTypes::PendingMintedAdjustment(updated),
            )?;
        }

        recovered += 1;
    }

    Ok(recovered)
}
