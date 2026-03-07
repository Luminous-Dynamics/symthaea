//! Currency Factory Coordinator Zome
//!
//! Enables communities to create their own mutual credit currencies with
//! **Parameter Sovereignty** (custom names, limits, demurrage) while the
//! **Immutable Economic Physics** (zero-sum, balance limits, no pre-mining)
//! are enforced at the integrity level.
//!
//! Lifecycle: Draft → Active → Suspended → Active → Retired (terminal)
//!
//! Communities with >10 members require a governance proposal to create a currency.
//! The proposal ID is recorded in the CurrencyDefinition for audit.

use currency_mint_integrity::*;
use hdk::prelude::*;
use mycelix_finance_shared::anchor_hash;
use mycelix_finance_types::{compute_minted_demurrage, CurrencyStatus, MintedCurrencyParams};

// =============================================================================
// INPUT/OUTPUT TYPES
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateCurrencyInput {
    /// DID of the creating DAO/community
    pub dao_did: String,
    /// Community-chosen parameters
    pub params: MintedCurrencyParams,
    /// Governance proposal that authorized this (required for communities >10 members)
    pub governance_proposal_id: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ActivateCurrencyInput {
    pub currency_id: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordMintedExchangeInput {
    pub currency_id: String,
    pub receiver_did: String,
    pub hours: f32,
    pub service_description: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetMintedBalanceInput {
    pub currency_id: String,
    pub member_did: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MintedBalanceInfo {
    pub member_did: String,
    pub currency_id: String,
    pub currency_name: String,
    pub currency_symbol: String,
    pub balance: i32,
    pub credit_limit: i32,
    pub can_provide: bool,
    pub can_receive: bool,
    pub total_provided: f32,
    pub total_received: f32,
    pub exchange_count: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetMemberExchangesInput {
    pub currency_id: String,
    pub member_did: String,
    pub limit: Option<usize>,
    /// Only return exchanges with timestamp > this value (cursor for forward pagination)
    pub after_timestamp: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DemurrageReportEntry {
    pub member_did: String,
    pub current_balance: i32,
    pub pending_deduction: i32,
    pub effective_balance: i32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DemurrageReport {
    pub currency_id: String,
    pub currency_name: String,
    pub demurrage_rate: f64,
    pub total_pending_demurrage: i64,
    pub affected_members: u32,
    pub entries: Vec<DemurrageReportEntry>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OpenDisputeInput {
    pub exchange_id: String,
    pub reason: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResolveDisputeInput {
    pub exchange_id: String,
    /// true = accept dispute (reverse balances), false = reject (keep balances)
    pub accept: bool,
    pub resolution_reason: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AmendCurrencyParamsInput {
    pub currency_id: String,
    pub new_params: MintedCurrencyParams,
    pub governance_proposal_id: Option<String>,
}

// =============================================================================
// CURRENCY LIFECYCLE
// =============================================================================

/// Create a new community currency (starts in Draft status).
///
/// Parameters are validated against constitutional limits by the integrity zome.
/// Communities with >10 members MUST provide a governance_proposal_id.
/// Small communities (hearth-scale, ≤10) can create currencies without governance.
#[hdk_extern]
pub fn create_currency(input: CreateCurrencyInput) -> ExternResult<CurrencyDefinition> {
    if !input.dao_did.starts_with("did:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DAO must be a valid DID".into()
        )));
    }

    // Validate params (also validated by integrity, but fail fast here)
    if let Err(e) = input.params.validate() {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Invalid currency parameters: {}",
            e
        ))));
    }

    // Governance gate: communities with >10 members require a governance proposal
    let community_size = fetch_community_size(&input.dao_did);
    if community_size > 10 {
        if input.governance_proposal_id.is_none() {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Communities with >10 members ({} members) require a governance proposal to create a currency",
                community_size
            ))));
        }
        // Verify governance authorization
        match call(
            CallTargetCell::Local,
            ZomeName::from("tend"),
            FunctionName::from("verify_governance_agent"),
            None,
            (),
        ) {
            Ok(ZomeCallResponse::Ok(_)) => {} // Authorized
            Ok(other) => {
                return Err(wasm_error!(WasmErrorInner::Guest(format!(
                    "Governance authorization failed for currency creation: {:?}",
                    other
                ))));
            }
            Err(e) => {
                return Err(wasm_error!(WasmErrorInner::Guest(format!(
                    "Currency creation requires governance authorization for communities >10 members: {:?}", e
                ))));
            }
        }
    }

    let now = sys_time()?;
    let currency_id = format!(
        "currency:{}:{}:{}",
        input.dao_did,
        input.params.symbol,
        now.as_micros()
    );

    let def = CurrencyDefinition {
        id: currency_id.clone(),
        creator_dao_did: input.dao_did.clone(),
        governance_proposal_id: input.governance_proposal_id,
        params: input.params,
        status: CurrencyStatus::Draft,
        created_at: now,
    };

    let hash = create_entry(&EntryTypes::CurrencyDefinition(def.clone()))?;

    // Index: DAO → currencies
    create_link(
        anchor_hash(&format!("dao-currencies:{}", input.dao_did))?,
        hash.clone(),
        LinkTypes::DaoToCurrencies,
        (),
    )?;

    // Index: currency ID → definition (for O(1) lookup)
    create_link(
        anchor_hash(&format!("currency:{}", currency_id))?,
        hash,
        LinkTypes::CurrencyIdToDefinition,
        (),
    )?;

    Ok(def)
}

/// Activate a Draft currency (transitions Draft → Active).
///
/// Once active, members can record exchanges. Requires governance authorization
/// for communities with >10 members.
#[hdk_extern]
pub fn activate_currency(input: ActivateCurrencyInput) -> ExternResult<CurrencyDefinition> {
    let (record, def) = get_currency_inner(&input.currency_id)?;

    if def.status != CurrencyStatus::Draft {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Currency is {:?}, only Draft currencies can be activated",
            def.status
        ))));
    }

    let updated = CurrencyDefinition {
        status: CurrencyStatus::Active,
        ..def
    };

    update_entry(
        record.action_address().clone(),
        &EntryTypes::CurrencyDefinition(updated.clone()),
    )?;

    // Add to global active currencies index for cross-community discovery
    create_link(
        anchor_hash("all-active-currencies")?,
        record.action_address().clone(),
        LinkTypes::DaoToCurrencies,
        (),
    )?;

    Ok(updated)
}

/// Suspend an Active currency (transitions Active → Suspended).
#[hdk_extern]
pub fn suspend_currency(currency_id: String) -> ExternResult<CurrencyDefinition> {
    let (record, def) = get_currency_inner(&currency_id)?;

    if def.status != CurrencyStatus::Active {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only Active currencies can be suspended".into()
        )));
    }

    let updated = CurrencyDefinition {
        status: CurrencyStatus::Suspended,
        ..def
    };

    update_entry(
        record.action_address().clone(),
        &EntryTypes::CurrencyDefinition(updated.clone()),
    )?;
    Ok(updated)
}

/// Reactivate a Suspended currency (transitions Suspended → Active).
#[hdk_extern]
pub fn reactivate_currency(currency_id: String) -> ExternResult<CurrencyDefinition> {
    let (record, def) = get_currency_inner(&currency_id)?;

    if def.status != CurrencyStatus::Suspended {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only Suspended currencies can be reactivated".into()
        )));
    }

    let updated = CurrencyDefinition {
        status: CurrencyStatus::Active,
        ..def
    };

    update_entry(
        record.action_address().clone(),
        &EntryTypes::CurrencyDefinition(updated.clone()),
    )?;

    // Re-add to global active currencies index
    create_link(
        anchor_hash("all-active-currencies")?,
        record.action_address().clone(),
        LinkTypes::DaoToCurrencies,
        (),
    )?;

    Ok(updated)
}

/// Retire a currency permanently (terminal state — no new exchanges, balances frozen).
///
/// Only Active or Suspended currencies can be retired. Draft currencies must be
/// activated first (prevents bypassing governance gate via Draft → Retired).
#[hdk_extern]
pub fn retire_currency(currency_id: String) -> ExternResult<CurrencyDefinition> {
    let (record, def) = get_currency_inner(&currency_id)?;

    match def.status {
        CurrencyStatus::Active | CurrencyStatus::Suspended => {} // allowed
        CurrencyStatus::Retired => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Currency is already retired".into()
            )));
        }
        CurrencyStatus::Draft => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Draft currencies cannot be retired — activate first".into()
            )));
        }
    }

    let updated = CurrencyDefinition {
        status: CurrencyStatus::Retired,
        ..def
    };

    update_entry(
        record.action_address().clone(),
        &EntryTypes::CurrencyDefinition(updated.clone()),
    )?;
    Ok(updated)
}

/// Get a currency definition by ID.
#[hdk_extern]
pub fn get_currency(currency_id: String) -> ExternResult<Option<CurrencyDefinition>> {
    match get_currency_inner(&currency_id) {
        Ok((_, def)) => Ok(Some(def)),
        Err(_) => Ok(None),
    }
}

/// Get all currencies created by a DAO.
#[hdk_extern]
pub fn get_dao_currencies(dao_did: String) -> ExternResult<Vec<CurrencyDefinition>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("dao-currencies:{}", dao_did))?,
            LinkTypes::DaoToCurrencies,
        )?,
        GetStrategy::default(),
    )?;

    let mut currencies = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(def) = record
                    .entry()
                    .to_app_option::<CurrencyDefinition>()
                    .ok()
                    .flatten()
                {
                    currencies.push(def);
                }
            }
        }
    }
    Ok(currencies)
}

/// List all Active currencies across all DAOs (global discovery).
#[hdk_extern]
pub fn list_active_currencies(_: ()) -> ExternResult<Vec<CurrencyDefinition>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash("all-active-currencies")?,
            LinkTypes::DaoToCurrencies,
        )?,
        GetStrategy::default(),
    )?;

    let mut currencies = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(def) = record
                    .entry()
                    .to_app_option::<CurrencyDefinition>()
                    .ok()
                    .flatten()
                {
                    // Only include actually Active currencies (link may be stale)
                    if def.status == CurrencyStatus::Active {
                        currencies.push(def);
                    }
                }
            }
        }
    }
    Ok(currencies)
}

// =============================================================================
// EXCHANGE FUNCTIONS
// =============================================================================

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
        let exchange_links = get_links(
            LinkQuery::try_new(
                anchor_hash(&format!("mex:{}", input.currency_id))?,
                LinkTypes::CurrencyToExchanges,
            )?,
            GetStrategy::default(),
        )?;

        let mut today_count: u8 = 0;
        for link in exchange_links {
            if let Some(action_hash) = link.target.into_action_hash() {
                if let Some(record) = get(action_hash, GetOptions::default())? {
                    if let Some(ex) = record
                        .entry()
                        .to_app_option::<MintedExchange>()
                        .ok()
                        .flatten()
                    {
                        if ex.provider_did == provider_did
                            && ex.timestamp.as_micros() >= day_ago.as_micros()
                        {
                            today_count = today_count.saturating_add(1);
                        }
                    }
                }
            }
        }

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
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("mex:{}", currency_id))?,
            LinkTypes::CurrencyToExchanges,
        )?,
        GetStrategy::default(),
    )?;

    let mut pending = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(ex) = record
                    .entry()
                    .to_app_option::<MintedExchange>()
                    .ok()
                    .flatten()
                {
                    if !ex.confirmed {
                        pending.push(ex);
                    }
                }
            }
        }
    }
    Ok(pending)
}

/// List pending exchanges awaiting confirmation by a specific receiver.
#[hdk_extern]
pub fn list_pending_for_receiver(receiver_did: String) -> ExternResult<Vec<MintedExchange>> {
    // Get all currencies this receiver participates in via balance links
    // For now, scan all currencies the receiver has exchanges in
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("receiver-pending:{}", receiver_did))?,
            LinkTypes::CurrencyToExchanges,
        )?,
        GetStrategy::default(),
    )?;

    let mut pending = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(ex) = record
                    .entry()
                    .to_app_option::<MintedExchange>()
                    .ok()
                    .flatten()
                {
                    if !ex.confirmed && ex.receiver_did == receiver_did {
                        pending.push(ex);
                    }
                }
            }
        }
    }
    Ok(pending)
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
    // never updated for unconfirmed exchanges. Just return success.
    // The exchange entry remains on the DHT as an immutable audit record.
    Ok(true)
}

/// Find a minted exchange by ID.
fn find_minted_exchange(exchange_id: &str) -> ExternResult<(MintedExchange, ActionHash)> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("mex-id:{}", exchange_id))?,
            LinkTypes::CurrencyToExchanges,
        )?,
        GetStrategy::default(),
    )?;

    // Fall back to scanning currency exchanges
    if let Some(link) = links.first() {
        if let Some(link_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(link_hash, GetOptions::default())? {
                if let Some(ex) = record
                    .entry()
                    .to_app_option::<MintedExchange>()
                    .ok()
                    .flatten()
                {
                    return Ok((ex, record.action_address().clone()));
                }
            }
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest(format!(
        "Exchange {} not found",
        exchange_id
    ))))
}

/// Get a member's balance in a community-minted currency.
///
/// Transparently applies demurrage if the currency has a non-zero demurrage rate.
/// This ensures balances always reflect the current decay state without requiring
/// a separate cron job or explicit demurrage call.
#[hdk_extern]
pub fn get_minted_balance(input: GetMintedBalanceInput) -> ExternResult<MintedBalanceInfo> {
    let (_, def) = get_currency_inner(&input.currency_id)?;

    // Lazy demurrage: apply on read if currency has demurrage
    let effective_balance =
        if def.params.demurrage_rate > 0.0 && def.status == CurrencyStatus::Active {
            let result = apply_minted_demurrage(ApplyDemurrageInput {
                currency_id: input.currency_id.clone(),
                member_did: input.member_did.clone(),
            })?;
            result.new_balance
        } else {
            let bal =
                get_or_create_minted_balance(input.member_did.clone(), input.currency_id.clone())?;
            bal.balance
        };

    let bal = get_or_create_minted_balance(input.member_did, input.currency_id)?;

    Ok(MintedBalanceInfo {
        member_did: bal.member_did,
        currency_id: bal.currency_id,
        currency_name: def.params.name,
        currency_symbol: def.params.symbol,
        balance: effective_balance,
        credit_limit: def.params.credit_limit,
        can_provide: effective_balance < def.params.credit_limit,
        can_receive: effective_balance > -def.params.credit_limit,
        total_provided: bal.total_provided,
        total_received: bal.total_received,
        exchange_count: bal.exchange_count,
    })
}

/// Get recent exchanges for a community-minted currency.
///
/// Supports cursor-based pagination via `after_timestamp` — only returns
/// exchanges with timestamp > the cursor. Results sorted newest-first.
#[hdk_extern]
pub fn get_currency_exchanges(input: PaginatedCurrencyInput) -> ExternResult<Vec<MintedExchange>> {
    let limit = input.limit.unwrap_or(100);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("mex:{}", input.currency_id))?,
            LinkTypes::CurrencyToExchanges,
        )?,
        GetStrategy::default(),
    )?;

    let mut exchanges = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(ex) = record
                    .entry()
                    .to_app_option::<MintedExchange>()
                    .ok()
                    .flatten()
                {
                    // Cursor filter: skip exchanges at or before the cursor
                    if let Some(cursor) = &input.after_timestamp {
                        if ex.timestamp.as_micros() <= cursor.as_micros() {
                            continue;
                        }
                    }
                    exchanges.push(ex);
                }
            }
        }
    }

    // Sort newest-first for consistent pagination
    exchanges.sort_by(|a, b| b.timestamp.as_micros().cmp(&a.timestamp.as_micros()));
    exchanges.truncate(limit);
    Ok(exchanges)
}

/// Get aggregate stats for a community-minted currency.
///
/// Returns member count, total positive balances (credit), total negative balances (debt),
/// exchange count, and net sum (should always be 0 for zero-sum currencies).
#[hdk_extern]
pub fn get_currency_stats(currency_id: String) -> ExternResult<CurrencyStats> {
    let (_, def) = get_currency_inner(&currency_id)?;

    // Scan all balance links for this currency
    // Convention: balance anchors are "mbal:{currency_id}:{member_did}"
    // We can't enumerate all members, so scan exchanges to collect unique DIDs
    let exchange_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("mex:{}", currency_id))?,
            LinkTypes::CurrencyToExchanges,
        )?,
        GetStrategy::default(),
    )?;

    let mut member_dids = std::collections::HashSet::new();
    let mut total_exchanges = 0u64;
    let mut confirmed_count = 0u64;
    let mut pending_count = 0u64;

    for link in exchange_links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(ex) = record
                    .entry()
                    .to_app_option::<MintedExchange>()
                    .ok()
                    .flatten()
                {
                    total_exchanges += 1;
                    if ex.confirmed {
                        confirmed_count += 1;
                    } else {
                        pending_count += 1;
                    }
                    member_dids.insert(ex.provider_did);
                    member_dids.insert(ex.receiver_did);
                }
            }
        }
    }

    // Sum balances across all known members
    let mut total_credit: i64 = 0;
    let mut total_debt: i64 = 0;

    for did in &member_dids {
        let bal = get_or_create_minted_balance(did.clone(), currency_id.clone())?;
        if bal.balance > 0 {
            total_credit += bal.balance as i64;
        } else if bal.balance < 0 {
            total_debt += bal.balance as i64;
        }
    }

    Ok(CurrencyStats {
        currency_id,
        currency_name: def.params.name,
        currency_symbol: def.params.symbol,
        status: def.status,
        member_count: member_dids.len() as u32,
        total_credit,
        total_debt,
        net_sum: total_credit + total_debt, // should be 0 for zero-sum
        total_exchanges,
        confirmed_exchanges: confirmed_count,
        pending_exchanges: pending_count,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CurrencyStats {
    pub currency_id: String,
    pub currency_name: String,
    pub currency_symbol: String,
    pub status: CurrencyStatus,
    pub member_count: u32,
    pub total_credit: i64,
    pub total_debt: i64,
    pub net_sum: i64,
    pub total_exchanges: u64,
    pub confirmed_exchanges: u64,
    pub pending_exchanges: u64,
}

/// Get all exchanges for a specific member in a currency (as provider or receiver).
#[hdk_extern]
pub fn get_member_exchanges(input: GetMemberExchangesInput) -> ExternResult<Vec<MintedExchange>> {
    let limit = input.limit.unwrap_or(100);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("mex:{}", input.currency_id))?,
            LinkTypes::CurrencyToExchanges,
        )?,
        GetStrategy::default(),
    )?;

    let mut exchanges = Vec::new();
    for link in links {
        if exchanges.len() >= limit {
            break;
        }
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(ex) = record
                    .entry()
                    .to_app_option::<MintedExchange>()
                    .ok()
                    .flatten()
                {
                    if ex.provider_did == input.member_did || ex.receiver_did == input.member_did {
                        exchanges.push(ex);
                    }
                }
            }
        }
    }
    Ok(exchanges)
}

/// Get a demurrage report for a currency without mutating any state.
///
/// Scans all member balances, computes pending demurrage for each, and
/// returns a summary. Useful for governance review before redistribution.
#[hdk_extern]
pub fn get_demurrage_report(currency_id: String) -> ExternResult<DemurrageReport> {
    let (_, def) = get_currency_inner(&currency_id)?;

    if def.params.demurrage_rate <= 0.0 {
        return Ok(DemurrageReport {
            currency_id,
            currency_name: def.params.name,
            demurrage_rate: def.params.demurrage_rate,
            total_pending_demurrage: 0,
            affected_members: 0,
            entries: Vec::new(),
        });
    }

    // Collect all member DIDs from exchange links
    let exchange_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("mex:{}", currency_id))?,
            LinkTypes::CurrencyToExchanges,
        )?,
        GetStrategy::default(),
    )?;

    let mut member_dids = std::collections::HashSet::new();
    for link in exchange_links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(ex) = record
                    .entry()
                    .to_app_option::<MintedExchange>()
                    .ok()
                    .flatten()
                {
                    member_dids.insert(ex.provider_did);
                    member_dids.insert(ex.receiver_did);
                }
            }
        }
    }

    let now = sys_time()?;
    let mut entries = Vec::new();
    let mut total_pending: i64 = 0;
    let mut affected: u32 = 0;

    for did in member_dids {
        let bal = get_or_create_minted_balance(did.clone(), currency_id.clone())?;
        if bal.balance <= 0 {
            continue;
        }

        let elapsed = now
            .as_micros()
            .saturating_sub(bal.last_activity.as_micros()) as u64
            / 1_000_000;
        let deduction = compute_minted_demurrage(bal.balance, def.params.demurrage_rate, elapsed);

        if deduction > 0 {
            affected += 1;
            total_pending += deduction as i64;
            entries.push(DemurrageReportEntry {
                member_did: did,
                current_balance: bal.balance,
                pending_deduction: deduction,
                effective_balance: bal.balance - deduction,
            });
        }
    }

    Ok(DemurrageReport {
        currency_id,
        currency_name: def.params.name,
        demurrage_rate: def.params.demurrage_rate,
        total_pending_demurrage: total_pending,
        affected_members: affected,
        entries,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CompostBalance {
    pub currency_id: String,
    pub accumulated: i32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RedistributeCompostResult {
    pub currency_id: String,
    pub total_redistributed: i32,
    pub recipients: u32,
    pub per_member_amount: i32,
    pub remainder_kept: i32,
}

/// Get the accumulated compost (demurrage) balance for a currency.
#[hdk_extern]
pub fn get_compost_balance(currency_id: String) -> ExternResult<CompostBalance> {
    let compost_did = format!("did:mycelix:__compost__:{}", currency_id);
    let bal = get_or_create_minted_balance(compost_did, currency_id.clone())?;
    Ok(CompostBalance {
        currency_id,
        accumulated: bal.balance,
    })
}

/// Redistribute accumulated compost evenly to all active members.
///
/// Splits the compost balance equally among all members with a balance entry
/// (excluding the compost pseudo-member). Remainder stays in compost.
/// Zero-sum is preserved: compost decreases by exactly what members gain.
#[hdk_extern]
pub fn redistribute_compost(currency_id: String) -> ExternResult<RedistributeCompostResult> {
    let (_, def) = get_currency_inner(&currency_id)?;
    if def.status != CurrencyStatus::Active {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Can only redistribute compost for Active currencies".into()
        )));
    }

    // Governance gate: communities with >10 members require authorization
    let community_size = fetch_community_size(&def.creator_dao_did);
    if community_size > 10 {
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
                    "Compost redistribution requires governance authorization for communities >10 members".into()
                )));
            }
        }
    }

    let compost_did = format!("did:mycelix:__compost__:{}", currency_id);
    let compost_bal = get_or_create_minted_balance(compost_did.clone(), currency_id.clone())?;

    if compost_bal.balance <= 0 {
        return Ok(RedistributeCompostResult {
            currency_id,
            total_redistributed: 0,
            recipients: 0,
            per_member_amount: 0,
            remainder_kept: 0,
        });
    }

    // Collect all member DIDs from exchange links (excluding compost pseudo-member)
    let exchange_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("mex:{}", currency_id))?,
            LinkTypes::CurrencyToExchanges,
        )?,
        GetStrategy::default(),
    )?;

    let mut member_dids = std::collections::HashSet::new();
    for link in exchange_links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(ex) = record
                    .entry()
                    .to_app_option::<MintedExchange>()
                    .ok()
                    .flatten()
                {
                    if !ex.provider_did.contains("__compost__") {
                        member_dids.insert(ex.provider_did);
                    }
                    if !ex.receiver_did.contains("__compost__") {
                        member_dids.insert(ex.receiver_did);
                    }
                }
            }
        }
    }

    let member_count = member_dids.len() as u32;
    if member_count == 0 {
        return Ok(RedistributeCompostResult {
            currency_id,
            total_redistributed: 0,
            recipients: 0,
            per_member_amount: 0,
            remainder_kept: compost_bal.balance,
        });
    }

    let per_member = compost_bal.balance / member_count as i32;
    if per_member <= 0 {
        return Ok(RedistributeCompostResult {
            currency_id,
            total_redistributed: 0,
            recipients: member_count,
            per_member_amount: 0,
            remainder_kept: compost_bal.balance,
        });
    }

    let total_distributed = per_member * member_count as i32;
    let remainder = compost_bal.balance - total_distributed;
    let now = sys_time()?;

    // Credit each member
    for did in &member_dids {
        let anchor_key = format!("mbal:{}:{}", currency_id, did);
        let links = get_links(
            LinkQuery::try_new(
                anchor_hash(&anchor_key)?,
                LinkTypes::CurrencyMemberToBalance,
            )?,
            GetStrategy::default(),
        )?;
        if let Some(link) = links.first() {
            if let Some(link_hash) = link.target.clone().into_action_hash() {
                if let Some(record) = get(link_hash, GetOptions::default())? {
                    if let Some(mut bal) = record
                        .entry()
                        .to_app_option::<MintedBalance>()
                        .ok()
                        .flatten()
                    {
                        bal.balance += per_member;
                        bal.last_activity = now;
                        update_entry(record.action_address().clone(), &bal)?;
                    }
                }
            }
        }
    }

    // Debit compost
    let compost_anchor = format!("mbal:{}:{}", currency_id, compost_did);
    let compost_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&compost_anchor)?,
            LinkTypes::CurrencyMemberToBalance,
        )?,
        GetStrategy::default(),
    )?;
    if let Some(link) = compost_links.first() {
        if let Some(link_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(link_hash, GetOptions::default())? {
                if let Some(mut bal) = record
                    .entry()
                    .to_app_option::<MintedBalance>()
                    .ok()
                    .flatten()
                {
                    bal.balance -= total_distributed;
                    bal.last_activity = now;
                    update_entry(record.action_address().clone(), &bal)?;
                }
            }
        }
    }

    Ok(RedistributeCompostResult {
        currency_id,
        total_redistributed: total_distributed,
        recipients: member_count,
        per_member_amount: per_member,
        remainder_kept: remainder,
    })
}

/// Get all minted currency balances for a member across all currencies.
///
/// Returns a portfolio view: one entry per currency the member participates in.
/// Uses the member-currencies index for O(1) lookups (falls back to pending links).
#[hdk_extern]
pub fn get_member_portfolio(member_did: String) -> ExternResult<Vec<MintedBalanceInfo>> {
    let mut portfolio = Vec::new();
    let mut seen_currencies = std::collections::HashSet::new();

    // Primary: use the member-currencies index (populated on first balance creation)
    let member_currency_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("member-currencies:{}", member_did))?,
            LinkTypes::AnchorLinks,
        )?,
        GetStrategy::default(),
    )?;

    for link in member_currency_links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(def) = record
                    .entry()
                    .to_app_option::<CurrencyDefinition>()
                    .ok()
                    .flatten()
                {
                    seen_currencies.insert(def.id);
                }
            }
        }
    }

    // Get balance info for each known currency
    for currency_id in seen_currencies {
        match get_minted_balance(GetMintedBalanceInput {
            currency_id,
            member_did: member_did.clone(),
        }) {
            Ok(info) => portfolio.push(info),
            Err(_) => continue, // Currency may have been retired/deleted
        }
    }

    Ok(portfolio)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PaginatedCurrencyInput {
    pub currency_id: String,
    pub limit: Option<usize>,
    /// Only return exchanges with timestamp >= this value (cursor for forward pagination)
    pub after_timestamp: Option<Timestamp>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ApplyDemurrageInput {
    pub currency_id: String,
    pub member_did: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DemurrageResult {
    pub member_did: String,
    pub currency_id: String,
    pub previous_balance: i32,
    pub deduction: i32,
    pub new_balance: i32,
    pub demurrage_rate: f64,
}

/// Apply demurrage to a member's balance in a community-minted currency.
///
/// Demurrage only applies to positive balances (credits). Negative balances
/// (debts) are exempt — you don't get penalized for owing. The deduction is
/// computed from the time elapsed since the member's last activity.
///
/// Communities that set demurrage_rate = 0.0 are exempt (no-op).
#[hdk_extern]
pub fn apply_minted_demurrage(input: ApplyDemurrageInput) -> ExternResult<DemurrageResult> {
    let (_, def) = get_currency_inner(&input.currency_id)?;

    if def.status != CurrencyStatus::Active {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Demurrage can only be applied to Active currencies".into()
        )));
    }

    if def.params.demurrage_rate <= 0.0 {
        let bal =
            get_or_create_minted_balance(input.member_did.clone(), input.currency_id.clone())?;
        return Ok(DemurrageResult {
            member_did: input.member_did,
            currency_id: input.currency_id,
            previous_balance: bal.balance,
            deduction: 0,
            new_balance: bal.balance,
            demurrage_rate: 0.0,
        });
    }

    let bal = get_or_create_minted_balance(input.member_did.clone(), input.currency_id.clone())?;
    let now = sys_time()?;
    let elapsed = now
        .as_micros()
        .saturating_sub(bal.last_activity.as_micros()) as u64
        / 1_000_000; // Convert microseconds to seconds

    let deduction = compute_minted_demurrage(bal.balance, def.params.demurrage_rate, elapsed);

    if deduction > 0 {
        // Apply deduction by updating balance
        let anchor_key = format!("mbal:{}:{}", input.currency_id, input.member_did);
        let links = get_links(
            LinkQuery::try_new(
                anchor_hash(&anchor_key)?,
                LinkTypes::CurrencyMemberToBalance,
            )?,
            GetStrategy::default(),
        )?;

        if let Some(link) = links.first() {
            if let Some(link_hash) = link.target.clone().into_action_hash() {
                if let Some(record) = get(link_hash, GetOptions::default())? {
                    if let Some(mut updated_bal) = record
                        .entry()
                        .to_app_option::<MintedBalance>()
                        .ok()
                        .flatten()
                    {
                        updated_bal.balance -= deduction;
                        updated_bal.last_activity = now;
                        update_entry(record.action_address().clone(), &updated_bal)?;
                    }
                }
            }
        }

        // Credit deducted amount to compost pseudo-member to preserve zero-sum.
        // The compost balance accumulates demurrage for future redistribution.
        let compost_did = format!("did:mycelix:__compost__:{}", input.currency_id);
        let mut compost =
            get_or_create_minted_balance(compost_did.clone(), input.currency_id.clone())?;
        let compost_anchor = format!("mbal:{}:{}", input.currency_id, compost_did);
        let compost_links = get_links(
            LinkQuery::try_new(
                anchor_hash(&compost_anchor)?,
                LinkTypes::CurrencyMemberToBalance,
            )?,
            GetStrategy::default(),
        )?;
        if let Some(link) = compost_links.first() {
            if let Some(link_hash) = link.target.clone().into_action_hash() {
                if let Some(record) = get(link_hash, GetOptions::default())? {
                    compost.balance += deduction;
                    compost.last_activity = now;
                    update_entry(record.action_address().clone(), &compost)?;
                }
            }
        }
    }

    Ok(DemurrageResult {
        member_did: input.member_did,
        currency_id: input.currency_id,
        previous_balance: bal.balance,
        deduction,
        new_balance: bal.balance - deduction,
        demurrage_rate: def.params.demurrage_rate,
    })
}

// =============================================================================
// INTERNAL HELPERS
// =============================================================================

fn get_currency_inner(currency_id: &str) -> ExternResult<(Record, CurrencyDefinition)> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("currency:{}", currency_id))?,
            LinkTypes::CurrencyIdToDefinition,
        )?,
        GetStrategy::default(),
    )?;

    let link = links
        .first()
        .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
            "Currency not found: {}",
            currency_id
        ))))?;

    let action_hash =
        link.target
            .clone()
            .into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Invalid link target".into()
            )))?;

    let record = get(action_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Currency record not found".into())
    ))?;

    let def = record
        .entry()
        .to_app_option::<CurrencyDefinition>()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Deserialization error: {:?}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Currency entry missing".into()
        )))?;

    Ok((record, def))
}

fn get_or_create_minted_balance(
    member_did: String,
    currency_id: String,
) -> ExternResult<MintedBalance> {
    let anchor_key = format!("mbal:{}:{}", currency_id, member_did);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&anchor_key)?,
            LinkTypes::CurrencyMemberToBalance,
        )?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(bal) = record
                    .entry()
                    .to_app_option::<MintedBalance>()
                    .ok()
                    .flatten()
                {
                    return Ok(bal);
                }
            }
        }
    }

    // Create new zero balance
    let now = sys_time()?;
    let bal = MintedBalance {
        member_did: member_did.clone(),
        currency_id: currency_id.clone(),
        balance: 0,
        total_provided: 0.0,
        total_received: 0.0,
        exchange_count: 0,
        last_activity: now,
    };

    let hash = create_entry(&EntryTypes::MintedBalance(bal.clone()))?;
    create_link(
        anchor_hash(&anchor_key)?,
        hash,
        LinkTypes::CurrencyMemberToBalance,
        (),
    )?;

    // P0: Index member → currency for O(1) portfolio lookups.
    // Skip compost pseudo-members (internal bookkeeping, not real members).
    if !member_did.contains("__compost__") {
        if let Ok((currency_record, _)) = get_currency_inner(&currency_id) {
            create_link(
                anchor_hash(&format!("member-currencies:{}", member_did))?,
                currency_record.action_address().clone(),
                LinkTypes::AnchorLinks,
                (),
            )?;
        }
    }

    Ok(bal)
}

/// Fetch community member count via cross-zome call to governance/identity.
/// Falls back to 0 (permissive) if the cross-zome call is unreachable,
/// allowing standalone operation during development or when governance is offline.
fn fetch_community_size(dao_did: &str) -> u32 {
    // Try cross-zome call to governance bridge for community membership count
    match call(
        CallTargetCell::Local,
        ZomeName::from("finance_bridge"),
        FunctionName::from("get_community_member_count"),
        None,
        dao_did.to_string(),
    ) {
        Ok(ZomeCallResponse::Ok(result)) => result.decode::<u32>().unwrap_or(0),
        _ => 0, // Unreachable or error — permissive fallback
    }
}

fn update_minted_balance(
    member_did: &str,
    currency_id: &str,
    hours: f32,
    is_provider: bool,
) -> ExternResult<()> {
    let anchor_key = format!("mbal:{}:{}", currency_id, member_did);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&anchor_key)?,
            LinkTypes::CurrencyMemberToBalance,
        )?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(link_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(link_hash, GetOptions::default())? {
                if let Some(mut bal) = record
                    .entry()
                    .to_app_option::<MintedBalance>()
                    .ok()
                    .flatten()
                {
                    let now = sys_time()?;
                    if is_provider {
                        bal.balance += hours.round() as i32;
                        bal.total_provided += hours;
                    } else {
                        bal.balance -= hours.round() as i32;
                        bal.total_received += hours;
                    }
                    bal.exchange_count += 1;
                    bal.last_activity = now;
                    update_entry(record.action_address().clone(), &bal)?;
                }
            }
        }
    }
    Ok(())
}
