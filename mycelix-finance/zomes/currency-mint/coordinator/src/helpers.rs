//! Internal helper functions shared across coordinator modules.

use currency_mint_integrity::*;
use hdk::prelude::*;
use mycelix_finance_shared::anchor_hash;
use mycelix_finance_types::compute_minted_demurrage;

pub(crate) fn get_currency_inner(currency_id: &str) -> ExternResult<(Record, CurrencyDefinition)> {
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

    let original_hash =
        link.target
            .clone()
            .into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Invalid link target".into()
            )))?;

    // Follow the update chain to get the latest version of the entry.
    // The link always points to the original create action, but update_entry
    // creates new actions that form a chain: create → update1 → update2 → ...
    // Each update is recorded as an update of its predecessor, so we must
    // recursively follow the chain until we find an action with no updates.
    let record = follow_update_chain(original_hash)?;

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

pub(crate) fn get_or_create_minted_balance(
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
            if let Ok(record) = follow_update_chain(action_hash) {
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

    // Index member → currency for O(1) portfolio lookups.
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
/// Falls back to 0 (permissive) if the cross-zome call is unreachable.
pub(crate) fn fetch_community_size(dao_did: &str) -> u32 {
    match call(
        CallTargetCell::Local,
        ZomeName::from("finance_bridge"),
        FunctionName::from("get_community_member_count"),
        None,
        dao_did.to_string(),
    ) {
        Ok(ZomeCallResponse::Ok(result)) => result.decode::<u32>().unwrap_or(0),
        _ => 0,
    }
}

pub(crate) fn update_minted_balance(
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
            if let Ok(record) = follow_update_chain(link_hash) {
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

pub(crate) fn find_minted_exchange(
    exchange_id: &str,
) -> ExternResult<(MintedExchange, ActionHash)> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("mex-id:{}", exchange_id))?,
            LinkTypes::CurrencyToExchanges,
        )?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        if let Some(link_hash) = link.target.clone().into_action_hash() {
            if let Ok(record) = follow_update_chain(link_hash) {
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

pub(crate) fn find_dispute_record(
    exchange_id: &str,
) -> ExternResult<Option<(Record, MintedDispute)>> {
    let dispute_anchor = format!("dispute:{}", exchange_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&dispute_anchor)?, LinkTypes::ExchangeToDispute)?,
        GetStrategy::default(),
    )?;

    let Some(link) = links.first() else {
        return Ok(None);
    };

    let action_hash =
        link.target
            .clone()
            .into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Invalid dispute link target".into()
            )))?;

    let record = follow_update_chain(action_hash)?;

    let dispute = record
        .entry()
        .to_app_option::<MintedDispute>()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Deserialization error: {:?}",
                e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Dispute entry missing".into()
        )))?;

    Ok(Some((record, dispute)))
}

pub(crate) fn remove_from_active_index(currency_action_hash: &ActionHash) -> ExternResult<()> {
    let active_links = get_links(
        LinkQuery::try_new(
            anchor_hash("all-active-currencies")?,
            LinkTypes::DaoToCurrencies,
        )?,
        GetStrategy::default(),
    )?;
    for link in active_links {
        if link.target.clone().into_action_hash() == Some(currency_action_hash.clone()) {
            delete_link(link.create_link_hash, GetOptions::default())?;
        }
    }
    Ok(())
}

pub(crate) fn count_currency_exchanges(currency_id: &str) -> ExternResult<(u64, u64, u64)> {
    let exchange_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&format!("mex:{}", currency_id))?,
            LinkTypes::CurrencyToExchanges,
        )?,
        GetStrategy::default(),
    )?;

    let mut total = 0u64;
    let mut confirmed = 0u64;
    let mut pending = 0u64;

    for link in exchange_links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Ok(record) = follow_update_chain(action_hash) {
                if let Some(ex) = record
                    .entry()
                    .to_app_option::<MintedExchange>()
                    .ok()
                    .flatten()
                {
                    total += 1;
                    if ex.confirmed {
                        confirmed += 1;
                    } else {
                        pending += 1;
                    }
                }
            }
        }
    }

    Ok((total, confirmed, pending))
}

pub(crate) fn collect_currency_members(
    currency_id: &str,
) -> ExternResult<std::collections::HashSet<String>> {
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
            if let Ok(record) = follow_update_chain(action_hash) {
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
    Ok(member_dids)
}

pub(crate) fn compute_demurrage(balance: i32, rate: f64, elapsed_secs: u64) -> i32 {
    compute_minted_demurrage(balance, rate, elapsed_secs)
}

/// Recursively follow the Holochain update chain from an original action hash
/// to find the latest version of a record. Each `update_entry` creates a new
/// action that is recorded as an update of its predecessor.
pub(crate) fn follow_update_chain(action_hash: ActionHash) -> ExternResult<Record> {
    let mut current_hash = action_hash;
    loop {
        let details = get_details(current_hash.clone(), GetOptions::default())?.ok_or(
            wasm_error!(WasmErrorInner::Guest("Currency record not found".into())),
        )?;
        match details {
            Details::Record(record_details) => {
                if let Some(latest_update) = record_details.updates.last() {
                    // Continue following the chain
                    current_hash = latest_update.action_address().clone();
                } else {
                    // No more updates — this is the latest
                    return Ok(record_details.record);
                }
            }
            _ => {
                // Fallback: just get the record directly
                return get(current_hash, GetOptions::default())?.ok_or(wasm_error!(
                    WasmErrorInner::Guest("Currency record not found".into())
                ));
            }
        }
    }
}
