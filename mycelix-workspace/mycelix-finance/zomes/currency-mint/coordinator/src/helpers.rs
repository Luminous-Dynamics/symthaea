// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Internal helper functions shared across coordinator modules.

use currency_mint_integrity::*;
use hdk::prelude::*;
use mycelix_finance_shared::anchor_hash;
pub(crate) use mycelix_finance_shared::follow_update_chain;
pub(crate) use mycelix_finance_shared::pick_race_winner;
pub(crate) use mycelix_finance_shared::COMMUNITY_GOVERNANCE_THRESHOLD;
pub(crate) use mycelix_finance_shared::{validate_did_format, validate_id};

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

    // Pick the link with the lowest target ActionHash (deterministic winner)
    // to handle orphaned links from past race conditions.
    if let Some(action_hash) = links
        .iter()
        .filter_map(|l| l.target.clone().into_action_hash())
        .min()
    {
        let record = follow_update_chain(action_hash)?;
        let bal = record
            .entry()
            .to_app_option::<MintedBalance>()
            .map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Balance deserialization error for {}:{}: {:?}",
                    currency_id, member_did, e
                )))
            })?;
        if let Some(bal) = bal {
            return Ok(bal);
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
        hash.clone(),
        LinkTypes::CurrencyMemberToBalance,
        (),
    )?;

    // RC-12: Race condition guard — re-read links to detect concurrent creators.
    // If multiple links exist, deterministically pick the one with the lowest
    // target ActionHash. If we're not the winner, return the winner's entry.
    let recheck_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&anchor_key)?,
            LinkTypes::CurrencyMemberToBalance,
        )?,
        GetStrategy::default(),
    )?;
    if recheck_links.len() > 1 {
        if let Some(winner_link) = recheck_links
            .iter()
            .filter_map(|l| l.target.clone().into_action_hash().map(|h| (l, h)))
            .min_by(|(_, a), (_, b)| a.cmp(b))
        {
            let winner_hash = winner_link.1;
            if winner_hash != hash {
                // We lost the race — return the winner's entry instead
                let record = follow_update_chain(winner_hash)?;
                let winner_bal = record
                    .entry()
                    .to_app_option::<MintedBalance>()
                    .map_err(|e| {
                        wasm_error!(WasmErrorInner::Guest(format!(
                            "Balance deserialization error for {}:{}: {:?}",
                            currency_id, member_did, e
                        )))
                    })?
                    .ok_or(wasm_error!(WasmErrorInner::Guest(
                        "Winner balance entry missing".into()
                    )))?;
                return Ok(winner_bal);
            }
        }
    }

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
///
/// SECURITY: Fail-closed — returns error if the finance bridge is unreachable.
/// This is consistent with STRICT_GOVERNANCE_MODE=true in the finance bridge.
/// Previously returned 0 (permissive), which bypassed the governance proposal
/// requirement for communities >10 members.
pub(crate) fn fetch_community_size(dao_did: &str) -> ExternResult<u32> {
    match call(
        CallTargetCell::Local,
        ZomeName::from("finance_bridge"),
        FunctionName::from("get_community_member_count"),
        None,
        dao_did.to_string(),
    ) {
        Ok(ZomeCallResponse::Ok(result)) => {
            let count = result.decode::<u32>().unwrap_or(0);
            Ok(count)
        }
        Ok(other) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "fetch_community_size: finance_bridge returned unexpected response for {}: {:?}",
            dao_did, other
        )))),
        Err(e) => Err(wasm_error!(WasmErrorInner::Guest(format!(
            "fetch_community_size: finance_bridge unreachable for {}: {:?}",
            dao_did, e
        )))),
    }
}

pub(crate) fn update_minted_balance(
    member_did: &str,
    currency_id: &str,
    hours: f32,
    is_provider: bool,
) -> ExternResult<()> {
    let now = sys_time()?;
    mutate_balance(member_did, currency_id, |bal| {
        if is_provider {
            bal.balance += hours.round() as i32;
            bal.total_provided += hours;
        } else {
            bal.balance -= hours.round() as i32;
            bal.total_received += hours;
        }
        bal.exchange_count += 1;
        bal.last_activity = now;
    })?;
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

    let link = links
        .first()
        .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
            "Exchange {} not found",
            exchange_id
        ))))?;

    let link_hash =
        link.target
            .clone()
            .into_action_hash()
            .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
                "Invalid link target for exchange {}",
                exchange_id
            ))))?;

    let record = follow_update_chain(link_hash)?;

    let ex = record
        .entry()
        .to_app_option::<MintedExchange>()
        .map_err(|e| {
            wasm_error!(WasmErrorInner::Guest(format!(
                "Exchange {} deserialization error: {:?}",
                exchange_id, e
            )))
        })?
        .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
            "Exchange {} entry missing",
            exchange_id
        ))))?;

    Ok((ex, record.action_address().clone()))
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
        let Some(action_hash) = link.target.into_action_hash() else {
            continue; // deleted link
        };
        let record = follow_update_chain(action_hash)?;
        let ex = record
            .entry()
            .to_app_option::<MintedExchange>()
            .map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Exchange deserialization error in count for {}: {:?}",
                    currency_id, e
                )))
            })?;
        if let Some(ex) = ex {
            total += 1;
            if ex.confirmed {
                confirmed += 1;
            } else {
                pending += 1;
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
        let Some(action_hash) = link.target.into_action_hash() else {
            continue; // deleted link
        };
        let record = follow_update_chain(action_hash)?;
        let ex = record
            .entry()
            .to_app_option::<MintedExchange>()
            .map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Exchange deserialization error in members for {}: {:?}",
                    currency_id, e
                )))
            })?;
        if let Some(ex) = ex {
            if !ex.provider_did.contains("__compost__") {
                member_dids.insert(ex.provider_did);
            }
            if !ex.receiver_did.contains("__compost__") {
                member_dids.insert(ex.receiver_did);
            }
        }
    }
    Ok(member_dids)
}

/// Collect all entries of type `T` reachable via links from an anchor.
///
/// Follows the update chain for each link target, deserializes to `T`,
/// and returns all successfully decoded entries with their latest action hash.
/// Skips links that can't be resolved (deleted entries, network errors) but
/// logs deserialization errors via `debug!` for observability.
pub(crate) fn collect_linked_entries<T: TryFrom<SerializedBytes, Error = SerializedBytesError>>(
    anchor: &str,
    link_type: LinkTypes,
) -> ExternResult<Vec<(T, ActionHash)>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(anchor)?, link_type)?,
        GetStrategy::default(),
    )?;

    let mut entries = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Ok(record) = follow_update_chain(action_hash) {
                match record.entry().to_app_option::<T>() {
                    Ok(Some(entry)) => {
                        entries.push((entry, record.action_address().clone()));
                    }
                    Ok(None) => {} // No entry content (e.g. deleted)
                    Err(e) => {
                        debug!(
                            "collect_linked_entries: deserialization error for {}: {:?}",
                            anchor, e
                        );
                    }
                }
            }
        }
    }
    Ok(entries)
}

/// Maximum retries for optimistic-locking balance mutations.
/// If a concurrent update is detected (fork), we re-read and retry.
const MAX_BALANCE_RETRIES: usize = 3;

/// Mutate a member's balance entry in-place via a closure, with optimistic
/// locking and retry to handle concurrent updates (race conditions RC-1..RC-5).
///
/// Pattern:
/// 1. Follow update chain to get latest record + value
/// 2. Apply mutation
/// 3. `update_entry` using the record's action hash
/// 4. Re-read via `follow_update_chain` from the original link target
/// 5. If re-read value doesn't match what we wrote, a concurrent update won — retry
///
/// Returns `Ok(None)` if no balance link exists (no-op).
pub(crate) fn mutate_balance(
    member_did: &str,
    currency_id: &str,
    f: impl Fn(&mut MintedBalance),
) -> ExternResult<Option<MintedBalance>> {
    let anchor_key = format!("mbal:{}:{}", currency_id, member_did);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&anchor_key)?,
            LinkTypes::CurrencyMemberToBalance,
        )?,
        GetStrategy::default(),
    )?;

    // Pick the link with the lowest target ActionHash (deterministic winner)
    // to handle orphaned links from past race conditions (RC-12).
    let Some(original_hash) = links
        .iter()
        .filter_map(|l| l.target.clone().into_action_hash())
        .min()
    else {
        return Ok(None);
    };

    for attempt in 0..MAX_BALANCE_RETRIES {
        let record = follow_update_chain(original_hash.clone())?;
        let mut bal = record
            .entry()
            .to_app_option::<MintedBalance>()
            .map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Balance deserialization error for {}:{}: {:?}",
                    currency_id, member_did, e
                )))
            })?
            .ok_or(wasm_error!(WasmErrorInner::Guest(format!(
                "Balance entry missing for {}:{}",
                currency_id, member_did
            ))))?;

        f(&mut bal);
        let expected = bal.clone();
        update_entry(record.action_address().clone(), &bal)?;

        // Verify our update won: re-read from the original link target
        let verify_record = follow_update_chain(original_hash.clone())?;
        if let Ok(Some(actual)) = verify_record.entry().to_app_option::<MintedBalance>() {
            if actual.balance == expected.balance
                && actual.exchange_count == expected.exchange_count
            {
                return Ok(Some(expected));
            }
        }

        // Concurrent update detected — retry unless exhausted
        if attempt == MAX_BALANCE_RETRIES - 1 {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Balance update for {}:{} failed after {} retries due to concurrent modifications",
                currency_id, member_did, MAX_BALANCE_RETRIES
            ))));
        }
        debug!(
            "mutate_balance: concurrent update detected for {}:{}, retry {}/{}",
            currency_id,
            member_did,
            attempt + 1,
            MAX_BALANCE_RETRIES
        );
    }

    // Unreachable, but satisfies the compiler
    Err(wasm_error!(WasmErrorInner::Guest(
        "Balance update failed: retry loop exited unexpectedly".into()
    )))
}
