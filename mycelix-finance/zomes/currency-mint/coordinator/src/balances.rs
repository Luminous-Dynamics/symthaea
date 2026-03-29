// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Balance queries and portfolio views.

use currency_mint_integrity::*;
use hdk::prelude::*;
use mycelix_finance_types::{compute_minted_demurrage, CurrencyStatus};

use crate::helpers::*;
use crate::{GetMintedBalanceInput, MintedBalanceInfo};

/// Get a member's balance in a community-minted currency.
///
/// **Pure read** — no DHT writes are performed. Demurrage is computed in-memory
/// so the returned `effective_balance` reflects decay, but the stored balance is
/// unchanged. Call `apply_minted_demurrage` explicitly when you need to persist
/// demurrage (e.g., before exchanges, or via a periodic governance action).
///
/// RC-10 fix: previously this function called `apply_minted_demurrage` on every
/// read, triggering DHT writes and amplifying race conditions across the system.
#[hdk_extern]
pub fn get_minted_balance(input: GetMintedBalanceInput) -> ExternResult<MintedBalanceInfo> {
    let (_, def) = get_currency_inner(&input.currency_id)?;
    let bal = get_or_create_minted_balance(input.member_did, input.currency_id)?;

    // Compute demurrage in-memory only — no writes.
    let pending_demurrage =
        if def.params.demurrage_rate > 0.0 && def.status == CurrencyStatus::Active {
            let now = sys_time()?;
            let elapsed = now
                .as_micros()
                .saturating_sub(bal.last_activity.as_micros()) as u64
                / 1_000_000; // microseconds → seconds
            compute_minted_demurrage(bal.balance, def.params.demurrage_rate, elapsed)
        } else {
            0
        };

    let effective_balance = bal.balance - pending_demurrage;

    Ok(MintedBalanceInfo {
        member_did: bal.member_did,
        currency_id: bal.currency_id,
        currency_name: def.params.name,
        currency_symbol: def.params.symbol,
        raw_balance: bal.balance,
        effective_balance,
        pending_demurrage,
        last_activity: bal.last_activity,
        balance: effective_balance,
        credit_limit: def.params.credit_limit,
        can_provide: effective_balance < def.params.credit_limit,
        can_receive: effective_balance > -def.params.credit_limit,
        total_provided: bal.total_provided,
        total_received: bal.total_received,
        exchange_count: bal.exchange_count,
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
    let entries = collect_linked_entries::<CurrencyDefinition>(
        &format!("member-currencies:{}", member_did),
        LinkTypes::AnchorLinks,
    )?;
    for (def, _) in entries {
        seen_currencies.insert(def.id);
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
