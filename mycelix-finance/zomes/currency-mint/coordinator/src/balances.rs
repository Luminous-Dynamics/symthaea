//! Balance queries and portfolio views.

use currency_mint_integrity::*;
use hdk::prelude::*;
use mycelix_finance_types::CurrencyStatus;

use crate::helpers::*;
use crate::{ApplyDemurrageInput, GetMintedBalanceInput, MintedBalanceInfo};

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
            let result = crate::demurrage::apply_minted_demurrage(ApplyDemurrageInput {
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
