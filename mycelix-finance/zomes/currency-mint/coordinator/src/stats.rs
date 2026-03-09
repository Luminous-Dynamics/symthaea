//! Currency statistics, member listing, demurrage reports, and compost balance queries.

use hdk::prelude::*;
use mycelix_finance_types::compute_minted_demurrage;

use crate::helpers::*;
use crate::{CompostBalance, CurrencyStats, DemurrageReport, DemurrageReportEntry};

/// Get aggregate stats for a community-minted currency.
///
/// Returns member count, total positive balances (credit), total negative balances (debt),
/// exchange count, and net sum (should always be 0 for zero-sum currencies).
#[hdk_extern]
pub fn get_currency_stats(currency_id: String) -> ExternResult<CurrencyStats> {
    let (_, def) = get_currency_inner(&currency_id)?;

    let member_dids = collect_currency_members(&currency_id)?;
    let (total_exchanges, confirmed_count, pending_count) = count_currency_exchanges(&currency_id)?;

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

    // Include compost balance for complete zero-sum picture
    let compost_did = format!("did:mycelix:__compost__:{}", currency_id);
    let compost = get_or_create_minted_balance(compost_did, currency_id.clone())?;

    Ok(CurrencyStats {
        currency_id,
        currency_name: def.params.name,
        currency_symbol: def.params.symbol,
        status: def.status,
        member_count: member_dids.len() as u32,
        total_credit,
        total_debt,
        compost_balance: compost.balance as i64,
        net_sum: total_credit + total_debt + compost.balance as i64, // should be 0
        total_exchanges,
        confirmed_exchanges: confirmed_count,
        pending_exchanges: pending_count,
    })
}

/// List all member DIDs for a community-minted currency.
///
/// Returns unique member DIDs (excluding the compost pseudo-member) collected
/// from exchange history. Useful for governance flows and UI member lists.
#[hdk_extern]
pub fn list_currency_members(currency_id: String) -> ExternResult<Vec<String>> {
    // Verify currency exists
    let _ = get_currency_inner(&currency_id)?;
    let members = collect_currency_members(&currency_id)?;
    let mut sorted: Vec<String> = members.into_iter().collect();
    sorted.sort();
    Ok(sorted)
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

    let member_dids = collect_currency_members(&currency_id)?;

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
