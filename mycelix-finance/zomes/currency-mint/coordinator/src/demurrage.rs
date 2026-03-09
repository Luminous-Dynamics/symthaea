//! Demurrage application and compost redistribution.

use hdk::prelude::*;
use mycelix_finance_types::{compute_minted_demurrage, CurrencyStatus};

use crate::helpers::*;
use crate::{ApplyDemurrageInput, DemurrageResult, RedistributeCompostResult};

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
        // Apply deduction to member balance
        mutate_balance(&input.member_did, &input.currency_id, |b| {
            b.balance -= deduction;
            b.last_activity = now;
        })?;

        // Credit deducted amount to compost pseudo-member to preserve zero-sum.
        let compost_did = format!("did:mycelix:__compost__:{}", input.currency_id);
        // Ensure compost balance entry exists
        let _ = get_or_create_minted_balance(compost_did.clone(), input.currency_id.clone())?;
        mutate_balance(&compost_did, &input.currency_id, |b| {
            b.balance += deduction;
            b.last_activity = now;
        })?;
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

/// Apply demurrage to ALL positive-balance members of a currency at once.
///
/// Batch version of `apply_minted_demurrage`. Useful for scheduled governance
/// actions rather than relying solely on lazy-on-read demurrage.
/// Returns a result for each member that had a deduction applied.
#[hdk_extern]
pub fn apply_demurrage_all(currency_id: String) -> ExternResult<Vec<DemurrageResult>> {
    let (_, def) = get_currency_inner(&currency_id)?;

    if def.status != CurrencyStatus::Active {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Demurrage can only be applied to Active currencies".into()
        )));
    }

    if def.params.demurrage_rate <= 0.0 {
        return Ok(Vec::new());
    }

    let member_dids = collect_currency_members(&currency_id)?;

    let mut results = Vec::new();
    for did in member_dids {
        let result = apply_minted_demurrage(ApplyDemurrageInput {
            currency_id: currency_id.clone(),
            member_did: did,
        })?;
        if result.deduction > 0 {
            results.push(result);
        }
    }

    Ok(results)
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

    let member_dids = collect_currency_members(&currency_id)?;

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
        mutate_balance(did, &currency_id, |b| {
            b.balance += per_member;
            b.last_activity = now;
        })?;
    }

    // Debit compost
    mutate_balance(&compost_did, &currency_id, |b| {
        b.balance -= total_distributed;
        b.last_activity = now;
    })?;

    Ok(RedistributeCompostResult {
        currency_id,
        total_redistributed: total_distributed,
        recipients: member_count,
        per_member_amount: per_member,
        remainder_kept: remainder,
    })
}
