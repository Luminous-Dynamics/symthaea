// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dispute opening, resolution, and queries.

use currency_mint_integrity::*;
use hdk::prelude::*;
use mycelix_finance_shared::anchor_hash;
use mycelix_finance_types::CurrencyStatus;

use crate::helpers::*;
use crate::{OpenDisputeInput, ResolveDisputeInput};

/// Open a dispute on a confirmed exchange.
///
/// Only the provider or receiver of the exchange can open a dispute.
/// Creates a MintedDispute entry linked to the exchange. A dispute does
/// not reverse balances — that only happens on resolution (accept).
#[hdk_extern]
pub fn open_minted_dispute(input: OpenDisputeInput) -> ExternResult<MintedDispute> {
    let (exchange, _) = find_minted_exchange(&input.exchange_id)?;

    if !exchange.confirmed {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot dispute an unconfirmed exchange — cancel it instead".into()
        )));
    }

    // No disputes on retired currencies — balances are frozen, resolution can't reverse them
    let (_, def) = get_currency_inner(&exchange.currency_id)?;
    if def.status == CurrencyStatus::Retired {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot open disputes on retired currencies — balances are frozen".into()
        )));
    }

    // Only participants can open disputes
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);
    if caller_did != exchange.provider_did && caller_did != exchange.receiver_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only exchange participants can open a dispute".into()
        )));
    }

    // Check no existing dispute
    let dispute_anchor = format!("dispute:{}", input.exchange_id);
    let existing = get_links(
        LinkQuery::try_new(anchor_hash(&dispute_anchor)?, LinkTypes::ExchangeToDispute)?,
        GetStrategy::default(),
    )?;
    if !existing.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "A dispute already exists for this exchange".into()
        )));
    }

    let now = sys_time()?;
    let dispute = MintedDispute {
        exchange_id: input.exchange_id.clone(),
        opener_did: caller_did,
        reason: input.reason,
        resolved: None,
        resolver_did: None,
        resolution_reason: None,
        opened_at: now,
        resolved_at: None,
    };

    let hash = create_entry(&EntryTypes::MintedDispute(dispute.clone()))?;
    create_link(
        anchor_hash(&dispute_anchor)?,
        hash,
        LinkTypes::ExchangeToDispute,
        (),
    )?;

    Ok(dispute)
}

/// Resolve a dispute on a confirmed exchange.
///
/// - `accept = true`: reverses the exchange balances (provider -hours, receiver +hours)
/// - `accept = false`: keeps original balances (dispute rejected)
///
/// Requires governance authorization for communities with >10 members.
#[hdk_extern]
pub fn resolve_minted_dispute(input: ResolveDisputeInput) -> ExternResult<MintedDispute> {
    let (exchange, _) = find_minted_exchange(&input.exchange_id)?;

    let (dispute_record, dispute) = find_dispute_record(&input.exchange_id)?.ok_or(wasm_error!(
        WasmErrorInner::Guest("No dispute found for this exchange".into())
    ))?;

    if dispute.resolved.is_some() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "This dispute has already been resolved".into()
        )));
    }

    // Governance gate: communities above threshold require authorization
    let (_, def) = get_currency_inner(&exchange.currency_id)?;
    let community_size = fetch_community_size(&def.creator_dao_did)?;
    if community_size > COMMUNITY_GOVERNANCE_THRESHOLD {
        match call(
            CallTargetCell::Local,
            ZomeName::from("tend"),
            FunctionName::from("verify_governance_agent"),
            None,
            (),
        ) {
            Ok(ZomeCallResponse::Ok(_)) => {}
            _ => {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Dispute resolution requires governance authorization for communities >10 members"
                        .into()
                )));
            }
        }
    }

    let caller = agent_info()?.agent_initial_pubkey;
    let resolver_did = format!("did:mycelix:{}", caller);
    let now = sys_time()?;

    // If accepted, reverse the balances. Uses the same PendingMintedAdjustment
    // crash-recovery pattern as confirm_minted_exchange (reversed: true) —
    // without it, a crash between the two update_minted_balance calls below
    // would permanently break the zero-sum invariant with no way to detect
    // or fix it, since resolve_minted_dispute only marks the dispute
    // `resolved` (which is what guards against being re-run) after both
    // updates succeed.
    if input.accept {
        let pending_adj_hash = create_entry(&EntryTypes::PendingMintedAdjustment(
            PendingMintedAdjustment {
                exchange_id: exchange.id.clone(),
                provider_did: exchange.provider_did.clone(),
                receiver_did: exchange.receiver_did.clone(),
                hours: exchange.hours as f64,
                currency_id: exchange.currency_id.clone(),
                provider_completed: false,
                receiver_completed: false,
                created_at: now,
                reversed: true,
            },
        ))?;
        create_link(
            anchor_hash("pending-minted-adjustments")?,
            pending_adj_hash.clone(),
            LinkTypes::PendingMintedAdjustmentToExchange,
            (),
        )?;

        // Reverse: provider loses what they gained, receiver gets back what they lost
        update_minted_balance(
            &exchange.provider_did,
            &exchange.currency_id,
            exchange.hours,
            false, // provider now "receives" (balance decreases)
        )?;
        let pending_adj_hash = update_entry(
            pending_adj_hash,
            &EntryTypes::PendingMintedAdjustment(PendingMintedAdjustment {
                exchange_id: exchange.id.clone(),
                provider_did: exchange.provider_did.clone(),
                receiver_did: exchange.receiver_did.clone(),
                hours: exchange.hours as f64,
                currency_id: exchange.currency_id.clone(),
                provider_completed: true,
                receiver_completed: false,
                created_at: now,
                reversed: true,
            }),
        )?;

        update_minted_balance(
            &exchange.receiver_did,
            &exchange.currency_id,
            exchange.hours,
            true, // receiver now "provides" (balance increases)
        )?;
        update_entry(
            pending_adj_hash,
            &EntryTypes::PendingMintedAdjustment(PendingMintedAdjustment {
                exchange_id: exchange.id.clone(),
                provider_did: exchange.provider_did.clone(),
                receiver_did: exchange.receiver_did.clone(),
                hours: exchange.hours as f64,
                currency_id: exchange.currency_id.clone(),
                provider_completed: true,
                receiver_completed: true,
                created_at: now,
                reversed: true,
            }),
        )?;
    }

    let resolved_dispute = MintedDispute {
        resolved: Some(input.accept),
        resolver_did: Some(resolver_did),
        resolution_reason: Some(input.resolution_reason),
        resolved_at: Some(now),
        ..dispute
    };

    update_entry(
        dispute_record.action_address().clone(),
        &EntryTypes::MintedDispute(resolved_dispute.clone()),
    )?;

    Ok(resolved_dispute)
}

/// Get an existing dispute for an exchange, if one exists.
#[hdk_extern]
pub fn get_dispute(exchange_id: String) -> ExternResult<Option<MintedDispute>> {
    Ok(find_dispute_record(&exchange_id)?.map(|(_, dispute)| dispute))
}
