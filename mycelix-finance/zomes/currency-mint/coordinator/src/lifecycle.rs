//! Currency lifecycle management: creation, activation, suspension, retirement, discovery, amendment.

use currency_mint_integrity::*;
use hdk::prelude::*;
use mycelix_finance_shared::anchor_hash;
use mycelix_finance_types::CurrencyStatus;

use crate::helpers::*;
use crate::{ActivateCurrencyInput, AmendCurrencyParamsInput, CreateCurrencyInput};

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

    // Remove from active currencies index — suspended currencies shouldn't appear in discovery
    remove_from_active_index(record.action_address())?;

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

    // Re-add to global active currencies index (guard against duplicate links)
    let active_base = anchor_hash("all-active-currencies")?;
    let existing_active = get_links(
        LinkQuery::try_new(active_base.clone(), LinkTypes::DaoToCurrencies)?,
        GetStrategy::default(),
    )?;
    let already_linked = existing_active
        .iter()
        .any(|l| l.target.clone().into_action_hash() == Some(record.action_address().clone()));
    if !already_linked {
        create_link(
            active_base,
            record.action_address().clone(),
            LinkTypes::DaoToCurrencies,
            (),
        )?;
    }

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

    // Clean up the active currencies index to prevent stale link accumulation
    remove_from_active_index(record.action_address())?;

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
    let entries = collect_linked_entries::<CurrencyDefinition>(
        &format!("dao-currencies:{}", dao_did),
        LinkTypes::DaoToCurrencies,
    )?;
    Ok(entries.into_iter().map(|(def, _)| def).collect())
}

/// List all Active currencies across all DAOs (global discovery).
#[hdk_extern]
pub fn list_active_currencies(_: ()) -> ExternResult<Vec<CurrencyDefinition>> {
    let entries = collect_linked_entries::<CurrencyDefinition>(
        "all-active-currencies",
        LinkTypes::DaoToCurrencies,
    )?;
    // Only include actually Active currencies (link may be stale)
    Ok(entries
        .into_iter()
        .map(|(def, _)| def)
        .filter(|def| def.status == CurrencyStatus::Active)
        .collect())
}

/// Search active currencies by name or symbol (case-insensitive substring match).
///
/// Scans the global active currency index. Useful for multi-community discovery
/// when a member wants to find currencies to join.
#[hdk_extern]
pub fn search_currencies(query: String) -> ExternResult<Vec<CurrencyDefinition>> {
    let query_lower = query.to_lowercase();
    let entries = collect_linked_entries::<CurrencyDefinition>(
        "all-active-currencies",
        LinkTypes::DaoToCurrencies,
    )?;
    Ok(entries
        .into_iter()
        .map(|(def, _)| def)
        .filter(|def| {
            def.status == CurrencyStatus::Active
                && (def.params.name.to_lowercase().contains(&query_lower)
                    || def.params.symbol.to_lowercase().contains(&query_lower))
        })
        .collect())
}

/// Amend the parameters of an existing currency.
///
/// Validates the new parameters against constitutional limits. Only Draft or Active
/// currencies can be amended. Communities with >10 members require governance
/// authorization (proposal ID).
#[hdk_extern]
pub fn amend_currency_params(input: AmendCurrencyParamsInput) -> ExternResult<CurrencyDefinition> {
    let (record, def) = get_currency_inner(&input.currency_id)?;

    match def.status {
        CurrencyStatus::Draft | CurrencyStatus::Active => {}
        CurrencyStatus::Suspended => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Cannot amend parameters while currency is Suspended".into()
            )));
        }
        CurrencyStatus::Retired => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Cannot amend parameters of a Retired currency".into()
            )));
        }
    }

    // Validate new params against constitutional limits
    if let Err(e) = input.new_params.validate() {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "New parameters violate constitutional limits: {}",
            e
        ))));
    }

    // Governance gate
    let community_size = fetch_community_size(&def.creator_dao_did);
    if community_size > 10 {
        if input.governance_proposal_id.is_none() {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Communities with >10 members require a governance proposal to amend parameters"
                    .into()
            )));
        }
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
                    "Parameter amendment requires governance authorization for communities >10 members"
                        .into()
                )));
            }
        }
    }

    let updated = CurrencyDefinition {
        params: input.new_params,
        governance_proposal_id: input.governance_proposal_id.or(def.governance_proposal_id),
        ..def
    };

    update_entry(
        record.action_address().clone(),
        &EntryTypes::CurrencyDefinition(updated.clone()),
    )?;

    Ok(updated)
}
