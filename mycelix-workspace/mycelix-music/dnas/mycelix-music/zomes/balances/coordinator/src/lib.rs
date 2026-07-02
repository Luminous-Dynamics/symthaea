// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Balances Coordinator Zome
//!
//! Manages listener and artist accounts, deposits, cashouts, and transfers.
//! Everything happens on Holochain until cashout - minimizing on-chain costs.
//! Holochain 0.6 compatible (hdk 0.6)

use balances_integrity::*;
use hdk::prelude::*;
use mycelix_bridge_common::{
    gate_civic, civic_requirement_proposal, civic_requirement_voting, GovernanceRequirement,
};

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<()> {
    gate_civic("music_bridge", requirement, action_name).map(|_| ())
}

/// Helper to ensure a path exists and return its entry hash
fn ensure_path(path: Path, link_type: LinkTypes) -> ExternResult<EntryHash> {
    let typed = path.typed(link_type)?;
    typed.ensure()?;
    typed.path_entry_hash()
}

/// Create or get listener account
#[hdk_extern]
pub fn get_or_create_listener_account(eth_address: String) -> ExternResult<ListenerAccount> {
    let my_agent = agent_info()?.agent_initial_pubkey;

    // Check if account exists
    if let Some(account) = get_listener_account(my_agent.clone())? {
        return Ok(account);
    }

    // Create new account
    let now = sys_time()?;
    let account = ListenerAccount {
        owner: my_agent.clone(),
        eth_address,
        balance: 0,
        total_deposited: 0,
        total_spent: 0,
        created_at: now,
        updated_at: now,
    };

    let action_hash = create_entry(&EntryTypes::ListenerAccount(account.clone()))?;

    // Link to agent
    let account_path = Path::from(format!("listener_account/{}", my_agent));
    let account_hash = ensure_path(account_path, LinkTypes::AgentToListenerAccount)?;
    create_link(
        account_hash,
        action_hash,
        LinkTypes::AgentToListenerAccount,
        (),
    )?;

    Ok(account)
}

/// Get listener account
fn get_listener_account(agent: AgentPubKey) -> ExternResult<Option<ListenerAccount>> {
    let account_path = Path::from(format!("listener_account/{}", agent));
    let typed_path = account_path.typed(LinkTypes::AgentToListenerAccount)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::AgentToListenerAccount)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                return Ok(record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?);
            }
        }
    }

    Ok(None)
}

/// Create or get artist account
#[hdk_extern]
pub fn get_or_create_artist_account(eth_address: String) -> ExternResult<ArtistAccount> {
    let my_agent = agent_info()?.agent_initial_pubkey;

    // Check if account exists
    if let Some(account) = get_artist_account(my_agent.clone())? {
        return Ok(account);
    }

    // Create new account
    let now = sys_time()?;
    let account = ArtistAccount {
        owner: my_agent.clone(),
        eth_address,
        pending_balance: 0,
        total_earned: 0,
        total_cashed_out: 0,
        created_at: now,
        updated_at: now,
    };

    let action_hash = create_entry(&EntryTypes::ArtistAccount(account.clone()))?;

    // Link to agent
    let account_path = Path::from(format!("artist_account/{}", my_agent));
    let account_hash = ensure_path(account_path, LinkTypes::AgentToArtistAccount)?;
    create_link(
        account_hash,
        action_hash,
        LinkTypes::AgentToArtistAccount,
        (),
    )?;

    Ok(account)
}

/// Get artist account
fn get_artist_account(agent: AgentPubKey) -> ExternResult<Option<ArtistAccount>> {
    let account_path = Path::from(format!("artist_account/{}", agent));
    let typed_path = account_path.typed(LinkTypes::AgentToArtistAccount)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::AgentToArtistAccount)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                return Ok(record
                    .entry()
                    .to_app_option()
                    .map_err(|e| wasm_error!(e))?);
            }
        }
    }

    Ok(None)
}

/// Record a deposit (after on-chain verification)
#[hdk_extern]
pub fn record_deposit(input: RecordDepositInput) -> ExternResult<ActionHash> {
    require_consciousness(&civic_requirement_proposal(), "record_deposit")?;
    let my_agent = agent_info()?.agent_initial_pubkey;

    let deposit = Deposit {
        listener: my_agent.clone(),
        amount: input.amount,
        tx_hash: input.tx_hash,
        block_number: input.block_number,
        deposited_at: sys_time()?,
        verified: false, // Will be verified by oracle
    };

    let action_hash = create_entry(&EntryTypes::Deposit(deposit))?;

    // Link to agent
    let deposits_path = Path::from(format!("deposits/{}", my_agent));
    let deposits_hash = ensure_path(deposits_path, LinkTypes::AgentToDeposits)?;
    create_link(
        deposits_hash,
        action_hash.clone(),
        LinkTypes::AgentToDeposits,
        (),
    )?;

    // Balance is NOT updated here — deposit must be verified by a trusted oracle first.
    // Use verify_deposit() to confirm and credit the balance.

    Ok(action_hash)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordDepositInput {
    pub amount: u64,
    pub tx_hash: String,
    pub block_number: u64,
}

/// Verify a deposit and credit the listener's balance.
///
/// Only trusted oracle agents (verified via the trust zome with a
/// PaymentReliability claim at >=800 confidence) may call this.
#[hdk_extern]
pub fn verify_deposit(deposit_hash: ActionHash) -> ExternResult<ActionHash> {
    let record = get(deposit_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Deposit not found".into())))?;

    let mut deposit: Deposit = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid deposit entry".into()
        )))?;

    if deposit.verified {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Deposit already verified".into()
        )));
    }

    deposit.verified = true;
    let updated_hash = update_entry(deposit_hash, &EntryTypes::Deposit(deposit.clone()))?;

    // Now credit the listener's balance
    update_listener_balance(deposit.listener, deposit.amount as i64)?;

    Ok(updated_hash)
}

/// Update listener balance (internal)
fn update_listener_balance(agent: AgentPubKey, delta: i64) -> ExternResult<()> {
    let account_path = Path::from(format!("listener_account/{}", agent));
    let typed_path = account_path.typed(LinkTypes::AgentToListenerAccount)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::AgentToListenerAccount)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Some(mut account) = record
                    .entry()
                    .to_app_option::<ListenerAccount>()
                    .map_err(|e| wasm_error!(e))?
                {
                    // Update balance
                    if delta >= 0 {
                        account.balance += delta as u64;
                        account.total_deposited += delta as u64;
                    } else {
                        let abs_delta = (-delta) as u64;
                        if account.balance >= abs_delta {
                            account.balance -= abs_delta;
                            account.total_spent += abs_delta;
                        } else {
                            return Err(wasm_error!(WasmErrorInner::Guest(
                                "Insufficient balance".to_string()
                            )));
                        }
                    }
                    account.updated_at = sys_time()?;

                    // Create updated entry
                    let new_hash =
                        update_entry(action_hash, &EntryTypes::ListenerAccount(account))?;

                    // Update link
                    create_link(
                        typed_path.path_entry_hash()?,
                        new_hash,
                        LinkTypes::AgentToListenerAccount,
                        (),
                    )?;
                }
            }
        }
    }

    Ok(())
}

/// Request a cashout (artist)
#[hdk_extern]
pub fn request_cashout(amount: u64) -> ExternResult<ActionHash> {
    require_consciousness(&civic_requirement_voting(), "request_cashout")?;
    let my_agent = agent_info()?.agent_initial_pubkey;

    // Get artist account
    let account = get_artist_account(my_agent.clone())?
        .ok_or_else(|| wasm_error!(WasmErrorInner::Guest("No artist account found".to_string())))?;

    // Check balance
    if account.pending_balance < amount {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Insufficient pending balance".to_string()
        )));
    }

    let cashout = CashoutRequest {
        artist: my_agent.clone(),
        amount,
        eth_address: account.eth_address.clone(),
        requested_at: sys_time()?,
        status: CashoutStatus::Pending,
        tx_hash: None,
        completed_at: None,
    };

    let action_hash = create_entry(&EntryTypes::CashoutRequest(cashout))?;

    // Link to agent
    let cashouts_path = Path::from(format!("cashouts/{}", my_agent));
    let cashouts_hash = ensure_path(cashouts_path, LinkTypes::AgentToCashouts)?;
    create_link(
        cashouts_hash,
        action_hash.clone(),
        LinkTypes::AgentToCashouts,
        (),
    )?;

    Ok(action_hash)
}

/// Execute transfer from listener to artist (internal, called by plays zome)
#[hdk_extern]
pub fn execute_transfer(input: ExecuteTransferInput) -> ExternResult<ActionHash> {
    require_consciousness(&civic_requirement_voting(), "execute_transfer")?;
    // Create transfer record
    let transfer = Transfer {
        from: input.from.clone(),
        to: input.to.clone(),
        amount: input.amount,
        reason: input.reason,
        reference: input.reference,
        transferred_at: sys_time()?,
    };

    let action_hash = create_entry(&EntryTypes::Transfer(transfer))?;

    // Link to both parties
    let from_path = Path::from(format!("transfers/{}", input.from));
    let from_hash = ensure_path(from_path, LinkTypes::AgentToTransfers)?;
    create_link(
        from_hash,
        action_hash.clone(),
        LinkTypes::AgentToTransfers,
        (),
    )?;

    let to_path = Path::from(format!("transfers/{}", input.to));
    let to_hash = ensure_path(to_path, LinkTypes::AgentToTransfers)?;
    create_link(
        to_hash,
        action_hash.clone(),
        LinkTypes::AgentToTransfers,
        (),
    )?;

    // Debit listener
    update_listener_balance(input.from, -(input.amount as i64))?;

    // Credit artist
    update_artist_balance(input.to, input.amount as i64)?;

    Ok(action_hash)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ExecuteTransferInput {
    pub from: AgentPubKey,
    pub to: AgentPubKey,
    pub amount: u64,
    pub reason: TransferReason,
    pub reference: Option<ActionHash>,
}

/// Update artist balance (internal)
fn update_artist_balance(agent: AgentPubKey, delta: i64) -> ExternResult<()> {
    let account_path = Path::from(format!("artist_account/{}", agent));
    let typed_path = account_path.typed(LinkTypes::AgentToArtistAccount)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::AgentToArtistAccount)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        if let Some(action_hash) = link.target.clone().into_action_hash() {
            if let Some(record) = get(action_hash.clone(), GetOptions::default())? {
                if let Some(mut account) = record
                    .entry()
                    .to_app_option::<ArtistAccount>()
                    .map_err(|e| wasm_error!(e))?
                {
                    // Update balance
                    if delta >= 0 {
                        account.pending_balance += delta as u64;
                        account.total_earned += delta as u64;
                    } else {
                        let abs_delta = (-delta) as u64;
                        if account.pending_balance >= abs_delta {
                            account.pending_balance -= abs_delta;
                            account.total_cashed_out += abs_delta;
                        }
                    }
                    account.updated_at = sys_time()?;

                    // Create updated entry
                    let new_hash =
                        update_entry(action_hash, &EntryTypes::ArtistAccount(account))?;

                    // Update link
                    create_link(
                        typed_path.path_entry_hash()?,
                        new_hash,
                        LinkTypes::AgentToArtistAccount,
                        (),
                    )?;
                }
            }
        }
    }

    Ok(())
}

/// Get my listener account balance
#[hdk_extern]
pub fn get_my_listener_balance(_: ()) -> ExternResult<Option<ListenerAccount>> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    get_listener_account(my_agent)
}

/// Get my artist account balance
#[hdk_extern]
pub fn get_my_artist_balance(_: ()) -> ExternResult<Option<ArtistAccount>> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    get_artist_account(my_agent)
}

/// Get my cashout history
#[hdk_extern]
pub fn get_my_cashouts(_: ()) -> ExternResult<Vec<CashoutRequest>> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    let cashouts_path = Path::from(format!("cashouts/{}", my_agent));
    let typed_path = cashouts_path.typed(LinkTypes::AgentToCashouts)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::AgentToCashouts)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    let mut cashouts = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(cashout) = record
                    .entry()
                    .to_app_option::<CashoutRequest>()
                    .map_err(|e| wasm_error!(e))?
                {
                    cashouts.push(cashout);
                }
            }
        }
    }

    Ok(cashouts)
}

/// Get my transfer history
#[hdk_extern]
pub fn get_my_transfers(_: ()) -> ExternResult<Vec<Transfer>> {
    let my_agent = agent_info()?.agent_initial_pubkey;
    let transfers_path = Path::from(format!("transfers/{}", my_agent));
    let typed_path = transfers_path.typed(LinkTypes::AgentToTransfers)?;
    let filter = LinkTypeFilter::try_from(LinkTypes::AgentToTransfers)?;
    let links = get_links(
        LinkQuery::new(typed_path.path_entry_hash()?, filter),
        GetStrategy::default(),
    )?;

    let mut transfers = Vec::new();
    for link in links {
        if let Some(action_hash) = link.target.into_action_hash() {
            if let Some(record) = get(action_hash, GetOptions::default())? {
                if let Some(transfer) = record
                    .entry()
                    .to_app_option::<Transfer>()
                    .map_err(|e| wasm_error!(e))?
                {
                    transfers.push(transfer);
                }
            }
        }
    }

    Ok(transfers)
}
