// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Circles Coordinator Zome
//!
//! This zome provides coordinator functions for community credit circles
//! in the Mycelix Mutual Aid hApp. Implements mutual credit with automatic clearing.

use circles_integrity::*;
use hdk::prelude::*;
use mutualaid_common::*;
use std::collections::HashMap;

// =============================================================================
// INPUT TYPES
// =============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct CreateCircleInput {
    pub name: String,
    pub description: String,
    pub currency_name: String,
    pub currency_symbol: String,
    pub default_credit_limit: i64,
    pub max_credit_limit: i64,
    pub transaction_fee_percent: f64,
    pub demurrage_rate_percent: f64,
    pub geographic_scope: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct JoinCircleInput {
    pub circle_hash: ActionHash,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TransferInput {
    pub circle_hash: ActionHash,
    pub to: AgentPubKey,
    pub amount: i64,
    pub memo: String,
    pub transaction_type: TransactionType,
    pub related_exchange_hash: Option<ActionHash>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AdjustCreditLimitInput {
    pub circle_hash: ActionHash,
    pub member: AgentPubKey,
    pub new_limit: i64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ClearingInput {
    pub circle_hash: ActionHash,
}

// =============================================================================
// CIRCLE MANAGEMENT
// =============================================================================

/// Create a new credit circle
#[hdk_extern]
pub fn create_circle(input: CreateCircleInput) -> ExternResult<Record> {
    let founder = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let circle = CreditCircle {
        id: generate_id("circle"),
        name: input.name,
        description: input.description,
        currency_name: input.currency_name,
        currency_symbol: input.currency_symbol,
        default_credit_limit: input.default_credit_limit,
        max_credit_limit: input.max_credit_limit,
        transaction_fee_percent: input.transaction_fee_percent,
        demurrage_rate_percent: input.demurrage_rate_percent,
        geographic_scope: input.geographic_scope,
        founders: vec![founder.clone()],
        rules_hash: None,
        created_at: Timestamp::from_micros(now.as_micros() as i64),
        active: true,
    };

    let action_hash = create_entry(EntryTypes::CreditCircle(circle.clone()))?;

    // Create founder's credit line
    let credit_line = CreditLine {
        circle_hash: action_hash.clone(),
        member: founder.clone(),
        credit_limit: input.default_credit_limit,
        balance: 0,
        total_credit_extended: 0,
        total_credit_received: 0,
        joined_at: Timestamp::from_micros(now.as_micros() as i64),
        status: CreditLineStatus::Active,
        last_activity: Timestamp::from_micros(now.as_micros() as i64),
    };

    let line_hash = create_entry(EntryTypes::CreditLine(credit_line))?;

    // Create links
    create_link(
        action_hash.clone(),
        founder.clone(),
        LinkTypes::CircleToMembers,
        (),
    )?;
    create_link(
        founder.clone(),
        action_hash.clone(),
        LinkTypes::MemberToCircles,
        (),
    )?;
    create_link(founder, line_hash, LinkTypes::MemberToCreditLines, ())?;

    // Link to all circles
    let all_anchor = all_circles_anchor()?;
    create_link(all_anchor, action_hash.clone(), LinkTypes::AllCircles, ())?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created circle".to_string()
    )))
}

/// Get a circle by hash
#[hdk_extern]
pub fn get_circle(hash: ActionHash) -> ExternResult<Option<Record>> {
    get(hash, GetOptions::default())
}

/// Get all circles
#[hdk_extern]
pub fn get_all_circles(_: ()) -> ExternResult<Vec<Record>> {
    let anchor = all_circles_anchor()?;
    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AllCircles)?,
        GetStrategy::default(),
    )?;

    let mut circles = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                circles.push(record);
            }
        }
    }

    Ok(circles)
}

/// Get circles I'm a member of
#[hdk_extern]
pub fn get_my_circles(_: ()) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::MemberToCircles)?,
        GetStrategy::default(),
    )?;

    let mut circles = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                circles.push(record);
            }
        }
    }

    Ok(circles)
}

// =============================================================================
// MEMBERSHIP
// =============================================================================

/// Join a credit circle
#[hdk_extern]
pub fn join_circle(input: JoinCircleInput) -> ExternResult<Record> {
    let member = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // Get the circle to check default credit limit
    let circle_record = get(input.circle_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Circle not found".to_string())),
    )?;

    let circle: CreditCircle = circle_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse circle".to_string()
        )))?;

    if !circle.active {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot join inactive circle".to_string()
        )));
    }

    // Check if already a member
    let existing_lines = get_my_credit_line_for_circle(input.circle_hash.clone())?;
    if existing_lines.is_some() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Already a member of this circle".to_string()
        )));
    }

    // Create credit line for new member
    let credit_line = CreditLine {
        circle_hash: input.circle_hash.clone(),
        member: member.clone(),
        credit_limit: circle.default_credit_limit,
        balance: 0,
        total_credit_extended: 0,
        total_credit_received: 0,
        joined_at: Timestamp::from_micros(now.as_micros() as i64),
        status: CreditLineStatus::Active,
        last_activity: Timestamp::from_micros(now.as_micros() as i64),
    };

    let line_hash = create_entry(EntryTypes::CreditLine(credit_line))?;

    // Create links
    create_link(
        input.circle_hash.clone(),
        member.clone(),
        LinkTypes::CircleToMembers,
        (),
    )?;
    create_link(
        member.clone(),
        input.circle_hash.clone(),
        LinkTypes::MemberToCircles,
        (),
    )?;
    create_link(
        member,
        line_hash.clone(),
        LinkTypes::MemberToCreditLines,
        (),
    )?;

    get(line_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created credit line".to_string()
    )))
}

/// Get my credit line for a specific circle
#[hdk_extern]
pub fn get_my_credit_line_for_circle(circle_hash: ActionHash) -> ExternResult<Option<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::MemberToCreditLines)?,
        GetStrategy::default(),
    )?;

    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(line) = record.entry().to_app_option::<CreditLine>().ok().flatten() {
                    if line.circle_hash == circle_hash {
                        return Ok(Some(record));
                    }
                }
            }
        }
    }

    Ok(None)
}

/// Get all members of a circle
#[hdk_extern]
pub fn get_circle_members(circle_hash: ActionHash) -> ExternResult<Vec<AgentPubKey>> {
    let links = get_links(
        LinkQuery::try_new(circle_hash, LinkTypes::CircleToMembers)?,
        GetStrategy::default(),
    )?;

    let mut members = Vec::new();
    for link in links {
        if let Some(agent) = link.target.into_agent_pub_key() {
            members.push(agent);
        }
    }

    Ok(members)
}

// =============================================================================
// TRANSACTIONS
// =============================================================================

/// Transfer credits to another member
#[hdk_extern]
pub fn transfer(input: TransferInput) -> ExternResult<Record> {
    let from = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    // Get sender's credit line
    let from_line_record =
        get_my_credit_line_for_circle(input.circle_hash.clone())?.ok_or(wasm_error!(
            WasmErrorInner::Guest("You are not a member of this circle".to_string())
        ))?;

    let mut from_line: CreditLine = from_line_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse credit line".to_string()
        )))?;

    // Get recipient's credit line
    let to_line_record = get_credit_line_for_member(input.circle_hash.clone(), input.to.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
        "Recipient is not a member of this circle".to_string()
    )))?;

    let mut to_line: CreditLine = to_line_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse credit line".to_string()
        )))?;

    // Check if sender has enough credit
    let new_from_balance = from_line.balance - input.amount;
    if new_from_balance < -(from_line.credit_limit as i64) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Insufficient credit".to_string()
        )));
    }

    // Update balances
    from_line.balance = new_from_balance;
    from_line.total_credit_extended += input.amount as u64;
    from_line.last_activity = Timestamp::from_micros(now.as_micros() as i64);

    to_line.balance += input.amount;
    to_line.total_credit_received += input.amount as u64;
    to_line.last_activity = Timestamp::from_micros(now.as_micros() as i64);

    // Create transaction record
    let tx = CreditTransaction {
        id: generate_id("tx"),
        circle_hash: input.circle_hash.clone(),
        from: from.clone(),
        to: input.to.clone(),
        amount: input.amount,
        transaction_type: input.transaction_type,
        memo: input.memo,
        related_exchange_hash: input.related_exchange_hash,
        created_at: Timestamp::from_micros(now.as_micros() as i64),
        confirmed: true,
    };

    let tx_hash = create_entry(EntryTypes::CreditTransaction(tx))?;

    // Update credit lines
    let from_line_hash = from_line_record.action_address().clone();
    update_entry(from_line_hash, EntryTypes::CreditLine(from_line))?;

    let to_line_hash = to_line_record.action_address().clone();
    update_entry(to_line_hash, EntryTypes::CreditLine(to_line))?;

    // Create links
    create_link(
        input.circle_hash,
        tx_hash.clone(),
        LinkTypes::CircleToTransactions,
        (),
    )?;
    create_link(from, tx_hash.clone(), LinkTypes::MemberToTransactions, ())?;
    create_link(
        input.to,
        tx_hash.clone(),
        LinkTypes::MemberToTransactions,
        (),
    )?;

    get(tx_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve created transaction".to_string()
    )))
}

/// Get credit line for a specific member in a circle
fn get_credit_line_for_member(
    circle_hash: ActionHash,
    member: AgentPubKey,
) -> ExternResult<Option<Record>> {
    let links = get_links(
        LinkQuery::try_new(member, LinkTypes::MemberToCreditLines)?,
        GetStrategy::default(),
    )?;

    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(line) = record.entry().to_app_option::<CreditLine>().ok().flatten() {
                    if line.circle_hash == circle_hash {
                        return Ok(Some(record));
                    }
                }
            }
        }
    }

    Ok(None)
}

/// Get my transactions in a circle
#[hdk_extern]
pub fn get_my_transactions(circle_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let agent = agent_info()?.agent_initial_pubkey;
    let links = get_links(
        LinkQuery::try_new(agent, LinkTypes::MemberToTransactions)?,
        GetStrategy::default(),
    )?;

    let mut transactions = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                if let Some(tx) = record
                    .entry()
                    .to_app_option::<CreditTransaction>()
                    .ok()
                    .flatten()
                {
                    if tx.circle_hash == circle_hash {
                        transactions.push(record);
                    }
                }
            }
        }
    }

    Ok(transactions)
}

/// Get all transactions in a circle
#[hdk_extern]
pub fn get_circle_transactions(circle_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(circle_hash, LinkTypes::CircleToTransactions)?,
        GetStrategy::default(),
    )?;

    let mut transactions = Vec::new();
    for link in links {
        if let Some(hash) = link.target.into_action_hash() {
            if let Some(record) = get(hash, GetOptions::default())? {
                transactions.push(record);
            }
        }
    }

    Ok(transactions)
}

// =============================================================================
// BALANCE QUERIES
// =============================================================================

/// Get my balance in a specific circle
#[hdk_extern]
pub fn get_my_balance_in_circle(circle_hash: ActionHash) -> ExternResult<Balance> {
    let agent = agent_info()?.agent_initial_pubkey;
    let line_record = get_my_credit_line_for_circle(circle_hash.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("You are not a member of this circle".to_string())
    ))?;

    let line: CreditLine = line_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse credit line".to_string()
        )))?;

    let credit_available = if line.balance < 0 {
        line.credit_limit as i64 + line.balance
    } else {
        line.credit_limit as i64
    };

    Ok(Balance {
        member: agent,
        circle_hash,
        balance: line.balance,
        credit_available,
        as_of: line.last_activity,
    })
}

/// Get all balances in a circle
#[hdk_extern]
pub fn get_circle_balances(circle_hash: ActionHash) -> ExternResult<Vec<Balance>> {
    let members = get_circle_members(circle_hash.clone())?;
    let now = sys_time()?;

    let mut balances = Vec::new();
    for member in members {
        if let Some(line_record) = get_credit_line_for_member(circle_hash.clone(), member.clone())?
        {
            if let Some(line) = line_record
                .entry()
                .to_app_option::<CreditLine>()
                .ok()
                .flatten()
            {
                let credit_available = if line.balance < 0 {
                    line.credit_limit as i64 + line.balance
                } else {
                    line.credit_limit as i64
                };

                balances.push(Balance {
                    member,
                    circle_hash: circle_hash.clone(),
                    balance: line.balance,
                    credit_available,
                    as_of: Timestamp::from_micros(now.as_micros() as i64),
                });
            }
        }
    }

    Ok(balances)
}

// =============================================================================
// AUTOMATIC CLEARING
// =============================================================================

/// Run automatic credit clearing for a circle
/// This finds cycles of debt and clears them automatically
#[hdk_extern]
pub fn run_clearing(input: ClearingInput) -> ExternResult<Vec<Record>> {
    let now = sys_time()?;
    let members = get_circle_members(input.circle_hash.clone())?;

    // Build a map of balances
    let mut balances: HashMap<AgentPubKey, i64> = HashMap::new();
    for member in &members {
        if let Some(line_record) =
            get_credit_line_for_member(input.circle_hash.clone(), member.clone())?
        {
            if let Some(line) = line_record
                .entry()
                .to_app_option::<CreditLine>()
                .ok()
                .flatten()
            {
                balances.insert(member.clone(), line.balance);
            }
        }
    }

    // Find members with negative balances (debtors) and positive balances (creditors)
    let mut debtors: Vec<(AgentPubKey, i64)> = Vec::new();
    let mut creditors: Vec<(AgentPubKey, i64)> = Vec::new();

    for (member, balance) in &balances {
        if *balance < 0 {
            debtors.push((member.clone(), balance.abs()));
        } else if *balance > 0 {
            creditors.push((member.clone(), *balance));
        }
    }

    // Simple clearing: match debtors with creditors
    let mut clearing_transactions = Vec::new();

    for (debtor, mut debt) in debtors {
        for (creditor, credit) in creditors.iter_mut() {
            if debt == 0 || *credit == 0 {
                continue;
            }

            let clear_amount = debt.min(*credit);
            debt -= clear_amount;
            *credit -= clear_amount;

            // Create clearing transaction
            let tx = CreditTransaction {
                id: generate_id("clearing"),
                circle_hash: input.circle_hash.clone(),
                from: debtor.clone(),
                to: creditor.clone(),
                amount: clear_amount,
                transaction_type: TransactionType::Clearing,
                memo: "Automatic credit clearing".to_string(),
                related_exchange_hash: None,
                created_at: Timestamp::from_micros(now.as_micros() as i64),
                confirmed: true,
            };

            let tx_hash = create_entry(EntryTypes::CreditTransaction(tx))?;
            create_link(
                input.circle_hash.clone(),
                tx_hash.clone(),
                LinkTypes::CircleToTransactions,
                (),
            )?;

            if let Some(record) = get(tx_hash, GetOptions::default())? {
                clearing_transactions.push(record);
            }

            // Update credit lines
            if let Some(debtor_line_record) =
                get_credit_line_for_member(input.circle_hash.clone(), debtor.clone())?
            {
                if let Some(mut line) = debtor_line_record
                    .entry()
                    .to_app_option::<CreditLine>()
                    .ok()
                    .flatten()
                {
                    line.balance += clear_amount;
                    line.last_activity = Timestamp::from_micros(now.as_micros() as i64);
                    let line_hash = debtor_line_record.action_address().clone();
                    update_entry(line_hash, EntryTypes::CreditLine(line))?;
                }
            }

            if let Some(creditor_line_record) =
                get_credit_line_for_member(input.circle_hash.clone(), creditor.clone())?
            {
                if let Some(mut line) = creditor_line_record
                    .entry()
                    .to_app_option::<CreditLine>()
                    .ok()
                    .flatten()
                {
                    line.balance -= clear_amount;
                    line.last_activity = Timestamp::from_micros(now.as_micros() as i64);
                    let line_hash = creditor_line_record.action_address().clone();
                    update_entry(line_hash, EntryTypes::CreditLine(line))?;
                }
            }
        }
    }

    Ok(clearing_transactions)
}

// =============================================================================
// CREDIT LIMIT MANAGEMENT
// =============================================================================

/// Adjust a member's credit limit (requires governance approval in production)
#[hdk_extern]
pub fn adjust_credit_limit(input: AdjustCreditLimitInput) -> ExternResult<Record> {
    let now = sys_time()?;

    // Get circle to check max limit
    let circle_record = get(input.circle_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Circle not found".to_string())),
    )?;

    let circle: CreditCircle = circle_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse circle".to_string()
        )))?;

    if input.new_limit > circle.max_credit_limit {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "New limit exceeds maximum allowed".to_string()
        )));
    }

    // Get member's credit line
    let line_record = get_credit_line_for_member(input.circle_hash.clone(), input.member.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Member not found in this circle".to_string()
        )))?;

    let mut line: CreditLine = line_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not parse credit line".to_string()
        )))?;

    // Check if new limit would put member over limit
    if line.balance < 0 && line.balance.abs() > input.new_limit {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot reduce limit below current debt".to_string()
        )));
    }

    line.credit_limit = input.new_limit;
    line.last_activity = Timestamp::from_micros(now.as_micros() as i64);

    let line_hash = line_record.action_address().clone();
    let new_hash = update_entry(line_hash, EntryTypes::CreditLine(line))?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not retrieve updated credit line".to_string()
    )))
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Generate a unique ID
fn generate_id(prefix: &str) -> String {
    let now = sys_time().unwrap_or(Timestamp::from_micros(0));
    let agent = agent_info()
        .map(|info| info.agent_initial_pubkey.to_string())
        .unwrap_or_default();
    format!(
        "{}_{}_{}",
        prefix,
        now.as_micros(),
        &agent[..8.min(agent.len())]
    )
}

/// Anchor for all circles
fn all_circles_anchor() -> ExternResult<EntryHash> {
    let anchor_bytes = SerializedBytes::from(UnsafeBytes::from(
        "circles_anchor:all_circles".as_bytes().to_vec(),
    ));
    hash_entry(Entry::App(AppEntryBytes(anchor_bytes)))
}
