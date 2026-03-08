//! Circles Coordinator Zome
//!
//! This zome provides coordinator functions for community credit circles
//! in the Mycelix Mutual Aid hApp. Implements mutual credit with automatic clearing.

use hdk::prelude::*;
use mutualaid_circles_integrity::*;
use mutualaid_common::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_basic, requirement_for_proposal, GovernanceEligibility,
    GovernanceRequirement,
};
use std::collections::HashMap;

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("commons_bridge", requirement, action_name)
}

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

fn get_latest_record(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    let Some(details) = get_details(action_hash, GetOptions::default())? else {
        return Ok(None);
    };
    match details {
        Details::Record(record_details) => {
            if record_details.updates.is_empty() {
                Ok(Some(record_details.record))
            } else {
                let latest_update = &record_details.updates[record_details.updates.len() - 1];
                let latest_hash = latest_update.action_address().clone();
                get_latest_record(latest_hash)
            }
        }
        Details::Entry(_) => Ok(None),
    }
}

#[hdk_extern]
pub fn create_circle(input: CreateCircleInput) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_basic(), "create_circle")?;
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

    get_latest_record(action_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
            if let Some(record) = get_latest_record(hash)? {
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
            if let Some(record) = get_latest_record(hash)? {
                circles.push(record);
            }
        }
    }

    Ok(circles)
}

// =============================================================================
// UPDATE FUNCTIONS
// =============================================================================

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateCreditCircleInput {
    pub original_action_hash: ActionHash,
    pub updated_entry: CreditCircle,
}

/// Update a credit circle entry (general-purpose update replacing the whole entry)
#[hdk_extern]
pub fn update_circle(input: UpdateCreditCircleInput) -> ExternResult<ActionHash> {
    let _eligibility = require_consciousness(&requirement_for_proposal(), "update_circle")?;
    update_entry(
        input.original_action_hash,
        EntryTypes::CreditCircle(input.updated_entry),
    )
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct UpdateCreditLineInput {
    pub original_action_hash: ActionHash,
    pub updated_entry: CreditLine,
}

/// Update a credit line entry (general-purpose update replacing the whole entry)
#[hdk_extern]
pub fn update_credit_line(input: UpdateCreditLineInput) -> ExternResult<ActionHash> {
    let _eligibility = require_consciousness(&requirement_for_proposal(), "update_credit_line")?;
    update_entry(
        input.original_action_hash,
        EntryTypes::CreditLine(input.updated_entry),
    )
}

// =============================================================================
// MEMBERSHIP
// =============================================================================

/// Join a credit circle
#[hdk_extern]
pub fn join_circle(input: JoinCircleInput) -> ExternResult<Record> {
    let _eligibility = require_consciousness(&requirement_for_basic(), "join_circle")?;
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

    get_latest_record(line_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
            if let Some(record) = get_latest_record(hash)? {
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
    let _eligibility = require_consciousness(&requirement_for_basic(), "transfer")?;
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

    get_latest_record(tx_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
            if let Some(record) = get_latest_record(hash)? {
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
            if let Some(record) = get_latest_record(hash)? {
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
            if let Some(record) = get_latest_record(hash)? {
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
                    line.credit_limit + line.balance
                } else {
                    line.credit_limit
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
    let _eligibility = require_consciousness(&requirement_for_proposal(), "run_clearing")?;
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

            if let Some(record) = get_latest_record(tx_hash)? {
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
    let _eligibility = require_consciousness(&requirement_for_proposal(), "adjust_credit_limit")?;
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

    get_latest_record(new_hash)?.ok_or(wasm_error!(WasmErrorInner::Guest(
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

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0xAA; 36])
    }

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xBB; 36])
    }

    fn fake_agent_2() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0xCC; 36])
    }

    // ── Input struct serde roundtrip tests ─────────────────────────────

    #[test]
    fn create_circle_input_serde_roundtrip() {
        let input = CreateCircleInput {
            name: "Community Circle".to_string(),
            description: "A circle for our neighborhood".to_string(),
            currency_name: "CommCoin".to_string(),
            currency_symbol: "CC".to_string(),
            default_credit_limit: 1000,
            max_credit_limit: 5000,
            transaction_fee_percent: 1.5,
            demurrage_rate_percent: 0.5,
            geographic_scope: Some("Downtown".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateCircleInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, "Community Circle");
        assert_eq!(decoded.currency_name, "CommCoin");
        assert_eq!(decoded.currency_symbol, "CC");
        assert_eq!(decoded.default_credit_limit, 1000);
        assert_eq!(decoded.max_credit_limit, 5000);
        assert_eq!(decoded.transaction_fee_percent, 1.5);
        assert_eq!(decoded.demurrage_rate_percent, 0.5);
        assert_eq!(decoded.geographic_scope, Some("Downtown".to_string()));
    }

    #[test]
    fn create_circle_input_no_geographic_scope() {
        let input = CreateCircleInput {
            name: "Global Circle".to_string(),
            description: "Worldwide".to_string(),
            currency_name: "Globe".to_string(),
            currency_symbol: "GL".to_string(),
            default_credit_limit: 500,
            max_credit_limit: 2000,
            transaction_fee_percent: 0.0,
            demurrage_rate_percent: 0.0,
            geographic_scope: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateCircleInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.geographic_scope, None);
        assert_eq!(decoded.transaction_fee_percent, 0.0);
    }

    #[test]
    fn create_circle_input_empty_strings() {
        let input = CreateCircleInput {
            name: String::new(),
            description: String::new(),
            currency_name: String::new(),
            currency_symbol: String::new(),
            default_credit_limit: 0,
            max_credit_limit: 0,
            transaction_fee_percent: 0.0,
            demurrage_rate_percent: 0.0,
            geographic_scope: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateCircleInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.name.is_empty());
        assert!(decoded.currency_name.is_empty());
    }

    #[test]
    fn create_circle_input_unicode_fields() {
        let input = CreateCircleInput {
            name: "\u{1F30D} Global \u{1F91D}".to_string(),
            description: "\u{4E92}\u{52A9}".to_string(),
            currency_name: "\u{00A5}en".to_string(),
            currency_symbol: "\u{00A5}".to_string(),
            default_credit_limit: 100,
            max_credit_limit: 1000,
            transaction_fee_percent: 0.0,
            demurrage_rate_percent: 0.0,
            geographic_scope: Some("\u{6771}\u{4EAC}".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateCircleInput = serde_json::from_str(&json).unwrap();
        assert!(decoded.name.contains('\u{1F30D}'));
        assert_eq!(decoded.currency_symbol, "\u{00A5}");
    }

    #[test]
    fn transfer_input_serde_roundtrip() {
        let input = TransferInput {
            circle_hash: fake_hash(),
            to: fake_agent(),
            amount: 42,
            memo: "For gardening help".to_string(),
            transaction_type: TransactionType::Payment,
            related_exchange_hash: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: TransferInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.amount, 42);
        assert_eq!(decoded.memo, "For gardening help");
        assert_eq!(decoded.related_exchange_hash, None);
    }

    #[test]
    fn transfer_input_with_related_exchange_hash() {
        let related = ActionHash::from_raw_36(vec![0xEE; 36]);
        let input = TransferInput {
            circle_hash: fake_hash(),
            to: fake_agent(),
            amount: 100,
            memo: "exchange".to_string(),
            transaction_type: TransactionType::Payment,
            related_exchange_hash: Some(related.clone()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: TransferInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.related_exchange_hash, Some(related));
    }

    #[test]
    fn transfer_input_zero_amount() {
        let input = TransferInput {
            circle_hash: fake_hash(),
            to: fake_agent(),
            amount: 0,
            memo: String::new(),
            transaction_type: TransactionType::Adjustment,
            related_exchange_hash: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: TransferInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.amount, 0);
        assert!(decoded.memo.is_empty());
    }

    #[test]
    fn transfer_input_negative_amount() {
        let input = TransferInput {
            circle_hash: fake_hash(),
            to: fake_agent(),
            amount: -500,
            memo: "refund".to_string(),
            transaction_type: TransactionType::Adjustment,
            related_exchange_hash: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: TransferInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.amount, -500);
    }

    #[test]
    fn transfer_input_all_transaction_types() {
        let types = vec![
            TransactionType::Payment,
            TransactionType::Repayment,
            TransactionType::Gift,
            TransactionType::Clearing,
            TransactionType::Fee,
            TransactionType::Demurrage,
            TransactionType::Adjustment,
        ];
        for tx_type in types {
            let input = TransferInput {
                circle_hash: fake_hash(),
                to: fake_agent(),
                amount: 10,
                memo: "test".to_string(),
                transaction_type: tx_type.clone(),
                related_exchange_hash: None,
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: TransferInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.transaction_type, tx_type);
        }
    }

    #[test]
    fn adjust_credit_limit_input_serde_roundtrip() {
        let input = AdjustCreditLimitInput {
            circle_hash: fake_hash(),
            member: fake_agent_2(),
            new_limit: 3000,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AdjustCreditLimitInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_limit, 3000);
    }

    #[test]
    fn adjust_credit_limit_input_zero_limit() {
        let input = AdjustCreditLimitInput {
            circle_hash: fake_hash(),
            member: fake_agent(),
            new_limit: 0,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AdjustCreditLimitInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_limit, 0);
    }

    #[test]
    fn adjust_credit_limit_input_max_i64() {
        let input = AdjustCreditLimitInput {
            circle_hash: fake_hash(),
            member: fake_agent(),
            new_limit: i64::MAX,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: AdjustCreditLimitInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_limit, i64::MAX);
    }

    #[test]
    fn clearing_input_serde_roundtrip() {
        let input = ClearingInput {
            circle_hash: fake_hash(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: ClearingInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.circle_hash, fake_hash());
    }

    #[test]
    fn join_circle_input_serde_roundtrip() {
        let circle_hash = ActionHash::from_raw_36(vec![0xDD; 36]);
        let input = JoinCircleInput {
            circle_hash: circle_hash.clone(),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: JoinCircleInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.circle_hash, circle_hash);
    }

    // ── Integrity entry struct serde roundtrip tests ───────────────────

    #[test]
    fn credit_circle_serde_roundtrip() {
        let circle = CreditCircle {
            id: "circle-001".to_string(),
            name: "Test Circle".to_string(),
            description: "A test".to_string(),
            currency_name: "TestCoin".to_string(),
            currency_symbol: "TC".to_string(),
            default_credit_limit: 1000,
            max_credit_limit: 5000,
            transaction_fee_percent: 1.0,
            demurrage_rate_percent: 0.5,
            geographic_scope: Some("Local".to_string()),
            founders: vec![fake_agent()],
            rules_hash: None,
            created_at: Timestamp::from_micros(0),
            active: true,
        };
        let json = serde_json::to_string(&circle).unwrap();
        let decoded: CreditCircle = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "circle-001");
        assert_eq!(decoded.founders.len(), 1);
        assert!(decoded.active);
        assert!(decoded.rules_hash.is_none());
    }

    #[test]
    fn credit_line_serde_roundtrip() {
        let line = CreditLine {
            circle_hash: fake_hash(),
            member: fake_agent(),
            credit_limit: 1000,
            balance: -250,
            total_credit_extended: 500,
            total_credit_received: 750,
            joined_at: Timestamp::from_micros(0),
            status: CreditLineStatus::Active,
            last_activity: Timestamp::from_micros(100),
        };
        let json = serde_json::to_string(&line).unwrap();
        let decoded: CreditLine = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.credit_limit, 1000);
        assert_eq!(decoded.balance, -250);
        assert_eq!(decoded.total_credit_extended, 500);
        assert_eq!(decoded.status, CreditLineStatus::Active);
    }

    #[test]
    fn balance_serde_roundtrip() {
        let balance = Balance {
            member: fake_agent(),
            circle_hash: fake_hash(),
            balance: 42,
            credit_available: 958,
            as_of: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&balance).unwrap();
        let decoded: Balance = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.balance, 42);
        assert_eq!(decoded.credit_available, 958);
    }

    #[test]
    fn balance_negative_balance_serde() {
        let balance = Balance {
            member: fake_agent(),
            circle_hash: fake_hash(),
            balance: -500,
            credit_available: 500,
            as_of: Timestamp::from_micros(0),
        };
        let json = serde_json::to_string(&balance).unwrap();
        let decoded: Balance = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.balance, -500);
    }

    // ── Integrity enum serde roundtrip tests ──────────────────────────

    #[test]
    fn transaction_type_all_variants_serde() {
        let variants = vec![
            TransactionType::Payment,
            TransactionType::Repayment,
            TransactionType::Gift,
            TransactionType::Clearing,
            TransactionType::Fee,
            TransactionType::Demurrage,
            TransactionType::Adjustment,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: TransactionType = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    #[test]
    fn credit_line_status_all_variants_serde() {
        let variants = vec![
            CreditLineStatus::Active,
            CreditLineStatus::Frozen,
            CreditLineStatus::Suspended,
            CreditLineStatus::Closed,
        ];
        for variant in &variants {
            let json = serde_json::to_string(variant).unwrap();
            let decoded: CreditLineStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(&decoded, variant);
        }
    }

    // ── Clone / equality tests ────────────────────────────────────────

    #[test]
    fn transaction_type_clone_eq() {
        let tt = TransactionType::Clearing;
        let cloned = tt.clone();
        assert_eq!(tt, cloned);
    }

    #[test]
    fn credit_line_status_clone_eq() {
        let status = CreditLineStatus::Frozen;
        let cloned = status.clone();
        assert_eq!(status, cloned);
    }

    // ── Edge case tests ───────────────────────────────────────────────

    #[test]
    fn create_circle_input_extreme_fee_values() {
        let input = CreateCircleInput {
            name: "Edge".to_string(),
            description: String::new(),
            currency_name: "E".to_string(),
            currency_symbol: "E".to_string(),
            default_credit_limit: i64::MAX,
            max_credit_limit: i64::MAX,
            transaction_fee_percent: f64::MAX,
            demurrage_rate_percent: f64::MIN_POSITIVE,
            geographic_scope: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: CreateCircleInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.default_credit_limit, i64::MAX);
        assert_eq!(decoded.transaction_fee_percent, f64::MAX);
    }

    #[test]
    fn transfer_input_i64_max_amount() {
        let input = TransferInput {
            circle_hash: fake_hash(),
            to: fake_agent(),
            amount: i64::MAX,
            memo: "max".to_string(),
            transaction_type: TransactionType::Payment,
            related_exchange_hash: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: TransferInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.amount, i64::MAX);
    }

    #[test]
    fn transfer_input_i64_min_amount() {
        let input = TransferInput {
            circle_hash: fake_hash(),
            to: fake_agent(),
            amount: i64::MIN,
            memo: "min".to_string(),
            transaction_type: TransactionType::Adjustment,
            related_exchange_hash: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: TransferInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.amount, i64::MIN);
    }

    // ── UpdateCreditCircleInput tests ────────────────────────────────

    #[test]
    fn update_credit_circle_input_struct_construction() {
        let input = UpdateCreditCircleInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            updated_entry: CreditCircle {
                id: "circle-updated".to_string(),
                name: "Updated Circle".to_string(),
                description: "Updated description".to_string(),
                currency_name: "NewCoin".to_string(),
                currency_symbol: "NC".to_string(),
                default_credit_limit: 2000,
                max_credit_limit: 10000,
                transaction_fee_percent: 2.0,
                demurrage_rate_percent: 1.0,
                geographic_scope: Some("Uptown".to_string()),
                founders: vec![fake_agent()],
                rules_hash: None,
                created_at: Timestamp::from_micros(0),
                active: true,
            },
        };
        assert_eq!(
            input.original_action_hash,
            ActionHash::from_raw_36(vec![0xdb; 36])
        );
        assert_eq!(input.updated_entry.name, "Updated Circle");
        assert_eq!(input.updated_entry.default_credit_limit, 2000);
    }

    #[test]
    fn update_credit_circle_input_serde_roundtrip() {
        let input = UpdateCreditCircleInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xab; 36]),
            updated_entry: CreditCircle {
                id: "circle-serde".to_string(),
                name: "Serde Circle".to_string(),
                description: "Testing serde".to_string(),
                currency_name: "TestCoin".to_string(),
                currency_symbol: "TC".to_string(),
                default_credit_limit: 500,
                max_credit_limit: 5000,
                transaction_fee_percent: 0.0,
                demurrage_rate_percent: 0.0,
                geographic_scope: None,
                founders: vec![fake_agent(), fake_agent_2()],
                rules_hash: Some(ActionHash::from_raw_36(vec![0xDD; 36])),
                created_at: Timestamp::from_micros(100),
                active: false,
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateCreditCircleInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.original_action_hash, input.original_action_hash);
        assert_eq!(decoded.updated_entry.id, "circle-serde");
        assert!(!decoded.updated_entry.active);
        assert!(decoded.updated_entry.rules_hash.is_some());
        assert_eq!(decoded.updated_entry.founders.len(), 2);
    }

    #[test]
    fn update_credit_circle_input_clone_is_equal() {
        let input = UpdateCreditCircleInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xef; 36]),
            updated_entry: CreditCircle {
                id: "circle-clone".to_string(),
                name: "Clone Test".to_string(),
                description: "Cloning".to_string(),
                currency_name: "C".to_string(),
                currency_symbol: "C".to_string(),
                default_credit_limit: 100,
                max_credit_limit: 1000,
                transaction_fee_percent: 0.5,
                demurrage_rate_percent: 0.1,
                geographic_scope: None,
                founders: vec![fake_agent()],
                rules_hash: None,
                created_at: Timestamp::from_micros(0),
                active: true,
            },
        };
        let cloned = input.clone();
        assert_eq!(cloned.original_action_hash, input.original_action_hash);
        assert_eq!(cloned.updated_entry.id, input.updated_entry.id);
        assert_eq!(cloned.updated_entry.name, input.updated_entry.name);
    }

    // ── UpdateCreditLineInput tests ─────────────────────────────────

    #[test]
    fn update_credit_line_input_struct_construction() {
        let input = UpdateCreditLineInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xdb; 36]),
            updated_entry: CreditLine {
                circle_hash: fake_hash(),
                member: fake_agent(),
                credit_limit: 2000,
                balance: 500,
                total_credit_extended: 1000,
                total_credit_received: 1500,
                joined_at: Timestamp::from_micros(0),
                status: CreditLineStatus::Active,
                last_activity: Timestamp::from_micros(200),
            },
        };
        assert_eq!(
            input.original_action_hash,
            ActionHash::from_raw_36(vec![0xdb; 36])
        );
        assert_eq!(input.updated_entry.credit_limit, 2000);
        assert_eq!(input.updated_entry.balance, 500);
    }

    #[test]
    fn update_credit_line_input_serde_roundtrip() {
        let input = UpdateCreditLineInput {
            original_action_hash: ActionHash::from_raw_36(vec![0xcc; 36]),
            updated_entry: CreditLine {
                circle_hash: fake_hash(),
                member: fake_agent_2(),
                credit_limit: 3000,
                balance: -1000,
                total_credit_extended: 2000,
                total_credit_received: 1000,
                joined_at: Timestamp::from_micros(50),
                status: CreditLineStatus::Frozen,
                last_activity: Timestamp::from_micros(300),
            },
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateCreditLineInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.original_action_hash, input.original_action_hash);
        assert_eq!(decoded.updated_entry.balance, -1000);
        assert_eq!(decoded.updated_entry.status, CreditLineStatus::Frozen);
    }

    #[test]
    fn update_credit_line_input_clone_is_equal() {
        let input = UpdateCreditLineInput {
            original_action_hash: ActionHash::from_raw_36(vec![0x22; 36]),
            updated_entry: CreditLine {
                circle_hash: fake_hash(),
                member: fake_agent(),
                credit_limit: 500,
                balance: 0,
                total_credit_extended: 0,
                total_credit_received: 0,
                joined_at: Timestamp::from_micros(0),
                status: CreditLineStatus::Suspended,
                last_activity: Timestamp::from_micros(0),
            },
        };
        let cloned = input.clone();
        assert_eq!(cloned.original_action_hash, input.original_action_hash);
        assert_eq!(
            cloned.updated_entry.credit_limit,
            input.updated_entry.credit_limit
        );
        assert_eq!(cloned.updated_entry.status, input.updated_entry.status);
    }
}
