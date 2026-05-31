// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Flow Coordinator Zome
//! Business logic for water allocation, H2O credits, and transactions

use flow_integrity::*;
use hdk::prelude::*;

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Helper to collect records from links
fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            records.push(record);
        }
    }
    Ok(records)
}

// ============================================================================
// WATER SOURCE MANAGEMENT
// ============================================================================

/// Register a new water source
#[hdk_extern]
pub fn register_source(source: WaterSource) -> ExternResult<Record> {
    if source.id.is_empty() || source.id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Source ID must be 1-256 characters".into()
        )));
    }
    if source.name.is_empty() || source.name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Source name must be 1-256 characters".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::WaterSource(source.clone()))?;

    // Link to all sources anchor
    create_entry(&EntryTypes::Anchor(Anchor("all_sources".to_string())))?;
    create_link(
        anchor_hash("all_sources")?,
        action_hash.clone(),
        LinkTypes::AllSources,
        (),
    )?;

    // Link to source type anchor
    let type_anchor = format!("source_type:{:?}", source.source_type);
    create_entry(&EntryTypes::Anchor(Anchor(type_anchor.clone())))?;
    create_link(
        anchor_hash(&type_anchor)?,
        action_hash.clone(),
        LinkTypes::SourceTypeToSource,
        (),
    )?;

    // Link steward to source
    create_link(
        source.steward,
        action_hash.clone(),
        LinkTypes::StewardToSource,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created water source".into()
    )))
}

/// Get a water source record by action hash
#[hdk_extern]
pub fn get_source(action_hash: ActionHash) -> ExternResult<Option<Record>> {
    get(action_hash, GetOptions::default())
}

/// Get all registered water sources
#[hdk_extern]
pub fn get_all_sources(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_sources")?, LinkTypes::AllSources)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

/// Get the current status of a water source
#[hdk_extern]
pub fn get_source_status(action_hash: ActionHash) -> ExternResult<SourceStatus> {
    let record = get(action_hash, GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Source not found".into())
    ))?;
    let source: WaterSource = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid source entry".into()
        )))?;
    Ok(source.status)
}

/// Update source status (steward only)
#[hdk_extern]
pub fn update_source_status(input: UpdateSourceStatusInput) -> ExternResult<Record> {
    let agent_info = agent_info()?;
    let record = get(input.source_hash.clone(), GetOptions::default())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Source not found".into())
    ))?;
    let mut source: WaterSource = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid source entry".into()
        )))?;

    if source.steward != agent_info.agent_initial_pubkey {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the steward can update source status".into()
        )));
    }

    source.status = input.new_status;
    let new_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::WaterSource(source),
    )?;

    get(new_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated source".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateSourceStatusInput {
    pub source_hash: ActionHash,
    pub new_status: SourceStatus,
}

// ============================================================================
// SHARE ALLOCATION
// ============================================================================

/// Allocate a water share from a source to a holder
#[hdk_extern]
pub fn allocate_shares(share: WaterShare) -> ExternResult<Record> {
    let agent_info = agent_info()?;

    // Verify caller is steward of the source
    let source_record = get(share.source_hash.clone(), GetOptions::default())?.ok_or(
        wasm_error!(WasmErrorInner::Guest("Water source not found".into())),
    )?;
    let source: WaterSource = source_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid source entry".into()
        )))?;

    if source.steward != agent_info.agent_initial_pubkey {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the source steward can allocate shares".into()
        )));
    }

    if source.status == SourceStatus::Depleted || source.status == SourceStatus::Contaminated {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot allocate from a depleted or contaminated source".into()
        )));
    }

    let action_hash = create_entry(&EntryTypes::WaterShare(share.clone()))?;

    // Link source to share
    create_link(
        share.source_hash.clone(),
        action_hash.clone(),
        LinkTypes::SourceToShare,
        (),
    )?;

    // Link holder to share
    create_link(
        share.holder.clone(),
        action_hash.clone(),
        LinkTypes::HolderToShare,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created share".into()
    )))
}

/// Get all allocations for a given water source
#[hdk_extern]
pub fn get_source_allocations(source_hash: ActionHash) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(source_hash, LinkTypes::SourceToShare)?,
        GetStrategy::default(),
    )?;
    records_from_links(links)
}

// ============================================================================
// H2O CREDIT MANAGEMENT
// ============================================================================

/// Get or initialize H2O credit balance for the calling agent
#[hdk_extern]
pub fn get_my_balance(_: ()) -> ExternResult<H2OCredit> {
    let agent_info = agent_info()?;
    let agent_key = agent_info.agent_initial_pubkey.clone();

    let links = get_links(
        LinkQuery::try_new(agent_key.clone(), LinkTypes::AgentToCredit)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.into_iter().last() {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        let record = get(action_hash, GetOptions::default())?.ok_or(wasm_error!(
            WasmErrorInner::Guest("Credit record not found".into())
        ))?;
        let credit: H2OCredit = record
            .entry()
            .to_app_option()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Invalid credit entry".into()
            )))?;
        Ok(credit)
    } else {
        // Initialize zero balance
        let credit = H2OCredit {
            holder: agent_key.clone(),
            balance_liters: 0,
            total_earned: 0,
            total_spent: 0,
        };
        let action_hash = create_entry(&EntryTypes::H2OCredit(credit.clone()))?;
        create_link(agent_key, action_hash, LinkTypes::AgentToCredit, ())?;
        Ok(credit)
    }
}

/// Transfer H2O credits to another agent
#[hdk_extern]
pub fn transfer_credits(input: TransferCreditsInput) -> ExternResult<Record> {
    let agent_info = agent_info()?;
    let from_agent = agent_info.agent_initial_pubkey.clone();

    if from_agent == input.to_agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot transfer credits to yourself".into()
        )));
    }
    if input.liters == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Transfer amount must be greater than zero".into()
        )));
    }

    // Get sender balance
    let sender_balance = get_my_balance(())?;
    if sender_balance.balance_liters < input.liters as i64 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Insufficient H2O credit balance".into()
        )));
    }

    let now = sys_time()?;

    // Create transaction record
    let tx = WaterTransaction {
        from_agent: from_agent.clone(),
        to_agent: input.to_agent.clone(),
        liters: input.liters,
        credit_type: input.transaction_type,
        timestamp: now,
        source_hash: input.source_hash,
    };
    let tx_hash = create_entry(&EntryTypes::WaterTransaction(tx))?;

    // Link transaction to sender and receiver
    create_link(
        from_agent.clone(),
        tx_hash.clone(),
        LinkTypes::AgentToTransactionSent,
        (),
    )?;
    create_link(
        input.to_agent.clone(),
        tx_hash.clone(),
        LinkTypes::AgentToTransactionReceived,
        (),
    )?;

    // Update sender balance
    let updated_sender = H2OCredit {
        holder: from_agent.clone(),
        balance_liters: sender_balance.balance_liters - input.liters as i64,
        total_earned: sender_balance.total_earned,
        total_spent: sender_balance.total_spent + input.liters,
    };
    update_agent_credit(&from_agent, updated_sender)?;

    get(tx_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created transaction".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TransferCreditsInput {
    pub to_agent: AgentPubKey,
    pub liters: u64,
    pub transaction_type: TransactionType,
    pub source_hash: Option<ActionHash>,
}

/// Record water usage against a source
#[hdk_extern]
pub fn record_usage(input: RecordUsageInput) -> ExternResult<Record> {
    let agent_info = agent_info()?;
    let agent_key = agent_info.agent_initial_pubkey.clone();
    let now = sys_time()?;

    let usage = UsageRecord {
        agent: agent_key.clone(),
        source_hash: input.source_hash.clone(),
        liters_used: input.liters_used,
        usage_category: input.usage_category,
        recorded_at: now,
        meter_reference: input.meter_reference,
    };

    let action_hash = create_entry(&EntryTypes::UsageRecord(usage))?;

    // Link source to usage
    create_link(
        input.source_hash,
        action_hash.clone(),
        LinkTypes::SourceToUsage,
        (),
    )?;

    // Link agent to usage
    create_link(
        agent_key.clone(),
        action_hash.clone(),
        LinkTypes::AgentToUsage,
        (),
    )?;

    // Debit from agent credit balance
    let balance = get_my_balance(())?;
    let updated = H2OCredit {
        holder: agent_key.clone(),
        balance_liters: balance.balance_liters - input.liters_used as i64,
        total_earned: balance.total_earned,
        total_spent: balance.total_spent + input.liters_used,
    };
    update_agent_credit(&agent_key, updated)?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find created usage record".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordUsageInput {
    pub source_hash: ActionHash,
    pub liters_used: u64,
    pub usage_category: WaterClassification,
    pub meter_reference: Option<String>,
}

// ============================================================================
// INTERNAL HELPERS
// ============================================================================

/// Update an agent's credit balance (creates new entry, relinks)
fn update_agent_credit(agent: &AgentPubKey, new_credit: H2OCredit) -> ExternResult<ActionHash> {
    // Find existing credit link
    let links = get_links(
        LinkQuery::try_new(agent.clone(), LinkTypes::AgentToCredit)?,
        GetStrategy::default(),
    )?;

    let action_hash = if let Some(link) = links.into_iter().last() {
        let old_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        update_entry(old_hash, &EntryTypes::H2OCredit(new_credit))?
    } else {
        let h = create_entry(&EntryTypes::H2OCredit(new_credit))?;
        create_link(agent.clone(), h.clone(), LinkTypes::AgentToCredit, ())?;
        h
    };

    Ok(action_hash)
}
