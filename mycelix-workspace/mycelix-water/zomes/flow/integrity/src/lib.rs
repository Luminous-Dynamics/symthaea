// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Flow Integrity Zome
//! Water allocation, H2O credits, and water economics
//!
//! The FLOW pillar manages water sources, share allocations,
//! credit balances, and transaction records for community water systems.

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

// ============================================================================
// WATER SOURCE
// ============================================================================

/// Type of water source
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum WaterSourceType {
    Municipal,
    Well,
    Spring,
    Rainwater,
    Aquifer,
    River,
    Lake,
    Recycled,
    Desalinated,
}

/// Operational status of a water source
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum SourceStatus {
    Active,
    Seasonal,
    Depleted,
    Contaminated,
    UnderMaintenance,
}

/// A registered water source in the community system
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct WaterSource {
    /// Unique identifier for this source
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Type of water source
    pub source_type: WaterSourceType,
    /// Maximum capacity in liters
    pub max_capacity_liters: u64,
    /// Natural recharge rate in liters per day
    pub recharge_rate_liters_per_day: u64,
    /// GPS latitude
    pub location_lat: f64,
    /// GPS longitude
    pub location_lon: f64,
    /// Agent responsible for this source
    pub steward: AgentPubKey,
    /// Current operational status
    pub status: SourceStatus,
}

// ============================================================================
// WATER SHARES
// ============================================================================

/// How water is allocated from a source
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AllocationType {
    /// Fixed volume per period
    Fixed,
    /// Proportional to total available
    Proportional,
    /// Priority-based during scarcity
    Priority,
    /// Emergency allocation
    Emergency,
}

/// Classification of water use
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum WaterClassification {
    Potable,
    Cooking,
    Hygiene,
    Irrigation,
    Industrial,
    Recreation,
    Greywater,
}

/// A share of water allocated from a source to a holder
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct WaterShare {
    /// Hash of the WaterSource this share draws from
    pub source_hash: ActionHash,
    /// Agent holding this share
    pub holder: AgentPubKey,
    /// How this allocation works
    pub allocation_type: AllocationType,
    /// Volume allocated per period in liters
    pub volume_per_period_liters: u64,
    /// Period length in days
    pub period_days: u32,
    /// Priority level (0 = highest priority)
    pub priority: u8,
    /// What this water is used for
    pub usage_category: WaterClassification,
}

// ============================================================================
// H2O CREDITS
// ============================================================================

/// H2O credit balance for an agent
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct H2OCredit {
    /// Agent holding these credits
    pub holder: AgentPubKey,
    /// Current balance in liters (can go negative for overdraft)
    pub balance_liters: i64,
    /// Total credits ever earned
    pub total_earned: u64,
    /// Total credits ever spent
    pub total_spent: u64,
}

// ============================================================================
// WATER TRANSACTIONS
// ============================================================================

/// Type of water credit transaction
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TransactionType {
    /// Regular allocation from source
    Allocation,
    /// Peer-to-peer transfer
    Transfer,
    /// Purchase of credits
    Purchase,
    /// Donation of credits
    Donation,
    /// Emergency allocation
    Emergency,
}

/// A record of water credit movement between agents
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct WaterTransaction {
    /// Sending agent
    pub from_agent: AgentPubKey,
    /// Receiving agent
    pub to_agent: AgentPubKey,
    /// Volume in liters
    pub liters: u64,
    /// Type of transaction
    pub credit_type: TransactionType,
    /// When this transaction occurred
    pub timestamp: Timestamp,
    /// Optional link to water source
    pub source_hash: Option<ActionHash>,
}

// ============================================================================
// USAGE RECORD
// ============================================================================

/// Record of actual water usage against an allocation
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct UsageRecord {
    /// Agent who used the water
    pub agent: AgentPubKey,
    /// Source from which water was drawn
    pub source_hash: ActionHash,
    /// Liters consumed
    pub liters_used: u64,
    /// What the water was used for
    pub usage_category: WaterClassification,
    /// When usage was recorded
    pub recorded_at: Timestamp,
    /// Optional meter reading or sensor data reference
    pub meter_reference: Option<String>,
}

// ============================================================================
// ENTRY & LINK TYPE REGISTRATION
// ============================================================================

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    WaterSource(WaterSource),
    WaterShare(WaterShare),
    H2OCredit(H2OCredit),
    WaterTransaction(WaterTransaction),
    UsageRecord(UsageRecord),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Anchor to all water sources
    AllSources,
    /// Source type anchor to sources of that type
    SourceTypeToSource,
    /// Steward agent to their sources
    StewardToSource,
    /// Source to its allocated shares
    SourceToShare,
    /// Holder agent to their shares
    HolderToShare,
    /// Agent to their credit balance
    AgentToCredit,
    /// Agent to their transactions (sent)
    AgentToTransactionSent,
    /// Agent to their transactions (received)
    AgentToTransactionReceived,
    /// Source to usage records
    SourceToUsage,
    /// Agent to their usage records
    AgentToUsage,
}

// ============================================================================
// VALIDATION
// ============================================================================

#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::WaterSource(source) => validate_create_water_source(action, source),
                EntryTypes::WaterShare(share) => validate_create_water_share(action, share),
                EntryTypes::H2OCredit(credit) => validate_create_h2o_credit(action, credit),
                EntryTypes::WaterTransaction(tx) => validate_create_water_transaction(action, tx),
                EntryTypes::UsageRecord(usage) => validate_create_usage_record(action, usage),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::WaterSource(source) => {
                    validate_update_water_source(action, source, original_action_hash)
                }
                EntryTypes::H2OCredit(credit) => {
                    validate_update_h2o_credit(action, credit, original_action_hash)
                }
                _ => Ok(ValidateCallbackResult::Valid),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::AllSources => Ok(ValidateCallbackResult::Valid),
            LinkTypes::SourceTypeToSource => Ok(ValidateCallbackResult::Valid),
            LinkTypes::StewardToSource => Ok(ValidateCallbackResult::Valid),
            LinkTypes::SourceToShare => Ok(ValidateCallbackResult::Valid),
            LinkTypes::HolderToShare => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToCredit => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToTransactionSent => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToTransactionReceived => Ok(ValidateCallbackResult::Valid),
            LinkTypes::SourceToUsage => Ok(ValidateCallbackResult::Valid),
            LinkTypes::AgentToUsage => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type: _,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => Ok(ValidateCallbackResult::Valid),
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

fn validate_create_water_source(
    _action: Create,
    source: WaterSource,
) -> ExternResult<ValidateCallbackResult> {
    if source.id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Water source ID cannot be empty".into(),
        ));
    }
    if source.name.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Water source name cannot be empty".into(),
        ));
    }
    if source.max_capacity_liters == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Max capacity must be greater than zero".into(),
        ));
    }
    if source.location_lat < -90.0 || source.location_lat > 90.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Latitude must be between -90 and 90".into(),
        ));
    }
    if source.location_lon < -180.0 || source.location_lon > 180.0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Longitude must be between -180 and 180".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_water_source(
    _action: Update,
    source: WaterSource,
    original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash)?;
    let original_source: WaterSource = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original water source not found".into()
        )))?;

    if source.id != original_source.id {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change water source ID".into(),
        ));
    }
    if source.steward != original_source.steward {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change water source steward via update; use transfer".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_water_share(
    _action: Create,
    share: WaterShare,
) -> ExternResult<ValidateCallbackResult> {
    if share.volume_per_period_liters == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Share volume must be greater than zero".into(),
        ));
    }
    if share.period_days == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Period days must be greater than zero".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_h2o_credit(
    _action: Create,
    _credit: H2OCredit,
) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

fn validate_update_h2o_credit(
    _action: Update,
    _credit: H2OCredit,
    _original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_water_transaction(
    _action: Create,
    tx: WaterTransaction,
) -> ExternResult<ValidateCallbackResult> {
    if tx.liters == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Transaction volume must be greater than zero".into(),
        ));
    }
    if tx.from_agent == tx.to_agent {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot transfer credits to self".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_usage_record(
    _action: Create,
    usage: UsageRecord,
) -> ExternResult<ValidateCallbackResult> {
    if usage.liters_used == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Usage liters must be greater than zero".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}
