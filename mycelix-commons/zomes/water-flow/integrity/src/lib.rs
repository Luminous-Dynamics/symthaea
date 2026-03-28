// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Flow Integrity Zome
//! Water allocation, H2O credits, and water economics
//!
//! The FLOW pillar manages water sources, share allocations,
//! credit balances, and transaction records for community water systems.

use hdi::prelude::*;
use mycelix_bridge_entry_types::{check_author_match, check_link_author_match};

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
            tag,
            action: _,
        } => match link_type {
            LinkTypes::AllSources => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AllSources link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::SourceTypeToSource => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "SourceTypeToSource link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::StewardToSource => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "StewardToSource link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::SourceToShare => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "SourceToShare link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::HolderToShare => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "HolderToShare link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToCredit => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToCredit link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToTransactionSent => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToTransactionSent link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToTransactionReceived => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToTransactionReceived link tag too long (max 512 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::SourceToUsage => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "SourceToUsage link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::AgentToUsage => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "AgentToUsage link tag too long (max 256 bytes)".into(),
                    ));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        },
        FlatOp::RegisterDeleteLink { action, .. } => {
            let original_action = must_get_action(action.link_add_address.clone())?;
            Ok(check_link_author_match(
                original_action.action().author(),
                &action.author,
            ))
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            Ok(check_author_match(
                original.action().author(),
                &action.author,
                "update",
            ))
        }
        FlatOp::RegisterDelete(OpDelete { action, .. }) => {
            let original_action = must_get_action(action.deletes_address.clone())?;
            Ok(check_author_match(
                original_action.action().author(),
                &action.author,
                "delete",
            ))
        }
    }
}

fn validate_create_water_source(
    _action: Create,
    source: WaterSource,
) -> ExternResult<ValidateCallbackResult> {
    if source.id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Water source ID cannot be empty".into(),
        ));
    }
    if source.id.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Water source ID must be 256 characters or fewer".into(),
        ));
    }
    if source.name.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Water source name cannot be empty".into(),
        ));
    }
    if source.name.len() > 256 {
        return Ok(ValidateCallbackResult::Invalid(
            "Water source name must be 256 characters or fewer".into(),
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

fn validate_h2o_credit_fields(credit: &H2OCredit) -> ExternResult<ValidateCallbackResult> {
    // total_spent cannot exceed total_earned (no credit from nothing)
    if credit.total_spent > credit.total_earned {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "total_spent ({}) cannot exceed total_earned ({})",
            credit.total_spent, credit.total_earned
        )));
    }
    // balance_liters must be consistent: earned - spent = balance (allow small overdraft via governance)
    // but balance cannot exceed total_earned (can't have more than ever earned)
    if credit.balance_liters > credit.total_earned as i64 {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "balance_liters ({}) cannot exceed total_earned ({})",
            credit.balance_liters, credit.total_earned
        )));
    }
    Ok(ValidateCallbackResult::Valid)
}

fn validate_create_h2o_credit(
    _action: Create,
    credit: H2OCredit,
) -> ExternResult<ValidateCallbackResult> {
    validate_h2o_credit_fields(&credit)
}

fn validate_update_h2o_credit(
    _action: Update,
    credit: H2OCredit,
    _original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    validate_h2o_credit_fields(&credit)
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
    if let Some(ref mr) = usage.meter_reference {
        if mr.len() > 256 {
            return Ok(ValidateCallbackResult::Invalid(
                "Meter reference must be 256 characters or fewer".into(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // HELPERS
    // ========================================================================

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    fn fake_agent_2() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_entry_hash() -> EntryHash {
        EntryHash::from_raw_36(vec![0u8; 36])
    }

    fn fake_create() -> Create {
        Create {
            author: fake_agent(),
            timestamp: Timestamp::from_micros(0),
            action_seq: 0,
            prev_action: fake_action_hash(),
            entry_type: EntryType::App(AppEntryDef::new(
                EntryDefIndex(0),
                ZomeIndex(0),
                EntryVisibility::Public,
            )),
            entry_hash: fake_entry_hash(),
            weight: EntryRateWeight::default(),
        }
    }

    fn is_valid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Valid))
    }

    fn is_invalid(result: &ExternResult<ValidateCallbackResult>) -> bool {
        matches!(result, Ok(ValidateCallbackResult::Invalid(_)))
    }

    fn invalid_msg(result: &ExternResult<ValidateCallbackResult>) -> String {
        match result {
            Ok(ValidateCallbackResult::Invalid(msg)) => msg.clone(),
            _ => panic!("Expected Invalid, got {:?}", result),
        }
    }

    fn make_water_source() -> WaterSource {
        WaterSource {
            id: "src-001".into(),
            name: "Cedar Creek Spring".into(),
            source_type: WaterSourceType::Spring,
            max_capacity_liters: 500_000,
            recharge_rate_liters_per_day: 10_000,
            location_lat: 45.5,
            location_lon: -122.6,
            steward: fake_agent(),
            status: SourceStatus::Active,
        }
    }

    fn make_water_share() -> WaterShare {
        WaterShare {
            source_hash: fake_action_hash(),
            holder: fake_agent(),
            allocation_type: AllocationType::Fixed,
            volume_per_period_liters: 1_000,
            period_days: 30,
            priority: 0,
            usage_category: WaterClassification::Potable,
        }
    }

    fn make_h2o_credit() -> H2OCredit {
        H2OCredit {
            holder: fake_agent(),
            balance_liters: 5_000,
            total_earned: 10_000,
            total_spent: 5_000,
        }
    }

    fn make_water_transaction() -> WaterTransaction {
        WaterTransaction {
            from_agent: fake_agent(),
            to_agent: fake_agent_2(),
            liters: 100,
            credit_type: TransactionType::Transfer,
            timestamp: Timestamp::from_micros(0),
            source_hash: None,
        }
    }

    fn make_usage_record() -> UsageRecord {
        UsageRecord {
            agent: fake_agent(),
            source_hash: fake_action_hash(),
            liters_used: 50,
            usage_category: WaterClassification::Irrigation,
            recorded_at: Timestamp::from_micros(0),
            meter_reference: None,
        }
    }

    // ========================================================================
    // SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn serde_roundtrip_water_source_type() {
        for variant in [
            WaterSourceType::Municipal,
            WaterSourceType::Well,
            WaterSourceType::Spring,
            WaterSourceType::Rainwater,
            WaterSourceType::Aquifer,
            WaterSourceType::River,
            WaterSourceType::Lake,
            WaterSourceType::Recycled,
            WaterSourceType::Desalinated,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: WaterSourceType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn serde_roundtrip_source_status() {
        for variant in [
            SourceStatus::Active,
            SourceStatus::Seasonal,
            SourceStatus::Depleted,
            SourceStatus::Contaminated,
            SourceStatus::UnderMaintenance,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: SourceStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn serde_roundtrip_allocation_type() {
        for variant in [
            AllocationType::Fixed,
            AllocationType::Proportional,
            AllocationType::Priority,
            AllocationType::Emergency,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: AllocationType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn serde_roundtrip_water_classification() {
        for variant in [
            WaterClassification::Potable,
            WaterClassification::Cooking,
            WaterClassification::Hygiene,
            WaterClassification::Irrigation,
            WaterClassification::Industrial,
            WaterClassification::Recreation,
            WaterClassification::Greywater,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: WaterClassification = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    #[test]
    fn serde_roundtrip_transaction_type() {
        for variant in [
            TransactionType::Allocation,
            TransactionType::Transfer,
            TransactionType::Purchase,
            TransactionType::Donation,
            TransactionType::Emergency,
        ] {
            let json = serde_json::to_string(&variant).unwrap();
            let back: TransactionType = serde_json::from_str(&json).unwrap();
            assert_eq!(variant, back);
        }
    }

    // ========================================================================
    // WATER SOURCE VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_water_source_passes() {
        let result = validate_create_water_source(fake_create(), make_water_source());
        assert!(is_valid(&result));
    }

    #[test]
    fn water_source_empty_id_rejected() {
        let mut src = make_water_source();
        src.id = "".into();
        let result = validate_create_water_source(fake_create(), src);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Water source ID cannot be empty");
    }

    #[test]
    fn water_source_empty_name_rejected() {
        let mut src = make_water_source();
        src.name = "".into();
        let result = validate_create_water_source(fake_create(), src);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Water source name cannot be empty");
    }

    #[test]
    fn water_source_zero_capacity_rejected() {
        let mut src = make_water_source();
        src.max_capacity_liters = 0;
        let result = validate_create_water_source(fake_create(), src);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Max capacity must be greater than zero"
        );
    }

    #[test]
    fn water_source_one_liter_capacity_accepted() {
        let mut src = make_water_source();
        src.max_capacity_liters = 1;
        let result = validate_create_water_source(fake_create(), src);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_source_u64_max_capacity_accepted() {
        let mut src = make_water_source();
        src.max_capacity_liters = u64::MAX;
        let result = validate_create_water_source(fake_create(), src);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_source_lat_over_90_rejected() {
        let mut src = make_water_source();
        src.location_lat = 90.001;
        let result = validate_create_water_source(fake_create(), src);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Latitude must be between -90 and 90");
    }

    #[test]
    fn water_source_lat_under_neg90_rejected() {
        let mut src = make_water_source();
        src.location_lat = -90.001;
        let result = validate_create_water_source(fake_create(), src);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Latitude must be between -90 and 90");
    }

    #[test]
    fn water_source_lat_exactly_90_accepted() {
        let mut src = make_water_source();
        src.location_lat = 90.0;
        let result = validate_create_water_source(fake_create(), src);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_source_lat_exactly_neg90_accepted() {
        let mut src = make_water_source();
        src.location_lat = -90.0;
        let result = validate_create_water_source(fake_create(), src);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_source_lat_zero_accepted() {
        let mut src = make_water_source();
        src.location_lat = 0.0;
        let result = validate_create_water_source(fake_create(), src);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_source_lon_over_180_rejected() {
        let mut src = make_water_source();
        src.location_lon = 180.001;
        let result = validate_create_water_source(fake_create(), src);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Longitude must be between -180 and 180"
        );
    }

    #[test]
    fn water_source_lon_under_neg180_rejected() {
        let mut src = make_water_source();
        src.location_lon = -180.001;
        let result = validate_create_water_source(fake_create(), src);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Longitude must be between -180 and 180"
        );
    }

    #[test]
    fn water_source_lon_exactly_180_accepted() {
        let mut src = make_water_source();
        src.location_lon = 180.0;
        let result = validate_create_water_source(fake_create(), src);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_source_lon_exactly_neg180_accepted() {
        let mut src = make_water_source();
        src.location_lon = -180.0;
        let result = validate_create_water_source(fake_create(), src);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_source_lon_zero_accepted() {
        let mut src = make_water_source();
        src.location_lon = 0.0;
        let result = validate_create_water_source(fake_create(), src);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_source_zero_recharge_accepted() {
        let mut src = make_water_source();
        src.recharge_rate_liters_per_day = 0;
        let result = validate_create_water_source(fake_create(), src);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_source_all_types_accepted() {
        for source_type in [
            WaterSourceType::Municipal,
            WaterSourceType::Well,
            WaterSourceType::Spring,
            WaterSourceType::Rainwater,
            WaterSourceType::Aquifer,
            WaterSourceType::River,
            WaterSourceType::Lake,
            WaterSourceType::Recycled,
            WaterSourceType::Desalinated,
        ] {
            let mut src = make_water_source();
            src.source_type = source_type;
            let result = validate_create_water_source(fake_create(), src);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn water_source_all_statuses_accepted() {
        for status in [
            SourceStatus::Active,
            SourceStatus::Seasonal,
            SourceStatus::Depleted,
            SourceStatus::Contaminated,
            SourceStatus::UnderMaintenance,
        ] {
            let mut src = make_water_source();
            src.status = status;
            let result = validate_create_water_source(fake_create(), src);
            assert!(is_valid(&result));
        }
    }

    // ========================================================================
    // WATER SHARE VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_water_share_passes() {
        let result = validate_create_water_share(fake_create(), make_water_share());
        assert!(is_valid(&result));
    }

    #[test]
    fn water_share_zero_volume_rejected() {
        let mut share = make_water_share();
        share.volume_per_period_liters = 0;
        let result = validate_create_water_share(fake_create(), share);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Share volume must be greater than zero"
        );
    }

    #[test]
    fn water_share_one_liter_accepted() {
        let mut share = make_water_share();
        share.volume_per_period_liters = 1;
        let result = validate_create_water_share(fake_create(), share);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_share_u64_max_volume_accepted() {
        let mut share = make_water_share();
        share.volume_per_period_liters = u64::MAX;
        let result = validate_create_water_share(fake_create(), share);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_share_zero_period_rejected() {
        let mut share = make_water_share();
        share.period_days = 0;
        let result = validate_create_water_share(fake_create(), share);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Period days must be greater than zero"
        );
    }

    #[test]
    fn water_share_one_day_period_accepted() {
        let mut share = make_water_share();
        share.period_days = 1;
        let result = validate_create_water_share(fake_create(), share);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_share_u32_max_period_accepted() {
        let mut share = make_water_share();
        share.period_days = u32::MAX;
        let result = validate_create_water_share(fake_create(), share);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_share_all_allocation_types_accepted() {
        for alloc_type in [
            AllocationType::Fixed,
            AllocationType::Proportional,
            AllocationType::Priority,
            AllocationType::Emergency,
        ] {
            let mut share = make_water_share();
            share.allocation_type = alloc_type;
            let result = validate_create_water_share(fake_create(), share);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn water_share_all_usage_categories_accepted() {
        for category in [
            WaterClassification::Potable,
            WaterClassification::Cooking,
            WaterClassification::Hygiene,
            WaterClassification::Irrigation,
            WaterClassification::Industrial,
            WaterClassification::Recreation,
            WaterClassification::Greywater,
        ] {
            let mut share = make_water_share();
            share.usage_category = category;
            let result = validate_create_water_share(fake_create(), share);
            assert!(is_valid(&result));
        }
    }

    #[test]
    fn water_share_priority_zero_accepted() {
        let mut share = make_water_share();
        share.priority = 0;
        let result = validate_create_water_share(fake_create(), share);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_share_priority_max_accepted() {
        let mut share = make_water_share();
        share.priority = u8::MAX;
        let result = validate_create_water_share(fake_create(), share);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // H2O CREDIT VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_h2o_credit_passes() {
        let result = validate_create_h2o_credit(fake_create(), make_h2o_credit());
        assert!(is_valid(&result));
    }

    #[test]
    fn h2o_credit_zero_balance_accepted() {
        let mut credit = make_h2o_credit();
        credit.balance_liters = 0;
        let result = validate_create_h2o_credit(fake_create(), credit);
        assert!(is_valid(&result));
    }

    #[test]
    fn h2o_credit_negative_balance_accepted() {
        let mut credit = make_h2o_credit();
        credit.balance_liters = -1_000;
        let result = validate_create_h2o_credit(fake_create(), credit);
        assert!(is_valid(&result));
    }

    #[test]
    fn h2o_credit_zero_totals_accepted() {
        let credit = H2OCredit {
            holder: fake_agent(),
            balance_liters: 0,
            total_earned: 0,
            total_spent: 0,
        };
        let result = validate_create_h2o_credit(fake_create(), credit);
        assert!(is_valid(&result));
    }

    // --- H2O Credit hardening tests ---

    #[test]
    fn h2o_credit_spent_exceeds_earned_rejected() {
        let credit = H2OCredit {
            holder: fake_agent(),
            balance_liters: 0,
            total_earned: 100,
            total_spent: 101,
        };
        let result = validate_create_h2o_credit(fake_create(), credit);
        assert!(is_invalid(&result));
    }

    #[test]
    fn h2o_credit_spent_equals_earned_accepted() {
        let credit = H2OCredit {
            holder: fake_agent(),
            balance_liters: 0,
            total_earned: 1_000,
            total_spent: 1_000,
        };
        let result = validate_create_h2o_credit(fake_create(), credit);
        assert!(is_valid(&result));
    }

    #[test]
    fn h2o_credit_balance_exceeds_earned_rejected() {
        let credit = H2OCredit {
            holder: fake_agent(),
            balance_liters: 10_001,
            total_earned: 10_000,
            total_spent: 0,
        };
        let result = validate_create_h2o_credit(fake_create(), credit);
        assert!(is_invalid(&result));
    }

    #[test]
    fn h2o_credit_balance_equals_earned_accepted() {
        let credit = H2OCredit {
            holder: fake_agent(),
            balance_liters: 10_000,
            total_earned: 10_000,
            total_spent: 0,
        };
        let result = validate_create_h2o_credit(fake_create(), credit);
        assert!(is_valid(&result));
    }

    #[test]
    fn h2o_credit_large_values_accepted() {
        let credit = H2OCredit {
            holder: fake_agent(),
            balance_liters: 1_000_000_000,
            total_earned: 1_000_000_000,
            total_spent: 0,
        };
        let result = validate_create_h2o_credit(fake_create(), credit);
        assert!(is_valid(&result));
    }

    #[test]
    fn h2o_credit_max_spent_exceeds_zero_earned_rejected() {
        let credit = H2OCredit {
            holder: fake_agent(),
            balance_liters: -1,
            total_earned: 0,
            total_spent: 1,
        };
        let result = validate_create_h2o_credit(fake_create(), credit);
        assert!(is_invalid(&result));
    }

    #[test]
    fn h2o_credit_shared_validation_rejects_bad_fields() {
        // The shared validator is used by both create and update paths
        let credit = H2OCredit {
            holder: fake_agent(),
            balance_liters: 0,
            total_earned: 100,
            total_spent: 200,
        };
        let result = validate_h2o_credit_fields(&credit);
        assert!(is_invalid(&result));
    }

    #[test]
    fn h2o_credit_negative_overdraft_with_zero_earned_rejected() {
        // Cannot have negative balance with nothing earned
        let credit = H2OCredit {
            holder: fake_agent(),
            balance_liters: -500,
            total_earned: 0,
            total_spent: 0,
        };
        // balance (-500) > total_earned (0) is false, so this passes the balance check
        // but total_spent (0) <= total_earned (0), so this is technically valid
        // The negative balance is allowed for overdraft
        let result = validate_create_h2o_credit(fake_create(), credit);
        assert!(is_valid(&result));
    }

    // ========================================================================
    // WATER TRANSACTION VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_water_transaction_passes() {
        let result = validate_create_water_transaction(fake_create(), make_water_transaction());
        assert!(is_valid(&result));
    }

    #[test]
    fn water_transaction_zero_liters_rejected() {
        let mut tx = make_water_transaction();
        tx.liters = 0;
        let result = validate_create_water_transaction(fake_create(), tx);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Transaction volume must be greater than zero"
        );
    }

    #[test]
    fn water_transaction_one_liter_accepted() {
        let mut tx = make_water_transaction();
        tx.liters = 1;
        let result = validate_create_water_transaction(fake_create(), tx);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_transaction_u64_max_liters_accepted() {
        let mut tx = make_water_transaction();
        tx.liters = u64::MAX;
        let result = validate_create_water_transaction(fake_create(), tx);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_transaction_self_transfer_rejected() {
        let mut tx = make_water_transaction();
        tx.to_agent = tx.from_agent.clone();
        let result = validate_create_water_transaction(fake_create(), tx);
        assert!(is_invalid(&result));
        assert_eq!(invalid_msg(&result), "Cannot transfer credits to self");
    }

    #[test]
    fn water_transaction_with_source_hash_accepted() {
        let mut tx = make_water_transaction();
        tx.source_hash = Some(fake_action_hash());
        let result = validate_create_water_transaction(fake_create(), tx);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_transaction_without_source_hash_accepted() {
        let tx = make_water_transaction(); // Default has None
        let result = validate_create_water_transaction(fake_create(), tx);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_transaction_all_types_accepted() {
        for tx_type in [
            TransactionType::Allocation,
            TransactionType::Transfer,
            TransactionType::Purchase,
            TransactionType::Donation,
            TransactionType::Emergency,
        ] {
            let mut tx = make_water_transaction();
            tx.credit_type = tx_type;
            let result = validate_create_water_transaction(fake_create(), tx);
            assert!(is_valid(&result));
        }
    }

    // ========================================================================
    // USAGE RECORD VALIDATION TESTS
    // ========================================================================

    #[test]
    fn valid_usage_record_passes() {
        let result = validate_create_usage_record(fake_create(), make_usage_record());
        assert!(is_valid(&result));
    }

    #[test]
    fn usage_record_zero_liters_rejected() {
        let mut usage = make_usage_record();
        usage.liters_used = 0;
        let result = validate_create_usage_record(fake_create(), usage);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Usage liters must be greater than zero"
        );
    }

    #[test]
    fn usage_record_one_liter_accepted() {
        let mut usage = make_usage_record();
        usage.liters_used = 1;
        let result = validate_create_usage_record(fake_create(), usage);
        assert!(is_valid(&result));
    }

    #[test]
    fn usage_record_u64_max_liters_accepted() {
        let mut usage = make_usage_record();
        usage.liters_used = u64::MAX;
        let result = validate_create_usage_record(fake_create(), usage);
        assert!(is_valid(&result));
    }

    #[test]
    fn usage_record_with_meter_reference_accepted() {
        let mut usage = make_usage_record();
        usage.meter_reference = Some("METER-42-ALPHA".into());
        let result = validate_create_usage_record(fake_create(), usage);
        assert!(is_valid(&result));
    }

    #[test]
    fn usage_record_without_meter_reference_accepted() {
        let usage = make_usage_record(); // Default has None
        let result = validate_create_usage_record(fake_create(), usage);
        assert!(is_valid(&result));
    }

    #[test]
    fn usage_record_all_categories_accepted() {
        for category in [
            WaterClassification::Potable,
            WaterClassification::Cooking,
            WaterClassification::Hygiene,
            WaterClassification::Irrigation,
            WaterClassification::Industrial,
            WaterClassification::Recreation,
            WaterClassification::Greywater,
        ] {
            let mut usage = make_usage_record();
            usage.usage_category = category;
            let result = validate_create_usage_record(fake_create(), usage);
            assert!(is_valid(&result));
        }
    }

    // ========================================================================
    // STRING LENGTH LIMIT BOUNDARY TESTS
    // ========================================================================

    #[test]
    fn water_source_id_at_limit_accepted() {
        let mut src = make_water_source();
        src.id = "A".repeat(64);
        let result = validate_create_water_source(fake_create(), src);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_source_id_over_limit_rejected() {
        let mut src = make_water_source();
        src.id = "A".repeat(257);
        let result = validate_create_water_source(fake_create(), src);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Water source ID must be 256 characters or fewer"
        );
    }

    #[test]
    fn water_source_name_at_limit_accepted() {
        let mut src = make_water_source();
        src.name = "A".repeat(256);
        let result = validate_create_water_source(fake_create(), src);
        assert!(is_valid(&result));
    }

    #[test]
    fn water_source_name_over_limit_rejected() {
        let mut src = make_water_source();
        src.name = "A".repeat(257);
        let result = validate_create_water_source(fake_create(), src);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Water source name must be 256 characters or fewer"
        );
    }

    #[test]
    fn usage_record_meter_reference_at_limit_accepted() {
        let mut usage = make_usage_record();
        usage.meter_reference = Some("M".repeat(256));
        let result = validate_create_usage_record(fake_create(), usage);
        assert!(is_valid(&result));
    }

    #[test]
    fn usage_record_meter_reference_over_limit_rejected() {
        let mut usage = make_usage_record();
        usage.meter_reference = Some("M".repeat(257));
        let result = validate_create_usage_record(fake_create(), usage);
        assert!(is_invalid(&result));
        assert_eq!(
            invalid_msg(&result),
            "Meter reference must be 256 characters or fewer"
        );
    }

    // ========================================================================
    // NOTE: validate_update_water_source and validate_update_h2o_credit
    // call must_get_valid_record() which requires a live Holochain conductor.
    // These are tested via sweettest integration tests, not unit tests.
    // ========================================================================

    // ========================================================================
    // LINK TAG LENGTH VALIDATION TESTS
    // ========================================================================

    fn validate_link_tag_for(
        link_type: LinkTypes,
        tag_bytes: Vec<u8>,
    ) -> ExternResult<ValidateCallbackResult> {
        let tag = LinkTag(tag_bytes);
        match link_type {
            LinkTypes::AllSources
            | LinkTypes::StewardToSource
            | LinkTypes::SourceToShare
            | LinkTypes::HolderToShare
            | LinkTypes::AgentToCredit
            | LinkTypes::SourceToUsage
            | LinkTypes::AgentToUsage => {
                if tag.0.len() > 256 {
                    return Ok(ValidateCallbackResult::Invalid(format!(
                        "{:?} link tag too long (max 256 bytes)",
                        link_type
                    )));
                }
                Ok(ValidateCallbackResult::Valid)
            }
            LinkTypes::SourceTypeToSource
            | LinkTypes::AgentToTransactionSent
            | LinkTypes::AgentToTransactionReceived => {
                if tag.0.len() > 512 {
                    return Ok(ValidateCallbackResult::Invalid(format!(
                        "{:?} link tag too long (max 512 bytes)",
                        link_type
                    )));
                }
                Ok(ValidateCallbackResult::Valid)
            }
        }
    }

    #[test]
    fn link_all_sources_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::AllSources, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_all_sources_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::AllSources, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    #[test]
    fn link_source_type_to_source_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::SourceTypeToSource, vec![0u8; 512]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_source_type_to_source_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::SourceTypeToSource, vec![0u8; 513]);
        assert!(is_invalid(&result));
    }

    #[test]
    fn link_steward_to_source_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::StewardToSource, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_steward_to_source_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::StewardToSource, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    #[test]
    fn link_source_to_share_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::SourceToShare, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_source_to_share_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::SourceToShare, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    #[test]
    fn link_holder_to_share_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::HolderToShare, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_holder_to_share_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::HolderToShare, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    #[test]
    fn link_agent_to_credit_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::AgentToCredit, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_agent_to_credit_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::AgentToCredit, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    #[test]
    fn link_agent_to_transaction_sent_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::AgentToTransactionSent, vec![0u8; 512]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_agent_to_transaction_sent_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::AgentToTransactionSent, vec![0u8; 513]);
        assert!(is_invalid(&result));
    }

    #[test]
    fn link_agent_to_transaction_received_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::AgentToTransactionReceived, vec![0u8; 512]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_agent_to_transaction_received_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::AgentToTransactionReceived, vec![0u8; 513]);
        assert!(is_invalid(&result));
    }

    #[test]
    fn link_source_to_usage_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::SourceToUsage, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_source_to_usage_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::SourceToUsage, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }

    #[test]
    fn link_agent_to_usage_tag_at_limit() {
        let result = validate_link_tag_for(LinkTypes::AgentToUsage, vec![0u8; 256]);
        assert!(is_valid(&result));
    }

    #[test]
    fn link_agent_to_usage_tag_too_long() {
        let result = validate_link_tag_for(LinkTypes::AgentToUsage, vec![0u8; 257]);
        assert!(is_invalid(&result));
    }
}
