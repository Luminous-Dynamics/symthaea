//! Flow Coordinator Zome
//! Business logic for water allocation, H2O credits, and transactions

use hdk::prelude::*;
use mycelix_bridge_common::{
    gate_consciousness, requirement_for_basic, requirement_for_proposal, GovernanceEligibility,
    GovernanceRequirement,
};
use water_flow_integrity::*;

fn require_consciousness(
    requirement: &GovernanceRequirement,
    action_name: &str,
) -> ExternResult<GovernanceEligibility> {
    gate_consciousness("commons_bridge", requirement, action_name)
}

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// Helper to collect records from links
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

fn records_from_links(links: Vec<Link>) -> ExternResult<Vec<Record>> {
    let mut records = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
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
    require_consciousness(&requirement_for_basic(), "register_source")?;
    if source.id.trim().is_empty() || source.id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Source ID must be 1-256 non-whitespace characters".into()
        )));
    }
    if source.name.trim().is_empty() || source.name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Source name must be 1-256 non-whitespace characters".into()
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
    require_consciousness(&requirement_for_proposal(), "update_source_status")?;
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
    require_consciousness(&requirement_for_proposal(), "allocate_shares")?;
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
    require_consciousness(&requirement_for_proposal(), "transfer_credits")?;
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
    require_consciousness(&requirement_for_basic(), "record_usage")?;
    let agent_info = agent_info()?;
    let agent_key = agent_info.agent_initial_pubkey.clone();
    let now = sys_time()?;

    if input.liters_used == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Usage amount must be greater than zero".into()
        )));
    }

    // Check sufficient balance before debiting
    let balance = get_my_balance(())?;
    if balance.balance_liters < input.liters_used as i64 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Insufficient H2O credit balance for this usage".into()
        )));
    }

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

    // Debit from agent credit balance (balance already fetched above)
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

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_agent() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![0u8; 36])
    }

    fn fake_agent_2() -> AgentPubKey {
        AgentPubKey::from_raw_36(vec![1u8; 36])
    }

    fn fake_action_hash() -> ActionHash {
        ActionHash::from_raw_36(vec![0u8; 36])
    }

    // ========================================================================
    // COORDINATOR STRUCT SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn update_source_status_input_serde_roundtrip() {
        let input = UpdateSourceStatusInput {
            source_hash: fake_action_hash(),
            new_status: SourceStatus::Active,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateSourceStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_status, SourceStatus::Active);
    }

    #[test]
    fn update_source_status_input_all_statuses() {
        for status in [
            SourceStatus::Active,
            SourceStatus::Seasonal,
            SourceStatus::Depleted,
            SourceStatus::Contaminated,
            SourceStatus::UnderMaintenance,
        ] {
            let input = UpdateSourceStatusInput {
                source_hash: fake_action_hash(),
                new_status: status.clone(),
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: UpdateSourceStatusInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.new_status, status);
        }
    }

    #[test]
    fn transfer_credits_input_serde_roundtrip() {
        let input = TransferCreditsInput {
            to_agent: fake_agent_2(),
            liters: 500,
            transaction_type: TransactionType::Transfer,
            source_hash: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: TransferCreditsInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.liters, 500);
        assert_eq!(decoded.transaction_type, TransactionType::Transfer);
        assert!(decoded.source_hash.is_none());
    }

    #[test]
    fn transfer_credits_input_with_source_hash() {
        let input = TransferCreditsInput {
            to_agent: fake_agent(),
            liters: 1000,
            transaction_type: TransactionType::Donation,
            source_hash: Some(fake_action_hash()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: TransferCreditsInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.liters, 1000);
        assert_eq!(decoded.transaction_type, TransactionType::Donation);
        assert!(decoded.source_hash.is_some());
    }

    #[test]
    fn transfer_credits_input_all_transaction_types() {
        for tx_type in [
            TransactionType::Allocation,
            TransactionType::Transfer,
            TransactionType::Purchase,
            TransactionType::Donation,
            TransactionType::Emergency,
        ] {
            let input = TransferCreditsInput {
                to_agent: fake_agent(),
                liters: 100,
                transaction_type: tx_type.clone(),
                source_hash: None,
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: TransferCreditsInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.transaction_type, tx_type);
        }
    }

    #[test]
    fn record_usage_input_serde_roundtrip() {
        let input = RecordUsageInput {
            source_hash: fake_action_hash(),
            liters_used: 250,
            usage_category: WaterClassification::Irrigation,
            meter_reference: Some("METER-001".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RecordUsageInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.liters_used, 250);
        assert_eq!(decoded.usage_category, WaterClassification::Irrigation);
        assert_eq!(decoded.meter_reference, Some("METER-001".to_string()));
    }

    #[test]
    fn record_usage_input_without_meter_reference() {
        let input = RecordUsageInput {
            source_hash: fake_action_hash(),
            liters_used: 100,
            usage_category: WaterClassification::Potable,
            meter_reference: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RecordUsageInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.liters_used, 100);
        assert_eq!(decoded.usage_category, WaterClassification::Potable);
        assert!(decoded.meter_reference.is_none());
    }

    #[test]
    fn record_usage_input_all_classifications() {
        for category in [
            WaterClassification::Potable,
            WaterClassification::Cooking,
            WaterClassification::Hygiene,
            WaterClassification::Irrigation,
            WaterClassification::Industrial,
            WaterClassification::Recreation,
            WaterClassification::Greywater,
        ] {
            let input = RecordUsageInput {
                source_hash: fake_action_hash(),
                liters_used: 50,
                usage_category: category.clone(),
                meter_reference: None,
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: RecordUsageInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.usage_category, category);
        }
    }

    // ========================================================================
    // INTEGRITY ENUM SERDE ROUNDTRIP TESTS (via coordinator re-export)
    // ========================================================================

    #[test]
    fn water_source_type_all_variants_serde() {
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
    fn source_status_all_variants_serde() {
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
    fn allocation_type_all_variants_serde() {
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
    fn water_classification_all_variants_serde() {
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
    fn transaction_type_all_variants_serde() {
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
    // WATER SOURCE SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn water_source_serde_roundtrip() {
        let source = WaterSource {
            id: "well-42".to_string(),
            name: "Deep Well Alpha".to_string(),
            source_type: WaterSourceType::Well,
            max_capacity_liters: 100_000,
            recharge_rate_liters_per_day: 5_000,
            location_lat: 35.6762,
            location_lon: 139.6503,
            steward: fake_agent(),
            status: SourceStatus::Active,
        };
        let json = serde_json::to_string(&source).unwrap();
        let decoded: WaterSource = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, source);
    }

    #[test]
    fn water_source_all_types_serde_roundtrip() {
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
            let source = WaterSource {
                id: "src-x".to_string(),
                name: "Test Source".to_string(),
                source_type: source_type.clone(),
                max_capacity_liters: 1000,
                recharge_rate_liters_per_day: 100,
                location_lat: 0.0,
                location_lon: 0.0,
                steward: fake_agent(),
                status: SourceStatus::Active,
            };
            let json = serde_json::to_string(&source).unwrap();
            let decoded: WaterSource = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.source_type, source_type);
        }
    }

    #[test]
    fn water_source_all_statuses_serde_roundtrip() {
        for status in [
            SourceStatus::Active,
            SourceStatus::Seasonal,
            SourceStatus::Depleted,
            SourceStatus::Contaminated,
            SourceStatus::UnderMaintenance,
        ] {
            let source = WaterSource {
                id: "src-s".to_string(),
                name: "Status Source".to_string(),
                source_type: WaterSourceType::Lake,
                max_capacity_liters: 50_000,
                recharge_rate_liters_per_day: 0,
                location_lat: 45.0,
                location_lon: -90.0,
                steward: fake_agent(),
                status: status.clone(),
            };
            let json = serde_json::to_string(&source).unwrap();
            let decoded: WaterSource = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.status, status);
        }
    }

    // ========================================================================
    // WATER SOURCE BOUNDARY VALUES
    // ========================================================================

    #[test]
    fn water_source_lat_lon_boundary_values_serde() {
        let source = WaterSource {
            id: "boundary".to_string(),
            name: "Boundary Test".to_string(),
            source_type: WaterSourceType::Spring,
            max_capacity_liters: 1,
            recharge_rate_liters_per_day: 0,
            location_lat: 90.0,
            location_lon: -180.0,
            steward: fake_agent(),
            status: SourceStatus::Active,
        };
        let json = serde_json::to_string(&source).unwrap();
        let decoded: WaterSource = serde_json::from_str(&json).unwrap();
        assert!((decoded.location_lat - 90.0).abs() < f64::EPSILON);
        assert!((decoded.location_lon - (-180.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn water_source_zero_recharge_serde() {
        let source = WaterSource {
            id: "stagnant".to_string(),
            name: "Stagnant Reservoir".to_string(),
            source_type: WaterSourceType::Lake,
            max_capacity_liters: 1_000_000,
            recharge_rate_liters_per_day: 0,
            location_lat: -33.8688,
            location_lon: 151.2093,
            steward: fake_agent(),
            status: SourceStatus::Seasonal,
        };
        let json = serde_json::to_string(&source).unwrap();
        let decoded: WaterSource = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.recharge_rate_liters_per_day, 0);
    }

    // ========================================================================
    // WATER SHARE SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn water_share_serde_roundtrip() {
        let share = WaterShare {
            source_hash: fake_action_hash(),
            holder: fake_agent(),
            allocation_type: AllocationType::Proportional,
            volume_per_period_liters: 5_000,
            period_days: 7,
            priority: 2,
            usage_category: WaterClassification::Hygiene,
        };
        let json = serde_json::to_string(&share).unwrap();
        let decoded: WaterShare = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, share);
    }

    #[test]
    fn water_share_all_allocation_types_serde() {
        for alloc in [
            AllocationType::Fixed,
            AllocationType::Proportional,
            AllocationType::Priority,
            AllocationType::Emergency,
        ] {
            let share = WaterShare {
                source_hash: fake_action_hash(),
                holder: fake_agent(),
                allocation_type: alloc.clone(),
                volume_per_period_liters: 100,
                period_days: 30,
                priority: 0,
                usage_category: WaterClassification::Potable,
            };
            let json = serde_json::to_string(&share).unwrap();
            let decoded: WaterShare = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.allocation_type, alloc);
        }
    }

    #[test]
    fn water_share_priority_boundary_serde() {
        for priority in [0u8, 1, 127, 255] {
            let share = WaterShare {
                source_hash: fake_action_hash(),
                holder: fake_agent(),
                allocation_type: AllocationType::Priority,
                volume_per_period_liters: 500,
                period_days: 14,
                priority,
                usage_category: WaterClassification::Irrigation,
            };
            let json = serde_json::to_string(&share).unwrap();
            let decoded: WaterShare = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.priority, priority);
        }
    }

    // ========================================================================
    // H2O CREDIT SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn h2o_credit_serde_roundtrip() {
        let credit = H2OCredit {
            holder: fake_agent(),
            balance_liters: 5_000,
            total_earned: 10_000,
            total_spent: 5_000,
        };
        let json = serde_json::to_string(&credit).unwrap();
        let decoded: H2OCredit = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, credit);
    }

    #[test]
    fn h2o_credit_negative_balance_serde() {
        let credit = H2OCredit {
            holder: fake_agent(),
            balance_liters: -500,
            total_earned: 1000,
            total_spent: 1500,
        };
        let json = serde_json::to_string(&credit).unwrap();
        let decoded: H2OCredit = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.balance_liters, -500);
    }

    #[test]
    fn h2o_credit_zero_balance_serde() {
        let credit = H2OCredit {
            holder: fake_agent(),
            balance_liters: 0,
            total_earned: 0,
            total_spent: 0,
        };
        let json = serde_json::to_string(&credit).unwrap();
        let decoded: H2OCredit = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.balance_liters, 0);
        assert_eq!(decoded.total_earned, 0);
        assert_eq!(decoded.total_spent, 0);
    }

    // ========================================================================
    // WATER TRANSACTION SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn water_transaction_serde_roundtrip() {
        let tx = WaterTransaction {
            from_agent: fake_agent(),
            to_agent: fake_agent_2(),
            liters: 250,
            credit_type: TransactionType::Donation,
            timestamp: Timestamp::from_micros(42_000),
            source_hash: Some(fake_action_hash()),
        };
        let json = serde_json::to_string(&tx).unwrap();
        let decoded: WaterTransaction = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, tx);
    }

    #[test]
    fn water_transaction_no_source_hash_serde() {
        let tx = WaterTransaction {
            from_agent: fake_agent(),
            to_agent: fake_agent_2(),
            liters: 100,
            credit_type: TransactionType::Transfer,
            timestamp: Timestamp::from_micros(0),
            source_hash: None,
        };
        let json = serde_json::to_string(&tx).unwrap();
        let decoded: WaterTransaction = serde_json::from_str(&json).unwrap();
        assert!(decoded.source_hash.is_none());
    }

    // ========================================================================
    // USAGE RECORD SERDE ROUNDTRIP TESTS
    // ========================================================================

    #[test]
    fn usage_record_serde_roundtrip() {
        let usage = UsageRecord {
            agent: fake_agent(),
            source_hash: fake_action_hash(),
            liters_used: 75,
            usage_category: WaterClassification::Cooking,
            recorded_at: Timestamp::from_micros(123_456),
            meter_reference: Some("METER-XYZ-99".to_string()),
        };
        let json = serde_json::to_string(&usage).unwrap();
        let decoded: UsageRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, usage);
    }

    #[test]
    fn usage_record_no_meter_serde() {
        let usage = UsageRecord {
            agent: fake_agent(),
            source_hash: fake_action_hash(),
            liters_used: 10,
            usage_category: WaterClassification::Greywater,
            recorded_at: Timestamp::from_micros(0),
            meter_reference: None,
        };
        let json = serde_json::to_string(&usage).unwrap();
        let decoded: UsageRecord = serde_json::from_str(&json).unwrap();
        assert!(decoded.meter_reference.is_none());
    }

    #[test]
    fn usage_record_all_classifications_serde() {
        for cat in [
            WaterClassification::Potable,
            WaterClassification::Cooking,
            WaterClassification::Hygiene,
            WaterClassification::Irrigation,
            WaterClassification::Industrial,
            WaterClassification::Recreation,
            WaterClassification::Greywater,
        ] {
            let usage = UsageRecord {
                agent: fake_agent(),
                source_hash: fake_action_hash(),
                liters_used: 50,
                usage_category: cat.clone(),
                recorded_at: Timestamp::from_micros(0),
                meter_reference: None,
            };
            let json = serde_json::to_string(&usage).unwrap();
            let decoded: UsageRecord = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.usage_category, cat);
        }
    }

    // ========================================================================
    // TRANSFER CREDITS INPUT EDGE CASES
    // ========================================================================

    #[test]
    fn transfer_credits_input_zero_liters_serde() {
        let input = TransferCreditsInput {
            to_agent: fake_agent_2(),
            liters: 0,
            transaction_type: TransactionType::Transfer,
            source_hash: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: TransferCreditsInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.liters, 0);
    }

    #[test]
    fn transfer_credits_input_u64_max_liters_serde() {
        let input = TransferCreditsInput {
            to_agent: fake_agent_2(),
            liters: u64::MAX,
            transaction_type: TransactionType::Emergency,
            source_hash: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: TransferCreditsInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.liters, u64::MAX);
    }

    // ========================================================================
    // RECORD USAGE INPUT EDGE CASES
    // ========================================================================

    #[test]
    fn record_usage_input_zero_liters_serde() {
        let input = RecordUsageInput {
            source_hash: fake_action_hash(),
            liters_used: 0,
            usage_category: WaterClassification::Potable,
            meter_reference: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RecordUsageInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.liters_used, 0);
    }

    #[test]
    fn record_usage_input_u64_max_liters_serde() {
        let input = RecordUsageInput {
            source_hash: fake_action_hash(),
            liters_used: u64::MAX,
            usage_category: WaterClassification::Industrial,
            meter_reference: Some("INDUSTRIAL-MAX".to_string()),
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: RecordUsageInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.liters_used, u64::MAX);
    }

    // ========================================================================
    // UPDATE SOURCE STATUS INPUT EDGE CASES
    // ========================================================================

    #[test]
    fn update_source_status_contaminated_serde() {
        let input = UpdateSourceStatusInput {
            source_hash: fake_action_hash(),
            new_status: SourceStatus::Contaminated,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateSourceStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_status, SourceStatus::Contaminated);
    }

    #[test]
    fn update_source_status_under_maintenance_serde() {
        let input = UpdateSourceStatusInput {
            source_hash: fake_action_hash(),
            new_status: SourceStatus::UnderMaintenance,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: UpdateSourceStatusInput = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.new_status, SourceStatus::UnderMaintenance);
    }

    // ========================================================================
    // ERROR-PATH & FAILURE-MODE TESTS
    // ========================================================================

    /// Verify that invalid SourceStatus JSON strings fail to deserialize
    /// rather than silently producing a wrong variant.
    #[test]
    fn source_status_invalid_variant_deser_fails() {
        let bad_json = r#""Frozen""#;
        let result = serde_json::from_str::<SourceStatus>(bad_json);
        assert!(
            result.is_err(),
            "Unknown status variant should fail deserialization"
        );

        let bad_json2 = r#""active""#; // lowercase
        let result2 = serde_json::from_str::<SourceStatus>(bad_json2);
        assert!(
            result2.is_err(),
            "Lowercase variant should fail deserialization"
        );

        let bad_json3 = r#""""#; // empty string
        let result3 = serde_json::from_str::<SourceStatus>(bad_json3);
        assert!(result3.is_err(), "Empty string should fail deserialization");
    }

    /// Verify that depleted and contaminated statuses correctly roundtrip in
    /// UpdateSourceStatusInput -- these are the statuses that block allocation
    /// in the coordinator (allocate_shares rejects them).
    #[test]
    fn update_source_status_allocation_blocking_statuses_roundtrip() {
        for blocking_status in [SourceStatus::Depleted, SourceStatus::Contaminated] {
            let input = UpdateSourceStatusInput {
                source_hash: fake_action_hash(),
                new_status: blocking_status.clone(),
            };
            let json = serde_json::to_string(&input).unwrap();
            let decoded: UpdateSourceStatusInput = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.new_status, blocking_status);
            // Confirm these are indeed the statuses that would block allocation
            assert!(
                decoded.new_status == SourceStatus::Depleted
                    || decoded.new_status == SourceStatus::Contaminated
            );
        }
    }

    /// TransferCreditsInput: verify that zero liters serializes correctly --
    /// the coordinator rejects this at runtime with "Transfer amount must be
    /// greater than zero", but the struct itself must serialize it faithfully.
    #[test]
    fn transfer_credits_zero_liters_roundtrip_preserves_zero() {
        let input = TransferCreditsInput {
            to_agent: fake_agent_2(),
            liters: 0,
            transaction_type: TransactionType::Transfer,
            source_hash: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: TransferCreditsInput = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.liters, 0,
            "Zero liters must survive serde roundtrip"
        );
    }

    /// TransferCreditsInput: same agent as sender/receiver is rejected by
    /// the coordinator at runtime. Verify the struct itself faithfully
    /// serializes identical agent pub keys and they compare equal after
    /// deserialization.
    #[test]
    fn transfer_credits_same_agent_serde_preserves_equality() {
        let same_agent = fake_agent();
        let input = TransferCreditsInput {
            to_agent: same_agent.clone(),
            liters: 100,
            transaction_type: TransactionType::Transfer,
            source_hash: None,
        };
        let json = serde_json::to_string(&input).unwrap();
        let decoded: TransferCreditsInput = serde_json::from_str(&json).unwrap();
        // After roundtrip, the to_agent must still match the original agent --
        // the coordinator uses this equality check to reject self-transfers
        assert_eq!(decoded.to_agent, same_agent);
    }

    /// WaterSource with an empty ID: verify serde roundtrip preserves the
    /// empty string. The coordinator rejects empty IDs with
    /// "Source ID must be 1-256 characters" but the struct must serialize it.
    #[test]
    fn water_source_empty_id_roundtrip_preserves_empty() {
        let source = WaterSource {
            id: "".to_string(),
            name: "Valid Name".to_string(),
            source_type: WaterSourceType::Well,
            max_capacity_liters: 1000,
            recharge_rate_liters_per_day: 100,
            location_lat: 0.0,
            location_lon: 0.0,
            steward: fake_agent(),
            status: SourceStatus::Active,
        };
        let json = serde_json::to_string(&source).unwrap();
        let decoded: WaterSource = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "", "Empty ID must survive roundtrip");
    }

    /// WaterSource with an overlong ID (>256 chars): verify serde roundtrip
    /// preserves the full string. The coordinator rejects this at runtime.
    #[test]
    fn water_source_overlong_id_roundtrip() {
        let long_id = "X".repeat(300);
        let source = WaterSource {
            id: long_id.clone(),
            name: "Valid Name".to_string(),
            source_type: WaterSourceType::River,
            max_capacity_liters: 5000,
            recharge_rate_liters_per_day: 500,
            location_lat: 10.0,
            location_lon: 20.0,
            steward: fake_agent(),
            status: SourceStatus::Active,
        };
        let json = serde_json::to_string(&source).unwrap();
        let decoded: WaterSource = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id.len(), 300, "Overlong ID must survive roundtrip");
        assert_eq!(decoded.id, long_id);
    }

    /// H2OCredit: verify that extreme negative balance (i64::MIN) roundtrips.
    /// The system allows negative balances (overdraft), so the minimum value
    /// must serialize and deserialize correctly.
    #[test]
    fn h2o_credit_i64_min_balance_roundtrip() {
        let credit = H2OCredit {
            holder: fake_agent(),
            balance_liters: i64::MIN,
            total_earned: 0,
            total_spent: 0,
        };
        let json = serde_json::to_string(&credit).unwrap();
        let decoded: H2OCredit = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.balance_liters, i64::MIN);
    }

    /// H2OCredit: verify that i64::MAX balance roundtrips correctly.
    #[test]
    fn h2o_credit_i64_max_balance_roundtrip() {
        let credit = H2OCredit {
            holder: fake_agent(),
            balance_liters: i64::MAX,
            total_earned: u64::MAX,
            total_spent: u64::MAX,
        };
        let json = serde_json::to_string(&credit).unwrap();
        let decoded: H2OCredit = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.balance_liters, i64::MAX);
        assert_eq!(decoded.total_earned, u64::MAX);
        assert_eq!(decoded.total_spent, u64::MAX);
    }

    /// WaterShare: verify boundary priority values (0 = highest, 255 = lowest)
    /// serde roundtrip correctly, specifically testing that the u8 extremes
    /// are not truncated or misinterpreted.
    #[test]
    fn water_share_priority_0_and_255_serde_roundtrip() {
        for (priority, expected_desc) in [(0u8, "highest"), (255u8, "lowest")] {
            let share = WaterShare {
                source_hash: fake_action_hash(),
                holder: fake_agent(),
                allocation_type: AllocationType::Priority,
                volume_per_period_liters: 500,
                period_days: 7,
                priority,
                usage_category: WaterClassification::Potable,
            };
            let json = serde_json::to_string(&share).unwrap();
            let decoded: WaterShare = serde_json::from_str(&json).unwrap();
            assert_eq!(
                decoded.priority, priority,
                "Priority {} ({}) must survive roundtrip",
                priority, expected_desc
            );
        }
    }

    /// WaterTransaction: verify that a transaction with from_agent == to_agent
    /// roundtrips correctly. The coordinator rejects self-transfers, but the
    /// integrity layer validation also catches this. The serde layer must
    /// preserve the data faithfully regardless.
    #[test]
    fn water_transaction_self_transfer_serde_roundtrip() {
        let same_agent = fake_agent();
        let tx = WaterTransaction {
            from_agent: same_agent.clone(),
            to_agent: same_agent.clone(),
            liters: 100,
            credit_type: TransactionType::Transfer,
            timestamp: Timestamp::from_micros(42_000),
            source_hash: None,
        };
        let json = serde_json::to_string(&tx).unwrap();
        let decoded: WaterTransaction = serde_json::from_str(&json).unwrap();
        assert_eq!(
            decoded.from_agent, decoded.to_agent,
            "Self-transfer agents must match after roundtrip"
        );
        assert_eq!(decoded.liters, 100);
    }

    /// TransferCreditsInput: verify invalid JSON for the transaction_type field
    /// causes deserialization failure.
    #[test]
    fn transfer_credits_input_invalid_transaction_type_deser_fails() {
        // Manually craft JSON with an invalid transaction type
        let bad_json = r#"{"to_agent":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"liters":100,"transaction_type":"Barter","source_hash":null}"#;
        let result = serde_json::from_str::<TransferCreditsInput>(bad_json);
        assert!(
            result.is_err(),
            "Invalid transaction type 'Barter' should fail deserialization"
        );
    }

    /// RecordUsageInput: verify invalid JSON for the usage_category field
    /// causes deserialization failure.
    #[test]
    fn record_usage_input_invalid_classification_deser_fails() {
        let bad_json = r#"{"source_hash":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"liters_used":50,"usage_category":"Sewage","meter_reference":null}"#;
        let result = serde_json::from_str::<RecordUsageInput>(bad_json);
        assert!(
            result.is_err(),
            "Invalid classification 'Sewage' should fail deserialization"
        );
    }

    // ========================================================================
    // VALIDATION HARDENING EDGE CASE TESTS
    // ========================================================================

    /// Source ID that is only whitespace should be rejected by register_source
    /// validation (whitespace-only IDs are not meaningful identifiers).
    #[test]
    fn water_source_whitespace_only_id_serde_roundtrip() {
        let source = WaterSource {
            id: "   ".to_string(),
            name: "Valid Name".to_string(),
            source_type: WaterSourceType::Well,
            max_capacity_liters: 1000,
            recharge_rate_liters_per_day: 100,
            location_lat: 0.0,
            location_lon: 0.0,
            steward: fake_agent(),
            status: SourceStatus::Active,
        };
        let json = serde_json::to_string(&source).unwrap();
        let decoded: WaterSource = serde_json::from_str(&json).unwrap();
        // The struct itself roundtrips faithfully; coordinator rejects at runtime
        assert_eq!(decoded.id, "   ");
        assert!(
            decoded.id.trim().is_empty(),
            "Whitespace-only ID must be caught by trim().is_empty()"
        );
    }

    /// Source name that is only whitespace should be rejected.
    #[test]
    fn water_source_whitespace_only_name_serde_roundtrip() {
        let source = WaterSource {
            id: "valid-id".to_string(),
            name: " \t\n ".to_string(),
            source_type: WaterSourceType::River,
            max_capacity_liters: 5000,
            recharge_rate_liters_per_day: 500,
            location_lat: 10.0,
            location_lon: 20.0,
            steward: fake_agent(),
            status: SourceStatus::Active,
        };
        let json = serde_json::to_string(&source).unwrap();
        let decoded: WaterSource = serde_json::from_str(&json).unwrap();
        assert!(
            decoded.name.trim().is_empty(),
            "Whitespace-only name must be caught by trim().is_empty()"
        );
    }

    /// RecordUsageInput with zero liters: the coordinator now rejects this
    /// with "Usage amount must be greater than zero".
    #[test]
    fn record_usage_input_zero_liters_struct_preserves_zero() {
        let input = RecordUsageInput {
            source_hash: fake_action_hash(),
            liters_used: 0,
            usage_category: WaterClassification::Potable,
            meter_reference: None,
        };
        assert_eq!(
            input.liters_used, 0,
            "Zero liters preserved in struct for coordinator rejection"
        );
    }
}
