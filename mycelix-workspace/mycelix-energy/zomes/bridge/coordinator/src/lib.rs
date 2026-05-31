// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Energy Bridge Coordinator Zome
//!
//! Cross-hApp communication for Terra Atlas integration, investment tracking,
//! and regenerative exit coordination.

use energy_bridge_integrity::*;
use hdk::prelude::*;

const ENERGY_HAPP_ID: &str = "mycelix-energy";

/// Create or retrieve an anchor entry hash for deterministic link bases
fn anchor_hash(anchor_string: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_string.to_string());
    let _ = create_entry(&EntryTypes::Anchor(anchor.clone()));
    hash_entry(&anchor)
}

/// Sync project from Terra Atlas
#[hdk_extern]
pub fn sync_terra_atlas_project(input: SyncProjectInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let project = TerraAtlasProject {
        id: format!("project:{}:{}", input.terra_atlas_id, now.as_micros()),
        terra_atlas_id: input.terra_atlas_id.clone(),
        name: input.name,
        project_type: input.project_type,
        location: input.location,
        capacity_mw: input.capacity_mw,
        total_investment: input.total_investment,
        current_investment: input.current_investment,
        status: input.status,
        regenerative_progress: 0.0,
        synced_at: now,
    };

    let hash = create_entry(&EntryTypes::TerraAtlasProject(project))?;

    create_link(
        anchor_hash("all_projects")?,
        hash.clone(),
        LinkTypes::AllProjects,
        (),
    )?;

    broadcast_energy_event(BroadcastEnergyEventInput {
        event_type: EnergyEventType::ProjectDiscovered,
        project_id: input.terra_atlas_id,
        payload: "{}".to_string(),
    })?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Project not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SyncProjectInput {
    pub terra_atlas_id: String,
    pub name: String,
    pub project_type: EnergyType,
    pub location: GeoLocation,
    pub capacity_mw: f64,
    pub total_investment: u64,
    pub current_investment: u64,
    pub status: ProjectStatus,
}

/// Record investment from finance hApp
#[hdk_extern]
pub fn record_investment(input: RecordInvestmentInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let investment = InvestmentRecord {
        id: format!(
            "invest:{}:{}:{}",
            input.project_id,
            input.investor_did,
            now.as_micros()
        ),
        project_id: input.project_id.clone(),
        investor_did: input.investor_did.clone(),
        amount: input.amount,
        currency: input.currency,
        source_happ: input.source_happ,
        investment_type: input.investment_type,
        created_at: now,
    };

    let hash = create_entry(&EntryTypes::InvestmentRecord(investment))?;

    create_link(
        anchor_hash(&input.project_id)?,
        hash.clone(),
        LinkTypes::ProjectToInvestments,
        (),
    )?;

    create_link(
        anchor_hash(&input.investor_did)?,
        hash.clone(),
        LinkTypes::DidToInvestments,
        (),
    )?;

    broadcast_energy_event(BroadcastEnergyEventInput {
        event_type: EnergyEventType::InvestmentReceived,
        project_id: input.project_id,
        payload: serde_json::json!({
            "amount": input.amount,
            "investor": input.investor_did,
        })
        .to_string(),
    })?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Investment not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordInvestmentInput {
    pub project_id: String,
    pub investor_did: String,
    pub amount: u64,
    pub currency: String,
    pub source_happ: String,
    pub investment_type: InvestmentType,
}

/// Record regenerative milestone
#[hdk_extern]
pub fn record_milestone(input: RecordMilestoneInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let milestone = RegenerativeMilestone {
        id: format!(
            "milestone:{}:{:?}:{}",
            input.project_id,
            input.milestone_type,
            now.as_micros()
        ),
        project_id: input.project_id.clone(),
        milestone_type: input.milestone_type.clone(),
        community_readiness: input.community_readiness,
        operator_certification: input.operator_certification,
        financial_sustainability: input.financial_sustainability,
        achieved_at: Some(now),
        verified_by: input.verified_by,
    };

    let hash = create_entry(&EntryTypes::RegenerativeMilestone(milestone))?;

    create_link(
        anchor_hash(&input.project_id)?,
        hash.clone(),
        LinkTypes::ProjectToMilestones,
        (),
    )?;

    broadcast_energy_event(BroadcastEnergyEventInput {
        event_type: EnergyEventType::MilestoneAchieved,
        project_id: input.project_id,
        payload: serde_json::json!({
            "milestone_type": format!("{:?}", input.milestone_type),
            "community_readiness": input.community_readiness,
        })
        .to_string(),
    })?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Milestone not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordMilestoneInput {
    pub project_id: String,
    pub milestone_type: MilestoneType,
    pub community_readiness: f64,
    pub operator_certification: bool,
    pub financial_sustainability: f64,
    pub verified_by: Option<String>,
}

/// Get project investments
#[hdk_extern]
pub fn get_project_investments(project_id: String) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&project_id)?, LinkTypes::ProjectToInvestments)?,
        GetStrategy::default(),
    )?;

    let mut investments = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            investments.push(record);
        }
    }

    Ok(investments)
}

/// Get investments by DID
#[hdk_extern]
pub fn get_investments_by_did(did: String) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&did)?, LinkTypes::DidToInvestments)?,
        GetStrategy::default(),
    )?;

    let mut investments = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            investments.push(record);
        }
    }

    Ok(investments)
}

/// Broadcast energy event
#[hdk_extern]
pub fn broadcast_energy_event(input: BroadcastEnergyEventInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let event = EnergyBridgeEvent {
        id: format!("event:{:?}:{}", input.event_type, now.as_micros()),
        event_type: input.event_type,
        project_id: input.project_id,
        payload: input.payload,
        source: ENERGY_HAPP_ID.to_string(),
        timestamp: now,
    };

    let hash = create_entry(&EntryTypes::EnergyBridgeEvent(event))?;

    create_link(
        anchor_hash("recent_events")?,
        hash.clone(),
        LinkTypes::RecentEvents,
        (),
    )?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Event not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BroadcastEnergyEventInput {
    pub event_type: EnergyEventType,
    pub project_id: String,
    pub payload: String,
}

// =============================================================================
// Production Metrics (for bidirectional sync)
// =============================================================================

/// Record production metrics for an energy project
///
/// Called periodically (e.g., monthly) to record actual energy generation.
/// This data is used for:
/// - Verifying project performance
/// - Calculating returns for investors
/// - Triggering regenerative transition conditions
#[hdk_extern]
pub fn record_production_update(input: RecordProductionInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let record = ProductionRecord {
        id: format!(
            "prod:{}:{}:{}",
            input.project_id,
            input.period_start.as_micros(),
            now.as_micros()
        ),
        project_id: input.project_id.clone(),
        terra_atlas_id: input.terra_atlas_id.clone(),
        period_start: input.period_start,
        period_end: input.period_end,
        energy_generated_mwh: input.energy_generated_mwh,
        capacity_factor: input.capacity_factor,
        revenue_generated: input.revenue_generated,
        currency: input.currency.clone(),
        grid_injection_mwh: input.grid_injection_mwh,
        self_consumption_mwh: input.self_consumption_mwh,
        recorded_at: now,
        verified_by: input.verified_by.clone(),
    };

    let hash = create_entry(&EntryTypes::ProductionRecord(record))?;

    // Link to project
    create_link(
        anchor_hash(&input.project_id)?,
        hash.clone(),
        LinkTypes::ProjectToProduction,
        (),
    )?;

    // Queue for sync to Terra Atlas
    queue_sync_to_terra_atlas(QueueSyncInput {
        sync_type: SyncType::ProductionMetrics,
        project_id: input.project_id.clone(),
        payload: serde_json::json!({
            "terra_atlas_id": input.terra_atlas_id,
            "period_start": input.period_start.as_micros(),
            "period_end": input.period_end.as_micros(),
            "energy_generated_mwh": input.energy_generated_mwh,
            "capacity_factor": input.capacity_factor,
            "revenue_generated": input.revenue_generated,
            "currency": input.currency,
        })
        .to_string(),
    })?;

    // Broadcast event
    broadcast_energy_event(BroadcastEnergyEventInput {
        event_type: EnergyEventType::ProductionUpdate,
        project_id: input.project_id,
        payload: serde_json::json!({
            "energy_mwh": input.energy_generated_mwh,
            "capacity_factor": input.capacity_factor,
            "revenue": input.revenue_generated,
        })
        .to_string(),
    })?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Production record not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordProductionInput {
    pub project_id: String,
    pub terra_atlas_id: String,
    pub period_start: Timestamp,
    pub period_end: Timestamp,
    pub energy_generated_mwh: f64,
    pub capacity_factor: f64,
    pub revenue_generated: u64,
    pub currency: String,
    pub grid_injection_mwh: f64,
    pub self_consumption_mwh: f64,
    pub verified_by: Option<String>,
}

/// Get production history for a project
#[hdk_extern]
pub fn get_project_production_history(project_id: String) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&project_id)?, LinkTypes::ProjectToProduction)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            records.push(record);
        }
    }

    Ok(records)
}

// =============================================================================
// Investment Aggregation
// =============================================================================

/// Get total investment for a project
#[hdk_extern]
pub fn get_project_investment_total(project_id: String) -> ExternResult<InvestmentSummary> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&project_id)?, LinkTypes::ProjectToInvestments)?,
        GetStrategy::default(),
    )?;

    let mut total_amount: u64 = 0;
    let mut investor_count: u32 = 0;
    let mut investors: Vec<String> = Vec::new();
    let mut by_type: std::collections::HashMap<String, u64> = std::collections::HashMap::new();

    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            if let Some(investment) = record
                .entry()
                .to_app_option::<InvestmentRecord>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Parse error: {:?}", e))))?
            {
                total_amount += investment.amount;
                if !investors.contains(&investment.investor_did) {
                    investors.push(investment.investor_did.clone());
                    investor_count += 1;
                }
                let type_key = format!("{:?}", investment.investment_type);
                *by_type.entry(type_key).or_insert(0) += investment.amount;
            }
        }
    }

    Ok(InvestmentSummary {
        project_id,
        total_amount,
        investor_count,
        by_type: by_type.into_iter().collect(),
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct InvestmentSummary {
    pub project_id: String,
    pub total_amount: u64,
    pub investor_count: u32,
    pub by_type: Vec<(String, u64)>,
}

// =============================================================================
// Sync Queue Management (for bidirectional sync to Terra Atlas)
// =============================================================================

/// Queue data for sync to Terra Atlas
///
/// Creates a PendingSyncRecord that external sync services can poll.
/// The sync service reads pending records, pushes to Terra Atlas API,
/// then calls `mark_sync_complete()` to update status.
#[hdk_extern]
pub fn queue_sync_to_terra_atlas(input: QueueSyncInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let record = PendingSyncRecord {
        id: format!(
            "sync:{:?}:{}:{}",
            input.sync_type,
            input.project_id,
            now.as_micros()
        ),
        sync_type: input.sync_type,
        target_system: "terra-atlas".to_string(),
        payload: input.payload,
        created_at: now,
        synced_at: None,
        retry_count: 0,
        last_error: None,
    };

    let hash = create_entry(&EntryTypes::PendingSyncRecord(record))?;

    create_link(
        anchor_hash("pending_syncs")?,
        hash.clone(),
        LinkTypes::PendingSyncs,
        (),
    )?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Sync record not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct QueueSyncInput {
    pub sync_type: SyncType,
    pub project_id: String,
    pub payload: String,
}

/// Get all pending syncs (for external sync service)
#[hdk_extern]
pub fn get_pending_syncs(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("pending_syncs")?, LinkTypes::PendingSyncs)?,
        GetStrategy::default(),
    )?;

    let mut records = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        if let Some(record) = get(hash.clone(), GetOptions::default())? {
            // Only include records that haven't been synced yet
            if let Some(sync_record) = record
                .entry()
                .to_app_option::<PendingSyncRecord>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Parse error: {:?}", e))))?
            {
                if sync_record.synced_at.is_none() {
                    records.push(record);
                }
            }
        }
    }

    Ok(records)
}

/// Mark a sync as complete (called by external sync service)
#[hdk_extern]
pub fn mark_sync_complete(input: MarkSyncCompleteInput) -> ExternResult<bool> {
    // Create completion event for audit trail
    let now = sys_time()?;

    broadcast_energy_event(BroadcastEnergyEventInput {
        event_type: EnergyEventType::SyncPending, // Using existing type for sync status
        project_id: input.sync_id.clone(),
        payload: serde_json::json!({
            "status": if input.success { "completed" } else { "failed" },
            "error": input.error,
            "synced_at": now.as_micros(),
        })
        .to_string(),
    })?;

    Ok(true)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MarkSyncCompleteInput {
    pub sync_id: String,
    pub success: bool,
    pub error: Option<String>,
}

// =============================================================================
// Project Management
// =============================================================================

/// Update project status
#[hdk_extern]
pub fn update_project_status(input: UpdateProjectStatusInput) -> ExternResult<Record> {
    // Queue status change for sync
    queue_sync_to_terra_atlas(QueueSyncInput {
        sync_type: SyncType::StatusChange,
        project_id: input.project_id.clone(),
        payload: serde_json::json!({
            "terra_atlas_id": input.terra_atlas_id,
            "new_status": format!("{:?}", input.new_status),
            "reason": input.reason,
        })
        .to_string(),
    })?;

    // Broadcast event
    broadcast_energy_event(BroadcastEnergyEventInput {
        event_type: EnergyEventType::StatusChanged,
        project_id: input.project_id.clone(),
        payload: serde_json::json!({
            "new_status": format!("{:?}", input.new_status),
            "reason": input.reason,
        })
        .to_string(),
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct UpdateProjectStatusInput {
    pub project_id: String,
    pub terra_atlas_id: String,
    pub new_status: ProjectStatus,
    pub reason: Option<String>,
}

/// Get all synced projects
#[hdk_extern]
pub fn get_all_projects(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("all_projects")?, LinkTypes::AllProjects)?,
        GetStrategy::default(),
    )?;

    let mut projects = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            projects.push(record);
        }
    }

    Ok(projects)
}

/// Get project milestones
#[hdk_extern]
pub fn get_project_milestones(project_id: String) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&project_id)?, LinkTypes::ProjectToMilestones)?,
        GetStrategy::default(),
    )?;

    let mut milestones = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            milestones.push(record);
        }
    }

    Ok(milestones)
}

/// Get recent events
#[hdk_extern]
pub fn get_recent_events(limit: u32) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("recent_events")?, LinkTypes::RecentEvents)?,
        GetStrategy::default(),
    )?;

    let mut events = Vec::new();
    for link in links.into_iter().rev().take(limit as usize) {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            events.push(record);
        }
    }

    Ok(events)
}

// =============================================================================
// Regenerative Transition Management
// =============================================================================

/// Initiate regenerative transition
///
/// Starts the process of transitioning ownership from investors to community.
/// Per the Luminous Chimera Model:
/// - Condition-based (not time-based) transitions
/// - Reserve account must be funded (3-5% annual)
/// - Community must demonstrate readiness
/// - Minimum investor returns met
#[hdk_extern]
pub fn initiate_transition(input: InitiateTransitionInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let record = TransitionRecord {
        id: format!("transition:{}:{}", input.project_id, now.as_micros()),
        project_id: input.project_id.clone(),
        terra_atlas_id: input.terra_atlas_id.clone(),
        from_ownership_pct: input.current_community_ownership,
        to_ownership_pct: input.target_ownership_pct,
        community_did: input.community_did.clone(),
        reserve_account_balance: input.reserve_account_balance,
        conditions_met: input.conditions_met.clone(),
        conditions_pending: input.conditions_pending.clone(),
        status: TransitionStatus::Proposed,
        initiated_at: now,
        completed_at: None,
    };

    let hash = create_entry(&EntryTypes::TransitionRecord(record))?;

    // Link to project
    create_link(
        anchor_hash(&input.project_id)?,
        hash.clone(),
        LinkTypes::ProjectToTransitions,
        (),
    )?;

    // Link to active transitions
    create_link(
        anchor_hash("active_transitions")?,
        hash.clone(),
        LinkTypes::ActiveTransitions,
        (),
    )?;

    // Queue for sync
    queue_sync_to_terra_atlas(QueueSyncInput {
        sync_type: SyncType::TransitionProgress,
        project_id: input.project_id.clone(),
        payload: serde_json::json!({
            "terra_atlas_id": input.terra_atlas_id,
            "current_ownership": input.current_community_ownership,
            "target_ownership": input.target_ownership_pct,
            "community_did": input.community_did,
            "status": "Proposed",
        })
        .to_string(),
    })?;

    // Broadcast
    broadcast_energy_event(BroadcastEnergyEventInput {
        event_type: EnergyEventType::TransitionInitiated,
        project_id: input.project_id,
        payload: serde_json::json!({
            "from_pct": input.current_community_ownership,
            "to_pct": input.target_ownership_pct,
            "community": input.community_did,
        })
        .to_string(),
    })?;

    get(hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Transition record not found".into()
    )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct InitiateTransitionInput {
    pub project_id: String,
    pub terra_atlas_id: String,
    pub community_did: String,
    pub current_community_ownership: f64,
    pub target_ownership_pct: f64,
    pub reserve_account_balance: u64,
    pub conditions_met: Vec<String>,
    pub conditions_pending: Vec<String>,
}

/// Complete a regenerative transition
#[hdk_extern]
pub fn complete_transition(input: CompleteTransitionInput) -> ExternResult<Record> {
    // Queue completion for sync
    queue_sync_to_terra_atlas(QueueSyncInput {
        sync_type: SyncType::TransitionProgress,
        project_id: input.project_id.clone(),
        payload: serde_json::json!({
            "terra_atlas_id": input.terra_atlas_id,
            "final_ownership": input.final_ownership_pct,
            "status": "Completed",
        })
        .to_string(),
    })?;

    // Broadcast completion
    broadcast_energy_event(BroadcastEnergyEventInput {
        event_type: EnergyEventType::CommunityOwnershipComplete,
        project_id: input.project_id,
        payload: serde_json::json!({
            "final_ownership": input.final_ownership_pct,
            "community": input.community_did,
        })
        .to_string(),
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CompleteTransitionInput {
    pub transition_id: String,
    pub project_id: String,
    pub terra_atlas_id: String,
    pub community_did: String,
    pub final_ownership_pct: f64,
}

/// Get active transitions
#[hdk_extern]
pub fn get_active_transitions(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash("active_transitions")?,
            LinkTypes::ActiveTransitions,
        )?,
        GetStrategy::default(),
    )?;

    let mut transitions = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            transitions.push(record);
        }
    }

    Ok(transitions)
}

/// Get project transitions
#[hdk_extern]
pub fn get_project_transitions(project_id: String) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&project_id)?, LinkTypes::ProjectToTransitions)?,
        GetStrategy::default(),
    )?;

    let mut transitions = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            transitions.push(record);
        }
    }

    Ok(transitions)
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Test Helpers
    // =========================================================================

    fn create_test_timestamp() -> Timestamp {
        Timestamp::from_micros(1704067200000000) // 2024-01-01 00:00:00 UTC
    }

    fn valid_geo_location() -> GeoLocation {
        GeoLocation {
            latitude: 37.7749,
            longitude: -122.4194,
            region: "California".to_string(),
            country: "USA".to_string(),
        }
    }

    // =========================================================================
    // SyncProjectInput Tests
    // =========================================================================

    fn valid_sync_project_input() -> SyncProjectInput {
        SyncProjectInput {
            terra_atlas_id: "TA-2024-001".to_string(),
            name: "Sahara Solar Farm".to_string(),
            project_type: EnergyType::Solar,
            location: valid_geo_location(),
            capacity_mw: 500.0,
            total_investment: 100_000_000,
            current_investment: 50_000_000,
            status: ProjectStatus::Funding,
        }
    }

    #[test]
    fn test_sync_project_input_valid() {
        let input = valid_sync_project_input();
        assert!(!input.terra_atlas_id.is_empty());
        assert!(!input.name.is_empty());
        assert!(input.capacity_mw > 0.0);
    }

    #[test]
    fn test_sync_project_input_all_energy_types() {
        let types = vec![
            EnergyType::Solar,
            EnergyType::Wind,
            EnergyType::Hydro,
            EnergyType::Geothermal,
            EnergyType::Nuclear,
            EnergyType::Storage,
            EnergyType::Mixed,
        ];
        for energy_type in types {
            let input = SyncProjectInput {
                project_type: energy_type.clone(),
                ..valid_sync_project_input()
            };
            assert_eq!(input.project_type, energy_type);
        }
    }

    #[test]
    fn test_sync_project_input_all_statuses() {
        let statuses = vec![
            ProjectStatus::Discovery,
            ProjectStatus::Funding,
            ProjectStatus::Development,
            ProjectStatus::Operational,
            ProjectStatus::Transitioning,
            ProjectStatus::CommunityOwned,
        ];
        for status in statuses {
            let input = SyncProjectInput {
                status: status.clone(),
                ..valid_sync_project_input()
            };
            assert_eq!(input.status, status);
        }
    }

    #[test]
    fn test_sync_project_input_funding_progress() {
        let input = valid_sync_project_input();
        let progress = (input.current_investment as f64 / input.total_investment as f64) * 100.0;
        assert_eq!(progress, 50.0);
    }

    #[test]
    fn test_sync_project_input_serialization() {
        let input = valid_sync_project_input();
        let json = serde_json::to_string(&input);
        assert!(json.is_ok());
    }

    // =========================================================================
    // RecordInvestmentInput Tests
    // =========================================================================

    fn valid_record_investment_input() -> RecordInvestmentInput {
        RecordInvestmentInput {
            project_id: "project:solar_farm_alpha".to_string(),
            investor_did: "did:mycelix:investor1".to_string(),
            amount: 50000,
            currency: "USD".to_string(),
            source_happ: "mycelix-finance".to_string(),
            investment_type: InvestmentType::Equity,
        }
    }

    #[test]
    fn test_record_investment_input_valid() {
        let input = valid_record_investment_input();
        assert!(input.investor_did.starts_with("did:mycelix:"));
        assert!(input.amount > 0);
    }

    #[test]
    fn test_record_investment_input_all_types() {
        let types = vec![
            InvestmentType::Equity,
            InvestmentType::Loan,
            InvestmentType::Grant,
            InvestmentType::CommunityShare,
        ];
        for inv_type in types {
            let input = RecordInvestmentInput {
                investment_type: inv_type.clone(),
                ..valid_record_investment_input()
            };
            assert_eq!(input.investment_type, inv_type);
        }
    }

    #[test]
    fn test_record_investment_input_various_amounts() {
        let amounts = vec![100, 1000, 10000, 100000, 1_000_000];
        for amount in amounts {
            let input = RecordInvestmentInput {
                amount,
                ..valid_record_investment_input()
            };
            assert!(input.amount > 0);
        }
    }

    #[test]
    fn test_record_investment_input_various_currencies() {
        let currencies = vec!["USD", "EUR", "GBP", "CHF", "BTC"];
        for currency in currencies {
            let input = RecordInvestmentInput {
                currency: currency.to_string(),
                ..valid_record_investment_input()
            };
            assert_eq!(input.currency, currency);
        }
    }

    // =========================================================================
    // RecordMilestoneInput Tests
    // =========================================================================

    fn valid_record_milestone_input() -> RecordMilestoneInput {
        RecordMilestoneInput {
            project_id: "project:solar_farm_alpha".to_string(),
            milestone_type: MilestoneType::CommunityFormation,
            community_readiness: 0.8,
            operator_certification: true,
            financial_sustainability: 0.75,
            verified_by: Some("did:mycelix:verifier1".to_string()),
        }
    }

    #[test]
    fn test_record_milestone_input_valid() {
        let input = valid_record_milestone_input();
        assert!(!input.project_id.is_empty());
        assert!(input.community_readiness >= 0.0 && input.community_readiness <= 1.0);
    }

    #[test]
    fn test_record_milestone_input_all_types() {
        let types = vec![
            MilestoneType::CommunityFormation,
            MilestoneType::OperatorTraining,
            MilestoneType::FinancialIndependence,
            MilestoneType::GovernanceEstablished,
            MilestoneType::FullTransition,
        ];
        for milestone_type in types {
            let input = RecordMilestoneInput {
                milestone_type: milestone_type.clone(),
                ..valid_record_milestone_input()
            };
            assert_eq!(input.milestone_type, milestone_type);
        }
    }

    #[test]
    fn test_record_milestone_input_with_verifier() {
        let input = valid_record_milestone_input();
        assert!(input.verified_by.is_some());
    }

    #[test]
    fn test_record_milestone_input_without_verifier() {
        let input = RecordMilestoneInput {
            verified_by: None,
            ..valid_record_milestone_input()
        };
        assert!(input.verified_by.is_none());
    }

    // =========================================================================
    // RecordProductionInput Tests
    // =========================================================================

    fn valid_record_production_input() -> RecordProductionInput {
        RecordProductionInput {
            project_id: "project:solar_farm_alpha".to_string(),
            terra_atlas_id: "TA-2024-001".to_string(),
            period_start: create_test_timestamp(),
            period_end: Timestamp::from_micros(1706745600000000),
            energy_generated_mwh: 1500.0,
            capacity_factor: 0.25,
            revenue_generated: 150000,
            currency: "USD".to_string(),
            grid_injection_mwh: 1200.0,
            self_consumption_mwh: 300.0,
            verified_by: Some("did:mycelix:grid_operator".to_string()),
        }
    }

    #[test]
    fn test_record_production_input_valid() {
        let input = valid_record_production_input();
        assert!(!input.project_id.is_empty());
        assert!(input.energy_generated_mwh >= 0.0);
        assert!(input.capacity_factor >= 0.0 && input.capacity_factor <= 1.0);
    }

    #[test]
    fn test_record_production_input_energy_distribution() {
        let input = valid_record_production_input();
        let total = input.grid_injection_mwh + input.self_consumption_mwh;
        assert!(total <= input.energy_generated_mwh * 1.01); // 1% tolerance
    }

    #[test]
    fn test_record_production_input_period_ordering() {
        let input = valid_record_production_input();
        assert!(input.period_end.as_micros() > input.period_start.as_micros());
    }

    #[test]
    fn test_record_production_input_serialization() {
        let input = valid_record_production_input();
        let json = serde_json::to_string(&input);
        assert!(json.is_ok());
    }

    // =========================================================================
    // BroadcastEnergyEventInput Tests
    // =========================================================================

    fn valid_broadcast_energy_event_input() -> BroadcastEnergyEventInput {
        BroadcastEnergyEventInput {
            event_type: EnergyEventType::InvestmentReceived,
            project_id: "project:solar_farm_alpha".to_string(),
            payload: r#"{"amount": 50000, "investor": "did:mycelix:investor1"}"#.to_string(),
        }
    }

    #[test]
    fn test_broadcast_energy_event_input_valid() {
        let input = valid_broadcast_energy_event_input();
        assert!(!input.project_id.is_empty());
        assert!(!input.payload.is_empty());
    }

    #[test]
    fn test_broadcast_energy_event_input_all_types() {
        let types = vec![
            EnergyEventType::ProjectDiscovered,
            EnergyEventType::InvestmentReceived,
            EnergyEventType::MilestoneAchieved,
            EnergyEventType::TransitionInitiated,
            EnergyEventType::CommunityOwnershipComplete,
            EnergyEventType::ProductionUpdate,
            EnergyEventType::StatusChanged,
            EnergyEventType::SyncPending,
        ];
        for event_type in types {
            let input = BroadcastEnergyEventInput {
                event_type: event_type.clone(),
                ..valid_broadcast_energy_event_input()
            };
            assert_eq!(input.event_type, event_type);
        }
    }

    // =========================================================================
    // QueueSyncInput Tests
    // =========================================================================

    fn valid_queue_sync_input() -> QueueSyncInput {
        QueueSyncInput {
            sync_type: SyncType::ProductionMetrics,
            project_id: "project:solar_farm_alpha".to_string(),
            payload: r#"{"energy_mwh": 1500}"#.to_string(),
        }
    }

    #[test]
    fn test_queue_sync_input_valid() {
        let input = valid_queue_sync_input();
        assert!(!input.project_id.is_empty());
        assert!(!input.payload.is_empty());
    }

    #[test]
    fn test_queue_sync_input_all_types() {
        let types = vec![
            SyncType::InvestmentUpdate,
            SyncType::ProductionMetrics,
            SyncType::MilestoneProgress,
            SyncType::StatusChange,
            SyncType::TransitionProgress,
        ];
        for sync_type in types {
            let input = QueueSyncInput {
                sync_type: sync_type.clone(),
                ..valid_queue_sync_input()
            };
            assert_eq!(input.sync_type, sync_type);
        }
    }

    // =========================================================================
    // MarkSyncCompleteInput Tests
    // =========================================================================

    fn valid_mark_sync_complete_input() -> MarkSyncCompleteInput {
        MarkSyncCompleteInput {
            sync_id: "sync:ProductionMetrics:project1:123456".to_string(),
            success: true,
            error: None,
        }
    }

    #[test]
    fn test_mark_sync_complete_input_success() {
        let input = valid_mark_sync_complete_input();
        assert!(input.success);
        assert!(input.error.is_none());
    }

    #[test]
    fn test_mark_sync_complete_input_failure() {
        let input = MarkSyncCompleteInput {
            success: false,
            error: Some("Connection timeout".to_string()),
            ..valid_mark_sync_complete_input()
        };
        assert!(!input.success);
        assert!(input.error.is_some());
    }

    // =========================================================================
    // UpdateProjectStatusInput Tests
    // =========================================================================

    fn valid_update_project_status_input() -> UpdateProjectStatusInput {
        UpdateProjectStatusInput {
            project_id: "project:solar_farm_alpha".to_string(),
            terra_atlas_id: "TA-2024-001".to_string(),
            new_status: ProjectStatus::Development,
            reason: Some("Funding target reached".to_string()),
        }
    }

    #[test]
    fn test_update_project_status_input_valid() {
        let input = valid_update_project_status_input();
        assert!(!input.project_id.is_empty());
        assert!(!input.terra_atlas_id.is_empty());
    }

    #[test]
    fn test_update_project_status_input_all_statuses() {
        let statuses = vec![
            ProjectStatus::Discovery,
            ProjectStatus::Funding,
            ProjectStatus::Development,
            ProjectStatus::Operational,
            ProjectStatus::Transitioning,
            ProjectStatus::CommunityOwned,
        ];
        for status in statuses {
            let input = UpdateProjectStatusInput {
                new_status: status.clone(),
                ..valid_update_project_status_input()
            };
            assert_eq!(input.new_status, status);
        }
    }

    #[test]
    fn test_update_project_status_input_with_reason() {
        let input = valid_update_project_status_input();
        assert!(input.reason.is_some());
    }

    #[test]
    fn test_update_project_status_input_without_reason() {
        let input = UpdateProjectStatusInput {
            reason: None,
            ..valid_update_project_status_input()
        };
        assert!(input.reason.is_none());
    }

    // =========================================================================
    // InitiateTransitionInput Tests
    // =========================================================================

    fn valid_initiate_transition_input() -> InitiateTransitionInput {
        InitiateTransitionInput {
            project_id: "project:solar_farm_alpha".to_string(),
            terra_atlas_id: "TA-2024-001".to_string(),
            community_did: "did:mycelix:community1".to_string(),
            current_community_ownership: 25.0,
            target_ownership_pct: 50.0,
            reserve_account_balance: 100000,
            conditions_met: vec![
                "CommunityReadiness".to_string(),
                "FinancialSustainability".to_string(),
            ],
            conditions_pending: vec!["GovernanceMaturity".to_string()],
        }
    }

    #[test]
    fn test_initiate_transition_input_valid() {
        let input = valid_initiate_transition_input();
        assert!(input.community_did.starts_with("did:mycelix:"));
        assert!(input.target_ownership_pct > input.current_community_ownership);
    }

    #[test]
    fn test_initiate_transition_input_conditions() {
        let input = valid_initiate_transition_input();
        assert!(!input.conditions_met.is_empty());
        assert!(!input.conditions_pending.is_empty());
    }

    #[test]
    fn test_initiate_transition_input_to_100_percent() {
        let input = InitiateTransitionInput {
            current_community_ownership: 75.0,
            target_ownership_pct: 100.0,
            ..valid_initiate_transition_input()
        };
        assert_eq!(input.target_ownership_pct, 100.0);
    }

    // =========================================================================
    // CompleteTransitionInput Tests
    // =========================================================================

    fn valid_complete_transition_input() -> CompleteTransitionInput {
        CompleteTransitionInput {
            transition_id: "transition:project1:123456".to_string(),
            project_id: "project:solar_farm_alpha".to_string(),
            terra_atlas_id: "TA-2024-001".to_string(),
            community_did: "did:mycelix:community1".to_string(),
            final_ownership_pct: 100.0,
        }
    }

    #[test]
    fn test_complete_transition_input_valid() {
        let input = valid_complete_transition_input();
        assert!(!input.transition_id.is_empty());
        assert!(input.community_did.starts_with("did:mycelix:"));
    }

    #[test]
    fn test_complete_transition_input_full_ownership() {
        let input = valid_complete_transition_input();
        assert_eq!(input.final_ownership_pct, 100.0);
    }

    #[test]
    fn test_complete_transition_input_partial_ownership() {
        let input = CompleteTransitionInput {
            final_ownership_pct: 51.0,
            ..valid_complete_transition_input()
        };
        assert!(input.final_ownership_pct > 50.0);
        assert!(input.final_ownership_pct < 100.0);
    }

    // =========================================================================
    // InvestmentSummary Tests
    // =========================================================================

    #[test]
    fn test_investment_summary_creation() {
        let summary = InvestmentSummary {
            project_id: "project:solar_farm_alpha".to_string(),
            total_amount: 5_000_000,
            investor_count: 100,
            by_type: vec![
                ("Equity".to_string(), 3_000_000),
                ("Loan".to_string(), 1_500_000),
                ("Grant".to_string(), 500_000),
            ],
        };
        assert!(!summary.project_id.is_empty());
        assert!(summary.investor_count > 0);
    }

    #[test]
    fn test_investment_summary_type_breakdown() {
        let summary = InvestmentSummary {
            project_id: "project:solar_farm_alpha".to_string(),
            total_amount: 5_000_000,
            investor_count: 100,
            by_type: vec![
                ("Equity".to_string(), 3_000_000),
                ("Loan".to_string(), 1_500_000),
                ("Grant".to_string(), 500_000),
            ],
        };
        let total_by_type: u64 = summary.by_type.iter().map(|(_, v)| v).sum();
        assert_eq!(total_by_type, summary.total_amount);
    }

    #[test]
    fn test_investment_summary_empty() {
        let summary = InvestmentSummary {
            project_id: "project:new_project".to_string(),
            total_amount: 0,
            investor_count: 0,
            by_type: vec![],
        };
        assert_eq!(summary.investor_count, 0);
        assert_eq!(summary.total_amount, 0);
    }

    // =========================================================================
    // Business Logic Edge Cases
    // =========================================================================

    #[test]
    fn test_project_id_format() {
        let terra_atlas_id = "TA-2024-001";
        let timestamp = 1704067200000000_u64;
        let id = format!("project:{}:{}", terra_atlas_id, timestamp);
        assert!(id.starts_with("project:"));
    }

    #[test]
    fn test_investment_id_format() {
        let project_id = "project1";
        let investor_did = "did:mycelix:investor1";
        let timestamp = 1704067200000000_u64;
        let id = format!("invest:{}:{}:{}", project_id, investor_did, timestamp);
        assert!(id.starts_with("invest:"));
    }

    #[test]
    fn test_milestone_id_format() {
        let project_id = "project1";
        let milestone_type = MilestoneType::CommunityFormation;
        let timestamp = 1704067200000000_u64;
        let id = format!(
            "milestone:{}:{:?}:{}",
            project_id, milestone_type, timestamp
        );
        assert!(id.starts_with("milestone:"));
    }

    #[test]
    fn test_event_id_format() {
        let event_type = EnergyEventType::InvestmentReceived;
        let timestamp = 1704067200000000_u64;
        let id = format!("event:{:?}:{}", event_type, timestamp);
        assert!(id.starts_with("event:"));
    }

    #[test]
    fn test_sync_id_format() {
        let sync_type = SyncType::ProductionMetrics;
        let project_id = "project1";
        let timestamp = 1704067200000000_u64;
        let id = format!("sync:{:?}:{}:{}", sync_type, project_id, timestamp);
        assert!(id.starts_with("sync:"));
    }

    #[test]
    fn test_transition_id_format() {
        let project_id = "project1";
        let timestamp = 1704067200000000_u64;
        let id = format!("transition:{}:{}", project_id, timestamp);
        assert!(id.starts_with("transition:"));
    }

    #[test]
    fn test_json_payload_creation() {
        let payload = serde_json::json!({
            "amount": 50000,
            "investor": "did:mycelix:investor1",
        });
        let payload_str = payload.to_string();
        assert!(!payload_str.is_empty());
    }

    #[test]
    fn test_deserialization_roundtrip() {
        let input = valid_sync_project_input();
        let json = serde_json::to_string(&input).unwrap();
        let deserialized: SyncProjectInput = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.terra_atlas_id, input.terra_atlas_id);
        assert_eq!(deserialized.capacity_mw, input.capacity_mw);
    }

    #[test]
    fn test_production_metrics_json_payload() {
        let input = valid_record_production_input();
        let payload = serde_json::json!({
            "terra_atlas_id": input.terra_atlas_id,
            "period_start": input.period_start.as_micros(),
            "period_end": input.period_end.as_micros(),
            "energy_generated_mwh": input.energy_generated_mwh,
            "capacity_factor": input.capacity_factor,
            "revenue_generated": input.revenue_generated,
            "currency": input.currency,
        });
        assert!(!payload.to_string().is_empty());
    }

    #[test]
    fn test_energy_happ_id_constant() {
        assert_eq!(ENERGY_HAPP_ID, "mycelix-energy");
    }
}
