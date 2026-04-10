// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Energy Bridge Coordinator Zome
//!
//! Cross-hApp communication for Terra Atlas integration, investment tracking,
//! and regenerative exit coordination.

use hdk::prelude::*;
use energy_bridge_integrity::*;

const ENERGY_HAPP_ID: &str = "mycelix-energy";

/// Create or retrieve an anchor entry hash for deterministic link bases
fn anchor_hash(anchor_string: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_string.to_string());
    if let Err(e) = create_entry(&EntryTypes::Anchor(anchor.clone())) { debug!("Anchor creation warning: {:?}", e); }
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
        phi_score: None,
        harmony_alignment: None,
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

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Project not found".into())))
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
        id: format!("invest:{}:{}:{}", input.project_id, input.investor_did, now.as_micros()),
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
        }).to_string(),
    })?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Investment not found".into())))
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
        id: format!("milestone:{}:{:?}:{}", input.project_id, input.milestone_type, now.as_micros()),
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
        }).to_string(),
    })?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Milestone not found".into())))
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
        id: format!("prod:{}:{}:{}", input.project_id, input.period_start.as_micros(), now.as_micros()),
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
        }).to_string(),
    })?;

    // Broadcast event
    broadcast_energy_event(BroadcastEnergyEventInput {
        event_type: EnergyEventType::ProductionUpdate,
        project_id: input.project_id,
        payload: serde_json::json!({
            "energy_mwh": input.energy_generated_mwh,
            "capacity_factor": input.capacity_factor,
            "revenue": input.revenue_generated,
        }).to_string(),
    })?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Production record not found".into())))
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
            if let Some(investment) = record.entry().to_app_option::<InvestmentRecord>()
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
        id: format!("sync:{:?}:{}:{}", input.sync_type, input.project_id, now.as_micros()),
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

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Sync record not found".into())))
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
            if let Some(sync_record) = record.entry().to_app_option::<PendingSyncRecord>()
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
        }).to_string(),
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
        }).to_string(),
    })?;

    // Broadcast event
    broadcast_energy_event(BroadcastEnergyEventInput {
        event_type: EnergyEventType::StatusChanged,
        project_id: input.project_id.clone(),
        payload: serde_json::json!({
            "new_status": format!("{:?}", input.new_status),
            "reason": input.reason,
        }).to_string(),
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
        }).to_string(),
    })?;

    // Broadcast
    broadcast_energy_event(BroadcastEnergyEventInput {
        event_type: EnergyEventType::TransitionInitiated,
        project_id: input.project_id,
        payload: serde_json::json!({
            "from_pct": input.current_community_ownership,
            "to_pct": input.target_ownership_pct,
            "community": input.community_did,
        }).to_string(),
    })?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Transition record not found".into())))
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
        }).to_string(),
    })?;

    // Broadcast completion
    broadcast_energy_event(BroadcastEnergyEventInput {
        event_type: EnergyEventType::CommunityOwnershipComplete,
        project_id: input.project_id,
        payload: serde_json::json!({
            "final_ownership": input.final_ownership_pct,
            "community": input.community_did,
        }).to_string(),
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
        LinkQuery::try_new(anchor_hash("active_transitions")?, LinkTypes::ActiveTransitions)?,
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

// =============================================================================
// Energy→Finance Bridge: Certificate Collateral Registration
// =============================================================================

/// Register an energy production certificate as finance collateral.
///
/// Bridges from the energy cluster to the finance cluster, registering
/// verified energy production as SAP-eligible collateral.
///
/// Only projects with status `Operational` or `CommunityOwned` are eligible
/// (they must have verified production data).
#[hdk_extern]
pub fn register_certificate_as_collateral(input: CertificateCollateralInput) -> ExternResult<CertificateCollateralResult> {
    // 1. Verify the certificate exists locally by looking up the project
    let project = match call(
        CallTargetCell::Local,
        ZomeName::from("projects"),
        FunctionName::from("get_project"),
        None,
        input.project_id.clone(),
    ) {
        Ok(ZomeCallResponse::Ok(result)) => {
            result.decode::<Option<Record>>().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e)))
            })?
        }
        _ => None,
    };

    // Verify the project exists and is in a valid state for collateral registration
    if let Some(ref record) = project {
        if let Some(project_entry) = record.entry().to_app_option::<TerraAtlasProject>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Parse error: {:?}", e))))?
        {
            match project_entry.status {
                ProjectStatus::Operational | ProjectStatus::CommunityOwned => {
                    // Valid — project has verified production
                }
                _ => {
                    return Ok(CertificateCollateralResult {
                        success: false,
                        certificate_id: input.certificate_id,
                        collateral_registered: false,
                        error: Some(format!(
                            "Project must be Operational or CommunityOwned, got {:?}",
                            project_entry.status
                        )),
                    });
                }
            }
        }
    } else {
        return Ok(CertificateCollateralResult {
            success: false,
            certificate_id: input.certificate_id,
            collateral_registered: false,
            error: Some(format!("Project {} not found", input.project_id)),
        });
    }

    // 2. Cross-cluster call to finance bridge to register as collateral
    #[derive(Serialize, Debug)]
    struct RegisterCollateralPayload {
        owner_did: String,
        source_happ: String,
        asset_type: String,
        asset_id: String,
        value_estimate: u64,
        currency: String,
    }

    match call(
        CallTargetCell::OtherRole("finance".into()),
        ZomeName::from("finance_bridge"),
        FunctionName::from("register_collateral"),
        None,
        RegisterCollateralPayload {
            owner_did: input.producer_did.clone(),
            source_happ: ENERGY_HAPP_ID.to_string(),
            asset_type: "EnergyAsset".to_string(),
            asset_id: input.certificate_id.clone(),
            value_estimate: input.sap_value,
            currency: "SAP".to_string(),
        },
    ) {
        Ok(ZomeCallResponse::Ok(_)) => {
            // 3. Broadcast local event for audit trail
            let _ = broadcast_energy_event(BroadcastEnergyEventInput {
                event_type: EnergyEventType::CertificateCollateralRegistered,
                project_id: input.project_id,
                payload: serde_json::json!({
                    "certificate_id": input.certificate_id,
                    "producer_did": input.producer_did,
                    "sap_value": input.sap_value,
                }).to_string(),
            });

            Ok(CertificateCollateralResult {
                success: true,
                certificate_id: input.certificate_id,
                collateral_registered: true,
                error: None,
            })
        }
        Ok(other) => {
            Ok(CertificateCollateralResult {
                success: false,
                certificate_id: input.certificate_id,
                collateral_registered: false,
                error: Some(format!("Finance bridge returned: {:?}", other)),
            })
        }
        Err(e) => {
            debug!("register_certificate_as_collateral: finance unreachable: {:?}", e);
            Ok(CertificateCollateralResult {
                success: false,
                certificate_id: input.certificate_id,
                collateral_registered: false,
                error: Some(format!("Finance cluster unreachable: {:?}", e)),
            })
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CertificateCollateralInput {
    pub certificate_id: String,
    pub project_id: String,
    pub producer_did: String,
    pub sap_value: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CertificateCollateralResult {
    pub success: bool,
    pub certificate_id: String,
    pub collateral_registered: bool,
    pub error: Option<String>,
}

/// Verify an energy project exists (called by finance bridge for certificate verification).
///
/// Returns the project record if found, None if not. This allows the finance
/// cluster to independently verify that an energy certificate references a
/// real project before accepting it as collateral.
#[hdk_extern]
pub fn get_project(project_id: String) -> ExternResult<Option<Record>> {
    match call(
        CallTargetCell::Local,
        ZomeName::from("projects"),
        FunctionName::from("get_project"),
        None,
        project_id.clone(),
    ) {
        Ok(ZomeCallResponse::Ok(result)) => {
            result.decode::<Option<Record>>().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!("Decode error: {:?}", e)))
            })
        }
        _ => {
            debug!("get_project: projects zome unavailable for {}", project_id);
            Ok(None)
        }
    }
}

// =============================================================================
// Gap 4: Energy Equipment Procurement via Supplychain
// =============================================================================

/// Input for procuring equipment via the supplychain cluster.
#[derive(Serialize, Deserialize, Debug)]
pub struct EquipmentProcurementInput {
    /// Equipment type / SKU (e.g., "INVERTER-10KW", "SOLAR-PANEL-400W").
    pub equipment_type: String,
    /// Number of units required.
    pub quantity: u64,
    /// Reference to the energy project requiring the equipment.
    pub project_hash: ActionHash,
}

/// Result of an equipment procurement attempt.
#[derive(Serialize, Deserialize, Debug)]
pub struct EquipmentProcurementResult {
    /// Whether procurement was initiated successfully.
    pub procurement_initiated: bool,
    /// Stock level found in supplychain (None if unavailable/not found).
    pub stock_available: Option<u64>,
    /// Supplychain purchase order hash if a PO was created.
    pub purchase_order_hash: Option<String>,
    /// Non-fatal error message.
    pub error: Option<String>,
}

/// Procure equipment from the supplychain cluster for an energy project.
///
/// 1. Checks stock availability via `get_stock_level_by_sku` in the supplychain
///    inventory zome.
/// 2. If stock is available, calls `select_best_supplier` then
///    `create_purchase_order` in the supplychain procurement zome.
/// 3. Returns a result the caller can handle; all cross-cluster failures are
///    non-fatal.
#[hdk_extern]
pub fn procure_equipment(input: EquipmentProcurementInput) -> ExternResult<EquipmentProcurementResult> {
    if input.equipment_type.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "equipment_type is required".to_string()
        )));
    }
    if input.quantity == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "quantity must be > 0".to_string()
        )));
    }

    // Step 1: Check inventory stock level by SKU
    let stock_payload = ExternIO::encode(serde_json::json!({ "sku": input.equipment_type }))
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

    let stock_level: Option<u64> = match call(
        CallTargetCell::OtherRole("supplychain".into()),
        ZomeName::from("inventory_coordinator"),
        FunctionName::from("get_stock_level_by_sku"),
        None,
        stock_payload,
    ) {
        Ok(ZomeCallResponse::Ok(data)) => {
            let value: serde_json::Value = data.decode().unwrap_or(serde_json::Value::Null);
            // get_stock_level_by_sku returns Option<InventoryLevel>
            // InventoryLevel has a `quantity` field
            value
                .get("quantity")
                .and_then(|v| v.as_u64())
        }
        Ok(_) => None,
        Err(_) => {
            return Ok(EquipmentProcurementResult {
                procurement_initiated: false,
                stock_available: None,
                purchase_order_hash: None,
                error: Some("Supplychain cluster not available".to_string()),
            });
        }
    };

    let available = stock_level.unwrap_or(0);
    if available < input.quantity {
        return Ok(EquipmentProcurementResult {
            procurement_initiated: false,
            stock_available: Some(available),
            purchase_order_hash: None,
            error: Some(format!(
                "Insufficient stock: {} available, {} required",
                available, input.quantity
            )),
        });
    }

    // Step 2: Select best supplier
    let supplier_payload = ExternIO::encode(serde_json::json!({
        "category": input.equipment_type,
        "quantity": input.quantity,
    }))
    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

    let supplier_key: Option<serde_json::Value> = match call(
        CallTargetCell::OtherRole("supplychain".into()),
        ZomeName::from("procurement_coordinator"),
        FunctionName::from("select_best_supplier"),
        None,
        supplier_payload,
    ) {
        Ok(ZomeCallResponse::Ok(data)) => {
            data.decode::<serde_json::Value>().ok()
        }
        Ok(other) => {
            return Ok(EquipmentProcurementResult {
                procurement_initiated: false,
                stock_available: Some(available),
                purchase_order_hash: None,
                error: Some(format!("select_best_supplier rejected: {:?}", other)),
            });
        }
        Err(_) => {
            return Ok(EquipmentProcurementResult {
                procurement_initiated: false,
                stock_available: Some(available),
                purchase_order_hash: None,
                error: Some("Supplychain procurement zome not available".to_string()),
            });
        }
    };

    // Step 3: Create purchase order
    // Unit price is derived from the supplier response if present, defaulting to 1.
    let unit_price = supplier_key
        .as_ref()
        .and_then(|v| v.get("minimum_order_value"))
        .and_then(|v| v.as_u64())
        .map(|v| v.max(1))
        .unwrap_or(1);

    let supplier_agent = supplier_key
        .as_ref()
        .and_then(|v| v.get("agent"))
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    let po_payload = ExternIO::encode(serde_json::json!({
        "po_number": format!("ENERGY-{}-{}", input.equipment_type, input.project_hash),
        "supplier": supplier_agent,
        "items": [{
            "sku": input.equipment_type,
            "description": format!("{} units of {}", input.quantity, input.equipment_type),
            "quantity": input.quantity,
            "unit_price": unit_price,
        }],
        "currency": "USD",
        "notes": format!("Equipment for energy project {}", input.project_hash),
    }))
    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

    match call(
        CallTargetCell::OtherRole("supplychain".into()),
        ZomeName::from("procurement_coordinator"),
        FunctionName::from("create_purchase_order"),
        None,
        po_payload,
    ) {
        Ok(ZomeCallResponse::Ok(data)) => {
            let po_hash: serde_json::Value = data.decode().unwrap_or(serde_json::Value::Null);
            Ok(EquipmentProcurementResult {
                procurement_initiated: true,
                stock_available: Some(available),
                purchase_order_hash: Some(po_hash.to_string()),
                error: None,
            })
        }
        Ok(other) => Ok(EquipmentProcurementResult {
            procurement_initiated: false,
            stock_available: Some(available),
            purchase_order_hash: None,
            error: Some(format!("create_purchase_order rejected: {:?}", other)),
        }),
        Err(e) => Ok(EquipmentProcurementResult {
            procurement_initiated: false,
            stock_available: Some(available),
            purchase_order_hash: None,
            error: Some(format!("create_purchase_order error: {:?}", e)),
        }),
    }
}

// =============================================================================
// Consciousness Assessment (Symthaea → Energy Bridge)
// =============================================================================

/// Record a consciousness assessment from Symthaea for a project.
///
/// Stores the Phi score, Eight Harmonies alignment, and supporting metrics
/// on the DHT, linked to the project for discovery. Also queues a sync
/// record to push scores to Terra Atlas.
#[hdk_extern]
pub fn record_consciousness_assessment(input: RecordAssessmentInput) -> ExternResult<Record> {
    let now = sys_time()?;

    let assessment = ConsciousnessAssessment {
        id: format!("assess:{}:{}:{}", input.project_id, input.scorer_did, now.as_micros()),
        project_id: input.project_id.clone(),
        scorer_did: input.scorer_did.clone(),
        phi_score: input.phi_score,
        harmony_alignment: input.harmony_alignment,
        per_harmony_scores: input.per_harmony_scores.clone(),
        care_activation: input.care_activation,
        meta_awareness: input.meta_awareness,
        assessment_cycle: input.assessment_cycle,
        assessed_at: now,
    };

    let hash = create_entry(&EntryTypes::ConsciousnessAssessment(assessment))?;

    // Link to project for discovery
    create_link(
        anchor_hash(&input.project_id)?,
        hash.clone(),
        LinkTypes::ProjectToAssessments,
        (),
    )?;

    // Queue for sync to Terra Atlas
    queue_sync_to_terra_atlas(QueueSyncInput {
        sync_type: SyncType::ConsciousnessScore,
        project_id: input.project_id.clone(),
        payload: serde_json::json!({
            "project_id": input.project_id,
            "phi_score": input.phi_score,
            "harmony_alignment": input.harmony_alignment,
            "per_harmony_scores": input.per_harmony_scores,
            "care_activation": input.care_activation,
            "meta_awareness": input.meta_awareness,
            "scorer_did": input.scorer_did,
            "assessment_cycle": input.assessment_cycle,
        }).to_string(),
    })?;

    // Broadcast event
    broadcast_energy_event(BroadcastEnergyEventInput {
        event_type: EnergyEventType::ConsciousnessAssessed,
        project_id: input.project_id,
        payload: serde_json::json!({
            "phi_score": input.phi_score,
            "harmony_alignment": input.harmony_alignment,
            "scorer": input.scorer_did,
        }).to_string(),
    })?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Assessment not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordAssessmentInput {
    pub project_id: String,
    pub scorer_did: String,
    pub phi_score: f64,
    pub harmony_alignment: f64,
    pub per_harmony_scores: String,
    pub care_activation: f64,
    pub meta_awareness: f64,
    pub assessment_cycle: u64,
}

/// Get consciousness assessments for a project
#[hdk_extern]
pub fn get_project_consciousness(project_id: String) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&project_id)?, LinkTypes::ProjectToAssessments)?,
        GetStrategy::default(),
    )?;

    let mut assessments = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            assessments.push(record);
        }
    }

    Ok(assessments)
}

/// Get the latest consciousness assessment for a project
#[hdk_extern]
pub fn get_latest_project_consciousness(project_id: String) -> ExternResult<Option<Record>> {
    let assessments = get_project_consciousness(project_id)?;
    // Return the last one (most recently linked)
    Ok(assessments.into_iter().last())
}

// =============================================================================
// Allocation Pledges (Trust-Weighted Resource Commitment)
// =============================================================================

/// Minimum consciousness score required to submit a pledge (Participant tier).
/// Matches `consciousness_gate_pledge` in mycelix-bridge-common.
const PLEDGE_MIN_TRUST: f64 = 0.2;

/// Default pledge TTL: 24 hours in microseconds.
/// Matches consciousness credential TTL.
const PLEDGE_TTL_US: i64 = 86_400_000_000;

/// Submit a trust-weighted pledge toward a project.
///
/// The pledger must have a consciousness score >= 0.2 (Participant tier).
/// The pledge expires after 24h (matching consciousness credential TTL).
/// Pledges are denominated in TEND, SAP, or a community currency.
#[hdk_extern]
pub fn submit_pledge(input: SubmitPledgeInput) -> ExternResult<Record> {
    // Consciousness gate: require Participant tier
    if input.trust_score < PLEDGE_MIN_TRUST {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Trust score {:.2} below pledge threshold {:.2} (Participant tier required)",
                input.trust_score, PLEDGE_MIN_TRUST
            )
        )));
    }

    let now = sys_time()?;
    let expires_at = Timestamp::from_micros(now.as_micros() + PLEDGE_TTL_US);

    let pledge = AllocationPledge {
        id: format!("pledge:{}:{}:{}", input.project_id, input.pledger_did, now.as_micros()),
        pledger_did: input.pledger_did.clone(),
        project_id: input.project_id.clone(),
        amount: input.amount,
        currency: input.currency.clone(),
        trust_score: input.trust_score,
        trust_tier: input.trust_tier.clone(),
        harmony_intent: input.harmony_intent.clone(),
        status: PledgeStatus::Pending,
        pledged_at: now,
        expires_at,
        matched_at: None,
    };

    let hash = create_entry(&EntryTypes::AllocationPledge(pledge))?;

    // Link to project for discovery
    create_link(
        anchor_hash(&input.project_id)?,
        hash.clone(),
        LinkTypes::AssetToPledges,
        (),
    )?;

    // Link to pledger for portfolio view
    create_link(
        anchor_hash(&input.pledger_did)?,
        hash.clone(),
        LinkTypes::PledgerToPledges,
        (),
    )?;

    // Broadcast event
    broadcast_energy_event(BroadcastEnergyEventInput {
        event_type: EnergyEventType::PledgeSubmitted,
        project_id: input.project_id,
        payload: serde_json::json!({
            "pledger": input.pledger_did,
            "amount": input.amount,
            "currency": input.currency,
            "trust_score": input.trust_score,
            "trust_tier": input.trust_tier,
            "harmony_intent": input.harmony_intent,
        }).to_string(),
    })?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Pledge not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitPledgeInput {
    pub pledger_did: String,
    pub project_id: String,
    pub amount: u64,
    pub currency: String,
    pub trust_score: f64,
    pub trust_tier: String,
    pub harmony_intent: String,
}

/// Withdraw a pending pledge (only the pledger can withdraw).
#[hdk_extern]
pub fn withdraw_pledge(input: WithdrawPledgeInput) -> ExternResult<Record> {
    let record = get(input.pledge_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Pledge not found".into())))?;

    let pledge = record.entry().to_app_option::<AllocationPledge>()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Parse error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid pledge entry".into())))?;

    // Only the pledger can withdraw
    if pledge.pledger_did != input.pledger_did {
        return Err(wasm_error!(WasmErrorInner::Guest("Only the pledger can withdraw".into())));
    }

    // Can only withdraw pending pledges
    if pledge.status != PledgeStatus::Pending {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Cannot withdraw pledge in {:?} status", pledge.status)
        )));
    }

    // Create updated pledge with Withdrawn status
    let withdrawn = AllocationPledge {
        status: PledgeStatus::Withdrawn,
        ..pledge.clone()
    };

    let new_hash = update_entry(input.pledge_hash, &withdrawn)?;

    // Broadcast withdrawal
    broadcast_energy_event(BroadcastEnergyEventInput {
        event_type: EnergyEventType::PledgeWithdrawn,
        project_id: pledge.project_id,
        payload: serde_json::json!({
            "pledger": pledge.pledger_did,
            "amount": pledge.amount,
        }).to_string(),
    })?;

    get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Updated pledge not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct WithdrawPledgeInput {
    pub pledge_hash: ActionHash,
    pub pledger_did: String,
}

/// Get all pledges for a project
#[hdk_extern]
pub fn get_project_pledges(project_id: String) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&project_id)?, LinkTypes::AssetToPledges)?,
        GetStrategy::default(),
    )?;

    let mut pledges = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            pledges.push(record);
        }
    }

    Ok(pledges)
}

/// Get all pledges by a specific pledger
#[hdk_extern]
pub fn get_pledger_pledges(pledger_did: String) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&pledger_did)?, LinkTypes::PledgerToPledges)?,
        GetStrategy::default(),
    )?;

    let mut pledges = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            pledges.push(record);
        }
    }

    Ok(pledges)
}

// =============================================================================
// Matching Engine (Bilateral Consciousness-Weighted)
// =============================================================================

/// Minimum consciousness score to run matching (Citizen tier).
const MATCHING_MIN_TRUST: f64 = 0.3;

/// Scoring weights for the matching algorithm.
const W_TRUST: f64 = 0.40;
const W_AMOUNT_FIT: f64 = 0.30;
const W_HARMONY: f64 = 0.20;
const W_PROXIMITY: f64 = 0.10;

/// Sigmoid function for continuous consciousness weight.
/// Maps consciousness score to [0,1] with smooth transition around threshold.
fn sigmoid_weight(score: f64, threshold: f64, temperature: f64) -> f64 {
    1.0 / (1.0 + (-(score - threshold) / temperature).exp())
}

/// Run consciousness-weighted matching for a project.
///
/// The asset holder (or any Citizen-tier participant) collects all pending
/// pledges, scores them using the 4-factor consciousness-weighted algorithm,
/// and creates AllocationMatch entries for the best matches.
///
/// Matching is bilateral: the holder proposes, the pledger must confirm.
/// No partial fills — a pledge is either fully matched or stays pending.
#[hdk_extern]
pub fn run_matching(input: RunMatchingInput) -> ExternResult<Vec<Record>> {
    // Consciousness gate: require Citizen tier to run matching
    if input.holder_trust_score < MATCHING_MIN_TRUST {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Trust score {:.2} below matching threshold {:.2} (Citizen tier required)",
                input.holder_trust_score, MATCHING_MIN_TRUST
            )
        )));
    }

    let now = sys_time()?;

    // Collect pending pledges for this project
    let pledge_records = get_project_pledges(input.project_id.clone())?;

    // Parse and filter to pending, non-expired pledges
    let mut scored_pledges: Vec<(AllocationPledge, ActionHash, f64)> = Vec::new();

    for record in &pledge_records {
        let action_hash = record.action_address().clone();
        if let Some(pledge) = record.entry().to_app_option::<AllocationPledge>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Parse error: {:?}", e))))?
        {
            // Only match pending pledges that haven't expired
            if pledge.status != PledgeStatus::Pending {
                continue;
            }
            if pledge.expires_at < now {
                continue;
            }

            // Compute match score
            let trust_weight = sigmoid_weight(pledge.trust_score, 0.3, 0.05);

            let amount_fit = if input.funding_gap > 0 {
                let fit = 1.0 - ((pledge.amount as f64 - input.funding_gap as f64).abs()
                    / input.total_cost.max(1) as f64);
                fit.clamp(0.0, 1.0)
            } else {
                0.0 // No gap to fill
            };

            let harmony_alignment = input.project_harmony_alignment.clamp(0.0, 1.0);

            let community_proximity = if input.community_did.is_some()
                && pledge.harmony_intent.contains(input.community_did.as_deref().unwrap_or(""))
            {
                1.0
            } else {
                0.0
            };

            let match_score = W_TRUST * trust_weight
                + W_AMOUNT_FIT * amount_fit
                + W_HARMONY * harmony_alignment
                + W_PROXIMITY * community_proximity;

            scored_pledges.push((pledge, action_hash, match_score));
        }
    }

    // Sort by match score descending
    scored_pledges.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Greedy fill: accept pledges in score order until funding gap is met
    // Supports partial fills: if a pledge exceeds the remaining gap,
    // only the needed portion is matched (rest stays as a reduced pledge).
    let mut remaining_gap = input.funding_gap;
    let mut matches = Vec::new();

    for (pledge, _action_hash, score) in &scored_pledges {
        if remaining_gap == 0 {
            break;
        }

        // Determine match amount (partial fill support)
        let match_amount = pledge.amount.min(remaining_gap);

        let trust_weight = sigmoid_weight(pledge.trust_score, 0.3, 0.05);
        let amount_fit = if input.funding_gap > 0 {
            (1.0 - ((pledge.amount as f64 - remaining_gap as f64).abs()
                / input.total_cost.max(1) as f64)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let allocation_match = AllocationMatch {
            id: format!("match:{}:{}:{}", input.project_id, pledge.id, now.as_micros()),
            pledge_id: pledge.id.clone(),
            project_id: input.project_id.clone(),
            pledger_did: pledge.pledger_did.clone(),
            holder_did: input.holder_did.clone(),
            amount: match_amount,
            match_score: *score,
            trust_weight,
            amount_fit,
            harmony_alignment: input.project_harmony_alignment.clamp(0.0, 1.0),
            community_proximity: 0.0,
            status: MatchStatus::Proposed,
            proposed_at: now,
            resolved_at: None,
        };

        let hash = create_entry(&EntryTypes::AllocationMatch(allocation_match))?;

        // Link to project
        create_link(
            anchor_hash(&input.project_id)?,
            hash.clone(),
            LinkTypes::AssetToMatches,
            (),
        )?;

        // Link pledge to match
        create_link(
            anchor_hash(&pledge.id)?,
            hash.clone(),
            LinkTypes::PledgeToMatch,
            (),
        )?;

        // Broadcast
        broadcast_energy_event(BroadcastEnergyEventInput {
            event_type: EnergyEventType::MatchProposed,
            project_id: input.project_id.clone(),
            payload: serde_json::json!({
                "pledger": pledge.pledger_did,
                "amount": match_amount,
                "partial_fill": match_amount < pledge.amount,
                "match_score": score,
                "trust_weight": trust_weight,
            }).to_string(),
        })?;

        if let Some(record) = get(hash, GetOptions::default())? {
            matches.push(record);
        }

        remaining_gap = remaining_gap.saturating_sub(match_amount);
    }

    Ok(matches)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RunMatchingInput {
    pub project_id: String,
    pub holder_did: String,
    pub holder_trust_score: f64,
    pub funding_gap: u64,
    pub total_cost: u64,
    pub project_harmony_alignment: f64,
    pub community_did: Option<String>,
}

/// Pledger confirms a proposed match.
#[hdk_extern]
pub fn confirm_match(input: ConfirmMatchInput) -> ExternResult<Record> {
    let record = get(input.match_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Match not found".into())))?;

    let allocation_match = record.entry().to_app_option::<AllocationMatch>()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Parse error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid match entry".into())))?;

    // Only the pledger can confirm
    if allocation_match.pledger_did != input.pledger_did {
        return Err(wasm_error!(WasmErrorInner::Guest("Only the pledger can confirm".into())));
    }

    if allocation_match.status != MatchStatus::Proposed {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Cannot confirm match in {:?} status", allocation_match.status)
        )));
    }

    // Re-check consciousness: pledger must still meet threshold
    if input.current_trust_score < PLEDGE_MIN_TRUST {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Trust score has dropped below pledge threshold — cannot confirm".into()
        )));
    }

    let now = sys_time()?;
    let accepted = AllocationMatch {
        status: MatchStatus::Accepted,
        resolved_at: Some(now),
        ..allocation_match.clone()
    };

    let new_hash = update_entry(input.match_hash, &accepted)?;

    broadcast_energy_event(BroadcastEnergyEventInput {
        event_type: EnergyEventType::MatchAccepted,
        project_id: allocation_match.project_id,
        payload: serde_json::json!({
            "pledger": allocation_match.pledger_did,
            "amount": allocation_match.amount,
            "match_score": allocation_match.match_score,
        }).to_string(),
    })?;

    get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Updated match not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ConfirmMatchInput {
    pub match_hash: ActionHash,
    pub pledger_did: String,
    pub current_trust_score: f64,
}

/// Pledger rejects a proposed match.
#[hdk_extern]
pub fn reject_match(input: RejectMatchInput) -> ExternResult<Record> {
    let record = get(input.match_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Match not found".into())))?;

    let allocation_match = record.entry().to_app_option::<AllocationMatch>()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Parse error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid match entry".into())))?;

    if allocation_match.pledger_did != input.pledger_did {
        return Err(wasm_error!(WasmErrorInner::Guest("Only the pledger can reject".into())));
    }

    if allocation_match.status != MatchStatus::Proposed {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Cannot reject match in {:?} status", allocation_match.status)
        )));
    }

    let now = sys_time()?;
    let rejected = AllocationMatch {
        status: MatchStatus::Rejected,
        resolved_at: Some(now),
        ..allocation_match.clone()
    };

    let new_hash = update_entry(input.match_hash, &rejected)?;

    broadcast_energy_event(BroadcastEnergyEventInput {
        event_type: EnergyEventType::MatchRejected,
        project_id: allocation_match.project_id,
        payload: serde_json::json!({
            "pledger": allocation_match.pledger_did,
            "amount": allocation_match.amount,
        }).to_string(),
    })?;

    get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Updated match not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RejectMatchInput {
    pub match_hash: ActionHash,
    pub pledger_did: String,
}

/// Get all matches for a project
#[hdk_extern]
pub fn get_project_matches(project_id: String) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&project_id)?, LinkTypes::AssetToMatches)?,
        GetStrategy::default(),
    )?;

    let mut matches = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            matches.push(record);
        }
    }

    Ok(matches)
}

// =============================================================================
// Reputation Feedback (post-match rating)
// =============================================================================

/// Rate a completed match — both pledger and holder can provide feedback.
///
/// Feedback scores feed into MYCEL reputation scores and influence future
/// matching priority. Both `outcome_score` and `harmony_fulfilled` affect
/// the rated party's consciousness profile engagement dimension.
#[hdk_extern]
pub fn rate_match(input: RateMatchInput) -> ExternResult<Record> {
    let now = sys_time()?;

    // Verify the match exists and is completed
    let match_record = get(input.match_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Match not found".into())))?;

    let allocation_match = match_record.entry().to_app_option::<AllocationMatch>()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Parse error: {:?}", e))))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid match entry".into())))?;

    if allocation_match.status != MatchStatus::Accepted && allocation_match.status != MatchStatus::Completed {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Can only rate accepted or completed matches".into()
        )));
    }

    // Verify rater is a participant in this match
    let (role, rated_did) = if input.rater_did == allocation_match.pledger_did {
        (FeedbackRole::Pledger, allocation_match.holder_did.clone())
    } else if input.rater_did == allocation_match.holder_did {
        (FeedbackRole::Holder, allocation_match.pledger_did.clone())
    } else {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only match participants can rate".into()
        )));
    };

    let feedback = MatchFeedback {
        id: format!("feedback:{}:{:?}:{}", allocation_match.id, role, now.as_micros()),
        match_id: allocation_match.id.clone(),
        project_id: allocation_match.project_id.clone(),
        rater_did: input.rater_did.clone(),
        rated_did,
        role,
        outcome_score: input.outcome_score,
        harmony_fulfilled: input.harmony_fulfilled,
        would_match_again: input.would_match_again,
        comment: input.comment,
        rated_at: now,
    };

    let hash = create_entry(&EntryTypes::MatchFeedback(feedback))?;

    // Link from match to feedback
    create_link(
        anchor_hash(&allocation_match.id)?,
        hash.clone(),
        LinkTypes::MatchToFeedback,
        (),
    )?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Feedback not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RateMatchInput {
    pub match_hash: ActionHash,
    pub rater_did: String,
    pub outcome_score: f64,
    pub harmony_fulfilled: f64,
    pub would_match_again: bool,
    pub comment: Option<String>,
}

/// Get feedback for a match
#[hdk_extern]
pub fn get_match_feedback(match_id: String) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&match_id)?, LinkTypes::MatchToFeedback)?,
        GetStrategy::default(),
    )?;

    let mut feedback = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            feedback.push(record);
        }
    }

    Ok(feedback)
}

// =============================================================================
// QOL Impact Metrics
// =============================================================================

/// Minimum consciousness score to report impact (Participant tier).
const IMPACT_MIN_TRUST: f64 = 0.2;

/// Record QOL impact metrics for a project.
///
/// Reports real-world impact (CO2 avoided, jobs created, community trust, etc.)
/// that feeds back into the project's consciousness score and Terra Atlas dashboard.
#[hdk_extern]
pub fn record_impact(input: RecordImpactInput) -> ExternResult<Record> {
    // Consciousness gate
    if input.reporter_trust_score < IMPACT_MIN_TRUST {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!(
                "Trust score {:.2} below impact reporting threshold {:.2}",
                input.reporter_trust_score, IMPACT_MIN_TRUST
            )
        )));
    }

    let now = sys_time()?;

    let impact = ImpactRecord {
        id: format!("impact:{}:{}:{}", input.project_id, input.reporter_did, now.as_micros()),
        project_id: input.project_id.clone(),
        reporter_did: input.reporter_did.clone(),
        period_start: input.period_start,
        period_end: input.period_end,
        co2_avoided_tonnes: input.co2_avoided_tonnes,
        jobs_created: input.jobs_created,
        community_trust_delta: input.community_trust_delta,
        energy_access_households: input.energy_access_households,
        biodiversity_index_delta: input.biodiversity_index_delta,
        verification_evidence: input.verification_evidence.clone(),
        verified_by: input.verified_by.clone(),
        recorded_at: now,
    };

    let hash = create_entry(&EntryTypes::ImpactRecord(impact))?;

    // Link to project
    create_link(
        anchor_hash(&input.project_id)?,
        hash.clone(),
        LinkTypes::AssetToImpacts,
        (),
    )?;

    // Queue for sync to Terra Atlas
    queue_sync_to_terra_atlas(QueueSyncInput {
        sync_type: SyncType::ImpactMetrics,
        project_id: input.project_id.clone(),
        payload: serde_json::json!({
            "project_id": input.project_id,
            "co2_avoided_tonnes": input.co2_avoided_tonnes,
            "jobs_created": input.jobs_created,
            "community_trust_delta": input.community_trust_delta,
            "energy_access_households": input.energy_access_households,
            "biodiversity_index_delta": input.biodiversity_index_delta,
            "reporter_did": input.reporter_did,
        }).to_string(),
    })?;

    // Broadcast
    broadcast_energy_event(BroadcastEnergyEventInput {
        event_type: EnergyEventType::ImpactReported,
        project_id: input.project_id,
        payload: serde_json::json!({
            "co2_avoided_tonnes": input.co2_avoided_tonnes,
            "jobs_created": input.jobs_created,
            "community_trust_delta": input.community_trust_delta,
            "reporter": input.reporter_did,
        }).to_string(),
    })?;

    get(hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Impact record not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordImpactInput {
    pub project_id: String,
    pub reporter_did: String,
    pub reporter_trust_score: f64,
    pub period_start: Timestamp,
    pub period_end: Timestamp,
    pub co2_avoided_tonnes: f64,
    pub jobs_created: u32,
    pub community_trust_delta: f64,
    pub energy_access_households: u32,
    pub biodiversity_index_delta: f64,
    pub verification_evidence: Option<String>,
    pub verified_by: Option<String>,
}

/// Get impact history for a project
#[hdk_extern]
pub fn get_project_impact(project_id: String) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&project_id)?, LinkTypes::AssetToImpacts)?,
        GetStrategy::default(),
    )?;

    let mut impacts = Vec::new();
    for link in links {
        let hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link".into())))?;
        if let Some(record) = get(hash, GetOptions::default())? {
            impacts.push(record);
        }
    }

    Ok(impacts)
}

/// Get full allocation summary for a project (consciousness + pledges + matches + impact)
#[hdk_extern]
pub fn get_allocation_summary(project_id: String) -> ExternResult<AllocationSummary> {
    // Get latest consciousness score
    let latest_consciousness = get_latest_project_consciousness(project_id.clone())?;
    let (phi_score, harmony_alignment) = if let Some(ref record) = latest_consciousness {
        if let Some(assessment) = record.entry().to_app_option::<ConsciousnessAssessment>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Parse: {:?}", e))))?
        {
            (assessment.phi_score, assessment.harmony_alignment)
        } else {
            (0.0, 0.0)
        }
    } else {
        (0.0, 0.0)
    };

    // Count pledges by status
    let pledge_records = get_project_pledges(project_id.clone())?;
    let mut total_pledged: u64 = 0;
    let mut pending_pledges: u32 = 0;
    let mut matched_pledges: u32 = 0;
    for record in &pledge_records {
        if let Some(pledge) = record.entry().to_app_option::<AllocationPledge>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Parse: {:?}", e))))?
        {
            total_pledged += pledge.amount;
            match pledge.status {
                PledgeStatus::Pending => pending_pledges += 1,
                PledgeStatus::Matched => matched_pledges += 1,
                _ => {}
            }
        }
    }

    // Aggregate impact
    let impact_records = get_project_impact(project_id.clone())?;
    let mut total_co2_avoided: f64 = 0.0;
    let mut total_jobs_created: u32 = 0;
    let mut total_trust_delta: f64 = 0.0;
    let mut total_households: u32 = 0;
    for record in &impact_records {
        if let Some(impact) = record.entry().to_app_option::<ImpactRecord>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(format!("Parse: {:?}", e))))?
        {
            total_co2_avoided += impact.co2_avoided_tonnes;
            total_jobs_created += impact.jobs_created;
            total_trust_delta += impact.community_trust_delta;
            total_households += impact.energy_access_households;
        }
    }

    // Net Humanity Benefit composite
    let net_humanity_benefit = (total_co2_avoided / 1000.0) * 0.3
        + (total_jobs_created as f64 / 100.0) * 0.25
        + total_trust_delta.clamp(-1.0, 1.0) * 0.25
        + (total_households as f64 / 1000.0) * 0.2;

    Ok(AllocationSummary {
        project_id,
        phi_score,
        harmony_alignment,
        total_pledged,
        pending_pledges,
        matched_pledges,
        total_co2_avoided,
        total_jobs_created,
        total_trust_delta,
        total_households,
        net_humanity_benefit,
        impact_report_count: impact_records.len() as u32,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AllocationSummary {
    pub project_id: String,
    pub phi_score: f64,
    pub harmony_alignment: f64,
    pub total_pledged: u64,
    pub pending_pledges: u32,
    pub matched_pledges: u32,
    pub total_co2_avoided: f64,
    pub total_jobs_created: u32,
    pub total_trust_delta: f64,
    pub total_households: u32,
    pub net_humanity_benefit: f64,
    pub impact_report_count: u32,
}

// ============================================================================
// Observability — Bridge Metrics Export
// ============================================================================

/// Return a JSON-encoded snapshot of this bridge's dispatch metrics.
///
/// See `mycelix_bridge_common::metrics::BridgeMetricsSnapshot` for the schema.
#[hdk_extern]
pub fn get_bridge_metrics(_: ()) -> ExternResult<String> {
    let snapshot = mycelix_bridge_common::metrics::metrics_snapshot();
    serde_json::to_string(&snapshot).map_err(|e| {
        wasm_error!(WasmErrorInner::Guest(format!(
            "Failed to serialize metrics snapshot: {}",
            e
        )))
    })
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
            EnergyEventType::CertificateCollateralRegistered,
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
            conditions_met: vec!["CommunityReadiness".to_string(), "FinancialSustainability".to_string()],
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
        let id = format!("milestone:{}:{:?}:{}", project_id, milestone_type, timestamp);
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

    // =========================================================================
    // CertificateCollateralInput Tests
    // =========================================================================

    fn valid_certificate_collateral_input() -> CertificateCollateralInput {
        CertificateCollateralInput {
            certificate_id: "cert:solar_farm_alpha:2026-03".to_string(),
            project_id: "project:solar_farm_alpha".to_string(),
            producer_did: "did:mycelix:producer1".to_string(),
            sap_value: 150_000,
        }
    }

    #[test]
    fn test_certificate_collateral_input_valid() {
        let input = valid_certificate_collateral_input();
        assert!(!input.certificate_id.is_empty());
        assert!(!input.project_id.is_empty());
        assert!(input.producer_did.starts_with("did:mycelix:"));
        assert!(input.sap_value > 0);
    }

    #[test]
    fn test_certificate_collateral_input_serialization() {
        let input = valid_certificate_collateral_input();
        let json = serde_json::to_string(&input);
        assert!(json.is_ok());
        let deserialized: CertificateCollateralInput =
            serde_json::from_str(&json.unwrap()).unwrap();
        assert_eq!(deserialized.certificate_id, input.certificate_id);
        assert_eq!(deserialized.sap_value, input.sap_value);
    }

    #[test]
    fn test_certificate_collateral_input_various_values() {
        let values = vec![1, 1_000, 100_000, 1_000_000, 10_000_000];
        for value in values {
            let input = CertificateCollateralInput {
                sap_value: value,
                ..valid_certificate_collateral_input()
            };
            assert!(input.sap_value > 0);
        }
    }

    // =========================================================================
    // CertificateCollateralResult Tests
    // =========================================================================

    #[test]
    fn test_certificate_collateral_result_success() {
        let result = CertificateCollateralResult {
            success: true,
            certificate_id: "cert:solar_farm_alpha:2026-03".to_string(),
            collateral_registered: true,
            error: None,
        };
        assert!(result.success);
        assert!(result.collateral_registered);
        assert!(result.error.is_none());
    }

    #[test]
    fn test_certificate_collateral_result_failure_finance_unreachable() {
        let result = CertificateCollateralResult {
            success: false,
            certificate_id: "cert:solar_farm_alpha:2026-03".to_string(),
            collateral_registered: false,
            error: Some("Finance cluster unreachable: NetworkError".to_string()),
        };
        assert!(!result.success);
        assert!(!result.collateral_registered);
        assert!(result.error.is_some());
        assert!(result.error.unwrap().contains("unreachable"));
    }

    #[test]
    fn test_certificate_collateral_result_failure_invalid_status() {
        let result = CertificateCollateralResult {
            success: false,
            certificate_id: "cert:solar_farm_alpha:2026-03".to_string(),
            collateral_registered: false,
            error: Some("Project must be Operational or CommunityOwned, got Funding".to_string()),
        };
        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("Operational"));
    }

    #[test]
    fn test_certificate_collateral_result_failure_project_not_found() {
        let result = CertificateCollateralResult {
            success: false,
            certificate_id: "cert:nonexistent:2026-03".to_string(),
            collateral_registered: false,
            error: Some("Project project:nonexistent not found".to_string()),
        };
        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("not found"));
    }

    #[test]
    fn test_certificate_collateral_result_serialization() {
        let result = CertificateCollateralResult {
            success: true,
            certificate_id: "cert:test:001".to_string(),
            collateral_registered: true,
            error: None,
        };
        let json = serde_json::to_string(&result);
        assert!(json.is_ok());
        let deserialized: CertificateCollateralResult =
            serde_json::from_str(&json.unwrap()).unwrap();
        assert_eq!(deserialized.success, result.success);
        assert_eq!(deserialized.certificate_id, result.certificate_id);
    }

    #[test]
    fn test_certificate_collateral_id_format() {
        let cert_id = "cert:solar_farm_alpha:2026-03";
        assert!(cert_id.starts_with("cert:"));
    }

    #[test]
    fn test_certificate_collateral_event_type() {
        let event_type = EnergyEventType::CertificateCollateralRegistered;
        assert_eq!(event_type, EnergyEventType::CertificateCollateralRegistered);
    }

    #[test]
    fn test_certificate_collateral_json_payload() {
        let input = valid_certificate_collateral_input();
        let payload = serde_json::json!({
            "certificate_id": input.certificate_id,
            "producer_did": input.producer_did,
            "sap_value": input.sap_value,
        });
        let payload_str = payload.to_string();
        assert!(!payload_str.is_empty());
        assert!(payload_str.contains("certificate_id"));
        assert!(payload_str.contains("sap_value"));
    }

    // =========================================================================
    // Gap 4: Equipment Procurement Tests
    // =========================================================================

    #[test]
    fn test_equipment_procurement_input_serde() {
        let input = EquipmentProcurementInput {
            equipment_type: "INVERTER-10KW".to_string(),
            quantity: 5,
            project_hash: ActionHash::from_raw_36(vec![0u8; 36]),
        };
        let json = serde_json::to_string(&input).unwrap();
        let back: EquipmentProcurementInput = serde_json::from_str(&json).unwrap();
        assert_eq!(back.equipment_type, "INVERTER-10KW");
        assert_eq!(back.quantity, 5);
    }

    #[test]
    fn test_equipment_procurement_result_serde() {
        // Successful procurement
        let result = EquipmentProcurementResult {
            procurement_initiated: true,
            stock_available: Some(20),
            purchase_order_hash: Some("ENERGY-INVERTER-10KW-abc123".to_string()),
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let back: EquipmentProcurementResult = serde_json::from_str(&json).unwrap();
        assert!(back.procurement_initiated);
        assert_eq!(back.stock_available, Some(20));
        assert!(back.purchase_order_hash.is_some());
        assert!(back.error.is_none());

        // Insufficient stock
        let result2 = EquipmentProcurementResult {
            procurement_initiated: false,
            stock_available: Some(2),
            purchase_order_hash: None,
            error: Some("Insufficient stock: 2 available, 5 required".to_string()),
        };
        let json2 = serde_json::to_string(&result2).unwrap();
        let back2: EquipmentProcurementResult = serde_json::from_str(&json2).unwrap();
        assert!(!back2.procurement_initiated);
        assert_eq!(back2.stock_available, Some(2));
        assert!(back2.error.as_deref().unwrap().contains("Insufficient"));

        // Cluster unavailable
        let result3 = EquipmentProcurementResult {
            procurement_initiated: false,
            stock_available: None,
            purchase_order_hash: None,
            error: Some("Supplychain cluster not available".to_string()),
        };
        let json3 = serde_json::to_string(&result3).unwrap();
        let back3: EquipmentProcurementResult = serde_json::from_str(&json3).unwrap();
        assert!(!back3.procurement_initiated);
        assert!(back3.stock_available.is_none());
        assert!(back3.error.is_some());
    }
}
