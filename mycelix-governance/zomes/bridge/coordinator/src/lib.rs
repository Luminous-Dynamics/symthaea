//! Governance Bridge Coordinator Zome
//!
//! Cross-hApp communication for proposal queries, voting status,
//! and execution requests across the Mycelix ecosystem.
//!
//! ## Consciousness Metrics Integration
//!
//! This module provides coordinator functions for Symthaea consciousness metrics:
//! - `record_consciousness_snapshot`: Store Φ measurements from Symthaea
//! - `verify_consciousness_gate`: Check Φ threshold for governance actions
//! - `assess_value_alignment`: Evaluate proposal alignment with Seven Harmonies
//! - `get_agent_consciousness_history`: Track consciousness over time
//!
//! Updated to use HDK 0.6 patterns

use hdk::prelude::*;
use governance_bridge_integrity::*;

const GOVERNANCE_HAPP_ID: &str = "mycelix-governance";
const SYMTHAEA_SOURCE: &str = "symthaea";

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    Ok(InitCallbackResult::Pass)
}

/// Query governance from another hApp
#[hdk_extern]
pub fn query_governance(input: QueryGovernanceInput) -> ExternResult<QueryGovernanceResult> {
    if input.source_happ.is_empty() || input.source_happ.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Source hApp must be 1-256 characters".into())));
    }

    let now = sys_time()?;

    let query = GovernanceQuery {
        id: format!("query:{}:{}", input.source_happ, now.as_micros()),
        query_type: input.query_type.clone(),
        source_happ: input.source_happ.clone(),
        parameters: serde_json::to_string(&input.parameters).unwrap_or_default(),
        queried_at: now,
    };
    create_entry(&EntryTypes::GovernanceQuery(query))?;

    match input.query_type {
        GovernanceQueryType::ActiveProposals => get_active_proposals_internal(),
        GovernanceQueryType::ProposalById => {
            let id = input.parameters.get("proposal_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            get_proposal_by_id_internal(id)
        }
        GovernanceQueryType::VotingEligibility => {
            let did = input.parameters.get("did")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            check_voting_eligibility_internal(did)
        }
        _ => Ok(QueryGovernanceResult {
            success: false,
            data: None,
            error: Some("Query type not implemented".into()),
        }),
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct QueryGovernanceInput {
    pub source_happ: String,
    pub query_type: GovernanceQueryType,
    pub parameters: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct QueryGovernanceResult {
    pub success: bool,
    pub data: Option<serde_json::Value>,
    pub error: Option<String>,
}

fn get_active_proposals_internal() -> ExternResult<QueryGovernanceResult> {
    // Query active proposals via anchor
    let links = get_links(
        LinkQuery::try_new(anchor_hash("active_proposals")?, LinkTypes::ActiveProposals)?,
        GetStrategy::default(),
    )?;

    let mut proposals = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(proposal_ref) = record
                .entry()
                .to_app_option::<ProposalReference>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                proposals.push(serde_json::json!({
                    "proposal_id": proposal_ref.proposal_id,
                    "title": proposal_ref.title,
                    "status": format!("{:?}", proposal_ref.status),
                    "votes_for": proposal_ref.votes_for,
                    "votes_against": proposal_ref.votes_against,
                }));
            }
        }
    }

    Ok(QueryGovernanceResult {
        success: true,
        data: Some(serde_json::json!({ "proposals": proposals })),
        error: None,
    })
}

fn get_proposal_by_id_internal(id: &str) -> ExternResult<QueryGovernanceResult> {
    // Cross-zome call to proposals coordinator
    let call_result = call(
        CallTargetCell::Local,
        ZomeName::from("proposals"),
        FunctionName::from("get_proposal"),
        None,
        ExternIO::encode(id.to_string())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
    );

    match call_result {
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            if let Ok(maybe_record) = extern_io.decode::<Option<Record>>() {
                Ok(QueryGovernanceResult {
                    success: true,
                    data: Some(serde_json::json!({
                        "proposal_id": id,
                        "found": maybe_record.is_some(),
                    })),
                    error: None,
                })
            } else {
                Ok(QueryGovernanceResult {
                    success: true,
                    data: Some(serde_json::json!({"proposal_id": id, "found": false})),
                    error: Some("Failed to decode proposal response".into()),
                })
            }
        }
        Ok(ZomeCallResponse::NetworkError(e)) => Ok(QueryGovernanceResult {
            success: false,
            data: None,
            error: Some(format!("Network error querying proposal: {}", e)),
        }),
        _ => Ok(QueryGovernanceResult {
            success: true,
            data: Some(serde_json::json!({"proposal_id": id, "found": false})),
            error: Some("Proposals zome unavailable".into()),
        }),
    }
}

fn check_voting_eligibility_internal(did: &str) -> ExternResult<QueryGovernanceResult> {
    // Cross-zome call to councils coordinator to check membership
    let call_result = call(
        CallTargetCell::Local,
        ZomeName::from("councils"),
        FunctionName::from("get_member_councils"),
        None,
        ExternIO::encode(did.to_string())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
    );

    match call_result {
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            if let Ok(councils) = extern_io.decode::<Vec<Record>>() {
                let eligible = !councils.is_empty();
                Ok(QueryGovernanceResult {
                    success: true,
                    data: Some(serde_json::json!({
                        "did": did,
                        "eligible": eligible,
                        "council_count": councils.len(),
                        "voting_power": if eligible { 1.0 } else { 0.0 },
                    })),
                    error: None,
                })
            } else {
                // Couldn't decode response — fail closed (not eligible)
                Ok(QueryGovernanceResult {
                    success: false,
                    data: Some(serde_json::json!({
                        "did": did,
                        "eligible": false,
                        "voting_power": 0.0,
                    })),
                    error: Some("Could not decode council membership response".into()),
                })
            }
        }
        Ok(ZomeCallResponse::NetworkError(e)) => Ok(QueryGovernanceResult {
            success: false,
            data: None,
            error: Some(format!("Network error checking eligibility: {}", e)),
        }),
        _ => {
            // Councils zome unavailable — fail closed (not eligible)
            Ok(QueryGovernanceResult {
                success: false,
                data: Some(serde_json::json!({
                    "did": did,
                    "eligible": false,
                    "voting_power": 0.0,
                })),
                error: Some("Councils zome unavailable — cannot verify eligibility".into()),
            })
        }
    }
}

/// Request execution from governance to another hApp
#[hdk_extern]
pub fn request_execution(input: RequestExecutionInput) -> ExternResult<Record> {
    // Input validation
    if input.proposal_id.is_empty() || input.proposal_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Proposal ID must be 1-256 characters".into())));
    }
    if input.target_happ.is_empty() || input.target_happ.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Target hApp must be 1-256 characters".into())));
    }
    if input.action.is_empty() || input.action.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Action must be 1-256 characters".into())));
    }

    let now = sys_time()?;

    let request = ExecutionRequest {
        id: format!("exec:{}:{}:{}", input.proposal_id, input.target_happ, now.as_micros()),
        proposal_id: input.proposal_id.clone(),
        target_happ: input.target_happ.clone(),
        action: input.action.clone(),
        parameters: serde_json::to_string(&input.parameters).unwrap_or_default(),
        status: ExecutionStatus::Pending,
        requested_at: now,
        executed_at: None,
    };

    let action_hash = create_entry(&EntryTypes::ExecutionRequest(request.clone()))?;

    // Create anchor and link for O(1) lookup by execution ID
    let eid_anchor = format!("eid:{}", request.id);
    create_entry(&EntryTypes::Anchor(Anchor(eid_anchor.clone())))?;
    create_link(
        anchor_hash(&eid_anchor)?,
        action_hash.clone(),
        LinkTypes::ExecutionById,
        (),
    )?;

    // Create anchor and link hApp to execution
    let happ_anchor = format!("happ:{}", input.target_happ);
    create_entry(&EntryTypes::Anchor(Anchor(happ_anchor.clone())))?;
    create_link(
        anchor_hash(&happ_anchor)?,
        action_hash.clone(),
        LinkTypes::HappToExecutions,
        (),
    )?;

    // Create anchor and link proposal to execution
    let proposal_anchor = format!("proposal_exec:{}", input.proposal_id);
    create_entry(&EntryTypes::Anchor(Anchor(proposal_anchor.clone())))?;
    create_link(
        anchor_hash(&proposal_anchor)?,
        action_hash.clone(),
        LinkTypes::ProposalToExecutions,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Execution not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RequestExecutionInput {
    pub proposal_id: String,
    pub target_happ: String,
    pub action: String,
    pub parameters: serde_json::Value,
}

/// Broadcast governance event
#[hdk_extern]
pub fn broadcast_governance_event(input: BroadcastGovernanceEventInput) -> ExternResult<Record> {
    // Input validation
    if input.subject.is_empty() || input.subject.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Subject must be 1-256 characters".into())));
    }
    if input.payload.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("Payload must be at most 4096 characters".into())));
    }
    if let Some(ref proposal_id) = input.proposal_id {
        if proposal_id.is_empty() || proposal_id.len() > 256 {
            return Err(wasm_error!(WasmErrorInner::Guest("Proposal ID must be 1-256 characters".into())));
        }
    }

    let now = sys_time()?;

    let event = GovernanceBridgeEvent {
        id: format!("event:{:?}:{}", input.event_type, now.as_micros()),
        event_type: input.event_type,
        proposal_id: input.proposal_id,
        subject: input.subject,
        payload: input.payload,
        source_happ: GOVERNANCE_HAPP_ID.to_string(),
        timestamp: now,
    };

    let action_hash = create_entry(&EntryTypes::GovernanceBridgeEvent(event))?;

    // Create anchor and link to recent events
    create_entry(&EntryTypes::Anchor(Anchor("recent_events".to_string())))?;
    create_link(
        anchor_hash("recent_events")?,
        action_hash.clone(),
        LinkTypes::RecentEvents,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Event not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct BroadcastGovernanceEventInput {
    pub event_type: GovernanceEventType,
    pub proposal_id: Option<String>,
    pub subject: String,
    pub payload: String,
}

/// Get pending executions for a hApp
#[hdk_extern]
pub fn get_pending_executions(target_happ: String) -> ExternResult<Vec<Record>> {
    let happ_anchor = format!("happ:{}", target_happ);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&happ_anchor)?, LinkTypes::HappToExecutions)?,
        GetStrategy::default(),
    )?;

    let mut executions = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            // Filter to only pending executions
            if let Some(exec) = record
                .entry()
                .to_app_option::<ExecutionRequest>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if exec.status == ExecutionStatus::Pending {
                    executions.push(record);
                }
            }
        }
    }

    Ok(executions)
}

/// Get recent governance events
#[hdk_extern]
pub fn get_recent_events(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("recent_events")?, LinkTypes::RecentEvents)?,
        GetStrategy::default(),
    )?;

    let mut events = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            events.push(record);
        }
    }

    // Sort by timestamp (most recent first)
    events.sort_by(|a, b| b.action().timestamp().cmp(&a.action().timestamp()));

    // Return last 50 events
    Ok(events.into_iter().take(50).collect())
}

/// Acknowledge execution (called by target hApp)
#[hdk_extern]
pub fn acknowledge_execution(input: AcknowledgeExecutionInput) -> ExternResult<bool> {
    // Input validation
    if input.execution_id.is_empty() || input.execution_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Execution ID must be 1-256 characters".into())));
    }
    if let Some(ref result) = input.result {
        if result.len() > 4096 {
            return Err(wasm_error!(WasmErrorInner::Guest("Result must be at most 4096 characters".into())));
        }
    }

    // Find the execution request by ID via O(1) link-based lookup
    let eid_anchor = format!("eid:{}", input.execution_id);
    let mut found = false;

    if let Ok(entry_hash) = anchor_hash(&eid_anchor) {
        if let Ok(links) = get_links(
            LinkQuery::try_new(entry_hash, LinkTypes::ExecutionById)?,
            GetStrategy::default(),
        ) {
            if let Some(link) = links.into_iter().max_by_key(|l| l.timestamp) {
                if let Ok(ah) = ActionHash::try_from(link.target) {
                    if let Some(record) = get(ah, GetOptions::default())? {
                        if let Some(exec) = record
                            .entry()
                            .to_app_option::<ExecutionRequest>()
                            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                        {
                            let updated_exec = ExecutionRequest {
                                status: input.status.clone(),
                                executed_at: Some(sys_time()?),
                                ..exec
                            };
                            update_entry(
                                record.action_address().clone(),
                                &EntryTypes::ExecutionRequest(updated_exec),
                            )?;
                            found = true;
                        }
                    }
                }
            }
        }
    }

    // Fallback: O(n) chain scan for execution requests created before the link was added
    if !found {
        let filter = ChainQueryFilter::new()
            .entry_type(EntryType::App(AppEntryDef::try_from(
                UnitEntryTypes::ExecutionRequest,
            )?))
            .include_entries(true);

        let records = query(filter)?;

        for record in records {
            if let Some(exec) = record
                .entry()
                .to_app_option::<ExecutionRequest>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if exec.id == input.execution_id {
                    let updated_exec = ExecutionRequest {
                        status: input.status.clone(),
                        executed_at: Some(sys_time()?),
                        ..exec
                    };
                    update_entry(
                        record.action_address().clone(),
                        &EntryTypes::ExecutionRequest(updated_exec),
                    )?;
                    found = true;
                    break;
                }
            }
        }
    }

    Ok(found)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AcknowledgeExecutionInput {
    pub execution_id: String,
    pub status: ExecutionStatus,
    pub result: Option<String>,
}

/// Publish a proposal reference for cross-hApp visibility
#[hdk_extern]
pub fn publish_proposal_reference(input: PublishProposalReferenceInput) -> ExternResult<Record> {
    // Input validation
    if input.proposal_id.is_empty() || input.proposal_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Proposal ID must be 1-256 characters".into())));
    }
    if input.title.is_empty() || input.title.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Title must be 1-256 characters".into())));
    }

    // Check status before moving into struct
    let is_active = input.status == ProposalStatus::Active;

    let proposal_ref = ProposalReference {
        proposal_id: input.proposal_id.clone(),
        title: input.title,
        proposal_type: input.proposal_type,
        status: input.status,
        votes_for: input.votes_for,
        votes_against: input.votes_against,
        ends_at: input.ends_at,
        created_at: sys_time()?,
    };

    let action_hash = create_entry(&EntryTypes::ProposalReference(proposal_ref))?;

    // Link to active proposals if status is Active
    if is_active {
        create_entry(&EntryTypes::Anchor(Anchor("active_proposals".to_string())))?;
        create_link(
            anchor_hash("active_proposals")?,
            action_hash.clone(),
            LinkTypes::ActiveProposals,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find proposal reference".into()
        )))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PublishProposalReferenceInput {
    pub proposal_id: String,
    pub title: String,
    pub proposal_type: ProposalType,
    pub status: ProposalStatus,
    pub votes_for: u64,
    pub votes_against: u64,
    pub ends_at: Timestamp,
}

/// Transfer credits between accounts (cross-zome entry point)
///
/// Called by the execution zome's `GovernanceAction::TransferCredits` dispatch.
/// Records the transfer as a governance event. Actual fund movement is handled
/// by the fund allocation system in the execution zome.
#[hdk_extern]
pub fn transfer_credits(input: TransferCreditsInput) -> ExternResult<Record> {
    if input.from.is_empty() || input.from.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("From account must be 1-256 characters".into())));
    }
    if input.to.is_empty() || input.to.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("To account must be 1-256 characters".into())));
    }
    if input.amount <= 0.0 || !input.amount.is_finite() {
        return Err(wasm_error!(WasmErrorInner::Guest("Amount must be positive and finite".into())));
    }

    // Record the transfer as a governance event
    broadcast_governance_event(BroadcastGovernanceEventInput {
        event_type: GovernanceEventType::ProposalExecuted,
        proposal_id: None,
        subject: format!("Credit transfer: {} -> {} ({} credits)", input.from, input.to, input.amount),
        payload: serde_json::to_string(&serde_json::json!({
            "type": "transfer_credits",
            "from": input.from,
            "to": input.to,
            "amount": input.amount,
        })).unwrap_or_default(),
    })
}

/// Input for credit transfers via cross-zome call
#[derive(Serialize, Deserialize, Debug)]
pub struct TransferCreditsInput {
    pub from: String,
    pub to: String,
    pub amount: f64,
}

/// Get events by proposal ID - for filtering events related to a specific proposal
#[hdk_extern]
pub fn get_events_by_proposal(proposal_id: String) -> ExternResult<Vec<Record>> {
    // Get all recent events and filter by proposal_id
    let links = get_links(
        LinkQuery::try_new(anchor_hash("recent_events")?, LinkTypes::RecentEvents)?,
        GetStrategy::default(),
    )?;

    let mut events = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            // Check if event relates to this proposal
            if let Some(event) = record
                .entry()
                .to_app_option::<GovernanceBridgeEvent>()
                .ok()
                .flatten()
            {
                if event.proposal_id.as_ref() == Some(&proposal_id) {
                    events.push(record);
                }
            }
        }

        if events.len() >= 50 {
            break;
        }
    }

    Ok(events)
}

// =============================================================================
// CONSCIOUSNESS METRICS COORDINATOR FUNCTIONS
// =============================================================================

/// Record a consciousness snapshot from Symthaea
///
/// Stores the Φ measurement and related metrics for an agent at a point in time.
/// Used to establish consciousness state before governance actions.
#[hdk_extern]
pub fn record_consciousness_snapshot(input: RecordSnapshotInput) -> ExternResult<Record> {
    // Input validation
    if input.phi < 0.0 || input.phi > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest("Phi must be between 0.0 and 1.0".into())));
    }
    if input.meta_awareness < 0.0 || input.meta_awareness > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest("Meta awareness must be between 0.0 and 1.0".into())));
    }
    if input.self_model_accuracy < 0.0 || input.self_model_accuracy > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest("Self model accuracy must be between 0.0 and 1.0".into())));
    }
    if input.coherence < 0.0 || input.coherence > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest("Coherence must be between 0.0 and 1.0".into())));
    }
    if input.affective_valence < -1.0 || input.affective_valence > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest("Affective valence must be between -1.0 and 1.0".into())));
    }
    if input.care_activation < 0.0 || input.care_activation > 1.0 {
        return Err(wasm_error!(WasmErrorInner::Guest("Care activation must be between 0.0 and 1.0".into())));
    }
    if let Some(ref source) = input.source {
        if source.is_empty() || source.len() > 256 {
            return Err(wasm_error!(WasmErrorInner::Guest("Source must be 1-256 characters".into())));
        }
    }

    let now = sys_time()?;
    let agent_info = agent_info()?;
    let agent_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);

    let snapshot = ConsciousnessSnapshot {
        id: format!("snapshot:{}:{}", agent_did, now.as_micros()),
        agent_did: agent_did.clone(),
        phi: input.phi,
        meta_awareness: input.meta_awareness,
        self_model_accuracy: input.self_model_accuracy,
        coherence: input.coherence,
        affective_valence: input.affective_valence,
        care_activation: input.care_activation,
        captured_at: now,
        source: input.source.unwrap_or_else(|| SYMTHAEA_SOURCE.to_string()),
    };

    let action_hash = create_entry(&EntryTypes::ConsciousnessSnapshot(snapshot.clone()))?;

    // Link from agent to snapshot
    let agent_anchor = format!("agent:{}", agent_did);
    create_entry(&EntryTypes::Anchor(Anchor(agent_anchor.clone())))?;
    create_link(
        anchor_hash(&agent_anchor)?,
        action_hash.clone(),
        LinkTypes::AgentToSnapshots,
        (),
    )?;

    // Link to recent snapshots
    create_entry(&EntryTypes::Anchor(Anchor("recent_snapshots".to_string())))?;
    create_link(
        anchor_hash("recent_snapshots")?,
        action_hash.clone(),
        LinkTypes::RecentSnapshots,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Snapshot not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RecordSnapshotInput {
    pub phi: f64,
    pub meta_awareness: f64,
    pub self_model_accuracy: f64,
    pub coherence: f64,
    pub affective_valence: f64,
    pub care_activation: f64,
    pub source: Option<String>,
}

/// Verify consciousness gate for a governance action
///
/// Checks if the agent's current Φ meets the threshold for the requested action type.
/// Returns a gate verification record that can be referenced by governance actions.
#[hdk_extern]
pub fn verify_consciousness_gate(input: VerifyGateInput) -> ExternResult<GateVerificationResult> {
    let now = sys_time()?;
    let agent_info = agent_info()?;
    let agent_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);

    // Get the latest snapshot for this agent
    let latest_snapshot = get_latest_agent_snapshot(&agent_did)?;

    let (_snapshot, snapshot_id, phi) = match latest_snapshot {
        Some((_record, snap)) => (Some(snap.clone()), snap.id.clone(), snap.phi),
        None => {
            // No snapshot available - gate fails
            let gate = ConsciousnessGate {
                id: format!("gate:{}:{}", agent_did, now.as_micros()),
                agent_did: agent_did.clone(),
                action_type: input.action_type,
                snapshot_id: "none".to_string(),
                phi_at_verification: 0.0,
                required_phi: input.action_type.phi_threshold(),
                passed: false,
                failure_reason: Some("No consciousness snapshot available".to_string()),
                action_id: input.action_id.clone(),
                verified_at: now,
            };

            let action_hash = create_entry(&EntryTypes::ConsciousnessGate(gate.clone()))?;

            // Link from agent to gate
            let agent_anchor = format!("agent:{}", agent_did);
            create_entry(&EntryTypes::Anchor(Anchor(agent_anchor.clone())))?;
            create_link(
                anchor_hash(&agent_anchor)?,
                action_hash,
                LinkTypes::AgentToGates,
                (),
            )?;

            return Ok(GateVerificationResult {
                passed: false,
                phi: 0.0,
                required_phi: input.action_type.phi_threshold(),
                action_type: input.action_type,
                failure_reason: Some("No consciousness snapshot available".to_string()),
                gate_id: gate.id,
            });
        }
    };

    let required_phi = input.action_type.phi_threshold();
    let passed = phi >= required_phi;
    let failure_reason = if passed {
        None
    } else {
        Some(format!(
            "Φ {} below threshold {} for {:?}",
            phi, required_phi, input.action_type
        ))
    };

    // Create gate verification record
    let gate = ConsciousnessGate {
        id: format!("gate:{}:{}", agent_did, now.as_micros()),
        agent_did: agent_did.clone(),
        action_type: input.action_type,
        snapshot_id,
        phi_at_verification: phi,
        required_phi,
        passed,
        failure_reason: failure_reason.clone(),
        action_id: input.action_id,
        verified_at: now,
    };

    let action_hash = create_entry(&EntryTypes::ConsciousnessGate(gate.clone()))?;

    // Link from agent to gate
    let agent_anchor = format!("agent:{}", agent_did);
    create_entry(&EntryTypes::Anchor(Anchor(agent_anchor.clone())))?;
    create_link(
        anchor_hash(&agent_anchor)?,
        action_hash,
        LinkTypes::AgentToGates,
        (),
    )?;

    Ok(GateVerificationResult {
        passed,
        phi,
        required_phi,
        action_type: input.action_type,
        failure_reason,
        gate_id: gate.id,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VerifyGateInput {
    pub action_type: GovernanceActionType,
    pub action_id: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GateVerificationResult {
    pub passed: bool,
    pub phi: f64,
    pub required_phi: f64,
    pub action_type: GovernanceActionType,
    pub failure_reason: Option<String>,
    pub gate_id: String,
}

/// Assess value alignment of a proposal
///
/// Evaluates how well a proposal aligns with the Seven Harmonies,
/// using the agent's current consciousness state.
#[hdk_extern]
pub fn assess_value_alignment(input: AssessAlignmentInput) -> ExternResult<Record> {
    // Input validation
    if input.proposal_id.is_empty() || input.proposal_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Proposal ID must be 1-256 characters".into())));
    }
    if input.proposal_content.is_empty() || input.proposal_content.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("Proposal content must be 1-4096 characters".into())));
    }

    let now = sys_time()?;
    let agent_info = agent_info()?;
    let agent_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);

    // Get the latest snapshot for this agent
    let latest_snapshot = get_latest_agent_snapshot(&agent_did)?;

    let snapshot_id = match &latest_snapshot {
        Some((_, snap)) => snap.id.clone(),
        None => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "No consciousness snapshot available for alignment assessment".into()
            )));
        }
    };

    // Calculate harmony scores (in production, this would analyze the proposal content)
    let harmony_scores = calculate_harmony_scores(&input.proposal_content);

    // Calculate overall alignment
    let overall_alignment: f64 = harmony_scores.iter().map(|h| h.score).sum::<f64>()
        / harmony_scores.len() as f64;

    // Detect violations (scores below threshold)
    let violations: Vec<String> = harmony_scores
        .iter()
        .filter(|h| h.score < -0.3)
        .map(|h| format!("{}: severe misalignment ({:.2})", h.harmony, h.score))
        .collect();

    // Calculate authenticity (based on CARE activation if available)
    let authenticity = match &latest_snapshot {
        Some((_, snap)) => snap.care_activation,
        None => 0.5,
    };

    // Determine recommendation
    let recommendation = determine_recommendation(overall_alignment, authenticity, &violations);

    // Create assessment entry
    let assessment = ValueAlignmentAssessment {
        id: format!("alignment:{}:{}:{}", agent_did, input.proposal_id, now.as_micros()),
        agent_did: agent_did.clone(),
        proposal_id: input.proposal_id.clone(),
        overall_alignment,
        harmony_scores,
        authenticity,
        violations,
        recommendation,
        snapshot_id,
        assessed_at: now,
    };

    let action_hash = create_entry(&EntryTypes::ValueAlignmentAssessment(assessment))?;

    // Link from proposal to alignment
    let proposal_anchor = format!("proposal:{}", input.proposal_id);
    create_entry(&EntryTypes::Anchor(Anchor(proposal_anchor.clone())))?;
    create_link(
        anchor_hash(&proposal_anchor)?,
        action_hash.clone(),
        LinkTypes::ProposalToAlignments,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Assessment not found".into())))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct AssessAlignmentInput {
    pub proposal_id: String,
    pub proposal_content: String,
}

/// Get agent's consciousness history
#[hdk_extern]
pub fn get_agent_consciousness_history(agent_did: String) -> ExternResult<Option<Record>> {
    let agent_anchor = format!("agent:history:{}", agent_did);
    let anchor = match anchor_hash(&agent_anchor) {
        Ok(h) => h,
        Err(_) => return Ok(None),
    };

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AgentToHistory)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.last() {
        let action_hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        return get(action_hash, GetOptions::default());
    }

    Ok(None)
}

/// Get recent consciousness snapshots for an agent
#[hdk_extern]
pub fn get_agent_snapshots(input: GetAgentSnapshotsInput) -> ExternResult<Vec<Record>> {
    let agent_anchor = format!("agent:{}", input.agent_did);
    let anchor = match anchor_hash(&agent_anchor) {
        Ok(h) => h,
        Err(_) => return Ok(vec![]),
    };

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AgentToSnapshots)?,
        GetStrategy::default(),
    )?;

    let mut snapshots = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            snapshots.push(record);
        }

        if snapshots.len() >= input.limit.unwrap_or(50) as usize {
            break;
        }
    }

    // Sort by timestamp (most recent first)
    snapshots.sort_by(|a, b| b.action().timestamp().cmp(&a.action().timestamp()));

    Ok(snapshots)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetAgentSnapshotsInput {
    pub agent_did: String,
    pub limit: Option<u32>,
}

/// Get value alignments for a proposal
#[hdk_extern]
pub fn get_proposal_alignments(proposal_id: String) -> ExternResult<Vec<Record>> {
    let proposal_anchor = format!("proposal:{}", proposal_id);
    let anchor = match anchor_hash(&proposal_anchor) {
        Ok(h) => h,
        Err(_) => return Ok(vec![]),
    };

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::ProposalToAlignments)?,
        GetStrategy::default(),
    )?;

    let mut alignments = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            alignments.push(record);
        }
    }

    Ok(alignments)
}

/// Get current Φ threshold requirements
#[hdk_extern]
pub fn get_phi_thresholds(_: ()) -> ExternResult<PhiThresholds> {
    Ok(PhiThresholds {
        basic: GovernanceActionType::Basic.phi_threshold(),
        proposal_submission: GovernanceActionType::ProposalSubmission.phi_threshold(),
        voting: GovernanceActionType::Voting.phi_threshold(),
        constitutional: GovernanceActionType::Constitutional.phi_threshold(),
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PhiThresholds {
    pub basic: f64,
    pub proposal_submission: f64,
    pub voting: f64,
    pub constitutional: f64,
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Get the latest consciousness snapshot for an agent
fn get_latest_agent_snapshot(
    agent_did: &str,
) -> ExternResult<Option<(Record, ConsciousnessSnapshot)>> {
    let agent_anchor = format!("agent:{}", agent_did);
    let anchor = match anchor_hash(&agent_anchor) {
        Ok(h) => h,
        Err(_) => return Ok(None),
    };

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AgentToSnapshots)?,
        GetStrategy::default(),
    )?;

    // Get all snapshots and find the most recent
    let mut latest: Option<(Record, ConsciousnessSnapshot, Timestamp)> = None;

    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            if let Some(snapshot) = record
                .entry()
                .to_app_option::<ConsciousnessSnapshot>()
                .ok()
                .flatten()
            {
                let ts = snapshot.captured_at;
                match &latest {
                    Some((_, _, latest_ts)) if ts > *latest_ts => {
                        latest = Some((record, snapshot, ts));
                    }
                    None => {
                        latest = Some((record, snapshot, ts));
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(latest.map(|(r, s, _)| (r, s)))
}

/// Calculate harmony scores for proposal content
fn calculate_harmony_scores(content: &str) -> Vec<HarmonyScore> {
    let content_lower = content.to_lowercase();

    // Seven Harmonies with keyword-based scoring
    let harmonies = [
        ("Resonant Coherence", vec!["integration", "wholeness", "coherent", "unified"]),
        ("Pan-Sentient Flourishing", vec!["flourishing", "wellbeing", "care", "compassion", "help"]),
        ("Integral Wisdom", vec!["wisdom", "truth", "knowledge", "understanding", "verify"]),
        ("Infinite Play", vec!["creative", "play", "possibility", "experiment", "novel"]),
        ("Universal Interconnectedness", vec!["connection", "network", "relationship", "web"]),
        ("Sacred Reciprocity", vec!["reciprocity", "balance", "exchange", "mutual", "fair"]),
        ("Evolutionary Progression", vec!["evolution", "growth", "progress", "development"]),
    ];

    harmonies
        .iter()
        .map(|(name, keywords)| {
            let keyword_count = keywords
                .iter()
                .filter(|kw| content_lower.contains(*kw))
                .count();

            // Base score of 0.0, increase for each keyword found
            let score = (keyword_count as f64 * 0.2).min(1.0);

            // Check for negative indicators
            let negative_keywords = ["harm", "damage", "destroy", "exclude", "discriminate"];
            let negative_count = negative_keywords
                .iter()
                .filter(|kw| content_lower.contains(*kw))
                .count();

            let final_score = score - (negative_count as f64 * 0.3);

            HarmonyScore {
                harmony: name.to_string(),
                score: final_score.clamp(-1.0, 1.0),
            }
        })
        .collect()
}

/// Determine governance recommendation based on alignment and authenticity
fn determine_recommendation(
    alignment: f64,
    authenticity: f64,
    violations: &[String],
) -> GovernanceRecommendation {
    // Strong oppose if violations detected
    if !violations.is_empty() {
        return GovernanceRecommendation::StrongOppose;
    }

    // Cannot evaluate if authenticity is too low
    if authenticity < 0.2 {
        return GovernanceRecommendation::CannotEvaluate;
    }

    // Weighted score: 60% alignment, 40% authenticity
    let combined = alignment * 0.6 + authenticity * 0.4;

    match combined {
        c if c > 0.7 => GovernanceRecommendation::StrongSupport,
        c if c > 0.3 => GovernanceRecommendation::Support,
        c if c > -0.3 => GovernanceRecommendation::Neutral,
        c if c > -0.7 => GovernanceRecommendation::Oppose,
        _ => GovernanceRecommendation::StrongOppose,
    }
}

// =============================================================================
// WEIGHTED CONSENSUS COORDINATOR FUNCTIONS
// =============================================================================

/// Register as a consensus participant
///
/// Creates K-Vector and FederatedReputation entries for the calling agent,
/// establishing them as a participant in weighted consensus voting.
#[hdk_extern]
pub fn register_consensus_participant(_: ()) -> ExternResult<Record> {
    let now = sys_time()?;
    let agent_info = agent_info()?;
    let agent_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);

    // Create initial K-Vector
    let k_vector = KVector::new_participant(agent_did.clone(), now);
    let k_vector_hash = create_entry(&EntryTypes::KVector(k_vector.clone()))?;

    // Link agent to K-Vector
    let agent_anchor = format!("agent:{}", agent_did);
    create_entry(&EntryTypes::Anchor(Anchor(agent_anchor.clone())))?;
    create_link(
        anchor_hash(&agent_anchor)?,
        k_vector_hash.clone(),
        LinkTypes::AgentToKVector,
        (),
    )?;

    // Create initial FederatedReputation
    let fed_rep = FederatedReputation::new_participant(agent_did.clone(), now);
    let fed_rep_hash = create_entry(&EntryTypes::FederatedReputation(fed_rep.clone()))?;

    // Link agent to FederatedReputation
    create_link(
        anchor_hash(&agent_anchor)?,
        fed_rep_hash.clone(),
        LinkTypes::AgentToFederatedRep,
        (),
    )?;

    // Get latest consciousness snapshot (if any) for initial phi
    let phi = match get_latest_agent_snapshot(&agent_did)? {
        Some((_, snapshot)) => snapshot.phi,
        None => 0.3, // Default for new participants
    };

    // Create ConsensusParticipant
    let participant = ConsensusParticipant {
        agent_did: agent_did.clone(),
        k_vector_id: format!("kvec:{}:{}", agent_did, now.as_micros()),
        federated_rep_id: Some(format!("fedrep:{}:{}", agent_did, now.as_micros())),
        reputation: k_vector.k_r,
        voting_weight: k_vector.voting_weight(),
        matl_score: 0.5, // Initial MATL score
        phi,
        federated_score: fed_rep.aggregated_score,
        is_active: true,
        rounds_participated: 0,
        successful_votes: 0,
        slashing_events: 0,
        last_slashing_at: None,
        streak_count: 0,
        registered_at: now,
        last_active_at: now,
    };

    let participant_hash = create_entry(&EntryTypes::ConsensusParticipant(participant))?;

    // Link agent to participant
    create_link(
        anchor_hash(&agent_anchor)?,
        participant_hash.clone(),
        LinkTypes::AgentToParticipant,
        (),
    )?;

    // Link to active participants
    create_entry(&EntryTypes::Anchor(Anchor("active_participants".to_string())))?;
    create_link(
        anchor_hash("active_participants")?,
        participant_hash.clone(),
        LinkTypes::ActiveParticipants,
        (),
    )?;

    get(participant_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Participant not found".into())))
}

/// Calculate holistic vote weight for a proposal
///
/// Computes the full holistic weight: Reputation² × (0.7 + 0.3 × Φ) × (1 + 0.2 × HarmonicAlignment)
#[hdk_extern]
pub fn calculate_holistic_vote_weight(input: CalculateWeightInput) -> ExternResult<HolisticVotingWeight> {
    let agent_info = agent_info()?;
    let agent_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);

    // Get participant's current state
    let participant = get_agent_participant(&agent_did)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not registered as participant".into())))?;

    // Get latest consciousness Φ
    let phi = match get_latest_agent_snapshot(&agent_did)? {
        Some((_, snapshot)) => snapshot.phi,
        None => participant.phi,
    };

    // Calculate weight with harmonic alignment
    let weight = HolisticVotingWeight::calculate(
        participant.reputation,
        phi,
        input.harmonic_alignment.unwrap_or(0.0),
    );

    Ok(weight)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CalculateWeightInput {
    pub harmonic_alignment: Option<f64>,
}

/// Cast a weighted vote on a proposal
///
/// Creates a WeightedVote entry with full holistic weight calculation.
/// Verifies consciousness gate before allowing the vote.
#[hdk_extern]
pub fn cast_weighted_vote(input: CastWeightedVoteInput) -> ExternResult<WeightedVoteResult> {
    // Input validation
    if input.proposal_id.is_empty() || input.proposal_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Proposal ID must be 1-256 characters".into())));
    }
    if let Some(ref reason) = input.reason {
        if reason.len() > 4096 {
            return Err(wasm_error!(WasmErrorInner::Guest("Reason must be at most 4096 characters".into())));
        }
    }

    let now = sys_time()?;
    let agent_info = agent_info()?;
    let agent_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);

    // Get participant's current state
    let participant = get_agent_participant(&agent_did)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not registered as participant".into())))?;

    // Check if participant can vote
    if !participant.can_vote() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Participant cannot vote (inactive or low reputation)".into()
        )));
    }

    // Get adaptive threshold for proposal type
    let threshold = AdaptiveThreshold::for_proposal_type(&input.proposal_type);

    // Get latest consciousness Φ
    let phi = match get_latest_agent_snapshot(&agent_did)? {
        Some((_, snapshot)) => snapshot.phi,
        None => participant.phi,
    };

    // Verify consciousness gate for this proposal type
    if !threshold.voter_meets_phi_requirement(phi) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Consciousness Φ ({:.2}) below threshold ({:.2}) for {:?} proposals",
            phi, threshold.min_voter_phi, input.proposal_type
        ))));
    }

    // Check cooldown
    if participant.in_cooldown(now) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Participant in cooldown period after slashing".into()
        )));
    }

    // Calculate holistic weight
    let holistic_weight = HolisticVotingWeight::calculate(
        participant.effective_reputation(now),
        phi,
        input.harmonic_alignment.unwrap_or(0.0),
    );

    // Create the vote
    let vote = WeightedVote {
        id: format!("vote:{}:{}:{}", agent_did, input.proposal_id, now.as_micros()),
        proposal_id: input.proposal_id.clone(),
        round: input.round,
        voter_did: agent_did.clone(),
        decision: input.decision,
        reputation: participant.effective_reputation(now),
        weight: holistic_weight.final_weight,
        phi,
        reason: input.reason,
        voted_at: now,
        signature: format!("sig:{}:{}", agent_did, now.as_micros()),
    };

    let vote_hash = create_entry(&EntryTypes::WeightedVote(vote.clone()))?;

    // Link vote to proposal round
    let round_anchor = format!("round:{}:{}", input.proposal_id, input.round);
    create_entry(&EntryTypes::Anchor(Anchor(round_anchor.clone())))?;
    create_link(
        anchor_hash(&round_anchor)?,
        vote_hash.clone(),
        LinkTypes::RoundToVotes,
        (),
    )?;

    // Link vote to agent
    let agent_anchor = format!("agent:{}", agent_did);
    create_link(
        anchor_hash(&agent_anchor)?,
        vote_hash,
        LinkTypes::AgentToVotes,
        (),
    )?;

    Ok(WeightedVoteResult {
        vote_id: vote.id,
        weight: holistic_weight.final_weight,
        weight_breakdown: holistic_weight.calculation_breakdown,
        decision: input.decision,
        phi_at_vote: phi,
        proposal_type: input.proposal_type,
        threshold_required: threshold.base_threshold,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CastWeightedVoteInput {
    pub proposal_id: String,
    pub proposal_type: ProposalType,
    pub round: u64,
    pub decision: VoteDecision,
    pub harmonic_alignment: Option<f64>,
    pub reason: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct WeightedVoteResult {
    pub vote_id: String,
    pub weight: f64,
    pub weight_breakdown: String,
    pub decision: VoteDecision,
    pub phi_at_vote: f64,
    pub proposal_type: ProposalType,
    pub threshold_required: f64,
}

/// Get adaptive threshold for a proposal type
#[hdk_extern]
pub fn get_adaptive_threshold(proposal_type: ProposalType) -> ExternResult<AdaptiveThreshold> {
    Ok(AdaptiveThreshold::for_proposal_type(&proposal_type))
}

/// Get participant's current status including streak, cooldown, and effective reputation
#[hdk_extern]
pub fn get_participant_status(_: ()) -> ExternResult<ParticipantStatus> {
    let now = sys_time()?;
    let agent_info = agent_info()?;
    let agent_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);

    let participant = get_agent_participant(&agent_did)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Not registered as participant".into())))?;

    // Get latest consciousness Φ
    let current_phi = match get_latest_agent_snapshot(&agent_did)? {
        Some((_, snapshot)) => snapshot.phi,
        None => participant.phi,
    };

    let effective_rep = participant.effective_reputation(now);
    let in_cooldown = participant.in_cooldown(now);
    let streak_bonus = participant.streak_bonus();

    // Check voting eligibility for each proposal type
    let can_vote_standard = participant.can_vote() &&
        AdaptiveThreshold::for_proposal_type(&ProposalType::Standard)
            .voter_meets_phi_requirement(current_phi);

    let can_vote_emergency = participant.can_vote() &&
        AdaptiveThreshold::for_proposal_type(&ProposalType::Emergency)
            .voter_meets_phi_requirement(current_phi);

    let can_vote_constitutional = participant.can_vote() &&
        AdaptiveThreshold::for_proposal_type(&ProposalType::Constitutional)
            .voter_meets_phi_requirement(current_phi);

    Ok(ParticipantStatus {
        agent_did,
        is_active: participant.is_active,
        base_reputation: participant.reputation,
        effective_reputation: effective_rep,
        streak_count: participant.streak_count,
        streak_bonus,
        in_cooldown,
        current_phi,
        federated_score: participant.federated_score,
        rounds_participated: participant.rounds_participated,
        successful_votes: participant.successful_votes,
        success_rate: participant.success_rate(),
        slashing_events: participant.slashing_events,
        can_vote_standard,
        can_vote_emergency,
        can_vote_constitutional,
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ParticipantStatus {
    pub agent_did: String,
    pub is_active: bool,
    pub base_reputation: f64,
    pub effective_reputation: f64,
    pub streak_count: u64,
    pub streak_bonus: f64,
    pub in_cooldown: bool,
    pub current_phi: f64,
    pub federated_score: f64,
    pub rounds_participated: u64,
    pub successful_votes: u64,
    pub success_rate: f64,
    pub slashing_events: u64,
    pub can_vote_standard: bool,
    pub can_vote_emergency: bool,
    pub can_vote_constitutional: bool,
}

/// Update federated reputation signals from external modules
///
/// ## Domain Boundary Enforcement
///
/// All input values are clamped to valid ranges [0.0, 1.0] for scores.
/// Per Commons Charter v1.0 Article II:
/// - Finance domain signals (stake, payments, escrow) can contribute at most 5% to aggregated score
/// - This is enforced in FederatedReputation.calculate_aggregated(), not here
/// - All normalized values are clamped to prevent manipulation
#[hdk_extern]
pub fn update_federated_reputation(input: UpdateFederatedReputationInput) -> ExternResult<Record> {
    let now = sys_time()?;
    let agent_info = agent_info()?;
    let agent_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);

    // Get current federated reputation
    let (current_record, mut fed_rep) = get_agent_federated_reputation(&agent_did)?
        .ok_or(wasm_error!(WasmErrorInner::Guest("No federated reputation found".into())))?;

    // Update only the fields that are provided, with domain boundary clamping
    // All f64 scores are clamped to [0.0, 1.0] to prevent manipulation

    // Identity domain signals
    if let Some(v) = input.identity_verification {
        fed_rep.identity_verification = v.clamp(0.0, 1.0);
    }
    if let Some(v) = input.credential_count {
        fed_rep.credential_count = v.min(100); // Cap at 100 credentials
    }
    if let Some(v) = input.credential_quality {
        fed_rep.credential_quality = v.clamp(0.0, 1.0);
    }

    // Knowledge domain signals
    if let Some(v) = input.epistemic_contributions {
        fed_rep.epistemic_contributions = v.clamp(0.0, 1.0);
    }
    if let Some(v) = input.factcheck_accuracy {
        fed_rep.factcheck_accuracy = v.clamp(0.0, 1.0);
    }
    if let Some(v) = input.dark_spots_resolved {
        fed_rep.dark_spots_resolved = v.min(1000); // Cap at 1000
    }

    // Finance domain signals (these contribute max 5% to final score)
    if let Some(v) = input.stake_weight {
        fed_rep.stake_weight = v.clamp(0.0, 1.0);
    }
    if let Some(v) = input.payment_reliability {
        fed_rep.payment_reliability = v.clamp(0.0, 1.0);
    }
    if let Some(v) = input.escrow_completion_rate {
        fed_rep.escrow_completion_rate = v.clamp(0.0, 1.0);
    }

    // FL domain signals
    if let Some(v) = input.pogq_score {
        fed_rep.pogq_score = v.clamp(0.0, 1.0);
    }
    if let Some(v) = input.fl_contributions {
        fed_rep.fl_contributions = v.min(10000); // Cap at 10000
    }
    if let Some(v) = input.byzantine_clean_rate {
        fed_rep.byzantine_clean_rate = v.clamp(0.0, 1.0);
    }

    // Governance domain signals
    if let Some(v) = input.voting_participation {
        fed_rep.voting_participation = v.clamp(0.0, 1.0);
    }
    if let Some(v) = input.proposal_success_rate {
        fed_rep.proposal_success_rate = v.clamp(0.0, 1.0);
    }
    if let Some(v) = input.consensus_alignment {
        fed_rep.consensus_alignment = v.clamp(0.0, 1.0);
    }

    // Recalculate aggregated score (domain boundaries enforced in calculate_aggregated)
    fed_rep.refresh_aggregation(now);

    // Update the entry
    let updated_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::FederatedReputation(fed_rep),
    )?;

    get(updated_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Updated reputation not found".into())))
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct UpdateFederatedReputationInput {
    pub identity_verification: Option<f64>,
    pub credential_count: Option<u64>,
    pub credential_quality: Option<f64>,
    pub epistemic_contributions: Option<f64>,
    pub factcheck_accuracy: Option<f64>,
    pub dark_spots_resolved: Option<u64>,
    pub stake_weight: Option<f64>,
    pub payment_reliability: Option<f64>,
    pub escrow_completion_rate: Option<f64>,
    pub pogq_score: Option<f64>,
    pub fl_contributions: Option<u64>,
    pub byzantine_clean_rate: Option<f64>,
    pub voting_participation: Option<f64>,
    pub proposal_success_rate: Option<f64>,
    pub consensus_alignment: Option<f64>,
}

/// Get votes for a consensus round
#[hdk_extern]
pub fn get_round_votes(input: GetRoundVotesInput) -> ExternResult<Vec<Record>> {
    let round_anchor = format!("round:{}:{}", input.proposal_id, input.round);
    let anchor = match anchor_hash(&round_anchor) {
        Ok(h) => h,
        Err(_) => return Ok(vec![]),
    };

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::RoundToVotes)?,
        GetStrategy::default(),
    )?;

    let mut votes = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            votes.push(record);
        }
    }

    Ok(votes)
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetRoundVotesInput {
    pub proposal_id: String,
    pub round: u64,
}

/// Calculate consensus round result
#[hdk_extern]
pub fn calculate_round_result(input: CalculateRoundResultInput) -> ExternResult<RoundResult> {
    let votes = get_round_votes(GetRoundVotesInput {
        proposal_id: input.proposal_id.clone(),
        round: input.round,
    })?;

    let threshold = AdaptiveThreshold::for_proposal_type(&input.proposal_type);

    let mut total_weight = 0.0;
    let mut weighted_approvals = 0.0;
    let mut weighted_rejections = 0.0;
    let mut vote_count = 0u64;

    for record in &votes {
        if let Some(vote) = record
            .entry()
            .to_app_option::<WeightedVote>()
            .ok()
            .flatten()
        {
            total_weight += vote.weight;
            vote_count += 1;

            match vote.decision {
                VoteDecision::Approve => weighted_approvals += vote.weight,
                VoteDecision::Reject => weighted_rejections += vote.weight,
                VoteDecision::Abstain => {} // Abstains add to total but not to either side
            }
        }
    }

    let required_threshold = threshold.calculate_threshold(total_weight);
    let consensus_reached = weighted_approvals > required_threshold;
    let rejected = weighted_rejections > required_threshold;

    let approval_percentage = if total_weight > 0.0 {
        (weighted_approvals / total_weight) * 100.0
    } else {
        0.0
    };

    let quorum_met = threshold.quorum_met(vote_count, input.eligible_voters);

    let result = if !quorum_met {
        "quorum_not_met"
    } else if consensus_reached {
        "approved"
    } else if rejected {
        "rejected"
    } else {
        "pending"
    };

    Ok(RoundResult {
        proposal_id: input.proposal_id,
        round: input.round,
        proposal_type: input.proposal_type,
        total_weight,
        weighted_approvals,
        weighted_rejections,
        vote_count,
        required_threshold,
        approval_percentage,
        quorum_met,
        consensus_reached,
        rejected,
        result: result.to_string(),
    })
}

#[derive(Serialize, Deserialize, Debug)]
pub struct CalculateRoundResultInput {
    pub proposal_id: String,
    pub round: u64,
    pub proposal_type: ProposalType,
    pub eligible_voters: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RoundResult {
    pub proposal_id: String,
    pub round: u64,
    pub proposal_type: ProposalType,
    pub total_weight: f64,
    pub weighted_approvals: f64,
    pub weighted_rejections: f64,
    pub vote_count: u64,
    pub required_threshold: f64,
    pub approval_percentage: f64,
    pub quorum_met: bool,
    pub consensus_reached: bool,
    pub rejected: bool,
    pub result: String,
}

// =============================================================================
// CROSS-CLUSTER DISPATCH TO PERSONAL
// =============================================================================

/// Allowed zomes in the personal cluster that governance can call
const ALLOWED_PERSONAL_ZOMES: &[&str] = &["personal_bridge"];

/// Dispatch a call to the personal cluster via OtherRole
///
/// Used by governance to request credential presentations (Phi, K-vector,
/// identity proofs) from the agent's personal vault via the personal bridge.
#[hdk_extern]
pub fn dispatch_personal_call(input: DispatchPersonalCallInput) -> ExternResult<ExternIO> {
    // Validate zome is in allowlist
    if !ALLOWED_PERSONAL_ZOMES.contains(&input.zome_name.as_str()) {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Zome '{}' not in ALLOWED_PERSONAL_ZOMES",
            input.zome_name
        ))));
    }

    // Validate function name length
    if input.fn_name.is_empty() || input.fn_name.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Function name must be 1-256 characters".into()
        )));
    }

    // Call the personal cluster via OtherRole
    match call(
        CallTargetCell::OtherRole("personal".into()),
        ZomeName::from(input.zome_name),
        FunctionName::from(input.fn_name),
        None,
        input.payload,
    )? {
        ZomeCallResponse::Ok(io) => Ok(io),
        ZomeCallResponse::NetworkError(e) => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Network error calling personal cluster: {}", e)
        ))),
        other => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Personal cluster call failed: {:?}", other)
        ))),
    }
}

/// Input for dispatching a call to the personal cluster
#[derive(Serialize, Deserialize, Debug)]
pub struct DispatchPersonalCallInput {
    pub zome_name: String,
    pub fn_name: String,
    pub payload: ExternIO,
}

/// Request Phi credential from agent's personal vault
///
/// Convenience wrapper around dispatch_personal_call for the common case
/// of requesting a Phi attestation for consciousness-gated governance.
#[hdk_extern]
pub fn request_phi_credential(_: ()) -> ExternResult<ExternIO> {
    match call(
        CallTargetCell::OtherRole("personal".into()),
        ZomeName::from("personal_bridge"),
        FunctionName::from("present_phi_credential"),
        None,
        ExternIO::encode(())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
    )? {
        ZomeCallResponse::Ok(io) => Ok(io),
        ZomeCallResponse::NetworkError(e) => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Network error calling personal cluster: {}", e)
        ))),
        other => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Personal cluster call failed: {:?}", other)
        ))),
    }
}

/// Request K-vector trust credential from agent's personal vault
///
/// Convenience wrapper for requesting K-vector trust data for weighted voting.
#[hdk_extern]
pub fn request_k_vector(_: ()) -> ExternResult<ExternIO> {
    match call(
        CallTargetCell::OtherRole("personal".into()),
        ZomeName::from("personal_bridge"),
        FunctionName::from("present_k_vector"),
        None,
        ExternIO::encode(())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
    )? {
        ZomeCallResponse::Ok(io) => Ok(io),
        ZomeCallResponse::NetworkError(e) => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Network error calling personal cluster: {}", e)
        ))),
        other => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Personal cluster call failed: {:?}", other)
        ))),
    }
}

/// Request identity proof from agent's personal vault
///
/// Convenience wrapper for requesting ZK identity proof without full profile disclosure.
#[hdk_extern]
pub fn request_identity_proof(_: ()) -> ExternResult<ExternIO> {
    match call(
        CallTargetCell::OtherRole("personal".into()),
        ZomeName::from("personal_bridge"),
        FunctionName::from("present_identity_proof"),
        None,
        ExternIO::encode(())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
    )? {
        ZomeCallResponse::Ok(io) => Ok(io),
        ZomeCallResponse::NetworkError(e) => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Network error calling personal cluster: {}", e)
        ))),
        other => Err(wasm_error!(WasmErrorInner::Guest(
            format!("Personal cluster call failed: {:?}", other)
        ))),
    }
}

// =============================================================================
// ADDITIONAL HELPER FUNCTIONS
// =============================================================================

/// Get the consensus participant for an agent
fn get_agent_participant(agent_did: &str) -> ExternResult<Option<ConsensusParticipant>> {
    let agent_anchor = format!("agent:{}", agent_did);
    let anchor = match anchor_hash(&agent_anchor) {
        Ok(h) => h,
        Err(_) => return Ok(None),
    };

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AgentToParticipant)?,
        GetStrategy::default(),
    )?;

    // Get the most recent participant record
    if let Some(link) = links.last() {
        let action_hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            return record
                .entry()
                .to_app_option::<ConsensusParticipant>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())));
        }
    }

    Ok(None)
}

/// Get the federated reputation for an agent
fn get_agent_federated_reputation(
    agent_did: &str,
) -> ExternResult<Option<(Record, FederatedReputation)>> {
    let agent_anchor = format!("agent:{}", agent_did);
    let anchor = match anchor_hash(&agent_anchor) {
        Ok(h) => h,
        Err(_) => return Ok(None),
    };

    let links = get_links(
        LinkQuery::try_new(anchor, LinkTypes::AgentToFederatedRep)?,
        GetStrategy::default(),
    )?;

    // Get the most recent federated reputation record
    if let Some(link) = links.last() {
        let action_hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(action_hash, GetOptions::default())? {
            let fed_rep = record
                .entry()
                .to_app_option::<FederatedReputation>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?;

            if let Some(fr) = fed_rep {
                return Ok(Some((record, fr)));
            }
        }
    }

    Ok(None)
}
