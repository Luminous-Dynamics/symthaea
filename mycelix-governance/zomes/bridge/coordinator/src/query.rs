use super::*;

#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    Ok(InitCallbackResult::Pass)
}

/// Query governance from another hApp
#[hdk_extern]
pub fn query_governance(input: QueryGovernanceInput) -> ExternResult<QueryGovernanceResult> {
    if input.source_happ.is_empty() || input.source_happ.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Source hApp must be 1-256 characters".into()
        )));
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
            let id = input
                .parameters
                .get("proposal_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            get_proposal_by_id_internal(id)
        }
        GovernanceQueryType::VotingEligibility => {
            let did = input
                .parameters
                .get("did")
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
    match governance_utils::call_local_best_effort("proposals", "get_proposal", id.to_string())? {
        Some(extern_io) => {
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
        None => Ok(QueryGovernanceResult {
            success: true,
            data: Some(serde_json::json!({"proposal_id": id, "found": false})),
            error: Some("Proposals zome unavailable".into()),
        }),
    }
}

fn check_voting_eligibility_internal(did: &str) -> ExternResult<QueryGovernanceResult> {
    match governance_utils::call_local_best_effort(
        "councils",
        "get_member_councils",
        did.to_string(),
    )? {
        Some(extern_io) => {
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
                Ok(QueryGovernanceResult {
                    success: false,
                    data: Some(
                        serde_json::json!({"did": did, "eligible": false, "voting_power": 0.0}),
                    ),
                    error: Some("Could not decode council membership response".into()),
                })
            }
        }
        None => Ok(QueryGovernanceResult {
            success: false,
            data: Some(serde_json::json!({"did": did, "eligible": false, "voting_power": 0.0})),
            error: Some("Councils zome unavailable — cannot verify eligibility".into()),
        }),
    }
}

/// Request execution from governance to another hApp
#[hdk_extern]
pub fn request_execution(input: RequestExecutionInput) -> ExternResult<Record> {
    check_execution_request_input(&input).map_err(|e| wasm_error!(WasmErrorInner::Guest(e)))?;

    // Phi gate: execution requests require ProposalSubmission level (Φ ≥ 0.3)
    let agent_info = agent_info()?;
    let agent_did = format!("did:mycelix:{}", agent_info.agent_initial_pubkey);
    let phi = match get_latest_agent_snapshot(&agent_did)? {
        Some((_, snapshot)) => snapshot.consciousness_level,
        None => 0.0,
    };
    let required = get_dynamic_consciousness_gate(&GovernanceActionType::ProposalSubmission)?;
    if phi < required {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Consciousness Φ ({:.2}) below threshold ({:.2}) for execution requests",
            phi, required
        ))));
    }

    let now = sys_time()?;

    let request = ExecutionRequest {
        id: format!(
            "exec:{}:{}:{}",
            input.proposal_id,
            input.target_happ,
            now.as_micros()
        ),
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Execution not found".into()
    )))
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
    check_broadcast_event_input(&input).map_err(|e| wasm_error!(WasmErrorInner::Guest(e)))?;

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
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Execution ID must be 1-256 characters".into()
        )));
    }
    if let Some(ref result) = input.result {
        if result.len() > 4096 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Result must be at most 4096 characters".into()
            )));
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
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Proposal ID must be 1-256 characters".into()
        )));
    }
    if input.title.is_empty() || input.title.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Title must be 1-256 characters".into()
        )));
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
    check_transfer_credits_input(&input).map_err(|e| wasm_error!(WasmErrorInner::Guest(e)))?;

    // Record the transfer as a governance event
    broadcast_governance_event(BroadcastGovernanceEventInput {
        event_type: GovernanceEventType::ProposalExecuted,
        proposal_id: None,
        subject: format!(
            "Credit transfer: {} -> {} ({} credits)",
            input.from, input.to, input.amount
        ),
        payload: serde_json::to_string(&serde_json::json!({
            "type": "transfer_credits",
            "from": input.from,
            "to": input.to,
            "amount": input.amount,
        }))
        .unwrap_or_default(),
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
