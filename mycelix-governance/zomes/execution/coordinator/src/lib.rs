//! Execution Coordinator Zome
//! Business logic for proposal execution
//!
//! Updated to use HDK 0.6 patterns

use hdk::prelude::*;
use execution_integrity::*;

/// Helper to get an anchor entry hash
fn anchor_hash(anchor_str: &str) -> ExternResult<EntryHash> {
    let anchor = Anchor(anchor_str.to_string());
    hash_entry(&EntryTypes::Anchor(anchor))
}

/// O(1) link-based lookup: find a timelock record by its string ID.
/// Falls back to O(n) chain scan if the link is missing (backwards compat).
fn find_timelock_by_id(timelock_id: &str) -> ExternResult<Record> {
    // Try link-based lookup first (O(1))
    let anchor_key = format!("tl:{}", timelock_id);
    if let Ok(entry_hash) = anchor_hash(&anchor_key) {
        if let Ok(links) = get_links(
            LinkQuery::try_new(entry_hash, LinkTypes::TimelockById)?,
            GetStrategy::default(),
        ) {
            if let Some(link) = links.into_iter().max_by_key(|l| l.timestamp) {
                if let Ok(ah) = ActionHash::try_from(link.target) {
                    if let Some(record) = get(ah, GetOptions::default())? {
                        return Ok(record);
                    }
                }
            }
        }
    }

    // Fallback: O(n) chain scan for timelocks created before the link was added
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Timelock,
        )?))
        .include_entries(true);

    let records = query(filter)?;

    for record in records {
        if let Some(tl) = record
            .entry()
            .to_app_option::<Timelock>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if tl.id == timelock_id {
                return Ok(record);
            }
        }
    }

    Err(wasm_error!(WasmErrorInner::Guest(
        "Timelock not found".into()
    )))
}

#[hdk_extern]
pub fn init(_: ()) -> ExternResult<InitCallbackResult> {
    // Pre-create the pending_timelocks anchor so queries never fail on empty DNA
    let anchor = Anchor("pending_timelocks".to_string());
    create_entry(&EntryTypes::Anchor(anchor))?;
    Ok(InitCallbackResult::Pass)
}

/// Create a timelock for an approved proposal
#[hdk_extern]
pub fn create_timelock(input: CreateTimelockInput) -> ExternResult<Record> {
    // Input validation
    if input.proposal_id.is_empty() || input.proposal_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Proposal ID must be 1-256 characters".into())));
    }
    if input.actions.is_empty() || input.actions.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("Actions must be 1-4096 characters".into())));
    }
    if input.duration_hours == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest("Duration must be at least 1 hour".into())));
    }

    let now = sys_time()?;
    let timelock_id = format!("timelock:{}:{}", input.proposal_id, now.as_micros());

    let timelock = Timelock {
        id: timelock_id,
        proposal_id: input.proposal_id.clone(),
        actions: input.actions,
        started: now,
        expires: Timestamp::from_micros(now.as_micros() as i64 + (input.duration_hours as i64 * 3600 * 1_000_000)),
        status: TimelockStatus::Pending,
        cancellation_reason: None,
    };

    let tl_id = timelock.id.clone();
    let action_hash = create_entry(&EntryTypes::Timelock(timelock))?;

    // Create anchor and link proposal to timelock
    let proposal_anchor = format!("proposal_timelock:{}", input.proposal_id);
    create_entry(&EntryTypes::Anchor(Anchor(proposal_anchor.clone())))?;
    create_link(
        anchor_hash(&proposal_anchor)?,
        action_hash.clone(),
        LinkTypes::ProposalToTimelock,
        (),
    )?;

    // Create anchor and link for O(1) lookup by timelock ID
    let tl_anchor = format!("tl:{}", tl_id);
    create_entry(&EntryTypes::Anchor(Anchor(tl_anchor.clone())))?;
    create_link(
        anchor_hash(&tl_anchor)?,
        action_hash.clone(),
        LinkTypes::TimelockById,
        (),
    )?;

    // Create anchor and link to pending timelocks
    create_entry(&EntryTypes::Anchor(Anchor("pending_timelocks".to_string())))?;
    create_link(
        anchor_hash("pending_timelocks")?,
        action_hash.clone(),
        LinkTypes::PendingTimelocks,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find timelock".into()
        )))
}

/// Input for creating a timelock
#[derive(Serialize, Deserialize, Debug)]
pub struct CreateTimelockInput {
    pub proposal_id: String,
    pub actions: String,
    pub duration_hours: u32,
}

/// Get timelock for a proposal
#[hdk_extern]
pub fn get_proposal_timelock(proposal_id: String) -> ExternResult<Option<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&format!("proposal_timelock:{}", proposal_id))?, LinkTypes::ProposalToTimelock)?,
        GetStrategy::default(),
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    let latest_link = links.into_iter().max_by_key(|l| l.timestamp);
    if let Some(link) = latest_link {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        return get(action_hash, GetOptions::default());
    }

    Ok(None)
}

/// Mark a timelock as ready for execution (after signature verification)
///
/// Transitions a timelock from Pending to Ready once pre-conditions are met
/// (e.g., threshold signature obtained, waiting period elapsed).
#[hdk_extern]
pub fn mark_timelock_ready(input: MarkTimelockReadyInput) -> ExternResult<Record> {
    if input.timelock_id.is_empty() || input.timelock_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Timelock ID must be 1-256 characters".into())));
    }

    // Find the timelock via O(1) link-based lookup
    let current_record = find_timelock_by_id(&input.timelock_id)?;

    let current_timelock: Timelock = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid timelock entry".into()
        )))?;

    if current_timelock.status != TimelockStatus::Pending {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Can only mark Pending timelocks as Ready, current status: {:?}",
            current_timelock.status
        ))));
    }

    let ready_timelock = Timelock {
        status: TimelockStatus::Ready,
        ..current_timelock
    };

    let action_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::Timelock(ready_timelock),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find updated timelock".into()
        )))
}

/// Input for marking a timelock as ready
#[derive(Serialize, Deserialize, Debug)]
pub struct MarkTimelockReadyInput {
    pub timelock_id: String,
}

/// Execute a ready timelock
#[hdk_extern]
pub fn execute_timelock(input: ExecuteTimelockInput) -> ExternResult<Record> {
    // Input validation
    if input.timelock_id.is_empty() || input.timelock_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Timelock ID must be 1-256 characters".into())));
    }
    if input.executor_did.is_empty() || input.executor_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Executor DID must be 1-256 characters".into())));
    }

    // Verify the executor DID matches the calling agent
    let agent = agent_info()?;
    let expected_did = format!("did:mycelix:{}", agent.agent_initial_pubkey);
    if input.executor_did != expected_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Executor DID must match the calling agent".into()
        )));
    }

    // Find the timelock via O(1) link-based lookup
    let current_record = find_timelock_by_id(&input.timelock_id)?;

    let current_timelock: Timelock = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid timelock entry".into()
        )))?;

    // Verify timelock is ready
    let now = sys_time()?;
    if now < current_timelock.expires {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Timelock has not expired yet".into()
        )));
    }

    // Timelock must be in Ready status (transitioned via mark_timelock_ready after
    // signature verification). Fall back to accepting Pending if threshold-signing
    // zome is not installed (graceful degradation).
    match current_timelock.status {
        TimelockStatus::Ready => {
            // Normal path — timelock was marked ready after signature verification
        }
        TimelockStatus::Pending => {
            // Check if threshold-signing zome is installed
            let sig_check = call(
                CallTargetCell::Local,
                ZomeName::from("threshold_signing"),
                FunctionName::from("get_proposal_signature"),
                None,
                ExternIO::encode(current_timelock.proposal_id.clone())
                    .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
            );

            match sig_check {
                Ok(ZomeCallResponse::Ok(extern_io)) => {
                    // Threshold-signing zome is installed — require Ready status
                    if let Ok(maybe_record) = extern_io.decode::<Option<Record>>() {
                        if maybe_record.is_none() {
                            return Err(wasm_error!(WasmErrorInner::Guest(
                                format!(
                                    "No verified threshold signature found for proposal '{}'. \
                                     Call mark_timelock_ready after obtaining a signature.",
                                    current_timelock.proposal_id
                                )
                            )));
                        }
                        // Signature exists — allow execution from Pending (backwards compat)
                    }
                }
                Ok(ZomeCallResponse::NetworkError(e)) => {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        format!("Network error checking threshold signature: {}", e)
                    )));
                }
                _ => {
                    // Threshold-signing zome not installed — graceful degradation
                    let _ = emit_signal(&serde_json::json!({
                        "type": "GovernanceWarning",
                        "warning": "threshold_signing_unavailable",
                        "message": format!(
                            "Threshold-signing zome not installed. Executing proposal '{}' without signature verification.",
                            current_timelock.proposal_id
                        ),
                    }));
                }
            }
        }
        other => {
            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                "Timelock must be in Ready or Pending status, current: {:?}",
                other
            ))));
        }
    }

    // Execute the actions via cross-zome dispatch
    let execution_result = execute_actions(&current_timelock.actions)?;

    let execution_id = format!("execution:{}:{}", input.timelock_id, now.as_micros());

    let execution = Execution {
        id: execution_id,
        timelock_id: input.timelock_id.clone(),
        proposal_id: current_timelock.proposal_id.clone(),
        executor: input.executor_did,
        status: if execution_result.success {
            ExecutionStatus::Success
        } else {
            ExecutionStatus::Failed
        },
        result: execution_result.result,
        error: execution_result.error,
        executed_at: now,
    };

    let action_hash = create_entry(&EntryTypes::Execution(execution))?;

    // Update timelock status
    let updated_timelock = Timelock {
        id: current_timelock.id.clone(),
        proposal_id: current_timelock.proposal_id.clone(),
        actions: current_timelock.actions.clone(),
        started: current_timelock.started,
        expires: current_timelock.expires,
        status: if execution_result.success {
            TimelockStatus::Executed
        } else {
            TimelockStatus::Failed
        },
        cancellation_reason: None,
    };

    update_entry(
        current_record.action_address().clone(),
        &EntryTypes::Timelock(updated_timelock),
    )?;

    // Clean up pending_timelocks link (timelock is no longer pending)
    if let Ok(pending_links) = get_links(
        LinkQuery::try_new(anchor_hash("pending_timelocks")?, LinkTypes::PendingTimelocks)?,
        GetStrategy::default(),
    ) {
        for link in pending_links {
            if let Ok(target_hash) = ActionHash::try_from(link.target.clone()) {
                if let Ok(Some(record)) = get(target_hash, GetOptions::default()) {
                    if let Some(tl) = record.entry().to_app_option::<Timelock>().ok().flatten() {
                        if tl.id == current_timelock.id {
                            let _ = delete_link(link.create_link_hash, GetOptions::default());
                        }
                    }
                }
            }
        }
    }

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find execution".into()
        )))
}

/// Input for executing a timelock
#[derive(Serialize, Deserialize, Debug)]
pub struct ExecuteTimelockInput {
    pub timelock_id: String,
    pub executor_did: String,
}

/// Result of executing actions
struct ActionExecutionResult {
    success: bool,
    result: Option<String>,
    error: Option<String>,
}

/// Typed governance action — deserialized from the actions JSON string
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
enum GovernanceAction {
    TransferCredits {
        from: String,
        to: String,
        amount: f64,
    },
    UpdateParameter {
        parameter: String,
        value: String,
    },
    EmitEvent {
        event: String,
        #[serde(default)]
        payload: serde_json::Value,
    },
}

impl GovernanceAction {
    /// Validate action parameters
    fn validate(&self) -> Result<(), String> {
        match self {
            GovernanceAction::TransferCredits { from, to, amount } => {
                if from.is_empty() {
                    return Err("TransferCredits: 'from' is required".to_string());
                }
                if to.is_empty() {
                    return Err("TransferCredits: 'to' is required".to_string());
                }
                if *amount <= 0.0 {
                    return Err(format!("TransferCredits: amount must be positive, got {}", amount));
                }
                if !amount.is_finite() {
                    return Err("TransferCredits: amount must be finite".to_string());
                }
                Ok(())
            }
            GovernanceAction::UpdateParameter { parameter, .. } => {
                if parameter.is_empty() {
                    return Err("UpdateParameter: 'parameter' name is required".to_string());
                }
                Ok(())
            }
            GovernanceAction::EmitEvent { .. } => Ok(()),
        }
    }

    /// Execute the action via cross-zome dispatch
    fn execute(&self) -> ExternResult<String> {
        match self {
            GovernanceAction::TransferCredits { from, to, amount } => {
                let transfer_input = serde_json::json!({
                    "from": from,
                    "to": to,
                    "amount": amount,
                });
                match call(
                    CallTargetCell::Local,
                    ZomeName::from("governance_bridge"),
                    FunctionName::from("transfer_credits"),
                    None,
                    ExternIO::encode(transfer_input)
                        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
                )? {
                    ZomeCallResponse::Ok(_) => {
                        Ok(format!("TransferCredits: {} -> {} ({} credits) [executed]", from, to, amount))
                    }
                    other => {
                        // Bridge unavailable or call failed — log but don't hard-fail
                        Ok(format!(
                            "TransferCredits: {} -> {} ({} credits) [bridge unavailable: {:?}]",
                            from, to, amount, other
                        ))
                    }
                }
            }
            GovernanceAction::UpdateParameter { parameter, value } => {
                let update_input = serde_json::json!({
                    "parameter": parameter,
                    "value": value,
                });
                match call(
                    CallTargetCell::Local,
                    ZomeName::from("constitution"),
                    FunctionName::from("update_parameter"),
                    None,
                    ExternIO::encode(update_input)
                        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
                )? {
                    ZomeCallResponse::Ok(_) => {
                        Ok(format!("UpdateParameter: {} = {} [executed]", parameter, value))
                    }
                    other => {
                        Ok(format!(
                            "UpdateParameter: {} = {} [constitution zome unavailable: {:?}]",
                            parameter, value, other
                        ))
                    }
                }
            }
            GovernanceAction::EmitEvent { event, payload } => {
                // Emit as a governance signal to connected clients
                let _ = emit_signal(&serde_json::json!({
                    "type": "GovernanceActionExecuted",
                    "event": event,
                    "payload": payload,
                }));
                Ok(format!("EmitEvent: {} [emitted]", event))
            }
        }
    }
}

/// Execute actions parsed from JSON via cross-zome dispatch
fn execute_actions(actions_json: &str) -> ExternResult<ActionExecutionResult> {
    // Parse as typed enum array (or single action)
    let actions: Vec<GovernanceAction> = match serde_json::from_str(actions_json) {
        Ok(a) => a,
        Err(_) => {
            match serde_json::from_str::<GovernanceAction>(actions_json) {
                Ok(v) => vec![v],
                Err(e) => {
                    return Ok(ActionExecutionResult {
                        success: false,
                        result: None,
                        error: Some(format!("Failed to parse actions: {}. Expected GovernanceAction with type TransferCredits, UpdateParameter, or EmitEvent", e)),
                    });
                }
            }
        }
    };

    let mut results = Vec::new();

    for (i, action) in actions.iter().enumerate() {
        if let Err(msg) = action.validate() {
            return Ok(ActionExecutionResult {
                success: false,
                result: Some(format!("Executed {} of {} actions before failure", i, actions.len())),
                error: Some(format!("Action {}: {}", i, msg)),
            });
        }
        match action.execute() {
            Ok(description) => results.push(description),
            Err(e) => {
                return Ok(ActionExecutionResult {
                    success: false,
                    result: Some(format!("Executed {} of {} actions before failure", i, actions.len())),
                    error: Some(format!("Action {} execution failed: {}", i, e)),
                });
            }
        }
    }

    Ok(ActionExecutionResult {
        success: true,
        result: Some(results.join("; ")),
        error: None,
    })
}

/// Cancel a timelock (guardian veto)
#[hdk_extern]
pub fn veto_timelock(input: VetoTimelockInput) -> ExternResult<Record> {
    // Input validation
    if input.timelock_id.is_empty() || input.timelock_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Timelock ID must be 1-256 characters".into())));
    }
    if input.guardian_did.is_empty() || input.guardian_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Guardian DID must be 1-256 characters".into())));
    }
    if input.reason.is_empty() || input.reason.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest("Veto reason must be 1-4096 characters".into())));
    }

    // Verify the guardian DID matches the calling agent
    let agent = agent_info()?;
    let expected_did = format!("did:mycelix:{}", agent.agent_initial_pubkey);
    if input.guardian_did != expected_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Guardian DID must match the calling agent".into()
        )));
    }

    // Verify guardian role: caller must be a member of at least one council
    let guardian_check = call(
        CallTargetCell::Local,
        ZomeName::from("councils"),
        FunctionName::from("get_member_councils"),
        None,
        ExternIO::encode(input.guardian_did.clone())
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?,
    );

    match guardian_check {
        Ok(ZomeCallResponse::Ok(extern_io)) => {
            if let Ok(councils) = extern_io.decode::<Vec<Record>>() {
                if councils.is_empty() {
                    return Err(wasm_error!(WasmErrorInner::Guest(
                        "Only council members (guardians) can veto timelocks".into()
                    )));
                }
            }
        }
        _ => {
            // Councils zome unavailable — fail closed for veto power
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Cannot verify guardian role: councils zome unavailable".into()
            )));
        }
    }

    let now = sys_time()?;
    let veto_id = format!("veto:{}:{}", input.timelock_id, now.as_micros());

    let veto = GuardianVeto {
        id: veto_id,
        timelock_id: input.timelock_id.clone(),
        guardian: input.guardian_did.clone(),
        reason: input.reason.clone(),
        vetoed_at: now,
    };

    let action_hash = create_entry(&EntryTypes::GuardianVeto(veto))?;

    // Update timelock status to cancelled via O(1) link-based lookup
    let tl_record = find_timelock_by_id(&input.timelock_id)?;
    let tl: Timelock = tl_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid timelock entry".into())))?;
    if matches!(tl.status, TimelockStatus::Pending | TimelockStatus::Ready) {
        let cancelled = Timelock {
            status: TimelockStatus::Cancelled,
            cancellation_reason: Some(input.reason.clone()),
            ..tl
        };
        update_entry(
            tl_record.action_address().clone(),
            &EntryTypes::Timelock(cancelled),
        )?;
    }

    // Clean up pending_timelocks link (timelock was vetoed/cancelled)
    if let Ok(pending_links) = get_links(
        LinkQuery::try_new(anchor_hash("pending_timelocks")?, LinkTypes::PendingTimelocks)?,
        GetStrategy::default(),
    ) {
        for link in pending_links {
            if let Ok(target_hash) = ActionHash::try_from(link.target.clone()) {
                if let Ok(Some(record)) = get(target_hash, GetOptions::default()) {
                    if let Some(tl) = record.entry().to_app_option::<Timelock>().ok().flatten() {
                        if tl.id == input.timelock_id {
                            let _ = delete_link(link.create_link_hash, GetOptions::default());
                        }
                    }
                }
            }
        }
    }

    // Create anchor and link guardian to veto
    let guardian_anchor = format!("guardian:{}", input.guardian_did);
    create_entry(&EntryTypes::Anchor(Anchor(guardian_anchor.clone())))?;
    create_link(
        anchor_hash(&guardian_anchor)?,
        action_hash.clone(),
        LinkTypes::GuardianToVeto,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Could not find veto".into()
        )))
}

/// Input for vetoing a timelock
#[derive(Serialize, Deserialize, Debug)]
pub struct VetoTimelockInput {
    pub timelock_id: String,
    pub guardian_did: String,
    pub reason: String,
}

/// Lock funds in escrow for a proposal's execution.
///
/// Called after a proposal is approved and before a timelock is created.
/// Creates a `FundAllocation` entry with status `Locked` and links it
/// to the proposal.
#[hdk_extern]
pub fn lock_proposal_funds(input: LockFundsInput) -> ExternResult<Record> {
    if input.proposal_id.is_empty() || input.proposal_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest("Proposal ID must be 1-256 characters".into())));
    }
    if input.source_account.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest("Source account is required".into())));
    }
    if input.amount <= 0.0 || !input.amount.is_finite() {
        return Err(wasm_error!(WasmErrorInner::Guest("Amount must be positive and finite".into())));
    }

    // Check for existing locked allocation for this proposal
    if let Some(_existing) = find_fund_allocation_for_proposal(&input.proposal_id)? {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Funds already locked for proposal '{}'", input.proposal_id)
        )));
    }

    let now = sys_time()?;
    let alloc_id = format!("alloc:{}:{}", input.proposal_id, now.as_micros());

    let alloc = FundAllocation {
        id: alloc_id,
        proposal_id: input.proposal_id.clone(),
        timelock_id: input.timelock_id.unwrap_or_default(),
        source_account: input.source_account,
        amount: input.amount,
        currency: input.currency.unwrap_or_else(|| "credits".to_string()),
        locked_at: now,
        status: AllocationStatus::Locked,
        status_reason: None,
    };

    let action_hash = create_entry(&EntryTypes::FundAllocation(alloc))?;

    // Link proposal to fund allocation
    let alloc_anchor = format!("fund_alloc:{}", input.proposal_id);
    create_entry(&EntryTypes::Anchor(Anchor(alloc_anchor.clone())))?;
    create_link(
        anchor_hash(&alloc_anchor)?,
        action_hash.clone(),
        LinkTypes::ProposalToFundAllocation,
        (),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find fund allocation".into())))
}

/// Input for locking funds
#[derive(Serialize, Deserialize, Debug)]
pub struct LockFundsInput {
    pub proposal_id: String,
    pub timelock_id: Option<String>,
    pub source_account: String,
    pub amount: f64,
    pub currency: Option<String>,
}

/// Release locked funds after successful execution
#[hdk_extern]
pub fn release_locked_funds(input: ReleaseFundsInput) -> ExternResult<Record> {
    let (record, alloc) = find_fund_allocation_for_proposal(&input.proposal_id)?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            format!("No fund allocation found for proposal '{}'", input.proposal_id)
        )))?;

    if alloc.status != AllocationStatus::Locked {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Allocation is not locked (current status: {:?})", alloc.status)
        )));
    }

    let released = FundAllocation {
        status: AllocationStatus::Released,
        status_reason: Some(input.reason.unwrap_or_else(|| "Execution completed successfully".to_string())),
        ..alloc
    };

    let action_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::FundAllocation(released),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find updated allocation".into())))
}

/// Input for releasing funds
#[derive(Serialize, Deserialize, Debug)]
pub struct ReleaseFundsInput {
    pub proposal_id: String,
    pub reason: Option<String>,
}

/// Refund locked funds (e.g., after veto or expiration)
#[hdk_extern]
pub fn refund_locked_funds(input: RefundFundsInput) -> ExternResult<Record> {
    let (record, alloc) = find_fund_allocation_for_proposal(&input.proposal_id)?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            format!("No fund allocation found for proposal '{}'", input.proposal_id)
        )))?;

    if alloc.status != AllocationStatus::Locked {
        return Err(wasm_error!(WasmErrorInner::Guest(
            format!("Allocation is not locked (current status: {:?})", alloc.status)
        )));
    }

    let refunded = FundAllocation {
        status: AllocationStatus::Refunded,
        status_reason: Some(input.reason),
        ..alloc
    };

    let action_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::FundAllocation(refunded),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest("Could not find updated allocation".into())))
}

/// Input for refunding funds
#[derive(Serialize, Deserialize, Debug)]
pub struct RefundFundsInput {
    pub proposal_id: String,
    pub reason: String,
}

/// Query fund allocation status for a proposal
#[hdk_extern]
pub fn get_fund_allocation(proposal_id: String) -> ExternResult<Option<Record>> {
    Ok(find_fund_allocation_for_proposal(&proposal_id)?.map(|(r, _)| r))
}

/// Internal helper: find the active fund allocation for a proposal
fn find_fund_allocation_for_proposal(proposal_id: &str) -> ExternResult<Option<(Record, FundAllocation)>> {
    let alloc_anchor = format!("fund_alloc:{}", proposal_id);
    let links = get_links(
        LinkQuery::try_new(anchor_hash(&alloc_anchor)?, LinkTypes::ProposalToFundAllocation)?,
        GetStrategy::default(),
    )?;

    // Find the most recent allocation
    let latest_link = links.into_iter().max_by_key(|l| l.timestamp);
    if let Some(link) = latest_link {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            let alloc: FundAllocation = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or(wasm_error!(WasmErrorInner::Guest("Invalid allocation entry".into())))?;
            return Ok(Some((record, alloc)));
        }
    }
    Ok(None)
}

/// Get pending timelocks
#[hdk_extern]
pub fn get_pending_timelocks(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(anchor_hash("pending_timelocks")?, LinkTypes::PendingTimelocks)?,
        GetStrategy::default(),
    )?;

    let mut timelocks = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            // Filter to only actually pending timelocks
            if let Some(tl) = record
                .entry()
                .to_app_option::<Timelock>()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            {
                if tl.status == TimelockStatus::Pending {
                    timelocks.push(record);
                }
            }
        }
    }

    Ok(timelocks)
}
