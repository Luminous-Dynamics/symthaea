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

    // Find the timelock
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::Timelock,
        )?))
        .include_entries(true);

    let records = query(filter)?;

    let mut timelock_record: Option<Record> = None;
    for record in records {
        if let Some(tl) = record
            .entry()
            .to_app_option::<Timelock>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if tl.id == input.timelock_id {
                timelock_record = Some(record);
                break;
            }
        }
    }

    let current_record = timelock_record.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Timelock not found".into()
    )))?;

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

    if current_timelock.status != TimelockStatus::Pending {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Timelock is not in pending status".into()
        )));
    }

    // Execute the actions (simplified - would parse and execute in production)
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

    /// Execute the action and return a description
    fn execute(&self) -> String {
        match self {
            GovernanceAction::TransferCredits { from, to, amount } => {
                // In a full implementation, this would call the finance zome
                format!("TransferCredits: {} -> {} ({} credits)", from, to, amount)
            }
            GovernanceAction::UpdateParameter { parameter, value } => {
                format!("UpdateParameter: {} = {}", parameter, value)
            }
            GovernanceAction::EmitEvent { event, payload } => {
                format!("EmitEvent: {} (payload: {})", event, payload)
            }
        }
    }
}

/// Execute actions parsed from JSON
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
        results.push(action.execute());
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

    // Update timelock status to cancelled
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
            if tl.id == input.timelock_id && tl.status == TimelockStatus::Pending {
                let cancelled = Timelock {
                    id: tl.id,
                    proposal_id: tl.proposal_id,
                    actions: tl.actions,
                    started: tl.started,
                    expires: tl.expires,
                    status: TimelockStatus::Cancelled,
                    cancellation_reason: Some(input.reason.clone()),
                };
                update_entry(
                    record.action_address().clone(),
                    &EntryTypes::Timelock(cancelled),
                )?;
                break;
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
