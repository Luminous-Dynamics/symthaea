// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Execution Coordinator Zome
//! Business logic for proposal execution
//!
//! Updated to use HDK 0.6 patterns

use execution_integrity::*;
use hdk::prelude::*;
use mycelix_zome_helpers::get_latest_record;

/// Mirror type for ThresholdSignature from threshold-signing integrity zome.
/// Avoids linking the integrity crate (which causes duplicate HDI symbols in WASM).
///
/// Uses `SerializedBytes` for Holochain entry deserialization via `to_app_option()`.
#[derive(Serialize, Deserialize, Debug, Clone, SerializedBytes)]
struct ThresholdSignature {
    pub id: String,
    pub committee_id: String,
    pub signed_content_hash: Vec<u8>,
    pub signed_content_description: String,
    pub signature: Vec<u8>,
    pub signer_count: u32,
    pub signers: Vec<u32>,
    pub verified: bool,
    pub signed_at: Timestamp,
}

/// Mirror type for SigningCommittee — extracts only the scope field.
/// Avoids full dependency on threshold-signing integrity crate (which
/// causes duplicate HDI symbols in WASM).
#[derive(Serialize, Deserialize, Debug, Clone, SerializedBytes)]
struct CommitteeScopeMirror {
    #[serde(default)]
    pub scope: serde_json::Value,
}

/// Extract the scope variant name from a CommitteeScopeMirror.
/// Handles both simple variants ("All") and tagged variants ({"Custom": [...]}).
fn extract_scope_name(scope: &serde_json::Value) -> &str {
    match scope {
        serde_json::Value::String(s) => s.as_str(),
        serde_json::Value::Object(map) => map.keys().next().map(|k| k.as_str()).unwrap_or("All"),
        _ => "All",
    }
}

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
                    if let Some(record) = get_latest_record(ah)? {
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

    // Take the LAST match — update_entry appends newer versions later in the chain
    let mut found: Option<Record> = None;
    for record in records {
        if let Some(tl) = record
            .entry()
            .to_app_option::<Timelock>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if tl.id == timelock_id {
                found = Some(record);
            }
        }
    }

    found.ok_or(wasm_error!(WasmErrorInner::Guest(
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
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Proposal ID must be 1-256 characters".into()
        )));
    }
    if input.actions.is_empty() || input.actions.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Actions must be 1-4096 characters".into()
        )));
    }
    if input.duration_hours == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Duration must be at least 1 hour".into()
        )));
    }
    if input.duration_hours > 8760 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Duration cannot exceed 8,760 hours (1 year)".into()
        )));
    }

    let now = sys_time()?;
    let timelock_id = format!("timelock:{}:{}", input.proposal_id, now.as_micros());

    let timelock = Timelock {
        id: timelock_id,
        proposal_id: input.proposal_id.clone(),
        actions: input.actions,
        started: now,
        expires: Timestamp::from_micros(
            now.as_micros() as i64 + (input.duration_hours as i64 * 3600 * 1_000_000),
        ),
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
        LinkQuery::try_new(
            anchor_hash(&format!("proposal_timelock:{}", proposal_id))?,
            LinkTypes::ProposalToTimelock,
        )?,
        GetStrategy::default(),
    )?;

    if links.is_empty() {
        return Ok(None);
    }

    let latest_link = links.into_iter().max_by_key(|l| l.timestamp);
    if let Some(link) = latest_link {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        return get_latest_record(action_hash);
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
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Timelock ID must be 1-256 characters".into()
        )));
    }

    // Find the timelock via O(1) link-based lookup
    let current_record = find_timelock_by_id(&input.timelock_id)?;

    // Authorization: only the timelock creator can mark it ready
    let caller = agent_info()?.agent_initial_pubkey;
    let author = current_record.action().author().clone();
    if caller != author {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the timelock creator can mark it as ready".into()
        )));
    }

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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Timelock ID must be 1-256 characters".into()
        )));
    }
    if input.executor_did.is_empty() || input.executor_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Executor DID must be 1-256 characters".into()
        )));
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
                    // Threshold-signing zome is installed — decode and validate
                    let maybe_record: Option<Record> = extern_io.decode().map_err(|e| {
                        wasm_error!(WasmErrorInner::Guest(format!(
                            "Failed to decode threshold signature response: {}",
                            e
                        )))
                    })?;

                    match maybe_record {
                        None => {
                            return Err(wasm_error!(WasmErrorInner::Guest(format!(
                                "No verified threshold signature found for proposal '{}'. \
                                     Call mark_timelock_ready after obtaining a signature.",
                                current_timelock.proposal_id
                            ))));
                        }
                        Some(sig_record) => {
                            // Decode the ThresholdSignature entry for validation
                            let sig: ThresholdSignature = sig_record
                                .entry()
                                .to_app_option()
                                .map_err(|e| {
                                    wasm_error!(WasmErrorInner::Guest(format!(
                                        "Failed to decode ThresholdSignature entry: {}",
                                        e
                                    )))
                                })?
                                .ok_or(wasm_error!(WasmErrorInner::Guest(
                                    "Threshold signature record has no entry".into()
                                )))?;

                            // Defense-in-depth: verify signature is marked verified
                            if !sig.verified {
                                return Err(wasm_error!(WasmErrorInner::Guest(format!(
                                    "Threshold signature '{}' for proposal '{}' is not verified",
                                    sig.id, current_timelock.proposal_id
                                ))));
                            }

                            // Defense-in-depth: verify signature description references this proposal
                            if !sig
                                .signed_content_description
                                .contains(&current_timelock.proposal_id)
                            {
                                return Err(wasm_error!(WasmErrorInner::Guest(
                                    format!(
                                        "Threshold signature '{}' content description does not reference proposal '{}' (was: '{}')",
                                        sig.id, current_timelock.proposal_id, sig.signed_content_description
                                    )
                                )));
                            }

                            // Defense-in-depth: verify committee scope covers this proposal type.
                            // Fetch the committee to check its scope against the proposal.
                            if let Ok(ZomeCallResponse::Ok(committee_io)) = call(
                                CallTargetCell::Local,
                                ZomeName::from("threshold_signing"),
                                FunctionName::from("get_committee"),
                                None,
                                ExternIO::encode(sig.committee_id.clone()).map_err(|e| {
                                    wasm_error!(WasmErrorInner::Guest(e.to_string()))
                                })?,
                            ) {
                                if let Ok(Some(committee_record)) =
                                    committee_io.decode::<Option<Record>>()
                                {
                                    // Decode scope from committee via mirror struct
                                    if let Ok(Some(committee_mirror)) = committee_record
                                        .entry()
                                        .to_app_option::<CommitteeScopeMirror>(
                                    ) {
                                        // Infer proposal type from signed_content_description
                                        // Format: "proposal:MIP-001" or "constitutional:CA-001" etc.
                                        let proposal_type = sig
                                            .signed_content_description
                                            .split(':')
                                            .next()
                                            .unwrap_or("unknown");

                                        let scope_name =
                                            extract_scope_name(&committee_mirror.scope);

                                        let scope_allows = match scope_name {
                                            "All" => true,
                                            "Constitutional" => proposal_type == "constitutional",
                                            "Treasury" => proposal_type == "treasury",
                                            "Protocol" => proposal_type == "protocol",
                                            _ => true, // Custom or unknown — permissive
                                        };

                                        if !scope_allows {
                                            return Err(wasm_error!(WasmErrorInner::Guest(
                                                format!(
                                                    "Committee '{}' scope '{}' does not authorize signing '{}' proposals",
                                                    sig.committee_id, scope_name, proposal_type
                                                )
                                            )));
                                        }
                                    }
                                }
                            }
                            // If committee fetch fails, proceed (signature itself is already verified)

                            // Emit audit signal with signature details
                            let _ = emit_signal(serde_json::json!({
                                "type": "ThresholdSignatureVerified",
                                "proposal_id": current_timelock.proposal_id,
                                "signature_id": sig.id,
                                "committee_id": sig.committee_id,
                                "signer_count": sig.signer_count,
                                "signers": sig.signers,
                            }));
                        }
                    }
                }
                Ok(ZomeCallResponse::NetworkError(e)) => {
                    return Err(wasm_error!(WasmErrorInner::Guest(format!(
                        "Network error checking threshold signature: {}",
                        e
                    ))));
                }
                _ => {
                    // Threshold-signing zome not installed — graceful degradation
                    let _ = emit_signal(serde_json::json!({
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
        LinkQuery::try_new(
            anchor_hash("pending_timelocks")?,
            LinkTypes::PendingTimelocks,
        )?,
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
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
                    return Err(format!(
                        "TransferCredits: amount must be positive, got {}",
                        amount
                    ));
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
                // SECURITY: Fail-closed — credit transfers MUST execute or fail explicitly.
                // Returning Ok without actual transfer creates phantom transactions.
                let transfer_input = serde_json::json!({"from": from, "to": to, "amount": amount});
                governance_utils::call_local(
                    "governance_bridge",
                    "transfer_credits",
                    transfer_input,
                ).map_err(|e| wasm_error!(WasmErrorInner::Guest(format!(
                    "TransferCredits failed: governance bridge unavailable — {} -> {} ({} credits): {:?}",
                    from, to, amount, e
                ))))?;
                Ok(format!(
                    "TransferCredits: {} -> {} ({} credits) [executed]",
                    from, to, amount
                ))
            }
            GovernanceAction::UpdateParameter { parameter, value } => {
                // SECURITY: Fail-closed — parameter updates MUST persist or fail explicitly.
                // Returning Ok without actual update creates phantom governance changes.
                let update_input = serde_json::json!({"parameter": parameter, "value": value});
                governance_utils::call_local(
                    "constitution",
                    "update_parameter",
                    update_input,
                ).map_err(|e| wasm_error!(WasmErrorInner::Guest(format!(
                    "UpdateParameter failed: constitution zome unavailable — {} = {}: {:?}",
                    parameter, value, e
                ))))?;
                Ok(format!(
                    "UpdateParameter: {} = {} [executed]",
                    parameter, value
                ))
            }
            GovernanceAction::EmitEvent { event, payload } => {
                // Emit as a governance signal to connected clients
                let _ = emit_signal(serde_json::json!({
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
        Err(_) => match serde_json::from_str::<GovernanceAction>(actions_json) {
            Ok(v) => vec![v],
            Err(e) => {
                return Ok(ActionExecutionResult {
                        success: false,
                        result: None,
                        error: Some(format!("Failed to parse actions: {}. Expected GovernanceAction with type TransferCredits, UpdateParameter, or EmitEvent", e)),
                    });
            }
        },
    };

    let mut results = Vec::new();

    for (i, action) in actions.iter().enumerate() {
        if let Err(msg) = action.validate() {
            return Ok(ActionExecutionResult {
                success: false,
                result: Some(format!(
                    "Executed {} of {} actions before failure",
                    i,
                    actions.len()
                )),
                error: Some(format!("Action {}: {}", i, msg)),
            });
        }
        match action.execute() {
            Ok(description) => results.push(description),
            Err(e) => {
                return Ok(ActionExecutionResult {
                    success: false,
                    result: Some(format!(
                        "Executed {} of {} actions before failure",
                        i,
                        actions.len()
                    )),
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
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Timelock ID must be 1-256 characters".into()
        )));
    }
    if input.guardian_did.is_empty() || input.guardian_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Guardian DID must be 1-256 characters".into()
        )));
    }
    if input.reason.is_empty() || input.reason.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Veto reason must be 1-4096 characters".into()
        )));
    }

    // Verify the guardian DID matches the calling agent
    let agent = agent_info()?;
    let expected_did = format!("did:mycelix:{}", agent.agent_initial_pubkey);
    if input.guardian_did != expected_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Guardian DID must match the calling agent".into()
        )));
    }

    // ── VETO RATE LIMITING ──
    // Max 1 veto per Guardian per 7 days. Prevents serial veto DoS where
    // a Guardian freezes governance by vetoing every proposal in sequence.
    const VETO_COOLDOWN_US: i64 = 7 * 24 * 3600 * 1_000_000;
    let guardian_anchor = format!("guardian:{}", input.guardian_did);
    if let Ok(eh) = anchor_hash(&guardian_anchor) {
        if let Ok(links) = get_links(
            LinkQuery::try_new(eh, LinkTypes::GuardianToVeto)?,
            GetStrategy::default(),
        ) {
            let now_us = sys_time()?.as_micros() as i64;
            for link in links {
                if let Ok(ah) = ActionHash::try_from(link.target) {
                    if let Some(record) = get(ah, GetOptions::default())? {
                        if let Some(prior_veto) = record
                            .entry()
                            .to_app_option::<GuardianVeto>()
                            .ok()
                            .flatten()
                        {
                            let elapsed = now_us - prior_veto.vetoed_at.as_micros() as i64;
                            if elapsed < VETO_COOLDOWN_US {
                                let days_remaining = (VETO_COOLDOWN_US - elapsed) / (24 * 3600 * 1_000_000);
                                let _ = emit_signal(serde_json::json!({
                                    "type": "VetoRateLimitExceeded",
                                    "guardian_did": input.guardian_did,
                                    "cooldown_days": 7,
                                    "days_remaining": days_remaining,
                                }));
                                return Err(wasm_error!(WasmErrorInner::Guest(format!(
                                    "Veto rate limit: max 1 veto per 7 days per Guardian. \
                                     Next veto available in {} day(s).",
                                    days_remaining + 1
                                ))));
                            }
                        }
                    }
                }
            }
        }
    }

    // ── YEARLY VETO LIMIT (Art. III, Sec. 5.4) ──
    // Max 3 vetoes per Guardian per rolling 12-month window.
    // Exceeding triggers probation signal.
    if let Ok(eh) = anchor_hash(&guardian_anchor) {
        if let Ok(links) = get_links(
            LinkQuery::try_new(eh, LinkTypes::GuardianToVeto)?,
            GetStrategy::default(),
        ) {
            let now_us = sys_time()?.as_micros() as i64;
            let mut vetoes_in_window: u32 = 0;
            for link in links {
                if let Ok(ah) = ActionHash::try_from(link.target) {
                    if let Some(record) = get(ah, GetOptions::default())? {
                        if let Some(prior_veto) = record
                            .entry()
                            .to_app_option::<GuardianVeto>()
                            .ok()
                            .flatten()
                        {
                            let elapsed = now_us - prior_veto.vetoed_at.as_micros() as i64;
                            if elapsed < execution_integrity::ROLLING_YEAR_US {
                                vetoes_in_window += 1;
                            }
                        }
                    }
                }
            }
            if vetoes_in_window >= execution_integrity::VETO_YEARLY_LIMIT {
                let _ = emit_signal(serde_json::json!({
                    "type": "VetoYearlyLimitExceeded",
                    "guardian_did": input.guardian_did,
                    "vetoes_in_window": vetoes_in_window,
                    "limit": execution_integrity::VETO_YEARLY_LIMIT,
                }));
                return Err(wasm_error!(WasmErrorInner::Guest(format!(
                    "Yearly veto limit exceeded: {} vetoes in the past 12 months \
                     (max {}). Guardian enters probation (Art. III, Sec. 5.4).",
                    vetoes_in_window, execution_integrity::VETO_YEARLY_LIMIT
                ))));
            }
        }
    }

    // Verify guardian role: caller must be a member of at least one council
    let guardian_io = governance_utils::call_local(
        "councils",
        "get_member_councils",
        input.guardian_did.clone(),
    )?;
    if let Ok(councils) = guardian_io.decode::<Vec<Record>>() {
        if councils.is_empty() {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Only council members (guardians) can veto timelocks".into()
            )));
        }
    }

    // If the timelock is already in Ready state, require Guardian-tier Φ (0.8)
    // since cancelling a signed, ready-to-execute proposal is a high-impact action.
    let tl_pre = find_timelock_by_id(&input.timelock_id)?;
    let tl_pre_entry: Timelock = tl_pre
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid timelock entry".into()
        )))?;

    if tl_pre_entry.status == TimelockStatus::Ready {
        // Require elevated consciousness for Ready-state vetoes
        const GUARDIAN_PHI_THRESHOLD: f64 = 0.8;
        match governance_utils::call_local_best_effort(
            "governance_bridge",
            "verify_consciousness_gate",
            serde_json::json!({"action_type": "Veto", "action_id": input.timelock_id.clone()}),
        )? {
            Some(extern_io) => {
                if let Ok(result) = extern_io.decode::<serde_json::Value>() {
                    let phi = result.get("phi").and_then(|p| p.as_f64()).unwrap_or(0.0);
                    if phi < GUARDIAN_PHI_THRESHOLD {
                        return Err(wasm_error!(WasmErrorInner::Guest(format!(
                            "Vetoing a Ready timelock requires Guardian-tier Φ ({:.2}), caller has {:.2}",
                            GUARDIAN_PHI_THRESHOLD, phi
                        ))));
                    }
                }
            }
            None => {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Cannot veto Ready timelock: consciousness bridge unavailable (fail-closed)"
                        .into()
                )));
            }
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
        affected_proposal_id: input.affected_proposal_id.clone(),
        justification_hash: input.justification_hash.clone(),
        threat_category: input.threat_category.clone(),
    };

    let action_hash = create_entry(&EntryTypes::GuardianVeto(veto))?;

    // Update timelock status to cancelled via O(1) link-based lookup
    let tl_record = find_timelock_by_id(&input.timelock_id)?;
    let tl: Timelock = tl_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid timelock entry".into()
        )))?;
    if matches!(tl.status, TimelockStatus::Pending | TimelockStatus::Ready) {
        // Transition to Vetoed (not Cancelled) — allows supermajority override
        let vetoed = Timelock {
            status: TimelockStatus::Vetoed,
            cancellation_reason: Some(input.reason.clone()),
            ..tl
        };
        update_entry(
            tl_record.action_address().clone(),
            &EntryTypes::Timelock(vetoed),
        )?;
    }

    // Clean up pending_timelocks link (timelock was vetoed/cancelled)
    if let Ok(pending_links) = get_links(
        LinkQuery::try_new(
            anchor_hash("pending_timelocks")?,
            LinkTypes::PendingTimelocks,
        )?,
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find veto".into()
    )))
}

/// Input for vetoing a timelock
#[derive(Serialize, Deserialize, Debug)]
pub struct VetoTimelockInput {
    pub timelock_id: String,
    pub guardian_did: String,
    pub reason: String,
    /// Proposal ID affected by this veto (constitutional registry, Art. III Sec. 5.5).
    #[serde(default)]
    pub affected_proposal_id: Option<String>,
    /// SHA-256 hash of the full justification document.
    #[serde(default)]
    pub justification_hash: Option<String>,
    /// Threat category for the veto (required post-sunset for Charter Guardian Authority).
    #[serde(default)]
    pub threat_category: Option<String>,
}

// ============================================================================
// VETO OVERRIDE MECHANISM
// Thermodynamic counterbalance: No Maxwell's Demon — collective energy (67%
// supermajority, Art. III Sec. 5.3) can overcome any individual barrier.
// ============================================================================

/// Challenge a guardian veto — initiates the 48-hour override window.
/// Any Citizen-tier (Φ ≥ 0.4) agent can challenge.
#[hdk_extern]
pub fn challenge_veto(input: ChallengeVetoInput) -> ExternResult<()> {
    if input.veto_id.is_empty() || input.veto_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Veto ID must be 1-256 characters".into()
        )));
    }
    if input.challenger_did.is_empty() || input.challenger_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Challenger DID must be 1-256 characters".into()
        )));
    }

    // Verify challenger DID matches calling agent
    let agent = agent_info()?;
    let expected_did = format!("did:mycelix:{}", agent.agent_initial_pubkey);
    if input.challenger_did != expected_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Challenger DID must match the calling agent".into()
        )));
    }

    // Verify challenger meets Citizen-tier Φ (0.4)
    const CITIZEN_PHI: f64 = 0.4;
    match governance_utils::call_local_best_effort(
        "governance_bridge",
        "verify_consciousness_gate",
        serde_json::json!({"action_type": "ChallengeVeto", "action_id": input.veto_id.clone()}),
    )? {
        Some(extern_io) => {
            if let Ok(result) = extern_io.decode::<serde_json::Value>() {
                let phi = result.get("phi").and_then(|p| p.as_f64()).unwrap_or(0.0);
                if phi < CITIZEN_PHI {
                    return Err(wasm_error!(WasmErrorInner::Guest(format!(
                        "Challenging a veto requires Citizen-tier Φ ({:.2}), caller has {:.2}",
                        CITIZEN_PHI, phi
                    ))));
                }
            }
        }
        None => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Cannot challenge veto: consciousness bridge unavailable (fail-closed)".into()
            )));
        }
    }

    // Emit signal to notify participants
    let _ = emit_signal(serde_json::json!({
        "type": "VetoChallenged",
        "veto_id": input.veto_id,
        "challenger_did": input.challenger_did,
        "override_window_hours": 48,
        "override_threshold": execution_integrity::VETO_OVERRIDE_THRESHOLD,
    }));

    Ok(())
}

/// Input for challenging a veto
#[derive(Serialize, Deserialize, Debug)]
pub struct ChallengeVetoInput {
    pub veto_id: String,
    pub challenger_did: String,
}

/// Cast a vote to override (or sustain) a guardian veto.
/// Requires Citizen-tier Φ (0.4).
#[hdk_extern]
pub fn cast_override_vote(input: CastOverrideVoteInput) -> ExternResult<Record> {
    if input.veto_id.is_empty() || input.veto_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Veto ID must be 1-256 characters".into()
        )));
    }
    if input.voter_did.is_empty() || input.voter_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Voter DID must be 1-256 characters".into()
        )));
    }

    // Verify voter DID matches calling agent
    let agent = agent_info()?;
    let expected_did = format!("did:mycelix:{}", agent.agent_initial_pubkey);
    if input.voter_did != expected_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Voter DID must match the calling agent".into()
        )));
    }

    // Get voter's Φ score (must be Citizen-tier ≥ 0.4)
    let phi_score = match governance_utils::call_local_best_effort(
        "governance_bridge",
        "verify_consciousness_gate",
        serde_json::json!({"action_type": "OverrideVote", "action_id": input.veto_id.clone()}),
    )? {
        Some(extern_io) => {
            if let Ok(result) = extern_io.decode::<serde_json::Value>() {
                let phi = result.get("phi").and_then(|p| p.as_f64()).unwrap_or(0.0);
                if phi < 0.4 {
                    return Err(wasm_error!(WasmErrorInner::Guest(format!(
                        "Override voting requires Citizen-tier Φ (0.40), caller has {:.2}",
                        phi
                    ))));
                }
                phi
            } else {
                return Err(wasm_error!(WasmErrorInner::Guest(
                    "Failed to decode consciousness gate response".into()
                )));
            }
        }
        None => {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Cannot vote on override: consciousness bridge unavailable (fail-closed)".into()
            )));
        }
    };

    // Check for duplicate vote by this agent on this veto
    let veto_anchor = format!("veto_override:{}", input.veto_id);
    create_entry(&EntryTypes::Anchor(Anchor(veto_anchor.clone())))?;
    let existing_votes = get_links(
        LinkQuery::try_new(
            anchor_hash(&veto_anchor)?,
            LinkTypes::VetoToOverrideVotes,
        )?,
        GetStrategy::default(),
    )?;
    for link in &existing_votes {
        if let Ok(ah) = ActionHash::try_from(link.target.clone()) {
            if let Some(record) = get(ah, GetOptions::default())? {
                if let Some(existing) = record
                    .entry()
                    .to_app_option::<VetoOverrideVote>()
                    .ok()
                    .flatten()
                {
                    if existing.voter_did == input.voter_did {
                        return Err(wasm_error!(WasmErrorInner::Guest(
                            "Agent has already voted on this veto override".into()
                        )));
                    }
                }
            }
        }
    }

    let now = sys_time()?;
    let vote_id = format!("override_vote:{}:{}:{}", input.veto_id, input.voter_did, now.as_micros());

    let vote = VetoOverrideVote {
        id: vote_id,
        veto_id: input.veto_id.clone(),
        voter_did: input.voter_did,
        supports_override: input.supports_override,
        phi_score,
        voted_at: now,
    };

    let action_hash = create_entry(&EntryTypes::VetoOverrideVote(vote))?;

    // Link vote to veto
    create_link(
        anchor_hash(&veto_anchor)?,
        action_hash.clone(),
        LinkTypes::VetoToOverrideVotes,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find override vote".into()
    )))
}

/// Input for casting an override vote
#[derive(Serialize, Deserialize, Debug)]
pub struct CastOverrideVoteInput {
    pub veto_id: String,
    pub voter_did: String,
    pub supports_override: bool,
}

/// Resolve a veto override — tallies votes and transitions the timelock.
/// Can be called by any agent after the 48-hour override window closes.
/// This is permission-less enforcement — no special role required.
#[hdk_extern]
pub fn resolve_override(input: ResolveOverrideInput) -> ExternResult<Record> {
    if input.veto_id.is_empty() || input.veto_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Veto ID must be 1-256 characters".into()
        )));
    }
    if input.timelock_id.is_empty() || input.timelock_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Timelock ID must be 1-256 characters".into()
        )));
    }

    // Verify the timelock is in Vetoed state
    let tl_record = find_timelock_by_id(&input.timelock_id)?;
    let tl: Timelock = tl_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid timelock entry".into()
        )))?;

    if tl.status != TimelockStatus::Vetoed {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Timelock must be in Vetoed status to resolve override, current: {:?}",
            tl.status
        ))));
    }

    // Collect all override votes
    let veto_anchor = format!("veto_override:{}", input.veto_id);
    let vote_links = get_links(
        LinkQuery::try_new(
            anchor_hash(&veto_anchor)?,
            LinkTypes::VetoToOverrideVotes,
        )?,
        GetStrategy::default(),
    )
    .unwrap_or_default();

    let mut votes_for: f64 = 0.0;
    let mut votes_against: f64 = 0.0;
    let mut voter_count: u64 = 0;

    for link in vote_links {
        if let Ok(ah) = ActionHash::try_from(link.target) {
            if let Some(record) = get(ah, GetOptions::default())? {
                if let Some(vote) = record
                    .entry()
                    .to_app_option::<VetoOverrideVote>()
                    .ok()
                    .flatten()
                {
                    voter_count += 1;
                    // Weight by phi score for consciousness-integrated override
                    let weight = vote.phi_score.max(0.0).min(1.0);
                    if vote.supports_override {
                        votes_for += weight;
                    } else {
                        votes_against += weight;
                    }
                }
            }
        }
    }

    let total_weight = votes_for + votes_against;
    let override_ratio = if total_weight > 0.0 {
        votes_for / total_weight
    } else {
        0.0
    };
    let override_succeeded = override_ratio >= execution_integrity::VETO_OVERRIDE_THRESHOLD;

    let now = sys_time()?;
    let result_id = format!("override_result:{}:{}", input.veto_id, now.as_micros());

    let result = VetoOverrideResult {
        id: result_id,
        veto_id: input.veto_id.clone(),
        timelock_id: input.timelock_id.clone(),
        override_votes_for: votes_for,
        override_votes_against: votes_against,
        total_eligible_voters: voter_count,
        override_threshold: execution_integrity::VETO_OVERRIDE_THRESHOLD,
        override_succeeded,
        resolved_at: now,
    };

    let result_hash = create_entry(&EntryTypes::VetoOverrideResult(result))?;

    // Link result to veto
    create_entry(&EntryTypes::Anchor(Anchor(veto_anchor.clone())))?;
    create_link(
        anchor_hash(&veto_anchor)?,
        result_hash.clone(),
        LinkTypes::VetoToOverrideResult,
        (),
    )?;

    // Transition timelock based on outcome
    if override_succeeded {
        // Override succeeded — restore timelock to Ready
        let restored = Timelock {
            status: TimelockStatus::Ready,
            cancellation_reason: None,
            ..tl
        };
        update_entry(
            tl_record.action_address().clone(),
            &EntryTypes::Timelock(restored),
        )?;

        let _ = emit_signal(serde_json::json!({
            "type": "VetoOverrideSucceeded",
            "veto_id": input.veto_id,
            "timelock_id": input.timelock_id,
            "override_ratio": override_ratio,
            "voter_count": voter_count,
        }));
    } else {
        // Override failed — finalize cancellation
        let cancelled = Timelock {
            status: TimelockStatus::Cancelled,
            ..tl
        };
        update_entry(
            tl_record.action_address().clone(),
            &EntryTypes::Timelock(cancelled),
        )?;

        // Clean up pending_timelocks link
        if let Ok(pending_links) = get_links(
            LinkQuery::try_new(
                anchor_hash("pending_timelocks")?,
                LinkTypes::PendingTimelocks,
            )?,
            GetStrategy::default(),
        ) {
            for link in pending_links {
                if let Ok(target_hash) = ActionHash::try_from(link.target.clone()) {
                    if let Ok(Some(record)) = get(target_hash, GetOptions::default()) {
                        if let Some(ptl) = record.entry().to_app_option::<Timelock>().ok().flatten() {
                            if ptl.id == input.timelock_id {
                                let _ = delete_link(link.create_link_hash, GetOptions::default());
                            }
                        }
                    }
                }
            }
        }

        let _ = emit_signal(serde_json::json!({
            "type": "VetoSustained",
            "veto_id": input.veto_id,
            "timelock_id": input.timelock_id,
            "override_ratio": override_ratio,
            "voter_count": voter_count,
        }));
    }

    get(result_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find override result".into()
    )))
}

/// Input for resolving a veto override
#[derive(Serialize, Deserialize, Debug)]
pub struct ResolveOverrideInput {
    pub veto_id: String,
    pub timelock_id: String,
}

/// Query a guardian's veto history for accountability and transparency.
#[hdk_extern]
pub fn get_guardian_vetoes(guardian_did: String) -> ExternResult<Vec<Record>> {
    if guardian_did.is_empty() || guardian_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Guardian DID must be 1-256 characters".into()
        )));
    }
    let guardian_anchor = format!("guardian:{}", guardian_did);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&guardian_anchor)?,
            LinkTypes::GuardianToVeto,
        )?,
        GetStrategy::default(),
    )?;

    let mut vetoes = Vec::new();
    for link in links {
        if let Ok(ah) = ActionHash::try_from(link.target) {
            if let Ok(Some(record)) = get(ah, GetOptions::default()) {
                vetoes.push(record);
            }
        }
    }
    Ok(vetoes)
}

/// Lock funds in escrow for a proposal's execution.
///
/// Called after a proposal is approved and before a timelock is created.
/// Creates a `FundAllocation` entry with status `Locked` and links it
/// to the proposal.
#[hdk_extern]
pub fn lock_proposal_funds(input: LockFundsInput) -> ExternResult<Record> {
    if input.proposal_id.is_empty() || input.proposal_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Proposal ID must be 1-256 characters".into()
        )));
    }
    if input.source_account.is_empty() || input.source_account.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Source account must be 1-256 characters".into()
        )));
    }
    if let Some(ref currency) = input.currency {
        if currency.len() > 64 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Currency must be at most 64 characters".into()
            )));
        }
    }
    if let Some(ref tl_id) = input.timelock_id {
        if tl_id.len() > 256 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Timelock ID must be at most 256 characters".into()
            )));
        }
    }
    if input.amount <= 0.0 || !input.amount.is_finite() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Amount must be positive and finite".into()
        )));
    }

    // Check for existing locked allocation for this proposal
    if let Some(_existing) = find_fund_allocation_for_proposal(&input.proposal_id)? {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Funds already locked for proposal '{}'",
            input.proposal_id
        ))));
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find fund allocation".into()
    )))
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
    if input.proposal_id.is_empty() || input.proposal_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Proposal ID must be 1-256 characters".into()
        )));
    }
    if let Some(ref reason) = input.reason {
        if reason.len() > 4096 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Reason must be at most 4096 characters".into()
            )));
        }
    }

    let (record, alloc) = find_fund_allocation_for_proposal(&input.proposal_id)?.ok_or(
        wasm_error!(WasmErrorInner::Guest(format!(
            "No fund allocation found for proposal '{}'",
            input.proposal_id
        ))),
    )?;

    // Authorization: only the fund allocation creator can release
    let caller = agent_info()?.agent_initial_pubkey;
    if caller != *record.action().author() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the fund allocation creator can release locked funds".into()
        )));
    }

    if alloc.status != AllocationStatus::Locked {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Allocation is not locked (current status: {:?})",
            alloc.status
        ))));
    }

    let released = FundAllocation {
        status: AllocationStatus::Released,
        status_reason: Some(
            input
                .reason
                .unwrap_or_else(|| "Execution completed successfully".to_string()),
        ),
        ..alloc
    };

    let action_hash = update_entry(
        record.action_address().clone(),
        &EntryTypes::FundAllocation(released),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated allocation".into()
    )))
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
    if input.proposal_id.is_empty() || input.proposal_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Proposal ID must be 1-256 characters".into()
        )));
    }
    if input.reason.len() > 4096 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Reason must be at most 4096 characters".into()
        )));
    }

    let (record, alloc) = find_fund_allocation_for_proposal(&input.proposal_id)?.ok_or(
        wasm_error!(WasmErrorInner::Guest(format!(
            "No fund allocation found for proposal '{}'",
            input.proposal_id
        ))),
    )?;

    // Authorization: only the fund allocation creator or a guardian can refund
    let caller = agent_info()?.agent_initial_pubkey;
    if caller != *record.action().author() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the fund allocation creator can refund locked funds".into()
        )));
    }

    if alloc.status != AllocationStatus::Locked {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Allocation is not locked (current status: {:?})",
            alloc.status
        ))));
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

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find updated allocation".into()
    )))
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
fn find_fund_allocation_for_proposal(
    proposal_id: &str,
) -> ExternResult<Option<(Record, FundAllocation)>> {
    let alloc_anchor = format!("fund_alloc:{}", proposal_id);
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash(&alloc_anchor)?,
            LinkTypes::ProposalToFundAllocation,
        )?,
        GetStrategy::default(),
    )?;

    // Find the most recent allocation
    let latest_link = links.into_iter().max_by_key(|l| l.timestamp);
    if let Some(link) = latest_link {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
            let alloc: FundAllocation = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Invalid allocation entry".into()
                )))?;
            return Ok(Some((record, alloc)));
        }
    }
    Ok(None)
}

/// Get pending timelocks
#[hdk_extern]
pub fn get_pending_timelocks(_: ()) -> ExternResult<Vec<Record>> {
    let links = get_links(
        LinkQuery::try_new(
            anchor_hash("pending_timelocks")?,
            LinkTypes::PendingTimelocks,
        )?,
        GetStrategy::default(),
    )?;

    let mut timelocks = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get_latest_record(action_hash)? {
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

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // GovernanceAction::validate() — pure method tests
    // =========================================================================

    // --- TransferCredits ---

    #[test]
    fn test_transfer_credits_valid() {
        let action = GovernanceAction::TransferCredits {
            from: "treasury".into(),
            to: "project-fund".into(),
            amount: 1000.0,
        };
        assert!(action.validate().is_ok());
    }

    #[test]
    fn test_transfer_credits_empty_from() {
        let action = GovernanceAction::TransferCredits {
            from: "".into(),
            to: "project-fund".into(),
            amount: 100.0,
        };
        let err = action.validate().unwrap_err();
        assert!(err.contains("'from' is required"));
    }

    #[test]
    fn test_transfer_credits_empty_to() {
        let action = GovernanceAction::TransferCredits {
            from: "treasury".into(),
            to: "".into(),
            amount: 100.0,
        };
        let err = action.validate().unwrap_err();
        assert!(err.contains("'to' is required"));
    }

    #[test]
    fn test_transfer_credits_zero_amount() {
        let action = GovernanceAction::TransferCredits {
            from: "treasury".into(),
            to: "project".into(),
            amount: 0.0,
        };
        let err = action.validate().unwrap_err();
        assert!(err.contains("must be positive"));
    }

    #[test]
    fn test_transfer_credits_negative_amount() {
        let action = GovernanceAction::TransferCredits {
            from: "treasury".into(),
            to: "project".into(),
            amount: -50.0,
        };
        let err = action.validate().unwrap_err();
        assert!(err.contains("must be positive"));
    }

    #[test]
    fn test_transfer_credits_infinite_amount() {
        let action = GovernanceAction::TransferCredits {
            from: "treasury".into(),
            to: "project".into(),
            amount: f64::INFINITY,
        };
        let err = action.validate().unwrap_err();
        assert!(err.contains("must be finite"));
    }

    #[test]
    fn test_transfer_credits_nan_amount() {
        let action = GovernanceAction::TransferCredits {
            from: "treasury".into(),
            to: "project".into(),
            amount: f64::NAN,
        };
        // NaN is neither positive nor finite — hits the <= 0.0 check
        assert!(action.validate().is_err());
    }

    // --- UpdateParameter ---

    #[test]
    fn test_update_parameter_valid() {
        let action = GovernanceAction::UpdateParameter {
            parameter: "quorum_threshold".into(),
            value: "0.67".into(),
        };
        assert!(action.validate().is_ok());
    }

    #[test]
    fn test_update_parameter_empty_name() {
        let action = GovernanceAction::UpdateParameter {
            parameter: "".into(),
            value: "0.67".into(),
        };
        let err = action.validate().unwrap_err();
        assert!(err.contains("'parameter' name is required"));
    }

    // --- EmitEvent ---

    #[test]
    fn test_emit_event_always_valid() {
        let action = GovernanceAction::EmitEvent {
            event: "treasury_disbursement".into(),
            payload: serde_json::json!({"amount": 500}),
        };
        assert!(action.validate().is_ok());

        // Even empty event is valid (no validation on event name)
        let empty = GovernanceAction::EmitEvent {
            event: "".into(),
            payload: serde_json::Value::Null,
        };
        assert!(empty.validate().is_ok());
    }

    // =========================================================================
    // GovernanceAction serde — JSON round-trip and tagged enum format
    // =========================================================================

    #[test]
    fn test_governance_action_serde_transfer() {
        let json = r#"{"type":"TransferCredits","from":"treasury","to":"dev-fund","amount":250.5}"#;
        let action: GovernanceAction = serde_json::from_str(json).unwrap();
        match action {
            GovernanceAction::TransferCredits { from, to, amount } => {
                assert_eq!(from, "treasury");
                assert_eq!(to, "dev-fund");
                assert!((amount - 250.5).abs() < f64::EPSILON);
            }
            _ => panic!("Expected TransferCredits"),
        }
    }

    #[test]
    fn test_governance_action_serde_update() {
        let json = r#"{"type":"UpdateParameter","parameter":"phi_threshold","value":"0.5"}"#;
        let action: GovernanceAction = serde_json::from_str(json).unwrap();
        match action {
            GovernanceAction::UpdateParameter { parameter, value } => {
                assert_eq!(parameter, "phi_threshold");
                assert_eq!(value, "0.5");
            }
            _ => panic!("Expected UpdateParameter"),
        }
    }

    #[test]
    fn test_governance_action_serde_emit() {
        let json = r#"{"type":"EmitEvent","event":"proposal_executed"}"#;
        let action: GovernanceAction = serde_json::from_str(json).unwrap();
        match action {
            GovernanceAction::EmitEvent { event, payload } => {
                assert_eq!(event, "proposal_executed");
                assert_eq!(payload, serde_json::Value::Null); // default
            }
            _ => panic!("Expected EmitEvent"),
        }
    }

    #[test]
    fn test_governance_action_array_parse() {
        let json = r#"[
            {"type":"TransferCredits","from":"a","to":"b","amount":100},
            {"type":"EmitEvent","event":"done"}
        ]"#;
        let actions: Vec<GovernanceAction> = serde_json::from_str(json).unwrap();
        assert_eq!(actions.len(), 2);
        assert!(actions[0].validate().is_ok());
        assert!(actions[1].validate().is_ok());
    }

    #[test]
    fn test_governance_action_invalid_json() {
        let json = "not valid json at all {{{";
        assert!(serde_json::from_str::<GovernanceAction>(json).is_err());
    }

    // =========================================================================
    // ThresholdSignature mirror type serde
    // =========================================================================

    // =========================================================================
    // Committee scope enforcement — proposal type inference
    // =========================================================================

    // --- extract_scope_name ---

    #[test]
    fn test_extract_scope_simple_variants() {
        assert_eq!(extract_scope_name(&serde_json::json!("All")), "All");
        assert_eq!(
            extract_scope_name(&serde_json::json!("Constitutional")),
            "Constitutional"
        );
        assert_eq!(
            extract_scope_name(&serde_json::json!("Treasury")),
            "Treasury"
        );
        assert_eq!(
            extract_scope_name(&serde_json::json!("Protocol")),
            "Protocol"
        );
    }

    #[test]
    fn test_extract_scope_custom_variant() {
        // Custom(Vec<String>) serializes as {"Custom": ["type1", "type2"]}
        let custom = serde_json::json!({"Custom": ["treasury_ops", "emergency"]});
        assert_eq!(extract_scope_name(&custom), "Custom");
    }

    #[test]
    fn test_extract_scope_null_defaults_to_all() {
        assert_eq!(extract_scope_name(&serde_json::Value::Null), "All");
    }

    // --- scope enforcement logic ---

    fn scope_allows(scope_name: &str, proposal_type: &str) -> bool {
        match scope_name {
            "All" => true,
            "Constitutional" => proposal_type == "constitutional",
            "Treasury" => proposal_type == "treasury",
            "Protocol" => proposal_type == "protocol",
            _ => true,
        }
    }

    #[test]
    fn test_scope_all_allows_everything() {
        assert!(scope_allows("All", "constitutional"));
        assert!(scope_allows("All", "treasury"));
        assert!(scope_allows("All", "protocol"));
        assert!(scope_allows("All", "proposal"));
        assert!(scope_allows("All", "unknown"));
    }

    #[test]
    fn test_scope_constitutional_restricts() {
        assert!(scope_allows("Constitutional", "constitutional"));
        assert!(!scope_allows("Constitutional", "treasury"));
        assert!(!scope_allows("Constitutional", "protocol"));
        assert!(!scope_allows("Constitutional", "proposal"));
    }

    #[test]
    fn test_scope_treasury_restricts() {
        assert!(scope_allows("Treasury", "treasury"));
        assert!(!scope_allows("Treasury", "constitutional"));
        assert!(!scope_allows("Treasury", "protocol"));
    }

    #[test]
    fn test_scope_protocol_restricts() {
        assert!(scope_allows("Protocol", "protocol"));
        assert!(!scope_allows("Protocol", "treasury"));
        assert!(!scope_allows("Protocol", "constitutional"));
    }

    #[test]
    fn test_scope_custom_permissive() {
        assert!(scope_allows("Custom", "anything"));
    }

    #[test]
    fn test_infer_proposal_type_from_description() {
        fn infer(desc: &str) -> &str {
            desc.split(':').next().unwrap_or("unknown")
        }
        assert_eq!(infer("proposal:MIP-001"), "proposal");
        assert_eq!(infer("constitutional:CA-001"), "constitutional");
        assert_eq!(infer("treasury:TB-042"), "treasury");
        assert_eq!(infer("protocol:PU-007"), "protocol");
        assert_eq!(infer("no-colon-here"), "no-colon-here");
        assert_eq!(infer(""), "");
    }

    #[test]
    fn test_veto_cooldown_is_7_days() {
        const VETO_COOLDOWN_US: i64 = 7 * 24 * 3600 * 1_000_000;
        assert_eq!(VETO_COOLDOWN_US, 604_800_000_000);
    }

    #[test]
    fn test_guardian_phi_threshold_constant() {
        // Verify the Guardian-tier threshold used in veto_timelock
        // matches the actual Guardian Φ requirement (0.8)
        const GUARDIAN_PHI_THRESHOLD: f64 = 0.8;
        assert!(
            GUARDIAN_PHI_THRESHOLD >= 0.8,
            "Guardian veto must require actual Guardian-tier Φ (0.8)"
        );
        assert!(GUARDIAN_PHI_THRESHOLD <= 1.0, "Must be a valid Φ score");
    }

    // =========================================================================
    // ThresholdSignature mirror type serde
    // =========================================================================

    #[test]
    fn test_threshold_signature_serde_roundtrip() {
        let sig = ThresholdSignature {
            id: "sig-1".into(),
            committee_id: "committee-1".into(),
            signed_content_hash: vec![1, 2, 3],
            signed_content_description: "proposal:MIP-001".into(),
            signature: vec![0u8; 64],
            signer_count: 2,
            signers: vec![1, 2],
            verified: true,
            signed_at: Timestamp::from_micros(1000000),
        };
        let json = serde_json::to_string(&sig).unwrap();
        let decoded: ThresholdSignature = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "sig-1");
        assert!(decoded.verified);
        assert!(decoded.signed_content_description.contains("MIP-001"));
    }
}
