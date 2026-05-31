// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Social Recovery Coordinator Zome
//! Business logic for DID social recovery
//!
//! Updated to use HDK 0.6 patterns

use hdk::prelude::*;
use recovery_integrity::*;

/// Create a deterministic entry hash from a string identifier
/// This is used for link bases when we need to link from string IDs
fn string_to_entry_hash(s: &str) -> EntryHash {
    let bytes: Vec<u8> = holo_hash::blake2b_256(s.as_bytes())
        .into_iter()
        .chain([0u8; 4])
        .collect();
    // Safety: blake2b_256 returns 32 bytes + 4 padding = 36, matching from_raw_36
    EntryHash::from_raw_36(bytes)
}

/// Set up recovery for a DID
#[hdk_extern]
pub fn setup_recovery(input: SetupRecoveryInput) -> ExternResult<Record> {
    if input.did.is_empty() || input.did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DID must be 1-256 characters".into()
        )));
    }
    if input.trustees.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "At least one trustee is required".into()
        )));
    }
    if input.trustees.len() > 50 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Maximum 50 trustees allowed".into()
        )));
    }
    for trustee in &input.trustees {
        if trustee.is_empty() || trustee.len() > 256 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Each trustee DID must be 1-256 characters".into()
            )));
        }
    }
    if input.threshold == 0 || input.threshold > input.trustees.len() as u32 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Threshold must be between 1 and the number of trustees".into()
        )));
    }
    let agent_info = agent_info()?;
    let now = sys_time()?;

    let config = RecoveryConfig {
        did: input.did.clone(),
        owner: agent_info.agent_initial_pubkey,
        trustees: input.trustees,
        threshold: input.threshold,
        time_lock: input.time_lock.unwrap_or(86400 * 7), // Default 7 days
        active: true,
        created: now,
        updated: now,
    };

    let action_hash = create_entry(&EntryTypes::RecoveryConfig(config.clone()))?;

    // Link DID to config
    create_link(
        string_to_entry_hash(&input.did),
        action_hash.clone(),
        LinkTypes::DidToRecoveryConfig,
        (),
    )?;

    // Link each trustee to config
    for trustee in &config.trustees {
        create_link(
            string_to_entry_hash(trustee),
            action_hash.clone(),
            LinkTypes::TrusteeToConfig,
            (),
        )?;
    }

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find recovery config".into()
    )))
}

/// Input for setting up recovery
#[derive(Serialize, Deserialize, Debug)]
pub struct SetupRecoveryInput {
    pub did: String,
    pub trustees: Vec<String>,
    pub threshold: u32,
    pub time_lock: Option<u64>,
}

/// Get recovery config for a DID
#[hdk_extern]
pub fn get_recovery_config(did: String) -> ExternResult<Option<Record>> {
    if did.is_empty() || did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DID must be 1-256 characters".into()
        )));
    }
    let did_hash = string_to_entry_hash(&did);
    let links = get_links(
        LinkQuery::try_new(did_hash, LinkTypes::DidToRecoveryConfig)?,
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

/// Initiate a recovery request (trustee only)
#[hdk_extern]
pub fn initiate_recovery(input: InitiateRecoveryInput) -> ExternResult<Record> {
    if input.did.is_empty() || input.did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "DID must be 1-256 characters".into()
        )));
    }
    if input.initiator_did.is_empty() || input.initiator_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Initiator DID must be 1-256 characters".into()
        )));
    }
    if input.reason.is_empty() || input.reason.len() > 2048 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Reason must be 1-2048 characters".into()
        )));
    }
    // Verify recovery config exists
    let config_record = get_recovery_config(input.did.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("No recovery config found for this DID".into())
    ))?;

    let config: RecoveryConfig = config_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid recovery config".into()
        )))?;

    // Verify initiator is a trustee
    if !config.trustees.contains(&input.initiator_did) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only trustees can initiate recovery".into()
        )));
    }

    let now = sys_time()?;
    let request_id = format!("recovery:{}:{}", input.did, now.as_micros());

    let request = RecoveryRequest {
        id: request_id.clone(),
        did: input.did.clone(),
        new_agent: input.new_agent,
        initiated_by: input.initiator_did.clone(),
        reason: input.reason,
        status: RecoveryStatus::Pending,
        created: now,
        time_lock_expires: None,
    };

    let action_hash = create_entry(&EntryTypes::RecoveryRequest(request))?;

    // Link DID to request
    create_link(
        string_to_entry_hash(&input.did),
        action_hash.clone(),
        LinkTypes::DidToRecoveryRequest,
        (),
    )?;

    // Create initial approval vote from initiator
    let vote = RecoveryVote {
        request_id,
        trustee: input.initiator_did,
        vote: VoteDecision::Approve,
        comment: Some("Initiated recovery".to_string()),
        voted_at: now,
    };

    let vote_hash = create_entry(&EntryTypes::RecoveryVote(vote))?;

    create_link(
        action_hash.clone(),
        vote_hash,
        LinkTypes::RequestToVotes,
        (),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find recovery request".into()
    )))
}

/// Input for initiating recovery
#[derive(Serialize, Deserialize, Debug)]
pub struct InitiateRecoveryInput {
    pub did: String,
    pub initiator_did: String,
    pub new_agent: AgentPubKey,
    pub reason: String,
}

/// Vote on a recovery request
#[hdk_extern]
pub fn vote_on_recovery(input: VoteOnRecoveryInput) -> ExternResult<Record> {
    if input.request_id.is_empty() || input.request_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Request ID must be 1-256 characters".into()
        )));
    }
    if input.trustee_did.is_empty() || input.trustee_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Trustee DID must be 1-256 characters".into()
        )));
    }
    if let Some(ref comment) = input.comment {
        if comment.len() > 2048 {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Comment must be under 2048 characters".into()
            )));
        }
    }
    let now = sys_time()?;

    let vote = RecoveryVote {
        request_id: input.request_id.clone(),
        trustee: input.trustee_did,
        vote: input.vote,
        comment: input.comment,
        voted_at: now,
    };

    let action_hash = create_entry(&EntryTypes::RecoveryVote(vote))?;

    // Link to request
    let request_hash = string_to_entry_hash(&input.request_id);
    create_link(
        request_hash,
        action_hash.clone(),
        LinkTypes::RequestToVotes,
        (),
    )?;

    // Check if threshold is reached
    check_and_update_request_status(input.request_id)?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find vote".into()
    )))
}

/// Input for voting on recovery
#[derive(Serialize, Deserialize, Debug)]
pub struct VoteOnRecoveryInput {
    pub request_id: String,
    pub trustee_did: String,
    pub vote: VoteDecision,
    pub comment: Option<String>,
}

/// Get votes for a recovery request
#[hdk_extern]
pub fn get_recovery_votes(request_id: String) -> ExternResult<Vec<Record>> {
    if request_id.is_empty() || request_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Request ID must be 1-256 characters".into()
        )));
    }
    let request_hash = string_to_entry_hash(&request_id);
    let links = get_links(
        LinkQuery::try_new(request_hash, LinkTypes::RequestToVotes)?,
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

/// Check threshold and update request status
fn check_and_update_request_status(request_id: String) -> ExternResult<()> {
    // Get all votes for this request
    let vote_records = get_recovery_votes(request_id.clone())?;

    let mut approve_count = 0u32;
    let mut reject_count = 0u32;

    for record in vote_records {
        if let Some(vote) = record
            .entry()
            .to_app_option::<RecoveryVote>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            match vote.vote {
                VoteDecision::Approve => approve_count += 1,
                VoteDecision::Reject => reject_count += 1,
                VoteDecision::Abstain => {}
            }
        }
    }

    // Find the recovery request to get the DID
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::RecoveryRequest,
        )?))
        .include_entries(true);

    let records = query(filter)?;

    let mut request_record: Option<Record> = None;
    let mut request_data: Option<RecoveryRequest> = None;
    for record in records {
        if let Some(req) = record
            .entry()
            .to_app_option::<RecoveryRequest>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if req.id == request_id {
                request_data = Some(req);
                request_record = Some(record);
                break;
            }
        }
    }

    let (current_record, current_request) = match (request_record, request_data) {
        (Some(r), Some(d)) => (r, d),
        _ => return Ok(()), // Request not found on this agent's chain
    };

    // Only process pending requests
    if current_request.status != RecoveryStatus::Pending {
        return Ok(());
    }

    // Get the recovery config to check threshold
    let config_record = get_recovery_config(current_request.did.clone())?;
    let config: RecoveryConfig = match config_record {
        Some(rec) => rec
            .entry()
            .to_app_option()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
            .ok_or(wasm_error!(WasmErrorInner::Guest(
                "Invalid recovery config".into()
            )))?,
        None => return Ok(()), // No config found
    };

    let total_trustees = config.trustees.len() as u32;

    if approve_count >= config.threshold {
        // Threshold reached — approve and set time lock
        let now = sys_time()?;
        let time_lock_expires =
            Timestamp::from_micros(now.as_micros() as i64 + (config.time_lock as i64 * 1_000_000));

        let approved_request = RecoveryRequest {
            id: current_request.id,
            did: current_request.did,
            new_agent: current_request.new_agent,
            initiated_by: current_request.initiated_by,
            reason: current_request.reason,
            status: RecoveryStatus::Approved,
            created: current_request.created,
            time_lock_expires: Some(time_lock_expires),
        };

        update_entry(
            current_record.action_address().clone(),
            &EntryTypes::RecoveryRequest(approved_request),
        )?;
    } else if reject_count > total_trustees - config.threshold {
        // Impossible to reach threshold — reject
        let rejected_request = RecoveryRequest {
            id: current_request.id,
            did: current_request.did,
            new_agent: current_request.new_agent,
            initiated_by: current_request.initiated_by,
            reason: current_request.reason,
            status: RecoveryStatus::Rejected,
            created: current_request.created,
            time_lock_expires: None,
        };

        update_entry(
            current_record.action_address().clone(),
            &EntryTypes::RecoveryRequest(rejected_request),
        )?;
    }

    Ok(())
}

/// Pure threshold decision logic — no HDK calls, fully testable.
/// Returns Some(true) for approved, Some(false) for rejected, None for still pending.
pub fn evaluate_threshold(
    approve_count: u32,
    reject_count: u32,
    total_trustees: u32,
    threshold: u32,
) -> Option<bool> {
    if approve_count >= threshold {
        Some(true)
    } else if total_trustees >= threshold && reject_count > total_trustees - threshold {
        Some(false)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approved_exact_threshold() {
        assert_eq!(evaluate_threshold(3, 0, 5, 3), Some(true));
    }

    #[test]
    fn test_approved_above_threshold() {
        assert_eq!(evaluate_threshold(4, 0, 5, 3), Some(true));
    }

    #[test]
    fn test_rejected_impossible_to_reach() {
        // 5 trustees, threshold 3, need 3 approvals
        // 3 rejected → only 2 remain → can't reach 3
        assert_eq!(evaluate_threshold(0, 3, 5, 3), Some(false));
    }

    #[test]
    fn test_still_pending() {
        // 5 trustees, threshold 3, 2 approved 1 rejected → still possible
        assert_eq!(evaluate_threshold(2, 1, 5, 3), None);
    }

    #[test]
    fn test_single_trustee_approved() {
        assert_eq!(evaluate_threshold(1, 0, 1, 1), Some(true));
    }

    #[test]
    fn test_single_trustee_rejected() {
        assert_eq!(evaluate_threshold(0, 1, 1, 1), Some(false));
    }

    #[test]
    fn test_no_votes_yet() {
        assert_eq!(evaluate_threshold(0, 0, 5, 3), None);
    }

    #[test]
    fn test_all_abstain_stays_pending() {
        // Abstains don't count as approve or reject
        assert_eq!(evaluate_threshold(0, 0, 5, 3), None);
    }

    #[test]
    fn test_unanimous_approval() {
        assert_eq!(evaluate_threshold(5, 0, 5, 3), Some(true));
    }

    #[test]
    fn test_boundary_reject_not_yet() {
        // 5 trustees, threshold 3 → need >2 rejects to make impossible
        // 2 rejects → still 3 potential approvals
        assert_eq!(evaluate_threshold(0, 2, 5, 3), None);
    }

    #[test]
    fn test_threshold_equals_total() {
        // All must approve; 1 reject means impossible
        assert_eq!(evaluate_threshold(0, 1, 3, 3), Some(false));
        assert_eq!(evaluate_threshold(3, 0, 3, 3), Some(true));
        assert_eq!(evaluate_threshold(2, 0, 3, 3), None);
    }
}

/// Execute recovery (after time lock)
#[hdk_extern]
pub fn execute_recovery(request_id: String) -> ExternResult<Record> {
    if request_id.is_empty() || request_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Request ID must be 1-256 characters".into()
        )));
    }
    // Find the request
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::RecoveryRequest,
        )?))
        .include_entries(true);

    let records = query(filter)?;

    let mut request_record: Option<Record> = None;
    for record in records {
        if let Some(req) = record
            .entry()
            .to_app_option::<RecoveryRequest>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if req.id == request_id {
                request_record = Some(record);
                break;
            }
        }
    }

    let current_record = request_record.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Recovery request not found".into()
    )))?;

    let current_request: RecoveryRequest = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid recovery request".into()
        )))?;

    // Verify status allows execution
    if current_request.status != RecoveryStatus::Approved
        && current_request.status != RecoveryStatus::ReadyToExecute
    {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Recovery request is not approved".into()
        )));
    }

    // Verify time lock has expired
    let now = sys_time()?;
    if let Some(expires) = current_request.time_lock_expires {
        if now < expires {
            return Err(wasm_error!(WasmErrorInner::Guest(
                "Time lock has not expired".into()
            )));
        }
    } else {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Time lock not set".into()
        )));
    }

    // Update request to completed
    let completed_request = RecoveryRequest {
        id: current_request.id,
        did: current_request.did,
        new_agent: current_request.new_agent,
        initiated_by: current_request.initiated_by,
        reason: current_request.reason,
        status: RecoveryStatus::Completed,
        created: current_request.created,
        time_lock_expires: current_request.time_lock_expires,
    };

    let action_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::RecoveryRequest(completed_request),
    )?;

    // Note: The actual DID update would be handled by the did_registry zome
    // This would require a cross-zome call in a full implementation

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find completed request".into()
    )))
}

/// Cancel a recovery request (owner only, before execution)
#[hdk_extern]
pub fn cancel_recovery(request_id: String) -> ExternResult<Record> {
    if request_id.is_empty() || request_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Request ID must be 1-256 characters".into()
        )));
    }
    let agent_info = agent_info()?;

    // Find the request
    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::RecoveryRequest,
        )?))
        .include_entries(true);

    let records = query(filter)?;

    let mut request_record: Option<Record> = None;
    for record in records {
        if let Some(req) = record
            .entry()
            .to_app_option::<RecoveryRequest>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if req.id == request_id {
                request_record = Some(record);
                break;
            }
        }
    }

    let current_record = request_record.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Recovery request not found".into()
    )))?;

    let current_request: RecoveryRequest = current_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid recovery request".into()
        )))?;

    // Get recovery config to verify owner
    let config_record = get_recovery_config(current_request.did.clone())?.ok_or(wasm_error!(
        WasmErrorInner::Guest("Recovery config not found".into())
    ))?;

    let config: RecoveryConfig = config_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Invalid recovery config".into()
        )))?;

    // Verify caller is owner
    if config.owner != agent_info.agent_initial_pubkey {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only owner can cancel recovery".into()
        )));
    }

    // Verify not already completed
    if current_request.status == RecoveryStatus::Completed {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot cancel completed recovery".into()
        )));
    }

    // Update to cancelled
    let cancelled_request = RecoveryRequest {
        id: current_request.id,
        did: current_request.did,
        new_agent: current_request.new_agent,
        initiated_by: current_request.initiated_by,
        reason: current_request.reason,
        status: RecoveryStatus::Cancelled,
        created: current_request.created,
        time_lock_expires: current_request.time_lock_expires,
    };

    let action_hash = update_entry(
        current_record.action_address().clone(),
        &EntryTypes::RecoveryRequest(cancelled_request),
    )?;

    get(action_hash, GetOptions::default())?.ok_or(wasm_error!(WasmErrorInner::Guest(
        "Could not find cancelled request".into()
    )))
}

/// Get pending recovery requests for a trustee
#[hdk_extern]
pub fn get_trustee_responsibilities(trustee_did: String) -> ExternResult<Vec<Record>> {
    if trustee_did.is_empty() || trustee_did.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Trustee DID must be 1-256 characters".into()
        )));
    }
    let trustee_hash = string_to_entry_hash(&trustee_did);
    let links = get_links(
        LinkQuery::try_new(trustee_hash, LinkTypes::TrusteeToConfig)?,
        GetStrategy::default(),
    )?;

    let mut configs = Vec::new();
    for link in links {
        let action_hash = ActionHash::try_from(link.target)
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
        if let Some(record) = get(action_hash, GetOptions::default())? {
            configs.push(record);
        }
    }

    Ok(configs)
}
