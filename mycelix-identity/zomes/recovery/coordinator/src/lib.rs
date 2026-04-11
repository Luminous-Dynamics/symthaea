// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Social Recovery Coordinator Zome
//! Business logic for DID social recovery
//!
//! Updated to use HDK 0.6 patterns
//!
//! ## MFA Integration
//! - Verifies trustee MFA status before allowing votes
//! - Enrolls SocialRecovery factor when setting up recovery
//! - Updates MFA state when recovery is executed

use hdk::prelude::*;
use recovery_integrity::*;
use subtle::ConstantTimeEq;

// =============================================================================
// MFA CROSS-ZOME INTEGRATION
// =============================================================================

/// MFA assurance level (mirrors mfa_integrity::AssuranceLevel)
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize, PartialOrd)]
pub enum AssuranceLevel {
    Anonymous,
    Basic,
    Verified,
    HighlyAssured,
    ConstitutionallyCritical,
}

/// Check if a DID has sufficient MFA assurance for recovery operations
fn verify_mfa_assurance_for_recovery(did: &str) -> ExternResult<bool> {
    let response = call(
        CallTargetCell::Local,
        ZomeName::new("mfa"),
        FunctionName::new("get_mfa_assurance_score"),
        None,
        did.to_string(),
    )?;

    match response {
        ZomeCallResponse::Ok(result) => {
            let score: f64 = result.decode().map_err(|e| {
                wasm_error!(WasmErrorInner::Guest(format!(
                    "Failed to decode MFA score: {:?}",
                    e
                )))
            })?;
            // Require at least Basic level (0.25) for recovery operations
            Ok(score >= 0.25)
        }
        ZomeCallResponse::Unauthorized(_, _, _, _) => {
            // MFA zome not installed — this is a configuration issue, not a security bypass.
            // Fail safe: deny rather than silently allow.
            warn!(
                "MFA zome unauthorized — denying recovery operation (MFA zome must be installed)"
            );
            Err(wasm_error!(WasmErrorInner::Guest(
                "MFA verification unavailable (zome unauthorized). Recovery denied for safety."
                    .into()
            )))
        }
        ZomeCallResponse::NetworkError(err) => {
            warn!(
                "MFA zome network error: {} — denying recovery operation",
                err
            );
            Err(wasm_error!(WasmErrorInner::Guest(format!(
                "MFA verification failed (network error: {}). Retry later.",
                err
            ))))
        }
        ZomeCallResponse::CountersigningSession(err) => {
            warn!(
                "MFA countersigning error: {} — denying recovery operation",
                err
            );
            Err(wasm_error!(WasmErrorInner::Guest(format!(
                "MFA verification failed (countersigning: {}). Retry later.",
                err
            ))))
        }
        ZomeCallResponse::AuthenticationFailed(_, _) => {
            warn!("MFA authentication failed — denying recovery operation");
            Err(wasm_error!(WasmErrorInner::Guest(
                "MFA verification failed (authentication). Recovery denied for safety.".into()
            )))
        }
    }
}

/// Enroll SocialRecovery factor in MFA state
fn enroll_social_recovery_factor(did: &str, trustees: &[String]) -> ExternResult<()> {
    #[derive(Serialize, Deserialize, Debug)]
    struct EnrollFactorInput {
        did: String,
        factor_type: String, // "SocialRecovery"
        factor_id: String,
        metadata: String,
        reason: String,
    }

    let factor_id = format!(
        "social_recovery:{}",
        &holo_hash::blake2b_256(did.as_bytes())
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>()[..16]
    );

    let metadata = serde_json::json!({
        "trustees": trustees,
        "trustee_count": trustees.len()
    })
    .to_string();

    let input = EnrollFactorInput {
        did: did.to_string(),
        factor_type: "SocialRecovery".to_string(),
        factor_id,
        metadata,
        reason: "Social recovery setup".to_string(),
    };

    let response = call(
        CallTargetCell::Local,
        ZomeName::new("mfa"),
        FunctionName::new("enroll_factor"),
        None,
        input,
    )?;

    match response {
        ZomeCallResponse::Ok(_) => Ok(()),
        ZomeCallResponse::Unauthorized(_, _, _, _) => {
            warn!("MFA zome unauthorized - recovery setup WITHOUT MFA factor enrollment");
            Ok(())
        }
        ZomeCallResponse::NetworkError(err) => {
            warn!(
                "MFA network error during factor enrollment: {} - recovery setup WITHOUT MFA",
                err
            );
            Ok(())
        }
        ZomeCallResponse::CountersigningSession(err) => {
            warn!("MFA countersigning error during factor enrollment: {} - recovery setup WITHOUT MFA", err);
            Ok(())
        }
        ZomeCallResponse::AuthenticationFailed(_, _) => {
            warn!("MFA authentication failed - recovery setup WITHOUT MFA factor enrollment");
            Ok(())
        }
    }
}

/// Notify bridge of successful recovery execution so other hApps are informed
fn notify_bridge_of_recovery(did: &str, new_agent: &AgentPubKey) -> ExternResult<()> {
    #[derive(Serialize, Deserialize, Debug)]
    struct BroadcastEventInput {
        event_type: String,
        subject: String,
        payload: String,
        source_happ: String,
    }

    let payload = serde_json::json!({
        "did": did,
        "new_agent": format!("{}", new_agent),
        "event": "recovery_completed",
    })
    .to_string();

    let input = BroadcastEventInput {
        event_type: "DidRecovered".to_string(),
        subject: did.to_string(),
        payload,
        source_happ: "mycelix-identity".to_string(),
    };

    let response = call(
        CallTargetCell::Local,
        ZomeName::new("identity_bridge"),
        FunctionName::new("broadcast_event"),
        None,
        input,
    )?;

    match response {
        ZomeCallResponse::Ok(_) => Ok(()),
        _ => {
            debug!(
                "Bridge notification failed for recovery of {} - non-critical",
                did
            );
            Ok(())
        }
    }
}

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

    // Enroll SocialRecovery factor in MFA state
    // This allows social recovery to contribute to the identity's assurance level
    if let Err(e) = enroll_social_recovery_factor(&input.did, &config.trustees) {
        debug!("Failed to enroll SocialRecovery factor in MFA: {:?}", e);
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

    // Rate limit: Only 1 active (Pending/Approved/ReadyToExecute) recovery request per DID
    let did_hash = string_to_entry_hash(&input.did);
    let request_links = get_links(
        LinkQuery::try_new(did_hash, LinkTypes::DidToRecoveryRequest)?,
        GetStrategy::default(),
    )?;
    for link in &request_links {
        if let Ok(action_hash) = ActionHash::try_from(link.target.clone()) {
            if let Ok(Some(record)) = get(action_hash, GetOptions::default()) {
                if let Ok(Some(req)) = record.entry().to_app_option::<RecoveryRequest>() {
                    match req.status {
                        RecoveryStatus::Pending
                        | RecoveryStatus::Approved
                        | RecoveryStatus::ReadyToExecute => {
                            return Err(wasm_error!(WasmErrorInner::Guest(
                                "An active recovery request already exists for this DID. Cancel it first.".into()
                            )));
                        }
                        _ => {} // Completed/Rejected/Cancelled — allow new request
                    }
                }
            }
        }
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

    // Verify initiator is a trustee (constant-time to prevent timing side-channel)
    let is_trustee = config.trustees.iter().fold(0u8, |acc, trustee| {
        acc | trustee
            .as_bytes()
            .ct_eq(input.initiator_did.as_bytes())
            .unwrap_u8()
    });
    if is_trustee == 0 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only trustees can initiate recovery".into()
        )));
    }

    // Verify initiator has sufficient MFA assurance
    if !verify_mfa_assurance_for_recovery(&input.initiator_did)? {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Initiator does not meet MFA requirements (minimum Basic level required)".into()
        )));
    }

    let now = sys_time()?;
    let request_id = format!("recovery:{}:{}", input.did, now.as_micros());

    // Save values for bridge event before they are moved
    let did_for_event = input.did.clone();
    let initiator_for_event = input.initiator_did.clone();

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

    // Save request_id for bridge event before it's moved
    let request_id_for_event = request_id.clone();

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

    // Broadcast RecoveryInitiated event to bridge for ecosystem-wide awareness
    {
        #[derive(Serialize, Deserialize, Debug)]
        struct BroadcastEventInput {
            event_type: String,
            subject: String,
            payload: String,
            source_happ: String,
        }

        let payload = serde_json::json!({
            "did": did_for_event,
            "initiated_by": initiator_for_event,
            "request_id": request_id_for_event,
            "event": "recovery_initiated",
        })
        .to_string();

        let event_input = BroadcastEventInput {
            event_type: "RecoveryInitiated".to_string(),
            subject: did_for_event.to_string(),
            payload,
            source_happ: "mycelix-identity".to_string(),
        };

        match call(
            CallTargetCell::Local,
            ZomeName::new("identity_bridge"),
            FunctionName::new("broadcast_event"),
            None,
            event_input,
        ) {
            Ok(ZomeCallResponse::Ok(_)) => {}
            _ => {
                debug!("Bridge notification failed for RecoveryInitiated event - non-critical");
            }
        }
    }

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

    // Verify caller is the claimed trustee
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);
    if input.trustee_did != caller_did {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Caller must be the claimed trustee".into()
        )));
    }

    // Verify voter has sufficient MFA assurance
    if !verify_mfa_assurance_for_recovery(&input.trustee_did)? {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Voter does not meet MFA requirements (minimum Basic level required)".into()
        )));
    }

    // Prevent duplicate votes: check if this trustee already voted on this request
    let existing_votes = get_recovery_votes(input.request_id.clone())?;
    for record in &existing_votes {
        if let Some(existing_vote) = record
            .entry()
            .to_app_option::<RecoveryVote>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if existing_vote.trustee == input.trustee_did {
                return Err(wasm_error!(WasmErrorInner::Guest(format!(
                    "Trustee {} has already voted on this recovery request",
                    input.trustee_did
                ))));
            }
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
                // Keep iterating — update_entry appends newer versions later in the chain
                request_data = Some(req);
                request_record = Some(record);
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
                // Keep iterating — update_entry appends newer versions later in the chain
                request_record = Some(record);
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

    // Verify caller is the designated new agent or the recovery initiator
    let caller = agent_info()?.agent_initial_pubkey;
    let caller_did = format!("did:mycelix:{}", caller);
    if caller != current_request.new_agent && caller_did != current_request.initiated_by {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the designated new agent or recovery initiator can execute recovery".into()
        )));
    }

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

    // Save values for MFA notification before moving
    let did_for_mfa = current_request.did.clone();
    let new_agent_for_mfa = current_request.new_agent.clone();

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

    // Broadcast recovery event to bridge so other hApps are informed
    if let Err(e) = notify_bridge_of_recovery(&did_for_mfa, &new_agent_for_mfa) {
        debug!("Failed to notify bridge of recovery execution: {:?}", e);
    }

    // DID transfer completes when the new agent calls did_registry::claim_recovered_did().
    // This two-step pattern is required by Holochain's agent-centric architecture:
    // only the new agent can create entries on their own source chain.

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
                // Keep iterating — update_entry appends newer versions later in the chain
                request_record = Some(record);
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

/// Get a specific recovery request by ID
#[hdk_extern]
pub fn get_recovery_request(request_id: String) -> ExternResult<Option<Record>> {
    if request_id.is_empty() || request_id.len() > 256 {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Request ID must be 1-256 characters".into(),
        )));
    }

    let filter = ChainQueryFilter::new()
        .entry_type(EntryType::App(AppEntryDef::try_from(
            UnitEntryTypes::RecoveryRequest,
        )?))
        .include_entries(true);

    let mut found: Option<Record> = None;
    for record in query(filter)? {
        if let Some(req) = record
            .entry()
            .to_app_option::<RecoveryRequest>()
            .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        {
            if req.id == request_id {
                // Keep iterating — update_entry appends newer versions later in the chain
                found = Some(record);
            }
        }
    }

    Ok(found)
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

// =============================================================================
// PROGRESSIVE RECOVERY — Self-Recovery Functions
// =============================================================================

/// Input for creating self-recovery config.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CreateSelfRecoveryInput {
    pub did: String,
}

/// Auto-create a self-recovery config for a new DID.
///
/// Called by `create_did()` — gives every user a recovery fallback from Day 0.
/// Starts with zero anchors (user enrolls them progressively) and a conservative
/// 7-day time lock.
#[hdk_extern]
pub fn create_self_recovery(input: CreateSelfRecoveryInput) -> ExternResult<Record> {
    if !input.did.starts_with("did:mycelix:") {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Invalid DID format".into()
        )));
    }

    let agent = agent_info()?.agent_initial_pubkey;
    let now = sys_time()?;

    let config = SelfRecoveryConfig {
        did: input.did.clone(),
        owner: agent,
        anchors: vec![],
        anchor_threshold: 1, // Will be meaningful once anchors are enrolled
        time_lock: SELF_RECOVERY_DEFAULT_TIME_LOCK, // 7 days
        active: true,
        superseded_by_social: false,
        created: now,
        updated: now,
    };

    let action_hash = create_entry(&EntryTypes::SelfRecoveryConfig(config.clone()))?;
    let did_hash = string_to_entry_hash(&input.did);
    create_link(
        did_hash,
        action_hash.clone(),
        LinkTypes::DidToSelfRecoveryConfig,
        LinkTag::new("self_recovery"),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Failed to get created self-recovery config".into()
        )))
}

/// Input for adding a verification anchor.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AddVerificationAnchorInput {
    pub did: String,
    pub anchor: VerificationAnchor,
}

/// Add a verification anchor to self-recovery config.
///
/// Called when a user verifies a phone number, email, passkey, or device.
/// Updates the self-recovery time lock based on anchor count:
/// - 1 anchor: 14 days
/// - 2 anchors: 7 days
/// - 3+ anchors: 72 hours (minimum)
#[hdk_extern]
pub fn add_verification_anchor(input: AddVerificationAnchorInput) -> ExternResult<Record> {
    let did_hash = string_to_entry_hash(&input.did);
    let links = get_links(
        LinkQuery::try_new(did_hash, LinkTypes::DidToSelfRecoveryConfig)?,
        GetStrategy::default(),
    )?;

    let link = links.first().ok_or(wasm_error!(WasmErrorInner::Guest(
        "No self-recovery config found for this DID".into()
    )))?;
    let original_hash = ActionHash::try_from(link.target.clone())
        .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

    let record = get(original_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Self-recovery config record not found".into()
        )))?;

    let mut config: SelfRecoveryConfig = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Failed to decode self-recovery config".into()
        )))?;

    // Verify caller is owner
    let agent = agent_info()?.agent_initial_pubkey;
    if config.owner != agent {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Only the DID owner can add verification anchors".into()
        )));
    }

    // Add the anchor
    config.anchors.push(input.anchor);
    config.updated = sys_time()?;

    // Adjust time lock based on anchor count (more anchors = shorter lock)
    config.time_lock = match config.anchors.len() {
        0 => SELF_RECOVERY_DEFAULT_TIME_LOCK,        // 7 days (shouldn't reach here)
        1 => 14 * 24 * 3600,                          // 14 days
        2 => SELF_RECOVERY_DEFAULT_TIME_LOCK,         // 7 days
        _ => SELF_RECOVERY_MIN_TIME_LOCK,             // 72 hours
    };

    // Adjust threshold: require majority of anchors
    config.anchor_threshold = ((config.anchors.len() as f64 * 0.5).ceil() as u32).max(1);

    let new_hash = update_entry(original_hash, &config)?;
    get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Failed to get updated self-recovery config".into()
        )))
}

/// Get self-recovery config for a DID.
#[hdk_extern]
pub fn get_self_recovery_config(did: String) -> ExternResult<Option<Record>> {
    let did_hash = string_to_entry_hash(&did);
    let links = get_links(
        LinkQuery::try_new(did_hash, LinkTypes::DidToSelfRecoveryConfig)?,
        GetStrategy::default(),
    )?;

    match links.first() {
        Some(link) => {
            let hash = ActionHash::try_from(link.target.clone())
                .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;
            get(hash, GetOptions::default())
        }
        None => Ok(None),
    }
}

/// Mark self-recovery as superseded by social recovery.
///
/// Called by the Hearth auto-recovery when social recovery is configured.
/// Self-recovery remains available as a fallback.
#[hdk_extern]
pub fn mark_self_recovery_superseded(did: String) -> ExternResult<()> {
    let did_hash = string_to_entry_hash(&did);
    let links = get_links(
        LinkQuery::try_new(did_hash, LinkTypes::DidToSelfRecoveryConfig)?,
        GetStrategy::default(),
    )?;

    if let Some(link) = links.first() {
        let hash = ActionHash::try_from(link.target.clone())
            .map_err(|_| wasm_error!(WasmErrorInner::Guest("Invalid link target".into())))?;

        if let Some(record) = get(hash.clone(), GetOptions::default())? {
            let mut config: SelfRecoveryConfig = record
                .entry()
                .to_app_option()
                .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
                .ok_or(wasm_error!(WasmErrorInner::Guest(
                    "Failed to decode config".into()
                )))?;

            config.superseded_by_social = true;
            config.updated = sys_time()?;
            update_entry(hash, &config)?;
        }
    }

    Ok(())
}

/// Input for initiating self-recovery.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct InitiateSelfRecoveryInput {
    pub did: String,
    pub new_agent: AgentPubKey,
    pub reason: String,
    /// Initial verified anchor (must match one in the config)
    pub initial_anchor: VerificationAnchor,
}

/// Initiate a self-recovery request.
///
/// Requires proving control of at least one verification anchor.
/// The request enters a time-locked waiting period before it can execute.
#[hdk_extern]
pub fn initiate_self_recovery(input: InitiateSelfRecoveryInput) -> ExternResult<Record> {
    // Fetch self-recovery config
    let config_record = get_self_recovery_config(input.did.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "No self-recovery config found".into()
        )))?;

    let config: SelfRecoveryConfig = config_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Failed to decode config".into()
        )))?;

    if !config.active {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Self-recovery is not active for this DID".into()
        )));
    }

    if config.anchors.is_empty() {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Cannot initiate self-recovery without enrolled verification anchors".into()
        )));
    }

    // Verify the initial anchor matches one in the config
    if !config.anchors.contains(&input.initial_anchor) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Verification anchor does not match any enrolled anchor".into()
        )));
    }

    let now = sys_time()?;
    let request_id = format!(
        "self-recovery-{}-{}",
        input.did.chars().skip(12).take(8).collect::<String>(),
        now.as_micros()
    );

    let request = SelfRecoveryRequest {
        id: request_id,
        did: input.did.clone(),
        new_agent: input.new_agent,
        verified_anchors: vec![input.initial_anchor],
        status: RecoveryStatus::Pending,
        created: now,
        time_lock_expires: None, // Set when anchor threshold met
        reason: input.reason,
    };

    let action_hash = create_entry(&EntryTypes::SelfRecoveryRequest(request))?;
    let did_hash = string_to_entry_hash(&input.did);
    create_link(
        did_hash,
        action_hash.clone(),
        LinkTypes::DidToSelfRecoveryRequest,
        LinkTag::new("self_recovery_request"),
    )?;

    get(action_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Failed to get created self-recovery request".into()
        )))
}

/// Input for verifying an additional anchor during self-recovery.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VerifySelfRecoveryAnchorInput {
    pub request_action_hash: ActionHash,
    pub anchor: VerificationAnchor,
}

/// Verify an additional anchor on an active self-recovery request.
///
/// When the anchor threshold is met, the request transitions to Approved
/// and the time lock countdown begins.
#[hdk_extern]
pub fn verify_self_recovery_anchor(
    input: VerifySelfRecoveryAnchorInput,
) -> ExternResult<Record> {
    let record = get(input.request_action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Self-recovery request not found".into()
        )))?;

    let mut request: SelfRecoveryRequest = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Failed to decode request".into()
        )))?;

    if request.status != RecoveryStatus::Pending {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Can only verify anchors on pending requests".into()
        )));
    }

    // Get the config to check threshold
    let config_record = get_self_recovery_config(request.did.clone())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Self-recovery config not found".into()
        )))?;
    let config: SelfRecoveryConfig = config_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Failed to decode config".into()
        )))?;

    // Verify anchor matches config
    if !config.anchors.contains(&input.anchor) {
        return Err(wasm_error!(WasmErrorInner::Guest(
            "Anchor does not match any enrolled anchor".into()
        )));
    }

    // Don't double-count
    if !request.verified_anchors.contains(&input.anchor) {
        request.verified_anchors.push(input.anchor);
    }

    // Check if threshold met → transition to Approved
    if request.verified_anchors.len() as u32 >= config.anchor_threshold {
        request.status = RecoveryStatus::Approved;
        let now = sys_time()?;
        let expires = Timestamp::from_micros(
            now.as_micros() + (config.time_lock as i64 * 1_000_000),
        );
        request.time_lock_expires = Some(expires);
    }

    let new_hash = update_entry(input.request_action_hash, &request)?;
    get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Failed to get updated request".into()
        )))
}

/// Execute a self-recovery request after time lock expires.
///
/// Completes the identity rotation — same effect as social recovery execution.
#[hdk_extern]
pub fn execute_self_recovery(request_action_hash: ActionHash) -> ExternResult<Record> {
    let record = get(request_action_hash.clone(), GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Self-recovery request not found".into()
        )))?;

    let mut request: SelfRecoveryRequest = record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Failed to decode request".into()
        )))?;

    // Must be Approved (time lock set)
    if request.status != RecoveryStatus::Approved {
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Cannot execute self-recovery in {:?} status (must be Approved)",
            request.status
        ))));
    }

    // Check time lock has expired
    let now = sys_time()?;
    let expires = request.time_lock_expires.ok_or(wasm_error!(
        WasmErrorInner::Guest("No time lock expiry set".into())
    ))?;
    if now < expires {
        let remaining_secs = (expires.as_micros() - now.as_micros()) / 1_000_000;
        return Err(wasm_error!(WasmErrorInner::Guest(format!(
            "Time lock has not expired — {} seconds remaining",
            remaining_secs
        ))));
    }

    // Execute: mark as completed
    request.status = RecoveryStatus::Completed;
    let new_hash = update_entry(request_action_hash, &request)?;

    // Cross-zome call to DID registry to rotate the key
    let _ = call(
        CallTargetCell::Local,
        ZomeName::new("did_registry"),
        FunctionName::new("claim_recovered_did"),
        None,
        request.did.clone(),
    );

    get(new_hash, GetOptions::default())?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Failed to get completed request".into()
        )))
}
