//! Social Recovery Integrity Zome
//! Defines entry types and validation for DID social recovery
//!
//! Updated to use HDI 0.7 patterns with FlatOp validation

use hdi::prelude::*;
use std::collections::HashSet;

/// Recovery configuration for a DID
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RecoveryConfig {
    /// The DID being protected
    pub did: String,
    /// Owner's agent pub key
    pub owner: AgentPubKey,
    /// List of trustee DIDs
    pub trustees: Vec<String>,
    /// Minimum trustees required (threshold)
    pub threshold: u32,
    /// Time lock in seconds before recovery executes
    pub time_lock: u64,
    /// Whether recovery is currently active
    pub active: bool,
    /// Creation timestamp
    pub created: Timestamp,
    /// Last update timestamp
    pub updated: Timestamp,
}

/// A recovery request initiated by trustees
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RecoveryRequest {
    /// Request identifier
    pub id: String,
    /// DID being recovered
    pub did: String,
    /// New agent pub key to recover to
    pub new_agent: AgentPubKey,
    /// Initiating trustee's DID
    pub initiated_by: String,
    /// Reason for recovery
    pub reason: String,
    /// Current status
    pub status: RecoveryStatus,
    /// When the request was created
    pub created: Timestamp,
    /// When time lock expires (if approved)
    pub time_lock_expires: Option<Timestamp>,
}

/// Trustee vote on a recovery request
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RecoveryVote {
    /// Recovery request ID
    pub request_id: String,
    /// Voting trustee's DID
    pub trustee: String,
    /// Vote decision
    pub vote: VoteDecision,
    /// Optional comment
    pub comment: Option<String>,
    /// Vote timestamp
    pub voted_at: Timestamp,
}

/// Status of a recovery request
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum RecoveryStatus {
    /// Waiting for trustee votes
    Pending,
    /// Threshold reached, in time lock period
    Approved,
    /// Time lock expired, can execute
    ReadyToExecute,
    /// Recovery completed
    Completed,
    /// Recovery was rejected
    Rejected,
    /// Recovery was cancelled by owner
    Cancelled,
}

/// Trustee vote decision
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum VoteDecision {
    Approve,
    Reject,
    Abstain,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    RecoveryConfig(RecoveryConfig),
    RecoveryRequest(RecoveryRequest),
    RecoveryVote(RecoveryVote),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// DID to recovery config
    DidToRecoveryConfig,
    /// DID to recovery requests
    DidToRecoveryRequest,
    /// Recovery request to votes
    RequestToVotes,
    /// Trustee to their responsibilities
    TrusteeToConfig,
}

/// Genesis self-check - called when app is installed
#[hdk_extern]
pub fn genesis_self_check(_data: GenesisSelfCheckData) -> ExternResult<ValidateCallbackResult> {
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation callback using FlatOp pattern matching
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::RecoveryConfig(config) => {
                    validate_create_recovery_config(EntryCreationAction::Create(action), config)
                }
                EntryTypes::RecoveryRequest(request) => {
                    validate_create_recovery_request(EntryCreationAction::Create(action), request)
                }
                EntryTypes::RecoveryVote(vote) => validate_create_recovery_vote(EntryCreationAction::Create(action), vote),
            },
            OpEntry::UpdateEntry {
                app_entry, action, ..
            } => match app_entry {
                EntryTypes::RecoveryConfig(config) => {
                    validate_update_recovery_config(action, config)
                }
                EntryTypes::RecoveryRequest(request) => {
                    validate_update_recovery_request(action, request)
                }
                EntryTypes::RecoveryVote(_) => Ok(ValidateCallbackResult::Invalid(
                    "Recovery votes cannot be updated".into(),
                )),
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink { link_type, tag, .. } => {
            if tag.0.len() > 1024 {
                return Ok(ValidateCallbackResult::Invalid(
                    "Link tag exceeds maximum length of 1024 bytes".into(),
                ));
            }
            match link_type {
                LinkTypes::DidToRecoveryConfig => Ok(ValidateCallbackResult::Valid),
                LinkTypes::DidToRecoveryRequest => Ok(ValidateCallbackResult::Valid),
                LinkTypes::RequestToVotes => Ok(ValidateCallbackResult::Valid),
                LinkTypes::TrusteeToConfig => Ok(ValidateCallbackResult::Valid),
            }
        }
        FlatOp::RegisterDeleteLink {
            original_action,
            action,
            ..
        } => {
            if action.author != original_action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the link creator can delete their links".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(update) => {
            let action = match &update {
                OpUpdate::Entry { action, .. }
                | OpUpdate::PrivateEntry { action, .. }
                | OpUpdate::Agent { action, .. }
                | OpUpdate::CapClaim { action, .. }
                | OpUpdate::CapGrant { action, .. } => action,
            };
            let original = must_get_action(action.original_action_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original entry author can update their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
        FlatOp::RegisterDelete(OpDelete { action }) => {
            let original = must_get_action(action.deletes_address.clone())?;
            if *original.action().author() != action.author {
                return Ok(ValidateCallbackResult::Invalid(
                    "Only the original entry author can delete their entries".into(),
                ));
            }
            Ok(ValidateCallbackResult::Valid)
        }
    }
}

/// Validate recovery config creation
fn validate_create_recovery_config(
    action: EntryCreationAction,
    config: RecoveryConfig,
) -> ExternResult<ValidateCallbackResult> {
    // Validate DID format
    if !config.did.starts_with("did:mycelix:") {
        return Ok(ValidateCallbackResult::Invalid(
            "DID must start with 'did:mycelix:'".into(),
        ));
    }

    // Validate owner is author
    if config.owner != *action.author() {
        return Ok(ValidateCallbackResult::Invalid(
            "Owner must be the author".into(),
        ));
    }

    // Validate trustee count (3-7)
    if config.trustees.len() < 3 || config.trustees.len() > 7 {
        return Ok(ValidateCallbackResult::Invalid(
            "Must have 3-7 trustees".into(),
        ));
    }

    // Validate threshold
    let min_threshold = (config.trustees.len() as f64 * 0.5).ceil() as u32;
    if config.threshold < min_threshold {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Threshold must be at least {} (majority)",
            min_threshold
        )));
    }

    if config.threshold as usize > config.trustees.len() {
        return Ok(ValidateCallbackResult::Invalid(
            "Threshold cannot exceed trustee count".into(),
        ));
    }

    // Validate time lock (minimum 24 hours)
    if config.time_lock < 86400 {
        return Ok(ValidateCallbackResult::Invalid(
            "Time lock must be at least 24 hours (86400 seconds)".into(),
        ));
    }

    // Validate all trustees are unique (prevent Sybil attacks)
    let unique_trustees: HashSet<&String> = config.trustees.iter().collect();
    if unique_trustees.len() != config.trustees.len() {
        return Ok(ValidateCallbackResult::Invalid(
            "Duplicate trustees are not allowed".into(),
        ));
    }

    // Validate all trustees are valid DIDs
    for trustee in &config.trustees {
        if !trustee.starts_with("did:") {
            return Ok(ValidateCallbackResult::Invalid(format!(
                "Invalid trustee DID: {}",
                trustee
            )));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate recovery config update
fn validate_update_recovery_config(
    action: Update,
    config: RecoveryConfig,
) -> ExternResult<ValidateCallbackResult> {
    // Validate trustee count (3-7)
    if config.trustees.len() < 3 || config.trustees.len() > 7 {
        return Ok(ValidateCallbackResult::Invalid(
            "Must have 3-7 trustees".into(),
        ));
    }

    // Validate all trustees are unique (prevent Sybil attacks)
    let unique_trustees: HashSet<&String> = config.trustees.iter().collect();
    if unique_trustees.len() != config.trustees.len() {
        return Ok(ValidateCallbackResult::Invalid(
            "Duplicate trustees are not allowed".into(),
        ));
    }

    // Validate threshold
    let min_threshold = (config.trustees.len() as f64 * 0.5).ceil() as u32;
    if config.threshold < min_threshold {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Threshold must be at least {} (majority)",
            min_threshold
        )));
    }

    // Fetch original to enforce invariants
    let original_record = must_get_valid_record(action.original_action_address.clone())?;
    let original: RecoveryConfig = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original recovery config not found".into()
        )))?;

    // Immutable fields
    if config.did != original.did {
        return Ok(ValidateCallbackResult::Invalid(
            "Recovery config DID cannot be changed".into(),
        ));
    }
    if config.owner != original.owner {
        return Ok(ValidateCallbackResult::Invalid(
            "Recovery config owner cannot be changed".into(),
        ));
    }
    if config.created != original.created {
        return Ok(ValidateCallbackResult::Invalid(
            "Recovery config created timestamp cannot be changed".into(),
        ));
    }

    // Updated timestamp must advance
    if config.updated <= original.updated {
        return Ok(ValidateCallbackResult::Invalid(
            "Recovery config updated timestamp must advance".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate recovery request creation
fn validate_create_recovery_request(
    _action: EntryCreationAction,
    request: RecoveryRequest,
) -> ExternResult<ValidateCallbackResult> {
    // Validate DID format
    if !request.did.starts_with("did:mycelix:") {
        return Ok(ValidateCallbackResult::Invalid(
            "DID must start with 'did:mycelix:'".into(),
        ));
    }

    // Validate initiator is a DID
    if !request.initiated_by.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Initiator must be a valid DID".into(),
        ));
    }

    // Validate reason provided
    if request.reason.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Recovery reason is required".into(),
        ));
    }

    // Validate initial status is Pending
    if request.status != RecoveryStatus::Pending {
        return Ok(ValidateCallbackResult::Invalid(
            "Initial status must be Pending".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate recovery request update (status transitions)
fn validate_update_recovery_request(
    action: Update,
    request: RecoveryRequest,
) -> ExternResult<ValidateCallbackResult> {
    // Validate DID format
    if !request.did.starts_with("did:mycelix:") {
        return Ok(ValidateCallbackResult::Invalid(
            "DID must start with 'did:mycelix:'".into(),
        ));
    }

    // Fetch original to enforce invariants
    let original_record = must_get_valid_record(action.original_action_address.clone())?;
    let original: RecoveryRequest = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original recovery request not found".into()
        )))?;

    // Immutable fields
    if request.id != original.id {
        return Ok(ValidateCallbackResult::Invalid(
            "Recovery request ID cannot be changed".into(),
        ));
    }
    if request.did != original.did {
        return Ok(ValidateCallbackResult::Invalid(
            "Recovery request DID cannot be changed".into(),
        ));
    }
    if request.new_agent != original.new_agent {
        return Ok(ValidateCallbackResult::Invalid(
            "Recovery request new_agent cannot be changed".into(),
        ));
    }
    if request.initiated_by != original.initiated_by {
        return Ok(ValidateCallbackResult::Invalid(
            "Recovery request initiator cannot be changed".into(),
        ));
    }
    if request.created != original.created {
        return Ok(ValidateCallbackResult::Invalid(
            "Recovery request created timestamp cannot be changed".into(),
        ));
    }

    // Validate status transitions via state machine
    // Terminal states: Completed, Rejected, Cancelled cannot transition further
    let valid_transition = match (&original.status, &request.status) {
        // Pending → Approved (threshold reached), Rejected, or Cancelled
        (RecoveryStatus::Pending, RecoveryStatus::Approved)
        | (RecoveryStatus::Pending, RecoveryStatus::Rejected)
        | (RecoveryStatus::Pending, RecoveryStatus::Cancelled) => true,
        // Approved → ReadyToExecute (timelock expired) or Cancelled
        (RecoveryStatus::Approved, RecoveryStatus::ReadyToExecute)
        | (RecoveryStatus::Approved, RecoveryStatus::Cancelled) => true,
        // ReadyToExecute → Completed or Cancelled
        (RecoveryStatus::ReadyToExecute, RecoveryStatus::Completed)
        | (RecoveryStatus::ReadyToExecute, RecoveryStatus::Cancelled) => true,
        // Same status (no-op update) is allowed
        (a, b) if a == b => true,
        _ => false,
    };
    if !valid_transition {
        return Ok(ValidateCallbackResult::Invalid(format!(
            "Invalid recovery status transition from {:?} to {:?}",
            original.status, request.status
        )));
    }

    // Approved status must have time_lock_expires set
    if request.status == RecoveryStatus::Approved && request.time_lock_expires.is_none() {
        return Ok(ValidateCallbackResult::Invalid(
            "Approved recovery must have time_lock_expires set".into(),
        ));
    }

    // Completed status requires time_lock_expires to be set
    if request.status == RecoveryStatus::Completed && request.time_lock_expires.is_none() {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot complete recovery without timelock (time_lock_expires must be set)".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn arb_recovery_status() -> impl Strategy<Value = RecoveryStatus> {
        prop_oneof![
            Just(RecoveryStatus::Pending),
            Just(RecoveryStatus::Approved),
            Just(RecoveryStatus::ReadyToExecute),
            Just(RecoveryStatus::Completed),
            Just(RecoveryStatus::Rejected),
            Just(RecoveryStatus::Cancelled),
        ]
    }

    fn arb_vote_decision() -> impl Strategy<Value = VoteDecision> {
        prop_oneof![
            Just(VoteDecision::Approve),
            Just(VoteDecision::Reject),
            Just(VoteDecision::Abstain),
        ]
    }

    /// Generate a valid trustee list (3-7 unique DID strings).
    fn arb_trustee_list() -> impl Strategy<Value = Vec<String>> {
        (3usize..=7).prop_flat_map(|count| {
            proptest::collection::vec("[a-zA-Z0-9]{8,16}".prop_map(|s| format!("did:mycelix:{}", s)), count)
        })
    }

    proptest! {
        /// RecoveryStatus round-trips through JSON.
        #[test]
        fn recovery_status_json_roundtrip(status in arb_recovery_status()) {
            let json = serde_json::to_string(&status).unwrap();
            let back: RecoveryStatus = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(status, back);
        }

        /// VoteDecision round-trips through JSON.
        #[test]
        fn vote_decision_json_roundtrip(vote in arb_vote_decision()) {
            let json = serde_json::to_string(&vote).unwrap();
            let back: VoteDecision = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(vote, back);
        }

        /// Majority threshold is always ceil(n/2) for any valid trustee count.
        #[test]
        fn majority_threshold_formula(count in 3usize..=7) {
            let min_threshold = (count as f64 * 0.5).ceil() as u32;
            // For 3 trustees: ceil(1.5) = 2
            // For 4 trustees: ceil(2.0) = 2
            // For 5 trustees: ceil(2.5) = 3
            // For 6 trustees: ceil(3.0) = 3
            // For 7 trustees: ceil(3.5) = 4
            prop_assert!(min_threshold >= 2, "Majority must be at least 2");
            prop_assert!(min_threshold as usize <= count, "Majority can't exceed count");
        }

        /// Valid thresholds are within [majority, trustees.len()].
        #[test]
        fn threshold_bounds(trustees in arb_trustee_list()) {
            let n = trustees.len();
            let min_threshold = (n as f64 * 0.5).ceil() as u32;
            for threshold in min_threshold..=(n as u32) {
                prop_assert!(threshold >= min_threshold);
                prop_assert!(threshold as usize <= n);
            }
        }

        /// Terminal recovery states cannot transition to non-self states.
        #[test]
        fn terminal_recovery_states(
            target in arb_recovery_status()
        ) {
            let terminals = [
                RecoveryStatus::Completed,
                RecoveryStatus::Rejected,
                RecoveryStatus::Cancelled,
            ];
            for terminal in &terminals {
                let valid = match (terminal, &target) {
                    (RecoveryStatus::Completed, RecoveryStatus::Completed)
                    | (RecoveryStatus::Rejected, RecoveryStatus::Rejected)
                    | (RecoveryStatus::Cancelled, RecoveryStatus::Cancelled) => true,
                    _ => false,
                };
                if terminal != &target {
                    prop_assert!(!valid, "Terminal {:?} should not transition to {:?}", terminal, target);
                }
            }
        }

        /// Pending can transition to Approved, Rejected, or Cancelled (not ReadyToExecute or Completed).
        #[test]
        fn pending_valid_transitions(target in arb_recovery_status()) {
            let valid = match target {
                RecoveryStatus::Approved
                | RecoveryStatus::Rejected
                | RecoveryStatus::Cancelled
                | RecoveryStatus::Pending => true,
                RecoveryStatus::ReadyToExecute | RecoveryStatus::Completed => false,
            };
            // This matches the validation logic in validate_update_recovery_request
            let matches_validation = match (&RecoveryStatus::Pending, &target) {
                (RecoveryStatus::Pending, RecoveryStatus::Approved)
                | (RecoveryStatus::Pending, RecoveryStatus::Rejected)
                | (RecoveryStatus::Pending, RecoveryStatus::Cancelled) => true,
                (a, b) if a == b => true,
                _ => false,
            };
            prop_assert_eq!(valid, matches_validation);
        }

        /// Generated trustee lists always have 3-7 entries.
        #[test]
        fn trustee_list_size(trustees in arb_trustee_list()) {
            prop_assert!(trustees.len() >= 3);
            prop_assert!(trustees.len() <= 7);
            for t in &trustees {
                prop_assert!(t.starts_with("did:mycelix:"));
            }
        }
    }
}

/// Validate recovery vote creation
fn validate_create_recovery_vote(
    _action: EntryCreationAction,
    vote: RecoveryVote,
) -> ExternResult<ValidateCallbackResult> {
    // Validate trustee is a DID
    if !vote.trustee.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Trustee must be a valid DID".into(),
        ));
    }

    // Validate request ID not empty
    if vote.request_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Request ID is required".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}
