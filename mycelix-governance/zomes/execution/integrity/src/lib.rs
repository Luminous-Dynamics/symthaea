//! Execution Integrity Zome
//! Defines entry types and validation for proposal execution
//!
//! Updated to use HDI 0.7 patterns

use hdi::prelude::*;

/// Anchor entry for deterministic link bases
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Anchor(pub String);

/// Timelock for approved proposals
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Timelock {
    /// Timelock identifier
    pub id: String,
    /// Proposal ID
    pub proposal_id: String,
    /// Actions to execute (JSON)
    pub actions: String,
    /// When timelock started
    pub started: Timestamp,
    /// When timelock expires
    pub expires: Timestamp,
    /// Timelock status
    pub status: TimelockStatus,
    /// Cancellation reason if cancelled
    pub cancellation_reason: Option<String>,
}

/// Status of a timelock
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum TimelockStatus {
    /// Waiting for timelock to expire
    Pending,
    /// Ready to execute
    Ready,
    /// Successfully executed
    Executed,
    /// Cancelled before execution
    Cancelled,
    /// Execution failed
    Failed,
}

/// Execution record for a proposal
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Execution {
    /// Execution identifier
    pub id: String,
    /// Timelock ID
    pub timelock_id: String,
    /// Proposal ID
    pub proposal_id: String,
    /// Executor's DID
    pub executor: String,
    /// Execution status
    pub status: ExecutionStatus,
    /// Result data (JSON)
    pub result: Option<String>,
    /// Error message if failed
    pub error: Option<String>,
    /// Execution timestamp
    pub executed_at: Timestamp,
}

/// Status of an execution
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ExecutionStatus {
    Success,
    PartialSuccess,
    Failed,
}

/// Guardian veto (for emergency cancellation)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GuardianVeto {
    /// Veto identifier
    pub id: String,
    /// Timelock ID being vetoed
    pub timelock_id: String,
    /// Guardian's DID
    pub guardian: String,
    /// Reason for veto
    pub reason: String,
    /// Veto timestamp
    pub vetoed_at: Timestamp,
}

/// Status of a fund allocation
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AllocationStatus {
    /// Funds locked in escrow awaiting execution
    Locked,
    /// Funds released after successful execution
    Released,
    /// Funds returned after veto or expiration
    Refunded,
}

/// Fund allocation — tracks locked funds for a proposal's execution
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct FundAllocation {
    /// Allocation identifier
    pub id: String,
    /// Associated proposal ID
    pub proposal_id: String,
    /// Associated timelock ID
    pub timelock_id: String,
    /// Account the funds are locked from
    pub source_account: String,
    /// Amount locked
    pub amount: f64,
    /// Currency denomination (e.g., "credits")
    pub currency: String,
    /// When funds were locked
    pub locked_at: Timestamp,
    /// Current allocation status
    pub status: AllocationStatus,
    /// Reason for status change (refund reason, release confirmation, etc.)
    pub status_reason: Option<String>,
}

#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    Anchor(Anchor),
    Timelock(Timelock),
    Execution(Execution),
    GuardianVeto(GuardianVeto),
    FundAllocation(FundAllocation),
}

#[hdk_link_types]
pub enum LinkTypes {
    /// Proposal to timelock
    ProposalToTimelock,
    /// Timelock to execution
    TimelockToExecution,
    /// Pending timelocks
    PendingTimelocks,
    /// Guardian to vetoes
    GuardianToVeto,
    /// Proposal to fund allocation
    ProposalToFundAllocation,
}

/// HDI 0.7 single validation callback using FlatOp pattern
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op.flattened::<EntryTypes, LinkTypes>()? {
        FlatOp::StoreEntry(store_entry) => match store_entry {
            OpEntry::CreateEntry { app_entry, action } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Timelock(timelock) => validate_create_timelock(action, timelock),
                EntryTypes::Execution(execution) => validate_create_execution(action, execution),
                EntryTypes::GuardianVeto(veto) => validate_create_veto(action, veto),
                EntryTypes::FundAllocation(alloc) => validate_create_fund_allocation(action, alloc),
            },
            OpEntry::UpdateEntry {
                app_entry,
                action,
                original_action_hash,
                original_entry_hash: _,
            } => match app_entry {
                EntryTypes::Anchor(_) => Ok(ValidateCallbackResult::Valid),
                EntryTypes::Timelock(timelock) => {
                    validate_update_timelock(action, timelock, original_action_hash)
                }
                EntryTypes::Execution(_) => {
                    // Executions cannot be updated once created
                    Ok(ValidateCallbackResult::Invalid(
                        "Execution records cannot be modified".into(),
                    ))
                }
                EntryTypes::GuardianVeto(_) => {
                    // Vetoes cannot be updated
                    Ok(ValidateCallbackResult::Invalid(
                        "Vetoes cannot be modified".into(),
                    ))
                }
                EntryTypes::FundAllocation(alloc) => {
                    validate_update_fund_allocation(action, alloc, original_action_hash)
                }
            },
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterCreateLink {
            link_type,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            LinkTypes::ProposalToTimelock => Ok(ValidateCallbackResult::Valid),
            LinkTypes::TimelockToExecution => Ok(ValidateCallbackResult::Valid),
            LinkTypes::PendingTimelocks => Ok(ValidateCallbackResult::Valid),
            LinkTypes::GuardianToVeto => Ok(ValidateCallbackResult::Valid),
            LinkTypes::ProposalToFundAllocation => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::RegisterDeleteLink {
            link_type,
            original_action: _,
            base_address: _,
            target_address: _,
            tag: _,
            action: _,
        } => match link_type {
            // Allow removing from pending list when executed/cancelled
            LinkTypes::PendingTimelocks => Ok(ValidateCallbackResult::Valid),
            _ => Ok(ValidateCallbackResult::Valid),
        },
        FlatOp::StoreRecord(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterAgentActivity(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterUpdate(_) => Ok(ValidateCallbackResult::Valid),
        FlatOp::RegisterDelete(_) => Ok(ValidateCallbackResult::Valid),
    }
}

/// Validate timelock creation
fn validate_create_timelock(
    _action: Create,
    timelock: Timelock,
) -> ExternResult<ValidateCallbackResult> {
    // Validate expiry is after start
    if timelock.expires <= timelock.started {
        return Ok(ValidateCallbackResult::Invalid(
            "Timelock expiry must be after start".into(),
        ));
    }

    // Validate actions is valid JSON
    if serde_json::from_str::<serde_json::Value>(&timelock.actions).is_err() {
        return Ok(ValidateCallbackResult::Invalid(
            "Actions must be valid JSON".into(),
        ));
    }

    // Validate initial status is Pending
    if timelock.status != TimelockStatus::Pending {
        return Ok(ValidateCallbackResult::Invalid(
            "Initial timelock status must be Pending".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate timelock update
fn validate_update_timelock(
    _action: Update,
    timelock: Timelock,
    original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    // Get original timelock for comparison
    let original_record = must_get_valid_record(original_action_hash)?;
    let original_timelock: Timelock = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original timelock not found".into()
        )))?;

    // Cannot change proposal ID
    if timelock.proposal_id != original_timelock.proposal_id {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change timelock proposal ID".into(),
        ));
    }

    // Validate status transitions
    match (&original_timelock.status, &timelock.status) {
        (TimelockStatus::Pending, TimelockStatus::Ready) => Ok(ValidateCallbackResult::Valid),
        (TimelockStatus::Pending, TimelockStatus::Cancelled) => Ok(ValidateCallbackResult::Valid),
        (TimelockStatus::Ready, TimelockStatus::Executed) => Ok(ValidateCallbackResult::Valid),
        (TimelockStatus::Ready, TimelockStatus::Failed) => Ok(ValidateCallbackResult::Valid),
        (TimelockStatus::Ready, TimelockStatus::Cancelled) => Ok(ValidateCallbackResult::Valid),
        _ => Ok(ValidateCallbackResult::Invalid(
            "Invalid timelock status transition".into(),
        )),
    }
}

/// Validate execution creation
fn validate_create_execution(
    _action: Create,
    execution: Execution,
) -> ExternResult<ValidateCallbackResult> {
    // Validate executor is a DID
    if !execution.executor.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Executor must be a valid DID".into(),
        ));
    }

    // Validate result is valid JSON if provided
    if let Some(ref result) = execution.result {
        if serde_json::from_str::<serde_json::Value>(result).is_err() {
            return Ok(ValidateCallbackResult::Invalid(
                "Result must be valid JSON".into(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate guardian veto creation
fn validate_create_veto(
    _action: Create,
    veto: GuardianVeto,
) -> ExternResult<ValidateCallbackResult> {
    // Validate guardian is a DID
    if !veto.guardian.starts_with("did:") {
        return Ok(ValidateCallbackResult::Invalid(
            "Guardian must be a valid DID".into(),
        ));
    }

    // Validate reason provided
    if veto.reason.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Veto reason is required".into(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate fund allocation creation
fn validate_create_fund_allocation(
    _action: Create,
    alloc: FundAllocation,
) -> ExternResult<ValidateCallbackResult> {
    if alloc.amount <= 0.0 || !alloc.amount.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Fund allocation amount must be positive and finite".into(),
        ));
    }
    if alloc.source_account.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Source account is required".into(),
        ));
    }
    if alloc.currency.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Currency is required".into(),
        ));
    }
    if alloc.status != AllocationStatus::Locked {
        return Ok(ValidateCallbackResult::Invalid(
            "Initial allocation status must be Locked".into(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate fund allocation update (status transitions)
fn validate_update_fund_allocation(
    _action: Update,
    alloc: FundAllocation,
    original_action_hash: ActionHash,
) -> ExternResult<ValidateCallbackResult> {
    let original_record = must_get_valid_record(original_action_hash)?;
    let original: FundAllocation = original_record
        .entry()
        .to_app_option()
        .map_err(|e| wasm_error!(WasmErrorInner::Guest(e.to_string())))?
        .ok_or(wasm_error!(WasmErrorInner::Guest(
            "Original fund allocation not found".into()
        )))?;

    // Cannot change proposal or amount
    if alloc.proposal_id != original.proposal_id {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change allocation proposal ID".into(),
        ));
    }
    if (alloc.amount - original.amount).abs() > f64::EPSILON {
        return Ok(ValidateCallbackResult::Invalid(
            "Cannot change allocation amount".into(),
        ));
    }

    // Valid transitions: Locked → Released, Locked → Refunded
    match (&original.status, &alloc.status) {
        (AllocationStatus::Locked, AllocationStatus::Released) => Ok(ValidateCallbackResult::Valid),
        (AllocationStatus::Locked, AllocationStatus::Refunded) => Ok(ValidateCallbackResult::Valid),
        _ => Ok(ValidateCallbackResult::Invalid(format!(
            "Invalid allocation status transition: {:?} → {:?}",
            original.status, alloc.status
        ))),
    }
}
