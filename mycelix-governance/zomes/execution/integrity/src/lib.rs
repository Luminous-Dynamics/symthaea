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
    /// O(1) lookup: timelock ID anchor → timelock record
    TimelockById,
}

// ---------------------------------------------------------------------------
// Pure check functions (no HDI host calls — unit-testable)
// ---------------------------------------------------------------------------

/// Check that a new timelock is valid: expires > started, valid JSON actions,
/// and initial status is Pending.
pub fn check_create_timelock(timelock: &Timelock) -> Result<(), String> {
    if timelock.expires <= timelock.started {
        return Err("Timelock expiry must be after start".into());
    }
    if serde_json::from_str::<serde_json::Value>(&timelock.actions).is_err() {
        return Err("Actions must be valid JSON".into());
    }
    if timelock.status != TimelockStatus::Pending {
        return Err("Initial timelock status must be Pending".into());
    }
    Ok(())
}

/// Check that a timelock update is valid: immutable proposal_id, and only
/// whitelisted status transitions are allowed.
pub fn check_update_timelock(original: &Timelock, updated: &Timelock) -> Result<(), String> {
    if updated.proposal_id != original.proposal_id {
        return Err("Cannot change timelock proposal ID".into());
    }
    match (&original.status, &updated.status) {
        (TimelockStatus::Pending, TimelockStatus::Ready)
        | (TimelockStatus::Pending, TimelockStatus::Cancelled)
        | (TimelockStatus::Ready, TimelockStatus::Executed)
        | (TimelockStatus::Ready, TimelockStatus::Failed)
        | (TimelockStatus::Ready, TimelockStatus::Cancelled) => Ok(()),
        _ => Err("Invalid timelock status transition".into()),
    }
}

/// Check that a new execution record is valid: executor starts with "did:",
/// and result (if present) is valid JSON.
pub fn check_create_execution(execution: &Execution) -> Result<(), String> {
    if !execution.executor.starts_with("did:") {
        return Err("Executor must be a valid DID".into());
    }
    if let Some(ref result) = execution.result {
        if serde_json::from_str::<serde_json::Value>(result).is_err() {
            return Err("Result must be valid JSON".into());
        }
    }
    Ok(())
}

/// Check that a new guardian veto is valid: guardian starts with "did:" and
/// reason is not empty.
pub fn check_create_veto(veto: &GuardianVeto) -> Result<(), String> {
    if !veto.guardian.starts_with("did:") {
        return Err("Guardian must be a valid DID".into());
    }
    if veto.reason.is_empty() {
        return Err("Veto reason is required".into());
    }
    Ok(())
}

/// Check that a new fund allocation is valid: amount > 0 and finite,
/// source_account not empty, currency not empty, initial status is Locked.
pub fn check_create_fund_allocation(alloc: &FundAllocation) -> Result<(), String> {
    if alloc.amount <= 0.0 || !alloc.amount.is_finite() {
        return Err("Fund allocation amount must be positive and finite".into());
    }
    if alloc.source_account.is_empty() {
        return Err("Source account is required".into());
    }
    if alloc.currency.is_empty() {
        return Err("Currency is required".into());
    }
    if alloc.status != AllocationStatus::Locked {
        return Err("Initial allocation status must be Locked".into());
    }
    Ok(())
}

/// Check that a fund allocation update is valid: immutable proposal_id,
/// immutable amount, and only Locked→Released or Locked→Refunded transitions.
pub fn check_update_fund_allocation(
    original: &FundAllocation,
    updated: &FundAllocation,
) -> Result<(), String> {
    if updated.proposal_id != original.proposal_id {
        return Err("Cannot change allocation proposal ID".into());
    }
    if (updated.amount - original.amount).abs() > f64::EPSILON {
        return Err("Cannot change allocation amount".into());
    }
    match (&original.status, &updated.status) {
        (AllocationStatus::Locked, AllocationStatus::Released)
        | (AllocationStatus::Locked, AllocationStatus::Refunded) => Ok(()),
        _ => Err(format!(
            "Invalid allocation status transition: {:?} → {:?}",
            original.status, updated.status
        )),
    }
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
            LinkTypes::TimelockById => Ok(ValidateCallbackResult::Valid),
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
    match check_create_timelock(&timelock) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(reason) => Ok(ValidateCallbackResult::Invalid(reason)),
    }
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

    match check_update_timelock(&original_timelock, &timelock) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(reason) => Ok(ValidateCallbackResult::Invalid(reason)),
    }
}

/// Validate execution creation
fn validate_create_execution(
    _action: Create,
    execution: Execution,
) -> ExternResult<ValidateCallbackResult> {
    match check_create_execution(&execution) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(reason) => Ok(ValidateCallbackResult::Invalid(reason)),
    }
}

/// Validate guardian veto creation
fn validate_create_veto(
    _action: Create,
    veto: GuardianVeto,
) -> ExternResult<ValidateCallbackResult> {
    match check_create_veto(&veto) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(reason) => Ok(ValidateCallbackResult::Invalid(reason)),
    }
}

/// Validate fund allocation creation
fn validate_create_fund_allocation(
    _action: Create,
    alloc: FundAllocation,
) -> ExternResult<ValidateCallbackResult> {
    match check_create_fund_allocation(&alloc) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(reason) => Ok(ValidateCallbackResult::Invalid(reason)),
    }
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

    match check_update_fund_allocation(&original, &alloc) {
        Ok(()) => Ok(ValidateCallbackResult::Valid),
        Err(reason) => Ok(ValidateCallbackResult::Invalid(reason)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(micros: i64) -> Timestamp {
        Timestamp::from_micros(micros)
    }

    fn make_timelock() -> Timelock {
        Timelock {
            id: "tl-1".into(),
            proposal_id: "prop-1".into(),
            actions: r#"["transfer"]"#.into(),
            started: ts(1_000_000),
            expires: ts(2_000_000),
            status: TimelockStatus::Pending,
            cancellation_reason: None,
        }
    }

    fn make_execution() -> Execution {
        Execution {
            id: "ex-1".into(),
            timelock_id: "tl-1".into(),
            proposal_id: "prop-1".into(),
            executor: "did:key:z6Mk".into(),
            status: ExecutionStatus::Success,
            result: Some(r#"{"ok":true}"#.into()),
            error: None,
            executed_at: ts(3_000_000),
        }
    }

    fn make_veto() -> GuardianVeto {
        GuardianVeto {
            id: "v-1".into(),
            timelock_id: "tl-1".into(),
            guardian: "did:key:z6Guardian".into(),
            reason: "Emergency safety concern".into(),
            vetoed_at: ts(1_500_000),
        }
    }

    fn make_fund_allocation() -> FundAllocation {
        FundAllocation {
            id: "fa-1".into(),
            proposal_id: "prop-1".into(),
            timelock_id: "tl-1".into(),
            source_account: "treasury-main".into(),
            amount: 1000.0,
            currency: "credits".into(),
            locked_at: ts(1_000_000),
            status: AllocationStatus::Locked,
            status_reason: None,
        }
    }

    // ---- Timelock creation tests ----

    #[test]
    fn test_valid_timelock_accepted() {
        assert!(check_create_timelock(&make_timelock()).is_ok());
    }

    #[test]
    fn test_timelock_expiry_after_start() {
        let mut tl = make_timelock();
        // expires == started
        tl.expires = tl.started;
        assert_eq!(
            check_create_timelock(&tl).unwrap_err(),
            "Timelock expiry must be after start"
        );

        // expires < started
        tl.expires = ts(500_000);
        assert_eq!(
            check_create_timelock(&tl).unwrap_err(),
            "Timelock expiry must be after start"
        );
    }

    #[test]
    fn test_timelock_actions_must_be_json() {
        let mut tl = make_timelock();
        tl.actions = "not json {{{".into();
        assert_eq!(
            check_create_timelock(&tl).unwrap_err(),
            "Actions must be valid JSON"
        );
    }

    #[test]
    fn test_timelock_initial_status_pending() {
        let mut tl = make_timelock();
        tl.status = TimelockStatus::Ready;
        assert_eq!(
            check_create_timelock(&tl).unwrap_err(),
            "Initial timelock status must be Pending"
        );
    }

    // ---- Timelock update / status transition tests ----

    #[test]
    fn test_timelock_valid_status_transitions() {
        let original = make_timelock();

        // Pending → Ready
        let mut updated = original.clone();
        updated.status = TimelockStatus::Ready;
        assert!(check_update_timelock(&original, &updated).is_ok());

        // Pending → Cancelled
        let mut updated = original.clone();
        updated.status = TimelockStatus::Cancelled;
        assert!(check_update_timelock(&original, &updated).is_ok());

        // Ready → Executed
        let mut ready = original.clone();
        ready.status = TimelockStatus::Ready;
        let mut updated = ready.clone();
        updated.status = TimelockStatus::Executed;
        assert!(check_update_timelock(&ready, &updated).is_ok());

        // Ready → Failed
        let mut updated = ready.clone();
        updated.status = TimelockStatus::Failed;
        assert!(check_update_timelock(&ready, &updated).is_ok());

        // Ready → Cancelled
        let mut updated = ready.clone();
        updated.status = TimelockStatus::Cancelled;
        assert!(check_update_timelock(&ready, &updated).is_ok());
    }

    #[test]
    fn test_timelock_invalid_status_transition() {
        let original = make_timelock(); // Pending
        let mut updated = original.clone();
        updated.status = TimelockStatus::Executed; // Pending → Executed not allowed
        assert_eq!(
            check_update_timelock(&original, &updated).unwrap_err(),
            "Invalid timelock status transition"
        );
    }

    // ---- Execution tests ----

    #[test]
    fn test_execution_executor_must_be_did() {
        let mut ex = make_execution();
        ex.executor = "agent:abc".into();
        assert_eq!(
            check_create_execution(&ex).unwrap_err(),
            "Executor must be a valid DID"
        );

        // Valid DID passes
        ex.executor = "did:key:z6Mk".into();
        assert!(check_create_execution(&ex).is_ok());
    }

    // ---- Veto tests ----

    #[test]
    fn test_veto_guardian_must_be_did() {
        let mut v = make_veto();
        v.guardian = "not-a-did".into();
        assert_eq!(
            check_create_veto(&v).unwrap_err(),
            "Guardian must be a valid DID"
        );
    }

    #[test]
    fn test_veto_reason_required() {
        let mut v = make_veto();
        v.reason = String::new();
        assert_eq!(
            check_create_veto(&v).unwrap_err(),
            "Veto reason is required"
        );
    }

    // ---- Fund allocation tests ----

    #[test]
    fn test_fund_allocation_amount_positive() {
        let mut fa = make_fund_allocation();
        fa.amount = 0.0;
        assert_eq!(
            check_create_fund_allocation(&fa).unwrap_err(),
            "Fund allocation amount must be positive and finite"
        );

        fa.amount = -50.0;
        assert_eq!(
            check_create_fund_allocation(&fa).unwrap_err(),
            "Fund allocation amount must be positive and finite"
        );

        fa.amount = f64::NAN;
        assert_eq!(
            check_create_fund_allocation(&fa).unwrap_err(),
            "Fund allocation amount must be positive and finite"
        );

        fa.amount = f64::INFINITY;
        assert_eq!(
            check_create_fund_allocation(&fa).unwrap_err(),
            "Fund allocation amount must be positive and finite"
        );
    }

    #[test]
    fn test_fund_allocation_initial_status_locked() {
        let mut fa = make_fund_allocation();
        fa.status = AllocationStatus::Released;
        assert_eq!(
            check_create_fund_allocation(&fa).unwrap_err(),
            "Initial allocation status must be Locked"
        );
    }

    #[test]
    fn test_fund_allocation_valid_transitions() {
        let original = make_fund_allocation(); // Locked

        // Locked → Released
        let mut updated = original.clone();
        updated.status = AllocationStatus::Released;
        assert!(check_update_fund_allocation(&original, &updated).is_ok());

        // Locked → Refunded
        let mut updated = original.clone();
        updated.status = AllocationStatus::Refunded;
        assert!(check_update_fund_allocation(&original, &updated).is_ok());

        // Released → Refunded should fail
        let mut released = original.clone();
        released.status = AllocationStatus::Released;
        let mut updated = released.clone();
        updated.status = AllocationStatus::Refunded;
        assert!(check_update_fund_allocation(&released, &updated).is_err());
    }
}
