//! One-shot runtime authorization and unknown-outcome reconciliation.
//!
//! This crate validates mutation state transitions. It does not call tools,
//! persist a database, provide host locks, verify digital signatures, sandbox
//! an operator or promise exactly-once execution across process crashes.
//!
//! Production hosts must durably persist each returned state before continuing.
//! A dispatched attempt with an ambiguous outcome blocks new mutation
//! authorization until an external receipt proves a committed output or a
//! definite no-mutation result.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

pub const MUSE_RUNTIME_GATE_SCHEMA_V1: &str = "muse-runtime-gate/v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateErrorKindV1 {
    InvalidInput,
    Unauthorized,
    StaleState,
    DuplicateUse,
    BudgetExceeded,
    OutcomeUnresolved,
    InvalidTransition,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GateErrorV1 {
    pub kind: GateErrorKindV1,
    pub message: String,
}

impl GateErrorV1 {
    fn new(kind: GateErrorKindV1, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
    fn invalid(message: impl Into<String>) -> Self {
        Self::new(GateErrorKindV1::InvalidInput, message)
    }
    fn stale(message: impl Into<String>) -> Self {
        Self::new(GateErrorKindV1::StaleState, message)
    }
    fn transition(message: impl Into<String>) -> Self {
        Self::new(GateErrorKindV1::InvalidTransition, message)
    }
}

impl Display for GateErrorV1 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}
impl Error for GateErrorV1 {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MutationScopeV1 {
    SampleInterval,
    Phrase,
    Stem,
    Global,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttemptStatusV1 {
    Authorized,
    Dispatched,
    Committed,
    FailedNoMutation,
    OutcomeUnknown,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResourceBudgetV1 {
    pub compute_units: f64,
    pub latency_ms: u64,
    pub risk_units: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeHeadV1 {
    pub current_audio_sha256: String,
    pub registry_snapshot_sha256: String,
    pub session_ledger_sha256: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RemainingBudgetV1 {
    pub compute_units: f64,
    pub latency_ms: u64,
    pub risk_units: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegisteredOperatorV1 {
    pub operator_id: String,
    pub version: String,
    pub artifact_sha256: String,
    pub capabilities: Vec<String>,
    pub maximum_scope: MutationScopeV1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegisteredObserverV1 {
    pub metric_id: String,
    pub observer_profile_sha256: String,
    pub evaluator_artifact_sha256: String,
    pub calibration_receipt_sha256: String,
    pub independently_qualified: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthorityPolicyV1 {
    pub registry_snapshot_sha256: String,
    pub operators: Vec<RegisteredOperatorV1>,
    pub observers: Vec<RegisteredObserverV1>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreconditionReceiptV1 {
    pub precondition_id: String,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObserverEvidenceV1 {
    pub metric_id: String,
    pub observer_profile_sha256: String,
    pub evaluator_artifact_sha256: String,
    pub calibration_receipt_sha256: String,
    pub independently_qualified: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UserAuthorizationV1 {
    pub receipt_sha256: String,
    pub plan_sha256: String,
    pub plan_node_id: String,
    pub parent_audio_sha256: String,
    pub action_sha256: String,
    pub maximum_scope: MutationScopeV1,
    pub explicit_authorization: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MutationRequestV1 {
    pub grant_id_sha256: String,
    pub plan_sha256: String,
    pub plan_node_id: String,
    pub registry_snapshot_sha256: String,
    pub parent_audio_sha256: String,
    pub session_ledger_sha256: String,
    pub action_sha256: String,
    pub operator_id: String,
    pub operator_version: String,
    pub operator_artifact_sha256: String,
    pub required_capabilities: Vec<String>,
    pub requested_scope: MutationScopeV1,
    pub required_precondition_ids: Vec<String>,
    pub precondition_receipts: Vec<PreconditionReceiptV1>,
    pub required_observer_metric_ids: Vec<String>,
    pub observers: Vec<ObserverEvidenceV1>,
    pub user_authorization: UserAuthorizationV1,
    pub estimated_cost: ResourceBudgetV1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionGrantV1 {
    pub grant_id_sha256: String,
    pub plan_sha256: String,
    pub plan_node_id: String,
    pub registry_snapshot_sha256: String,
    pub parent_audio_sha256: String,
    pub session_ledger_sha256: String,
    pub action_sha256: String,
    pub operator_id: String,
    pub operator_version: String,
    pub operator_artifact_sha256: String,
    pub authorization_receipt_sha256: String,
    pub maximum_scope: MutationScopeV1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionAttemptV1 {
    pub grant: ExecutionGrantV1,
    pub status: AttemptStatusV1,
    pub dispatch_receipt_sha256: Option<String>,
    pub output_audio_sha256: Option<String>,
    pub execution_receipt_sha256: Option<String>,
    pub failure_receipt_sha256: Option<String>,
    pub reconciliation_receipt_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeStateV1 {
    pub schema: String,
    pub head: RuntimeHeadV1,
    pub remaining_budget: RemainingBudgetV1,
    pub attempts: Vec<ExecutionAttemptV1>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DispatchReceiptV1 {
    pub grant_id_sha256: String,
    pub parent_audio_sha256: String,
    pub action_sha256: String,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitReceiptV1 {
    pub grant_id_sha256: String,
    pub parent_audio_sha256: String,
    pub action_sha256: String,
    pub output_audio_sha256: String,
    pub execution_receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DefiniteFailureReceiptV1 {
    pub grant_id_sha256: String,
    pub parent_audio_sha256: String,
    pub action_sha256: String,
    pub failure_receipt_sha256: String,
    pub proves_no_mutation: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReconciliationOutcomeV1 {
    Committed { output_audio_sha256: String, execution_receipt_sha256: String },
    NoMutation { failure_receipt_sha256: String },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReconciliationReceiptV1 {
    pub grant_id_sha256: String,
    pub original_parent_audio_sha256: String,
    pub action_sha256: String,
    pub receipt_sha256: String,
    pub outcome: ReconciliationOutcomeV1,
}

pub fn initialize_runtime(
    head: RuntimeHeadV1,
    remaining_budget: RemainingBudgetV1,
) -> Result<RuntimeStateV1, GateErrorV1> {
    validate_digest(&head.current_audio_sha256, "current_audio_sha256")?;
    validate_digest(&head.registry_snapshot_sha256, "registry_snapshot_sha256")?;
    validate_digest(&head.session_ledger_sha256, "session_ledger_sha256")?;
    validate_nonnegative(remaining_budget.compute_units, "compute budget")?;
    validate_nonnegative(remaining_budget.risk_units, "risk budget")?;
    Ok(RuntimeStateV1 {
        schema: MUSE_RUNTIME_GATE_SCHEMA_V1.to_owned(),
        head,
        remaining_budget,
        attempts: Vec::new(),
    })
}

/// Returns a new state with a one-use grant and its resource reservation.
/// Persist the returned state durably before dispatching any external operator.
pub fn authorize_mutation(
    state: &RuntimeStateV1,
    policy: &AuthorityPolicyV1,
    request: &MutationRequestV1,
) -> Result<(RuntimeStateV1, ExecutionGrantV1), GateErrorV1> {
    validate_state(state)?;
    validate_policy(policy)?;
    validate_request(request)?;

    if state.attempts.iter().any(|attempt| matches!(
        attempt.status,
        AttemptStatusV1::Authorized | AttemptStatusV1::Dispatched | AttemptStatusV1::OutcomeUnknown
    )) {
        return Err(GateErrorV1::new(
            GateErrorKindV1::OutcomeUnresolved,
            "an earlier mutation is authorized, dispatched or unresolved; reconcile it first",
        ));
    }

    if state.head.registry_snapshot_sha256 != request.registry_snapshot_sha256
        || state.head.registry_snapshot_sha256 != policy.registry_snapshot_sha256
    {
        return Err(GateErrorV1::stale("registry snapshot is not the exact current registry"));
    }
    if state.head.current_audio_sha256 != request.parent_audio_sha256 {
        return Err(GateErrorV1::stale("parent audio is not the exact current runtime head"));
    }
    if state.head.session_ledger_sha256 != request.session_ledger_sha256 {
        return Err(GateErrorV1::stale("session ledger digest is not current"));
    }

    let operator = policy.operators.iter().find(|operator| {
        operator.operator_id == request.operator_id
            && operator.version == request.operator_version
            && operator.artifact_sha256 == request.operator_artifact_sha256
    }).ok_or_else(|| GateErrorV1::new(
        GateErrorKindV1::Unauthorized,
        "operator ID/version/artifact is not registered",
    ))?;

    if !request.required_capabilities.iter().all(|capability| operator.capabilities.contains(capability)) {
        return Err(GateErrorV1::new(
            GateErrorKindV1::Unauthorized,
            "operator does not satisfy all required capabilities",
        ));
    }
    if request.requested_scope > operator.maximum_scope {
        return Err(GateErrorV1::new(
            GateErrorKindV1::Unauthorized,
            "operator locality exceeds the requested mutation scope",
        ));
    }

    let auth = &request.user_authorization;
    if !auth.explicit_authorization
        || auth.plan_sha256 != request.plan_sha256
        || auth.plan_node_id != request.plan_node_id
        || auth.parent_audio_sha256 != request.parent_audio_sha256
        || auth.action_sha256 != request.action_sha256
        || auth.maximum_scope != request.requested_scope
    {
        return Err(GateErrorV1::new(
            GateErrorKindV1::Unauthorized,
            "user authorization must explicitly bind this exact plan/node/parent/action/scope",
        ));
    }

    validate_preconditions(
        &request.required_precondition_ids,
        &request.precondition_receipts,
    )?;
    validate_observers(
        &request.required_observer_metric_ids,
        &request.observers,
        &request.operator_artifact_sha256,
        &policy.observers,
    )?;

    validate_nonnegative(request.estimated_cost.compute_units, "estimated compute")?;
    validate_nonnegative(request.estimated_cost.risk_units, "estimated risk")?;
    if request.estimated_cost.compute_units > state.remaining_budget.compute_units
        || request.estimated_cost.latency_ms > state.remaining_budget.latency_ms
        || request.estimated_cost.risk_units > state.remaining_budget.risk_units
    {
        return Err(GateErrorV1::new(
            GateErrorKindV1::BudgetExceeded,
            "estimated resource cost exceeds remaining budget",
        ));
    }

    if state.attempts.iter().any(|attempt| attempt.grant.grant_id_sha256 == request.grant_id_sha256) {
        return Err(GateErrorV1::new(
            GateErrorKindV1::DuplicateUse,
            "grant ID was already used in this runtime ledger",
        ));
    }

    let grant = ExecutionGrantV1 {
        grant_id_sha256: request.grant_id_sha256.clone(),
        plan_sha256: request.plan_sha256.clone(),
        plan_node_id: request.plan_node_id.clone(),
        registry_snapshot_sha256: request.registry_snapshot_sha256.clone(),
        parent_audio_sha256: request.parent_audio_sha256.clone(),
        session_ledger_sha256: request.session_ledger_sha256.clone(),
        action_sha256: request.action_sha256.clone(),
        operator_id: request.operator_id.clone(),
        operator_version: request.operator_version.clone(),
        operator_artifact_sha256: request.operator_artifact_sha256.clone(),
        authorization_receipt_sha256: auth.receipt_sha256.clone(),
        maximum_scope: request.requested_scope,
    };

    let mut next = state.clone();
    next.remaining_budget.compute_units -= request.estimated_cost.compute_units;
    next.remaining_budget.latency_ms -= request.estimated_cost.latency_ms;
    next.remaining_budget.risk_units -= request.estimated_cost.risk_units;
    next.attempts.push(ExecutionAttemptV1 {
        grant: grant.clone(),
        status: AttemptStatusV1::Authorized,
        dispatch_receipt_sha256: None,
        output_audio_sha256: None,
        execution_receipt_sha256: None,
        failure_receipt_sha256: None,
        reconciliation_receipt_sha256: None,
    });
    Ok((next, grant))
}

/// Persist this returned DISPATCHED state before calling the operator.
pub fn mark_dispatched(
    state: &RuntimeStateV1,
    grant: &ExecutionGrantV1,
    receipt: &DispatchReceiptV1,
) -> Result<RuntimeStateV1, GateErrorV1> {
    validate_state(state)?;
    validate_digest(&receipt.receipt_sha256, "dispatch receipt")?;
    validate_receipt_binding(
        &grant.grant_id_sha256,
        &grant.parent_audio_sha256,
        &grant.action_sha256,
        &receipt.grant_id_sha256,
        &receipt.parent_audio_sha256,
        &receipt.action_sha256,
    )?;
    let mut next = state.clone();
    let attempt = find_attempt_mut(&mut next, &grant.grant_id_sha256)?;
    ensure_same_grant(&attempt.grant, grant)?;
    if attempt.status != AttemptStatusV1::Authorized || attempt.dispatch_receipt_sha256.is_some() {
        return Err(GateErrorV1::new(
            GateErrorKindV1::DuplicateUse,
            "grant is not in its one-use AUTHORIZED state",
        ));
    }
    attempt.status = AttemptStatusV1::Dispatched;
    attempt.dispatch_receipt_sha256 = Some(receipt.receipt_sha256.clone());
    Ok(next)
}

/// Only a valid commit advances the runtime audio head.
pub fn commit_execution(
    state: &RuntimeStateV1,
    grant: &ExecutionGrantV1,
    receipt: &CommitReceiptV1,
) -> Result<RuntimeStateV1, GateErrorV1> {
    validate_state(state)?;
    validate_digest(&receipt.output_audio_sha256, "output audio digest")?;
    validate_digest(&receipt.execution_receipt_sha256, "execution receipt")?;
    validate_receipt_binding(
        &grant.grant_id_sha256,
        &grant.parent_audio_sha256,
        &grant.action_sha256,
        &receipt.grant_id_sha256,
        &receipt.parent_audio_sha256,
        &receipt.action_sha256,
    )?;
    if state.head.current_audio_sha256 != grant.parent_audio_sha256 {
        return Err(GateErrorV1::stale("runtime audio head changed before commit"));
    }
    let mut next = state.clone();
    let attempt = find_attempt_mut(&mut next, &grant.grant_id_sha256)?;
    ensure_same_grant(&attempt.grant, grant)?;
    if attempt.status != AttemptStatusV1::Dispatched {
        return Err(GateErrorV1::transition("only a DISPATCHED attempt may commit directly"));
    }
    attempt.status = AttemptStatusV1::Committed;
    attempt.output_audio_sha256 = Some(receipt.output_audio_sha256.clone());
    attempt.execution_receipt_sha256 = Some(receipt.execution_receipt_sha256.clone());
    next.head.current_audio_sha256 = receipt.output_audio_sha256.clone();
    Ok(next)
}

/// A definite failure is retry-safe only when an independent receipt proves
/// that no output was committed. Otherwise the state must remain unresolved.
/// Cancel an unused AUTHORIZED grant before dispatch. This is a definite
/// no-mutation transition; the reservation is not refunded in v1.
pub fn cancel_undispatched_authorization(
    state: &RuntimeStateV1,
    grant: &ExecutionGrantV1,
    cancellation_receipt_sha256: &str,
) -> Result<RuntimeStateV1, GateErrorV1> {
    validate_state(state)?;
    validate_digest(cancellation_receipt_sha256, "cancellation receipt")?;
    let mut next = state.clone();
    let attempt = find_attempt_mut(&mut next, &grant.grant_id_sha256)?;
    ensure_same_grant(&attempt.grant, grant)?;
    if attempt.status != AttemptStatusV1::Authorized {
        return Err(GateErrorV1::transition(
            "only an undispatched AUTHORIZED grant can be cancelled",
        ));
    }
    attempt.status = AttemptStatusV1::FailedNoMutation;
    attempt.failure_receipt_sha256 = Some(cancellation_receipt_sha256.to_owned());
    Ok(next)
}

pub fn record_definite_failure(
    state: &RuntimeStateV1,
    grant: &ExecutionGrantV1,
    receipt: &DefiniteFailureReceiptV1,
) -> Result<RuntimeStateV1, GateErrorV1> {
    validate_state(state)?;
    validate_digest(&receipt.failure_receipt_sha256, "failure receipt")?;
    if !receipt.proves_no_mutation {
        return Err(GateErrorV1::new(
            GateErrorKindV1::OutcomeUnresolved,
            "failure receipt does not prove no mutation; use OUTCOME_UNKNOWN",
        ));
    }
    validate_receipt_binding(
        &grant.grant_id_sha256,
        &grant.parent_audio_sha256,
        &grant.action_sha256,
        &receipt.grant_id_sha256,
        &receipt.parent_audio_sha256,
        &receipt.action_sha256,
    )?;
    let mut next = state.clone();
    let attempt = find_attempt_mut(&mut next, &grant.grant_id_sha256)?;
    ensure_same_grant(&attempt.grant, grant)?;
    if attempt.status != AttemptStatusV1::Dispatched {
        return Err(GateErrorV1::transition("only a DISPATCHED attempt may fail definitively"));
    }
    attempt.status = AttemptStatusV1::FailedNoMutation;
    attempt.failure_receipt_sha256 = Some(receipt.failure_receipt_sha256.clone());
    Ok(next)
}

pub fn mark_outcome_unknown(
    state: &RuntimeStateV1,
    grant: &ExecutionGrantV1,
) -> Result<RuntimeStateV1, GateErrorV1> {
    validate_state(state)?;
    let mut next = state.clone();
    let attempt = find_attempt_mut(&mut next, &grant.grant_id_sha256)?;
    ensure_same_grant(&attempt.grant, grant)?;
    if attempt.status != AttemptStatusV1::Dispatched {
        return Err(GateErrorV1::transition("only a DISPATCHED attempt may become OUTCOME_UNKNOWN"));
    }
    attempt.status = AttemptStatusV1::OutcomeUnknown;
    Ok(next)
}

/// The only legal transition out of OUTCOME_UNKNOWN.
pub fn reconcile_unknown_outcome(
    state: &RuntimeStateV1,
    grant: &ExecutionGrantV1,
    receipt: &ReconciliationReceiptV1,
) -> Result<RuntimeStateV1, GateErrorV1> {
    validate_state(state)?;
    validate_digest(&receipt.receipt_sha256, "reconciliation receipt")?;
    validate_receipt_binding(
        &grant.grant_id_sha256,
        &grant.parent_audio_sha256,
        &grant.action_sha256,
        &receipt.grant_id_sha256,
        &receipt.original_parent_audio_sha256,
        &receipt.action_sha256,
    )?;
    if state.head.current_audio_sha256 != grant.parent_audio_sha256 {
        return Err(GateErrorV1::stale("runtime head changed while outcome was unknown"));
    }
    let mut next = state.clone();
    let attempt = find_attempt_mut(&mut next, &grant.grant_id_sha256)?;
    ensure_same_grant(&attempt.grant, grant)?;
    if attempt.status != AttemptStatusV1::OutcomeUnknown {
        return Err(GateErrorV1::transition("only OUTCOME_UNKNOWN may be reconciled"));
    }
    match &receipt.outcome {
        ReconciliationOutcomeV1::Committed {
            output_audio_sha256,
            execution_receipt_sha256,
        } => {
            validate_digest(output_audio_sha256, "reconciled output digest")?;
            validate_digest(execution_receipt_sha256, "reconciled execution receipt")?;
            attempt.status = AttemptStatusV1::Committed;
            attempt.output_audio_sha256 = Some(output_audio_sha256.clone());
            attempt.execution_receipt_sha256 = Some(execution_receipt_sha256.clone());
            attempt.reconciliation_receipt_sha256 = Some(receipt.receipt_sha256.clone());
            next.head.current_audio_sha256 = output_audio_sha256.clone();
        }
        ReconciliationOutcomeV1::NoMutation { failure_receipt_sha256 } => {
            validate_digest(failure_receipt_sha256, "reconciled failure receipt")?;
            attempt.status = AttemptStatusV1::FailedNoMutation;
            attempt.failure_receipt_sha256 = Some(failure_receipt_sha256.clone());
            attempt.reconciliation_receipt_sha256 = Some(receipt.receipt_sha256.clone());
        }
    }
    Ok(next)
}

fn validate_state(state: &RuntimeStateV1) -> Result<(), GateErrorV1> {
    if state.schema != MUSE_RUNTIME_GATE_SCHEMA_V1 {
        return Err(GateErrorV1::invalid("runtime state schema mismatch"));
    }
    validate_digest(&state.head.current_audio_sha256, "state current audio digest")?;
    validate_digest(&state.head.registry_snapshot_sha256, "state registry digest")?;
    validate_digest(&state.head.session_ledger_sha256, "state session digest")?;
    validate_nonnegative(state.remaining_budget.compute_units, "remaining compute")?;
    validate_nonnegative(state.remaining_budget.risk_units, "remaining risk")?;
    let mut ids = BTreeSet::new();
    let mut outstanding = 0usize;
    for attempt in &state.attempts {
        validate_grant(&attempt.grant)?;
        if !ids.insert(attempt.grant.grant_id_sha256.as_str()) {
            return Err(GateErrorV1::invalid("duplicate grant ID in runtime state"));
        }
        match attempt.status {
            AttemptStatusV1::Authorized => {
                outstanding += 1;
                if attempt.dispatch_receipt_sha256.is_some()
                    || attempt.output_audio_sha256.is_some()
                    || attempt.execution_receipt_sha256.is_some()
                    || attempt.failure_receipt_sha256.is_some()
                {
                    return Err(GateErrorV1::invalid("AUTHORIZED attempt contains later receipts"));
                }
            }
            AttemptStatusV1::Dispatched | AttemptStatusV1::OutcomeUnknown => {
                outstanding += 1;
                let d = attempt.dispatch_receipt_sha256.as_deref()
                    .ok_or_else(|| GateErrorV1::invalid("dispatched attempt lacks a dispatch receipt"))?;
                validate_digest(d, "dispatch receipt")?;
                if attempt.output_audio_sha256.is_some()
                    || attempt.execution_receipt_sha256.is_some()
                    || attempt.failure_receipt_sha256.is_some()
                {
                    return Err(GateErrorV1::invalid("nonterminal attempt contains terminal receipts"));
                }
            }
            AttemptStatusV1::Committed => {
                let o = attempt.output_audio_sha256.as_deref()
                    .ok_or_else(|| GateErrorV1::invalid("committed attempt lacks output digest"))?;
                let e = attempt.execution_receipt_sha256.as_deref()
                    .ok_or_else(|| GateErrorV1::invalid("committed attempt lacks execution receipt"))?;
                validate_digest(o, "committed output digest")?;
                validate_digest(e, "execution receipt")?;
                if attempt.failure_receipt_sha256.is_some() {
                    return Err(GateErrorV1::invalid("committed attempt contains failure receipt"));
                }
            }
            AttemptStatusV1::FailedNoMutation => {
                let f = attempt.failure_receipt_sha256.as_deref()
                    .ok_or_else(|| GateErrorV1::invalid("failed attempt lacks failure receipt"))?;
                validate_digest(f, "failure receipt")?;
                if attempt.output_audio_sha256.is_some() || attempt.execution_receipt_sha256.is_some() {
                    return Err(GateErrorV1::invalid("failed attempt contains committed output"));
                }
            }
        }
        if let Some(d) = &attempt.reconciliation_receipt_sha256 {
            validate_digest(d, "reconciliation receipt")?;
        }
    }
    if outstanding > 1 {
        return Err(GateErrorV1::invalid("runtime state contains more than one unresolved attempt"));
    }
    Ok(())
}

fn validate_policy(policy: &AuthorityPolicyV1) -> Result<(), GateErrorV1> {
    validate_digest(&policy.registry_snapshot_sha256, "policy registry digest")?;
    let mut seen = BTreeSet::new();
    for operator in &policy.operators {
        require_text(&operator.operator_id, "operator ID")?;
        require_text(&operator.version, "operator version")?;
        validate_digest(&operator.artifact_sha256, "operator artifact digest")?;
        validate_unique_texts(&operator.capabilities, "operator capabilities")?;
        if !seen.insert((operator.operator_id.as_str(), operator.version.as_str(), operator.artifact_sha256.as_str())) {
            return Err(GateErrorV1::invalid("duplicate operator profile in authority policy"));
        }
    }
    let mut observer_keys = BTreeSet::new();
    for observer in &policy.observers {
        require_text(&observer.metric_id, "registered observer metric ID")?;
        validate_digest(&observer.observer_profile_sha256, "registered observer profile digest")?;
        validate_digest(&observer.evaluator_artifact_sha256, "registered evaluator artifact digest")?;
        validate_digest(&observer.calibration_receipt_sha256, "registered observer calibration receipt")?;
        if !observer_keys.insert((observer.metric_id.as_str(), observer.observer_profile_sha256.as_str())) {
            return Err(GateErrorV1::invalid("duplicate observer profile in authority policy"));
        }
    }
    Ok(())
}

fn validate_request(request: &MutationRequestV1) -> Result<(), GateErrorV1> {
    validate_digest(&request.grant_id_sha256, "grant ID")?;
    validate_digest(&request.plan_sha256, "plan digest")?;
    require_text(&request.plan_node_id, "plan node ID")?;
    validate_digest(&request.registry_snapshot_sha256, "request registry digest")?;
    validate_digest(&request.parent_audio_sha256, "parent audio digest")?;
    validate_digest(&request.session_ledger_sha256, "session ledger digest")?;
    validate_digest(&request.action_sha256, "action digest")?;
    require_text(&request.operator_id, "operator ID")?;
    require_text(&request.operator_version, "operator version")?;
    validate_digest(&request.operator_artifact_sha256, "operator artifact digest")?;
    validate_unique_texts(&request.required_capabilities, "required capabilities")?;
    validate_unique_texts(&request.required_precondition_ids, "required preconditions")?;
    validate_unique_texts(&request.required_observer_metric_ids, "required observer metrics")?;
    if request.required_capabilities.is_empty() || request.required_observer_metric_ids.is_empty() {
        return Err(GateErrorV1::invalid("capabilities and observer obligations must be nonempty"));
    }
    validate_nonnegative(request.estimated_cost.compute_units, "estimated compute cost")?;
    validate_nonnegative(request.estimated_cost.risk_units, "estimated risk cost")?;

    validate_unique_texts(&request.required_capabilities, "required capabilities")?;
    validate_unique_texts(&request.required_precondition_ids, "required precondition IDs")?;
    validate_unique_texts(&request.required_observer_metric_ids, "required observer metric IDs")?;
    let auth = &request.user_authorization;
    validate_digest(&auth.receipt_sha256, "user authorization receipt")?;
    validate_digest(&auth.plan_sha256, "authorized plan digest")?;
    require_text(&auth.plan_node_id, "authorized plan node ID")?;
    validate_digest(&auth.parent_audio_sha256, "authorized parent audio digest")?;
    validate_digest(&auth.action_sha256, "authorized action digest")?;
    if !auth.explicit_authorization
        || auth.plan_sha256 != request.plan_sha256
        || auth.plan_node_id != request.plan_node_id
        || auth.parent_audio_sha256 != request.parent_audio_sha256
        || auth.action_sha256 != request.action_sha256
        || auth.maximum_scope != request.requested_scope
    {
        return Err(GateErrorV1::new(
            GateErrorKindV1::Unauthorized,
            "user authorization receipt must explicitly bind this exact plan node, parent, action and least-privilege scope",
        ));
    }
    Ok(())
}

fn validate_grant(grant: &ExecutionGrantV1) -> Result<(), GateErrorV1> {
    validate_digest(&grant.grant_id_sha256, "grant ID")?;
    validate_digest(&grant.plan_sha256, "plan digest")?;
    require_text(&grant.plan_node_id, "plan node ID")?;
    validate_digest(&grant.registry_snapshot_sha256, "grant registry digest")?;
    validate_digest(&grant.parent_audio_sha256, "grant parent digest")?;
    validate_digest(&grant.session_ledger_sha256, "grant session digest")?;
    validate_digest(&grant.action_sha256, "grant action digest")?;
    require_text(&grant.operator_id, "grant operator ID")?;
    require_text(&grant.operator_version, "grant operator version")?;
    validate_digest(&grant.operator_artifact_sha256, "grant operator digest")?;
    validate_digest(&grant.authorization_receipt_sha256, "grant authorization receipt")?;
    Ok(())
}

fn validate_preconditions(
    required: &[String],
    receipts: &[PreconditionReceiptV1],
) -> Result<(), GateErrorV1> {
    let mut seen = BTreeSet::new();
    for receipt in receipts {
        require_text(&receipt.precondition_id, "precondition ID")?;
        validate_digest(&receipt.receipt_sha256, "precondition receipt")?;
        if !seen.insert(receipt.precondition_id.as_str()) {
            return Err(GateErrorV1::invalid("duplicate precondition receipt"));
        }
    }
    let expected: BTreeSet<&str> = required.iter().map(String::as_str).collect();
    if seen != expected {
        return Err(GateErrorV1::new(
            GateErrorKindV1::Unauthorized,
            "precondition receipt set does not exactly match required preconditions",
        ));
    }
    Ok(())
}

fn validate_observers(
    required: &[String],
    observers: &[ObserverEvidenceV1],
    operator_digest: &str,
    registered: &[RegisteredObserverV1],
) -> Result<(), GateErrorV1> {
    let mut seen = BTreeSet::new();
    for observer in observers {
        require_text(&observer.metric_id, "observer metric ID")?;
        validate_digest(&observer.observer_profile_sha256, "observer profile digest")?;
        validate_digest(&observer.evaluator_artifact_sha256, "evaluator artifact digest")?;
        validate_digest(&observer.calibration_receipt_sha256, "observer calibration receipt")?;
        let profile = registered.iter().find(|profile| {
            profile.metric_id == observer.metric_id
                && profile.observer_profile_sha256 == observer.observer_profile_sha256
                && profile.evaluator_artifact_sha256 == observer.evaluator_artifact_sha256
                && profile.calibration_receipt_sha256 == observer.calibration_receipt_sha256
        });
        if !observer.independently_qualified
            || observer.evaluator_artifact_sha256 == operator_digest
            || profile.is_none_or(|profile| !profile.independently_qualified)
        {
            return Err(GateErrorV1::new(
                GateErrorKindV1::Unauthorized,
                "observer is not present in the frozen authority registry, is not qualified or reuses the operator artifact",
            ));
        }
        if !seen.insert(observer.metric_id.as_str()) {
            return Err(GateErrorV1::invalid("duplicate observer metric"));
        }
    }
    let expected: BTreeSet<&str> = required.iter().map(String::as_str).collect();
    if seen != expected {
        return Err(GateErrorV1::new(
            GateErrorKindV1::Unauthorized,
            "observer evidence set does not exactly match required metrics",
        ));
    }
    Ok(())
}

fn validate_receipt_binding(
    grant_id: &str,
    parent: &str,
    action: &str,
    receipt_grant_id: &str,
    receipt_parent: &str,
    receipt_action: &str,
) -> Result<(), GateErrorV1> {
    if grant_id != receipt_grant_id || parent != receipt_parent || action != receipt_action {
        return Err(GateErrorV1::invalid("execution receipt does not bind exact grant, parent and action"));
    }
    Ok(())
}

fn find_attempt_mut<'a>(
    state: &'a mut RuntimeStateV1,
    grant_id: &str,
) -> Result<&'a mut ExecutionAttemptV1, GateErrorV1> {
    state.attempts.iter_mut().find(|attempt| attempt.grant.grant_id_sha256 == grant_id)
        .ok_or_else(|| GateErrorV1::new(GateErrorKindV1::Unauthorized, "grant not found in runtime state"))
}

fn ensure_same_grant(
    stored: &ExecutionGrantV1,
    supplied: &ExecutionGrantV1,
) -> Result<(), GateErrorV1> {
    if stored != supplied {
        Err(GateErrorV1::new(GateErrorKindV1::Unauthorized, "supplied grant differs from persisted grant"))
    } else {
        Ok(())
    }
}

fn validate_unique_texts(values: &[String], field: &str) -> Result<(), GateErrorV1> {
    let mut seen = BTreeSet::new();
    for value in values {
        require_text(value, field)?;
        if !seen.insert(value.as_str()) {
            return Err(GateErrorV1::invalid(format!("{field} contains duplicates")));
        }
    }
    Ok(())
}

fn require_text(value: &str, field: &str) -> Result<(), GateErrorV1> {
    if value.trim().is_empty() {
        Err(GateErrorV1::invalid(format!("{field} must not be empty")))
    } else {
        Ok(())
    }
}

fn validate_nonnegative(value: f64, field: &str) -> Result<(), GateErrorV1> {
    if !value.is_finite() || value < 0.0 {
        Err(GateErrorV1::invalid(format!("{field} must be finite and nonnegative")))
    } else {
        Ok(())
    }
}

fn validate_digest(value: &str, field: &str) -> Result<(), GateErrorV1> {
    let valid = value.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64 && hex.bytes().all(|byte| {
            byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
        })
    });
    if !valid {
        Err(GateErrorV1::invalid(format!(
            "{field} must be sha256: followed by 64 lowercase hexadecimal characters"
        )))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(c: char) -> String { format!("sha256:{}", c.to_string().repeat(64)) }

    fn state() -> RuntimeStateV1 {
        initialize_runtime(
            RuntimeHeadV1 {
                current_audio_sha256: digest('a'),
                registry_snapshot_sha256: digest('b'),
                session_ledger_sha256: digest('c'),
            },
            RemainingBudgetV1 { compute_units: 100.0, latency_ms: 10_000, risk_units: 10.0 },
        ).unwrap()
    }

    fn policy() -> AuthorityPolicyV1 {
        AuthorityPolicyV1 {
            registry_snapshot_sha256: digest('b'),
            operators: vec![RegisteredOperatorV1 {
                operator_id: "formant-local-editor".into(),
                version: "1.0".into(),
                artifact_sha256: digest('d'),
                capabilities: vec!["vocal.local-edit".into(), "vocal.phoneme-edit".into()],
                maximum_scope: MutationScopeV1::Phrase,
            }],
            observers: vec![RegisteredObserverV1 {
                metric_id: "phoneme-error-rate".into(),
                observer_profile_sha256: digest('3'),
                evaluator_artifact_sha256: digest('4'),
                calibration_receipt_sha256: digest('5'),
                independently_qualified: true,
            }],
        }
    }

    fn request() -> MutationRequestV1 {
        MutationRequestV1 {
            grant_id_sha256: digest('e'),
            plan_sha256: digest('f'),
            plan_node_id: "replace-word".into(),
            registry_snapshot_sha256: digest('b'),
            parent_audio_sha256: digest('a'),
            session_ledger_sha256: digest('c'),
            action_sha256: digest('1'),
            operator_id: "formant-local-editor".into(),
            operator_version: "1.0".into(),
            operator_artifact_sha256: digest('d'),
            required_capabilities: vec!["vocal.local-edit".into(), "vocal.phoneme-edit".into()],
            requested_scope: MutationScopeV1::Phrase,
            required_precondition_ids: vec!["alignment.resolved".into()],
            precondition_receipts: vec![PreconditionReceiptV1 {
                precondition_id: "alignment.resolved".into(),
                receipt_sha256: digest('2'),
            }],
            required_observer_metric_ids: vec!["phoneme-error-rate".into()],
            observers: vec![ObserverEvidenceV1 {
                metric_id: "phoneme-error-rate".into(),
                observer_profile_sha256: digest('3'),
                evaluator_artifact_sha256: digest('4'),
                calibration_receipt_sha256: digest('5'),
                independently_qualified: true,
            }],
            user_authorization: UserAuthorizationV1 {
                receipt_sha256: digest('6'),
                plan_sha256: digest('f'),
                plan_node_id: "replace-word".into(),
                parent_audio_sha256: digest('a'),
                action_sha256: digest('1'),
                maximum_scope: MutationScopeV1::Phrase,
                explicit_authorization: true,
            },
            estimated_cost: ResourceBudgetV1 { compute_units: 10.0, latency_ms: 500, risk_units: 0.1 },
        }
    }

    fn dispatch(grant: &ExecutionGrantV1) -> DispatchReceiptV1 {
        DispatchReceiptV1 {
            grant_id_sha256: grant.grant_id_sha256.clone(),
            parent_audio_sha256: grant.parent_audio_sha256.clone(),
            action_sha256: grant.action_sha256.clone(),
            receipt_sha256: digest('7'),
        }
    }

    fn commit(grant: &ExecutionGrantV1) -> CommitReceiptV1 {
        CommitReceiptV1 {
            grant_id_sha256: grant.grant_id_sha256.clone(),
            parent_audio_sha256: grant.parent_audio_sha256.clone(),
            action_sha256: grant.action_sha256.clone(),
            output_audio_sha256: digest('8'),
            execution_receipt_sha256: digest('9'),
        }
    }

    fn authorized() -> (RuntimeStateV1, ExecutionGrantV1) {
        authorize_mutation(&state(), &policy(), &request()).unwrap()
    }

    #[test]
    fn authorizes_dispatches_and_commits_exact_output() {
        let (s, g) = authorized();
        let dispatched = mark_dispatched(&s, &g, &dispatch(&g)).unwrap();
        let committed = commit_execution(&dispatched, &g, &commit(&g)).unwrap();
        assert_eq!(committed.attempts[0].status, AttemptStatusV1::Committed);
        assert_eq!(committed.head.current_audio_sha256, digest('8'));
        assert_eq!(committed.remaining_budget.compute_units, 90.0);
    }

    #[test]
    fn rejects_grant_reuse_after_commit() {
        let (s, g) = authorized();
        let dispatched = mark_dispatched(&s, &g, &dispatch(&g)).unwrap();
        let committed = commit_execution(&dispatched, &g, &commit(&g)).unwrap();
        assert_eq!(mark_dispatched(&committed, &g, &dispatch(&g)).unwrap_err().kind, GateErrorKindV1::DuplicateUse);
    }

    #[test]
    fn rejects_forged_or_changed_grant_fields() {
        let (s, g) = authorized();
        let mut forged = g.clone();
        forged.action_sha256 = digest('0');
        assert_eq!(mark_dispatched(&s, &forged, &dispatch(&forged)).unwrap_err().kind, GateErrorKindV1::Unauthorized);
    }

    #[test]
    fn rejects_stale_parent_registry_and_session() {
        for dimension in 0..3 {
            let mut r = request();
            match dimension {
                0 => { r.parent_audio_sha256 = digest('0'); r.user_authorization.parent_audio_sha256 = digest('0'); }
                1 => r.registry_snapshot_sha256 = digest('0'),
                _ => r.session_ledger_sha256 = digest('0'),
            }
            assert!(authorize_mutation(&state(), &policy(), &r).is_err());
        }
    }

    #[test]
    fn rejects_unregistered_capability_and_global_scope() {
        let mut missing = request();
        missing.required_capabilities.push("vocal.global".into());
        assert_eq!(authorize_mutation(&state(), &policy(), &missing).unwrap_err().kind, GateErrorKindV1::Unauthorized);
        let mut global = request();
        global.requested_scope = MutationScopeV1::Global;
        global.user_authorization.maximum_scope = MutationScopeV1::Global;
        assert_eq!(authorize_mutation(&state(), &policy(), &global).unwrap_err().kind, GateErrorKindV1::Unauthorized);
    }

    #[test]
    fn rejects_authorization_receipt_bound_to_different_plan_or_parent() {
        let mut wrong_plan = request();
        wrong_plan.user_authorization.plan_sha256 = digest('0');
        assert_eq!(
            authorize_mutation(&state(), &policy(), &wrong_plan).unwrap_err().kind,
            GateErrorKindV1::Unauthorized
        );
        let mut wrong_parent = request();
        wrong_parent.user_authorization.parent_audio_sha256 = digest('0');
        assert_eq!(
            authorize_mutation(&state(), &policy(), &wrong_parent).unwrap_err().kind,
            GateErrorKindV1::Unauthorized
        );
    }

    #[test]
    fn observer_must_match_frozen_authority_registry_profile() {
        let mut wrong_profile = request();
        wrong_profile.observers[0].observer_profile_sha256 = digest('0');
        assert_eq!(
            authorize_mutation(&state(), &policy(), &wrong_profile).unwrap_err().kind,
            GateErrorKindV1::Unauthorized
        );
    }

    #[test]
    fn cancel_before_dispatch_closes_grant_and_allows_a_new_one() {
        let (authorized, grant) = authorized();
        let cancelled = cancel_undispatched_authorization(
            &authorized, &grant, &digest('a')
        ).unwrap();
        assert_eq!(cancelled.attempts[0].status, AttemptStatusV1::FailedNoMutation);
        let mut retry = request();
        retry.grant_id_sha256 = digest('c');
        assert!(authorize_mutation(&cancelled, &policy(), &retry).is_ok());
    }

    #[test]
    fn requires_explicit_authorization_bound_to_exact_action_and_plan() {
        let mut absent = request();
        absent.user_authorization.explicit_authorization = false;
        assert!(authorize_mutation(&state(), &policy(), &absent).is_err());
        let mut changed = request();
        changed.user_authorization.action_sha256 = digest('0');
        assert!(authorize_mutation(&state(), &policy(), &changed).is_err());
    }

    #[test]
    fn requires_exact_precondition_and_observer_coverage() {
        let mut no_precondition = request();
        no_precondition.precondition_receipts.clear();
        assert!(authorize_mutation(&state(), &policy(), &no_precondition).is_err());
        let mut extra_observer = request();
        extra_observer.observers.push(extra_observer.observers[0].clone());
        assert!(authorize_mutation(&state(), &policy(), &extra_observer).is_err());
    }

    #[test]
    fn observer_must_be_independent_and_artifact_distinct() {
        let mut shared = request();
        shared.observers[0].evaluator_artifact_sha256 = digest('d');
        assert!(authorize_mutation(&state(), &policy(), &shared).is_err());
        let mut not_qualified = request();
        not_qualified.observers[0].independently_qualified = false;
        assert!(authorize_mutation(&state(), &policy(), &not_qualified).is_err());
    }

    #[test]
    fn reserves_budget_at_authorization_before_dispatch() {
        let (s, _) = authorized();
        assert_eq!(s.remaining_budget.compute_units, 90.0);
        assert_eq!(s.remaining_budget.latency_ms, 9_500);
    }

    #[test]
    fn blocks_new_authorization_during_unknown_outcome() {
        let (s, g) = authorized();
        let dispatched = mark_dispatched(&s, &g, &dispatch(&g)).unwrap();
        let unknown = mark_outcome_unknown(&dispatched, &g).unwrap();
        let mut retry = request();
        retry.grant_id_sha256 = digest('0');
        assert_eq!(authorize_mutation(&unknown, &policy(), &retry).unwrap_err().kind, GateErrorKindV1::OutcomeUnresolved);
    }

    #[test]
    fn reconciles_committed_unknown_output_exactly_once() {
        let (s, g) = authorized();
        let dispatched = mark_dispatched(&s, &g, &dispatch(&g)).unwrap();
        let unknown = mark_outcome_unknown(&dispatched, &g).unwrap();
        let receipt = ReconciliationReceiptV1 {
            grant_id_sha256: g.grant_id_sha256.clone(),
            original_parent_audio_sha256: g.parent_audio_sha256.clone(),
            action_sha256: g.action_sha256.clone(),
            receipt_sha256: digest('a'),
            outcome: ReconciliationOutcomeV1::Committed {
                output_audio_sha256: digest('8'),
                execution_receipt_sha256: digest('9'),
            },
        };
        let reconciled = reconcile_unknown_outcome(&unknown, &g, &receipt).unwrap();
        assert_eq!(reconciled.head.current_audio_sha256, digest('8'));
        assert_eq!(reconcile_unknown_outcome(&reconciled, &g, &receipt).unwrap_err().kind, GateErrorKindV1::InvalidTransition);
    }

    #[test]
    fn reconciliation_no_mutation_permits_safe_retry() {
        let (s, g) = authorized();
        let dispatched = mark_dispatched(&s, &g, &dispatch(&g)).unwrap();
        let unknown = mark_outcome_unknown(&dispatched, &g).unwrap();
        let receipt = ReconciliationReceiptV1 {
            grant_id_sha256: g.grant_id_sha256.clone(),
            original_parent_audio_sha256: g.parent_audio_sha256.clone(),
            action_sha256: g.action_sha256.clone(),
            receipt_sha256: digest('a'),
            outcome: ReconciliationOutcomeV1::NoMutation { failure_receipt_sha256: digest('b') },
        };
        let reconciled = reconcile_unknown_outcome(&unknown, &g, &receipt).unwrap();
        let mut retry = request();
        retry.grant_id_sha256 = digest('c');
        assert!(authorize_mutation(&reconciled, &policy(), &retry).is_ok());
    }

    #[test]
    fn failure_without_no_mutation_proof_remains_unresolved() {
        let (s, g) = authorized();
        let dispatched = mark_dispatched(&s, &g, &dispatch(&g)).unwrap();
        let result = record_definite_failure(&dispatched, &g, &DefiniteFailureReceiptV1 {
            grant_id_sha256: g.grant_id_sha256.clone(),
            parent_audio_sha256: g.parent_audio_sha256.clone(),
            action_sha256: g.action_sha256.clone(),
            failure_receipt_sha256: digest('a'),
            proves_no_mutation: false,
        });
        assert_eq!(result.unwrap_err().kind, GateErrorKindV1::OutcomeUnresolved);
    }

    #[test]
    fn commit_before_dispatch_is_rejected() {
        let (s, g) = authorized();
        assert_eq!(commit_execution(&s, &g, &commit(&g)).unwrap_err().kind, GateErrorKindV1::InvalidTransition);
    }

    #[test]
    fn definite_no_mutation_failure_is_retained_and_retryable() {
        let (s, g) = authorized();
        let dispatched = mark_dispatched(&s, &g, &dispatch(&g)).unwrap();
        let failed = record_definite_failure(&dispatched, &g, &DefiniteFailureReceiptV1 {
            grant_id_sha256: g.grant_id_sha256.clone(),
            parent_audio_sha256: g.parent_audio_sha256.clone(),
            action_sha256: g.action_sha256.clone(),
            failure_receipt_sha256: digest('a'),
            proves_no_mutation: true,
        }).unwrap();
        assert_eq!(failed.attempts[0].status, AttemptStatusV1::FailedNoMutation);
        let mut retry = request();
        retry.grant_id_sha256 = digest('c');
        assert!(authorize_mutation(&failed, &policy(), &retry).is_ok());
    }

    #[test]
    fn wrong_commit_parent_does_not_advance_audio_head() {
        let (s, g) = authorized();
        let dispatched = mark_dispatched(&s, &g, &dispatch(&g)).unwrap();
        let mut bad = commit(&g);
        bad.parent_audio_sha256 = digest('0');
        assert!(commit_execution(&dispatched, &g, &bad).is_err());
        assert_eq!(dispatched.head.current_audio_sha256, digest('a'));
    }

    #[test]
    fn changed_reconciliation_binding_is_rejected() {
        let (s, g) = authorized();
        let dispatched = mark_dispatched(&s, &g, &dispatch(&g)).unwrap();
        let unknown = mark_outcome_unknown(&dispatched, &g).unwrap();
        let bad = ReconciliationReceiptV1 {
            grant_id_sha256: g.grant_id_sha256.clone(),
            original_parent_audio_sha256: digest('0'),
            action_sha256: g.action_sha256.clone(),
            receipt_sha256: digest('a'),
            outcome: ReconciliationOutcomeV1::NoMutation { failure_receipt_sha256: digest('b') },
        };
        assert!(reconcile_unknown_outcome(&unknown, &g, &bad).is_err());
    }

    #[test]
    fn repeated_transitions_cannot_reserve_budget_twice() {
        let (authorized, g) = authorized();
        let dispatched = mark_dispatched(&authorized, &g, &dispatch(&g)).unwrap();
        assert_eq!(dispatched.remaining_budget.compute_units, 90.0);
        let unknown = mark_outcome_unknown(&dispatched, &g).unwrap();
        assert_eq!(unknown.remaining_budget.compute_units, 90.0);
    }
}
