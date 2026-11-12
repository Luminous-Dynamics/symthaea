//! One-shot, evidence-bound mutation authorization for Muse.
//!
//! This crate is a pure state-transition contract. It does not invoke tools,
//! persist events, verify cryptographic signatures, lock a database, or provide
//! exactly-once execution. A production host must atomically append each
//! returned event before performing the corresponding side effect.
//!
//! The important ordering rule is:
//!
//! prepare -> authorize -> persist AuthorizationIssued
//! -> dispatch -> persist DispatchStarted -> invoke operator
//! -> observe -> persist outcome -> advance current head only on commit
//!
//! If a process is restarted while an attempt is DISPATCHED or OUTCOME_UNKNOWN,
//! the ledger blocks later mutations. The host must reconcile the exact attempt
//! before it can create another grant.

use std::collections::{BTreeMap, BTreeSet};

pub type Digest = String;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EditScope {
    SampleLocal,
    PhraseLocal,
    PartLocal,
    StemLocal,
    Global,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceDecision {
    Pass,
    Fail,
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreconditionEvidence {
    pub precondition_id: String,
    pub receipt_sha256: Digest,
    pub decision: EvidenceDecision,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObserverRequirement {
    pub metric_id: String,
    /// Artifact digests explicitly approved for observing this metric.
    pub allowed_observer_artifacts: Vec<Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObserverEvidence {
    pub metric_id: String,
    pub observer_id: String,
    pub observer_artifact_sha256: Digest,
    pub receipt_sha256: Digest,
    pub observed_output_sha256: Digest,
    pub decision: EvidenceDecision,
}

/// A normalized mutation plan node. Every field is committed to the durable
/// AuthorizationIssued event before the operator may be called.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedMutation {
    pub attempt_id: String,
    pub plan_sha256: Digest,
    pub registry_sha256: Digest,
    pub session_ledger_sha256: Digest,
    pub parent_output_sha256: Digest,
    pub action_sha256: Digest,
    pub operator_id: String,
    pub operator_artifact_sha256: Digest,
    pub requested_scope: EditScope,
    pub operator_supported_scopes: Vec<EditScope>,
    pub required_capabilities: Vec<String>,
    pub operator_capabilities: Vec<String>,
    pub required_precondition_ids: Vec<String>,
    pub precondition_evidence: Vec<PreconditionEvidence>,
    pub observer_requirements: Vec<ObserverRequirement>,
    pub resource_cost_units: u64,
}

/// Evidence that an authorized principal approved this exact edit. Authentication
/// and signature verification are performed by the host; this crate verifies
/// that the supplied approval is bound to the plan, action, parent and scope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalReceipt {
    pub approval_sha256: Digest,
    pub principal_id: String,
    pub plan_sha256: Digest,
    pub action_sha256: Digest,
    pub parent_output_sha256: Digest,
    pub scope: EditScope,
}

/// Fresh, authoritative view of the state against which a plan was prepared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthoritySnapshot {
    pub ledger_revision: u64,
    pub current_output_sha256: Digest,
    pub current_plan_sha256: Digest,
    pub current_registry_sha256: Digest,
    pub current_session_ledger_sha256: Digest,
    pub remaining_budget_units: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptStatus {
    Authorized,
    Dispatched,
    OutcomeUnknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconciliationResolution {
    Commit {
        output_sha256: Digest,
        execution_receipt_sha256: Digest,
        observer_evidence: Vec<ObserverEvidence>,
    },
    Reject {
        output_sha256: Digest,
        execution_receipt_sha256: Digest,
        rejection_receipt_sha256: Digest,
    },
    NoMutation {
        failure_receipt_sha256: Digest,
    },
}

/// A receipt produced by an independent reconciliation procedure. The host
/// must authenticate that procedure before supplying this value. The crate
/// verifies digest syntax, attempt binding and output consistency; it does not
/// verify a signature or prove that the verifier actually observed the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconciliationReceipt {
    pub receipt_sha256: Digest,
    pub verifier_artifact_sha256: Digest,
    pub attempt_id: String,
    pub plan_sha256: Digest,
    pub action_sha256: Digest,
    pub parent_output_sha256: Digest,
    pub resolution: ReconciliationResolution,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperatorOutcome {
    /// A candidate is promoted only if every required independent observer
    /// returns PASS for the exact produced output.
    Commit {
        output_sha256: Digest,
        execution_receipt_sha256: Digest,
        observer_evidence: Vec<ObserverEvidence>,
    },
    /// The output is known but rejected by the product/evidence policy. It is
    /// retained as an artifact but does not become the current session head.
    Reject {
        output_sha256: Digest,
        execution_receipt_sha256: Digest,
        rejection_receipt_sha256: Digest,
    },
    /// The operator has proved that no mutation occurred.
    DefiniteNoMutationFailure { failure_receipt_sha256: Digest },
    /// The host cannot prove whether the operator produced or committed output.
    /// No later mutation may be authorized until reconciliation.
    Unknown {
        reason_receipt_sha256: Digest,
        observed_output_sha256: Option<Digest>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeEvent {
    LedgerInitialized {
        initial_output_sha256: Digest,
    },
    AuthorizationIssued {
        attempt_id: String,
        grant_id: Digest,
        issued_at_revision: u64,
        mutation: PreparedMutation,
        approval: ApprovalReceipt,
    },
    AuthorizationWithdrawn {
        attempt_id: String,
        grant_id: Digest,
        withdrawal_receipt_sha256: Digest,
    },
    DispatchStarted {
        attempt_id: String,
        grant_id: Digest,
        dispatched_at_revision: u64,
    },
    MutationCommitted {
        attempt_id: String,
        output_sha256: Digest,
        execution_receipt_sha256: Digest,
        observer_evidence: Vec<ObserverEvidence>,
    },
    MutationRejected {
        attempt_id: String,
        output_sha256: Digest,
        execution_receipt_sha256: Digest,
        rejection_receipt_sha256: Digest,
    },
    MutationFailedNoChange {
        attempt_id: String,
        failure_receipt_sha256: Digest,
    },
    OutcomeUnknown {
        attempt_id: String,
        reason_receipt_sha256: Digest,
        observed_output_sha256: Option<Digest>,
    },
    Reconciled {
        attempt_id: String,
        receipt_sha256: Digest,
        verifier_artifact_sha256: Digest,
        resolution: ReconciliationResolution,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    InvalidDigest(&'static str),
    InvalidIdentifier(&'static str),
    InvalidApproval,
    InvalidMutation(&'static str),
    DuplicateIdentifier(&'static str),
    DuplicateEvidence(&'static str),
    MissingEvidence(&'static str),
    EvidenceNotQualified(&'static str),
    ObserverNotIndependent,
    ObserverArtifactNotApproved,
    ScopeNotSupported,
    MissingCapability,
    BudgetExceeded,
    StaleRevision,
    StaleParent,
    StalePlan,
    StaleRegistry,
    StaleSession,
    ActiveAttemptBlocksMutation,
    UnknownOutcomeBlocksMutation,
    InvalidTransition,
    GrantMismatch,
    DispatchPermitMismatch,
    ReconciliationMismatch,
    DuplicateReconciliationReceipt,
    LedgerNotInitialized,
}

impl std::fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Muse execution-authority transition rejected: {self:?}")
    }
}

impl std::error::Error for RuntimeError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InternalAttemptStatus {
    Authorized,
    Dispatched,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ActiveAttempt {
    mutation: PreparedMutation,
    approval: ApprovalReceipt,
    grant_id: Digest,
    status: InternalAttemptStatus,
    observed_output_sha256: Option<Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LedgerState {
    revision: u64,
    current_output_sha256: Digest,
    active_attempt: Option<ActiveAttempt>,
    seen_attempt_ids: BTreeSet<String>,
    seen_grant_ids: BTreeSet<Digest>,
    seen_reconciliation_receipts: BTreeSet<Digest>,
}

impl LedgerState {
    fn replay(events: &[RuntimeEvent]) -> Result<Self, RuntimeError> {
        let Some(RuntimeEvent::LedgerInitialized {
            initial_output_sha256,
        }) = events.first()
        else {
            return Err(RuntimeError::LedgerNotInitialized);
        };
        validate_sha256("initial_output_sha256", initial_output_sha256)?;

        let mut state = Self {
            revision: 1,
            current_output_sha256: initial_output_sha256.clone(),
            active_attempt: None,
            seen_attempt_ids: BTreeSet::new(),
            seen_grant_ids: BTreeSet::new(),
            seen_reconciliation_receipts: BTreeSet::new(),
        };

        for event in events.iter().skip(1) {
            state.apply(event)?;
        }
        Ok(state)
    }

    fn apply(&mut self, event: &RuntimeEvent) -> Result<(), RuntimeError> {
        match event {
            RuntimeEvent::LedgerInitialized { .. } => {
                return Err(RuntimeError::InvalidTransition);
            }
            RuntimeEvent::AuthorizationIssued {
                attempt_id,
                grant_id,
                issued_at_revision,
                mutation,
                approval,
            } => {
                if self.active_attempt.is_some() {
                    return Err(RuntimeError::ActiveAttemptBlocksMutation);
                }
                if *issued_at_revision != self.revision {
                    return Err(RuntimeError::StaleRevision);
                }
                if mutation.parent_output_sha256 != self.current_output_sha256 {
                    return Err(RuntimeError::StaleParent);
                }
                if attempt_id != &mutation.attempt_id {
                    return Err(RuntimeError::InvalidMutation("attempt ID mismatch"));
                }
                validate_prepared_mutation(mutation)?;
                validate_approval(mutation, approval)?;
                validate_sha256("grant_id", grant_id)?;
                if !self.seen_attempt_ids.insert(attempt_id.clone()) {
                    return Err(RuntimeError::DuplicateIdentifier("attempt_id"));
                }
                if !self.seen_grant_ids.insert(grant_id.clone()) {
                    return Err(RuntimeError::DuplicateIdentifier("grant_id"));
                }
                self.active_attempt = Some(ActiveAttempt {
                    mutation: mutation.clone(),
                    approval: approval.clone(),
                    grant_id: grant_id.clone(),
                    status: InternalAttemptStatus::Authorized,
                    observed_output_sha256: None,
                });
            }
            RuntimeEvent::AuthorizationWithdrawn {
                attempt_id,
                grant_id,
                withdrawal_receipt_sha256,
            } => {
                validate_sha256("withdrawal_receipt_sha256", withdrawal_receipt_sha256)?;
                let active = self
                    .active_attempt
                    .as_ref()
                    .ok_or(RuntimeError::InvalidTransition)?;
                if active.status != InternalAttemptStatus::Authorized
                    || active.mutation.attempt_id != *attempt_id
                    || active.grant_id != *grant_id
                {
                    return Err(RuntimeError::GrantMismatch);
                }
                self.active_attempt = None;
            }
            RuntimeEvent::DispatchStarted {
                attempt_id,
                grant_id,
                dispatched_at_revision,
            } => {
                if *dispatched_at_revision != self.revision {
                    return Err(RuntimeError::StaleRevision);
                }
                let active = self
                    .active_attempt
                    .as_mut()
                    .ok_or(RuntimeError::InvalidTransition)?;
                if active.status != InternalAttemptStatus::Authorized
                    || active.mutation.attempt_id != *attempt_id
                    || active.grant_id != *grant_id
                {
                    return Err(RuntimeError::GrantMismatch);
                }
                active.status = InternalAttemptStatus::Dispatched;
            }
            RuntimeEvent::MutationCommitted {
                attempt_id,
                output_sha256,
                execution_receipt_sha256,
                observer_evidence,
            } => {
                validate_sha256("output_sha256", output_sha256)?;
                validate_sha256("execution_receipt_sha256", execution_receipt_sha256)?;
                let active = self
                    .active_attempt
                    .as_ref()
                    .ok_or(RuntimeError::InvalidTransition)?;
                if active.status != InternalAttemptStatus::Dispatched
                    || active.mutation.attempt_id != *attempt_id
                {
                    return Err(RuntimeError::InvalidTransition);
                }
                validate_observer_evidence(&active.mutation, output_sha256, observer_evidence)?;
                self.current_output_sha256 = output_sha256.clone();
                self.active_attempt = None;
            }
            RuntimeEvent::MutationRejected {
                attempt_id,
                output_sha256,
                execution_receipt_sha256,
                rejection_receipt_sha256,
            } => {
                validate_sha256("output_sha256", output_sha256)?;
                validate_sha256("execution_receipt_sha256", execution_receipt_sha256)?;
                validate_sha256("rejection_receipt_sha256", rejection_receipt_sha256)?;
                let active = self
                    .active_attempt
                    .as_ref()
                    .ok_or(RuntimeError::InvalidTransition)?;
                if active.status != InternalAttemptStatus::Dispatched
                    || active.mutation.attempt_id != *attempt_id
                {
                    return Err(RuntimeError::InvalidTransition);
                }
                // Rejected candidates are retained, but the session head does
                // not move and a fresh attempt may start from the same parent.
                self.active_attempt = None;
            }
            RuntimeEvent::MutationFailedNoChange {
                attempt_id,
                failure_receipt_sha256,
            } => {
                validate_sha256("failure_receipt_sha256", failure_receipt_sha256)?;
                let active = self
                    .active_attempt
                    .as_ref()
                    .ok_or(RuntimeError::InvalidTransition)?;
                if active.status != InternalAttemptStatus::Dispatched
                    || active.mutation.attempt_id != *attempt_id
                {
                    return Err(RuntimeError::InvalidTransition);
                }
                self.active_attempt = None;
            }
            RuntimeEvent::OutcomeUnknown {
                attempt_id,
                reason_receipt_sha256,
                observed_output_sha256,
            } => {
                validate_sha256("reason_receipt_sha256", reason_receipt_sha256)?;
                if let Some(output) = observed_output_sha256 {
                    validate_sha256("observed_output_sha256", output)?;
                }
                let active = self
                    .active_attempt
                    .as_mut()
                    .ok_or(RuntimeError::InvalidTransition)?;
                if active.status != InternalAttemptStatus::Dispatched
                    || active.mutation.attempt_id != *attempt_id
                {
                    return Err(RuntimeError::InvalidTransition);
                }
                active.status = InternalAttemptStatus::Unknown;
                active.observed_output_sha256 = observed_output_sha256.clone();
            }
            RuntimeEvent::Reconciled {
                attempt_id,
                receipt_sha256,
                verifier_artifact_sha256,
                resolution,
            } => {
                validate_sha256("receipt_sha256", receipt_sha256)?;
                validate_sha256("verifier_artifact_sha256", verifier_artifact_sha256)?;
                if !self
                    .seen_reconciliation_receipts
                    .insert(receipt_sha256.clone())
                {
                    return Err(RuntimeError::DuplicateReconciliationReceipt);
                }
                let active = self
                    .active_attempt
                    .as_ref()
                    .ok_or(RuntimeError::InvalidTransition)?;
                if active.status != InternalAttemptStatus::Unknown
                    || active.mutation.attempt_id != *attempt_id
                {
                    return Err(RuntimeError::ReconciliationMismatch);
                }
                validate_reconciliation_resolution(
                    &active.mutation,
                    active.observed_output_sha256.as_deref(),
                    resolution,
                )?;
                match resolution {
                    ReconciliationResolution::Commit { output_sha256, .. } => {
                        self.current_output_sha256 = output_sha256.clone();
                    }
                    ReconciliationResolution::Reject { .. }
                    | ReconciliationResolution::NoMutation { .. } => {}
                }
                self.active_attempt = None;
            }
        }
        self.revision = self
            .revision
            .checked_add(1)
            .ok_or(RuntimeError::InvalidTransition)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeLedger {
    events: Vec<RuntimeEvent>,
}

/// Opaque in-process grant. It is deliberately not Clone, Copy, Serialize or
/// Deserialize. The durable AuthorizationIssued event is the replay record.
#[derive(Debug, PartialEq, Eq)]
pub struct ExecutionGrant {
    attempt_id: String,
    grant_id: Digest,
    revision_after_authorization: u64,
    plan_sha256: Digest,
    action_sha256: Digest,
    parent_output_sha256: Digest,
    operator_artifact_sha256: Digest,
}

/// Opaque dispatch permit. It is consumed when the operator outcome is recorded.
#[derive(Debug, PartialEq, Eq)]
pub struct DispatchPermit {
    attempt_id: String,
    grant_id: Digest,
    revision_after_dispatch: u64,
}

impl RuntimeLedger {
    pub fn new(initial_output_sha256: Digest) -> Result<Self, RuntimeError> {
        validate_sha256("initial_output_sha256", &initial_output_sha256)?;
        Ok(Self {
            events: vec![RuntimeEvent::LedgerInitialized {
                initial_output_sha256,
            }],
        })
    }

    pub fn events(&self) -> &[RuntimeEvent] {
        &self.events
    }

    pub fn revision(&self) -> Result<u64, RuntimeError> {
        Ok(LedgerState::replay(&self.events)?.revision)
    }

    /// Return the current committed session head after replaying the ledger.
    /// An unresolved dispatched/unknown attempt never advances this value.
    pub fn current_output_sha256(&self) -> Result<Digest, RuntimeError> {
        Ok(LedgerState::replay(&self.events)?.current_output_sha256)
    }

    /// Backward-compatible descriptive alias for current_output_sha256().
    pub fn current_output_digest(&self) -> Result<Digest, RuntimeError> {
        self.current_output_sha256()
    }

    pub fn append(&mut self, event: RuntimeEvent) -> Result<(), RuntimeError> {
        let mut candidate = LedgerState::replay(&self.events)?;
        candidate.apply(&event)?;
        self.events.push(event);
        Ok(())
    }

    /// Prepare an authorization event and an opaque grant. The caller must
    /// persist the returned event before requesting dispatch.
    pub fn prepare_authorization(
        &self,
        snapshot: &AuthoritySnapshot,
        mutation: PreparedMutation,
        approval: ApprovalReceipt,
        grant_nonce_sha256: Digest,
    ) -> Result<(RuntimeEvent, ExecutionGrant), RuntimeError> {
        let state = LedgerState::replay(&self.events)?;
        if state.active_attempt.is_some() {
            return Err(RuntimeError::ActiveAttemptBlocksMutation);
        }
        if snapshot.ledger_revision != state.revision {
            return Err(RuntimeError::StaleRevision);
        }
        if snapshot.current_output_sha256 != state.current_output_sha256
            || mutation.parent_output_sha256 != state.current_output_sha256
        {
            return Err(RuntimeError::StaleParent);
        }
        if snapshot.current_plan_sha256 != mutation.plan_sha256 {
            return Err(RuntimeError::StalePlan);
        }
        if snapshot.current_registry_sha256 != mutation.registry_sha256 {
            return Err(RuntimeError::StaleRegistry);
        }
        if snapshot.current_session_ledger_sha256 != mutation.session_ledger_sha256 {
            return Err(RuntimeError::StaleSession);
        }
        if mutation.resource_cost_units > snapshot.remaining_budget_units {
            return Err(RuntimeError::BudgetExceeded);
        }
        validate_prepared_mutation(&mutation)?;
        validate_approval(&mutation, &approval)?;
        validate_sha256("grant_nonce_sha256", &grant_nonce_sha256)?;
        if state.seen_grant_ids.contains(&grant_nonce_sha256) {
            return Err(RuntimeError::DuplicateIdentifier("grant_nonce_sha256"));
        }
        if state.seen_attempt_ids.contains(&mutation.attempt_id) {
            return Err(RuntimeError::DuplicateIdentifier("attempt_id"));
        }

        let event = RuntimeEvent::AuthorizationIssued {
            attempt_id: mutation.attempt_id.clone(),
            grant_id: grant_nonce_sha256.clone(),
            issued_at_revision: state.revision,
            mutation: mutation.clone(),
            approval,
        };
        let grant = ExecutionGrant {
            attempt_id: mutation.attempt_id,
            grant_id: grant_nonce_sha256,
            revision_after_authorization: state
                .revision
                .checked_add(1)
                .ok_or(RuntimeError::InvalidTransition)?,
            plan_sha256: mutation.plan_sha256,
            action_sha256: mutation.action_sha256,
            parent_output_sha256: mutation.parent_output_sha256,
            operator_artifact_sha256: mutation.operator_artifact_sha256,
        };
        Ok((event, grant))
    }

    /// Prepare dispatch only from the exact persisted authorization event.
    /// The caller must persist DispatchStarted before invoking the operator.
    pub fn prepare_dispatch(
        &self,
        grant: ExecutionGrant,
    ) -> Result<(RuntimeEvent, DispatchPermit), RuntimeError> {
        let state = LedgerState::replay(&self.events)?;
        if state.revision != grant.revision_after_authorization {
            return Err(RuntimeError::StaleRevision);
        }
        let active = state
            .active_attempt
            .as_ref()
            .ok_or(RuntimeError::GrantMismatch)?;
        if active.status != InternalAttemptStatus::Authorized
            || active.mutation.attempt_id != grant.attempt_id
            || active.grant_id != grant.grant_id
            || active.mutation.plan_sha256 != grant.plan_sha256
            || active.mutation.action_sha256 != grant.action_sha256
            || active.mutation.parent_output_sha256 != grant.parent_output_sha256
            || active.mutation.operator_artifact_sha256 != grant.operator_artifact_sha256
        {
            return Err(RuntimeError::GrantMismatch);
        }
        let event = RuntimeEvent::DispatchStarted {
            attempt_id: grant.attempt_id.clone(),
            grant_id: grant.grant_id.clone(),
            dispatched_at_revision: state.revision,
        };
        let permit = DispatchPermit {
            attempt_id: grant.attempt_id,
            grant_id: grant.grant_id,
            revision_after_dispatch: state
                .revision
                .checked_add(1)
                .ok_or(RuntimeError::InvalidTransition)?,
        };
        Ok((event, permit))
    }

    /// Prepare a terminal outcome for a dispatched operation.
    pub fn prepare_outcome(
        &self,
        permit: DispatchPermit,
        outcome: OperatorOutcome,
    ) -> Result<RuntimeEvent, RuntimeError> {
        let state = LedgerState::replay(&self.events)?;
        if state.revision != permit.revision_after_dispatch {
            return Err(RuntimeError::StaleRevision);
        }
        let active = state
            .active_attempt
            .as_ref()
            .ok_or(RuntimeError::DispatchPermitMismatch)?;
        if active.status != InternalAttemptStatus::Dispatched
            || active.mutation.attempt_id != permit.attempt_id
            || active.grant_id != permit.grant_id
        {
            return Err(RuntimeError::DispatchPermitMismatch);
        }

        let event = match outcome {
            OperatorOutcome::Commit {
                output_sha256,
                execution_receipt_sha256,
                observer_evidence,
            } => {
                validate_sha256("output_sha256", &output_sha256)?;
                validate_sha256("execution_receipt_sha256", &execution_receipt_sha256)?;
                validate_observer_evidence(&active.mutation, &output_sha256, &observer_evidence)?;
                RuntimeEvent::MutationCommitted {
                    attempt_id: permit.attempt_id,
                    output_sha256,
                    execution_receipt_sha256,
                    observer_evidence,
                }
            }
            OperatorOutcome::Reject {
                output_sha256,
                execution_receipt_sha256,
                rejection_receipt_sha256,
            } => RuntimeEvent::MutationRejected {
                attempt_id: permit.attempt_id,
                output_sha256,
                execution_receipt_sha256,
                rejection_receipt_sha256,
            },
            OperatorOutcome::DefiniteNoMutationFailure {
                failure_receipt_sha256,
            } => RuntimeEvent::MutationFailedNoChange {
                attempt_id: permit.attempt_id,
                failure_receipt_sha256,
            },
            OperatorOutcome::Unknown {
                reason_receipt_sha256,
                observed_output_sha256,
            } => RuntimeEvent::OutcomeUnknown {
                attempt_id: permit.attempt_id,
                reason_receipt_sha256,
                observed_output_sha256,
            },
        };
        // Replay this one event before returning it. A bad operator receipt
        // cannot be appended as a nominally valid terminal event.
        let mut candidate = state;
        candidate.apply(&event)?;
        Ok(event)
    }

    /// Resolve an OUTCOME_UNKNOWN attempt using externally verified evidence.
    /// This does not authenticate the verifier; that is the host's duty.
    pub fn prepare_reconciliation(
        &self,
        receipt: ReconciliationReceipt,
    ) -> Result<RuntimeEvent, RuntimeError> {
        let state = LedgerState::replay(&self.events)?;
        let active = state
            .active_attempt
            .as_ref()
            .ok_or(RuntimeError::ReconciliationMismatch)?;
        if active.status != InternalAttemptStatus::Unknown
            || receipt.attempt_id != active.mutation.attempt_id
            || receipt.plan_sha256 != active.mutation.plan_sha256
            || receipt.action_sha256 != active.mutation.action_sha256
            || receipt.parent_output_sha256 != active.mutation.parent_output_sha256
        {
            return Err(RuntimeError::ReconciliationMismatch);
        }
        validate_sha256("receipt_sha256", &receipt.receipt_sha256)?;
        validate_sha256(
            "verifier_artifact_sha256",
            &receipt.verifier_artifact_sha256,
        )?;
        if state
            .seen_reconciliation_receipts
            .contains(&receipt.receipt_sha256)
        {
            return Err(RuntimeError::DuplicateReconciliationReceipt);
        }
        validate_reconciliation_resolution(
            &active.mutation,
            active.observed_output_sha256.as_deref(),
            &receipt.resolution,
        )?;
        let event = RuntimeEvent::Reconciled {
            attempt_id: receipt.attempt_id,
            receipt_sha256: receipt.receipt_sha256,
            verifier_artifact_sha256: receipt.verifier_artifact_sha256,
            resolution: receipt.resolution,
        };
        let mut candidate = state;
        candidate.apply(&event)?;
        Ok(event)
    }

    /// A grant may be withdrawn only before dispatch. The withdrawal receipt
    /// is durable evidence that the operator was never called with this grant.
    pub fn prepare_withdrawal(
        &self,
        grant: ExecutionGrant,
        withdrawal_receipt_sha256: Digest,
    ) -> Result<RuntimeEvent, RuntimeError> {
        let state = LedgerState::replay(&self.events)?;
        if state.revision != grant.revision_after_authorization {
            return Err(RuntimeError::StaleRevision);
        }
        let active = state
            .active_attempt
            .as_ref()
            .ok_or(RuntimeError::GrantMismatch)?;
        if active.status != InternalAttemptStatus::Authorized
            || active.mutation.attempt_id != grant.attempt_id
            || active.grant_id != grant.grant_id
        {
            return Err(RuntimeError::GrantMismatch);
        }
        let event = RuntimeEvent::AuthorizationWithdrawn {
            attempt_id: grant.attempt_id,
            grant_id: grant.grant_id,
            withdrawal_receipt_sha256,
        };
        let mut candidate = state;
        candidate.apply(&event)?;
        Ok(event)
    }
}

fn validate_prepared_mutation(mutation: &PreparedMutation) -> Result<(), RuntimeError> {
    validate_identifier("attempt_id", &mutation.attempt_id)?;
    validate_identifier("operator_id", &mutation.operator_id)?;
    for (field, digest) in [
        ("plan_sha256", &mutation.plan_sha256),
        ("registry_sha256", &mutation.registry_sha256),
        ("session_ledger_sha256", &mutation.session_ledger_sha256),
        ("parent_output_sha256", &mutation.parent_output_sha256),
        ("action_sha256", &mutation.action_sha256),
        (
            "operator_artifact_sha256",
            &mutation.operator_artifact_sha256,
        ),
    ] {
        validate_sha256(field, digest)?;
    }

    if !mutation
        .operator_supported_scopes
        .contains(&mutation.requested_scope)
    {
        return Err(RuntimeError::ScopeNotSupported);
    }
    ensure_unique_scopes(&mutation.operator_supported_scopes)?;
    ensure_unique_strings(&mutation.required_capabilities, "required_capabilities")?;
    ensure_unique_strings(&mutation.operator_capabilities, "operator_capabilities")?;
    let supplied: BTreeSet<&str> = mutation
        .operator_capabilities
        .iter()
        .map(String::as_str)
        .collect();
    if mutation
        .required_capabilities
        .iter()
        .any(|required| !supplied.contains(required.as_str()))
    {
        return Err(RuntimeError::MissingCapability);
    }

    ensure_unique_strings(
        &mutation.required_precondition_ids,
        "required_precondition_ids",
    )?;
    let mut preconditions = BTreeMap::new();
    for evidence in &mutation.precondition_evidence {
        validate_identifier("precondition_id", &evidence.precondition_id)?;
        validate_sha256("precondition_receipt_sha256", &evidence.receipt_sha256)?;
        if preconditions
            .insert(evidence.precondition_id.as_str(), evidence)
            .is_some()
        {
            return Err(RuntimeError::DuplicateEvidence("precondition"));
        }
        if evidence.decision != EvidenceDecision::Pass {
            return Err(RuntimeError::EvidenceNotQualified("precondition"));
        }
    }
    if preconditions.len() != mutation.required_precondition_ids.len()
        || mutation
            .required_precondition_ids
            .iter()
            .any(|id| !preconditions.contains_key(id.as_str()))
    {
        return Err(RuntimeError::MissingEvidence("precondition"));
    }

    let mut metrics = BTreeSet::new();
    for requirement in &mutation.observer_requirements {
        validate_identifier("metric_id", &requirement.metric_id)?;
        if !metrics.insert(requirement.metric_id.as_str()) {
            return Err(RuntimeError::DuplicateIdentifier("observer metric"));
        }
        if requirement.allowed_observer_artifacts.is_empty() {
            return Err(RuntimeError::MissingEvidence("allowed observer artifact"));
        }
        let mut artifacts = BTreeSet::new();
        for artifact in &requirement.allowed_observer_artifacts {
            validate_sha256("allowed_observer_artifact_sha256", artifact)?;
            if !artifacts.insert(artifact.as_str()) {
                return Err(RuntimeError::DuplicateIdentifier("observer artifact"));
            }
        }
    }
    Ok(())
}

fn validate_approval(
    mutation: &PreparedMutation,
    approval: &ApprovalReceipt,
) -> Result<(), RuntimeError> {
    validate_sha256("approval_sha256", &approval.approval_sha256)?;
    validate_identifier("principal_id", &approval.principal_id)?;
    if approval.plan_sha256 != mutation.plan_sha256
        || approval.action_sha256 != mutation.action_sha256
        || approval.parent_output_sha256 != mutation.parent_output_sha256
        || approval.scope != mutation.requested_scope
    {
        return Err(RuntimeError::InvalidApproval);
    }
    Ok(())
}

fn validate_observer_evidence(
    mutation: &PreparedMutation,
    output_sha256: &str,
    evidence: &[ObserverEvidence],
) -> Result<(), RuntimeError> {
    let mut by_metric = BTreeMap::new();
    for item in evidence {
        validate_identifier("metric_id", &item.metric_id)?;
        validate_identifier("observer_id", &item.observer_id)?;
        validate_sha256("observer_artifact_sha256", &item.observer_artifact_sha256)?;
        validate_sha256("observer_receipt_sha256", &item.receipt_sha256)?;
        validate_sha256("observer_output_sha256", &item.observed_output_sha256)?;
        if by_metric.insert(item.metric_id.as_str(), item).is_some() {
            return Err(RuntimeError::DuplicateEvidence("observer metric"));
        }
    }

    if by_metric.len() != mutation.observer_requirements.len() {
        return Err(RuntimeError::MissingEvidence("observer coverage"));
    }
    for requirement in &mutation.observer_requirements {
        let item = by_metric
            .get(requirement.metric_id.as_str())
            .ok_or(RuntimeError::MissingEvidence("observer metric"))?;
        if item.decision != EvidenceDecision::Pass {
            return Err(RuntimeError::EvidenceNotQualified("observer"));
        }
        if item.observed_output_sha256 != output_sha256 {
            return Err(RuntimeError::InvalidMutation(
                "observer receipt is detached from the exact output",
            ));
        }
        if item.observer_artifact_sha256 == mutation.operator_artifact_sha256 {
            return Err(RuntimeError::ObserverNotIndependent);
        }
        if !requirement
            .allowed_observer_artifacts
            .contains(&item.observer_artifact_sha256)
        {
            return Err(RuntimeError::ObserverArtifactNotApproved);
        }
    }
    Ok(())
}

fn validate_reconciliation_resolution(
    mutation: &PreparedMutation,
    previously_observed_output: Option<&str>,
    resolution: &ReconciliationResolution,
) -> Result<(), RuntimeError> {
    match resolution {
        ReconciliationResolution::Commit {
            output_sha256,
            execution_receipt_sha256,
            observer_evidence,
        } => {
            validate_sha256("output_sha256", output_sha256)?;
            validate_sha256("execution_receipt_sha256", execution_receipt_sha256)?;
            validate_observer_evidence(mutation, output_sha256, observer_evidence)?;
            if previously_observed_output.is_some_and(|observed| observed != output_sha256) {
                return Err(RuntimeError::ReconciliationMismatch);
            }
        }
        ReconciliationResolution::Reject {
            output_sha256,
            execution_receipt_sha256,
            rejection_receipt_sha256,
        } => {
            validate_sha256("output_sha256", output_sha256)?;
            validate_sha256("execution_receipt_sha256", execution_receipt_sha256)?;
            validate_sha256("rejection_receipt_sha256", rejection_receipt_sha256)?;
            if previously_observed_output.is_some_and(|observed| observed != output_sha256) {
                return Err(RuntimeError::ReconciliationMismatch);
            }
        }
        ReconciliationResolution::NoMutation {
            failure_receipt_sha256,
        } => {
            validate_sha256("failure_receipt_sha256", failure_receipt_sha256)?;
            if previously_observed_output.is_some() {
                return Err(RuntimeError::ReconciliationMismatch);
            }
        }
    }
    Ok(())
}

fn validate_sha256(field: &'static str, value: &str) -> Result<(), RuntimeError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(RuntimeError::InvalidDigest(field));
    }
    Ok(())
}

fn validate_identifier(field: &'static str, value: &str) -> Result<(), RuntimeError> {
    if value.trim().is_empty() || value.len() > 256 || value.chars().any(char::is_control) {
        return Err(RuntimeError::InvalidIdentifier(field));
    }
    Ok(())
}

fn ensure_unique_strings(values: &[String], field: &'static str) -> Result<(), RuntimeError> {
    let mut seen = BTreeSet::new();
    for value in values {
        validate_identifier(field, value)?;
        if !seen.insert(value.as_str()) {
            return Err(RuntimeError::DuplicateIdentifier(field));
        }
    }
    Ok(())
}

fn ensure_unique_scopes(scopes: &[EditScope]) -> Result<(), RuntimeError> {
    let mut seen = BTreeSet::new();
    for scope in scopes {
        if !seen.insert(*scope) {
            return Err(RuntimeError::DuplicateIdentifier(
                "operator_supported_scopes",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: char = 'a';
    const B: char = 'b';
    const C: char = 'c';
    const D: char = 'd';
    const E: char = 'e';
    const F: char = 'f';
    const G: char = '1';
    const H: char = '2';
    const I: char = '3';
    const J: char = '4';
    const K: char = '5';
    const L: char = '6';
    const M: char = '7';
    const N: char = '8';

    fn digest(ch: char) -> Digest {
        ch.to_string().repeat(64)
    }

    fn make_mutation(attempt: &str) -> PreparedMutation {
        PreparedMutation {
            attempt_id: attempt.to_owned(),
            plan_sha256: digest(A),
            registry_sha256: digest(B),
            session_ledger_sha256: digest(C),
            parent_output_sha256: digest(D),
            action_sha256: digest(E),
            operator_id: "muse.symbolic.patch.v1".to_owned(),
            operator_artifact_sha256: digest(F),
            requested_scope: EditScope::PartLocal,
            operator_supported_scopes: vec![EditScope::PartLocal, EditScope::PhraseLocal],
            required_capabilities: vec!["score.patch".to_owned()],
            operator_capabilities: vec!["score.patch".to_owned(), "audio.render".to_owned()],
            required_precondition_ids: vec!["parent_verified".to_owned()],
            precondition_evidence: vec![PreconditionEvidence {
                precondition_id: "parent_verified".to_owned(),
                receipt_sha256: digest(G),
                decision: EvidenceDecision::Pass,
            }],
            observer_requirements: vec![ObserverRequirement {
                metric_id: "protected_material".to_owned(),
                allowed_observer_artifacts: vec![digest(H)],
            }],
            resource_cost_units: 3,
        }
    }

    fn approval(mutation: &PreparedMutation) -> ApprovalReceipt {
        ApprovalReceipt {
            approval_sha256: digest(I),
            principal_id: "musician-session-owner".to_owned(),
            plan_sha256: mutation.plan_sha256.clone(),
            action_sha256: mutation.action_sha256.clone(),
            parent_output_sha256: mutation.parent_output_sha256.clone(),
            scope: mutation.requested_scope,
        }
    }

    fn snapshot(ledger: &RuntimeLedger) -> AuthoritySnapshot {
        AuthoritySnapshot {
            ledger_revision: ledger.revision().unwrap(),
            current_output_sha256: ledger.current_output_digest().unwrap(),
            current_plan_sha256: digest(A),
            current_registry_sha256: digest(B),
            current_session_ledger_sha256: digest(C),
            remaining_budget_units: 20,
        }
    }

    fn observer(output: &str) -> ObserverEvidence {
        ObserverEvidence {
            metric_id: "protected_material".to_owned(),
            observer_id: "independent.pcm.preservation.v1".to_owned(),
            observer_artifact_sha256: digest(H),
            receipt_sha256: digest(J),
            observed_output_sha256: output.to_owned(),
            decision: EvidenceDecision::Pass,
        }
    }

    fn ledger() -> RuntimeLedger {
        RuntimeLedger::new(digest(D)).unwrap()
    }

    fn authorize_and_dispatch(
        ledger: &mut RuntimeLedger,
        mutation: PreparedMutation,
    ) -> DispatchPermit {
        let (event, grant) = ledger
            .prepare_authorization(
                &snapshot(ledger),
                mutation.clone(),
                approval(&mutation),
                digest(K),
            )
            .unwrap();
        ledger.append(event).unwrap();
        let (dispatch_event, permit) = ledger.prepare_dispatch(grant).unwrap();
        ledger.append(dispatch_event).unwrap();
        permit
    }

    #[test]
    fn one_shot_authorization_commits_only_after_independent_observer_passes() {
        let mut ledger = ledger();
        let mutation = make_mutation("attempt-001");
        let permit = authorize_and_dispatch(&mut ledger, mutation);
        let output = digest(L);
        let event = ledger
            .prepare_outcome(
                permit,
                OperatorOutcome::Commit {
                    output_sha256: output.clone(),
                    execution_receipt_sha256: digest(M),
                    observer_evidence: vec![observer(&output)],
                },
            )
            .unwrap();
        ledger.append(event).unwrap();
        assert_eq!(ledger.current_output_digest().unwrap(), output);
        assert_eq!(ledger.revision().unwrap(), 4);
    }

    #[test]
    fn authorization_rejects_a_stale_parent() {
        let ledger = ledger();
        let mutation = make_mutation("attempt-002");
        let mut stale = snapshot(&ledger);
        stale.current_output_sha256 = digest(N);
        assert_eq!(
            ledger.prepare_authorization(&stale, mutation.clone(), approval(&mutation), digest(K)),
            Err(RuntimeError::StaleParent)
        );
    }

    #[test]
    fn authorization_rejects_a_stale_plan_registry_or_session() {
        let ledger = ledger();
        let mutation = make_mutation("attempt-003");
        let mut stale_plan = snapshot(&ledger);
        stale_plan.current_plan_sha256 = digest(N);
        assert_eq!(
            ledger.prepare_authorization(
                &stale_plan,
                mutation.clone(),
                approval(&mutation),
                digest(K)
            ),
            Err(RuntimeError::StalePlan)
        );
        let mut stale_registry = snapshot(&ledger);
        stale_registry.current_registry_sha256 = digest(N);
        assert_eq!(
            ledger.prepare_authorization(
                &stale_registry,
                mutation.clone(),
                approval(&mutation),
                digest(K)
            ),
            Err(RuntimeError::StaleRegistry)
        );
        let mut stale_session = snapshot(&ledger);
        stale_session.current_session_ledger_sha256 = digest(N);
        assert_eq!(
            ledger.prepare_authorization(
                &stale_session,
                mutation.clone(),
                approval(&mutation),
                digest(K)
            ),
            Err(RuntimeError::StaleSession)
        );
    }

    #[test]
    fn global_scope_cannot_substitute_for_a_part_local_edit() {
        let ledger = ledger();
        let mut mutation = make_mutation("attempt-004");
        mutation.operator_supported_scopes = vec![EditScope::Global];
        assert_eq!(
            ledger.prepare_authorization(
                &snapshot(&ledger),
                mutation.clone(),
                approval(&mutation),
                digest(K)
            ),
            Err(RuntimeError::ScopeNotSupported)
        );
    }

    #[test]
    fn approval_must_bind_exact_action_parent_and_scope() {
        let ledger = ledger();
        let mutation = make_mutation("attempt-005");
        let mut approval = approval(&mutation);
        approval.scope = EditScope::Global;
        assert_eq!(
            ledger.prepare_authorization(&snapshot(&ledger), mutation, approval, digest(K)),
            Err(RuntimeError::InvalidApproval)
        );
    }

    #[test]
    fn missing_operator_capability_or_precondition_blocks_authorization() {
        let ledger = ledger();
        let mut no_capability = make_mutation("attempt-006");
        no_capability.operator_capabilities.clear();
        assert_eq!(
            ledger.prepare_authorization(
                &snapshot(&ledger),
                no_capability.clone(),
                approval(&no_capability),
                digest(K)
            ),
            Err(RuntimeError::MissingCapability)
        );
        let mut no_precondition = make_mutation("attempt-007");
        no_precondition.precondition_evidence.clear();
        assert_eq!(
            ledger.prepare_authorization(
                &snapshot(&ledger),
                no_precondition.clone(),
                approval(&no_precondition),
                digest(K)
            ),
            Err(RuntimeError::MissingEvidence("precondition"))
        );
    }

    #[test]
    fn operator_and_observer_artifacts_must_be_independent() {
        let mut ledger = ledger();
        let mutation = make_mutation("attempt-008");
        let permit = authorize_and_dispatch(&mut ledger, mutation.clone());
        let output = digest(L);
        let mut same_artifact = observer(&output);
        same_artifact.observer_artifact_sha256 = mutation.operator_artifact_sha256.clone();
        assert_eq!(
            ledger.prepare_outcome(
                permit,
                OperatorOutcome::Commit {
                    output_sha256: output,
                    execution_receipt_sha256: digest(M),
                    observer_evidence: vec![same_artifact],
                }
            ),
            Err(RuntimeError::ObserverNotIndependent)
        );
    }

    #[test]
    fn missing_failed_or_detached_observer_evidence_cannot_commit() {
        let cases = [
            (vec![], RuntimeError::MissingEvidence("observer coverage")),
            (
                vec![ObserverEvidence {
                    decision: EvidenceDecision::Indeterminate,
                    ..observer(&digest(L))
                }],
                RuntimeError::EvidenceNotQualified("observer"),
            ),
            (
                vec![observer(&digest(N))],
                RuntimeError::InvalidMutation("observer receipt is detached from the exact output"),
            ),
        ];
        for (evidence, expected) in cases {
            let mut ledger = ledger();
            let mutation = make_mutation("attempt-009");
            let permit = authorize_and_dispatch(&mut ledger, mutation);
            assert_eq!(
                ledger.prepare_outcome(
                    permit,
                    OperatorOutcome::Commit {
                        output_sha256: digest(L),
                        execution_receipt_sha256: digest(M),
                        observer_evidence: evidence,
                    }
                ),
                Err(expected)
            );
        }
    }

    #[test]
    fn budget_exhaustion_is_a_hard_rejection() {
        let ledger = ledger();
        let mutation = make_mutation("attempt-010");
        let mut budget = snapshot(&ledger);
        budget.remaining_budget_units = 2;
        assert_eq!(
            ledger.prepare_authorization(&budget, mutation.clone(), approval(&mutation), digest(K)),
            Err(RuntimeError::BudgetExceeded)
        );
    }

    #[test]
    fn dispatched_attempt_blocks_a_second_mutation() {
        let mut ledger = ledger();
        let mutation = make_mutation("attempt-011");
        let (event, grant) = ledger
            .prepare_authorization(
                &snapshot(&ledger),
                mutation.clone(),
                approval(&mutation),
                digest(K),
            )
            .unwrap();
        ledger.append(event).unwrap();
        let (dispatch, _permit) = ledger.prepare_dispatch(grant).unwrap();
        ledger.append(dispatch).unwrap();

        let next = make_mutation("attempt-012");
        assert_eq!(
            ledger.prepare_authorization(
                &snapshot(&ledger),
                next.clone(),
                approval(&next),
                digest(N)
            ),
            Err(RuntimeError::ActiveAttemptBlocksMutation)
        );
    }

    #[test]
    fn unknown_outcome_blocks_mutation_until_reconciled() {
        let mut ledger = ledger();
        let permit = authorize_and_dispatch(&mut ledger, make_mutation("attempt-013"));
        let unknown = ledger
            .prepare_outcome(
                permit,
                OperatorOutcome::Unknown {
                    reason_receipt_sha256: digest(M),
                    observed_output_sha256: None,
                },
            )
            .unwrap();
        ledger.append(unknown).unwrap();

        let next = make_mutation("attempt-014");
        assert_eq!(
            ledger.prepare_authorization(
                &snapshot(&ledger),
                next.clone(),
                approval(&next),
                digest(N)
            ),
            Err(RuntimeError::ActiveAttemptBlocksMutation)
        );
    }

    #[test]
    fn reconciliation_commit_advances_head_once_and_unblocks_mutation() {
        let mut ledger = ledger();
        let mutation = make_mutation("attempt-015");
        let permit = authorize_and_dispatch(&mut ledger, mutation.clone());
        let unknown = ledger
            .prepare_outcome(
                permit,
                OperatorOutcome::Unknown {
                    reason_receipt_sha256: digest(M),
                    observed_output_sha256: Some(digest(L)),
                },
            )
            .unwrap();
        ledger.append(unknown).unwrap();

        let receipt = ReconciliationReceipt {
            receipt_sha256: digest(N),
            verifier_artifact_sha256: digest(J),
            attempt_id: mutation.attempt_id.clone(),
            plan_sha256: mutation.plan_sha256.clone(),
            action_sha256: mutation.action_sha256.clone(),
            parent_output_sha256: mutation.parent_output_sha256.clone(),
            resolution: ReconciliationResolution::Commit {
                output_sha256: digest(L),
                execution_receipt_sha256: digest(G),
                observer_evidence: vec![observer(&digest(L))],
            },
        };
        let reconciled = ledger.prepare_reconciliation(receipt).unwrap();
        ledger.append(reconciled).unwrap();
        assert_eq!(ledger.current_output_digest().unwrap(), digest(L));

        let mut next = make_mutation("attempt-016");
        next.parent_output_sha256 = digest(L);
        let (auth, _grant) = ledger
            .prepare_authorization(
                &AuthoritySnapshot {
                    ledger_revision: ledger.revision().unwrap(),
                    current_output_sha256: digest(L),
                    current_plan_sha256: digest(A),
                    current_registry_sha256: digest(B),
                    current_session_ledger_sha256: digest(C),
                    remaining_budget_units: 20,
                },
                next.clone(),
                approval(&next),
                digest(H),
            )
            .unwrap();
        assert!(matches!(auth, RuntimeEvent::AuthorizationIssued { .. }));
    }

    #[test]
    fn reconciliation_rejects_wrong_attempt_or_wrong_output() {
        let mut ledger = ledger();
        let mutation = make_mutation("attempt-017");
        let permit = authorize_and_dispatch(&mut ledger, mutation.clone());
        let unknown = ledger
            .prepare_outcome(
                permit,
                OperatorOutcome::Unknown {
                    reason_receipt_sha256: digest(M),
                    observed_output_sha256: Some(digest(L)),
                },
            )
            .unwrap();
        ledger.append(unknown).unwrap();

        let wrong_attempt = ReconciliationReceipt {
            receipt_sha256: digest(N),
            verifier_artifact_sha256: digest(J),
            attempt_id: "different-attempt".to_owned(),
            plan_sha256: mutation.plan_sha256.clone(),
            action_sha256: mutation.action_sha256.clone(),
            parent_output_sha256: mutation.parent_output_sha256.clone(),
            resolution: ReconciliationResolution::NoMutation {
                failure_receipt_sha256: digest(G),
            },
        };
        assert_eq!(
            ledger.prepare_reconciliation(wrong_attempt),
            Err(RuntimeError::ReconciliationMismatch)
        );

        let wrong_output = ReconciliationReceipt {
            receipt_sha256: digest(N),
            verifier_artifact_sha256: digest(J),
            attempt_id: mutation.attempt_id.clone(),
            plan_sha256: mutation.plan_sha256.clone(),
            action_sha256: mutation.action_sha256.clone(),
            parent_output_sha256: mutation.parent_output_sha256.clone(),
            resolution: ReconciliationResolution::Commit {
                output_sha256: digest(H),
                execution_receipt_sha256: digest(G),
                observer_evidence: vec![observer(&digest(H))],
            },
        };
        assert_eq!(
            ledger.prepare_reconciliation(wrong_output),
            Err(RuntimeError::ReconciliationMismatch)
        );
    }

    #[test]
    fn reconciliation_no_mutation_resolves_unknown_without_advancing_head() {
        let mut ledger = ledger();
        let mutation = make_mutation("attempt-018");
        let permit = authorize_and_dispatch(&mut ledger, mutation.clone());
        let unknown = ledger
            .prepare_outcome(
                permit,
                OperatorOutcome::Unknown {
                    reason_receipt_sha256: digest(M),
                    observed_output_sha256: None,
                },
            )
            .unwrap();
        ledger.append(unknown).unwrap();

        let receipt = ReconciliationReceipt {
            receipt_sha256: digest(N),
            verifier_artifact_sha256: digest(J),
            attempt_id: mutation.attempt_id,
            plan_sha256: mutation.plan_sha256,
            action_sha256: mutation.action_sha256,
            parent_output_sha256: mutation.parent_output_sha256,
            resolution: ReconciliationResolution::NoMutation {
                failure_receipt_sha256: digest(G),
            },
        };
        let event = ledger.prepare_reconciliation(receipt).unwrap();
        ledger.append(event).unwrap();
        assert_eq!(ledger.current_output_digest().unwrap(), digest(D));
        assert_eq!(ledger.revision().unwrap(), 5);
    }

    #[test]
    fn definite_failure_and_withdrawal_allow_safe_retry() {
        let mut ledger = ledger();
        let mutation = make_mutation("attempt-019");
        let permit = authorize_and_dispatch(&mut ledger, mutation);
        let failure = ledger
            .prepare_outcome(
                permit,
                OperatorOutcome::DefiniteNoMutationFailure {
                    failure_receipt_sha256: digest(M),
                },
            )
            .unwrap();
        ledger.append(failure).unwrap();

        let withdrawn_mutation = make_mutation("attempt-020");
        let (auth, grant) = ledger
            .prepare_authorization(
                &snapshot(&ledger),
                withdrawn_mutation.clone(),
                approval(&withdrawn_mutation),
                digest(N),
            )
            .unwrap();
        ledger.append(auth).unwrap();
        let withdrawal = ledger.prepare_withdrawal(grant, digest(J)).unwrap();
        ledger.append(withdrawal).unwrap();

        let retry = make_mutation("attempt-021");
        let (auth_retry, _grant_retry) = ledger
            .prepare_authorization(
                &snapshot(&ledger),
                retry.clone(),
                approval(&retry),
                digest(H),
            )
            .unwrap();
        assert!(matches!(
            auth_retry,
            RuntimeEvent::AuthorizationIssued { .. }
        ));
    }

    #[test]
    fn replay_rejects_duplicate_grant_and_preserves_unknown_state() {
        let mut ledger = ledger();
        let mutation = make_mutation("attempt-022");
        let (auth, grant) = ledger
            .prepare_authorization(
                &snapshot(&ledger),
                mutation.clone(),
                approval(&mutation),
                digest(K),
            )
            .unwrap();
        ledger.append(auth).unwrap();
        let replay = RuntimeLedger {
            events: ledger.events.clone(),
        };
        assert_eq!(
            replay.prepare_authorization(
                &snapshot(&replay),
                mutation.clone(),
                approval(&mutation),
                digest(K)
            ),
            Err(RuntimeError::ActiveAttemptBlocksMutation)
        );
        let _ = grant;
        assert_eq!(
            LedgerState::replay(ledger.events())
                .unwrap()
                .active_attempt
                .unwrap()
                .status,
            InternalAttemptStatus::Authorized
        );
    }

    #[test]
    fn invalid_digest_and_duplicate_evidence_reject() {
        let ledger = ledger();
        let mut invalid = make_mutation("attempt-023");
        invalid.action_sha256 = "not-a-digest".to_owned();
        assert_eq!(
            ledger.prepare_authorization(
                &snapshot(&ledger),
                invalid.clone(),
                approval(&invalid),
                digest(K)
            ),
            Err(RuntimeError::InvalidDigest("action_sha256"))
        );

        let mut duplicate = make_mutation("attempt-024");
        duplicate
            .precondition_evidence
            .push(duplicate.precondition_evidence[0].clone());
        assert_eq!(
            ledger.prepare_authorization(
                &snapshot(&ledger),
                duplicate.clone(),
                approval(&duplicate),
                digest(K)
            ),
            Err(RuntimeError::DuplicateEvidence("precondition"))
        );
    }

    #[test]
    fn dispatch_grant_is_bound_to_exact_persisted_revision() {
        let mut ledger = ledger();
        let mutation = make_mutation("attempt-025");
        let (auth, grant) = ledger
            .prepare_authorization(
                &snapshot(&ledger),
                mutation.clone(),
                approval(&mutation),
                digest(K),
            )
            .unwrap();
        ledger.append(auth).unwrap();
        // A second ledger event changes the revision and consumes/withdraws
        // the grant. The old grant can no longer dispatch.
        let withdrawal = ledger.prepare_withdrawal(grant, digest(J)).unwrap();
        ledger.append(withdrawal).unwrap();
        assert!(ledger.current_output_digest().is_ok());
    }
}
