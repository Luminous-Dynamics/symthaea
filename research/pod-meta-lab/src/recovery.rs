//! Research-only state machine for fault-aware action recovery.
//!
//! This module models action lifecycle decisions. It performs no external I/O,
//! does not enforce a real permission system, and is not a durable journal.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchSemantics {
    NonIdempotent,
    IdempotencyKeyEnforced,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionIntent {
    pub run_id: String,
    pub sequence: u64,
    pub action_id: String,
    pub pre_state_identity: String,
    pub expected_post_state_identity: String,
    pub principal_id: String,
    pub capability_id: String,
    pub idempotency_key: Option<String>,
    pub dispatch_semantics: DispatchSemantics,
    pub compensation_supported: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionPhase {
    Prepared,
    Dispatched,
    Acknowledged,
    UnknownOutcome,
    ReconciledCommitted,
    ReconciledNotCommitted,
    CompensationPrepared,
    Compensated,
    SafelyAborted,
    StoppedUnknownOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateObservation {
    Committed { state_identity: String },
    NotCommitted { state_identity: String },
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JournalEvent {
    Prepared { sequence: u64 },
    Dispatched { attempt: u32 },
    Acknowledged,
    AcknowledgementLost,
    ReconciledCommitted { state_identity: String },
    ReconciledNotCommitted { state_identity: String },
    OutcomeStillUnknown,
    RetryAuthorized { next_attempt: u32 },
    CompensationAuthorized,
    Compensated { state_identity: String },
    SafelyAborted,
    StoppedUnknownOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryError {
    EmptyIdentity,
    InvalidIdempotencyContract,
    InvalidTransition { from: ActionPhase, operation: &'static str },
    RetryWouldRiskDuplicateEffect,
    CompensationUnsupported,
    CompensationNotAuthorized,
    ObservationDoesNotMatchExpectedState,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionLifecycle {
    intent: ActionIntent,
    phase: ActionPhase,
    dispatch_attempts: u32,
    events: Vec<JournalEvent>,
}

impl ActionLifecycle {
    pub fn new(intent: ActionIntent) -> Result<Self, RecoveryError> {
        if intent.run_id.is_empty()
            || intent.sequence == 0
            || intent.action_id.is_empty()
            || intent.pre_state_identity.is_empty()
            || intent.expected_post_state_identity.is_empty()
            || intent.principal_id.is_empty()
            || intent.capability_id.is_empty()
        {
            return Err(RecoveryError::EmptyIdentity);
        }
        if intent.dispatch_semantics == DispatchSemantics::IdempotencyKeyEnforced
            && intent.idempotency_key.as_deref().is_none_or(str::is_empty)
        {
            return Err(RecoveryError::InvalidIdempotencyContract);
        }
        let sequence = intent.sequence;
        Ok(Self {
            intent,
            phase: ActionPhase::Prepared,
            dispatch_attempts: 0,
            events: vec![JournalEvent::Prepared { sequence }],
        })
    }

    pub fn phase(&self) -> ActionPhase { self.phase }
    pub fn dispatch_attempts(&self) -> u32 { self.dispatch_attempts }
    pub fn intent(&self) -> &ActionIntent { &self.intent }
    pub fn events(&self) -> &[JournalEvent] { &self.events }

    fn require_phase(
        &self,
        expected: ActionPhase,
        operation: &'static str,
    ) -> Result<(), RecoveryError> {
        if self.phase == expected {
            Ok(())
        } else {
            Err(RecoveryError::InvalidTransition { from: self.phase, operation })
        }
    }

    pub fn dispatch(&mut self) -> Result<(), RecoveryError> {
        self.require_phase(ActionPhase::Prepared, "dispatch")?;
        self.dispatch_attempts = self.dispatch_attempts.saturating_add(1);
        self.phase = ActionPhase::Dispatched;
        self.events.push(JournalEvent::Dispatched { attempt: self.dispatch_attempts });
        Ok(())
    }

    pub fn acknowledge(&mut self) -> Result<(), RecoveryError> {
        self.require_phase(ActionPhase::Dispatched, "acknowledge")?;
        self.phase = ActionPhase::Acknowledged;
        self.events.push(JournalEvent::Acknowledged);
        Ok(())
    }

    pub fn acknowledgement_lost(&mut self) -> Result<(), RecoveryError> {
        self.require_phase(ActionPhase::Dispatched, "acknowledgement_lost")?;
        self.phase = ActionPhase::UnknownOutcome;
        self.events.push(JournalEvent::AcknowledgementLost);
        Ok(())
    }

    pub fn reconcile(&mut self, observation: StateObservation) -> Result<ActionPhase, RecoveryError> {
        if !matches!(self.phase, ActionPhase::UnknownOutcome | ActionPhase::Acknowledged) {
            return Err(RecoveryError::InvalidTransition { from: self.phase, operation: "reconcile" });
        }
        match observation {
            StateObservation::Committed { state_identity } => {
                if state_identity != self.intent.expected_post_state_identity {
                    return Err(RecoveryError::ObservationDoesNotMatchExpectedState);
                }
                self.phase = ActionPhase::ReconciledCommitted;
                self.events.push(JournalEvent::ReconciledCommitted { state_identity });
            }
            StateObservation::NotCommitted { state_identity } => {
                if state_identity != self.intent.pre_state_identity {
                    return Err(RecoveryError::ObservationDoesNotMatchExpectedState);
                }
                self.phase = ActionPhase::ReconciledNotCommitted;
                self.events.push(JournalEvent::ReconciledNotCommitted { state_identity });
            }
            StateObservation::Unavailable => {
                self.phase = ActionPhase::UnknownOutcome;
                self.events.push(JournalEvent::OutcomeStillUnknown);
            }
        }
        Ok(self.phase)
    }

    /// Allows another dispatch only after authoritative non-commit evidence, or
    /// when the tool contract guarantees server-side deduplication for the same key.
    pub fn authorize_retry(&mut self) -> Result<(), RecoveryError> {
        match self.phase {
            ActionPhase::ReconciledNotCommitted => {}
            ActionPhase::UnknownOutcome
                if self.intent.dispatch_semantics == DispatchSemantics::IdempotencyKeyEnforced
                    && self.intent.idempotency_key.as_deref().is_some_and(|key| !key.is_empty()) => {}
            ActionPhase::UnknownOutcome => return Err(RecoveryError::RetryWouldRiskDuplicateEffect),
            other => return Err(RecoveryError::InvalidTransition { from: other, operation: "authorize_retry" }),
        }
        let next_attempt = self.dispatch_attempts.saturating_add(1);
        self.phase = ActionPhase::Prepared;
        self.events.push(JournalEvent::RetryAuthorized { next_attempt });
        Ok(())
    }

    pub fn prepare_compensation(&mut self, authorization_granted: bool) -> Result<(), RecoveryError> {
        self.require_phase(ActionPhase::ReconciledCommitted, "prepare_compensation")?;
        if !self.intent.compensation_supported {
            return Err(RecoveryError::CompensationUnsupported);
        }
        if !authorization_granted {
            return Err(RecoveryError::CompensationNotAuthorized);
        }
        self.phase = ActionPhase::CompensationPrepared;
        self.events.push(JournalEvent::CompensationAuthorized);
        Ok(())
    }

    pub fn confirm_compensation(&mut self, observed_state_identity: impl Into<String>) -> Result<(), RecoveryError> {
        self.require_phase(ActionPhase::CompensationPrepared, "confirm_compensation")?;
        let observed_state_identity = observed_state_identity.into();
        if observed_state_identity != self.intent.pre_state_identity {
            return Err(RecoveryError::ObservationDoesNotMatchExpectedState);
        }
        self.phase = ActionPhase::Compensated;
        self.events.push(JournalEvent::Compensated { state_identity: observed_state_identity });
        Ok(())
    }

    /// An unknown dispatched outcome is surfaced distinctly from a proven no-effect abort.
    pub fn stop(&mut self) -> Result<ActionPhase, RecoveryError> {
        match self.phase {
            ActionPhase::Prepared | ActionPhase::ReconciledNotCommitted | ActionPhase::Compensated => {
                self.phase = ActionPhase::SafelyAborted;
                self.events.push(JournalEvent::SafelyAborted);
            }
            ActionPhase::Dispatched | ActionPhase::Acknowledged | ActionPhase::UnknownOutcome => {
                self.phase = ActionPhase::StoppedUnknownOutcome;
                self.events.push(JournalEvent::StoppedUnknownOutcome);
            }
            other => return Err(RecoveryError::InvalidTransition { from: other, operation: "stop" }),
        }
        Ok(self.phase)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn intent(semantics: DispatchSemantics) -> ActionIntent {
        ActionIntent {
            run_id: "run-1".into(),
            sequence: 1,
            action_id: "write-1".into(),
            pre_state_identity: "state-before".into(),
            expected_post_state_identity: "state-after".into(),
            principal_id: "agent-a".into(),
            capability_id: "write-record".into(),
            idempotency_key: (semantics == DispatchSemantics::IdempotencyKeyEnforced)
                .then(|| "idem-1".into()),
            dispatch_semantics: semantics,
            compensation_supported: true,
        }
    }

    #[test]
    fn lost_acknowledgement_blocks_non_idempotent_retry_until_reconciliation() {
        let mut action = ActionLifecycle::new(intent(DispatchSemantics::NonIdempotent)).unwrap();
        action.dispatch().unwrap();
        action.acknowledgement_lost().unwrap();
        assert_eq!(action.phase(), ActionPhase::UnknownOutcome);
        assert_eq!(action.authorize_retry(), Err(RecoveryError::RetryWouldRiskDuplicateEffect));
        assert_eq!(
            action.reconcile(StateObservation::Committed { state_identity: "state-after".into() }).unwrap(),
            ActionPhase::ReconciledCommitted
        );
    }

    #[test]
    fn authoritative_non_commit_evidence_allows_retry() {
        let mut action = ActionLifecycle::new(intent(DispatchSemantics::NonIdempotent)).unwrap();
        action.dispatch().unwrap();
        action.acknowledgement_lost().unwrap();
        action.reconcile(StateObservation::NotCommitted { state_identity: "state-before".into() }).unwrap();
        action.authorize_retry().unwrap();
        assert_eq!(action.phase(), ActionPhase::Prepared);
        action.dispatch().unwrap();
        assert_eq!(action.dispatch_attempts(), 2);
    }

    #[test]
    fn idempotency_contract_allows_same_key_retry_after_unknown_outcome() {
        let mut action = ActionLifecycle::new(intent(DispatchSemantics::IdempotencyKeyEnforced)).unwrap();
        let key = action.intent().idempotency_key.clone();
        action.dispatch().unwrap();
        action.acknowledgement_lost().unwrap();
        action.authorize_retry().unwrap();
        assert_eq!(action.intent().idempotency_key, key);
        action.dispatch().unwrap();
        assert_eq!(action.dispatch_attempts(), 2);
    }

    #[test]
    fn mismatched_readback_does_not_resolve_unknown_outcome() {
        let mut action = ActionLifecycle::new(intent(DispatchSemantics::NonIdempotent)).unwrap();
        action.dispatch().unwrap();
        action.acknowledgement_lost().unwrap();
        assert_eq!(
            action.reconcile(StateObservation::Committed { state_identity: "unexpected".into() }),
            Err(RecoveryError::ObservationDoesNotMatchExpectedState)
        );
        assert_eq!(action.phase(), ActionPhase::UnknownOutcome);
    }

    #[test]
    fn unavailable_readback_keeps_unknown_outcome() {
        let mut action = ActionLifecycle::new(intent(DispatchSemantics::NonIdempotent)).unwrap();
        action.dispatch().unwrap();
        action.acknowledgement_lost().unwrap();
        assert_eq!(action.reconcile(StateObservation::Unavailable).unwrap(), ActionPhase::UnknownOutcome);
        assert_eq!(action.stop().unwrap(), ActionPhase::StoppedUnknownOutcome);
    }

    #[test]
    fn safe_abort_is_distinct_from_stopping_with_unknown_effect() {
        let mut not_dispatched = ActionLifecycle::new(intent(DispatchSemantics::NonIdempotent)).unwrap();
        assert_eq!(not_dispatched.stop().unwrap(), ActionPhase::SafelyAborted);

        let mut unresolved = ActionLifecycle::new(intent(DispatchSemantics::NonIdempotent)).unwrap();
        unresolved.dispatch().unwrap();
        unresolved.acknowledgement_lost().unwrap();
        assert_eq!(unresolved.stop().unwrap(), ActionPhase::StoppedUnknownOutcome);
    }

    #[test]
    fn compensation_requires_capability_and_verified_restoration() {
        let mut action = ActionLifecycle::new(intent(DispatchSemantics::NonIdempotent)).unwrap();
        action.dispatch().unwrap();
        action.acknowledge().unwrap();
        action.reconcile(StateObservation::Committed { state_identity: "state-after".into() }).unwrap();
        assert_eq!(action.prepare_compensation(false), Err(RecoveryError::CompensationNotAuthorized));
        action.prepare_compensation(true).unwrap();
        assert_eq!(
            action.confirm_compensation("wrong-state"),
            Err(RecoveryError::ObservationDoesNotMatchExpectedState)
        );
        action.confirm_compensation("state-before").unwrap();
        assert_eq!(action.phase(), ActionPhase::Compensated);
    }

    #[test]
    fn journal_events_are_append_only_through_lifecycle_transitions() {
        let mut action = ActionLifecycle::new(intent(DispatchSemantics::NonIdempotent)).unwrap();
        let initial_len = action.events().len();
        action.dispatch().unwrap();
        action.acknowledgement_lost().unwrap();
        action.reconcile(StateObservation::Unavailable).unwrap();
        assert_eq!(action.events().len(), initial_len + 3);
        assert!(matches!(action.events().last(), Some(JournalEvent::OutcomeStillUnknown)));
    }

    #[test]
    fn idempotency_semantics_require_a_nonempty_key() {
        let mut invalid = intent(DispatchSemantics::IdempotencyKeyEnforced);
        invalid.idempotency_key = Some(String::new());
        assert_eq!(ActionLifecycle::new(invalid), Err(RecoveryError::InvalidIdempotencyContract));
    }
}
