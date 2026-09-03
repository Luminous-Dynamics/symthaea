// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Affine execution transaction for committed operational restore.
//!
//! Restore execution is a single owner-scoped transaction. A committed restore
//! is consumed into one session; the session owns the exact validated checkpoint
//! source, every unused action permit, and every executor-earned receipt.
//! Executor failure aborts the session and drops all remaining execution
//! authority. Productive activation may eventually consume only a fully
//! completed session token, which retains the same exact source.

use super::restore_actions::{
    canonical_plan_for_decision, EvidenceRestorePolicy, RestoreAction, RestoreDomainPlan,
    RestorePlanError,
};
use super::restore_admission::{
    CommittedOperationalRestore, OperationalRestoreSource, RestoreDigest, RestoreGenerationFence,
};
use super::restore_semantics::RestoreDomain;
use crate::actuator_isolation::ActuatorIsolationSupervisor;
use crate::operator_authority::OperatorAuthority;
use crate::temporal_assurance::TemporalAssuranceSupervisor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RestoreExecutionBinding {
    checkpoint_digest: RestoreDigest,
    fence: RestoreGenerationFence,
}

impl RestoreExecutionBinding {
    fn from_committed(committed: &CommittedOperationalRestore) -> Self {
        Self {
            checkpoint_digest: committed.checkpoint_digest(),
            fence: committed.fence(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RestoreActionOutcome {
    Applied,
    Requalified,
    Reconciled,
    Dropped,
}

/// Evidence that one exact canonical restore action completed.
///
/// Deliberately not `Clone`, `Copy`, `Serialize` or `Deserialize`. There is no
/// generic production constructor. Receipts stay owned by the execution session.
#[derive(Debug, PartialEq, Eq)]
struct RestoreActionReceipt {
    binding: RestoreExecutionBinding,
    domain: RestoreDomain,
    action: RestoreAction,
    outcome: RestoreActionOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RestoreExecutorError {
    PermitMismatch {
        expected_domain: RestoreDomain,
        expected_action: RestoreAction,
        actual_domain: RestoreDomain,
        actual_action: RestoreAction,
    },
}

/// Single-use authority to execute one exact canonical restore obligation.
///
/// This is an implementation detail of `RestoreExecutionSession`; it never
/// escapes the session through the productive API.
#[derive(Debug)]
struct RestoreActionPermit {
    binding: RestoreExecutionBinding,
    domain: RestoreDomain,
    action: RestoreAction,
}

impl RestoreActionPermit {
    fn execute_operator_replay_merge(
        self,
        live: &mut OperatorAuthority,
        checkpoint: &OperatorAuthority,
    ) -> Result<RestoreActionReceipt, RestoreExecutorError> {
        let expected_domain = RestoreDomain::OperatorAuthority;
        let expected_action = RestoreAction::MergeEvidence(EvidenceRestorePolicy::ReplayBarrier);
        if self.domain != expected_domain || self.action != expected_action {
            return Err(RestoreExecutorError::PermitMismatch {
                expected_domain,
                expected_action,
                actual_domain: self.domain,
                actual_action: self.action,
            });
        }

        let Self {
            binding,
            domain,
            action,
        } = self;
        live.merge_restore_replay_evidence_from(checkpoint);
        Ok(RestoreActionReceipt {
            binding,
            domain,
            action,
            outcome: RestoreActionOutcome::Applied,
        })
    }

    fn execute_actuator_authority_join(
        self,
        live: &mut ActuatorIsolationSupervisor,
        checkpoint: &ActuatorIsolationSupervisor,
    ) -> Result<RestoreActionReceipt, RestoreExecutorError> {
        let expected_domain = RestoreDomain::ActuatorIsolation;
        let expected_action = RestoreAction::PreserveOrNarrowAuthority;
        if self.domain != expected_domain || self.action != expected_action {
            return Err(RestoreExecutorError::PermitMismatch {
                expected_domain,
                expected_action,
                actual_domain: self.domain,
                actual_action: self.action,
            });
        }

        let Self {
            binding,
            domain,
            action,
        } = self;
        live.preserve_restore_isolation_latches_from(checkpoint);
        Ok(RestoreActionReceipt {
            binding,
            domain,
            action,
            outcome: RestoreActionOutcome::Applied,
        })
    }

    fn execute_temporal_authority_join(
        self,
        live: &mut TemporalAssuranceSupervisor,
        checkpoint: &TemporalAssuranceSupervisor,
    ) -> Result<RestoreActionReceipt, RestoreExecutorError> {
        let expected_domain = RestoreDomain::TemporalAssurance;
        let expected_action = RestoreAction::PreserveOrNarrowAuthority;
        if self.domain != expected_domain || self.action != expected_action {
            return Err(RestoreExecutorError::PermitMismatch {
                expected_domain,
                expected_action,
                actual_domain: self.domain,
                actual_action: self.action,
            });
        }

        let Self {
            binding,
            domain,
            action,
        } = self;
        live.preserve_restore_hold_latch_from(checkpoint);
        Ok(RestoreActionReceipt {
            binding,
            domain,
            action,
            outcome: RestoreActionOutcome::Applied,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RestoreExecutionSessionState {
    Open,
    Aborted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RestoreSessionError {
    PlanUnavailable {
        domain: RestoreDomain,
        error: RestorePlanError,
    },
    Aborted,
    MissingPermit {
        domain: RestoreDomain,
        action: RestoreAction,
    },
    Executor(RestoreExecutorError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RestoreExecutionError {
    WrongBinding {
        domain: RestoreDomain,
        action: RestoreAction,
    },
    MissingReceipt {
        domain: RestoreDomain,
        action: RestoreAction,
    },
    DuplicateReceipt {
        domain: RestoreDomain,
        action: RestoreAction,
    },
    UnexpectedReceipt {
        domain: RestoreDomain,
        action: RestoreAction,
    },
    OutcomeMismatch {
        domain: RestoreDomain,
        action: RestoreAction,
        outcome: RestoreActionOutcome,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RestoreFinishError {
    Aborted,
    Incomplete { remaining_permits: usize },
    InvalidExecution(RestoreExecutionError),
}

/// Fully executed restore transaction.
///
/// Deliberately affine and non-serializable. The exact validated checkpoint
/// source survives into this token so final activation can consume the same
/// source that preparation, commit, executors and receipts were bound to.
pub(super) struct CompletedRestoreExecution {
    _binding: RestoreExecutionBinding,
    _source: OperationalRestoreSource,
    _receipts: Vec<RestoreActionReceipt>,
}

/// One committed restore creates exactly one execution attempt.
///
/// `begin` consumes the non-Clone `CommittedOperationalRestore`, preventing one
/// committed authority decision from spawning multiple independent permit sets.
/// It also takes ownership of the exact validated checkpoint source. Executors
/// therefore have no API for supplying replacement checkpoint-domain objects.
/// The session itself is non-Clone/non-Copy/non-serializable.
pub(super) struct RestoreExecutionSession {
    binding: RestoreExecutionBinding,
    source: OperationalRestoreSource,
    plans: Vec<RestoreDomainPlan>,
    permits: Vec<RestoreActionPermit>,
    receipts: Vec<RestoreActionReceipt>,
    state: RestoreExecutionSessionState,
}

impl RestoreExecutionSession {
    pub(super) fn begin(
        committed: CommittedOperationalRestore,
    ) -> Result<Self, RestoreSessionError> {
        let binding = RestoreExecutionBinding::from_committed(&committed);
        let plans = canonical_plans_for_committed(&committed).map_err(plan_session_error)?;
        let permits = permits_for_plans(binding, &plans);
        let source = committed.into_source();
        debug_assert_eq!(binding.checkpoint_digest, source.digest());
        Ok(Self {
            binding,
            source,
            plans,
            permits,
            receipts: Vec::new(),
            state: RestoreExecutionSessionState::Open,
        })
    }

    pub(super) const fn state(&self) -> RestoreExecutionSessionState {
        self.state
    }

    pub(super) fn remaining_permits(&self) -> usize {
        self.permits.len()
    }

    pub(super) fn receipt_count(&self) -> usize {
        self.receipts.len()
    }

    fn abort(&mut self) {
        self.state = RestoreExecutionSessionState::Aborted;
        self.permits.clear();
    }

    fn take_exact_permit(
        &mut self,
        domain: RestoreDomain,
        action: RestoreAction,
    ) -> Result<RestoreActionPermit, RestoreSessionError> {
        if self.state == RestoreExecutionSessionState::Aborted {
            return Err(RestoreSessionError::Aborted);
        }
        let Some(index) = self
            .permits
            .iter()
            .position(|permit| permit.domain == domain && permit.action == action)
        else {
            self.abort();
            return Err(RestoreSessionError::MissingPermit { domain, action });
        };
        Ok(self.permits.remove(index))
    }

    /// Execute and internally record the operator replay barrier obligation using
    /// only the operator source owned by this exact restore session.
    pub(super) fn execute_operator_replay_merge(
        &mut self,
        live: &mut OperatorAuthority,
    ) -> Result<(), RestoreSessionError> {
        let domain = RestoreDomain::OperatorAuthority;
        let action = RestoreAction::MergeEvidence(EvidenceRestorePolicy::ReplayBarrier);
        let permit = self.take_exact_permit(domain, action)?;
        let result = {
            let checkpoint = &self.source.checkpoint().operator_authority;
            permit.execute_operator_replay_merge(live, checkpoint)
        };
        match result {
            Ok(receipt) => {
                self.receipts.push(receipt);
                Ok(())
            }
            Err(error) => {
                self.abort();
                Err(RestoreSessionError::Executor(error))
            }
        }
    }

    /// Preserve the union of live and session-owned checkpoint actuator
    /// isolation authority.
    ///
    /// This consumes only `ActuatorIsolation + PreserveOrNarrowAuthority`. The
    /// remaining actuator evidence/requalification/reconciliation obligations
    /// stay in the session and continue to block completion.
    pub(super) fn execute_actuator_authority_join(
        &mut self,
        live: &mut ActuatorIsolationSupervisor,
    ) -> Result<(), RestoreSessionError> {
        let domain = RestoreDomain::ActuatorIsolation;
        let action = RestoreAction::PreserveOrNarrowAuthority;
        let permit = self.take_exact_permit(domain, action)?;
        let result = {
            let checkpoint = &self.source.checkpoint().actuator_isolation;
            permit.execute_actuator_authority_join(live, checkpoint)
        };
        match result {
            Ok(receipt) => {
                self.receipts.push(receipt);
                Ok(())
            }
            Err(error) => {
                self.abort();
                Err(RestoreSessionError::Executor(error))
            }
        }
    }

    /// Preserve temporal review authority from the exact session-owned source
    /// without importing historical runtime measurements as current truth.
    ///
    /// This consumes only `TemporalAssurance + PreserveOrNarrowAuthority`.
    /// Replay/counterexample/history merge, fresh requalification and final
    /// reconciliation remain separate unconsumed obligations.
    pub(super) fn execute_temporal_authority_join(
        &mut self,
        live: &mut TemporalAssuranceSupervisor,
    ) -> Result<(), RestoreSessionError> {
        let domain = RestoreDomain::TemporalAssurance;
        let action = RestoreAction::PreserveOrNarrowAuthority;
        let permit = self.take_exact_permit(domain, action)?;
        let result = {
            let checkpoint = &self.source.checkpoint().temporal;
            permit.execute_temporal_authority_join(live, checkpoint)
        };
        match result {
            Ok(receipt) => {
                self.receipts.push(receipt);
                Ok(())
            }
            Err(error) => {
                self.abort();
                Err(RestoreSessionError::Executor(error))
            }
        }
    }

    /// Consume a completed session into the only token eligible for later
    /// productive activation.
    ///
    /// There is intentionally no all-green fixture yet: until every canonical
    /// action has a concrete executor, ordinary sessions remain incomplete and
    /// this function must fail closed.
    pub(super) fn finish(self) -> Result<CompletedRestoreExecution, RestoreFinishError> {
        if self.state == RestoreExecutionSessionState::Aborted {
            return Err(RestoreFinishError::Aborted);
        }
        if !self.permits.is_empty() {
            return Err(RestoreFinishError::Incomplete {
                remaining_permits: self.permits.len(),
            });
        }
        validate_execution(self.binding, &self.plans, &self.receipts)
            .map_err(RestoreFinishError::InvalidExecution)?;
        Ok(CompletedRestoreExecution {
            _binding: self.binding,
            _source: self.source,
            _receipts: self.receipts,
        })
    }
}

fn canonical_plans_for_committed(
    committed: &CommittedOperationalRestore,
) -> Result<Vec<RestoreDomainPlan>, RestorePlanError> {
    committed
        .decisions()
        .iter()
        .map(|decision| canonical_plan_for_decision(*decision))
        .collect()
}

fn plan_domain(error: RestorePlanError) -> RestoreDomain {
    match error {
        RestorePlanError::DecisionNotAdmissible { domain, .. }
        | RestorePlanError::DuplicateAction { domain, .. }
        | RestorePlanError::MissingSemanticAction { domain, .. }
        | RestorePlanError::MissingVerdictAction { domain, .. }
        | RestorePlanError::UnexpectedAction { domain, .. }
        | RestorePlanError::EvidencePolicyUnderspecified { domain }
        | RestorePlanError::MissingEvidencePolicy { domain, .. }
        | RestorePlanError::UnexpectedEvidencePolicy { domain, .. }
        | RestorePlanError::VerdictActionMismatch { domain, .. } => domain,
    }
}

fn plan_session_error(error: RestorePlanError) -> RestoreSessionError {
    RestoreSessionError::PlanUnavailable {
        domain: plan_domain(error),
        error,
    }
}

fn permits_for_plans(
    binding: RestoreExecutionBinding,
    plans: &[RestoreDomainPlan],
) -> Vec<RestoreActionPermit> {
    plans
        .iter()
        .flat_map(|plan| {
            plan.actions().iter().copied().map(move |action| RestoreActionPermit {
                binding,
                domain: plan.domain(),
                action,
            })
        })
        .collect()
}

fn outcome_matches(action: RestoreAction, outcome: RestoreActionOutcome) -> bool {
    match action {
        RestoreAction::ReplaceValidatedHistorical
        | RestoreAction::PreserveOrNarrowAuthority
        | RestoreAction::MergeEvidence(_) => outcome == RestoreActionOutcome::Applied,
        RestoreAction::RequalifyFromCurrentInputs => outcome == RestoreActionOutcome::Requalified,
        RestoreAction::ReconcileBeforeActivation => outcome == RestoreActionOutcome::Reconciled,
        RestoreAction::DropEphemeral => outcome == RestoreActionOutcome::Dropped,
    }
}

fn validate_receipts_for_plan(
    binding: RestoreExecutionBinding,
    plan: &RestoreDomainPlan,
    receipts: &[RestoreActionReceipt],
) -> Result<(), RestoreExecutionError> {
    for receipt in receipts.iter().filter(|receipt| receipt.domain == plan.domain()) {
        if receipt.binding != binding {
            return Err(RestoreExecutionError::WrongBinding {
                domain: receipt.domain,
                action: receipt.action,
            });
        }
        if !plan.actions().contains(&receipt.action) {
            return Err(RestoreExecutionError::UnexpectedReceipt {
                domain: receipt.domain,
                action: receipt.action,
            });
        }
        if !outcome_matches(receipt.action, receipt.outcome) {
            return Err(RestoreExecutionError::OutcomeMismatch {
                domain: receipt.domain,
                action: receipt.action,
                outcome: receipt.outcome,
            });
        }
    }

    for action in plan.actions() {
        let matching = receipts
            .iter()
            .filter(|receipt| receipt.domain == plan.domain() && receipt.action == *action)
            .count();
        match matching {
            0 => {
                return Err(RestoreExecutionError::MissingReceipt {
                    domain: plan.domain(),
                    action: *action,
                });
            }
            1 => {}
            _ => {
                return Err(RestoreExecutionError::DuplicateReceipt {
                    domain: plan.domain(),
                    action: *action,
                });
            }
        }
    }
    Ok(())
}

fn validate_execution(
    binding: RestoreExecutionBinding,
    plans: &[RestoreDomainPlan],
    receipts: &[RestoreActionReceipt],
) -> Result<(), RestoreExecutionError> {
    for plan in plans {
        validate_receipts_for_plan(binding, plan, receipts)?;
    }
    for receipt in receipts {
        if !plans.iter().any(|plan| {
            plan.domain() == receipt.domain && plan.actions().contains(&receipt.action)
        }) {
            return Err(RestoreExecutionError::UnexpectedReceipt {
                domain: receipt.domain,
                action: receipt.action,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actuator_isolation::PhysicalActuator;
    use crate::embodiment::SubterraneanEmbodiment;
    use crate::operational_checkpoint::restore_admission::{
        commit_operational_restore, prepare_operational_restore, OperationalRestoreSource,
        RestoreAdmissionVerdict, RestoreDomainDecision, RestorePreparationContext,
    };
    use crate::operational_checkpoint::restore_semantics::OPERATIONAL_RESTORE_CONTRACTS;
    use crate::operator_protocol::{
        AuthenticationLevel, OperatorCommand, OperatorCommandEnvelope, OperatorId, OperatorRole,
    };
    use crate::plan_freshness::RuntimeRevisions;
    use crate::temporal_assurance::{TemporalAuthority, TemporalRuntimeFrame};
    use symthaea_core::genesis::GenesisSeed;

    fn digest(byte: u8) -> RestoreDigest {
        RestoreDigest::new([byte; 32])
    }

    fn fence() -> RestoreGenerationFence {
        RestoreGenerationFence::new(1, 2, 3, 4, 5, digest(6))
    }

    fn alternate_fence() -> RestoreGenerationFence {
        RestoreGenerationFence::new(1, 2, 3, 99, 5, digest(6))
    }

    fn decisions() -> Vec<RestoreDomainDecision> {
        OPERATIONAL_RESTORE_CONTRACTS
            .iter()
            .map(|contract| {
                RestoreDomainDecision::new(
                    contract.domain,
                    RestoreAdmissionVerdict::ReconciliationRequired,
                )
            })
            .collect()
    }

    fn source(phrase: &str) -> OperationalRestoreSource {
        let mut checkpoint =
            SubterraneanEmbodiment::new(&GenesisSeed::from_phrase(phrase)).operational_checkpoint();

        // Give the owned source concrete replay evidence.
        checkpoint
            .operator_authority
            .ingest(
                OperatorCommandEnvelope {
                    operator: OperatorId(91),
                    role: OperatorRole::SafetyOfficer,
                    authentication: AuthenticationLevel::HardwareBacked,
                    epoch: 1,
                    sequence: 4,
                    proposal_id: 91,
                    issued_step: 0,
                    expires_step: 100,
                    command: OperatorCommand::HoldPosition,
                },
                0,
                true,
            )
            .expect("source replay evidence");

        // Give the same owned source one actuator latch.
        checkpoint
            .actuator_isolation
            .force_health_for_test(PhysicalActuator::LeftTrack, 0.0);
        let mut command = crate::types::SubterraneanCommand::zero();
        command.set_left_track(1.0);
        let state = crate::types::SubterraneanState::home();
        for _ in 0..64 {
            checkpoint.actuator_isolation.observe(&command, &state, &state);
            if checkpoint
                .actuator_isolation
                .report()
                .is_isolated(PhysicalActuator::LeftTrack)
            {
                break;
            }
        }
        assert!(
            checkpoint
                .actuator_isolation
                .report()
                .is_isolated(PhysicalActuator::LeftTrack)
        );

        // Give the source a temporal review hold without importing any of its
        // measurements as live truth during the later authority join.
        checkpoint.temporal.assess(
            0.005,
            0,
            RuntimeRevisions::default(),
            &TemporalRuntimeFrame::default(),
            true,
            false,
        );
        assert!(checkpoint.temporal.hold_latched());

        OperationalRestoreSource::capture(checkpoint).expect("valid owned restore source")
    }

    fn committed_with(
        phrase: &str,
        live_fence: RestoreGenerationFence,
    ) -> CommittedOperationalRestore {
        let prepared = prepare_operational_restore(
            RestorePreparationContext::new(source(phrase), live_fence),
            decisions(),
        )
        .expect("prepare");
        commit_operational_restore(prepared, live_fence).expect("commit")
    }

    fn committed() -> CommittedOperationalRestore {
        committed_with("restore-execution-source", fence())
    }

    #[test]
    fn begin_owns_exact_canonical_obligation_set_and_source_identity() {
        let expected = decisions()
            .iter()
            .map(|decision| canonical_plan_for_decision(*decision).unwrap().actions().len())
            .sum::<usize>();
        let session = RestoreExecutionSession::begin(committed()).expect("session");
        assert_eq!(session.state(), RestoreExecutionSessionState::Open);
        assert_eq!(session.remaining_permits(), expected);
        assert_eq!(session.receipt_count(), 0);
        assert_eq!(session.binding.checkpoint_digest, session.source.digest());
    }

    #[test]
    fn operator_replay_execution_consumes_only_session_owned_source() {
        let mut session = RestoreExecutionSession::begin(committed()).expect("session");
        let before = session.remaining_permits();
        let mut live = OperatorAuthority::default();
        session
            .execute_operator_replay_merge(&mut live)
            .expect("operator replay execution");
        assert_eq!(session.remaining_permits(), before - 1);
        assert_eq!(session.receipt_count(), 1);
        assert_eq!(session.state(), RestoreExecutionSessionState::Open);
        let receipt = &session.receipts[0];
        assert_eq!(receipt.binding, session.binding);
        assert_eq!(receipt.domain, RestoreDomain::OperatorAuthority);
        assert_eq!(
            receipt.action,
            RestoreAction::MergeEvidence(EvidenceRestorePolicy::ReplayBarrier)
        );
        assert_eq!(receipt.outcome, RestoreActionOutcome::Applied);

        // Sequence 4 came only from the session-owned checkpoint. If its replay
        // evidence was actually merged, replaying it against live must fail.
        let replay = live.ingest(
            OperatorCommandEnvelope {
                operator: OperatorId(91),
                role: OperatorRole::SafetyOfficer,
                authentication: AuthenticationLevel::HardwareBacked,
                epoch: 1,
                sequence: 4,
                proposal_id: 92,
                issued_step: 0,
                expires_step: 100,
                command: OperatorCommand::EmergencyStop,
            },
            0,
            true,
        );
        assert_eq!(
            replay.err(),
            Some(crate::operator_authority::OperatorAuthorityRejection::Replay)
        );
    }

    #[test]
    fn actuator_authority_executor_uses_owned_source_and_earns_only_one_receipt() {
        let mut session = RestoreExecutionSession::begin(committed()).expect("session");
        let before_permits = session.remaining_permits();
        let before_receipts = session.receipt_count();
        let mut live = ActuatorIsolationSupervisor::default();

        session
            .execute_actuator_authority_join(&mut live)
            .expect("actuator authority join");

        assert!(live.report().is_isolated(PhysicalActuator::LeftTrack));
        assert_eq!(session.remaining_permits(), before_permits - 1);
        assert_eq!(session.receipt_count(), before_receipts + 1);
        let receipt = session.receipts.last().expect("actuator receipt");
        assert_eq!(receipt.domain, RestoreDomain::ActuatorIsolation);
        assert_eq!(receipt.action, RestoreAction::PreserveOrNarrowAuthority);
        assert_eq!(receipt.outcome, RestoreActionOutcome::Applied);
        assert_eq!(session.state(), RestoreExecutionSessionState::Open);
    }

    #[test]
    fn temporal_authority_executor_uses_owned_source_and_earns_only_one_receipt() {
        let mut session = RestoreExecutionSession::begin(committed()).expect("session");
        let before_permits = session.remaining_permits();
        let before_receipts = session.receipt_count();
        let mut live = TemporalAssuranceSupervisor::default();

        session
            .execute_temporal_authority_join(&mut live)
            .expect("temporal authority join");

        assert!(live.hold_latched());
        assert_eq!(live.last().authority, TemporalAuthority::HoldForReview);
        assert_eq!(live.clean_dwell_steps(), 0);
        assert_eq!(session.remaining_permits(), before_permits - 1);
        assert_eq!(session.receipt_count(), before_receipts + 1);
        let receipt = session.receipts.last().expect("temporal receipt");
        assert_eq!(receipt.domain, RestoreDomain::TemporalAssurance);
        assert_eq!(receipt.action, RestoreAction::PreserveOrNarrowAuthority);
        assert_eq!(receipt.outcome, RestoreActionOutcome::Applied);
        assert_eq!(session.state(), RestoreExecutionSessionState::Open);
    }

    #[test]
    fn duplicate_executor_attempt_aborts_and_destroys_remaining_authority() {
        let mut session = RestoreExecutionSession::begin(committed()).expect("session");
        let mut live = OperatorAuthority::default();
        session
            .execute_operator_replay_merge(&mut live)
            .expect("first execution");
        assert_eq!(
            session.execute_operator_replay_merge(&mut live).err(),
            Some(RestoreSessionError::MissingPermit {
                domain: RestoreDomain::OperatorAuthority,
                action: RestoreAction::MergeEvidence(EvidenceRestorePolicy::ReplayBarrier),
            })
        );
        assert_eq!(session.state(), RestoreExecutionSessionState::Aborted);
        assert_eq!(session.remaining_permits(), 0);
        assert_eq!(session.receipt_count(), 1);
    }

    #[test]
    fn aborted_session_rejects_all_future_execution() {
        let mut session = RestoreExecutionSession::begin(committed()).expect("session");
        let mut live = OperatorAuthority::default();
        session
            .execute_operator_replay_merge(&mut live)
            .expect("first execution");
        let _ = session.execute_operator_replay_merge(&mut live);
        assert_eq!(
            session.execute_operator_replay_merge(&mut live).err(),
            Some(RestoreSessionError::Aborted)
        );
    }

    #[test]
    fn incomplete_session_cannot_finish() {
        let session = RestoreExecutionSession::begin(committed()).expect("session");
        let remaining = session.remaining_permits();
        assert_eq!(
            session.finish().err(),
            Some(RestoreFinishError::Incomplete {
                remaining_permits: remaining,
            })
        );
    }

    #[test]
    fn aborted_session_cannot_finish() {
        let mut session = RestoreExecutionSession::begin(committed()).expect("session");
        let mut live = OperatorAuthority::default();
        session
            .execute_operator_replay_merge(&mut live)
            .expect("first execution");
        let _ = session.execute_operator_replay_merge(&mut live);
        assert_eq!(session.finish().err(), Some(RestoreFinishError::Aborted));
    }

    #[test]
    fn same_source_under_different_live_fence_creates_different_session_binding() {
        let first =
            RestoreExecutionSession::begin(committed_with("same-source", fence())).expect("first");
        let second = RestoreExecutionSession::begin(committed_with(
            "same-source",
            alternate_fence(),
        ))
        .expect("second");
        assert_eq!(first.binding.checkpoint_digest, second.binding.checkpoint_digest);
        assert_ne!(first.binding, second.binding);
    }

    #[test]
    fn different_source_creates_different_session_binding_under_same_live_fence() {
        let first = RestoreExecutionSession::begin(committed_with("source-a", fence())).expect("a");
        let second = RestoreExecutionSession::begin(committed_with("source-b", fence())).expect("b");
        assert_ne!(first.binding.checkpoint_digest, second.binding.checkpoint_digest);
        assert_ne!(first.binding, second.binding);
    }

    #[test]
    fn finish_validator_still_rejects_missing_receipts_even_if_permits_are_test_cleared() {
        let mut session = RestoreExecutionSession::begin(committed()).expect("session");
        session.permits.clear();
        assert!(matches!(
            session.finish(),
            Err(RestoreFinishError::InvalidExecution(
                RestoreExecutionError::MissingReceipt { .. }
            ))
        ));
    }
}
