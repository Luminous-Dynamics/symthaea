// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Affine execution transaction for committed operational restore.
//!
//! Restore execution is a single owner-scoped transaction. A committed restore
//! is consumed into one session; the session owns every unused action permit and
//! every executor-earned receipt. Executor failure aborts the session and drops
//! all remaining execution authority. Productive activation may eventually
//! consume only a fully completed session token.

use super::restore_actions::{
    canonical_plan_for_decision, EvidenceRestorePolicy, RestoreAction, RestoreDomainPlan,
    RestorePlanError,
};
use super::restore_admission::{CommittedOperationalRestore, RestoreDigest, RestoreGenerationFence};
use super::restore_semantics::RestoreDomain;
use crate::operator_authority::OperatorAuthority;

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
/// Deliberately affine and non-serializable. A later activation integration may
/// consume this token, but checkpoint bytes can never deserialize into it.
/// Its fields remain opaque until that activation boundary is implemented.
pub(super) struct CompletedRestoreExecution {
    _binding: RestoreExecutionBinding,
    _receipts: Vec<RestoreActionReceipt>,
}

/// One committed restore creates exactly one execution attempt.
///
/// `begin` consumes the non-Clone `CommittedOperationalRestore`, preventing one
/// committed authority decision from spawning multiple independent permit sets.
/// The session itself is non-Clone/non-Copy/non-serializable.
pub(super) struct RestoreExecutionSession {
    binding: RestoreExecutionBinding,
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
        Ok(Self {
            binding,
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

    /// Execute and internally record the first concrete restore receipt.
    ///
    /// The raw permit and resulting receipt never leave the session. Any failure
    /// aborts the complete transaction and clears every unused permit.
    pub(super) fn execute_operator_replay_merge(
        &mut self,
        live: &mut OperatorAuthority,
        checkpoint: &OperatorAuthority,
    ) -> Result<(), RestoreSessionError> {
        let domain = RestoreDomain::OperatorAuthority;
        let action = RestoreAction::MergeEvidence(EvidenceRestorePolicy::ReplayBarrier);
        let permit = self.take_exact_permit(domain, action)?;
        match permit.execute_operator_replay_merge(live, checkpoint) {
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
    use crate::operational_checkpoint::restore_admission::{
        commit_operational_restore, prepare_operational_restore, RestoreAdmissionVerdict,
        RestoreDomainDecision, RestorePreparationContext,
    };
    use crate::operational_checkpoint::restore_semantics::OPERATIONAL_RESTORE_CONTRACTS;

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

    fn committed_with(
        checkpoint_byte: u8,
        live_fence: RestoreGenerationFence,
    ) -> CommittedOperationalRestore {
        let prepared = prepare_operational_restore(
            RestorePreparationContext::new(digest(checkpoint_byte), live_fence),
            decisions(),
        )
        .expect("prepare");
        commit_operational_restore(prepared, live_fence).expect("commit")
    }

    fn committed() -> CommittedOperationalRestore {
        committed_with(7, fence())
    }

    #[test]
    fn begin_owns_exact_canonical_obligation_set() {
        let expected = decisions()
            .iter()
            .map(|decision| canonical_plan_for_decision(*decision).unwrap().actions().len())
            .sum::<usize>();
        let session = RestoreExecutionSession::begin(committed()).expect("session");
        assert_eq!(session.state(), RestoreExecutionSessionState::Open);
        assert_eq!(session.remaining_permits(), expected);
        assert_eq!(session.receipt_count(), 0);
    }

    #[test]
    fn operator_replay_execution_consumes_permit_and_stores_receipt_internally() {
        let mut session = RestoreExecutionSession::begin(committed()).expect("session");
        let before = session.remaining_permits();
        let mut live = OperatorAuthority::default();
        let checkpoint = OperatorAuthority::default();
        session
            .execute_operator_replay_merge(&mut live, &checkpoint)
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
    }

    #[test]
    fn duplicate_executor_attempt_aborts_and_destroys_remaining_authority() {
        let mut session = RestoreExecutionSession::begin(committed()).expect("session");
        let mut live = OperatorAuthority::default();
        let checkpoint = OperatorAuthority::default();
        session
            .execute_operator_replay_merge(&mut live, &checkpoint)
            .expect("first execution");
        assert_eq!(
            session
                .execute_operator_replay_merge(&mut live, &checkpoint)
                .err(),
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
        let checkpoint = OperatorAuthority::default();
        session
            .execute_operator_replay_merge(&mut live, &checkpoint)
            .expect("first execution");
        let _ = session.execute_operator_replay_merge(&mut live, &checkpoint);
        assert_eq!(
            session
                .execute_operator_replay_merge(&mut live, &checkpoint)
                .err(),
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
        let checkpoint = OperatorAuthority::default();
        session
            .execute_operator_replay_merge(&mut live, &checkpoint)
            .expect("first execution");
        let _ = session.execute_operator_replay_merge(&mut live, &checkpoint);
        assert_eq!(session.finish().err(), Some(RestoreFinishError::Aborted));
    }

    #[test]
    fn same_checkpoint_under_different_live_fence_creates_different_session_binding() {
        let first = RestoreExecutionSession::begin(committed_with(7, fence())).expect("first");
        let second = RestoreExecutionSession::begin(committed_with(7, alternate_fence())).expect("second");
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
