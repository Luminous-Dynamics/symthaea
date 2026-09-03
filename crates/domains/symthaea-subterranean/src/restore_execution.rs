// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Receipt-complete execution contract for committed operational restore.
//!
//! A committed restore is not activation authority by itself. Every typed
//! restore action must execute against the exact committed generation binding
//! and produce one executor-authenticated receipt before activation may proceed.

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
    /// Bind execution only to an actual committed restore transaction. Raw
    /// digest/fence construction stays unavailable so sibling modules cannot
    /// self-mint a successful execution context.
    pub(super) fn from_committed(committed: &CommittedOperationalRestore) -> Self {
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
/// generic production constructor. A receipt may be created only by an executor
/// path that consumes the matching affine `RestoreActionPermit` and performs the
/// corresponding mutation/requalification before returning success.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct RestoreActionReceipt {
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
/// Deliberately not `Clone`, `Copy`, `Serialize` or `Deserialize`. Fields are
/// private and raw construction is unavailable outside this module.
#[derive(Debug)]
pub(super) struct RestoreActionPermit {
    binding: RestoreExecutionBinding,
    domain: RestoreDomain,
    action: RestoreAction,
}

impl RestoreActionPermit {
    pub(super) const fn domain(&self) -> RestoreDomain {
        self.domain
    }

    pub(super) const fn action(&self) -> RestoreAction {
        self.action
    }

    /// Execute the first concrete receipt-bearing restore action.
    ///
    /// This method is intentionally action-specific. It consumes the permit,
    /// checks the exact canonical obligation, performs the owner-local operator
    /// replay merge, and only then constructs the success receipt. A mismatched
    /// permit is consumed fail-closed and cannot mutate operator state.
    pub(super) fn execute_operator_replay_merge(
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

/// Transaction-owned set of still-unconsumed restore action permits.
///
/// A permit disappears from this set when taken. The type is affine and has no
/// serialization surface, so a checkpoint cannot deserialize into execution
/// authority and the same obligation cannot be issued twice by this set.
pub(super) struct RestoreExecutionPermitSet {
    binding: RestoreExecutionBinding,
    permits: Vec<RestoreActionPermit>,
}

impl RestoreExecutionPermitSet {
    pub(super) fn remaining(&self) -> usize {
        self.permits.len()
    }

    pub(super) fn take(
        &mut self,
        domain: RestoreDomain,
        action: RestoreAction,
    ) -> Result<RestoreActionPermit, RestorePermitError> {
        let Some(index) = self
            .permits
            .iter()
            .position(|permit| permit.domain == domain && permit.action == action)
        else {
            return Err(RestorePermitError::MissingPermit { domain, action });
        };
        Ok(self.permits.remove(index))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RestorePermitError {
    PlanUnavailable {
        domain: RestoreDomain,
        error: RestorePlanError,
    },
    MissingPermit {
        domain: RestoreDomain,
        action: RestoreAction,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RestoreExecutionError {
    PlanUnavailable {
        domain: RestoreDomain,
        error: RestorePlanError,
    },
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

fn canonical_plans_for_committed(
    committed: &CommittedOperationalRestore,
) -> Result<Vec<RestoreDomainPlan>, RestorePlanError> {
    committed
        .decisions()
        .iter()
        .map(|decision| canonical_plan_for_decision(*decision))
        .collect()
}

/// Issue the complete affine action-permit set for one committed restore.
///
/// The caller does not supply domains or actions. Every permit is derived from
/// the committed decisions and canonical RA-20/RA-24 plans.
pub(super) fn issue_restore_action_permits(
    committed: &CommittedOperationalRestore,
) -> Result<RestoreExecutionPermitSet, RestorePermitError> {
    let binding = RestoreExecutionBinding::from_committed(committed);
    let plans = canonical_plans_for_committed(committed).map_err(|error| {
        let domain = match error {
            RestorePlanError::DecisionNotAdmissible { domain, .. }
            | RestorePlanError::DuplicateAction { domain, .. }
            | RestorePlanError::MissingSemanticAction { domain, .. }
            | RestorePlanError::MissingVerdictAction { domain, .. }
            | RestorePlanError::UnexpectedAction { domain, .. }
            | RestorePlanError::EvidencePolicyUnderspecified { domain }
            | RestorePlanError::MissingEvidencePolicy { domain, .. }
            | RestorePlanError::UnexpectedEvidencePolicy { domain, .. }
            | RestorePlanError::VerdictActionMismatch { domain, .. } => domain,
        };
        RestorePermitError::PlanUnavailable { domain, error }
    })?;

    let permits = plans
        .iter()
        .flat_map(|plan| {
            plan.actions().iter().copied().map(move |action| RestoreActionPermit {
                binding,
                domain: plan.domain(),
                action,
            })
        })
        .collect();

    Ok(RestoreExecutionPermitSet { binding, permits })
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

/// Activation gate for a complete restore transaction.
///
/// Plans are derived canonically from the committed RA-19 decisions. Callers
/// provide only executor-produced receipts; they cannot choose restore semantics
/// or mint success evidence through this module.
pub(super) fn validate_restore_execution(
    committed: &CommittedOperationalRestore,
    receipts: &[RestoreActionReceipt],
) -> Result<(), RestoreExecutionError> {
    let binding = RestoreExecutionBinding::from_committed(committed);
    let plans = canonical_plans_for_committed(committed).map_err(|error| {
        let domain = match error {
            RestorePlanError::DecisionNotAdmissible { domain, .. }
            | RestorePlanError::DuplicateAction { domain, .. }
            | RestorePlanError::MissingSemanticAction { domain, .. }
            | RestorePlanError::MissingVerdictAction { domain, .. }
            | RestorePlanError::UnexpectedAction { domain, .. }
            | RestorePlanError::EvidencePolicyUnderspecified { domain }
            | RestorePlanError::MissingEvidencePolicy { domain, .. }
            | RestorePlanError::UnexpectedEvidencePolicy { domain, .. }
            | RestorePlanError::VerdictActionMismatch { domain, .. } => domain,
        };
        RestoreExecutionError::PlanUnavailable { domain, error }
    })?;

    for plan in &plans {
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
    use crate::operational_checkpoint::restore_actions::canonical_plan_for_decision;
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

    fn operator_plan() -> RestoreDomainPlan {
        canonical_plan_for_decision(RestoreDomainDecision::new(
            RestoreDomain::OperatorAuthority,
            RestoreAdmissionVerdict::ReconciliationRequired,
        ))
        .expect("operator canonical plan")
    }

    fn test_receipt(
        binding: RestoreExecutionBinding,
        domain: RestoreDomain,
        action: RestoreAction,
        outcome: RestoreActionOutcome,
    ) -> RestoreActionReceipt {
        RestoreActionReceipt {
            binding,
            domain,
            action,
            outcome,
        }
    }

    fn operator_receipts(binding: RestoreExecutionBinding) -> Vec<RestoreActionReceipt> {
        vec![
            test_receipt(
                binding,
                RestoreDomain::OperatorAuthority,
                RestoreAction::PreserveOrNarrowAuthority,
                RestoreActionOutcome::Applied,
            ),
            test_receipt(
                binding,
                RestoreDomain::OperatorAuthority,
                RestoreAction::MergeEvidence(EvidenceRestorePolicy::ReplayBarrier),
                RestoreActionOutcome::Applied,
            ),
            test_receipt(
                binding,
                RestoreDomain::OperatorAuthority,
                RestoreAction::ReconcileBeforeActivation,
                RestoreActionOutcome::Reconciled,
            ),
            test_receipt(
                binding,
                RestoreDomain::OperatorAuthority,
                RestoreAction::DropEphemeral,
                RestoreActionOutcome::Dropped,
            ),
        ]
    }

    #[test]
    fn permit_set_exactly_matches_canonical_obligations() {
        let committed = committed();
        let expected = committed
            .decisions()
            .iter()
            .map(|decision| canonical_plan_for_decision(*decision).unwrap().actions().len())
            .sum::<usize>();
        let permits = issue_restore_action_permits(&committed).expect("canonical permits");
        assert_eq!(permits.remaining(), expected);
        assert_eq!(permits.binding, RestoreExecutionBinding::from_committed(&committed));
    }

    #[test]
    fn taking_permit_consumes_exact_obligation_once() {
        let committed = committed();
        let mut permits = issue_restore_action_permits(&committed).expect("canonical permits");
        let before = permits.remaining();
        let permit = permits
            .take(
                RestoreDomain::OperatorAuthority,
                RestoreAction::MergeEvidence(EvidenceRestorePolicy::ReplayBarrier),
            )
            .expect("operator replay permit");
        assert_eq!(permit.domain(), RestoreDomain::OperatorAuthority);
        assert_eq!(
            permit.action(),
            RestoreAction::MergeEvidence(EvidenceRestorePolicy::ReplayBarrier)
        );
        assert_eq!(permit.binding, RestoreExecutionBinding::from_committed(&committed));
        assert_eq!(permits.remaining(), before - 1);
        assert_eq!(
            permits
                .take(
                    RestoreDomain::OperatorAuthority,
                    RestoreAction::MergeEvidence(EvidenceRestorePolicy::ReplayBarrier),
                )
                .err(),
            Some(RestorePermitError::MissingPermit {
                domain: RestoreDomain::OperatorAuthority,
                action: RestoreAction::MergeEvidence(EvidenceRestorePolicy::ReplayBarrier),
            })
        );
    }

    #[test]
    fn action_not_in_canonical_plan_has_no_permit() {
        let committed = committed();
        let mut permits = issue_restore_action_permits(&committed).expect("canonical permits");
        assert_eq!(
            permits
                .take(RestoreDomain::Controller, RestoreAction::DropEphemeral)
                .err(),
            Some(RestorePermitError::MissingPermit {
                domain: RestoreDomain::Controller,
                action: RestoreAction::DropEphemeral,
            })
        );
    }

    #[test]
    fn operator_replay_executor_mints_receipt_only_after_owner_merge_path() {
        let committed = committed();
        let binding = RestoreExecutionBinding::from_committed(&committed);
        let mut permits = issue_restore_action_permits(&committed).expect("canonical permits");
        let permit = permits
            .take(
                RestoreDomain::OperatorAuthority,
                RestoreAction::MergeEvidence(EvidenceRestorePolicy::ReplayBarrier),
            )
            .expect("operator replay permit");
        let mut live = OperatorAuthority::default();
        let checkpoint = OperatorAuthority::default();
        let receipt = permit
            .execute_operator_replay_merge(&mut live, &checkpoint)
            .expect("matching executor");
        assert_eq!(receipt.binding, binding);
        assert_eq!(receipt.domain, RestoreDomain::OperatorAuthority);
        assert_eq!(
            receipt.action,
            RestoreAction::MergeEvidence(EvidenceRestorePolicy::ReplayBarrier)
        );
        assert_eq!(receipt.outcome, RestoreActionOutcome::Applied);
    }

    #[test]
    fn wrong_permit_cannot_mint_operator_replay_receipt() {
        let committed = committed();
        let mut permits = issue_restore_action_permits(&committed).expect("canonical permits");
        let wrong = permits
            .take(RestoreDomain::Controller, RestoreAction::ReplaceValidatedHistorical)
            .expect("controller permit");
        let mut live = OperatorAuthority::default();
        let checkpoint = OperatorAuthority::default();
        assert_eq!(
            wrong
                .execute_operator_replay_merge(&mut live, &checkpoint)
                .err(),
            Some(RestoreExecutorError::PermitMismatch {
                expected_domain: RestoreDomain::OperatorAuthority,
                expected_action: RestoreAction::MergeEvidence(
                    EvidenceRestorePolicy::ReplayBarrier,
                ),
                actual_domain: RestoreDomain::Controller,
                actual_action: RestoreAction::ReplaceValidatedHistorical,
            })
        );
    }

    #[test]
    fn audited_operator_receipts_validate_against_exact_commit_binding() {
        let committed = committed();
        let binding = RestoreExecutionBinding::from_committed(&committed);
        assert_eq!(
            validate_receipts_for_plan(binding, &operator_plan(), &operator_receipts(binding)),
            Ok(())
        );
    }

    #[test]
    fn missing_action_receipt_is_rejected() {
        let committed = committed();
        let binding = RestoreExecutionBinding::from_committed(&committed);
        let mut receipts = operator_receipts(binding);
        receipts.pop();
        assert!(matches!(
            validate_receipts_for_plan(binding, &operator_plan(), &receipts),
            Err(RestoreExecutionError::MissingReceipt { .. })
        ));
    }

    #[test]
    fn receipt_from_other_committed_context_is_rejected() {
        let committed = committed();
        let other = committed_with(99, fence());
        let expected = RestoreExecutionBinding::from_committed(&committed);
        let wrong = RestoreExecutionBinding::from_committed(&other);
        assert_ne!(expected, wrong);
        assert!(matches!(
            validate_receipts_for_plan(expected, &operator_plan(), &operator_receipts(wrong)),
            Err(RestoreExecutionError::WrongBinding { .. })
        ));
    }

    #[test]
    fn same_checkpoint_under_different_live_fence_is_different_transaction() {
        let first = committed_with(7, fence());
        let second = committed_with(7, alternate_fence());
        assert_eq!(first.checkpoint_digest(), second.checkpoint_digest());
        assert_ne!(
            RestoreExecutionBinding::from_committed(&first),
            RestoreExecutionBinding::from_committed(&second)
        );
    }

    #[test]
    fn permit_from_other_committed_context_carries_other_binding() {
        let first = committed_with(7, fence());
        let second = committed_with(7, alternate_fence());
        let mut first_permits = issue_restore_action_permits(&first).expect("first permits");
        let mut second_permits = issue_restore_action_permits(&second).expect("second permits");
        let first_permit = first_permits
            .take(RestoreDomain::Controller, RestoreAction::ReplaceValidatedHistorical)
            .unwrap();
        let second_permit = second_permits
            .take(RestoreDomain::Controller, RestoreAction::ReplaceValidatedHistorical)
            .unwrap();
        assert_ne!(first_permit.binding, second_permit.binding);
    }

    #[test]
    fn duplicate_receipt_is_rejected() {
        let committed = committed();
        let binding = RestoreExecutionBinding::from_committed(&committed);
        let mut receipts = operator_receipts(binding);
        receipts.push(test_receipt(
            binding,
            RestoreDomain::OperatorAuthority,
            RestoreAction::PreserveOrNarrowAuthority,
            RestoreActionOutcome::Applied,
        ));
        assert!(matches!(
            validate_receipts_for_plan(binding, &operator_plan(), &receipts),
            Err(RestoreExecutionError::DuplicateReceipt { .. })
        ));
    }

    #[test]
    fn action_requires_semantically_matching_outcome() {
        let committed = committed();
        let binding = RestoreExecutionBinding::from_committed(&committed);
        let mut receipts = operator_receipts(binding);
        receipts[3] = test_receipt(
            binding,
            RestoreDomain::OperatorAuthority,
            RestoreAction::DropEphemeral,
            RestoreActionOutcome::Applied,
        );
        assert!(matches!(
            validate_receipts_for_plan(binding, &operator_plan(), &receipts),
            Err(RestoreExecutionError::OutcomeMismatch { .. })
        ));
    }

    #[test]
    fn complete_semantic_planning_still_requires_execution_receipts() {
        let committed = committed();
        assert_eq!(
            validate_restore_execution(&committed, &[]),
            Err(RestoreExecutionError::MissingReceipt {
                domain: RestoreDomain::Controller,
                action: RestoreAction::ReplaceValidatedHistorical,
            })
        );
    }
}
