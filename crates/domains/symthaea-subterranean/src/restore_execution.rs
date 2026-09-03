// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Receipt-complete execution contract for committed operational restore.
//!
//! A committed restore is not activation authority by itself. Every typed
//! restore action must execute against the exact committed generation binding
//! and produce one owner-minted receipt before activation may proceed.

use super::restore_actions::{
    canonical_plan_for_decision, RestoreAction, RestoreDomainPlan, RestorePlanError,
};
use super::restore_admission::{CommittedOperationalRestore, RestoreDigest, RestoreGenerationFence};
use super::restore_semantics::RestoreDomain;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RestoreActionReceipt {
    binding: RestoreExecutionBinding,
    domain: RestoreDomain,
    action: RestoreAction,
    outcome: RestoreActionOutcome,
}

impl RestoreActionReceipt {
    pub(super) const fn new(
        binding: RestoreExecutionBinding,
        domain: RestoreDomain,
        action: RestoreAction,
        outcome: RestoreActionOutcome,
    ) -> Self {
        Self {
            binding,
            domain,
            action,
            outcome,
        }
    }
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
    for receipt in receipts {
        if receipt.binding != binding {
            return Err(RestoreExecutionError::WrongBinding {
                domain: receipt.domain,
                action: receipt.action,
            });
        }
        if receipt.domain != plan.domain() || !plan.actions().contains(&receipt.action) {
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
/// provide only receipts; they cannot choose or replace restore semantics after
/// commit. All plans are derived before any receipt is evaluated so an
/// under-specified domain fails for its semantic reason rather than because an
/// earlier audited domain happens to lack a receipt.
pub(super) fn validate_restore_execution(
    committed: &CommittedOperationalRestore,
    receipts: &[RestoreActionReceipt],
) -> Result<(), RestoreExecutionError> {
    let binding = RestoreExecutionBinding::from_committed(committed);
    let plans = committed
        .decisions()
        .iter()
        .map(|decision| {
            canonical_plan_for_decision(*decision).map_err(|error| {
                RestoreExecutionError::PlanUnavailable {
                    domain: decision.domain(),
                    error,
                }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    for plan in &plans {
        let domain_receipts = receipts
            .iter()
            .copied()
            .filter(|receipt| receipt.domain == plan.domain())
            .collect::<Vec<_>>();
        validate_receipts_for_plan(binding, plan, &domain_receipts)?;
    }

    for receipt in receipts {
        if !plans.iter().any(|plan| plan.domain() == receipt.domain) {
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
    use crate::operational_checkpoint::restore_actions::{
        canonical_plan_for_decision, EvidenceRestorePolicy,
    };
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

    fn operator_receipts(binding: RestoreExecutionBinding) -> Vec<RestoreActionReceipt> {
        vec![
            RestoreActionReceipt::new(
                binding,
                RestoreDomain::OperatorAuthority,
                RestoreAction::PreserveOrNarrowAuthority,
                RestoreActionOutcome::Applied,
            ),
            RestoreActionReceipt::new(
                binding,
                RestoreDomain::OperatorAuthority,
                RestoreAction::MergeEvidence(EvidenceRestorePolicy::ReplayBarrier),
                RestoreActionOutcome::Applied,
            ),
            RestoreActionReceipt::new(
                binding,
                RestoreDomain::OperatorAuthority,
                RestoreAction::ReconcileBeforeActivation,
                RestoreActionOutcome::Reconciled,
            ),
            RestoreActionReceipt::new(
                binding,
                RestoreDomain::OperatorAuthority,
                RestoreAction::DropEphemeral,
                RestoreActionOutcome::Dropped,
            ),
        ]
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
    fn duplicate_receipt_is_rejected() {
        let committed = committed();
        let binding = RestoreExecutionBinding::from_committed(&committed);
        let mut receipts = operator_receipts(binding);
        receipts.push(receipts[0]);
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
        receipts[3] = RestoreActionReceipt::new(
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
    fn full_activation_stays_blocked_at_first_unaudited_domain() {
        let committed = committed();
        assert_eq!(
            validate_restore_execution(&committed, &[]),
            Err(RestoreExecutionError::PlanUnavailable {
                domain: RestoreDomain::ActuatorIsolation,
                error: RestorePlanError::EvidencePolicyUnderspecified {
                    domain: RestoreDomain::ActuatorIsolation,
                },
            })
        );
    }
}
