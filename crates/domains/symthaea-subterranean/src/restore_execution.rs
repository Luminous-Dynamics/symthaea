// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Receipt-complete execution contract for committed operational restore.
//!
//! A committed restore is not activation authority by itself. Every typed
//! restore action must execute against the exact committed generation binding
//! and produce one owner-minted receipt before activation may proceed.

use super::restore_actions::{RestoreAction, RestoreDomainPlan};
use super::restore_admission::{CommittedOperationalRestore, RestoreDigest, RestoreGenerationFence};
use super::restore_semantics::RestoreDomain;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RestoreExecutionBinding {
    checkpoint_digest: RestoreDigest,
    fence: RestoreGenerationFence,
}

impl RestoreExecutionBinding {
    pub(super) const fn new(
        checkpoint_digest: RestoreDigest,
        fence: RestoreGenerationFence,
    ) -> Self {
        Self {
            checkpoint_digest,
            fence,
        }
    }

    pub(super) fn from_committed(committed: &CommittedOperationalRestore) -> Self {
        Self::new(committed.checkpoint_digest(), committed.fence())
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
    MissingPlan(RestoreDomain),
    DuplicatePlan(RestoreDomain),
    VerdictMismatch(RestoreDomain),
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

fn validate_plan_coverage(
    committed: &CommittedOperationalRestore,
    plans: &[RestoreDomainPlan],
) -> Result<(), RestoreExecutionError> {
    for decision in committed.decisions() {
        let mut matching = plans.iter().filter(|plan| plan.domain() == decision.domain());
        let Some(plan) = matching.next() else {
            return Err(RestoreExecutionError::MissingPlan(decision.domain()));
        };
        if matching.next().is_some() {
            return Err(RestoreExecutionError::DuplicatePlan(decision.domain()));
        }
        if plan.verdict() != decision.verdict() {
            return Err(RestoreExecutionError::VerdictMismatch(decision.domain()));
        }
    }
    Ok(())
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
/// This function intentionally fails until every committed domain has an
/// action-complete RA-20 plan. Several domains remain fail-closed while their
/// evidence polarity is still being audited, so there is deliberately no
/// synthetic all-green fixture in this tranche.
pub(super) fn validate_restore_execution(
    committed: &CommittedOperationalRestore,
    plans: &[RestoreDomainPlan],
    receipts: &[RestoreActionReceipt],
) -> Result<(), RestoreExecutionError> {
    validate_plan_coverage(committed, plans)?;
    let binding = RestoreExecutionBinding::from_committed(committed);

    for plan in plans {
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
    use crate::operational_checkpoint::restore_actions::EvidenceRestorePolicy;
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

    fn decisions() -> Vec<RestoreDomainDecision> {
        OPERATIONAL_RESTORE_CONTRACTS
            .iter()
            .map(|contract| {
                let verdict = if matches!(
                    contract.domain,
                    RestoreDomain::Controller
                        | RestoreDomain::Mission
                        | RestoreDomain::OperatorAuthority
                ) {
                    RestoreAdmissionVerdict::ReconciliationRequired
                } else {
                    RestoreAdmissionVerdict::ProvenNonWidening
                };
                RestoreDomainDecision::new(contract.domain, verdict)
            })
            .collect()
    }

    fn committed() -> CommittedOperationalRestore {
        let prepared = prepare_operational_restore(
            RestorePreparationContext::new(digest(7), fence()),
            decisions(),
        )
        .expect("prepare");
        commit_operational_restore(prepared, fence()).expect("commit")
    }

    fn historical_reconcile_plan(domain: RestoreDomain) -> RestoreDomainPlan {
        RestoreDomainPlan::new(
            RestoreDomainDecision::new(domain, RestoreAdmissionVerdict::ReconciliationRequired),
            vec![
                RestoreAction::ReplaceValidatedHistorical,
                RestoreAction::ReconcileBeforeActivation,
            ],
        )
        .expect("historical reconcile plan")
    }

    fn operator_plan() -> RestoreDomainPlan {
        RestoreDomainPlan::new(
            RestoreDomainDecision::new(
                RestoreDomain::OperatorAuthority,
                RestoreAdmissionVerdict::ReconciliationRequired,
            ),
            vec![
                RestoreAction::PreserveOrNarrowAuthority,
                RestoreAction::MergeEvidence(EvidenceRestorePolicy::ReplayBarrier),
                RestoreAction::ReconcileBeforeActivation,
                RestoreAction::DropEphemeral,
            ],
        )
        .expect("operator plan")
    }

    fn audited_prefix_plans() -> Vec<RestoreDomainPlan> {
        vec![
            historical_reconcile_plan(RestoreDomain::Controller),
            historical_reconcile_plan(RestoreDomain::Mission),
            operator_plan(),
        ]
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
    fn receipt_from_other_commit_binding_is_rejected() {
        let committed = committed();
        let binding = RestoreExecutionBinding::from_committed(&committed);
        let wrong = RestoreExecutionBinding::new(digest(99), fence());
        assert!(matches!(
            validate_receipts_for_plan(wrong, &operator_plan(), &operator_receipts(binding)),
            Err(RestoreExecutionError::WrongBinding { .. })
        ));
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
    fn partial_audited_plan_set_cannot_activate_full_restore() {
        let committed = committed();
        assert_eq!(
            validate_restore_execution(&committed, &audited_prefix_plans(), &[]),
            Err(RestoreExecutionError::MissingPlan(
                RestoreDomain::DegradedSupervisor
            ))
        );
    }

    #[test]
    fn duplicate_domain_plan_is_rejected_before_execution() {
        let committed = committed();
        let mut plans = audited_prefix_plans();
        plans.insert(1, historical_reconcile_plan(RestoreDomain::Controller));
        assert_eq!(
            validate_restore_execution(&committed, &plans, &[]),
            Err(RestoreExecutionError::DuplicatePlan(
                RestoreDomain::Controller
            ))
        );
    }
}
