// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Action-complete restore planning and evidence-polarity contracts.
//!
//! RA-17 says *which semantic obligations* each checkpoint domain carries.
//! RA-19 says *whether* a proposed restore can proceed and binds the decision to
//! an exact live generation. This module closes the translation gap: a domain
//! plan is valid only when its typed actions discharge every registered restore
//! semantic, unsupported actions are absent, and evidence merge uses an
//! explicitly audited polarity.

use super::restore_admission::{RestoreAdmissionVerdict, RestoreDomainDecision};
use super::restore_semantics::{contract_for, RestoreDomain, RestoreSemantics};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum EvidenceRestorePolicy {
    /// Replay/sequence barriers may only move forward per identity/source.
    ReplayBarrier,
    /// Adverse evidence that justifies restriction must not be rolled backward.
    RestrictionSupporting,
    /// Known contradictions/counterexamples survive restore.
    CounterexamplePreserving,
    /// Evidence that supports widening/recovery must be re-earned from fresh
    /// observations rather than resurrected from a stale checkpoint.
    RecoverySupportingFreshOnly,
    /// Diagnostic history that is explicitly non-authoritative.
    NeutralHistory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum RestoreAction {
    ReplaceValidatedHistorical,
    PreserveOrNarrowAuthority,
    MergeEvidence(EvidenceRestorePolicy),
    RequalifyFromCurrentInputs,
    ReconcileBeforeActivation,
    DropEphemeral,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RestoreDomainPlan {
    decision: RestoreDomainDecision,
    actions: Vec<RestoreAction>,
}

impl RestoreDomainPlan {
    pub(super) fn new(
        decision: RestoreDomainDecision,
        actions: Vec<RestoreAction>,
    ) -> Result<Self, RestorePlanError> {
        validate_actions(decision, actions)
    }

    pub(super) const fn domain(&self) -> RestoreDomain {
        self.decision.domain()
    }

    pub(super) const fn verdict(&self) -> RestoreAdmissionVerdict {
        self.decision.verdict()
    }

    pub(super) fn actions(&self) -> &[RestoreAction] {
        &self.actions
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RestorePlanError {
    DecisionNotAdmissible {
        domain: RestoreDomain,
        verdict: RestoreAdmissionVerdict,
    },
    DuplicateAction {
        domain: RestoreDomain,
        action: RestoreAction,
    },
    MissingSemanticAction {
        domain: RestoreDomain,
        semantic: RestoreSemantics,
    },
    MissingVerdictAction {
        domain: RestoreDomain,
        verdict: RestoreAdmissionVerdict,
        action: RestoreAction,
    },
    UnexpectedAction {
        domain: RestoreDomain,
        action: RestoreAction,
    },
    EvidencePolicyUnderspecified {
        domain: RestoreDomain,
    },
    MissingEvidencePolicy {
        domain: RestoreDomain,
        policy: EvidenceRestorePolicy,
    },
    UnexpectedEvidencePolicy {
        domain: RestoreDomain,
        policy: EvidenceRestorePolicy,
    },
    VerdictActionMismatch {
        domain: RestoreDomain,
        verdict: RestoreAdmissionVerdict,
        action: RestoreAction,
    },
}

fn required_evidence_policies(domain: RestoreDomain) -> Option<&'static [EvidenceRestorePolicy]> {
    match domain {
        RestoreDomain::OperatorAuthority => Some(&[EvidenceRestorePolicy::ReplayBarrier]),
        RestoreDomain::DegradedSupervisor => Some(&[
            EvidenceRestorePolicy::RestrictionSupporting,
            EvidenceRestorePolicy::RecoverySupportingFreshOnly,
        ]),
        RestoreDomain::SensorFusion
        | RestoreDomain::ActuatorIsolation
        | RestoreDomain::PartitionRecovery
        | RestoreDomain::TemporalAssurance => None,
        RestoreDomain::Controller
        | RestoreDomain::Mission
        | RestoreDomain::UpdateManager
        | RestoreDomain::FieldEnvelope => Some(&[]),
    }
}

fn has_action(actions: &[RestoreAction], target: RestoreAction) -> bool {
    actions.contains(&target)
}

fn has_any_evidence_action(actions: &[RestoreAction]) -> bool {
    actions
        .iter()
        .any(|action| matches!(action, RestoreAction::MergeEvidence(_)))
}

fn semantic_action_present(semantic: RestoreSemantics, actions: &[RestoreAction]) -> bool {
    match semantic {
        RestoreSemantics::HistoricalReplace => {
            has_action(actions, RestoreAction::ReplaceValidatedHistorical)
        }
        RestoreSemantics::AuthorityMonotone => {
            has_action(actions, RestoreAction::PreserveOrNarrowAuthority)
        }
        RestoreSemantics::EvidenceMerge => has_any_evidence_action(actions),
        RestoreSemantics::DerivedRequalify => {
            has_action(actions, RestoreAction::RequalifyFromCurrentInputs)
        }
        RestoreSemantics::TransitionReconcile => {
            has_action(actions, RestoreAction::ReconcileBeforeActivation)
        }
        RestoreSemantics::EphemeralDrop => {
            has_action(actions, RestoreAction::DropEphemeral)
        }
    }
}

fn action_supported(
    semantic_set: &[RestoreSemantics],
    verdict: RestoreAdmissionVerdict,
    action: RestoreAction,
) -> bool {
    match action {
        RestoreAction::ReplaceValidatedHistorical => {
            semantic_set.contains(&RestoreSemantics::HistoricalReplace)
        }
        RestoreAction::PreserveOrNarrowAuthority => {
            semantic_set.contains(&RestoreSemantics::AuthorityMonotone)
        }
        RestoreAction::MergeEvidence(_) => {
            semantic_set.contains(&RestoreSemantics::EvidenceMerge)
        }
        RestoreAction::RequalifyFromCurrentInputs => {
            semantic_set.contains(&RestoreSemantics::DerivedRequalify)
        }
        RestoreAction::ReconcileBeforeActivation => {
            semantic_set.contains(&RestoreSemantics::TransitionReconcile)
                || verdict == RestoreAdmissionVerdict::ReconciliationRequired
        }
        RestoreAction::DropEphemeral => {
            semantic_set.contains(&RestoreSemantics::EphemeralDrop)
        }
    }
}

fn validate_verdict_action_consistency(
    domain: RestoreDomain,
    verdict: RestoreAdmissionVerdict,
    actions: &[RestoreAction],
) -> Result<(), RestorePlanError> {
    if matches!(
        verdict,
        RestoreAdmissionVerdict::Widening | RestoreAdmissionVerdict::NotProvable
    ) {
        return Err(RestorePlanError::DecisionNotAdmissible { domain, verdict });
    }

    match verdict {
        RestoreAdmissionVerdict::ReconciliationRequired => {
            if !has_action(actions, RestoreAction::ReconcileBeforeActivation) {
                return Err(RestorePlanError::MissingVerdictAction {
                    domain,
                    verdict,
                    action: RestoreAction::ReconcileBeforeActivation,
                });
            }
        }
        RestoreAdmissionVerdict::ConservativeRequalification => {
            if !has_action(actions, RestoreAction::RequalifyFromCurrentInputs) {
                return Err(RestorePlanError::MissingVerdictAction {
                    domain,
                    verdict,
                    action: RestoreAction::RequalifyFromCurrentInputs,
                });
            }
        }
        RestoreAdmissionVerdict::ProvenNonWidening => {}
        RestoreAdmissionVerdict::Widening | RestoreAdmissionVerdict::NotProvable => unreachable!(),
    }

    if has_action(actions, RestoreAction::ReconcileBeforeActivation)
        && verdict != RestoreAdmissionVerdict::ReconciliationRequired
    {
        return Err(RestorePlanError::VerdictActionMismatch {
            domain,
            verdict,
            action: RestoreAction::ReconcileBeforeActivation,
        });
    }

    if has_action(actions, RestoreAction::RequalifyFromCurrentInputs)
        && !matches!(
            verdict,
            RestoreAdmissionVerdict::ConservativeRequalification
                | RestoreAdmissionVerdict::ReconciliationRequired
        )
    {
        return Err(RestorePlanError::VerdictActionMismatch {
            domain,
            verdict,
            action: RestoreAction::RequalifyFromCurrentInputs,
        });
    }
    Ok(())
}

fn validate_actions(
    decision: RestoreDomainDecision,
    mut actions: Vec<RestoreAction>,
) -> Result<RestoreDomainPlan, RestorePlanError> {
    let domain = decision.domain();
    let verdict = decision.verdict();

    validate_verdict_action_consistency(domain, verdict, &actions)?;

    actions.sort_unstable();
    for pair in actions.windows(2) {
        if pair[0] == pair[1] {
            return Err(RestorePlanError::DuplicateAction {
                domain,
                action: pair[0],
            });
        }
    }

    let contract = contract_for(domain);
    for action in &actions {
        if !action_supported(contract.semantics, verdict, *action) {
            return Err(RestorePlanError::UnexpectedAction {
                domain,
                action: *action,
            });
        }
    }

    for semantic in contract.semantics {
        if !semantic_action_present(*semantic, &actions) {
            return Err(RestorePlanError::MissingSemanticAction {
                domain,
                semantic: *semantic,
            });
        }
    }

    if contract.semantics.contains(&RestoreSemantics::EvidenceMerge) {
        let Some(required) = required_evidence_policies(domain) else {
            return Err(RestorePlanError::EvidencePolicyUnderspecified { domain });
        };
        for policy in required {
            if !has_action(&actions, RestoreAction::MergeEvidence(*policy)) {
                return Err(RestorePlanError::MissingEvidencePolicy {
                    domain,
                    policy: *policy,
                });
            }
        }
        for action in &actions {
            if let RestoreAction::MergeEvidence(policy) = action {
                if !required.contains(policy) {
                    return Err(RestorePlanError::UnexpectedEvidencePolicy {
                        domain,
                        policy: *policy,
                    });
                }
            }
        }
    } else if let Some(RestoreAction::MergeEvidence(policy)) = actions
        .iter()
        .find(|action| matches!(action, RestoreAction::MergeEvidence(_)))
        .copied()
    {
        return Err(RestorePlanError::UnexpectedEvidencePolicy { domain, policy });
    }

    Ok(RestoreDomainPlan { decision, actions })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decision(
        domain: RestoreDomain,
        verdict: RestoreAdmissionVerdict,
    ) -> RestoreDomainDecision {
        RestoreDomainDecision::new(domain, verdict)
    }

    #[test]
    fn operator_plan_must_discharge_all_obligations_and_reconciliation_barrier() {
        let plan = RestoreDomainPlan::new(
            decision(
                RestoreDomain::OperatorAuthority,
                RestoreAdmissionVerdict::ReconciliationRequired,
            ),
            vec![
                RestoreAction::PreserveOrNarrowAuthority,
                RestoreAction::MergeEvidence(EvidenceRestorePolicy::ReplayBarrier),
                RestoreAction::DropEphemeral,
                RestoreAction::ReconcileBeforeActivation,
            ],
        )
        .expect("complete operator plan");
        assert_eq!(plan.domain(), RestoreDomain::OperatorAuthority);
        assert_eq!(plan.verdict(), RestoreAdmissionVerdict::ReconciliationRequired);
        assert!(plan.actions().contains(&RestoreAction::DropEphemeral));
        assert!(
            plan.actions()
                .contains(&RestoreAction::ReconcileBeforeActivation)
        );
    }

    #[test]
    fn reconciliation_verdict_requires_non_activation_barrier() {
        assert_eq!(
            RestoreDomainPlan::new(
                decision(
                    RestoreDomain::OperatorAuthority,
                    RestoreAdmissionVerdict::ReconciliationRequired,
                ),
                vec![
                    RestoreAction::PreserveOrNarrowAuthority,
                    RestoreAction::MergeEvidence(EvidenceRestorePolicy::ReplayBarrier),
                    RestoreAction::DropEphemeral,
                ],
            )
            .err(),
            Some(RestorePlanError::MissingVerdictAction {
                domain: RestoreDomain::OperatorAuthority,
                verdict: RestoreAdmissionVerdict::ReconciliationRequired,
                action: RestoreAction::ReconcileBeforeActivation,
            })
        );
    }

    #[test]
    fn conservative_requalification_verdict_requires_requalification_action() {
        assert_eq!(
            RestoreDomainPlan::new(
                decision(
                    RestoreDomain::FieldEnvelope,
                    RestoreAdmissionVerdict::ConservativeRequalification,
                ),
                vec![],
            )
            .err(),
            Some(RestorePlanError::MissingVerdictAction {
                domain: RestoreDomain::FieldEnvelope,
                verdict: RestoreAdmissionVerdict::ConservativeRequalification,
                action: RestoreAction::RequalifyFromCurrentInputs,
            })
        );
    }

    #[test]
    fn operator_plan_cannot_omit_replay_merge() {
        assert_eq!(
            RestoreDomainPlan::new(
                decision(
                    RestoreDomain::OperatorAuthority,
                    RestoreAdmissionVerdict::ReconciliationRequired,
                ),
                vec![
                    RestoreAction::PreserveOrNarrowAuthority,
                    RestoreAction::DropEphemeral,
                    RestoreAction::ReconcileBeforeActivation,
                ],
            )
            .err(),
            Some(RestorePlanError::MissingSemanticAction {
                domain: RestoreDomain::OperatorAuthority,
                semantic: RestoreSemantics::EvidenceMerge,
            })
        );
    }

    #[test]
    fn operator_plan_rejects_wrong_evidence_polarity() {
        assert_eq!(
            RestoreDomainPlan::new(
                decision(
                    RestoreDomain::OperatorAuthority,
                    RestoreAdmissionVerdict::ReconciliationRequired,
                ),
                vec![
                    RestoreAction::PreserveOrNarrowAuthority,
                    RestoreAction::MergeEvidence(
                        EvidenceRestorePolicy::RecoverySupportingFreshOnly,
                    ),
                    RestoreAction::DropEphemeral,
                    RestoreAction::ReconcileBeforeActivation,
                ],
            )
            .err(),
            Some(RestorePlanError::MissingEvidencePolicy {
                domain: RestoreDomain::OperatorAuthority,
                policy: EvidenceRestorePolicy::ReplayBarrier,
            })
        );
    }

    #[test]
    fn operator_plan_rejects_unjustified_historical_replace() {
        assert_eq!(
            RestoreDomainPlan::new(
                decision(
                    RestoreDomain::OperatorAuthority,
                    RestoreAdmissionVerdict::ReconciliationRequired,
                ),
                vec![
                    RestoreAction::ReplaceValidatedHistorical,
                    RestoreAction::PreserveOrNarrowAuthority,
                    RestoreAction::MergeEvidence(EvidenceRestorePolicy::ReplayBarrier),
                    RestoreAction::DropEphemeral,
                    RestoreAction::ReconcileBeforeActivation,
                ],
            )
            .err(),
            Some(RestorePlanError::UnexpectedAction {
                domain: RestoreDomain::OperatorAuthority,
                action: RestoreAction::ReplaceValidatedHistorical,
            })
        );
    }

    #[test]
    fn degraded_plan_requires_adverse_and_fresh_recovery_polarities() {
        let plan = RestoreDomainPlan::new(
            decision(
                RestoreDomain::DegradedSupervisor,
                RestoreAdmissionVerdict::ProvenNonWidening,
            ),
            vec![
                RestoreAction::PreserveOrNarrowAuthority,
                RestoreAction::MergeEvidence(EvidenceRestorePolicy::RestrictionSupporting),
                RestoreAction::MergeEvidence(
                    EvidenceRestorePolicy::RecoverySupportingFreshOnly,
                ),
            ],
        )
        .expect("audited degraded plan");
        assert_eq!(plan.domain(), RestoreDomain::DegradedSupervisor);
    }

    #[test]
    fn degraded_plan_cannot_resurrect_recovery_credit_by_omission() {
        assert_eq!(
            RestoreDomainPlan::new(
                decision(
                    RestoreDomain::DegradedSupervisor,
                    RestoreAdmissionVerdict::ProvenNonWidening,
                ),
                vec![
                    RestoreAction::PreserveOrNarrowAuthority,
                    RestoreAction::MergeEvidence(
                        EvidenceRestorePolicy::RestrictionSupporting,
                    ),
                ],
            )
            .err(),
            Some(RestorePlanError::MissingEvidencePolicy {
                domain: RestoreDomain::DegradedSupervisor,
                policy: EvidenceRestorePolicy::RecoverySupportingFreshOnly,
            })
        );
    }

    #[test]
    fn unaudited_evidence_domain_fails_closed() {
        assert_eq!(
            RestoreDomainPlan::new(
                decision(
                    RestoreDomain::SensorFusion,
                    RestoreAdmissionVerdict::ReconciliationRequired,
                ),
                vec![
                    RestoreAction::MergeEvidence(EvidenceRestorePolicy::ReplayBarrier),
                    RestoreAction::ReconcileBeforeActivation,
                ],
            )
            .err(),
            Some(RestorePlanError::EvidencePolicyUnderspecified {
                domain: RestoreDomain::SensorFusion,
            })
        );
    }

    #[test]
    fn every_registered_semantic_has_a_required_action_family() {
        let mapping = [
            (
                RestoreSemantics::HistoricalReplace,
                RestoreAction::ReplaceValidatedHistorical,
            ),
            (
                RestoreSemantics::AuthorityMonotone,
                RestoreAction::PreserveOrNarrowAuthority,
            ),
            (
                RestoreSemantics::EvidenceMerge,
                RestoreAction::MergeEvidence(EvidenceRestorePolicy::NeutralHistory),
            ),
            (
                RestoreSemantics::DerivedRequalify,
                RestoreAction::RequalifyFromCurrentInputs,
            ),
            (
                RestoreSemantics::TransitionReconcile,
                RestoreAction::ReconcileBeforeActivation,
            ),
            (
                RestoreSemantics::EphemeralDrop,
                RestoreAction::DropEphemeral,
            ),
        ];
        for semantic in [
            RestoreSemantics::HistoricalReplace,
            RestoreSemantics::AuthorityMonotone,
            RestoreSemantics::EvidenceMerge,
            RestoreSemantics::DerivedRequalify,
            RestoreSemantics::TransitionReconcile,
            RestoreSemantics::EphemeralDrop,
        ] {
            assert!(mapping.iter().any(|(registered, _)| *registered == semantic));
        }
    }

    #[test]
    fn reconciliation_action_requires_reconciliation_verdict() {
        assert_eq!(
            RestoreDomainPlan::new(
                decision(
                    RestoreDomain::UpdateManager,
                    RestoreAdmissionVerdict::ProvenNonWidening,
                ),
                vec![RestoreAction::ReconcileBeforeActivation],
            )
            .err(),
            Some(RestorePlanError::VerdictActionMismatch {
                domain: RestoreDomain::UpdateManager,
                verdict: RestoreAdmissionVerdict::ProvenNonWidening,
                action: RestoreAction::ReconcileBeforeActivation,
            })
        );
    }

    #[test]
    fn derived_requalification_cannot_claim_plain_non_widening() {
        assert_eq!(
            RestoreDomainPlan::new(
                decision(
                    RestoreDomain::FieldEnvelope,
                    RestoreAdmissionVerdict::ProvenNonWidening,
                ),
                vec![RestoreAction::RequalifyFromCurrentInputs],
            )
            .err(),
            Some(RestorePlanError::VerdictActionMismatch {
                domain: RestoreDomain::FieldEnvelope,
                verdict: RestoreAdmissionVerdict::ProvenNonWidening,
                action: RestoreAction::RequalifyFromCurrentInputs,
            })
        );
    }

    #[test]
    fn duplicate_actions_are_rejected() {
        assert_eq!(
            RestoreDomainPlan::new(
                decision(
                    RestoreDomain::OperatorAuthority,
                    RestoreAdmissionVerdict::ReconciliationRequired,
                ),
                vec![
                    RestoreAction::PreserveOrNarrowAuthority,
                    RestoreAction::PreserveOrNarrowAuthority,
                    RestoreAction::MergeEvidence(EvidenceRestorePolicy::ReplayBarrier),
                    RestoreAction::DropEphemeral,
                    RestoreAction::ReconcileBeforeActivation,
                ],
            )
            .err(),
            Some(RestorePlanError::DuplicateAction {
                domain: RestoreDomain::OperatorAuthority,
                action: RestoreAction::PreserveOrNarrowAuthority,
            })
        );
    }
}
