// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Action-complete restore planning and evidence-polarity contracts.
//!
//! RA-17 defines semantic obligations; RA-19 binds an admission decision to a
//! live generation. This module proves the translation into executable-shaped
//! restore actions is complete, exclusive, and evidence-polarity aware.

use super::restore_admission::{RestoreAdmissionVerdict, RestoreDomainDecision};
use super::restore_semantics::{contract_for, RestoreDomain, RestoreSemantics};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum EvidenceRestorePolicy {
    ReplayBarrier,
    RestrictionSupporting,
    CounterexamplePreserving,
    RecoverySupportingFreshOnly,
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

/// Evidence polarity is audited domain-by-domain. `None` means the domain is
/// intentionally blocked until its state machine has been inspected.
fn required_evidence_policies(domain: RestoreDomain) -> Option<&'static [EvidenceRestorePolicy]> {
    match domain {
        RestoreDomain::OperatorAuthority => Some(&[EvidenceRestorePolicy::ReplayBarrier]),
        RestoreDomain::DegradedSupervisor => Some(&[
            EvidenceRestorePolicy::RestrictionSupporting,
            EvidenceRestorePolicy::RecoverySupportingFreshOnly,
        ]),
        RestoreDomain::SensorFusion => Some(&[
            EvidenceRestorePolicy::ReplayBarrier,
            EvidenceRestorePolicy::RestrictionSupporting,
            EvidenceRestorePolicy::RecoverySupportingFreshOnly,
        ]),
        RestoreDomain::ActuatorIsolation
        | RestoreDomain::PartitionRecovery
        | RestoreDomain::TemporalAssurance => None,
        RestoreDomain::Controller
        | RestoreDomain::Mission
        | RestoreDomain::UpdateManager
        | RestoreDomain::FieldEnvelope => Some(&[]),
    }
}

fn has(actions: &[RestoreAction], action: RestoreAction) -> bool {
    actions.contains(&action)
}

fn has_evidence(actions: &[RestoreAction]) -> bool {
    actions
        .iter()
        .any(|action| matches!(action, RestoreAction::MergeEvidence(_)))
}

fn semantic_present(semantic: RestoreSemantics, actions: &[RestoreAction]) -> bool {
    match semantic {
        RestoreSemantics::HistoricalReplace => has(actions, RestoreAction::ReplaceValidatedHistorical),
        RestoreSemantics::AuthorityMonotone => has(actions, RestoreAction::PreserveOrNarrowAuthority),
        RestoreSemantics::EvidenceMerge => has_evidence(actions),
        RestoreSemantics::DerivedRequalify => has(actions, RestoreAction::RequalifyFromCurrentInputs),
        RestoreSemantics::TransitionReconcile => has(actions, RestoreAction::ReconcileBeforeActivation),
        RestoreSemantics::EphemeralDrop => has(actions, RestoreAction::DropEphemeral),
    }
}

fn action_supported(
    semantics: &[RestoreSemantics],
    verdict: RestoreAdmissionVerdict,
    action: RestoreAction,
) -> bool {
    match action {
        RestoreAction::ReplaceValidatedHistorical => semantics.contains(&RestoreSemantics::HistoricalReplace),
        RestoreAction::PreserveOrNarrowAuthority => semantics.contains(&RestoreSemantics::AuthorityMonotone),
        RestoreAction::MergeEvidence(_) => semantics.contains(&RestoreSemantics::EvidenceMerge),
        RestoreAction::RequalifyFromCurrentInputs => semantics.contains(&RestoreSemantics::DerivedRequalify),
        RestoreAction::ReconcileBeforeActivation => {
            semantics.contains(&RestoreSemantics::TransitionReconcile)
                || verdict == RestoreAdmissionVerdict::ReconciliationRequired
        }
        RestoreAction::DropEphemeral => semantics.contains(&RestoreSemantics::EphemeralDrop),
    }
}

fn validate_verdict(
    domain: RestoreDomain,
    verdict: RestoreAdmissionVerdict,
    actions: &[RestoreAction],
) -> Result<(), RestorePlanError> {
    if matches!(verdict, RestoreAdmissionVerdict::Widening | RestoreAdmissionVerdict::NotProvable) {
        return Err(RestorePlanError::DecisionNotAdmissible { domain, verdict });
    }

    match verdict {
        RestoreAdmissionVerdict::ReconciliationRequired => {
            if !has(actions, RestoreAction::ReconcileBeforeActivation) {
                return Err(RestorePlanError::MissingVerdictAction {
                    domain,
                    verdict,
                    action: RestoreAction::ReconcileBeforeActivation,
                });
            }
        }
        RestoreAdmissionVerdict::ConservativeRequalification => {
            if !has(actions, RestoreAction::RequalifyFromCurrentInputs) {
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

    if has(actions, RestoreAction::ReconcileBeforeActivation)
        && verdict != RestoreAdmissionVerdict::ReconciliationRequired
    {
        return Err(RestorePlanError::VerdictActionMismatch {
            domain,
            verdict,
            action: RestoreAction::ReconcileBeforeActivation,
        });
    }
    if has(actions, RestoreAction::RequalifyFromCurrentInputs)
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
    validate_verdict(domain, verdict, &actions)?;

    actions.sort_unstable();
    for pair in actions.windows(2) {
        if pair[0] == pair[1] {
            return Err(RestorePlanError::DuplicateAction { domain, action: pair[0] });
        }
    }

    let contract = contract_for(domain);
    for action in &actions {
        if !action_supported(contract.semantics, verdict, *action) {
            return Err(RestorePlanError::UnexpectedAction { domain, action: *action });
        }
    }
    for semantic in contract.semantics {
        if !semantic_present(*semantic, &actions) {
            return Err(RestorePlanError::MissingSemanticAction { domain, semantic: *semantic });
        }
    }

    if contract.semantics.contains(&RestoreSemantics::EvidenceMerge) {
        let Some(required) = required_evidence_policies(domain) else {
            return Err(RestorePlanError::EvidencePolicyUnderspecified { domain });
        };
        for policy in required {
            if !has(&actions, RestoreAction::MergeEvidence(*policy)) {
                return Err(RestorePlanError::MissingEvidencePolicy { domain, policy: *policy });
            }
        }
        for action in &actions {
            if let RestoreAction::MergeEvidence(policy) = action {
                if !required.contains(policy) {
                    return Err(RestorePlanError::UnexpectedEvidencePolicy { domain, policy: *policy });
                }
            }
        }
    } else if let Some(RestoreAction::MergeEvidence(policy)) = actions
        .iter()
        .copied()
        .find(|action| matches!(action, RestoreAction::MergeEvidence(_)))
    {
        return Err(RestorePlanError::UnexpectedEvidencePolicy { domain, policy });
    }

    Ok(RestoreDomainPlan { decision, actions })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(domain: RestoreDomain, verdict: RestoreAdmissionVerdict) -> RestoreDomainDecision {
        RestoreDomainDecision::new(domain, verdict)
    }

    fn operator_actions() -> Vec<RestoreAction> {
        vec![
            RestoreAction::PreserveOrNarrowAuthority,
            RestoreAction::MergeEvidence(EvidenceRestorePolicy::ReplayBarrier),
            RestoreAction::ReconcileBeforeActivation,
            RestoreAction::DropEphemeral,
        ]
    }

    #[test]
    fn operator_plan_is_action_complete() {
        let plan = RestoreDomainPlan::new(
            d(RestoreDomain::OperatorAuthority, RestoreAdmissionVerdict::ReconciliationRequired),
            operator_actions(),
        )
        .unwrap();
        assert_eq!(plan.domain(), RestoreDomain::OperatorAuthority);
        assert_eq!(plan.verdict(), RestoreAdmissionVerdict::ReconciliationRequired);
    }

    #[test]
    fn reconciliation_verdict_requires_non_activation_barrier() {
        let mut actions = operator_actions();
        actions.retain(|a| *a != RestoreAction::ReconcileBeforeActivation);
        assert_eq!(
            RestoreDomainPlan::new(
                d(RestoreDomain::OperatorAuthority, RestoreAdmissionVerdict::ReconciliationRequired),
                actions,
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
    fn operator_cannot_smuggle_historical_replace() {
        let mut actions = operator_actions();
        actions.push(RestoreAction::ReplaceValidatedHistorical);
        assert_eq!(
            RestoreDomainPlan::new(
                d(RestoreDomain::OperatorAuthority, RestoreAdmissionVerdict::ReconciliationRequired),
                actions,
            )
            .err(),
            Some(RestorePlanError::UnexpectedAction {
                domain: RestoreDomain::OperatorAuthority,
                action: RestoreAction::ReplaceValidatedHistorical,
            })
        );
    }

    #[test]
    fn degraded_plan_distinguishes_adverse_and_recovery_evidence() {
        RestoreDomainPlan::new(
            d(RestoreDomain::DegradedSupervisor, RestoreAdmissionVerdict::ReconciliationRequired),
            vec![
                RestoreAction::PreserveOrNarrowAuthority,
                RestoreAction::MergeEvidence(EvidenceRestorePolicy::RestrictionSupporting),
                RestoreAction::MergeEvidence(EvidenceRestorePolicy::RecoverySupportingFreshOnly),
                RestoreAction::ReconcileBeforeActivation,
            ],
        )
        .unwrap();
    }

    #[test]
    fn degraded_plan_cannot_omit_fresh_only_recovery_policy() {
        assert_eq!(
            RestoreDomainPlan::new(
                d(RestoreDomain::DegradedSupervisor, RestoreAdmissionVerdict::ReconciliationRequired),
                vec![
                    RestoreAction::PreserveOrNarrowAuthority,
                    RestoreAction::MergeEvidence(EvidenceRestorePolicy::RestrictionSupporting),
                    RestoreAction::ReconcileBeforeActivation,
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
    fn sensor_plan_covers_replay_reliability_fresh_report_and_policy() {
        let plan = RestoreDomainPlan::new(
            d(RestoreDomain::SensorFusion, RestoreAdmissionVerdict::ReconciliationRequired),
            vec![
                RestoreAction::MergeEvidence(EvidenceRestorePolicy::ReplayBarrier),
                RestoreAction::MergeEvidence(EvidenceRestorePolicy::RestrictionSupporting),
                RestoreAction::MergeEvidence(EvidenceRestorePolicy::RecoverySupportingFreshOnly),
                RestoreAction::RequalifyFromCurrentInputs,
                RestoreAction::ReconcileBeforeActivation,
            ],
        )
        .unwrap();
        assert_eq!(plan.domain(), RestoreDomain::SensorFusion);
    }

    #[test]
    fn sensor_plan_cannot_restore_previous_report_as_current_truth() {
        assert_eq!(
            RestoreDomainPlan::new(
                d(RestoreDomain::SensorFusion, RestoreAdmissionVerdict::ReconciliationRequired),
                vec![
                    RestoreAction::MergeEvidence(EvidenceRestorePolicy::ReplayBarrier),
                    RestoreAction::MergeEvidence(EvidenceRestorePolicy::RestrictionSupporting),
                    RestoreAction::MergeEvidence(EvidenceRestorePolicy::RecoverySupportingFreshOnly),
                    RestoreAction::ReconcileBeforeActivation,
                ],
            )
            .err(),
            Some(RestorePlanError::MissingSemanticAction {
                domain: RestoreDomain::SensorFusion,
                semantic: RestoreSemantics::DerivedRequalify,
            })
        );
    }

    #[test]
    fn unaudited_actuator_evidence_fails_closed() {
        assert_eq!(
            RestoreDomainPlan::new(
                d(RestoreDomain::ActuatorIsolation, RestoreAdmissionVerdict::ReconciliationRequired),
                vec![
                    RestoreAction::PreserveOrNarrowAuthority,
                    RestoreAction::MergeEvidence(EvidenceRestorePolicy::RestrictionSupporting),
                    RestoreAction::ReconcileBeforeActivation,
                ],
            )
            .err(),
            Some(RestorePlanError::EvidencePolicyUnderspecified {
                domain: RestoreDomain::ActuatorIsolation,
            })
        );
    }

    #[test]
    fn field_envelope_requires_requalification_and_policy_reconcile() {
        RestoreDomainPlan::new(
            d(RestoreDomain::FieldEnvelope, RestoreAdmissionVerdict::ReconciliationRequired),
            vec![
                RestoreAction::RequalifyFromCurrentInputs,
                RestoreAction::ReconcileBeforeActivation,
            ],
        )
        .unwrap();
    }

    #[test]
    fn proven_non_widening_cannot_carry_reconciliation_action() {
        assert_eq!(
            RestoreDomainPlan::new(
                d(RestoreDomain::UpdateManager, RestoreAdmissionVerdict::ProvenNonWidening),
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
    fn duplicate_actions_are_rejected() {
        let mut actions = operator_actions();
        actions.push(RestoreAction::PreserveOrNarrowAuthority);
        assert_eq!(
            RestoreDomainPlan::new(
                d(RestoreDomain::OperatorAuthority, RestoreAdmissionVerdict::ReconciliationRequired),
                actions,
            )
            .err(),
            Some(RestorePlanError::DuplicateAction {
                domain: RestoreDomain::OperatorAuthority,
                action: RestoreAction::PreserveOrNarrowAuthority,
            })
        );
    }
}
