// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Operator-authority restore comparison.
//!
//! The active operator constraint has a small, explicit restrictiveness order,
//! but `OperatorAuthority` also contains replay evidence (`last_sequence`) that
//! must merge monotonically. Until that evidence merge is implemented, a safe
//! constraint comparison is **not** sufficient to claim a fully non-widening
//! operator restore.

use crate::operator_authority::{OperatorAuthority, OperatorConstraint};

use super::restore_admission::{RestoreAdmissionVerdict, RestoreDomainDecision};
use super::restore_semantics::RestoreDomain;

/// Compare only the active operator constraint.
///
/// Exact equality is safe. A strictly higher restrictiveness rank is a proven
/// narrowing. A lower rank widens authority. Equal-rank but unequal values are
/// deliberately `NotProvable`: today that covers `None <-> Mission(_)` and
/// different mission intents, whose mission-level effects are not represented
/// by the scalar safety rank.
pub(super) fn operator_constraint_restore_verdict(
    current: OperatorConstraint,
    candidate: OperatorConstraint,
) -> RestoreAdmissionVerdict {
    if candidate == current {
        return RestoreAdmissionVerdict::ProvenNonWidening;
    }

    let current_rank = current.restrictiveness_rank();
    let candidate_rank = candidate.restrictiveness_rank();
    if candidate_rank > current_rank {
        RestoreAdmissionVerdict::ProvenNonWidening
    } else if candidate_rank < current_rank {
        RestoreAdmissionVerdict::Widening
    } else {
        RestoreAdmissionVerdict::NotProvable
    }
}

/// Produce the RA-19 decision for the complete `OperatorAuthority` domain.
///
/// Even when the active constraint is proven non-widening, the full domain is
/// currently `ReconciliationRequired` because stale checkpoint replay history
/// must be merged rather than replaced. This prevents a partial constraint-only
/// fix from reopening already-consumed operator sequences.
pub(super) fn operator_authority_restore_decision(
    current: &OperatorAuthority,
    candidate: &OperatorAuthority,
) -> RestoreDomainDecision {
    let constraint_verdict =
        operator_constraint_restore_verdict(current.constraint(), candidate.constraint());
    let verdict = match constraint_verdict {
        RestoreAdmissionVerdict::Widening => RestoreAdmissionVerdict::Widening,
        RestoreAdmissionVerdict::NotProvable => RestoreAdmissionVerdict::NotProvable,
        RestoreAdmissionVerdict::ProvenNonWidening
        | RestoreAdmissionVerdict::ConservativeRequalification
        | RestoreAdmissionVerdict::ReconciliationRequired => {
            RestoreAdmissionVerdict::ReconciliationRequired
        }
    };
    RestoreDomainDecision::new(RestoreDomain::OperatorAuthority, verdict)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mission::SubterraneanMissionIntent;
    use crate::operator_protocol::{
        AuthenticationLevel, OperatorCommand, OperatorCommandEnvelope, OperatorId, OperatorRole,
    };

    fn command(sequence: u64, command: OperatorCommand) -> OperatorCommandEnvelope {
        OperatorCommandEnvelope {
            operator: OperatorId(7),
            role: OperatorRole::SafetyOfficer,
            authentication: AuthenticationLevel::HardwareBacked,
            epoch: 1,
            sequence,
            proposal_id: sequence,
            issued_step: 0,
            expires_step: 100,
            command,
        }
    }

    #[test]
    fn exact_constraint_is_non_widening() {
        for constraint in [
            OperatorConstraint::None,
            OperatorConstraint::ReturnHome,
            OperatorConstraint::HoldPosition,
            OperatorConstraint::MaintenanceLock,
            OperatorConstraint::EmergencyStop,
            OperatorConstraint::Mission(SubterraneanMissionIntent::Explore),
        ] {
            assert_eq!(
                operator_constraint_restore_verdict(constraint, constraint),
                RestoreAdmissionVerdict::ProvenNonWidening
            );
        }
    }

    #[test]
    fn stronger_rank_is_non_widening_and_weaker_rank_is_widening() {
        let ordered = [
            OperatorConstraint::None,
            OperatorConstraint::ReturnHome,
            OperatorConstraint::HoldPosition,
            OperatorConstraint::MaintenanceLock,
            OperatorConstraint::EmergencyStop,
        ];
        for (current_index, current) in ordered.iter().copied().enumerate() {
            for (candidate_index, candidate) in ordered.iter().copied().enumerate() {
                let verdict = operator_constraint_restore_verdict(current, candidate);
                if candidate_index > current_index {
                    assert_eq!(verdict, RestoreAdmissionVerdict::ProvenNonWidening);
                } else if candidate_index < current_index {
                    assert_eq!(verdict, RestoreAdmissionVerdict::Widening);
                } else {
                    assert_eq!(verdict, RestoreAdmissionVerdict::ProvenNonWidening);
                }
            }
        }
    }

    #[test]
    fn equal_rank_mission_changes_are_not_provable() {
        assert_eq!(
            operator_constraint_restore_verdict(
                OperatorConstraint::None,
                OperatorConstraint::Mission(SubterraneanMissionIntent::Explore),
            ),
            RestoreAdmissionVerdict::NotProvable
        );
        assert_eq!(
            operator_constraint_restore_verdict(
                OperatorConstraint::Mission(SubterraneanMissionIntent::Explore),
                OperatorConstraint::None,
            ),
            RestoreAdmissionVerdict::NotProvable
        );
        assert_eq!(
            operator_constraint_restore_verdict(
                OperatorConstraint::Mission(SubterraneanMissionIntent::Explore),
                OperatorConstraint::Mission(SubterraneanMissionIntent::AssistPeer),
            ),
            RestoreAdmissionVerdict::NotProvable
        );
    }

    #[test]
    fn constraint_safe_operator_domain_still_requires_replay_reconciliation() {
        let current = OperatorAuthority::default();
        let candidate = OperatorAuthority::default();
        assert_eq!(
            operator_authority_restore_decision(&current, &candidate),
            RestoreDomainDecision::new(
                RestoreDomain::OperatorAuthority,
                RestoreAdmissionVerdict::ReconciliationRequired,
            )
        );
    }

    #[test]
    fn weaker_candidate_constraint_widens_complete_operator_domain() {
        let mut current = OperatorAuthority::default();
        current
            .ingest(command(1, OperatorCommand::EmergencyStop), 0, true)
            .expect("emergency stop should be accepted");
        let candidate = OperatorAuthority::default();
        assert_eq!(
            operator_authority_restore_decision(&current, &candidate),
            RestoreDomainDecision::new(
                RestoreDomain::OperatorAuthority,
                RestoreAdmissionVerdict::Widening,
            )
        );
    }
}
