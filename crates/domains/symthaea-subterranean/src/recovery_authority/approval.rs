// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Point-of-use qualification for human recovery approvals.
//!
//! A signed [`RecoveryApprovalEnvelopeV1`] is portable evidence that a human
//! approved a proposal. It is not, by itself, proof that the proposal is still
//! justified by the live system when the approval is counted toward quorum.
//! This module re-derives the live recovery basis immediately before admission
//! and wraps the approval in a host-local, non-serializable, non-cloneable token.
//!
//! The token is valid only for the exact host `now_step` at which it was
//! qualified. This prevents a caller from caching a prior "safe" evaluation and
//! presenting it after the control loop has advanced.

use super::{
    QualifiedRecoveryBasisV1, RecoveryApprovalEnvelopeV1, RecoveryHostBindingV1,
    RecoveryProposalRejection, RecoveryProposalV1, RecoveryQualificationRejection,
    requalify_recovery_proposal,
};
use crate::embodiment::SubterraneanEmbodiment;
use crate::operator_authority::{
    OperatorAuthority, OperatorAuthorityRejection, OperatorDecision,
};
use crate::operator_protocol::OperatorId;

/// Host-local, one-shot proof that one exact approval was requalified against
/// live runtime state at `qualified_at_step`.
///
/// Deliberately does not implement `Clone`, `Copy`, `Serialize`, or
/// `Deserialize`. Portable evidence and live admission authority remain distinct.
#[derive(Debug)]
#[must_use = "qualified recovery approval must be consumed by the authority gate or discarded"]
pub struct QualifiedRecoveryApprovalV1 {
    approval: RecoveryApprovalEnvelopeV1,
    qualified_at_step: u64,
}

impl QualifiedRecoveryApprovalV1 {
    pub const fn operator(&self) -> OperatorId {
        self.approval.operator
    }

    pub const fn proposal_id(&self) -> u64 {
        self.approval.proposal.proposal_id()
    }

    pub const fn qualified_at_step(&self) -> u64 {
        self.qualified_at_step
    }

    pub const fn proposal(&self) -> RecoveryProposalV1 {
        self.approval.proposal
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryApprovalQualificationRejection {
    ApprovalNotYetIssued,
    Proposal(RecoveryProposalRejection),
    Basis(RecoveryQualificationRejection),
}

impl From<RecoveryProposalRejection> for RecoveryApprovalQualificationRejection {
    fn from(value: RecoveryProposalRejection) -> Self {
        Self::Proposal(value)
    }
}

impl From<RecoveryQualificationRejection> for RecoveryApprovalQualificationRejection {
    fn from(value: RecoveryQualificationRejection) -> Self {
        Self::Basis(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryApprovalAdmissionRejection {
    StaleQualification {
        qualified_at_step: u64,
        now_step: u64,
    },
    Authority(OperatorAuthorityRejection),
}

impl From<OperatorAuthorityRejection> for RecoveryApprovalAdmissionRejection {
    fn from(value: OperatorAuthorityRejection) -> Self {
        Self::Authority(value)
    }
}

/// Recompute the live basis for the proposal carried by `approval` and produce
/// a one-shot host-local approval token.
///
/// This check intentionally does not require every independent safety source to
/// be nominal. It requires the exact proposal basis to remain valid. Independent
/// partition/degraded/temporal/capability restrictions remain encoded in that
/// basis and continue to constrain the platform after operator recovery.
pub fn qualify_recovery_approval(
    embodiment: &SubterraneanEmbodiment,
    host: RecoveryHostBindingV1,
    approval: RecoveryApprovalEnvelopeV1,
    now_step: u64,
) -> Result<QualifiedRecoveryApprovalV1, RecoveryApprovalQualificationRejection> {
    if approval.approval_issued_step > now_step {
        return Err(RecoveryApprovalQualificationRejection::ApprovalNotYetIssued);
    }
    approval.validate_proposal_time()?;
    approval
        .proposal
        .validate(now_step, embodiment.operator_constraint())?;

    let _basis: QualifiedRecoveryBasisV1 =
        requalify_recovery_proposal(embodiment, host, approval.proposal)?;

    Ok(QualifiedRecoveryApprovalV1 {
        approval,
        qualified_at_step: now_step,
    })
}

impl OperatorAuthority {
    /// Internal point-of-use quorum primitive. The token is consumed by value,
    /// and advancing the host control step requires re-running live qualification.
    /// Public recovery admission must be owned by the embodiment/control-plane
    /// that owns the exact authority instance and evidence source.
    pub(crate) fn approve_qualified_recovery(
        &mut self,
        qualified: QualifiedRecoveryApprovalV1,
        now_step: u64,
    ) -> Result<OperatorDecision, RecoveryApprovalAdmissionRejection> {
        if now_step != qualified.qualified_at_step {
            return Err(RecoveryApprovalAdmissionRejection::StaleQualification {
                qualified_at_step: qualified.qualified_at_step,
                now_step,
            });
        }
        Ok(self.approve_recovery(qualified.approval, now_step)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{RecoveryDigest, qualify_recovery_basis};
    use crate::operator_authority::OperatorConstraint;
    use crate::operator_protocol::{
        AuthenticationLevel, OperatorCommand, OperatorCommandEnvelope, OperatorRole,
    };
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

    fn host() -> RecoveryHostBindingV1 {
        RecoveryHostBindingV1::new(RecoveryDigest([23; 32]), 7, 11)
            .expect("valid host binding")
    }

    fn command(
        operator: u64,
        sequence: u64,
        proposal_id: u64,
        command: OperatorCommand,
        issued_step: u64,
    ) -> OperatorCommandEnvelope {
        OperatorCommandEnvelope {
            operator: OperatorId(operator),
            role: OperatorRole::SafetyOfficer,
            authentication: AuthenticationLevel::HardwareBacked,
            epoch: 1,
            sequence,
            proposal_id,
            issued_step,
            expires_step: issued_step.saturating_add(1_000),
            command,
        }
    }

    fn approval(
        operator: u64,
        sequence: u64,
        proposal: RecoveryProposalV1,
        issued_step: u64,
    ) -> RecoveryApprovalEnvelopeV1 {
        RecoveryApprovalEnvelopeV1 {
            operator: OperatorId(operator),
            role: OperatorRole::SafetyOfficer,
            authentication: AuthenticationLevel::HardwareBacked,
            epoch: 1,
            sequence,
            approval_issued_step: issued_step,
            proposal,
        }
    }

    fn step_once(embodiment: &mut SubterraneanEmbodiment, seed: u64) {
        let thought = ContinuousHV::random(HDC_DIMENSION, seed);
        let _ = embodiment.step(&thought, 0.005, 0.9);
    }

    fn prepared_hold(
        phrase: &str,
    ) -> (
        SubterraneanEmbodiment,
        OperatorAuthority,
        RecoveryProposalV1,
        u64,
    ) {
        let genesis = GenesisSeed::from_phrase(phrase);
        let mut embodiment = SubterraneanEmbodiment::new(&genesis);
        embodiment
            .ingest_operator_command(command(90, 1, 1, OperatorCommand::HoldPosition, 0))
            .expect("embodiment hold should be accepted");
        step_once(&mut embodiment, 101);

        let basis = qualify_recovery_basis(&embodiment, host()).expect("basis should qualify");
        let now_step = basis.evidence_step();

        let mut authority = OperatorAuthority::default();
        authority
            .ingest(
                command(99, 1, 1, OperatorCommand::HoldPosition, 0),
                now_step,
                true,
            )
            .expect("standalone authority hold should be accepted");
        assert_eq!(authority.constraint(), OperatorConstraint::HoldPosition);

        let proposal = authority
            .issue_qualified_recovery_proposal(
                basis,
                17,
                now_step,
                now_step.saturating_add(100),
            )
            .expect("qualified proposal should issue");

        (embodiment, authority, proposal, now_step)
    }

    #[test]
    fn approval_must_be_requalified_at_point_of_use() {
        let (embodiment, mut authority, proposal, now_step) =
            prepared_hold("live-recovery-approval");
        let token = qualify_recovery_approval(
            &embodiment,
            host(),
            approval(1, 1, proposal, now_step),
            now_step,
        )
        .expect("live approval should qualify");

        assert!(matches!(
            authority
                .approve_qualified_recovery(token, now_step)
                .expect("qualified approval should count"),
            OperatorDecision::PendingQuorum {
                approvals: 1,
                required: 2
            }
        ));
    }

    #[test]
    fn qualified_token_expires_when_host_step_advances() {
        let (embodiment, mut authority, proposal, now_step) =
            prepared_hold("stale-live-recovery-approval");
        let token = qualify_recovery_approval(
            &embodiment,
            host(),
            approval(1, 1, proposal, now_step),
            now_step,
        )
        .expect("live approval should qualify");

        assert_eq!(
            authority.approve_qualified_recovery(token, now_step.saturating_add(1)),
            Err(RecoveryApprovalAdmissionRejection::StaleQualification {
                qualified_at_step: now_step,
                now_step: now_step.saturating_add(1),
            })
        );
        assert_eq!(authority.pending_approvals(proposal.proposal_id()), 0);
    }

    #[test]
    fn stable_live_basis_can_support_second_human_on_later_step() {
        let (mut embodiment, mut authority, proposal, first_step) =
            prepared_hold("stable-live-recovery-quorum");

        let first = qualify_recovery_approval(
            &embodiment,
            host(),
            approval(1, 1, proposal, first_step),
            first_step,
        )
        .expect("first approval should qualify");
        assert!(matches!(
            authority
                .approve_qualified_recovery(first, first_step)
                .expect("first approval should count"),
            OperatorDecision::PendingQuorum { .. }
        ));

        step_once(&mut embodiment, 102);
        let second_step = embodiment
            .evidence_records()
            .last()
            .expect("evidence after second step")
            .step;
        assert!(second_step >= first_step);

        let second = qualify_recovery_approval(
            &embodiment,
            host(),
            approval(2, 1, proposal, second_step),
            second_step,
        )
        .expect("semantically unchanged basis should still qualify");
        assert_eq!(
            authority
                .approve_qualified_recovery(second, second_step)
                .expect("second approval should clear quorum"),
            OperatorDecision::Cleared
        );
        assert_eq!(authority.constraint(), OperatorConstraint::None);
    }

    #[test]
    fn material_constraint_change_invalidates_old_reviewed_proposal() {
        let (mut embodiment, _authority, proposal, now_step) =
            prepared_hold("changed-live-recovery-basis");
        embodiment
            .ingest_operator_command(command(
                91,
                1,
                2,
                OperatorCommand::EmergencyStop,
                now_step,
            ))
            .expect("more restrictive emergency stop should be accepted");
        step_once(&mut embodiment, 103);
        let changed_step = embodiment
            .evidence_records()
            .last()
            .expect("changed evidence")
            .step;

        assert!(matches!(
            qualify_recovery_approval(
                &embodiment,
                host(),
                approval(1, 1, proposal, changed_step),
                changed_step,
            ),
            Err(RecoveryApprovalQualificationRejection::Proposal(
                RecoveryProposalRejection::ActiveConstraintMismatch
            ))
        ));
    }

    #[test]
    fn future_dated_human_approval_cannot_be_qualified_early() {
        let (embodiment, _authority, proposal, now_step) =
            prepared_hold("future-live-recovery-approval");
        assert!(matches!(
            qualify_recovery_approval(
                &embodiment,
                host(),
                approval(1, 1, proposal, now_step.saturating_add(1)),
                now_step,
            ),
            Err(RecoveryApprovalQualificationRejection::ApprovalNotYetIssued)
        ));
    }
}
