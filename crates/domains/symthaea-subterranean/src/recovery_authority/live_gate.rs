// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Host-authoritative point-of-use recovery admission.
//!
//! The lower-level approval token in `approval.rs` is useful for characterizing
//! the distinction between portable approval evidence and live authority, but a
//! caller-supplied `now_step` is not itself a trustworthy freshness source. This
//! module therefore derives the admission step from the live embodiment's latest
//! qualified evidence and immediately forwards the approval into the host-owned
//! authority state machine.
//!
//! No reusable "safety passed" token crosses this boundary. If the embodiment
//! advances or its material recovery basis changes, the next approval attempt
//! re-evaluates that new state.

use super::{
    RecoveryApprovalEnvelopeV1, RecoveryHostBindingV1, RecoveryProposalRejection,
    RecoveryQualificationRejection, requalify_recovery_proposal,
};
use crate::embodiment::SubterraneanEmbodiment;
use crate::operator_authority::{
    OperatorAuthority, OperatorAuthorityRejection, OperatorDecision,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveRecoveryAdmissionRejection {
    ApprovalAfterLatestEvidence {
        approval_issued_step: u64,
        evidence_step: u64,
    },
    Proposal(RecoveryProposalRejection),
    Basis(RecoveryQualificationRejection),
    Authority(OperatorAuthorityRejection),
}

impl From<RecoveryProposalRejection> for LiveRecoveryAdmissionRejection {
    fn from(value: RecoveryProposalRejection) -> Self {
        Self::Proposal(value)
    }
}

impl From<RecoveryQualificationRejection> for LiveRecoveryAdmissionRejection {
    fn from(value: RecoveryQualificationRejection) -> Self {
        Self::Basis(value)
    }
}

impl From<OperatorAuthorityRejection> for LiveRecoveryAdmissionRejection {
    fn from(value: OperatorAuthorityRejection) -> Self {
        Self::Authority(value)
    }
}

impl OperatorAuthority {
    /// Requalify and admit one recovery approval using the latest evidence step
    /// from the live embodiment as the authoritative admission time.
    ///
    /// This method intentionally does not accept `now_step` from the caller.
    /// The latest qualifying evidence must be at least as recent as the human
    /// approval itself, and the reviewed proposal must still match current live
    /// state before the approval is allowed to enter quorum.
    pub fn approve_recovery_from_live_state(
        &mut self,
        embodiment: &SubterraneanEmbodiment,
        host: RecoveryHostBindingV1,
        approval: RecoveryApprovalEnvelopeV1,
    ) -> Result<OperatorDecision, LiveRecoveryAdmissionRejection> {
        approval.validate_proposal_time()?;

        let basis = requalify_recovery_proposal(embodiment, host, approval.proposal)?;
        let evidence_step = basis.evidence_step();

        if approval.approval_issued_step > evidence_step {
            return Err(LiveRecoveryAdmissionRejection::ApprovalAfterLatestEvidence {
                approval_issued_step: approval.approval_issued_step,
                evidence_step,
            });
        }

        approval
            .proposal
            .validate(evidence_step, embodiment.operator_constraint())?;

        Ok(self.approve_recovery(approval, evidence_step)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator_authority::OperatorConstraint;
    use crate::operator_protocol::{
        AuthenticationLevel, OperatorCommand, OperatorCommandEnvelope, OperatorId, OperatorRole,
    };
    use crate::operator_authority::recovery_authority::{
        RecoveryApprovalEnvelopeV1, RecoveryDigest, RecoveryProposalV1, qualify_recovery_basis,
    };
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

    fn host() -> RecoveryHostBindingV1 {
        RecoveryHostBindingV1::new(RecoveryDigest([31; 32]), 7, 11)
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

    fn step_once(embodiment: &mut SubterraneanEmbodiment, seed: u64) -> u64 {
        let thought = ContinuousHV::random(HDC_DIMENSION, seed);
        let _ = embodiment.step(&thought, 0.005, 0.9);
        embodiment
            .evidence_records()
            .last()
            .expect("evidence after step")
            .step
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
        let evidence_step = step_once(&mut embodiment, 201);

        let basis = qualify_recovery_basis(&embodiment, host()).expect("basis should qualify");
        assert_eq!(basis.evidence_step(), evidence_step);

        let mut authority = OperatorAuthority::default();
        authority
            .ingest(
                command(99, 1, 1, OperatorCommand::HoldPosition, 0),
                evidence_step,
                true,
            )
            .expect("standalone authority hold should be accepted");
        assert_eq!(authority.constraint(), OperatorConstraint::HoldPosition);

        let proposal = authority
            .issue_qualified_recovery_proposal(
                basis,
                41,
                evidence_step,
                evidence_step.saturating_add(100),
            )
            .expect("qualified proposal should issue");

        (embodiment, authority, proposal, evidence_step)
    }

    #[test]
    fn live_gate_derives_admission_time_from_embodiment() {
        let (embodiment, mut authority, proposal, evidence_step) =
            prepared_hold("live-gate-derived-time");

        let decision = authority
            .approve_recovery_from_live_state(
                &embodiment,
                host(),
                approval(1, 1, proposal, evidence_step),
            )
            .expect("current approval should enter quorum");

        assert!(matches!(
            decision,
            OperatorDecision::PendingQuorum {
                approvals: 1,
                required: 2
            }
        ));
    }

    #[test]
    fn approval_newer_than_latest_evidence_cannot_enter_quorum() {
        let (embodiment, mut authority, proposal, evidence_step) =
            prepared_hold("live-gate-fresh-evidence");
        let approval_step = evidence_step.saturating_add(1);

        assert_eq!(
            authority.approve_recovery_from_live_state(
                &embodiment,
                host(),
                approval(1, 1, proposal, approval_step),
            ),
            Err(LiveRecoveryAdmissionRejection::ApprovalAfterLatestEvidence {
                approval_issued_step: approval_step,
                evidence_step,
            })
        );
        assert_eq!(authority.pending_approvals(proposal.proposal_id()), 0);
    }

    #[test]
    fn fresh_evidence_after_human_approval_allows_admission() {
        let (mut embodiment, mut authority, proposal, first_evidence_step) =
            prepared_hold("live-gate-evidence-after-approval");
        let approval_step = first_evidence_step.saturating_add(1);

        let later_evidence_step = step_once(&mut embodiment, 202);
        assert!(later_evidence_step >= approval_step);

        let decision = authority
            .approve_recovery_from_live_state(
                &embodiment,
                host(),
                approval(1, 1, proposal, approval_step),
            )
            .expect("fresh evidence should allow the approval to enter quorum");

        assert!(matches!(decision, OperatorDecision::PendingQuorum { .. }));
    }

    #[test]
    fn material_live_state_change_rejects_old_proposal_before_quorum() {
        let (mut embodiment, mut authority, proposal, evidence_step) =
            prepared_hold("live-gate-material-change");
        embodiment
            .ingest_operator_command(command(
                91,
                1,
                2,
                OperatorCommand::EmergencyStop,
                evidence_step,
            ))
            .expect("emergency stop should be accepted");
        let changed_step = step_once(&mut embodiment, 203);

        assert!(matches!(
            authority.approve_recovery_from_live_state(
                &embodiment,
                host(),
                approval(1, 1, proposal, changed_step),
            ),
            Err(LiveRecoveryAdmissionRejection::Basis(
                RecoveryQualificationRejection::ProposalBasisMismatch
            ))
                | Err(LiveRecoveryAdmissionRejection::Basis(
                    RecoveryQualificationRejection::EvidenceConstraintMismatch
                ))
        ));
        assert_eq!(authority.pending_approvals(proposal.proposal_id()), 0);
    }
}
