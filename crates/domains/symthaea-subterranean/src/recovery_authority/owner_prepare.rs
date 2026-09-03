// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Borrow-safe preparation for embodiment-owned recovery transitions.
//!
//! `SubterraneanEmbodiment` owns both the live evidence source and the
//! authoritative `OperatorAuthority`. A public owner-level recovery method must
//! therefore avoid holding an immutable borrow of the whole embodiment while it
//! mutably borrows the internal authority field.
//!
//! This module separates those phases:
//!
//! 1. derive and validate a host-local prepared transition from immutable live
//!    embodiment state;
//! 2. end that immutable borrow;
//! 3. consume the prepared transition through the internal authority state
//!    machine.
//!
//! The prepared types are deliberately non-serializable and non-cloneable. They
//! are implementation capabilities for the eventual embodiment-owned API, not
//! portable authority and not public recovery entry points.

use super::{
    RecoveryApprovalEnvelopeV1, RecoveryHostBindingV1, RecoveryProposalRejection,
    RecoveryProposalV1, RecoveryQualificationRejection, qualify_recovery_basis,
    requalify_recovery_proposal,
};
use crate::embodiment::SubterraneanEmbodiment;
use crate::operator_authority::{
    OperatorAuthority, OperatorAuthorityRejection, OperatorDecision,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecoveryOwnerPreparationRejection {
    Proposal(RecoveryProposalRejection),
    Basis(RecoveryQualificationRejection),
    ApprovalAfterLatestEvidence {
        approval_issued_step: u64,
        evidence_step: u64,
    },
}

impl From<RecoveryProposalRejection> for RecoveryOwnerPreparationRejection {
    fn from(value: RecoveryProposalRejection) -> Self {
        Self::Proposal(value)
    }
}

impl From<RecoveryQualificationRejection> for RecoveryOwnerPreparationRejection {
    fn from(value: RecoveryQualificationRejection) -> Self {
        Self::Basis(value)
    }
}

#[derive(Debug)]
#[must_use = "prepared recovery issuance must be consumed by the authoritative owner or discarded"]
pub(crate) struct PreparedRecoveryIssuanceV1 {
    proposal: RecoveryProposalV1,
    issue_step: u64,
}

impl PreparedRecoveryIssuanceV1 {
    pub(crate) const fn proposal(&self) -> RecoveryProposalV1 {
        self.proposal
    }

    pub(crate) const fn issue_step(&self) -> u64 {
        self.issue_step
    }
}

#[derive(Debug)]
#[must_use = "prepared recovery admission must be consumed by the authoritative owner or discarded"]
pub(crate) struct PreparedRecoveryAdmissionV1 {
    approval: RecoveryApprovalEnvelopeV1,
    evidence_step: u64,
}

impl PreparedRecoveryAdmissionV1 {
    pub(crate) const fn proposal_id(&self) -> u64 {
        self.approval.proposal.proposal_id()
    }

    pub(crate) const fn evidence_step(&self) -> u64 {
        self.evidence_step
    }
}

/// Prepare exact proposal issuance from immutable live embodiment state.
///
/// The eventual `SubterraneanEmbodiment` public wrapper can call this first,
/// allowing the immutable borrow to end before mutating its internal
/// `operator_authority` field.
pub(crate) fn prepare_recovery_issuance(
    embodiment: &SubterraneanEmbodiment,
    host: RecoveryHostBindingV1,
    proposal_id: u64,
    expires_step: u64,
) -> Result<PreparedRecoveryIssuanceV1, RecoveryOwnerPreparationRejection> {
    let basis = qualify_recovery_basis(embodiment, host)?;
    let issue_step = basis.evidence_step();
    let host = basis.host();
    let proposal = RecoveryProposalV1::new(
        proposal_id,
        basis.active_constraint(),
        basis.safety_snapshot_digest(),
        basis.evidence_snapshot_digest(),
        host.deployment_identity_digest(),
        host.controller_epoch(),
        host.control_plane_generation(),
        issue_step,
        expires_step,
    );
    proposal.validate(issue_step, embodiment.operator_constraint())?;
    Ok(PreparedRecoveryIssuanceV1 {
        proposal,
        issue_step,
    })
}

/// Prepare one approval for quorum admission from immutable live embodiment
/// state. Evidence supporting the widening must be at least as recent as the
/// human approval itself.
pub(crate) fn prepare_recovery_admission(
    embodiment: &SubterraneanEmbodiment,
    host: RecoveryHostBindingV1,
    approval: RecoveryApprovalEnvelopeV1,
) -> Result<PreparedRecoveryAdmissionV1, RecoveryOwnerPreparationRejection> {
    approval.validate_proposal_time()?;
    let basis = requalify_recovery_proposal(embodiment, host, approval.proposal)?;
    let evidence_step = basis.evidence_step();
    if approval.approval_issued_step > evidence_step {
        return Err(
            RecoveryOwnerPreparationRejection::ApprovalAfterLatestEvidence {
                approval_issued_step: approval.approval_issued_step,
                evidence_step,
            },
        );
    }
    approval
        .proposal
        .validate(evidence_step, embodiment.operator_constraint())?;
    Ok(PreparedRecoveryAdmissionV1 {
        approval,
        evidence_step,
    })
}

impl OperatorAuthority {
    /// Internal mutation phase for an issuance already prepared from live state.
    pub(crate) fn issue_prepared_recovery(
        &mut self,
        prepared: PreparedRecoveryIssuanceV1,
    ) -> Result<RecoveryProposalV1, OperatorAuthorityRejection> {
        self.issue_recovery_proposal(prepared.proposal, prepared.issue_step)
    }

    /// Internal mutation phase for an approval already prepared from live state.
    pub(crate) fn admit_prepared_recovery(
        &mut self,
        prepared: PreparedRecoveryAdmissionV1,
    ) -> Result<OperatorDecision, OperatorAuthorityRejection> {
        self.approve_recovery(prepared.approval, prepared.evidence_step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator_authority::OperatorConstraint;
    use crate::operator_protocol::{
        AuthenticationLevel, OperatorCommand, OperatorCommandEnvelope, OperatorId, OperatorRole,
    };
    use super::super::RecoveryDigest;
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

    fn host() -> RecoveryHostBindingV1 {
        RecoveryHostBindingV1::new(RecoveryDigest([41; 32]), 7, 11)
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
    ) -> (SubterraneanEmbodiment, OperatorAuthority, RecoveryProposalV1, u64) {
        let genesis = GenesisSeed::from_phrase(phrase);
        let mut embodiment = SubterraneanEmbodiment::new(&genesis);
        embodiment
            .ingest_operator_command(command(90, 1, 1, OperatorCommand::HoldPosition, 0))
            .expect("embodiment hold should be accepted");
        let evidence_step = step_once(&mut embodiment, 501);

        let issuance = prepare_recovery_issuance(
            &embodiment,
            host(),
            61,
            evidence_step.saturating_add(100),
        )
        .expect("issuance should prepare");

        let mut authority = OperatorAuthority::default();
        authority
            .ingest(
                command(99, 1, 1, OperatorCommand::HoldPosition, 0),
                evidence_step,
                true,
            )
            .expect("characterization authority hold should be accepted");
        assert_eq!(authority.constraint(), OperatorConstraint::HoldPosition);
        let proposal = authority
            .issue_prepared_recovery(issuance)
            .expect("prepared issuance should enter internal state machine");

        (embodiment, authority, proposal, evidence_step)
    }

    #[test]
    fn prepared_issuance_uses_live_evidence_step() {
        let genesis = GenesisSeed::from_phrase("owner-preparation-issuance");
        let mut embodiment = SubterraneanEmbodiment::new(&genesis);
        embodiment
            .ingest_operator_command(command(90, 1, 1, OperatorCommand::HoldPosition, 0))
            .expect("hold should be accepted");
        let evidence_step = step_once(&mut embodiment, 502);
        let prepared = prepare_recovery_issuance(
            &embodiment,
            host(),
            62,
            evidence_step.saturating_add(100),
        )
        .expect("issuance should prepare");
        assert_eq!(prepared.issue_step(), evidence_step);
        assert_eq!(prepared.proposal().issued_step(), evidence_step);
    }

    #[test]
    fn prepared_admission_requires_evidence_after_approval() {
        let (mut embodiment, _authority, proposal, evidence_step) =
            prepared_hold("owner-preparation-freshness");
        let approval_step = evidence_step.saturating_add(1);
        assert!(matches!(
            prepare_recovery_admission(
                &embodiment,
                host(),
                approval(1, 1, proposal, approval_step),
            ),
            Err(RecoveryOwnerPreparationRejection::ApprovalAfterLatestEvidence { .. })
        ));
        let later_step = step_once(&mut embodiment, 503);
        assert!(later_step >= approval_step);
        let prepared = prepare_recovery_admission(
            &embodiment,
            host(),
            approval(1, 1, proposal, approval_step),
        )
        .expect("fresh evidence should prepare admission");
        assert_eq!(prepared.evidence_step(), later_step);
        assert_eq!(prepared.proposal_id(), proposal.proposal_id());
    }

    #[test]
    fn prepared_admissions_can_drive_internal_quorum_without_requalification_token_reuse() {
        let (mut embodiment, mut authority, proposal, first_step) =
            prepared_hold("owner-preparation-quorum");

        let first = prepare_recovery_admission(
            &embodiment,
            host(),
            approval(1, 1, proposal, first_step),
        )
        .expect("first approval should prepare");
        assert!(matches!(
            authority
                .admit_prepared_recovery(first)
                .expect("first prepared approval should count"),
            OperatorDecision::PendingQuorum { .. }
        ));

        let second_step = step_once(&mut embodiment, 504);
        let second = prepare_recovery_admission(
            &embodiment,
            host(),
            approval(2, 1, proposal, second_step),
        )
        .expect("second approval should prepare from fresh state");
        assert_eq!(
            authority
                .admit_prepared_recovery(second)
                .expect("second prepared approval should clear characterization authority"),
            OperatorDecision::Cleared
        );
    }
}
