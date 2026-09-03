// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bound recovery proposals for human authority widening.
//!
//! A recovery quorum must approve the exact restriction and evidence snapshot it
//! reviewed. This module does not verify signatures or the provenance of opaque
//! digests; those remain responsibilities of the upstream trust/security boundary.
//!
//! ## Public API boundary
//!
//! `OperatorAuthority` remains a public type for ordinary restriction/narrowing
//! operations, but recovery widening is intentionally not a downstream-callable
//! method on an arbitrary authority instance. The eventual positive public
//! widening path is owned by `SubterraneanEmbodiment` (see HA-06A / #331).
//!
//! This normal doctest makes the crate path explicit so the compile-fail tests
//! below cannot succeed merely because the crate import itself is wrong.
//!
//! ```no_run
//! use symthaea_subterranean::OperatorAuthority;
//! let _authority = OperatorAuthority::default();
//! ```
//!
//! Raw proposal issuance is internal:
//!
//! ```compile_fail
//! use symthaea_subterranean::OperatorAuthority;
//! use symthaea_subterranean::operator_authority::recovery_authority::{
//!     RecoveryDigest, RecoveryProposalV1,
//! };
//! use symthaea_subterranean::OperatorConstraint;
//!
//! let mut authority = OperatorAuthority::default();
//! let proposal = RecoveryProposalV1::new(
//!     1,
//!     OperatorConstraint::HoldPosition,
//!     RecoveryDigest([1; 32]),
//!     RecoveryDigest([2; 32]),
//!     RecoveryDigest([3; 32]),
//!     1,
//!     1,
//!     1,
//!     10,
//! );
//! let _ = authority.issue_recovery_proposal(proposal, 1);
//! ```
//!
//! Raw approval admission is internal:
//!
//! ```compile_fail
//! use symthaea_subterranean::OperatorAuthority;
//! let _raw_approve = OperatorAuthority::approve_recovery;
//! ```
//!
//! Qualified issuance/admission on arbitrary authority instances are also
//! internal until the embodiment-owned wrapper exists:
//!
//! ```compile_fail
//! use symthaea_subterranean::OperatorAuthority;
//! let _qualified_issue = OperatorAuthority::issue_qualified_recovery_proposal;
//! ```
//!
//! ```compile_fail
//! use symthaea_subterranean::OperatorAuthority;
//! let _qualified_approve = OperatorAuthority::approve_qualified_recovery;
//! ```
//!
//! The evidence-derived live helper is likewise internal because its two-object
//! signature is not itself an ownership guarantee:
//!
//! ```compile_fail
//! use symthaea_subterranean::OperatorAuthority;
//! let _live_gate = OperatorAuthority::approve_recovery_from_live_state;
//! ```

pub mod approval;
pub mod live_gate;
pub mod qualification;

pub use approval::{
    QualifiedRecoveryApprovalV1, RecoveryApprovalAdmissionRejection,
    RecoveryApprovalQualificationRejection, qualify_recovery_approval,
};
pub use live_gate::LiveRecoveryAdmissionRejection;
pub use qualification::{
    QualifiedRecoveryBasisV1, RecoveryHostBindingV1, RecoveryQualificationRejection,
    qualify_recovery_basis, requalify_recovery_proposal,
};

use crate::operator_authority::OperatorConstraint;
use crate::operator_protocol::{
    AuthenticationLevel, OperatorCommand, OperatorCommandEnvelope, OperatorId, OperatorRole,
};
use serde::{Deserialize, Serialize};

pub const RECOVERY_PROPOSAL_SCHEMA_VERSION: u16 = 1;
const RECOVERY_PROPOSAL_DOMAIN: &[u8] = b"symthaea-subterranean/recovery-proposal-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct RecoveryDigest(pub [u8; 32]);

impl RecoveryDigest {
    pub const fn is_valid(self) -> bool {
        let mut index = 0;
        while index < self.0.len() {
            if self.0[index] != 0 {
                return true;
            }
            index += 1;
        }
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryProposalV1 {
    schema_version: u16,
    proposal_id: u64,
    active_constraint: OperatorConstraint,
    target_constraint: OperatorConstraint,
    safety_snapshot_digest: RecoveryDigest,
    evidence_snapshot_digest: RecoveryDigest,
    deployment_identity_digest: RecoveryDigest,
    controller_epoch: u64,
    control_plane_generation: u64,
    issued_step: u64,
    expires_step: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryProposalRejection {
    InvalidSchema,
    InvalidProposalId,
    InvalidDigest,
    InvalidLifetime,
    NotYetValid,
    Expired,
    ActiveConstraintMismatch,
    UnsupportedActiveConstraint,
    UnsupportedTargetConstraint,
    InvalidControllerEpoch,
    InvalidControlPlaneGeneration,
    ApprovalPredatesProposal,
}

impl RecoveryProposalV1 {
    /// Construct portable proposal evidence. Construction alone does not make a
    /// proposal authoritative: the trusted owner must explicitly issue the exact
    /// proposal before any approvals can count toward recovery.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        proposal_id: u64,
        active_constraint: OperatorConstraint,
        safety_snapshot_digest: RecoveryDigest,
        evidence_snapshot_digest: RecoveryDigest,
        deployment_identity_digest: RecoveryDigest,
        controller_epoch: u64,
        control_plane_generation: u64,
        issued_step: u64,
        expires_step: u64,
    ) -> Self {
        Self {
            schema_version: RECOVERY_PROPOSAL_SCHEMA_VERSION,
            proposal_id,
            active_constraint,
            target_constraint: OperatorConstraint::None,
            safety_snapshot_digest,
            evidence_snapshot_digest,
            deployment_identity_digest,
            controller_epoch,
            control_plane_generation,
            issued_step,
            expires_step,
        }
    }

    pub const fn proposal_id(self) -> u64 {
        self.proposal_id
    }

    pub const fn active_constraint(self) -> OperatorConstraint {
        self.active_constraint
    }

    pub const fn target_constraint(self) -> OperatorConstraint {
        self.target_constraint
    }

    pub const fn safety_snapshot_digest(self) -> RecoveryDigest {
        self.safety_snapshot_digest
    }

    pub const fn evidence_snapshot_digest(self) -> RecoveryDigest {
        self.evidence_snapshot_digest
    }

    pub const fn deployment_identity_digest(self) -> RecoveryDigest {
        self.deployment_identity_digest
    }

    pub const fn controller_epoch(self) -> u64 {
        self.controller_epoch
    }

    pub const fn control_plane_generation(self) -> u64 {
        self.control_plane_generation
    }

    pub const fn issued_step(self) -> u64 {
        self.issued_step
    }

    pub const fn expires_step(self) -> u64 {
        self.expires_step
    }

    pub fn validate(
        self,
        now_step: u64,
        current_constraint: OperatorConstraint,
    ) -> Result<(), RecoveryProposalRejection> {
        if self.schema_version != RECOVERY_PROPOSAL_SCHEMA_VERSION {
            return Err(RecoveryProposalRejection::InvalidSchema);
        }
        if self.proposal_id == 0 {
            return Err(RecoveryProposalRejection::InvalidProposalId);
        }
        if !self.safety_snapshot_digest.is_valid()
            || !self.evidence_snapshot_digest.is_valid()
            || !self.deployment_identity_digest.is_valid()
        {
            return Err(RecoveryProposalRejection::InvalidDigest);
        }
        if self.expires_step < self.issued_step {
            return Err(RecoveryProposalRejection::InvalidLifetime);
        }
        if now_step < self.issued_step {
            return Err(RecoveryProposalRejection::NotYetValid);
        }
        if now_step > self.expires_step {
            return Err(RecoveryProposalRejection::Expired);
        }
        if self.active_constraint != current_constraint {
            return Err(RecoveryProposalRejection::ActiveConstraintMismatch);
        }
        if matches!(
            self.active_constraint,
            OperatorConstraint::None | OperatorConstraint::Mission(_)
        ) {
            return Err(RecoveryProposalRejection::UnsupportedActiveConstraint);
        }
        if self.target_constraint != OperatorConstraint::None {
            return Err(RecoveryProposalRejection::UnsupportedTargetConstraint);
        }
        if self.controller_epoch == 0 {
            return Err(RecoveryProposalRejection::InvalidControllerEpoch);
        }
        if self.control_plane_generation == 0 {
            return Err(RecoveryProposalRejection::InvalidControlPlaneGeneration);
        }
        Ok(())
    }

    /// Canonical bytes for an upstream signer/hasher. This function provides
    /// deterministic field binding, not cryptographic authentication by itself.
    pub fn canonical_bytes(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(160);
        out.extend_from_slice(RECOVERY_PROPOSAL_DOMAIN);
        out.extend_from_slice(&self.schema_version.to_be_bytes());
        out.extend_from_slice(&self.proposal_id.to_be_bytes());
        out.extend_from_slice(&self.active_constraint.code().to_be_bytes());
        out.extend_from_slice(&self.target_constraint.code().to_be_bytes());
        out.extend_from_slice(&self.safety_snapshot_digest.0);
        out.extend_from_slice(&self.evidence_snapshot_digest.0);
        out.extend_from_slice(&self.deployment_identity_digest.0);
        out.extend_from_slice(&self.controller_epoch.to_be_bytes());
        out.extend_from_slice(&self.control_plane_generation.to_be_bytes());
        out.extend_from_slice(&self.issued_step.to_be_bytes());
        out.extend_from_slice(&self.expires_step.to_be_bytes());
        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryApprovalEnvelopeV1 {
    pub operator: OperatorId,
    pub role: OperatorRole,
    pub authentication: AuthenticationLevel,
    pub epoch: u64,
    pub sequence: u64,
    pub approval_issued_step: u64,
    pub proposal: RecoveryProposalV1,
}

impl RecoveryApprovalEnvelopeV1 {
    pub const fn as_command_envelope(self) -> OperatorCommandEnvelope {
        OperatorCommandEnvelope {
            operator: self.operator,
            role: self.role,
            authentication: self.authentication,
            epoch: self.epoch,
            sequence: self.sequence,
            proposal_id: self.proposal.proposal_id,
            issued_step: self.approval_issued_step,
            expires_step: self.proposal.expires_step,
            command: OperatorCommand::ResumeNominal,
        }
    }

    pub const fn validate_proposal_time(self) -> Result<(), RecoveryProposalRejection> {
        if self.approval_issued_step < self.proposal.issued_step {
            Err(RecoveryProposalRejection::ApprovalPredatesProposal)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> RecoveryDigest {
        RecoveryDigest([byte; 32])
    }

    #[test]
    fn canonical_bytes_change_when_bound_evidence_changes() {
        let a = RecoveryProposalV1::new(
            9,
            OperatorConstraint::EmergencyStop,
            digest(1),
            digest(2),
            digest(3),
            4,
            5,
            10,
            20,
        );
        let b = RecoveryProposalV1::new(
            9,
            OperatorConstraint::EmergencyStop,
            digest(1),
            digest(7),
            digest(3),
            4,
            5,
            10,
            20,
        );
        assert_ne!(a.canonical_bytes(), b.canonical_bytes());
    }

    #[test]
    fn proposal_is_bound_to_exact_active_constraint() {
        let proposal = RecoveryProposalV1::new(
            9,
            OperatorConstraint::HoldPosition,
            digest(1),
            digest(2),
            digest(3),
            4,
            5,
            10,
            20,
        );
        assert_eq!(
            proposal.validate(12, OperatorConstraint::EmergencyStop),
            Err(RecoveryProposalRejection::ActiveConstraintMismatch)
        );
    }

    #[test]
    fn approval_cannot_claim_to_predate_its_proposal() {
        let proposal = RecoveryProposalV1::new(
            9,
            OperatorConstraint::HoldPosition,
            digest(1),
            digest(2),
            digest(3),
            4,
            5,
            10,
            20,
        );
        let approval = RecoveryApprovalEnvelopeV1 {
            operator: OperatorId(1),
            role: OperatorRole::SafetyOfficer,
            authentication: AuthenticationLevel::HardwareBacked,
            epoch: 1,
            sequence: 1,
            approval_issued_step: 9,
            proposal,
        };
        assert_eq!(
            approval.validate_proposal_time(),
            Err(RecoveryProposalRejection::ApprovalPredatesProposal)
        );
    }
}
