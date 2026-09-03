// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Host-local qualification for operator-authority recovery proposals.
//!
//! A portable [`RecoveryProposalV1`] is evidence about a proposed widening;
//! it is not itself proof that the running system is currently safe to widen.
//! This module derives the safety/evidence commitments from the live
//! [`SubterraneanEmbodiment`] and returns a non-serializable qualification
//! object. Independent degraded, partition, temporal, capability, and field
//! constraints are committed into the basis but are not required to be nominal:
//! clearing an operator restriction must not impersonate clearing another
//! authority source.
//!
//! `RecoveryHostBindingV1` intentionally does not verify cryptographic
//! provenance. Its deployment/controller/control-plane identities must come
//! from the upstream trusted host/Xenia boundary.

use super::{RecoveryDigest, RecoveryProposalRejection, RecoveryProposalV1};
use crate::embodiment::SubterraneanEmbodiment;
use crate::operator_authority::{
    OperatorAuthority, OperatorAuthorityRejection, OperatorConstraint,
};
use crate::safety::SubterraneanHazard;
use crate::update_control::UpdateState;

const SAFETY_BASIS_DOMAIN: &[u8] = b"symthaea-subterranean/recovery-qualified-safety-v1";
const EVIDENCE_BASIS_DOMAIN: &[u8] = b"symthaea-subterranean/recovery-qualified-evidence-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecoveryHostBindingV1 {
    deployment_identity_digest: RecoveryDigest,
    controller_epoch: u64,
    control_plane_generation: u64,
}

impl RecoveryHostBindingV1 {
    pub fn new(
        deployment_identity_digest: RecoveryDigest,
        controller_epoch: u64,
        control_plane_generation: u64,
    ) -> Result<Self, RecoveryQualificationRejection> {
        let value = Self {
            deployment_identity_digest,
            controller_epoch,
            control_plane_generation,
        };
        if !value.is_valid() {
            return Err(RecoveryQualificationRejection::InvalidHostBinding);
        }
        Ok(value)
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

    pub const fn is_valid(self) -> bool {
        self.deployment_identity_digest.is_valid()
            && self.controller_epoch > 0
            && self.control_plane_generation > 0
    }
}

/// Host-local proof that the currently observed runtime state is sufficient to
/// *propose* removal of the exact active operator restriction.
///
/// This type deliberately does not implement `Serialize`/`Deserialize`. It must
/// be reconstructed from current live state after restart or material change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QualifiedRecoveryBasisV1 {
    active_constraint: OperatorConstraint,
    safety_snapshot_digest: RecoveryDigest,
    evidence_snapshot_digest: RecoveryDigest,
    host: RecoveryHostBindingV1,
    evidence_step: u64,
}

impl QualifiedRecoveryBasisV1 {
    pub const fn active_constraint(self) -> OperatorConstraint {
        self.active_constraint
    }

    pub const fn safety_snapshot_digest(self) -> RecoveryDigest {
        self.safety_snapshot_digest
    }

    pub const fn evidence_snapshot_digest(self) -> RecoveryDigest {
        self.evidence_snapshot_digest
    }

    pub const fn host(self) -> RecoveryHostBindingV1 {
        self.host
    }

    pub const fn evidence_step(self) -> u64 {
        self.evidence_step
    }

    pub fn matches_proposal(self, proposal: RecoveryProposalV1) -> bool {
        proposal.active_constraint() == self.active_constraint
            && proposal.target_constraint() == OperatorConstraint::None
            && proposal.safety_snapshot_digest() == self.safety_snapshot_digest
            && proposal.evidence_snapshot_digest() == self.evidence_snapshot_digest
            && proposal.deployment_identity_digest() == self.host.deployment_identity_digest
            && proposal.controller_epoch() == self.host.controller_epoch
            && proposal.control_plane_generation() == self.host.control_plane_generation
    }

    fn proposal(
        self,
        proposal_id: u64,
        issued_step: u64,
        expires_step: u64,
    ) -> RecoveryProposalV1 {
        RecoveryProposalV1::new(
            proposal_id,
            self.active_constraint,
            self.safety_snapshot_digest,
            self.evidence_snapshot_digest,
            self.host.deployment_identity_digest,
            self.host.controller_epoch,
            self.host.control_plane_generation,
            issued_step,
            expires_step,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryQualificationRejection {
    InvalidHostBinding,
    NoActiveRestriction,
    PhysicalHazardActive,
    StateIntegrityInvalid,
    CriticalSensorEvidenceUnavailable,
    InvariantViolationActive,
    UpdateTransitionActive,
    MissingEvidence,
    EvidenceConstraintMismatch,
    EvidenceHazardActive,
    EvidenceSensorDegraded,
    EvidenceInvariantViolation,
    ProposalBasisMismatch,
}

fn append_bool(hasher: &mut blake3::Hasher, value: bool) {
    hasher.update(&[u8::from(value)]);
}

fn append_u16(hasher: &mut blake3::Hasher, value: u16) {
    hasher.update(&value.to_be_bytes());
}

fn append_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_be_bytes());
}

fn append_usize(hasher: &mut blake3::Hasher, value: usize) {
    append_u64(hasher, u64::try_from(value).unwrap_or(u64::MAX));
}

fn append_str(hasher: &mut blake3::Hasher, value: &str) {
    append_usize(hasher, value.len());
    hasher.update(value.as_bytes());
}

fn digest(hasher: blake3::Hasher) -> RecoveryDigest {
    RecoveryDigest(*hasher.finalize().as_bytes())
}

/// Evaluate the exact current runtime basis for clearing only the active
/// operator restriction.
///
/// Independent authority sources are intentionally not required to be nominal.
/// For example, a partition reconciliation hold may remain after an operator
/// emergency stop is cleared. Those states are committed into the returned
/// basis so a material change forces a fresh proposal/review.
pub fn qualify_recovery_basis(
    embodiment: &SubterraneanEmbodiment,
    host: RecoveryHostBindingV1,
) -> Result<QualifiedRecoveryBasisV1, RecoveryQualificationRejection> {
    if !host.is_valid() {
        return Err(RecoveryQualificationRejection::InvalidHostBinding);
    }

    let active_constraint = embodiment.operator_constraint();
    if matches!(
        active_constraint,
        OperatorConstraint::None | OperatorConstraint::Mission(_)
    ) {
        return Err(RecoveryQualificationRejection::NoActiveRestriction);
    }

    if embodiment.last_hazard().primary != SubterraneanHazard::None {
        return Err(RecoveryQualificationRejection::PhysicalHazardActive);
    }

    // `update_preconditions` recomputes the same physical-hazard + simulator
    // integrity check used by the embodiment's existing update boundary. Since
    // the explicit hazard was checked above, a false result here means the
    // physical state integrity report is not valid.
    if !embodiment.update_preconditions().physical_hazard_clear {
        return Err(RecoveryQualificationRejection::StateIntegrityInvalid);
    }

    let sensor = embodiment.sensor_fusion_report();
    if sensor.requires_fail_closed() {
        return Err(RecoveryQualificationRejection::CriticalSensorEvidenceUnavailable);
    }

    let invariant = embodiment.invariant_assessment();
    if !invariant.passed() {
        return Err(RecoveryQualificationRejection::InvariantViolationActive);
    }

    if matches!(
        embodiment.update_state(),
        Some(UpdateState::Staged | UpdateState::PendingHealth | UpdateState::RollbackRequired)
    ) {
        return Err(RecoveryQualificationRejection::UpdateTransitionActive);
    }

    let records = embodiment.evidence_records();
    let latest = records
        .last()
        .ok_or(RecoveryQualificationRejection::MissingEvidence)?;

    if latest.authority.operator_constraint != active_constraint.label() {
        return Err(RecoveryQualificationRejection::EvidenceConstraintMismatch);
    }
    if latest.raw_hazard != SubterraneanHazard::None.label()
        || latest.latched_hazard != SubterraneanHazard::None.label()
    {
        return Err(RecoveryQualificationRejection::EvidenceHazardActive);
    }
    if latest.sensor_quality.critical_degraded_channels > 0
        || latest.survivability.critical_channels_without_quorum > 0
    {
        return Err(RecoveryQualificationRejection::EvidenceSensorDegraded);
    }
    if !latest.certification.invariant_violations.is_empty()
        || latest.certification.invariant_command_modified
    {
        return Err(RecoveryQualificationRejection::EvidenceInvariantViolation);
    }

    let partition = embodiment.partition_recovery_assessment();
    let temporal = embodiment.temporal_assessment();
    let capability = embodiment.capability_profile();
    let envelope = embodiment.field_envelope_assessment();
    let isolation = embodiment.actuator_isolation_report();

    // Commit discrete safety-relevant state rather than volatile raw sensor
    // values. This lets two operators review one stable proposal while still
    // invalidating it when the authority/safety topology materially changes.
    let mut safety_hasher = blake3::Hasher::new();
    safety_hasher.update(SAFETY_BASIS_DOMAIN);
    append_u16(&mut safety_hasher, active_constraint.code());
    append_usize(&mut safety_hasher, sensor.accepted_sources);
    append_usize(&mut safety_hasher, sensor.isolated_sources);
    append_usize(
        &mut safety_hasher,
        sensor.critical_channels_without_quorum,
    );
    append_bool(&mut safety_hasher, invariant.passed());
    append_bool(&mut safety_hasher, invariant.command_modified);
    append_usize(&mut safety_hasher, invariant.violations.len());
    for violation in &invariant.violations {
        append_str(&mut safety_hasher, violation.code());
    }
    append_str(&mut safety_hasher, embodiment.degraded_mode().label());
    append_str(&mut safety_hasher, partition.mode.label());
    append_str(&mut safety_hasher, temporal.authority.label());
    append_str(&mut safety_hasher, capability.disposition.label());
    append_str(&mut safety_hasher, envelope.mode.label());
    append_usize(&mut safety_hasher, isolation.isolated_count);
    match embodiment.update_state() {
        Some(state) => {
            append_bool(&mut safety_hasher, true);
            append_u16(&mut safety_hasher, state.code());
        }
        None => append_bool(&mut safety_hasher, false),
    }
    let safety_snapshot_digest = digest(safety_hasher);

    let mut evidence_hasher = blake3::Hasher::new();
    evidence_hasher.update(EVIDENCE_BASIS_DOMAIN);
    append_str(
        &mut evidence_hasher,
        latest.authority.operator_constraint.as_str(),
    );
    append_str(&mut evidence_hasher, latest.raw_hazard.as_str());
    append_str(&mut evidence_hasher, latest.latched_hazard.as_str());
    append_usize(
        &mut evidence_hasher,
        latest.sensor_quality.critical_degraded_channels,
    );
    append_usize(
        &mut evidence_hasher,
        latest.survivability.critical_channels_without_quorum,
    );
    append_usize(
        &mut evidence_hasher,
        latest.survivability.isolated_actuators,
    );
    append_str(
        &mut evidence_hasher,
        latest.survivability.envelope_mode.as_str(),
    );
    append_str(
        &mut evidence_hasher,
        latest.survivability.capability_disposition.as_str(),
    );
    append_str(
        &mut evidence_hasher,
        latest.survivability.partition_mode.as_str(),
    );
    append_str(
        &mut evidence_hasher,
        latest.authority.degraded_mode.as_str(),
    );
    match latest.authority.update_state.as_deref() {
        Some(state) => {
            append_bool(&mut evidence_hasher, true);
            append_str(&mut evidence_hasher, state);
        }
        None => append_bool(&mut evidence_hasher, false),
    }
    append_bool(
        &mut evidence_hasher,
        latest.certification.invariant_command_modified,
    );
    let mut evidence_violations = latest.certification.invariant_violations.clone();
    evidence_violations.sort();
    append_usize(&mut evidence_hasher, evidence_violations.len());
    for violation in evidence_violations {
        append_str(&mut evidence_hasher, violation.as_str());
    }
    append_bool(&mut evidence_hasher, latest.return_path.feasible);
    let evidence_snapshot_digest = digest(evidence_hasher);

    Ok(QualifiedRecoveryBasisV1 {
        active_constraint,
        safety_snapshot_digest,
        evidence_snapshot_digest,
        host,
        evidence_step: latest.step,
    })
}

/// Recompute the live recovery basis and require an existing portable proposal
/// to match it exactly. A future embodiment-level approval wrapper should call
/// this before forwarding the approval into the internal authority state machine.
pub fn requalify_recovery_proposal(
    embodiment: &SubterraneanEmbodiment,
    host: RecoveryHostBindingV1,
    proposal: RecoveryProposalV1,
) -> Result<QualifiedRecoveryBasisV1, RecoveryQualificationRejection> {
    let basis = qualify_recovery_basis(embodiment, host)?;
    if !basis.matches_proposal(proposal) {
        return Err(RecoveryQualificationRejection::ProposalBasisMismatch);
    }
    Ok(basis)
}

impl OperatorAuthority {
    /// Internal qualified issuance primitive. The safety/evidence commitments are
    /// copied from a host-local qualified basis rather than supplied as arbitrary
    /// digests. Public issuance must be owned by the embodiment/control-plane
    /// that owns the authoritative `OperatorAuthority` instance.
    pub(crate) fn issue_qualified_recovery_proposal(
        &mut self,
        basis: QualifiedRecoveryBasisV1,
        proposal_id: u64,
        now_step: u64,
        expires_step: u64,
    ) -> Result<RecoveryProposalV1, OperatorAuthorityRejection> {
        if now_step < basis.evidence_step {
            return Err(OperatorAuthorityRejection::RecoveryProposal(
                RecoveryProposalRejection::NotYetValid,
            ));
        }
        let proposal = basis.proposal(proposal_id, now_step, expires_step);
        self.issue_recovery_proposal(proposal, now_step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator_protocol::{
        AuthenticationLevel, OperatorCommand, OperatorCommandEnvelope, OperatorId, OperatorRole,
    };
    use symthaea_core::genesis::GenesisSeed;
    use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

    fn host() -> RecoveryHostBindingV1 {
        RecoveryHostBindingV1::new(RecoveryDigest([9; 32]), 7, 11).expect("valid host binding")
    }

    fn operator_command(command: OperatorCommand) -> OperatorCommandEnvelope {
        OperatorCommandEnvelope {
            operator: OperatorId(1),
            role: OperatorRole::SafetyOfficer,
            authentication: AuthenticationLevel::HardwareBacked,
            epoch: 1,
            sequence: 1,
            proposal_id: 1,
            issued_step: 0,
            expires_step: 1_000,
            command,
        }
    }

    fn step_once(embodiment: &mut SubterraneanEmbodiment, seed: u64) {
        let thought = ContinuousHV::random(HDC_DIMENSION, seed);
        let _ = embodiment.step(&thought, 0.005, 0.9);
    }

    #[test]
    fn nominal_state_has_nothing_to_recover() {
        let genesis = GenesisSeed::from_phrase("qualified-recovery-nominal");
        let mut embodiment = SubterraneanEmbodiment::new(&genesis);
        step_once(&mut embodiment, 1);
        assert_eq!(
            qualify_recovery_basis(&embodiment, host()),
            Err(RecoveryQualificationRejection::NoActiveRestriction)
        );
    }

    #[test]
    fn active_hold_requires_evidence_recorded_after_the_restriction() {
        let genesis = GenesisSeed::from_phrase("qualified-recovery-needs-evidence");
        let mut embodiment = SubterraneanEmbodiment::new(&genesis);
        step_once(&mut embodiment, 2);
        embodiment
            .ingest_operator_command(operator_command(OperatorCommand::HoldPosition))
            .expect("hold should be accepted");
        assert_eq!(
            qualify_recovery_basis(&embodiment, host()),
            Err(RecoveryQualificationRejection::EvidenceConstraintMismatch)
        );
    }

    #[test]
    fn live_hold_can_produce_a_host_qualified_basis() {
        let genesis = GenesisSeed::from_phrase("qualified-recovery-live-hold");
        let mut embodiment = SubterraneanEmbodiment::new(&genesis);
        embodiment
            .ingest_operator_command(operator_command(OperatorCommand::HoldPosition))
            .expect("hold should be accepted");
        step_once(&mut embodiment, 3);

        let basis = qualify_recovery_basis(&embodiment, host()).expect("basis should qualify");
        assert_eq!(basis.active_constraint(), OperatorConstraint::HoldPosition);
        assert!(basis.safety_snapshot_digest().is_valid());
        assert!(basis.evidence_snapshot_digest().is_valid());
    }

    #[test]
    fn qualified_basis_builds_proposal_from_live_commitments() {
        let genesis = GenesisSeed::from_phrase("qualified-recovery-proposal");
        let mut embodiment = SubterraneanEmbodiment::new(&genesis);
        embodiment
            .ingest_operator_command(operator_command(OperatorCommand::HoldPosition))
            .expect("hold should be accepted");
        step_once(&mut embodiment, 4);

        let basis = qualify_recovery_basis(&embodiment, host()).expect("basis should qualify");
        let proposal = basis.proposal(17, basis.evidence_step(), basis.evidence_step() + 100);
        assert!(basis.matches_proposal(proposal));
    }

    #[test]
    fn changed_host_generation_invalidates_an_old_proposal_basis() {
        let genesis = GenesisSeed::from_phrase("qualified-recovery-generation");
        let mut embodiment = SubterraneanEmbodiment::new(&genesis);
        embodiment
            .ingest_operator_command(operator_command(OperatorCommand::HoldPosition))
            .expect("hold should be accepted");
        step_once(&mut embodiment, 5);

        let basis = qualify_recovery_basis(&embodiment, host()).expect("basis should qualify");
        let proposal = basis.proposal(21, basis.evidence_step(), basis.evidence_step() + 100);
        let changed_host = RecoveryHostBindingV1::new(RecoveryDigest([9; 32]), 7, 12)
            .expect("changed host binding remains structurally valid");
        assert_eq!(
            requalify_recovery_proposal(&embodiment, changed_host, proposal),
            Err(RecoveryQualificationRejection::ProposalBasisMismatch)
        );
    }

    #[test]
    fn zero_host_identity_is_rejected() {
        assert_eq!(
            RecoveryHostBindingV1::new(RecoveryDigest([0; 32]), 1, 1),
            Err(RecoveryQualificationRejection::InvalidHostBinding)
        );
    }
}
