// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Replay-resistant operator authority and safety-monotonic command constraints.

#[path = "recovery_authority.rs"]
pub mod recovery_authority;

use crate::embodiment::MotorSafetyLevel;
use crate::mission::SubterraneanMissionIntent;
use crate::operator_protocol::{
    OperatorCommand, OperatorCommandEnvelope, OperatorCommandRejection, OperatorId,
    OperatorTrustPolicy,
};
use crate::types::{SubterraneanCommand, SubterraneanState};
use recovery_authority::{
    RecoveryApprovalEnvelopeV1, RecoveryProposalRejection, RecoveryProposalV1,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperatorConstraint {
    None,
    EmergencyStop,
    HoldPosition,
    ReturnHome,
    Mission(SubterraneanMissionIntent),
    MaintenanceLock,
}

impl OperatorConstraint {
    pub const fn code(self) -> u16 {
        match self {
            Self::None => 0,
            Self::EmergencyStop => 1,
            Self::HoldPosition => 2,
            Self::ReturnHome => 3,
            Self::Mission(intent) => 10 + intent.index() as u16,
            Self::MaintenanceLock => 30,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::EmergencyStop => "emergency_stop",
            Self::HoldPosition => "hold_position",
            Self::ReturnHome => "return_home",
            Self::Mission(_) => "mission",
            Self::MaintenanceLock => "maintenance_lock",
        }
    }

    pub const fn mission_override(self) -> Option<SubterraneanMissionIntent> {
        match self {
            Self::EmergencyStop | Self::HoldPosition | Self::MaintenanceLock => Some(SubterraneanMissionIntent::HoldPosition),
            Self::ReturnHome => Some(SubterraneanMissionIntent::ReturnHome),
            Self::Mission(intent) => Some(intent),
            Self::None => None,
        }
    }

    pub const fn restrictiveness_rank(self) -> u8 {
        match self {
            Self::None | Self::Mission(_) => 0,
            Self::ReturnHome => 1,
            Self::HoldPosition => 2,
            Self::MaintenanceLock => 3,
            Self::EmergencyStop => 4,
        }
    }

    pub const fn more_restrictive(self, other: Self) -> Self {
        if other.restrictiveness_rank() > self.restrictiveness_rank() { other } else { self }
    }

    pub const fn can_replace_without_recovery(self, requested: Self) -> bool {
        match requested {
            Self::None => matches!(self, Self::None),
            Self::Mission(_) => matches!(self, Self::None | Self::Mission(_)),
            _ => requested.restrictiveness_rank() >= self.restrictiveness_rank(),
        }
    }

    pub const fn safety_floor(self) -> Option<MotorSafetyLevel> {
        match self {
            Self::EmergencyStop | Self::MaintenanceLock => Some(MotorSafetyLevel::Red),
            Self::HoldPosition => Some(MotorSafetyLevel::Orange),
            Self::None | Self::ReturnHome | Self::Mission(_) => None,
        }
    }

    pub fn constrain_nominal(self, mut command: SubterraneanCommand, state: &SubterraneanState) -> SubterraneanCommand {
        match self {
            Self::None | Self::Mission(_) => {}
            Self::ReturnHome => {
                command.set_cutter_head(0.0);
                command.set_auger_feed(0.0);
                command.set_left_track(command.left_track().min(-0.25));
                command.set_right_track(command.right_track().min(-0.25));
            }
            Self::EmergencyStop | Self::HoldPosition | Self::MaintenanceLock => {
                command.set_cutter_head(0.0);
                command.set_auger_feed(0.0);
                command.set_left_track(0.0);
                command.set_right_track(0.0);
                command.set_ballast_trim(0.0);
                command.recovery = Default::default();
                command.set_thermal_pump(if state.cutter_temp_c() >= 85.0 { 0.6 } else { 0.0 });
            }
        }
        command.sanitize();
        command
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperatorDecision {
    Applied(OperatorConstraint),
    PendingQuorum { approvals: usize, required: usize },
    Cleared,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorAuthorityRejection {
    Protocol(OperatorCommandRejection),
    StaleEpoch,
    Replay,
    ConflictingProposal,
    PhysicalHazardActive,
    WouldWeakenActiveConstraint,
    RecoveryProposalRequired,
    RecoveryProposalNotIssued,
    RecoveryProposal(RecoveryProposalRejection),
}

impl From<OperatorCommandRejection> for OperatorAuthorityRejection {
    fn from(value: OperatorCommandRejection) -> Self { Self::Protocol(value) }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PendingRecoveryApproval {
    proposal: RecoveryProposalV1,
    operators: BTreeSet<OperatorId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorAuthority {
    policy: OperatorTrustPolicy,
    last_sequence: BTreeMap<OperatorId, (u64, u64)>,
    /// Host-local issuance state. Portable checkpoints/evidence must not
    /// resurrect an outstanding authority-widening offer after restart.
    #[serde(skip, default)]
    issued_recovery: BTreeMap<u64, RecoveryProposalV1>,
    /// Host-local partial quorum state. A restart requires a fresh proposal and
    /// fresh approvals rather than replaying a pre-restart widening decision.
    #[serde(skip, default)]
    pending_resume: BTreeMap<u64, PendingRecoveryApproval>,
    constraint: OperatorConstraint,
    last_applied_proposal: Option<u64>,
    accepted_commands: u64,
    rejected_commands: u64,
}

impl OperatorAuthority {
    pub fn new(policy: OperatorTrustPolicy) -> Self {
        Self {
            policy,
            last_sequence: BTreeMap::new(),
            issued_recovery: BTreeMap::new(),
            pending_resume: BTreeMap::new(),
            constraint: OperatorConstraint::None,
            last_applied_proposal: None,
            accepted_commands: 0,
            rejected_commands: 0,
        }
    }

    pub fn constraint(&self) -> OperatorConstraint { self.constraint }

    pub fn validate(&self) -> bool {
        self.policy.maximum_command_age_steps > 0
            && self.policy.recovery_quorum >= 2
            && self.last_sequence.keys().all(|operator| operator.is_valid())
            && self.issued_recovery.iter().all(|(id, proposal)| *id == proposal.proposal_id())
            && self.pending_resume.iter().all(|(proposal_id, pending)| {
                *proposal_id == pending.proposal.proposal_id()
                    && self.issued_recovery.get(proposal_id) == Some(&pending.proposal)
                    && !pending.operators.is_empty()
                    && pending.operators.iter().all(|operator| operator.is_valid())
            })
    }

    pub fn accepted_commands(&self) -> u64 { self.accepted_commands }
    pub fn rejected_commands(&self) -> u64 { self.rejected_commands }
    pub fn pending_approvals(&self, proposal_id: u64) -> usize {
        self.pending_resume.get(&proposal_id).map_or(0, |pending| pending.operators.len())
    }
    pub fn last_applied_proposal(&self) -> Option<u64> { self.last_applied_proposal }
    pub fn issued_recovery_proposal(&self, proposal_id: u64) -> Option<RecoveryProposalV1> {
        self.issued_recovery.get(&proposal_id).copied()
    }

    fn reject<T>(&mut self, rejection: OperatorAuthorityRejection) -> Result<T, OperatorAuthorityRejection> {
        self.rejected_commands = self.rejected_commands.saturating_add(1);
        Err(rejection)
    }

    fn accept_sequence(&mut self, envelope: OperatorCommandEnvelope) -> Result<(), OperatorAuthorityRejection> {
        if let Some((epoch, sequence)) = self.last_sequence.get(&envelope.operator).copied() {
            if envelope.epoch < epoch { return self.reject(OperatorAuthorityRejection::StaleEpoch); }
            if envelope.epoch == epoch && envelope.sequence <= sequence { return self.reject(OperatorAuthorityRejection::Replay); }
        }
        self.last_sequence.insert(envelope.operator, (envelope.epoch, envelope.sequence));
        Ok(())
    }

    fn expire_pending(&mut self, now_step: u64) {
        self.issued_recovery.retain(|_, proposal| now_step <= proposal.expires_step());
        self.pending_resume.retain(|id, pending| {
            now_step <= pending.proposal.expires_step()
                && self.issued_recovery.get(id) == Some(&pending.proposal)
        });
    }

    fn invalidate_recovery_state(&mut self) {
        self.pending_resume.clear();
        self.issued_recovery.clear();
    }

    fn apply_non_recovery_constraint(
        &mut self,
        requested: OperatorConstraint,
        proposal_id: u64,
    ) -> Result<OperatorDecision, OperatorAuthorityRejection> {
        if !self.constraint.can_replace_without_recovery(requested) {
            return self.reject(OperatorAuthorityRejection::WouldWeakenActiveConstraint);
        }
        self.invalidate_recovery_state();
        self.constraint = requested;
        self.last_applied_proposal = Some(proposal_id);
        Ok(OperatorDecision::Applied(requested))
    }

    /// Register the exact proposal the local trusted host is willing to let
    /// operators review. This is an internal state-machine primitive: public
    /// recovery issuance must be owned by the embodiment/control-plane that also
    /// owns the live evidence source.
    pub(crate) fn issue_recovery_proposal(
        &mut self,
        proposal: RecoveryProposalV1,
        now_step: u64,
    ) -> Result<RecoveryProposalV1, OperatorAuthorityRejection> {
        if let Err(error) = proposal.validate(now_step, self.constraint) {
            return self.reject(OperatorAuthorityRejection::RecoveryProposal(error));
        }
        if let Some(existing) = self.issued_recovery.get(&proposal.proposal_id()) {
            if *existing != proposal {
                return self.reject(OperatorAuthorityRejection::ConflictingProposal);
            }
            return Ok(*existing);
        }
        self.issued_recovery.insert(proposal.proposal_id(), proposal);
        Ok(proposal)
    }

    /// Ordinary command ingestion. The legacy hazard boolean is retained for
    /// source compatibility only and has no ability to widen authority.
    pub fn ingest(
        &mut self,
        envelope: OperatorCommandEnvelope,
        now_step: u64,
        _legacy_physical_hazard_clear: bool,
    ) -> Result<OperatorDecision, OperatorAuthorityRejection> {
        let envelope = match self.policy.validate_metadata(envelope, now_step) {
            Ok(value) => value,
            Err(error) => return self.reject(error.into()),
        };
        self.accept_sequence(envelope)?;
        self.expire_pending(now_step);
        let decision = match envelope.command {
            OperatorCommand::EmergencyStop => self.apply_non_recovery_constraint(OperatorConstraint::EmergencyStop, envelope.proposal_id)?,
            OperatorCommand::HoldPosition => self.apply_non_recovery_constraint(OperatorConstraint::HoldPosition, envelope.proposal_id)?,
            OperatorCommand::ReturnHome => self.apply_non_recovery_constraint(OperatorConstraint::ReturnHome, envelope.proposal_id)?,
            OperatorCommand::SetMission(intent) => self.apply_non_recovery_constraint(OperatorConstraint::Mission(intent), envelope.proposal_id)?,
            OperatorCommand::EnterMaintenance => self.apply_non_recovery_constraint(OperatorConstraint::MaintenanceLock, envelope.proposal_id)?,
            OperatorCommand::ResumeNominal => return self.reject(OperatorAuthorityRejection::RecoveryProposalRequired),
        };
        self.accepted_commands = self.accepted_commands.saturating_add(1);
        Ok(decision)
    }

    /// Internal quorum transition. Public recovery admission must first be
    /// qualified by the authoritative embodiment/control-plane owner.
    pub(crate) fn approve_recovery(
        &mut self,
        approval: RecoveryApprovalEnvelopeV1,
        now_step: u64,
    ) -> Result<OperatorDecision, OperatorAuthorityRejection> {
        let metadata = match self.policy.validate_metadata(approval.as_command_envelope(), now_step) {
            Ok(value) => value,
            Err(error) => return self.reject(error.into()),
        };
        self.expire_pending(now_step);

        if let Err(error) = approval.validate_proposal_time() {
            return self.reject(OperatorAuthorityRejection::RecoveryProposal(error));
        }
        if let Err(error) = approval.proposal.validate(now_step, self.constraint) {
            return self.reject(OperatorAuthorityRejection::RecoveryProposal(error));
        }
        let proposal_id = approval.proposal.proposal_id();
        if self.issued_recovery.get(&proposal_id) != Some(&approval.proposal) {
            return self.reject(OperatorAuthorityRejection::RecoveryProposalNotIssued);
        }
        // Consume replay sequence only after the proposal itself is known to be
        // temporally valid, currently issued, and exactly bound to this state.
        self.accept_sequence(metadata)?;

        let pending = self.pending_resume.entry(proposal_id).or_insert_with(|| PendingRecoveryApproval {
            proposal: approval.proposal,
            operators: BTreeSet::new(),
        });
        if pending.proposal != approval.proposal {
            return self.reject(OperatorAuthorityRejection::ConflictingProposal);
        }
        pending.operators.insert(approval.operator);
        let approvals = pending.operators.len();
        let decision = if approvals >= self.policy.recovery_quorum.max(2) {
            self.constraint = approval.proposal.target_constraint();
            self.pending_resume.remove(&proposal_id);
            self.issued_recovery.remove(&proposal_id);
            self.last_applied_proposal = Some(proposal_id);
            OperatorDecision::Cleared
        } else {
            OperatorDecision::PendingQuorum { approvals, required: self.policy.recovery_quorum.max(2) }
        };
        self.accepted_commands = self.accepted_commands.saturating_add(1);
        Ok(decision)
    }

    pub(crate) fn reset_runtime(&mut self) {
        self.invalidate_recovery_state();
        self.constraint = OperatorConstraint::None;
        self.last_applied_proposal = None;
    }
}

impl Default for OperatorAuthority {
    fn default() -> Self { Self::new(OperatorTrustPolicy::default()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator_protocol::{AuthenticationLevel, OperatorRole};
    use recovery_authority::{RecoveryApprovalEnvelopeV1, RecoveryDigest};

    fn command(operator: u64, sequence: u64, proposal_id: u64, command: OperatorCommand) -> OperatorCommandEnvelope {
        OperatorCommandEnvelope {
            operator: OperatorId(operator), role: OperatorRole::SafetyOfficer,
            authentication: AuthenticationLevel::HardwareBacked, epoch: 1, sequence,
            proposal_id, issued_step: 10, expires_step: 100, command,
        }
    }
    fn proposal(id: u64, active: OperatorConstraint) -> RecoveryProposalV1 {
        RecoveryProposalV1::new(id, active, RecoveryDigest([1; 32]), RecoveryDigest([2; 32]), RecoveryDigest([3; 32]), 1, 1, 10, 100)
    }
    fn approval(operator: u64, sequence: u64, proposal: RecoveryProposalV1) -> RecoveryApprovalEnvelopeV1 {
        RecoveryApprovalEnvelopeV1 {
            operator: OperatorId(operator), role: OperatorRole::SafetyOfficer,
            authentication: AuthenticationLevel::HardwareBacked, epoch: 1, sequence,
            approval_issued_step: 20, proposal,
        }
    }

    #[test]
    fn unissued_proposal_cannot_start_quorum_or_consume_sequence() {
        let mut authority = OperatorAuthority::default();
        authority.ingest(command(1, 1, 1, OperatorCommand::EmergencyStop), 20, true).unwrap();
        let p = proposal(9, OperatorConstraint::EmergencyStop);
        let candidate = approval(1, 2, p);
        assert_eq!(
            authority.approve_recovery(candidate, 21),
            Err(OperatorAuthorityRejection::RecoveryProposalNotIssued)
        );
        authority.issue_recovery_proposal(p, 21).unwrap();
        assert!(matches!(
            authority.approve_recovery(candidate, 21).unwrap(),
            OperatorDecision::PendingQuorum { approvals: 1, required: 2 }
        ));
    }

    #[test]
    fn issued_exact_proposal_can_reach_quorum() {
        let mut authority = OperatorAuthority::default();
        authority.ingest(command(1, 1, 1, OperatorCommand::HoldPosition), 20, true).unwrap();
        let p = proposal(9, OperatorConstraint::HoldPosition);
        authority.issue_recovery_proposal(p, 20).unwrap();
        assert!(matches!(authority.approve_recovery(approval(1, 2, p), 21).unwrap(), OperatorDecision::PendingQuorum { approvals: 1, required: 2 }));
        assert_eq!(authority.approve_recovery(approval(2, 1, p), 22), Ok(OperatorDecision::Cleared));
        assert_eq!(authority.constraint(), OperatorConstraint::None);
    }

    #[test]
    fn approval_predating_proposal_is_rejected_without_consuming_sequence() {
        let mut authority = OperatorAuthority::default();
        authority.ingest(command(1, 1, 1, OperatorCommand::HoldPosition), 20, true).unwrap();
        let p = proposal(9, OperatorConstraint::HoldPosition);
        authority.issue_recovery_proposal(p, 20).unwrap();
        let mut bad = approval(1, 2, p);
        bad.approval_issued_step = 9;
        assert_eq!(
            authority.approve_recovery(bad, 21),
            Err(OperatorAuthorityRejection::RecoveryProposal(
                RecoveryProposalRejection::ApprovalPredatesProposal
            ))
        );
        assert!(matches!(
            authority.approve_recovery(approval(1, 2, p), 21).unwrap(),
            OperatorDecision::PendingQuorum { approvals: 1, required: 2 }
        ));
    }

    #[test]
    fn changed_evidence_same_id_cannot_join_issued_proposal() {
        let mut authority = OperatorAuthority::default();
        authority.ingest(command(1, 1, 1, OperatorCommand::HoldPosition), 20, true).unwrap();
        let p = proposal(9, OperatorConstraint::HoldPosition);
        authority.issue_recovery_proposal(p, 20).unwrap();
        authority.approve_recovery(approval(1, 2, p), 21).unwrap();
        let changed = RecoveryProposalV1::new(9, OperatorConstraint::HoldPosition, RecoveryDigest([1; 32]), RecoveryDigest([8; 32]), RecoveryDigest([3; 32]), 1, 1, 10, 100);
        assert_eq!(authority.approve_recovery(approval(2, 1, changed), 22), Err(OperatorAuthorityRejection::RecoveryProposalNotIssued));
        assert_eq!(authority.constraint(), OperatorConstraint::HoldPosition);
    }

    #[test]
    fn new_restriction_invalidates_issued_and_pending_recovery() {
        let mut authority = OperatorAuthority::default();
        authority.ingest(command(1, 1, 1, OperatorCommand::HoldPosition), 20, true).unwrap();
        let p = proposal(9, OperatorConstraint::HoldPosition);
        authority.issue_recovery_proposal(p, 20).unwrap();
        authority.approve_recovery(approval(1, 2, p), 21).unwrap();
        authority.ingest(command(2, 1, 2, OperatorCommand::EmergencyStop), 22, true).unwrap();
        assert_eq!(authority.pending_approvals(9), 0);
        assert_eq!(authority.issued_recovery_proposal(9), None);
        assert_eq!(authority.constraint(), OperatorConstraint::EmergencyStop);
    }

    #[test]
    fn live_recovery_state_is_not_deserialized() {
        let mut authority = OperatorAuthority::default();
        authority.ingest(command(1, 1, 1, OperatorCommand::HoldPosition), 20, true).unwrap();
        let p = proposal(9, OperatorConstraint::HoldPosition);
        authority.issue_recovery_proposal(p, 20).unwrap();
        authority.approve_recovery(approval(1, 2, p), 21).unwrap();
        assert_eq!(authority.pending_approvals(9), 1);
        assert_eq!(authority.issued_recovery_proposal(9), Some(p));

        let encoded = serde_json::to_vec(&authority).expect("authority should serialize");
        let restored: OperatorAuthority = serde_json::from_slice(&encoded).expect("authority should deserialize");
        assert_eq!(restored.constraint(), OperatorConstraint::HoldPosition);
        assert_eq!(restored.pending_approvals(9), 0);
        assert_eq!(restored.issued_recovery_proposal(9), None);
    }

    #[test]
    fn legacy_resume_cannot_clear_authority() {
        let mut authority = OperatorAuthority::default();
        authority.ingest(command(1, 1, 1, OperatorCommand::EmergencyStop), 20, true).unwrap();
        assert_eq!(authority.ingest(command(1, 2, 7, OperatorCommand::ResumeNominal), 21, true), Err(OperatorAuthorityRejection::RecoveryProposalRequired));
    }

    #[test]
    fn ordinary_commands_remain_monotone() {
        let mut authority = OperatorAuthority::default();
        authority.ingest(command(1, 1, 1, OperatorCommand::EmergencyStop), 20, true).unwrap();
        assert_eq!(authority.ingest(command(1, 2, 2, OperatorCommand::ReturnHome), 21, true), Err(OperatorAuthorityRejection::WouldWeakenActiveConstraint));
    }
}
