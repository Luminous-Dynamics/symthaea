// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Replay-resistant operator authority and safety-monotonic command constraints.

use crate::embodiment::MotorSafetyLevel;
use crate::mission::SubterraneanMissionIntent;
use crate::operator_protocol::{
    OperatorCommand, OperatorCommandEnvelope, OperatorCommandRejection, OperatorId,
    OperatorTrustPolicy,
};
use crate::types::{SubterraneanCommand, SubterraneanState};
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
            Self::EmergencyStop | Self::HoldPosition | Self::MaintenanceLock => {
                Some(SubterraneanMissionIntent::HoldPosition)
            }
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
        if other.restrictiveness_rank() > self.restrictiveness_rank() {
            other
        } else {
            self
        }
    }

    /// Whether `requested` may replace this constraint without going through the
    /// dedicated recovery protocol.
    ///
    /// Ordinary operator commands are safety-monotonic: they may preserve or
    /// increase restriction, but never widen authority. Mission changes are only
    /// allowed while the current operator state is `None` or another mission.
    /// `None` is recovery-only once any explicit operator state has been applied.
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

    /// Restrict the learned nominal command before physical recovery planning.
    /// Hazard recovery remains downstream and may add cooling, dewatering,
    /// sealing, support or withdrawal required to preserve the platform.
    pub fn constrain_nominal(
        self,
        mut command: SubterraneanCommand,
        state: &SubterraneanState,
    ) -> SubterraneanCommand {
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
                command.set_thermal_pump(if state.cutter_temp_c() >= 85.0 {
                    0.6
                } else {
                    0.0
                });
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
}

impl From<OperatorCommandRejection> for OperatorAuthorityRejection {
    fn from(value: OperatorCommandRejection) -> Self {
        Self::Protocol(value)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PendingRecoveryApproval {
    expires_step: u64,
    operators: BTreeSet<OperatorId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorAuthority {
    policy: OperatorTrustPolicy,
    last_sequence: BTreeMap<OperatorId, (u64, u64)>,
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
            pending_resume: BTreeMap::new(),
            constraint: OperatorConstraint::None,
            last_applied_proposal: None,
            accepted_commands: 0,
            rejected_commands: 0,
        }
    }

    pub fn constraint(&self) -> OperatorConstraint {
        self.constraint
    }

    pub fn validate(&self) -> bool {
        self.policy.maximum_command_age_steps > 0
            && self.policy.recovery_quorum >= 2
            && self
                .last_sequence
                .keys()
                .all(|operator| operator.is_valid())
            && self.pending_resume.values().all(|pending| {
                !pending.operators.is_empty()
                    && pending.operators.iter().all(|operator| operator.is_valid())
            })
    }

    pub fn accepted_commands(&self) -> u64 {
        self.accepted_commands
    }

    pub fn rejected_commands(&self) -> u64 {
        self.rejected_commands
    }

    pub fn pending_approvals(&self, proposal_id: u64) -> usize {
        self.pending_resume
            .get(&proposal_id)
            .map_or(0, |pending| pending.operators.len())
    }

    pub fn last_applied_proposal(&self) -> Option<u64> {
        self.last_applied_proposal
    }

    fn reject<T>(
        &mut self,
        rejection: OperatorAuthorityRejection,
    ) -> Result<T, OperatorAuthorityRejection> {
        self.rejected_commands = self.rejected_commands.saturating_add(1);
        Err(rejection)
    }

    fn accept_sequence(
        &mut self,
        envelope: OperatorCommandEnvelope,
    ) -> Result<(), OperatorAuthorityRejection> {
        if let Some((epoch, sequence)) = self.last_sequence.get(&envelope.operator).copied() {
            if envelope.epoch < epoch {
                return self.reject(OperatorAuthorityRejection::StaleEpoch);
            }
            if envelope.epoch == epoch && envelope.sequence <= sequence {
                return self.reject(OperatorAuthorityRejection::Replay);
            }
        }
        self.last_sequence
            .insert(envelope.operator, (envelope.epoch, envelope.sequence));
        Ok(())
    }

    fn expire_pending(&mut self, now_step: u64) {
        self.pending_resume
            .retain(|_, pending| now_step <= pending.expires_step);
    }

    fn apply_non_recovery_constraint(
        &mut self,
        requested: OperatorConstraint,
        proposal_id: u64,
    ) -> Result<OperatorDecision, OperatorAuthorityRejection> {
        if !self.constraint.can_replace_without_recovery(requested) {
            return self.reject(OperatorAuthorityRejection::WouldWeakenActiveConstraint);
        }

        // A fresh operator constraint invalidates approvals collected to clear an
        // earlier state. This includes an idempotent repeated stop: a newly
        // asserted stop must not inherit an older recovery quorum.
        self.pending_resume.clear();
        self.constraint = requested;
        self.last_applied_proposal = Some(proposal_id);
        Ok(OperatorDecision::Applied(requested))
    }

    pub fn ingest(
        &mut self,
        envelope: OperatorCommandEnvelope,
        now_step: u64,
        physical_hazard_clear: bool,
    ) -> Result<OperatorDecision, OperatorAuthorityRejection> {
        let envelope = match self.policy.validate_metadata(envelope, now_step) {
            Ok(value) => value,
            Err(error) => return self.reject(error.into()),
        };
        self.accept_sequence(envelope)?;
        self.expire_pending(now_step);

        let decision = match envelope.command {
            OperatorCommand::EmergencyStop => self.apply_non_recovery_constraint(
                OperatorConstraint::EmergencyStop,
                envelope.proposal_id,
            )?,
            OperatorCommand::HoldPosition => self.apply_non_recovery_constraint(
                OperatorConstraint::HoldPosition,
                envelope.proposal_id,
            )?,
            OperatorCommand::ReturnHome => self.apply_non_recovery_constraint(
                OperatorConstraint::ReturnHome,
                envelope.proposal_id,
            )?,
            OperatorCommand::SetMission(intent) => self.apply_non_recovery_constraint(
                OperatorConstraint::Mission(intent),
                envelope.proposal_id,
            )?,
            OperatorCommand::EnterMaintenance => self.apply_non_recovery_constraint(
                OperatorConstraint::MaintenanceLock,
                envelope.proposal_id,
            )?,
            OperatorCommand::ResumeNominal => {
                if !physical_hazard_clear {
                    return self.reject(OperatorAuthorityRejection::PhysicalHazardActive);
                }
                let pending = self
                    .pending_resume
                    .entry(envelope.proposal_id)
                    .or_insert_with(|| PendingRecoveryApproval {
                        expires_step: envelope.expires_step,
                        operators: BTreeSet::new(),
                    });
                if pending.expires_step != envelope.expires_step {
                    return self.reject(OperatorAuthorityRejection::ConflictingProposal);
                }
                pending.operators.insert(envelope.operator);
                let approvals = pending.operators.len();
                if approvals >= self.policy.recovery_quorum.max(2) {
                    self.constraint = OperatorConstraint::None;
                    self.pending_resume.remove(&envelope.proposal_id);
                    self.last_applied_proposal = Some(envelope.proposal_id);
                    OperatorDecision::Cleared
                } else {
                    OperatorDecision::PendingQuorum {
                        approvals,
                        required: self.policy.recovery_quorum.max(2),
                    }
                }
            }
        };
        self.accepted_commands = self.accepted_commands.saturating_add(1);
        Ok(decision)
    }

    pub fn reset_runtime(&mut self) {
        self.pending_resume.clear();
        self.constraint = OperatorConstraint::None;
        self.last_applied_proposal = None;
    }
}

impl Default for OperatorAuthority {
    fn default() -> Self {
        Self::new(OperatorTrustPolicy::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator_protocol::{AuthenticationLevel, OperatorRole};

    fn command(
        operator: u64,
        sequence: u64,
        proposal_id: u64,
        command: OperatorCommand,
    ) -> OperatorCommandEnvelope {
        OperatorCommandEnvelope {
            operator: OperatorId(operator),
            role: OperatorRole::SafetyOfficer,
            authentication: AuthenticationLevel::HardwareBacked,
            epoch: 1,
            sequence,
            proposal_id,
            issued_step: 10,
            expires_step: 100,
            command,
        }
    }

    #[test]
    fn replayed_operator_sequence_is_rejected() {
        let mut authority = OperatorAuthority::default();
        let value = command(1, 1, 1, OperatorCommand::HoldPosition);
        assert!(authority.ingest(value, 20, true).is_ok());
        assert_eq!(
            authority.ingest(value, 21, true),
            Err(OperatorAuthorityRejection::Replay)
        );
    }

    #[test]
    fn one_operator_cannot_resume_after_stop() {
        let mut authority = OperatorAuthority::default();
        authority
            .ingest(command(1, 1, 1, OperatorCommand::EmergencyStop), 20, true)
            .expect("stop is valid");
        let decision = authority
            .ingest(command(1, 2, 7, OperatorCommand::ResumeNominal), 21, true)
            .expect("first approval is valid");
        assert_eq!(
            decision,
            OperatorDecision::PendingQuorum {
                approvals: 1,
                required: 2
            }
        );
        assert_eq!(authority.constraint(), OperatorConstraint::EmergencyStop);
    }

    #[test]
    fn two_independent_approvals_clear_restrictive_constraint() {
        let mut authority = OperatorAuthority::default();
        authority
            .ingest(command(1, 1, 1, OperatorCommand::HoldPosition), 20, true)
            .expect("hold is valid");
        authority
            .ingest(command(1, 2, 9, OperatorCommand::ResumeNominal), 21, true)
            .expect("first approval is valid");
        let decision = authority
            .ingest(command(2, 1, 9, OperatorCommand::ResumeNominal), 22, true)
            .expect("second approval is valid");
        assert_eq!(decision, OperatorDecision::Cleared);
        assert_eq!(authority.constraint(), OperatorConstraint::None);
    }

    #[test]
    fn physical_hazard_blocks_resume_even_with_valid_approval() {
        let mut authority = OperatorAuthority::default();
        let result = authority.ingest(command(1, 1, 9, OperatorCommand::ResumeNominal), 20, false);
        assert_eq!(
            result,
            Err(OperatorAuthorityRejection::PhysicalHazardActive)
        );
    }

    #[test]
    fn emergency_stop_cannot_be_weakened_by_ordinary_commands() {
        let mut authority = OperatorAuthority::default();
        authority
            .ingest(command(1, 1, 1, OperatorCommand::EmergencyStop), 20, true)
            .expect("stop is valid");

        let weaker = [
            OperatorCommand::HoldPosition,
            OperatorCommand::ReturnHome,
            OperatorCommand::SetMission(SubterraneanMissionIntent::Explore),
            OperatorCommand::EnterMaintenance,
        ];

        for (index, requested) in weaker.into_iter().enumerate() {
            let result = authority.ingest(
                command(1, index as u64 + 2, index as u64 + 2, requested),
                21 + index as u64,
                true,
            );
            assert_eq!(
                result,
                Err(OperatorAuthorityRejection::WouldWeakenActiveConstraint)
            );
            assert_eq!(authority.constraint(), OperatorConstraint::EmergencyStop);
        }
    }

    #[test]
    fn stronger_non_recovery_constraint_is_allowed() {
        let mut authority = OperatorAuthority::default();
        authority
            .ingest(command(1, 1, 1, OperatorCommand::ReturnHome), 20, true)
            .expect("return is valid");
        let decision = authority
            .ingest(command(1, 2, 2, OperatorCommand::EmergencyStop), 21, true)
            .expect("stronger stop must remain available");
        assert_eq!(
            decision,
            OperatorDecision::Applied(OperatorConstraint::EmergencyStop)
        );
        assert_eq!(authority.constraint(), OperatorConstraint::EmergencyStop);
    }

    #[test]
    fn mission_changes_remain_available_at_mission_level() {
        let mut authority = OperatorAuthority::default();
        authority
            .ingest(
                command(
                    1,
                    1,
                    1,
                    OperatorCommand::SetMission(SubterraneanMissionIntent::Explore),
                ),
                20,
                true,
            )
            .expect("initial mission is valid");
        let decision = authority
            .ingest(
                command(
                    1,
                    2,
                    2,
                    OperatorCommand::SetMission(SubterraneanMissionIntent::ProbeAhead),
                ),
                21,
                true,
            )
            .expect("mission-level change is non-widening");
        assert_eq!(
            decision,
            OperatorDecision::Applied(OperatorConstraint::Mission(
                SubterraneanMissionIntent::ProbeAhead
            ))
        );
    }

    #[test]
    fn new_restriction_invalidates_existing_resume_approvals() {
        let mut authority = OperatorAuthority::default();
        authority
            .ingest(command(1, 1, 1, OperatorCommand::HoldPosition), 20, true)
            .expect("hold is valid");
        let first = authority
            .ingest(command(1, 2, 9, OperatorCommand::ResumeNominal), 21, true)
            .expect("first recovery approval is valid");
        assert_eq!(
            first,
            OperatorDecision::PendingQuorum {
                approvals: 1,
                required: 2
            }
        );
        assert_eq!(authority.pending_approvals(9), 1);

        authority
            .ingest(command(2, 1, 2, OperatorCommand::EmergencyStop), 22, true)
            .expect("new emergency stop is valid");
        assert_eq!(authority.pending_approvals(9), 0);

        let stale_second = authority
            .ingest(command(2, 2, 9, OperatorCommand::ResumeNominal), 23, true)
            .expect("approval starts a fresh quorum after the new restriction");
        assert_eq!(
            stale_second,
            OperatorDecision::PendingQuorum {
                approvals: 1,
                required: 2
            }
        );
        assert_eq!(authority.constraint(), OperatorConstraint::EmergencyStop);
    }

    #[test]
    fn repeated_emergency_stop_invalidates_pending_recovery() {
        let mut authority = OperatorAuthority::default();
        authority
            .ingest(command(1, 1, 1, OperatorCommand::EmergencyStop), 20, true)
            .expect("initial stop is valid");
        authority
            .ingest(command(1, 2, 9, OperatorCommand::ResumeNominal), 21, true)
            .expect("first recovery approval is valid");
        assert_eq!(authority.pending_approvals(9), 1);

        authority
            .ingest(command(2, 1, 2, OperatorCommand::EmergencyStop), 22, true)
            .expect("reasserted stop is valid");
        assert_eq!(authority.pending_approvals(9), 0);
        assert_eq!(authority.constraint(), OperatorConstraint::EmergencyStop);
    }

    #[test]
    fn hold_removes_motion_but_preserves_needed_cooling() {
        let mut state = SubterraneanState::home();
        state.channels[crate::types::CUTTER_TEMP_C] = 100.0;
        let mut command = SubterraneanCommand::zero();
        command.set_cutter_head(1.0);
        command.set_left_track(1.0);
        command.set_right_track(1.0);
        let constrained = OperatorConstraint::HoldPosition.constrain_nominal(command, &state);
        assert_eq!(constrained.cutter_head(), 0.0);
        assert_eq!(constrained.left_track(), 0.0);
        assert!(constrained.thermal_pump() >= 0.6);
    }
}
