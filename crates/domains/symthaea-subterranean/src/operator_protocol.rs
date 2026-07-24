// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Operator command protocol and trust-boundary validation.
//!
//! This module does not verify cryptographic signatures. It accepts an
//! authentication level asserted by an upstream transport/security boundary,
//! validates freshness, role, sequence and command semantics, and refuses to
//! treat unverified metadata as operational authority.

use crate::mission::SubterraneanMissionIntent;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct OperatorId(pub u64);

impl OperatorId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    pub const fn is_valid(self) -> bool {
        self.0 != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OperatorRole {
    Observer,
    Operator,
    Supervisor,
    SafetyOfficer,
}

impl OperatorRole {
    pub const fn can_direct_mission(self) -> bool {
        matches!(
            self,
            Self::Operator | Self::Supervisor | Self::SafetyOfficer
        )
    }

    pub const fn can_approve_recovery(self) -> bool {
        matches!(self, Self::Supervisor | Self::SafetyOfficer)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AuthenticationLevel {
    Unverified,
    TransportAuthenticated,
    HardwareBacked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperatorCommand {
    EmergencyStop,
    HoldPosition,
    ReturnHome,
    SetMission(SubterraneanMissionIntent),
    EnterMaintenance,
    ResumeNominal,
}

impl OperatorCommand {
    pub const fn code(self) -> u16 {
        match self {
            Self::EmergencyStop => 1,
            Self::HoldPosition => 2,
            Self::ReturnHome => 3,
            Self::SetMission(intent) => 10 + intent.index() as u16,
            Self::EnterMaintenance => 30,
            Self::ResumeNominal => 31,
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::EmergencyStop => "emergency_stop",
            Self::HoldPosition => "hold_position",
            Self::ReturnHome => "return_home",
            Self::SetMission(_) => "set_mission",
            Self::EnterMaintenance => "enter_maintenance",
            Self::ResumeNominal => "resume_nominal",
        }
    }

    pub const fn is_restrictive(self) -> bool {
        matches!(
            self,
            Self::EmergencyStop | Self::HoldPosition | Self::ReturnHome | Self::EnterMaintenance
        )
    }

    pub const fn requires_quorum(self) -> bool {
        matches!(self, Self::ResumeNominal)
    }

    pub const fn minimum_authentication(self) -> AuthenticationLevel {
        match self {
            Self::EmergencyStop => AuthenticationLevel::TransportAuthenticated,
            Self::HoldPosition
            | Self::ReturnHome
            | Self::SetMission(_)
            | Self::EnterMaintenance => AuthenticationLevel::TransportAuthenticated,
            Self::ResumeNominal => AuthenticationLevel::HardwareBacked,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorCommandEnvelope {
    pub operator: OperatorId,
    pub role: OperatorRole,
    pub authentication: AuthenticationLevel,
    pub epoch: u64,
    pub sequence: u64,
    /// Shared identifier used when a command requires multiple independent
    /// approvals. Zero is invalid for quorum-gated commands.
    pub proposal_id: u64,
    pub issued_step: u64,
    pub expires_step: u64,
    pub command: OperatorCommand,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorTrustPolicy {
    pub maximum_command_age_steps: u64,
    pub maximum_future_skew_steps: u64,
    pub recovery_quorum: usize,
}

impl Default for OperatorTrustPolicy {
    fn default() -> Self {
        Self {
            maximum_command_age_steps: 2_000,
            maximum_future_skew_steps: 20,
            recovery_quorum: 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorCommandRejection {
    InvalidOperator,
    ObserverHasNoAuthority,
    InsufficientRole,
    InsufficientAuthentication,
    InvalidLifetime,
    NotYetValid,
    Expired,
    TooOld,
    InvalidProposal,
}

impl OperatorTrustPolicy {
    pub fn validate_metadata(
        self,
        envelope: OperatorCommandEnvelope,
        now_step: u64,
    ) -> Result<OperatorCommandEnvelope, OperatorCommandRejection> {
        if !envelope.operator.is_valid() {
            return Err(OperatorCommandRejection::InvalidOperator);
        }
        if envelope.role == OperatorRole::Observer {
            return Err(OperatorCommandRejection::ObserverHasNoAuthority);
        }
        if !envelope.role.can_direct_mission() {
            return Err(OperatorCommandRejection::InsufficientRole);
        }
        if envelope.command.requires_quorum() && !envelope.role.can_approve_recovery() {
            return Err(OperatorCommandRejection::InsufficientRole);
        }
        if envelope.authentication < envelope.command.minimum_authentication() {
            return Err(OperatorCommandRejection::InsufficientAuthentication);
        }
        if envelope.expires_step < envelope.issued_step {
            return Err(OperatorCommandRejection::InvalidLifetime);
        }
        if envelope.issued_step > now_step.saturating_add(self.maximum_future_skew_steps) {
            return Err(OperatorCommandRejection::NotYetValid);
        }
        if now_step > envelope.expires_step {
            return Err(OperatorCommandRejection::Expired);
        }
        if now_step.saturating_sub(envelope.issued_step) > self.maximum_command_age_steps {
            return Err(OperatorCommandRejection::TooOld);
        }
        if envelope.command.requires_quorum() && envelope.proposal_id == 0 {
            return Err(OperatorCommandRejection::InvalidProposal);
        }
        Ok(envelope)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope(command: OperatorCommand) -> OperatorCommandEnvelope {
        OperatorCommandEnvelope {
            operator: OperatorId(7),
            role: OperatorRole::Supervisor,
            authentication: AuthenticationLevel::HardwareBacked,
            epoch: 1,
            sequence: 1,
            proposal_id: 9,
            issued_step: 100,
            expires_step: 200,
            command,
        }
    }

    #[test]
    fn observer_cannot_acquire_actuation_authority() {
        let mut value = envelope(OperatorCommand::HoldPosition);
        value.role = OperatorRole::Observer;
        assert_eq!(
            OperatorTrustPolicy::default().validate_metadata(value, 120),
            Err(OperatorCommandRejection::ObserverHasNoAuthority)
        );
    }

    #[test]
    fn resume_requires_hardware_backed_authentication() {
        let mut value = envelope(OperatorCommand::ResumeNominal);
        value.authentication = AuthenticationLevel::TransportAuthenticated;
        assert_eq!(
            OperatorTrustPolicy::default().validate_metadata(value, 120),
            Err(OperatorCommandRejection::InsufficientAuthentication)
        );
    }

    #[test]
    fn stale_command_is_rejected_even_when_expiry_is_far_away() {
        let mut value = envelope(OperatorCommand::ReturnHome);
        value.issued_step = 1;
        value.expires_step = 10_000;
        let policy = OperatorTrustPolicy {
            maximum_command_age_steps: 100,
            ..Default::default()
        };
        assert_eq!(
            policy.validate_metadata(value, 500),
            Err(OperatorCommandRejection::TooOld)
        );
    }
}
