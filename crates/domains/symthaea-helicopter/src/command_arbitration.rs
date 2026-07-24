// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic arbitration across guidance, learned, operator, and safety commands.
//!
//! Multiple controllers must not write the actuator vector by last-writer-wins.
//! This arbiter applies fixed source precedence, per-channel authority masks,
//! leases, future-skew checks, motor-cut protection, and explicit conflict
//! evidence. It does not prove upstream command correctness.

use serde::{Deserialize, Serialize};

use crate::types::HelicopterCommand;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CommandSource {
    LearnedResidual,
    Guidance,
    Operator,
    EnvelopeProtection,
    EmergencyLanding,
    HardwareWatchdog,
}

impl CommandSource {
    pub const fn priority(self) -> u8 {
        match self {
            Self::LearnedResidual => 10,
            Self::Guidance => 20,
            Self::Operator => 30,
            Self::EnvelopeProtection => 70,
            Self::EmergencyLanding => 80,
            Self::HardwareWatchdog => 100,
        }
    }

    pub const fn may_disarm(self) -> bool {
        matches!(self, Self::EmergencyLanding | Self::HardwareWatchdog)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CommandProposal {
    pub source: CommandSource,
    pub issued_at_s: f64,
    pub expires_at_s: f64,
    /// collective, cyclic_lon, cyclic_lat, pedal, thrust, tail_rotor
    pub authority_mask: [bool; 6],
    pub command: HelicopterCommand,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CommandArbiterConfig {
    pub maximum_future_skew_s: f64,
    pub conflict_tolerance: f64,
    pub protected_motor_floor: f64,
}

impl Default for CommandArbiterConfig {
    fn default() -> Self {
        Self {
            maximum_future_skew_s: 0.010,
            conflict_tolerance: 1.0e-4,
            protected_motor_floor: 0.05,
        }
    }
}

impl CommandArbiterConfig {
    fn validate(&self) -> Result<(), CommandArbitrationError> {
        if !self.maximum_future_skew_s.is_finite()
            || self.maximum_future_skew_s < 0.0
            || !self.conflict_tolerance.is_finite()
            || self.conflict_tolerance < 0.0
            || !self.protected_motor_floor.is_finite()
            || !(0.0..=1.0).contains(&self.protected_motor_floor)
        {
            return Err(CommandArbitrationError::InvalidConfiguration);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandArbitrationError {
    InvalidConfiguration,
    InvalidTime,
    InvalidProposal,
    NonFiniteCommand,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProposalRejectionReason {
    Expired,
    FutureDated,
    InvalidLease,
    UnauthorizedMotorCut,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RejectedCommandProposal {
    pub source: CommandSource,
    pub reason: ProposalRejectionReason,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CommandArbitrationResult {
    pub command: HelicopterCommand,
    pub selected_sources: [Option<CommandSource>; 6],
    pub rejected: Vec<RejectedCommandProposal>,
    pub conflict_channels: [bool; 6],
    pub disarmed_by: Option<CommandSource>,
}

#[derive(Debug, Clone)]
pub struct HelicopterCommandArbiter {
    config: CommandArbiterConfig,
}

impl Default for HelicopterCommandArbiter {
    fn default() -> Self {
        Self::new(CommandArbiterConfig::default())
            .expect("default command-arbiter configuration must remain valid")
    }
}

impl HelicopterCommandArbiter {
    pub fn new(config: CommandArbiterConfig) -> Result<Self, CommandArbitrationError> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn arbitrate(
        &self,
        now_s: f64,
        fallback: HelicopterCommand,
        proposals: &[CommandProposal],
    ) -> Result<CommandArbitrationResult, CommandArbitrationError> {
        self.config.validate()?;
        if !now_s.is_finite() || now_s < 0.0 {
            return Err(CommandArbitrationError::InvalidTime);
        }
        let fallback_values = command_values(fallback);
        if !fallback_values.iter().all(|value| value.is_finite()) {
            return Err(CommandArbitrationError::NonFiniteCommand);
        }

        let mut values = fallback_values;
        let mut selected_sources = [None; 6];
        let mut selected_priorities = [0u8; 6];
        let mut conflict_channels = [false; 6];
        let mut rejected = Vec::new();
        let mut disarmed_by: Option<CommandSource> = None;

        for proposal in proposals {
            let proposal_values = command_values(proposal.command);
            if !proposal_values.iter().all(|value| value.is_finite()) {
                return Err(CommandArbitrationError::NonFiniteCommand);
            }
            if !proposal.issued_at_s.is_finite()
                || !proposal.expires_at_s.is_finite()
                || proposal.issued_at_s < 0.0
                || proposal.expires_at_s <= proposal.issued_at_s
            {
                rejected.push(RejectedCommandProposal {
                    source: proposal.source,
                    reason: ProposalRejectionReason::InvalidLease,
                });
                continue;
            }
            if proposal.expires_at_s <= now_s {
                rejected.push(RejectedCommandProposal {
                    source: proposal.source,
                    reason: ProposalRejectionReason::Expired,
                });
                continue;
            }
            if proposal.issued_at_s > now_s + self.config.maximum_future_skew_s {
                rejected.push(RejectedCommandProposal {
                    source: proposal.source,
                    reason: ProposalRejectionReason::FutureDated,
                });
                continue;
            }

            let unauthorized_motor_cut = !proposal.source.may_disarm()
                && ((proposal.authority_mask[4]
                    && proposal_values[4] < self.config.protected_motor_floor)
                    || (proposal.authority_mask[5]
                        && proposal_values[5] < self.config.protected_motor_floor));
            if unauthorized_motor_cut {
                rejected.push(RejectedCommandProposal {
                    source: proposal.source,
                    reason: ProposalRejectionReason::UnauthorizedMotorCut,
                });
                continue;
            }

            let priority = proposal.source.priority();
            for channel in 0..6 {
                if !proposal.authority_mask[channel] {
                    continue;
                }
                if priority > selected_priorities[channel] {
                    values[channel] = proposal_values[channel];
                    selected_sources[channel] = Some(proposal.source);
                    selected_priorities[channel] = priority;
                    conflict_channels[channel] = false;
                } else if priority == selected_priorities[channel]
                    && selected_sources[channel].is_some()
                    && (values[channel] - proposal_values[channel]).abs()
                        > self.config.conflict_tolerance
                {
                    values[channel] = fallback_values[channel];
                    selected_sources[channel] = None;
                    conflict_channels[channel] = true;
                }
            }

            if proposal.source.may_disarm()
                && proposal.authority_mask[4]
                && proposal.authority_mask[5]
                && proposal_values[4] <= self.config.protected_motor_floor
                && proposal_values[5] <= self.config.protected_motor_floor
            {
                disarmed_by = match disarmed_by {
                    Some(existing) if existing.priority() >= priority => Some(existing),
                    _ => Some(proposal.source),
                };
            }
        }

        Ok(CommandArbitrationResult {
            command: values_to_command(values).clamped(),
            selected_sources,
            rejected,
            conflict_channels,
            disarmed_by,
        })
    }
}

fn command_values(command: HelicopterCommand) -> [f64; 6] {
    command.to_ctrl()
}

fn values_to_command(values: [f64; 6]) -> HelicopterCommand {
    HelicopterCommand {
        collective: values[0] as f32,
        cyclic_lon: values[1] as f32,
        cyclic_lat: values[2] as f32,
        pedal: values[3] as f32,
        thrust: values[4] as f32,
        tail_rotor: values[5] as f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proposal(source: CommandSource, command: HelicopterCommand) -> CommandProposal {
        CommandProposal {
            source,
            issued_at_s: 1.0,
            expires_at_s: 2.0,
            authority_mask: [true; 6],
            command,
        }
    }

    #[test]
    fn emergency_command_overrides_guidance() {
        let arbiter = HelicopterCommandArbiter::default();
        let guidance = proposal(
            CommandSource::Guidance,
            HelicopterCommand {
                cyclic_lat: 0.5,
                ..HelicopterCommand::hover()
            },
        );
        let emergency = proposal(
            CommandSource::EmergencyLanding,
            HelicopterCommand {
                cyclic_lat: -0.2,
                ..HelicopterCommand::hover()
            },
        );
        let result = arbiter
            .arbitrate(1.5, HelicopterCommand::hover(), &[guidance, emergency])
            .unwrap();
        assert!((result.command.cyclic_lat + 0.2).abs() < 1.0e-6);
        assert_eq!(
            result.selected_sources[2],
            Some(CommandSource::EmergencyLanding)
        );
    }

    #[test]
    fn stale_proposal_is_rejected() {
        let arbiter = HelicopterCommandArbiter::default();
        let mut stale = proposal(CommandSource::Operator, HelicopterCommand::hover());
        stale.expires_at_s = 1.1;
        let result = arbiter
            .arbitrate(1.5, HelicopterCommand::hover(), &[stale])
            .unwrap();
        assert_eq!(result.rejected[0].reason, ProposalRejectionReason::Expired);
    }

    #[test]
    fn learned_controller_cannot_cut_both_rotors() {
        let arbiter = HelicopterCommandArbiter::default();
        let cut = proposal(CommandSource::LearnedResidual, HelicopterCommand::zero());
        let fallback = HelicopterCommand::hover();
        let result = arbiter.arbitrate(1.5, fallback, &[cut]).unwrap();
        assert_eq!(result.command.thrust, fallback.thrust);
        assert_eq!(
            result.rejected[0].reason,
            ProposalRejectionReason::UnauthorizedMotorCut
        );
    }

    #[test]
    fn watchdog_can_disarm() {
        let arbiter = HelicopterCommandArbiter::default();
        let cut = proposal(CommandSource::HardwareWatchdog, HelicopterCommand::zero());
        let result = arbiter
            .arbitrate(1.5, HelicopterCommand::hover(), &[cut])
            .unwrap();
        assert_eq!(result.command.thrust, 0.0);
        assert_eq!(result.disarmed_by, Some(CommandSource::HardwareWatchdog));
    }

    #[test]
    fn equal_priority_disagreement_falls_back_per_channel() {
        let arbiter = HelicopterCommandArbiter::default();
        let first = proposal(
            CommandSource::Operator,
            HelicopterCommand {
                pedal: 0.4,
                ..HelicopterCommand::hover()
            },
        );
        let second = proposal(
            CommandSource::Operator,
            HelicopterCommand {
                pedal: -0.4,
                ..HelicopterCommand::hover()
            },
        );
        let fallback = HelicopterCommand::hover();
        let result = arbiter.arbitrate(1.5, fallback, &[first, second]).unwrap();
        assert_eq!(result.command.pedal, fallback.pedal);
        assert!(result.conflict_channels[3]);
    }
}
