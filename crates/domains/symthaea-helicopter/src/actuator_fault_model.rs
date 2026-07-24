// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic actuator fault propagation for control-allocation campaigns.
//!
//! The allocator reports residual authority, but qualification also needs a
//! plant-side model that can make commanded and realized actuator positions
//! diverge. Faults here operate after command arbitration and before ordinary
//! servo/governor dynamics.

use serde::{Deserialize, Serialize};

use crate::control_allocation::ActuatorHealth;
use crate::types::HelicopterCommand;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActuatorChannel {
    Collective,
    CyclicLongitudinal,
    CyclicLateral,
    Pedal,
    MainRotorGovernor,
    TailRotorGovernor,
}

impl ActuatorChannel {
    pub const ALL: [Self; 6] = [
        Self::Collective,
        Self::CyclicLongitudinal,
        Self::CyclicLateral,
        Self::Pedal,
        Self::MainRotorGovernor,
        Self::TailRotorGovernor,
    ];

    const fn index(self) -> usize {
        match self {
            Self::Collective => 0,
            Self::CyclicLongitudinal => 1,
            Self::CyclicLateral => 2,
            Self::Pedal => 3,
            Self::MainRotorGovernor => 4,
            Self::TailRotorGovernor => 5,
        }
    }

    const fn is_unsigned(self) -> bool {
        matches!(self, Self::MainRotorGovernor | Self::TailRotorGovernor)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ActuatorFaultMode {
    Bias {
        offset: f64,
    },
    GainLoss {
        gain: f64,
    },
    Jammed {
        position: f64,
    },
    Deadband {
        width: f64,
    },
    Runaway {
        rate_per_s: f64,
        direction: f64,
    },
    Intermittent {
        period_s: f64,
        available_fraction: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScheduledActuatorFault {
    pub channel: ActuatorChannel,
    pub mode: ActuatorFaultMode,
    pub start_time_s: f64,
    pub end_time_s: Option<f64>,
}

impl ScheduledActuatorFault {
    fn validate(&self) -> Result<(), ActuatorFaultError> {
        if !self.start_time_s.is_finite() || self.start_time_s < 0.0 {
            return Err(ActuatorFaultError::InvalidConfiguration);
        }
        if self
            .end_time_s
            .is_some_and(|end| !end.is_finite() || end <= self.start_time_s)
        {
            return Err(ActuatorFaultError::InvalidConfiguration);
        }
        match self.mode {
            ActuatorFaultMode::Bias { offset } => {
                if !offset.is_finite() {
                    return Err(ActuatorFaultError::InvalidConfiguration);
                }
            }
            ActuatorFaultMode::GainLoss { gain } => {
                if !gain.is_finite() || !(0.0..=1.0).contains(&gain) {
                    return Err(ActuatorFaultError::InvalidConfiguration);
                }
            }
            ActuatorFaultMode::Jammed { position } => {
                if !position.is_finite() || !valid_channel_value(self.channel, position) {
                    return Err(ActuatorFaultError::InvalidConfiguration);
                }
            }
            ActuatorFaultMode::Deadband { width } => {
                if !width.is_finite() || !(0.0..=1.0).contains(&width) {
                    return Err(ActuatorFaultError::InvalidConfiguration);
                }
            }
            ActuatorFaultMode::Runaway {
                rate_per_s,
                direction,
            } => {
                if !rate_per_s.is_finite()
                    || rate_per_s <= 0.0
                    || !direction.is_finite()
                    || direction.abs() < 1.0e-12
                {
                    return Err(ActuatorFaultError::InvalidConfiguration);
                }
            }
            ActuatorFaultMode::Intermittent {
                period_s,
                available_fraction,
            } => {
                if !period_s.is_finite()
                    || period_s <= 0.0
                    || !available_fraction.is_finite()
                    || !(0.0..=1.0).contains(&available_fraction)
                {
                    return Err(ActuatorFaultError::InvalidConfiguration);
                }
            }
        }
        Ok(())
    }

    fn active_at(&self, time_s: f64) -> bool {
        time_s >= self.start_time_s && self.end_time_s.is_none_or(|end| time_s < end)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActuatorFaultError {
    InvalidConfiguration,
    NonFiniteCommand,
    NonFiniteTime,
    TimeWentBackwards,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ActuatorFaultEvidence {
    pub applied_cycles: u64,
    pub faulted_cycles: u64,
    pub active_fault_applications: u64,
    pub maximum_absolute_divergence: [f64; 6],
    pub fail_safe_requests: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActuatorFaultOutput {
    pub realized_command: HelicopterCommand,
    pub estimated_health: ActuatorHealth,
    pub active_faults: usize,
    pub divergence: [f64; 6],
    pub fail_safe_required: bool,
}

#[derive(Debug, Clone)]
pub struct ActuatorFaultModel {
    faults: Vec<ScheduledActuatorFault>,
    last_time_s: Option<f64>,
    last_realized: HelicopterCommand,
    evidence: ActuatorFaultEvidence,
}

impl ActuatorFaultModel {
    pub fn new(faults: Vec<ScheduledActuatorFault>) -> Result<Self, ActuatorFaultError> {
        for fault in &faults {
            fault.validate()?;
        }
        Ok(Self {
            faults,
            last_time_s: None,
            last_realized: HelicopterCommand::zero(),
            evidence: ActuatorFaultEvidence {
                applied_cycles: 0,
                faulted_cycles: 0,
                active_fault_applications: 0,
                maximum_absolute_divergence: [0.0; 6],
                fail_safe_requests: 0,
            },
        })
    }

    pub fn evidence(&self) -> ActuatorFaultEvidence {
        self.evidence
    }

    pub fn reset(&mut self) {
        self.last_time_s = None;
        self.last_realized = HelicopterCommand::zero();
        self.evidence = ActuatorFaultEvidence {
            applied_cycles: 0,
            faulted_cycles: 0,
            active_fault_applications: 0,
            maximum_absolute_divergence: [0.0; 6],
            fail_safe_requests: 0,
        };
    }

    pub fn apply(
        &mut self,
        requested: HelicopterCommand,
        time_s: f64,
    ) -> Result<ActuatorFaultOutput, ActuatorFaultError> {
        if !time_s.is_finite() || time_s < 0.0 {
            return Err(ActuatorFaultError::NonFiniteTime);
        }
        if self.last_time_s.is_some_and(|last| time_s < last) {
            return Err(ActuatorFaultError::TimeWentBackwards);
        }
        let requested_values = command_values(requested);
        if !requested_values.iter().all(|value| value.is_finite()) {
            return Err(ActuatorFaultError::NonFiniteCommand);
        }
        let dt_s = self.last_time_s.map(|last| time_s - last).unwrap_or(0.0);
        let last_values = command_values(self.last_realized);
        let mut realized = requested_values;
        let mut health = [1.0_f64; 6];
        let mut active_faults = 0usize;
        let mut fail_safe_required = false;

        for fault in &self.faults {
            if !fault.active_at(time_s) {
                continue;
            }
            active_faults += 1;
            self.evidence.active_fault_applications =
                self.evidence.active_fault_applications.saturating_add(1);
            let index = fault.channel.index();
            match fault.mode {
                ActuatorFaultMode::Bias { offset } => {
                    realized[index] += offset;
                    health[index] = health[index].min(0.75_f64);
                }
                ActuatorFaultMode::GainLoss { gain } => {
                    realized[index] *= gain;
                    health[index] = health[index].min(gain);
                }
                ActuatorFaultMode::Jammed { position } => {
                    realized[index] = position;
                    health[index] = 0.0;
                    fail_safe_required |= matches!(
                        fault.channel,
                        ActuatorChannel::Collective
                            | ActuatorChannel::MainRotorGovernor
                            | ActuatorChannel::TailRotorGovernor
                    );
                }
                ActuatorFaultMode::Deadband { width } => {
                    if realized[index].abs() < width {
                        realized[index] = 0.0;
                    }
                    health[index] = health[index].min((1.0 - width).max(0.0));
                }
                ActuatorFaultMode::Runaway {
                    rate_per_s,
                    direction,
                } => {
                    realized[index] =
                        last_values[index] + direction.signum() * rate_per_s * dt_s.max(0.0);
                    health[index] = 0.0;
                    fail_safe_required = true;
                }
                ActuatorFaultMode::Intermittent {
                    period_s,
                    available_fraction,
                } => {
                    let phase = ((time_s - fault.start_time_s) / period_s).rem_euclid(1.0);
                    if phase >= available_fraction {
                        realized[index] = last_values[index];
                    }
                    health[index] = health[index].min(available_fraction);
                }
            }
            realized[index] = clamp_channel(fault.channel, realized[index]);
        }

        let realized_command = values_to_command(realized).clamped();
        let realized_values = command_values(realized_command);
        let mut divergence = [0.0; 6];
        for index in 0..6 {
            divergence[index] = realized_values[index] - requested_values[index];
            self.evidence.maximum_absolute_divergence[index] =
                self.evidence.maximum_absolute_divergence[index].max(divergence[index].abs());
        }
        self.evidence.applied_cycles = self.evidence.applied_cycles.saturating_add(1);
        if active_faults > 0 {
            self.evidence.faulted_cycles = self.evidence.faulted_cycles.saturating_add(1);
        }
        if fail_safe_required {
            self.evidence.fail_safe_requests = self.evidence.fail_safe_requests.saturating_add(1);
        }
        self.last_time_s = Some(time_s);
        self.last_realized = realized_command;

        Ok(ActuatorFaultOutput {
            realized_command,
            estimated_health: ActuatorHealth {
                collective: health[0],
                cyclic_lon: health[1],
                cyclic_lat: health[2],
                pedal: health[3],
                main_rotor: health[4],
                tail_rotor: health[5],
            },
            active_faults,
            divergence,
            fail_safe_required,
        })
    }
}

fn valid_channel_value(channel: ActuatorChannel, value: f64) -> bool {
    if channel.is_unsigned() {
        (0.0..=1.0).contains(&value)
    } else {
        (-1.0..=1.0).contains(&value)
    }
}

fn clamp_channel(channel: ActuatorChannel, value: f64) -> f64 {
    if channel.is_unsigned() {
        value.clamp(0.0, 1.0)
    } else {
        value.clamp(-1.0, 1.0)
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

    #[test]
    fn jammed_collective_diverges_and_requests_fail_safe() {
        let mut model = ActuatorFaultModel::new(vec![ScheduledActuatorFault {
            channel: ActuatorChannel::Collective,
            mode: ActuatorFaultMode::Jammed { position: 0.1 },
            start_time_s: 1.0,
            end_time_s: None,
        }])
        .unwrap();
        let output = model.apply(HelicopterCommand::hover(), 1.0).unwrap();
        assert!((output.realized_command.collective - 0.1).abs() < 1.0e-6);
        assert_eq!(output.estimated_health.collective, 0.0);
        assert!(output.fail_safe_required);
    }

    #[test]
    fn gain_loss_produces_conservative_health() {
        let mut model = ActuatorFaultModel::new(vec![ScheduledActuatorFault {
            channel: ActuatorChannel::CyclicLateral,
            mode: ActuatorFaultMode::GainLoss { gain: 0.4 },
            start_time_s: 0.0,
            end_time_s: None,
        }])
        .unwrap();
        let command = HelicopterCommand {
            cyclic_lat: 0.5,
            ..HelicopterCommand::hover()
        };
        let output = model.apply(command, 0.0).unwrap();
        assert!((output.realized_command.cyclic_lat - 0.2).abs() < 1.0e-6);
        assert_eq!(output.estimated_health.cyclic_lat, 0.4);
    }

    #[test]
    fn runaway_uses_elapsed_time_and_saturates() {
        let mut model = ActuatorFaultModel::new(vec![ScheduledActuatorFault {
            channel: ActuatorChannel::Pedal,
            mode: ActuatorFaultMode::Runaway {
                rate_per_s: 2.0,
                direction: 1.0,
            },
            start_time_s: 0.0,
            end_time_s: None,
        }])
        .unwrap();
        model.apply(HelicopterCommand::zero(), 0.0).unwrap();
        let output = model.apply(HelicopterCommand::zero(), 1.0).unwrap();
        assert_eq!(output.realized_command.pedal, 1.0);
        assert!(output.fail_safe_required);
    }

    #[test]
    fn inactive_fault_passes_command_through() {
        let mut model = ActuatorFaultModel::new(vec![ScheduledActuatorFault {
            channel: ActuatorChannel::MainRotorGovernor,
            mode: ActuatorFaultMode::Jammed { position: 0.0 },
            start_time_s: 10.0,
            end_time_s: None,
        }])
        .unwrap();
        let command = HelicopterCommand::hover();
        let output = model.apply(command, 1.0).unwrap();
        assert_eq!(output.realized_command.thrust, command.thrust);
        assert_eq!(output.active_faults, 0);
    }

    #[test]
    fn backward_time_fails_closed() {
        let mut model = ActuatorFaultModel::new(Vec::new()).unwrap();
        model.apply(HelicopterCommand::hover(), 2.0).unwrap();
        assert_eq!(
            model.apply(HelicopterCommand::hover(), 1.0).unwrap_err(),
            ActuatorFaultError::TimeWentBackwards
        );
    }
}
