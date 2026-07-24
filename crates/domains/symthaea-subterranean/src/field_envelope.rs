// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Continuous power and thermal command envelope for partial-failure survival.
//!
//! Safety planning chooses *what* should happen. This layer constrains *how
//! much* demand the current power, thermal, coolant and component state can
//! physically sustain without turning a recoverable fault into a total loss.

use crate::maintenance::MaintenanceAssessment;
use crate::types::{SubterraneanCommand, SubterraneanState};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldEnvelopeMode {
    Nominal,
    Derated,
    CriticalPower,
    ThermalProtection,
    SurvivalHold,
}

impl FieldEnvelopeMode {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Nominal => "nominal",
            Self::Derated => "derated",
            Self::CriticalPower => "critical_power",
            Self::ThermalProtection => "thermal_protection",
            Self::SurvivalHold => "survival_hold",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FieldEnvelopePolicy {
    pub derate_battery_ratio: f64,
    pub critical_battery_ratio: f64,
    pub survival_battery_ratio: f64,
    pub thermal_derate_c: f64,
    pub thermal_stop_c: f64,
    pub minimum_coolant_health: f64,
}

impl Default for FieldEnvelopePolicy {
    fn default() -> Self {
        Self {
            derate_battery_ratio: 0.4,
            critical_battery_ratio: 0.2,
            survival_battery_ratio: 0.08,
            thermal_derate_c: 105.0,
            thermal_stop_c: 135.0,
            minimum_coolant_health: 0.2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FieldEnvelopeAssessment {
    pub mode: FieldEnvelopeMode,
    pub power_margin: f64,
    pub thermal_margin: f64,
    pub cutter_cap: f32,
    pub auger_cap: f32,
    pub track_cap: f32,
    pub recovery_cap: f32,
    pub cooling_floor: f32,
    pub mission_work_allowed: bool,
}

impl FieldEnvelopeAssessment {
    pub const fn nominal() -> Self {
        Self {
            mode: FieldEnvelopeMode::Nominal,
            power_margin: 1.0,
            thermal_margin: 1.0,
            cutter_cap: 1.0,
            auger_cap: 1.0,
            track_cap: 1.0,
            recovery_cap: 1.0,
            cooling_floor: 0.0,
            mission_work_allowed: true,
        }
    }

    pub fn constrain(self, mut command: SubterraneanCommand) -> SubterraneanCommand {
        command.set_cutter_head(
            command
                .cutter_head()
                .clamp(-self.cutter_cap, self.cutter_cap),
        );
        command.set_auger_feed(command.auger_feed().clamp(-self.auger_cap, self.auger_cap));
        command.set_left_track(command.left_track().clamp(-self.track_cap, self.track_cap));
        command.set_right_track(command.right_track().clamp(-self.track_cap, self.track_cap));
        command.recovery.dewatering_pump = command
            .recovery
            .dewatering_pump
            .clamp(0.0, self.recovery_cap);
        command.recovery.sealant_injector = command
            .recovery
            .sealant_injector
            .clamp(0.0, self.recovery_cap);
        command.recovery.relay_deployer = command
            .recovery
            .relay_deployer
            .clamp(0.0, self.recovery_cap);
        command.recovery.roof_support = command.recovery.roof_support.clamp(0.0, self.recovery_cap);
        command.set_thermal_pump(
            command
                .thermal_pump()
                .max(self.cooling_floor)
                .clamp(0.0, 1.0),
        );
        command.sanitize();
        command
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldEnvelopeSupervisor {
    policy: FieldEnvelopePolicy,
    last_assessment: FieldEnvelopeAssessment,
    transitions: u64,
}

impl FieldEnvelopeSupervisor {
    pub fn new(policy: FieldEnvelopePolicy) -> Self {
        Self {
            policy,
            last_assessment: FieldEnvelopeAssessment::nominal(),
            transitions: 0,
        }
    }

    pub fn validate(&self) -> bool {
        let values = [
            self.policy.derate_battery_ratio,
            self.policy.critical_battery_ratio,
            self.policy.survival_battery_ratio,
            self.policy.minimum_coolant_health,
        ];
        values
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
            && self.policy.survival_battery_ratio <= self.policy.critical_battery_ratio
            && self.policy.critical_battery_ratio <= self.policy.derate_battery_ratio
            && self.policy.thermal_derate_c.is_finite()
            && self.policy.thermal_stop_c.is_finite()
            && self.policy.thermal_derate_c < self.policy.thermal_stop_c
            && self.last_assessment.power_margin.is_finite()
            && self.last_assessment.thermal_margin.is_finite()
    }

    pub fn assess(
        &mut self,
        state: &SubterraneanState,
        coolant_health: f64,
        maintenance: MaintenanceAssessment,
    ) -> FieldEnvelopeAssessment {
        let battery = state.battery_ratio().clamp(0.0, 1.0);
        let temperature = state.cutter_temp_c().max(0.0);
        let coolant_health = if coolant_health.is_finite() {
            coolant_health.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let power_margin = if self.policy.derate_battery_ratio <= 0.0 {
            battery
        } else {
            (battery / self.policy.derate_battery_ratio).clamp(0.0, 1.0)
        };
        let thermal_margin = ((self.policy.thermal_stop_c - temperature)
            / (self.policy.thermal_stop_c - self.policy.thermal_derate_c))
            .clamp(0.0, 1.0);

        let mut assessment =
            if battery <= self.policy.survival_battery_ratio || !maintenance.mobility_available {
                FieldEnvelopeAssessment {
                    mode: FieldEnvelopeMode::SurvivalHold,
                    power_margin,
                    thermal_margin,
                    cutter_cap: 0.0,
                    auger_cap: 0.0,
                    track_cap: if maintenance.mobility_available {
                        0.2
                    } else {
                        0.0
                    },
                    recovery_cap: 0.35,
                    cooling_floor: 0.0,
                    mission_work_allowed: false,
                }
            } else if temperature >= self.policy.thermal_stop_c
                || coolant_health < self.policy.minimum_coolant_health
                || !maintenance.cooling_available
            {
                FieldEnvelopeAssessment {
                    mode: FieldEnvelopeMode::ThermalProtection,
                    power_margin,
                    thermal_margin,
                    cutter_cap: 0.0,
                    auger_cap: 0.0,
                    track_cap: 0.35,
                    recovery_cap: 0.7,
                    cooling_floor: if maintenance.cooling_available {
                        0.85
                    } else {
                        0.0
                    },
                    mission_work_allowed: false,
                }
            } else if battery <= self.policy.critical_battery_ratio {
                FieldEnvelopeAssessment {
                    mode: FieldEnvelopeMode::CriticalPower,
                    power_margin,
                    thermal_margin,
                    cutter_cap: 0.1,
                    auger_cap: 0.2,
                    track_cap: 0.4,
                    recovery_cap: 0.6,
                    cooling_floor: if temperature > self.policy.thermal_derate_c {
                        0.65
                    } else {
                        0.0
                    },
                    mission_work_allowed: false,
                }
            } else if battery <= self.policy.derate_battery_ratio
                || temperature >= self.policy.thermal_derate_c
                || maintenance.maintenance_due
            {
                let cap = power_margin.min(thermal_margin).clamp(0.25, 0.75) as f32;
                FieldEnvelopeAssessment {
                    mode: FieldEnvelopeMode::Derated,
                    power_margin,
                    thermal_margin,
                    cutter_cap: cap * 0.8,
                    auger_cap: cap,
                    track_cap: cap.max(0.4),
                    recovery_cap: cap.max(0.5),
                    cooling_floor: if temperature >= self.policy.thermal_derate_c {
                        0.55
                    } else {
                        0.0
                    },
                    mission_work_allowed: false,
                }
            } else {
                FieldEnvelopeAssessment {
                    power_margin,
                    thermal_margin,
                    ..FieldEnvelopeAssessment::nominal()
                }
            };

        if state.cutter_temp_c() >= self.policy.thermal_derate_c && maintenance.cooling_available {
            assessment.cooling_floor = assessment.cooling_floor.max(0.55);
        }
        if assessment.mode != self.last_assessment.mode {
            self.transitions = self.transitions.saturating_add(1);
        }
        self.last_assessment = assessment;
        assessment
    }

    pub const fn last_assessment(&self) -> FieldEnvelopeAssessment {
        self.last_assessment
    }

    pub const fn transitions(&self) -> u64 {
        self.transitions
    }

    pub fn reset_runtime(&mut self) {
        self.last_assessment = FieldEnvelopeAssessment::nominal();
    }
}

impl Default for FieldEnvelopeSupervisor {
    fn default() -> Self {
        Self::new(FieldEnvelopePolicy::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BATTERY_RATIO, CUTTER_TEMP_C};

    #[test]
    fn critical_power_removes_boring_before_mobility() {
        let mut state = SubterraneanState::home();
        state.channels[BATTERY_RATIO] = 0.15;
        let mut supervisor = FieldEnvelopeSupervisor::default();
        let assessment = supervisor.assess(&state, 1.0, MaintenanceAssessment::nominal());
        assert_eq!(assessment.mode, FieldEnvelopeMode::CriticalPower);
        let mut command = SubterraneanCommand::zero();
        command.set_cutter_head(1.0);
        command.set_left_track(1.0);
        let command = assessment.constrain(command);
        assert!(command.cutter_head() <= 0.1);
        assert!(command.left_track() >= command.cutter_head());
    }

    #[test]
    fn thermal_protection_stops_cutting_and_preserves_cooling() {
        let mut state = SubterraneanState::home();
        state.channels[CUTTER_TEMP_C] = 150.0;
        let mut supervisor = FieldEnvelopeSupervisor::default();
        let assessment = supervisor.assess(&state, 1.0, MaintenanceAssessment::nominal());
        let mut command = SubterraneanCommand::zero();
        command.set_cutter_head(1.0);
        let command = assessment.constrain(command);
        assert_eq!(command.cutter_head(), 0.0);
        assert!(command.thermal_pump() >= 0.85);
    }

    #[test]
    fn failed_cooling_does_not_invent_pump_authority() {
        let mut state = SubterraneanState::home();
        state.channels[CUTTER_TEMP_C] = 150.0;
        let mut maintenance = MaintenanceAssessment::nominal();
        maintenance.cooling_available = false;
        let mut supervisor = FieldEnvelopeSupervisor::default();
        let assessment = supervisor.assess(&state, 1.0, maintenance);
        assert_eq!(assessment.cooling_floor, 0.0);
    }
}
