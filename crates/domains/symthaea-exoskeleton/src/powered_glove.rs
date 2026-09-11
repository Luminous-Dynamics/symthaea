// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Powered EVA glove and hand digital twin (SX-018).
//!
//! This is a human-assist research model, not an autonomous gripper or a
//! flight-qualified glove. Pressurization adds closing resistance; powered
//! tendons may reduce human workload or deliberately add exercise resistance.
//! Any loss of powered authority removes active force. Reference coefficients
//! carry simulation evidence only.

use serde::{Deserialize, Serialize};

use crate::space_exosuit::ExosuitEvidenceLevel;

pub const NUM_GLOVE_DIGITS: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(usize)]
pub enum HandDigit {
    Thumb = 0,
    Index = 1,
    Middle = 2,
    Ring = 3,
    Little = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoweredGloveMode {
    Transparent,
    FineManipulation,
    GripAssist,
    HoldAssist,
    ExerciseResistance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoweredGloveFailState {
    Nominal,
    Degraded,
    PassiveBackdrivable,
    ServiceRequired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoweredGloveFault {
    InvalidConfig,
    InvalidState,
    InvalidCommand,
    InvalidTimeStep,
    PowerUnavailable,
    ElectronicsFault,
    SensorConfidenceLow,
    DriveHealthLow,
    PressureIntegrityLow,
    ReleasePathBlocked,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PoweredGloveConfig {
    pub max_assist_force_n: [f64; NUM_GLOVE_DIGITS],
    pub max_exercise_resistance_n: [f64; NUM_GLOVE_DIGITS],
    pub pressure_resistance_coeff_n_pa: [f64; NUM_GLOVE_DIGITS],
    pub passive_breakaway_force_n: [f64; NUM_GLOVE_DIGITS],
    /// Tactility proxy before pressure, abrasion, sensing, and mode effects.
    pub passive_tactile_fraction: [f64; NUM_GLOVE_DIGITS],
    pub tactile_pressure_loss_per_pa: f64,
    pub fine_assist_fraction: f64,
    pub grip_assist_fraction: f64,
    pub hold_assist_fraction: f64,
    pub motor_efficiency: f64,
    pub static_hold_w_per_n: f64,
    pub max_electrical_power_w: f64,
    pub fatigue_reference_force_n: f64,
    pub fatigue_time_constant_s: f64,
    pub fatigue_recovery_time_constant_s: f64,
    pub min_sensor_confidence: f64,
    pub min_drive_health: f64,
    pub min_pressure_integrity: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl PoweredGloveConfig {
    pub fn simulation_reference() -> Self {
        Self {
            max_assist_force_n: [35.0, 30.0, 30.0, 25.0, 20.0],
            max_exercise_resistance_n: [18.0, 16.0, 16.0, 14.0, 12.0],
            pressure_resistance_coeff_n_pa: [2.2e-4, 2.0e-4, 2.0e-4, 1.8e-4, 1.6e-4],
            passive_breakaway_force_n: [2.5, 2.2, 2.2, 2.0, 1.8],
            passive_tactile_fraction: [0.72, 0.76, 0.76, 0.72, 0.68],
            tactile_pressure_loss_per_pa: 2.0e-6,
            fine_assist_fraction: 0.30,
            grip_assist_fraction: 0.70,
            hold_assist_fraction: 0.90,
            motor_efficiency: 0.72,
            static_hold_w_per_n: 0.08,
            max_electrical_power_w: 45.0,
            fatigue_reference_force_n: 30.0,
            fatigue_time_constant_s: 900.0,
            fatigue_recovery_time_constant_s: 600.0,
            min_sensor_confidence: 0.80,
            min_drive_health: 0.70,
            min_pressure_integrity: 0.95,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.max_assist_force_n
            .iter()
            .chain(self.max_exercise_resistance_n.iter())
            .chain(self.pressure_resistance_coeff_n_pa.iter())
            .chain(self.passive_breakaway_force_n.iter())
            .all(|v| v.is_finite() && *v >= 0.0)
            && self
                .passive_tactile_fraction
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            && self.tactile_pressure_loss_per_pa.is_finite()
            && self.tactile_pressure_loss_per_pa >= 0.0
            && [
                self.fine_assist_fraction,
                self.grip_assist_fraction,
                self.hold_assist_fraction,
                self.motor_efficiency,
                self.min_sensor_confidence,
                self.min_drive_health,
                self.min_pressure_integrity,
            ]
            .into_iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(&v))
            && self.motor_efficiency > 0.0
            && self.fine_assist_fraction <= self.grip_assist_fraction
            && self.grip_assist_fraction <= self.hold_assist_fraction
            && self.static_hold_w_per_n.is_finite()
            && self.static_hold_w_per_n >= 0.0
            && self.max_electrical_power_w.is_finite()
            && self.max_electrical_power_w >= 0.0
            && self.fatigue_reference_force_n.is_finite()
            && self.fatigue_reference_force_n > 0.0
            && self.fatigue_time_constant_s.is_finite()
            && self.fatigue_time_constant_s > 0.0
            && self.fatigue_recovery_time_constant_s.is_finite()
            && self.fatigue_recovery_time_constant_s > 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PoweredGloveState {
    pub fatigue: [f64; NUM_GLOVE_DIGITS],
    pub drive_health: [f64; NUM_GLOVE_DIGITS],
    pub sensor_confidence: [f64; NUM_GLOVE_DIGITS],
    /// Surface/outer-layer condition proxy [0,1]. Used only for tactile degradation.
    pub palm_abrasion_health: [f64; NUM_GLOVE_DIGITS],
    pub pressure_integrity: f64,
    pub power_available: bool,
    pub electronics_healthy: bool,
    /// Mechanical release/backdrive path must remain available without power.
    pub passive_release_path_clear: bool,
    pub suit_pressure_pa: f64,
    pub ambient_pressure_pa: f64,
}

impl PoweredGloveState {
    pub fn simulation_reference() -> Self {
        Self {
            fatigue: [0.0; NUM_GLOVE_DIGITS],
            drive_health: [1.0; NUM_GLOVE_DIGITS],
            sensor_confidence: [1.0; NUM_GLOVE_DIGITS],
            palm_abrasion_health: [1.0; NUM_GLOVE_DIGITS],
            pressure_integrity: 1.0,
            power_available: true,
            electronics_healthy: true,
            passive_release_path_clear: true,
            suit_pressure_pa: 30_000.0,
            ambient_pressure_pa: 0.0,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.fatigue
            .iter()
            .chain(self.drive_health.iter())
            .chain(self.sensor_confidence.iter())
            .chain(self.palm_abrasion_health.iter())
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            && self.pressure_integrity.is_finite()
            && (0.0..=1.0).contains(&self.pressure_integrity)
            && self.suit_pressure_pa.is_finite()
            && self.suit_pressure_pa >= 0.0
            && self.ambient_pressure_pa.is_finite()
            && self.ambient_pressure_pa >= 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PoweredGloveCommand {
    pub mode: PoweredGloveMode,
    pub requested_contact_force_n: [f64; NUM_GLOVE_DIGITS],
    pub tendon_speed_m_s: [f64; NUM_GLOVE_DIGITS],
    pub exercise_resistance_fraction: f64,
}

impl PoweredGloveCommand {
    pub fn transparent(requested_contact_force_n: [f64; NUM_GLOVE_DIGITS]) -> Self {
        Self {
            mode: PoweredGloveMode::Transparent,
            requested_contact_force_n,
            tendon_speed_m_s: [0.0; NUM_GLOVE_DIGITS],
            exercise_resistance_fraction: 0.0,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.requested_contact_force_n
            .iter()
            .chain(self.tendon_speed_m_s.iter())
            .all(|v| v.is_finite() && *v >= 0.0)
            && self.exercise_resistance_fraction.is_finite()
            && (0.0..=1.0).contains(&self.exercise_resistance_fraction)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PoweredGloveStep {
    pub fail_state: PoweredGloveFailState,
    pub faults: [Option<PoweredGloveFault>; 6],
    pub pressure_resistance_force_n: [f64; NUM_GLOVE_DIGITS],
    pub exercise_resistance_force_n: [f64; NUM_GLOVE_DIGITS],
    pub assist_force_n: [f64; NUM_GLOVE_DIGITS],
    pub human_required_force_n: [f64; NUM_GLOVE_DIGITS],
    pub tactile_fidelity: [f64; NUM_GLOVE_DIGITS],
    pub electrical_power_w: f64,
    pub backdrivable: bool,
    pub fatigue: [f64; NUM_GLOVE_DIGITS],
    pub evidence: ExosuitEvidenceLevel,
}

#[derive(Debug, Clone)]
pub struct PoweredGloveTwin {
    config: PoweredGloveConfig,
    state: PoweredGloveState,
}

impl PoweredGloveTwin {
    pub fn new(config: PoweredGloveConfig, state: PoweredGloveState) -> Result<Self, PoweredGloveFault> {
        if !config.is_valid() {
            return Err(PoweredGloveFault::InvalidConfig);
        }
        if !state.is_valid() {
            return Err(PoweredGloveFault::InvalidState);
        }
        Ok(Self { config, state })
    }

    pub fn simulation_reference() -> Self {
        Self::new(
            PoweredGloveConfig::simulation_reference(),
            PoweredGloveState::simulation_reference(),
        )
        .expect("reference powered glove must be valid")
    }

    pub fn config(&self) -> &PoweredGloveConfig {
        &self.config
    }

    pub fn state(&self) -> &PoweredGloveState {
        &self.state
    }

    pub fn state_mut_for_fault_injection(&mut self) -> &mut PoweredGloveState {
        &mut self.state
    }

    pub fn step(
        &mut self,
        command: PoweredGloveCommand,
        dt_s: f64,
    ) -> Result<PoweredGloveStep, PoweredGloveFault> {
        if !self.config.is_valid() {
            return Err(PoweredGloveFault::InvalidConfig);
        }
        if !self.state.is_valid() {
            return Err(PoweredGloveFault::InvalidState);
        }
        if !command.is_valid() {
            return Err(PoweredGloveFault::InvalidCommand);
        }
        if !dt_s.is_finite() || dt_s <= 0.0 {
            return Err(PoweredGloveFault::InvalidTimeStep);
        }

        let delta_p = (self.state.suit_pressure_pa - self.state.ambient_pressure_pa).max(0.0);
        let faults = self.authority_faults();
        let authority_denied = faults.iter().flatten().next().is_some();

        let mut pressure = [0.0; NUM_GLOVE_DIGITS];
        let mut exercise = [0.0; NUM_GLOVE_DIGITS];
        let mut assist = [0.0; NUM_GLOVE_DIGITS];

        for i in 0..NUM_GLOVE_DIGITS {
            pressure[i] = self.config.passive_breakaway_force_n[i]
                + self.config.pressure_resistance_coeff_n_pa[i] * delta_p;
        }

        if !authority_denied {
            if command.mode == PoweredGloveMode::ExerciseResistance {
                for i in 0..NUM_GLOVE_DIGITS {
                    exercise[i] = self.config.max_exercise_resistance_n[i]
                        * self.state.drive_health[i]
                        * command.exercise_resistance_fraction;
                }
            } else {
                let fraction = match command.mode {
                    PoweredGloveMode::Transparent => 0.0,
                    PoweredGloveMode::FineManipulation => self.config.fine_assist_fraction,
                    PoweredGloveMode::GripAssist => self.config.grip_assist_fraction,
                    PoweredGloveMode::HoldAssist => self.config.hold_assist_fraction,
                    PoweredGloveMode::ExerciseResistance => 0.0,
                };
                for i in 0..NUM_GLOVE_DIGITS {
                    let gross_load = command.requested_contact_force_n[i] + pressure[i];
                    assist[i] = (self.config.max_assist_force_n[i]
                        * self.state.drive_health[i]
                        * fraction)
                        .min(gross_load);
                }
            }
        }

        let raw_power = active_power_w(
            &assist,
            &exercise,
            &command.tendon_speed_m_s,
            self.config.motor_efficiency,
            self.config.static_hold_w_per_n,
        );
        if raw_power > self.config.max_electrical_power_w && raw_power > 0.0 {
            let scale = self.config.max_electrical_power_w / raw_power;
            for value in assist.iter_mut().chain(exercise.iter_mut()) {
                *value *= scale;
            }
        }
        let electrical_power_w = active_power_w(
            &assist,
            &exercise,
            &command.tendon_speed_m_s,
            self.config.motor_efficiency,
            self.config.static_hold_w_per_n,
        );

        let mut human = [0.0; NUM_GLOVE_DIGITS];
        let mut tactile = [0.0; NUM_GLOVE_DIGITS];
        let pressure_tactile_factor =
            (1.0 - self.config.tactile_pressure_loss_per_pa * delta_p).clamp(0.0, 1.0);
        let mode_tactile_factor = match command.mode {
            PoweredGloveMode::Transparent | PoweredGloveMode::FineManipulation => 1.0,
            PoweredGloveMode::GripAssist => 0.96,
            PoweredGloveMode::HoldAssist => 0.92,
            PoweredGloveMode::ExerciseResistance => 0.96,
        };

        for i in 0..NUM_GLOVE_DIGITS {
            human[i] = (command.requested_contact_force_n[i] + pressure[i] + exercise[i] - assist[i])
                .max(0.0);
            self.state.fatigue[i] = update_fatigue(
                self.state.fatigue[i],
                human[i],
                dt_s,
                &self.config,
            );
            tactile[i] = (self.config.passive_tactile_fraction[i]
                * pressure_tactile_factor
                * self.state.sensor_confidence[i]
                * self.state.palm_abrasion_health[i]
                * mode_tactile_factor)
                .clamp(0.0, 1.0);
        }

        let fail_state = if authority_denied {
            if self.state.passive_release_path_clear {
                PoweredGloveFailState::PassiveBackdrivable
            } else {
                PoweredGloveFailState::ServiceRequired
            }
        } else if self.state.drive_health.iter().any(|h| *h < 0.90)
            || self.state.sensor_confidence.iter().any(|c| *c < 0.90)
        {
            PoweredGloveFailState::Degraded
        } else {
            PoweredGloveFailState::Nominal
        };

        Ok(PoweredGloveStep {
            fail_state,
            faults,
            pressure_resistance_force_n: pressure,
            exercise_resistance_force_n: exercise,
            assist_force_n: assist,
            human_required_force_n: human,
            tactile_fidelity: tactile,
            electrical_power_w,
            backdrivable: self.state.passive_release_path_clear,
            fatigue: self.state.fatigue,
            evidence: self.config.evidence,
        })
    }

    fn authority_faults(&self) -> [Option<PoweredGloveFault>; 6] {
        let mut faults = [None; 6];
        let mut n = 0;
        let mut push = |fault| {
            if n < faults.len() {
                faults[n] = Some(fault);
                n += 1;
            }
        };
        if !self.state.power_available {
            push(PoweredGloveFault::PowerUnavailable);
        }
        if !self.state.electronics_healthy {
            push(PoweredGloveFault::ElectronicsFault);
        }
        if self
            .state
            .sensor_confidence
            .iter()
            .any(|v| *v < self.config.min_sensor_confidence)
        {
            push(PoweredGloveFault::SensorConfidenceLow);
        }
        if self
            .state
            .drive_health
            .iter()
            .any(|v| *v < self.config.min_drive_health)
        {
            push(PoweredGloveFault::DriveHealthLow);
        }
        if self.state.pressure_integrity < self.config.min_pressure_integrity {
            push(PoweredGloveFault::PressureIntegrityLow);
        }
        if !self.state.passive_release_path_clear {
            push(PoweredGloveFault::ReleasePathBlocked);
        }
        faults
    }
}

fn active_power_w(
    assist: &[f64; NUM_GLOVE_DIGITS],
    exercise: &[f64; NUM_GLOVE_DIGITS],
    speed: &[f64; NUM_GLOVE_DIGITS],
    efficiency: f64,
    static_hold_w_per_n: f64,
) -> f64 {
    assist
        .iter()
        .zip(exercise.iter())
        .zip(speed.iter())
        .map(|((a, r), v)| {
            let force = a + r;
            force * v / efficiency + force * static_hold_w_per_n
        })
        .sum()
}

fn update_fatigue(
    current: f64,
    human_force_n: f64,
    dt_s: f64,
    config: &PoweredGloveConfig,
) -> f64 {
    let normalized = human_force_n / config.fatigue_reference_force_n;
    let accumulation = normalized * normalized * dt_s / config.fatigue_time_constant_s;
    let recovery = if normalized < 0.10 {
        current * dt_s / config.fatigue_recovery_time_constant_s
    } else {
        0.0
    };
    (current + accumulation - recovery).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grip(mode: PoweredGloveMode) -> PoweredGloveCommand {
        PoweredGloveCommand {
            mode,
            requested_contact_force_n: [20.0; NUM_GLOVE_DIGITS],
            tendon_speed_m_s: [0.02; NUM_GLOVE_DIGITS],
            exercise_resistance_fraction: 0.0,
        }
    }

    #[test]
    fn grip_assist_reduces_human_force() {
        let mut transparent = PoweredGloveTwin::simulation_reference();
        let mut assisted = PoweredGloveTwin::simulation_reference();
        let a = transparent.step(grip(PoweredGloveMode::Transparent), 1.0).unwrap();
        let b = assisted.step(grip(PoweredGloveMode::GripAssist), 1.0).unwrap();
        assert!(b.human_required_force_n.iter().sum::<f64>()
            < a.human_required_force_n.iter().sum::<f64>());
        assert!(b.electrical_power_w > 0.0);
    }

    #[test]
    fn loss_of_power_removes_all_active_force_and_is_backdrivable() {
        let mut glove = PoweredGloveTwin::simulation_reference();
        glove.state_mut_for_fault_injection().power_available = false;
        let mut cmd = grip(PoweredGloveMode::ExerciseResistance);
        cmd.exercise_resistance_fraction = 1.0;
        let step = glove.step(cmd, 1.0).unwrap();
        assert_eq!(step.fail_state, PoweredGloveFailState::PassiveBackdrivable);
        assert!(step.assist_force_n.iter().all(|v| *v == 0.0));
        assert!(step.exercise_resistance_force_n.iter().all(|v| *v == 0.0));
        assert_eq!(step.electrical_power_w, 0.0);
        assert!(step.backdrivable);
    }

    #[test]
    fn blocked_release_path_removes_powered_authority() {
        let mut glove = PoweredGloveTwin::simulation_reference();
        glove.state_mut_for_fault_injection().passive_release_path_clear = false;
        let step = glove.step(grip(PoweredGloveMode::HoldAssist), 1.0).unwrap();
        assert_eq!(step.fail_state, PoweredGloveFailState::ServiceRequired);
        assert!(step.assist_force_n.iter().all(|v| *v == 0.0));
        assert!(!step.backdrivable);
    }

    #[test]
    fn higher_pressure_increases_hand_load_and_reduces_tactile_proxy() {
        let mut low = PoweredGloveTwin::simulation_reference();
        let mut high = PoweredGloveTwin::simulation_reference();
        low.state_mut_for_fault_injection().suit_pressure_pa = 20_000.0;
        high.state_mut_for_fault_injection().suit_pressure_pa = 50_000.0;
        let a = low.step(grip(PoweredGloveMode::Transparent), 1.0).unwrap();
        let b = high.step(grip(PoweredGloveMode::Transparent), 1.0).unwrap();
        assert!(b.human_required_force_n.iter().sum::<f64>()
            > a.human_required_force_n.iter().sum::<f64>());
        assert!(b.tactile_fidelity.iter().sum::<f64>() < a.tactile_fidelity.iter().sum::<f64>());
    }

    #[test]
    fn exercise_mode_adds_loading_and_consumes_power() {
        let mut glove = PoweredGloveTwin::simulation_reference();
        let mut cmd = grip(PoweredGloveMode::ExerciseResistance);
        cmd.exercise_resistance_fraction = 1.0;
        let step = glove.step(cmd, 1.0).unwrap();
        assert!(step.exercise_resistance_force_n.iter().sum::<f64>() > 0.0);
        assert!(step.assist_force_n.iter().all(|v| *v == 0.0));
        assert!(step.electrical_power_w > 0.0);
    }

    #[test]
    fn fine_manipulation_preserves_more_tactile_proxy_than_hold_assist() {
        let mut fine = PoweredGloveTwin::simulation_reference();
        let mut hold = PoweredGloveTwin::simulation_reference();
        let a = fine.step(grip(PoweredGloveMode::FineManipulation), 1.0).unwrap();
        let b = hold.step(grip(PoweredGloveMode::HoldAssist), 1.0).unwrap();
        assert!(a.tactile_fidelity.iter().sum::<f64>() > b.tactile_fidelity.iter().sum::<f64>());
    }

    #[test]
    fn electrical_ceiling_scales_active_force() {
        let mut config = PoweredGloveConfig::simulation_reference();
        config.max_electrical_power_w = 2.0;
        let mut glove = PoweredGloveTwin::new(config, PoweredGloveState::simulation_reference()).unwrap();
        let step = glove.step(grip(PoweredGloveMode::HoldAssist), 1.0).unwrap();
        assert!(step.electrical_power_w <= 2.0 + 1e-9);
    }

    #[test]
    fn low_pressure_integrity_denies_powered_grip() {
        let mut glove = PoweredGloveTwin::simulation_reference();
        glove.state_mut_for_fault_injection().pressure_integrity = 0.80;
        let step = glove.step(grip(PoweredGloveMode::HoldAssist), 1.0).unwrap();
        assert!(step.assist_force_n.iter().all(|v| *v == 0.0));
        assert!(step.faults.contains(&Some(PoweredGloveFault::PressureIntegrityLow)));
    }

    #[test]
    fn powered_assist_reduces_repeated_work_fatigue() {
        let mut unassisted = PoweredGloveTwin::simulation_reference();
        let mut assisted = PoweredGloveTwin::simulation_reference();
        for _ in 0..120 {
            unassisted.step(grip(PoweredGloveMode::Transparent), 1.0).unwrap();
            assisted.step(grip(PoweredGloveMode::GripAssist), 1.0).unwrap();
        }
        assert!(assisted.state().fatigue.iter().sum::<f64>()
            < unassisted.state().fatigue.iter().sum::<f64>());
    }
}
