// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Powered EVA glove and hand digital twin (SX-018).
//!
//! The glove is modeled as a human-assist device, not an autonomous gripper.
//! Pressurization adds closing resistance, powered tendons may reduce human
//! effort, and exercise mode may deliberately add resistance. Any loss of
//! powered authority removes active force rather than increasing grip.
//!
//! The reference coefficients are simulation inputs only. They are not xEMU,
//! AxEMU, EMU, or human-rating data.

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
    /// No powered force. The wearer moves the glove through its passive load.
    Transparent,
    /// Low-authority assistance intended to preserve fine manipulation.
    FineManipulation,
    /// Moderate assistance for ordinary gripping/tool use.
    GripAssist,
    /// Higher assistance for sustained static grip, still bounded and backdrivable.
    HoldAssist,
    /// Deliberately adds closing resistance for low-g exercise/training.
    ExerciseResistance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoweredGloveFailState {
    Nominal,
    Degraded,
    /// Powered force is removed and the mechanism is expected to be mechanically backdrivable.
    PassiveBackdrivable,
    /// Powered force is removed but the declared passive-release path is obstructed.
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
    ReleasePathBlocked,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PoweredGloveConfig {
    /// Maximum tendon assist force per digit, N.
    pub max_assist_force_n: [f64; NUM_GLOVE_DIGITS],
    /// Maximum powered exercise resistance per digit, N.
    pub max_exercise_resistance_n: [f64; NUM_GLOVE_DIGITS],
    /// Pressure-induced closing resistance coefficient, N / Pa.
    pub pressure_resistance_coeff_n_pa: [f64; NUM_GLOVE_DIGITS],
    /// Passive breakaway/friction force in the garment/tendon routing, N.
    pub passive_breakaway_force_n: [f64; NUM_GLOVE_DIGITS],
    /// Fraction of the nominal maximum assist allowed in fine-manipulation mode.
    pub fine_assist_fraction: f64,
    /// Fraction allowed in ordinary grip-assist mode.
    pub grip_assist_fraction: f64,
    /// Fraction allowed in sustained-hold mode.
    pub hold_assist_fraction: f64,
    /// Tendon/drive mechanical-to-electrical efficiency proxy.
    pub motor_efficiency: f64,
    /// Static holding electrical power proxy, W per N of assist.
    pub static_hold_w_per_n: f64,
    /// Glove-wide electrical ceiling. Assistance scales down to respect it.
    pub max_electrical_power_w: f64,
    /// Human reference force for normalized fatigue accumulation, N per digit.
    pub fatigue_reference_force_n: f64,
    /// Time constant for fatigue accumulation at reference load, s.
    pub fatigue_time_constant_s: f64,
    /// Time constant for recovery at near-zero load, s.
    pub fatigue_recovery_time_constant_s: f64,
    /// Minimum trusted sensor confidence [0,1].
    pub min_sensor_confidence: f64,
    /// Minimum powered drive health [0,1].
    pub min_drive_health: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl PoweredGloveConfig {
    pub fn simulation_reference() -> Self {
        Self {
            max_assist_force_n: [35.0, 30.0, 30.0, 25.0, 20.0],
            max_exercise_resistance_n: [18.0, 16.0, 16.0, 14.0, 12.0],
            pressure_resistance_coeff_n_pa: [2.2e-4, 2.0e-4, 2.0e-4, 1.8e-4, 1.6e-4],
            passive_breakaway_force_n: [2.5, 2.2, 2.2, 2.0, 1.8],
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
            && [
                self.fine_assist_fraction,
                self.grip_assist_fraction,
                self.hold_assist_fraction,
                self.motor_efficiency,
                self.min_sensor_confidence,
                self.min_drive_health,
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
    /// Human hand fatigue estimate per digit [0,1].
    pub fatigue: [f64; NUM_GLOVE_DIGITS],
    /// Powered tendon/drive health per digit [0,1].
    pub drive_health: [f64; NUM_GLOVE_DIGITS],
    /// Confidence in force/position/tendon sensing per digit [0,1].
    pub sensor_confidence: [f64; NUM_GLOVE_DIGITS],
    /// Whether powered-glove energy is available.
    pub power_available: bool,
    /// Whether the glove electronics/safety channel is healthy.
    pub electronics_healthy: bool,
    /// True only if the mechanism has a clear mechanical backdrive/release path.
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
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            && self.suit_pressure_pa.is_finite()
            && self.suit_pressure_pa >= 0.0
            && self.ambient_pressure_pa.is_finite()
            && self.ambient_pressure_pa >= 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PoweredGloveCommand {
    pub mode: PoweredGloveMode,
    /// Desired net contact/grip force per digit, N.
    pub requested_contact_force_n: [f64; NUM_GLOVE_DIGITS],
    /// Absolute tendon travel speed proxy per digit, m/s.
    pub tendon_speed_m_s: [f64; NUM_GLOVE_DIGITS],
    /// Requested fraction of available exercise resistance [0,1].
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
    pub faults: [Option<PoweredGloveFault>; 5],
    pub pressure_resistance_force_n: [f64; NUM_GLOVE_DIGITS],
    pub exercise_resistance_force_n: [f64; NUM_GLOVE_DIGITS],
    pub assist_force_n: [f64; NUM_GLOVE_DIGITS],
    pub human_required_force_n: [f64; NUM_GLOVE_DIGITS],
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

        let pressure_delta_pa = (self.state.suit_pressure_pa - self.state.ambient_pressure_pa).max(0.0);
        let mut pressure_resistance_force_n = [0.0; NUM_GLOVE_DIGITS];
        let mut exercise_resistance_force_n = [0.0; NUM_GLOVE_DIGITS];
        for i in 0..NUM_GLOVE_DIGITS {
            pressure_resistance_force_n[i] = self.config.passive_breakaway_force_n[i]
                + self.config.pressure_resistance_coeff_n_pa[i] * pressure_delta_pa;
            if command.mode == PoweredGloveMode::ExerciseResistance {
                exercise_resistance_force_n[i] = self.config.max_exercise_resistance_n[i]
                    * command.exercise_resistance_fraction;
            }
        }

        let faults = self.authority_faults();
        let authority_denied = faults.iter().flatten().next().is_some();
        let fail_state = if authority_denied {
            if self.state.passive_release_path_clear {
                PoweredGloveFailState::PassiveBackdrivable
            } else {
                PoweredGloveFailState::ServiceRequired
            }
        } else if self
            .state
            .drive_health
            .iter()
            .any(|h| *h < 0.90)
            || self.state.sensor_confidence.iter().any(|c| *c < 0.90)
        {
            PoweredGloveFailState::Degraded
        } else {
            PoweredGloveFailState::Nominal
        };

        let mode_fraction = match command.mode {
            PoweredGloveMode::Transparent | PoweredGloveMode::ExerciseResistance => 0.0,
            PoweredGloveMode::FineManipulation => self.config.fine_assist_fraction,
            PoweredGloveMode::GripAssist => self.config.grip_assist_fraction,
            PoweredGloveMode::HoldAssist => self.config.hold_assist_fraction,
        };

        let mut assist_force_n = [0.0; NUM_GLOVE_DIGITS];
        if !authority_denied {
            for i in 0..NUM_GLOVE_DIGITS {
                let gross_human_load = command.requested_contact_force_n[i]
                    + pressure_resistance_force_n[i]
                    + exercise_resistance_force_n[i];
                assist_force_n[i] = (self.config.max_assist_force_n[i]
                    * self.state.drive_health[i]
                    * mode_fraction)
                    .min(gross_human_load);
            }
        }

        let raw_power_w = electrical_power_w(
            &assist_force_n,
            &command.tendon_speed_m_s,
            self.config.motor_efficiency,
            self.config.static_hold_w_per_n,
        );
        if raw_power_w > self.config.max_electrical_power_w && raw_power_w > 0.0 {
            let scale = self.config.max_electrical_power_w / raw_power_w;
            for force in &mut assist_force_n {
                *force *= scale;
            }
        }
        let electrical_power_w = electrical_power_w(
            &assist_force_n,
            &command.tendon_speed_m_s,
            self.config.motor_efficiency,
            self.config.static_hold_w_per_n,
        );

        let mut human_required_force_n = [0.0; NUM_GLOVE_DIGITS];
        for i in 0..NUM_GLOVE_DIGITS {
            human_required_force_n[i] = (command.requested_contact_force_n[i]
                + pressure_resistance_force_n[i]
                + exercise_resistance_force_n[i]
                - assist_force_n[i])
                .max(0.0);
            self.state.fatigue[i] = update_fatigue(
                self.state.fatigue[i],
                human_required_force_n[i],
                dt_s,
                &self.config,
            );
        }

        Ok(PoweredGloveStep {
            fail_state,
            faults,
            pressure_resistance_force_n,
            exercise_resistance_force_n,
            assist_force_n,
            human_required_force_n,
            electrical_power_w,
            backdrivable: self.state.passive_release_path_clear,
            fatigue: self.state.fatigue,
            evidence: self.config.evidence,
        })
    }

    fn authority_faults(&self) -> [Option<PoweredGloveFault>; 5] {
        let mut faults = [None; 5];
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
        if !self.state.passive_release_path_clear {
            push(PoweredGloveFault::ReleasePathBlocked);
        }
        faults
    }
}

fn electrical_power_w(
    assist_force_n: &[f64; NUM_GLOVE_DIGITS],
    tendon_speed_m_s: &[f64; NUM_GLOVE_DIGITS],
    efficiency: f64,
    static_hold_w_per_n: f64,
) -> f64 {
    assist_force_n
        .iter()
        .zip(tendon_speed_m_s.iter())
        .map(|(force, speed)| force * speed / efficiency + force * static_hold_w_per_n)
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
        let a = transparent
            .step(grip(PoweredGloveMode::Transparent), 1.0)
            .unwrap();
        let b = assisted.step(grip(PoweredGloveMode::GripAssist), 1.0).unwrap();
        assert!(b.human_required_force_n.iter().sum::<f64>()
            < a.human_required_force_n.iter().sum::<f64>());
        assert!(b.electrical_power_w > 0.0);
    }

    #[test]
    fn loss_of_power_removes_assist_and_leaves_glove_backdrivable() {
        let mut glove = PoweredGloveTwin::simulation_reference();
        glove.state_mut_for_fault_injection().power_available = false;
        let step = glove.step(grip(PoweredGloveMode::HoldAssist), 1.0).unwrap();
        assert_eq!(step.fail_state, PoweredGloveFailState::PassiveBackdrivable);
        assert!(step.assist_force_n.iter().all(|force| *force == 0.0));
        assert_eq!(step.electrical_power_w, 0.0);
        assert!(step.backdrivable);
        assert!(step.faults.contains(&Some(PoweredGloveFault::PowerUnavailable)));
    }

    #[test]
    fn blocked_release_path_removes_powered_authority() {
        let mut glove = PoweredGloveTwin::simulation_reference();
        glove.state_mut_for_fault_injection().passive_release_path_clear = false;
        let step = glove.step(grip(PoweredGloveMode::HoldAssist), 1.0).unwrap();
        assert_eq!(step.fail_state, PoweredGloveFailState::ServiceRequired);
        assert!(step.assist_force_n.iter().all(|force| *force == 0.0));
        assert!(!step.backdrivable);
        assert!(step.faults.contains(&Some(PoweredGloveFault::ReleasePathBlocked)));
    }

    #[test]
    fn higher_pressure_increases_transparent_hand_load() {
        let mut low = PoweredGloveTwin::simulation_reference();
        let mut high = PoweredGloveTwin::simulation_reference();
        low.state_mut_for_fault_injection().suit_pressure_pa = 20_000.0;
        high.state_mut_for_fault_injection().suit_pressure_pa = 50_000.0;
        let low_step = low.step(grip(PoweredGloveMode::Transparent), 1.0).unwrap();
        let high_step = high.step(grip(PoweredGloveMode::Transparent), 1.0).unwrap();
        assert!(high_step.human_required_force_n.iter().sum::<f64>()
            > low_step.human_required_force_n.iter().sum::<f64>());
    }

    #[test]
    fn exercise_mode_increases_human_loading_without_assist() {
        let mut glove = PoweredGloveTwin::simulation_reference();
        let mut cmd = grip(PoweredGloveMode::ExerciseResistance);
        cmd.exercise_resistance_fraction = 1.0;
        let step = glove.step(cmd, 1.0).unwrap();
        assert!(step.exercise_resistance_force_n.iter().sum::<f64>() > 0.0);
        assert!(step.assist_force_n.iter().all(|force| *force == 0.0));
        assert_eq!(step.electrical_power_w, 0.0);
    }

    #[test]
    fn fine_manipulation_has_less_authority_than_grip_assist() {
        let mut fine = PoweredGloveTwin::simulation_reference();
        let mut grip_glove = PoweredGloveTwin::simulation_reference();
        let a = fine
            .step(grip(PoweredGloveMode::FineManipulation), 1.0)
            .unwrap();
        let b = grip_glove
            .step(grip(PoweredGloveMode::GripAssist), 1.0)
            .unwrap();
        assert!(a.assist_force_n.iter().sum::<f64>() < b.assist_force_n.iter().sum::<f64>());
    }

    #[test]
    fn electrical_ceiling_scales_assist() {
        let mut config = PoweredGloveConfig::simulation_reference();
        config.max_electrical_power_w = 2.0;
        let mut glove = PoweredGloveTwin::new(config, PoweredGloveState::simulation_reference()).unwrap();
        let step = glove.step(grip(PoweredGloveMode::HoldAssist), 1.0).unwrap();
        assert!(step.electrical_power_w <= 2.0 + 1e-9);
    }

    #[test]
    fn powered_assist_reduces_accumulated_fatigue_in_repeated_work() {
        let mut unassisted = PoweredGloveTwin::simulation_reference();
        let mut assisted = PoweredGloveTwin::simulation_reference();
        for _ in 0..120 {
            unassisted
                .step(grip(PoweredGloveMode::Transparent), 1.0)
                .unwrap();
            assisted
                .step(grip(PoweredGloveMode::GripAssist), 1.0)
                .unwrap();
        }
        assert!(assisted.state().fatigue.iter().sum::<f64>()
            < unassisted.state().fatigue.iter().sum::<f64>());
    }
}
