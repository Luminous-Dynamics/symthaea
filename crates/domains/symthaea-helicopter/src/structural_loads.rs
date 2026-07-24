// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reduced-order structural-load and vibration evidence.
//!
//! Flight-control success is not sufficient when rotor harmonics, hub moments,
//! or repeated load cycles exceed the airframe's declared envelope. This module
//! converts timestamped load observations into bounded RMS vibration, peak-load,
//! and cumulative fatigue evidence. It is an assurance model, not a substitute
//! for finite-element analysis, ground resonance testing, or certified loads data.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StructuralLoadConfig {
    pub rotor_blade_count: u8,
    /// Convert the simulator's governor RPM into physical main-rotor RPM.
    pub physical_rotor_rpm_scale: f64,
    pub rms_time_constant_s: f64,
    pub max_normal_load_factor_g: f64,
    pub max_hub_moment_nm: f64,
    pub max_vibration_rms_mps2: f64,
    pub fatigue_reference_amplitude: f64,
    pub fatigue_exponent: f64,
    pub maximum_cumulative_damage: f64,
}

impl Default for StructuralLoadConfig {
    fn default() -> Self {
        Self {
            rotor_blade_count: 4,
            physical_rotor_rpm_scale: 0.12,
            rms_time_constant_s: 1.0,
            max_normal_load_factor_g: 2.5,
            max_hub_moment_nm: 18_000.0,
            max_vibration_rms_mps2: 4.0,
            fatigue_reference_amplitude: 1.0,
            fatigue_exponent: 4.0,
            maximum_cumulative_damage: 1.0,
        }
    }
}

impl StructuralLoadConfig {
    pub fn validate(&self) -> Result<(), StructuralLoadError> {
        if self.rotor_blade_count == 0
            || !self.physical_rotor_rpm_scale.is_finite()
            || !(0.0..=1.0).contains(&self.physical_rotor_rpm_scale)
            || !self.rms_time_constant_s.is_finite()
            || self.rms_time_constant_s <= 0.0
            || !self.max_normal_load_factor_g.is_finite()
            || self.max_normal_load_factor_g <= 1.0
            || !self.max_hub_moment_nm.is_finite()
            || self.max_hub_moment_nm <= 0.0
            || !self.max_vibration_rms_mps2.is_finite()
            || self.max_vibration_rms_mps2 <= 0.0
            || !self.fatigue_reference_amplitude.is_finite()
            || self.fatigue_reference_amplitude <= 0.0
            || !self.fatigue_exponent.is_finite()
            || self.fatigue_exponent < 1.0
            || !self.maximum_cumulative_damage.is_finite()
            || self.maximum_cumulative_damage <= 0.0
        {
            return Err(StructuralLoadError::InvalidConfiguration);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StructuralLoadObservation {
    pub monotonic_time_s: f64,
    pub rotor_rpm: f64,
    pub normal_acceleration_mps2: f64,
    pub lateral_acceleration_mps2: f64,
    pub longitudinal_acceleration_mps2: f64,
    pub hub_roll_moment_nm: f64,
    pub hub_pitch_moment_nm: f64,
    pub hub_yaw_moment_nm: f64,
}

impl StructuralLoadObservation {
    fn validate(&self) -> Result<(), StructuralLoadError> {
        if [
            self.monotonic_time_s,
            self.rotor_rpm,
            self.normal_acceleration_mps2,
            self.lateral_acceleration_mps2,
            self.longitudinal_acceleration_mps2,
            self.hub_roll_moment_nm,
            self.hub_pitch_moment_nm,
            self.hub_yaw_moment_nm,
        ]
        .iter()
        .any(|value| !value.is_finite())
            || self.monotonic_time_s < 0.0
            || self.rotor_rpm < 0.0
        {
            return Err(StructuralLoadError::NonFiniteObservation);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StructuralLoadState {
    Nominal,
    Caution,
    LimitExceeded,
    FatigueLimitExceeded,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StructuralLoadEvidence {
    pub state: StructuralLoadState,
    pub samples: u64,
    pub one_per_rev_hz: f64,
    pub blade_pass_hz: f64,
    pub normal_load_factor_g: f64,
    pub hub_moment_magnitude_nm: f64,
    pub vibration_rms_mps2: f64,
    pub peak_normal_load_factor_g: f64,
    pub peak_hub_moment_nm: f64,
    pub cumulative_fatigue_damage: f64,
    pub limit_exceedances: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuralLoadError {
    InvalidConfiguration,
    NonFiniteObservation,
    TimeWentBackwards,
}

#[derive(Debug, Clone)]
pub struct StructuralLoadMonitor {
    config: StructuralLoadConfig,
    last_time_s: Option<f64>,
    vibration_mean_square: f64,
    evidence: StructuralLoadEvidence,
}

impl StructuralLoadMonitor {
    pub fn new(config: StructuralLoadConfig) -> Result<Self, StructuralLoadError> {
        config.validate()?;
        Ok(Self {
            config,
            last_time_s: None,
            vibration_mean_square: 0.0,
            evidence: StructuralLoadEvidence {
                state: StructuralLoadState::Nominal,
                samples: 0,
                one_per_rev_hz: 0.0,
                blade_pass_hz: 0.0,
                normal_load_factor_g: 1.0,
                hub_moment_magnitude_nm: 0.0,
                vibration_rms_mps2: 0.0,
                peak_normal_load_factor_g: 1.0,
                peak_hub_moment_nm: 0.0,
                cumulative_fatigue_damage: 0.0,
                limit_exceedances: 0,
            },
        })
    }

    pub fn evidence(&self) -> StructuralLoadEvidence {
        self.evidence
    }

    pub fn observe(
        &mut self,
        observation: StructuralLoadObservation,
    ) -> Result<StructuralLoadEvidence, StructuralLoadError> {
        self.config.validate()?;
        observation.validate()?;
        if self
            .last_time_s
            .is_some_and(|last| observation.monotonic_time_s < last)
        {
            return Err(StructuralLoadError::TimeWentBackwards);
        }
        let dt_s = self
            .last_time_s
            .map(|last| observation.monotonic_time_s - last)
            .unwrap_or(0.0);
        self.last_time_s = Some(observation.monotonic_time_s);

        let physical_rotor_rpm = observation.rotor_rpm * self.config.physical_rotor_rpm_scale;
        let one_per_rev_hz = physical_rotor_rpm / 60.0;
        let blade_pass_hz = one_per_rev_hz * f64::from(self.config.rotor_blade_count);
        let normal_load_factor_g = observation.normal_acceleration_mps2.abs() / 9.81;
        let hub_moment_magnitude_nm = magnitude3([
            observation.hub_roll_moment_nm,
            observation.hub_pitch_moment_nm,
            observation.hub_yaw_moment_nm,
        ]);
        let vibration_amplitude_mps2 = magnitude3([
            observation.lateral_acceleration_mps2,
            observation.longitudinal_acceleration_mps2,
            observation.normal_acceleration_mps2 - 9.81,
        ]);
        let alpha = if dt_s <= 0.0 {
            1.0
        } else {
            1.0 - (-dt_s / self.config.rms_time_constant_s).exp()
        };
        self.vibration_mean_square +=
            alpha * (vibration_amplitude_mps2.powi(2) - self.vibration_mean_square);
        self.vibration_mean_square = self.vibration_mean_square.max(0.0);
        let vibration_rms_mps2 = self.vibration_mean_square.sqrt();

        if dt_s > 0.0 {
            let normalized_amplitude =
                (vibration_amplitude_mps2 / self.config.fatigue_reference_amplitude).max(0.0);
            let cycles = blade_pass_hz * dt_s;
            self.evidence.cumulative_fatigue_damage +=
                normalized_amplitude.powf(self.config.fatigue_exponent) * cycles * 1.0e-9;
        }

        let instantaneous_limit = normal_load_factor_g > self.config.max_normal_load_factor_g
            || hub_moment_magnitude_nm > self.config.max_hub_moment_nm
            || vibration_rms_mps2 > self.config.max_vibration_rms_mps2;
        if instantaneous_limit {
            self.evidence.limit_exceedances = self.evidence.limit_exceedances.saturating_add(1);
        }
        let caution = normal_load_factor_g > self.config.max_normal_load_factor_g * 0.8
            || hub_moment_magnitude_nm > self.config.max_hub_moment_nm * 0.8
            || vibration_rms_mps2 > self.config.max_vibration_rms_mps2 * 0.8;
        let state =
            if self.evidence.cumulative_fatigue_damage > self.config.maximum_cumulative_damage {
                StructuralLoadState::FatigueLimitExceeded
            } else if instantaneous_limit {
                StructuralLoadState::LimitExceeded
            } else if caution {
                StructuralLoadState::Caution
            } else {
                StructuralLoadState::Nominal
            };

        self.evidence.state = state;
        self.evidence.samples = self.evidence.samples.saturating_add(1);
        self.evidence.one_per_rev_hz = one_per_rev_hz;
        self.evidence.blade_pass_hz = blade_pass_hz;
        self.evidence.normal_load_factor_g = normal_load_factor_g;
        self.evidence.hub_moment_magnitude_nm = hub_moment_magnitude_nm;
        self.evidence.vibration_rms_mps2 = vibration_rms_mps2;
        self.evidence.peak_normal_load_factor_g = self
            .evidence
            .peak_normal_load_factor_g
            .max(normal_load_factor_g);
        self.evidence.peak_hub_moment_nm = self
            .evidence
            .peak_hub_moment_nm
            .max(hub_moment_magnitude_nm);
        Ok(self.evidence)
    }
}

fn magnitude3(values: [f64; 3]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nominal(time_s: f64) -> StructuralLoadObservation {
        StructuralLoadObservation {
            monotonic_time_s: time_s,
            rotor_rpm: 3_300.0,
            normal_acceleration_mps2: 9.81,
            lateral_acceleration_mps2: 0.2,
            longitudinal_acceleration_mps2: 0.1,
            hub_roll_moment_nm: 1_000.0,
            hub_pitch_moment_nm: 500.0,
            hub_yaw_moment_nm: 200.0,
        }
    }

    #[test]
    fn reports_rotor_harmonic_frequencies() {
        let mut monitor = StructuralLoadMonitor::new(StructuralLoadConfig::default()).unwrap();
        let evidence = monitor.observe(nominal(0.0)).unwrap();
        assert!((evidence.one_per_rev_hz - 6.6).abs() < 1.0e-9);
        assert!((evidence.blade_pass_hz - 26.4).abs() < 1.0e-9);
    }

    #[test]
    fn high_hub_moment_is_limit_exceedance() {
        let mut monitor = StructuralLoadMonitor::new(StructuralLoadConfig::default()).unwrap();
        let mut observation = nominal(0.0);
        observation.hub_roll_moment_nm = 20_000.0;
        let evidence = monitor.observe(observation).unwrap();
        assert_eq!(evidence.state, StructuralLoadState::LimitExceeded);
        assert_eq!(evidence.limit_exceedances, 1);
    }

    #[test]
    fn fatigue_damage_is_monotonic() {
        let mut monitor = StructuralLoadMonitor::new(StructuralLoadConfig::default()).unwrap();
        monitor.observe(nominal(0.0)).unwrap();
        let first = monitor
            .observe(nominal(1.0))
            .unwrap()
            .cumulative_fatigue_damage;
        let second = monitor
            .observe(nominal(2.0))
            .unwrap()
            .cumulative_fatigue_damage;
        assert!(second >= first);
    }

    #[test]
    fn time_reversal_fails_closed() {
        let mut monitor = StructuralLoadMonitor::new(StructuralLoadConfig::default()).unwrap();
        monitor.observe(nominal(2.0)).unwrap();
        assert_eq!(
            monitor.observe(nominal(1.0)),
            Err(StructuralLoadError::TimeWentBackwards)
        );
    }
}
