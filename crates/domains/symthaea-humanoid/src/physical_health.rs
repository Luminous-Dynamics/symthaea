// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Morphology-generic physical-health evidence for humanoid authority.
//!
//! This module is intentionally upstream of any concrete HAL. A hardware backend
//! may report actuator/current/temperature/power evidence here, but it remains
//! responsible for its own independent drive-local limits, fuses, watchdogs,
//! contactors, thermal cut-outs, and e-stop behavior.
//!
//! The values derived here are authority **derating**, not certification claims.

use serde::{Deserialize, Serialize};

use crate::execution::HumanoidAuthorityEnvelope;
use crate::morphology::HumanoidMorphology;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ActuatorHealthSample {
    pub enabled: bool,
    pub feedback_valid: bool,
    pub current_a: f64,
    pub current_limit_a: f64,
    /// Missing temperature feedback is represented explicitly rather than by a
    /// fabricated nominal value.
    pub temperature_c: Option<f64>,
    pub warning_temperature_c: f64,
    pub shutdown_temperature_c: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PowerHealthSample {
    pub bus_voltage_v: f64,
    pub minimum_bus_voltage_v: f64,
    pub nominal_bus_voltage_v: f64,
    pub pack_current_a: f64,
    pub pack_current_limit_a: f64,
    pub state_of_charge: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidPhysicalHealthFrame {
    pub morphology: HumanoidMorphology,
    pub sequence: u64,
    pub sampled_at_s: f64,
    pub received_at_s: f64,
    pub calibration_fingerprint: u64,
    pub actuators: Vec<ActuatorHealthSample>,
    pub power: PowerHealthSample,
    /// Critical diagnostics may latch this independently of the scalar margins.
    pub latched_fault: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PhysicalHealthAuthorityConfig {
    /// Age below which freshness does not derate authority.
    pub full_freshness_age_s: f64,
    /// Age at or above which physical authority is zero.
    pub maximum_frame_age_s: f64,
    /// Current-utilization fraction where progressive derating begins.
    pub current_derate_start_fraction: f64,
    /// Authority ceiling when actuator temperature feedback is unavailable.
    pub missing_temperature_authority: f32,
    /// State of charge at or below which physical authority is zero.
    pub minimum_state_of_charge: f64,
    /// State of charge at or above which SoC does not derate authority.
    pub full_state_of_charge: f64,
}

impl Default for PhysicalHealthAuthorityConfig {
    fn default() -> Self {
        // Engineering defaults only. A qualified hardware profile must replace
        // these with limits derived from the actual drives, pack, cooling, and
        // fault-tolerant time interval.
        Self {
            full_freshness_age_s: 0.02,
            maximum_frame_age_s: 0.10,
            current_derate_start_fraction: 0.80,
            missing_temperature_authority: 0.35,
            minimum_state_of_charge: 0.10,
            full_state_of_charge: 0.30,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalHealthError {
    MorphologyMismatch,
    NonFiniteTimestamp,
    TimestampRegression,
    InvalidCalibration,
    InvalidActuatorEvidence,
    InvalidPowerEvidence,
    InvalidConfig,
}

impl std::fmt::Display for PhysicalHealthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::MorphologyMismatch => "physical-health actuator count does not match morphology",
            Self::NonFiniteTimestamp => "physical-health timestamp is non-finite",
            Self::TimestampRegression => "physical-health timestamp ordering is invalid",
            Self::InvalidCalibration => "physical-health calibration fingerprint is invalid",
            Self::InvalidActuatorEvidence => "physical-health actuator evidence is invalid",
            Self::InvalidPowerEvidence => "physical-health power evidence is invalid",
            Self::InvalidConfig => "physical-health authority configuration is invalid",
        };
        f.write_str(message)
    }
}

impl std::error::Error for PhysicalHealthError {}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HumanoidPhysicalHealthEnvelope {
    pub sequence: u64,
    pub frame_age_s: f64,
    pub freshness_authority: f32,
    pub actuator_authority: f32,
    pub power_authority: f32,
    pub latched_fault: bool,
}

impl HumanoidPhysicalHealthEnvelope {
    pub fn physical_authority(self) -> f32 {
        if self.latched_fault {
            return 0.0;
        }
        [
            self.freshness_authority,
            self.actuator_authority,
            self.power_authority,
        ]
        .into_iter()
        .map(restrictive_unit_interval)
        .fold(1.0, f32::min)
    }

    /// Tighten the physical component of an existing authority envelope without
    /// overwriting a pre-existing stricter limit.
    pub fn restrict_authority(
        self,
        mut authority: HumanoidAuthorityEnvelope,
    ) -> HumanoidAuthorityEnvelope {
        let limit = self.physical_authority();
        authority.physical = restrictive_unit_interval(authority.physical).min(limit);
        authority
    }
}

impl HumanoidPhysicalHealthFrame {
    pub fn evaluate(
        &self,
        expected_morphology: HumanoidMorphology,
        now_s: f64,
        config: PhysicalHealthAuthorityConfig,
    ) -> Result<HumanoidPhysicalHealthEnvelope, PhysicalHealthError> {
        validate_config(config)?;
        if self.morphology != expected_morphology
            || self.actuators.len() != expected_morphology.num_actuators()
        {
            return Err(PhysicalHealthError::MorphologyMismatch);
        }
        if !self.sampled_at_s.is_finite()
            || !self.received_at_s.is_finite()
            || !now_s.is_finite()
        {
            return Err(PhysicalHealthError::NonFiniteTimestamp);
        }
        if self.received_at_s < self.sampled_at_s || now_s < self.received_at_s {
            return Err(PhysicalHealthError::TimestampRegression);
        }
        if self.calibration_fingerprint == 0 {
            return Err(PhysicalHealthError::InvalidCalibration);
        }

        let frame_age_s = now_s - self.sampled_at_s;
        let freshness_authority = descending_margin(
            frame_age_s,
            config.full_freshness_age_s,
            config.maximum_frame_age_s,
        ) as f32;

        let mut actuator_authority = 1.0f32;
        for actuator in &self.actuators {
            if !actuator.enabled || !actuator.feedback_valid {
                actuator_authority = 0.0;
                break;
            }
            if !actuator.current_a.is_finite()
                || !actuator.current_limit_a.is_finite()
                || actuator.current_limit_a <= 0.0
                || !actuator.warning_temperature_c.is_finite()
                || !actuator.shutdown_temperature_c.is_finite()
                || actuator.warning_temperature_c >= actuator.shutdown_temperature_c
                || actuator
                    .temperature_c
                    .is_some_and(|temperature| !temperature.is_finite())
            {
                return Err(PhysicalHealthError::InvalidActuatorEvidence);
            }

            let current_utilization = actuator.current_a.abs() / actuator.current_limit_a;
            let current_authority = descending_margin(
                current_utilization,
                config.current_derate_start_fraction,
                1.0,
            ) as f32;
            let thermal_authority = match actuator.temperature_c {
                Some(temperature) => descending_margin(
                    temperature,
                    actuator.warning_temperature_c,
                    actuator.shutdown_temperature_c,
                ) as f32,
                None => restrictive_unit_interval(config.missing_temperature_authority),
            };
            actuator_authority = actuator_authority
                .min(current_authority)
                .min(thermal_authority);
        }

        let power = self.power;
        if !power.bus_voltage_v.is_finite()
            || !power.minimum_bus_voltage_v.is_finite()
            || !power.nominal_bus_voltage_v.is_finite()
            || power.minimum_bus_voltage_v <= 0.0
            || power.minimum_bus_voltage_v >= power.nominal_bus_voltage_v
            || !power.pack_current_a.is_finite()
            || !power.pack_current_limit_a.is_finite()
            || power.pack_current_limit_a <= 0.0
            || !power.state_of_charge.is_finite()
            || !(0.0..=1.0).contains(&power.state_of_charge)
        {
            return Err(PhysicalHealthError::InvalidPowerEvidence);
        }

        let voltage_authority = ascending_margin(
            power.bus_voltage_v,
            power.minimum_bus_voltage_v,
            power.nominal_bus_voltage_v,
        ) as f32;
        let pack_current_authority = descending_margin(
            power.pack_current_a.abs() / power.pack_current_limit_a,
            config.current_derate_start_fraction,
            1.0,
        ) as f32;
        let soc_authority = ascending_margin(
            power.state_of_charge,
            config.minimum_state_of_charge,
            config.full_state_of_charge,
        ) as f32;
        let power_authority = voltage_authority
            .min(pack_current_authority)
            .min(soc_authority);

        Ok(HumanoidPhysicalHealthEnvelope {
            sequence: self.sequence,
            frame_age_s,
            freshness_authority,
            actuator_authority,
            power_authority,
            latched_fault: self.latched_fault,
        })
    }
}

fn validate_config(config: PhysicalHealthAuthorityConfig) -> Result<(), PhysicalHealthError> {
    if !config.full_freshness_age_s.is_finite()
        || !config.maximum_frame_age_s.is_finite()
        || config.full_freshness_age_s < 0.0
        || config.full_freshness_age_s >= config.maximum_frame_age_s
        || !config.current_derate_start_fraction.is_finite()
        || !(0.0..1.0).contains(&config.current_derate_start_fraction)
        || !config.missing_temperature_authority.is_finite()
        || !(0.0..=1.0).contains(&config.missing_temperature_authority)
        || !config.minimum_state_of_charge.is_finite()
        || !config.full_state_of_charge.is_finite()
        || !(0.0..1.0).contains(&config.minimum_state_of_charge)
        || !(0.0..=1.0).contains(&config.full_state_of_charge)
        || config.minimum_state_of_charge >= config.full_state_of_charge
    {
        return Err(PhysicalHealthError::InvalidConfig);
    }
    Ok(())
}

fn restrictive_unit_interval(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn descending_margin(value: f64, full_until: f64, zero_at: f64) -> f64 {
    if !value.is_finite()
        || !full_until.is_finite()
        || !zero_at.is_finite()
        || full_until >= zero_at
    {
        return 0.0;
    }
    if value <= full_until {
        1.0
    } else if value >= zero_at {
        0.0
    } else {
        1.0 - (value - full_until) / (zero_at - full_until)
    }
}

fn ascending_margin(value: f64, zero_until: f64, full_at: f64) -> f64 {
    if !value.is_finite()
        || !zero_until.is_finite()
        || !full_at.is_finite()
        || zero_until >= full_at
    {
        return 0.0;
    }
    if value <= zero_until {
        0.0
    } else if value >= full_at {
        1.0
    } else {
        (value - zero_until) / (full_at - zero_until)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nominal_frame() -> HumanoidPhysicalHealthFrame {
        let morphology = HumanoidMorphology::Dmc21;
        HumanoidPhysicalHealthFrame {
            morphology,
            sequence: 7,
            sampled_at_s: 1.0,
            received_at_s: 1.0,
            calibration_fingerprint: 42,
            actuators: vec![
                ActuatorHealthSample {
                    enabled: true,
                    feedback_valid: true,
                    current_a: 0.2,
                    current_limit_a: 2.0,
                    temperature_c: Some(40.0),
                    warning_temperature_c: 70.0,
                    shutdown_temperature_c: 90.0,
                };
                morphology.num_actuators()
            ],
            power: PowerHealthSample {
                bus_voltage_v: 48.0,
                minimum_bus_voltage_v: 40.0,
                nominal_bus_voltage_v: 48.0,
                pack_current_a: 5.0,
                pack_current_limit_a: 40.0,
                state_of_charge: 0.8,
            },
            latched_fault: false,
        }
    }

    #[test]
    fn nominal_health_fully_admits_physical_authority() {
        let frame = nominal_frame();
        let envelope = frame
            .evaluate(frame.morphology, 1.0, PhysicalHealthAuthorityConfig::default())
            .unwrap();
        assert_eq!(envelope.physical_authority(), 1.0);
    }

    #[test]
    fn one_hot_actuator_derates_whole_body_authority() {
        let mut frame = nominal_frame();
        frame.actuators[3].temperature_c = Some(80.0);
        let envelope = frame
            .evaluate(frame.morphology, 1.0, PhysicalHealthAuthorityConfig::default())
            .unwrap();
        assert!((envelope.physical_authority() - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn missing_temperature_is_explicitly_conservative() {
        let mut frame = nominal_frame();
        frame.actuators[0].temperature_c = None;
        let envelope = frame
            .evaluate(frame.morphology, 1.0, PhysicalHealthAuthorityConfig::default())
            .unwrap();
        assert!((envelope.physical_authority() - 0.35).abs() < 1.0e-6);
    }

    #[test]
    fn stale_or_latched_health_revokes_physical_authority() {
        let frame = nominal_frame();
        let stale = frame
            .evaluate(frame.morphology, 1.2, PhysicalHealthAuthorityConfig::default())
            .unwrap();
        assert_eq!(stale.physical_authority(), 0.0);

        let mut faulted = nominal_frame();
        faulted.latched_fault = true;
        let envelope = faulted
            .evaluate(faulted.morphology, 1.0, PhysicalHealthAuthorityConfig::default())
            .unwrap();
        assert_eq!(envelope.physical_authority(), 0.0);
    }

    #[test]
    fn physical_health_only_restricts_existing_authority() {
        let mut frame = nominal_frame();
        frame.actuators[0].temperature_c = Some(80.0);
        let envelope = frame
            .evaluate(frame.morphology, 1.0, PhysicalHealthAuthorityConfig::default())
            .unwrap();
        let authority = envelope.restrict_authority(HumanoidAuthorityEnvelope {
            physical: 0.2,
            ..HumanoidAuthorityEnvelope::fully_admitted()
        });
        assert!((authority.physical - 0.2).abs() < 1.0e-6);
    }
}
