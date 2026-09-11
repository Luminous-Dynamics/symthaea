// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Zoned lunar-dust degradation twin for SX-017.
//!
//! Dust is not represented as a single health percentage. The model keeps
//! separate visor, joint, seal, radiator, textile, and suitport-interface
//! states. Electrodynamic dust-shield (EDS) mitigation is modeled as an
//! optional powered service with explicit evidence, not as guaranteed removal.
//!
//! All default coefficients are simulation assumptions. Measured coefficients
//! can be substituted without changing the state/authority model.

use serde::{Deserialize, Serialize};

use crate::space_exosuit::ExosuitEvidenceLevel;

pub const DUST_ZONE_COUNT: usize = 6;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(usize)]
pub enum DustZone {
    Visor = 0,
    Joint = 1,
    Seal = 2,
    Radiator = 3,
    OuterTextile = 4,
    SuitportInterface = 5,
}

impl DustZone {
    pub const ALL: [Self; DUST_ZONE_COUNT] = [
        Self::Visor,
        Self::Joint,
        Self::Seal,
        Self::Radiator,
        Self::OuterTextile,
        Self::SuitportInterface,
    ];

    pub const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DustZoneCoefficients {
    /// Increase in friction multiplier per g/m^2 retained dust.
    pub friction_per_g_m2: f64,
    /// Increase in seal-risk proxy per g/m^2 retained dust.
    pub seal_risk_per_g_m2: f64,
    /// Fractional optical-transmission loss per g/m^2 retained dust.
    pub optical_loss_per_g_m2: f64,
    /// Fractional heat-rejection loss per g/m^2 retained dust.
    pub heat_rejection_loss_per_g_m2: f64,
    /// Fractional outer-textile optical/thermal penalty per g/m^2.
    pub textile_optical_penalty_per_g_m2: f64,
    /// Additional generalized wear per abrasive cycle.
    pub abrasion_wear_per_cycle: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl DustZoneCoefficients {
    pub const fn zero() -> Self {
        Self {
            friction_per_g_m2: 0.0,
            seal_risk_per_g_m2: 0.0,
            optical_loss_per_g_m2: 0.0,
            heat_rejection_loss_per_g_m2: 0.0,
            textile_optical_penalty_per_g_m2: 0.0,
            abrasion_wear_per_cycle: 0.0,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn is_valid(&self) -> bool {
        [
            self.friction_per_g_m2,
            self.seal_risk_per_g_m2,
            self.optical_loss_per_g_m2,
            self.heat_rejection_loss_per_g_m2,
            self.textile_optical_penalty_per_g_m2,
            self.abrasion_wear_per_cycle,
        ]
        .into_iter()
        .all(|value| value.is_finite() && value >= 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EdsZoneConfig {
    pub available: bool,
    pub enabled: bool,
    /// Fraction of new deposition removed during this simulation step [0,1].
    pub removal_fraction: f64,
    /// Electrical power while active.
    pub power_w: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl EdsZoneConfig {
    pub const fn disabled() -> Self {
        Self {
            available: false,
            enabled: false,
            removal_fraction: 0.0,
            power_w: 0.0,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn simulation_reference() -> Self {
        Self {
            available: true,
            enabled: true,
            removal_fraction: 0.70,
            power_w: 4.0,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.removal_fraction.is_finite()
            && (0.0..=1.0).contains(&self.removal_fraction)
            && self.power_w.is_finite()
            && self.power_w >= 0.0
            && (!self.enabled || self.available)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DustModelConfig {
    pub coefficients: [DustZoneCoefficients; DUST_ZONE_COUNT],
    pub eds: [EdsZoneConfig; DUST_ZONE_COUNT],
    /// Fraction of retained suitport-interface load transferred inward on a
    /// docking/ingress cycle before decontamination [0,1].
    pub suitport_transfer_fraction: f64,
    /// Fraction of suitport-interface load removed by the declared external
    /// decontamination process before transfer [0,1].
    pub suitport_decon_fraction: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl DustModelConfig {
    pub fn simulation_reference() -> Self {
        let mut coefficients = [DustZoneCoefficients::zero(); DUST_ZONE_COUNT];

        coefficients[DustZone::Visor.index()].optical_loss_per_g_m2 = 0.030;
        coefficients[DustZone::Joint.index()].friction_per_g_m2 = 0.040;
        coefficients[DustZone::Joint.index()].abrasion_wear_per_cycle = 0.0005;
        coefficients[DustZone::Seal.index()].seal_risk_per_g_m2 = 0.025;
        coefficients[DustZone::Seal.index()].abrasion_wear_per_cycle = 0.0008;
        coefficients[DustZone::Radiator.index()].heat_rejection_loss_per_g_m2 = 0.025;
        coefficients[DustZone::OuterTextile.index()].textile_optical_penalty_per_g_m2 = 0.015;
        coefficients[DustZone::OuterTextile.index()].abrasion_wear_per_cycle = 0.0003;
        coefficients[DustZone::SuitportInterface.index()].seal_risk_per_g_m2 = 0.010;

        let mut eds = [EdsZoneConfig::disabled(); DUST_ZONE_COUNT];
        eds[DustZone::Visor.index()] = EdsZoneConfig::simulation_reference();
        eds[DustZone::Joint.index()] = EdsZoneConfig::simulation_reference();
        eds[DustZone::Seal.index()] = EdsZoneConfig::simulation_reference();
        eds[DustZone::Radiator.index()] = EdsZoneConfig::simulation_reference();
        eds[DustZone::SuitportInterface.index()] = EdsZoneConfig::simulation_reference();

        Self {
            coefficients,
            eds,
            suitport_transfer_fraction: 0.05,
            suitport_decon_fraction: 0.80,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.coefficients.iter().all(DustZoneCoefficients::is_valid)
            && self.eds.iter().all(EdsZoneConfig::is_valid)
            && self.suitport_transfer_fraction.is_finite()
            && (0.0..=1.0).contains(&self.suitport_transfer_fraction)
            && self.suitport_decon_fraction.is_finite()
            && (0.0..=1.0).contains(&self.suitport_decon_fraction)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DustZoneState {
    pub retained_dust_g_m2: f64,
    pub friction_multiplier: f64,
    pub seal_risk: f64,
    pub optical_transmission: f64,
    pub heat_rejection_fraction: f64,
    pub textile_optical_factor: f64,
    pub accumulated_wear: f64,
}

impl Default for DustZoneState {
    fn default() -> Self {
        Self {
            retained_dust_g_m2: 0.0,
            friction_multiplier: 1.0,
            seal_risk: 0.0,
            optical_transmission: 1.0,
            heat_rejection_fraction: 1.0,
            textile_optical_factor: 1.0,
            accumulated_wear: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DustExposure {
    /// New external deposition presented to each declared zone during the step.
    pub deposition_g_m2: [f64; DUST_ZONE_COUNT],
    /// Mechanical abrasive cycles (joint bends, contact cycles, etc.) by zone.
    pub abrasive_cycles: [f64; DUST_ZONE_COUNT],
    pub duration_s: f64,
    /// Whether this step includes a suitport docking/ingress transfer event.
    pub suitport_cycle: bool,
}

impl DustExposure {
    pub fn uniform(deposition_g_m2: f64, duration_s: f64) -> Self {
        Self {
            deposition_g_m2: [deposition_g_m2; DUST_ZONE_COUNT],
            abrasive_cycles: [0.0; DUST_ZONE_COUNT],
            duration_s,
            suitport_cycle: false,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.duration_s.is_finite()
            && self.duration_s > 0.0
            && self
                .deposition_g_m2
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
            && self
                .abrasive_cycles
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DustStepReport {
    pub retained_new_dust_g_m2: [f64; DUST_ZONE_COUNT],
    pub eds_power_w: f64,
    pub eds_energy_wh: f64,
    pub transferred_inside_g_m2: f64,
    pub visor_transmission: f64,
    pub joint_friction_multiplier: f64,
    pub seal_risk: f64,
    pub radiator_heat_rejection_fraction: f64,
    pub textile_optical_factor: f64,
    pub evidence: ExosuitEvidenceLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DustModelError {
    InvalidConfig,
    InvalidExposure,
}

#[derive(Debug, Clone)]
pub struct DustDegradationTwin {
    config: DustModelConfig,
    state: [DustZoneState; DUST_ZONE_COUNT],
}

impl DustDegradationTwin {
    pub fn new(config: DustModelConfig) -> Result<Self, DustModelError> {
        if !config.is_valid() {
            return Err(DustModelError::InvalidConfig);
        }
        Ok(Self {
            config,
            state: [DustZoneState::default(); DUST_ZONE_COUNT],
        })
    }

    pub fn simulation_reference() -> Self {
        Self::new(DustModelConfig::simulation_reference())
            .expect("reference dust model must be valid")
    }

    pub fn state(&self, zone: DustZone) -> &DustZoneState {
        &self.state[zone.index()]
    }

    pub fn config_mut_for_trade_study(&mut self) -> &mut DustModelConfig {
        &mut self.config
    }

    pub fn step(&mut self, exposure: DustExposure) -> Result<DustStepReport, DustModelError> {
        if !self.config.is_valid() {
            return Err(DustModelError::InvalidConfig);
        }
        if !exposure.is_valid() {
            return Err(DustModelError::InvalidExposure);
        }

        let mut retained_new = [0.0; DUST_ZONE_COUNT];
        let mut eds_power_w = 0.0;

        for zone in DustZone::ALL {
            let index = zone.index();
            let eds = self.config.eds[index];
            let removal = if eds.available && eds.enabled {
                eds_power_w += eds.power_w;
                eds.removal_fraction
            } else {
                0.0
            };
            let retained = exposure.deposition_g_m2[index] * (1.0 - removal);
            retained_new[index] = retained;

            let state = &mut self.state[index];
            state.retained_dust_g_m2 += retained;

            let coefficients = self.config.coefficients[index];
            state.friction_multiplier =
                1.0 + coefficients.friction_per_g_m2 * state.retained_dust_g_m2;
            state.seal_risk =
                (coefficients.seal_risk_per_g_m2 * state.retained_dust_g_m2).clamp(0.0, 1.0);
            state.optical_transmission =
                (1.0 - coefficients.optical_loss_per_g_m2 * state.retained_dust_g_m2)
                    .clamp(0.0, 1.0);
            state.heat_rejection_fraction =
                (1.0 - coefficients.heat_rejection_loss_per_g_m2 * state.retained_dust_g_m2)
                    .clamp(0.0, 1.0);
            state.textile_optical_factor =
                (1.0 - coefficients.textile_optical_penalty_per_g_m2 * state.retained_dust_g_m2)
                    .clamp(0.0, 1.0);
            state.accumulated_wear +=
                coefficients.abrasion_wear_per_cycle * exposure.abrasive_cycles[index];
        }

        let transferred_inside_g_m2 = if exposure.suitport_cycle {
            let interface = &mut self.state[DustZone::SuitportInterface.index()];
            let after_decon =
                interface.retained_dust_g_m2 * (1.0 - self.config.suitport_decon_fraction);
            let transferred = after_decon * self.config.suitport_transfer_fraction;
            interface.retained_dust_g_m2 = (interface.retained_dust_g_m2 - transferred).max(0.0);
            transferred
        } else {
            0.0
        };

        Ok(DustStepReport {
            retained_new_dust_g_m2: retained_new,
            eds_power_w,
            eds_energy_wh: eds_power_w * exposure.duration_s / 3600.0,
            transferred_inside_g_m2,
            visor_transmission: self.state[DustZone::Visor.index()].optical_transmission,
            joint_friction_multiplier: self.state[DustZone::Joint.index()].friction_multiplier,
            seal_risk: self.state[DustZone::Seal.index()].seal_risk,
            radiator_heat_rejection_fraction: self.state[DustZone::Radiator.index()]
                .heat_rejection_fraction,
            textile_optical_factor: self.state[DustZone::OuterTextile.index()]
                .textile_optical_factor,
            evidence: self.config.evidence,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eds_reduces_retained_deposition_when_enabled() {
        let exposure = DustExposure::uniform(1.0, 60.0);

        let mut mitigated = DustDegradationTwin::simulation_reference();
        let mitigated_report = mitigated.step(exposure).unwrap();

        let mut config = DustModelConfig::simulation_reference();
        config.eds = [EdsZoneConfig::disabled(); DUST_ZONE_COUNT];
        let mut unmitigated = DustDegradationTwin::new(config).unwrap();
        let unmitigated_report = unmitigated.step(exposure).unwrap();

        assert!(
            mitigated_report.retained_new_dust_g_m2[DustZone::Visor.index()]
                < unmitigated_report.retained_new_dust_g_m2[DustZone::Visor.index()]
        );
        assert!(mitigated_report.eds_power_w > 0.0);
    }

    #[test]
    fn dust_degrades_distinct_functions_by_zone() {
        let mut twin = DustDegradationTwin::simulation_reference();
        let report = twin.step(DustExposure::uniform(2.0, 60.0)).unwrap();
        assert!(report.visor_transmission < 1.0);
        assert!(report.joint_friction_multiplier > 1.0);
        assert!(report.radiator_heat_rejection_fraction < 1.0);
        assert!(report.textile_optical_factor < 1.0);
    }

    #[test]
    fn suitport_cycle_reports_contamination_transfer_after_decon() {
        let mut twin = DustDegradationTwin::simulation_reference();
        let mut exposure = DustExposure::uniform(2.0, 60.0);
        exposure.suitport_cycle = true;
        let report = twin.step(exposure).unwrap();
        assert!(report.transferred_inside_g_m2 >= 0.0);
        assert!(report.transferred_inside_g_m2 < 2.0);
    }

    #[test]
    fn invalid_eds_claim_is_rejected() {
        let mut config = DustModelConfig::simulation_reference();
        config.eds[DustZone::Visor.index()].removal_fraction = 1.2;
        assert_eq!(
            DustDegradationTwin::new(config).unwrap_err(),
            DustModelError::InvalidConfig
        );
    }

    #[test]
    fn default_coefficients_are_explicitly_simulation_evidence() {
        let config = DustModelConfig::simulation_reference();
        assert!(config
            .coefficients
            .iter()
            .all(|entry| entry.evidence == ExosuitEvidenceLevel::Simulation));
    }
}
