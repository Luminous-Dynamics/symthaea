// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Glove-specific lunar-dust durability submodel for SX-018.
//!
//! Generic suit `Joint`/`OuterTextile` dust state is intentionally not reused
//! as glove state. The hand has distinct palm, finger-joint, tendon-path, and
//! wrist/pressure-seal contamination paths. Coefficients are simulation-only
//! placeholders until glove-specific regolith, bend-cycle, abrasion, seal, and
//! EDS measurements replace them.

use serde::{Deserialize, Serialize};

use crate::dust::EdsZoneConfig;
use crate::powered_glove::{PoweredGloveState, NUM_GLOVE_DIGITS};
use crate::space_exosuit::ExosuitEvidenceLevel;

pub const GLOVE_DUST_ZONE_COUNT: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(usize)]
pub enum GloveDustZone {
    Palm = 0,
    FingerJoint = 1,
    TendonPath = 2,
    WristSeal = 3,
}

impl GloveDustZone {
    pub const ALL: [Self; GLOVE_DUST_ZONE_COUNT] = [
        Self::Palm,
        Self::FingerJoint,
        Self::TendonPath,
        Self::WristSeal,
    ];

    pub const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GloveDustCoefficients {
    /// Fraction of presented dust retained before active mitigation [0,1].
    pub retention_fraction: f64,
    /// Palm/dexterity wear accumulated per abrasive cycle.
    pub wear_per_cycle: f64,
    /// Sensor-confidence loss per retained g/m^2.
    pub sensor_loss_per_g_m2: f64,
    /// Powered tendon/drive health loss per retained g/m^2.
    pub drive_loss_per_g_m2: f64,
    /// Seal-risk proxy increase per retained g/m^2.
    pub seal_risk_per_g_m2: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl GloveDustCoefficients {
    pub const fn zero() -> Self {
        Self {
            retention_fraction: 0.0,
            wear_per_cycle: 0.0,
            sensor_loss_per_g_m2: 0.0,
            drive_loss_per_g_m2: 0.0,
            seal_risk_per_g_m2: 0.0,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.retention_fraction.is_finite()
            && (0.0..=1.0).contains(&self.retention_fraction)
            && [
                self.wear_per_cycle,
                self.sensor_loss_per_g_m2,
                self.drive_loss_per_g_m2,
                self.seal_risk_per_g_m2,
            ]
            .into_iter()
            .all(|v| v.is_finite() && v >= 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GloveDustConfig {
    pub coefficients: [GloveDustCoefficients; GLOVE_DUST_ZONE_COUNT],
    pub eds: [EdsZoneConfig; GLOVE_DUST_ZONE_COUNT],
    pub evidence: ExosuitEvidenceLevel,
}

impl GloveDustConfig {
    pub fn simulation_reference() -> Self {
        let mut coefficients = [GloveDustCoefficients::zero(); GLOVE_DUST_ZONE_COUNT];
        coefficients[GloveDustZone::Palm.index()] = GloveDustCoefficients {
            retention_fraction: 0.55,
            wear_per_cycle: 0.0004,
            sensor_loss_per_g_m2: 0.012,
            drive_loss_per_g_m2: 0.0,
            seal_risk_per_g_m2: 0.0,
            evidence: ExosuitEvidenceLevel::Simulation,
        };
        coefficients[GloveDustZone::FingerJoint.index()] = GloveDustCoefficients {
            retention_fraction: 0.60,
            wear_per_cycle: 0.0005,
            sensor_loss_per_g_m2: 0.008,
            drive_loss_per_g_m2: 0.006,
            seal_risk_per_g_m2: 0.0,
            evidence: ExosuitEvidenceLevel::Simulation,
        };
        coefficients[GloveDustZone::TendonPath.index()] = GloveDustCoefficients {
            retention_fraction: 0.35,
            wear_per_cycle: 0.0002,
            sensor_loss_per_g_m2: 0.004,
            drive_loss_per_g_m2: 0.015,
            seal_risk_per_g_m2: 0.0,
            evidence: ExosuitEvidenceLevel::Simulation,
        };
        coefficients[GloveDustZone::WristSeal.index()] = GloveDustCoefficients {
            retention_fraction: 0.40,
            wear_per_cycle: 0.0002,
            sensor_loss_per_g_m2: 0.0,
            drive_loss_per_g_m2: 0.0,
            seal_risk_per_g_m2: 0.020,
            evidence: ExosuitEvidenceLevel::Simulation,
        };

        // EDS is enabled only on palm/finger surfaces in the simulation
        // reference. This is a trade-study assumption, not a hardware claim.
        let mut eds = [EdsZoneConfig::disabled(); GLOVE_DUST_ZONE_COUNT];
        eds[GloveDustZone::Palm.index()] = EdsZoneConfig::simulation_reference();
        eds[GloveDustZone::FingerJoint.index()] = EdsZoneConfig::simulation_reference();

        Self {
            coefficients,
            eds,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.coefficients.iter().all(GloveDustCoefficients::is_valid)
            && self.eds.iter().all(EdsZoneConfig::is_valid)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GloveDustExposure {
    pub deposition_g_m2: [f64; GLOVE_DUST_ZONE_COUNT],
    pub abrasive_cycles: [f64; GLOVE_DUST_ZONE_COUNT],
    pub duration_s: f64,
}

impl GloveDustExposure {
    pub fn uniform(deposition_g_m2: f64, abrasive_cycles: f64, duration_s: f64) -> Self {
        Self {
            deposition_g_m2: [deposition_g_m2; GLOVE_DUST_ZONE_COUNT],
            abrasive_cycles: [abrasive_cycles; GLOVE_DUST_ZONE_COUNT],
            duration_s,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.duration_s.is_finite()
            && self.duration_s > 0.0
            && self
                .deposition_g_m2
                .iter()
                .chain(self.abrasive_cycles.iter())
                .all(|v| v.is_finite() && *v >= 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GloveDustZoneState {
    pub retained_dust_g_m2: f64,
    pub accumulated_wear: f64,
}

impl Default for GloveDustZoneState {
    fn default() -> Self {
        Self {
            retained_dust_g_m2: 0.0,
            accumulated_wear: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GloveDustStepReport {
    pub retained_dust_g_m2: [f64; GLOVE_DUST_ZONE_COUNT],
    pub palm_abrasion_health: f64,
    pub sensor_confidence_multiplier: f64,
    pub drive_health_multiplier: f64,
    pub wrist_seal_risk: f64,
    pub eds_power_w: f64,
    pub eds_energy_wh: f64,
    pub evidence: ExosuitEvidenceLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GloveDustError {
    InvalidConfig,
    InvalidExposure,
}

#[derive(Debug, Clone)]
pub struct GloveDustTwin {
    config: GloveDustConfig,
    state: [GloveDustZoneState; GLOVE_DUST_ZONE_COUNT],
}

impl GloveDustTwin {
    pub fn new(config: GloveDustConfig) -> Result<Self, GloveDustError> {
        if !config.is_valid() {
            return Err(GloveDustError::InvalidConfig);
        }
        Ok(Self {
            config,
            state: [GloveDustZoneState::default(); GLOVE_DUST_ZONE_COUNT],
        })
    }

    pub fn simulation_reference() -> Self {
        Self::new(GloveDustConfig::simulation_reference())
            .expect("reference glove dust model must be valid")
    }

    pub fn state(&self, zone: GloveDustZone) -> &GloveDustZoneState {
        &self.state[zone.index()]
    }

    pub fn step(&mut self, exposure: GloveDustExposure) -> Result<GloveDustStepReport, GloveDustError> {
        if !self.config.is_valid() {
            return Err(GloveDustError::InvalidConfig);
        }
        if !exposure.is_valid() {
            return Err(GloveDustError::InvalidExposure);
        }

        let mut eds_power_w = 0.0;
        for zone in GloveDustZone::ALL {
            let i = zone.index();
            let c = self.config.coefficients[i];
            let eds = self.config.eds[i];
            let removal = if eds.available && eds.enabled {
                eds_power_w += eds.power_w;
                eds.removal_fraction
            } else {
                0.0
            };
            let retained_new = exposure.deposition_g_m2[i] * c.retention_fraction * (1.0 - removal);
            self.state[i].retained_dust_g_m2 += retained_new;
            self.state[i].accumulated_wear += c.wear_per_cycle * exposure.abrasive_cycles[i];
        }

        let palm = self.state[GloveDustZone::Palm.index()];
        let finger = self.state[GloveDustZone::FingerJoint.index()];
        let tendon = self.state[GloveDustZone::TendonPath.index()];
        let seal = self.state[GloveDustZone::WristSeal.index()];

        let palm_abrasion_health = (1.0 - palm.accumulated_wear).clamp(0.0, 1.0);
        let sensor_loss = self.config.coefficients[GloveDustZone::Palm.index()].sensor_loss_per_g_m2
            * palm.retained_dust_g_m2
            + self.config.coefficients[GloveDustZone::FingerJoint.index()].sensor_loss_per_g_m2
                * finger.retained_dust_g_m2
            + self.config.coefficients[GloveDustZone::TendonPath.index()].sensor_loss_per_g_m2
                * tendon.retained_dust_g_m2;
        let drive_loss = self.config.coefficients[GloveDustZone::FingerJoint.index()].drive_loss_per_g_m2
            * finger.retained_dust_g_m2
            + self.config.coefficients[GloveDustZone::TendonPath.index()].drive_loss_per_g_m2
                * tendon.retained_dust_g_m2;
        let wrist_seal_risk = (self.config.coefficients[GloveDustZone::WristSeal.index()]
            .seal_risk_per_g_m2
            * seal.retained_dust_g_m2)
            .clamp(0.0, 1.0);

        Ok(GloveDustStepReport {
            retained_dust_g_m2: [
                palm.retained_dust_g_m2,
                finger.retained_dust_g_m2,
                tendon.retained_dust_g_m2,
                seal.retained_dust_g_m2,
            ],
            palm_abrasion_health,
            sensor_confidence_multiplier: (1.0 - sensor_loss).clamp(0.0, 1.0),
            drive_health_multiplier: (1.0 - drive_loss).clamp(0.0, 1.0),
            wrist_seal_risk,
            eds_power_w,
            eds_energy_wh: eds_power_w * exposure.duration_s / 3600.0,
            evidence: self.config.evidence,
        })
    }
}

/// Apply cumulative dust-derived degradation to a powered-glove trade-study
/// state without overwriting a more severe independent fault already present.
pub fn apply_glove_dust_report(state: &mut PoweredGloveState, report: &GloveDustStepReport) {
    for i in 0..NUM_GLOVE_DIGITS {
        state.palm_abrasion_health[i] =
            state.palm_abrasion_health[i].min(report.palm_abrasion_health);
        state.sensor_confidence[i] =
            state.sensor_confidence[i].min(report.sensor_confidence_multiplier);
        state.drive_health[i] = state.drive_health[i].min(report.drive_health_multiplier);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GloveDustRecommendation {
    Continue,
    DegradeDexterity,
    ReturnForService,
    InvalidState,
}

pub fn glove_dust_recommendation(report: &GloveDustStepReport) -> GloveDustRecommendation {
    let values = [
        report.palm_abrasion_health,
        report.sensor_confidence_multiplier,
        report.drive_health_multiplier,
        report.wrist_seal_risk,
        report.eds_power_w,
    ];
    if values.iter().any(|v| !v.is_finite()) {
        return GloveDustRecommendation::InvalidState;
    }
    // Simulation-only operational thresholds, not human-rating limits.
    if report.wrist_seal_risk >= 0.20
        || report.palm_abrasion_health <= 0.60
        || report.drive_health_multiplier <= 0.70
    {
        GloveDustRecommendation::ReturnForService
    } else if report.palm_abrasion_health <= 0.85
        || report.sensor_confidence_multiplier <= 0.90
        || report.drive_health_multiplier <= 0.90
    {
        GloveDustRecommendation::DegradeDexterity
    } else {
        GloveDustRecommendation::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::powered_glove::PoweredGloveState;

    #[test]
    fn glove_specific_dust_degrades_hand_relevant_channels() {
        let mut twin = GloveDustTwin::simulation_reference();
        let report = twin
            .step(GloveDustExposure::uniform(4.0, 200.0, 600.0))
            .unwrap();
        assert!(report.palm_abrasion_health < 1.0);
        assert!(report.sensor_confidence_multiplier < 1.0);
        assert!(report.drive_health_multiplier < 1.0);
        assert!(report.eds_power_w > 0.0);
    }

    #[test]
    fn dust_report_can_derate_powered_glove_without_masking_worse_faults() {
        let mut twin = GloveDustTwin::simulation_reference();
        let report = twin
            .step(GloveDustExposure::uniform(5.0, 300.0, 600.0))
            .unwrap();
        let mut state = PoweredGloveState::simulation_reference();
        state.drive_health[0] = 0.50;
        apply_glove_dust_report(&mut state, &report);
        assert_eq!(state.drive_health[0], 0.50);
        assert!(state.sensor_confidence[1] < 1.0);
        assert!(state.palm_abrasion_health[2] < 1.0);
    }

    #[test]
    fn disabling_eds_increases_retained_surface_dust() {
        let exposure = GloveDustExposure::uniform(2.0, 10.0, 60.0);
        let mut mitigated = GloveDustTwin::simulation_reference();
        let a = mitigated.step(exposure).unwrap();

        let mut config = GloveDustConfig::simulation_reference();
        config.eds = [EdsZoneConfig::disabled(); GLOVE_DUST_ZONE_COUNT];
        let mut unmitigated = GloveDustTwin::new(config).unwrap();
        let b = unmitigated.step(exposure).unwrap();

        assert!(a.retained_dust_g_m2[GloveDustZone::Palm.index()]
            < b.retained_dust_g_m2[GloveDustZone::Palm.index()]);
    }
}
