// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pressure-boundary damage, leak estimation, localization, and patch research twin (SX-021).
//!
//! This module is simulation/research software only. It observes and predicts
//! pressure loss; it never commands oxygen regulators, suit valves, purge,
//! pressure relief, or any other survival-critical hardware. Numerical
//! coefficients and thresholds remain explicit simulation inputs until replaced
//! by declared hardware/human-rating evidence.

use serde::{Deserialize, Serialize};

use crate::space_exosuit::ExosuitEvidenceLevel;

pub const PRESSURE_ZONE_COUNT: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(usize)]
pub enum PressureZone {
    Torso = 0,
    Helmet = 1,
    LeftArm = 2,
    RightArm = 3,
    LeftLeg = 4,
    RightLeg = 5,
    LeftGlove = 6,
    RightGlove = 7,
    PlssInterface = 8,
    SuitportInterface = 9,
}

impl PressureZone {
    pub const ALL: [Self; PRESSURE_ZONE_COUNT] = [
        Self::Torso,
        Self::Helmet,
        Self::LeftArm,
        Self::RightArm,
        Self::LeftLeg,
        Self::RightLeg,
        Self::LeftGlove,
        Self::RightGlove,
        Self::PlssInterface,
        Self::SuitportInterface,
    ];

    pub const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PressureDamageSource {
    MmodImpact,
    SecondaryRegolithEjecta,
    AbrasionPuncture,
    CutOrTear,
    SealLeak,
    JointLeak,
    ConnectorDamage,
    BladderDamage,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatchState {
    None,
    TemporaryApplied,
    TemporaryVerified,
    PermanentRepair,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PressureDamageSite {
    pub zone: PressureZone,
    pub source: PressureDamageSource,
    /// Effective leak-orifice area used by the gas-loss model, m^2.
    pub effective_orifice_area_m2: f64,
    /// Fractional area growth rate, 1/s. Zero means no modeled propagation.
    pub fractional_growth_rate_s: f64,
    /// Multiplier applied to leak area after a patch. 1.0 means no benefit;
    /// 0.0 means idealized complete sealing. This is evidence-bearing input,
    /// not a claim about any patch technology.
    pub patch_residual_fraction: f64,
    pub patch_state: PatchState,
    pub evidence: ExosuitEvidenceLevel,
}

impl PressureDamageSite {
    pub fn is_valid(&self) -> bool {
        self.effective_orifice_area_m2.is_finite()
            && self.effective_orifice_area_m2 >= 0.0
            && self.fractional_growth_rate_s.is_finite()
            && self.fractional_growth_rate_s >= 0.0
            && self.patch_residual_fraction.is_finite()
            && (0.0..=1.0).contains(&self.patch_residual_fraction)
    }

    pub fn effective_area_after_patch_m2(&self) -> f64 {
        let residual = match self.patch_state {
            PatchState::None => 1.0,
            PatchState::TemporaryApplied
            | PatchState::TemporaryVerified
            | PatchState::PermanentRepair => self.patch_residual_fraction,
        };
        self.effective_orifice_area_m2 * residual
    }

    pub fn propagate(&mut self, dt_s: f64) -> bool {
        if !self.is_valid() || !dt_s.is_finite() || dt_s <= 0.0 {
            return false;
        }
        self.effective_orifice_area_m2 *= 1.0 + self.fractional_growth_rate_s * dt_s;
        self.effective_orifice_area_m2.is_finite()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PressureSensorReading {
    pub pressure_pa: f64,
    pub confidence: f64,
    pub age_s: f64,
}

impl PressureSensorReading {
    pub fn is_valid(&self) -> bool {
        self.pressure_pa.is_finite()
            && self.pressure_pa >= 0.0
            && self.confidence.is_finite()
            && (0.0..=1.0).contains(&self.confidence)
            && self.age_s.is_finite()
            && self.age_s >= 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LeakLocalizationReading {
    pub zone: PressureZone,
    /// Generic local leak/anomaly score [0,1]. The sensor modality belongs to
    /// the declared test protocol (acoustic, distributed pressure, flow, etc.).
    pub anomaly_score: f64,
    pub confidence: f64,
    pub age_s: f64,
}

impl LeakLocalizationReading {
    pub fn is_valid(&self) -> bool {
        self.anomaly_score.is_finite()
            && (0.0..=1.0).contains(&self.anomaly_score)
            && self.confidence.is_finite()
            && (0.0..=1.0).contains(&self.confidence)
            && self.age_s.is_finite()
            && self.age_s >= 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PressureIntegrityConfig {
    /// Effective gas volume of the pressurized suit, m^3.
    pub free_volume_m3: f64,
    /// Specific gas constant used by the research leak model, J/(kg K).
    pub specific_gas_constant_j_kg_k: f64,
    /// Heat-capacity ratio for the research leak model.
    pub gamma: f64,
    /// Discharge coefficient multiplying idealized orifice mass flow.
    pub discharge_coefficient: f64,
    pub warning_pressure_pa: f64,
    pub critical_pressure_pa: f64,
    /// Pressure below which time-to-critical is reported as zero.
    pub minimum_model_pressure_pa: f64,
    pub max_sensor_age_s: f64,
    pub min_sensor_confidence: f64,
    /// Maximum relative disagreement among pressure sensors before the estimate
    /// is considered contradictory.
    pub max_relative_pressure_disagreement: f64,
    pub min_localization_confidence: f64,
    pub min_localization_anomaly_score: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl PressureIntegrityConfig {
    pub fn simulation_reference() -> Self {
        Self {
            free_volume_m3: 0.12,
            specific_gas_constant_j_kg_k: 287.0,
            gamma: 1.4,
            discharge_coefficient: 0.70,
            warning_pressure_pa: 24_000.0,
            critical_pressure_pa: 18_000.0,
            minimum_model_pressure_pa: 1_000.0,
            max_sensor_age_s: 2.0,
            min_sensor_confidence: 0.95,
            max_relative_pressure_disagreement: 0.03,
            min_localization_confidence: 0.80,
            min_localization_anomaly_score: 0.50,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.free_volume_m3.is_finite()
            && self.free_volume_m3 > 0.0
            && self.specific_gas_constant_j_kg_k.is_finite()
            && self.specific_gas_constant_j_kg_k > 0.0
            && self.gamma.is_finite()
            && self.gamma > 1.0
            && self.discharge_coefficient.is_finite()
            && (0.0..=1.0).contains(&self.discharge_coefficient)
            && self.warning_pressure_pa.is_finite()
            && self.critical_pressure_pa.is_finite()
            && self.minimum_model_pressure_pa.is_finite()
            && self.warning_pressure_pa > self.critical_pressure_pa
            && self.critical_pressure_pa > self.minimum_model_pressure_pa
            && self.minimum_model_pressure_pa > 0.0
            && self.max_sensor_age_s.is_finite()
            && self.max_sensor_age_s >= 0.0
            && self.min_sensor_confidence.is_finite()
            && (0.0..=1.0).contains(&self.min_sensor_confidence)
            && self.max_relative_pressure_disagreement.is_finite()
            && self.max_relative_pressure_disagreement >= 0.0
            && self.min_localization_confidence.is_finite()
            && (0.0..=1.0).contains(&self.min_localization_confidence)
            && self.min_localization_anomaly_score.is_finite()
            && (0.0..=1.0).contains(&self.min_localization_anomaly_score)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PressureIntegritySeverity {
    Nominal,
    Monitor,
    ReturnToSafeHaven,
    ImmediateEmergency,
    InvalidState,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PressureIntegrityAssessment {
    pub estimated_pressure_pa: f64,
    pub pressure_disagreement_fraction: f64,
    pub total_effective_leak_area_m2: f64,
    pub estimated_mass_loss_kg_s: f64,
    pub estimated_pressure_loss_pa_s: f64,
    pub time_to_warning_s: Option<f64>,
    pub time_to_critical_s: Option<f64>,
    pub localized_zone: Option<PressureZone>,
    pub localization_confidence: f64,
    pub severity: PressureIntegritySeverity,
    /// Conservative normalized summary suitable for the existing assist
    /// supervisor. This is not a material-strength or qualification metric.
    pub integrity_fraction: f64,
    pub evidence: ExosuitEvidenceLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PressureIntegrityError {
    InvalidConfig,
    InvalidGasState,
    NoValidPressureSensors,
    ContradictoryPressureSensors,
    InvalidDamageSite,
    InvalidLocalizationReading,
}

#[derive(Debug, Clone)]
pub struct PressureIntegrityTwin {
    config: PressureIntegrityConfig,
    damage_sites: Vec<PressureDamageSite>,
}

impl PressureIntegrityTwin {
    pub fn new(config: PressureIntegrityConfig) -> Result<Self, PressureIntegrityError> {
        if !config.is_valid() {
            return Err(PressureIntegrityError::InvalidConfig);
        }
        Ok(Self {
            config,
            damage_sites: Vec::new(),
        })
    }

    pub fn simulation_reference() -> Self {
        Self::new(PressureIntegrityConfig::simulation_reference())
            .expect("reference pressure-integrity config must be valid")
    }

    pub fn config(&self) -> &PressureIntegrityConfig {
        &self.config
    }

    pub fn damage_sites(&self) -> &[PressureDamageSite] {
        &self.damage_sites
    }

    pub fn add_damage_site(
        &mut self,
        site: PressureDamageSite,
    ) -> Result<usize, PressureIntegrityError> {
        if !site.is_valid() {
            return Err(PressureIntegrityError::InvalidDamageSite);
        }
        self.damage_sites.push(site);
        Ok(self.damage_sites.len() - 1)
    }

    pub fn apply_patch(
        &mut self,
        index: usize,
        state: PatchState,
        residual_fraction: f64,
        evidence: ExosuitEvidenceLevel,
    ) -> Result<(), PressureIntegrityError> {
        if !residual_fraction.is_finite() || !(0.0..=1.0).contains(&residual_fraction) {
            return Err(PressureIntegrityError::InvalidDamageSite);
        }
        let site = self
            .damage_sites
            .get_mut(index)
            .ok_or(PressureIntegrityError::InvalidDamageSite)?;
        site.patch_state = state;
        site.patch_residual_fraction = residual_fraction;
        site.evidence = site.evidence.min(evidence);
        Ok(())
    }

    pub fn propagate_damage(&mut self, dt_s: f64) -> Result<(), PressureIntegrityError> {
        if !dt_s.is_finite() || dt_s <= 0.0 {
            return Err(PressureIntegrityError::InvalidDamageSite);
        }
        for site in &mut self.damage_sites {
            if !site.propagate(dt_s) {
                return Err(PressureIntegrityError::InvalidDamageSite);
            }
        }
        Ok(())
    }

    pub fn assess(
        &self,
        sensors: &[PressureSensorReading],
        localization: &[LeakLocalizationReading],
        ambient_pressure_pa: f64,
        gas_temperature_k: f64,
        observed_makeup_flow_kg_s: f64,
    ) -> Result<PressureIntegrityAssessment, PressureIntegrityError> {
        if !self.config.is_valid() {
            return Err(PressureIntegrityError::InvalidConfig);
        }
        if !ambient_pressure_pa.is_finite()
            || ambient_pressure_pa < 0.0
            || !gas_temperature_k.is_finite()
            || gas_temperature_k <= 0.0
            || !observed_makeup_flow_kg_s.is_finite()
            || observed_makeup_flow_kg_s < 0.0
        {
            return Err(PressureIntegrityError::InvalidGasState);
        }
        if self.damage_sites.iter().any(|site| !site.is_valid()) {
            return Err(PressureIntegrityError::InvalidDamageSite);
        }
        if localization.iter().any(|reading| !reading.is_valid()) {
            return Err(PressureIntegrityError::InvalidLocalizationReading);
        }

        let (estimated_pressure_pa, disagreement) = self.fuse_pressure_sensors(sensors)?;
        let total_area = self
            .damage_sites
            .iter()
            .map(PressureDamageSite::effective_area_after_patch_m2)
            .sum::<f64>();
        let mass_loss = compressible_orifice_mass_flow_kg_s(
            estimated_pressure_pa,
            ambient_pressure_pa,
            gas_temperature_k,
            total_area,
            &self.config,
        );
        let net_mass_loss = (mass_loss - observed_makeup_flow_kg_s).max(0.0);
        let pressure_loss_pa_s = net_mass_loss
            * self.config.specific_gas_constant_j_kg_k
            * gas_temperature_k
            / self.config.free_volume_m3;

        let time_to_warning_s = time_to_threshold(
            estimated_pressure_pa,
            self.config.warning_pressure_pa,
            pressure_loss_pa_s,
        );
        let time_to_critical_s = time_to_threshold(
            estimated_pressure_pa,
            self.config.critical_pressure_pa,
            pressure_loss_pa_s,
        );
        let (localized_zone, localization_confidence) =
            self.localize(localization, estimated_pressure_pa)?;

        let severity = classify_severity(
            estimated_pressure_pa,
            time_to_warning_s,
            time_to_critical_s,
            total_area,
            &self.config,
        );
        let integrity_fraction = integrity_summary(
            estimated_pressure_pa,
            total_area,
            time_to_critical_s,
            &self.config,
        );

        Ok(PressureIntegrityAssessment {
            estimated_pressure_pa,
            pressure_disagreement_fraction: disagreement,
            total_effective_leak_area_m2: total_area,
            estimated_mass_loss_kg_s: mass_loss,
            estimated_pressure_loss_pa_s: pressure_loss_pa_s,
            time_to_warning_s,
            time_to_critical_s,
            localized_zone,
            localization_confidence,
            severity,
            integrity_fraction,
            evidence: self.config.evidence,
        })
    }

    fn fuse_pressure_sensors(
        &self,
        sensors: &[PressureSensorReading],
    ) -> Result<(f64, f64), PressureIntegrityError> {
        let valid = sensors
            .iter()
            .copied()
            .filter(|reading| {
                reading.is_valid()
                    && reading.age_s <= self.config.max_sensor_age_s
                    && reading.confidence >= self.config.min_sensor_confidence
            })
            .collect::<Vec<_>>();
        if valid.is_empty() {
            return Err(PressureIntegrityError::NoValidPressureSensors);
        }
        let weight_sum = valid.iter().map(|reading| reading.confidence).sum::<f64>();
        let estimate = valid
            .iter()
            .map(|reading| reading.pressure_pa * reading.confidence)
            .sum::<f64>()
            / weight_sum;
        let min = valid
            .iter()
            .map(|reading| reading.pressure_pa)
            .fold(f64::INFINITY, f64::min);
        let max = valid
            .iter()
            .map(|reading| reading.pressure_pa)
            .fold(0.0, f64::max);
        let disagreement = if estimate > 0.0 {
            (max - min) / estimate
        } else if max == 0.0 {
            0.0
        } else {
            f64::INFINITY
        };
        if !disagreement.is_finite()
            || disagreement > self.config.max_relative_pressure_disagreement
        {
            return Err(PressureIntegrityError::ContradictoryPressureSensors);
        }
        Ok((estimate, disagreement))
    }

    fn localize(
        &self,
        localization: &[LeakLocalizationReading],
        _pressure_pa: f64,
    ) -> Result<(Option<PressureZone>, f64), PressureIntegrityError> {
        let mut best: Option<LeakLocalizationReading> = None;
        for reading in localization {
            if reading.age_s > self.config.max_sensor_age_s
                || reading.confidence < self.config.min_localization_confidence
                || reading.anomaly_score < self.config.min_localization_anomaly_score
            {
                continue;
            }
            if best
                .map(|current| {
                    reading.anomaly_score * reading.confidence
                        > current.anomaly_score * current.confidence
                })
                .unwrap_or(true)
            {
                best = Some(*reading);
            }
        }
        Ok(best
            .map(|reading| (Some(reading.zone), reading.confidence))
            .unwrap_or((None, 0.0)))
    }
}

fn compressible_orifice_mass_flow_kg_s(
    upstream_pressure_pa: f64,
    downstream_pressure_pa: f64,
    temperature_k: f64,
    area_m2: f64,
    config: &PressureIntegrityConfig,
) -> f64 {
    if area_m2 <= 0.0 || upstream_pressure_pa <= downstream_pressure_pa {
        return 0.0;
    }
    let gamma = config.gamma;
    let r = config.specific_gas_constant_j_kg_k;
    let pressure_ratio = (downstream_pressure_pa / upstream_pressure_pa).clamp(0.0, 1.0);
    let critical_ratio = (2.0 / (gamma + 1.0)).powf(gamma / (gamma - 1.0));
    let flow_factor = if pressure_ratio <= critical_ratio {
        (gamma / r / temperature_k).sqrt()
            * (2.0 / (gamma + 1.0)).powf((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    } else {
        let term = (2.0 * gamma / (r * temperature_k * (gamma - 1.0)))
            * (pressure_ratio.powf(2.0 / gamma)
                - pressure_ratio.powf((gamma + 1.0) / gamma));
        term.max(0.0).sqrt()
    };
    config.discharge_coefficient * area_m2 * upstream_pressure_pa * flow_factor
}

fn time_to_threshold(current_pa: f64, threshold_pa: f64, loss_pa_s: f64) -> Option<f64> {
    if current_pa <= threshold_pa {
        Some(0.0)
    } else if loss_pa_s > 0.0 {
        Some((current_pa - threshold_pa) / loss_pa_s)
    } else {
        None
    }
}

fn classify_severity(
    pressure_pa: f64,
    time_to_warning_s: Option<f64>,
    time_to_critical_s: Option<f64>,
    total_area_m2: f64,
    config: &PressureIntegrityConfig,
) -> PressureIntegritySeverity {
    if !pressure_pa.is_finite() || !total_area_m2.is_finite() {
        return PressureIntegritySeverity::InvalidState;
    }
    if pressure_pa <= config.critical_pressure_pa
        || time_to_critical_s.is_some_and(|time| time <= 120.0)
    {
        PressureIntegritySeverity::ImmediateEmergency
    } else if pressure_pa <= config.warning_pressure_pa
        || time_to_warning_s.is_some_and(|time| time <= 300.0)
        || total_area_m2 > 0.0
    {
        PressureIntegritySeverity::ReturnToSafeHaven
    } else if total_area_m2 > 0.0 {
        PressureIntegritySeverity::Monitor
    } else {
        PressureIntegritySeverity::Nominal
    }
}

fn integrity_summary(
    pressure_pa: f64,
    total_area_m2: f64,
    time_to_critical_s: Option<f64>,
    config: &PressureIntegrityConfig,
) -> f64 {
    if !pressure_pa.is_finite() || !total_area_m2.is_finite() {
        return 0.0;
    }
    let pressure_fraction = if pressure_pa >= config.warning_pressure_pa {
        1.0
    } else {
        ((pressure_pa - config.minimum_model_pressure_pa)
            / (config.warning_pressure_pa - config.minimum_model_pressure_pa))
            .clamp(0.0, 1.0)
    };
    let time_fraction = time_to_critical_s
        .map(|time| (time / 600.0).clamp(0.0, 1.0))
        .unwrap_or(1.0);
    let leak_penalty = (1.0 / (1.0 + total_area_m2 * 1.0e6)).clamp(0.0, 1.0);
    pressure_fraction.min(time_fraction).min(leak_penalty)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sensors(pressure_pa: f64) -> [PressureSensorReading; 3] {
        [
            PressureSensorReading { pressure_pa, confidence: 0.99, age_s: 0.1 },
            PressureSensorReading { pressure_pa: pressure_pa * 1.001, confidence: 0.98, age_s: 0.1 },
            PressureSensorReading { pressure_pa: pressure_pa * 0.999, confidence: 0.97, age_s: 0.2 },
        ]
    }

    fn leak(zone: PressureZone, area_m2: f64) -> PressureDamageSite {
        PressureDamageSite {
            zone,
            source: PressureDamageSource::MmodImpact,
            effective_orifice_area_m2: area_m2,
            fractional_growth_rate_s: 0.0,
            patch_residual_fraction: 1.0,
            patch_state: PatchState::None,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    #[test]
    fn intact_suit_has_no_modeled_leak_and_nominal_severity() {
        let twin = PressureIntegrityTwin::simulation_reference();
        let result = twin.assess(&sensors(30_000.0), &[], 0.0, 295.0, 0.0).unwrap();
        assert_eq!(result.total_effective_leak_area_m2, 0.0);
        assert_eq!(result.estimated_mass_loss_kg_s, 0.0);
        assert_eq!(result.severity, PressureIntegritySeverity::Nominal);
        assert!(result.time_to_critical_s.is_none());
    }

    #[test]
    fn puncture_produces_finite_time_to_critical() {
        let mut twin = PressureIntegrityTwin::simulation_reference();
        twin.add_damage_site(leak(PressureZone::Torso, 1.0e-6)).unwrap();
        let result = twin.assess(&sensors(30_000.0), &[], 0.0, 295.0, 0.0).unwrap();
        assert!(result.estimated_mass_loss_kg_s > 0.0);
        assert!(result.estimated_pressure_loss_pa_s > 0.0);
        assert!(result.time_to_critical_s.is_some());
        assert!(result.integrity_fraction < 1.0);
    }

    #[test]
    fn verified_patch_reduces_leak_and_extends_predicted_time() {
        let mut twin = PressureIntegrityTwin::simulation_reference();
        let index = twin.add_damage_site(leak(PressureZone::LeftGlove, 1.0e-6)).unwrap();
        let before = twin.assess(&sensors(30_000.0), &[], 0.0, 295.0, 0.0).unwrap();
        twin.apply_patch(index, PatchState::TemporaryVerified, 0.10, ExosuitEvidenceLevel::Simulation)
            .unwrap();
        let after = twin.assess(&sensors(30_000.0), &[], 0.0, 295.0, 0.0).unwrap();
        assert!(after.estimated_mass_loss_kg_s < before.estimated_mass_loss_kg_s);
        assert!(after.time_to_critical_s.unwrap() > before.time_to_critical_s.unwrap());
    }

    #[test]
    fn pressure_sensor_disagreement_fails_closed() {
        let twin = PressureIntegrityTwin::simulation_reference();
        let readings = [
            PressureSensorReading { pressure_pa: 30_000.0, confidence: 0.99, age_s: 0.1 },
            PressureSensorReading { pressure_pa: 20_000.0, confidence: 0.99, age_s: 0.1 },
        ];
        assert_eq!(
            twin.assess(&readings, &[], 0.0, 295.0, 0.0),
            Err(PressureIntegrityError::ContradictoryPressureSensors)
        );
    }

    #[test]
    fn localization_uses_fresh_high_confidence_anomaly() {
        let mut twin = PressureIntegrityTwin::simulation_reference();
        twin.add_damage_site(leak(PressureZone::RightArm, 5.0e-7)).unwrap();
        let localization = [
            LeakLocalizationReading { zone: PressureZone::Torso, anomaly_score: 0.55, confidence: 0.85, age_s: 0.1 },
            LeakLocalizationReading { zone: PressureZone::RightArm, anomaly_score: 0.90, confidence: 0.95, age_s: 0.1 },
        ];
        let result = twin.assess(&sensors(30_000.0), &localization, 0.0, 295.0, 0.0).unwrap();
        assert_eq!(result.localized_zone, Some(PressureZone::RightArm));
        assert_eq!(result.localization_confidence, 0.95);
    }

    #[test]
    fn observed_makeup_flow_can_cancel_small_net_pressure_loss_without_hiding_leak() {
        let mut twin = PressureIntegrityTwin::simulation_reference();
        twin.add_damage_site(leak(PressureZone::Torso, 2.0e-7)).unwrap();
        let baseline = twin.assess(&sensors(30_000.0), &[], 0.0, 295.0, 0.0).unwrap();
        let compensated = twin
            .assess(
                &sensors(30_000.0),
                &[],
                0.0,
                295.0,
                baseline.estimated_mass_loss_kg_s * 1.1,
            )
            .unwrap();
        assert_eq!(compensated.estimated_pressure_loss_pa_s, 0.0);
        assert!(compensated.total_effective_leak_area_m2 > 0.0);
        assert_ne!(compensated.severity, PressureIntegritySeverity::Nominal);
    }
}
