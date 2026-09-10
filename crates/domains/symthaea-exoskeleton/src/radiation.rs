// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Operational EVA radiation-control model.
//!
//! A spacesuit has limited shielding mass. This model therefore treats
//! radiation safety primarily as measurement, forecasting, exposure budgeting,
//! and safe-haven timing. Shield descriptors record material/areal density but
//! never invent an attenuation factor without a radiation-transport result.

use serde::{Deserialize, Serialize};

use crate::space_exosuit::ExosuitEvidenceLevel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RadiationMaterialClass {
    LowZHydrogenous,
    Water,
    StructuralComposite,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RadiationShieldDescriptor {
    pub material: RadiationMaterialClass,
    /// Areal density, g/cm^2. This is geometry/material metadata only.
    pub areal_density_g_cm2: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl RadiationShieldDescriptor {
    pub fn is_valid(&self) -> bool {
        self.areal_density_g_cm2.is_finite() && self.areal_density_g_cm2 >= 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RadiationObservation {
    /// Measured personal dose rate at the wearer, mSv/h.
    pub personal_dose_rate_msv_h: f64,
    /// Accumulated dose assigned to the current EVA, mSv.
    pub eva_cumulative_dose_msv: f64,
    /// Forecast upper-bound dose rate over the planning horizon, mSv/h.
    pub forecast_upper_rate_msv_h: f64,
    /// Travel time to a materially better-shielded safe haven, minutes.
    pub safe_haven_time_min: f64,
    /// Whether space-weather operations have declared an energetic-particle alert.
    pub energetic_particle_alert: bool,
    /// Active dosimeter is healthy and its reading is current.
    pub dosimeter_healthy: bool,
}

impl RadiationObservation {
    pub fn is_valid(&self) -> bool {
        self.personal_dose_rate_msv_h.is_finite()
            && self.personal_dose_rate_msv_h >= 0.0
            && self.eva_cumulative_dose_msv.is_finite()
            && self.eva_cumulative_dose_msv >= 0.0
            && self.forecast_upper_rate_msv_h.is_finite()
            && self.forecast_upper_rate_msv_h >= 0.0
            && self.safe_haven_time_min.is_finite()
            && self.safe_haven_time_min >= 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RadiationOperationalPolicy {
    /// Program-defined EVA dose budget for this planning interval, mSv.
    pub eva_dose_budget_msv: f64,
    /// Program-defined rate at which immediate shelter action is required.
    pub immediate_shelter_rate_msv_h: f64,
    /// Fraction of total EVA dose budget at which return should begin.
    pub return_fraction: f64,
    /// Conservative multiplier applied to forecast rate while estimating dose
    /// incurred during return to shelter.
    pub forecast_margin: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl RadiationOperationalPolicy {
    pub fn simulation_reference() -> Self {
        Self {
            eva_dose_budget_msv: 1.0,
            immediate_shelter_rate_msv_h: 5.0,
            return_fraction: 0.5,
            forecast_margin: 2.0,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.eva_dose_budget_msv.is_finite()
            && self.eva_dose_budget_msv > 0.0
            && self.immediate_shelter_rate_msv_h.is_finite()
            && self.immediate_shelter_rate_msv_h > 0.0
            && self.return_fraction.is_finite()
            && (0.0..=1.0).contains(&self.return_fraction)
            && self.forecast_margin.is_finite()
            && self.forecast_margin >= 1.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RadiationAction {
    Continue,
    ReturnToSafeHaven,
    ImmediateShelter,
    InstrumentFault,
    InvalidState,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RadiationDecision {
    pub action: RadiationAction,
    /// Conservative predicted dose accumulated before shelter is reached, mSv.
    pub predicted_dose_at_shelter_msv: f64,
    pub remaining_budget_msv: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct RadiationExposureManager {
    policy: RadiationOperationalPolicy,
}

impl RadiationExposureManager {
    pub fn new(policy: RadiationOperationalPolicy) -> Option<Self> {
        policy.is_valid().then_some(Self { policy })
    }

    pub fn simulation_reference() -> Self {
        Self::new(RadiationOperationalPolicy::simulation_reference())
            .expect("reference radiation policy must be valid")
    }

    pub fn evaluate(&self, observation: RadiationObservation) -> RadiationDecision {
        if !self.policy.is_valid() || !observation.is_valid() {
            return RadiationDecision {
                action: RadiationAction::InvalidState,
                predicted_dose_at_shelter_msv: f64::INFINITY,
                remaining_budget_msv: 0.0,
            };
        }
        if !observation.dosimeter_healthy {
            return RadiationDecision {
                action: RadiationAction::InstrumentFault,
                predicted_dose_at_shelter_msv: f64::INFINITY,
                remaining_budget_msv: 0.0,
            };
        }

        let return_hours = observation.safe_haven_time_min / 60.0;
        let rate_for_return = observation
            .personal_dose_rate_msv_h
            .max(observation.forecast_upper_rate_msv_h)
            * self.policy.forecast_margin;
        let predicted_dose_at_shelter_msv =
            observation.eva_cumulative_dose_msv + rate_for_return * return_hours;
        let remaining_budget_msv =
            (self.policy.eva_dose_budget_msv - observation.eva_cumulative_dose_msv).max(0.0);

        let action = if observation.energetic_particle_alert
            || observation.personal_dose_rate_msv_h >= self.policy.immediate_shelter_rate_msv_h
            || predicted_dose_at_shelter_msv >= self.policy.eva_dose_budget_msv
        {
            RadiationAction::ImmediateShelter
        } else if observation.eva_cumulative_dose_msv
            >= self.policy.eva_dose_budget_msv * self.policy.return_fraction
        {
            RadiationAction::ReturnToSafeHaven
        } else {
            RadiationAction::Continue
        };

        RadiationDecision {
            action,
            predicted_dose_at_shelter_msv,
            remaining_budget_msv,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nominal() -> RadiationObservation {
        RadiationObservation {
            personal_dose_rate_msv_h: 0.05,
            eva_cumulative_dose_msv: 0.1,
            forecast_upper_rate_msv_h: 0.08,
            safe_haven_time_min: 15.0,
            energetic_particle_alert: false,
            dosimeter_healthy: true,
        }
    }

    #[test]
    fn nominal_exposure_can_continue() {
        let d = RadiationExposureManager::simulation_reference().evaluate(nominal());
        assert_eq!(d.action, RadiationAction::Continue);
    }

    #[test]
    fn energetic_particle_alert_demands_immediate_shelter() {
        let mut o = nominal();
        o.energetic_particle_alert = true;
        let d = RadiationExposureManager::simulation_reference().evaluate(o);
        assert_eq!(d.action, RadiationAction::ImmediateShelter);
    }

    #[test]
    fn failed_dosimeter_fails_closed() {
        let mut o = nominal();
        o.dosimeter_healthy = false;
        let d = RadiationExposureManager::simulation_reference().evaluate(o);
        assert_eq!(d.action, RadiationAction::InstrumentFault);
    }

    #[test]
    fn shield_metadata_never_claims_spectrum_independent_attenuation() {
        let shield = RadiationShieldDescriptor {
            material: RadiationMaterialClass::LowZHydrogenous,
            areal_density_g_cm2: 1.0,
            evidence: ExosuitEvidenceLevel::Simulation,
        };
        assert!(shield.is_valid());
    }
}
