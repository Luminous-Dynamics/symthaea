// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bearing coupling from durability state into EVA resource models.
//!
//! This module deliberately remains a thin composition layer. The dust twin
//! owns contamination/degradation state, PLSS owns life-support state, and the
//! mission harness owns task/resource accounting. This layer only translates a
//! declared dust result into explicit mechanical, thermal, and electrical
//! modifiers so those systems do not acquire hidden cross-dependencies.

use serde::{Deserialize, Serialize};

use crate::dust::DustStepReport;
use crate::eva_mission::EvaMissionSegment;
use crate::plss::{PlssReferenceTwin, PlssStepError, PlssStepInput};
use crate::space_exosuit::ExosuitEvidenceLevel;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DurabilityMissionModifiers {
    /// Multiplier applied to gross task mechanical power. A value above one
    /// represents additional effort caused by contaminated/more resistive joints.
    pub mechanical_load_multiplier: f64,
    /// Fraction of nominal PLSS heat rejection still available [0,1].
    pub radiator_heat_rejection_fraction: f64,
    /// EDS electrical load assigned to the mission bus, W.
    pub additional_mission_power_w: f64,
    /// Optical transmission available to the wearer [0,1].
    pub visor_transmission: f64,
    /// Generalized seal-risk proxy [0,1].
    pub seal_risk: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl DurabilityMissionModifiers {
    pub fn clean_reference() -> Self {
        Self {
            mechanical_load_multiplier: 1.0,
            radiator_heat_rejection_fraction: 1.0,
            additional_mission_power_w: 0.0,
            visor_transmission: 1.0,
            seal_risk: 0.0,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn from_dust(report: &DustStepReport) -> Option<Self> {
        let value = Self {
            mechanical_load_multiplier: report.joint_friction_multiplier,
            radiator_heat_rejection_fraction: report.radiator_heat_rejection_fraction,
            additional_mission_power_w: report.eds_power_w,
            visor_transmission: report.visor_transmission,
            seal_risk: report.seal_risk,
            evidence: report.evidence,
        };
        value.is_valid().then_some(value)
    }

    pub fn is_valid(&self) -> bool {
        self.mechanical_load_multiplier.is_finite()
            && self.mechanical_load_multiplier >= 1.0
            && self.radiator_heat_rejection_fraction.is_finite()
            && (0.0..=1.0).contains(&self.radiator_heat_rejection_fraction)
            && self.additional_mission_power_w.is_finite()
            && self.additional_mission_power_w >= 0.0
            && self.visor_transmission.is_finite()
            && (0.0..=1.0).contains(&self.visor_transmission)
            && self.seal_risk.is_finite()
            && (0.0..=1.0).contains(&self.seal_risk)
    }

    /// Return a cloned task segment with durability-driven task work and EDS
    /// electrical demand made explicit. The base segment is never mutated.
    pub fn adjust_segment(&self, base: &EvaMissionSegment) -> Option<EvaMissionSegment> {
        if !self.is_valid() || !base.is_valid() {
            return None;
        }
        let mut adjusted = base.clone();
        adjusted.gross_positive_mechanical_power_w *= self.mechanical_load_multiplier;
        adjusted.mission_power_w += self.additional_mission_power_w;
        adjusted.is_valid().then_some(adjusted)
    }

    /// Apply the dust-derived radiator derating through the PLSS's explicit
    /// environmental step path. This remains simulation only and does not
    /// command thermal hardware.
    pub fn step_plss(
        &self,
        plss: &mut PlssReferenceTwin,
        input: PlssStepInput,
    ) -> Result<(), PlssStepError> {
        if !self.is_valid() {
            return Err(PlssStepError::InvalidInput);
        }
        plss.step_with_heat_rejection_fraction(
            input,
            self.radiator_heat_rejection_fraction,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DurabilityOperationalPolicy {
    /// Below this visor transmission, degraded work is recommended.
    pub degrade_visor_transmission: f64,
    /// Below this visor transmission, service/return is recommended.
    pub return_visor_transmission: f64,
    /// Above this seal-risk proxy, service/return is recommended.
    pub return_seal_risk: f64,
    /// Below this heat-rejection fraction, service/return is recommended.
    pub return_heat_rejection_fraction: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl DurabilityOperationalPolicy {
    /// Simulation thresholds for deterministic trade studies only.
    pub fn simulation_reference() -> Self {
        Self {
            degrade_visor_transmission: 0.85,
            return_visor_transmission: 0.60,
            return_seal_risk: 0.20,
            return_heat_rejection_fraction: 0.60,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.degrade_visor_transmission.is_finite()
            && (0.0..=1.0).contains(&self.degrade_visor_transmission)
            && self.return_visor_transmission.is_finite()
            && (0.0..=self.degrade_visor_transmission).contains(&self.return_visor_transmission)
            && self.return_seal_risk.is_finite()
            && (0.0..=1.0).contains(&self.return_seal_risk)
            && self.return_heat_rejection_fraction.is_finite()
            && (0.0..=1.0).contains(&self.return_heat_rejection_fraction)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DurabilityRecommendation {
    Continue,
    DegradeWork,
    ReturnForService,
    InvalidState,
}

pub fn durability_recommendation(
    modifiers: DurabilityMissionModifiers,
    policy: DurabilityOperationalPolicy,
) -> DurabilityRecommendation {
    if !modifiers.is_valid() || !policy.is_valid() {
        return DurabilityRecommendation::InvalidState;
    }

    if modifiers.visor_transmission <= policy.return_visor_transmission
        || modifiers.seal_risk >= policy.return_seal_risk
        || modifiers.radiator_heat_rejection_fraction <= policy.return_heat_rejection_fraction
    {
        DurabilityRecommendation::ReturnForService
    } else if modifiers.visor_transmission <= policy.degrade_visor_transmission
        || modifiers.mechanical_load_multiplier > 1.0
        || modifiers.additional_mission_power_w > 0.0
    {
        DurabilityRecommendation::DegradeWork
    } else {
        DurabilityRecommendation::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dust::{DustDegradationTwin, DustExposure};
    use crate::eva_mission::EvaMissionPhase;
    use crate::metabolism::{HumanMetabolicModel, HumanWorkload};

    fn plss_input() -> PlssStepInput {
        let metabolism = HumanMetabolicModel::reference()
            .estimate(HumanWorkload {
                positive_mechanical_power_w: 100.0,
                negative_mechanical_power_w: 0.0,
            })
            .unwrap();
        PlssStepInput {
            metabolism,
            equipment_heat_w: 100.0,
            humidity_generation_per_min: 0.2,
            dt_s: 60.0,
        }
    }

    #[test]
    fn dust_increases_task_work_and_eds_mission_power() {
        let mut twin = DustDegradationTwin::simulation_reference();
        let report = twin.step(DustExposure::uniform(2.0, 60.0)).unwrap();
        let modifiers = DurabilityMissionModifiers::from_dust(&report).unwrap();
        let base = EvaMissionSegment::lunar_reference(
            "dusty-work",
            EvaMissionPhase::SurfaceWork,
            60.0,
        );
        let adjusted = modifiers.adjust_segment(&base).unwrap();
        assert!(adjusted.gross_positive_mechanical_power_w >= base.gross_positive_mechanical_power_w);
        assert!(adjusted.mission_power_w > base.mission_power_w);
    }

    #[test]
    fn radiator_dust_increases_plss_thermal_store() {
        let mut dust = DustDegradationTwin::simulation_reference();
        let report = dust.step(DustExposure::uniform(4.0, 60.0)).unwrap();
        let modifiers = DurabilityMissionModifiers::from_dust(&report).unwrap();

        let mut clean_plss = PlssReferenceTwin::simulation_reference();
        let mut dusty_plss = PlssReferenceTwin::simulation_reference();
        clean_plss.step(plss_input()).unwrap();
        modifiers.step_plss(&mut dusty_plss, plss_input()).unwrap();

        assert!(dusty_plss.state().thermal_store_k > clean_plss.state().thermal_store_k);
    }

    #[test]
    fn severe_thermal_derating_recommends_service_return() {
        let mut modifiers = DurabilityMissionModifiers::clean_reference();
        modifiers.radiator_heat_rejection_fraction = 0.40;
        assert_eq!(
            durability_recommendation(
                modifiers,
                DurabilityOperationalPolicy::simulation_reference()
            ),
            DurabilityRecommendation::ReturnForService
        );
    }

    #[test]
    fn invalid_state_fails_closed_to_invalid_recommendation() {
        let mut modifiers = DurabilityMissionModifiers::clean_reference();
        modifiers.seal_risk = f64::NAN;
        assert_eq!(
            durability_recommendation(
                modifiers,
                DurabilityOperationalPolicy::simulation_reference()
            ),
            DurabilityRecommendation::InvalidState
        );
    }
}
