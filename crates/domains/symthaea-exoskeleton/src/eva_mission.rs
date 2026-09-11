// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integrated EVA mission harness for the Space Exosuit research program.
//!
//! This module composes the existing metabolism, PLSS, protected power, and
//! radiation models into one deterministic simulation lineage. It is a
//! research harness, not a flight controller: it never commands PLSS hardware
//! or rescue propulsion and all default values remain simulation evidence.

use serde::{Deserialize, Serialize};

use crate::metabolism::{HumanMetabolicModel, HumanWorkload, MetabolicEstimate};
use crate::plss::{PlssReferenceTwin, PlssStepError, PlssStepInput};
use crate::power::{MultiBusPowerSystem, PowerAllocation, PowerRequest};
use crate::radiation::{
    RadiationAction, RadiationDecision, RadiationExposureManager, RadiationObservation,
};
use crate::space_exosuit::ExosuitEvidenceLevel;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvaMissionPhase {
    TraverseOut,
    SurfaceWork,
    TraverseBack,
    Contingency,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaMissionSegment {
    pub name: String,
    pub phase: EvaMissionPhase,
    pub duration_s: f64,
    /// Positive external mechanical power required by the task before assist.
    pub gross_positive_mechanical_power_w: f64,
    /// Human eccentric/negative-work demand before any future recovery model.
    pub negative_mechanical_power_w: f64,
    /// Requested fraction of positive task work supplied by the exoskeleton.
    pub requested_assist_fraction: f64,
    /// Mechanical output / electrical input for the assist path.
    pub assist_motor_efficiency: f64,
    /// Mobility electronics/controls load that is paid before assist power.
    pub mobility_overhead_w: f64,
    pub survival_power_w: f64,
    pub mission_power_w: f64,
    /// Non-mobility equipment heat coupled into the PLSS thermal model.
    pub non_actuator_equipment_heat_w: f64,
    pub humidity_generation_per_min: f64,
    pub regenerative_power_w: f64,
    pub external_charger_w: f64,
    pub personal_dose_rate_msv_h: f64,
    pub forecast_upper_rate_msv_h: f64,
    pub safe_haven_time_min: f64,
    pub energetic_particle_alert: bool,
    pub dosimeter_healthy: bool,
    pub evidence: ExosuitEvidenceLevel,
}

impl EvaMissionSegment {
    pub fn is_valid(&self) -> bool {
        !self.name.trim().is_empty()
            && self.duration_s.is_finite()
            && self.duration_s > 0.0
            && self.gross_positive_mechanical_power_w.is_finite()
            && self.gross_positive_mechanical_power_w >= 0.0
            && self.negative_mechanical_power_w.is_finite()
            && self.negative_mechanical_power_w >= 0.0
            && self.requested_assist_fraction.is_finite()
            && (0.0..=1.0).contains(&self.requested_assist_fraction)
            && self.assist_motor_efficiency.is_finite()
            && self.assist_motor_efficiency > 0.0
            && self.assist_motor_efficiency <= 1.0
            && [
                self.mobility_overhead_w,
                self.survival_power_w,
                self.mission_power_w,
                self.non_actuator_equipment_heat_w,
                self.humidity_generation_per_min,
                self.regenerative_power_w,
                self.external_charger_w,
                self.personal_dose_rate_msv_h,
                self.forecast_upper_rate_msv_h,
                self.safe_haven_time_min,
            ]
            .into_iter()
            .all(|v| v.is_finite() && v >= 0.0)
    }

    pub fn lunar_reference(name: impl Into<String>, phase: EvaMissionPhase, duration_s: f64) -> Self {
        Self {
            name: name.into(),
            phase,
            duration_s,
            gross_positive_mechanical_power_w: 100.0,
            negative_mechanical_power_w: 10.0,
            requested_assist_fraction: 0.40,
            assist_motor_efficiency: 0.75,
            mobility_overhead_w: 25.0,
            survival_power_w: 180.0,
            mission_power_w: 40.0,
            non_actuator_equipment_heat_w: 35.0,
            humidity_generation_per_min: 0.20,
            regenerative_power_w: 0.0,
            external_charger_w: 0.0,
            personal_dose_rate_msv_h: 0.05,
            forecast_upper_rate_msv_h: 0.08,
            safe_haven_time_min: 15.0,
            energetic_particle_alert: false,
            dosimeter_healthy: true,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EvaMissionDisposition {
    Continue,
    DegradeMission,
    ReturnToSafeHaven,
    ImmediateShelter,
    AbortEva,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvaMissionAbortReason {
    InvalidSegment,
    InvalidMetabolicState,
    PowerModelFault,
    SurvivalPowerUnavailable,
    LifeSupportFailure(PlssStepError),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaMissionSegmentReport {
    pub name: String,
    pub phase: EvaMissionPhase,
    pub disposition: EvaMissionDisposition,
    pub abort_reason: Option<EvaMissionAbortReason>,
    pub power: Option<PowerAllocation>,
    pub metabolism: Option<MetabolicEstimate>,
    pub radiation: Option<RadiationDecision>,
    pub actual_assist_fraction: f64,
    pub human_positive_mechanical_power_w: f64,
    pub actuator_waste_heat_w: f64,
    pub oxygen_consumed_l: f64,
    pub cumulative_radiation_msv: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EvaMissionTotals {
    pub elapsed_s: f64,
    pub oxygen_consumed_l: f64,
    pub co2_generated_l: f64,
    pub metabolic_energy_wh: f64,
    pub human_positive_mechanical_energy_wh: f64,
    pub survival_energy_wh: f64,
    pub mobility_energy_wh: f64,
    pub mission_energy_wh: f64,
    pub cumulative_radiation_msv: f64,
    pub degraded_segments: u32,
}

impl Default for EvaMissionTotals {
    fn default() -> Self {
        Self {
            elapsed_s: 0.0,
            oxygen_consumed_l: 0.0,
            co2_generated_l: 0.0,
            metabolic_energy_wh: 0.0,
            human_positive_mechanical_energy_wh: 0.0,
            survival_energy_wh: 0.0,
            mobility_energy_wh: 0.0,
            mission_energy_wh: 0.0,
            cumulative_radiation_msv: 0.0,
            degraded_segments: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvaMissionReport {
    pub disposition: EvaMissionDisposition,
    pub totals: EvaMissionTotals,
    pub segments: Vec<EvaMissionSegmentReport>,
    pub evidence: ExosuitEvidenceLevel,
}

#[derive(Debug, Clone)]
pub struct IntegratedEvaMission {
    metabolic: HumanMetabolicModel,
    plss: PlssReferenceTwin,
    power: MultiBusPowerSystem,
    radiation: RadiationExposureManager,
    totals: EvaMissionTotals,
}

impl IntegratedEvaMission {
    pub fn simulation_reference() -> Self {
        Self {
            metabolic: HumanMetabolicModel::reference(),
            plss: PlssReferenceTwin::simulation_reference(),
            power: MultiBusPowerSystem::simulation_reference(),
            radiation: RadiationExposureManager::simulation_reference(),
            totals: EvaMissionTotals::default(),
        }
    }

    pub fn plss_mut_for_fault_injection(&mut self) -> &mut PlssReferenceTwin {
        &mut self.plss
    }

    pub fn power_mut_for_fault_injection(&mut self) -> &mut MultiBusPowerSystem {
        &mut self.power
    }

    pub fn totals(&self) -> &EvaMissionTotals {
        &self.totals
    }

    pub fn run(&mut self, segments: &[EvaMissionSegment]) -> EvaMissionReport {
        let mut reports = Vec::with_capacity(segments.len());
        let mut disposition = EvaMissionDisposition::Continue;

        for segment in segments {
            let report = self.step(segment);
            disposition = disposition.max(report.disposition);
            let stop = matches!(
                report.disposition,
                EvaMissionDisposition::ReturnToSafeHaven
                    | EvaMissionDisposition::ImmediateShelter
                    | EvaMissionDisposition::AbortEva
            );
            reports.push(report);
            if stop {
                break;
            }
        }

        EvaMissionReport {
            disposition,
            totals: self.totals,
            segments: reports,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }

    pub fn step(&mut self, segment: &EvaMissionSegment) -> EvaMissionSegmentReport {
        if !segment.is_valid() {
            return self.abort_segment(segment, EvaMissionAbortReason::InvalidSegment);
        }

        let requested_assist_mechanical_w =
            segment.gross_positive_mechanical_power_w * segment.requested_assist_fraction;
        let requested_assist_electrical_w =
            requested_assist_mechanical_w / segment.assist_motor_efficiency;
        let requested_mobility_w = segment.mobility_overhead_w + requested_assist_electrical_w;

        let allocation = match self.power.step(PowerRequest {
            survival_w: segment.survival_power_w,
            mobility_w: requested_mobility_w,
            mission_w: segment.mission_power_w,
            regenerative_w: segment.regenerative_power_w,
            external_charger_w: segment.external_charger_w,
            dt_s: segment.duration_s,
        }) {
            Ok(value) => value,
            Err(_) => return self.abort_segment(segment, EvaMissionAbortReason::PowerModelFault),
        };

        self.accumulate_power(&allocation, segment.duration_s);
        self.totals.elapsed_s += segment.duration_s;

        if !allocation.survival_satisfied {
            return self.abort_segment_with_power(
                segment,
                allocation,
                EvaMissionAbortReason::SurvivalPowerUnavailable,
            );
        }

        // Mobility overhead is treated as the first mobility load. Only power
        // above that overhead can produce mechanical assistance.
        let electrical_for_assist_w =
            (allocation.mobility_delivered_w - segment.mobility_overhead_w).max(0.0);
        let actual_assist_mechanical_w = (electrical_for_assist_w
            * segment.assist_motor_efficiency)
            .min(segment.gross_positive_mechanical_power_w);
        let actual_assist_fraction = if segment.gross_positive_mechanical_power_w > 0.0 {
            actual_assist_mechanical_w / segment.gross_positive_mechanical_power_w
        } else {
            0.0
        };
        let human_positive_mechanical_power_w =
            (segment.gross_positive_mechanical_power_w - actual_assist_mechanical_w).max(0.0);
        let actuator_waste_heat_w =
            (electrical_for_assist_w - actual_assist_mechanical_w).max(0.0);

        let metabolism = match self.metabolic.estimate(HumanWorkload {
            positive_mechanical_power_w: human_positive_mechanical_power_w,
            negative_mechanical_power_w: segment.negative_mechanical_power_w,
        }) {
            Some(value) => value,
            None => {
                return self.abort_segment_with_power(
                    segment,
                    allocation,
                    EvaMissionAbortReason::InvalidMetabolicState,
                )
            }
        };

        let o2_before = self.plss.state().primary_o2_remaining_l
            + self.plss.state().secondary_o2_remaining_l;
        if let Err(err) = self.plss.step(PlssStepInput {
            metabolism,
            equipment_heat_w: segment.non_actuator_equipment_heat_w
                + segment.mobility_overhead_w
                + actuator_waste_heat_w,
            humidity_generation_per_min: segment.humidity_generation_per_min,
            dt_s: segment.duration_s,
        }) {
            return self.abort_segment_with_details(
                segment,
                allocation,
                metabolism,
                actual_assist_fraction,
                human_positive_mechanical_power_w,
                actuator_waste_heat_w,
                EvaMissionAbortReason::LifeSupportFailure(err),
            );
        }
        let o2_after = self.plss.state().primary_o2_remaining_l
            + self.plss.state().secondary_o2_remaining_l;
        let oxygen_consumed_l = (o2_before - o2_after).max(0.0);

        let hours = segment.duration_s / 3600.0;
        self.totals.oxygen_consumed_l += oxygen_consumed_l;
        self.totals.co2_generated_l += metabolism.co2_l_min * segment.duration_s / 60.0;
        self.totals.metabolic_energy_wh += metabolism.metabolic_power_w * hours;
        self.totals.human_positive_mechanical_energy_wh +=
            human_positive_mechanical_power_w * hours;
        self.totals.cumulative_radiation_msv += segment.personal_dose_rate_msv_h * hours;

        let radiation = self.radiation.evaluate(RadiationObservation {
            personal_dose_rate_msv_h: segment.personal_dose_rate_msv_h,
            eva_cumulative_dose_msv: self.totals.cumulative_radiation_msv,
            forecast_upper_rate_msv_h: segment.forecast_upper_rate_msv_h,
            safe_haven_time_min: segment.safe_haven_time_min,
            energetic_particle_alert: segment.energetic_particle_alert,
            dosimeter_healthy: segment.dosimeter_healthy,
        });

        let mut disposition = match radiation.action {
            RadiationAction::Continue => EvaMissionDisposition::Continue,
            RadiationAction::ReturnToSafeHaven
            | RadiationAction::InstrumentFault
            | RadiationAction::InvalidState => EvaMissionDisposition::ReturnToSafeHaven,
            RadiationAction::ImmediateShelter => EvaMissionDisposition::ImmediateShelter,
        };

        if !allocation.mobility_satisfied || !allocation.mission_satisfied {
            disposition = disposition.max(EvaMissionDisposition::DegradeMission);
            self.totals.degraded_segments += 1;
        }

        EvaMissionSegmentReport {
            name: segment.name.clone(),
            phase: segment.phase,
            disposition,
            abort_reason: None,
            power: Some(allocation),
            metabolism: Some(metabolism),
            radiation: Some(radiation),
            actual_assist_fraction,
            human_positive_mechanical_power_w,
            actuator_waste_heat_w,
            oxygen_consumed_l,
            cumulative_radiation_msv: self.totals.cumulative_radiation_msv,
        }
    }

    fn accumulate_power(&mut self, allocation: &PowerAllocation, dt_s: f64) {
        let h = dt_s / 3600.0;
        self.totals.survival_energy_wh += allocation.survival_delivered_w * h;
        self.totals.mobility_energy_wh += allocation.mobility_delivered_w * h;
        self.totals.mission_energy_wh += allocation.mission_delivered_w * h;
    }

    fn abort_segment(
        &self,
        segment: &EvaMissionSegment,
        reason: EvaMissionAbortReason,
    ) -> EvaMissionSegmentReport {
        EvaMissionSegmentReport {
            name: segment.name.clone(),
            phase: segment.phase,
            disposition: EvaMissionDisposition::AbortEva,
            abort_reason: Some(reason),
            power: None,
            metabolism: None,
            radiation: None,
            actual_assist_fraction: 0.0,
            human_positive_mechanical_power_w: 0.0,
            actuator_waste_heat_w: 0.0,
            oxygen_consumed_l: 0.0,
            cumulative_radiation_msv: self.totals.cumulative_radiation_msv,
        }
    }

    fn abort_segment_with_power(
        &self,
        segment: &EvaMissionSegment,
        power: PowerAllocation,
        reason: EvaMissionAbortReason,
    ) -> EvaMissionSegmentReport {
        let mut report = self.abort_segment(segment, reason);
        report.power = Some(power);
        report
    }

    #[allow(clippy::too_many_arguments)]
    fn abort_segment_with_details(
        &self,
        segment: &EvaMissionSegment,
        power: PowerAllocation,
        metabolism: MetabolicEstimate,
        actual_assist_fraction: f64,
        human_positive_mechanical_power_w: f64,
        actuator_waste_heat_w: f64,
        reason: EvaMissionAbortReason,
    ) -> EvaMissionSegmentReport {
        EvaMissionSegmentReport {
            name: segment.name.clone(),
            phase: segment.phase,
            disposition: EvaMissionDisposition::AbortEva,
            abort_reason: Some(reason),
            power: Some(power),
            metabolism: Some(metabolism),
            radiation: None,
            actual_assist_fraction,
            human_positive_mechanical_power_w,
            actuator_waste_heat_w,
            oxygen_consumed_l: 0.0,
            cumulative_radiation_msv: self.totals.cumulative_radiation_msv,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_hour_work(assist: f64) -> EvaMissionSegment {
        let mut s = EvaMissionSegment::lunar_reference(
            "work",
            EvaMissionPhase::SurfaceWork,
            3600.0,
        );
        s.gross_positive_mechanical_power_w = 120.0;
        s.requested_assist_fraction = assist;
        s.personal_dose_rate_msv_h = 0.02;
        s.forecast_upper_rate_msv_h = 0.03;
        s.safe_haven_time_min = 10.0;
        s
    }

    #[test]
    fn nominal_lunar_eva_completes() {
        let mut mission = IntegratedEvaMission::simulation_reference();
        let plan = [
            EvaMissionSegment::lunar_reference("outbound", EvaMissionPhase::TraverseOut, 600.0),
            EvaMissionSegment::lunar_reference("work", EvaMissionPhase::SurfaceWork, 1200.0),
            EvaMissionSegment::lunar_reference("return", EvaMissionPhase::TraverseBack, 600.0),
        ];
        let report = mission.run(&plan);
        assert!(matches!(
            report.disposition,
            EvaMissionDisposition::Continue | EvaMissionDisposition::DegradeMission
        ));
        assert_eq!(report.segments.len(), 3);
        assert!(report.totals.oxygen_consumed_l > 0.0);
        assert!(report.totals.mobility_energy_wh > 0.0);
    }

    #[test]
    fn assist_trades_mobility_energy_for_lower_oxygen_use() {
        let mut no_assist = IntegratedEvaMission::simulation_reference();
        let mut assisted = IntegratedEvaMission::simulation_reference();
        let a = no_assist.run(&[one_hour_work(0.0)]);
        let b = assisted.run(&[one_hour_work(0.50)]);
        assert!(b.totals.oxygen_consumed_l < a.totals.oxygen_consumed_l);
        assert!(b.totals.mobility_energy_wh > a.totals.mobility_energy_wh);
        assert!(b.totals.human_positive_mechanical_energy_wh
            < a.totals.human_positive_mechanical_energy_wh);
    }

    #[test]
    fn radiation_alert_forces_immediate_shelter() {
        let mut mission = IntegratedEvaMission::simulation_reference();
        let mut s = EvaMissionSegment::lunar_reference(
            "solar-event",
            EvaMissionPhase::Contingency,
            60.0,
        );
        s.energetic_particle_alert = true;
        let report = mission.run(&[s]);
        assert_eq!(report.disposition, EvaMissionDisposition::ImmediateShelter);
    }

    #[test]
    fn protected_survival_reserve_aborts_before_optional_load_can_mask_it() {
        let mut mission = IntegratedEvaMission::simulation_reference();
        let floor = mission.power.config().survival.reserve_floor_wh;
        mission
            .power_mut_for_fault_injection()
            .config_mut_for_fault_injection()
            .survival
            .energy_wh = floor;
        let report = mission.run(&[EvaMissionSegment::lunar_reference(
            "power-starved",
            EvaMissionPhase::Contingency,
            60.0,
        )]);
        assert_eq!(report.disposition, EvaMissionDisposition::AbortEva);
        assert_eq!(
            report.segments[0].abort_reason,
            Some(EvaMissionAbortReason::SurvivalPowerUnavailable)
        );
    }

    #[test]
    fn mission_bus_loss_degrades_instead_of_consuming_survival_power() {
        let mut mission = IntegratedEvaMission::simulation_reference();
        let floor = mission.power.config().mission.reserve_floor_wh;
        mission
            .power_mut_for_fault_injection()
            .config_mut_for_fault_injection()
            .mission
            .energy_wh = floor;
        let report = mission.run(&[EvaMissionSegment::lunar_reference(
            "mission-bus-lost",
            EvaMissionPhase::SurfaceWork,
            60.0,
        )]);
        assert_eq!(report.disposition, EvaMissionDisposition::DegradeMission);
        assert!(report.segments[0].power.unwrap().survival_satisfied);
    }
}
