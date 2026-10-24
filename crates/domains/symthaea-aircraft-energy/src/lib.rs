// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic, evidence-bounded screening of aircraft mission energy and power.
//!
//! This crate checks arithmetic constraints for a frozen mission profile. It is
//! not an aircraft performance solver, battery safety model, power-dispatch
//! optimizer, airworthiness assessment, or operating authorization.
//!
//! System energy and power are effective values at the declared aircraft bus.
//! The caller must account for conversion losses, thermal/aging derates,
//! integration limits, and provenance in its upstream model. The aggregate
//! envelope screen does not model source dispatch or prove that hybrid sources
//! can operate in parallel.

use std::collections::HashSet;

pub mod design_space;
pub mod mission_model_contract;
pub mod scenario_sweep;

const SECONDS_PER_HOUR: f64 = 3_600.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DutyClass {
    /// Must be supportable by the declared continuous bus-power rating.
    Continuous,
    /// Must be supportable by the peak rating; durations count toward the
    /// mission's aggregate peak-duration budget.
    Peak,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MissionSegment {
    pub id: String,
    /// Segment duration in seconds.
    pub duration_s: Option<f64>,
    /// Required aircraft-bus power in watts, including loads in the mission boundary.
    pub required_bus_power_w: Option<f64>,
    pub duty_class: Option<DutyClass>,
}

impl MissionSegment {
    pub fn new(
        id: impl Into<String>,
        duration_s: f64,
        required_bus_power_w: f64,
        duty_class: DutyClass,
    ) -> Self {
        Self {
            id: id.into(),
            duration_s: Some(duration_s),
            required_bus_power_w: Some(required_bus_power_w),
            duty_class: Some(duty_class),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MissionProfile {
    pub id: String,
    pub segments: Vec<MissionSegment>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnergySystemEnvelope {
    /// Complete installed mass within the declared system boundary, kg.
    pub installed_mass_kg: Option<f64>,
    /// Energy deliverable at the aircraft bus before reserve is withheld, Wh.
    /// Relevant derating and conversion losses must already be included.
    pub usable_energy_at_bus_wh: Option<f64>,
    /// Fraction of deliverable energy that must remain after the mission, [0, 1).
    pub reserve_fraction: Option<f64>,
    /// Continuous electrical output available at the aircraft bus, W.
    pub max_continuous_bus_power_w: Option<f64>,
    /// Peak electrical output available at the aircraft bus, W.
    pub max_peak_bus_power_w: Option<f64>,
    /// Aggregate peak-power duration budget for this mission, seconds.
    pub max_peak_duration_s: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatteryPackSpec {
    pub installed_pack_mass_kg: f64,
    /// Complete installed energy-system mass in the declared boundary, kg.
    /// Must include the pack plus required conversion, cooling, protection,
    /// mounting, and distribution hardware represented by that boundary.
    pub installed_system_mass_kg: f64,
    /// Pack-level (not cell-level) specific energy, Wh/kg.
    pub pack_specific_energy_wh_per_kg: f64,
    /// Fraction of pack energy permitted by the declared usable-energy rule, [0, 1].
    pub usable_fraction: f64,
    /// Capacity fraction retained at the declared operating temperature, [0, 1].
    pub temperature_capacity_fraction: f64,
    /// End-of-life state of health, [0, 1].
    pub end_of_life_soh_fraction: f64,
    /// Fraction of pack energy available at the aircraft bus after discharge losses, [0, 1].
    pub discharge_efficiency_fraction: f64,
    /// Pack-level continuous discharge specific power, W/kg.
    pub continuous_specific_power_w_per_kg: f64,
    /// Pack-level peak discharge specific power, W/kg.
    pub peak_specific_power_w_per_kg: f64,
    /// Power derating for the declared temperature/aging operating point, [0, 1].
    pub power_derating_fraction: f64,
    pub reserve_fraction: f64,
    /// Aggregate peak-power duration budget for the mission, seconds.
    pub max_peak_duration_s: f64,
}

impl BatteryPackSpec {
    /// Convert pack data to a bus-level screening envelope using explicit
    /// multiplicative capacity and power derating. This is bookkeeping, not
    /// a battery electrochemistry or thermal-transient model.
    pub fn to_envelope(&self) -> Result<EnergySystemEnvelope, Vec<Finding>> {
        let mut findings = Vec::new();
        check_positive(&mut findings, "battery.installed_pack_mass_kg", self.installed_pack_mass_kg);
        check_positive(&mut findings, "battery.installed_system_mass_kg", self.installed_system_mass_kg);
        if self.installed_system_mass_kg < self.installed_pack_mass_kg {
            findings.push(Finding::InvalidInput {
                field: "battery.installed_system_mass_kg".into(),
                reason: "complete installed system mass cannot be less than pack mass".into(),
            });
        }
        check_positive(
            &mut findings,
            "battery.pack_specific_energy_wh_per_kg",
            self.pack_specific_energy_wh_per_kg,
        );
        check_fraction(&mut findings, "battery.usable_fraction", self.usable_fraction, true);
        check_fraction(
            &mut findings,
            "battery.temperature_capacity_fraction",
            self.temperature_capacity_fraction,
            true,
        );
        check_fraction(
            &mut findings,
            "battery.end_of_life_soh_fraction",
            self.end_of_life_soh_fraction,
            true,
        );
        check_fraction(
            &mut findings,
            "battery.discharge_efficiency_fraction",
            self.discharge_efficiency_fraction,
            true,
        );
        check_non_negative(
            &mut findings,
            "battery.continuous_specific_power_w_per_kg",
            self.continuous_specific_power_w_per_kg,
        );
        check_non_negative(
            &mut findings,
            "battery.peak_specific_power_w_per_kg",
            self.peak_specific_power_w_per_kg,
        );
        check_fraction(
            &mut findings,
            "battery.power_derating_fraction",
            self.power_derating_fraction,
            true,
        );
        check_fraction(&mut findings, "battery.reserve_fraction", self.reserve_fraction, false);
        check_positive(&mut findings, "battery.max_peak_duration_s", self.max_peak_duration_s);
        if self.peak_specific_power_w_per_kg < self.continuous_specific_power_w_per_kg {
            findings.push(Finding::InvalidInput {
                field: "battery.peak_specific_power_w_per_kg".into(),
                reason: "peak specific power must be at least continuous specific power".into(),
            });
        }
        if !findings.is_empty() {
            return Err(findings);
        }

        let usable_energy_at_bus_wh = self.installed_pack_mass_kg
            * self.pack_specific_energy_wh_per_kg
            * self.usable_fraction
            * self.temperature_capacity_fraction
            * self.end_of_life_soh_fraction
            * self.discharge_efficiency_fraction;
        let max_continuous_bus_power_w = self.installed_pack_mass_kg
            * self.continuous_specific_power_w_per_kg
            * self.power_derating_fraction;
        let max_peak_bus_power_w = self.installed_pack_mass_kg
            * self.peak_specific_power_w_per_kg
            * self.power_derating_fraction;

        for (field, value) in [
            ("battery.derived_usable_energy_at_bus_wh", usable_energy_at_bus_wh),
            ("battery.derived_max_continuous_bus_power_w", max_continuous_bus_power_w),
            ("battery.derived_max_peak_bus_power_w", max_peak_bus_power_w),
        ] {
            if !value.is_finite() {
                return Err(vec![Finding::InvalidInput {
                    field: field.into(),
                    reason: "derived value is not finite".into(),
                }]);
            }
        }

        Ok(EnergySystemEnvelope {
            installed_mass_kg: Some(self.installed_system_mass_kg),
            usable_energy_at_bus_wh: Some(usable_energy_at_bus_wh),
            reserve_fraction: Some(self.reserve_fraction),
            max_continuous_bus_power_w: Some(max_continuous_bus_power_w),
            max_peak_bus_power_w: Some(max_peak_bus_power_w),
            max_peak_duration_s: Some(self.max_peak_duration_s),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Finding {
    MissingInput { field: String },
    InvalidInput { field: String, reason: String },
    EmptyMission,
    DuplicateSegmentId { id: String },
    EnergyShortfall { required_wh: f64, available_after_reserve_wh: f64 },
    ContinuousPowerShortfall { segment_id: String, required_w: f64, available_w: f64 },
    PeakPowerShortfall { segment_id: String, required_w: f64, available_w: f64 },
    PeakDurationExceeded { required_s: f64, available_s: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    /// The aggregate energy/power checks pass only within this crate's model.
    FeasibleWithinScreen,
    /// At least one explicitly modeled arithmetic constraint fails.
    InfeasibleWithinScreen,
    /// Required evidence/inputs are missing or invalid; this is never a pass.
    Indeterminate,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScreeningResult {
    pub disposition: Disposition,
    pub required_mission_energy_wh: Option<f64>,
    pub available_energy_after_reserve_wh: Option<f64>,
    pub energy_margin_wh: Option<f64>,
    pub minimum_continuous_power_margin_w: Option<f64>,
    pub peak_power_margin_w: Option<f64>,
    pub peak_duration_margin_s: Option<f64>,
    pub installed_mass_kg: Option<f64>,
    pub findings: Vec<Finding>,
}

impl ScreeningResult {
    fn indeterminate(findings: Vec<Finding>) -> Self {
        Self {
            disposition: Disposition::Indeterminate,
            required_mission_energy_wh: None,
            available_energy_after_reserve_wh: None,
            energy_margin_wh: None,
            minimum_continuous_power_margin_w: None,
            peak_power_margin_w: None,
            peak_duration_margin_s: None,
            installed_mass_kg: None,
            findings,
        }
    }
}

/// Compare a mission profile with aggregate bus-level energy and power limits.
///
/// This model does not calculate source dispatch, thermal transients, voltage
/// stability, battery electrochemistry, aerodynamics, generator fuel depletion,
/// or hybrid-bus integration. A passing result is not a flightworthiness,
/// complete-aircraft feasibility, or operating-authority claim.
pub fn evaluate_mission(
    mission: &MissionProfile,
    system: &EnergySystemEnvelope,
) -> ScreeningResult {
    let mut findings = Vec::new();
    if mission.id.trim().is_empty() {
        findings.push(Finding::InvalidInput {
            field: "mission.id".into(),
            reason: "must not be empty".into(),
        });
    }
    if mission.segments.is_empty() {
        findings.push(Finding::EmptyMission);
    }

    let installed_mass_kg = required_input(&mut findings, "system.installed_mass_kg", system.installed_mass_kg);
    let usable_energy_at_bus_wh = required_input(
        &mut findings,
        "system.usable_energy_at_bus_wh",
        system.usable_energy_at_bus_wh,
    );
    let reserve_fraction = required_input(&mut findings, "system.reserve_fraction", system.reserve_fraction);
    let max_continuous_bus_power_w = required_input(
        &mut findings,
        "system.max_continuous_bus_power_w",
        system.max_continuous_bus_power_w,
    );
    let max_peak_bus_power_w = required_input(
        &mut findings,
        "system.max_peak_bus_power_w",
        system.max_peak_bus_power_w,
    );
    let max_peak_duration_s = required_input(
        &mut findings,
        "system.max_peak_duration_s",
        system.max_peak_duration_s,
    );

    let mut ids = HashSet::new();
    for (index, segment) in mission.segments.iter().enumerate() {
        if segment.id.trim().is_empty() {
            findings.push(Finding::InvalidInput {
                field: format!("mission.segments[{index}].id"),
                reason: "must not be empty".into(),
            });
        } else if !ids.insert(segment.id.clone()) {
            findings.push(Finding::DuplicateSegmentId { id: segment.id.clone() });
        }
        let _ = required_input(
            &mut findings,
            &format!("mission.segments[{index}].duration_s"),
            segment.duration_s,
        );
        let _ = required_input(
            &mut findings,
            &format!("mission.segments[{index}].required_bus_power_w"),
            segment.required_bus_power_w,
        );
        if segment.duty_class.is_none() {
            findings.push(Finding::MissingInput {
                field: format!("mission.segments[{index}].duty_class"),
            });
        }
    }

    if let Some(value) = installed_mass_kg {
        check_positive(&mut findings, "system.installed_mass_kg", value);
    }
    if let Some(value) = usable_energy_at_bus_wh {
        check_non_negative(&mut findings, "system.usable_energy_at_bus_wh", value);
    }
    if let Some(value) = reserve_fraction {
        check_fraction(&mut findings, "system.reserve_fraction", value, false);
    }
    if let Some(value) = max_continuous_bus_power_w {
        check_non_negative(&mut findings, "system.max_continuous_bus_power_w", value);
    }
    if let Some(value) = max_peak_bus_power_w {
        check_non_negative(&mut findings, "system.max_peak_bus_power_w", value);
    }
    if let Some(value) = max_peak_duration_s {
        check_positive(&mut findings, "system.max_peak_duration_s", value);
    }
    if let (Some(continuous), Some(peak)) = (max_continuous_bus_power_w, max_peak_bus_power_w) {
        if peak < continuous {
            findings.push(Finding::InvalidInput {
                field: "system.max_peak_bus_power_w".into(),
                reason: "peak bus power must be at least continuous bus power".into(),
            });
        }
    }
    for (index, segment) in mission.segments.iter().enumerate() {
        if let Some(duration) = segment.duration_s {
            check_positive(&mut findings, &format!("mission.segments[{index}].duration_s"), duration);
        }
        if let Some(power) = segment.required_bus_power_w {
            check_non_negative(&mut findings, &format!("mission.segments[{index}].required_bus_power_w"), power);
        }
    }

    if !findings.is_empty() {
        return ScreeningResult::indeterminate(findings);
    }

    let (
        Some(mass),
        Some(energy),
        Some(reserve),
        Some(continuous),
        Some(peak),
        Some(peak_duration_budget),
    ) = (
        installed_mass_kg,
        usable_energy_at_bus_wh,
        reserve_fraction,
        max_continuous_bus_power_w,
        max_peak_bus_power_w,
        max_peak_duration_s,
    ) else {
        return ScreeningResult::indeterminate(vec![Finding::MissingInput {
            field: "system.required_values".into(),
        }]);
    };

    let mut required_energy_wh = 0.0;
    let mut min_continuous_margin_w: Option<f64> = None;
    let mut peak_power_margin_w: Option<f64> = None;
    let mut total_peak_duration_s = 0.0;
    let mut physical_findings = Vec::new();

    for segment in &mission.segments {
        let (Some(duration_s), Some(required_power_w), Some(duty_class)) = (
            segment.duration_s,
            segment.required_bus_power_w,
            segment.duty_class,
        ) else {
            return ScreeningResult::indeterminate(vec![Finding::MissingInput {
                field: format!("mission.segments[{}]", segment.id),
            }]);
        };

        let segment_energy_wh = required_power_w * duration_s / SECONDS_PER_HOUR;
        required_energy_wh += segment_energy_wh;
        if !segment_energy_wh.is_finite() || !required_energy_wh.is_finite() {
            return ScreeningResult::indeterminate(vec![Finding::InvalidInput {
                field: "mission.required_energy_wh".into(),
                reason: "energy accumulation overflowed or became non-finite".into(),
            }]);
        }

        match duty_class {
            DutyClass::Continuous => {
                let margin = continuous - required_power_w;
                min_continuous_margin_w = Some(min_continuous_margin_w.map_or(margin, |x| x.min(margin)));
                if margin < 0.0 {
                    physical_findings.push(Finding::ContinuousPowerShortfall {
                        segment_id: segment.id.clone(),
                        required_w: required_power_w,
                        available_w: continuous,
                    });
                }
            }
            DutyClass::Peak => {
                total_peak_duration_s += duration_s;
                let margin = peak - required_power_w;
                peak_power_margin_w = Some(peak_power_margin_w.map_or(margin, |x| x.min(margin)));
                if margin < 0.0 {
                    physical_findings.push(Finding::PeakPowerShortfall {
                        segment_id: segment.id.clone(),
                        required_w: required_power_w,
                        available_w: peak,
                    });
                }
            }
        }
    }

    let available_after_reserve_wh = energy * (1.0 - reserve);
    if !available_after_reserve_wh.is_finite() || !total_peak_duration_s.is_finite() {
        return ScreeningResult::indeterminate(vec![Finding::InvalidInput {
            field: "derived_mission_limits".into(),
            reason: "derived value is not finite".into(),
        }]);
    }
    let energy_margin_wh = available_after_reserve_wh - required_energy_wh;
    let peak_duration_margin_s = peak_duration_budget - total_peak_duration_s;
    if energy_margin_wh < 0.0 {
        physical_findings.push(Finding::EnergyShortfall {
            required_wh: required_energy_wh,
            available_after_reserve_wh,
        });
    }
    if peak_duration_margin_s < 0.0 {
        physical_findings.push(Finding::PeakDurationExceeded {
            required_s: total_peak_duration_s,
            available_s: peak_duration_budget,
        });
    }

    ScreeningResult {
        disposition: if physical_findings.is_empty() {
            Disposition::FeasibleWithinScreen
        } else {
            Disposition::InfeasibleWithinScreen
        },
        required_mission_energy_wh: Some(required_energy_wh),
        available_energy_after_reserve_wh: Some(available_after_reserve_wh),
        energy_margin_wh: Some(energy_margin_wh),
        minimum_continuous_power_margin_w: min_continuous_margin_w,
        peak_power_margin_w,
        peak_duration_margin_s: Some(peak_duration_margin_s),
        installed_mass_kg: Some(mass),
        findings: physical_findings,
    }
}

fn required_input(findings: &mut Vec<Finding>, field: &str, value: Option<f64>) -> Option<f64> {
    if value.is_none() {
        findings.push(Finding::MissingInput { field: field.into() });
    }
    value
}

fn check_positive(findings: &mut Vec<Finding>, field: &str, value: f64) {
    if !value.is_finite() || value <= 0.0 {
        findings.push(Finding::InvalidInput {
            field: field.into(),
            reason: "must be finite and greater than zero".into(),
        });
    }
}

fn check_non_negative(findings: &mut Vec<Finding>, field: &str, value: f64) {
    if !value.is_finite() || value < 0.0 {
        findings.push(Finding::InvalidInput {
            field: field.into(),
            reason: "must be finite and non-negative".into(),
        });
    }
}

fn check_fraction(findings: &mut Vec<Finding>, field: &str, value: f64, allow_one: bool) {
    let upper_invalid = if allow_one { value > 1.0 } else { value >= 1.0 };
    if !value.is_finite() || value < 0.0 || upper_invalid {
        let interval = if allow_one { "[0, 1]" } else { "[0, 1)" };
        findings.push(Finding::InvalidInput {
            field: field.into(),
            reason: format!("must be finite and lie in {interval}"),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mission(segments: Vec<MissionSegment>) -> MissionProfile {
        MissionProfile { id: "utility-cargo-reference-v0".into(), segments }
    }

    fn system(energy_wh: f64, reserve: f64, continuous_w: f64, peak_w: f64, peak_s: f64) -> EnergySystemEnvelope {
        EnergySystemEnvelope {
            installed_mass_kg: Some(100.0),
            usable_energy_at_bus_wh: Some(energy_wh),
            reserve_fraction: Some(reserve),
            max_continuous_bus_power_w: Some(continuous_w),
            max_peak_bus_power_w: Some(peak_w),
            max_peak_duration_s: Some(peak_s),
        }
    }

    #[test]
    fn battery_pack_derates_are_multiplicative_and_explicit() {
        let pack = BatteryPackSpec {
            installed_pack_mass_kg: 100.0,
            installed_system_mass_kg: 130.0,
            pack_specific_energy_wh_per_kg: 300.0,
            usable_fraction: 0.9,
            temperature_capacity_fraction: 0.8,
            end_of_life_soh_fraction: 0.85,
            discharge_efficiency_fraction: 0.95,
            continuous_specific_power_w_per_kg: 1_000.0,
            peak_specific_power_w_per_kg: 1_500.0,
            power_derating_fraction: 0.8,
            reserve_fraction: 0.2,
            max_peak_duration_s: 60.0,
        };
        let envelope = pack.to_envelope().expect("valid pack should map to an envelope");
        assert!((envelope.usable_energy_at_bus_wh.unwrap() - 17_442.0).abs() < 1e-9);
        assert_eq!(envelope.max_continuous_bus_power_w, Some(80_000.0));
        assert_eq!(envelope.max_peak_bus_power_w, Some(120_000.0));
    }

    #[test]
    fn nominal_mission_passes_only_the_declared_screen() {
        let profile = mission(vec![MissionSegment::new("cruise", 1_200.0, 10_000.0, DutyClass::Continuous)]);
        let result = evaluate_mission(&profile, &system(10_000.0, 0.2, 15_000.0, 20_000.0, 60.0));
        assert_eq!(result.disposition, Disposition::FeasibleWithinScreen);
        assert!((result.required_mission_energy_wh.unwrap() - (10_000.0 / 3.0)).abs() < 1e-9);
        assert!((result.available_energy_after_reserve_wh.unwrap() - 8_000.0).abs() < 1e-9);
    }

    #[test]
    fn energy_shortfall_fails_even_when_power_is_sufficient() {
        let profile = mission(vec![MissionSegment::new("cruise", 3_600.0, 1_000.0, DutyClass::Continuous)]);
        let result = evaluate_mission(&profile, &system(1_000.0, 0.2, 2_000.0, 3_000.0, 100.0));
        assert_eq!(result.disposition, Disposition::InfeasibleWithinScreen);
        assert!(result.findings.iter().any(|f| matches!(f, Finding::EnergyShortfall { .. })));
    }

    #[test]
    fn peak_power_shortfall_fails_even_when_total_energy_is_sufficient() {
        let profile = mission(vec![MissionSegment::new("takeoff", 10.0, 2_000.0, DutyClass::Peak)]);
        let result = evaluate_mission(&profile, &system(50_000.0, 0.1, 1_000.0, 1_500.0, 30.0));
        assert_eq!(result.disposition, Disposition::InfeasibleWithinScreen);
        assert!(result.findings.iter().any(|f| matches!(f, Finding::PeakPowerShortfall { .. })));
        assert!(result.findings.iter().all(|f| !matches!(f, Finding::EnergyShortfall { .. })));
    }

    #[test]
    fn continuous_power_shortfall_is_not_hidden_by_peak_rating() {
        let profile = mission(vec![MissionSegment::new("climb", 300.0, 2_000.0, DutyClass::Continuous)]);
        let result = evaluate_mission(&profile, &system(50_000.0, 0.1, 1_500.0, 3_000.0, 30.0));
        assert!(result.findings.iter().any(|f| matches!(f, Finding::ContinuousPowerShortfall { .. })));
    }

    #[test]
    fn cumulative_peak_time_is_bounded() {
        let profile = mission(vec![
            MissionSegment::new("takeoff", 20.0, 1_500.0, DutyClass::Peak),
            MissionSegment::new("go_around", 20.0, 1_500.0, DutyClass::Peak),
        ]);
        let result = evaluate_mission(&profile, &system(50_000.0, 0.1, 1_000.0, 2_000.0, 30.0));
        assert!(result.findings.iter().any(|f| matches!(f, Finding::PeakDurationExceeded { .. })));
    }

    #[test]
    fn missing_system_evidence_is_indeterminate_not_a_pass() {
        let profile = mission(vec![MissionSegment::new("cruise", 60.0, 100.0, DutyClass::Continuous)]);
        let mut incomplete = system(1_000.0, 0.1, 500.0, 1_000.0, 30.0);
        incomplete.usable_energy_at_bus_wh = None;
        let result = evaluate_mission(&profile, &incomplete);
        assert_eq!(result.disposition, Disposition::Indeterminate);
        assert!(result.findings.iter().any(|f| matches!(f, Finding::MissingInput { field } if field == "system.usable_energy_at_bus_wh")));
        assert!(result.required_mission_energy_wh.is_none());
    }

    #[test]
    fn invalid_nan_is_indeterminate() {
        let profile = mission(vec![MissionSegment::new("cruise", 60.0, f64::NAN, DutyClass::Continuous)]);
        let result = evaluate_mission(&profile, &system(1_000.0, 0.1, 500.0, 1_000.0, 30.0));
        assert_eq!(result.disposition, Disposition::Indeterminate);
    }

    #[test]
    fn duplicate_segment_ids_are_rejected() {
        let profile = mission(vec![
            MissionSegment::new("cruise", 10.0, 100.0, DutyClass::Continuous),
            MissionSegment::new("cruise", 10.0, 100.0, DutyClass::Continuous),
        ]);
        assert_eq!(
            evaluate_mission(&profile, &system(1_000.0, 0.1, 200.0, 400.0, 30.0)).disposition,
            Disposition::Indeterminate
        );
    }

    #[test]
    fn missing_duty_class_is_indeterminate() {
        let profile = mission(vec![MissionSegment {
            id: "unclassified".into(),
            duration_s: Some(10.0),
            required_bus_power_w: Some(100.0),
            duty_class: None,
        }]);
        let result = evaluate_mission(&profile, &system(1_000.0, 0.1, 200.0, 400.0, 30.0));
        assert_eq!(result.disposition, Disposition::Indeterminate);
        assert!(result.findings.iter().any(|f| matches!(f, Finding::MissingInput { field } if field.ends_with("duty_class"))));
    }

    #[test]
    fn reserve_of_one_is_rejected_as_indeterminate() {
        let profile = mission(vec![MissionSegment::new("cruise", 60.0, 100.0, DutyClass::Continuous)]);
        let mut invalid = system(1_000.0, 0.1, 200.0, 400.0, 30.0);
        invalid.reserve_fraction = Some(1.0);
        assert_eq!(evaluate_mission(&profile, &invalid).disposition, Disposition::Indeterminate);
    }

    #[test]
    fn energy_arithmetic_overflow_is_indeterminate() {
        let profile = mission(vec![MissionSegment::new(
            "overflow",
            f64::MAX,
            f64::MAX,
            DutyClass::Continuous,
        )]);
        let result = evaluate_mission(&profile, &system(1.0e300, 0.1, f64::MAX, f64::MAX, f64::MAX));
        assert_eq!(result.disposition, Disposition::Indeterminate);
        assert!(result.findings.iter().any(|f| matches!(f, Finding::InvalidInput { field, .. } if field == "mission.required_energy_wh")));
    }

    #[test]
    fn zero_duration_segments_are_rejected() {
        let profile = mission(vec![MissionSegment::new("instantaneous", 0.0, 100.0, DutyClass::Peak)]);
        assert_eq!(evaluate_mission(&profile, &system(1_000.0, 0.1, 200.0, 400.0, 30.0)).disposition, Disposition::Indeterminate);
    }

    #[test]
    fn complete_installed_mass_cannot_omit_pack_mass() {
        let pack = BatteryPackSpec {
            installed_pack_mass_kg: 100.0,
            installed_system_mass_kg: 90.0,
            pack_specific_energy_wh_per_kg: 300.0,
            usable_fraction: 0.9,
            temperature_capacity_fraction: 0.9,
            end_of_life_soh_fraction: 0.9,
            discharge_efficiency_fraction: 0.95,
            continuous_specific_power_w_per_kg: 500.0,
            peak_specific_power_w_per_kg: 800.0,
            power_derating_fraction: 0.8,
            reserve_fraction: 0.2,
            max_peak_duration_s: 20.0,
        };
        assert!(pack.to_envelope().is_err());
    }

    #[test]
    fn invalid_pack_derating_is_rejected_before_envelope_creation() {
        let pack = BatteryPackSpec {
            installed_pack_mass_kg: 10.0,
            installed_system_mass_kg: 12.0,
            pack_specific_energy_wh_per_kg: 300.0,
            usable_fraction: 1.1,
            temperature_capacity_fraction: 0.8,
            end_of_life_soh_fraction: 0.9,
            discharge_efficiency_fraction: 0.95,
            continuous_specific_power_w_per_kg: 500.0,
            peak_specific_power_w_per_kg: 800.0,
            power_derating_fraction: 0.8,
            reserve_fraction: 0.2,
            max_peak_duration_s: 20.0,
        };
        assert!(pack.to_envelope().is_err());
    }
}
