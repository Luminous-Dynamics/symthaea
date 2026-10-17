// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mass-coupled, reduced-order sizing orchestration for battery-electric missions.
//!
//! An upstream, versioned mission model must regenerate phase demand for every
//! candidate gross mass and its exact configuration. This module uses the upper
//! declared power bounds to size a battery pack against energy, continuous-power,
//! and peak-power requirements, updates installed-system mass, and iterates until
//! mass closure or an explicit stop condition.
//!
//! This is an orchestration/arithmetic layer, not an aerodynamic or propulsion
//! solver, battery electrochemistry model, structural analysis, certification,
//! or airworthiness tool. A returned convergence result means only that the
//! caller-supplied model and these equations reached arithmetic closure.

use std::collections::BTreeSet;

use super::{
    DutyClass, EnergySystemEnvelope,
    mission_model_contract::{MissionPhaseDemand, PowerUncertaintyBounds},
};

/// Fixed aircraft mass not including the energy system sized by this module.
/// Keep mass terms mutually exclusive; do not count the propulsion or consumables
/// again in fixed_vehicle_mass_kg.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedMassBreakdown {
    pub fixed_vehicle_mass_kg: f64,
    pub propulsion_installed_mass_kg: f64,
    pub consumables_mass_kg: f64,
    pub payload_mass_kg: f64,
    pub maximum_takeoff_mass_limit_kg: f64,
}

impl FixedMassBreakdown {
    pub fn without_energy_system_kg(&self) -> f64 {
        self.fixed_vehicle_mass_kg
            + self.propulsion_installed_mass_kg
            + self.consumables_mass_kg
            + self.payload_mass_kg
    }
}

/// Pack-level properties plus the multiplier from pack mass to full installed
/// energy-system mass. Values are explicit inputs, not battery predictions.
#[derive(Debug, Clone, PartialEq)]
pub struct BatterySizingParameters {
    pub pack_specific_energy_wh_per_kg: f64,
    pub usable_fraction: f64,
    pub temperature_capacity_fraction: f64,
    pub end_of_life_soh_fraction: f64,
    pub discharge_efficiency_fraction: f64,
    pub continuous_specific_power_w_per_kg: f64,
    pub peak_specific_power_w_per_kg: f64,
    pub power_derating_fraction: f64,
    pub reserve_fraction: f64,
    /// Installed-system mass / battery-pack mass; must be at least 1.
    pub installed_system_mass_multiplier: f64,
    pub max_peak_duration_s: f64,
}

/// Numerical convergence controls. A relaxed fixed-point iteration can fail
/// to converge; that result remains visible and must not be called a pass.
#[derive(Debug, Clone, PartialEq)]
pub struct CouplingSettings {
    pub max_iterations: usize,
    pub absolute_mass_tolerance_kg: f64,
    pub relative_mass_tolerance: f64,
    /// Relaxation factor in (0, 1]; values below 1 may stabilize an iteration.
    pub relaxation_factor: f64,
}

/// Stable identities for one candidate on one mission family. The model
/// implementation is responsible for binding these IDs, route, environment,
/// geometry, aero/propulsion maps and source revisions to real artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoupledMissionContext {
    pub candidate_id: String,
    pub configuration_id: String,
    pub mission_family_id: String,
}

/// Adapter to an external/reduced-order mission-demand implementation.
///
/// Implementations must recalculate demand at the supplied gross mass using
/// candidate-specific geometry/configuration and the frozen route/environment.
/// The trait is intentionally model-agnostic: this crate does not invent
/// aerodynamic coefficients or default performance maps.
pub trait MassSensitiveMissionModel {
    fn phase_demands(
        &self,
        context: &CoupledMissionContext,
        gross_mass_kg: f64,
    ) -> Result<Vec<MissionPhaseDemand>, String>;
}

#[derive(Debug, Clone, PartialEq)]
pub enum CouplingFinding {
    InvalidInput { field: String, reason: String },
    ModelFailure { iteration: usize, reason: String },
    InvalidPhase { iteration: usize, phase_id: String, reason: String },
    EmptyPhaseList { iteration: usize },
    DuplicatePhaseId { iteration: usize, phase_id: String },
    PeakDurationExceeded { required_s: f64, available_s: f64 },
    TakeoffMassLimitExceeded { required_kg: f64, limit_kg: f64 },
    NonConverged { iterations: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CouplingDisposition {
    ConvergedWithinModel,
    TakeoffMassLimitExceeded,
    PeakDurationExceeded,
    ModelFailure,
    InvalidDemand,
    NonConverged,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoupledIteration {
    pub iteration: usize,
    /// Gross mass used by the mission model for the phase demands in this row.
    pub evaluated_gross_mass_kg: f64,
    pub worst_case_mission_energy_wh: f64,
    pub peak_duration_s: f64,
    pub energy_limited_pack_mass_kg: f64,
    pub continuous_power_limited_pack_mass_kg: f64,
    pub peak_power_limited_pack_mass_kg: f64,
    pub required_pack_mass_kg: f64,
    pub required_installed_energy_system_mass_kg: f64,
    /// Fixed mass + calculated installed-system mass; this is the next fixed-point target.
    pub target_takeoff_mass_kg: f64,
    /// Target takeoff mass minus the mass at which demands were generated.
    pub mass_residual_kg: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CoupledSizingReport {
    pub disposition: CouplingDisposition,
    pub context: CoupledMissionContext,
    pub iteration_trace: Vec<CoupledIteration>,
    /// Final mass at which the mission model was evaluated, if demand was valid.
    pub evaluated_gross_mass_kg: Option<f64>,
    /// Mass implied by final sizing. Difference from evaluated mass is exposed.
    pub required_takeoff_mass_kg: Option<f64>,
    pub mass_closure_residual_kg: Option<f64>,
    pub required_pack_mass_kg: Option<f64>,
    pub required_installed_energy_system_mass_kg: Option<f64>,
    pub energy_system_envelope: Option<EnergySystemEnvelope>,
    pub final_phase_demands: Vec<MissionPhaseDemand>,
    pub worst_case_mission_energy_wh: Option<f64>,
    pub findings: Vec<CouplingFinding>,
}

#[derive(Debug, Clone, PartialEq)]
struct SizingCalculation {
    required_energy_wh: f64,
    peak_duration_s: f64,
    energy_limited_pack_mass_kg: f64,
    continuous_power_limited_pack_mass_kg: f64,
    peak_power_limited_pack_mass_kg: f64,
    required_pack_mass_kg: f64,
    installed_mass_kg: f64,
    target_takeoff_mass_kg: f64,
    envelope: EnergySystemEnvelope,
}

fn validate_inputs(
    context: &CoupledMissionContext,
    fixed_mass: &FixedMassBreakdown,
    battery: &BatterySizingParameters,
    settings: &CouplingSettings,
    initial_installed_energy_system_mass_kg: f64,
) -> Vec<CouplingFinding> {
    let mut findings = Vec::new();
    for (field, value) in [
        ("candidate_id", context.candidate_id.as_str()),
        ("configuration_id", context.configuration_id.as_str()),
        ("mission_family_id", context.mission_family_id.as_str()),
    ] {
        if value.trim().is_empty() {
            findings.push(CouplingFinding::InvalidInput {
                field: field.into(),
                reason: "must not be empty".into(),
            });
        }
    }

    for (field, value) in [
        ("mass.fixed_vehicle_mass_kg", fixed_mass.fixed_vehicle_mass_kg),
        ("mass.maximum_takeoff_mass_limit_kg", fixed_mass.maximum_takeoff_mass_limit_kg),
        ("battery.pack_specific_energy_wh_per_kg", battery.pack_specific_energy_wh_per_kg),
        ("battery.continuous_specific_power_w_per_kg", battery.continuous_specific_power_w_per_kg),
        ("battery.peak_specific_power_w_per_kg", battery.peak_specific_power_w_per_kg),
        ("battery.max_peak_duration_s", battery.max_peak_duration_s),
        ("settings.absolute_mass_tolerance_kg", settings.absolute_mass_tolerance_kg),
    ] {
        if !value.is_finite() || value <= 0.0 {
            findings.push(CouplingFinding::InvalidInput {
                field: field.into(),
                reason: "must be finite and greater than zero".into(),
            });
        }
    }
    for (field, value) in [
        ("mass.propulsion_installed_mass_kg", fixed_mass.propulsion_installed_mass_kg),
        ("mass.consumables_mass_kg", fixed_mass.consumables_mass_kg),
        ("mass.payload_mass_kg", fixed_mass.payload_mass_kg),
        ("initial_installed_energy_system_mass_kg", initial_installed_energy_system_mass_kg),
    ] {
        if !value.is_finite() || value < 0.0 {
            findings.push(CouplingFinding::InvalidInput {
                field: field.into(),
                reason: "must be finite and non-negative".into(),
            });
        }
    }
    for (field, value) in [
        ("battery.usable_fraction", battery.usable_fraction),
        ("battery.temperature_capacity_fraction", battery.temperature_capacity_fraction),
        ("battery.end_of_life_soh_fraction", battery.end_of_life_soh_fraction),
        ("battery.discharge_efficiency_fraction", battery.discharge_efficiency_fraction),
        ("battery.power_derating_fraction", battery.power_derating_fraction),
    ] {
        if !value.is_finite() || value <= 0.0 || value > 1.0 {
            findings.push(CouplingFinding::InvalidInput {
                field: field.into(),
                reason: "must be finite and lie in (0, 1] for sizing".into(),
            });
        }
    }
    if !battery.reserve_fraction.is_finite()
        || battery.reserve_fraction < 0.0
        || battery.reserve_fraction >= 1.0
    {
        findings.push(CouplingFinding::InvalidInput {
            field: "battery.reserve_fraction".into(),
            reason: "must be finite and lie in [0, 1)".into(),
        });
    }
    if !battery.installed_system_mass_multiplier.is_finite()
        || battery.installed_system_mass_multiplier < 1.0
    {
        findings.push(CouplingFinding::InvalidInput {
            field: "battery.installed_system_mass_multiplier".into(),
            reason: "must be finite and at least 1.0".into(),
        });
    }
    if settings.max_iterations == 0 {
        findings.push(CouplingFinding::InvalidInput {
            field: "settings.max_iterations".into(),
            reason: "must be greater than zero".into(),
        });
    }
    if !settings.relative_mass_tolerance.is_finite() || settings.relative_mass_tolerance < 0.0 {
        findings.push(CouplingFinding::InvalidInput {
            field: "settings.relative_mass_tolerance".into(),
            reason: "must be finite and non-negative".into(),
        });
    }
    if !settings.relaxation_factor.is_finite()
        || settings.relaxation_factor <= 0.0
        || settings.relaxation_factor > 1.0
    {
        findings.push(CouplingFinding::InvalidInput {
            field: "settings.relaxation_factor".into(),
            reason: "must be finite and in (0, 1]".into(),
        });
    }

    let fixed_total = fixed_mass.without_energy_system_kg();
    if !fixed_total.is_finite() || fixed_total <= 0.0 {
        findings.push(CouplingFinding::InvalidInput {
            field: "mass.fixed_total_kg".into(),
            reason: "summed non-energy mass must be finite and greater than zero".into(),
        });
    } else if fixed_total > fixed_mass.maximum_takeoff_mass_limit_kg {
        findings.push(CouplingFinding::InvalidInput {
            field: "mass.fixed_total_kg".into(),
            reason: "fixed mass already exceeds maximum takeoff mass".into(),
        });
    }
    let initial_total = fixed_total + initial_installed_energy_system_mass_kg;
    if !initial_total.is_finite() {
        findings.push(CouplingFinding::InvalidInput {
            field: "mass.initial_takeoff_mass_kg".into(),
            reason: "sum overflowed or became non-finite".into(),
        });
    } else if initial_total > fixed_mass.maximum_takeoff_mass_limit_kg {
        findings.push(CouplingFinding::InvalidInput {
            field: "mass.initial_takeoff_mass_kg".into(),
            reason: "initial guess exceeds maximum takeoff mass".into(),
        });
    }

    let max_peak_mass = battery.peak_specific_power_w_per_kg;
    let continuous_mass = battery.continuous_specific_power_w_per_kg;
    if max_peak_mass < continuous_mass {
        findings.push(CouplingFinding::InvalidInput {
            field: "battery.peak_specific_power_w_per_kg".into(),
            reason: "peak specific power must be at least continuous specific power".into(),
        });
    }
    findings
}

fn validate_phases(
    phases: &[MissionPhaseDemand],
    iteration: usize,
) -> Result<(), Vec<CouplingFinding>> {
    let mut findings = Vec::new();
    if phases.is_empty() {
        return Err(vec![CouplingFinding::EmptyPhaseList { iteration }]);
    }
    let mut seen = BTreeSet::new();
    let mut positive_demand_exists = false;
    for phase in phases {
        let invalid = |reason: &str| CouplingFinding::InvalidPhase {
            iteration,
            phase_id: phase.id.clone(),
            reason: reason.into(),
        };
        if phase.id.trim().is_empty() {
            findings.push(invalid("phase ID must not be empty"));
        }
        if !seen.insert(phase.id.clone()) {
            findings.push(CouplingFinding::DuplicatePhaseId {
                iteration,
                phase_id: phase.id.clone(),
            });
        }
        if !phase.duration_s.is_finite() || phase.duration_s <= 0.0 {
            findings.push(invalid("duration must be finite and greater than zero"));
        }
        if !phase.required_bus_power_w.is_finite() || phase.required_bus_power_w < 0.0 {
            findings.push(invalid("nominal bus power must be finite and non-negative"));
        }
        let bounds = &phase.power_uncertainty;
        if !bounds.lower_bound_w.is_finite()
            || !bounds.upper_bound_w.is_finite()
            || bounds.lower_bound_w < 0.0
            || bounds.upper_bound_w < bounds.lower_bound_w
            || phase.required_bus_power_w < bounds.lower_bound_w
            || phase.required_bus_power_w > bounds.upper_bound_w
        {
            findings.push(invalid("uncertainty bounds must satisfy 0 <= lower <= nominal <= upper"));
        }
        if bounds.method.trim().is_empty() {
            findings.push(invalid("power uncertainty method must not be empty"));
        }
        if phase.power_uncertainty.upper_bound_w > 0.0 {
            positive_demand_exists = true;
        }
    }
    if !positive_demand_exists {
        findings.push(CouplingFinding::InvalidInput {
            field: "mission.phase_demands".into(),
            reason: "at least one phase must have positive upper-bound bus power".into(),
        });
    }
    if findings.is_empty() { Ok(()) } else { Err(findings) }
}

fn calculate_sizing(
    fixed_mass: &FixedMassBreakdown,
    battery: &BatterySizingParameters,
    phases: &[MissionPhaseDemand],
) -> Result<SizingCalculation, CouplingFinding> {
    let mut required_energy_wh = 0.0_f64;
    let mut max_continuous_power_w = 0.0_f64;
    let mut max_peak_power_w = 0.0_f64;
    let mut peak_duration_s = 0.0_f64;

    for phase in phases {
        let upper_w = phase.power_uncertainty.upper_bound_w;
        let energy_wh = upper_w * phase.duration_s / 3_600.0;
        required_energy_wh += energy_wh;
        match phase.duty_class {
            DutyClass::Continuous => max_continuous_power_w = max_continuous_power_w.max(upper_w),
            DutyClass::Peak => {
                max_peak_power_w = max_peak_power_w.max(upper_w);
                peak_duration_s += phase.duration_s;
            }
        }
        if !energy_wh.is_finite()
            || !required_energy_wh.is_finite()
            || !peak_duration_s.is_finite()
        {
            return Err(CouplingFinding::InvalidInput {
                field: "mission.derived_demand".into(),
                reason: "energy or peak-duration accumulation overflowed".into(),
            });
        }
    }

    if peak_duration_s > battery.max_peak_duration_s {
        return Err(CouplingFinding::PeakDurationExceeded {
            required_s: peak_duration_s,
            available_s: battery.max_peak_duration_s,
        });
    }

    let usable_specific_energy_after_reserve =
        battery.pack_specific_energy_wh_per_kg
            * battery.usable_fraction
            * battery.temperature_capacity_fraction
            * battery.end_of_life_soh_fraction
            * battery.discharge_efficiency_fraction
            * (1.0 - battery.reserve_fraction);
    let effective_continuous_specific_power =
        battery.continuous_specific_power_w_per_kg * battery.power_derating_fraction;
    let effective_peak_specific_power =
        battery.peak_specific_power_w_per_kg * battery.power_derating_fraction;
    for (field, value) in [
        ("battery.usable_specific_energy_after_reserve", usable_specific_energy_after_reserve),
        ("battery.effective_continuous_specific_power", effective_continuous_specific_power),
        ("battery.effective_peak_specific_power", effective_peak_specific_power),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(CouplingFinding::InvalidInput {
                field: field.into(),
                reason: "derived sizing denominator must be finite and greater than zero".into(),
            });
        }
    }

    let energy_limited_pack_mass_kg = required_energy_wh / usable_specific_energy_after_reserve;
    let continuous_power_limited_pack_mass_kg =
        max_continuous_power_w / effective_continuous_specific_power;
    let peak_power_limited_pack_mass_kg = max_peak_power_w / effective_peak_specific_power;
    let required_pack_mass_kg = energy_limited_pack_mass_kg
        .max(continuous_power_limited_pack_mass_kg)
        .max(peak_power_limited_pack_mass_kg);
    let installed_mass_kg = required_pack_mass_kg * battery.installed_system_mass_multiplier;
    let target_takeoff_mass_kg = fixed_mass.without_energy_system_kg() + installed_mass_kg;
    let available_energy_at_bus_wh = required_pack_mass_kg
        * battery.pack_specific_energy_wh_per_kg
        * battery.usable_fraction
        * battery.temperature_capacity_fraction
        * battery.end_of_life_soh_fraction
        * battery.discharge_efficiency_fraction;
    let max_continuous_bus_power_w =
        required_pack_mass_kg * battery.continuous_specific_power_w_per_kg * battery.power_derating_fraction;
    let max_peak_bus_power_w =
        required_pack_mass_kg * battery.peak_specific_power_w_per_kg * battery.power_derating_fraction;

    for (field, value) in [
        ("battery.energy_limited_pack_mass_kg", energy_limited_pack_mass_kg),
        ("battery.continuous_power_limited_pack_mass_kg", continuous_power_limited_pack_mass_kg),
        ("battery.peak_power_limited_pack_mass_kg", peak_power_limited_pack_mass_kg),
        ("battery.required_pack_mass_kg", required_pack_mass_kg),
        ("battery.installed_mass_kg", installed_mass_kg),
        ("mass.target_takeoff_mass_kg", target_takeoff_mass_kg),
        ("battery.available_energy_at_bus_wh", available_energy_at_bus_wh),
        ("battery.max_continuous_bus_power_w", max_continuous_bus_power_w),
        ("battery.max_peak_bus_power_w", max_peak_bus_power_w),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(CouplingFinding::InvalidInput {
                field: field.into(),
                reason: "derived sizing result must be finite and non-negative".into(),
            });
        }
    }

    Ok(SizingCalculation {
        required_energy_wh,
        peak_duration_s,
        energy_limited_pack_mass_kg,
        continuous_power_limited_pack_mass_kg,
        peak_power_limited_pack_mass_kg,
        required_pack_mass_kg,
        installed_mass_kg,
        target_takeoff_mass_kg,
        envelope: EnergySystemEnvelope {
            installed_mass_kg: Some(installed_mass_kg),
            usable_energy_at_bus_wh: Some(available_energy_at_bus_wh),
            reserve_fraction: Some(battery.reserve_fraction),
            max_continuous_bus_power_w: Some(max_continuous_bus_power_w),
            max_peak_bus_power_w: Some(max_peak_bus_power_w),
            max_peak_duration_s: Some(battery.max_peak_duration_s),
        },
    })
}

/// Size a battery-electric candidate by iterating the mass ↔ mission-demand
/// coupling. The upstream model is called afresh at each current gross mass;
/// every phase's upper uncertainty bound is used for the arithmetic sizing
/// envelope. Missing evidence, model errors, peak-duration overruns, mass-limit
/// overruns, and non-convergence never return a converged disposition.
///
/// `initial_installed_energy_system_mass_kg` is only an initial guess. It does
/// not count as a claim about the final battery mass.
pub fn solve_mass_coupled_battery<M: MassSensitiveMissionModel>(
    context: &CoupledMissionContext,
    fixed_mass: &FixedMassBreakdown,
    battery: &BatterySizingParameters,
    settings: &CouplingSettings,
    initial_installed_energy_system_mass_kg: f64,
    model: &M,
) -> Result<CoupledSizingReport, Vec<CouplingFinding>> {
    let input_findings = validate_inputs(
        context, fixed_mass, battery, settings, initial_installed_energy_system_mass_kg,
    );
    if !input_findings.is_empty() { return Err(input_findings); }

    let mut current_gross_mass_kg =
        fixed_mass.without_energy_system_kg() + initial_installed_energy_system_mass_kg;
    let mut iteration_trace = Vec::with_capacity(settings.max_iterations.min(256));
    let mut findings = Vec::new();

    for iteration in 1..=settings.max_iterations {
        let phases = match model.phase_demands(context, current_gross_mass_kg) {
            Ok(phases) => phases,
            Err(reason) => {
                findings.push(CouplingFinding::ModelFailure { iteration, reason });
                return Ok(CoupledSizingReport {
                    disposition: CouplingDisposition::ModelFailure,
                    context: context.clone(),
                    iteration_trace,
                    evaluated_gross_mass_kg: Some(current_gross_mass_kg),
                    required_takeoff_mass_kg: None,
                    mass_closure_residual_kg: None,
                    required_pack_mass_kg: None,
                    required_installed_energy_system_mass_kg: None,
                    energy_system_envelope: None,
                    final_phase_demands: Vec::new(),
                    worst_case_mission_energy_wh: None,
                    findings,
                });
            }
        };
        if let Err(phase_findings) = validate_phases(&phases, iteration) {
            let disposition = CouplingDisposition::InvalidDemand;
            findings.extend(phase_findings);
            return Ok(CoupledSizingReport {
                disposition,
                context: context.clone(),
                iteration_trace,
                evaluated_gross_mass_kg: Some(current_gross_mass_kg),
                required_takeoff_mass_kg: None,
                mass_closure_residual_kg: None,
                required_pack_mass_kg: None,
                required_installed_energy_system_mass_kg: None,
                energy_system_envelope: None,
                final_phase_demands: phases,
                worst_case_mission_energy_wh: None,
                findings,
            });
        }

        let sizing = match calculate_sizing(fixed_mass, battery, &phases) {
            Ok(sizing) => sizing,
            Err(finding) => {
                let disposition = if matches!(finding, CouplingFinding::PeakDurationExceeded { .. }) {
                    CouplingDisposition::PeakDurationExceeded
                } else {
                    CouplingDisposition::InvalidDemand
                };
                findings.push(finding);
                return Ok(CoupledSizingReport {
                    disposition,
                    context: context.clone(),
                    iteration_trace,
                    evaluated_gross_mass_kg: Some(current_gross_mass_kg),
                    required_takeoff_mass_kg: None,
                    mass_closure_residual_kg: None,
                    required_pack_mass_kg: None,
                    required_installed_energy_system_mass_kg: None,
                    energy_system_envelope: None,
                    final_phase_demands: phases,
                    worst_case_mission_energy_wh: None,
                    findings,
                });
            }
        };

        let residual_kg = sizing.target_takeoff_mass_kg - current_gross_mass_kg;
        iteration_trace.push(CoupledIteration {
            iteration,
            evaluated_gross_mass_kg: current_gross_mass_kg,
            worst_case_mission_energy_wh: sizing.required_energy_wh,
            peak_duration_s: sizing.peak_duration_s,
            energy_limited_pack_mass_kg: sizing.energy_limited_pack_mass_kg,
            continuous_power_limited_pack_mass_kg: sizing.continuous_power_limited_pack_mass_kg,
            peak_power_limited_pack_mass_kg: sizing.peak_power_limited_pack_mass_kg,
            required_pack_mass_kg: sizing.required_pack_mass_kg,
            required_installed_energy_system_mass_kg: sizing.installed_mass_kg,
            target_takeoff_mass_kg: sizing.target_takeoff_mass_kg,
            mass_residual_kg: residual_kg,
        });

        if sizing.target_takeoff_mass_kg > fixed_mass.maximum_takeoff_mass_limit_kg {
            findings.push(CouplingFinding::TakeoffMassLimitExceeded {
                required_kg: sizing.target_takeoff_mass_kg,
                limit_kg: fixed_mass.maximum_takeoff_mass_limit_kg,
            });
            return Ok(CoupledSizingReport {
                disposition: CouplingDisposition::TakeoffMassLimitExceeded,
                context: context.clone(),
                iteration_trace,
                evaluated_gross_mass_kg: Some(current_gross_mass_kg),
                required_takeoff_mass_kg: Some(sizing.target_takeoff_mass_kg),
                mass_closure_residual_kg: Some(residual_kg),
                required_pack_mass_kg: Some(sizing.required_pack_mass_kg),
                required_installed_energy_system_mass_kg: Some(sizing.installed_mass_kg),
                energy_system_envelope: Some(sizing.envelope),
                final_phase_demands: phases,
                worst_case_mission_energy_wh: Some(sizing.required_energy_wh),
                findings,
            });
        }

        let tolerance_kg = settings.absolute_mass_tolerance_kg
            + settings.relative_mass_tolerance * current_gross_mass_kg.max(sizing.target_takeoff_mass_kg);
        if residual_kg.abs() <= tolerance_kg {
            return Ok(CoupledSizingReport {
                disposition: CouplingDisposition::ConvergedWithinModel,
                context: context.clone(),
                iteration_trace,
                evaluated_gross_mass_kg: Some(current_gross_mass_kg),
                required_takeoff_mass_kg: Some(sizing.target_takeoff_mass_kg),
                mass_closure_residual_kg: Some(residual_kg),
                required_pack_mass_kg: Some(sizing.required_pack_mass_kg),
                required_installed_energy_system_mass_kg: Some(sizing.installed_mass_kg),
                energy_system_envelope: Some(sizing.envelope),
                final_phase_demands: phases,
                worst_case_mission_energy_wh: Some(sizing.required_energy_wh),
                findings,
            });
        }

        let step = settings.relaxation_factor * residual_kg;
        let next_mass = current_gross_mass_kg + step;
        if !next_mass.is_finite() || next_mass <= 0.0 {
            findings.push(CouplingFinding::InvalidInput {
                field: "iteration.next_gross_mass_kg".into(),
                reason: "mass iteration overflowed or became non-positive".into(),
            });
            return Ok(CoupledSizingReport {
                disposition: CouplingDisposition::InvalidDemand,
                context: context.clone(),
                iteration_trace,
                evaluated_gross_mass_kg: Some(current_gross_mass_kg),
                required_takeoff_mass_kg: Some(sizing.target_takeoff_mass_kg),
                mass_closure_residual_kg: Some(residual_kg),
                required_pack_mass_kg: Some(sizing.required_pack_mass_kg),
                required_installed_energy_system_mass_kg: Some(sizing.installed_mass_kg),
                energy_system_envelope: Some(sizing.envelope),
                final_phase_demands: phases,
                worst_case_mission_energy_wh: Some(sizing.required_energy_wh),
                findings,
            });
        }
        current_gross_mass_kg = next_mass;
    }

    findings.push(CouplingFinding::NonConverged { iterations: settings.max_iterations });
    let last = iteration_trace.last();
    Ok(CoupledSizingReport {
        disposition: CouplingDisposition::NonConverged,
        context: context.clone(),
        iteration_trace,
        evaluated_gross_mass_kg: last.map(|row| row.evaluated_gross_mass_kg),
        required_takeoff_mass_kg: last.map(|row| row.target_takeoff_mass_kg),
        mass_closure_residual_kg: last.map(|row| row.mass_residual_kg),
        required_pack_mass_kg: last.map(|row| row.required_pack_mass_kg),
        required_installed_energy_system_mass_kg: last.map(|row| row.required_installed_energy_system_mass_kg),
        energy_system_envelope: None,
        final_phase_demands: Vec::new(),
        worst_case_mission_energy_wh: last.map(|row| row.worst_case_mission_energy_wh),
        findings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SimpleMassSensitiveModel {
        base_power_w: f64,
        watts_per_kg: f64,
        duration_s: f64,
    }

    impl MassSensitiveMissionModel for SimpleMassSensitiveModel {
        fn phase_demands(
            &self,
            _context: &CoupledMissionContext,
            gross_mass_kg: f64,
        ) -> Result<Vec<MissionPhaseDemand>, String> {
            let power = self.base_power_w + self.watts_per_kg * gross_mass_kg;
            Ok(vec![
                MissionPhaseDemand {
                    id: "cruise".into(),
                    duration_s: self.duration_s,
                    required_bus_power_w: power,
                    power_uncertainty: PowerUncertaintyBounds {
                        lower_bound_w: power * 0.95,
                        upper_bound_w: power * 1.05,
                        method: "synthetic-test-bound".into(),
                    },
                    duty_class: DutyClass::Continuous,
                },
                MissionPhaseDemand {
                    id: "climb".into(),
                    duration_s: 120.0,
                    required_bus_power_w: power * 1.4,
                    power_uncertainty: PowerUncertaintyBounds {
                        lower_bound_w: power * 1.3,
                        upper_bound_w: power * 1.5,
                        method: "synthetic-test-bound".into(),
                    },
                    duty_class: DutyClass::Peak,
                },
            ])
        }
    }

    fn context() -> CoupledMissionContext {
        CoupledMissionContext {
            candidate_id: "synthetic-aircraft-a".into(),
            configuration_id: "synthetic-config-r1".into(),
            mission_family_id: "synthetic-mission-v1".into(),
        }
    }

    fn fixed_mass() -> FixedMassBreakdown {
        FixedMassBreakdown {
            fixed_vehicle_mass_kg: 500.0,
            propulsion_installed_mass_kg: 35.0,
            consumables_mass_kg: 25.0,
            payload_mass_kg: 100.0,
            maximum_takeoff_mass_limit_kg: 950.0,
        }
    }

    fn battery() -> BatterySizingParameters {
        BatterySizingParameters {
            pack_specific_energy_wh_per_kg: 300.0,
            usable_fraction: 0.9,
            temperature_capacity_fraction: 0.95,
            end_of_life_soh_fraction: 0.8,
            discharge_efficiency_fraction: 0.95,
            continuous_specific_power_w_per_kg: 1_500.0,
            peak_specific_power_w_per_kg: 2_500.0,
            power_derating_fraction: 0.85,
            reserve_fraction: 0.2,
            installed_system_mass_multiplier: 1.35,
            max_peak_duration_s: 900.0,
        }
    }

    fn settings() -> CouplingSettings {
        CouplingSettings {
            max_iterations: 100,
            absolute_mass_tolerance_kg: 0.01,
            relative_mass_tolerance: 0.00001,
            relaxation_factor: 0.5,
        }
    }

    #[test]
    fn mass_feedback_changes_demand_and_closes_iteration() {
        let model = SimpleMassSensitiveModel {
            base_power_w: 4_000.0,
            watts_per_kg: 1.0,
            duration_s: 1_800.0,
        };
        let report = solve_mass_coupled_battery(
            &context(), &fixed_mass(), &battery(), &settings(), 50.0, &model,
        ).unwrap();
        assert_eq!(report.disposition, CouplingDisposition::ConvergedWithinModel);
        assert!(report.iteration_trace.len() > 1);
        assert!(report.mass_closure_residual_kg.unwrap().abs() <= 0.02);
        assert!(report.final_phase_demands[0].required_bus_power_w > 4_600.0);
        assert!(report.required_installed_energy_system_mass_kg.unwrap() > 0.0);
    }

    #[test]
    fn constant_demand_still_sizes_pack_for_peak_power() {
        let model = SimpleMassSensitiveModel {
            base_power_w: 5_000.0,
            watts_per_kg: 0.0,
            duration_s: 3_600.0,
        };
        let report = solve_mass_coupled_battery(
            &context(), &fixed_mass(), &battery(), &settings(), 50.0, &model,
        ).unwrap();
        assert_eq!(report.disposition, CouplingDisposition::ConvergedWithinModel);
        let trace = report.iteration_trace.last().unwrap();
        assert!(trace.peak_power_limited_pack_mass_kg > 0.0);
        assert!(trace.required_pack_mass_kg >= trace.peak_power_limited_pack_mass_kg);
    }

    #[test]
    fn missing_or_invalid_phases_do_not_pass() {
        struct EmptyModel;
        impl MassSensitiveMissionModel for EmptyModel {
            fn phase_demands(&self, _: &CoupledMissionContext, _: f64) -> Result<Vec<MissionPhaseDemand>, String> {
                Ok(Vec::new())
            }
        }
        let report = solve_mass_coupled_battery(
            &context(), &fixed_mass(), &battery(), &settings(), 50.0, &EmptyModel,
        ).unwrap();
        assert_eq!(report.disposition, CouplingDisposition::InvalidDemand);
        assert!(!report.findings.is_empty());
    }

    #[test]
    fn model_error_is_indeterminate_not_a_pass() {
        struct FailingModel;
        impl MassSensitiveMissionModel for FailingModel {
            fn phase_demands(&self, _: &CoupledMissionContext, _: f64) -> Result<Vec<MissionPhaseDemand>, String> {
                Err("synthetic missing aerodynamic map".into())
            }
        }
        let report = solve_mass_coupled_battery(
            &context(), &fixed_mass(), &battery(), &settings(), 50.0, &FailingModel,
        ).unwrap();
        assert_eq!(report.disposition, CouplingDisposition::ModelFailure);
        assert!(report.energy_system_envelope.is_none());
    }

    #[test]
    fn peak_duration_overrun_is_hard_stop() {
        let model = SimpleMassSensitiveModel {
            base_power_w: 4_000.0,
            watts_per_kg: 0.0,
            duration_s: 3_600.0,
        };
        let mut short_peak_budget = battery();
        short_peak_budget.max_peak_duration_s = 60.0;
        let report = solve_mass_coupled_battery(
            &context(), &fixed_mass(), &short_peak_budget, &settings(), 50.0, &model,
        ).unwrap();
        assert_eq!(report.disposition, CouplingDisposition::PeakDurationExceeded);
    }

    #[test]
    fn takeoff_mass_limit_is_not_hidden_by_energy_margin() {
        let mut mass = fixed_mass();
        mass.maximum_takeoff_mass_limit_kg = 650.0;
        let model = SimpleMassSensitiveModel {
            base_power_w: 4_000.0,
            watts_per_kg: 0.0,
            duration_s: 3_600.0,
        };
        mass.maximum_takeoff_mass_limit_kg = 675.0;
        let report = solve_mass_coupled_battery(
            &context(), &mass, &battery(), &settings(), 10.0, &model,
        ).unwrap();
        assert_eq!(report.disposition, CouplingDisposition::TakeoffMassLimitExceeded);
    }

    #[test]
    fn nonconvergence_is_explicit() {
        let model = SimpleMassSensitiveModel {
            base_power_w: 4_000.0,
            watts_per_kg: 5.0,
            duration_s: 7_200.0,
        };
        let mut short = settings();
        short.max_iterations = 1;
        let report = solve_mass_coupled_battery(
            &context(), &fixed_mass(), &battery(), &short, 10.0, &model,
        ).unwrap();
        assert_eq!(report.disposition, CouplingDisposition::NonConverged);
    }

    #[test]
    fn sizing_uses_declared_upper_power_bounds() {
        let model = SimpleMassSensitiveModel {
            base_power_w: 4_000.0,
            watts_per_kg: 0.0,
            duration_s: 1_800.0,
        };
        let report = solve_mass_coupled_battery(
            &context(), &fixed_mass(), &battery(), &settings(), 50.0, &model,
        ).unwrap();
        let expected_energy = report.final_phase_demands.iter()
            .map(|phase| phase.power_uncertainty.upper_bound_w * phase.duration_s / 3_600.0)
            .sum::<f64>();
        assert!((report.worst_case_mission_energy_wh.unwrap() - expected_energy).abs() < 1e-8);
    }

    #[test]
    fn invalid_reserve_and_mass_terms_fail_closed() {
        let mut bad_battery = battery();
        bad_battery.reserve_fraction = 1.0;
        assert!(solve_mass_coupled_battery(
            &context(), &fixed_mass(), &bad_battery, &settings(), 50.0,
            &SimpleMassSensitiveModel { base_power_w: 4_000.0, watts_per_kg: 0.0, duration_s: 1_800.0 },
        ).is_err());

        let mut bad_mass = fixed_mass();
        bad_mass.payload_mass_kg = f64::NAN;
        assert!(solve_mass_coupled_battery(
            &context(), &bad_mass, &battery(), &settings(), 50.0,
            &SimpleMassSensitiveModel { base_power_w: 4_000.0, watts_per_kg: 0.0, duration_s: 1_800.0 },
        ).is_err());
    }

    #[test]
    fn candidate_and_configuration_identity_are_preserved() {
        let model = SimpleMassSensitiveModel {
            base_power_w: 4_000.0,
            watts_per_kg: 0.0,
            duration_s: 1_800.0,
        };
        let mut ctx = context();
        ctx.candidate_id = "synthetic-aircraft-b".into();
        let report = solve_mass_coupled_battery(
            &ctx, &fixed_mass(), &battery(), &settings(), 50.0, &model,
        ).unwrap();
        assert_eq!(report.context.candidate_id, "synthetic-aircraft-b");
        assert_eq!(report.context.configuration_id, "synthetic-config-r1");
    }
}
