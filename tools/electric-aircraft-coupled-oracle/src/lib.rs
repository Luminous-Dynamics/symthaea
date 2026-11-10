// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent, standard-library-only scalar electro-thermal reference oracle.
//!
//! This package is deliberately separate from the Symthaea mission kernel. It
//! uses a safeguarded bisection solve of a one-node implicit thermal balance,
//! rather than reusing the production thermal-network or mission code.
//!
//! The reference plant freezes OCV at the start-of-substep SOC, models
//! temperature-dependent pack resistance, solves the low-current root of
//! P = I * (OCV - I*R), and solves the implicit thermal balance:
//!
//! C (T_next - T_now) / dt + G (T_next - T_ambient) = I(T_next)^2 R(T_next)
//!
//! SOC is then decremented using the substep chemical energy draw. This is a
//! deliberately small reference problem, not a cell electrochemistry model,
//! aircraft model, battery safety model or certification means of compliance.

use std::fmt;

pub const MODEL_VERSION: &str = "electric-aircraft-coupled-reference-oracle-v0.1.0";
const SECONDS_PER_HOUR: f64 = 3_600.0;
const MAX_SUBSTEPS_PER_SEGMENT: usize = 1_000_000;

/// Scalar plant and numerical bounds for the independent reference problem.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OracleConfig {
    /// Usable pack capacity at the reference condition (Wh).
    pub capacity_wh: f64,
    /// Open-circuit voltage at the SOC reference point (V).
    pub nominal_ocv_v: f64,
    /// Linear OCV slope with SOC (V per unit SOC).
    pub ocv_soc_slope_v: f64,
    pub soc_reference: f64,
    /// Equivalent pack DC resistance at the reference temperature (ohm).
    pub resistance_reference_ohm: f64,
    /// Linear resistance coefficient (1/K); must be nonnegative in v0.1.
    pub resistance_temperature_coefficient_per_k: f64,
    pub reference_temperature_k: f64,
    /// Lumped thermal capacitance (J/K).
    pub thermal_capacity_j_per_k: f64,
    /// Conductance to a fixed ambient boundary (W/K).
    pub thermal_conductance_w_per_k: f64,
    pub ambient_temperature_k: f64,
    pub minimum_temperature_k: f64,
    pub maximum_temperature_k: f64,
    pub reserve_soc: f64,
    pub maximum_continuous_power_w: f64,
    pub maximum_continuous_current_a: f64,
    /// Absolute temperature-bracket width at convergence (K).
    pub temperature_tolerance_k: f64,
    /// Maximum bisection iterations.
    pub maximum_iterations: usize,
    /// Maximum absolute electrical and thermal energy residual (J).
    pub energy_balance_tolerance_j: f64,
}

impl OracleConfig {
    pub fn validate(&self) -> Result<(), OracleError> {
        let positive = [
            self.capacity_wh,
            self.nominal_ocv_v,
            self.resistance_reference_ohm,
            self.reference_temperature_k,
            self.thermal_capacity_j_per_k,
            self.ambient_temperature_k,
            self.minimum_temperature_k,
            self.maximum_temperature_k,
            self.maximum_continuous_power_w,
            self.maximum_continuous_current_a,
            self.temperature_tolerance_k,
            self.energy_balance_tolerance_j,
        ];
        if positive.iter().any(|value| !value.is_finite() || *value <= 0.0) {
            return Err(OracleError::InvalidConfiguration(
                "positive plant and solver values must be finite and greater than zero",
            ));
        }
        if !self.ocv_soc_slope_v.is_finite()
            || !self.soc_reference.is_finite()
            || !self.resistance_temperature_coefficient_per_k.is_finite()
            || self.ocv_soc_slope_v < 0.0
            || !(0.0..=1.0).contains(&self.soc_reference)
            || self.resistance_temperature_coefficient_per_k < 0.0
        {
            return Err(OracleError::InvalidConfiguration(
                "SOC reference and constitutive coefficients are invalid",
            ));
        }
        if !self.thermal_conductance_w_per_k.is_finite()
            || self.thermal_conductance_w_per_k < 0.0
            || !self.reserve_soc.is_finite()
            || !(0.0..1.0).contains(&self.reserve_soc)
            || self.minimum_temperature_k >= self.maximum_temperature_k
            || self.maximum_iterations == 0
        {
            return Err(OracleError::InvalidConfiguration(
                "thermal bounds, reserve, conductance or iteration count are invalid",
            ));
        }
        if self.ambient_temperature_k < self.minimum_temperature_k
            || self.ambient_temperature_k > self.maximum_temperature_k
        {
            return Err(OracleError::InvalidConfiguration(
                "ambient temperature must lie inside the declared temperature domain",
            ));
        }
        Ok(())
    }

    fn open_circuit_voltage_v(&self, soc: f64) -> f64 {
        self.nominal_ocv_v + self.ocv_soc_slope_v * (soc - self.soc_reference)
    }

    fn resistance_ohm(&self, temperature_k: f64) -> f64 {
        self.resistance_reference_ohm
            * (1.0 + self.resistance_temperature_coefficient_per_k
                * (temperature_k - self.reference_temperature_k))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OracleState {
    pub temperature_k: f64,
    pub soc: f64,
}

impl OracleState {
    fn validate(&self, config: &OracleConfig) -> Result<(), OracleError> {
        if !self.temperature_k.is_finite()
            || self.temperature_k < config.minimum_temperature_k
            || self.temperature_k > config.maximum_temperature_k
        {
            return Err(OracleError::InitialTemperatureOutsideDomain {
                temperature_k: self.temperature_k,
                minimum_temperature_k: config.minimum_temperature_k,
                maximum_temperature_k: config.maximum_temperature_k,
            });
        }
        if !self.soc.is_finite() || self.soc < config.reserve_soc || self.soc > 1.0 {
            return Err(OracleError::InitialSocOutsideDomain {
                soc: self.soc,
                reserve_soc: config.reserve_soc,
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubstepInput {
    /// Required battery-terminal power. Demand is never silently reduced.
    pub terminal_power_w: f64,
    pub duration_s: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElectricalOperatingPoint {
    pub open_circuit_voltage_v: f64,
    pub resistance_ohm: f64,
    pub current_a: f64,
    pub terminal_voltage_v: f64,
    pub terminal_power_w: f64,
    pub resistive_heat_w: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubstepReport {
    pub next_state: OracleState,
    pub operating_point: ElectricalOperatingPoint,
    pub terminal_energy_wh: f64,
    pub chemical_energy_wh: f64,
    pub resistive_heat_j: f64,
    pub ambient_rejected_energy_j: f64,
    pub stored_thermal_energy_change_j: f64,
    /// Chemical energy minus terminal energy minus ohmic heat (J).
    pub electrical_energy_balance_residual_j: f64,
    /// Ohmic heat minus ambient rejection minus stored thermal energy (J).
    pub thermal_energy_balance_residual_j: f64,
    pub temperature_iterations: usize,
    pub final_temperature_bracket_width_k: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OracleError {
    InvalidConfiguration(&'static str),
    InvalidSubstep(&'static str),
    InitialTemperatureOutsideDomain {
        temperature_k: f64,
        minimum_temperature_k: f64,
        maximum_temperature_k: f64,
    },
    InitialSocOutsideDomain {
        soc: f64,
        reserve_soc: f64,
    },
    TemperatureBelowDomain {
        lower_temperature_k: f64,
        thermal_residual_w: f64,
    },
    ThermalLimitExceeded {
        maximum_temperature_k: f64,
        thermal_residual_w: f64,
    },
    ElectricalInfeasible {
        temperature_k: f64,
        discriminant_v2: f64,
    },
    ElectroThermalNoEquilibrium {
        maximum_electrically_feasible_temperature_k: f64,
        thermal_residual_w: f64,
    },
    PowerLimitExceeded {
        requested_power_w: f64,
        maximum_power_w: f64,
    },
    CurrentLimitExceeded {
        requested_current_a: f64,
        maximum_current_a: f64,
    },
    ReserveInfringed {
        resulting_soc: f64,
        reserve_soc: f64,
    },
    IterationLimit {
        maximum_iterations: usize,
        bracket_width_k: f64,
        thermal_residual_w: f64,
    },
    EnergyBalanceExceeded {
        electrical_residual_j: f64,
        thermal_residual_j: f64,
        maximum_absolute_residual_j: f64,
    },
    MissionInvalid(&'static str),
    MissionStepFailed {
        segment_id: String,
        substep_index: usize,
        source: Box<OracleError>,
    },
}

impl fmt::Display for OracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for OracleError {}

#[derive(Debug, Clone, PartialEq)]
pub struct MissionSegment {
    pub id: String,
    /// Required propulsion bus power at the battery terminals (W).
    pub propulsion_terminal_power_w: f64,
    /// Separate electrical demand for battery/propulsion thermal management (W).
    /// It is counted against pack energy but is not added to pack I²R heat twice.
    pub thermal_management_power_w: f64,
    pub duration_s: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SegmentSummary {
    pub id: String,
    pub substeps: usize,
    pub propulsion_terminal_energy_wh: f64,
    pub thermal_management_energy_wh: f64,
    pub terminal_energy_wh: f64,
    pub chemical_energy_wh: f64,
    pub resistive_heat_j: f64,
    pub maximum_temperature_k: f64,
    pub final_soc: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MissionReport {
    pub model_version: &'static str,
    pub segments: Vec<SegmentSummary>,
    pub final_state: OracleState,
    pub total_propulsion_terminal_energy_wh: f64,
    pub total_thermal_management_energy_wh: f64,
    pub total_terminal_energy_wh: f64,
    /// Fraction of battery-terminal mission energy consumed by thermal management.
    pub thermal_management_energy_fraction: f64,
    pub total_chemical_energy_wh: f64,
    pub total_resistive_heat_j: f64,
    pub total_ambient_rejected_energy_j: f64,
    pub maximum_temperature_k: f64,
    /// Complete substep history is retained for deterministic replay and audit.
    pub substeps: Vec<SubstepReport>,
}

/// Solve one electro-thermal substep with an independent safeguarded bisection.
///
/// OCV is evaluated at the start-of-substep SOC and held over this short
/// interval. Resistance varies linearly with the implicit end temperature.
/// The lower-current solution is selected for the terminal-power quadratic.
/// The resulting SOC update is committed only after electrical, thermal,
/// reserve and energy-balance checks all pass.
pub fn solve_substep_bisection(
    config: &OracleConfig,
    state: OracleState,
    input: SubstepInput,
) -> Result<SubstepReport, OracleError> {
    config.validate()?;
    state.validate(config)?;
    if !input.terminal_power_w.is_finite()
        || input.terminal_power_w <= 0.0
        || !input.duration_s.is_finite()
        || input.duration_s <= 0.0
    {
        return Err(OracleError::InvalidSubstep(
            "terminal power and duration must be finite and positive",
        ));
    }
    if input.terminal_power_w > config.maximum_continuous_power_w {
        return Err(OracleError::PowerLimitExceeded {
            requested_power_w: input.terminal_power_w,
            maximum_power_w: config.maximum_continuous_power_w,
        });
    }

    let ocv_v = config.open_circuit_voltage_v(state.soc);
    if !ocv_v.is_finite() || ocv_v <= 0.0 {
        return Err(OracleError::InvalidConfiguration(
            "resolved open-circuit voltage must be finite and positive",
        ));
    }

    let residual_at = |temperature_k: f64| -> Result<(f64, ElectricalOperatingPoint), OracleError> {
        let resistance_ohm = config.resistance_ohm(temperature_k);
        if !resistance_ohm.is_finite() || resistance_ohm <= 0.0 {
            return Err(OracleError::InvalidConfiguration(
                "resolved resistance must be finite and positive",
            ));
        }
        let discriminant = ocv_v * ocv_v
            - 4.0 * resistance_ohm * input.terminal_power_w;
        if !discriminant.is_finite() || discriminant < 0.0 {
            return Err(OracleError::ElectricalInfeasible {
                temperature_k,
                discriminant_v2: discriminant,
            });
        }
        // Algebraically equivalent to the small root, but avoids cancellation.
        let current_a = if resistance_ohm <= f64::EPSILON {
            input.terminal_power_w / ocv_v
        } else {
            2.0 * input.terminal_power_w / (ocv_v + discriminant.sqrt())
        };
        let terminal_voltage_v = ocv_v - current_a * resistance_ohm;
        let terminal_power_w = current_a * terminal_voltage_v;
        let resistive_heat_w = current_a * current_a * resistance_ohm;
        let operating_point = ElectricalOperatingPoint {
            open_circuit_voltage_v: ocv_v,
            resistance_ohm,
            current_a,
            terminal_voltage_v,
            terminal_power_w,
            resistive_heat_w,
        };
        if [
            current_a,
            terminal_voltage_v,
            terminal_power_w,
            resistive_heat_w,
        ]
        .iter()
        .any(|value| !value.is_finite())
            || current_a < 0.0
            || terminal_voltage_v <= 0.0
        {
            return Err(OracleError::InvalidSubstep(
                "electrical operating point is non-finite or nonphysical",
            ));
        }

        // Backward-Euler thermal balance, expressed as residual power (W).
        let storage_rate = config.thermal_capacity_j_per_k / input.duration_s;
        let residual_w = storage_rate * (temperature_k - state.temperature_k)
            + config.thermal_conductance_w_per_k
                * (temperature_k - config.ambient_temperature_k)
            - resistive_heat_w;
        if !residual_w.is_finite() {
            return Err(OracleError::InvalidSubstep(
                "thermal residual is non-finite",
            ));
        }
        Ok((residual_w, operating_point))
    };

    let mut lower_k = config.minimum_temperature_k;
    let mut upper_k = config.maximum_temperature_k;
    let (mut lower_residual_w, _) = residual_at(lower_k)?;
    if lower_residual_w > 0.0 {
        return Err(OracleError::TemperatureBelowDomain {
            lower_temperature_k: lower_k,
            thermal_residual_w: lower_residual_w,
        });
    }

    let mut iterations = 0;
    let upper_evaluation = residual_at(upper_k);
    let mut upper_residual_w = match upper_evaluation {
        Ok((residual_w, _)) => residual_w,
        Err(OracleError::ElectricalInfeasible { .. }) => {
            // The electrical power domain can close before the configured
            // thermal ceiling. Find its highest admissible temperature rather
            // than rejecting a mission whose thermal equilibrium lies below it.
            let mut electrically_feasible_k = lower_k;
            let mut electrically_infeasible_k = upper_k;
            while electrically_infeasible_k - electrically_feasible_k
                > config.temperature_tolerance_k
                && iterations < config.maximum_iterations
            {
                let midpoint_k =
                    0.5 * (electrically_feasible_k + electrically_infeasible_k);
                let resistance_ohm = config.resistance_ohm(midpoint_k);
                let discriminant = ocv_v * ocv_v
                    - 4.0 * resistance_ohm * input.terminal_power_w;
                iterations += 1;
                if discriminant.is_finite() && discriminant >= 0.0 {
                    electrically_feasible_k = midpoint_k;
                } else {
                    electrically_infeasible_k = midpoint_k;
                }
            }
            if electrically_infeasible_k - electrically_feasible_k
                > config.temperature_tolerance_k
            {
                let (residual_w, _) = residual_at(electrically_feasible_k)?;
                return Err(OracleError::IterationLimit {
                    maximum_iterations: config.maximum_iterations,
                    bracket_width_k: electrically_infeasible_k - electrically_feasible_k,
                    thermal_residual_w: residual_w,
                });
            }

            upper_k = electrically_feasible_k;
            let (residual_w, _) = residual_at(upper_k)?;
            if residual_w < 0.0 {
                return Err(OracleError::ElectroThermalNoEquilibrium {
                    maximum_electrically_feasible_temperature_k: upper_k,
                    thermal_residual_w: residual_w,
                });
            }
            residual_w
        }
        Err(other) => return Err(other),
    };

    if upper_residual_w < 0.0 {
        return Err(OracleError::ThermalLimitExceeded {
            maximum_temperature_k: upper_k,
            thermal_residual_w: upper_residual_w,
        });
    }

    let mut thermal_iterations = 0;
    while upper_k - lower_k > config.temperature_tolerance_k
        && thermal_iterations < config.maximum_iterations
    {
        let midpoint_k = 0.5 * (lower_k + upper_k);
        let (midpoint_residual_w, _) = residual_at(midpoint_k)?;
        thermal_iterations += 1;
        if midpoint_residual_w > 0.0 {
            upper_k = midpoint_k;
            upper_residual_w = midpoint_residual_w;
        } else {
            lower_k = midpoint_k;
            lower_residual_w = midpoint_residual_w;
        }
    }
    iterations += thermal_iterations;

    let bracket_width_k = upper_k - lower_k;
    if bracket_width_k > config.temperature_tolerance_k {
        let residual_w = if lower_residual_w.abs() < upper_residual_w.abs() {
            lower_residual_w
        } else {
            upper_residual_w
        };
        return Err(OracleError::IterationLimit {
            maximum_iterations: config.maximum_iterations,
            bracket_width_k,
            thermal_residual_w: residual_w,
        });
    }

    let next_temperature_k = 0.5 * (lower_k + upper_k);
    let (thermal_residual_w, operating_point) = residual_at(next_temperature_k)?;
    if operating_point.current_a > config.maximum_continuous_current_a + 1e-9 {
        return Err(OracleError::CurrentLimitExceeded {
            requested_current_a: operating_point.current_a,
            maximum_current_a: config.maximum_continuous_current_a,
        });
    }
    if next_temperature_k > config.maximum_temperature_k {
        return Err(OracleError::ThermalLimitExceeded {
            maximum_temperature_k: config.maximum_temperature_k,
            thermal_residual_w,
        });
    }

    let duration_h = input.duration_s / SECONDS_PER_HOUR;
    let terminal_energy_wh = input.terminal_power_w * duration_h;
    let chemical_energy_wh =
        operating_point.current_a * operating_point.open_circuit_voltage_v * duration_h;
    let resistive_heat_j = operating_point.resistive_heat_w * input.duration_s;
    let electrical_residual_j = chemical_energy_wh * SECONDS_PER_HOUR
        - terminal_energy_wh * SECONDS_PER_HOUR
        - resistive_heat_j;
    let ambient_rejected_energy_j = config.thermal_conductance_w_per_k
        * (next_temperature_k - config.ambient_temperature_k)
        * input.duration_s;
    let stored_thermal_energy_change_j = config.thermal_capacity_j_per_k
        * (next_temperature_k - state.temperature_k);
    let thermal_residual_j =
        resistive_heat_j - ambient_rejected_energy_j - stored_thermal_energy_change_j;

    let next_soc = state.soc - chemical_energy_wh / config.capacity_wh;
    if !next_soc.is_finite() || next_soc < config.reserve_soc - 1e-12 {
        return Err(OracleError::ReserveInfringed {
            resulting_soc: next_soc,
            reserve_soc: config.reserve_soc,
        });
    }

    let maximum_residual_j = electrical_residual_j
        .abs()
        .max(thermal_residual_j.abs());
    if !maximum_residual_j.is_finite()
        || maximum_residual_j > config.energy_balance_tolerance_j
    {
        return Err(OracleError::EnergyBalanceExceeded {
            electrical_residual_j,
            thermal_residual_j,
            maximum_absolute_residual_j: maximum_residual_j,
        });
    }

    Ok(SubstepReport {
        next_state: OracleState {
            temperature_k: next_temperature_k,
            soc: next_soc,
        },
        operating_point,
        terminal_energy_wh,
        chemical_energy_wh,
        resistive_heat_j,
        ambient_rejected_energy_j,
        stored_thermal_energy_change_j,
        electrical_energy_balance_residual_j: electrical_residual_j,
        thermal_energy_balance_residual_j: thermal_residual_j,
        temperature_iterations: iterations,
        final_temperature_bracket_width_k: bracket_width_k,
    })
}

/// Integrate a piecewise-constant terminal-power mission while preserving every
/// successful substep. Any failure aborts the mission and returns its exact
/// segment/substep location; no partial report is returned as a feasible mission.
pub fn simulate_mission(
    config: &OracleConfig,
    initial_state: OracleState,
    segments: &[MissionSegment],
    maximum_step_s: f64,
) -> Result<MissionReport, OracleError> {
    config.validate()?;
    initial_state.validate(config)?;
    if segments.is_empty() || !maximum_step_s.is_finite() || maximum_step_s <= 0.0 {
        return Err(OracleError::MissionInvalid(
            "mission must contain segments and a finite positive maximum step",
        ));
    }

    let mut state = initial_state;
    let mut summaries = Vec::with_capacity(segments.len());
    let mut substeps = Vec::new();
    let mut total_propulsion_terminal_energy_wh = 0.0;
    let mut total_thermal_management_energy_wh = 0.0;
    let mut total_terminal_energy_wh = 0.0;
    let mut total_chemical_energy_wh = 0.0;
    let mut total_resistive_heat_j = 0.0;
    let mut total_ambient_rejected_energy_j = 0.0;
    let mut mission_maximum_temperature_k = initial_state.temperature_k;

    for segment in segments {
        if segment.id.trim().is_empty()
            || !segment.duration_s.is_finite()
            || segment.duration_s <= 0.0
            || !segment.propulsion_terminal_power_w.is_finite()
            || segment.propulsion_terminal_power_w < 0.0
            || !segment.thermal_management_power_w.is_finite()
            || segment.thermal_management_power_w < 0.0
            || !(segment.propulsion_terminal_power_w + segment.thermal_management_power_w).is_finite()
            || segment.propulsion_terminal_power_w + segment.thermal_management_power_w <= 0.0
        {
            return Err(OracleError::MissionInvalid(
                "segment ID, duration and separately declared propulsion / thermal-management powers must be valid",
            ));
        }
        let total_terminal_power_w =
            segment.propulsion_terminal_power_w + segment.thermal_management_power_w;
        let substep_count_f = (segment.duration_s / maximum_step_s).ceil();
        if !substep_count_f.is_finite()
            || substep_count_f < 1.0
            || substep_count_f > MAX_SUBSTEPS_PER_SEGMENT as f64
        {
            return Err(OracleError::MissionInvalid(
                "segment substep count is outside the supported range",
            ));
        }
        let substep_count = substep_count_f as usize;
        let dt_s = segment.duration_s / substep_count as f64;
        let mut segment_propulsion_terminal_energy_wh = 0.0;
        let mut segment_thermal_management_energy_wh = 0.0;
        let mut segment_terminal_energy_wh = 0.0;
        let mut segment_chemical_energy_wh = 0.0;
        let mut segment_resistive_heat_j = 0.0;
        let mut segment_maximum_temperature_k = state.temperature_k;

        for substep_index in 0..substep_count {
            let report = solve_substep_bisection(
                config,
                state,
                SubstepInput {
                    terminal_power_w: total_terminal_power_w,
                    duration_s: dt_s,
                },
            )
            .map_err(|source| OracleError::MissionStepFailed {
                segment_id: segment.id.clone(),
                substep_index,
                source: Box::new(source),
            })?;
            state = report.next_state;
            let propulsion_energy_wh =
                segment.propulsion_terminal_power_w * dt_s / SECONDS_PER_HOUR;
            let thermal_management_energy_wh =
                segment.thermal_management_power_w * dt_s / SECONDS_PER_HOUR;
            segment_propulsion_terminal_energy_wh += propulsion_energy_wh;
            segment_thermal_management_energy_wh += thermal_management_energy_wh;
            segment_terminal_energy_wh += report.terminal_energy_wh;
            segment_chemical_energy_wh += report.chemical_energy_wh;
            segment_resistive_heat_j += report.resistive_heat_j;
            total_propulsion_terminal_energy_wh += propulsion_energy_wh;
            total_thermal_management_energy_wh += thermal_management_energy_wh;
            total_terminal_energy_wh += report.terminal_energy_wh;
            total_chemical_energy_wh += report.chemical_energy_wh;
            total_resistive_heat_j += report.resistive_heat_j;
            total_ambient_rejected_energy_j += report.ambient_rejected_energy_j;
            segment_maximum_temperature_k =
                segment_maximum_temperature_k.max(state.temperature_k);
            mission_maximum_temperature_k =
                mission_maximum_temperature_k.max(state.temperature_k);
            substeps.push(report);
        }

        summaries.push(SegmentSummary {
            id: segment.id.clone(),
            substeps: substep_count,
            propulsion_terminal_energy_wh: segment_propulsion_terminal_energy_wh,
            thermal_management_energy_wh: segment_thermal_management_energy_wh,
            terminal_energy_wh: segment_terminal_energy_wh,
            chemical_energy_wh: segment_chemical_energy_wh,
            resistive_heat_j: segment_resistive_heat_j,
            maximum_temperature_k: segment_maximum_temperature_k,
            final_soc: state.soc,
        });
    }

    Ok(MissionReport {
        model_version: MODEL_VERSION,
        segments: summaries,
        final_state: state,
        total_propulsion_terminal_energy_wh,
        total_thermal_management_energy_wh,
        total_terminal_energy_wh,
        thermal_management_energy_fraction: total_thermal_management_energy_wh
            / total_terminal_energy_wh,
        total_chemical_energy_wh,
        total_resistive_heat_j,
        total_ambient_rejected_energy_j,
        maximum_temperature_k: mission_maximum_temperature_k,
        substeps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> OracleConfig {
        OracleConfig {
            capacity_wh: 40_000.0,
            nominal_ocv_v: 400.0,
            ocv_soc_slope_v: 0.0,
            soc_reference: 0.5,
            resistance_reference_ohm: 0.04,
            resistance_temperature_coefficient_per_k: 0.0,
            reference_temperature_k: 298.15,
            thermal_capacity_j_per_k: 80_000.0,
            thermal_conductance_w_per_k: 100.0,
            ambient_temperature_k: 303.15,
            minimum_temperature_k: 273.15,
            maximum_temperature_k: 353.15,
            reserve_soc: 0.20,
            maximum_continuous_power_w: 150_000.0,
            maximum_continuous_current_a: 500.0,
            temperature_tolerance_k: 1e-9,
            maximum_iterations: 80,
            energy_balance_tolerance_j: 1e-3,
        }
    }

    fn initial_state() -> OracleState {
        OracleState {
            temperature_k: 298.15,
            soc: 0.95,
        }
    }

    #[test]
    fn constant_resistance_matches_independent_closed_form_solution() {
        let model = config();
        let state = initial_state();
        let input = SubstepInput {
            terminal_power_w: 40_000.0,
            duration_s: 60.0,
        };
        let report = solve_substep_bisection(&model, state, input).unwrap();

        let ocv = model.nominal_ocv_v;
        let resistance = model.resistance_reference_ohm;
        let discriminant = ocv * ocv - 4.0 * resistance * input.terminal_power_w;
        let expected_current =
            2.0 * input.terminal_power_w / (ocv + discriminant.sqrt());
        let expected_heat_w = expected_current * expected_current * resistance;
        let storage_rate = model.thermal_capacity_j_per_k / input.duration_s;
        let expected_temperature = (storage_rate * state.temperature_k
            + model.thermal_conductance_w_per_k * model.ambient_temperature_k
            + expected_heat_w)
            / (storage_rate + model.thermal_conductance_w_per_k);

        assert!((report.operating_point.current_a - expected_current).abs() < 1e-10);
        assert!((report.next_state.temperature_k - expected_temperature).abs() < 2e-8);
        assert!(report.electrical_energy_balance_residual_j.abs() < 1e-8);
        assert!(report.thermal_energy_balance_residual_j.abs() < 1e-3);
    }

    #[test]
    fn temperature_resistance_feedback_increases_current_and_heat() {
        let state = initial_state();
        let input = SubstepInput {
            terminal_power_w: 100_000.0,
            duration_s: 30.0,
        };
        let feedback_off = solve_substep_bisection(&config(), state, input).unwrap();

        let mut feedback_on = config();
        feedback_on.resistance_temperature_coefficient_per_k = 0.015;
        let feedback_enabled = solve_substep_bisection(&feedback_on, state, input).unwrap();

        assert!(feedback_enabled.next_state.temperature_k > feedback_off.next_state.temperature_k);
        assert!(feedback_enabled.operating_point.current_a > feedback_off.operating_point.current_a);
        assert!(feedback_enabled.resistive_heat_j > feedback_off.resistive_heat_j);
    }

    #[test]
    fn no_real_battery_current_root_is_electrical_infeasibility() {
        let mut model = config();
        model.nominal_ocv_v = 300.0;
        model.resistance_reference_ohm = 0.20;
        model.resistance_temperature_coefficient_per_k = 0.001;
        assert!(matches!(
            solve_substep_bisection(
                &model,
                initial_state(),
                SubstepInput { terminal_power_w: 120_000.0, duration_s: 10.0 }
            ),
            Err(OracleError::ElectricalInfeasible { .. })
        ));
    }

    #[test]
    fn lower_temperature_equilibrium_survives_an_invalid_hot_domain_tail() {
        let mut model = config();
        model.resistance_reference_ohm = 0.10;
        model.resistance_temperature_coefficient_per_k = 0.035;
        let report = solve_substep_bisection(
            &model,
            initial_state(),
            SubstepInput { terminal_power_w: 140_000.0, duration_s: 10.0 },
        )
        .unwrap();

        assert!(report.next_state.temperature_k > initial_state().temperature_k);
        assert!(report.next_state.temperature_k < model.maximum_temperature_k);
        assert!(report.operating_point.terminal_voltage_v > 0.0);
    }

    #[test]
    fn no_equilibrium_before_electrical_power_domain_closes_is_distinct() {
        let mut model = config();
        model.resistance_reference_ohm = 0.10;
        model.resistance_temperature_coefficient_per_k = 0.035;
        model.thermal_capacity_j_per_k = 1_000.0;
        model.thermal_conductance_w_per_k = 0.0;
        assert!(matches!(
            solve_substep_bisection(
                &model,
                initial_state(),
                SubstepInput { terminal_power_w: 140_000.0, duration_s: 600.0 }
            ),
            Err(OracleError::ElectroThermalNoEquilibrium { .. })
        ));
    }

    #[test]
    fn excess_heat_is_reported_as_thermal_limit_not_clipped_temperature() {
        let mut model = config();
        model.thermal_capacity_j_per_k = 1_000.0;
        model.thermal_conductance_w_per_k = 5.0;
        model.maximum_temperature_k = 310.0;
        assert!(matches!(
            solve_substep_bisection(
                &model,
                initial_state(),
                SubstepInput { terminal_power_w: 35_000.0, duration_s: 60.0 }
            ),
            Err(OracleError::ThermalLimitExceeded { .. })
        ));
    }

    #[test]
    fn reserve_infringement_is_distinct_from_electrical_and_thermal_failure() {
        let mut model = config();
        model.capacity_wh = 1_000.0;
        assert!(matches!(
            solve_substep_bisection(
                &model,
                initial_state(),
                SubstepInput { terminal_power_w: 20_000.0, duration_s: 3_600.0 }
            ),
            Err(OracleError::ReserveInfringed { .. })
        ));
    }

    #[test]
    fn iteration_limit_returns_residual_and_bracket_not_a_result() {
        let mut model = config();
        model.maximum_iterations = 1;
        model.temperature_tolerance_k = 1e-12;
        assert!(matches!(
            solve_substep_bisection(
                &model,
                initial_state(),
                SubstepInput { terminal_power_w: 70_000.0, duration_s: 300.0 }
            ),
            Err(OracleError::IterationLimit { maximum_iterations: 1, .. })
        ));
    }

    #[test]
    fn current_and_terminal_power_limits_are_explicit() {
        let mut power_limited = config();
        power_limited.maximum_continuous_power_w = 10_000.0;
        assert!(matches!(
            solve_substep_bisection(
                &power_limited,
                initial_state(),
                SubstepInput { terminal_power_w: 20_000.0, duration_s: 10.0 }
            ),
            Err(OracleError::PowerLimitExceeded { .. })
        ));

        let mut current_limited = config();
        current_limited.maximum_continuous_current_a = 10.0;
        assert!(matches!(
            solve_substep_bisection(
                &current_limited,
                initial_state(),
                SubstepInput { terminal_power_w: 20_000.0, duration_s: 10.0 }
            ),
            Err(OracleError::CurrentLimitExceeded { .. })
        ));
    }

    #[test]
    fn out_of_domain_initial_state_is_rejected() {
        let mut state = initial_state();
        state.temperature_k = 260.0;
        assert!(matches!(
            solve_substep_bisection(
                &config(),
                state,
                SubstepInput { terminal_power_w: 10_000.0, duration_s: 10.0 }
            ),
            Err(OracleError::InitialTemperatureOutsideDomain { .. })
        ));
    }

    #[test]
    fn mission_replay_preserves_segments_and_exact_substep_history() {
        let mission = vec![
            MissionSegment { id: "climb".into(), propulsion_terminal_power_w: 80_000.0, thermal_management_power_w: 0.0, duration_s: 120.0 },
            MissionSegment { id: "cruise".into(), propulsion_terminal_power_w: 35_000.0, thermal_management_power_w: 0.0, duration_s: 300.0 },
        ];
        let report = simulate_mission(&config(), initial_state(), &mission, 10.0).unwrap();

        assert_eq!(report.segments.len(), 2);
        assert_eq!(report.segments[0].substeps, 12);
        assert_eq!(report.segments[1].substeps, 30);
        assert_eq!(report.substeps.len(), 42);
        assert_eq!(report.model_version, MODEL_VERSION);
        assert!(report.total_chemical_energy_wh > report.total_terminal_energy_wh);
        assert!(report.maximum_temperature_k > initial_state().temperature_k);
    }

    #[test]
    fn thermal_management_energy_is_separately_charged_to_the_mission() {
        let mission = vec![MissionSegment {
            id: "hot-climb".into(),
            propulsion_terminal_power_w: 80_000.0,
            thermal_management_power_w: 20_000.0,
            duration_s: 120.0,
        }];
        let report = simulate_mission(&config(), initial_state(), &mission, 10.0).unwrap();

        assert!((report.total_propulsion_terminal_energy_wh - 2_666.666_666_7).abs() < 1e-6);
        assert!((report.total_thermal_management_energy_wh - 666.666_666_7).abs() < 1e-6);
        assert!((report.total_terminal_energy_wh - 3_333.333_333_3).abs() < 1e-6);
        assert!((report.thermal_management_energy_fraction - 0.20).abs() < 1e-10);
        assert!(
            (report.total_propulsion_terminal_energy_wh
                + report.total_thermal_management_energy_wh
                - report.total_terminal_energy_wh)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn timestep_refinement_converges_for_the_frozen_synthetic_mission() {
        let mission = vec![
            MissionSegment { id: "climb".into(), propulsion_terminal_power_w: 100_000.0, thermal_management_power_w: 0.0, duration_s: 120.0 },
            MissionSegment { id: "cruise".into(), propulsion_terminal_power_w: 40_000.0, thermal_management_power_w: 0.0, duration_s: 600.0 },
        ];
        let coarse = simulate_mission(&config(), initial_state(), &mission, 10.0).unwrap();
        let fine = simulate_mission(&config(), initial_state(), &mission, 1.0).unwrap();

        assert!((coarse.final_state.temperature_k - fine.final_state.temperature_k).abs() < 0.03);
        assert!((coarse.final_state.soc - fine.final_state.soc).abs() < 2e-4);
        assert!((coarse.total_chemical_energy_wh - fine.total_chemical_energy_wh).abs() < 0.01);
    }

    #[test]
    fn a_failed_substep_does_not_return_a_partial_mission_report() {
        let mission = vec![
            MissionSegment { id: "cruise".into(), propulsion_terminal_power_w: 10_000.0, thermal_management_power_w: 0.0, duration_s: 30.0 },
            MissionSegment { id: "impossible-climb".into(), propulsion_terminal_power_w: 200_000.0, thermal_management_power_w: 0.0, duration_s: 30.0 },
        ];
        assert!(matches!(
            simulate_mission(&config(), initial_state(), &mission, 10.0),
            Err(OracleError::MissionStepFailed { segment_id, .. })
                if segment_id == "impossible-climb"
        ));
    }
}
