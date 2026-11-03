#![forbid(unsafe_code)]

//! First-principles engineering primitives for the InstantClean dishwasher
//! research programme.
//!
//! This crate models measurable engineering quantities only: sensible heating,
//! electrical resource budgets, nozzle hydraulics and multi-objective candidate
//! comparison. It does not predict soil removal, microbiological safety, spray
//! breakup or CFD. Those require calibrated experiments and validated solvers.

pub mod chemistry;
pub mod quality;
pub mod service;
pub mod system_window;
pub mod validation;

pub mod energy {
    pub const WATER_CP_KJ_PER_KG_K: f64 = 4.186;
    const KJ_PER_KWH: f64 = 3_600.0;
    const WATT_SECONDS_PER_KWH: f64 = 3_600_000.0;

    /// Lumped thermal load for one wash stage. Water mass uses 1 L ~= 1 kg.
    /// Effective solid mass is the measured heat-coupled fraction of the load.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct ThermalLoad {
        pub process_water_l: f64,
        pub inlet_temp_c: f64,
        pub target_temp_c: f64,
        pub effective_solid_mass_kg: f64,
        pub solid_cp_kj_per_kg_k: f64,
        pub heater_efficiency: f64,
        pub recovered_heat_fraction: f64,
        pub heater_power_w: f64,
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct ThermalEstimate {
        pub sensible_heat_kwh: f64,
        pub heater_electricity_kwh: f64,
        pub ideal_heat_up_seconds: f64,
        pub temperature_rise_k: f64,
    }

    fn finite_non_negative(value: f64) -> bool {
        value.is_finite() && value >= 0.0
    }

    /// Ideal sensible-heat requirement before cabinet, pipe, evaporation and
    /// other real-world losses.
    pub fn estimate_thermal_load(load: ThermalLoad) -> Result<ThermalEstimate, &'static str> {
        if !finite_non_negative(load.process_water_l)
            || !finite_non_negative(load.effective_solid_mass_kg)
            || !finite_non_negative(load.solid_cp_kj_per_kg_k)
            || !load.inlet_temp_c.is_finite()
            || !load.target_temp_c.is_finite()
        {
            return Err("thermal quantities must be finite and non-negative where applicable");
        }
        if !(load.heater_efficiency.is_finite()
            && load.heater_efficiency > 0.0
            && load.heater_efficiency <= 1.0)
        {
            return Err("heater_efficiency must be in (0, 1]");
        }
        if !(load.recovered_heat_fraction.is_finite()
            && (0.0..=1.0).contains(&load.recovered_heat_fraction))
        {
            return Err("recovered_heat_fraction must be in [0, 1]");
        }
        if !finite_non_negative(load.heater_power_w) {
            return Err("heater_power_w must be finite and non-negative");
        }

        let delta_t = (load.target_temp_c - load.inlet_temp_c).max(0.0);
        let water_heat_kj = load.process_water_l * WATER_CP_KJ_PER_KG_K * delta_t;
        let solid_heat_kj =
            load.effective_solid_mass_kg * load.solid_cp_kj_per_kg_k * delta_t;
        let sensible_heat_kwh = (water_heat_kj + solid_heat_kj) / KJ_PER_KWH;
        let heater_electricity_kwh = sensible_heat_kwh
            * (1.0 - load.recovered_heat_fraction)
            / load.heater_efficiency;

        let ideal_heat_up_seconds = if heater_electricity_kwh == 0.0 {
            0.0
        } else if load.heater_power_w > 0.0 {
            heater_electricity_kwh * WATT_SECONDS_PER_KWH / load.heater_power_w
        } else {
            return Err("heater_power_w must be positive when heat is required");
        };

        if !sensible_heat_kwh.is_finite()
            || !heater_electricity_kwh.is_finite()
            || !ideal_heat_up_seconds.is_finite()
        {
            return Err("derived thermal result overflowed the supported numeric range");
        }

        Ok(ThermalEstimate {
            sensible_heat_kwh,
            heater_electricity_kwh,
            ideal_heat_up_seconds,
            temperature_rise_k: delta_t,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn water_only(volume_l: f64, start_c: f64, target_c: f64) -> ThermalLoad {
            ThermalLoad {
                process_water_l: volume_l,
                inlet_temp_c: start_c,
                target_temp_c: target_c,
                effective_solid_mass_kg: 0.0,
                solid_cp_kj_per_kg_k: 0.84,
                heater_efficiency: 1.0,
                recovered_heat_fraction: 0.0,
                heater_power_w: 1_000.0,
            }
        }

        #[test]
        fn one_litre_water_20_to_60c_matches_sensible_heat_equation() {
            let estimate = estimate_thermal_load(water_only(1.0, 20.0, 60.0)).unwrap();
            assert!((estimate.sensible_heat_kwh - 0.046_511_1).abs() < 1e-6);
        }

        #[test]
        fn no_heat_when_inlet_is_already_hotter_than_target() {
            let estimate = estimate_thermal_load(water_only(1.0, 65.0, 55.0)).unwrap();
            assert_eq!(estimate.sensible_heat_kwh, 0.0);
            assert_eq!(estimate.ideal_heat_up_seconds, 0.0);
        }

        #[test]
        fn recovered_heat_reduces_electrical_input_not_thermal_demand() {
            let mut load = water_only(1.0, 20.0, 55.0);
            let no_recovery = estimate_thermal_load(load).unwrap();
            load.recovered_heat_fraction = 0.5;
            let recovered = estimate_thermal_load(load).unwrap();
            assert_eq!(no_recovery.sensible_heat_kwh, recovered.sensible_heat_kwh);
            assert!((recovered.heater_electricity_kwh
                - no_recovery.heater_electricity_kwh * 0.5)
                .abs()
                < 1e-12);
        }

        #[test]
        fn rejects_zero_heater_efficiency() {
            let mut load = water_only(1.0, 20.0, 55.0);
            load.heater_efficiency = 0.0;
            assert!(estimate_thermal_load(load).is_err());
        }
    }
}

pub mod cycle {
    use super::energy::{estimate_thermal_load, ThermalEstimate, ThermalLoad};

    const WATT_SECONDS_PER_KWH: f64 = 3_600_000.0;

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct AuxiliaryLoads {
        pub pump_power_w: f64,
        pub pump_runtime_s: f64,
        pub controls_power_w: f64,
        pub controls_runtime_s: f64,
        pub drying_energy_kwh: f64,
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct CycleInput {
        pub thermal_load: ThermalLoad,
        pub auxiliary: AuxiliaryLoads,
        /// Fresh water consumed, distinct from recirculated process-water volume.
        pub fresh_water_l: f64,
        pub cycle_duration_s: f64,
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct CycleBudget {
        pub thermal: ThermalEstimate,
        pub pump_energy_kwh: f64,
        pub controls_energy_kwh: f64,
        pub drying_energy_kwh: f64,
        pub total_electricity_kwh: f64,
        pub fresh_water_l: f64,
        pub cycle_duration_s: f64,
        pub heater_can_finish_within_cycle: bool,
    }

    fn valid_non_negative(value: f64) -> bool {
        value.is_finite() && value >= 0.0
    }

    pub fn estimate_cycle(input: CycleInput) -> Result<CycleBudget, &'static str> {
        let a = input.auxiliary;
        if !valid_non_negative(a.pump_power_w)
            || !valid_non_negative(a.pump_runtime_s)
            || !valid_non_negative(a.controls_power_w)
            || !valid_non_negative(a.controls_runtime_s)
            || !valid_non_negative(a.drying_energy_kwh)
            || !valid_non_negative(input.fresh_water_l)
            || !valid_non_negative(input.cycle_duration_s)
        {
            return Err("cycle resource inputs must be finite and non-negative");
        }

        let thermal = estimate_thermal_load(input.thermal_load)?;
        let pump_energy_kwh = a.pump_power_w * a.pump_runtime_s / WATT_SECONDS_PER_KWH;
        let controls_energy_kwh =
            a.controls_power_w * a.controls_runtime_s / WATT_SECONDS_PER_KWH;
        let total_electricity_kwh = thermal.heater_electricity_kwh
            + pump_energy_kwh
            + controls_energy_kwh
            + a.drying_energy_kwh;

        if !pump_energy_kwh.is_finite()
            || !controls_energy_kwh.is_finite()
            || !total_electricity_kwh.is_finite()
        {
            return Err("derived cycle energy overflowed the supported numeric range");
        }

        Ok(CycleBudget {
            thermal,
            pump_energy_kwh,
            controls_energy_kwh,
            drying_energy_kwh: a.drying_energy_kwh,
            total_electricity_kwh,
            fresh_water_l: input.fresh_water_l,
            cycle_duration_s: input.cycle_duration_s,
            heater_can_finish_within_cycle: thermal.ideal_heat_up_seconds
                <= input.cycle_duration_s,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn fresh_water_is_separate_from_recirculated_charge() {
            let input = CycleInput {
                thermal_load: ThermalLoad {
                    process_water_l: 0.5,
                    inlet_temp_c: 20.0,
                    target_temp_c: 55.0,
                    effective_solid_mass_kg: 0.2,
                    solid_cp_kj_per_kg_k: 0.84,
                    heater_efficiency: 0.92,
                    recovered_heat_fraction: 0.0,
                    heater_power_w: 900.0,
                },
                auxiliary: AuxiliaryLoads {
                    pump_power_w: 75.0,
                    pump_runtime_s: 120.0,
                    controls_power_w: 6.0,
                    controls_runtime_s: 150.0,
                    drying_energy_kwh: 0.0,
                },
                fresh_water_l: 0.6,
                cycle_duration_s: 150.0,
            };
            let budget = estimate_cycle(input).unwrap();
            assert_eq!(budget.fresh_water_l, 0.6);
            assert!(budget.thermal.sensible_heat_kwh > 0.0);
            assert!(budget.total_electricity_kwh > budget.thermal.heater_electricity_kwh);
        }

        #[test]
        fn flags_heater_that_cannot_reach_temperature_in_cycle_window() {
            let input = CycleInput {
                thermal_load: ThermalLoad {
                    process_water_l: 2.0,
                    inlet_temp_c: 20.0,
                    target_temp_c: 60.0,
                    effective_solid_mass_kg: 0.0,
                    solid_cp_kj_per_kg_k: 0.84,
                    heater_efficiency: 0.9,
                    recovered_heat_fraction: 0.0,
                    heater_power_w: 300.0,
                },
                auxiliary: AuxiliaryLoads {
                    pump_power_w: 0.0,
                    pump_runtime_s: 0.0,
                    controls_power_w: 0.0,
                    controls_runtime_s: 0.0,
                    drying_energy_kwh: 0.0,
                },
                fresh_water_l: 2.0,
                cycle_duration_s: 60.0,
            };
            assert!(!estimate_cycle(input).unwrap().heater_can_finish_within_cycle);
        }
    }
}

pub mod hydraulics {
    use symthaea_thermofluids::fluids::{flow_regime, reynolds_number, Regime};

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct NozzleOperatingPoint {
        pub orifice_diameter_m: f64,
        pub volumetric_flow_m3_s: f64,
        pub pressure_drop_pa: f64,
        pub fluid_density_kg_m3: f64,
        pub dynamic_viscosity_pa_s: f64,
        pub pump_efficiency: f64,
    }

    /// Single-phase bulk-flow estimates only; no spray atomization, cavitation,
    /// breakup, wall impact or impact-cleaning model is implied.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct NozzleEstimate {
        pub exit_velocity_m_s: f64,
        pub reynolds_number: f64,
        pub flow_regime: Regime,
        pub dynamic_pressure_pa: f64,
        pub hydraulic_power_w: f64,
        pub estimated_electrical_pump_power_w: f64,
    }

    pub fn estimate_nozzle(point: NozzleOperatingPoint) -> Result<NozzleEstimate, &'static str> {
        let positive = [
            point.orifice_diameter_m,
            point.fluid_density_kg_m3,
            point.dynamic_viscosity_pa_s,
        ];
        if positive.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err("diameter, density and viscosity must be finite and positive");
        }
        if !point.volumetric_flow_m3_s.is_finite() || point.volumetric_flow_m3_s < 0.0 {
            return Err("volumetric flow must be finite and non-negative");
        }
        if !point.pressure_drop_pa.is_finite() || point.pressure_drop_pa < 0.0 {
            return Err("pressure drop must be finite and non-negative");
        }
        if !(point.pump_efficiency.is_finite()
            && point.pump_efficiency > 0.0
            && point.pump_efficiency <= 1.0)
        {
            return Err("pump efficiency must be in (0, 1]");
        }

        let area_m2 = std::f64::consts::PI * point.orifice_diameter_m.powi(2) / 4.0;
        let exit_velocity_m_s = point.volumetric_flow_m3_s / area_m2;
        let re = reynolds_number(
            point.fluid_density_kg_m3,
            exit_velocity_m_s,
            point.orifice_diameter_m,
            point.dynamic_viscosity_pa_s,
        );
        let hydraulic_power_w = point.pressure_drop_pa * point.volumetric_flow_m3_s;
        let dynamic_pressure_pa =
            0.5 * point.fluid_density_kg_m3 * exit_velocity_m_s.powi(2);
        let estimated_electrical_pump_power_w = hydraulic_power_w / point.pump_efficiency;

        if !exit_velocity_m_s.is_finite()
            || !re.is_finite()
            || !dynamic_pressure_pa.is_finite()
            || !hydraulic_power_w.is_finite()
            || !estimated_electrical_pump_power_w.is_finite()
        {
            return Err("derived nozzle result overflowed the supported numeric range");
        }

        Ok(NozzleEstimate {
            exit_velocity_m_s,
            reynolds_number: re,
            flow_regime: flow_regime(re),
            dynamic_pressure_pa,
            hydraulic_power_w,
            estimated_electrical_pump_power_w,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn computes_velocity_and_pump_power_from_measured_flow() {
            let estimate = estimate_nozzle(NozzleOperatingPoint {
                orifice_diameter_m: 0.002,
                volumetric_flow_m3_s: 1.0e-5,
                pressure_drop_pa: 100_000.0,
                fluid_density_kg_m3: 1_000.0,
                dynamic_viscosity_pa_s: 1.0e-3,
                pump_efficiency: 0.5,
            })
            .unwrap();
            let expected_velocity =
                1.0e-5 / (std::f64::consts::PI * 0.002_f64.powi(2) / 4.0);
            assert!((estimate.exit_velocity_m_s - expected_velocity).abs() < 1e-9);
            assert!((estimate.hydraulic_power_w - 1.0).abs() < 1e-12);
            assert!((estimate.estimated_electrical_pump_power_w - 2.0).abs() < 1e-12);
        }

        #[test]
        fn rejects_zero_diameter() {
            let point = NozzleOperatingPoint {
                orifice_diameter_m: 0.0,
                volumetric_flow_m3_s: 1.0e-5,
                pressure_drop_pa: 100_000.0,
                fluid_density_kg_m3: 1_000.0,
                dynamic_viscosity_pa_s: 1.0e-3,
                pump_efficiency: 0.5,
            };
            assert!(estimate_nozzle(point).is_err());
        }
    }
}


/// Parallel hydraulic-network model using a measured quadratic pump curve and
/// measured/equivalent quadratic branch resistances.
///
/// This is a lumped-parameter model, not CFD. It is suitable for early design
/// screening when its pump curve and branch resistances are measured. It does
/// not model rotating-arm momentum, air entrainment, cavitation or spray breakup.
pub mod hydraulic_network {
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct PumpCurve {
        /// Shutoff pressure at zero flow.
        pub shutoff_pressure_pa: f64,
        /// Quadratic pump-curve coefficient: pressure = shutoff - k * Q_total^2.
        pub pressure_coefficient_pa_per_flow_squared: f64,
        /// Overall hydraulic efficiency, strictly between zero and one.
        pub efficiency: f64,
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct Branch {
        /// Equivalent pressure-loss coefficient: delta_p = R * q_branch^2.
        pub resistance_pa_per_flow_squared: f64,
        pub open: bool,
    }

    #[derive(Debug, Clone, PartialEq)]
    pub struct NetworkEstimate {
        pub operating_pressure_pa: f64,
        pub total_flow_m3_s: f64,
        pub branch_flows_m3_s: Vec<f64>,
        pub hydraulic_power_w: f64,
        pub estimated_electrical_pump_power_w: f64,
        pub iterations: u32,
    }

    fn total_flow_at_pressure(pressure_pa: f64, branches: &[Branch]) -> f64 {
        branches
            .iter()
            .filter(|branch| branch.open)
            .map(|branch| {
                (pressure_pa / branch.resistance_pa_per_flow_squared).sqrt()
            })
            .sum()
    }

    /// Solve the pump/system intersection with monotone bisection.
    ///
    /// A branch resistance should be obtained from measured pressure/flow data
    /// or a separately verified component model. One closed-loop network with
    /// at least one open branch is required; dead-heading is rejected.
    pub fn solve_parallel_network(
        pump: PumpCurve,
        branches: &[Branch],
        pressure_tolerance_pa: f64,
        max_iterations: u32,
    ) -> Result<NetworkEstimate, &'static str> {
        if !(pump.shutoff_pressure_pa.is_finite() && pump.shutoff_pressure_pa > 0.0) {
            return Err("pump shutoff pressure must be finite and positive");
        }
        if !(pump.pressure_coefficient_pa_per_flow_squared.is_finite()
            && pump.pressure_coefficient_pa_per_flow_squared >= 0.0)
        {
            return Err("pump pressure coefficient must be finite and non-negative");
        }
        if !(pump.efficiency.is_finite() && pump.efficiency > 0.0 && pump.efficiency <= 1.0) {
            return Err("pump efficiency must be in (0, 1]");
        }
        if branches.is_empty() || !branches.iter().any(|branch| branch.open) {
            return Err("at least one hydraulic branch must be open");
        }
        if branches.iter().any(|branch| {
            !branch.resistance_pa_per_flow_squared.is_finite()
                || branch.resistance_pa_per_flow_squared <= 0.0
        }) {
            return Err("every branch resistance must be finite and positive");
        }
        if !(pressure_tolerance_pa.is_finite() && pressure_tolerance_pa > 0.0) {
            return Err("pressure tolerance must be finite and positive");
        }
        if max_iterations == 0 {
            return Err("max_iterations must be positive");
        }

        // System pressure is in [0, shutoff_pressure]. The residual
        // f(p) = p + k * Q(p)^2 - p_shutoff is monotone increasing.
        let mut low = 0.0;
        let mut high = pump.shutoff_pressure_pa;
        let mut iterations = 0;

        if pump.pressure_coefficient_pa_per_flow_squared == 0.0 {
            // A perfectly flat ideal pump curve holds its shutoff pressure
            // independent of flow in this simplified model.
            low = high;
        } else {
            while (high - low) > pressure_tolerance_pa && iterations < max_iterations {
                let midpoint = low + (high - low) * 0.5;
                let flow = total_flow_at_pressure(midpoint, branches);
                let residual = midpoint
                    + pump.pressure_coefficient_pa_per_flow_squared * flow * flow
                    - pump.shutoff_pressure_pa;
                if !residual.is_finite() {
                    return Err("hydraulic network residual is non-finite");
                }
                if residual > 0.0 {
                    high = midpoint;
                } else {
                    low = midpoint;
                }
                iterations += 1;
            }
            if (high - low) > pressure_tolerance_pa {
                return Err("hydraulic network solver did not converge within iteration limit");
            }
        }

        let operating_pressure_pa = low + (high - low) * 0.5;
        let branch_flows_m3_s: Vec<f64> = branches
            .iter()
            .map(|branch| {
                if branch.open {
                    (operating_pressure_pa / branch.resistance_pa_per_flow_squared).sqrt()
                } else {
                    0.0
                }
            })
            .collect();
        let total_flow_m3_s: f64 = branch_flows_m3_s.iter().sum();
        let hydraulic_power_w = operating_pressure_pa * total_flow_m3_s;
        let estimated_electrical_pump_power_w = hydraulic_power_w / pump.efficiency;

        if !operating_pressure_pa.is_finite()
            || !total_flow_m3_s.is_finite()
            || !hydraulic_power_w.is_finite()
            || !estimated_electrical_pump_power_w.is_finite()
            || branch_flows_m3_s.iter().any(|flow| !flow.is_finite())
        {
            return Err("derived hydraulic-network result is non-finite");
        }

        Ok(NetworkEstimate {
            operating_pressure_pa,
            total_flow_m3_s,
            branch_flows_m3_s,
            hydraulic_power_w,
            estimated_electrical_pump_power_w,
            iterations,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn single_branch_matches_closed_form_pump_system_intersection() {
            let estimate = solve_parallel_network(
                PumpCurve {
                    shutoff_pressure_pa: 100_000.0,
                    pressure_coefficient_pa_per_flow_squared: 1.0e12,
                    efficiency: 0.5,
                },
                &[Branch {
                    resistance_pa_per_flow_squared: 1.0e12,
                    open: true,
                }],
                1.0e-5,
                128,
            )
            .unwrap();

            // With equal pump and branch coefficients, p = p_shutoff / 2.
            assert!((estimate.operating_pressure_pa - 50_000.0).abs() < 1e-3);
            assert!((estimate.total_flow_m3_s - 0.000_223_606_8).abs() < 1e-9);
            assert_eq!(estimate.branch_flows_m3_s.len(), 1);
        }

        #[test]
        fn zero_slope_pump_curve_uses_shutoff_pressure() {
            let estimate = solve_parallel_network(
                PumpCurve {
                    shutoff_pressure_pa: 80_000.0,
                    pressure_coefficient_pa_per_flow_squared: 0.0,
                    efficiency: 0.5,
                },
                &[Branch {
                    resistance_pa_per_flow_squared: 1.0e12,
                    open: true,
                }],
                1.0e-5,
                128,
            )
            .unwrap();
            assert!((estimate.operating_pressure_pa - 80_000.0).abs() < 1e-9);
        }

        #[test]
        fn closed_branch_has_zero_flow_and_opening_a_second_branch_increases_total_flow() {
            let pump = PumpCurve {
                shutoff_pressure_pa: 100_000.0,
                pressure_coefficient_pa_per_flow_squared: 1.0e12,
                efficiency: 0.6,
            };
            let one = solve_parallel_network(
                pump,
                &[Branch {
                    resistance_pa_per_flow_squared: 1.0e12,
                    open: true,
                }],
                1.0e-5,
                128,
            )
            .unwrap();
            let two = solve_parallel_network(
                pump,
                &[
                    Branch {
                        resistance_pa_per_flow_squared: 1.0e12,
                        open: true,
                    },
                    Branch {
                        resistance_pa_per_flow_squared: 1.0e12,
                        open: true,
                    },
                ],
                1.0e-5,
                128,
            )
            .unwrap();
            let isolated = solve_parallel_network(
                pump,
                &[
                    Branch {
                        resistance_pa_per_flow_squared: 1.0e12,
                        open: true,
                    },
                    Branch {
                        resistance_pa_per_flow_squared: 1.0e12,
                        open: false,
                    },
                ],
                1.0e-5,
                128,
            )
            .unwrap();

            assert!(two.total_flow_m3_s > one.total_flow_m3_s);
            assert_eq!(isolated.branch_flows_m3_s[1], 0.0);
            assert!((isolated.total_flow_m3_s - one.total_flow_m3_s).abs() < 1e-9);
        }

        #[test]
        fn rejects_deadheaded_network_and_invalid_resistance() {
            let pump = PumpCurve {
                shutoff_pressure_pa: 100_000.0,
                pressure_coefficient_pa_per_flow_squared: 1.0e12,
                efficiency: 0.5,
            };
            assert!(solve_parallel_network(
                pump,
                &[Branch {
                    resistance_pa_per_flow_squared: 1.0,
                    open: false,
                }],
                1.0,
                100,
            )
            .is_err());
            assert!(solve_parallel_network(
                pump,
                &[Branch {
                    resistance_pa_per_flow_squared: 0.0,
                    open: true,
                }],
                1.0,
                100,
            )
            .is_err());
        }
    }
}

/// Soil-specific empirical first-order cleaning kinetics.
///
/// The parameters must be fitted to repeatable physical tests for one soil //// surface / detergent family. This is a response model, not a universal cleaning
/// law, and it must not be used to certify hygiene or sanitization.
pub mod kinetics {
    const GAS_CONSTANT_J_PER_MOL_K: f64 = 8.314_462_618;

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct CleaningCalibration {
        pub reference_rate_s_inv: f64,
        pub reference_temperature_k: f64,
        pub activation_energy_j_per_mol: f64,
        pub reference_detergent_dose: f64,
        pub detergent_order: f64,
        pub reference_shear_rate: f64,
        pub shear_order: f64,
        pub min_temperature_k: f64,
        pub max_temperature_k: f64,
        pub min_detergent_dose: f64,
        pub max_detergent_dose: f64,
        pub min_shear_rate: f64,
        pub max_shear_rate: f64,
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct CleaningConditions {
        pub temperature_k: f64,
        pub detergent_dose: f64,
        pub shear_rate: f64,
        pub exposure_s: f64,
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct CleaningPrediction {
        pub rate_constant_s_inv: f64,
        pub residual_soil_fraction: f64,
        pub removed_soil_fraction: f64,
    }

    fn calibration_valid(cal: CleaningCalibration) -> bool {
        let positive = [
            cal.reference_rate_s_inv,
            cal.reference_temperature_k,
            cal.reference_detergent_dose,
            cal.reference_shear_rate,
            cal.min_temperature_k,
            cal.max_temperature_k,
            cal.min_detergent_dose,
            cal.max_detergent_dose,
            cal.min_shear_rate,
            cal.max_shear_rate,
        ];
        positive.iter().all(|v| v.is_finite() && *v > 0.0)
            && cal.activation_energy_j_per_mol.is_finite()
            && cal.activation_energy_j_per_mol >= 0.0
            && cal.detergent_order.is_finite()
            && cal.detergent_order >= 0.0
            && cal.shear_order.is_finite()
            && cal.shear_order >= 0.0
            && cal.min_temperature_k <= cal.reference_temperature_k
            && cal.reference_temperature_k <= cal.max_temperature_k
            && cal.min_detergent_dose <= cal.reference_detergent_dose
            && cal.reference_detergent_dose <= cal.max_detergent_dose
            && cal.min_shear_rate <= cal.reference_shear_rate
            && cal.reference_shear_rate <= cal.max_shear_rate
    }

    fn rate_constant(
        cal: CleaningCalibration,
        conditions: CleaningConditions,
    ) -> Result<f64, &'static str> {
        if !calibration_valid(cal) {
            return Err("cleaning calibration is invalid");
        }
        if !conditions.temperature_k.is_finite()
            || conditions.temperature_k < cal.min_temperature_k
            || conditions.temperature_k > cal.max_temperature_k
            || !conditions.detergent_dose.is_finite()
            || conditions.detergent_dose < cal.min_detergent_dose
            || conditions.detergent_dose > cal.max_detergent_dose
            || !conditions.shear_rate.is_finite()
            || conditions.shear_rate < cal.min_shear_rate
            || conditions.shear_rate > cal.max_shear_rate
        {
            return Err("conditions are outside the calibrated envelope; extrapolation rejected");
        }

        let temperature_term = cal.activation_energy_j_per_mol
            / GAS_CONSTANT_J_PER_MOL_K
            * (1.0 / cal.reference_temperature_k - 1.0 / conditions.temperature_k);
        let detergent_term = cal.detergent_order
            * (conditions.detergent_dose / cal.reference_detergent_dose).ln();
        let shear_term =
            cal.shear_order * (conditions.shear_rate / cal.reference_shear_rate).ln();
        let rate = cal.reference_rate_s_inv
            * (temperature_term + detergent_term + shear_term).exp();

        if !rate.is_finite() || rate <= 0.0 {
            return Err("derived cleaning rate is non-finite or non-positive");
        }
        Ok(rate)
    }

    /// Predict empirical residual soil using R(t) = exp(-k*t).
    pub fn predict(
        cal: CleaningCalibration,
        conditions: CleaningConditions,
    ) -> Result<CleaningPrediction, &'static str> {
        if !conditions.exposure_s.is_finite() || conditions.exposure_s < 0.0 {
            return Err("exposure time must be finite and non-negative");
        }
        let rate = rate_constant(cal, conditions)?;
        let residual = (-rate * conditions.exposure_s).exp();
        if !residual.is_finite() {
            return Err("derived residual-soil fraction is non-finite");
        }
        Ok(CleaningPrediction {
            rate_constant_s_inv: rate,
            residual_soil_fraction: residual,
            removed_soil_fraction: 1.0 - residual,
        })
    }

    /// Minimum exposure predicted to reach a requested residual fraction.
    /// The returned time is only valid inside the calibration envelope.
    pub fn exposure_for_target_residual(
        cal: CleaningCalibration,
        conditions: CleaningConditions,
        target_residual_fraction: f64,
    ) -> Result<f64, &'static str> {
        if !(target_residual_fraction.is_finite()
            && target_residual_fraction > 0.0
            && target_residual_fraction <= 1.0)
        {
            return Err("target residual fraction must be in (0, 1]");
        }
        let rate = rate_constant(cal, conditions)?;
        let time = -target_residual_fraction.ln() / rate;
        if !time.is_finite() {
            return Err("derived exposure time is non-finite");
        }
        Ok(time)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn calibration() -> CleaningCalibration {
            CleaningCalibration {
                reference_rate_s_inv: 0.01,
                reference_temperature_k: 323.15,
                activation_energy_j_per_mol: 20_000.0,
                reference_detergent_dose: 1.0,
                detergent_order: 0.5,
                reference_shear_rate: 100.0,
                shear_order: 0.3,
                min_temperature_k: 303.15,
                max_temperature_k: 343.15,
                min_detergent_dose: 0.5,
                max_detergent_dose: 1.5,
                min_shear_rate: 50.0,
                max_shear_rate: 200.0,
            }
        }

        #[test]
        fn reference_condition_matches_first_order_definition() {
            let prediction = predict(
                calibration(),
                CleaningConditions {
                    temperature_k: 323.15,
                    detergent_dose: 1.0,
                    shear_rate: 100.0,
                    exposure_s: 100.0,
                },
            )
            .unwrap();
            assert!((prediction.rate_constant_s_inv - 0.01).abs() < 1e-12);
            assert!((prediction.residual_soil_fraction - (-1.0_f64).exp()).abs() < 1e-12);
        }

        #[test]
        fn higher_temperature_dose_and_shear_raise_rate_with_positive_orders() {
            let cal = calibration();
            let baseline = CleaningConditions {
                temperature_k: 323.15,
                detergent_dose: 1.0,
                shear_rate: 100.0,
                exposure_s: 60.0,
            };
            let baseline_rate = predict(cal, baseline).unwrap().rate_constant_s_inv;
            let hotter = predict(
                cal,
                CleaningConditions {
                    temperature_k: 333.15,
                    ..baseline
                },
            )
            .unwrap()
            .rate_constant_s_inv;
            let more_detergent = predict(
                cal,
                CleaningConditions {
                    detergent_dose: 1.25,
                    ..baseline
                },
            )
            .unwrap()
            .rate_constant_s_inv;
            let more_shear = predict(
                cal,
                CleaningConditions {
                    shear_rate: 150.0,
                    ..baseline
                },
            )
            .unwrap()
            .rate_constant_s_inv;
            assert!(hotter > baseline_rate);
            assert!(more_detergent > baseline_rate);
            assert!(more_shear > baseline_rate);
        }

        #[test]
        fn rejects_conditions_outside_calibrated_envelope() {
            let result = predict(
                calibration(),
                CleaningConditions {
                    temperature_k: 353.15,
                    detergent_dose: 1.0,
                    shear_rate: 100.0,
                    exposure_s: 60.0,
                },
            );
            assert!(result.is_err());
        }

        #[test]
        fn target_time_inverts_the_first_order_response() {
            let cal = calibration();
            let conditions = CleaningConditions {
                temperature_k: 323.15,
                detergent_dose: 1.0,
                shear_rate: 100.0,
                exposure_s: 0.0,
            };
            let time = exposure_for_target_residual(cal, conditions, 0.1).unwrap();
            let prediction = predict(
                cal,
                CleaningConditions {
                    exposure_s: time,
                    ..conditions
                },
            )
            .unwrap();
            assert!((prediction.residual_soil_fraction - 0.1).abs() < 1e-12);
        }
    }
}

pub mod pareto {
    /// All four objective fields are minimized.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct CandidateMetrics {
        pub id: &'static str,
        pub residual_soil_fraction: f64,
        pub energy_kwh: f64,
        pub fresh_water_l: f64,
        pub cycle_duration_s: f64,
    }

    fn valid(candidate: CandidateMetrics) -> bool {
        [
            candidate.residual_soil_fraction,
            candidate.energy_kwh,
            candidate.fresh_water_l,
            candidate.cycle_duration_s,
        ]
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0)
            && candidate.residual_soil_fraction <= 1.0
    }

    pub fn dominates(a: CandidateMetrics, b: CandidateMetrics, tolerance: f64) -> bool {
        if !valid(a) || !valid(b) || !tolerance.is_finite() || tolerance < 0.0 {
            return false;
        }
        let av = [
            a.residual_soil_fraction,
            a.energy_kwh,
            a.fresh_water_l,
            a.cycle_duration_s,
        ];
        let bv = [
            b.residual_soil_fraction,
            b.energy_kwh,
            b.fresh_water_l,
            b.cycle_duration_s,
        ];
        let no_worse = av.iter().zip(bv.iter()).all(|(x, y)| *x <= *y + tolerance);
        let strictly_better = av.iter().zip(bv.iter()).any(|(x, y)| *x < *y - tolerance);
        no_worse && strictly_better
    }

    /// Return non-dominated candidates, preserving source order.
    pub fn pareto_front<'a>(
        candidates: &'a [CandidateMetrics],
        tolerance: f64,
    ) -> Result<Vec<&'a CandidateMetrics>, &'static str> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err("Pareto tolerance must be finite and non-negative");
        }
        Ok(candidates
            .iter()
            .enumerate()
            .filter(|(_, candidate)| valid(**candidate))
            .filter(|(candidate_index, candidate)| {
                !candidates.iter().enumerate().any(|(other_index, other)| {
                    other_index != *candidate_index
                        && dominates(*other, **candidate, tolerance)
                })
            })
            .map(|(_, candidate)| candidate)
            .collect())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn removes_dominated_candidate_but_keeps_tradeoff() {
            let candidates = [
                CandidateMetrics {
                    id: "baseline",
                    residual_soil_fraction: 0.10,
                    energy_kwh: 0.50,
                    fresh_water_l: 8.0,
                    cycle_duration_s: 1_800.0,
                },
                CandidateMetrics {
                    id: "dominated",
                    residual_soil_fraction: 0.12,
                    energy_kwh: 0.55,
                    fresh_water_l: 8.5,
                    cycle_duration_s: 1_900.0,
                },
                CandidateMetrics {
                    id: "faster",
                    residual_soil_fraction: 0.10,
                    energy_kwh: 0.60,
                    fresh_water_l: 7.0,
                    cycle_duration_s: 900.0,
                },
            ];
            let front = pareto_front(&candidates, 1e-12).unwrap();
            assert_eq!(front.len(), 2);
            assert!(front.iter().any(|candidate| candidate.id == "baseline"));
            assert!(front.iter().any(|candidate| candidate.id == "faster"));
            assert!(!front.iter().any(|candidate| candidate.id == "dominated"));
        }

        #[test]
        fn rejects_invalid_tolerance() {
            let candidates = [CandidateMetrics {
                id: "candidate",
                residual_soil_fraction: 0.1,
                energy_kwh: 0.5,
                fresh_water_l: 5.0,
                cycle_duration_s: 600.0,
            }];
            assert!(pareto_front(&candidates, f64::NAN).is_err());
            assert!(pareto_front(&candidates, -1.0).is_err());
        }

        #[test]
        fn duplicate_ids_do_not_hide_dominance() {
            let candidates = [
                CandidateMetrics {
                    id: "same-id",
                    residual_soil_fraction: 0.1,
                    energy_kwh: 0.5,
                    fresh_water_l: 5.0,
                    cycle_duration_s: 600.0,
                },
                CandidateMetrics {
                    id: "same-id",
                    residual_soil_fraction: 0.2,
                    energy_kwh: 0.6,
                    fresh_water_l: 6.0,
                    cycle_duration_s: 700.0,
                },
            ];
            let front = pareto_front(&candidates, 0.0).unwrap();
            assert_eq!(front.len(), 1);
            assert_eq!(front[0].residual_soil_fraction, 0.1);
        }

        #[test]
        fn rejects_non_finite_candidate() {
            let candidates = [
                CandidateMetrics {
                    id: "good",
                    residual_soil_fraction: 0.1,
                    energy_kwh: 0.5,
                    fresh_water_l: 5.0,
                    cycle_duration_s: 600.0,
                },
                CandidateMetrics {
                    id: "nan",
                    residual_soil_fraction: f64::NAN,
                    energy_kwh: 0.1,
                    fresh_water_l: 1.0,
                    cycle_duration_s: 10.0,
                },
            ];
            let front = pareto_front(&candidates, 0.0).unwrap();
            assert_eq!(front.len(), 1);
            assert_eq!(front[0].id, "good");
        }
    }
}
