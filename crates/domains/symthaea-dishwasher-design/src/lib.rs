#![forbid(unsafe_code)]

//! First-principles engineering primitives for the InstantClean dishwasher
//! research programme.
//!
//! This crate models measurable engineering quantities only: sensible heating,
//! electrical resource budgets, nozzle hydraulics and multi-objective candidate
//! comparison. It does not predict soil removal, microbiological safety, spray
//! breakup or CFD. Those require calibrated experiments and validated solvers.

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

        Ok(NozzleEstimate {
            exit_velocity_m_s,
            reynolds_number: re,
            flow_regime: flow_regime(re),
            dynamic_pressure_pa: 0.5
                * point.fluid_density_kg_m3
                * exit_velocity_m_s.powi(2),
            hydraulic_power_w,
            estimated_electrical_pump_power_w: hydraulic_power_w / point.pump_efficiency,
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
    ) -> Vec<&'a CandidateMetrics> {
        candidates
            .iter()
            .filter(|candidate| valid(**candidate))
            .filter(|candidate| {
                !candidates.iter().any(|other| {
                    other.id != candidate.id && dominates(*other, **candidate, tolerance)
                })
            })
            .collect()
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
            let front = pareto_front(&candidates, 1e-12);
            assert_eq!(front.len(), 2);
            assert!(front.iter().any(|candidate| candidate.id == "baseline"));
            assert!(front.iter().any(|candidate| candidate.id == "faster"));
            assert!(!front.iter().any(|candidate| candidate.id == "dominated"));
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
            let front = pareto_front(&candidates, 0.0);
            assert_eq!(front.len(), 1);
            assert_eq!(front[0].id, "good");
        }
    }
}
