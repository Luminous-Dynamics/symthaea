// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Ideal static actuator-disk reference bound for propulsor screening.
//!
//! This module implements only the ideal, incompressible, uniform-flow static
//! momentum-theory lower bound on induced air power. It does not predict real
//! propeller shaft/electrical power, blade-profile or swirl losses, motor
//! efficiency, installed-airframe interaction, noise, or forward-flight
//! performance. Never use this static bound as a cruise-performance model.
//!
//! For positive thrust T, density rho, and actuator disk area A:
//!
//! - induced velocity: vi = sqrt(T / (2 rho A))
//! - ideal induced power: Pi = T * vi
//!
//! The calculation is a sanity-check oracle for synthetic fixtures, not a
//! qualified propulsor or aircraft performance prediction.

/// Inputs to the ideal static actuator-disk relation.
///
/// thrust_n is the target static thrust in newtons, air_density_kg_m3 is
/// ambient air density in kg/m³, and disk_area_m2 is the total ideal disk
/// area in m². For interacting or overlapping propulsors, total area cannot
/// automatically be treated as independent additive disk area.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaticActuatorDiskInput {
    pub thrust_n: f64,
    pub air_density_kg_m3: f64,
    pub disk_area_m2: f64,
}

/// Idealized quantities derived from static actuator-disk momentum theory.
///
/// ideal_induced_power_w is power imparted to the air in the ideal model;
/// it is not shaft power or electrical input power.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaticActuatorDiskEstimate {
    pub induced_velocity_m_s: f64,
    pub ideal_induced_power_w: f64,
    pub disk_loading_n_m2: f64,
}

/// Fail-closed input/output errors for the ideal static reference equation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaticActuatorDiskError {
    NonFiniteInput { field: &'static str },
    NonPositiveInput { field: &'static str },
    DerivedValueOutOfRange,
}

/// Calculate ideal induced air power for an ideal static actuator disk.
///
/// Assumptions: stationary disk, incompressible uniform ambient air, ideal
/// momentum theory, positive thrust/area/density. Real propulsor power will
/// exceed this induced-power bound because this model excludes non-idealities.
/// This function is not valid as a cruise efficiency or thrust prediction.
pub fn ideal_static_actuator_disk(
    input: StaticActuatorDiskInput,
) -> Result<StaticActuatorDiskEstimate, StaticActuatorDiskError> {
    for (field, value) in [
        ("thrust_n", input.thrust_n),
        ("air_density_kg_m3", input.air_density_kg_m3),
        ("disk_area_m2", input.disk_area_m2),
    ] {
        if !value.is_finite() {
            return Err(StaticActuatorDiskError::NonFiniteInput { field });
        }
        if value <= 0.0 {
            return Err(StaticActuatorDiskError::NonPositiveInput { field });
        }
    }

    let induced_velocity_m_s =
        (input.thrust_n / (2.0 * input.air_density_kg_m3 * input.disk_area_m2)).sqrt();
    let ideal_induced_power_w = input.thrust_n * induced_velocity_m_s;
    let disk_loading_n_m2 = input.thrust_n / input.disk_area_m2;
    if !induced_velocity_m_s.is_finite()
        || induced_velocity_m_s <= 0.0
        || !ideal_induced_power_w.is_finite()
        || ideal_induced_power_w <= 0.0
        || !disk_loading_n_m2.is_finite()
        || disk_loading_n_m2 <= 0.0
    {
        return Err(StaticActuatorDiskError::DerivedValueOutOfRange);
    }

    Ok(StaticActuatorDiskEstimate {
        induced_velocity_m_s,
        ideal_induced_power_w,
        disk_loading_n_m2,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_answer_matches_independent_static_momentum_calculation() {
        let result = ideal_static_actuator_disk(StaticActuatorDiskInput {
            thrust_n: 1_000.0,
            air_density_kg_m3: 1.225,
            disk_area_m2: 1.0,
        })
        .unwrap();

        assert!((result.induced_velocity_m_s - 20.203_050_891_044_214).abs() < 1e-10);
        assert!((result.ideal_induced_power_w - 20_203.050_891_044_215).abs() < 1e-8);
        assert_eq!(result.disk_loading_n_m2, 1_000.0);
    }

    #[test]
    fn four_times_disk_area_halves_ideal_induced_power_at_fixed_thrust() {
        let small = ideal_static_actuator_disk(StaticActuatorDiskInput {
            thrust_n: 1_000.0,
            air_density_kg_m3: 1.225,
            disk_area_m2: 1.0,
        })
        .unwrap();
        let large = ideal_static_actuator_disk(StaticActuatorDiskInput {
            thrust_n: 1_000.0,
            air_density_kg_m3: 1.225,
            disk_area_m2: 4.0,
        })
        .unwrap();

        assert!((large.ideal_induced_power_w / small.ideal_induced_power_w - 0.5).abs() < 1e-12);
    }

    #[test]
    fn doubling_air_density_reduces_ideal_induced_power_by_sqrt_two() {
        let standard = ideal_static_actuator_disk(StaticActuatorDiskInput {
            thrust_n: 1_000.0,
            air_density_kg_m3: 1.225,
            disk_area_m2: 1.0,
        })
        .unwrap();
        let doubled_density = ideal_static_actuator_disk(StaticActuatorDiskInput {
            thrust_n: 1_000.0,
            air_density_kg_m3: 2.45,
            disk_area_m2: 1.0,
        })
        .unwrap();

        assert!(
            (doubled_density.ideal_induced_power_w / standard.ideal_induced_power_w
                - 1.0 / 2.0_f64.sqrt())
            .abs()
                < 1e-12
        );
    }

    #[test]
    fn non_finite_and_non_positive_inputs_fail_closed() {
        assert_eq!(
            ideal_static_actuator_disk(StaticActuatorDiskInput {
                thrust_n: f64::NAN,
                air_density_kg_m3: 1.225,
                disk_area_m2: 1.0,
            }),
            Err(StaticActuatorDiskError::NonFiniteInput { field: "thrust_n" })
        );
        assert_eq!(
            ideal_static_actuator_disk(StaticActuatorDiskInput {
                thrust_n: 1_000.0,
                air_density_kg_m3: 0.0,
                disk_area_m2: 1.0,
            }),
            Err(StaticActuatorDiskError::NonPositiveInput {
                field: "air_density_kg_m3"
            })
        );
        assert_eq!(
            ideal_static_actuator_disk(StaticActuatorDiskInput {
                thrust_n: 1_000.0,
                air_density_kg_m3: 1.225,
                disk_area_m2: f64::INFINITY,
            }),
            Err(StaticActuatorDiskError::NonFiniteInput { field: "disk_area_m2" })
        );
    }

    #[test]
    fn overflowed_derived_power_is_not_reported_as_a_bound() {
        assert_eq!(
            ideal_static_actuator_disk(StaticActuatorDiskInput {
                thrust_n: 1e308,
                air_density_kg_m3: 1.0,
                disk_area_m2: 1.0,
            }),
            Err(StaticActuatorDiskError::DerivedValueOutOfRange)
        );
    }
}
