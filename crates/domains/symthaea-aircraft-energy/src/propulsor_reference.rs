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

/// Inputs for ideal axial-flight actuator-disk momentum theory.
///
/// Unlike the static-only reference function, this model includes positive
/// forward speed. It remains an ideal actuator-disk approximation, not a
/// blade-element, CFD, measured propeller, or electrical-chain model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxialActuatorDiskInput {
    pub thrust_n: f64,
    pub air_density_kg_m3: f64,
    pub disk_area_m2: f64,
    /// Free-stream speed along the propulsor axis, m/s. Reverse flow is out
    /// of scope for this simple positive-thrust propeller model.
    pub forward_speed_m_s: f64,
}

/// Ideal axial-flow quantities. At zero forward speed the ideal propulsive
/// efficiency is undefined and is represented by None, not by zero percent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxialActuatorDiskEstimate {
    pub induced_velocity_m_s: f64,
    /// Ideal power transferred to the airflow at the disk; not shaft or
    /// electrical input power.
    pub ideal_air_power_w: f64,
    /// Thrust multiplied by forward speed; zero in static conditions.
    pub useful_thrust_power_w: f64,
    /// Useful thrust power / ideal air power for forward flight only.
    pub ideal_propulsive_efficiency: Option<f64>,
    pub disk_loading_n_m2: f64,
}

/// Calculate an ideal axial-flow actuator-disk reference for positive thrust.
///
/// The ideal momentum relation is T = 2 rho A vi (V + vi), so the induced
/// velocity is (sqrt(V^2 + 2T/(rho A)) - V)/2. A rationalized expression is
/// used for the high-speed branch to avoid catastrophic subtraction. This
/// remains an ideal lower-bound calculation and is invalid for crossflow,
/// strongly nonuniform inflow, reverse flow, or as a real propeller map.
pub fn ideal_axial_actuator_disk(
    input: AxialActuatorDiskInput,
) -> Result<AxialActuatorDiskEstimate, StaticActuatorDiskError> {
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
    if !input.forward_speed_m_s.is_finite() {
        return Err(StaticActuatorDiskError::NonFiniteInput {
            field: "forward_speed_m_s",
        });
    }
    if input.forward_speed_m_s < 0.0 {
        return Err(StaticActuatorDiskError::NonPositiveInput {
            field: "forward_speed_m_s",
        });
    }

    let thrust_per_density_area =
        input.thrust_n / input.air_density_kg_m3 / input.disk_area_m2;
    if !thrust_per_density_area.is_finite()
        || thrust_per_density_area <= 0.0
        || thrust_per_density_area > f64::MAX / 2.0
    {
        return Err(StaticActuatorDiskError::DerivedValueOutOfRange);
    }
    let velocity_scale = (2.0 * thrust_per_density_area).sqrt();
    if !velocity_scale.is_finite() || velocity_scale <= 0.0 {
        return Err(StaticActuatorDiskError::DerivedValueOutOfRange);
    }

    let speed = input.forward_speed_m_s;
    let root = speed.hypot(velocity_scale);
    // vi = 0.5 * velocity_scale^2 / (root + speed). Scale the
    // high-speed ratio so root + speed cannot overflow and subtraction
    // of nearly equal velocities is avoided.
    let induced_velocity_m_s = if speed > velocity_scale {
        0.5 * velocity_scale * (velocity_scale / speed) / (root / speed + 1.0)
    } else {
        0.5 * velocity_scale * (velocity_scale / (root + speed))
    };
    let disk_flow_speed = speed + induced_velocity_m_s;
    let ideal_air_power_w = input.thrust_n * disk_flow_speed;
    let useful_thrust_power_w = input.thrust_n * speed;
    let disk_loading_n_m2 = input.thrust_n / input.disk_area_m2;
    let ideal_propulsive_efficiency = if speed > 0.0 {
        Some(speed / disk_flow_speed)
    } else {
        None
    };

    if !induced_velocity_m_s.is_finite()
        || induced_velocity_m_s <= 0.0
        || !disk_flow_speed.is_finite()
        || !ideal_air_power_w.is_finite()
        || ideal_air_power_w <= 0.0
        || !useful_thrust_power_w.is_finite()
        || useful_thrust_power_w < 0.0
        || !disk_loading_n_m2.is_finite()
        || disk_loading_n_m2 <= 0.0
        || ideal_propulsive_efficiency.is_some_and(|eta| !eta.is_finite() || eta <= 0.0 || eta >= 1.0)
    {
        return Err(StaticActuatorDiskError::DerivedValueOutOfRange);
    }

    Ok(AxialActuatorDiskEstimate {
        induced_velocity_m_s,
        ideal_air_power_w,
        useful_thrust_power_w,
        ideal_propulsive_efficiency,
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
    #[test]
    fn axial_zero_speed_matches_static_bound_and_has_no_static_efficiency() {
        let input = StaticActuatorDiskInput {
            thrust_n: 1_000.0,
            air_density_kg_m3: 1.225,
            disk_area_m2: 1.0,
        };
        let static_result = ideal_static_actuator_disk(input).unwrap();
        let axial_result = ideal_axial_actuator_disk(AxialActuatorDiskInput {
            thrust_n: input.thrust_n,
            air_density_kg_m3: input.air_density_kg_m3,
            disk_area_m2: input.disk_area_m2,
            forward_speed_m_s: 0.0,
        })
        .unwrap();

        assert!((axial_result.induced_velocity_m_s - static_result.induced_velocity_m_s).abs() < 1e-12);
        assert!((axial_result.ideal_air_power_w - static_result.ideal_induced_power_w).abs() < 1e-8);
        assert_eq!(axial_result.useful_thrust_power_w, 0.0);
        assert_eq!(axial_result.ideal_propulsive_efficiency, None);
    }

    #[test]
    fn forward_flight_reports_separate_useful_power_and_efficiency() {
        let result = ideal_axial_actuator_disk(AxialActuatorDiskInput {
            thrust_n: 1_000.0,
            air_density_kg_m3: 1.225,
            disk_area_m2: 1.0,
            forward_speed_m_s: 50.0,
        })
        .unwrap();

        assert!(result.ideal_air_power_w > result.useful_thrust_power_w);
        assert_eq!(result.useful_thrust_power_w, 50_000.0);
        let eta = result.ideal_propulsive_efficiency.unwrap();
        assert!(eta > 0.0 && eta < 1.0);
        assert!((eta - result.useful_thrust_power_w / result.ideal_air_power_w).abs() < 1e-12);
    }

    #[test]
    fn ideal_propulsive_efficiency_increases_with_forward_speed_at_fixed_thrust() {
        let slow = ideal_axial_actuator_disk(AxialActuatorDiskInput {
            thrust_n: 1_000.0,
            air_density_kg_m3: 1.225,
            disk_area_m2: 1.0,
            forward_speed_m_s: 10.0,
        })
        .unwrap();
        let fast = ideal_axial_actuator_disk(AxialActuatorDiskInput {
            thrust_n: 1_000.0,
            air_density_kg_m3: 1.225,
            disk_area_m2: 1.0,
            forward_speed_m_s: 50.0,
        })
        .unwrap();

        assert!(fast.ideal_propulsive_efficiency.unwrap() > slow.ideal_propulsive_efficiency.unwrap());
    }

    #[test]
    fn negative_or_non_finite_forward_speed_fails_closed() {
        assert_eq!(
            ideal_axial_actuator_disk(AxialActuatorDiskInput {
                thrust_n: 1_000.0,
                air_density_kg_m3: 1.225,
                disk_area_m2: 1.0,
                forward_speed_m_s: -1.0,
            }),
            Err(StaticActuatorDiskError::NonPositiveInput {
                field: "forward_speed_m_s"
            })
        );
        assert_eq!(
            ideal_axial_actuator_disk(AxialActuatorDiskInput {
                thrust_n: 1_000.0,
                air_density_kg_m3: 1.225,
                disk_area_m2: 1.0,
                forward_speed_m_s: f64::INFINITY,
            }),
            Err(StaticActuatorDiskError::NonFiniteInput {
                field: "forward_speed_m_s"
            })
        );
    }

}
