// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Dimensionless propeller performance coefficients for fair, regime-aware
//! comparisons across operating points and geometries.
//!
//! This module computes the conventional advance ratio, thrust coefficient,
//! power coefficient, and useful-power/shaft-power ratio from explicitly
//! unit-labelled SI inputs. It does not claim that equal coefficients imply
//! dynamic similarity: Reynolds number, Mach number, blade geometry, inflow,
//! installation/wing interaction, and measurement-boundary differences may
//! still change performance.
//!
//! Propeller coefficient conventions:
//!   n  = rpm / 60                         [rev/s]
//!   J  = V / (n D)
//!   C_T = T / (rho n^2 D^4)
//!   C_P = P_shaft / (rho n^3 D^5)
//!   eta = T V / P_shaft, only for V > 0 and P_shaft > 0
//!
//! The torque/RPM consistency audit is separate. This module uses the caller's
//! declared shaft-power value; it does not silently replace it with a derived
//! value, infer electrical input, or qualify the source measurement.

/// Operating-point data needed for conventional propeller coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropellerCoefficientInput {
    /// Axial free-stream speed, m/s. Crossflow/inflow angle is not represented.
    pub forward_speed_m_s: f64,
    /// Rotational rate in revolutions per minute.
    pub rpm: f64,
    /// Propeller diameter, m.
    pub diameter_m: f64,
    /// Ambient air density, kg/m^3.
    pub air_density_kg_m3: f64,
    /// Axial thrust under the source's declared sign convention, N.
    pub thrust_n: f64,
    /// Shaft power at the declared shaft measurement boundary, W; not electrical power.
    pub shaft_power_w: f64,
    /// Optional source-defined reference Reynolds number. The characteristic
    /// length/velocity convention must be preserved in external provenance.
    pub reference_reynolds_number: Option<f64>,
    /// Optional source-supplied tip Mach number for this operating point.
    /// This function does not infer speed of sound or atmospheric temperature.
    pub tip_mach: Option<f64>,
}

/// Dimensionless values derived from one operating point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropellerCoefficients {
    /// Rotational frequency, rev/s.
    pub revolutions_per_second: f64,
    /// Advance ratio J = V / (n D).
    pub advance_ratio: f64,
    /// Thrust coefficient C_T = T / (rho n^2 D^4).
    pub thrust_coefficient: f64,
    /// Shaft-power coefficient C_P = P_shaft / (rho n^3 D^5).
    pub shaft_power_coefficient: f64,
    /// Useful axial thrust power divided by shaft power, T V / P_shaft.
    /// Undefined for static conditions or zero shaft power; never reported as
    /// a meaningful static-efficiency score.
    pub propulsive_efficiency: Option<f64>,
    /// Passed-through metadata for similarity review; no value is inferred.
    pub reference_reynolds_number: Option<f64>,
    /// Passed-through metadata for similarity review; no value is inferred.
    pub tip_mach: Option<f64>,
}

/// Input failures or non-representable coefficient calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropellerCoefficientError {
    NonFiniteInput { field: &'static str },
    NonPositiveInput { field: &'static str },
    NegativeInput { field: &'static str },
    InvalidOptionalMetadata { field: &'static str },
    DerivedValueOutOfRange { field: &'static str },
}

/// Compute dimensionless coefficients from one positive-RPM propeller operating
/// point. Tolerances, measurement uncertainty, Reynolds number and tip Mach
/// provenance are the caller's responsibility.
pub fn calculate_propeller_coefficients(
    input: PropellerCoefficientInput,
) -> Result<PropellerCoefficients, PropellerCoefficientError> {
    for (field, value) in [
        ("forward_speed_m_s", input.forward_speed_m_s),
        ("rpm", input.rpm),
        ("diameter_m", input.diameter_m),
        ("air_density_kg_m3", input.air_density_kg_m3),
        ("thrust_n", input.thrust_n),
        ("shaft_power_w", input.shaft_power_w),
    ] {
        if !value.is_finite() {
            return Err(PropellerCoefficientError::NonFiniteInput { field });
        }
    }

    for (field, value) in [
        ("rpm", input.rpm),
        ("diameter_m", input.diameter_m),
        ("air_density_kg_m3", input.air_density_kg_m3),
    ] {
        if value <= 0.0 {
            return Err(PropellerCoefficientError::NonPositiveInput { field });
        }
    }

    for (field, value) in [
        ("forward_speed_m_s", input.forward_speed_m_s),
        ("thrust_n", input.thrust_n),
        ("shaft_power_w", input.shaft_power_w),
    ] {
        if value < 0.0 {
            return Err(PropellerCoefficientError::NegativeInput { field });
        }
    }

    if let Some(value) = input.reference_reynolds_number {
        if !value.is_finite() || value <= 0.0 {
            return Err(PropellerCoefficientError::InvalidOptionalMetadata {
                field: "reference_reynolds_number",
            });
        }
    }
    if let Some(value) = input.tip_mach {
        if !value.is_finite() || value <= 0.0 {
            return Err(PropellerCoefficientError::InvalidOptionalMetadata {
                field: "tip_mach",
            });
        }
    }

    let n = input.rpm / 60.0;
    let d2 = input.diameter_m.powi(2);
    let d4 = d2 * d2;
    let d5 = d4 * input.diameter_m;
    let n2 = n * n;
    let n3 = n2 * n;
    let thrust_denominator = input.air_density_kg_m3 * n2 * d4;
    let power_denominator = input.air_density_kg_m3 * n3 * d5;

    for (field, value) in [
        ("revolutions_per_second", n),
        ("diameter_power_4", d4),
        ("diameter_power_5", d5),
        ("thrust_denominator", thrust_denominator),
        ("power_denominator", power_denominator),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(PropellerCoefficientError::DerivedValueOutOfRange { field });
        }
    }

    let advance_ratio = input.forward_speed_m_s / (n * input.diameter_m);
    let thrust_coefficient = input.thrust_n / thrust_denominator;
    let shaft_power_coefficient = input.shaft_power_w / power_denominator;
    let propulsive_efficiency = if input.forward_speed_m_s > 0.0 && input.shaft_power_w > 0.0 {
        let useful_thrust_power_w = input.thrust_n * input.forward_speed_m_s;
        let efficiency = useful_thrust_power_w / input.shaft_power_w;
        if !useful_thrust_power_w.is_finite() || !efficiency.is_finite() {
            return Err(PropellerCoefficientError::DerivedValueOutOfRange {
                field: "propulsive_efficiency",
            });
        }
        Some(efficiency)
    } else {
        None
    };

    for (field, value) in [
        ("advance_ratio", advance_ratio),
        ("thrust_coefficient", thrust_coefficient),
        ("shaft_power_coefficient", shaft_power_coefficient),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(PropellerCoefficientError::DerivedValueOutOfRange { field });
        }
    }

    Ok(PropellerCoefficients {
        revolutions_per_second: n,
        advance_ratio,
        thrust_coefficient,
        shaft_power_coefficient,
        propulsive_efficiency,
        reference_reynolds_number: input.reference_reynolds_number,
        tip_mach: input.tip_mach,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn known_point() -> PropellerCoefficientInput {
        PropellerCoefficientInput {
            forward_speed_m_s: 10.0,
            rpm: 600.0,
            diameter_m: 2.0,
            air_density_kg_m3: 1.25,
            thrust_n: 100.0,
            shaft_power_w: 5_000.0,
            reference_reynolds_number: Some(180_000.0),
            tip_mach: Some(0.22),
        }
    }

    #[test]
    fn matches_independent_known_answer_for_all_coefficients() {
        let result = calculate_propeller_coefficients(known_point()).unwrap();
        assert_eq!(result.revolutions_per_second, 10.0);
        assert!((result.advance_ratio - 0.5).abs() < 1e-14);
        assert!((result.thrust_coefficient - 0.05).abs() < 1e-14);
        assert!((result.shaft_power_coefficient - 0.125).abs() < 1e-14);
        assert!((result.propulsive_efficiency.unwrap() - 0.2).abs() < 1e-14);
        assert_eq!(result.reference_reynolds_number, Some(180_000.0));
        assert_eq!(result.tip_mach, Some(0.22));
    }

    #[test]
    fn conventional_efficiency_equals_j_ct_over_cp_when_defined() {
        let result = calculate_propeller_coefficients(known_point()).unwrap();
        let coefficient_ratio =
            result.advance_ratio * result.thrust_coefficient / result.shaft_power_coefficient;
        assert!((result.propulsive_efficiency.unwrap() - coefficient_ratio).abs() < 1e-14);
    }

    #[test]
    fn static_condition_has_zero_advance_ratio_and_no_propulsive_efficiency() {
        let mut input = known_point();
        input.forward_speed_m_s = 0.0;
        let result = calculate_propeller_coefficients(input).unwrap();
        assert_eq!(result.advance_ratio, 0.0);
        assert_eq!(result.propulsive_efficiency, None);
    }

    #[test]
    fn zero_shaft_power_does_not_produce_infinite_efficiency() {
        let mut input = known_point();
        input.shaft_power_w = 0.0;
        let result = calculate_propeller_coefficients(input).unwrap();
        assert_eq!(result.shaft_power_coefficient, 0.0);
        assert_eq!(result.propulsive_efficiency, None);
    }

    #[test]
    fn zero_thrust_is_a_valid_zero_coefficient_not_a_missing_value() {
        let mut input = known_point();
        input.thrust_n = 0.0;
        let result = calculate_propeller_coefficients(input).unwrap();
        assert_eq!(result.thrust_coefficient, 0.0);
        assert_eq!(result.propulsive_efficiency, Some(0.0));
    }

    #[test]
    fn missing_similarity_metadata_is_not_fabricated() {
        let mut input = known_point();
        input.reference_reynolds_number = None;
        input.tip_mach = None;
        let result = calculate_propeller_coefficients(input).unwrap();
        assert_eq!(result.reference_reynolds_number, None);
        assert_eq!(result.tip_mach, None);
    }

    #[test]
    fn rejects_non_finite_and_invalid_inputs() {
        let mut input = known_point();
        input.rpm = f64::NAN;
        assert_eq!(
            calculate_propeller_coefficients(input),
            Err(PropellerCoefficientError::NonFiniteInput { field: "rpm" })
        );

        let mut input = known_point();
        input.diameter_m = 0.0;
        assert_eq!(
            calculate_propeller_coefficients(input),
            Err(PropellerCoefficientError::NonPositiveInput { field: "diameter_m" })
        );

        let mut input = known_point();
        input.thrust_n = -1.0;
        assert_eq!(
            calculate_propeller_coefficients(input),
            Err(PropellerCoefficientError::NegativeInput { field: "thrust_n" })
        );
    }

    #[test]
    fn rejects_invalid_optional_similarity_metadata() {
        let mut input = known_point();
        input.reference_reynolds_number = Some(f64::NAN);
        assert_eq!(
            calculate_propeller_coefficients(input),
            Err(PropellerCoefficientError::InvalidOptionalMetadata {
                field: "reference_reynolds_number"
            })
        );

        let mut input = known_point();
        input.tip_mach = Some(0.0);
        assert_eq!(
            calculate_propeller_coefficients(input),
            Err(PropellerCoefficientError::InvalidOptionalMetadata { field: "tip_mach" })
        );
    }

    #[test]
    fn extreme_operating_values_fail_when_denominators_overflow() {
        let mut input = known_point();
        input.rpm = f64::MAX;
        assert!(matches!(
            calculate_propeller_coefficients(input),
            Err(PropellerCoefficientError::DerivedValueOutOfRange { .. })
        ));
    }

    #[test]
    fn efficiency_above_one_is_preserved_for_diagnostics_not_clamped() {
        let mut input = known_point();
        input.shaft_power_w = 500.0;
        let result = calculate_propeller_coefficients(input).unwrap();
        assert!((result.propulsive_efficiency.unwrap() - 2.0).abs() < 1e-14);
    }

    #[test]
    fn identical_pinned_point_is_deterministic() {
        let input = known_point();
        assert_eq!(
            calculate_propeller_coefficients(input).unwrap(),
            calculate_propeller_coefficients(input).unwrap()
        );
    }
}
