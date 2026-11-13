// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent ideal actuator-disk hover reference oracle.
//!
//! This crate computes a deliberately narrow analytical lower bound for
//! steady, axial, ideal hover flow. It is independent of the helicopter and
//! multirotor production simulators and contains no third-party dependencies.
//!
//! For total rotor thrust T, air density rho, and declared effective
//! non-overlapping disk area A:
//!
//!   disk_loading = T / A
//!   induced_velocity = sqrt(T / (2 * rho * A))
//!   ideal_induced_power = T * induced_velocity
//!
//! A measured figure of merit may optionally convert the ideal induced-power
//! lower bound into a rotor-shaft power estimate. The figure of merit must be
//! supplied explicitly; this crate never invents one.
//!
//! This is NOT a total rotorcraft power model, aircraft sizing tool, rotor
//! interaction model, flight controller, safety model, or certification tool.

#![forbid(unsafe_code)]

use std::fmt;

/// Inputs for the ideal steady-hover actuator-disk reference.
///
/// effective_disk_area_m2 must describe the effective non-overlapping disk
/// area for the declared rotor system. Overlapping/coaxial disks must not be
/// double-counted. The caller owns that geometry/evidence determination.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoverInput {
    /// Total vertical thrust demanded from the complete rotor system (N).
    pub total_thrust_n: f64,
    /// Local air density for the declared hover condition (kg/m^3).
    pub air_density_kg_m3: f64,
    /// Effective non-overlapping total disk area (m^2).
    pub effective_disk_area_m2: f64,
    /// Optional measured/modelled hover figure of merit in (0, 1].
    ///
    /// This is an explicit input, not inferred from geometry. If absent, only
    /// the ideal induced-power lower bound is returned.
    pub figure_of_merit: Option<f64>,
}

/// Analytical hover quantities returned by the reference oracle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoverEstimate {
    /// Total thrust divided by effective disk area (N/m^2 = Pa).
    pub disk_loading_pa: f64,
    /// Ideal induced air velocity at the actuator disk (m/s).
    pub induced_velocity_mps: f64,
    /// Ideal induced power lower bound, T * v_i (W).
    pub ideal_induced_power_w: f64,
    /// Ideal induced power per unit thrust (W/N = m/s).
    pub ideal_induced_power_per_thrust_mps: f64,
    /// Optional rotor-shaft power estimate, ideal power / figure of merit (W).
    ///
    /// This does not include transmission, motor, inverter/ESC, wiring,
    /// cooling, battery, or aircraft installation losses.
    pub rotor_shaft_power_estimate_w: Option<f64>,
    /// Figure of merit used for the optional shaft-power estimate.
    pub figure_of_merit: Option<f64>,
}

/// Explicit failure dispositions for malformed or numerically unsupported input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HoverOracleError {
    NonFiniteInput,
    NonPositiveThrust,
    NonPositiveAirDensity,
    NonPositiveDiskArea,
    InvalidFigureOfMerit,
    NumericalOverflow,
}

impl fmt::Display for HoverOracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for HoverOracleError {}

/// Evaluate the ideal actuator-disk hover reference.
///
/// The computation assumes steady, axial, ideal flow in hover. It is a
/// theoretical lower bound, not a prediction of installed electrical power.
pub fn estimate_ideal_hover(input: HoverInput) -> Result<HoverEstimate, HoverOracleError> {
    let values = [
        input.total_thrust_n,
        input.air_density_kg_m3,
        input.effective_disk_area_m2,
    ];
    if values.iter().any(|value| !value.is_finite()) {
        return Err(HoverOracleError::NonFiniteInput);
    }

    if input.total_thrust_n <= 0.0 {
        return Err(HoverOracleError::NonPositiveThrust);
    }
    if input.air_density_kg_m3 <= 0.0 {
        return Err(HoverOracleError::NonPositiveAirDensity);
    }
    if input.effective_disk_area_m2 <= 0.0 {
        return Err(HoverOracleError::NonPositiveDiskArea);
    }

    if let Some(fm) = input.figure_of_merit {
        if !fm.is_finite() {
            return Err(HoverOracleError::NonFiniteInput);
        }
        if !(0.0 < fm && fm <= 1.0) {
            return Err(HoverOracleError::InvalidFigureOfMerit);
        }
    }

    // Divide in stages and validate every derived quantity. This avoids
    // silently returning infinities for extreme but finite input magnitudes.
    let disk_loading_pa = input.total_thrust_n / input.effective_disk_area_m2;
    let momentum_denominator = 2.0 * input.air_density_kg_m3 * input.effective_disk_area_m2;
    let induced_velocity_squared = input.total_thrust_n / momentum_denominator;
    if !disk_loading_pa.is_finite()
        || !momentum_denominator.is_finite()
        || momentum_denominator <= 0.0
        || !induced_velocity_squared.is_finite()
        || induced_velocity_squared <= 0.0
    {
        return Err(HoverOracleError::NumericalOverflow);
    }

    let induced_velocity_mps = induced_velocity_squared.sqrt();
    let ideal_induced_power_w = input.total_thrust_n * induced_velocity_mps;
    if !induced_velocity_mps.is_finite()
        || !ideal_induced_power_w.is_finite()
        || induced_velocity_mps <= 0.0
        || ideal_induced_power_w <= 0.0
    {
        return Err(HoverOracleError::NumericalOverflow);
    }

    let rotor_shaft_power_estimate_w = input
        .figure_of_merit
        .map(|figure_of_merit| ideal_induced_power_w / figure_of_merit);
    if rotor_shaft_power_estimate_w.is_some_and(|power| !power.is_finite()) {
        return Err(HoverOracleError::NumericalOverflow);
    }

    Ok(HoverEstimate {
        disk_loading_pa,
        induced_velocity_mps,
        ideal_induced_power_w,
        ideal_induced_power_per_thrust_mps: induced_velocity_mps,
        rotor_shaft_power_estimate_w,
        figure_of_merit: input.figure_of_merit,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(thrust_n: f64, density: f64, area: f64) -> HoverInput {
        HoverInput {
            total_thrust_n: thrust_n,
            air_density_kg_m3: density,
            effective_disk_area_m2: area,
            figure_of_merit: None,
        }
    }

    #[test]
    fn simple_analytic_case_matches_frozen_reference() {
        // T=2 N, rho=1 kg/m^3, A=1 m^2:
        // v_i=sqrt(2/(2*1*1))=1 m/s; P_i=2 W; disk loading=2 Pa.
        let result = estimate_ideal_hover(input(2.0, 1.0, 1.0)).unwrap();
        assert_eq!(result.disk_loading_pa, 2.0);
        assert_eq!(result.induced_velocity_mps, 1.0);
        assert_eq!(result.ideal_induced_power_w, 2.0);
        assert_eq!(result.ideal_induced_power_per_thrust_mps, 1.0);
        assert_eq!(result.rotor_shaft_power_estimate_w, None);
    }

    #[test]
    fn explicit_figure_of_merit_only_changes_shaft_power_estimate() {
        let mut case = input(2.0, 1.0, 1.0);
        case.figure_of_merit = Some(0.5);
        let result = estimate_ideal_hover(case).unwrap();
        assert_eq!(result.ideal_induced_power_w, 2.0);
        assert_eq!(result.rotor_shaft_power_estimate_w, Some(4.0));
        assert_eq!(result.figure_of_merit, Some(0.5));
    }

    #[test]
    fn doubling_thrust_scales_ideal_power_by_two_to_three_halves() {
        let base = estimate_ideal_hover(input(2.0, 1.0, 1.0)).unwrap();
        let doubled = estimate_ideal_hover(input(4.0, 1.0, 1.0)).unwrap();
        let expected_ratio = 2.0_f64.powf(1.5);
        let actual_ratio = doubled.ideal_induced_power_w / base.ideal_induced_power_w;
        assert!((actual_ratio - expected_ratio).abs() < 1.0e-12);
    }

    #[test]
    fn doubling_disk_area_reduces_ideal_power_by_inverse_sqrt_two() {
        let base = estimate_ideal_hover(input(2.0, 1.0, 1.0)).unwrap();
        let larger_disk = estimate_ideal_hover(input(2.0, 1.0, 2.0)).unwrap();
        let expected_ratio = 1.0 / 2.0_f64.sqrt();
        let actual_ratio = larger_disk.ideal_induced_power_w / base.ideal_induced_power_w;
        assert!((actual_ratio - expected_ratio).abs() < 1.0e-12);
    }

    #[test]
    fn doubling_density_reduces_ideal_power_by_inverse_sqrt_two() {
        let base = estimate_ideal_hover(input(2.0, 1.0, 1.0)).unwrap();
        let denser_air = estimate_ideal_hover(input(2.0, 2.0, 1.0)).unwrap();
        let expected_ratio = 1.0 / 2.0_f64.sqrt();
        let actual_ratio = denser_air.ideal_induced_power_w / base.ideal_induced_power_w;
        assert!((actual_ratio - expected_ratio).abs() < 1.0e-12);
    }

    #[test]
    fn rejects_zero_and_non_finite_physical_inputs() {
        assert_eq!(
            estimate_ideal_hover(input(0.0, 1.0, 1.0)),
            Err(HoverOracleError::NonPositiveThrust)
        );
        assert_eq!(
            estimate_ideal_hover(input(1.0, 0.0, 1.0)),
            Err(HoverOracleError::NonPositiveAirDensity)
        );
        assert_eq!(
            estimate_ideal_hover(input(1.0, 1.0, 0.0)),
            Err(HoverOracleError::NonPositiveDiskArea)
        );
        assert_eq!(
            estimate_ideal_hover(input(f64::NAN, 1.0, 1.0)),
            Err(HoverOracleError::NonFiniteInput)
        );
    }

    #[test]
    fn rejects_unphysical_figure_of_merit() {
        for figure_of_merit in [0.0, -0.1, 1.01, f64::INFINITY] {
            let mut case = input(2.0, 1.0, 1.0);
            case.figure_of_merit = Some(figure_of_merit);
            assert!(estimate_ideal_hover(case).is_err());
        }
    }

    #[test]
    fn extreme_finite_inputs_fail_closed_on_derived_overflow() {
        let case = input(f64::MAX, f64::MIN_POSITIVE, f64::MIN_POSITIVE);
        assert_eq!(
            estimate_ideal_hover(case),
            Err(HoverOracleError::NumericalOverflow)
        );
    }
}
