// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Checked entry points for the crate's textbook fluid/thermal equations.
//!
//! The original scalar functions remain available for compatibility and small
//! analytical examples. These wrappers are the safer choice for user-supplied
//! values, external telemetry, or model pipelines because they validate
//! domains and reject non-finite results rather than letting NaN/infinity
//! silently propagate into an engineering calculation.
//!
//! These routines validate arithmetic inputs, not the real-world applicability
//! of the simplified equation (e.g. incompressibility, fully developed flow,
//! constant material properties, or the assumptions behind a lumped model).

use std::error::Error;
use std::fmt;

/// Why a checked textbook equation refused to return a physical quantity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermofluidsError {
    /// One or more inputs contained NaN or infinity.
    NonFiniteInput,
    /// An input was outside its equation's physical domain.
    InvalidDomain,
    /// Finite-looking inputs produced an overflow/NaN result.
    NonFiniteResult,
}

impl fmt::Display for ThermofluidsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteInput => write!(f, "input must be finite"),
            Self::InvalidDomain => write!(f, "input is outside the equation's physical domain"),
            Self::NonFiniteResult => write!(f, "equation produced a non-finite result"),
        }
    }
}

impl Error for ThermofluidsError {}

#[inline]
fn finite(inputs: &[f64]) -> Result<(), ThermofluidsError> {
    if inputs.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(ThermofluidsError::NonFiniteInput)
    }
}

#[inline]
fn result_finite(value: f64) -> Result<f64, ThermofluidsError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ThermofluidsError::NonFiniteResult)
    }
}

/// Checked Reynolds number, using velocity magnitude as characteristic speed.
///
/// Density, diameter and dynamic viscosity must be positive. Signed velocity
/// is accepted for a directed coordinate system; Reynolds number remains
/// nonnegative because it uses the speed magnitude. The returned value is
/// dimensionless. This function does not determine whether flow is laminar,
/// transitional or turbulent; use `fluids::flow_regime` separately.
pub fn reynolds_number(
    density_kg_per_m3: f64,
    velocity_m_per_s: f64,
    diameter_m: f64,
    dynamic_viscosity_pa_s: f64,
) -> Result<f64, ThermofluidsError> {
    finite(&[
        density_kg_per_m3,
        velocity_m_per_s,
        diameter_m,
        dynamic_viscosity_pa_s,
    ])?;
    if density_kg_per_m3 <= 0.0 || diameter_m <= 0.0 || dynamic_viscosity_pa_s <= 0.0 {
        return Err(ThermofluidsError::InvalidDomain);
    }
    result_finite(
        density_kg_per_m3 * velocity_m_per_s.abs() * diameter_m
            / dynamic_viscosity_pa_s,
    )
}

/// Checked Bernoulli total head, in metres of the stated fluid.
///
/// Pressure may be negative when expressed as gauge pressure; density must be
/// positive. The formula itself assumes ideal steady incompressible flow along
/// a streamline and does not include pumps, turbines, or losses.
pub fn bernoulli_head(
    pressure_pa: f64,
    velocity_m_per_s: f64,
    elevation_m: f64,
    density_kg_per_m3: f64,
) -> Result<f64, ThermofluidsError> {
    finite(&[pressure_pa, velocity_m_per_s, elevation_m, density_kg_per_m3])?;
    if density_kg_per_m3 <= 0.0 {
        return Err(ThermofluidsError::InvalidDomain);
    }
    result_finite(crate::fluids::bernoulli_head(
        pressure_pa,
        velocity_m_per_s,
        elevation_m,
        density_kg_per_m3,
    ))
}

/// Checked Darcy-Weisbach major head loss, in metres.
///
/// Friction factor and pipe length may be zero but cannot be negative;
/// diameter must be positive. The result is a nonnegative scalar loss and
/// does not include minor losses or pump/turbine head.
pub fn darcy_weisbach_head_loss(
    friction_factor: f64,
    length_m: f64,
    diameter_m: f64,
    velocity_m_per_s: f64,
) -> Result<f64, ThermofluidsError> {
    finite(&[friction_factor, length_m, diameter_m, velocity_m_per_s])?;
    if friction_factor < 0.0 || length_m < 0.0 || diameter_m <= 0.0 {
        return Err(ThermofluidsError::InvalidDomain);
    }
    let loss = crate::fluids::darcy_weisbach_head_loss(
        friction_factor,
        length_m,
        diameter_m,
        velocity_m_per_s,
    )?;
    result_finite(loss)
}

/// Checked continuity velocity, in metres per second.
///
/// Areas must be positive; inlet velocity may be signed. This is the
/// one-dimensional incompressible continuity relation and does not model
/// compressibility, leakage, transient storage, or multiple branches.
pub fn continuity_velocity(
    area_in_m2: f64,
    velocity_in_m_per_s: f64,
    area_out_m2: f64,
) -> Result<f64, ThermofluidsError> {
    finite(&[area_in_m2, velocity_in_m_per_s, area_out_m2])?;
    if area_in_m2 <= 0.0 || area_out_m2 <= 0.0 {
        return Err(ThermofluidsError::InvalidDomain);
    }
    result_finite(crate::fluids::continuity_velocity(
        area_in_m2,
        velocity_in_m_per_s,
        area_out_m2,
    ))
}

/// Checked Carnot upper bound for a heat engine, in [0, 1].
///
/// Both reservoirs must have positive absolute temperatures and the hot
/// reservoir must not be colder than the cold one. This is an ideal upper
/// bound, not a prediction of an actual engine's efficiency.
pub fn carnot_efficiency(
    cold_temperature_k: f64,
    hot_temperature_k: f64,
) -> Result<f64, ThermofluidsError> {
    finite(&[cold_temperature_k, hot_temperature_k])?;
    if cold_temperature_k <= 0.0
        || hot_temperature_k <= 0.0
        || hot_temperature_k < cold_temperature_k
    {
        return Err(ThermofluidsError::InvalidDomain);
    }
    result_finite(crate::thermal::carnot_efficiency(
        cold_temperature_k,
        hot_temperature_k,
    ))
}

/// Checked Fourier conduction heat rate, in watts.
///
/// Conductivity and area must be nonnegative; thickness must be positive;
/// temperature difference is signed, so heat-flow direction is retained.
pub fn conduction_heat_rate(
    conductivity_w_per_m_k: f64,
    area_m2: f64,
    delta_temperature_k: f64,
    thickness_m: f64,
) -> Result<f64, ThermofluidsError> {
    finite(&[
        conductivity_w_per_m_k,
        area_m2,
        delta_temperature_k,
        thickness_m,
    ])?;
    if conductivity_w_per_m_k < 0.0 || area_m2 < 0.0 || thickness_m <= 0.0 {
        return Err(ThermofluidsError::InvalidDomain);
    }
    result_finite(crate::thermal::conduction_heat_rate(
        conductivity_w_per_m_k,
        area_m2,
        delta_temperature_k,
        thickness_m,
    ))
}

/// Checked Newton cooling/heating heat rate, in watts.
///
/// The coefficient and area must be nonnegative; signed temperature difference
/// is preserved so a hotter ambient may heat the selected body.
pub fn convection_heat_rate(
    coefficient_w_per_m2_k: f64,
    area_m2: f64,
    delta_temperature_k: f64,
) -> Result<f64, ThermofluidsError> {
    finite(&[coefficient_w_per_m2_k, area_m2, delta_temperature_k])?;
    if coefficient_w_per_m2_k < 0.0 || area_m2 < 0.0 {
        return Err(ThermofluidsError::InvalidDomain);
    }
    result_finite(crate::thermal::convection_heat_rate(
        coefficient_w_per_m2_k,
        area_m2,
        delta_temperature_k,
    ))
}

/// Checked engine work from a heat input and an efficiency in [0, 1], in joules.
///
/// This checks numeric bounds only; it does not establish whether the chosen
/// efficiency is thermodynamically achievable for the engine's reservoirs.
pub fn engine_work(heat_input_j: f64, efficiency: f64) -> Result<f64, ThermofluidsError> {
    finite(&[heat_input_j, efficiency])?;
    if heat_input_j < 0.0 || !(0.0..=1.0).contains(&efficiency) {
        return Err(ThermofluidsError::InvalidDomain);
    }
    result_finite(crate::thermal::engine_work(heat_input_j, efficiency))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reynolds_number_is_nonnegative_for_reversed_flow() {
        let forward = reynolds_number(1000.0, 2.0, 0.05, 1.0e-3).unwrap();
        let reverse = reynolds_number(1000.0, -2.0, 0.05, 1.0e-3).unwrap();
        assert!((forward - 100_000.0).abs() < 1.0e-9);
        assert_eq!(reverse, forward);
        assert_eq!(reynolds_number(1000.0, 0.0, 0.05, 1.0e-3), Ok(0.0));
    }

    #[test]
    fn checked_fluid_functions_match_reference_examples() {
        assert!((bernoulli_head(200_000.0, 1.0, 0.0, 1000.0).unwrap() - 20.050_968_399_6).abs() < 1.0e-6);
        assert!((darcy_weisbach_head_loss(0.02, 100.0, 0.1, 2.0).unwrap() - 4.0775).abs() < 1.0e-3);
        assert_eq!(continuity_velocity(0.02, 1.0, 0.01), Ok(2.0));
        assert_eq!(continuity_velocity(0.02, -1.0, 0.01), Ok(-2.0));
    }

    #[test]
    fn checked_thermal_functions_match_reference_examples() {
        assert_eq!(carnot_efficiency(300.0, 600.0), Ok(0.5));
        assert_eq!(conduction_heat_rate(200.0, 1.0, 50.0, 0.1), Ok(100_000.0));
        assert_eq!(convection_heat_rate(10.0, 2.0, -5.0), Ok(-100.0));
        assert_eq!(engine_work(1000.0, 0.5), Ok(500.0));
    }

    #[test]
    fn non_finite_inputs_are_rejected_by_every_checked_equation() {
        assert_eq!(
            reynolds_number(f64::NAN, 1.0, 1.0, 1.0),
            Err(ThermofluidsError::NonFiniteInput)
        );
        assert_eq!(
            bernoulli_head(1.0, f64::INFINITY, 0.0, 1.0),
            Err(ThermofluidsError::NonFiniteInput)
        );
        assert_eq!(
            darcy_weisbach_head_loss(0.02, 100.0, 0.1, f64::NAN),
            Err(ThermofluidsError::NonFiniteInput)
        );
        assert_eq!(
            continuity_velocity(1.0, 1.0, f64::INFINITY),
            Err(ThermofluidsError::NonFiniteInput)
        );
        assert_eq!(
            carnot_efficiency(300.0, f64::NAN),
            Err(ThermofluidsError::NonFiniteInput)
        );
        assert_eq!(
            conduction_heat_rate(1.0, 1.0, f64::INFINITY, 1.0),
            Err(ThermofluidsError::NonFiniteInput)
        );
        assert_eq!(
            convection_heat_rate(1.0, f64::NAN, 1.0),
            Err(ThermofluidsError::NonFiniteInput)
        );
        assert_eq!(
            engine_work(1.0, f64::INFINITY),
            Err(ThermofluidsError::NonFiniteInput)
        );
    }

    #[test]
    fn physical_domains_are_enforced() {
        assert_eq!(
            reynolds_number(0.0, 1.0, 1.0, 1.0),
            Err(ThermofluidsError::InvalidDomain)
        );
        assert_eq!(
            reynolds_number(1.0, 1.0, 1.0, 0.0),
            Err(ThermofluidsError::InvalidDomain)
        );
        assert_eq!(
            bernoulli_head(0.0, 0.0, 0.0, -1.0),
            Err(ThermofluidsError::InvalidDomain)
        );
        assert_eq!(
            darcy_weisbach_head_loss(-0.1, 1.0, 1.0, 1.0),
            Err(ThermofluidsError::InvalidDomain)
        );
        assert_eq!(
            continuity_velocity(1.0, 1.0, 0.0),
            Err(ThermofluidsError::InvalidDomain)
        );
        assert_eq!(
            carnot_efficiency(350.0, 300.0),
            Err(ThermofluidsError::InvalidDomain)
        );
        assert_eq!(
            conduction_heat_rate(1.0, 1.0, 10.0, 0.0),
            Err(ThermofluidsError::InvalidDomain)
        );
        assert_eq!(
            convection_heat_rate(-1.0, 1.0, 10.0),
            Err(ThermofluidsError::InvalidDomain)
        );
        assert_eq!(
            engine_work(1.0, 1.01),
            Err(ThermofluidsError::InvalidDomain)
        );
    }

    #[test]
    fn finite_inputs_that_overflow_are_rejected() {
        assert_eq!(
            continuity_velocity(f64::MAX, 2.0, f64::MIN_POSITIVE),
            Err(ThermofluidsError::NonFiniteResult)
        );
        assert_eq!(
            engine_work(f64::MAX, 1.0),
            Ok(f64::MAX)
        );
        assert_eq!(
            conduction_heat_rate(f64::MAX, f64::MAX, 2.0, f64::MIN_POSITIVE),
            Err(ThermofluidsError::NonFiniteResult)
        );
    }
}
