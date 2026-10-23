// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Checked numeric-domain entry points for the textbook thermofluid formulas.
//!
//! These functions preserve the existing formula API in the fluids and thermal
//! modules, but reject non-finite or nonphysical input domains before
//! evaluation. They also reject non-finite computed outputs.
//!
//! Important limit: this module validates numeric domains, not units. Callers
//! must still provide coherent physical quantities (normally SI), and these
//! checks do not establish that a formula is applicable to a real flow or
//! heat-transfer regime. The shared dimensional/physical type system remains
//! coordinated through Symthaea issue #6868.

use crate::{fluids, thermal};
use std::error::Error;
use std::fmt;

/// Physical input whose numeric domain is being checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parameter {
    Density,
    Speed,
    Diameter,
    DynamicViscosity,
    ReynoldsNumber,
    Pressure,
    Velocity,
    Elevation,
    FrictionFactor,
    Length,
    InletArea,
    OutletArea,
    Conductivity,
    Area,
    TemperatureDifference,
    Thickness,
    ConvectionCoefficient,
    ColdTemperature,
    HotTemperature,
    HeatInput,
    Efficiency,
}

/// Numeric constraint that was violated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Violation {
    NonFinite,
    MustBePositive,
    MustBeNonNegative,
    MustBeAtLeastColdTemperature,
    OutsideUnitInterval,
}

/// Formula whose computed output was not finite.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Formula {
    ReynoldsNumber,
    BernoulliHead,
    DarcyWeisbachHeadLoss,
    ContinuityVelocity,
    ConductionHeatRate,
    ConvectionHeatRate,
    EngineWork,
}

/// Failure returned by a checked thermofluid function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationError {
    InvalidInput {
        parameter: Parameter,
        violation: Violation,
    },
    NonFiniteResult {
        formula: Formula,
    },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput {
                parameter,
                violation,
            } => write!(f, "invalid thermofluid input {parameter:?}: {violation:?}"),
            Self::NonFiniteResult { formula } => {
                write!(f, "thermofluid formula {formula:?} produced a non-finite result")
            }
        }
    }
}

impl Error for ValidationError {}

fn finite(value: f64, parameter: Parameter) -> Result<f64, ValidationError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ValidationError::InvalidInput {
            parameter,
            violation: Violation::NonFinite,
        })
    }
}

fn positive(value: f64, parameter: Parameter) -> Result<f64, ValidationError> {
    finite(value, parameter)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err(ValidationError::InvalidInput {
            parameter,
            violation: Violation::MustBePositive,
        })
    }
}

fn non_negative(value: f64, parameter: Parameter) -> Result<f64, ValidationError> {
    finite(value, parameter)?;
    if value >= 0.0 {
        Ok(value)
    } else {
        Err(ValidationError::InvalidInput {
            parameter,
            violation: Violation::MustBeNonNegative,
        })
    }
}

fn finite_result(value: f64, formula: Formula) -> Result<f64, ValidationError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ValidationError::NonFiniteResult { formula })
    }
}

/// Checked Reynolds number using non-negative speed, positive density/diameter,
/// and positive dynamic viscosity. Inputs must use a coherent unit system.
pub fn try_reynolds_number(
    density: f64,
    speed: f64,
    diameter: f64,
    dynamic_viscosity: f64,
) -> Result<f64, ValidationError> {
    positive(density, Parameter::Density)?;
    non_negative(speed, Parameter::Speed)?;
    positive(diameter, Parameter::Diameter)?;
    positive(dynamic_viscosity, Parameter::DynamicViscosity)?;
    finite_result(
        fluids::reynolds_number(density, speed, diameter, dynamic_viscosity),
        Formula::ReynoldsNumber,
    )
}

/// Checked pipe-flow regime classification.
pub fn try_flow_regime(reynolds: f64) -> Result<fluids::Regime, ValidationError> {
    non_negative(reynolds, Parameter::ReynoldsNumber)?;
    Ok(fluids::flow_regime(reynolds))
}

/// Checked Bernoulli total head. Pressure and elevation may be signed; density
/// must be positive. Velocity is squared by the underlying equation.
pub fn try_bernoulli_head(
    pressure: f64,
    velocity: f64,
    elevation: f64,
    density: f64,
) -> Result<f64, ValidationError> {
    finite(pressure, Parameter::Pressure)?;
    finite(velocity, Parameter::Velocity)?;
    finite(elevation, Parameter::Elevation)?;
    positive(density, Parameter::Density)?;
    finite_result(
        fluids::bernoulli_head(pressure, velocity, elevation, density),
        Formula::BernoulliHead,
    )
}

/// Checked Darcy–Weisbach head loss. Zero friction factor or pipe length
/// yields zero loss; diameter must remain positive.
pub fn try_darcy_weisbach_head_loss(
    friction_factor: f64,
    length: f64,
    diameter: f64,
    velocity: f64,
) -> Result<f64, ValidationError> {
    non_negative(friction_factor, Parameter::FrictionFactor)?;
    non_negative(length, Parameter::Length)?;
    positive(diameter, Parameter::Diameter)?;
    finite(velocity, Parameter::Velocity)?;
    finite_result(
        fluids::darcy_weisbach_head_loss(friction_factor, length, diameter, velocity),
        Formula::DarcyWeisbachHeadLoss,
    )
}

/// Checked continuity relation. Both cross-sectional areas must be positive;
/// velocity may be signed to preserve the caller's one-dimensional direction convention.
pub fn try_continuity_velocity(
    area_in: f64,
    velocity_in: f64,
    area_out: f64,
) -> Result<f64, ValidationError> {
    positive(area_in, Parameter::InletArea)?;
    finite(velocity_in, Parameter::Velocity)?;
    positive(area_out, Parameter::OutletArea)?;
    finite_result(
        fluids::continuity_velocity(area_in, velocity_in, area_out),
        Formula::ContinuityVelocity,
    )
}

/// Checked Carnot upper-bound efficiency. Temperatures are absolute kelvin;
/// the hot reservoir must not be colder than the cold reservoir.
pub fn try_carnot_efficiency(t_cold: f64, t_hot: f64) -> Result<f64, ValidationError> {
    positive(t_cold, Parameter::ColdTemperature)?;
    positive(t_hot, Parameter::HotTemperature)?;
    if t_hot < t_cold {
        return Err(ValidationError::InvalidInput {
            parameter: Parameter::HotTemperature,
            violation: Violation::MustBeAtLeastColdTemperature,
        });
    }
    Ok(thermal::carnot_efficiency(t_cold, t_hot))
}

/// Checked Fourier conduction rate. Conductivity and area may be zero for an
/// explicitly insulated/zero-area case; thickness must be positive.
pub fn try_conduction_heat_rate(
    conductivity: f64,
    area: f64,
    delta_temp: f64,
    thickness: f64,
) -> Result<f64, ValidationError> {
    non_negative(conductivity, Parameter::Conductivity)?;
    non_negative(area, Parameter::Area)?;
    finite(delta_temp, Parameter::TemperatureDifference)?;
    positive(thickness, Parameter::Thickness)?;
    finite_result(
        thermal::conduction_heat_rate(conductivity, area, delta_temp, thickness),
        Formula::ConductionHeatRate,
    )
}

/// Checked Newton cooling/convection rate. A signed temperature difference
/// preserves the chosen heat-flow direction convention.
pub fn try_convection_heat_rate(
    coefficient: f64,
    area: f64,
    delta_temp: f64,
) -> Result<f64, ValidationError> {
    non_negative(coefficient, Parameter::ConvectionCoefficient)?;
    non_negative(area, Parameter::Area)?;
    finite(delta_temp, Parameter::TemperatureDifference)?;
    finite_result(
        thermal::convection_heat_rate(coefficient, area, delta_temp),
        Formula::ConvectionHeatRate,
    )
}

/// Checked idealized heat-engine work. Efficiency is a fraction in [0, 1];
/// this does not assert that a real engine reaches that efficiency.
pub fn try_engine_work(heat_input: f64, efficiency: f64) -> Result<f64, ValidationError> {
    non_negative(heat_input, Parameter::HeatInput)?;
    finite(efficiency, Parameter::Efficiency)?;
    if !(0.0..=1.0).contains(&efficiency) {
        return Err(ValidationError::InvalidInput {
            parameter: Parameter::Efficiency,
            violation: Violation::OutsideUnitInterval,
        });
    }
    finite_result(
        thermal::engine_work(heat_input, efficiency),
        Formula::EngineWork,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_reynolds_matches_reference_and_rejects_invalid_domains() {
        let re = try_reynolds_number(1000.0, 2.0, 0.05, 1e-3).unwrap();
        assert!((re - 100_000.0).abs() < 1e-6);
        assert!(matches!(
            try_reynolds_number(f64::NAN, 2.0, 0.05, 1e-3),
            Err(ValidationError::InvalidInput {
                parameter: Parameter::Density,
                violation: Violation::NonFinite
            })
        ));
        assert!(matches!(
            try_reynolds_number(1000.0, 2.0, 0.05, 0.0),
            Err(ValidationError::InvalidInput {
                parameter: Parameter::DynamicViscosity,
                violation: Violation::MustBePositive
            })
        ));
        assert!(try_reynolds_number(1000.0, -1.0, 0.05, 1e-3).is_err());
    }

    #[test]
    fn checked_flow_regime_rejects_negative_and_non_finite_values() {
        assert_eq!(try_flow_regime(1000.0).unwrap(), fluids::Regime::Laminar);
        assert!(try_flow_regime(-1.0).is_err());
        assert!(try_flow_regime(f64::INFINITY).is_err());
    }

    #[test]
    fn checked_bernoulli_requires_positive_density() {
        let head = try_bernoulli_head(200_000.0, 1.0, 0.0, 1000.0).unwrap();
        assert!(head.is_finite());
        assert!(try_bernoulli_head(200_000.0, 1.0, 0.0, 0.0).is_err());
    }

    #[test]
    fn checked_darcy_weisbach_rejects_invalid_pipe_geometry() {
        let loss = try_darcy_weisbach_head_loss(0.02, 100.0, 0.1, 2.0).unwrap();
        assert!((loss - 4.0775).abs() < 1e-3);
        assert!(try_darcy_weisbach_head_loss(0.02, 100.0, 0.0, 2.0).is_err());
        assert!(try_darcy_weisbach_head_loss(0.02, -1.0, 0.1, 2.0).is_err());
        assert!(try_darcy_weisbach_head_loss(0.02, 100.0, 0.1, f64::NAN).is_err());
    }

    #[test]
    fn checked_continuity_rejects_zero_outlet_area() {
        assert_eq!(try_continuity_velocity(0.02, 1.0, 0.01).unwrap(), 2.0);
        assert!(try_continuity_velocity(0.02, 1.0, 0.0).is_err());
    }

    #[test]
    fn checked_carnot_requires_ordered_positive_kelvin_temperatures() {
        assert!((try_carnot_efficiency(300.0, 600.0).unwrap() - 0.5).abs() < 1e-12);
        assert!(try_carnot_efficiency(0.0, 600.0).is_err());
        assert!(try_carnot_efficiency(400.0, 300.0).is_err());
    }

    #[test]
    fn checked_conduction_and_convection_preserve_signed_heat_flow() {
        assert!(
            (try_conduction_heat_rate(200.0, 1.0, 50.0, 0.1).unwrap() - 100_000.0).abs()
                < 1e-6
        );
        assert_eq!(try_convection_heat_rate(10.0, 2.0, -5.0).unwrap(), -100.0);
        assert!(try_conduction_heat_rate(200.0, 1.0, 50.0, 0.0).is_err());
        assert!(try_convection_heat_rate(-1.0, 2.0, 5.0).is_err());
    }

    #[test]
    fn checked_engine_work_bounds_efficiency_and_heat_input() {
        assert_eq!(try_engine_work(1000.0, 0.5).unwrap(), 500.0);
        assert!(try_engine_work(-1.0, 0.5).is_err());
        assert!(try_engine_work(1000.0, 1.01).is_err());
        assert!(try_engine_work(1000.0, f64::NAN).is_err());
    }

    #[test]
    fn checked_functions_reject_non_finite_computed_results() {
        assert!(matches!(
            try_continuity_velocity(f64::MAX, f64::MAX, f64::MIN_POSITIVE),
            Err(ValidationError::NonFiniteResult {
                formula: Formula::ContinuityVelocity
            })
        ));
    }
}
