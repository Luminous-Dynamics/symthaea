// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Small analytic calibration and observation-diagnostic utilities.
//!
//! These routines invert equations already implemented by the crate. They are
//! deliberately not general-purpose statistical inference: callers remain
//! responsible for observation uncertainty, structural error, and provenance.

use crate::energy_balance::{STEFAN_BOLTZMANN, try_effective_emissivity_surface_temperature};
use crate::error::{ModelError, require_finite, require_fraction, require_positive};
use crate::transient::OneBoxClimateModel;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EffectiveEmissivityCalibration {
    pub effective_olr_emissivity: f64,
    pub reconstructed_temperature: f64,
    pub temperature_residual: f64,
}

/// Infer the effective outgoing-longwave emissivity that exactly closes a
/// zero-dimensional radiative balance at an observed temperature.
pub fn calibrate_effective_olr_emissivity(
    solar_flux: f64,
    albedo: f64,
    observed_temperature: f64,
) -> Result<EffectiveEmissivityCalibration, ModelError> {
    require_positive("solar_flux", solar_flux)?;
    require_fraction("albedo", albedo)?;
    require_positive("observed_temperature", observed_temperature)?;
    let absorbed = solar_flux * (1.0 - albedo) / 4.0;
    let effective_olr_emissivity = absorbed / (STEFAN_BOLTZMANN * observed_temperature.powi(4));
    require_fraction("effective_olr_emissivity", effective_olr_emissivity)?;
    if effective_olr_emissivity == 0.0 {
        return Err(ModelError::NonPositive {
            parameter: "effective_olr_emissivity",
            value: effective_olr_emissivity,
        });
    }
    let reconstructed_temperature =
        try_effective_emissivity_surface_temperature(solar_flux, albedo, effective_olr_emissivity)?;
    Ok(EffectiveEmissivityCalibration {
        effective_olr_emissivity,
        reconstructed_temperature,
        temperature_residual: observed_temperature - reconstructed_temperature,
    })
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OneBoxCalibration {
    pub model: OneBoxClimateModel,
    pub equilibrium_warming: f64,
    pub response_time_seconds: f64,
}

impl OneBoxCalibration {
    /// Recover `feedback = forcing / equilibrium_warming` and
    /// `heat_capacity = feedback * response_time`.
    pub fn from_equilibrium_and_response(
        baseline_temperature: f64,
        forcing: f64,
        equilibrium_temperature: f64,
        response_time_seconds: f64,
    ) -> Result<Self, ModelError> {
        require_positive("baseline_temperature", baseline_temperature)?;
        require_finite("forcing", forcing)?;
        require_positive("equilibrium_temperature", equilibrium_temperature)?;
        require_positive("response_time_seconds", response_time_seconds)?;
        let equilibrium_warming = equilibrium_temperature - baseline_temperature;
        if equilibrium_warming == 0.0 {
            return Err(ModelError::SingularCalibration {
                reason: "equilibrium warming is zero",
            });
        }
        let feedback = forcing / equilibrium_warming;
        require_positive("inferred_feedback", feedback)?;
        let heat_capacity = feedback * response_time_seconds;
        let model = OneBoxClimateModel::try_new(heat_capacity, feedback, baseline_temperature)?;
        Ok(Self {
            model,
            equilibrium_warming,
            response_time_seconds,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObservationErrorSummary {
    pub count: usize,
    pub mean_error: f64,
    pub mean_absolute_error: f64,
    pub root_mean_square_error: f64,
    pub maximum_absolute_error: f64,
}

/// Compare timestamped temperature observations against the exact one-box
/// constant-forcing trajectory.
pub fn one_box_constant_forcing_error(
    model: &OneBoxClimateModel,
    initial_temperature: f64,
    forcing: f64,
    observations: &[(f64, f64)],
) -> Result<ObservationErrorSummary, ModelError> {
    model.validate()?;
    require_positive("initial_temperature", initial_temperature)?;
    require_finite("forcing", forcing)?;
    if observations.is_empty() {
        return Err(ModelError::EmptySeries {
            series: "one-box temperature observations",
        });
    }

    let mut error_sum = 0.0;
    let mut absolute_sum = 0.0;
    let mut squared_sum = 0.0;
    let mut maximum_absolute_error: f64 = 0.0;
    for &(time_seconds, observed_temperature) in observations {
        require_positive_or_zero("observation_time_seconds", time_seconds)?;
        require_positive("observed_temperature", observed_temperature)?;
        let predicted = model.exact_constant_forcing(initial_temperature, forcing, time_seconds)?;
        let error = observed_temperature - predicted;
        error_sum += error;
        absolute_sum += error.abs();
        squared_sum += error * error;
        maximum_absolute_error = maximum_absolute_error.max(error.abs());
    }
    let count = observations.len();
    Ok(ObservationErrorSummary {
        count,
        mean_error: error_sum / count as f64,
        mean_absolute_error: absolute_sum / count as f64,
        root_mean_square_error: (squared_sum / count as f64).sqrt(),
        maximum_absolute_error,
    })
}

fn require_positive_or_zero(parameter: &'static str, value: f64) -> Result<(), ModelError> {
    require_finite(parameter, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(ModelError::OutOfRange {
            parameter,
            value,
            min: 0.0,
            max: f64::INFINITY,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::energy_balance::{EARTH_ALBEDO, SOLAR_CONSTANT_EARTH};
    use crate::transient::SECONDS_PER_YEAR;

    #[test]
    fn effective_emissivity_calibration_reconstructs_temperature() {
        let fit =
            calibrate_effective_olr_emissivity(SOLAR_CONSTANT_EARTH, EARTH_ALBEDO, 288.0).unwrap();
        assert!(fit.effective_olr_emissivity > 0.6);
        assert!(fit.effective_olr_emissivity < 0.7);
        assert!(fit.temperature_residual.abs() < 1e-12);
    }

    #[test]
    fn one_box_parameters_are_recovered_analytically() {
        let original = OneBoxClimateModel::earthlike();
        let forcing = 3.7;
        let fit = OneBoxCalibration::from_equilibrium_and_response(
            original.baseline_temperature,
            forcing,
            original.equilibrium_temperature(forcing),
            original.response_time(),
        )
        .unwrap();
        assert!((fit.model.feedback - original.feedback).abs() < 1e-12);
        assert!(
            (fit.model.heat_capacity - original.heat_capacity).abs()
                < 1e-6 * original.heat_capacity.abs()
        );
    }

    #[test]
    fn exact_observations_have_zero_residual() {
        let model = OneBoxClimateModel::earthlike();
        let observations: Vec<_> = [0.0, SECONDS_PER_YEAR, 10.0 * SECONDS_PER_YEAR]
            .into_iter()
            .map(|time| {
                (
                    time,
                    model.exact_constant_forcing(288.0, 3.7, time).unwrap(),
                )
            })
            .collect();
        let summary = one_box_constant_forcing_error(&model, 288.0, 3.7, &observations).unwrap();
        assert_eq!(summary.count, 3);
        assert!(summary.maximum_absolute_error < 1e-12);
    }

    #[test]
    fn zero_warming_calibration_is_singular() {
        assert!(matches!(
            OneBoxCalibration::from_equilibrium_and_response(288.0, 3.7, 288.0, 1.0),
            Err(ModelError::SingularCalibration { .. })
        ));
    }
}
