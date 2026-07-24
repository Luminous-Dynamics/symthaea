// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Analytic calibration helpers for population models.
//!
//! The logistic fit uses the exact linearization for a caller-supplied carrying
//! capacity. It is not a claim that ecological observation errors are Gaussian
//! in transformed space; the transformed residuals are exposed so callers can
//! judge that assumption.

use crate::error::{ModelError, require_finite, require_positive};
use crate::logistic::LogisticModel;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogisticCalibration {
    pub model: LogisticModel,
    pub initial_population: f64,
    pub transformed_intercept: f64,
    pub transformed_root_mean_square_error: f64,
    pub transformed_r_squared: f64,
}

/// Fit intrinsic growth and initial population with known carrying capacity.
///
/// For `0 < N < K`, the logistic equation linearizes to
/// `ln(N / (K-N)) = intercept + r t`.
pub fn fit_logistic_known_capacity(
    carrying_capacity: f64,
    observations: &[(f64, f64)],
) -> Result<LogisticCalibration, ModelError> {
    require_positive("carrying_capacity", carrying_capacity)?;
    if observations.len() < 2 {
        return Err(ModelError::InsufficientSamples {
            required: 2,
            found: observations.len(),
        });
    }

    let count = observations.len() as f64;
    let mut sum_time = 0.0;
    let mut sum_transformed = 0.0;
    let mut transformed = Vec::with_capacity(observations.len());
    for &(time, population) in observations {
        require_finite("observation_time", time)?;
        require_positive("observed_population", population)?;
        if population >= carrying_capacity {
            return Err(ModelError::OutOfRange {
                parameter: "observed_population",
                value: population,
                min: f64::MIN_POSITIVE,
                max: carrying_capacity,
            });
        }
        let value = (population / (carrying_capacity - population)).ln();
        sum_time += time;
        sum_transformed += value;
        transformed.push((time, value));
    }

    let mean_time = sum_time / count;
    let mean_transformed = sum_transformed / count;
    let mut time_variance = 0.0;
    let mut covariance = 0.0;
    for &(time, value) in &transformed {
        let centered_time = time - mean_time;
        time_variance += centered_time * centered_time;
        covariance += centered_time * (value - mean_transformed);
    }
    if time_variance <= f64::EPSILON * count.max(1.0) {
        return Err(ModelError::SingularCalibration {
            reason: "observation times have no resolvable spread",
        });
    }

    let intrinsic_growth_rate = covariance / time_variance;
    require_positive("fitted_intrinsic_growth_rate", intrinsic_growth_rate)?;
    let transformed_intercept = mean_transformed - intrinsic_growth_rate * mean_time;
    let initial_population = carrying_capacity / (1.0 + (-transformed_intercept).exp());
    let model = LogisticModel::try_new(intrinsic_growth_rate, carrying_capacity)?;

    let mut squared_residual = 0.0;
    let mut total_squared = 0.0;
    for &(time, value) in &transformed {
        let predicted = transformed_intercept + intrinsic_growth_rate * time;
        squared_residual += (value - predicted).powi(2);
        total_squared += (value - mean_transformed).powi(2);
    }
    let transformed_r_squared = if total_squared > 0.0 {
        1.0 - squared_residual / total_squared
    } else {
        1.0
    };

    Ok(LogisticCalibration {
        model,
        initial_population,
        transformed_intercept,
        transformed_root_mean_square_error: (squared_residual / count).sqrt(),
        transformed_r_squared,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_logistic_series_recovers_parameters() {
        let original = LogisticModel::try_new(0.35, 120.0).unwrap();
        let initial = 12.0;
        let observations: Vec<_> = [0.0, 1.0, 2.0, 4.0, 8.0]
            .into_iter()
            .map(|time| (time, original.population(initial, time).unwrap()))
            .collect();
        let fit = fit_logistic_known_capacity(120.0, &observations).unwrap();
        assert!((fit.model.intrinsic_growth_rate - 0.35).abs() < 1e-12);
        assert!((fit.initial_population - initial).abs() < 1e-10);
        assert!(fit.transformed_root_mean_square_error < 1e-12);
        assert!((fit.transformed_r_squared - 1.0).abs() < 1e-12);
    }

    #[test]
    fn repeated_times_are_singular() {
        let observations = [(1.0, 10.0), (1.0, 20.0)];
        assert!(matches!(
            fit_logistic_known_capacity(100.0, &observations),
            Err(ModelError::SingularCalibration { .. })
        ));
    }

    #[test]
    fn observations_at_or_above_capacity_are_rejected() {
        let observations = [(0.0, 10.0), (1.0, 100.0)];
        assert!(fit_logistic_known_capacity(100.0, &observations).is_err());
    }
}
