//! Declared uniform-delay, quadratic NG-RC reference baseline.
//!
//! Features are [1, delayed taps, all pairwise products of delayed taps].
//! This transparent variant is not claimed to reproduce every NG-RC paper
//! implementation and is not the EMD/KDE-based NGRC-MDS comparator.

use std::error::Error;
use std::fmt;

use crate::baseline::{solve_least_squares_qr, validate_history};

const MAX_DELAY_COUNT: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NgrcError {
    InvalidHistory,
    InvalidDelayCount,
    InvalidDelaySpacing,
    InvalidRidgePenalty,
    InsufficientSamples { required: usize, actual: usize },
    InsufficientForecastContext { required: usize, actual: usize },
    LinearSolveFailure,
    NumericalFailure,
}

impl fmt::Display for NgrcError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHistory => write!(formatter, "NG-RC history is empty or non-finite"),
            Self::InvalidDelayCount => write!(formatter, "NG-RC delay count is outside 1..=16"),
            Self::InvalidDelaySpacing => write!(formatter, "NG-RC delay spacing must be positive"),
            Self::InvalidRidgePenalty => write!(
                formatter,
                "NG-RC ridge penalty must be finite and non-negative"
            ),
            Self::InsufficientSamples { required, actual } => write!(
                formatter,
                "NG-RC requires at least {required} observations; received {actual}"
            ),
            Self::InsufficientForecastContext { required, actual } => write!(
                formatter,
                "NG-RC requires {required} context observations; received {actual}"
            ),
            Self::LinearSolveFailure => write!(formatter, "NG-RC readout solve failed"),
            Self::NumericalFailure => {
                write!(formatter, "NG-RC arithmetic produced a non-finite value")
            }
        }
    }
}

impl Error for NgrcError {}

#[derive(Clone, Debug, PartialEq)]
pub struct UniformDelayNgrc {
    delay_count: usize,
    delay_spacing: usize,
    training_observations: usize,
    coefficients: Vec<f64>,
}

impl UniformDelayNgrc {
    pub fn delay_count(&self) -> usize {
        self.delay_count
    }

    pub fn delay_spacing(&self) -> usize {
        self.delay_spacing
    }

    pub fn training_observations(&self) -> usize {
        self.training_observations
    }

    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Recursive free-running prediction using only the supplied as-of history.
    pub fn forecast(&self, history: &[f64], horizon: usize) -> Result<Vec<f64>, NgrcError> {
        validate_history(history).map_err(|_| NgrcError::InvalidHistory)?;
        if horizon == 0 {
            return Ok(Vec::new());
        }

        let required = context_span(self.delay_count, self.delay_spacing)? + 1;
        if history.len() < required {
            return Err(NgrcError::InsufficientForecastContext {
                required,
                actual: history.len(),
            });
        }

        let mut context = history.to_vec();
        let mut output = Vec::with_capacity(horizon);
        for _ in 0..horizon {
            let origin = context.len() - 1;
            let features = features_at(&context, origin, self.delay_count, self.delay_spacing)?;
            let value = features
                .iter()
                .zip(self.coefficients.iter())
                .fold(0.0, |sum, (feature, coefficient)| {
                    coefficient.mul_add(*feature, sum)
                });
            if !value.is_finite() {
                return Err(NgrcError::NumericalFailure);
            }
            context.push(value);
            output.push(value);
        }
        Ok(output)
    }
}

/// Fit one-step targets using only the supplied training history.
/// Ridge applies to all readout features except the intercept.
pub fn fit_uniform_delay_ngrc(
    history: &[f64],
    delay_count: usize,
    delay_spacing: usize,
    ridge_penalty: f64,
) -> Result<UniformDelayNgrc, NgrcError> {
    validate_history(history).map_err(|_| NgrcError::InvalidHistory)?;
    if delay_count == 0 || delay_count > MAX_DELAY_COUNT {
        return Err(NgrcError::InvalidDelayCount);
    }
    if delay_spacing == 0 {
        return Err(NgrcError::InvalidDelaySpacing);
    }
    if !ridge_penalty.is_finite() || ridge_penalty < 0.0 {
        return Err(NgrcError::InvalidRidgePenalty);
    }

    let span = context_span(delay_count, delay_spacing)?;
    let required_context = span + 1;
    if history.len() <= required_context {
        return Err(NgrcError::InsufficientSamples {
            required: required_context + 1,
            actual: history.len(),
        });
    }
    let feature_count = 1 + delay_count + (delay_count * (delay_count + 1)) / 2;
    let target_rows = history.len() - required_context;
    if ridge_penalty == 0.0 && target_rows < feature_count {
        return Err(NgrcError::InsufficientSamples {
            required: required_context + feature_count,
            actual: history.len(),
        });
    }

    let penalty_rows = if ridge_penalty > 0.0 {
        feature_count - 1
    } else {
        0
    };
    let mut matrix = Vec::with_capacity(target_rows + penalty_rows);
    let mut target = Vec::with_capacity(target_rows + penalty_rows);

    for origin in span..history.len() - 1 {
        matrix.push(features_at(history, origin, delay_count, delay_spacing)?);
        target.push(history[origin + 1]);
    }

    if ridge_penalty > 0.0 {
        let scale = ridge_penalty.sqrt();
        if !scale.is_finite() {
            return Err(NgrcError::NumericalFailure);
        }
        for column in 1..feature_count {
            let mut row = vec![0.0; feature_count];
            row[column] = scale;
            matrix.push(row);
            target.push(0.0);
        }
    }

    let coefficients =
        solve_least_squares_qr(matrix, target).map_err(|_| NgrcError::LinearSolveFailure)?;
    Ok(UniformDelayNgrc {
        delay_count,
        delay_spacing,
        training_observations: history.len(),
        coefficients,
    })
}

fn context_span(delay_count: usize, delay_spacing: usize) -> Result<usize, NgrcError> {
    delay_count
        .saturating_sub(1)
        .checked_mul(delay_spacing)
        .ok_or(NgrcError::NumericalFailure)
}

fn features_at(
    history: &[f64],
    origin: usize,
    delay_count: usize,
    delay_spacing: usize,
) -> Result<Vec<f64>, NgrcError> {
    let mut taps = Vec::with_capacity(delay_count);
    for tap in 0..delay_count {
        let offset = tap
            .checked_mul(delay_spacing)
            .ok_or(NgrcError::NumericalFailure)?;
        let index = origin
            .checked_sub(offset)
            .ok_or(NgrcError::InsufficientForecastContext {
                required: context_span(delay_count, delay_spacing)? + 1,
                actual: origin + 1,
            })?;
        taps.push(
            *history
                .get(index)
                .ok_or(NgrcError::InsufficientForecastContext {
                    required: context_span(delay_count, delay_spacing)? + 1,
                    actual: history.len(),
                })?,
        );
    }

    let mut features = Vec::with_capacity(1 + delay_count + (delay_count * (delay_count + 1)) / 2);
    features.push(1.0);
    features.extend(taps.iter().copied());
    for left in 0..taps.len() {
        for right in left..taps.len() {
            features.push(taps[left] * taps[right]);
        }
    }
    if features.iter().any(|feature| !feature.is_finite()) {
        return Err(NgrcError::NumericalFailure);
    }
    Ok(features)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logistic_history(length: usize) -> Vec<f64> {
        let mut values = vec![0.123_456_789];
        while values.len() < length {
            let previous = *values.last().unwrap();
            values.push(3.7 * previous * (1.0 - previous));
        }
        values
    }

    #[test]
    fn quadratic_readout_recovers_logistic_map_one_step() {
        let values = logistic_history(200);
        let training = &values[..160];
        let model = fit_uniform_delay_ngrc(training, 1, 1, 0.0).unwrap();
        assert_eq!(model.coefficients().len(), 3);
        assert!(model.coefficients()[0].abs() < 1e-8);
        assert!((model.coefficients()[1] - 3.7).abs() < 1e-8);
        assert!((model.coefficients()[2] + 3.7).abs() < 1e-8);
        let prediction = model.forecast(training, 1).unwrap();
        assert!((prediction[0] - values[160]).abs() < 1e-8);
    }

    #[test]
    fn recursive_forecast_is_finite_and_uses_declared_uniform_delays() {
        let values = logistic_history(180);
        let model = fit_uniform_delay_ngrc(&values[..150], 3, 2, 1e-8).unwrap();
        assert_eq!(model.delay_count(), 3);
        assert_eq!(model.delay_spacing(), 2);
        assert_eq!(model.training_observations(), 150);
        let forecast = model.forecast(&values[..150], 12).unwrap();
        assert_eq!(forecast.len(), 12);
        assert!(forecast.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn rejects_invalid_delay_design_and_penalty() {
        let values = logistic_history(40);
        assert_eq!(
            fit_uniform_delay_ngrc(&values, 0, 1, 0.0),
            Err(NgrcError::InvalidDelayCount)
        );
        assert_eq!(
            fit_uniform_delay_ngrc(&values, 2, 0, 0.0),
            Err(NgrcError::InvalidDelaySpacing)
        );
        assert_eq!(
            fit_uniform_delay_ngrc(&values, 2, 1, f64::NAN),
            Err(NgrcError::InvalidRidgePenalty)
        );
    }

    #[test]
    fn rejects_insufficient_training_and_forecast_context() {
        let values = logistic_history(80);
        assert!(matches!(
            fit_uniform_delay_ngrc(&values[..4], 4, 2, 0.0),
            Err(NgrcError::InsufficientSamples { .. })
        ));
        let model = fit_uniform_delay_ngrc(&values, 3, 2, 1e-8).unwrap();
        assert!(matches!(
            model.forecast(&values[..3], 1),
            Err(NgrcError::InsufficientForecastContext { .. })
        ));
    }
}
