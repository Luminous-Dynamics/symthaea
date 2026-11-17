//! Transparent univariate forecasting baselines.
//!
//! These implementations are deliberately small reference arms, not SOTA
//! predictors. They consume only an observed history and expose no evaluator
//! or future-state interface. Linear autoregression is fitted with
//! Householder QR; optional ridge regularization is applied to lag
//! coefficients (the intercept is not penalized).

use std::error::Error;
use std::fmt;

/// Failure modes for fitting or applying a transparent baseline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaselineError {
    EmptyHistory,
    NonFiniteObservation,
    InvalidAutoregressionOrder,
    InsufficientSamples { required: usize, actual: usize },
    InsufficientForecastContext { required: usize, actual: usize },
    InvalidRidgePenalty,
    RankDeficientDesign,
    NumericalFailure,
}

impl fmt::Display for BaselineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyHistory => write!(formatter, "observation history is empty"),
            Self::NonFiniteObservation => {
                write!(formatter, "observation history contains a non-finite value")
            }
            Self::InvalidAutoregressionOrder => {
                write!(formatter, "autoregression order must be positive")
            }
            Self::InsufficientSamples { required, actual } => write!(
                formatter,
                "autoregression requires at least {required} observations; received {actual}"
            ),
            Self::InsufficientForecastContext { required, actual } => write!(
                formatter,
                "forecast requires {required} context observations; received {actual}"
            ),
            Self::InvalidRidgePenalty => {
                write!(formatter, "ridge penalty must be finite and non-negative")
            }
            Self::RankDeficientDesign => {
                write!(formatter, "autoregression design matrix is rank deficient")
            }
            Self::NumericalFailure => {
                write!(formatter, "baseline arithmetic produced a non-finite value")
            }
        }
    }
}

impl Error for BaselineError {}

/// Identity of a fitted transparent baseline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BaselineKind {
    Persistence,
    Mean,
    LinearAutoregression,
}

/// A fitted baseline with immutable model parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct FittedBaseline {
    kind: BaselineKind,
    training_observations: usize,
    mean: Option<f64>,
    order: usize,
    coefficients: Vec<f64>,
}

impl FittedBaseline {
    pub fn kind(&self) -> BaselineKind {
        self.kind
    }

    pub fn training_observations(&self) -> usize {
        self.training_observations
    }

    /// Linear-AR coefficients in the order [intercept, lag_1, ..., lag_p].
    /// Returns an empty slice for persistence and mean baselines.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// Forecast from the supplied as-of history.
    ///
    /// Persistence uses the latest supplied observation. The mean baseline
    /// uses the mean learned by `fit_mean`. Linear AR recursively feeds each
    /// prediction back as the next lag. The caller is responsible for
    /// supplying only observations available at the decision time.
    pub fn forecast(
        &self,
        observed_history: &[f64],
        horizon: usize,
    ) -> Result<Vec<f64>, BaselineError> {
        validate_history(observed_history)?;

        if horizon == 0 {
            return Ok(Vec::new());
        }

        match self.kind {
            BaselineKind::Persistence => {
                let value = *observed_history.last().ok_or(BaselineError::EmptyHistory)?;
                Ok(vec![value; horizon])
            }
            BaselineKind::Mean => {
                let value = self.mean.ok_or(BaselineError::NumericalFailure)?;
                Ok(vec![value; horizon])
            }
            BaselineKind::LinearAutoregression => {
                if observed_history.len() < self.order {
                    return Err(BaselineError::InsufficientForecastContext {
                        required: self.order,
                        actual: observed_history.len(),
                    });
                }

                let mut context = observed_history.to_vec();
                let mut forecasts = Vec::with_capacity(horizon);

                for _ in 0..horizon {
                    let mut prediction = self.coefficients[0];
                    for lag in 1..=self.order {
                        let observation = context[context.len() - lag];
                        prediction = self.coefficients[lag].mul_add(observation, prediction);
                    }
                    if !prediction.is_finite() {
                        return Err(BaselineError::NumericalFailure);
                    }
                    context.push(prediction);
                    forecasts.push(prediction);
                }

                Ok(forecasts)
            }
        }
    }
}

/// Fit a persistence baseline. The forecast value is taken from the latest
/// context observation supplied to `forecast`, not frozen at training time.
pub fn fit_persistence(history: &[f64]) -> Result<FittedBaseline, BaselineError> {
    validate_history(history)?;
    Ok(FittedBaseline {
        kind: BaselineKind::Persistence,
        training_observations: history.len(),
        mean: None,
        order: 0,
        coefficients: Vec::new(),
    })
}

/// Fit a constant-mean forecast using a numerically checked online mean.
pub fn fit_mean(history: &[f64]) -> Result<FittedBaseline, BaselineError> {
    validate_history(history)?;

    let mut mean = 0.0;
    for (index, observation) in history.iter().copied().enumerate() {
        let count = (index + 1) as f64;
        let difference = observation - mean;
        mean += difference / count;
        if !mean.is_finite() {
            return Err(BaselineError::NumericalFailure);
        }
    }

    Ok(FittedBaseline {
        kind: BaselineKind::Mean,
        training_observations: history.len(),
        mean: Some(mean),
        order: 0,
        coefficients: Vec::new(),
    })
}

/// Fit an AR(p) model with an intercept using Householder QR.
///
/// The objective is ordinary least squares when `ridge_penalty == 0`.
/// Otherwise it adds `ridge_penalty * sum(lag_coefficient^2)`; the
/// intercept is intentionally unpenalized. The first `order` observations
/// form the initial lag context and are not target rows.
pub fn fit_linear_ar(
    history: &[f64],
    order: usize,
    ridge_penalty: f64,
) -> Result<FittedBaseline, BaselineError> {
    validate_history(history)?;

    if order == 0 {
        return Err(BaselineError::InvalidAutoregressionOrder);
    }
    if !ridge_penalty.is_finite() || ridge_penalty < 0.0 {
        return Err(BaselineError::InvalidRidgePenalty);
    }
    if history.len() <= order {
        return Err(BaselineError::InsufficientSamples {
            required: order + 1,
            actual: history.len(),
        });
    }

    let feature_count = order + 1;
    let target_rows = history.len() - order;
    if target_rows < feature_count {
        return Err(BaselineError::InsufficientSamples {
            required: 2 * order + 1,
            actual: history.len(),
        });
    }

    let penalty_rows = if ridge_penalty > 0.0 { order } else { 0 };
    let mut matrix = Vec::with_capacity(target_rows + penalty_rows);
    let mut target = Vec::with_capacity(target_rows + penalty_rows);

    for time in order..history.len() {
        let mut row = Vec::with_capacity(feature_count);
        row.push(1.0);
        for lag in 1..=order {
            row.push(history[time - lag]);
        }
        matrix.push(row);
        target.push(history[time]);
    }

    if ridge_penalty > 0.0 {
        let penalty_scale = ridge_penalty.sqrt();
        if !penalty_scale.is_finite() {
            return Err(BaselineError::NumericalFailure);
        }
        for coefficient in 1..=order {
            let mut row = vec![0.0; feature_count];
            row[coefficient] = penalty_scale;
            matrix.push(row);
            target.push(0.0);
        }
    }

    let coefficients = solve_least_squares_qr(matrix, target)?;
    Ok(FittedBaseline {
        kind: BaselineKind::LinearAutoregression,
        training_observations: history.len(),
        mean: None,
        order,
        coefficients,
    })
}

fn validate_history(history: &[f64]) -> Result<(), BaselineError> {
    if history.is_empty() {
        return Err(BaselineError::EmptyHistory);
    }
    if history.iter().any(|value| !value.is_finite()) {
        return Err(BaselineError::NonFiniteObservation);
    }
    Ok(())
}

/// Solve min ||A x - b||_2 using unpivoted Householder QR.
///
/// This implementation is intentionally dependency-free. It rejects a
/// numerically rank-deficient design rather than silently returning unstable
/// coefficients.
fn solve_least_squares_qr(
    mut matrix: Vec<Vec<f64>>,
    mut target: Vec<f64>,
) -> Result<Vec<f64>, BaselineError> {
    let row_count = matrix.len();
    let column_count = matrix
        .first()
        .map(Vec::len)
        .ok_or(BaselineError::RankDeficientDesign)?;

    if row_count < column_count
        || target.len() != row_count
        || matrix.iter().any(|row| row.len() != column_count)
    {
        return Err(BaselineError::RankDeficientDesign);
    }

    for pivot in 0..column_count {
        let column_norm = matrix
            .iter()
            .skip(pivot)
            .fold(0.0_f64, |norm, row| norm.hypot(row[pivot]));
        if !column_norm.is_finite() || column_norm <= f64::MIN_POSITIVE {
            return Err(BaselineError::RankDeficientDesign);
        }

        let leading = matrix[pivot][pivot];
        let alpha = if leading >= 0.0 {
            -column_norm
        } else {
            column_norm
        };

        let mut reflector: Vec<f64> = matrix.iter().skip(pivot).map(|row| row[pivot]).collect();
        reflector[0] -= alpha;

        let reflector_norm = reflector
            .iter()
            .fold(0.0_f64, |norm, value| norm.hypot(*value));
        if !reflector_norm.is_finite() || reflector_norm <= f64::MIN_POSITIVE {
            return Err(BaselineError::RankDeficientDesign);
        }
        for value in &mut reflector {
            *value /= reflector_norm;
        }

        for column in pivot..column_count {
            let projection: f64 = matrix
                .iter()
                .skip(pivot)
                .zip(reflector.iter())
                .map(|(row, direction)| row[column] * direction)
                .sum();
            let scale = 2.0 * projection;
            for (row, direction) in matrix.iter_mut().skip(pivot).zip(reflector.iter()) {
                row[column] -= scale * direction;
            }
        }

        let target_projection: f64 = target
            .iter()
            .skip(pivot)
            .zip(reflector.iter())
            .map(|(value, direction)| value * direction)
            .sum();
        let target_scale = 2.0 * target_projection;
        for (value, direction) in target.iter_mut().skip(pivot).zip(reflector.iter()) {
            *value -= target_scale * direction;
        }

        matrix[pivot][pivot] = alpha;
        for row in matrix.iter_mut().skip(pivot + 1) {
            row[pivot] = 0.0;
        }
    }

    let max_diagonal = (0..column_count)
        .map(|index| matrix[index][index].abs())
        .fold(0.0_f64, f64::max);
    let rank_tolerance = max_diagonal * f64::EPSILON * (row_count + column_count) as f64 * 8.0;

    let mut coefficients = vec![0.0; column_count];
    for row in (0..column_count).rev() {
        let known_contribution: f64 = matrix[row]
            .iter()
            .enumerate()
            .skip(row + 1)
            .map(|(column, value)| value * coefficients[column])
            .sum();
        let diagonal = matrix[row][row];
        if !diagonal.is_finite() || diagonal.abs() <= rank_tolerance {
            return Err(BaselineError::RankDeficientDesign);
        }
        coefficients[row] = (target[row] - known_contribution) / diagonal;
        if !coefficients[row].is_finite() {
            return Err(BaselineError::NumericalFailure);
        }
    }

    Ok(coefficients)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistence_uses_the_latest_as_of_observation() {
        let model = fit_persistence(&[1.0, 2.0, 3.0]).unwrap();
        let forecast = model.forecast(&[2.0, 4.0, 7.0], 3).unwrap();
        assert_eq!(model.kind(), BaselineKind::Persistence);
        assert_eq!(forecast, vec![7.0, 7.0, 7.0]);
    }

    #[test]
    fn mean_forecast_repeats_the_training_mean() {
        let model = fit_mean(&[1.0, 3.0, 5.0, 7.0]).unwrap();
        let forecast = model.forecast(&[20.0, 30.0], 2).unwrap();
        assert_eq!(model.kind(), BaselineKind::Mean);
        assert_eq!(forecast, vec![4.0, 4.0]);
    }

    #[test]
    fn linear_ar_recovers_a_known_second_order_process() {
        let mut history = vec![0.7, -0.2];
        for _ in 0..100 {
            let time = history.len();
            let next = 0.25 + 1.25 * history[time - 1] - 0.8 * history[time - 2];
            history.push(next);
        }

        let model = fit_linear_ar(&history, 2, 0.0).unwrap();
        assert_eq!(model.kind(), BaselineKind::LinearAutoregression);
        assert_eq!(model.coefficients().len(), 3);
        assert!((model.coefficients()[0] - 0.25).abs() < 1e-8);
        assert!((model.coefficients()[1] - 1.25).abs() < 1e-8);
        assert!((model.coefficients()[2] + 0.8).abs() < 1e-8);

        let mut expected_context = history.clone();
        let mut expected = Vec::new();
        for _ in 0..5 {
            let time = expected_context.len();
            let next = 0.25 + 1.25 * expected_context[time - 1] - 0.8 * expected_context[time - 2];
            expected_context.push(next);
            expected.push(next);
        }
        let forecast = model.forecast(&history, 5).unwrap();
        for (actual, reference) in forecast.iter().zip(expected.iter()) {
            assert!((actual - reference).abs() < 1e-8);
        }
    }

    #[test]
    fn ridge_regularized_ar_remains_finite() {
        let history: Vec<f64> = (0..50)
            .map(|index| (index as f64 * 0.17).sin() + 0.2 * (index as f64 * 0.031).cos())
            .collect();
        let model = fit_linear_ar(&history, 4, 1e-4).unwrap();
        let forecast = model.forecast(&history, 12).unwrap();
        assert_eq!(forecast.len(), 12);
        assert!(forecast.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn rejects_invalid_orders_samples_and_penalties() {
        assert_eq!(
            fit_linear_ar(&[1.0, 2.0], 0, 0.0),
            Err(BaselineError::InvalidAutoregressionOrder)
        );
        assert_eq!(
            fit_linear_ar(&[1.0, 2.0, 3.0, 4.0], 2, 0.0),
            Err(BaselineError::InsufficientSamples {
                required: 5,
                actual: 4,
            })
        );
        assert_eq!(
            fit_linear_ar(&[1.0, 2.0, 3.0, 4.0, 5.0], 1, f64::NAN),
            Err(BaselineError::InvalidRidgePenalty)
        );
    }

    #[test]
    fn rejects_nonfinite_observations_and_short_forecast_context() {
        assert_eq!(
            fit_mean(&[1.0, f64::NAN]),
            Err(BaselineError::NonFiniteObservation)
        );

        let model = fit_linear_ar(&[0.1, 0.3, 0.4, 0.6, 0.9], 2, 1e-4).unwrap();
        assert_eq!(
            model.forecast(&[0.1], 1),
            Err(BaselineError::InsufficientForecastContext {
                required: 2,
                actual: 1,
            })
        );
    }

    #[test]
    fn zero_length_forecast_is_empty() {
        let model = fit_mean(&[2.0, 4.0]).unwrap();
        assert_eq!(model.forecast(&[3.0], 0).unwrap(), Vec::<f64>::new());
    }
}
