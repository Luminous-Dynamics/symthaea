//! Sampling-aware metrics for irregular-time forecasting.
//!
//! Ordinary MSE/MAE estimate error under the observed timestamp distribution.
//! For irregular sampling this can over-weight densely observed periods. This
//! module additionally computes self-normalized inverse-density CSE/CAE using
//! a leave-one-out Gaussian KDE over the observed timestamps.
//!
//! Method reference: Li et al., "Beyond MSE: Rethinking the Evaluation Metric
//! and Benchmarking for Irregular Time Series Forecasting" (arXiv:2608.17293,
//! 2026). Bandwidth is evaluator-supplied and must be preregistered. These are
//! estimates, not exact integrals; publish MSE/MAE, CSE/CAE, bandwidth and ESS.

use super::continuous_risk::{
    integrate_piecewise_linear_risk, ContinuousRiskError, PiecewiseLinearRisk,
};
use super::irregular::{
    verify_irregular_trace, IrregularOracle, IrregularTask, IrregularVerificationError,
};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IrregularForecastPrediction {
    pub time_us: u64,
    pub fast: f64,
    pub medium: f64,
    pub slow: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CseKernelConfig {
    /// Gaussian KDE bandwidth in microseconds. Must be fixed before confirmatory runs.
    pub bandwidth_us: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IrregularMetricChannel {
    pub mse: f64,
    pub mae: f64,
    /// Self-normalized inverse-density weighted estimate of continuous-time squared error.
    pub cse: f64,
    /// Self-normalized inverse-density weighted estimate of continuous-time absolute error.
    pub cae: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IrregularForecastMetrics {
    pub sample_count: usize,
    pub bandwidth_us: u64,
    pub effective_sample_size: f64,
    /// Exact reference for the piecewise-linear residual path between observed endpoints.
    /// This does not establish correctness outside the observed support.
    pub piecewise_linear_reference: PiecewiseLinearRisk,
    pub fast: IrregularMetricChannel,
    pub medium: IrregularMetricChannel,
    pub slow: IrregularMetricChannel,
}

impl IrregularForecastMetrics {
    pub fn mean_mse(&self) -> f64 {
        (self.fast.mse + self.medium.mse + self.slow.mse) / 3.0
    }

    pub fn mean_mae(&self) -> f64 {
        (self.fast.mae + self.medium.mae + self.slow.mae) / 3.0
    }

    pub fn mean_cse(&self) -> f64 {
        (self.fast.cse + self.medium.cse + self.slow.cse) / 3.0
    }

    pub fn mean_cae(&self) -> f64 {
        (self.fast.cae + self.medium.cae + self.slow.cae) / 3.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum IrregularMetricError {
    InvalidTask(IrregularVerificationError),
    ZeroBandwidth,
    TooFewSamples { count: usize, minimum: usize },
    PredictionCountMismatch { expected: usize, actual: usize },
    TimestampMismatch { index: usize },
    NonFinitePrediction { index: usize },
    InvalidDensityEstimate { index: usize },
    ContinuousRisk(ContinuousRiskError),
    NonFiniteMetric,
}

impl fmt::Display for IrregularMetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTask(error) => write!(f, "cannot score invalid irregular task: {error}"),
            Self::ZeroBandwidth => write!(f, "CSE Gaussian KDE bandwidth must be greater than zero"),
            Self::TooFewSamples { count, minimum } => {
                write!(f, "CSE requires at least {minimum} samples, received {count}")
            }
            Self::PredictionCountMismatch { expected, actual } => write!(
                f,
                "irregular prediction count {actual} differs from expected count {expected}"
            ),
            Self::TimestampMismatch { index } => {
                write!(f, "irregular prediction timestamp differs at index {index}")
            }
            Self::NonFinitePrediction { index } => {
                write!(f, "irregular prediction contains NaN or infinity at index {index}")
            }
            Self::InvalidDensityEstimate { index } => {
                write!(f, "leave-one-out KDE density is invalid at sample {index}")
            }
            Self::ContinuousRisk(error) => write!(f, "continuous-time reference failed: {error}"),
            Self::NonFiniteMetric => write!(f, "irregular forecast metric is non-finite"),
        }
    }
}

impl std::error::Error for IrregularMetricError {}

/// Score one prediction at each observed timestamp.
///
/// For timestamp t_i, the leave-one-out KDE is
/// p_hat(t_i) = (1 / ((n-1) h sqrt(2 pi))) * sum_{j != i} exp(-0.5 * ((t_i-t_j)/h)^2).
/// The common KDE normalization cancels in the self-normalized inverse-density
/// weights. Log-sum-exp is used to avoid underflow for sparse timestamp gaps.
pub fn score_irregular_forecasts(
    task: &IrregularTask,
    oracle: &IrregularOracle,
    predictions: &[IrregularForecastPrediction],
    config: CseKernelConfig,
) -> Result<IrregularForecastMetrics, IrregularMetricError> {
    verify_irregular_trace(task, oracle).map_err(IrregularMetricError::InvalidTask)?;
    if config.bandwidth_us == 0 {
        return Err(IrregularMetricError::ZeroBandwidth);
    }
    let count = task.observations.len();
    if count < 2 {
        return Err(IrregularMetricError::TooFewSamples {
            count,
            minimum: 2,
        });
    }
    if predictions.len() != count {
        return Err(IrregularMetricError::PredictionCountMismatch {
            expected: count,
            actual: predictions.len(),
        });
    }

    let mut timestamps = Vec::with_capacity(count);
    let mut timestamps_us = Vec::with_capacity(count);
    let mut residuals = Vec::with_capacity(count);
    let mut ordinary_squared_sums = [0.0_f64; 3];
    let mut ordinary_absolute_sums = [0.0_f64; 3];

    for (index, ((observation, point), prediction)) in task
        .observations
        .iter()
        .zip(&oracle.points)
        .zip(predictions)
        .enumerate()
    {
        if prediction.time_us != observation.time_us || point.time_us != observation.time_us {
            return Err(IrregularMetricError::TimestampMismatch { index });
        }
        if !prediction.fast.is_finite()
            || !prediction.medium.is_finite()
            || !prediction.slow.is_finite()
        {
            return Err(IrregularMetricError::NonFinitePrediction { index });
        }
        timestamps.push(observation.time_us as f64);
        timestamps_us.push(observation.time_us);
        let predicted = [prediction.fast, prediction.medium, prediction.slow];
        let truth = [
            f64::from(point.latent[0]),
            f64::from(point.latent[1]),
            f64::from(point.latent[2]),
        ];
        let mut sample_residuals = [0.0_f64; 3];
        for channel in 0..3 {
            let error = predicted[channel] - truth[channel];
            sample_residuals[channel] = error;
            let absolute = error.abs();
            let squared = error * error;
            ordinary_absolute_sums[channel] += absolute;
            ordinary_squared_sums[channel] += squared;
        }
        residuals.push(sample_residuals);
    }

    let piecewise_linear_reference = integrate_piecewise_linear_risk(
        &timestamps_us,
        &residuals,
    ).map_err(IrregularMetricError::ContinuousRisk)?;

    let inverse_density_logs = leave_one_out_inverse_density_logs(
        &timestamps,
        config.bandwidth_us as f64,
    )?;
    let max_log_weight = inverse_density_logs
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_log_weight.is_finite() {
        return Err(IrregularMetricError::NonFiniteMetric);
    }
    let weights: Vec<f64> = inverse_density_logs
        .iter()
        .map(|log_weight| (log_weight - max_log_weight).exp())
        .collect();
    let weight_sum: f64 = weights.iter().sum();
    let squared_weight_sum: f64 = weights.iter().map(|weight| weight * weight).sum();
    if !weight_sum.is_finite() || weight_sum <= 0.0
        || !squared_weight_sum.is_finite() || squared_weight_sum <= 0.0
    {
        return Err(IrregularMetricError::NonFiniteMetric);
    }
    let effective_sample_size = (weight_sum * weight_sum) / squared_weight_sum;

    let mut channels = Vec::with_capacity(3);
    for channel in 0..3 {
        let mut weighted_squared = 0.0_f64;
        let mut weighted_absolute = 0.0_f64;
        for (index, sample_residuals) in residuals.iter().enumerate() {
            let error = sample_residuals[channel];
            weighted_squared += weights[index] * error * error;
            weighted_absolute += weights[index] * error.abs();
        }
        let channel_metrics = IrregularMetricChannel {
            mse: ordinary_squared_sums[channel] / count as f64,
            mae: ordinary_absolute_sums[channel] / count as f64,
            cse: weighted_squared / weight_sum,
            cae: weighted_absolute / weight_sum,
        };
        if !channel_metrics.mse.is_finite()
            || !channel_metrics.mae.is_finite()
            || !channel_metrics.cse.is_finite()
            || !channel_metrics.cae.is_finite()
        {
            return Err(IrregularMetricError::NonFiniteMetric);
        }
        channels.push(channel_metrics);
    }

    Ok(IrregularForecastMetrics {
        sample_count: count,
        bandwidth_us: config.bandwidth_us,
        effective_sample_size,
        piecewise_linear_reference,
        fast: channels[0].clone(),
        medium: channels[1].clone(),
        slow: channels[2].clone(),
    })
}

fn leave_one_out_inverse_density_logs(
    timestamps: &[f64],
    bandwidth_us: f64,
) -> Result<Vec<f64>, IrregularMetricError> {
    if timestamps.len() < 2 || !bandwidth_us.is_finite() || bandwidth_us <= 0.0 {
        return Err(IrregularMetricError::NonFiniteMetric);
    }
    let mut inverse_density_logs = Vec::with_capacity(timestamps.len());
    for (index, time) in timestamps.iter().enumerate() {
        let mut exponents = Vec::with_capacity(timestamps.len().saturating_sub(1));
        for (other_index, other_time) in timestamps.iter().enumerate() {
            if index == other_index {
                continue;
            }
            let scaled = (time - other_time) / bandwidth_us;
            exponents.push(-0.5 * scaled * scaled);
        }
        let maximum = exponents
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if !maximum.is_finite() {
            return Err(IrregularMetricError::InvalidDensityEstimate { index });
        }
        let sum: f64 = exponents
            .iter()
            .map(|exponent| (exponent - maximum).exp())
            .sum();
        if !sum.is_finite() || sum <= 0.0 {
            return Err(IrregularMetricError::InvalidDensityEstimate { index });
        }
        let log_density_without_common_constant = maximum + sum.ln();
        inverse_density_logs.push(-log_density_without_common_constant);
    }
    Ok(inverse_density_logs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::irregular::{generate_irregular_task, IrregularConfig};

    fn fixture() -> (IrregularTask, IrregularOracle) {
        generate_irregular_task(IrregularConfig::reference(0xC5E_2026)).unwrap()
    }

    fn oracle_predictions(
        task: &IrregularTask,
        oracle: &IrregularOracle,
    ) -> Vec<IrregularForecastPrediction> {
        task.observations
            .iter()
            .zip(&oracle.points)
            .map(|(observation, point)| IrregularForecastPrediction {
                time_us: observation.time_us,
                fast: f64::from(point.latent[0]),
                medium: f64::from(point.latent[1]),
                slow: f64::from(point.latent[2]),
            })
            .collect()
    }

    #[test]
    fn perfect_irregular_forecasts_have_zero_mse_mae_cse_and_cae() {
        let (task, oracle) = fixture();
        let predictions = oracle_predictions(&task, &oracle);
        let scores = score_irregular_forecasts(
            &task,
            &oracle,
            &predictions,
            CseKernelConfig { bandwidth_us: 20_000 },
        )
        .unwrap();
        assert_eq!(scores.fast.mse, 0.0);
        assert_eq!(scores.fast.mae, 0.0);
        assert_eq!(scores.fast.cse, 0.0);
        assert_eq!(scores.fast.cae, 0.0);
        assert_eq!(scores.piecewise_linear_reference.mean_squared_error, [0.0; 3]);
        assert_eq!(scores.piecewise_linear_reference.mean_absolute_error, [0.0; 3]);
        assert!(scores.effective_sample_size > 0.0);
        assert!(scores.effective_sample_size <= scores.sample_count as f64);
    }

    #[test]
    fn dense_observation_cluster_is_downweighted_relative_to_sparse_regions() {
        let timestamps = [0.0, 1.0, 2.0, 3.0, 4.0, 100.0, 200.0];
        let logs = leave_one_out_inverse_density_logs(&timestamps, 10.0).unwrap();
        let dense_average = logs[..5].iter().sum::<f64>() / 5.0;
        let sparse_average = logs[5..].iter().sum::<f64>() / 2.0;
        // Inverse-density log weights are larger in sparse regions.
        assert!(sparse_average > dense_average);
    }

    #[test]
    fn sampling_aware_cse_downweights_errors_in_densely_sampled_regions() {
        let mut config = IrregularConfig::reference(0xC5E_2027);
        config.slots = 512;
        config.missing_per_mille = 0;
        config.jitter_per_mille = 900;
        let (task, oracle) = generate_irregular_task(config).unwrap();
        let mut predictions = oracle_predictions(&task, &oracle);

        let timestamps = task.observations.iter()
            .map(|observation| observation.time_us as f64)
            .collect::<Vec<_>>();
        let inverse_density_logs =
            leave_one_out_inverse_density_logs(&timestamps, 10_000.0).unwrap();
        let mut density_order = inverse_density_logs.iter()
            .enumerate()
            .map(|(index, log_weight)| (*log_weight, index))
            .collect::<Vec<_>>();
        density_order.sort_by(|left, right| left.0.total_cmp(&right.0));
        // Lower inverse-density log weight means locally higher observation density.
        for (_, index) in density_order.into_iter().take(64) {
            predictions[index].fast += 100.0;
        }

        let scores = score_irregular_forecasts(
            &task,
            &oracle,
            &predictions,
            CseKernelConfig { bandwidth_us: 10_000 },
        )
        .unwrap();
        assert!(scores.fast.mse > 0.0);
        assert!(scores.fast.cse < scores.fast.mse);
        assert!(scores.effective_sample_size > 0.0);
    }

    #[test]
    fn constant_residuals_match_cse_and_exact_piecewise_linear_reference() {
        let (task, oracle) = fixture();
        let mut predictions = oracle_predictions(&task, &oracle);
        for prediction in &mut predictions {
            prediction.fast += 1.0;
            prediction.medium -= 2.0;
            prediction.slow += 0.5;
        }
        let scores = score_irregular_forecasts(
            &task,
            &oracle,
            &predictions,
            CseKernelConfig { bandwidth_us: 20_000 },
        )
        .unwrap();

        for (channel, expected_mse, expected_mae) in [
            (&scores.fast, 1.0, 1.0),
            (&scores.medium, 4.0, 2.0),
            (&scores.slow, 0.25, 0.5),
        ] {
            assert!((channel.mse - expected_mse).abs() < 1.0e-10);
            assert!((channel.mae - expected_mae).abs() < 1.0e-10);
            assert!((channel.cse - expected_mse).abs() < 1.0e-10);
            assert!((channel.cae - expected_mae).abs() < 1.0e-10);
        }
        assert_eq!(
            scores.piecewise_linear_reference.mean_squared_error,
            [1.0, 4.0, 0.25]
        );
        assert_eq!(
            scores.piecewise_linear_reference.mean_absolute_error,
            [1.0, 2.0, 0.5]
        );
    }

    #[test]
    fn zero_bandwidth_is_rejected() {
        let (task, oracle) = fixture();
        let predictions = oracle_predictions(&task, &oracle);
        assert_eq!(
            score_irregular_forecasts(
                &task,
                &oracle,
                &predictions,
                CseKernelConfig { bandwidth_us: 0 },
            ),
            Err(IrregularMetricError::ZeroBandwidth)
        );
    }

    #[test]
    fn too_few_samples_are_rejected() {
        let mut config = IrregularConfig::reference(0xC5E_2028);
        config.slots = 2;
        config.missing_per_mille = 0;
        config.jitter_per_mille = 0;
        let (mut task, mut oracle) = generate_irregular_task(config).unwrap();
        task.observations.truncate(1);
        oracle.points.truncate(1);
        oracle.scheduled_slots = 2;
        oracle.dropped_slots = vec![1];
        task.content_sha256 = super::super::irregular::irregular_content_digest(&task);
        oracle.content_sha256 = super::super::irregular::irregular_oracle_digest(&oracle);
        let predictions = oracle_predictions(&task, &oracle);
        assert_eq!(
            score_irregular_forecasts(
                &task,
                &oracle,
                &predictions,
                CseKernelConfig { bandwidth_us: 1_000 },
            ),
            Err(IrregularMetricError::TooFewSamples { count: 1, minimum: 2 })
        );
    }

    #[test]
    fn prediction_count_mismatch_is_rejected() {
        let (task, oracle) = fixture();
        let mut predictions = oracle_predictions(&task, &oracle);
        predictions.pop();
        assert!(matches!(
            score_irregular_forecasts(
                &task,
                &oracle,
                &predictions,
                CseKernelConfig { bandwidth_us: 20_000 },
            ),
            Err(IrregularMetricError::PredictionCountMismatch { .. })
        ));
    }

    #[test]
    fn changed_timestamp_and_nonfinite_predictions_are_rejected() {
        let (task, oracle) = fixture();
        let mut predictions = oracle_predictions(&task, &oracle);
        predictions[0].time_us += 1;
        assert!(matches!(
            score_irregular_forecasts(
                &task,
                &oracle,
                &predictions,
                CseKernelConfig { bandwidth_us: 20_000 },
            ),
            Err(IrregularMetricError::TimestampMismatch { .. })
        ));
        predictions = oracle_predictions(&task, &oracle);
        predictions[0].fast = f64::NAN;
        assert!(matches!(
            score_irregular_forecasts(
                &task,
                &oracle,
                &predictions,
                CseKernelConfig { bandwidth_us: 20_000 },
            ),
            Err(IrregularMetricError::NonFinitePrediction { .. })
        ));
    }
}
