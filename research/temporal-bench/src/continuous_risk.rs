//! Exact continuous-time risk for the declared piecewise-linear residual path.
//!
//! This is a calibration reference, not a claim about unobserved process values.
//! Each residual channel is linearly interpolated between adjacent observed
//! timestamps; squared and absolute loss are then integrated in closed form.

use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub struct PiecewiseLinearRisk {
    pub duration_us: u128,
    pub integrated_squared_error: [f64; 3],
    pub integrated_absolute_error: [f64; 3],
    pub mean_squared_error: [f64; 3],
    pub mean_absolute_error: [f64; 3],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContinuousRiskError {
    TooFewSamples { count: usize },
    LengthMismatch { timestamps: usize, residuals: usize },
    TimestampNotIncreasing { index: usize },
    NonFiniteResidual { sample: usize, channel: usize },
    NonFiniteIntegral,
}

impl fmt::Display for ContinuousRiskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewSamples { count } => write!(
                f,
                "piecewise-linear integration requires at least two samples, received {count}"
            ),
            Self::LengthMismatch { timestamps, residuals } => write!(
                f,
                "timestamp count {timestamps} differs from residual count {residuals}"
            ),
            Self::TimestampNotIncreasing { index } => write!(
                f,
                "timestamps must be strictly increasing; invalid segment ending at index {index}"
            ),
            Self::NonFiniteResidual { sample, channel } => write!(
                f,
                "residual at sample {sample}, channel {channel} is not finite"
            ),
            Self::NonFiniteIntegral => write!(f, "piecewise-linear risk integral is not finite"),
        }
    }
}

impl std::error::Error for ContinuousRiskError {}

/// Integrate squared and absolute residuals exactly for a piecewise-linear
/// residual function between observed endpoints.
///
/// For one segment of duration dt and endpoint residuals e0,e1:
///   integral(e^2) = dt * (e0^2 + e0*e1 + e1^2) / 3.
///
/// The absolute integral handles zero crossings explicitly. The reported
/// means are normalized by the total observed support duration. No
/// extrapolation before the first or after the last timestamp is performed.
pub fn integrate_piecewise_linear_risk(
    timestamps_us: &[u64],
    residuals: &[[f64; 3]],
) -> Result<PiecewiseLinearRisk, ContinuousRiskError> {
    if timestamps_us.len() != residuals.len() {
        return Err(ContinuousRiskError::LengthMismatch {
            timestamps: timestamps_us.len(),
            residuals: residuals.len(),
        });
    }
    if timestamps_us.len() < 2 {
        return Err(ContinuousRiskError::TooFewSamples {
            count: timestamps_us.len(),
        });
    }
    for (sample, values) in residuals.iter().enumerate() {
        for (channel, value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(ContinuousRiskError::NonFiniteResidual { sample, channel });
            }
        }
    }

    let mut duration_us = 0_u128;
    let mut integrated_squared_error = [0.0_f64; 3];
    let mut integrated_absolute_error = [0.0_f64; 3];

    for index in 0..timestamps_us.len() - 1 {
        let start = timestamps_us[index];
        let end = timestamps_us[index + 1];
        if end <= start {
            return Err(ContinuousRiskError::TimestampNotIncreasing { index: index + 1 });
        }
        let dt = end - start;
        duration_us = duration_us
            .checked_add(u128::from(dt))
            .ok_or(ContinuousRiskError::NonFiniteIntegral)?;
        let dt_f64 = dt as f64;

        for channel in 0..3 {
            let left = residuals[index][channel];
            let right = residuals[index + 1][channel];
            let squared_segment =
                dt_f64 * (left * left + left * right + right * right) / 3.0;
            let absolute_segment = integrate_linear_absolute(dt_f64, left, right);
            integrated_squared_error[channel] += squared_segment;
            integrated_absolute_error[channel] += absolute_segment;
        }
    }

    let duration = duration_us as f64;
    let mut mean_squared_error = [0.0_f64; 3];
    let mut mean_absolute_error = [0.0_f64; 3];
    for channel in 0..3 {
        mean_squared_error[channel] = integrated_squared_error[channel] / duration;
        mean_absolute_error[channel] = integrated_absolute_error[channel] / duration;
        if !integrated_squared_error[channel].is_finite()
            || !integrated_absolute_error[channel].is_finite()
            || !mean_squared_error[channel].is_finite()
            || !mean_absolute_error[channel].is_finite()
        {
            return Err(ContinuousRiskError::NonFiniteIntegral);
        }
    }

    Ok(PiecewiseLinearRisk {
        duration_us,
        integrated_squared_error,
        integrated_absolute_error,
        mean_squared_error,
        mean_absolute_error,
    })
}

fn integrate_linear_absolute(dt: f64, left: f64, right: f64) -> f64 {
    let left_abs = left.abs();
    let right_abs = right.abs();
    if left == 0.0 || right == 0.0 || left.is_sign_positive() == right.is_sign_positive() {
        return dt * (left_abs + right_abs) / 2.0;
    }
    let denominator = left_abs + right_abs;
    if denominator == 0.0 {
        return 0.0;
    }
    dt * (left_abs * left_abs + right_abs * right_abs) / (2.0 * denominator)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn risk(times: &[u64], residuals: &[[f64; 3]]) -> PiecewiseLinearRisk {
        integrate_piecewise_linear_risk(times, residuals).unwrap()
    }

    #[test]
    fn constant_positive_residual_has_unit_squared_and_absolute_risk() {
        let result = risk(&[0, 10], &[[1.0; 3], [1.0; 3]]);
        assert_eq!(result.duration_us, 10);
        assert_eq!(result.mean_squared_error, [1.0; 3]);
        assert_eq!(result.mean_absolute_error, [1.0; 3]);
    }

    #[test]
    fn linear_zero_to_two_residual_has_exact_four_thirds_squared_risk() {
        let result = risk(&[0, 3], &[[0.0; 3], [2.0; 3]]);
        for channel in 0..3 {
            assert!((result.mean_squared_error[channel] - 4.0 / 3.0).abs() < 1.0e-12);
            assert!((result.mean_absolute_error[channel] - 1.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn sign_crossing_uses_exact_absolute_area_not_endpoint_trapezoid() {
        let result = risk(&[0, 2], &[[-1.0; 3], [1.0; 3]]);
        for channel in 0..3 {
            assert!((result.mean_squared_error[channel] - 1.0 / 3.0).abs() < 1.0e-12);
            assert!((result.mean_absolute_error[channel] - 0.5).abs() < 1.0e-12);
        }
    }

    #[test]
    fn negative_residual_segment_matches_closed_form() {
        let result = risk(&[0, 6], &[[-3.0; 3], [-1.0; 3]]);
        for channel in 0..3 {
            assert!((result.mean_squared_error[channel] - 13.0 / 3.0).abs() < 1.0e-12);
            assert!((result.mean_absolute_error[channel] - 2.0).abs() < 1.0e-12);
        }
    }

    #[test]
    fn multiple_segments_are_duration_weighted() {
        let result = risk(
            &[0, 1, 4],
            &[[0.0; 3], [1.0; 3], [1.0; 3]],
        );
        for channel in 0..3 {
            // First interval contributes 1/3; the remaining three contribute 3.
            assert!((result.integrated_squared_error[channel] - (1.0 / 3.0 + 3.0)).abs() < 1.0e-12);
            assert!((result.mean_squared_error[channel] - (10.0 / 3.0 / 4.0)).abs() < 1.0e-12);
        }
    }

    #[test]
    fn duplicate_or_reversed_timestamps_are_rejected() {
        assert_eq!(
            integrate_piecewise_linear_risk(&[0, 0], &[[0.0; 3], [1.0; 3]]),
            Err(ContinuousRiskError::TimestampNotIncreasing { index: 1 })
        );
    }

    #[test]
    fn length_and_sample_count_mismatches_are_rejected() {
        assert_eq!(
            integrate_piecewise_linear_risk(&[0, 1], &[[0.0; 3]]),
            Err(ContinuousRiskError::LengthMismatch { timestamps: 2, residuals: 1 })
        );
        assert_eq!(
            integrate_piecewise_linear_risk(&[0], &[[0.0; 3]]),
            Err(ContinuousRiskError::TooFewSamples { count: 1 })
        );
    }

    #[test]
    fn nonfinite_residuals_are_rejected() {
        assert_eq!(
            integrate_piecewise_linear_risk(&[0, 1], &[[0.0; 3], [f64::NAN, 0.0, 0.0]]),
            Err(ContinuousRiskError::NonFiniteResidual { sample: 1, channel: 0 })
        );
    }
}
