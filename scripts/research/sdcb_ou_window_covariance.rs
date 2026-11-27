//! Exact covariance oracle for averages of a stationary Ornstein–Uhlenbeck factor.
//!
//! Build from the repository root:
//!   rustc --edition=2024 --test scripts/research/sdcb_ou_window_covariance.rs -o /tmp/sdcb-ou-window
//! Run:
//!   /tmp/sdcb-ou-window
//!
//! This is one covariance primitive for time-windowed measurements. It is not
//! a complete dynamic filter, covariance-parameter estimator, or metrology
//! assurance service. Times are finite seconds relative to a declared epoch.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeWindow {
    pub start_seconds: f64,
    pub end_seconds: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidWindow,
    InvalidStationaryVariance,
    InvalidCorrelationTime,
    InvalidWhiteNoiseIntensity,
    NumericFailure,
}

fn validate_window(window: TimeWindow) -> Result<(), Error> {
    if !window.start_seconds.is_finite()
        || !window.end_seconds.is_finite()
        || window.end_seconds <= window.start_seconds
    {
        return Err(Error::InvalidWindow);
    }
    Ok(())
}

/// Stable evaluation of exp(-x) + x - 1 for x >= 0.
fn exp_kernel_remainder(x: f64) -> f64 {
    if x < 1.0e-3 {
        // Taylor expansion avoids cancellation of the first-order terms.
        let x2 = x * x;
        x2 * (0.5
            + x * (-1.0 / 6.0
                + x * (1.0 / 24.0
                    + x * (-1.0 / 120.0
                        + x * (1.0 / 720.0
                            + x * (-1.0 / 5040.0
                                + x * (1.0 / 40320.0)))))))
    } else {
        (-x).exp_m1() + x
    }
}

fn one_minus_exp_neg(x: f64) -> f64 {
    -(-x).exp_m1()
}

/// Integral of exp(-|t-s|/tau) over the two windows.
fn ou_kernel_integral(
    first: TimeWindow,
    second: TimeWindow,
    tau: f64,
) -> Result<f64, Error> {
    let tau_squared = tau * tau;
    if !tau_squared.is_finite() {
        return Err(Error::NumericFailure);
    }

    // The disjoint-window form is factored and remains stable for large gaps.
    if first.end_seconds <= second.start_seconds {
        let gap = second.start_seconds - first.end_seconds;
        let first_length = first.end_seconds - first.start_seconds;
        let second_length = second.end_seconds - second.start_seconds;
        let value = tau_squared
            * (-gap / tau).exp()
            * one_minus_exp_neg(first_length / tau)
            * one_minus_exp_neg(second_length / tau);
        return value.is_finite().then_some(value).ok_or(Error::NumericFailure);
    }
    if second.end_seconds <= first.start_seconds {
        return ou_kernel_integral(second, first, tau);
    }

    // For overlapping windows, use the twice-integrated stationary OU kernel.
    // The constant terms cancel analytically in this four-term combination.
    let h = |seconds: f64| exp_kernel_remainder(seconds.abs() / tau);
    let integral = tau_squared
        * (h(first.end_seconds - second.start_seconds)
            - h(first.start_seconds - second.start_seconds)
            - h(first.end_seconds - second.end_seconds)
            + h(first.start_seconds - second.end_seconds));

    if !integral.is_finite() {
        return Err(Error::NumericFailure);
    }
    // A covariance integral is non-negative. Permit only rounding-scale
    // negative residue; materially negative values indicate numerical failure.
    let scale = tau_squared
        * (h(first.end_seconds - second.start_seconds).abs()
            + h(first.start_seconds - second.start_seconds).abs()
            + h(first.end_seconds - second.end_seconds).abs()
            + h(first.start_seconds - second.end_seconds).abs())
            .max(f64::MIN_POSITIVE);
    if integral < -64.0 * f64::EPSILON * scale {
        return Err(Error::NumericFailure);
    }
    Ok(integral.max(0.0))
}

/// Covariance of averages of one stationary OU process.
///
/// If Cov(z(t), z(s)) = stationary_variance * exp(-|t-s|/tau), then
/// this returns Cov(mean(z over first), mean(z over second)).
pub fn ou_average_covariance(
    first: TimeWindow,
    second: TimeWindow,
    stationary_variance: f64,
    correlation_time_seconds: f64,
) -> Result<f64, Error> {
    validate_window(first)?;
    validate_window(second)?;
    if !stationary_variance.is_finite() || stationary_variance < 0.0 {
        return Err(Error::InvalidStationaryVariance);
    }
    if !correlation_time_seconds.is_finite() || correlation_time_seconds <= 0.0 {
        return Err(Error::InvalidCorrelationTime);
    }

    let first_duration = first.end_seconds - first.start_seconds;
    let second_duration = second.end_seconds - second.start_seconds;
    let integral = ou_kernel_integral(first, second, correlation_time_seconds)?;
    let covariance = stationary_variance * integral / (first_duration * second_duration);
    covariance.is_finite().then_some(covariance).ok_or(Error::NumericFailure)
}

/// Build a symmetric covariance matrix for a registry of averaging windows.
pub fn ou_average_covariance_matrix(
    windows: &[TimeWindow],
    stationary_variance: f64,
    correlation_time_seconds: f64,
) -> Result<Vec<Vec<f64>>, Error> {
    if windows.is_empty() {
        return Err(Error::InvalidWindow);
    }
    for window in windows {
        validate_window(*window)?;
    }
    let n = windows.len();
    let mut matrix = vec![vec![0.0; n]; n];
    for row in 0..n {
        for column in row..n {
            let value = ou_average_covariance(
                windows[row],
                windows[column],
                stationary_variance,
                correlation_time_seconds,
            )?;
            matrix[row][column] = value;
            matrix[column][row] = value;
        }
    }
    Ok(matrix)
}

/// Covariance contributed by one shared continuous-time white-noise process
/// when each observation is an interval average with a rectangular kernel.
/// spectral_density has units variance * seconds.
pub fn white_noise_average_covariance(
    first: TimeWindow,
    second: TimeWindow,
    spectral_density: f64,
) -> Result<f64, Error> {
    validate_window(first)?;
    validate_window(second)?;
    if !spectral_density.is_finite() || spectral_density < 0.0 {
        return Err(Error::InvalidWhiteNoiseIntensity);
    }
    let overlap = (first.end_seconds.min(second.end_seconds)
        - first.start_seconds.max(second.start_seconds))
        .max(0.0);
    let value = spectral_density * overlap
        / ((first.end_seconds - first.start_seconds)
            * (second.end_seconds - second.start_seconds));
    value.is_finite().then_some(value).ok_or(Error::NumericFailure)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(start: f64, end: f64) -> TimeWindow {
        TimeWindow { start_seconds: start, end_seconds: end }
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.15}, expected={expected:.15}, tolerance={tolerance}"
        );
    }

    #[test]
    fn unit_window_variance_matches_closed_form() {
        let value = ou_average_covariance(window(0.0, 1.0), window(0.0, 1.0), 1.0, 1.0)
            .expect("valid OU window");
        close(value, 2.0 / std::f64::consts::E, 1.0e-14);
    }

    #[test]
    fn separated_windows_have_exponential_gap_decay() {
        let value = ou_average_covariance(window(0.0, 1.0), window(2.0, 3.0), 1.0, 1.0)
            .expect("valid OU windows");
        let expected = (-1.0_f64).exp() * (1.0 - (-1.0_f64).exp()).powi(2);
        close(value, expected, 1.0e-14);
    }

    #[test]
    fn overlapping_unequal_windows_match_known_answer() {
        let value = ou_average_covariance(window(0.0, 2.0), window(1.0, 2.0), 1.0, 1.0)
            .expect("valid overlapping windows");
        close(value, 0.5676676416183064, 1.0e-13);
    }

    #[test]
    fn duplicate_windows_are_perfectly_correlated() {
        let first = window(0.0, 2.0);
        let second = window(0.0, 2.0);
        let a = ou_average_covariance(first, second, 1.0, 1.0).expect("valid windows");
        let b = ou_average_covariance(first, first, 1.0, 1.0).expect("valid windows");
        close(a, b, 0.0);
    }

    #[test]
    fn covariance_matrix_is_symmetric_and_preserves_window_overlap() {
        let windows = [window(0.0, 2.0), window(1.0, 2.0), window(0.0, 2.0), window(3.0, 4.0)];
        let matrix = ou_average_covariance_matrix(&windows, 1.0, 1.0).expect("valid registry");
        close(matrix[0][1], matrix[1][0], 0.0);
        close(matrix[0][2], matrix[0][0], 0.0);
        close(matrix[0][0], 0.5676676416183064, 1.0e-13);
        close(matrix[1][1], 0.7357588823428847, 1.0e-13);
        close(matrix[0][3], 0.1005363642278503, 1.0e-13);
    }

    #[test]
    fn tiny_window_to_correlation_time_ratio_is_stable() {
        let value = ou_average_covariance(
            window(0.0, 1.0),
            window(0.0, 1.0),
            1.0,
            1.0e9,
        ).expect("valid long correlation time");
        close(value, 1.0, 1.0e-12);
    }

    #[test]
    fn shared_white_noise_covariance_tracks_overlap_and_duration() {
        let first = window(0.0, 2.0);
        let second = window(1.0, 2.0);
        let later = window(3.0, 4.0);
        close(white_noise_average_covariance(first, second, 0.2).unwrap(), 0.1, 1.0e-15);
        close(white_noise_average_covariance(first, later, 0.2).unwrap(), 0.0, 0.0);
    }

    #[test]
    fn invalid_windows_and_parameters_fail_closed() {
        assert_eq!(
            ou_average_covariance(window(1.0, 1.0), window(0.0, 1.0), 1.0, 1.0),
            Err(Error::InvalidWindow)
        );
        assert_eq!(
            ou_average_covariance(window(0.0, 1.0), window(0.0, 1.0), 1.0, 0.0),
            Err(Error::InvalidCorrelationTime)
        );
        assert_eq!(
            white_noise_average_covariance(window(0.0, 1.0), window(0.0, 1.0), -1.0),
            Err(Error::InvalidWhiteNoiseIntensity)
        );
    }
}
