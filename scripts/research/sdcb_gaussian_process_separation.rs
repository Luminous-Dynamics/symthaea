//! Finite-sample Gaussian-process discrimination for a first-order sensor.
//!
//! Standalone research reference. It preserves the full covariance matrix of
//! correlated time-series observations; autocovariance lags are never treated
//! as independent replicates.
//!
//! Build:
//!   rustc --edition=2024 --test scripts/research/sdcb_gaussian_process_separation.rs -o /tmp/sdcb-gp-separation
//!   /tmp/sdcb-gp-separation
//!
//! M0 uses positive-definite finite Gaussian covariance matrices. It does not
//! add jitter, project covariance matrices, or implement singular Gaussian
//! measures. Singular/unsupported cases fail closed.

pub const MAX_SAMPLES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeparationError {
    InvalidDimension,
    InvalidParameter,
    NonFiniteCovariance,
    DimensionMismatch,
    NotPositiveDefinite,
    NumericalFailure,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DenseCovariance {
    dimension: usize,
    values: [f64; MAX_SAMPLES * MAX_SAMPLES],
}

impl DenseCovariance {
    fn zero(dimension: usize) -> Result<Self, SeparationError> {
        if dimension == 0 || dimension > MAX_SAMPLES {
            return Err(SeparationError::InvalidDimension);
        }
        Ok(Self {
            dimension,
            values: [0.0; MAX_SAMPLES * MAX_SAMPLES],
        })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn get(&self, row: usize, column: usize) -> f64 {
        self.values[row * MAX_SAMPLES + column]
    }

    fn set(&mut self, row: usize, column: usize, value: f64) {
        self.values[row * MAX_SAMPLES + column] = value;
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionalDivergences {
    /// KL(N(0, A) || N(0, B)).
    pub a_to_b: f64,
    /// KL(N(0, B) || N(0, A)).
    pub b_to_a: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistinguishabilityBounds {
    pub directional_kl: DirectionalDivergences,
    /// Pinsker upper bound using the smaller valid directional KL bound.
    /// This remains an upper bound, not the exact total variation distance.
    pub total_variation_upper: f64,
    /// Equal-prior optimal classification accuracy upper bound (1 + TV)/2.
    pub equal_prior_accuracy_upper: f64,
}

fn response_autocovariance(
    lambda: f64,
    mu: f64,
    process_variance: f64,
    elapsed: f64,
) -> Result<f64, SeparationError> {
    if !lambda.is_finite()
        || !mu.is_finite()
        || !process_variance.is_finite()
        || !elapsed.is_finite()
        || lambda <= 0.0
        || mu <= 0.0
        || process_variance < 0.0
        || elapsed < 0.0
    {
        return Err(SeparationError::InvalidParameter);
    }

    let rate_sum = lambda + mu;
    if !rate_sum.is_finite() || rate_sum <= 0.0 {
        return Err(SeparationError::NumericalFailure);
    }

    let q = (mu * process_variance) / rate_sum;
    let slow_rate = lambda.min(mu);
    let rate_gap = (lambda - mu).abs();
    let x = rate_gap * elapsed;
    let slow_elapsed = slow_rate * elapsed;

    if !q.is_finite() || !x.is_finite() || !slow_elapsed.is_finite() {
        return Err(SeparationError::NumericalFailure);
    }

    // psi(x) = (1 - exp(-x))/x, with psi(0)=1. exp_m1 avoids
    // cancellation when the process and sensor rates are nearly equal.
    let psi = if x == 0.0 {
        1.0
    } else {
        -(-x).exp_m1() / x
    };

    let result = q * (-slow_elapsed).exp() * (1.0 + slow_elapsed * psi);
    if !result.is_finite() {
        return Err(SeparationError::NonFiniteCovariance);
    }
    Ok(result)
}

/// Build the sampled covariance K + readout_variance * I.
///
/// The samples are at t_i = i * sample_interval. The off-diagonal entries
/// retain the full stationary OU + first-order-sensor temporal dependence.
pub fn response_observation_covariance(
    lambda: f64,
    mu: f64,
    process_variance: f64,
    sample_interval: f64,
    readout_noise_variance: f64,
    samples: usize,
) -> Result<DenseCovariance, SeparationError> {
    if samples == 0 || samples > MAX_SAMPLES {
        return Err(SeparationError::InvalidDimension);
    }
    if !sample_interval.is_finite()
        || sample_interval <= 0.0
        || !readout_noise_variance.is_finite()
        || readout_noise_variance < 0.0
    {
        return Err(SeparationError::InvalidParameter);
    }

    let mut covariance = DenseCovariance::zero(samples)?;
    for row in 0..samples {
        for column in 0..=row {
            let lag = ((row - column) as f64) * sample_interval;
            let mut value =
                response_autocovariance(lambda, mu, process_variance, lag)?;
            if row == column {
                value += readout_noise_variance;
            }
            if !value.is_finite() {
                return Err(SeparationError::NonFiniteCovariance);
            }
            covariance.set(row, column, value);
            covariance.set(column, row, value);
        }
    }
    Ok(covariance)
}

fn cholesky(matrix: &DenseCovariance) -> Result<DenseCovariance, SeparationError> {
    let n = matrix.dimension;
    let mut lower = DenseCovariance::zero(n)?;

    for row in 0..n {
        for column in 0..=row {
            let mut value = matrix.get(row, column);
            for k in 0..column {
                value -= lower.get(row, k) * lower.get(column, k);
            }

            if row == column {
                // No jitter or repair: a failed factorization is returned.
                if !value.is_finite() || value <= 0.0 {
                    return Err(SeparationError::NotPositiveDefinite);
                }
                lower.set(row, column, value.sqrt());
            } else {
                let pivot = lower.get(column, column);
                if !pivot.is_finite() || pivot <= 0.0 {
                    return Err(SeparationError::NotPositiveDefinite);
                }
                let entry = value / pivot;
                if !entry.is_finite() {
                    return Err(SeparationError::NumericalFailure);
                }
                lower.set(row, column, entry);
            }
        }
    }
    Ok(lower)
}

fn log_determinant_from_cholesky(
    lower: &DenseCovariance,
) -> Result<f64, SeparationError> {
    let mut log_determinant = 0.0;
    for index in 0..lower.dimension {
        let diagonal = lower.get(index, index);
        if !diagonal.is_finite() || diagonal <= 0.0 {
            return Err(SeparationError::NotPositiveDefinite);
        }
        log_determinant += 2.0 * diagonal.ln();
    }
    if !log_determinant.is_finite() {
        return Err(SeparationError::NumericalFailure);
    }
    Ok(log_determinant)
}

/// Compute KL(N(0, A) || N(0, B)) by Cholesky and triangular solves.
///
/// Uses tr(B^-1 A) = || L_B^-1 L_A ||_F^2, where A=L_A L_A^T and
/// B=L_B L_B^T. No explicit matrix inverse is formed.
pub fn gaussian_kl(
    from: &DenseCovariance,
    to: &DenseCovariance,
) -> Result<f64, SeparationError> {
    if from.dimension != to.dimension {
        return Err(SeparationError::DimensionMismatch);
    }

    let n = from.dimension;
    let lower_from = cholesky(from)?;
    let lower_to = cholesky(to)?;
    let logdet_from = log_determinant_from_cholesky(&lower_from)?;
    let logdet_to = log_determinant_from_cholesky(&lower_to)?;

    // transformed = L_to^-1 * L_from, computed one column at a time.
    let mut transformed = DenseCovariance::zero(n)?;
    for column in 0..n {
        for row in 0..n {
            let mut value = lower_from.get(row, column);
            for k in 0..row {
                value -= lower_to.get(row, k) * transformed.get(k, column);
            }
            let pivot = lower_to.get(row, row);
            if !pivot.is_finite() || pivot <= 0.0 {
                return Err(SeparationError::NotPositiveDefinite);
            }
            let entry = value / pivot;
            if !entry.is_finite() {
                return Err(SeparationError::NumericalFailure);
            }
            transformed.set(row, column, entry);
        }
    }

    let mut trace_term = 0.0;
    for row in 0..n {
        for column in 0..n {
            let value = transformed.get(row, column);
            trace_term += value * value;
        }
    }

    let result = 0.5 * (trace_term - (n as f64) + logdet_to - logdet_from);
    if !result.is_finite() || result < 0.0 {
        return Err(SeparationError::NumericalFailure);
    }
    Ok(result)
}

pub fn compare_gaussian_processes(
    a: &DenseCovariance,
    b: &DenseCovariance,
) -> Result<DistinguishabilityBounds, SeparationError> {
    let a_to_b = gaussian_kl(a, b)?;
    let b_to_a = gaussian_kl(b, a)?;
    let tighter_valid_kl = a_to_b.min(b_to_a);
    let total_variation_upper = (tighter_valid_kl / 2.0).sqrt();
    let equal_prior_accuracy_upper = (1.0 + total_variation_upper) / 2.0;

    if !total_variation_upper.is_finite() || !equal_prior_accuracy_upper.is_finite() {
        return Err(SeparationError::NumericalFailure);
    }

    Ok(DistinguishabilityBounds {
        directional_kl: DirectionalDivergences { a_to_b, b_to_a },
        total_variation_upper,
        equal_prior_accuracy_upper,
    })
}

/// Unit-DC-gain first-order sensor response to a traceable unit step at its inlet.
pub fn normalized_sensor_inlet_step(mu: f64, elapsed: f64) -> Result<f64, SeparationError> {
    if !mu.is_finite() || mu <= 0.0 || !elapsed.is_finite() || elapsed < 0.0 {
        return Err(SeparationError::InvalidParameter);
    }
    let result = -(-mu * elapsed).exp_m1();
    if !result.is_finite() {
        return Err(SeparationError::NumericalFailure);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LAMBDA_A: f64 = 0.25;
    const MU_A: f64 = 4.0;
    const VAR_A: f64 = 9.0;
    const LAMBDA_B_NEAR: f64 = 4.04;
    const MU_B_NEAR: f64 = 0.25;
    const VAR_B_NEAR: f64 = 145.35529411764706;
    const DT: f64 = 0.5;
    const READOUT_VARIANCE: f64 = 0.1;

    fn near(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
        );
    }

    fn near_worlds(samples: usize, readout_variance: f64) -> (DenseCovariance, DenseCovariance) {
        let a = response_observation_covariance(
            LAMBDA_A, MU_A, VAR_A, DT, readout_variance, samples
        ).unwrap();
        let b = response_observation_covariance(
            LAMBDA_B_NEAR, MU_B_NEAR, VAR_B_NEAR, DT, readout_variance, samples
        ).unwrap();
        (a, b)
    }

    #[test]
    fn exact_pole_swap_preserves_every_sampled_joint_covariance() {
        let a = response_observation_covariance(0.25, 4.0, 9.0, DT, READOUT_VARIANCE, 64)
            .unwrap();
        let b = response_observation_covariance(4.0, 0.25, 144.0, DT, READOUT_VARIANCE, 64)
            .unwrap();
        for row in 0..64 {
            for column in 0..64 {
                assert_eq!(a.get(row, column), b.get(row, column));
            }
        }
        near(gaussian_kl(&a, &b).unwrap(), 0.0, 0.0);
        near(gaussian_kl(&b, &a).unwrap(), 0.0, 0.0);
    }

    #[test]
    fn near_swap_covariance_kernels_are_distinct() {
        let a = response_autocovariance(LAMBDA_A, MU_A, VAR_A, 0.5).unwrap();
        let b = response_autocovariance(
            LAMBDA_B_NEAR, MU_B_NEAR, VAR_B_NEAR, 0.5
        ).unwrap();
        near(a, 7.8971944422307278, 2.0e-14);
        near(b, 7.8942387289264293, 2.0e-14);
        assert!((a - b).abs() > 1.0e-3);
    }

    #[test]
    fn directional_kl_matches_high_precision_32_sample_reference() {
        let (a, b) = near_worlds(32, READOUT_VARIANCE);
        let result = compare_gaussian_processes(&a, &b).unwrap();
        near(
            result.directional_kl.a_to_b,
            0.00025683365293439291,
            2.0e-10,
        );
        near(
            result.directional_kl.b_to_a,
            0.00025805588272451917,
            2.0e-10,
        );
        near(
            result.total_variation_upper,
            0.011332114827656683,
            2.0e-9,
        );
        near(
            result.equal_prior_accuracy_upper,
            0.50566605741382834,
            2.0e-9,
        );
    }

    #[test]
    fn frozen_nested_sample_prefixes_match_high_precision_references() {
        let cases = [
            (8, 0.00005516551920161024, 0.00005542108725631821),
            (16, 0.00012238823044587051, 0.00012296601907905116),
            (32, 0.00025683365293439291, 0.00025805588272451917),
            (64, 0.0005257244979114377, 0.0005282356100154552),
        ];
        let mut previous = 0.0;
        for (samples, expected_ab, expected_ba) in cases {
            let (a, b) = near_worlds(samples, READOUT_VARIANCE);
            let result = compare_gaussian_processes(&a, &b).unwrap();
            near(result.directional_kl.a_to_b, expected_ab, 2.0e-10);
            near(result.directional_kl.b_to_a, expected_ba, 2.0e-10);
            assert!(result.directional_kl.a_to_b >= previous);
            previous = result.directional_kl.a_to_b;
        }
    }

    #[test]
    fn increasing_readout_noise_does_not_increase_kl() {
        let (a_low, b_low) = near_worlds(32, 0.1);
        let (a_mid, b_mid) = near_worlds(32, 0.25);
        let (a_high, b_high) = near_worlds(32, 0.5);
        let low = gaussian_kl(&a_low, &b_low).unwrap();
        let mid = gaussian_kl(&a_mid, &b_mid).unwrap();
        let high = gaussian_kl(&a_high, &b_high).unwrap();
        assert!(low > mid);
        assert!(mid > high);
    }

    #[test]
    fn same_permutation_of_both_covariances_preserves_kl() {
        let (a, b) = near_worlds(16, READOUT_VARIANCE);
        let original = gaussian_kl(&a, &b).unwrap();
        let order: Vec<usize> = (0..16).rev().collect();
        let mut permuted_a = DenseCovariance::zero(16).unwrap();
        let mut permuted_b = DenseCovariance::zero(16).unwrap();
        for row in 0..16 {
            for column in 0..16 {
                permuted_a.set(row, column, a.get(order[row], order[column]));
                permuted_b.set(row, column, b.get(order[row], order[column]));
            }
        }
        near(gaussian_kl(&permuted_a, &permuted_b).unwrap(), original, 2.0e-12);
    }

    #[test]
    fn inlet_step_is_a_strong_positive_control() {
        let a = normalized_sensor_inlet_step(4.0, 0.5).unwrap();
        let b = normalized_sensor_inlet_step(0.25, 0.5).unwrap();
        near(a, 0.8646647167633873, 2.0e-15);
        near(b, 0.1175030974154046, 2.0e-15);
        assert!((a - b).abs() > 0.7);
    }

    #[test]
    fn malformed_parameters_and_time_grids_reject() {
        assert_eq!(
            response_observation_covariance(0.0, 1.0, 1.0, DT, 0.1, 8).unwrap_err(),
            SeparationError::InvalidParameter
        );
        assert_eq!(
            response_observation_covariance(1.0, 1.0, 1.0, 0.0, 0.1, 8).unwrap_err(),
            SeparationError::InvalidParameter
        );
        assert_eq!(
            response_observation_covariance(1.0, 1.0, 1.0, DT, -0.1, 8).unwrap_err(),
            SeparationError::InvalidParameter
        );
        assert_eq!(
            response_observation_covariance(1.0, 1.0, 1.0, DT, 0.1, MAX_SAMPLES + 1)
                .unwrap_err(),
            SeparationError::InvalidDimension
        );
    }

    #[test]
    fn singular_covariance_rejects_without_jitter() {
        let mut singular = DenseCovariance::zero(2).unwrap();
        singular.set(0, 0, 1.0);
        singular.set(0, 1, 1.0);
        singular.set(1, 0, 1.0);
        singular.set(1, 1, 1.0);
        assert_eq!(
            gaussian_kl(&singular, &singular).unwrap_err(),
            SeparationError::NotPositiveDefinite
        );
    }

    #[test]
    fn dimension_mismatch_rejects() {
        let a = response_observation_covariance(1.0, 2.0, 1.0, DT, 0.1, 8).unwrap();
        let b = response_observation_covariance(1.0, 2.0, 1.0, DT, 0.1, 16).unwrap();
        assert_eq!(
            gaussian_kl(&a, &b).unwrap_err(),
            SeparationError::DimensionMismatch
        );
    }
}
