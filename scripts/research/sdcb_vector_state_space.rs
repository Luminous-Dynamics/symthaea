//! Exact bounded linear-SDE discretization using a Van Loan block exponential.
//!
//! Build from the repository root:
//!   rustc --edition=2024 --test scripts/research/sdcb_vector_state_space.rs -o /tmp/sdcb-vector-state
//! Run:
//!   /tmp/sdcb-vector-state
//!
//! The matrix exponential uses the diagonal [13/13] Padé approximant with
//! scaling and squaring. State dimensions are deliberately bounded. This
//! standalone reference is an evaluator, not a production filtering service.

const MAX_STATE_DIMENSION: usize = 8;
const THETA_13: f64 = 5.371_920_351_148_152;
const SYMMETRY_TOL: f64 = 1.0e-12;
const PSD_TOL: f64 = 1.0e-10;
const PIVOT_FLOOR: f64 = 1.0e-300;

pub type Matrix = Vec<Vec<f64>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    EmptyState,
    StateTooLarge,
    InvalidMatrixDimensions,
    NonFiniteMatrix,
    AsymmetricCovariance,
    CovarianceNotPositiveSemidefinite,
    InvalidElapsedTime,
    NumericFailure,
    InvalidMean,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteTransition {
    pub transition: Matrix,
    pub process_covariance: Matrix,
    pub elapsed_seconds: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GaussianState {
    pub mean: Vec<f64>,
    pub covariance: Matrix,
}

fn dimension(matrix: &[Vec<f64>]) -> Result<usize, Error> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        return Err(Error::InvalidMatrixDimensions);
    }
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Error::NonFiniteMatrix);
    }
    Ok(n)
}

fn zeros(rows: usize, columns: usize) -> Matrix {
    vec![vec![0.0; columns]; rows]
}

fn identity(n: usize) -> Matrix {
    let mut result = zeros(n, n);
    for i in 0..n {
        result[i][i] = 1.0;
    }
    result
}

fn transpose(matrix: &[Vec<f64>]) -> Matrix {
    let rows = matrix.len();
    let columns = matrix[0].len();
    let mut result = zeros(columns, rows);
    for i in 0..rows {
        for j in 0..columns {
            result[j][i] = matrix[i][j];
        }
    }
    result
}

fn add(left: &[Vec<f64>], right: &[Vec<f64>]) -> Result<Matrix, Error> {
    if left.len() != right.len()
        || left.iter().zip(right).any(|(a, b)| a.len() != b.len())
    {
        return Err(Error::InvalidMatrixDimensions);
    }
    let result: Matrix = left.iter().zip(right).map(|(a, b)| {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    }).collect();
    all_finite(&result)?;
    Ok(result)
}

fn subtract(left: &[Vec<f64>], right: &[Vec<f64>]) -> Result<Matrix, Error> {
    if left.len() != right.len()
        || left.iter().zip(right).any(|(a, b)| a.len() != b.len())
    {
        return Err(Error::InvalidMatrixDimensions);
    }
    let result: Matrix = left.iter().zip(right).map(|(a, b)| {
        a.iter().zip(b).map(|(x, y)| x - y).collect()
    }).collect();
    all_finite(&result)?;
    Ok(result)
}

fn scale(matrix: &[Vec<f64>], factor: f64) -> Result<Matrix, Error> {
    if !factor.is_finite() {
        return Err(Error::NumericFailure);
    }
    let result: Matrix = matrix.iter().map(|row| {
        row.iter().map(|value| value * factor).collect()
    }).collect();
    all_finite(&result)?;
    Ok(result)
}

fn multiply(left: &[Vec<f64>], right: &[Vec<f64>]) -> Result<Matrix, Error> {
    if left.is_empty() || right.is_empty()
        || left[0].len() != right.len()
        || left.iter().any(|row| row.len() != left[0].len())
        || right.iter().any(|row| row.len() != right[0].len())
    {
        return Err(Error::InvalidMatrixDimensions);
    }
    let rows = left.len();
    let inner = right.len();
    let columns = right[0].len();
    let mut result = zeros(rows, columns);
    for i in 0..rows {
        for k in 0..inner {
            let value = left[i][k];
            for j in 0..columns {
                result[i][j] += value * right[k][j];
            }
        }
    }
    all_finite(&result)?;
    Ok(result)
}

fn all_finite(matrix: &[Vec<f64>]) -> Result<(), Error> {
    if matrix.iter().flatten().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(Error::NumericFailure)
    }
}

fn one_norm(matrix: &[Vec<f64>]) -> f64 {
    let n = matrix.len();
    let mut maximum: f64 = 0.0;
    for column in 0..n {
        let sum: f64 = (0..n).map(|row| matrix[row][column].abs()).sum();
        maximum = maximum.max(sum);
    }
    maximum
}

/// Solve A X = B using Gaussian elimination with partial pivoting.
fn solve_linear(mut a: Matrix, mut b: Matrix) -> Result<Matrix, Error> {
    let n = dimension(&a)?;
    if b.len() != n || b.is_empty()
        || b.iter().any(|row| row.len() != b[0].len())
        || b.iter().flatten().any(|value| !value.is_finite())
    {
        return Err(Error::InvalidMatrixDimensions);
    }
    let rhs_columns = b[0].len();
    for pivot_column in 0..n {
        let mut pivot_row = pivot_column;
        let mut pivot_magnitude = a[pivot_row][pivot_column].abs();
        for row in (pivot_column + 1)..n {
            let magnitude = a[row][pivot_column].abs();
            if magnitude > pivot_magnitude {
                pivot_magnitude = magnitude;
                pivot_row = row;
            }
        }
        if !pivot_magnitude.is_finite() || pivot_magnitude <= PIVOT_FLOOR {
            return Err(Error::NumericFailure);
        }
        if pivot_row != pivot_column {
            a.swap(pivot_row, pivot_column);
            b.swap(pivot_row, pivot_column);
        }

        let pivot = a[pivot_column][pivot_column];
        for column in pivot_column..n {
            a[pivot_column][column] /= pivot;
        }
        for column in 0..rhs_columns {
            b[pivot_column][column] /= pivot;
        }

        for row in 0..n {
            if row == pivot_column {
                continue;
            }
            let factor = a[row][pivot_column];
            if factor == 0.0 {
                continue;
            }
            for column in pivot_column..n {
                a[row][column] -= factor * a[pivot_column][column];
            }
            for column in 0..rhs_columns {
                b[row][column] -= factor * b[pivot_column][column];
            }
        }
    }
    all_finite(&b)?;
    Ok(b)
}

/// Diagonal [13/13] Padé matrix exponential with scaling and squaring.
/// The threshold and coefficients follow the standard Higham/Van Loan approach.
pub fn matrix_exponential(matrix: &[Vec<f64>]) -> Result<Matrix, Error> {
    let n = dimension(matrix)?;
    let norm = one_norm(matrix);
    if !norm.is_finite() {
        return Err(Error::NumericFailure);
    }
    if norm == 0.0 {
        return Ok(identity(n));
    }

    let scaling = if norm > THETA_13 {
        (norm / THETA_13).log2().ceil() as i32
    } else {
        0
    };
    if !(0..=1023).contains(&scaling) {
        return Err(Error::NumericFailure);
    }
    let divisor = 2.0_f64.powi(scaling);
    if !divisor.is_finite() || divisor == 0.0 {
        return Err(Error::NumericFailure);
    }
    let a = scale(matrix, 1.0 / divisor)?;
    let a2 = multiply(&a, &a)?;
    let a4 = multiply(&a2, &a2)?;
    let a6 = multiply(&a4, &a2)?;
    let i = identity(n);

    const B: [f64; 14] = [
        64_764_752_532_480_000.0,
        32_382_376_266_240_000.0,
        7_771_770_303_897_600.0,
        1_187_353_796_428_800.0,
        129_060_195_264_000.0,
        10_559_470_521_600.0,
        670_442_572_800.0,
        33_522_128_640.0,
        1_323_241_920.0,
        40_840_800.0,
        960_960.0,
        16_380.0,
        182.0,
        1.0,
    ];

    let u_inner = add(
        &add(&scale(&a6, B[13])?, &scale(&a4, B[11])?)?,
        &scale(&a2, B[9])?,
    )?;
    let u_polynomial = add(
        &add(
            &add(
                &multiply(&a6, &u_inner)?,
                &scale(&a6, B[7])?,
            )?,
            &scale(&a4, B[5])?,
        )?,
        &add(&scale(&a2, B[3])?, &scale(&i, B[1])?)?,
    )?;
    let u = multiply(&a, &u_polynomial)?;

    let v_inner = add(
        &add(&scale(&a6, B[12])?, &scale(&a4, B[10])?)?,
        &scale(&a2, B[8])?,
    )?;
    let v_polynomial = add(
        &add(
            &add(
                &multiply(&a6, &v_inner)?,
                &scale(&a6, B[6])?,
            )?,
            &scale(&a4, B[4])?,
        )?,
        &add(&scale(&a2, B[2])?, &scale(&i, B[0])?)?,
    )?;

    let numerator = add(&v_polynomial, &u)?;
    let denominator = subtract(&v_polynomial, &u)?;
    let mut result = solve_linear(denominator, numerator)?;

    for _ in 0..scaling {
        result = multiply(&result, &result)?;
    }
    all_finite(&result)?;
    Ok(result)
}

/// Jacobi eigenvalue iteration used only to verify symmetric PSD matrices.
fn symmetric_eigenvalues(matrix: &[Vec<f64>]) -> Result<Vec<f64>, Error> {
    let n = dimension(matrix)?;
    let mut a = matrix.to_vec();
    if n == 1 {
        return Ok(vec![a[0][0]]);
    }
    for _ in 0..(100 * n * n) {
        let mut p = 0;
        let mut q = 1;
        let mut largest: f64 = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i][j].abs() > largest {
                    largest = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        let scale = one_norm(&a).max(f64::MIN_POSITIVE);
        if largest <= 1.0e-14 * scale {
            break;
        }
        let apq = a[p][q];
        let tau = (a[q][q] - a[p][p]) / (2.0 * apq);
        let sign = if tau >= 0.0 { 1.0 } else { -1.0 };
        let t = sign / (tau.abs() + (1.0 + tau * tau).sqrt());
        let cosine = 1.0 / (1.0 + t * t).sqrt();
        let sine = t * cosine;

        let app = a[p][p];
        let aqq = a[q][q];
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for k in 0..n {
            if k == p || k == q {
                continue;
            }
            let akp = a[k][p];
            let akq = a[k][q];
            let next_kp = cosine * akp - sine * akq;
            let next_kq = sine * akp + cosine * akq;
            a[k][p] = next_kp;
            a[p][k] = next_kp;
            a[k][q] = next_kq;
            a[q][k] = next_kq;
        }
    }
    Ok((0..n).map(|i| a[i][i]).collect())
}

fn validate_symmetric_psd(matrix: &[Vec<f64>]) -> Result<(), Error> {
    let n = dimension(matrix)?;
    let scale = one_norm(matrix).max(f64::MIN_POSITIVE);
    for i in 0..n {
        for j in 0..i {
            if (matrix[i][j] - matrix[j][i]).abs() > SYMMETRY_TOL * scale {
                return Err(Error::AsymmetricCovariance);
            }
        }
    }
    let eigenvalues = symmetric_eigenvalues(matrix)?;
    let tolerance = PSD_TOL * scale;
    if eigenvalues.iter().any(|value| *value < -tolerance) {
        return Err(Error::CovarianceNotPositiveSemidefinite);
    }
    Ok(())
}

/// Discretize dx = A x dt + G dW, where Qc = G G^T.
///
/// Uses:
/// exp([[A, Qc], [0, -A^T]] * dt) = [[F, U], [0, F^-T]]
/// and Q(dt) = U F^T.
pub fn discretize_linear_sde(
    generator: &[Vec<f64>],
    diffusion_covariance: &[Vec<f64>],
    elapsed_seconds: f64,
) -> Result<DiscreteTransition, Error> {
    let n = dimension(generator).map_err(|_| Error::EmptyState)?;
    if n > MAX_STATE_DIMENSION {
        return Err(Error::StateTooLarge);
    }
    if dimension(diffusion_covariance)? != n {
        return Err(Error::InvalidMatrixDimensions);
    }
    if !elapsed_seconds.is_finite() || elapsed_seconds < 0.0 {
        return Err(Error::InvalidElapsedTime);
    }
    validate_symmetric_psd(diffusion_covariance)?;

    if elapsed_seconds == 0.0 {
        return Ok(DiscreteTransition {
            transition: identity(n),
            process_covariance: zeros(n, n),
            elapsed_seconds,
        });
    }

    let mut van_loan = zeros(2 * n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            van_loan[i][j] = generator[i][j] * elapsed_seconds;
            van_loan[i][j + n] = diffusion_covariance[i][j] * elapsed_seconds;
            van_loan[i + n][j + n] = -generator[j][i] * elapsed_seconds;
        }
    }
    all_finite(&van_loan)?;
    let exponential = matrix_exponential(&van_loan)?;

    let mut transition = zeros(n, n);
    let mut upper_right = zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            transition[i][j] = exponential[i][j];
            upper_right[i][j] = exponential[i][j + n];
        }
    }
    let mut process_covariance = multiply(&upper_right, &transpose(&transition))?;

    let covariance_scale = one_norm(&process_covariance).max(f64::MIN_POSITIVE);
    for i in 0..n {
        for j in 0..i {
            if (process_covariance[i][j] - process_covariance[j][i]).abs()
                > 1.0e-8 * covariance_scale
            {
                return Err(Error::NumericFailure);
            }
            let symmetric = 0.5 * (process_covariance[i][j] + process_covariance[j][i]);
            process_covariance[i][j] = symmetric;
            process_covariance[j][i] = symmetric;
        }
    }
    validate_symmetric_psd(&process_covariance)?;

    Ok(DiscreteTransition {
        transition,
        process_covariance,
        elapsed_seconds,
    })
}

/// Advance a Gaussian state by one elapsed-time transition.
pub fn propagate_gaussian(
    state: &GaussianState,
    transition: &DiscreteTransition,
) -> Result<GaussianState, Error> {
    let n = dimension(&state.covariance)?;
    if state.mean.len() != n {
        return Err(Error::InvalidMean);
    }
    if transition.transition.len() != n
        || transition.transition.iter().any(|row| row.len() != n)
        || transition.process_covariance.len() != n
        || transition.process_covariance.iter().any(|row| row.len() != n)
        || state.mean.iter().any(|value| !value.is_finite())
    {
        return Err(Error::InvalidMatrixDimensions);
    }
    validate_symmetric_psd(&state.covariance)?;
    validate_symmetric_psd(&transition.process_covariance)?;

    let mean_matrix: Matrix = state.mean.iter().map(|value| vec![*value]).collect();
    let next_mean_matrix = multiply(&transition.transition, &mean_matrix)?;
    let next_mean = next_mean_matrix.iter().map(|row| row[0]).collect();

    let propagated = multiply(
        &multiply(&transition.transition, &state.covariance)?,
        &transpose(&transition.transition),
    )?;
    let next_covariance = add(&propagated, &transition.process_covariance)?;
    validate_symmetric_psd(&next_covariance)?;

    Ok(GaussianState {
        mean: next_mean,
        covariance: next_covariance,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.16}, expected={expected:.16}, tolerance={tolerance:e}"
        );
    }

    fn matrix_close(actual: &[Vec<f64>], expected: &[Vec<f64>], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (actual_row, expected_row) in actual.iter().zip(expected) {
            assert_eq!(actual_row.len(), expected_row.len());
            for (a, e) in actual_row.iter().zip(expected_row) {
                close(*a, *e, tolerance);
            }
        }
    }

    #[test]
    fn scalar_ou_matches_closed_form_transition_and_process_covariance() {
        let lambda = 0.7;
        let stationary_variance = 1.5;
        let qc = 2.0 * lambda * stationary_variance;
        let dt = 2.3;
        let result = discretize_linear_sde(&vec![vec![-lambda]], &vec![vec![qc]], dt).unwrap();

        close(result.transition[0][0], (-lambda * dt).exp(), 1.0e-13);
        close(
            result.process_covariance[0][0],
            stationary_variance * (1.0 - (-2.0 * lambda * dt).exp()),
            1.0e-12,
        );
    }

    #[test]
    fn coupled_two_factor_fixture_preserves_cross_covariance() {
        let a = vec![vec![-1.0, 0.25], vec![0.0, -0.5]];
        let qc = vec![vec![0.4, 0.12], vec![0.12, 0.3]];
        let result = discretize_linear_sde(&a, &qc, 1.25).unwrap();

        matrix_close(
            &result.transition,
            &[
                vec![0.28650479686019, 0.1243783158294001],
                vec![0.0, 0.5352614285189903],
            ],
            2.0e-13,
        );
        matrix_close(
            &result.process_covariance,
            &[
                vec![0.1995091523166423, 0.0900913798078700],
                vec![0.0900913798078700, 0.2140485609419429],
            ],
            2.0e-12,
        );
        assert!(result.process_covariance[0][1].abs() > 0.05);
    }

    #[test]
    fn non_normal_generator_matches_independent_quadrature_fixture() {
        let a = vec![vec![-1.0, 10.0], vec![0.0, -2.0]];
        let qc = vec![vec![0.2, 0.05], vec![0.05, 0.3]];
        let result = discretize_linear_sde(&a, &qc, 0.1).unwrap();

        matrix_close(
            &result.transition,
            &[
                vec![0.9048374180359595, 0.8610666495797772],
                vec![0.0, 0.8187307530779818],
            ],
            2.0e-13,
        );
        matrix_close(
            &result.process_covariance,
            &[
                vec![0.0303703939111182, 0.0162415101669830],
                vec![0.0162415101669830, 0.0247259965473271],
            ],
            2.0e-13,
        );
    }

    #[test]
    fn zero_elapsed_time_has_identity_transition_and_zero_noise() {
        let a = vec![vec![-1.0, 0.2], vec![0.0, -0.5]];
        let qc = vec![vec![1.0, 0.0], vec![0.0, 0.3]];
        let result = discretize_linear_sde(&a, &qc, 0.0).unwrap();
        assert_eq!(result.transition, identity(2));
        assert_eq!(result.process_covariance, zeros(2, 2));
    }

    #[test]
    fn stable_long_gap_converges_without_negative_variance() {
        let result = discretize_linear_sde(&vec![vec![-0.8]], &vec![vec![1.6]], 120.0).unwrap();
        close(result.transition[0][0], (-96.0_f64).exp(), 1.0e-55);
        close(result.process_covariance[0][0], 1.0, 1.0e-12);
        assert!(result.process_covariance[0][0] >= 0.0);
    }

    #[test]
    fn zero_drift_coordinate_remains_static_but_can_be_correlated() {
        let a = vec![vec![0.0, 0.0], [0.0, -1.0].to_vec()];
        let qc = vec![vec![0.0, 0.0], vec![0.0, 0.2]];
        let result = discretize_linear_sde(&a, &qc, 3.0).unwrap();
        close(result.transition[0][0], 1.0, 1.0e-14);
        close(result.transition[0][1], 0.0, 1.0e-14);
        close(result.process_covariance[0][0], 0.0, 1.0e-14);
        close(result.process_covariance[0][1], 0.0, 1.0e-14);
    }

    #[test]
    fn gaussian_propagation_preserves_cross_covariance() {
        let a = vec![vec![-1.0, 0.25], vec![0.0, -0.5]];
        let qc = vec![vec![0.4, 0.12], vec![0.12, 0.3]];
        let transition = discretize_linear_sde(&a, &qc, 1.25).unwrap();
        let state = GaussianState {
            mean: vec![1.0, -2.0],
            covariance: vec![vec![0.5, 0.1], vec![0.1, 0.8]],
        };
        let propagated = propagate_gaussian(&state, &transition).unwrap();
        close(propagated.mean[0], 0.0377478332013898, 2.0e-13);
        close(propagated.mean[1], -1.0705228570379806, 2.0e-13);
        close(propagated.covariance[0][1], 0.145812, 1.0e-5);
        close(propagated.covariance[0][1], propagated.covariance[1][0], 1.0e-14);
    }

    #[test]
    fn semigroup_composition_matches_single_long_transition() {
        let a = vec![vec![-1.0, 0.4], vec![-0.2, -0.7]];
        let qc = vec![vec![0.3, 0.08], vec![0.08, 0.2]];
        let first = discretize_linear_sde(&a, &qc, 0.4).unwrap();
        let second = discretize_linear_sde(&a, &qc, 0.9).unwrap();
        let combined = discretize_linear_sde(&a, &qc, 1.3).unwrap();

        let composed_f = multiply(&second.transition, &first.transition).unwrap();
        let composed_q = add(
            &multiply(
                &multiply(&second.transition, &first.process_covariance).unwrap(),
                &transpose(&second.transition),
            ).unwrap(),
            &second.process_covariance,
        ).unwrap();

        matrix_close(&composed_f, &combined.transition, 2.0e-12);
        matrix_close(&composed_q, &combined.process_covariance, 2.0e-12);
    }

    #[test]
    fn invalid_covariance_and_elapsed_time_fail_closed() {
        let a = vec![vec![-1.0, 0.0], vec![0.0, -1.0]];
        assert_eq!(
            discretize_linear_sde(&a, &vec![vec![1.0, 0.2], vec![0.0, 1.0]], 1.0),
            Err(Error::AsymmetricCovariance)
        );
        assert_eq!(
            discretize_linear_sde(&a, &vec![vec![1.0, 2.0], vec![2.0, 1.0]], 1.0),
            Err(Error::CovarianceNotPositiveSemidefinite)
        );
        assert_eq!(
            discretize_linear_sde(&a, &vec![vec![1.0, 0.0], vec![0.0, 1.0]], -1.0),
            Err(Error::InvalidElapsedTime)
        );
    }

    #[test]
    fn matrix_exponential_handles_zero_and_non_normal_matrices() {
        assert_eq!(matrix_exponential(&zeros(2, 2)).unwrap(), identity(2));
        let a = vec![vec![-1.0, 10.0], vec![0.0, -2.0]];
        let result = matrix_exponential(&a).unwrap();
        close(result[0][0], (-1.0_f64).exp(), 1.0e-13);
        close(result[1][1], (-2.0_f64).exp(), 1.0e-13);
        assert!(result[0][1] > 0.8);
    }
}
