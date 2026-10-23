// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Checked multivariate linear least squares for shared scientific consumers.
//!
//! V1 deliberately exposes a small numerical proposition: finite tall design
//! matrices are factorized with deterministic column-pivoted, reorthogonalized
//! modified Gram-Schmidt QR. Numerical rank and a diagonal-R condition proxy are
//! surfaced explicitly; coefficients are emitted only for full-column-rank
//! systems and are returned in the caller's original column order.
//!
//! Numerical full rank is **not** physical, causal, or experimental
//! identifiability. The implementation also performs no automatic centering,
//! scaling, standardization, or intercept insertion, so caller data scaling is
//! part of the numerical proposition.

/// Stable method identifier for the V1 factorization/solve profile.
pub const LINEAR_LEAST_SQUARES_METHOD_V1: &str =
    "column-pivoted-modified-gram-schmidt-reorthogonalized-v1";

/// Numerical policy for [`try_linear_least_squares`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearLeastSquaresProfile {
    /// A QR diagonal at or below this fraction of the largest original column
    /// norm is treated as numerically rank deficient.
    pub relative_rank_tolerance: f64,
    /// Full-rank fits with a diagonal-R condition proxy above this value are
    /// retained but classified as [`LinearLeastSquaresDisposition::IllConditioned`].
    pub max_condition_proxy: f64,
}

impl Default for LinearLeastSquaresProfile {
    fn default() -> Self {
        Self {
            relative_rank_tolerance: 1.0e-12,
            max_condition_proxy: 1.0e10,
        }
    }
}

impl LinearLeastSquaresProfile {
    fn validate(self) -> Result<Self, LinearModelError> {
        if !self.relative_rank_tolerance.is_finite()
            || self.relative_rank_tolerance <= 0.0
            || self.relative_rank_tolerance >= 1.0
        {
            return Err(LinearModelError::InvalidProfile(
                "relative_rank_tolerance must be finite and in (0, 1)",
            ));
        }
        if !self.max_condition_proxy.is_finite() || self.max_condition_proxy < 1.0 {
            return Err(LinearModelError::InvalidProfile(
                "max_condition_proxy must be finite and >= 1",
            ));
        }
        Ok(self)
    }
}

/// Numerical disposition of a linear least-squares attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearLeastSquaresDisposition {
    /// Every declared parameter column was numerically independent under the
    /// exact profile and the diagonal condition proxy stayed within its limit.
    FullColumnRank,
    /// Every declared parameter column was numerically independent, but the
    /// diagonal condition proxy exceeded the caller-declared warning limit.
    IllConditioned,
    /// At least one declared parameter column was numerically dependent under
    /// the exact rank tolerance. No unique coefficient vector is emitted.
    RankDeficient,
}

/// Result of one checked multivariate linear least-squares calculation.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearLeastSquaresResult {
    /// Exact algorithm family used by this implementation.
    pub method: &'static str,
    /// Exact caller-declared numerical profile used for the calculation.
    pub profile: LinearLeastSquaresProfile,
    /// Number of observations (rows).
    pub observations: usize,
    /// Number of declared parameters/features (columns).
    pub parameters: usize,
    /// Numerical column rank under the declared profile.
    pub rank: usize,
    /// Absolute rank threshold derived from the profile and largest original
    /// column norm.
    pub rank_threshold: f64,
    /// Original column indices in the deterministic pivot/factorization order.
    pub factorization_column_order: Vec<usize>,
    /// Ratio `max(abs(diag(R))) / min(abs(diag(R)))` over diagonals above the
    /// rank threshold. This is a deterministic diagnostic proxy, **not** the
    /// matrix 2-norm condition number.
    pub condition_proxy: Option<f64>,
    /// Fit disposition.
    pub disposition: LinearLeastSquaresDisposition,
    /// Unique coefficient vector for full-rank systems, restored to the exact
    /// caller input-column order. `None` for rank-deficient systems.
    pub coefficients: Option<Vec<f64>>,
    /// `y - Xβ` for solved full-rank systems. Empty when no unique coefficient
    /// vector is available.
    pub residuals: Vec<f64>,
    /// Residual sum of squares for solved full-rank systems.
    pub residual_sum_squares: Option<f64>,
    /// Root-mean-square residual over observations for solved full-rank systems.
    pub rmse: Option<f64>,
    /// `observations - parameters` for solved full-column-rank systems.
    pub residual_degrees_of_freedom: Option<usize>,
}

impl LinearLeastSquaresResult {
    /// Whether this result contains a unique coefficient vector under the exact
    /// numerical profile.
    pub fn is_solved(&self) -> bool {
        self.coefficients.is_some()
    }
}

/// Structural or numerical failure before a valid least-squares result exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinearModelError {
    EmptyDesign,
    EmptyParameters,
    OutcomeLength {
        observations: usize,
        outcomes: usize,
    },
    RaggedRow {
        row: usize,
        expected: usize,
        actual: usize,
    },
    NonFiniteDesign {
        row: usize,
        column: usize,
    },
    NonFiniteOutcome {
        index: usize,
    },
    Underdetermined {
        observations: usize,
        parameters: usize,
    },
    InvalidProfile(&'static str),
    NumericalFailure(&'static str),
}

impl std::fmt::Display for LinearModelError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyDesign => formatter.write_str("design matrix has no observations"),
            Self::EmptyParameters => formatter.write_str("design matrix has no parameters"),
            Self::OutcomeLength {
                observations,
                outcomes,
            } => write!(
                formatter,
                "outcome length {outcomes} does not match {observations} observations"
            ),
            Self::RaggedRow {
                row,
                expected,
                actual,
            } => write!(
                formatter,
                "design row {row} has {actual} columns; expected {expected}"
            ),
            Self::NonFiniteDesign { row, column } => {
                write!(formatter, "design value at ({row}, {column}) is non-finite")
            }
            Self::NonFiniteOutcome { index } => {
                write!(formatter, "outcome value at index {index} is non-finite")
            }
            Self::Underdetermined {
                observations,
                parameters,
            } => write!(
                formatter,
                "underdetermined design: {observations} observations for {parameters} parameters"
            ),
            Self::InvalidProfile(message) => write!(formatter, "invalid profile: {message}"),
            Self::NumericalFailure(message) => write!(formatter, "numerical failure: {message}"),
        }
    }
}

impl std::error::Error for LinearModelError {}

/// Fit `y = Xβ + ε` using deterministic column-pivoted, reorthogonalized
/// modified Gram-Schmidt QR.
///
/// # Semantics
///
/// - The design matrix is row-major: `design[observation][parameter]`.
/// - No intercept column is inserted automatically.
/// - No predictor centering/scaling/standardization is performed automatically.
/// - `observations < parameters` is rejected as underdetermined.
/// - Rank-deficient tall systems return a structured result with no coefficient
///   vector rather than selecting one arbitrary solution.
/// - A full-rank but poorly conditioned system may still return coefficients;
///   the result is explicitly classified as `IllConditioned`.
/// - Column pivoting is deterministic: the largest residual column norm wins;
///   equal norms preserve the earlier remaining input-column order.
///
/// Numerical full rank does not establish physical/causal identifiability.
pub fn try_linear_least_squares(
    design: &[Vec<f64>],
    outcomes: &[f64],
    profile: LinearLeastSquaresProfile,
) -> Result<LinearLeastSquaresResult, LinearModelError> {
    let profile = profile.validate()?;
    let observations = design.len();
    if observations == 0 {
        return Err(LinearModelError::EmptyDesign);
    }
    let parameters = design[0].len();
    if parameters == 0 {
        return Err(LinearModelError::EmptyParameters);
    }
    if outcomes.len() != observations {
        return Err(LinearModelError::OutcomeLength {
            observations,
            outcomes: outcomes.len(),
        });
    }
    if observations < parameters {
        return Err(LinearModelError::Underdetermined {
            observations,
            parameters,
        });
    }

    for (row_index, row) in design.iter().enumerate() {
        if row.len() != parameters {
            return Err(LinearModelError::RaggedRow {
                row: row_index,
                expected: parameters,
                actual: row.len(),
            });
        }
        if let Some(column) = row.iter().position(|value| !value.is_finite()) {
            return Err(LinearModelError::NonFiniteDesign {
                row: row_index,
                column,
            });
        }
    }
    if let Some(index) = outcomes.iter().position(|value| !value.is_finite()) {
        return Err(LinearModelError::NonFiniteOutcome { index });
    }

    let mut matrix_scale = 0.0_f64;
    for column in 0..parameters {
        matrix_scale = matrix_scale.max(original_column_norm(design, column)?);
    }
    let rank_threshold = profile.relative_rank_tolerance * matrix_scale;
    if !rank_threshold.is_finite() {
        return Err(LinearModelError::NumericalFailure(
            "rank threshold is non-finite",
        ));
    }

    let mut factorization_column_order = (0..parameters).collect::<Vec<_>>();
    let mut q_columns = vec![vec![0.0_f64; observations]; parameters];
    let mut r = vec![vec![0.0_f64; parameters]; parameters];
    let mut diagonal = vec![0.0_f64; parameters];
    let mut rank = 0usize;

    for factor_column in 0..parameters {
        let mut best_position = factor_column;
        let mut best_norm = -1.0_f64;

        for candidate_position in factor_column..parameters {
            let original_column = factorization_column_order[candidate_position];
            let (_, norm) = residualized_column(
                design,
                original_column,
                &q_columns,
                factor_column,
            )?;
            if norm > best_norm {
                best_norm = norm;
                best_position = candidate_position;
            }
        }

        factorization_column_order.swap(factor_column, best_position);
        let original_column = factorization_column_order[factor_column];
        let mut work = (0..observations)
            .map(|row| design[row][original_column])
            .collect::<Vec<_>>();

        for basis in 0..factor_column {
            let projection = dot(&q_columns[basis], &work)?;
            r[basis][factor_column] += projection;
            subtract_scaled(&mut work, &q_columns[basis], projection)?;
        }
        for basis in 0..factor_column {
            let correction = dot(&q_columns[basis], &work)?;
            r[basis][factor_column] += correction;
            subtract_scaled(&mut work, &q_columns[basis], correction)?;
        }

        let norm = stable_l2_norm(work.iter().copied())?;
        r[factor_column][factor_column] = norm;
        diagonal[factor_column] = norm;
        if norm > rank_threshold {
            rank += 1;
            for (target, value) in q_columns[factor_column].iter_mut().zip(work) {
                *target = value / norm;
            }
        }
    }

    let condition_proxy = nonzero_diagonal_condition_proxy(&diagonal, rank_threshold)?;
    if rank < parameters {
        return Ok(LinearLeastSquaresResult {
            method: LINEAR_LEAST_SQUARES_METHOD_V1,
            profile,
            observations,
            parameters,
            rank,
            rank_threshold,
            factorization_column_order,
            condition_proxy,
            disposition: LinearLeastSquaresDisposition::RankDeficient,
            coefficients: None,
            residuals: Vec::new(),
            residual_sum_squares: None,
            rmse: None,
            residual_degrees_of_freedom: None,
        });
    }

    let condition_proxy = condition_proxy.ok_or(LinearModelError::NumericalFailure(
        "full-rank factorization has no nonzero diagonal",
    ))?;

    let mut q_transpose_y = vec![0.0_f64; parameters];
    for column in 0..parameters {
        q_transpose_y[column] = dot(&q_columns[column], outcomes)?;
    }

    let mut pivoted_coefficients = vec![0.0_f64; parameters];
    for row in (0..parameters).rev() {
        let mut value = q_transpose_y[row];
        for column in (row + 1)..parameters {
            value -= r[row][column] * pivoted_coefficients[column];
        }
        let divisor = r[row][row];
        if !divisor.is_finite() || divisor.abs() <= rank_threshold {
            return Err(LinearModelError::NumericalFailure(
                "full-rank triangular solve encountered a singular diagonal",
            ));
        }
        pivoted_coefficients[row] = value / divisor;
        if !pivoted_coefficients[row].is_finite() {
            return Err(LinearModelError::NumericalFailure(
                "coefficient solve produced a non-finite value",
            ));
        }
    }

    let mut coefficients = vec![0.0_f64; parameters];
    for (factor_column, original_column) in factorization_column_order.iter().copied().enumerate() {
        coefficients[original_column] = pivoted_coefficients[factor_column];
    }

    let mut residuals = Vec::with_capacity(observations);
    let mut residual_sum_squares = 0.0_f64;
    for (row, outcome) in design.iter().zip(outcomes.iter().copied()) {
        let prediction = row
            .iter()
            .zip(coefficients.iter())
            .try_fold(0.0_f64, |accumulator, (feature, coefficient)| {
                let next = accumulator + feature * coefficient;
                next.is_finite().then_some(next).ok_or(
                    LinearModelError::NumericalFailure(
                        "prediction produced a non-finite value",
                    ),
                )
            })?;
        let residual = outcome - prediction;
        if !residual.is_finite() {
            return Err(LinearModelError::NumericalFailure(
                "residual produced a non-finite value",
            ));
        }
        residuals.push(residual);
        residual_sum_squares += residual * residual;
        if !residual_sum_squares.is_finite() {
            return Err(LinearModelError::NumericalFailure(
                "residual sum of squares overflowed",
            ));
        }
    }

    let rmse = (residual_sum_squares / observations as f64).sqrt();
    let disposition = if condition_proxy > profile.max_condition_proxy {
        LinearLeastSquaresDisposition::IllConditioned
    } else {
        LinearLeastSquaresDisposition::FullColumnRank
    };

    Ok(LinearLeastSquaresResult {
        method: LINEAR_LEAST_SQUARES_METHOD_V1,
        profile,
        observations,
        parameters,
        rank,
        rank_threshold,
        factorization_column_order,
        condition_proxy: Some(condition_proxy),
        disposition,
        coefficients: Some(coefficients),
        residuals,
        residual_sum_squares: Some(residual_sum_squares),
        rmse: Some(rmse),
        residual_degrees_of_freedom: Some(observations - parameters),
    })
}

fn original_column_norm(design: &[Vec<f64>], column: usize) -> Result<f64, LinearModelError> {
    stable_l2_norm(design.iter().map(|row| row[column]))
}

fn residualized_column(
    design: &[Vec<f64>],
    original_column: usize,
    q_columns: &[Vec<f64>],
    basis_count: usize,
) -> Result<(Vec<f64>, f64), LinearModelError> {
    let mut work = design
        .iter()
        .map(|row| row[original_column])
        .collect::<Vec<_>>();
    for basis in q_columns.iter().take(basis_count) {
        let projection = dot(basis, &work)?;
        subtract_scaled(&mut work, basis, projection)?;
    }
    for basis in q_columns.iter().take(basis_count) {
        let correction = dot(basis, &work)?;
        subtract_scaled(&mut work, basis, correction)?;
    }
    let norm = stable_l2_norm(work.iter().copied())?;
    Ok((work, norm))
}

fn stable_l2_norm(values: impl IntoIterator<Item = f64>) -> Result<f64, LinearModelError> {
    let mut scale = 0.0_f64;
    let mut sum_squares = 1.0_f64;
    let mut nonzero = false;
    for value in values {
        if !value.is_finite() {
            return Err(LinearModelError::NumericalFailure(
                "norm received a non-finite intermediate value",
            ));
        }
        let absolute = value.abs();
        if absolute == 0.0 {
            continue;
        }
        nonzero = true;
        if scale < absolute {
            let ratio = if scale == 0.0 { 0.0 } else { scale / absolute };
            sum_squares = 1.0 + sum_squares * ratio * ratio;
            scale = absolute;
        } else {
            let ratio = absolute / scale;
            sum_squares += ratio * ratio;
        }
    }
    if !nonzero {
        return Ok(0.0);
    }
    let norm = scale * sum_squares.sqrt();
    norm.is_finite().then_some(norm).ok_or(
        LinearModelError::NumericalFailure("norm computation overflowed"),
    )
}

fn dot(left: &[f64], right: &[f64]) -> Result<f64, LinearModelError> {
    let mut sum = 0.0_f64;
    for (left, right) in left.iter().zip(right) {
        sum += left * right;
        if !sum.is_finite() {
            return Err(LinearModelError::NumericalFailure(
                "dot product overflowed",
            ));
        }
    }
    Ok(sum)
}

fn subtract_scaled(
    target: &mut [f64],
    basis: &[f64],
    scale: f64,
) -> Result<(), LinearModelError> {
    for (target, basis) in target.iter_mut().zip(basis) {
        *target -= scale * basis;
        if !target.is_finite() {
            return Err(LinearModelError::NumericalFailure(
                "orthogonalization produced a non-finite value",
            ));
        }
    }
    Ok(())
}

fn nonzero_diagonal_condition_proxy(
    diagonal: &[f64],
    rank_threshold: f64,
) -> Result<Option<f64>, LinearModelError> {
    let mut minimum = f64::INFINITY;
    let mut maximum = 0.0_f64;
    for value in diagonal.iter().copied().map(f64::abs) {
        if value > rank_threshold {
            minimum = minimum.min(value);
            maximum = maximum.max(value);
        }
    }
    if !minimum.is_finite() {
        return Ok(None);
    }
    let condition = maximum / minimum;
    condition.is_finite().then_some(Some(condition)).ok_or(
        LinearModelError::NumericalFailure("condition proxy is non-finite"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solved_coefficients(result: &LinearLeastSquaresResult) -> &[f64] {
        result.coefficients.as_deref().expect("fixture should solve")
    }

    #[test]
    fn exact_affine_model_recovers_intercept_and_slope() {
        let design = vec![vec![1.0, 0.0], vec![1.0, 1.0], vec![1.0, 2.0], vec![1.0, 3.0]];
        let outcomes = [1.0, 3.0, 5.0, 7.0];
        let result = try_linear_least_squares(
            &design,
            &outcomes,
            LinearLeastSquaresProfile::default(),
        )
        .unwrap();
        assert_eq!(result.disposition, LinearLeastSquaresDisposition::FullColumnRank);
        assert_eq!(result.rank, 2);
        let coefficients = solved_coefficients(&result);
        assert!((coefficients[0] - 1.0).abs() < 1.0e-12);
        assert!((coefficients[1] - 2.0).abs() < 1.0e-12);
        assert!(result.residual_sum_squares.unwrap() < 1.0e-24);
        assert_eq!(result.residual_degrees_of_freedom, Some(2));
    }

    #[test]
    fn exact_multi_predictor_model_recovers_column_ordered_coefficients() {
        let design = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, -1.0],
        ];
        let outcomes = [3.0, -2.0, 1.0, 8.0];
        let result = try_linear_least_squares(
            &design,
            &outcomes,
            LinearLeastSquaresProfile::default(),
        )
        .unwrap();
        let coefficients = solved_coefficients(&result);
        assert!((coefficients[0] - 3.0).abs() < 1.0e-12);
        assert!((coefficients[1] + 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn pivoting_is_recorded_and_coefficients_return_to_input_order() {
        let design = vec![
            vec![1.0e-3, 1.0],
            vec![2.0e-3, 0.0],
            vec![3.0e-3, -1.0],
            vec![4.0e-3, 2.0],
        ];
        let outcomes = design
            .iter()
            .map(|row| 2.0 * row[0] + 3.0 * row[1])
            .collect::<Vec<_>>();
        let result = try_linear_least_squares(
            &design,
            &outcomes,
            LinearLeastSquaresProfile::default(),
        )
        .unwrap();
        assert_eq!(result.factorization_column_order[0], 1);
        let coefficients = solved_coefficients(&result);
        assert!((coefficients[0] - 2.0).abs() < 1.0e-10);
        assert!((coefficients[1] - 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn duplicate_column_is_rank_deficient_and_has_no_coefficients() {
        let design = vec![vec![1.0, 2.0], vec![2.0, 4.0], vec![3.0, 6.0]];
        let outcomes = [1.0, 2.0, 3.0];
        let result = try_linear_least_squares(
            &design,
            &outcomes,
            LinearLeastSquaresProfile::default(),
        )
        .unwrap();
        assert_eq!(result.disposition, LinearLeastSquaresDisposition::RankDeficient);
        assert_eq!(result.rank, 1);
        assert!(!result.is_solved());
        assert!(result.residuals.is_empty());
        assert_eq!(result.residual_degrees_of_freedom, None);
    }

    #[test]
    fn explicit_intercept_plus_constant_predictor_is_rank_deficient() {
        let design = vec![vec![1.0, 2.0], vec![1.0, 2.0], vec![1.0, 2.0]];
        let result = try_linear_least_squares(
            &design,
            &[1.0, 2.0, 3.0],
            LinearLeastSquaresProfile::default(),
        )
        .unwrap();
        assert_eq!(result.disposition, LinearLeastSquaresDisposition::RankDeficient);
        assert_eq!(result.rank, 1);
    }

    #[test]
    fn nearly_collinear_design_is_retained_but_flagged_ill_conditioned() {
        let design = vec![
            vec![1.0, 1.0 + 1.0e-9],
            vec![2.0, 2.0 - 1.0e-9],
            vec![3.0, 3.0 + 1.0e-9],
            vec![4.0, 4.0 - 1.0e-9],
        ];
        let outcomes = [2.0, 4.0, 6.0, 8.0];
        let profile = LinearLeastSquaresProfile {
            relative_rank_tolerance: 1.0e-14,
            max_condition_proxy: 1.0e6,
        };
        let result = try_linear_least_squares(&design, &outcomes, profile).unwrap();
        assert_eq!(result.rank, 2);
        assert_eq!(result.disposition, LinearLeastSquaresDisposition::IllConditioned);
        assert!(result.condition_proxy.unwrap() > profile.max_condition_proxy);
        assert!(result.is_solved());
    }

    #[test]
    fn underdetermined_design_rejects() {
        let design = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let error = try_linear_least_squares(
            &design,
            &[1.0, 2.0],
            LinearLeastSquaresProfile::default(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            LinearModelError::Underdetermined {
                observations: 2,
                parameters: 3,
            }
        );
    }

    #[test]
    fn ragged_and_outcome_shape_mismatch_reject() {
        let design = vec![vec![1.0, 2.0], vec![3.0]];
        assert_eq!(
            try_linear_least_squares(
                &design,
                &[1.0, 2.0],
                LinearLeastSquaresProfile::default(),
            )
            .unwrap_err(),
            LinearModelError::RaggedRow {
                row: 1,
                expected: 2,
                actual: 1,
            }
        );
        let design = vec![vec![1.0], vec![2.0]];
        assert_eq!(
            try_linear_least_squares(
                &design,
                &[1.0],
                LinearLeastSquaresProfile::default(),
            )
            .unwrap_err(),
            LinearModelError::OutcomeLength {
                observations: 2,
                outcomes: 1,
            }
        );
    }

    #[test]
    fn non_finite_input_rejects() {
        let design = vec![vec![1.0, f64::NAN], vec![1.0, 2.0]];
        assert_eq!(
            try_linear_least_squares(
                &design,
                &[1.0, 2.0],
                LinearLeastSquaresProfile::default(),
            )
            .unwrap_err(),
            LinearModelError::NonFiniteDesign { row: 0, column: 1 }
        );
        let design = vec![vec![1.0], vec![2.0]];
        assert_eq!(
            try_linear_least_squares(
                &design,
                &[1.0, f64::INFINITY],
                LinearLeastSquaresProfile::default(),
            )
            .unwrap_err(),
            LinearModelError::NonFiniteOutcome { index: 1 }
        );
    }

    #[test]
    fn column_permutation_permutates_coefficients_without_changing_model() {
        let design = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, -1.0],
        ];
        let swapped = design
            .iter()
            .map(|row| vec![row[1], row[0]])
            .collect::<Vec<_>>();
        let outcomes = [3.0, -2.0, 1.0, 8.0];
        let first = try_linear_least_squares(
            &design,
            &outcomes,
            LinearLeastSquaresProfile::default(),
        )
        .unwrap();
        let second = try_linear_least_squares(
            &swapped,
            &outcomes,
            LinearLeastSquaresProfile::default(),
        )
        .unwrap();
        let first = solved_coefficients(&first);
        let second = solved_coefficients(&second);
        assert!((first[0] - second[1]).abs() < 1.0e-12);
        assert!((first[1] - second[0]).abs() < 1.0e-12);
    }

    #[test]
    fn actuator_style_inertia_and_damping_fixture_is_recovered() {
        // tau = I * acceleration + b * velocity, with I=2 and b=0.5.
        let design = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![2.0, -1.0],
            vec![-1.0, 2.0],
            vec![1.5, -0.5],
        ];
        let outcomes = design
            .iter()
            .map(|row| 2.0 * row[0] + 0.5 * row[1])
            .collect::<Vec<_>>();
        let result = try_linear_least_squares(
            &design,
            &outcomes,
            LinearLeastSquaresProfile::default(),
        )
        .unwrap();
        let coefficients = solved_coefficients(&result);
        assert!((coefficients[0] - 2.0).abs() < 1.0e-12);
        assert!((coefficients[1] - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn noisy_overdetermined_fixture_reports_nonzero_residual() {
        let design = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let outcomes = [2.0, 4.1, 5.9, 8.2, 9.8];
        let result = try_linear_least_squares(
            &design,
            &outcomes,
            LinearLeastSquaresProfile::default(),
        )
        .unwrap();
        assert_eq!(result.disposition, LinearLeastSquaresDisposition::FullColumnRank);
        assert!(result.residual_sum_squares.unwrap() > 0.0);
        assert!(result.rmse.unwrap() > 0.0);
    }

    #[test]
    fn profile_validation_rejects_invalid_thresholds() {
        let design = vec![vec![1.0], vec![2.0]];
        let outcomes = [1.0, 2.0];
        for profile in [
            LinearLeastSquaresProfile {
                relative_rank_tolerance: 0.0,
                max_condition_proxy: 1.0e10,
            },
            LinearLeastSquaresProfile {
                relative_rank_tolerance: 1.0e-12,
                max_condition_proxy: 0.5,
            },
        ] {
            assert!(matches!(
                try_linear_least_squares(&design, &outcomes, profile),
                Err(LinearModelError::InvalidProfile(_))
            ));
        }
    }

    #[test]
    fn result_binds_method_profile_and_rank_threshold() {
        let profile = LinearLeastSquaresProfile::default();
        let result = try_linear_least_squares(&[vec![1.0], vec![2.0]], &[3.0, 6.0], profile)
            .unwrap();
        assert_eq!(result.method, LINEAR_LEAST_SQUARES_METHOD_V1);
        assert_eq!(result.profile, profile);
        assert!(result.rank_threshold.is_finite());
        assert!(result.rank_threshold >= 0.0);
    }
}
