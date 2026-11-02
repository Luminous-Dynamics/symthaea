//! Dependency-free reference utilities for information-seeking connectome experiments.
//!
//! This crate implements exact discrete expected information gain (EIG) for a
//! finite hypothesis set and candidate experiments. It is a mathematical
//! selector, not a biological model: callers supply priors, outcome likelihoods,
//! and costs. The code does not estimate or calibrate them.
//!
//! For hypothesis H and candidate experiment e with outcome Y:
//!
//!     EIG(e) = H(H) - E_Y[H(H | Y, e)] = I(H; Y | e)
//!
//! Entropy and EIG are reported in bits. Cost-normalized EIG is bits per unit of
//! caller-defined cost. It is not interchangeable with downstream decision
//! utility or regret.

use core::fmt;
use std::collections::HashSet;

const PROBABILITY_TOLERANCE: f64 = 1.0e-9;
const ROUND_OFF_TOLERANCE: f64 = 1.0e-12;

/// Errors are explicit so malformed priors and likelihoods are never silently
/// normalized into plausible-looking experiment rankings.
#[derive(Debug, Clone, PartialEq)]
pub enum DesignError {
    EmptyHypothesisSet,
    PriorCountMismatch { expected: usize, actual: usize },
    InvalidPrior { index: usize, value: f64 },
    PriorMassNotOne { sum: f64 },
    EmptyExperimentId,
    DuplicateExperimentId { experiment_id: String },
    InvalidExperimentCost { cost: f64 },
    NoCandidateExperiments,
    EmptyOutcomeSet,
    DuplicateOutcomeLabel { label: String },
    LikelihoodRowCountMismatch { expected: usize, actual: usize },
    LikelihoodOutcomeCountMismatch {
        hypothesis_index: usize,
        expected: usize,
        actual: usize,
    },
    InvalidLikelihood {
        hypothesis_index: usize,
        outcome_index: usize,
        value: f64,
    },
    LikelihoodMassNotOne { hypothesis_index: usize, sum: f64 },
    OutcomeIndexOutOfRange { index: usize, outcome_count: usize },
    ImpossibleObservedOutcome { outcome_index: usize },
    SelectedExperimentNotFound { experiment_id: String },
    EmptyActionSet,
    EmptyActionLabel,
    DuplicateActionLabel { label: String },
    DecisionUtilityRowCountMismatch { expected: usize, actual: usize },
    DecisionUtilityActionCountMismatch {
        hypothesis_index: usize,
        expected: usize,
        actual: usize,
    },
    InvalidDecisionUtility {
        hypothesis_index: usize,
        action_index: usize,
        value: f64,
    },
    EmptyParameterSet,
    EmptyMeasurementSet,
    MatrixDimensionMismatch {
        matrix: String,
        expected_rows: usize,
        actual_rows: usize,
        expected_columns: usize,
        actual_columns: usize,
    },
    InvalidMatrixValue {
        matrix: String,
        row: usize,
        column: usize,
        value: f64,
    },
    NonSymmetricMatrix {
        matrix: String,
        row: usize,
        column: usize,
    },
    MatrixNotPositiveDefinite {
        matrix: String,
        pivot: usize,
    },
    NumericalInstability,
}

impl fmt::Display for DesignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyHypothesisSet => write!(f, "hypothesis set must not be empty"),
            Self::PriorCountMismatch { expected, actual } => {
                write!(f, "expected {expected} hypothesis priors, received {actual}")
            }
            Self::InvalidPrior { index, value } => write!(f, "prior {index} is invalid: {value}"),
            Self::PriorMassNotOne { sum } => {
                write!(f, "hypothesis priors must sum to one, received {sum}")
            }
            Self::EmptyExperimentId => write!(f, "experiment id must not be empty"),
            Self::DuplicateExperimentId { experiment_id } => {
                write!(f, "duplicate experiment id {experiment_id:?}")
            }
            Self::InvalidExperimentCost { cost } => {
                write!(f, "experiment cost must be finite and positive, received {cost}")
            }
            Self::NoCandidateExperiments => write!(f, "candidate experiment set must not be empty"),
            Self::EmptyOutcomeSet => write!(f, "candidate experiment must declare at least one outcome"),
            Self::DuplicateOutcomeLabel { label } => {
                write!(f, "candidate experiment has duplicate outcome label {label:?}")
            }
            Self::LikelihoodRowCountMismatch { expected, actual } => {
                write!(f, "expected {expected} likelihood rows, received {actual}")
            }
            Self::LikelihoodOutcomeCountMismatch {
                hypothesis_index,
                expected,
                actual,
            } => write!(
                f,
                "hypothesis {hypothesis_index} has {actual} outcome likelihoods; expected {expected}"
            ),
            Self::InvalidLikelihood {
                hypothesis_index,
                outcome_index,
                value,
            } => write!(
                f,
                "likelihood for hypothesis {hypothesis_index}, outcome {outcome_index} is invalid: {value}"
            ),
            Self::LikelihoodMassNotOne { hypothesis_index, sum } => write!(
                f,
                "likelihoods for hypothesis {hypothesis_index} must sum to one, received {sum}"
            ),
            Self::OutcomeIndexOutOfRange { index, outcome_count } => {
                write!(f, "outcome index {index} is outside 0..{outcome_count}")
            }
            Self::ImpossibleObservedOutcome { outcome_index } => {
                write!(f, "outcome {outcome_index} has zero probability under every hypothesis")
            }
            Self::SelectedExperimentNotFound { experiment_id } => {
                write!(f, "selected experiment {experiment_id:?} is not in the candidate set")
            }
            Self::EmptyActionSet => write!(f, "decision action set must not be empty"),
            Self::EmptyActionLabel => write!(f, "decision action label must not be empty"),
            Self::DuplicateActionLabel { label } => {
                write!(f, "duplicate decision action label {label:?}")
            }
            Self::DecisionUtilityRowCountMismatch { expected, actual } => write!(
                f,
                "expected {expected} decision-utility rows, received {actual}"
            ),
            Self::DecisionUtilityActionCountMismatch {
                hypothesis_index,
                expected,
                actual,
            } => write!(
                f,
                "hypothesis {hypothesis_index} has {actual} action utilities; expected {expected}"
            ),
            Self::InvalidDecisionUtility {
                hypothesis_index,
                action_index,
                value,
            } => write!(
                f,
                "utility for hypothesis {hypothesis_index}, action {action_index} is invalid: {value}"
            ),
            Self::EmptyParameterSet => write!(f, "parameter set must not be empty"),
            Self::EmptyMeasurementSet => write!(f, "design must contain at least one measurement"),
            Self::MatrixDimensionMismatch {
                matrix,
                expected_rows,
                actual_rows,
                expected_columns,
                actual_columns,
            } => write!(
                f,
                "{matrix} dimensions are {actual_rows}x{actual_columns}; expected {expected_rows}x{expected_columns}"
            ),
            Self::InvalidMatrixValue { matrix, row, column, value } => write!(
                f,
                "{matrix}[{row},{column}] is non-finite: {value}"
            ),
            Self::NonSymmetricMatrix { matrix, row, column } => write!(
                f,
                "{matrix} is not symmetric at ({row},{column})"
            ),
            Self::MatrixNotPositiveDefinite { matrix, pivot } => write!(
                f,
                "{matrix} is not positive definite at Cholesky pivot {pivot}"
            ),
            Self::NumericalInstability => {
                write!(f, "information or decision-value calculation exceeded its numerical tolerance")
            }
        }
    }
}

impl std::error::Error for DesignError {}

/// A candidate experiment's likelihood model.
///
/// likelihoods_by_hypothesis[h][y] is P(Y = outcome[y] | H = h, experiment).
/// Every row must be a normalized probability distribution. The outcome order
/// is shared across all hypotheses.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateExperiment {
    pub id: String,
    /// Caller-defined resource cost. Must be finite and strictly positive.
    pub cost: f64,
    pub outcomes: Vec<String>,
    pub likelihoods_by_hypothesis: Vec<Vec<f64>>,
}

/// Information score for one candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InformationGainScore {
    pub prior_entropy_bits: f64,
    pub expected_posterior_entropy_bits: f64,
    pub information_gain_bits: f64,
    pub cost: f64,
    pub information_gain_bits_per_cost: f64,
}

/// Candidate and score after deterministic ranking.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedExperiment {
    pub experiment_id: String,
    pub score: InformationGainScore,
}

/// Opportunity regret relative to the best candidate in a declared set.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionRegret {
    pub selected_experiment_id: String,
    pub best_experiment_id: String,
    pub selected_utility_bits_per_cost: f64,
    pub best_utility_bits_per_cost: f64,
    pub regret_bits_per_cost: f64,
}

/// Decision problem used to calculate expected value of sample information.
///
/// `utilities_by_hypothesis[h][a]` is the utility of selecting action `a`
/// when hypothesis `h` is true. Utilities are caller-defined, finite values;
/// they need not be probabilities or non-negative.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionProblem {
    pub actions: Vec<String>,
    pub utilities_by_hypothesis: Vec<Vec<f64>>,
}

/// Expected value of sample information (EVSI) for one candidate experiment.
///
/// EVSI is measured in the caller's utility units. It is deliberately distinct
/// from EIG (bits): an experiment can distinguish hypotheses that imply the
/// same action and therefore have high EIG but zero decision value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecisionValueScore {
    pub prior_best_expected_utility: f64,
    pub expected_best_utility_after_experiment: f64,
    pub expected_value_of_sample_information: f64,
    pub cost: f64,
    pub expected_value_per_cost: f64,
}

/// Candidate and decision-value score after deterministic ranking.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedDecisionExperiment {
    pub experiment_id: String,
    pub score: DecisionValueScore,
}

/// Decision regret relative to the best candidate in the supplied finite pool.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionSelectionRegret {
    pub selected_experiment_id: String,
    pub best_experiment_id: String,
    pub selected_expected_value_per_cost: f64,
    pub best_expected_value_per_cost: f64,
    pub regret_per_cost: f64,
}

/// Gaussian prior over an unknown parameter vector.
///
/// The covariance must be finite, symmetric and strictly positive definite.
/// The prior mean is omitted because local linear-Gaussian information gain
/// depends on covariance, not the location of the mean.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterPrior {
    pub covariance: Vec<Vec<f64>>,
}

/// A candidate measurement design for a local linear-Gaussian observation model.
///
/// `sensitivities[measurement][parameter]` is the Jacobian of the expected
/// observation with respect to the unknown parameter vector. The observation
/// noise covariance may include temporal/spatial measurement correlations.
#[derive(Debug, Clone, PartialEq)]
pub struct LinearGaussianDesign {
    pub id: String,
    pub cost: f64,
    pub sensitivities: Vec<Vec<f64>>,
    pub noise_covariance: Vec<Vec<f64>>,
}

/// Parameter-identification scores for both D- and A-optimal design.
///
/// D-optimal information is mutual information in bits under the stated
/// Gaussian linear model. A-optimal reduction is the decrease in covariance
/// trace, in the squared units of the parameter vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParameterInformationScore {
    pub cost: f64,
    pub prior_log_determinant_covariance: f64,
    pub posterior_log_determinant_covariance: f64,
    pub d_optimal_information_bits: f64,
    pub d_optimal_bits_per_cost: f64,
    pub prior_covariance_trace: f64,
    pub posterior_covariance_trace: f64,
    pub a_optimal_trace_reduction: f64,
    pub a_optimal_reduction_per_cost: f64,
}

/// Objective used to rank parameter-identification experiments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterObjective {
    DOptimal,
    AOptimal,
}

/// Candidate and score after deterministic parameter-design ranking.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedParameterExperiment {
    pub experiment_id: String,
    pub score: ParameterInformationScore,
}

/// Opportunity regret against the best candidate in the supplied finite pool.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterSelectionRegret {
    pub selected_experiment_id: String,
    pub best_experiment_id: String,
    pub objective: ParameterObjective,
    pub selected_value_per_cost: f64,
    pub best_value_per_cost: f64,
    pub regret_per_cost: f64,
}

const MATRIX_SYMMETRY_TOLERANCE: f64 = 1.0e-10;

/// Validate square finite matrix structure and numerical symmetry.
fn validate_symmetric_matrix(
    matrix: &[Vec<f64>],
    name: &str,
) -> Result<usize, DesignError> {
    let dimension = matrix.len();
    if dimension == 0 {
        return Err(DesignError::EmptyParameterSet);
    }
    for (row_index, row) in matrix.iter().enumerate() {
        if row.len() != dimension {
            return Err(DesignError::MatrixDimensionMismatch {
                matrix: name.to_owned(),
                expected_rows: dimension,
                actual_rows: dimension,
                expected_columns: dimension,
                actual_columns: row.len(),
            });
        }
        for (column_index, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(DesignError::InvalidMatrixValue {
                    matrix: name.to_owned(),
                    row: row_index,
                    column: column_index,
                    value,
                });
            }
            let transpose = matrix[column_index][row_index];
            if !transpose.is_finite() {
                return Err(DesignError::InvalidMatrixValue {
                    matrix: name.to_owned(),
                    row: column_index,
                    column: row_index,
                    value: transpose,
                });
            }
            let scale = value.abs().max(transpose.abs()).max(1.0);
            if (value - transpose).abs() > MATRIX_SYMMETRY_TOLERANCE * scale {
                return Err(DesignError::NonSymmetricMatrix {
                    matrix: name.to_owned(),
                    row: row_index,
                    column: column_index,
                });
            }
        }
    }
    Ok(dimension)
}

/// Lower-triangular Cholesky factor L for A = L L^T.
fn cholesky(
    matrix: &[Vec<f64>],
    name: &str,
) -> Result<Vec<Vec<f64>>, DesignError> {
    let dimension = validate_symmetric_matrix(matrix, name)?;
    let mut lower = vec![vec![0.0; dimension]; dimension];
    for row in 0..dimension {
        for column in 0..=row {
            let mut value = matrix[row][column];
            for k in 0..column {
                value -= lower[row][k] * lower[column][k];
            }
            if !value.is_finite() {
                return Err(DesignError::NumericalInstability);
            }
            if row == column {
                if value <= 0.0 {
                    return Err(DesignError::MatrixNotPositiveDefinite {
                        matrix: name.to_owned(),
                        pivot: row,
                    });
                }
                lower[row][column] = value.sqrt();
            } else {
                let diagonal = lower[column][column];
                if diagonal <= 0.0 || !diagonal.is_finite() {
                    return Err(DesignError::MatrixNotPositiveDefinite {
                        matrix: name.to_owned(),
                        pivot: column,
                    });
                }
                lower[row][column] = value / diagonal;
            }
        }
    }
    Ok(lower)
}

/// Solve L L^T x = b for an SPD matrix using forward/back substitution.
fn solve_cholesky(lower: &[Vec<f64>], rhs: &[f64]) -> Result<Vec<f64>, DesignError> {
    let dimension = lower.len();
    if rhs.len() != dimension {
        return Err(DesignError::NumericalInstability);
    }
    let mut forward = vec![0.0; dimension];
    for row in 0..dimension {
        let mut value = rhs[row];
        for column in 0..row {
            value -= lower[row][column] * forward[column];
        }
        forward[row] = value / lower[row][row];
        if !forward[row].is_finite() {
            return Err(DesignError::NumericalInstability);
        }
    }
    let mut result = vec![0.0; dimension];
    for row in (0..dimension).rev() {
        let mut value = forward[row];
        for column in (row + 1)..dimension {
            value -= lower[column][row] * result[column];
        }
        result[row] = value / lower[row][row];
        if !result[row].is_finite() {
            return Err(DesignError::NumericalInstability);
        }
    }
    Ok(result)
}

fn inverse_spd(lower: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, DesignError> {
    let dimension = lower.len();
    let mut inverse = vec![vec![0.0; dimension]; dimension];
    for column in 0..dimension {
        let mut basis = vec![0.0; dimension];
        basis[column] = 1.0;
        let solution = solve_cholesky(lower, &basis)?;
        for row in 0..dimension {
            inverse[row][column] = solution[row];
        }
    }
    // Remove only floating-point asymmetry introduced by independent solves.
    for row in 0..dimension {
        for column in 0..row {
            let average = 0.5 * (inverse[row][column] + inverse[column][row]);
            inverse[row][column] = average;
            inverse[column][row] = average;
        }
    }
    Ok(inverse)
}

fn log_determinant_from_cholesky(lower: &[Vec<f64>]) -> Result<f64, DesignError> {
    let mut result = 0.0;
    for index in 0..lower.len() {
        let diagonal = lower[index][index];
        if diagonal <= 0.0 || !diagonal.is_finite() {
            return Err(DesignError::NumericalInstability);
        }
        result += 2.0 * diagonal.ln();
    }
    if result.is_finite() {
        Ok(result)
    } else {
        Err(DesignError::NumericalInstability)
    }
}

fn matrix_trace(matrix: &[Vec<f64>]) -> f64 {
    (0..matrix.len()).map(|index| matrix[index][index]).sum()
}

fn validate_parameter_design(
    prior: &ParameterPrior,
    design: &LinearGaussianDesign,
) -> Result<(usize, usize), DesignError> {
    if design.id.trim().is_empty() {
        return Err(DesignError::EmptyExperimentId);
    }
    if !design.cost.is_finite() || design.cost <= 0.0 {
        return Err(DesignError::InvalidExperimentCost { cost: design.cost });
    }
    let parameter_count = validate_symmetric_matrix(&prior.covariance, "prior covariance")?;
    cholesky(&prior.covariance, "prior covariance")?;

    let measurement_count = design.sensitivities.len();
    if measurement_count == 0 {
        return Err(DesignError::EmptyMeasurementSet);
    }
    for (row_index, row) in design.sensitivities.iter().enumerate() {
        if row.len() != parameter_count {
            return Err(DesignError::MatrixDimensionMismatch {
                matrix: "sensitivities".to_owned(),
                expected_rows: measurement_count,
                actual_rows: measurement_count,
                expected_columns: parameter_count,
                actual_columns: row.len(),
            });
        }
        for (column_index, &value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(DesignError::InvalidMatrixValue {
                    matrix: "sensitivities".to_owned(),
                    row: row_index,
                    column: column_index,
                    value,
                });
            }
        }
    }
    let noise_dimension = validate_symmetric_matrix(&design.noise_covariance, "noise covariance")?;
    if noise_dimension != measurement_count {
        return Err(DesignError::MatrixDimensionMismatch {
            matrix: "noise covariance".to_owned(),
            expected_rows: measurement_count,
            actual_rows: noise_dimension,
            expected_columns: measurement_count,
            actual_columns: noise_dimension,
        });
    }
    cholesky(&design.noise_covariance, "noise covariance")?;
    Ok((parameter_count, measurement_count))
}

/// Compute local Gaussian Fisher information and posterior parameter covariance.
///
/// Model: y = f(theta_0) + J (theta - theta_0) + epsilon,
/// epsilon ~ Normal(0, R), theta ~ Normal(theta_0, Sigma).
/// The expected Fisher information is J^T R^-1 J and the Gaussian posterior
/// covariance is (Sigma^-1 + J^T R^-1 J)^-1.
///
/// D-optimal information is 0.5 * log2(det(Sigma) / det(Sigma_post)) bits.
/// A-optimal utility is trace(Sigma) - trace(Sigma_post). This is a local
/// linear-Gaussian design calculation; it does not estimate J or R.
pub fn evaluate_parameter_design(
    prior: &ParameterPrior,
    design: &LinearGaussianDesign,
) -> Result<ParameterInformationScore, DesignError> {
    let (_parameter_count, measurement_count) = validate_parameter_design(prior, design)?;
    let prior_factor = cholesky(&prior.covariance, "prior covariance")?;
    let noise_factor = cholesky(&design.noise_covariance, "noise covariance")?;
    let prior_precision = inverse_spd(&prior_factor)?;

    // Solve R Z = J one parameter column at a time. This avoids explicitly
    // forming R^-1, which is less stable and unnecessary.
    let parameter_count = prior.covariance.len();
    let mut noise_whitened_sensitivities = vec![vec![0.0; parameter_count]; measurement_count];
    for parameter in 0..parameter_count {
        let rhs: Vec<f64> = design
            .sensitivities
            .iter()
            .map(|row| row[parameter])
            .collect();
        let solution = solve_cholesky(&noise_factor, &rhs)?;
        for measurement in 0..measurement_count {
            noise_whitened_sensitivities[measurement][parameter] = solution[measurement];
        }
    }

    // F = J^T R^-1 J.
    let mut fisher = vec![vec![0.0; parameter_count]; parameter_count];
    for row in 0..parameter_count {
        for column in 0..parameter_count {
            fisher[row][column] = (0..measurement_count)
                .map(|measurement| {
                    design.sensitivities[measurement][row]
                        * noise_whitened_sensitivities[measurement][column]
                })
                .sum();
        }
    }
    for row in 0..parameter_count {
        for column in 0..row {
            let average = 0.5 * (fisher[row][column] + fisher[column][row]);
            fisher[row][column] = average;
            fisher[column][row] = average;
        }
    }

    let mut posterior_precision = prior_precision.clone();
    for row in 0..parameter_count {
        for column in 0..parameter_count {
            posterior_precision[row][column] += fisher[row][column];
        }
    }
    let posterior_precision_factor = cholesky(&posterior_precision, "posterior precision")?;
    let posterior_covariance = inverse_spd(&posterior_precision_factor)?;

    let prior_log_determinant_covariance =
        log_determinant_from_cholesky(&prior_factor)?;
    let posterior_log_determinant_covariance =
        -log_determinant_from_cholesky(&posterior_precision_factor)?;
    let mut d_optimal_information_bits = 0.5
        * (prior_log_determinant_covariance - posterior_log_determinant_covariance)
        / std::f64::consts::LN_2;
    if d_optimal_information_bits < 0.0
        && d_optimal_information_bits >= -ROUND_OFF_TOLERANCE
    {
        d_optimal_information_bits = 0.0;
    }
    if !d_optimal_information_bits.is_finite() || d_optimal_information_bits < 0.0 {
        return Err(DesignError::NumericalInstability);
    }

    let prior_covariance_trace = matrix_trace(&prior.covariance);
    let posterior_covariance_trace = matrix_trace(&posterior_covariance);
    let mut a_optimal_trace_reduction =
        prior_covariance_trace - posterior_covariance_trace;
    if a_optimal_trace_reduction < 0.0
        && a_optimal_trace_reduction >= -ROUND_OFF_TOLERANCE
    {
        a_optimal_trace_reduction = 0.0;
    }
    if !prior_covariance_trace.is_finite()
        || !posterior_covariance_trace.is_finite()
        || !a_optimal_trace_reduction.is_finite()
        || a_optimal_trace_reduction < 0.0
    {
        return Err(DesignError::NumericalInstability);
    }

    let d_optimal_bits_per_cost = d_optimal_information_bits / design.cost;
    let a_optimal_reduction_per_cost = a_optimal_trace_reduction / design.cost;
    if !d_optimal_bits_per_cost.is_finite()
        || !a_optimal_reduction_per_cost.is_finite()
    {
        return Err(DesignError::NumericalInstability);
    }

    Ok(ParameterInformationScore {
        cost: design.cost,
        prior_log_determinant_covariance,
        posterior_log_determinant_covariance,
        d_optimal_information_bits,
        d_optimal_bits_per_cost,
        prior_covariance_trace,
        posterior_covariance_trace,
        a_optimal_trace_reduction,
        a_optimal_reduction_per_cost,
    })
}

fn parameter_score_values(
    score: &ParameterInformationScore,
    objective: ParameterObjective,
) -> (f64, f64) {
    match objective {
        ParameterObjective::DOptimal => (
            score.d_optimal_bits_per_cost,
            score.d_optimal_information_bits,
        ),
        ParameterObjective::AOptimal => (
            score.a_optimal_reduction_per_cost,
            score.a_optimal_trace_reduction,
        ),
    }
}

/// Rank candidates by D-optimal or A-optimal information per cost, then the
/// corresponding raw objective and stable ID.
pub fn rank_parameter_candidates(
    prior: &ParameterPrior,
    candidates: &[LinearGaussianDesign],
    objective: ParameterObjective,
) -> Result<Vec<RankedParameterExperiment>, DesignError> {
    if candidates.is_empty() {
        return Err(DesignError::NoCandidateExperiments);
    }
    let mut identifiers = HashSet::with_capacity(candidates.len());
    let mut ranked = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if !identifiers.insert(candidate.id.as_str()) {
            return Err(DesignError::DuplicateExperimentId {
                experiment_id: candidate.id.clone(),
            });
        }
        ranked.push(RankedParameterExperiment {
            experiment_id: candidate.id.clone(),
            score: evaluate_parameter_design(prior, candidate)?,
        });
    }
    ranked.sort_by(|left, right| {
        let (left_cost_adjusted, left_raw) =
            parameter_score_values(&left.score, objective);
        let (right_cost_adjusted, right_raw) =
            parameter_score_values(&right.score, objective);
        right_cost_adjusted
            .total_cmp(&left_cost_adjusted)
            .then_with(|| right_raw.total_cmp(&left_raw))
            .then_with(|| left.experiment_id.cmp(&right.experiment_id))
    });
    Ok(ranked)
}

/// Regret relative to the best candidate under the selected parameter objective.
pub fn parameter_selection_regret(
    prior: &ParameterPrior,
    candidates: &[LinearGaussianDesign],
    objective: ParameterObjective,
    selected_experiment_id: &str,
) -> Result<ParameterSelectionRegret, DesignError> {
    let ranked = rank_parameter_candidates(prior, candidates, objective)?;
    let best = ranked.first().ok_or(DesignError::NoCandidateExperiments)?;
    let selected = ranked
        .iter()
        .find(|candidate| candidate.experiment_id == selected_experiment_id)
        .ok_or_else(|| DesignError::SelectedExperimentNotFound {
            experiment_id: selected_experiment_id.to_owned(),
        })?;
    let (best_value, _) = parameter_score_values(&best.score, objective);
    let (selected_value, _) = parameter_score_values(&selected.score, objective);
    Ok(ParameterSelectionRegret {
        selected_experiment_id: selected.experiment_id.clone(),
        best_experiment_id: best.experiment_id.clone(),
        objective,
        selected_value_per_cost: selected_value,
        best_value_per_cost: best_value,
        regret_per_cost: (best_value - selected_value).max(0.0),
    })
}

/// Shannon entropy in bits. Input probabilities must be non-negative and sum
/// to one; all production calls validate this before evaluating entropy.
pub fn entropy_bits(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|&&probability| probability > 0.0)
        .map(|&probability| -probability * probability.log2())
        .sum()
}

fn validate_priors(priors: &[f64]) -> Result<(), DesignError> {
    if priors.is_empty() {
        return Err(DesignError::EmptyHypothesisSet);
    }

    let mut sum = 0.0;
    for (index, &prior) in priors.iter().enumerate() {
        if !prior.is_finite() || prior < 0.0 {
            return Err(DesignError::InvalidPrior { index, value: prior });
        }
        sum += prior;
    }

    if !sum.is_finite() || (sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(DesignError::PriorMassNotOne { sum });
    }
    Ok(())
}

fn validate_candidate(
    priors: &[f64],
    candidate: &CandidateExperiment,
) -> Result<usize, DesignError> {
    validate_priors(priors)?;

    if candidate.id.trim().is_empty() {
        return Err(DesignError::EmptyExperimentId);
    }
    if !candidate.cost.is_finite() || candidate.cost <= 0.0 {
        return Err(DesignError::InvalidExperimentCost { cost: candidate.cost });
    }
    if candidate.outcomes.is_empty() {
        return Err(DesignError::EmptyOutcomeSet);
    }

    let mut labels = HashSet::with_capacity(candidate.outcomes.len());
    for label in &candidate.outcomes {
        if !labels.insert(label.as_str()) {
            return Err(DesignError::DuplicateOutcomeLabel { label: label.clone() });
        }
    }

    if candidate.likelihoods_by_hypothesis.len() != priors.len() {
        return Err(DesignError::LikelihoodRowCountMismatch {
            expected: priors.len(),
            actual: candidate.likelihoods_by_hypothesis.len(),
        });
    }

    let outcome_count = candidate.outcomes.len();
    for (hypothesis_index, row) in candidate.likelihoods_by_hypothesis.iter().enumerate() {
        if row.len() != outcome_count {
            return Err(DesignError::LikelihoodOutcomeCountMismatch {
                hypothesis_index,
                expected: outcome_count,
                actual: row.len(),
            });
        }

        let mut row_sum = 0.0;
        for (outcome_index, &likelihood) in row.iter().enumerate() {
            if !likelihood.is_finite() || likelihood < 0.0 {
                return Err(DesignError::InvalidLikelihood {
                    hypothesis_index,
                    outcome_index,
                    value: likelihood,
                });
            }
            row_sum += likelihood;
        }

        if !row_sum.is_finite() || (row_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
            return Err(DesignError::LikelihoodMassNotOne {
                hypothesis_index,
                sum: row_sum,
            });
        }
    }

    Ok(outcome_count)
}

/// Calculate EIG exactly for a finite discrete hypothesis and outcome space.
pub fn evaluate_candidate(
    priors: &[f64],
    candidate: &CandidateExperiment,
) -> Result<InformationGainScore, DesignError> {
    let outcome_count = validate_candidate(priors, candidate)?;
    let prior_entropy = entropy_bits(priors);

    // P(y) = sum_h P(h) P(y | h, e).
    let mut outcome_marginals = vec![0.0; outcome_count];
    for (hypothesis_index, &prior) in priors.iter().enumerate() {
        for (outcome_index, &likelihood) in candidate.likelihoods_by_hypothesis[hypothesis_index]
            .iter()
            .enumerate()
        {
            outcome_marginals[outcome_index] += prior * likelihood;
        }
    }

    // E_y[H(H | y, e)] using Bayes' rule.
    let mut expected_posterior_entropy = 0.0;
    for (outcome_index, &marginal) in outcome_marginals.iter().enumerate() {
        if marginal <= 0.0 {
            continue;
        }

        let posterior: Vec<f64> = priors
            .iter()
            .enumerate()
            .map(|(hypothesis_index, &prior)| {
                prior * candidate.likelihoods_by_hypothesis[hypothesis_index][outcome_index]
                    / marginal
            })
            .collect();

        expected_posterior_entropy += marginal * entropy_bits(&posterior);
    }

    let mut information_gain = prior_entropy - expected_posterior_entropy;
    if information_gain < 0.0 && information_gain >= -ROUND_OFF_TOLERANCE {
        information_gain = 0.0;
    }
    if information_gain > prior_entropy
        && information_gain <= prior_entropy + ROUND_OFF_TOLERANCE
    {
        information_gain = prior_entropy;
    }
    if !information_gain.is_finite()
        || information_gain < 0.0
        || information_gain > prior_entropy + ROUND_OFF_TOLERANCE
    {
        return Err(DesignError::NumericalInstability);
    }

    let information_gain_bits_per_cost = information_gain / candidate.cost;
    if !information_gain_bits_per_cost.is_finite() {
        return Err(DesignError::NumericalInstability);
    }
    Ok(InformationGainScore {
        prior_entropy_bits: prior_entropy,
        expected_posterior_entropy_bits: expected_posterior_entropy,
        information_gain_bits: information_gain,
        cost: candidate.cost,
        information_gain_bits_per_cost,
    })
}

/// Return P(Y = outcomes[outcome_index] | experiment).
pub fn outcome_probability(
    priors: &[f64],
    candidate: &CandidateExperiment,
    outcome_index: usize,
) -> Result<f64, DesignError> {
    let outcome_count = validate_candidate(priors, candidate)?;
    if outcome_index >= outcome_count {
        return Err(DesignError::OutcomeIndexOutOfRange {
            index: outcome_index,
            outcome_count,
        });
    }

    Ok(priors
        .iter()
        .enumerate()
        .map(|(hypothesis_index, &prior)| {
            prior * candidate.likelihoods_by_hypothesis[hypothesis_index][outcome_index]
        })
        .sum())
}

/// Update the hypothesis prior after observing one candidate outcome.
pub fn posterior_for_observed_outcome(
    priors: &[f64],
    candidate: &CandidateExperiment,
    outcome_index: usize,
) -> Result<Vec<f64>, DesignError> {
    let marginal = outcome_probability(priors, candidate, outcome_index)?;
    if marginal <= 0.0 || !marginal.is_finite() {
        return Err(DesignError::ImpossibleObservedOutcome { outcome_index });
    }

    Ok(priors
        .iter()
        .enumerate()
        .map(|(hypothesis_index, &prior)| {
            prior * candidate.likelihoods_by_hypothesis[hypothesis_index][outcome_index]
                / marginal
        })
        .collect())
}

/// Rank by EIG per cost, then raw EIG, then ascending ID. The final ID
/// tie-break makes output stable regardless of input order.
pub fn rank_candidates(
    priors: &[f64],
    candidates: &[CandidateExperiment],
) -> Result<Vec<RankedExperiment>, DesignError> {
    validate_priors(priors)?;
    if candidates.is_empty() {
        return Err(DesignError::NoCandidateExperiments);
    }

    let mut ids = HashSet::with_capacity(candidates.len());
    let mut ranked = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if !ids.insert(candidate.id.as_str()) {
            return Err(DesignError::DuplicateExperimentId {
                experiment_id: candidate.id.clone(),
            });
        }
        ranked.push(RankedExperiment {
            experiment_id: candidate.id.clone(),
            score: evaluate_candidate(priors, candidate)?,
        });
    }

    ranked.sort_by(|left, right| {
        right
            .score
            .information_gain_bits_per_cost
            .total_cmp(&left.score.information_gain_bits_per_cost)
            .then_with(|| {
                right
                    .score
                    .information_gain_bits
                    .total_cmp(&left.score.information_gain_bits)
            })
            .then_with(|| left.experiment_id.cmp(&right.experiment_id))
    });

    Ok(ranked)
}

/// Compute opportunity regret under the EIG-per-cost objective. This is
/// regret relative to the best experiment in this finite candidate set, not
/// behavioural utility or real-world scientific value.
pub fn selection_regret(
    priors: &[f64],
    candidates: &[CandidateExperiment],
    selected_experiment_id: &str,
) -> Result<SelectionRegret, DesignError> {
    let ranked = rank_candidates(priors, candidates)?;
    let best = ranked.first().ok_or(DesignError::NoCandidateExperiments)?;
    let selected = ranked
        .iter()
        .find(|candidate| candidate.experiment_id == selected_experiment_id)
        .ok_or_else(|| DesignError::SelectedExperimentNotFound {
            experiment_id: selected_experiment_id.to_owned(),
        })?;

    let best_utility = best.score.information_gain_bits_per_cost;
    let selected_utility = selected.score.information_gain_bits_per_cost;
    let regret = (best_utility - selected_utility).max(0.0);

    Ok(SelectionRegret {
        selected_experiment_id: selected.experiment_id.clone(),
        best_experiment_id: best.experiment_id.clone(),
        selected_utility_bits_per_cost: selected_utility,
        best_utility_bits_per_cost: best_utility,
        regret_bits_per_cost: regret,
    })
}


fn validate_decision_problem(
    priors: &[f64],
    problem: &DecisionProblem,
) -> Result<(), DesignError> {
    validate_priors(priors)?;
    if problem.actions.is_empty() {
        return Err(DesignError::EmptyActionSet);
    }
    let mut action_labels = HashSet::with_capacity(problem.actions.len());
    for label in &problem.actions {
        if label.trim().is_empty() {
            return Err(DesignError::EmptyActionLabel);
        }
        if !action_labels.insert(label.as_str()) {
            return Err(DesignError::DuplicateActionLabel { label: label.clone() });
        }
    }
    if problem.utilities_by_hypothesis.len() != priors.len() {
        return Err(DesignError::DecisionUtilityRowCountMismatch {
            expected: priors.len(),
            actual: problem.utilities_by_hypothesis.len(),
        });
    }
    for (hypothesis_index, row) in problem.utilities_by_hypothesis.iter().enumerate() {
        if row.len() != problem.actions.len() {
            return Err(DesignError::DecisionUtilityActionCountMismatch {
                hypothesis_index,
                expected: problem.actions.len(),
                actual: row.len(),
            });
        }
        for (action_index, &utility) in row.iter().enumerate() {
            if !utility.is_finite() {
                return Err(DesignError::InvalidDecisionUtility {
                    hypothesis_index,
                    action_index,
                    value: utility,
                });
            }
        }
    }
    Ok(())
}

fn best_expected_utility(
    probabilities: &[f64],
    utilities_by_hypothesis: &[Vec<f64>],
    action_count: usize,
) -> Result<f64, DesignError> {
    let mut best = f64::NEG_INFINITY;
    for action_index in 0..action_count {
        let expected = probabilities
            .iter()
            .enumerate()
            .map(|(h, probability)| probability * utilities_by_hypothesis[h][action_index])
            .sum::<f64>();
        if !expected.is_finite() {
            return Err(DesignError::NumericalInstability);
        }
        best = best.max(expected);
    }
    if best.is_finite() {
        Ok(best)
    } else {
        Err(DesignError::NumericalInstability)
    }
}

/// Evaluate the expected value of sample information for a candidate.
///
/// This computes the expected improvement in the best available action after
/// observing the candidate's outcome, compared with choosing an action before
/// running the experiment. It is not the same objective as EIG or EIG/cost.
pub fn evaluate_decision_value(
    priors: &[f64],
    candidate: &CandidateExperiment,
    problem: &DecisionProblem,
) -> Result<DecisionValueScore, DesignError> {
    let outcome_count = validate_candidate(priors, candidate)?;
    validate_decision_problem(priors, problem)?;

    let prior_best = best_expected_utility(
        priors,
        &problem.utilities_by_hypothesis,
        problem.actions.len(),
    )?;

    let mut marginals = vec![0.0; outcome_count];
    for (h, &prior) in priors.iter().enumerate() {
        for (y, &likelihood) in candidate.likelihoods_by_hypothesis[h].iter().enumerate() {
            marginals[y] += prior * likelihood;
        }
    }

    let mut expected_post_experiment_best = 0.0;
    for (outcome_index, &marginal) in marginals.iter().enumerate() {
        if marginal <= 0.0 {
            continue;
        }
        let posterior: Vec<f64> = priors
            .iter()
            .enumerate()
            .map(|(h, &prior)| {
                prior * candidate.likelihoods_by_hypothesis[h][outcome_index] / marginal
            })
            .collect();
        let post_best = best_expected_utility(
            &posterior,
            &problem.utilities_by_hypothesis,
            problem.actions.len(),
        )?;
        expected_post_experiment_best += marginal * post_best;
    }

    let mut evsi = expected_post_experiment_best - prior_best;
    if evsi < 0.0 && evsi >= -ROUND_OFF_TOLERANCE {
        evsi = 0.0;
    }
    if !evsi.is_finite() || evsi < 0.0 {
        return Err(DesignError::NumericalInstability);
    }

    let expected_value_per_cost = evsi / candidate.cost;
    if !expected_value_per_cost.is_finite() {
        return Err(DesignError::NumericalInstability);
    }
    Ok(DecisionValueScore {
        prior_best_expected_utility: prior_best,
        expected_best_utility_after_experiment: expected_post_experiment_best,
        expected_value_of_sample_information: evsi,
        cost: candidate.cost,
        expected_value_per_cost,
    })
}

/// Rank candidate experiments by EVSI per cost, then raw EVSI, then stable ID.
pub fn rank_candidates_by_decision_value(
    priors: &[f64],
    candidates: &[CandidateExperiment],
    problem: &DecisionProblem,
) -> Result<Vec<RankedDecisionExperiment>, DesignError> {
    validate_priors(priors)?;
    validate_decision_problem(priors, problem)?;
    if candidates.is_empty() {
        return Err(DesignError::NoCandidateExperiments);
    }
    let mut ids = HashSet::with_capacity(candidates.len());
    let mut ranked = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if !ids.insert(candidate.id.as_str()) {
            return Err(DesignError::DuplicateExperimentId {
                experiment_id: candidate.id.clone(),
            });
        }
        ranked.push(RankedDecisionExperiment {
            experiment_id: candidate.id.clone(),
            score: evaluate_decision_value(priors, candidate, problem)?,
        });
    }
    ranked.sort_by(|left, right| {
        right
            .score
            .expected_value_per_cost
            .total_cmp(&left.score.expected_value_per_cost)
            .then_with(|| {
                right
                    .score
                    .expected_value_of_sample_information
                    .total_cmp(&left.score.expected_value_of_sample_information)
            })
            .then_with(|| left.experiment_id.cmp(&right.experiment_id))
    });
    Ok(ranked)
}

/// Compute decision-value regret against the best candidate in the declared pool.
///
/// This is an EVSI-per-cost opportunity gap. It is distinct from EIG-selection
/// regret and should not be presented as realized biological utility.
pub fn decision_selection_regret(
    priors: &[f64],
    candidates: &[CandidateExperiment],
    problem: &DecisionProblem,
    selected_experiment_id: &str,
) -> Result<DecisionSelectionRegret, DesignError> {
    let ranked = rank_candidates_by_decision_value(priors, candidates, problem)?;
    let best = ranked.first().ok_or(DesignError::NoCandidateExperiments)?;
    let selected = ranked
        .iter()
        .find(|candidate| candidate.experiment_id == selected_experiment_id)
        .ok_or_else(|| DesignError::SelectedExperimentNotFound {
            experiment_id: selected_experiment_id.to_owned(),
        })?;
    let best_utility = best.score.expected_value_per_cost;
    let selected_utility = selected.score.expected_value_per_cost;
    Ok(DecisionSelectionRegret {
        selected_experiment_id: selected.experiment_id.clone(),
        best_experiment_id: best.experiment_id.clone(),
        selected_expected_value_per_cost: selected_utility,
        best_expected_value_per_cost: best_utility,
        regret_per_cost: (best_utility - selected_utility).max(0.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binary_experiment(
        id: &str,
        cost: f64,
        response_given_h0: f64,
        response_given_h1: f64,
    ) -> CandidateExperiment {
        CandidateExperiment {
            id: id.to_owned(),
            cost,
            outcomes: vec!["response".to_owned(), "no_response".to_owned()],
            likelihoods_by_hypothesis: vec![
                vec![response_given_h0, 1.0 - response_given_h0],
                vec![response_given_h1, 1.0 - response_given_h1],
            ],
        }
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual:.15}, expected {expected:.15}, tolerance {tolerance}"
        );
    }

    #[test]
    fn eig_and_decision_value_are_distinct_experiment_objectives() {
        let priors = [0.25, 0.25, 0.25, 0.25];
        let problem = DecisionProblem {
            actions: vec!["choose_group_a".into(), "choose_group_b".into()],
            utilities_by_hypothesis: vec![
                vec![1.0, 0.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![0.0, 1.0],
            ],
        };

        // This experiment reveals which member of each utility-equivalent
        // pair is present, but never reveals which decision group is correct.
        let within_group = CandidateExperiment {
            id: "within-group-identity".into(),
            cost: 1.0,
            outcomes: vec!["member_0".into(), "member_1".into()],
            likelihoods_by_hypothesis: vec![
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
            ],
        };

        // This noisy experiment identifies the utility-relevant group with
        // 80% probability but has less total information about H.
        let group_signal = CandidateExperiment {
            id: "decision-relevant-group".into(),
            cost: 1.0,
            outcomes: vec!["group_a".into(), "group_b".into()],
            likelihoods_by_hypothesis: vec![
                vec![0.8, 0.2],
                vec![0.8, 0.2],
                vec![0.2, 0.8],
                vec![0.2, 0.8],
            ],
        };

        let eig_ranked = rank_candidates(
            &priors,
            &[within_group.clone(), group_signal.clone()],
        )
        .unwrap();
        assert_eq!(eig_ranked[0].experiment_id, "within-group-identity");
        close(eig_ranked[0].score.information_gain_bits, 1.0, 1.0e-12);

        let within_score = evaluate_decision_value(&priors, &within_group, &problem).unwrap();
        let group_score = evaluate_decision_value(&priors, &group_signal, &problem).unwrap();
        close(within_score.expected_value_of_sample_information, 0.0, 1.0e-12);
        close(group_score.expected_value_of_sample_information, 0.3, 1.0e-12);

        let decision_ranked = rank_candidates_by_decision_value(
            &priors,
            &[within_group.clone(), group_signal.clone()],
            &problem,
        )
        .unwrap();
        assert_eq!(decision_ranked[0].experiment_id, "decision-relevant-group");

        let regret = decision_selection_regret(
            &priors,
            &[within_group, group_signal],
            &problem,
            "within-group-identity",
        )
        .unwrap();
        assert_eq!(regret.best_experiment_id, "decision-relevant-group");
        close(regret.regret_per_cost, 0.3, 1.0e-12);
    }

    #[test]
    fn empty_and_duplicate_decision_action_labels_reject() {
        let candidate = binary_experiment("candidate", 1.0, 0.9, 0.1);
        let empty = DecisionProblem {
            actions: vec![" ".into()],
            utilities_by_hypothesis: vec![vec![1.0], vec![0.0]],
        };
        assert_eq!(
            evaluate_decision_value(&[0.5, 0.5], &candidate, &empty),
            Err(DesignError::EmptyActionLabel)
        );

        let duplicate = DecisionProblem {
            actions: vec!["same".into(), "same".into()],
            utilities_by_hypothesis: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        assert!(matches!(
            evaluate_decision_value(&[0.5, 0.5], &candidate, &duplicate),
            Err(DesignError::DuplicateActionLabel { .. })
        ));
    }

    #[test]
    fn cost_normalized_scores_reject_overflow() {
        let tiny_cost = f64::from_bits(1);
        let candidate = binary_experiment("tiny-cost", tiny_cost, 0.9, 0.1);
        assert_eq!(
            evaluate_candidate(&[0.5, 0.5], &candidate),
            Err(DesignError::NumericalInstability)
        );

        let problem = DecisionProblem {
            actions: vec!["a".into(), "b".into()],
            utilities_by_hypothesis: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        assert_eq!(
            evaluate_decision_value(&[0.5, 0.5], &candidate, &problem),
            Err(DesignError::NumericalInstability)
        );
    }

    #[test]
    fn invalid_decision_utility_shape_and_non_finite_values_reject() {
        let candidate = binary_experiment("candidate", 1.0, 0.9, 0.1);
        let bad_shape = DecisionProblem {
            actions: vec!["a".into(), "b".into()],
            utilities_by_hypothesis: vec![vec![1.0, 0.0]],
        };
        assert!(matches!(
            evaluate_decision_value(&[0.5, 0.5], &candidate, &bad_shape),
            Err(DesignError::DecisionUtilityRowCountMismatch { .. })
        ));

        let bad_value = DecisionProblem {
            actions: vec!["a".into()],
            utilities_by_hypothesis: vec![vec![f64::NAN], vec![0.0]],
        };
        assert!(matches!(
            evaluate_decision_value(&[0.5, 0.5], &candidate, &bad_value),
            Err(DesignError::InvalidDecisionUtility { .. })
        ));
    }

    fn scalar_parameter_design(id: &str, sensitivity: f64, noise_variance: f64, cost: f64) -> LinearGaussianDesign {
        LinearGaussianDesign {
            id: id.to_owned(),
            cost,
            sensitivities: vec![vec![sensitivity]],
            noise_covariance: vec![vec![noise_variance]],
        }
    }

    #[test]
    fn scalar_gaussian_design_matches_closed_form() {
        let prior = ParameterPrior {
            covariance: vec![vec![4.0]],
        };
        let design = scalar_parameter_design("scalar", 2.0, 1.0, 1.0);
        let score = evaluate_parameter_design(&prior, &design).unwrap();
        close(score.posterior_covariance_trace, 4.0 / 17.0, 1.0e-12);
        close(score.a_optimal_trace_reduction, 64.0 / 17.0, 1.0e-12);
        close(score.d_optimal_information_bits, 0.5 * 17.0_f64.log2(), 1.0e-12);
        close(score.prior_covariance_trace, 4.0, 1.0e-12);
    }

    #[test]
    fn d_and_a_optimal_designs_can_select_different_experiments() {
        let prior = ParameterPrior {
            covariance: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        let concentrated = LinearGaussianDesign {
            id: "concentrated".into(),
            cost: 1.0,
            sensitivities: vec![vec![10.0, 0.0]],
            noise_covariance: vec![vec![1.0]],
        };
        let balanced = LinearGaussianDesign {
            id: "balanced".into(),
            cost: 1.0,
            sensitivities: vec![vec![2.0, 0.0], vec![0.0, 2.0]],
            noise_covariance: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        let candidates = vec![concentrated, balanced];
        let d_ranked = rank_parameter_candidates(
            &prior,
            &candidates,
            ParameterObjective::DOptimal,
        ).unwrap();
        let a_ranked = rank_parameter_candidates(
            &prior,
            &candidates,
            ParameterObjective::AOptimal,
        ).unwrap();
        assert_eq!(d_ranked[0].experiment_id, "concentrated");
        assert_eq!(a_ranked[0].experiment_id, "balanced");
        close(d_ranked[0].score.d_optimal_information_bits, 0.5 * 101.0_f64.log2(), 1.0e-12);
        close(a_ranked[0].score.a_optimal_trace_reduction, 1.6, 1.0e-12);
    }

    #[test]
    fn zero_sensitivity_has_zero_parameter_information() {
        let prior = ParameterPrior {
            covariance: vec![vec![2.0, 0.5], vec![0.5, 1.0]],
        };
        let design = LinearGaussianDesign {
            id: "uninformative".into(),
            cost: 1.0,
            sensitivities: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
            noise_covariance: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        let score = evaluate_parameter_design(&prior, &design).unwrap();
        close(score.d_optimal_information_bits, 0.0, 1.0e-12);
        close(score.a_optimal_trace_reduction, 0.0, 1.0e-12);
    }

    #[test]
    fn correlated_observation_noise_is_accounted_for() {
        let prior = ParameterPrior {
            covariance: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        let design = LinearGaussianDesign {
            id: "correlated-noise".into(),
            cost: 1.0,
            sensitivities: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            noise_covariance: vec![vec![2.0, 0.5], vec![0.5, 1.0]],
        };
        let score = evaluate_parameter_design(&prior, &design).unwrap();
        assert!(score.d_optimal_information_bits > 0.0);
        assert!(score.a_optimal_trace_reduction > 0.0);
        assert!(score.posterior_covariance_trace < score.prior_covariance_trace);
    }

    #[test]
    fn parameter_design_rejects_non_positive_definite_prior_and_noise() {
        let singular_prior = ParameterPrior {
            covariance: vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        };
        let valid_noise = LinearGaussianDesign {
            id: "candidate".into(),
            cost: 1.0,
            sensitivities: vec![vec![1.0, 0.0]],
            noise_covariance: vec![vec![1.0]],
        };
        assert!(matches!(
            evaluate_parameter_design(&singular_prior, &valid_noise),
            Err(DesignError::MatrixNotPositiveDefinite { .. })
        ));

        let prior = ParameterPrior {
            covariance: vec![vec![1.0]],
        };
        let singular_noise = scalar_parameter_design("noise-singular", 1.0, 0.0, 1.0);
        assert!(matches!(
            evaluate_parameter_design(&prior, &singular_noise),
            Err(DesignError::MatrixNotPositiveDefinite { .. })
        ));
    }

    #[test]
    fn parameter_design_rejects_asymmetric_and_malformed_matrices() {
        let prior = ParameterPrior {
            covariance: vec![vec![1.0, 0.2], vec![0.1, 1.0]],
        };
        let design = LinearGaussianDesign {
            id: "candidate".into(),
            cost: 1.0,
            sensitivities: vec![vec![1.0, 0.0]],
            noise_covariance: vec![vec![1.0]],
        };
        assert!(matches!(
            evaluate_parameter_design(&prior, &design),
            Err(DesignError::NonSymmetricMatrix { .. })
        ));

        let prior = ParameterPrior {
            covariance: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        let malformed = LinearGaussianDesign {
            id: "bad-shape".into(),
            cost: 1.0,
            sensitivities: vec![vec![1.0]],
            noise_covariance: vec![vec![1.0]],
        };
        assert!(matches!(
            evaluate_parameter_design(&prior, &malformed),
            Err(DesignError::MatrixDimensionMismatch { .. })
        ));
    }

    #[test]
    fn parameter_design_cost_and_duplicate_ids_fail_closed() {
        let prior = ParameterPrior {
            covariance: vec![vec![1.0]],
        };
        let zero_cost = scalar_parameter_design("zero-cost", 1.0, 1.0, 0.0);
        assert!(matches!(
            evaluate_parameter_design(&prior, &zero_cost),
            Err(DesignError::InvalidExperimentCost { .. })
        ));
        let a = scalar_parameter_design("same", 1.0, 1.0, 1.0);
        let b = scalar_parameter_design("same", 2.0, 1.0, 1.0);
        assert!(matches!(
            rank_parameter_candidates(&prior, &[a, b], ParameterObjective::DOptimal),
            Err(DesignError::DuplicateExperimentId { .. })
        ));
    }

    #[test]
    fn parameter_selection_regret_is_zero_for_best_candidate() {
        let prior = ParameterPrior {
            covariance: vec![vec![1.0]],
        };
        let strong = scalar_parameter_design("strong", 2.0, 1.0, 1.0);
        let weak = scalar_parameter_design("weak", 1.0, 1.0, 1.0);
        let candidates = vec![strong, weak];
        let regret = parameter_selection_regret(
            &prior,
            &candidates,
            ParameterObjective::DOptimal,
            "strong",
        ).unwrap();
        assert_eq!(regret.best_experiment_id, "strong");
        close(regret.regret_per_cost, 0.0, 1.0e-12);
    }

    #[test]
    fn parameter_information_per_cost_overflow_is_rejected() {
        let prior = ParameterPrior {
            covariance: vec![vec![1.0]],
        };
        let design = scalar_parameter_design("tiny-cost", 1.0, 1.0, f64::from_bits(1));
        assert_eq!(
            evaluate_parameter_design(&prior, &design),
            Err(DesignError::NumericalInstability)
        );
    }

    #[test]
    fn uninformative_experiment_has_zero_eig() {
        let experiment = binary_experiment("uninformative", 1.0, 0.5, 0.5);
        let score = evaluate_candidate(&[0.5, 0.5], &experiment).unwrap();
        close(score.prior_entropy_bits, 1.0, 1.0e-12);
        close(score.expected_posterior_entropy_bits, 1.0, 1.0e-12);
        close(score.information_gain_bits, 0.0, 1.0e-12);
    }

    #[test]
    fn perfect_discrimination_yields_one_bit_for_equal_binary_prior() {
        let experiment = CandidateExperiment {
            id: "perfect".to_owned(),
            cost: 1.0,
            outcomes: vec!["left".to_owned(), "right".to_owned()],
            likelihoods_by_hypothesis: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        let score = evaluate_candidate(&[0.5, 0.5], &experiment).unwrap();
        close(score.information_gain_bits, 1.0, 1.0e-12);
        close(score.expected_posterior_entropy_bits, 0.0, 1.0e-12);
    }

    #[test]
    fn noisy_ninety_ten_discriminator_matches_analytic_eig() {
        let experiment = binary_experiment("noisy", 1.0, 0.9, 0.1);
        let score = evaluate_candidate(&[0.5, 0.5], &experiment).unwrap();
        close(score.information_gain_bits, 0.531_004_406_410_718_8, 1.0e-12);
    }

    #[test]
    fn perfect_discrimination_reduces_asymmetric_prior_to_zero_entropy() {
        let experiment = CandidateExperiment {
            id: "asymmetric-perfect".to_owned(),
            cost: 1.0,
            outcomes: vec!["h0".to_owned(), "h1".to_owned()],
            likelihoods_by_hypothesis: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        };
        let score = evaluate_candidate(&[0.8, 0.2], &experiment).unwrap();
        close(score.prior_entropy_bits, 0.721_928_094_887_362_3, 1.0e-12);
        close(score.information_gain_bits, score.prior_entropy_bits, 1.0e-12);
    }

    #[test]
    fn posterior_update_uses_bayes_rule() {
        let experiment = binary_experiment("bayes", 1.0, 0.8, 0.2);
        let posterior = posterior_for_observed_outcome(&[0.5, 0.5], &experiment, 0).unwrap();
        close(posterior[0], 0.8, 1.0e-12);
        close(posterior[1], 0.2, 1.0e-12);
    }

    #[test]
    fn cost_aware_ranking_can_prefer_less_raw_information() {
        let expensive = binary_experiment("expensive", 5.0, 0.9, 0.1);
        let cheap = binary_experiment("cheap", 1.0, 0.8, 0.2);
        let ranked = rank_candidates(&[0.5, 0.5], &[expensive, cheap]).unwrap();
        assert_eq!(ranked[0].experiment_id, "cheap");
        assert!(ranked[1].score.information_gain_bits > ranked[0].score.information_gain_bits);
        assert!(
            ranked[0].score.information_gain_bits_per_cost
                > ranked[1].score.information_gain_bits_per_cost
        );
    }

    #[test]
    fn candidate_order_does_not_change_ranked_tie_break() {
        let alpha = binary_experiment("alpha", 1.0, 0.9, 0.1);
        let zeta = binary_experiment("zeta", 1.0, 0.9, 0.1);
        let first = rank_candidates(&[0.5, 0.5], &[zeta.clone(), alpha.clone()]).unwrap();
        let second = rank_candidates(&[0.5, 0.5], &[alpha, zeta]).unwrap();
        assert_eq!(
            first.iter().map(|item| &item.experiment_id).collect::<Vec<_>>(),
            second.iter().map(|item| &item.experiment_id).collect::<Vec<_>>()
        );
        assert_eq!(first[0].experiment_id, "alpha");
    }

    #[test]
    fn selection_regret_is_zero_for_best_candidate_and_positive_otherwise() {
        let expensive = binary_experiment("expensive", 5.0, 0.9, 0.1);
        let cheap = binary_experiment("cheap", 1.0, 0.8, 0.2);
        let best = selection_regret(&[0.5, 0.5], &[expensive.clone(), cheap.clone()], "cheap")
            .unwrap();
        close(best.regret_bits_per_cost, 0.0, 1.0e-12);
        let regret = selection_regret(&[0.5, 0.5], &[expensive, cheap], "expensive").unwrap();
        assert!(regret.regret_bits_per_cost > 0.0);
        assert_eq!(regret.best_experiment_id, "cheap");
    }

    #[test]
    fn malformed_prior_fails_closed_without_renormalization() {
        let experiment = binary_experiment("candidate", 1.0, 0.9, 0.1);
        assert_eq!(
            evaluate_candidate(&[0.4, 0.4], &experiment),
            Err(DesignError::PriorMassNotOne { sum: 0.8 })
        );
    }

    #[test]
    fn malformed_likelihood_fails_closed_without_renormalization() {
        let mut experiment = binary_experiment("candidate", 1.0, 0.9, 0.1);
        experiment.likelihoods_by_hypothesis[0] = vec![0.9, 0.2];
        assert!(matches!(
            evaluate_candidate(&[0.5, 0.5], &experiment),
            Err(DesignError::LikelihoodMassNotOne { hypothesis_index: 0, .. })
        ));
    }

    #[test]
    fn non_finite_probability_and_non_positive_cost_are_rejected() {
        let mut experiment = binary_experiment("candidate", 1.0, 0.9, 0.1);
        experiment.likelihoods_by_hypothesis[0][0] = f64::NAN;
        assert!(matches!(
            evaluate_candidate(&[0.5, 0.5], &experiment),
            Err(DesignError::InvalidLikelihood {
                hypothesis_index: 0,
                outcome_index: 0,
                ..
            })
        ));
        let bad_cost = binary_experiment("bad-cost", 0.0, 0.9, 0.1);
        assert!(matches!(
            evaluate_candidate(&[0.5, 0.5], &bad_cost),
            Err(DesignError::InvalidExperimentCost { .. })
        ));
    }

    #[test]
    fn impossible_observed_outcome_has_no_arbitrary_posterior() {
        let experiment = CandidateExperiment {
            id: "impossible".to_owned(),
            cost: 1.0,
            outcomes: vec!["only".to_owned(), "never".to_owned()],
            likelihoods_by_hypothesis: vec![vec![1.0, 0.0], vec![1.0, 0.0]],
        };
        assert_eq!(
            posterior_for_observed_outcome(&[0.5, 0.5], &experiment, 1),
            Err(DesignError::ImpossibleObservedOutcome { outcome_index: 1 })
        );
    }

    #[test]
    fn duplicate_outcomes_and_unknown_selection_fail_closed() {
        let mut duplicate = binary_experiment("candidate", 1.0, 0.9, 0.1);
        duplicate.outcomes[1] = duplicate.outcomes[0].clone();
        assert!(matches!(
            evaluate_candidate(&[0.5, 0.5], &duplicate),
            Err(DesignError::DuplicateOutcomeLabel { .. })
        ));

        let valid = binary_experiment("valid", 1.0, 0.9, 0.1);
        assert!(matches!(
            selection_regret(&[0.5, 0.5], &[valid], "missing"),
            Err(DesignError::SelectedExperimentNotFound { .. })
        ));
    }

    #[test]
    fn duplicate_candidate_ids_are_rejected() {
        let first = binary_experiment("same", 1.0, 0.9, 0.1);
        let second = binary_experiment("same", 2.0, 0.8, 0.2);
        assert!(matches!(
            rank_candidates(&[0.5, 0.5], &[first, second]),
            Err(DesignError::DuplicateExperimentId { .. })
        ));
    }

    #[test]
    fn prior_entropy_bounds_information_gain() {
        let experiment = binary_experiment("bounded", 1.0, 1.0, 0.0);
        let score = evaluate_candidate(&[0.9, 0.1], &experiment).unwrap();
        assert!(score.information_gain_bits <= score.prior_entropy_bits);
        assert!(score.information_gain_bits >= 0.0);
    }
}
