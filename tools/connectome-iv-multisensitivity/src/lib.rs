//! Multi-instrument sample-normalized profiled Anderson–Rubin reference.
//!
//! Scope:
//! - one scalar endogenous treatment;
//! - r >= 1 scalar instruments observed jointly;
//! - an intercept and no other controls;
//! - fixed-design, conditional Gaussian, homoskedastic, independent rows;
//! - sample-normalized RMS support for the vector of direct instrument effects.
//!
//! Instruments are centered, and K = Z'Z is formed from the centered design.
//! For candidate structural effect b, gamma_hat(b) = K^-1 Z'(Y - bX).
//! The exact-exclusion AR statistic is
//!
//!     AR_exact(b) = gamma_hat' K gamma_hat / sigma_hat(b)^2.
//!
//! Under the sample-normalized ellipsoid gamma' K gamma / n <= g^2, the
//! general-r closed form is
//!
//!     pAR(b; g) = (sqrt(AR_exact(b)) - sqrt(n) * g / sigma_hat(b))_+^2.
//!
//! The candidate is not rejected when pAR / r <= the caller-supplied
//! F(r, n - 1 - r) critical value.
//!
//! This support depends on the realized instrument design Z. It is a
//! sample-dependent, design-conditional null, not automatically the same as
//! a fixed population bound on direct effects. The bound must be declared
//! before outcome unblinding.
//!
//! This is a synthetic mathematical reference. It does not audit
//! randomization/exclusion, estimate g, prove a biological mechanism, or
//! establish validity for clustered/heteroskedastic observations.

use core::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct MultiIvObservation {
    /// One value per instrument / independently assigned stimulation channel.
    pub instruments: Vec<f64>,
    pub treatment: f64,
    pub outcome: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultiIvDiagnostics {
    pub observation_count: usize,
    pub instrument_count: usize,
    /// K = Z_centered' Z_centered, in the caller's instrument coordinate system.
    pub instrument_gram: Vec<Vec<f64>>,
    /// First-stage coefficient vector from treatment on intercept and all instruments.
    pub first_stage_coefficients: Vec<f64>,
    /// Joint partial F statistic for all r first-stage coefficients.
    /// No universal pass/fail threshold is applied.
    pub first_stage_joint_f: f64,
    /// Reduced-form outcome-on-instruments coefficient vector.
    pub reduced_form_coefficients: Vec<f64>,
    /// Scalar 2SLS ratio; absent when the fitted first-stage quadratic form is zero.
    pub two_stage_least_squares: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MultiProfiledArResult {
    pub candidate_effect: f64,
    /// Candidate-compatible vector of direct instrument-to-outcome coefficients.
    pub implied_direct_effect_coefficients: Vec<f64>,
    /// sqrt(gamma_hat' K gamma_hat / n), the sample RMS of Z gamma_hat.
    pub implied_direct_effect_rms: f64,
    pub allowed_direct_effect_rms: f64,
    pub residual_sigma: f64,
    pub exact_exclusion_ar_statistic: f64,
    pub profiled_ar_statistic: f64,
    pub instrument_count: usize,
    pub exact_exclusion_ar_over_instruments: f64,
    pub profiled_ar_over_instruments: f64,
    pub f_critical_value: f64,
    pub accepted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MultiIvError {
    EmptyInstrumentSet,
    TooFewObservations {
        minimum: usize,
        actual: usize,
        instrument_count: usize,
    },
    InstrumentDimensionMismatch {
        observation_index: usize,
        expected: usize,
        actual: usize,
    },
    NonFiniteInput {
        observation_index: usize,
        field: &'static str,
        column: Option<usize>,
    },
    RankDeficientInstrumentSet {
        pivot_index: usize,
        pivot_value: f64,
    },
    InvalidCandidateEffect {
        value: f64,
    },
    InvalidDirectEffectBound {
        value: f64,
    },
    InvalidFCriticalValue {
        value: f64,
    },
    DegenerateFirstStageResidual,
    DegenerateOutcomeResidual {
        candidate_effect: f64,
    },
    NumericalOverflow {
        field: &'static str,
    },
}

impl fmt::Display for MultiIvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInstrumentSet => write!(f, "at least one instrument is required"),
            Self::TooFewObservations { minimum, actual, instrument_count } => write!(
                f,
                "at least {minimum} observations are required for {instrument_count} instruments; received {actual}"
            ),
            Self::InstrumentDimensionMismatch { observation_index, expected, actual } => write!(
                f,
                "observation {observation_index} has {actual} instruments; expected {expected}"
            ),
            Self::NonFiniteInput { observation_index, field, column } => {
                if let Some(column) = column {
                    write!(f, "observation {observation_index} has non-finite {field} at instrument column {column}")
                } else {
                    write!(f, "observation {observation_index} has non-finite {field}")
                }
            }
            Self::RankDeficientInstrumentSet { pivot_index, pivot_value } => write!(
                f,
                "instrument Gram matrix is rank deficient or numerically singular at Cholesky pivot {pivot_index}: {pivot_value}"
            ),
            Self::InvalidCandidateEffect { value } => write!(f, "candidate effect is invalid: {value}"),
            Self::InvalidDirectEffectBound { value } => {
                write!(f, "direct-effect RMS bound must be finite and non-negative: {value}")
            }
            Self::InvalidFCriticalValue { value } => {
                write!(f, "F critical value must be finite and positive: {value}")
            }
            Self::DegenerateFirstStageResidual => {
                write!(f, "first-stage residual variance is zero")
            }
            Self::DegenerateOutcomeResidual { candidate_effect } => write!(
                f, "outcome residual variance is zero for candidate effect {candidate_effect}"
            ),
            Self::NumericalOverflow { field } => write!(f, "numerical overflow while computing {field}"),
        }
    }
}

impl std::error::Error for MultiIvError {}

fn compensated_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for value in values {
        let adjusted = value - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    sum
}

fn compensated_dot(left: &[f64], right: &[f64]) -> f64 {
    compensated_sum(left.iter().zip(right).map(|(a, b)| a * b))
}

fn cholesky(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, MultiIvError> {
    let dimension = matrix.len();
    let mut lower = vec![vec![0.0; dimension]; dimension];

    for row in 0..dimension {
        for column in 0..=row {
            let correction = compensated_sum(
                (0..column).map(|index| lower[row][index] * lower[column][index]),
            );
            let value = matrix[row][column] - correction;

            if row == column {
                // Compare each residual pivot to its own original diagonal so
                // heterogeneous instrument units do not cause false rank failure.
                // This is a numerical conditioning guard, not a weak-IV test.
                let pivot_tolerance = 64.0 * f64::EPSILON * matrix[row][row];
                if !value.is_finite() || value <= pivot_tolerance {
                    return Err(MultiIvError::RankDeficientInstrumentSet {
                        pivot_index: row,
                        pivot_value: value,
                    });
                }
                lower[row][column] = value.sqrt();
            } else {
                let pivot = lower[column][column];
                let element = value / pivot;
                if !element.is_finite() {
                    return Err(MultiIvError::NumericalOverflow {
                        field: "Cholesky factor",
                    });
                }
                lower[row][column] = element;
            }
        }
    }
    Ok(lower)
}

fn solve_cholesky(
    lower: &[Vec<f64>],
    right_hand_side: &[f64],
) -> Result<Vec<f64>, MultiIvError> {
    let dimension = lower.len();
    let mut forward = vec![0.0; dimension];

    for row in 0..dimension {
        let correction = compensated_sum(
            (0..row).map(|column| lower[row][column] * forward[column]),
        );
        forward[row] = (right_hand_side[row] - correction) / lower[row][row];
    }

    let mut solution = vec![0.0; dimension];
    for row in (0..dimension).rev() {
        let correction = compensated_sum(
            ((row + 1)..dimension).map(|column| lower[column][row] * solution[column]),
        );
        solution[row] = (forward[row] - correction) / lower[row][row];
    }

    if solution.iter().any(|value| !value.is_finite()) {
        return Err(MultiIvError::NumericalOverflow {
            field: "positive-definite linear solve",
        });
    }
    Ok(solution)
}

fn matrix_vector_product(
    matrix: &[Vec<f64>],
    vector: &[f64],
) -> Result<Vec<f64>, MultiIvError> {
    let result: Vec<f64> = matrix
        .iter()
        .map(|row| compensated_dot(row, vector))
        .collect();
    if result.iter().any(|value| !value.is_finite()) {
        return Err(MultiIvError::NumericalOverflow {
            field: "matrix-vector product",
        });
    }
    Ok(result)
}

fn quadratic_form(
    vector: &[f64],
    matrix: &[Vec<f64>],
) -> Result<f64, MultiIvError> {
    let transformed = matrix_vector_product(matrix, vector)?;
    let value = compensated_dot(vector, &transformed);
    if !value.is_finite() {
        return Err(MultiIvError::NumericalOverflow {
            field: "quadratic form",
        });
    }
    Ok(value)
}

struct PreparedDesign {
    centered_instruments: Vec<Vec<f64>>,
    centered_treatment: Vec<f64>,
    centered_outcome: Vec<f64>,
    gram: Vec<Vec<f64>>,
    gram_cholesky: Vec<Vec<f64>>,
    diagnostics: MultiIvDiagnostics,
}

fn prepare_design(
    observations: &[MultiIvObservation],
) -> Result<PreparedDesign, MultiIvError> {
    if observations.is_empty() || observations[0].instruments.is_empty() {
        return Err(MultiIvError::EmptyInstrumentSet);
    }

    let instrument_count = observations[0].instruments.len();
    let observation_count = observations.len();
    let minimum = instrument_count + 2;
    if observation_count < minimum {
        return Err(MultiIvError::TooFewObservations {
            minimum,
            actual: observation_count,
            instrument_count,
        });
    }

    for (index, observation) in observations.iter().enumerate() {
        if observation.instruments.len() != instrument_count {
            return Err(MultiIvError::InstrumentDimensionMismatch {
                observation_index: index,
                expected: instrument_count,
                actual: observation.instruments.len(),
            });
        }
        for (column, value) in observation.instruments.iter().enumerate() {
            if !value.is_finite() {
                return Err(MultiIvError::NonFiniteInput {
                    observation_index: index,
                    field: "instrument",
                    column: Some(column),
                });
            }
        }
        for (field, value) in [
            ("treatment", observation.treatment),
            ("outcome", observation.outcome),
        ] {
            if !value.is_finite() {
                return Err(MultiIvError::NonFiniteInput {
                    observation_index: index,
                    field,
                    column: None,
                });
            }
        }
    }

    let instrument_means: Vec<f64> = (0..instrument_count)
        .map(|column| {
            compensated_sum(
                observations
                    .iter()
                    .map(|observation| observation.instruments[column]),
            ) / observation_count as f64
        })
        .collect();
    let treatment_mean = compensated_sum(
        observations.iter().map(|observation| observation.treatment),
    ) / observation_count as f64;
    let outcome_mean = compensated_sum(
        observations.iter().map(|observation| observation.outcome),
    ) / observation_count as f64;

    let centered_instruments: Vec<Vec<f64>> = observations
        .iter()
        .map(|observation| {
            (0..instrument_count)
                .map(|column| observation.instruments[column] - instrument_means[column])
                .collect()
        })
        .collect();
    let centered_treatment: Vec<f64> = observations
        .iter()
        .map(|observation| observation.treatment - treatment_mean)
        .collect();
    let centered_outcome: Vec<f64> = observations
        .iter()
        .map(|observation| observation.outcome - outcome_mean)
        .collect();

    let mut gram = vec![vec![0.0; instrument_count]; instrument_count];
    for row in 0..instrument_count {
        for column in row..instrument_count {
            let value = compensated_sum(
                centered_instruments
                    .iter()
                    .map(|instruments| instruments[row] * instruments[column]),
            );
            if !value.is_finite() {
                return Err(MultiIvError::NumericalOverflow {
                    field: "instrument Gram matrix",
                });
            }
            gram[row][column] = value;
            gram[column][row] = value;
        }
    }

    let gram_cholesky = cholesky(&gram)?;

    let first_stage_cross_product: Vec<f64> = (0..instrument_count)
        .map(|column| {
            compensated_sum(
                centered_instruments
                    .iter()
                    .zip(&centered_treatment)
                    .map(|(instruments, treatment)| instruments[column] * treatment),
            )
        })
        .collect();
    let reduced_form_cross_product: Vec<f64> = (0..instrument_count)
        .map(|column| {
            compensated_sum(
                centered_instruments
                    .iter()
                    .zip(&centered_outcome)
                    .map(|(instruments, outcome)| instruments[column] * outcome),
            )
        })
        .collect();

    let first_stage_coefficients =
        solve_cholesky(&gram_cholesky, &first_stage_cross_product)?;
    let reduced_form_coefficients =
        solve_cholesky(&gram_cholesky, &reduced_form_cross_product)?;

    let first_stage_residuals: Vec<f64> = centered_instruments
        .iter()
        .zip(&centered_treatment)
        .map(|(instruments, treatment)| {
            treatment - compensated_dot(instruments, &first_stage_coefficients)
        })
        .collect();
    let first_stage_sse = compensated_sum(
        first_stage_residuals.iter().map(|residual| residual * residual),
    );
    if !first_stage_sse.is_finite() {
        return Err(MultiIvError::NumericalOverflow {
            field: "first-stage residual sum of squares",
        });
    }
    if first_stage_sse <= 0.0 {
        return Err(MultiIvError::DegenerateFirstStageResidual);
    }

    let residual_degrees_of_freedom =
        (observation_count - 1 - instrument_count) as f64;
    let first_stage_variance = first_stage_sse / residual_degrees_of_freedom;
    let explained_treatment = quadratic_form(&first_stage_coefficients, &gram)?;
    let first_stage_joint_f =
        explained_treatment / instrument_count as f64 / first_stage_variance;

    let explained_treatment = explained_treatment.max(0.0);
    let two_stage_least_squares = if explained_treatment == 0.0 {
        None
    } else {
        let gram_delta = matrix_vector_product(&gram, &reduced_form_coefficients)?;
        let numerator = compensated_dot(&first_stage_coefficients, &gram_delta);
        let ratio = numerator / explained_treatment;
        if !ratio.is_finite() {
            return Err(MultiIvError::NumericalOverflow {
                field: "two-stage least-squares ratio",
            });
        }
        Some(ratio)
    };

    if !first_stage_joint_f.is_finite() || first_stage_joint_f < 0.0 {
        return Err(MultiIvError::NumericalOverflow {
            field: "joint first-stage F statistic",
        });
    }

    let diagnostics = MultiIvDiagnostics {
        observation_count,
        instrument_count,
        instrument_gram: gram.clone(),
        first_stage_coefficients,
        first_stage_joint_f,
        reduced_form_coefficients,
        two_stage_least_squares,
    };

    Ok(PreparedDesign {
        centered_instruments,
        centered_treatment,
        centered_outcome,
        gram,
        gram_cholesky,
        diagnostics,
    })
}

/// Compute the multi-instrument first-stage and reduced-form diagnostics.
/// The joint first-stage F statistic is reported continuously without a
/// universal threshold or a pass/fail label.
pub fn diagnose_multi_instrument(
    observations: &[MultiIvObservation],
) -> Result<MultiIvDiagnostics, MultiIvError> {
    Ok(prepare_design(observations)?.diagnostics)
}

/// Evaluate one candidate structural effect using the general-r closed form
/// under the sample-normalized ellipsoid gamma' K gamma / n <= g^2.
///
/// The returned test statistic is pAR/r. The caller supplies the upper-tail
/// critical value for F(r, n-1-r), fixed before outcomes are unblinded.
pub fn profile_multi_instrument_candidate(
    observations: &[MultiIvObservation],
    candidate_effect: f64,
    direct_effect_rms_bound: f64,
    f_critical_value: f64,
) -> Result<(MultiIvDiagnostics, MultiProfiledArResult), MultiIvError> {
    if !candidate_effect.is_finite() {
        return Err(MultiIvError::InvalidCandidateEffect {
            value: candidate_effect,
        });
    }
    if !direct_effect_rms_bound.is_finite() || direct_effect_rms_bound < 0.0 {
        return Err(MultiIvError::InvalidDirectEffectBound {
            value: direct_effect_rms_bound,
        });
    }
    if !f_critical_value.is_finite() || f_critical_value <= 0.0 {
        return Err(MultiIvError::InvalidFCriticalValue {
            value: f_critical_value,
        });
    }

    let prepared = prepare_design(observations)?;
    let diagnostics = prepared.diagnostics.clone();
    let observation_count = diagnostics.observation_count;
    let instrument_count = diagnostics.instrument_count;
    let residual_degrees_of_freedom =
        (observation_count - 1 - instrument_count) as f64;

    let candidate_residuals: Vec<f64> = prepared
        .centered_outcome
        .iter()
        .zip(&prepared.centered_treatment)
        .map(|(outcome, treatment)| outcome - candidate_effect * treatment)
        .collect();
    if candidate_residuals.iter().any(|value| !value.is_finite()) {
        return Err(MultiIvError::NumericalOverflow {
            field: "candidate residualization",
        });
    }

    let instrument_cross_product: Vec<f64> = (0..instrument_count)
        .map(|column| {
            compensated_sum(
                prepared
                    .centered_instruments
                    .iter()
                    .zip(&candidate_residuals)
                    .map(|(instruments, residual)| instruments[column] * residual),
            )
        })
        .collect();
    let implied_direct_effect_coefficients =
        solve_cholesky(&prepared.gram_cholesky, &instrument_cross_product)?;

    let fitted_residuals: Vec<f64> = prepared
        .centered_instruments
        .iter()
        .zip(&candidate_residuals)
        .map(|(instruments, residual)| {
            residual
                - compensated_dot(instruments, &implied_direct_effect_coefficients)
        })
        .collect();
    let residual_sse = compensated_sum(
        fitted_residuals.iter().map(|residual| residual * residual),
    );
    if !residual_sse.is_finite() {
        return Err(MultiIvError::NumericalOverflow {
            field: "candidate residual sum of squares",
        });
    }
    if residual_sse <= 0.0 {
        return Err(MultiIvError::DegenerateOutcomeResidual {
            candidate_effect,
        });
    }

    let residual_variance = residual_sse / residual_degrees_of_freedom;
    let residual_sigma = residual_variance.sqrt();
    let direct_effect_quadratic =
        quadratic_form(&implied_direct_effect_coefficients, &prepared.gram)?;
    if direct_effect_quadratic < 0.0 {
        return Err(MultiIvError::NumericalOverflow {
            field: "negative direct-effect quadratic form",
        });
    }

    let exact_exclusion_ar_statistic = direct_effect_quadratic / residual_variance;
    let exact_root = exact_exclusion_ar_statistic.sqrt();
    let permitted_root =
        (observation_count as f64).sqrt() * direct_effect_rms_bound / residual_sigma;
    let excess_root = (exact_root - permitted_root).max(0.0);
    let profiled_ar_statistic = excess_root * excess_root;

    let implied_direct_effect_rms =
        (direct_effect_quadratic / observation_count as f64).sqrt();
    let exact_exclusion_ar_over_instruments =
        exact_exclusion_ar_statistic / instrument_count as f64;
    let profiled_ar_over_instruments =
        profiled_ar_statistic / instrument_count as f64;

    for (field, value) in [
        ("residual sigma", residual_sigma),
        ("exact AR statistic", exact_exclusion_ar_statistic),
        ("profiled AR statistic", profiled_ar_statistic),
        ("direct effect RMS", implied_direct_effect_rms),
        ("exact AR / r", exact_exclusion_ar_over_instruments),
        ("profiled AR / r", profiled_ar_over_instruments),
    ] {
        if !value.is_finite() {
            return Err(MultiIvError::NumericalOverflow { field });
        }
    }

    let result = MultiProfiledArResult {
        candidate_effect,
        implied_direct_effect_coefficients,
        implied_direct_effect_rms,
        allowed_direct_effect_rms: direct_effect_rms_bound,
        residual_sigma,
        exact_exclusion_ar_statistic,
        profiled_ar_statistic,
        instrument_count,
        exact_exclusion_ar_over_instruments,
        profiled_ar_over_instruments,
        f_critical_value,
        accepted: profiled_ar_over_instruments <= f_critical_value,
    };

    Ok((diagnostics, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    const F95_DF2_DF5: f64 = 5.786_135_043_349_963;

    fn two_instrument_design(
        first_stage_one: f64,
        first_stage_two: f64,
        direct_one: f64,
        direct_two: f64,
    ) -> Vec<MultiIvObservation> {
        let z_one = [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0];
        let z_two = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
        let residual_pattern = [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0];

        (0..8)
            .map(|index| {
                let instruments = vec![z_one[index], z_two[index]];
                let treatment = first_stage_one * z_one[index]
                    + first_stage_two * z_two[index]
                    + 0.1 * residual_pattern[index];
                let outcome = 3.0 * treatment
                    + direct_one * z_one[index]
                    + direct_two * z_two[index]
                    + 0.2 * residual_pattern[index];
                MultiIvObservation {
                    instruments,
                    treatment,
                    outcome,
                }
            })
            .collect()
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual:.15}, expected {expected:.15}, tolerance {tolerance}"
        );
    }

    fn one_instrument_design() -> Vec<MultiIvObservation> {
        let first_stage_noise = [
            0.15, -0.15, 0.12, -0.12, 0.18,
            -0.18, 0.10, -0.10, 0.14, -0.14,
        ];
        let outcome_noise = [
            0.10, -0.10, 0.08, -0.08, 0.12,
            -0.12, 0.06, -0.06, 0.09, -0.09,
        ];
        let mut observations = Vec::with_capacity(20);
        for index in 0..10 {
            let magnitude = 0.05 + 0.1 * index as f64;
            for sign in [-1.0, 1.0] {
                let instrument = sign * magnitude;
                let treatment = 2.0 * instrument + first_stage_noise[index];
                let outcome = 3.0 * treatment + 0.6 * instrument + outcome_noise[index];
                observations.push(MultiIvObservation {
                    instruments: vec![instrument],
                    treatment,
                    outcome,
                });
            }
        }
        observations
    }

    #[test]
    fn two_instrument_gram_and_first_stage_diagnostics_have_known_answers() {
        let observations = two_instrument_design(2.0, 0.5, 0.6, -0.4);
        let diagnostics = diagnose_multi_instrument(&observations).unwrap();
        assert_eq!(diagnostics.instrument_count, 2);
        close(diagnostics.instrument_gram[0][0], 8.0, 1.0e-12);
        close(diagnostics.instrument_gram[1][1], 8.0, 1.0e-12);
        close(diagnostics.instrument_gram[0][1], 0.0, 1.0e-12);
        close(diagnostics.first_stage_coefficients[0], 2.0, 1.0e-12);
        close(diagnostics.first_stage_coefficients[1], 0.5, 1.0e-12);
        close(diagnostics.first_stage_joint_f, 1062.5, 1.0e-9);
        close(diagnostics.reduced_form_coefficients[0], 6.6, 1.0e-12);
        close(diagnostics.reduced_form_coefficients[1], 1.1, 1.0e-12);
        close(diagnostics.two_stage_least_squares.unwrap(), 3.0 + 8.0 / 34.0, 1.0e-12);
    }

    #[test]
    fn broad_rms_support_covers_the_vector_direct_effect() {
        let observations = two_instrument_design(2.0, 0.5, 0.6, -0.4);
        let (diagnostics, result) =
            profile_multi_instrument_candidate(&observations, 3.0, 0.8, F95_DF2_DF5)
                .unwrap();
        close(diagnostics.first_stage_joint_f, 1062.5, 1.0e-9);
        close(result.implied_direct_effect_coefficients[0], 0.6, 1.0e-12);
        close(result.implied_direct_effect_coefficients[1], -0.4, 1.0e-12);
        close(result.implied_direct_effect_rms, 0.721_110_255_092_797_9, 1.0e-12);
        close(result.profiled_ar_statistic, 0.0, 1.0e-12);
        assert!(result.accepted);
    }

    #[test]
    fn tight_vector_direct_effect_support_rejects_candidate() {
        let observations = two_instrument_design(2.0, 0.5, 0.6, -0.4);
        let (_, result) =
            profile_multi_instrument_candidate(&observations, 3.0, 0.3, F95_DF2_DF5)
                .unwrap();
        close(result.exact_exclusion_ar_statistic, 65.0, 1.0e-10);
        close(result.profiled_ar_statistic, 22.166_730_868_040_155, 1.0e-9);
        close(result.profiled_ar_over_instruments, 11.083_365_434_020_077, 1.0e-9);
        assert!(!result.accepted);
    }

    #[test]
    fn intermediate_vector_bound_reduces_statistic_and_can_retain_candidate() {
        let observations = two_instrument_design(2.0, 0.5, 0.6, -0.4);
        let (_, tight) =
            profile_multi_instrument_candidate(&observations, 3.0, 0.3, F95_DF2_DF5)
                .unwrap();
        let (_, intermediate) =
            profile_multi_instrument_candidate(&observations, 3.0, 0.5, F95_DF2_DF5)
                .unwrap();
        assert!(intermediate.profiled_ar_statistic < tight.profiled_ar_statistic);
        assert!(intermediate.accepted);
    }

    #[test]
    fn zero_radius_reduces_to_exact_exclusion_ar() {
        let observations = two_instrument_design(2.0, 0.5, 0.6, -0.4);
        let (_, result) =
            profile_multi_instrument_candidate(&observations, 3.0, 0.0, F95_DF2_DF5)
                .unwrap();
        close(
            result.profiled_ar_statistic,
            result.exact_exclusion_ar_statistic,
            1.0e-12,
        );
        close(
            result.profiled_ar_over_instruments,
            result.exact_exclusion_ar_over_instruments,
            1.0e-12,
        );
    }

    #[test]
    fn one_instrument_case_matches_scalar_p_ar_known_answer() {
        let observations = one_instrument_design();
        let diagnostics = diagnose_multi_instrument(&observations).unwrap();
        close(diagnostics.first_stage_joint_f, 1210.313_447_927_199_4, 1.0e-8);
        close(diagnostics.two_stage_least_squares.unwrap(), 3.3, 1.0e-12);

        let (_, result) =
            profile_multi_instrument_candidate(&observations, 3.0, 0.31, 4.413_873_419_170_566)
                .unwrap();
        close(result.implied_direct_effect_rms, 0.345_976_877_840_123_7, 1.0e-12);
        close(result.profiled_ar_statistic, 2.740_946_271_084_397, 1.0e-8);
        close(result.profiled_ar_over_instruments, result.profiled_ar_statistic, 1.0e-12);
    }

    #[test]
    fn weak_multi_instrument_first_stage_is_reported_without_threshold() {
        let observations = two_instrument_design(0.05, 0.02, 0.6, -0.4);
        let diagnostics = diagnose_multi_instrument(&observations).unwrap();
        assert!(diagnostics.first_stage_joint_f < 1.0);
        assert!(diagnostics.first_stage_joint_f >= 0.0);
    }

    #[test]
    fn zero_relevance_with_covering_bound_leaves_distant_candidates_unrejected() {
        let observations = two_instrument_design(0.0, 0.0, 0.6, -0.4);
        let diagnostics = diagnose_multi_instrument(&observations).unwrap();
        close(diagnostics.first_stage_joint_f, 0.0, 0.0);
        assert_eq!(diagnostics.two_stage_least_squares, None);

        let (_, at_three) =
            profile_multi_instrument_candidate(&observations, 3.0, 0.8, F95_DF2_DF5)
                .unwrap();
        let (_, at_ten) =
            profile_multi_instrument_candidate(&observations, 10.0, 0.8, F95_DF2_DF5)
                .unwrap();
        assert!(at_three.accepted);
        assert!(at_ten.accepted);
        close(
            at_three.implied_direct_effect_rms,
            at_ten.implied_direct_effect_rms,
            1.0e-12,
        );
    }

    #[test]
    fn instrument_scaling_preserves_sample_rms_p_ar_statistic() {
        let observations = two_instrument_design(2.0, 0.5, 0.6, -0.4);
        let scaled: Vec<MultiIvObservation> = observations
            .iter()
            .map(|item| MultiIvObservation {
                instruments: vec![10.0 * item.instruments[0], 0.1 * item.instruments[1]],
                treatment: item.treatment,
                outcome: item.outcome,
            })
            .collect();
        let (_, original) =
            profile_multi_instrument_candidate(&observations, 3.0, 0.8, F95_DF2_DF5)
                .unwrap();
        let (_, transformed) =
            profile_multi_instrument_candidate(&scaled, 3.0, 0.8, F95_DF2_DF5)
                .unwrap();
        close(
            original.profiled_ar_statistic,
            transformed.profiled_ar_statistic,
            1.0e-9,
        );
        close(
            original.implied_direct_effect_rms,
            transformed.implied_direct_effect_rms,
            1.0e-12,
        );
    }

    #[test]
    fn sample_order_does_not_change_multi_instrument_result() {
        let observations = two_instrument_design(2.0, 0.5, 0.6, -0.4);
        let reversed: Vec<MultiIvObservation> = observations.iter().cloned().rev().collect();
        let (_, original) =
            profile_multi_instrument_candidate(&observations, 3.0, 0.5, F95_DF2_DF5)
                .unwrap();
        let (_, permuted) =
            profile_multi_instrument_candidate(&reversed, 3.0, 0.5, F95_DF2_DF5)
                .unwrap();
        close(
            original.profiled_ar_statistic,
            permuted.profiled_ar_statistic,
            1.0e-12,
        );
    }

    #[test]
    fn collinear_instruments_fail_closed() {
        let observations: Vec<MultiIvObservation> = (0..8)
            .map(|index| {
                let instrument = index as f64 - 3.5;
                MultiIvObservation {
                    instruments: vec![instrument, 2.0 * instrument],
                    treatment: 3.0 * instrument + (index % 2) as f64,
                    outcome: 5.0 * instrument + (index % 2) as f64,
                }
            })
            .collect();
        assert!(matches!(
            diagnose_multi_instrument(&observations),
            Err(MultiIvError::RankDeficientInstrumentSet { .. })
        ));
    }

    #[test]
    fn mismatched_instrument_row_width_fails_closed() {
        let mut observations = two_instrument_design(2.0, 0.5, 0.6, -0.4);
        observations[0].instruments.pop();
        assert!(matches!(
            diagnose_multi_instrument(&observations),
            Err(MultiIvError::InstrumentDimensionMismatch { .. })
        ));
    }

    #[test]
    fn empty_instrument_set_fails_closed() {
        let observations = vec![
            MultiIvObservation { instruments: vec![], treatment: 0.0, outcome: 0.0 },
            MultiIvObservation { instruments: vec![], treatment: 1.0, outcome: 1.0 },
            MultiIvObservation { instruments: vec![], treatment: 2.0, outcome: 2.0 },
        ];
        assert_eq!(
            diagnose_multi_instrument(&observations),
            Err(MultiIvError::EmptyInstrumentSet)
        );
    }

    #[test]
    fn too_few_observations_for_instrument_count_fails_closed() {
        let observations = two_instrument_design(2.0, 0.5, 0.6, -0.4);
        assert!(matches!(
            diagnose_multi_instrument(&observations[..3]),
            Err(MultiIvError::TooFewObservations { instrument_count: 2, .. })
        ));
    }

    #[test]
    fn non_finite_instrument_treatment_or_outcome_fails_closed() {
        let mut observations = two_instrument_design(2.0, 0.5, 0.6, -0.4);
        observations[1].instruments[1] = f64::NAN;
        assert!(matches!(
            diagnose_multi_instrument(&observations),
            Err(MultiIvError::NonFiniteInput { field: "instrument", column: Some(1), .. })
        ));

        let mut observations = two_instrument_design(2.0, 0.5, 0.6, -0.4);
        observations[2].treatment = f64::INFINITY;
        assert!(matches!(
            diagnose_multi_instrument(&observations),
            Err(MultiIvError::NonFiniteInput { field: "treatment", .. })
        ));

        let mut observations = two_instrument_design(2.0, 0.5, 0.6, -0.4);
        observations[3].outcome = f64::NAN;
        assert!(matches!(
            diagnose_multi_instrument(&observations),
            Err(MultiIvError::NonFiniteInput { field: "outcome", .. })
        ));
    }

    #[test]
    fn invalid_bound_and_critical_value_fail_closed() {
        let observations = two_instrument_design(2.0, 0.5, 0.6, -0.4);
        assert!(matches!(
            profile_multi_instrument_candidate(&observations, 3.0, -0.1, F95_DF2_DF5),
            Err(MultiIvError::InvalidDirectEffectBound { .. })
        ));
        assert!(matches!(
            profile_multi_instrument_candidate(&observations, 3.0, 0.1, 0.0),
            Err(MultiIvError::InvalidFCriticalValue { .. })
        ));
    }

    #[test]
    fn invalid_candidate_effect_fails_closed() {
        let observations = two_instrument_design(2.0, 0.5, 0.6, -0.4);
        assert!(matches!(
            profile_multi_instrument_candidate(&observations, f64::NAN, 0.8, F95_DF2_DF5),
            Err(MultiIvError::InvalidCandidateEffect { .. })
        ));
    }

    #[test]
    fn degenerate_outcome_residual_fails_closed() {
        let mut observations = two_instrument_design(2.0, 0.5, 0.6, -0.4);
        for item in &mut observations {
            item.outcome = 3.0 * item.treatment
                + 0.6 * item.instruments[0]
                - 0.4 * item.instruments[1];
        }
        assert!(matches!(
            profile_multi_instrument_candidate(&observations, 3.0, 0.0, F95_DF2_DF5),
            Err(MultiIvError::DegenerateOutcomeResidual { .. })
        ));
    }

    #[test]
    fn p_ar_statistic_never_exceeds_exact_exclusion_statistic() {
        let observations = two_instrument_design(2.0, 0.5, 0.6, -0.4);
        let (_, exact) =
            profile_multi_instrument_candidate(&observations, 3.0, 0.0, F95_DF2_DF5)
                .unwrap();
        let (_, bounded) =
            profile_multi_instrument_candidate(&observations, 3.0, 0.3, F95_DF2_DF5)
                .unwrap();
        assert!(bounded.profiled_ar_statistic <= exact.profiled_ar_statistic);
    }

    #[test]
    fn heterogeneous_instrument_units_preserve_the_p_ar_statistic() {
        let observations = two_instrument_design(2.0, 0.5, 0.6, -0.4);
        let rescaled: Vec<MultiIvObservation> = observations
            .iter()
            .map(|item| MultiIvObservation {
                instruments: vec![1.0e6 * item.instruments[0], 1.0e-6 * item.instruments[1]],
                treatment: item.treatment,
                outcome: item.outcome,
            })
            .collect();
        let (_, original) =
            profile_multi_instrument_candidate(&observations, 3.0, 0.5, F95_DF2_DF5)
                .unwrap();
        let (_, changed_units) =
            profile_multi_instrument_candidate(&rescaled, 3.0, 0.5, F95_DF2_DF5)
                .unwrap();
        close(
            original.profiled_ar_statistic,
            changed_units.profiled_ar_statistic,
            1.0e-8,
        );
        close(
            original.implied_direct_effect_rms,
            changed_units.implied_direct_effect_rms,
            1.0e-12,
        );
    }

    #[test]
    fn permuting_instrument_columns_preserves_the_test() {
        let observations = two_instrument_design(2.0, 0.5, 0.6, -0.4);
        let permuted: Vec<MultiIvObservation> = observations
            .iter()
            .map(|item| MultiIvObservation {
                instruments: vec![item.instruments[1], item.instruments[0]],
                treatment: item.treatment,
                outcome: item.outcome,
            })
            .collect();
        let (_, original) =
            profile_multi_instrument_candidate(&observations, 3.0, 0.5, F95_DF2_DF5)
                .unwrap();
        let (_, changed_order) =
            profile_multi_instrument_candidate(&permuted, 3.0, 0.5, F95_DF2_DF5)
                .unwrap();
        close(
            original.profiled_ar_statistic,
            changed_order.profiled_ar_statistic,
            1.0e-12,
        );
        close(
            original.implied_direct_effect_rms,
            changed_order.implied_direct_effect_rms,
            1.0e-12,
        );
    }

}
