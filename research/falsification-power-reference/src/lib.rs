//! A small, dependency-free power oracle for declared Gaussian model discrepancies.
//!
//! This crate is intentionally narrower than a model-adequacy test or experiment
//! selector. It computes alternative-specific best-case power for a fixed
//! observation design when the discrepancy direction and iid Gaussian noise
//! variance are known to the evaluator. See the repository research note for
//! assumptions and claim boundaries.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub mod gaussian_terminal_induction;
pub mod mechanism_separation;
pub mod sequential;

/// Errors for invalid designs and power-envelope declarations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PowerError {
    EmptyObservationDesign,
    EmptyAlternativeSet,
    EmptyAlternativeId,
    DuplicateAlternativeId(String),
    MismatchedObservationCount { expected: usize, actual: usize },
    NonFiniteResidual { index: usize },
    InvalidSigma,
    InvalidAlpha,
    InvalidTargetPower,
}

impl fmt::Display for PowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyObservationDesign => write!(f, "observation design must not be empty"),
            Self::EmptyAlternativeSet => write!(f, "declared alternative set must not be empty"),
            Self::EmptyAlternativeId => write!(f, "alternative identifier must not be empty"),
            Self::DuplicateAlternativeId(id) => write!(f, "duplicate alternative identifier: {id}"),
            Self::MismatchedObservationCount { expected, actual } => {
                write!(
                    f,
                    "alternative has {actual} observations; expected {expected}"
                )
            }
            Self::NonFiniteResidual { index } => {
                write!(f, "expected residual at index {index} is not finite")
            }
            Self::InvalidSigma => write!(f, "noise sigma must be finite and strictly positive"),
            Self::InvalidAlpha => write!(f, "alpha must be finite and in (1e-12, 1 - 1e-12)"),
            Self::InvalidTargetPower => write!(
                f,
                "target power must be finite and strictly between alpha and 1"
            ),
        }
    }
}

impl Error for PowerError {}

/// One evaluator-declared alternative relative to the working model.
///
/// expected_residuals[i] is alternative prediction minus working-model
/// prediction at the i-th observation in the fixed design. These values are
/// evaluator-side data and must not be exposed to the research agent.
#[derive(Debug, Clone, PartialEq)]
pub struct DeclaredAlternative {
    pub id: String,
    pub expected_residuals: Vec<f64>,
}

impl DeclaredAlternative {
    pub fn new(id: impl Into<String>, expected_residuals: Vec<f64>) -> Self {
        Self {
            id: id.into(),
            expected_residuals,
        }
    }
}

/// Best-case power for one known discrepancy direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerResult {
    pub observations: usize,
    pub noncentrality: f64,
    pub critical_value: f64,
    pub power: f64,
}

/// Whether one alternative reaches the predeclared power target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerDisposition {
    TargetMet,
    Underpowered,
}

/// One row in a power-envelope report.
#[derive(Debug, Clone, PartialEq)]
pub struct AlternativePower {
    pub id: String,
    pub result: PowerResult,
    pub disposition: PowerDisposition,
}

/// Alternative-specific report for one fixed observation design.
///
/// target_met is true only when every declared alternative reaches the target.
/// The minimum is never hidden by averaging across alternatives.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerEnvelopeReport {
    pub alpha: f64,
    pub target_power: f64,
    pub minimum_power: f64,
    pub worst_case_alternative: String,
    pub target_met: bool,
    pub underpowered_alternatives: Vec<String>,
    pub alternatives: Vec<AlternativePower>,
}

/// Compute best-case power for a fixed, evaluator-declared Gaussian shift.
///
/// Assumptions:
/// - observations are independent with common, known standard deviation sigma;
/// - the null mean is the working-model prediction;
/// - the full discrepancy direction is known to the evaluator;
/// - the test is the one-sided likelihood-ratio test aligned with that known direction;
/// - the design, alpha and stopping rule are fixed (no sequential peeking).
///
/// This is an oracle diagnostic, not an operational adequacy test. Estimating
/// nuisance parameters, unknown/correlated noise, adaptive stopping,
/// multiplicity and imperfect discrepancy specification can lower actual
/// power and require separate qualification.
pub fn known_direction_power(
    expected_residuals: &[f64],
    sigma: f64,
    alpha: f64,
) -> Result<PowerResult, PowerError> {
    validate_sigma(sigma)?;
    validate_alpha(alpha)?;
    if expected_residuals.is_empty() {
        return Err(PowerError::EmptyObservationDesign);
    }

    let mut squared_signal_to_noise = 0.0;
    for (index, residual) in expected_residuals.iter().copied().enumerate() {
        if !residual.is_finite() {
            return Err(PowerError::NonFiniteResidual { index });
        }
        let standardized = residual / sigma;
        squared_signal_to_noise += standardized * standardized;
    }

    let noncentrality = squared_signal_to_noise.sqrt();
    let critical_value = normal_quantile(1.0 - alpha);
    let power = one_sided_power(noncentrality, critical_value);

    Ok(PowerResult {
        observations: expected_residuals.len(),
        noncentrality,
        critical_value,
        power,
    })
}

/// Evaluate every declared alternative under the same fixed design size.
///
/// All alternatives must provide one expected residual per observation in the
/// same design order. This checks vector lengths only: callers must bind the
/// same observation IDs, units, sensor basis and acquisition-cost accounting.
pub fn evaluate_power_envelope(
    alternatives: &[DeclaredAlternative],
    sigma: f64,
    alpha: f64,
    target_power: f64,
) -> Result<PowerEnvelopeReport, PowerError> {
    validate_sigma(sigma)?;
    validate_alpha(alpha)?;
    validate_target_power(alpha, target_power)?;
    let Some(first) = alternatives.first() else {
        return Err(PowerError::EmptyAlternativeSet);
    };
    let expected_observations = first.expected_residuals.len();
    if expected_observations == 0 {
        return Err(PowerError::EmptyObservationDesign);
    }

    let mut seen = BTreeSet::new();
    let mut rows = Vec::with_capacity(alternatives.len());
    for alternative in alternatives {
        if alternative.id.trim().is_empty() {
            return Err(PowerError::EmptyAlternativeId);
        }
        if !seen.insert(alternative.id.clone()) {
            return Err(PowerError::DuplicateAlternativeId(alternative.id.clone()));
        }
        if alternative.expected_residuals.len() != expected_observations {
            return Err(PowerError::MismatchedObservationCount {
                expected: expected_observations,
                actual: alternative.expected_residuals.len(),
            });
        }
        let result = known_direction_power(&alternative.expected_residuals, sigma, alpha)?;
        let disposition = if result.power >= target_power {
            PowerDisposition::TargetMet
        } else {
            PowerDisposition::Underpowered
        };
        rows.push(AlternativePower {
            id: alternative.id.clone(),
            result,
            disposition,
        });
    }

    let worst = rows
        .iter()
        .min_by(|left, right| left.result.power.total_cmp(&right.result.power))
        .expect("the non-empty alternative set was checked above");
    let minimum_power = worst.result.power;
    let worst_case_alternative = worst.id.clone();
    let underpowered_alternatives = rows
        .iter()
        .filter(|row| row.disposition == PowerDisposition::Underpowered)
        .map(|row| row.id.clone())
        .collect::<Vec<_>>();

    Ok(PowerEnvelopeReport {
        alpha,
        target_power,
        minimum_power,
        worst_case_alternative,
        target_met: underpowered_alternatives.is_empty(),
        underpowered_alternatives,
        alternatives: rows,
    })
}

/// Return the minimum number of iid replicates needed for a constant,
/// evaluator-known mean discrepancy to reach target_power.
///
/// None means no finite replicate count can reach the target (for example,
/// a zero discrepancy). The calculation uses the same known-variance,
/// one-sided Gaussian oracle as known_direction_power.
pub fn minimum_replicates_for_power(
    expected_residual: f64,
    sigma: f64,
    alpha: f64,
    target_power: f64,
) -> Result<Option<u64>, PowerError> {
    validate_sigma(sigma)?;
    validate_alpha(alpha)?;
    validate_target_power(alpha, target_power)?;
    if !expected_residual.is_finite() {
        return Err(PowerError::NonFiniteResidual { index: 0 });
    }
    if expected_residual == 0.0 {
        return Ok(None);
    }

    let effect_size = expected_residual.abs() / sigma;
    if !effect_size.is_finite() {
        return Ok(Some(1));
    }

    let critical_value = normal_quantile(1.0 - alpha);
    let power_at = |n: u64| one_sided_power(effect_size * (n as f64).sqrt(), critical_value);

    let mut high = 1_u64;
    while power_at(high) < target_power {
        if high > u64::MAX / 2 {
            return Ok(None);
        }
        high *= 2;
    }

    let mut low = 0_u64;
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        if power_at(middle) >= target_power {
            high = middle;
        } else {
            low = middle;
        }
    }
    Ok(Some(high))
}

fn validate_sigma(sigma: f64) -> Result<(), PowerError> {
    if !sigma.is_finite() || sigma <= 0.0 {
        Err(PowerError::InvalidSigma)
    } else {
        Ok(())
    }
}

fn validate_alpha(alpha: f64) -> Result<(), PowerError> {
    if !alpha.is_finite() || alpha <= 1.0e-12 || alpha >= 1.0 - 1.0e-12 {
        Err(PowerError::InvalidAlpha)
    } else {
        Ok(())
    }
}

fn validate_target_power(alpha: f64, target: f64) -> Result<(), PowerError> {
    if !target.is_finite() || target <= alpha || target >= 1.0 {
        Err(PowerError::InvalidTargetPower)
    } else {
        Ok(())
    }
}

fn one_sided_power(noncentrality: f64, critical_value: f64) -> f64 {
    if noncentrality.is_infinite() {
        return 1.0;
    }
    (1.0 - normal_cdf(critical_value - noncentrality)).clamp(0.0, 1.0)
}

/// Standard normal CDF using the Abramowitz-Stegun 7.1.26 approximation.
fn normal_cdf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.5;
    }
    if x == f64::INFINITY {
        return 1.0;
    }
    if x == f64::NEG_INFINITY {
        return 0.0;
    }

    let absolute = x.abs();
    let t = 1.0 / (1.0 + 0.231_641_9 * absolute);
    let density = 0.398_942_280_401_432_7 * (-0.5 * absolute * absolute).exp();
    let polynomial = t
        * (0.319_381_530
            + t * (-0.356_563_782
                + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let tail = density * polynomial;
    if x > 0.0 { 1.0 - tail } else { tail }
}

/// Acklam's rational approximation to the standard normal quantile.
fn normal_quantile(probability: f64) -> f64 {
    const A: [f64; 6] = [
        -39.696_830_286_653_76,
        220.946_098_424_520_5,
        -275.928_510_446_968_7,
        138.357_751_867_269,
        -30.664_798_066_147_16,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -54.476_098_798_224_06,
        161.585_836_858_040_9,
        -155.698_979_859_886_6,
        66.801_311_887_719_72,
        -13.280_681_552_885_72,
    ];
    const C: [f64; 6] = [
        -0.007_784_894_002_430_293,
        -0.322_396_458_041_136_5,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        0.007_784_695_709_041_462,
        0.322_467_129_070_039_8,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const LOW: f64 = 0.024_25;
    const HIGH: f64 = 1.0 - LOW;

    if probability < LOW {
        let q = (-2.0 * probability.ln()).sqrt();
        return (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }
    if probability <= HIGH {
        let q = probability - 0.5;
        let r = q * q;
        return (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0);
    }

    let q = (-2.0 * (1.0 - probability).ln()).sqrt();
    -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
        / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nominal_null_power_matches_alpha() {
        let result = known_direction_power(&[0.0, 0.0, 0.0], 0.15, 0.05).unwrap();
        assert!((result.power - 0.05).abs() < 2.0e-7);
    }

    #[test]
    fn critical_value_matches_one_sided_five_percent_test() {
        let result = known_direction_power(&[0.1], 0.15, 0.05).unwrap();
        assert!((result.critical_value - 1.644_853_626_95).abs() < 1.0e-7);
    }

    #[test]
    fn known_direction_power_matches_normal_reference_value() {
        let result = known_direction_power(&[0.24, 0.24, 0.24], 0.15, 0.05).unwrap();
        assert!((result.power - 0.870_007_728_878_440_5).abs() < 2.0e-7);
    }

    #[test]
    fn reversing_the_declared_discrepancy_does_not_change_one_sided_power() {
        let positive = known_direction_power(&[0.1, 0.2, 0.3], 0.15, 0.05).unwrap();
        let negative = known_direction_power(&[-0.1, -0.2, -0.3], 0.15, 0.05).unwrap();
        assert!((positive.power - negative.power).abs() < 1.0e-12);
    }

    #[test]
    fn additional_informative_observations_do_not_reduce_oracle_power() {
        let one = known_direction_power(&[0.12], 0.15, 0.05).unwrap();
        let two = known_direction_power(&[0.12, 0.12], 0.15, 0.05).unwrap();
        let three = known_direction_power(&[0.12, 0.12, 0.12], 0.15, 0.05).unwrap();
        assert!(one.power < two.power);
        assert!(two.power < three.power);
    }

    #[test]
    fn minimum_replicates_is_the_first_count_to_reach_target() {
        let count = minimum_replicates_for_power(0.24, 0.15, 0.05, 0.80)
            .unwrap()
            .unwrap();
        assert_eq!(count, 3);
        let two = known_direction_power(&[0.24; 2], 0.15, 0.05).unwrap();
        let three = known_direction_power(&[0.24; 3], 0.15, 0.05).unwrap();
        assert!(two.power < 0.80);
        assert!(three.power >= 0.80);
    }

    #[test]
    fn envelope_preserves_each_alternative_and_uses_the_worst_case() {
        let alternatives = [
            DeclaredAlternative::new("early", vec![0.24, 0.24, 0.24, 0.24]),
            DeclaredAlternative::new("late", vec![0.01, 0.01, 0.01, 0.01]),
        ];
        let report = evaluate_power_envelope(&alternatives, 0.15, 0.05, 0.80).unwrap();
        assert_eq!(report.worst_case_alternative, "late");
        assert_eq!(report.underpowered_alternatives, vec!["late"]);
        assert!(!report.target_met);
        assert_eq!(report.alternatives.len(), 2);
    }

    #[test]
    fn envelope_rejects_unequal_observation_budgets() {
        let alternatives = [
            DeclaredAlternative::new("a", vec![0.1, 0.2]),
            DeclaredAlternative::new("b", vec![0.1]),
        ];
        assert!(matches!(
            evaluate_power_envelope(&alternatives, 0.15, 0.05, 0.80),
            Err(PowerError::MismatchedObservationCount { .. })
        ));
    }

    #[test]
    fn zero_discrepancy_cannot_reach_power_above_alpha() {
        assert_eq!(
            minimum_replicates_for_power(0.0, 0.15, 0.05, 0.80).unwrap(),
            None
        );
    }

    #[test]
    fn malformed_inputs_fail_closed() {
        assert_eq!(
            known_direction_power(&[], 0.15, 0.05),
            Err(PowerError::EmptyObservationDesign)
        );
        assert_eq!(
            known_direction_power(&[0.1], 0.0, 0.05),
            Err(PowerError::InvalidSigma)
        );
        assert_eq!(
            known_direction_power(&[f64::NAN], 0.15, 0.05),
            Err(PowerError::NonFiniteResidual { index: 0 })
        );
        assert_eq!(
            known_direction_power(&[0.1], 0.15, 0.0),
            Err(PowerError::InvalidAlpha)
        );
        assert_eq!(
            minimum_replicates_for_power(0.1, 0.15, 0.05, 0.05),
            Err(PowerError::InvalidTargetPower)
        );
    }

    #[test]
    fn envelope_rejects_duplicate_ids() {
        let alternatives = [
            DeclaredAlternative::new("same", vec![0.1]),
            DeclaredAlternative::new("same", vec![0.2]),
        ];
        assert_eq!(
            evaluate_power_envelope(&alternatives, 0.15, 0.05, 0.80),
            Err(PowerError::DuplicateAlternativeId("same".to_owned()))
        );
    }
}
