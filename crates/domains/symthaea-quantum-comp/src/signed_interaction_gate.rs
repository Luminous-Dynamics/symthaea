//! Conservative signed-interaction assessment from independent evidence intervals.
//!
//! This is a new method motivated by the QARCH-012 audit. It is not the
//! published R-IDeA magnitude-only DeA gate and is not a theorem from that
//! paper. It bounds the two signed factors separately, multiplies intervals,
//! and only reports a direction when the entire product interval lies beyond
//! a preregistered margin.
//!
//! Inputs are:
//! - a current predictor value f_hat(x);
//! - a proxy prediction g(x) with a verified uniform bound
//!   |g(x)-bar_f(x)| <= tau2;
//! - an independent validation interval for f_star(x);
//! - lower coverage guarantees for both evidence intervals.
//!
//! The true function is never accepted as a point value by this API.
//! Artifact verification, statistical interval construction and the validity
//! of the tau2 certificate remain responsibilities of upstream verifiers.

use std::error::Error;
use std::fmt;

/// Conservative lower bound on joint coverage of two evidence statements.
///
/// If each statement holds with probability at least c1 and c2, the union
/// bound gives a joint coverage lower bound max(0, c1 + c2 - 1), without an
/// independence assumption.
pub fn joint_coverage_lower_bound(
    proxy_bound_coverage: f64,
    validation_interval_coverage: f64,
) -> Result<f64, SignedInteractionGateError> {
    validate_coverage(proxy_bound_coverage)?;
    validate_coverage(validation_interval_coverage)?;
    Ok((proxy_bound_coverage + validation_interval_coverage - 1.0).max(0.0))
}

/// Status of the signed cross-term interval relative to the declared margin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignedInteractionGateStatus {
    /// The entire interaction interval is strictly below -margin.
    CertifiedOffsetting,
    /// The entire interaction interval is strictly above +margin.
    CertifiedReinforcing,
    /// The entire interaction interval is contained in [-margin, +margin].
    NeutralWithinMargin,
    /// The interval overlaps more than one decision region.
    Ambiguous,
    /// Joint evidence coverage is below the preregistered minimum.
    InsufficientCoverage,
    /// An upstream verifier has not verified the proxy/validation artifacts.
    EvidenceNotVerified,
}

/// Inputs to the validation-bounded signed-interaction gate.
///
/// All artifact IDs must resolve in the upstream provenance system. The
/// evidence verification booleans are expected to be set only by that
/// verifier; this function does not independently fetch or validate artifacts.
#[derive(Debug, Clone, PartialEq)]
pub struct SignedInteractionGateInput {
    /// Stable design/candidate identifier.
    pub design_id: String,
    /// Physical/model domain identifier for the design point.
    pub domain_id: String,
    /// Current fitted predictor value at the design point.
    pub current_prediction: f64,
    /// Proxy predictor g(x) at the design point.
    pub proxy_prediction: f64,
    /// Certified uniform absolute error bound |g(x)-bar_f(x)| <= tau2.
    pub proxy_uniform_error_bound: f64,
    /// Lower coverage guarantee for the proxy uniform-error certificate.
    pub proxy_bound_coverage_lower: f64,
    /// Immutable artifact digest for proxy fitting/training.
    pub proxy_fit_artifact_digest: String,
    /// Immutable split ID used to train proxy g.
    pub proxy_fit_split_id: String,
    /// Immutable artifact digest for the proxy uniform-error certificate.
    pub proxy_bound_artifact_digest: String,
    /// Immutable held-out split ID used to validate the proxy error bound.
    pub proxy_bound_validation_split_id: String,
    /// Whether the upstream verifier validated the proxy bound for this domain.
    pub proxy_bound_evidence_verified: bool,
    /// Lower endpoint of the independent validation interval for f_star(x).
    pub true_function_interval_lower: f64,
    /// Upper endpoint of the independent validation interval for f_star(x).
    pub true_function_interval_upper: f64,
    /// Lower coverage guarantee for the validation interval.
    pub true_function_interval_coverage_lower: f64,
    /// Immutable artifact digest for the validation interval.
    pub true_function_validation_artifact_digest: String,
    /// Immutable split ID used to construct the true-function validation interval.
    pub true_function_validation_split_id: String,
    /// Whether the upstream verifier validated the interval construction.
    pub true_function_interval_evidence_verified: bool,
    /// Immutable evaluator artifact digest, reserved for downstream scoring.
    pub evaluator_artifact_digest: String,
    /// Immutable evaluator-only split ID, never available to selection.
    pub evaluator_split_id: String,
    /// Minimum joint coverage required before issuing a signed classification.
    pub minimum_joint_coverage: f64,
    /// Minimum absolute interaction magnitude required for sign certification.
    pub margin: f64,
}

/// Conservative interval for the signed product A=(f_hat-bar_f)(bar_f-f_star).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignedProductInterval {
    /// Lower endpoint across the four endpoint products.
    pub lower: f64,
    /// Upper endpoint across the four endpoint products.
    pub upper: f64,
}

/// Full signed-interaction assessment with interval and provenance summary.
#[derive(Debug, Clone, PartialEq)]
pub struct SignedInteractionGateReport {
    /// Stable design/candidate identifier.
    pub design_id: String,
    /// Interval for bar_f obtained from g and tau2.
    pub best_in_class_proxy_interval: SignedProductInterval,
    /// Interval for estimation error f_hat-bar_f.
    pub estimation_error_interval: SignedProductInterval,
    /// Interval for misspecification error bar_f-f_star.
    pub misspecification_error_interval: SignedProductInterval,
    /// Conservative interval for their signed product.
    pub signed_interaction_interval: SignedProductInterval,
    /// Conservative joint coverage lower bound.
    pub joint_coverage_lower_bound: f64,
    /// Minimum joint coverage required by the caller.
    pub minimum_joint_coverage: f64,
    /// Sign classification; never inferred from a midpoint alone.
    pub status: SignedInteractionGateStatus,
    /// Evidence artifact digests used in the assessment.
    pub evidence_artifact_digests: Vec<String>,
    /// Distinct data split IDs used by fit, certificates, validation and evaluation.
    pub evidence_split_ids: Vec<String>,
    /// Claim ceiling for this assessment.
    pub claim_ceiling: Vec<String>,
}

/// Input and arithmetic errors from the signed-interaction gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignedInteractionGateError {
    /// Design/domain or artifact metadata is missing.
    EmptyMetadata,
    /// Proxy fit, proxy bound, validation, or evaluator artifacts overlap.
    ArtifactDigestOverlap,
    /// A coverage probability or required coverage is outside [0, 1].
    InvalidCoverage,
    /// The proxy error bound or sign margin is invalid.
    InvalidNonNegativeBound,
    /// The true-function validation interval is malformed.
    InvalidTrueFunctionInterval,
    /// A prediction or interval endpoint is non-finite.
    NonFiniteInput,
    /// An interval endpoint or product overflowed.
    DerivedOverflow,
}

impl fmt::Display for SignedInteractionGateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyMetadata => "design, domain, and artifact metadata must be non-empty",
            Self::ArtifactDigestOverlap => {
                "proxy fit, proxy bound, validation, and evaluator artifacts must be distinct"
            }
            Self::InvalidCoverage => "coverage values must be finite and in [0, 1]",
            Self::InvalidNonNegativeBound => {
                "proxy error bound and sign margin must be finite and non-negative"
            }
            Self::InvalidTrueFunctionInterval => {
                "true-function validation interval must be finite and ordered"
            }
            Self::NonFiniteInput => "predictor values must be finite",
            Self::DerivedOverflow => "derived signed-interaction interval overflowed",
        };
        f.write_str(message)
    }
}

impl Error for SignedInteractionGateError {}

/// Assess the sign of the squared-error interaction using independent bounds.
///
/// Let:
/// E = f_hat - bar_f and M = bar_f - f_star.
/// The proxy certificate gives bar_f in [g-tau2, g+tau2]. The validation
/// interval gives f_star in [L,U]. We obtain conservative intervals for E
/// and M, then compute the minimum and maximum of the four endpoint products.
///
/// A signed status is issued only if both evidence statements are verified
/// and their union-bound joint coverage meets minimum_joint_coverage.
/// This is a conservative diagnostic under the supplied assumptions, not a
/// guarantee outside the declared design domain or target distribution.
pub fn assess_signed_interaction(
    input: &SignedInteractionGateInput,
) -> Result<SignedInteractionGateReport, SignedInteractionGateError> {
    validate_input(input)?;
    let coverage = joint_coverage_lower_bound(
        input.proxy_bound_coverage_lower,
        input.true_function_interval_coverage_lower,
    )?;

    let bar_lower = input.proxy_prediction - input.proxy_uniform_error_bound;
    let bar_upper = input.proxy_prediction + input.proxy_uniform_error_bound;
    let estimation = SignedProductInterval {
        lower: input.current_prediction - bar_upper,
        upper: input.current_prediction - bar_lower,
    };
    let misspecification = SignedProductInterval {
        lower: bar_lower - input.true_function_interval_upper,
        upper: bar_upper - input.true_function_interval_lower,
    };
    let interaction = interval_product(estimation, misspecification)?;

    let status = if !input.proxy_bound_evidence_verified
        || !input.true_function_interval_evidence_verified
    {
        SignedInteractionGateStatus::EvidenceNotVerified
    } else if coverage + 1.0e-12 < input.minimum_joint_coverage {
        SignedInteractionGateStatus::InsufficientCoverage
    } else if interaction.upper < -input.margin {
        SignedInteractionGateStatus::CertifiedOffsetting
    } else if interaction.lower > input.margin {
        SignedInteractionGateStatus::CertifiedReinforcing
    } else if interaction.lower >= -input.margin
        && interaction.upper <= input.margin
    {
        SignedInteractionGateStatus::NeutralWithinMargin
    } else {
        SignedInteractionGateStatus::Ambiguous
    };

    Ok(SignedInteractionGateReport {
        design_id: input.design_id.clone(),
        best_in_class_proxy_interval: SignedProductInterval {
            lower: bar_lower,
            upper: bar_upper,
        },
        estimation_error_interval: estimation,
        misspecification_error_interval: misspecification,
        signed_interaction_interval: interaction,
        joint_coverage_lower_bound: coverage,
        minimum_joint_coverage: input.minimum_joint_coverage,
        status,
        evidence_artifact_digests: vec![
            input.proxy_fit_artifact_digest.clone(),
            input.proxy_bound_artifact_digest.clone(),
            input.true_function_validation_artifact_digest.clone(),
            input.evaluator_artifact_digest.clone(),
        ],
        evidence_split_ids: vec![
            input.proxy_fit_split_id.clone(),
            input.proxy_bound_validation_split_id.clone(),
            input.true_function_validation_split_id.clone(),
            input.evaluator_split_id.clone(),
        ],
        claim_ceiling: vec![
            "Signed interaction assessment at the declared design point only".into(),
            "Valid only if proxy and validation coverage statements are true".into(),
            "Does not establish lower future test risk outside the declared domain".into(),
            "Not the published R-IDeA DeA gate or a quantum hardware claim".into(),
        ],
    })
}

fn validate_input(
    input: &SignedInteractionGateInput,
) -> Result<(), SignedInteractionGateError> {
    let metadata = [
        input.design_id.as_str(),
        input.domain_id.as_str(),
        input.proxy_fit_artifact_digest.as_str(),
        input.proxy_fit_split_id.as_str(),
        input.proxy_bound_artifact_digest.as_str(),
        input.proxy_bound_validation_split_id.as_str(),
        input.true_function_validation_artifact_digest.as_str(),
        input.true_function_validation_split_id.as_str(),
        input.evaluator_artifact_digest.as_str(),
        input.evaluator_split_id.as_str(),
    ];
    if metadata.iter().any(|value| value.trim().is_empty()) {
        return Err(SignedInteractionGateError::EmptyMetadata);
    }
    let artifact_ids = [
        input.proxy_fit_artifact_digest.as_str(),
        input.proxy_bound_artifact_digest.as_str(),
        input.true_function_validation_artifact_digest.as_str(),
        input.evaluator_artifact_digest.as_str(),
    ];
    let split_ids = [
        input.proxy_fit_split_id.as_str(),
        input.proxy_bound_validation_split_id.as_str(),
        input.true_function_validation_split_id.as_str(),
        input.evaluator_split_id.as_str(),
    ];
    if has_duplicate_artifacts(&artifact_ids) || has_duplicate_artifacts(&split_ids) {
        return Err(SignedInteractionGateError::ArtifactDigestOverlap);
    }
    if [
        input.current_prediction,
        input.proxy_prediction,
        input.true_function_interval_lower,
        input.true_function_interval_upper,
    ]
    .iter()
    .any(|value| !value.is_finite())
    {
        return Err(SignedInteractionGateError::NonFiniteInput);
    }
    if !input.proxy_uniform_error_bound.is_finite()
        || input.proxy_uniform_error_bound < 0.0
        || !input.margin.is_finite()
        || input.margin < 0.0
    {
        return Err(SignedInteractionGateError::InvalidNonNegativeBound);
    }
    if input.true_function_interval_lower > input.true_function_interval_upper {
        return Err(SignedInteractionGateError::InvalidTrueFunctionInterval);
    }
    validate_coverage(input.proxy_bound_coverage_lower)?;
    validate_coverage(input.true_function_interval_coverage_lower)?;
    validate_coverage(input.minimum_joint_coverage)?;
    let bar_lower = input.proxy_prediction - input.proxy_uniform_error_bound;
    let bar_upper = input.proxy_prediction + input.proxy_uniform_error_bound;
    if !bar_lower.is_finite() || !bar_upper.is_finite() {
        return Err(SignedInteractionGateError::DerivedOverflow);
    }
    Ok(())
}

fn validate_coverage(value: f64) -> Result<(), SignedInteractionGateError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(SignedInteractionGateError::InvalidCoverage);
    }
    Ok(())
}

fn has_duplicate_artifacts(values: &[&str]) -> bool {
    values.iter().enumerate().any(|(index, value)| {
        values[..index]
            .iter()
            .any(|previous| previous == value)
    })
}

fn interval_product(
    left: SignedProductInterval,
    right: SignedProductInterval,
) -> Result<SignedProductInterval, SignedInteractionGateError> {
    let products = [
        left.lower * right.lower,
        left.lower * right.upper,
        left.upper * right.lower,
        left.upper * right.upper,
    ];
    if products.iter().any(|value| !value.is_finite()) {
        return Err(SignedInteractionGateError::DerivedOverflow);
    }
    let lower = products.iter().copied().fold(f64::INFINITY, f64::min);
    let upper = products
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    Ok(SignedProductInterval { lower, upper })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(
        current_prediction: f64,
        proxy_prediction: f64,
        proxy_error: f64,
        true_lower: f64,
        true_upper: f64,
    ) -> SignedInteractionGateInput {
        SignedInteractionGateInput {
            design_id: "x0".into(),
            domain_id: "toy-domain-v1".into(),
            current_prediction,
            proxy_prediction,
            proxy_uniform_error_bound: proxy_error,
            proxy_bound_coverage_lower: 0.99,
            proxy_fit_artifact_digest: "proxy-fit-artifact".into(),
            proxy_fit_split_id: "proxy-fit-split".into(),
            proxy_bound_artifact_digest: "proxy-bound-artifact".into(),
            proxy_bound_validation_split_id: "proxy-bound-validation-split".into(),
            proxy_bound_evidence_verified: true,
            true_function_interval_lower: true_lower,
            true_function_interval_upper: true_upper,
            true_function_interval_coverage_lower: 0.95,
            true_function_validation_artifact_digest: "true-function-validation".into(),
            true_function_interval_evidence_verified: true,
            evaluator_artifact_digest: "evaluator-only-artifact".into(),
            minimum_joint_coverage: 0.9,
            margin: 0.0,
        }
    }

    #[test]
    fn certifies_offsetting_sign_when_interaction_interval_is_negative() {
        let report = assess_signed_interaction(&input(-1.0, 1.0, 0.0, 0.5, 0.5)).unwrap();
        assert_eq!(
            report.status,
            SignedInteractionGateStatus::CertifiedOffsetting
        );
        assert!((report.signed_interaction_interval.lower - (-1.0)).abs() < 1.0e-12);
        assert!((report.signed_interaction_interval.upper - (-1.0)).abs() < 1.0e-12);
        assert!((report.joint_coverage_lower_bound - 0.94).abs() < 1.0e-12);
    }

    #[test]
    fn certifies_reinforcing_sign_when_interaction_interval_is_positive() {
        let report = assess_signed_interaction(&input(3.0, 1.0, 0.0, 0.5, 0.5)).unwrap();
        assert_eq!(
            report.status,
            SignedInteractionGateStatus::CertifiedReinforcing
        );
        assert!((report.signed_interaction_interval.lower - 1.0).abs() < 1.0e-12);
        assert!((report.signed_interaction_interval.upper - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn uncertain_proxy_and_truth_intervals_force_ambiguity() {
        let report = assess_signed_interaction(&input(1.0, 1.0, 0.1, 0.4, 0.6)).unwrap();
        assert_eq!(report.status, SignedInteractionGateStatus::Ambiguous);
        assert!(report.signed_interaction_interval.lower < 0.0);
        assert!(report.signed_interaction_interval.upper > 0.0);
    }

    #[test]
    fn union_bound_requires_sufficient_joint_coverage() {
        let mut data = input(3.0, 1.0, 0.0, 0.5, 0.5);
        data.proxy_bound_coverage_lower = 0.95;
        data.true_function_interval_coverage_lower = 0.90;
        data.minimum_joint_coverage = 0.90;
        let report = assess_signed_interaction(&data).unwrap();
        assert!((report.joint_coverage_lower_bound - 0.85).abs() < 1.0e-12);
        assert_eq!(report.status, SignedInteractionGateStatus::InsufficientCoverage);
    }

    #[test]
    fn unverified_evidence_never_produces_a_signed_certificate() {
        let mut data = input(-1.0, 1.0, 0.0, 0.5, 0.5);
        data.true_function_interval_evidence_verified = false;
        let report = assess_signed_interaction(&data).unwrap();
        assert_eq!(report.status, SignedInteractionGateStatus::EvidenceNotVerified);
    }

    #[test]
    fn artifact_overlap_is_rejected() {
        let mut data = input(-1.0, 1.0, 0.0, 0.5, 0.5);
        data.evaluator_artifact_digest = data.true_function_validation_artifact_digest.clone();
        assert_eq!(
            assess_signed_interaction(&data),
            Err(SignedInteractionGateError::ArtifactDigestOverlap)
        );
    }

    #[test]
    fn overlapping_split_ids_are_rejected() {
        let mut data = input(-1.0, 1.0, 0.0, 0.5, 0.5);
        data.evaluator_split_id = data.true_function_validation_split_id.clone();
        assert_eq!(
            assess_signed_interaction(&data),
            Err(SignedInteractionGateError::ArtifactDigestOverlap)
        );
    }

    #[test]
    fn malformed_inputs_fail_closed() {
        let mut data = input(-1.0, 1.0, 0.0, 0.5, 0.5);
        data.true_function_interval_lower = 0.6;
        data.true_function_interval_upper = 0.5;
        assert_eq!(
            assess_signed_interaction(&data),
            Err(SignedInteractionGateError::InvalidTrueFunctionInterval)
        );

        let mut bad_bound = input(-1.0, 1.0, -0.1, 0.5, 0.5);
        assert_eq!(
            assess_signed_interaction(&bad_bound),
            Err(SignedInteractionGateError::InvalidNonNegativeBound)
        );
        bad_bound.margin = f64::NAN;
        assert_eq!(
            assess_signed_interaction(&bad_bound),
            Err(SignedInteractionGateError::InvalidNonNegativeBound)
        );
    }

    #[test]
    fn interval_product_contains_all_endpoint_products() {
        let report = assess_signed_interaction(&input(2.0, 1.0, 0.2, 0.1, 0.4)).unwrap();
        let e = report.estimation_error_interval;
        let m = report.misspecification_error_interval;
        for estimation in [e.lower, e.upper] {
            for misspecification in [m.lower, m.upper] {
                let product = estimation * misspecification;
                assert!(product >= report.signed_interaction_interval.lower);
                assert!(product <= report.signed_interaction_interval.upper);
            }
        }
    }
}