//! Paper-faithful mathematical primitives for representative/informative/
//! de-amplifying Bayesian experimental design (R-IDeA).
//!
//! Reference: Tang, Sloman & Kaski, AISTATS 2026, PMLR 300, pp. 3016-3024,
//! arXiv:2506.07805v3, Sections 3-4 and Appendix D.
//!
//! This module implements mathematical scoring primitives and the proxy
//! objective. It does not train/optimize the proxy predictor g, estimate EIG,
//! estimate a valid tau_2 bound, or prove that the theorem's assumptions hold
//! for a physical system. Those require a separately qualified pipeline.
//!
//! MMD convention: Appendix D defines biased empirical squared MMD with an
//! RBF kernel. The authors' released compute_mmd helper also returns this
//! squared quantity. The robust ratio therefore uses MMD^2 consistently in
//! numerator and denominator; it does not take a square root.
//!
//! The first adaptive step needs a non-empty design history for the R-I ratio.
//! A caller should use a declared initialization design before calling
//! score_r_i. An empty or zero-discrepancy baseline is not replaced by a
//! fabricated large constant.

use std::error::Error;
use std::fmt;

/// Numerical tolerance for a tiny negative MMD^2 caused by round-off.
const MMD_NEGATIVE_ROUNDOFF_TOLERANCE: f64 = 1.0e-12;

/// All validation and numerical failures from these R-IDeA primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RidaExactError {
    /// An empirical distribution or design history is empty.
    EmptySamples,
    /// Sample vectors do not have the same positive feature dimension.
    FeatureDimensionMismatch,
    /// A sample contains a NaN or infinity.
    NonFiniteSample,
    /// RBF bandwidth is invalid or its variance is not representable.
    InvalidBandwidth,
    /// A baseline MMD^2 of zero makes the robust ratio undefined.
    ZeroBaselineMmd,
    /// EIG must be finite and non-negative.
    InvalidExpectedInformationGain,
    /// Lambda must be finite and non-negative.
    InvalidRepresentativenessWeight,
    /// A derived MMD or acquisition score became invalid.
    InvalidDerivedScore,
    /// Predictor, proxy, and observation vectors must be non-empty and equally sized.
    ProxyDimensionMismatch,
    /// A predictor, proxy, or observed target contains a NaN or infinity.
    NonFiniteProxyValue,
    /// Tau must be finite and non-negative.
    InvalidThreshold,
    /// Kappa must be finite and strictly positive.
    InvalidSmoothingScale,
    /// Threshold parameters violate their theoretical sign/domain constraints.
    InvalidThresholdParameters,
    /// A finite threshold overflowed during derivation.
    ThresholdOverflow,
}

impl fmt::Display for RidaExactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptySamples => "empirical samples and design history must be non-empty",
            Self::FeatureDimensionMismatch => "sample feature dimensions must be equal and positive",
            Self::NonFiniteSample => "empirical sample coordinates must be finite",
            Self::InvalidBandwidth => {
                "RBF bandwidth must be finite and positive with representable variance"
            }
            Self::ZeroBaselineMmd => "baseline MMD^2 is zero; the R-I robust ratio is undefined",
            Self::InvalidExpectedInformationGain => {
                "expected information gain must be finite and non-negative"
            }
            Self::InvalidRepresentativenessWeight => {
                "representativeness weight lambda must be finite and non-negative"
            }
            Self::InvalidDerivedScore => "a derived MMD or acquisition score is invalid",
            Self::ProxyDimensionMismatch => "proxy inputs must be non-empty and have equal lengths",
            Self::NonFiniteProxyValue => "predictor, proxy, and observed values must be finite",
            Self::InvalidThreshold => "de-amplification threshold must be finite and non-negative",
            Self::InvalidSmoothingScale => "sigmoid smoothing scale kappa must be finite and positive",
            Self::InvalidThresholdParameters => {
                "threshold parameters require tau0 >= 0, B_infinity >= 0, c >= 2, and tau2 >= 0"
            }
            Self::ThresholdOverflow => "the derived de-amplification threshold overflowed",
        };
        f.write_str(message)
    }
}

impl Error for RidaExactError {}

/// Compute the biased empirical squared MMD between two finite sample sets.
///
/// Each element is one point in a common Euclidean feature space. The
/// estimator is exactly
///
/// MMD^2(P,Q) = mean(k(P,P)) + mean(k(Q,Q)) - 2 mean(k(P,Q)),
///
/// with k(x,y) = exp(-||x-y||^2 / (2 sigma^2)). This is the convention in
/// Appendix D of the AISTATS 2026 R-IDeA paper and the authors' released
/// reference code. The biased estimator includes diagonal terms, so it
/// remains defined for singleton sample sets.
pub fn empirical_squared_mmd_rbf(
    left: &[Vec<f64>],
    right: &[Vec<f64>],
    sigma: f64,
) -> Result<f64, RidaExactError> {
    validate_bandwidth(sigma)?;
    let dimension = validate_samples(left)?;
    let right_dimension = validate_samples(right)?;
    if dimension != right_dimension {
        return Err(RidaExactError::FeatureDimensionMismatch);
    }

    let left_count = left.len() as f64;
    let right_count = right.len() as f64;
    let mut left_left = 0.0;
    for left_sample in left {
        for other in left {
            left_left += rbf_kernel(left_sample, other, sigma);
        }
    }

    let mut right_right = 0.0;
    for right_sample in right {
        for other in right {
            right_right += rbf_kernel(right_sample, other, sigma);
        }
    }

    let mut cross = 0.0;
    for left_sample in left {
        for right_sample in right {
            cross += rbf_kernel(left_sample, right_sample, sigma);
        }
    }

    let left_term = left_left / (left_count * left_count);
    let right_term = right_right / (right_count * right_count);
    let cross_term = 2.0 * cross / (left_count * right_count);
    let raw = left_term + right_term - cross_term;
    if !raw.is_finite() {
        return Err(RidaExactError::InvalidDerivedScore);
    }
    if raw < -MMD_NEGATIVE_ROUNDOFF_TOLERANCE {
        return Err(RidaExactError::InvalidDerivedScore);
    }
    Ok(raw.max(0.0))
}

/// R-I (representative and informative) acquisition result for one design.
#[derive(Debug, Clone, PartialEq)]
pub struct RiAcquisitionScore {
    /// Expected information gain supplied by the caller, in its declared unit.
    pub expected_information_gain: f64,
    /// Biased squared MMD between the pre-existing history and target samples.
    pub baseline_squared_mmd: f64,
    /// Biased squared MMD after appending the candidate design to the history.
    pub candidate_squared_mmd: f64,
    /// Candidate/baseline squared-MMD ratio.
    pub robust_ratio: f64,
    /// Multiplicative representativeness correction: 1 - lambda * robust_ratio.
    pub representativeness_factor: f64,
    /// R-I acquisition score: EIG * representativeness_factor.
    pub score: f64,
}

/// Evaluate the paper's R-I acquisition score for one candidate design.
///
/// The implementation uses squared MMD in both ratio terms, matching the
/// Appendix D estimator and the authors' reference implementation. It refuses
/// an empty history or a zero baseline MMD^2 instead of introducing an
/// arbitrary denominator. Negative scores are preserved, not clamped.
pub fn score_r_i(
    history: &[Vec<f64>],
    candidate: &[f64],
    target_distribution_samples: &[Vec<f64>],
    expected_information_gain: f64,
    lambda: f64,
    sigma: f64,
) -> Result<RiAcquisitionScore, RidaExactError> {
    if history.is_empty() {
        return Err(RidaExactError::EmptySamples);
    }
    if !expected_information_gain.is_finite() || expected_information_gain < 0.0 {
        return Err(RidaExactError::InvalidExpectedInformationGain);
    }
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(RidaExactError::InvalidRepresentativenessWeight);
    }
    if candidate.is_empty() {
        return Err(RidaExactError::FeatureDimensionMismatch);
    }
    if candidate.iter().any(|value| !value.is_finite()) {
        return Err(RidaExactError::NonFiniteSample);
    }

    let history_dimension = validate_samples(history)?;
    let target_dimension = validate_samples(target_distribution_samples)?;
    if candidate.len() != history_dimension || history_dimension != target_dimension {
        return Err(RidaExactError::FeatureDimensionMismatch);
    }

    let baseline_squared_mmd =
        empirical_squared_mmd_rbf(history, target_distribution_samples, sigma)?;
    if baseline_squared_mmd == 0.0 {
        return Err(RidaExactError::ZeroBaselineMmd);
    }

    let mut updated_history = Vec::with_capacity(history.len() + 1);
    updated_history.extend(history.iter().cloned());
    updated_history.push(candidate.to_vec());
    let candidate_squared_mmd =
        empirical_squared_mmd_rbf(&updated_history, target_distribution_samples, sigma)?;
    let robust_ratio = candidate_squared_mmd / baseline_squared_mmd;
    let representativeness_factor = 1.0 - lambda * robust_ratio;
    let score = expected_information_gain * representativeness_factor;

    if !robust_ratio.is_finite()
        || !representativeness_factor.is_finite()
        || !score.is_finite()
    {
        return Err(RidaExactError::InvalidDerivedScore);
    }

    Ok(RiAcquisitionScore {
        expected_information_gain,
        baseline_squared_mmd,
        candidate_squared_mmd,
        robust_ratio,
        representativeness_factor,
        score,
    })
}

/// Result of deriving the paper's proxy de-amplification threshold.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeAmplificationThreshold {
    /// B_infinity = 0: model is well specified under the supplied bound.
    /// The de-amplification gate is not needed and should be neutral (weight 1).
    WellSpecified,
    /// Finite threshold tau = tau0 / B_infinity + c * B_infinity + tau2.
    Finite(f64),
}

/// Derive threshold tau from the paper's Theorem 2 and proxy-error bound.
///
/// For B_infinity > 0:
/// tau1 = tau0 / B_infinity + c * B_infinity,
/// tau = tau1 + tau2,
/// where tau0 >= 0, c >= 2 and tau2 >= 0.
///
/// If B_infinity is exactly zero, the model is well specified under the
/// supplied bound; the theoretical misspecification region is absent. We
/// return WellSpecified rather than divide by zero. Callers use a neutral
/// de-amplification weight of one in this case.
pub fn derive_de_amplification_threshold(
    tau0: f64,
    b_infinity: f64,
    c: f64,
    tau2: f64,
) -> Result<DeAmplificationThreshold, RidaExactError> {
    if !tau0.is_finite()
        || !b_infinity.is_finite()
        || !c.is_finite()
        || !tau2.is_finite()
        || tau0 < 0.0
        || b_infinity < 0.0
        || c < 2.0
        || tau2 < 0.0
    {
        return Err(RidaExactError::InvalidThresholdParameters);
    }
    if b_infinity == 0.0 {
        return Ok(DeAmplificationThreshold::WellSpecified);
    }

    let tau1 = tau0 / b_infinity + c * b_infinity;
    let tau = tau1 + tau2;
    if !tau1.is_finite() || !tau.is_finite() || tau < 0.0 {
        return Err(RidaExactError::ThresholdOverflow);
    }
    Ok(DeAmplificationThreshold::Finite(tau))
}

/// Calculate the paper's printed magnitude-only DeA sigmoid gate.
///
/// A well-specified model (threshold status WellSpecified) receives neutral
/// weight 1. Otherwise:
/// DeA(xi) = sigmoid((|f_hat(xi) - g(xi)| - tau) / kappa).
///
/// IMPORTANT: this formula uses an absolute gap and therefore cannot, by
/// itself, identify the sign of the generalization-error interaction. It
/// reproduces the printed acquisition term, not a signed certificate that a
/// sample de-amplifies error. The proposition/theorem sign relationship is
/// under explicit audit in QARCH-012. Do not use this value as proof that the
/// signed interaction is negative or that test error will decrease.
pub fn score_de_amplification(
    current_prediction: f64,
    proxy_prediction: f64,
    threshold: DeAmplificationThreshold,
    kappa: f64,
) -> Result<f64, RidaExactError> {
    if !current_prediction.is_finite() || !proxy_prediction.is_finite() {
        return Err(RidaExactError::NonFiniteProxyValue);
    }
    if !kappa.is_finite() || kappa <= 0.0 {
        return Err(RidaExactError::InvalidSmoothingScale);
    }

    match threshold {
        DeAmplificationThreshold::WellSpecified => Ok(1.0),
        DeAmplificationThreshold::Finite(tau) => {
            if !tau.is_finite() || tau < 0.0 {
                return Err(RidaExactError::InvalidThreshold);
            }
            let gap = (current_prediction - proxy_prediction).abs();
            let standardized = (gap - tau) / kappa;
            if standardized.is_nan() {
                return Err(RidaExactError::InvalidDerivedScore);
            }
            Ok(stable_sigmoid(standardized))
        }
    }
}

/// Full R-IDeA score for a candidate given its R-I score and predictor disagreement.
#[derive(Debug, Clone, PartialEq)]
pub struct RideaAcquisitionScore {
    /// R-I component (EIG and squared-MMD robust ratio).
    pub r_i: RiAcquisitionScore,
    /// DeA sigmoid weight.
    pub de_amplification_weight: f64,
    /// Full acquisition: R-I score multiplied by DeA.
    pub score: f64,
}

/// Combine a previously calculated R-I score with the DeA predictor gate.
pub fn score_r_idea(
    r_i: RiAcquisitionScore,
    current_prediction: f64,
    proxy_prediction: f64,
    threshold: DeAmplificationThreshold,
    kappa: f64,
) -> Result<RideaAcquisitionScore, RidaExactError> {
    let de_amplification_weight = score_de_amplification(
        current_prediction,
        proxy_prediction,
        threshold,
        kappa,
    )?;
    let score = r_i.score * de_amplification_weight;
    if !score.is_finite() {
        return Err(RidaExactError::InvalidDerivedScore);
    }
    Ok(RideaAcquisitionScore {
        r_i,
        de_amplification_weight,
        score,
    })
}

/// Sign of the cross-term in the squared-error decomposition.
///
/// For A = (f_hat - bar_f) * (bar_f - f_star), a negative value offsets the
/// sum of the separate misspecification and estimation squared errors; a
/// positive value reinforces that sum. This diagnostic is evaluator/oracle
/// only because it requires the true function value f_star.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignedErrorInteraction {
    /// The cross-term is negative and reduces the sum of the two squared components.
    Offsetting,
    /// The cross-term is positive and increases the sum of the two squared components.
    Reinforcing,
    /// The cross-term is exactly zero.
    Neutral,
}

/// Evaluator-only decomposition of one point's squared prediction error.
///
/// This structure requires f_star and must never be passed to deployment
/// selection or used to train the proxy. It is provided to test whether a
/// magnitude-only DeA gate actually identifies signed error cancellation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvaluatorOnlySignedErrorDiagnostic {
    /// bar_f - f_star: the signed misspecification error.
    pub misspecification_error: f64,
    /// f_hat - bar_f: the signed estimation error.
    pub estimation_error: f64,
    /// A = (f_hat - bar_f) * (bar_f - f_star), without the factor of two.
    pub signed_interaction: f64,
    /// 2*A: the cross-term in the squared-error identity.
    pub decomposition_cross_term: f64,
    /// (bar_f - f_star)^2.
    pub misspecification_squared_error: f64,
    /// (f_hat - bar_f)^2.
    pub estimation_squared_error: f64,
    /// Sum of the two squared components before the cross-term.
    pub component_sum: f64,
    /// (f_hat - f_star)^2: actual pointwise squared prediction error.
    pub actual_squared_error: f64,
    /// Whether the cross-term offsets or reinforces the two squared components.
    pub interaction: SignedErrorInteraction,
}

/// Compute the signed squared-error decomposition for one evaluator-only point.
///
/// Identity:
/// (f_hat - f_star)^2 = (bar_f - f_star)^2
///                    + (f_hat - bar_f)^2
///                    + 2*(bar_f - f_star)*(f_hat - bar_f).
///
/// The true value is an evaluator-only input. This function is not suitable
/// for deployed acquisition because f_star is unknown in real experiments.
pub fn evaluate_signed_interaction_oracle(
    current_prediction: f64,
    best_in_class_prediction: f64,
    true_prediction: f64,
) -> Result<EvaluatorOnlySignedErrorDiagnostic, RidaExactError> {
    if !current_prediction.is_finite()
        || !best_in_class_prediction.is_finite()
        || !true_prediction.is_finite()
    {
        return Err(RidaExactError::NonFiniteProxyValue);
    }

    let misspecification_error = best_in_class_prediction - true_prediction;
    let estimation_error = current_prediction - best_in_class_prediction;
    let signed_interaction = estimation_error * misspecification_error;
    let decomposition_cross_term = 2.0 * signed_interaction;
    let misspecification_squared_error = misspecification_error * misspecification_error;
    let estimation_squared_error = estimation_error * estimation_error;
    let component_sum = misspecification_squared_error + estimation_squared_error;
    let actual_squared_error =
        (current_prediction - true_prediction) * (current_prediction - true_prediction);
    let reconstructed_error = component_sum + decomposition_cross_term;

    if [
        misspecification_error,
        estimation_error,
        signed_interaction,
        decomposition_cross_term,
        misspecification_squared_error,
        estimation_squared_error,
        component_sum,
        actual_squared_error,
        reconstructed_error,
    ]
    .iter()
    .any(|value| !value.is_finite())
        || (actual_squared_error - reconstructed_error).abs()
            > 1.0e-10 * actual_squared_error.abs().max(1.0)
    {
        return Err(RidaExactError::InvalidDerivedScore);
    }

    let interaction = if signed_interaction < 0.0 {
        SignedErrorInteraction::Offsetting
    } else if signed_interaction > 0.0 {
        SignedErrorInteraction::Reinforcing
    } else {
        SignedErrorInteraction::Neutral
    };

    Ok(EvaluatorOnlySignedErrorDiagnostic {
        misspecification_error,
        estimation_error,
        signed_interaction,
        decomposition_cross_term,
        misspecification_squared_error,
        estimation_squared_error,
        component_sum,
        actual_squared_error,
        interaction,
    })
}

/// Exact paper Eq. (11) proxy-training objective and its components.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProxyTrainingObjective {
    /// Mean squared error between proxy predictions and observed outcomes.
    pub mean_squared_error: f64,
    /// Mean disagreement hinge penalty max(0, tau - |f_hat - g|).
    pub disagreement_hinge: f64,
    /// Sum of the two terms (paper Eq. 11).
    pub total: f64,
}

/// Evaluate the R-IDeA proxy objective for fixed current-predictor and proxy outputs.
///
/// This evaluates the objective but does not optimize g. The paper's Eq. (11)
/// is MSE(g(x), y) plus the disagreement hinge penalty. Optional weighting
/// and regularization present in the released reference implementation are
/// intentionally omitted from this paper-equation primitive; an optimizer
/// adapter must report any additional terms.
pub fn evaluate_proxy_training_objective(
    current_predictions: &[f64],
    proxy_predictions: &[f64],
    observed_targets: &[f64],
    tau: f64,
) -> Result<ProxyTrainingObjective, RidaExactError> {
    let count = current_predictions.len();
    if count == 0 || proxy_predictions.len() != count || observed_targets.len() != count {
        return Err(RidaExactError::ProxyDimensionMismatch);
    }
    if !tau.is_finite() || tau < 0.0 {
        return Err(RidaExactError::InvalidThreshold);
    }
    if current_predictions
        .iter()
        .chain(proxy_predictions)
        .chain(observed_targets)
        .any(|value| !value.is_finite())
    {
        return Err(RidaExactError::NonFiniteProxyValue);
    }

    let mut squared_error = 0.0;
    let mut disagreement_hinge = 0.0;
    for ((current, proxy), observed) in current_predictions
        .iter()
        .zip(proxy_predictions)
        .zip(observed_targets)
    {
        let residual = proxy - observed;
        let gap = (current - proxy).abs();
        squared_error += residual * residual;
        disagreement_hinge += (tau - gap).max(0.0);
    }
    let denominator = count as f64;
    let mean_squared_error = squared_error / denominator;
    let disagreement_hinge = disagreement_hinge / denominator;
    let total = mean_squared_error + disagreement_hinge;

    if !mean_squared_error.is_finite()
        || !disagreement_hinge.is_finite()
        || !total.is_finite()
    {
        return Err(RidaExactError::InvalidDerivedScore);
    }

    Ok(ProxyTrainingObjective {
        mean_squared_error,
        disagreement_hinge,
        total,
    })
}

fn validate_samples(samples: &[Vec<f64>]) -> Result<usize, RidaExactError> {
    if samples.is_empty() {
        return Err(RidaExactError::EmptySamples);
    }
    let dimension = samples[0].len();
    if dimension == 0 || samples.iter().any(|sample| sample.len() != dimension) {
        return Err(RidaExactError::FeatureDimensionMismatch);
    }
    if samples.iter().flatten().any(|coordinate| !coordinate.is_finite()) {
        return Err(RidaExactError::NonFiniteSample);
    }
    Ok(dimension)
}

fn validate_bandwidth(sigma: f64) -> Result<(), RidaExactError> {
    let variance = 2.0 * sigma * sigma;
    if !sigma.is_finite() || sigma <= 0.0 || !variance.is_finite() || variance <= 0.0 {
        return Err(RidaExactError::InvalidBandwidth);
    }
    Ok(())
}

fn rbf_kernel(left: &[f64], right: &[f64], sigma: f64) -> f64 {
    let variance = 2.0 * sigma * sigma;
    let mut squared_distance = 0.0;
    for (left_value, right_value) in left.iter().zip(right) {
        let difference = left_value - right_value;
        squared_distance += difference * difference;
    }
    if squared_distance.is_infinite() {
        return 0.0;
    }
    (-squared_distance / variance).exp()
}

fn stable_sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(value: f64) -> Vec<f64> {
        vec![value]
    }

    #[test]
    fn squared_mmd_is_zero_for_identical_empirical_distributions() {
        let samples = vec![point(0.0), point(1.0)];
        let actual = empirical_squared_mmd_rbf(&samples, &samples, 1.0).unwrap();
        assert!(actual.abs() < 1.0e-12);
    }

    #[test]
    fn squared_mmd_matches_independent_singleton_rbf_reference() {
        let actual =
            empirical_squared_mmd_rbf(&[point(0.0)], &[point(1.0)], 1.0).unwrap();
        let expected = 2.0 - 2.0 * (-0.5_f64).exp();
        assert!((actual - expected).abs() < 1.0e-12);
        assert!((actual - 0.786_938_680_574_733_2).abs() < 1.0e-12);
    }

    #[test]
    fn biased_squared_mmd_is_invariant_to_duplicate_empirical_points() {
        let singleton = empirical_squared_mmd_rbf(&[point(0.0)], &[point(1.0)], 1.0).unwrap();
        let duplicated =
            empirical_squared_mmd_rbf(&[point(0.0), point(0.0)], &[point(1.0)], 1.0)
                .unwrap();
        assert!((singleton - duplicated).abs() < 1.0e-12);
    }

    #[test]
    fn mmd_rejects_empty_mismatched_and_non_finite_samples() {
        assert_eq!(
            empirical_squared_mmd_rbf(&[], &[point(0.0)], 1.0),
            Err(RidaExactError::EmptySamples)
        );
        assert_eq!(
            empirical_squared_mmd_rbf(&[point(0.0)], &[vec![0.0, 1.0]], 1.0),
            Err(RidaExactError::FeatureDimensionMismatch)
        );
        assert_eq!(
            empirical_squared_mmd_rbf(&[point(f64::NAN)], &[point(0.0)], 1.0),
            Err(RidaExactError::NonFiniteSample)
        );
        assert_eq!(
            empirical_squared_mmd_rbf(&[point(0.0)], &[point(1.0)], 0.0),
            Err(RidaExactError::InvalidBandwidth)
        );
    }

    #[test]
    fn r_i_matches_paper_ratio_with_squared_mmd() {
        let history = vec![point(0.0)];
        let target = vec![point(0.0), point(1.0)];
        let score = score_r_i(&history, &[1.5], &target, 0.7, 0.5, 1.0).unwrap();
        assert!((score.baseline_squared_mmd - 0.196_734_670_143_683_18).abs() < 1.0e-12);
        assert!((score.candidate_squared_mmd - 0.058_751_548_707_702_16).abs() < 1.0e-12);
        assert!((score.robust_ratio - 0.298_633_426_760_995_2).abs() < 1.0e-12);
        assert!((score.score - 0.595_478_300_633_651_7).abs() < 1.0e-12);
    }

    #[test]
    fn zero_lambda_recovers_expected_information_gain() {
        let score = score_r_i(
            &[point(0.0)],
            &[1.5],
            &[point(0.0), point(1.0)],
            0.7,
            0.0,
            1.0,
        )
        .unwrap();
        assert!((score.score - 0.7).abs() < 1.0e-12);
        assert!((score.representativeness_factor - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn r_i_preserves_negative_scores_instead_of_clamping() {
        let score = score_r_i(
            &[point(0.0)],
            &[1.5],
            &[point(0.0), point(1.0)],
            1.0,
            5.0,
            1.0,
        )
        .unwrap();
        assert!(score.representativeness_factor < 0.0);
        assert!((score.score - score.representativeness_factor).abs() < 1.0e-12);
    }

    #[test]
    fn r_i_rejects_zero_baseline_and_empty_history() {
        assert_eq!(
            score_r_i(&[], &[1.0], &[point(0.0)], 1.0, 1.0, 1.0),
            Err(RidaExactError::EmptySamples)
        );
        assert_eq!(
            score_r_i(&[point(0.0)], &[1.0], &[point(0.0)], 1.0, 1.0, 1.0),
            Err(RidaExactError::ZeroBaselineMmd)
        );
    }

    #[test]
    fn threshold_matches_theorem_two_and_handles_well_specified_case() {
        assert_eq!(
            derive_de_amplification_threshold(0.5, 2.0, 2.0, 0.25).unwrap(),
            DeAmplificationThreshold::Finite(4.5)
        );
        assert_eq!(
            derive_de_amplification_threshold(0.5, 0.0, 2.0, 0.25).unwrap(),
            DeAmplificationThreshold::WellSpecified
        );
        assert_eq!(
            derive_de_amplification_threshold(0.5, 2.0, 1.99, 0.25),
            Err(RidaExactError::InvalidThresholdParameters)
        );
    }

    #[test]
    fn dea_sigmoid_is_stable_and_neutral_when_well_specified() {
        let actual = score_de_amplification(
            2.0,
            0.0,
            DeAmplificationThreshold::Finite(1.0),
            1.0,
        )
        .unwrap();
        assert!((actual - 0.731_058_578_630_004_9).abs() < 1.0e-12);
        assert_eq!(
            score_de_amplification(
                1.0e308,
                -1.0e308,
                DeAmplificationThreshold::Finite(1.0),
                1.0
            )
            .unwrap(),
            1.0
        );
        assert_eq!(
            score_de_amplification(100.0, 0.0, DeAmplificationThreshold::WellSpecified, 1.0)
                .unwrap(),
            1.0
        );
    }

    #[test]
    fn magnitude_dea_gate_cannot_identify_signed_error_offsetting() {
        let threshold = DeAmplificationThreshold::Finite(1.0);
        let kappa = 1.0;
        let left_gate = score_de_amplification(-1.0, 1.0, threshold, kappa).unwrap();
        let right_gate = score_de_amplification(3.0, 1.0, threshold, kappa).unwrap();
        assert!((left_gate - right_gate).abs() < 1.0e-12);
        assert!((left_gate - 0.731_058_578_630_004_9).abs() < 1.0e-12);

        let left = evaluate_signed_interaction_oracle(-1.0, 1.0, 0.5).unwrap();
        let right = evaluate_signed_interaction_oracle(3.0, 1.0, 0.5).unwrap();
        assert_eq!(left.interaction, SignedErrorInteraction::Offsetting);
        assert_eq!(right.interaction, SignedErrorInteraction::Reinforcing);
        assert!((left.signed_interaction - (-1.0)).abs() < 1.0e-12);
        assert!((right.signed_interaction - 1.0).abs() < 1.0e-12);
        assert!((left.actual_squared_error - 2.25).abs() < 1.0e-12);
        assert!((right.actual_squared_error - 6.25).abs() < 1.0e-12);

        let threshold_from_paper =
            derive_de_amplification_threshold(0.0, 0.5, 2.0, 0.0).unwrap();
        assert_eq!(threshold_from_paper, DeAmplificationThreshold::Finite(1.0));
    }

    #[test]
    fn signed_interaction_oracle_rejects_non_finite_truth() {
        assert_eq!(
            evaluate_signed_interaction_oracle(1.0, 0.5, f64::NAN),
            Err(RidaExactError::NonFiniteProxyValue)
        );
    }

    #[test]
    fn proxy_objective_matches_paper_equation_eleven() {
        let loss = evaluate_proxy_training_objective(
            &[2.0, 2.0],
            &[1.0, 4.0],
            &[1.0, 3.0],
            2.0,
        )
        .unwrap();
        assert!((loss.mean_squared_error - 0.5).abs() < 1.0e-12);
        assert!((loss.disagreement_hinge - 0.5).abs() < 1.0e-12);
        assert!((loss.total - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn full_r_idea_score_multiplies_r_i_by_dea_weight() {
        let r_i = score_r_i(
            &[point(0.0)],
            &[1.5],
            &[point(0.0), point(1.0)],
            0.7,
            0.5,
            1.0,
        )
        .unwrap();
        let full = score_r_idea(
            r_i,
            2.0,
            0.0,
            DeAmplificationThreshold::Finite(1.0),
            1.0,
        )
        .unwrap();
        assert!((full.score - full.r_i.score * full.de_amplification_weight).abs() < 1.0e-12);
    }

    #[test]
    fn proxy_objective_rejects_mismatched_or_non_finite_inputs() {
        assert_eq!(
            evaluate_proxy_training_objective(&[1.0], &[], &[1.0], 0.5),
            Err(RidaExactError::ProxyDimensionMismatch)
        );
        assert_eq!(
            evaluate_proxy_training_objective(&[f64::NAN], &[1.0], &[1.0], 0.5),
            Err(RidaExactError::NonFiniteProxyValue)
        );
    }
}