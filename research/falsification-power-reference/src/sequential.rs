//! Time-uniform Gaussian likelihood-ratio e-process references.
//!
//! The evaluator commits each alternative prediction before accepting the
//! corresponding observation. This makes chronology explicit in the local
//! protocol, although it cannot prove that a caller did not read an outcome
//! through an out-of-band channel.
//!
//! Validity requires the conditional simple null
//! Y_t | F_{t-1}, A_t ~ N(mu_t, sigma_t^2), where mu_t and sigma_t are known
//! predictable quantities and each alternative discrepancy d_t is committed
//! before Y_t. Unknown/estimated conditional means or variances, misspecified
//! state models, composite nulls and arbitrary physical trajectories require
//! separate methods and qualification.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

/// Invalid input or protocol transition for a sequential reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SequentialError {
    InvalidSigma,
    InvalidAlpha,
    EmptyObservationId,
    EmptyDecisionDigest,
    DuplicateObservationId(String),
    PredictionAlreadyPending,
    NoPredictionPending,
    ObservationIdMismatch { expected: String, actual: String },
    DecisionDigestMismatch,
    NonFiniteDiscrepancy,
    NonFiniteConditionalMean,
    NonFiniteObservation,
    EmptyAlternativeSet,
    EmptyAlternativeId,
    DuplicateAlternativeId(String),
    InvalidAlternativeWeight,
    AlternativeWeightsMustSumToOne,
    AlternativeCountMismatch { expected: usize, actual: usize },
    NumericalOverflow,
    ObservationCountOverflow,
}

impl fmt::Display for SequentialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSigma => write!(f, "sigma must be finite and strictly positive"),
            Self::InvalidAlpha => {
                write!(f, "alpha must be finite and strictly between zero and one")
            }
            Self::EmptyObservationId => write!(f, "observation ID must not be empty"),
            Self::EmptyDecisionDigest => write!(f, "decision digest must not be empty"),
            Self::DuplicateObservationId(id) => {
                write!(f, "observation ID was already consumed: {id}")
            }
            Self::PredictionAlreadyPending => {
                write!(f, "a prediction is already pending an observation")
            }
            Self::NoPredictionPending => write!(
                f,
                "an observation cannot be scored before a prediction is committed"
            ),
            Self::ObservationIdMismatch { expected, actual } => {
                write!(
                    f,
                    "observation ID mismatch: expected {expected}, received {actual}"
                )
            }
            Self::DecisionDigestMismatch => write!(
                f,
                "observation decision digest differs from the committed prediction"
            ),
            Self::NonFiniteDiscrepancy => write!(f, "predicted discrepancy must be finite"),
            Self::NonFiniteConditionalMean => write!(f, "null conditional mean must be finite"),
            Self::NonFiniteObservation => write!(f, "observed residual must be finite"),
            Self::EmptyAlternativeSet => write!(f, "alternative set must not be empty"),
            Self::EmptyAlternativeId => write!(f, "alternative ID must not be empty"),
            Self::DuplicateAlternativeId(id) => write!(f, "duplicate alternative ID: {id}"),
            Self::InvalidAlternativeWeight => write!(
                f,
                "alternative weights must be finite and strictly positive"
            ),
            Self::AlternativeWeightsMustSumToOne => {
                write!(f, "fixed alternative weights must sum to one")
            }
            Self::AlternativeCountMismatch { expected, actual } => {
                write!(
                    f,
                    "received {actual} alternative predictions; expected {expected}"
                )
            }
            Self::NumericalOverflow => {
                write!(f, "likelihood-ratio update exceeded finite f64 range")
            }
            Self::ObservationCountOverflow => write!(f, "observation counter overflow"),
        }
    }
}

impl Error for SequentialError {}

/// One update result. The p-value is based on the maximum e-value observed
/// through this step, not only the current e-value.
#[derive(Debug, Clone, PartialEq)]
pub struct SequentialUpdate {
    pub observation_id: String,
    pub decision_digest: String,
    pub alpha: f64,
    pub expected_residual: f64,
    pub null_conditional_mean: f64,
    pub sigma: f64,
    pub observations: u64,
    pub log_e_value: f64,
    pub maximum_log_e_value: f64,
    pub anytime_p_value: f64,
    pub crossed_alpha_boundary: bool,
}

/// A single evaluator-declared alternative that is fixed before the sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedAlternative {
    pub id: String,
    pub weight: f64,
}

/// One component of a fixed-mixture report.
#[derive(Debug, Clone, PartialEq)]
pub struct MixtureComponentUpdate {
    pub id: String,
    pub weight: f64,
    pub expected_residual: f64,
    pub log_e_value: f64,
}

/// Report for a predeclared fixed mixture of alternative-specific e-processes.
#[derive(Debug, Clone, PartialEq)]
pub struct MixtureSequentialUpdate {
    pub observation_id: String,
    pub decision_digest: String,
    pub alpha: f64,
    pub null_conditional_mean: f64,
    pub sigma: f64,
    pub observations: u64,
    pub components: Vec<MixtureComponentUpdate>,
    pub log_e_value: f64,
    pub maximum_log_e_value: f64,
    pub anytime_p_value: f64,
    pub crossed_alpha_boundary: bool,
}

#[derive(Debug, Clone)]
struct PendingPrediction {
    observation_id: String,
    decision_digest: String,
    residuals: Vec<f64>,
    null_conditional_mean: f64,
    sigma: f64,
}

fn validate_alpha_sigma(alpha: f64, sigma: f64) -> Result<(), SequentialError> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(SequentialError::InvalidSigma);
    }
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(SequentialError::InvalidAlpha);
    }
    Ok(())
}

fn validate_prediction_identity(
    observation_id: &str,
    decision_digest: &str,
) -> Result<(), SequentialError> {
    if observation_id.trim().is_empty() {
        return Err(SequentialError::EmptyObservationId);
    }
    if decision_digest.trim().is_empty() {
        return Err(SequentialError::EmptyDecisionDigest);
    }
    Ok(())
}

fn likelihood_ratio_increment(
    discrepancy: f64,
    observed_value: f64,
    null_conditional_mean: f64,
    sigma: f64,
) -> Result<f64, SequentialError> {
    if !discrepancy.is_finite() {
        return Err(SequentialError::NonFiniteDiscrepancy);
    }
    if !null_conditional_mean.is_finite() {
        return Err(SequentialError::NonFiniteConditionalMean);
    }
    if !observed_value.is_finite() {
        return Err(SequentialError::NonFiniteObservation);
    }

    // Under H0 the predictable innovation Y_t - mu_t is conditionally
    // N(0, sigma_t^2). The alternative changes its conditional mean by d_t.
    let innovation = observed_value - null_conditional_mean;
    if !innovation.is_finite() {
        return Err(SequentialError::NumericalOverflow);
    }
    let standardized_discrepancy = discrepancy / sigma;
    let standardized_innovation = innovation / sigma;
    let increment = standardized_discrepancy * standardized_innovation
        - 0.5 * standardized_discrepancy * standardized_discrepancy;
    if !increment.is_finite() {
        return Err(SequentialError::NumericalOverflow);
    }
    Ok(increment)
}

fn anytime_p_value(maximum_log_e_value: f64) -> f64 {
    (-maximum_log_e_value.max(0.0)).exp().min(1.0)
}

/// A predictable-prediction, single-alternative Gaussian e-process.
///
/// For each time t, the evaluator must call commit_prediction before observe.
/// Under the conditional Gaussian simple null, the likelihood-ratio increment
/// exp(d_t * (Y_t - mu_t) / sigma_t^2 - d_t^2 / (2*sigma_t^2)) has conditional
/// expectation 1 when mu_t, sigma_t and d_t are predictable and the conditional
/// Gaussian null is correctly specified. The product is therefore a
/// nonnegative martingale.
///
/// The anytime p-value is 1 / max_{s <= t} E_s, capped at one. Ville's
/// inequality controls the probability of ever crossing 1/alpha under the
/// declared null. This is not valid automatically for estimated sigma,
/// composite nulls, a misspecified conditional-mean/state model, or
/// non-predictable mu_t, sigma_t or d_t.
#[derive(Debug, Clone)]
pub struct PredictableGaussianEProcess {
    sigma: f64,
    alpha: f64,
    log_e_value: f64,
    maximum_log_e_value: f64,
    observations: u64,
    pending: Option<PendingPrediction>,
    consumed_observation_ids: BTreeSet<String>,
}

impl PredictableGaussianEProcess {
    pub fn new(sigma: f64, alpha: f64) -> Result<Self, SequentialError> {
        validate_alpha_sigma(alpha, sigma)?;
        Ok(Self {
            sigma,
            alpha,
            log_e_value: 0.0,
            maximum_log_e_value: 0.0,
            observations: 0,
            pending: None,
            consumed_observation_ids: BTreeSet::new(),
        })
    }

    /// Commit the discrepancy prediction and decision identity before the
    /// corresponding observation becomes available to this evaluator.
    pub fn commit_prediction(
        &mut self,
        observation_id: impl Into<String>,
        decision_digest: impl Into<String>,
        expected_residual: f64,
    ) -> Result<(), SequentialError> {
        self.commit_prediction_with_sigma(
            observation_id,
            decision_digest,
            expected_residual,
            self.sigma,
        )
    }

    /// Commit a per-observation conditional noise sigma with the alternative
    /// prediction. This supports known action-dependent measurement precision.
    pub fn commit_prediction_with_sigma(
        &mut self,
        observation_id: impl Into<String>,
        decision_digest: impl Into<String>,
        expected_residual: f64,
        sigma: f64,
    ) -> Result<(), SequentialError> {
        self.commit_prediction_with_conditional_gaussian(
            observation_id,
            decision_digest,
            expected_residual,
            0.0,
            sigma,
        )
    }

    /// Commit an alternative discrepancy and the known conditional Gaussian
    /// null N(mu_t, sigma_t^2) before the corresponding observation is released.
    ///
    /// mu_t and sigma_t may depend on the previous history/action, but must be
    /// fixed before the current outcome. This permits correctly specified
    /// predictable AR/state-dependent innovations; it does not estimate or
    /// validate the conditional-mean model.
    pub fn commit_prediction_with_conditional_gaussian(
        &mut self,
        observation_id: impl Into<String>,
        decision_digest: impl Into<String>,
        expected_residual: f64,
        null_conditional_mean: f64,
        sigma: f64,
    ) -> Result<(), SequentialError> {
        let observation_id = observation_id.into();
        let decision_digest = decision_digest.into();
        validate_prediction_identity(&observation_id, &decision_digest)?;
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(SequentialError::InvalidSigma);
        }
        if !expected_residual.is_finite() {
            return Err(SequentialError::NonFiniteDiscrepancy);
        }
        if !null_conditional_mean.is_finite() {
            return Err(SequentialError::NonFiniteConditionalMean);
        }
        if self.pending.is_some() {
            return Err(SequentialError::PredictionAlreadyPending);
        }
        if self.consumed_observation_ids.contains(&observation_id) {
            return Err(SequentialError::DuplicateObservationId(observation_id));
        }
        self.pending = Some(PendingPrediction {
            observation_id,
            decision_digest,
            residuals: vec![expected_residual],
            null_conditional_mean,
            sigma,
        });
        Ok(())
    }

    /// Consume the observation that matches the pending, previously committed
    /// prediction. A mismatch or invalid observation leaves the prediction
    /// pending and does not mutate the e-process.
    pub fn observe(
        &mut self,
        observation_id: &str,
        decision_digest: &str,
        observed_residual: f64,
    ) -> Result<SequentialUpdate, SequentialError> {
        if !observed_residual.is_finite() {
            return Err(SequentialError::NonFiniteObservation);
        }
        let pending = self
            .pending
            .as_ref()
            .ok_or(SequentialError::NoPredictionPending)?;
        if pending.observation_id != observation_id {
            return Err(SequentialError::ObservationIdMismatch {
                expected: pending.observation_id.clone(),
                actual: observation_id.to_owned(),
            });
        }
        if pending.decision_digest != decision_digest {
            return Err(SequentialError::DecisionDigestMismatch);
        }

        let increment =
            likelihood_ratio_increment(
                pending.residuals[0],
                observed_residual,
                pending.null_conditional_mean,
                pending.sigma,
            )?;
        let next_log_e_value = self.log_e_value + increment;
        if !next_log_e_value.is_finite() {
            return Err(SequentialError::NumericalOverflow);
        }
        let next_observations = self
            .observations
            .checked_add(1)
            .ok_or(SequentialError::ObservationCountOverflow)?;
        let next_maximum = self.maximum_log_e_value.max(next_log_e_value);
        let alpha_boundary = -self.alpha.ln();
        let anytime_p = anytime_p_value(next_maximum);
        let crossed = next_maximum >= alpha_boundary;

        let pending = self
            .pending
            .take()
            .expect("pending prediction was checked above");
        self.consumed_observation_ids
            .insert(pending.observation_id.clone());
        self.log_e_value = next_log_e_value;
        self.maximum_log_e_value = next_maximum;
        self.observations = next_observations;

        Ok(SequentialUpdate {
            observation_id: pending.observation_id,
            decision_digest: pending.decision_digest,
            alpha: self.alpha,
            expected_residual: pending.residuals[0],
            null_conditional_mean: pending.null_conditional_mean,
            sigma: pending.sigma,
            observations: self.observations,
            log_e_value: self.log_e_value,
            maximum_log_e_value: self.maximum_log_e_value,
            anytime_p_value: anytime_p,
            crossed_alpha_boundary: crossed,
        })
    }

    pub fn has_pending_prediction(&self) -> bool {
        self.pending.is_some()
    }
}

/// A fixed, predeclared mixture of alternative-specific Gaussian e-processes.
///
/// Weights are normalized at construction and then remain fixed. Each
/// alternative residual must be committed before the shared observation.
/// Under the same simple conditional Gaussian null, a fixed convex mixture
/// remains an e-process. Post-hoc selection of the best component is not
/// substituted for this mixture and is not covered by its validity guarantee.
#[derive(Debug, Clone)]
pub struct PredictableGaussianMixture {
    sigma: f64,
    alpha: f64,
    alternatives: Vec<WeightedAlternative>,
    weights: Vec<f64>,
    component_log_e_values: Vec<f64>,
    log_e_value: f64,
    maximum_log_e_value: f64,
    observations: u64,
    pending: Option<PendingPrediction>,
    consumed_observation_ids: BTreeSet<String>,
}

impl PredictableGaussianMixture {
    pub fn new(
        sigma: f64,
        alpha: f64,
        alternatives: Vec<WeightedAlternative>,
    ) -> Result<Self, SequentialError> {
        validate_alpha_sigma(alpha, sigma)?;
        if alternatives.is_empty() {
            return Err(SequentialError::EmptyAlternativeSet);
        }

        let mut seen = BTreeSet::new();
        let mut weight_sum = 0.0;
        for alternative in &alternatives {
            if alternative.id.trim().is_empty() {
                return Err(SequentialError::EmptyAlternativeId);
            }
            if !seen.insert(alternative.id.clone()) {
                return Err(SequentialError::DuplicateAlternativeId(
                    alternative.id.clone(),
                ));
            }
            if !alternative.weight.is_finite() || alternative.weight <= 0.0 {
                return Err(SequentialError::InvalidAlternativeWeight);
            }
            weight_sum += alternative.weight;
        }
        if !weight_sum.is_finite() || (weight_sum - 1.0).abs() > 1.0e-10 {
            return Err(SequentialError::AlternativeWeightsMustSumToOne);
        }

        let weights = alternatives
            .iter()
            .map(|alternative| alternative.weight / weight_sum)
            .collect::<Vec<_>>();
        let count = alternatives.len();
        Ok(Self {
            sigma,
            alpha,
            alternatives,
            weights,
            component_log_e_values: vec![0.0; count],
            log_e_value: 0.0,
            maximum_log_e_value: 0.0,
            observations: 0,
            pending: None,
            consumed_observation_ids: BTreeSet::new(),
        })
    }

    /// Commit every alternative prediction for one observation before
    /// accepting the corresponding shared outcome.
    pub fn commit_predictions(
        &mut self,
        observation_id: impl Into<String>,
        decision_digest: impl Into<String>,
        expected_residuals: &[f64],
    ) -> Result<(), SequentialError> {
        self.commit_predictions_with_sigma(
            observation_id,
            decision_digest,
            expected_residuals,
            self.sigma,
        )
    }

    /// Commit a per-observation conditional sigma shared by all alternatives.
    pub fn commit_predictions_with_sigma(
        &mut self,
        observation_id: impl Into<String>,
        decision_digest: impl Into<String>,
        expected_residuals: &[f64],
        sigma: f64,
    ) -> Result<(), SequentialError> {
        self.commit_predictions_with_conditional_gaussian(
            observation_id,
            decision_digest,
            expected_residuals,
            0.0,
            sigma,
        )
    }

    /// Commit a fixed alternative mixture and known conditional Gaussian null
    /// N(mu_t, sigma_t^2) before the shared observation is released.
    pub fn commit_predictions_with_conditional_gaussian(
        &mut self,
        observation_id: impl Into<String>,
        decision_digest: impl Into<String>,
        expected_residuals: &[f64],
        null_conditional_mean: f64,
        sigma: f64,
    ) -> Result<(), SequentialError> {
        let observation_id = observation_id.into();
        let decision_digest = decision_digest.into();
        validate_prediction_identity(&observation_id, &decision_digest)?;
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(SequentialError::InvalidSigma);
        }
        if !null_conditional_mean.is_finite() {
            return Err(SequentialError::NonFiniteConditionalMean);
        }
        if self.pending.is_some() {
            return Err(SequentialError::PredictionAlreadyPending);
        }
        if self.consumed_observation_ids.contains(&observation_id) {
            return Err(SequentialError::DuplicateObservationId(observation_id));
        }
        if expected_residuals.len() != self.alternatives.len() {
            return Err(SequentialError::AlternativeCountMismatch {
                expected: self.alternatives.len(),
                actual: expected_residuals.len(),
            });
        }
        if expected_residuals.iter().any(|value| !value.is_finite()) {
            return Err(SequentialError::NonFiniteDiscrepancy);
        }
        self.pending = Some(PendingPrediction {
            observation_id,
            decision_digest,
            residuals: expected_residuals.to_vec(),
            null_conditional_mean,
            sigma,
        });
        Ok(())
    }

    /// Consume the shared observation only when it matches the pending
    /// identity and all precommitted alternative predictions.
    pub fn observe(
        &mut self,
        observation_id: &str,
        decision_digest: &str,
        observed_residual: f64,
    ) -> Result<MixtureSequentialUpdate, SequentialError> {
        if !observed_residual.is_finite() {
            return Err(SequentialError::NonFiniteObservation);
        }
        let pending = self
            .pending
            .as_ref()
            .ok_or(SequentialError::NoPredictionPending)?;
        if pending.observation_id != observation_id {
            return Err(SequentialError::ObservationIdMismatch {
                expected: pending.observation_id.clone(),
                actual: observation_id.to_owned(),
            });
        }
        if pending.decision_digest != decision_digest {
            return Err(SequentialError::DecisionDigestMismatch);
        }

        let mut next_components = Vec::with_capacity(self.alternatives.len());
        for (discrepancy, current) in pending
            .residuals
            .iter()
            .copied()
            .zip(self.component_log_e_values.iter().copied())
        {
            let increment =
                likelihood_ratio_increment(
                    discrepancy,
                    observed_residual,
                    pending.null_conditional_mean,
                    pending.sigma,
                )?;
            let next = current + increment;
            if !next.is_finite() {
                return Err(SequentialError::NumericalOverflow);
            }
            next_components.push(next);
        }

        let weighted_logs = next_components
            .iter()
            .copied()
            .zip(self.weights.iter().copied())
            .map(|(log_e, weight)| log_e + weight.ln())
            .collect::<Vec<_>>();
        let next_log_e_value = log_sum_exp(&weighted_logs);
        if !next_log_e_value.is_finite() {
            return Err(SequentialError::NumericalOverflow);
        }
        let next_observations = self
            .observations
            .checked_add(1)
            .ok_or(SequentialError::ObservationCountOverflow)?;
        let next_maximum = self.maximum_log_e_value.max(next_log_e_value);
        let alpha_boundary = -self.alpha.ln();
        let anytime_p = anytime_p_value(next_maximum);
        let crossed = next_maximum >= alpha_boundary;

        let pending = self
            .pending
            .take()
            .expect("pending prediction was checked above");
        self.consumed_observation_ids
            .insert(pending.observation_id.clone());
        self.component_log_e_values = next_components;
        self.log_e_value = next_log_e_value;
        self.maximum_log_e_value = next_maximum;
        self.observations = next_observations;

        let components = self
            .alternatives
            .iter()
            .zip(self.weights.iter().copied())
            .zip(self.component_log_e_values.iter().copied())
            .zip(pending.residuals.iter().copied())
            .map(
                |(((alternative, weight), log_e_value), expected_residual)| {
                    MixtureComponentUpdate {
                        id: alternative.id.clone(),
                        weight,
                        expected_residual,
                        log_e_value,
                    }
                },
            )
            .collect();

        Ok(MixtureSequentialUpdate {
            observation_id: pending.observation_id,
            decision_digest: pending.decision_digest,
            alpha: self.alpha,
            null_conditional_mean: pending.null_conditional_mean,
            sigma: pending.sigma,
            observations: self.observations,
            components,
            log_e_value: self.log_e_value,
            maximum_log_e_value: self.maximum_log_e_value,
            anytime_p_value: anytime_p,
            crossed_alpha_boundary: crossed,
        })
    }

    pub fn has_pending_prediction(&self) -> bool {
        self.pending.is_some()
    }
}

fn log_sum_exp(values: &[f64]) -> f64 {
    let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    maximum
        + values
            .iter()
            .map(|value| (value - maximum).exp())
            .sum::<f64>()
            .ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_standard_normal(state: &mut u64) -> f64 {
        let mut next = || {
            *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut value = *state;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            value ^= value >> 31;
            ((value >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
        };
        let u1 = next();
        let u2 = next();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    #[test]
    fn known_gaussian_log_likelihood_increment_matches_reference() {
        let mut process = PredictableGaussianEProcess::new(0.15, 0.05).unwrap();
        process
            .commit_prediction("obs-1", "sha256:decision-1", 0.24)
            .unwrap();
        let result = process.observe("obs-1", "sha256:decision-1", 0.20).unwrap();
        assert!((result.log_e_value - 0.853_333_333_333_333_3).abs() < 1.0e-12);
        assert_eq!(result.observations, 1);
    }

    #[test]
    fn predictable_conditional_mean_innovation_matches_known_answer() {
        let mut process = PredictableGaussianEProcess::new(0.5, 0.05).unwrap();
        process
            .commit_prediction_with_conditional_gaussian(
                "obs-1",
                "sha256:decision-1",
                0.2,
                0.5,
                0.5,
            )
            .unwrap();
        let result = process.observe("obs-1", "sha256:decision-1", 0.7).unwrap();
        // d * (y - mu) / sigma^2 - d^2 / (2 sigma^2)
        // = 0.2 * 0.2 / 0.25 - 0.04 / 0.5 = 0.08.
        assert!((result.log_e_value - 0.08).abs() < 1.0e-12);
        assert_eq!(result.null_conditional_mean, 0.5);
        assert_eq!(result.expected_residual, 0.2);
        assert_eq!(result.sigma, 0.5);
    }

    #[test]
    fn predictable_ar1_null_is_controlled_but_zero_mean_misspecification_is_not() {
        const PATHS: u64 = 20_000;
        const HORIZON: u64 = 100;
        const RHO: f64 = 0.65;
        const DISCREPANCY: f64 = 0.22;
        const SIGMA: f64 = 0.80;

        let mut conditional_false_alarms = 0_u64;
        let mut zero_mean_false_alarms = 0_u64;

        for path in 0..PATHS {
            let mut state = 0x7a31_9cde_0428_b5f1_u64
                ^ path.wrapping_mul(0xd6e8_feb8_6659_fd93);
            let mut conditional =
                PredictableGaussianEProcess::new(SIGMA, 0.05).unwrap();
            let mut zero_mean = PredictableGaussianEProcess::new(SIGMA, 0.05).unwrap();
            let mut previous_observation = 0.0;
            let mut conditional_crossed = false;
            let mut zero_mean_crossed = false;

            for step in 0..HORIZON {
                // The AR(1) conditional mean is completely determined by
                // the previous observation and is committed before this draw.
                let conditional_mean = RHO * previous_observation;
                let observed = conditional_mean
                    + SIGMA * deterministic_standard_normal(&mut state);
                let conditional_id = format!("conditional-{path}-{step}");
                let zero_mean_id = format!("zero-mean-{path}-{step}");
                let conditional_digest = format!("conditional-decision-{path}-{step}");
                let zero_mean_digest = format!("zero-mean-decision-{path}-{step}");

                if !conditional_crossed {
                    conditional
                        .commit_prediction_with_conditional_gaussian(
                            &conditional_id,
                            &conditional_digest,
                            DISCREPANCY,
                            conditional_mean,
                            SIGMA,
                        )
                        .unwrap();
                    let update = conditional
                        .observe(&conditional_id, &conditional_digest, observed)
                        .unwrap();
                    conditional_crossed = update.crossed_alpha_boundary;
                }

                if !zero_mean_crossed {
                    zero_mean
                        .commit_prediction_with_sigma(
                            &zero_mean_id,
                            &zero_mean_digest,
                            DISCREPANCY,
                            SIGMA,
                        )
                        .unwrap();
                    let update = zero_mean
                        .observe(&zero_mean_id, &zero_mean_digest, observed)
                        .unwrap();
                    zero_mean_crossed = update.crossed_alpha_boundary;
                }

                previous_observation = observed;
                if conditional_crossed && zero_mean_crossed {
                    break;
                }
            }

            conditional_false_alarms += u64::from(conditional_crossed);
            zero_mean_false_alarms += u64::from(zero_mean_crossed);
        }

        let conditional_rate = conditional_false_alarms as f64 / PATHS as f64;
        let zero_mean_rate = zero_mean_false_alarms as f64 / PATHS as f64;
        assert!(
            conditional_rate <= 0.06,
            "conditional AR(1) null crossing rate {conditional_rate:.4} exceeded 0.06"
        );
        assert!(
            zero_mean_rate > 0.10,
            "zero-mean misspecification negative control was not detected: {zero_mean_rate:.4}"
        );
    }

    #[test]
    fn predictable_per_observation_sigma_is_used_in_the_likelihood_ratio() {
        let mut process = PredictableGaussianEProcess::new(0.15, 0.05).unwrap();
        process
            .commit_prediction_with_sigma("obs-1", "sha256:decision-1", 0.24, 0.15)
            .unwrap();
        let first = process.observe("obs-1", "sha256:decision-1", 0.20).unwrap();

        process
            .commit_prediction_with_sigma("obs-2", "sha256:decision-2", 0.24, 0.30)
            .unwrap();
        let second = process.observe("obs-2", "sha256:decision-2", 0.20).unwrap();

        assert!((first.log_e_value - 0.853_333_333_333_333_3).abs() < 1.0e-12);
        assert!((second.log_e_value - 1.066_666_666_666_666_7).abs() < 1.0e-12);
        assert_eq!(first.alpha, 0.05);
        assert_eq!(second.alpha, 0.05);
        assert_eq!(first.expected_residual, 0.24);
        assert_eq!(second.expected_residual, 0.24);
        assert_eq!(first.sigma, 0.15);
        assert_eq!(second.sigma, 0.30);
    }

    #[test]
    fn outcome_cannot_be_scored_before_prediction_is_committed() {
        let mut process = PredictableGaussianEProcess::new(0.15, 0.05).unwrap();
        assert_eq!(
            process.observe("obs-1", "sha256:decision-1", 0.20),
            Err(SequentialError::NoPredictionPending)
        );
        assert_eq!(process.observations, 0);
    }

    #[test]
    fn observation_must_match_committed_identity_and_digest() {
        let mut process = PredictableGaussianEProcess::new(0.15, 0.05).unwrap();
        process
            .commit_prediction("obs-1", "sha256:decision-1", 0.24)
            .unwrap();
        assert!(matches!(
            process.observe("obs-2", "sha256:decision-1", 0.20),
            Err(SequentialError::ObservationIdMismatch { .. })
        ));
        assert_eq!(
            process.observe("obs-1", "sha256:other", 0.20),
            Err(SequentialError::DecisionDigestMismatch)
        );
        assert_eq!(process.observations, 0);
        assert!(process.has_pending_prediction());
        let valid = process.observe("obs-1", "sha256:decision-1", 0.20).unwrap();
        assert_eq!(valid.observations, 1);
    }

    #[test]
    fn a_committed_prediction_cannot_be_replaced_before_observation() {
        let mut process = PredictableGaussianEProcess::new(0.15, 0.05).unwrap();
        process
            .commit_prediction("obs-1", "sha256:decision-1", 0.24)
            .unwrap();
        assert_eq!(
            process.commit_prediction("obs-2", "sha256:decision-2", 0.12),
            Err(SequentialError::PredictionAlreadyPending)
        );
        assert_eq!(process.observations, 0);
    }

    #[test]
    fn anytime_p_value_uses_the_running_maximum_not_current_evidence() {
        let mut process = PredictableGaussianEProcess::new(0.15, 0.05).unwrap();
        process
            .commit_prediction("obs-1", "sha256:decision-1", 0.24)
            .unwrap();
        let first = process.observe("obs-1", "sha256:decision-1", 0.50).unwrap();
        assert!(first.crossed_alpha_boundary);
        assert!(first.anytime_p_value <= 0.05);

        process
            .commit_prediction("obs-2", "sha256:decision-2", 0.24)
            .unwrap();
        let second = process
            .observe("obs-2", "sha256:decision-2", -0.50)
            .unwrap();
        assert!(second.log_e_value < first.log_e_value);
        assert_eq!(second.maximum_log_e_value, first.maximum_log_e_value);
        assert!(second.crossed_alpha_boundary);
        assert!(second.anytime_p_value <= 0.05);
    }

    #[test]
    fn predictable_action_can_depend_on_previous_observation() {
        let mut process = PredictableGaussianEProcess::new(0.15, 0.05).unwrap();
        process
            .commit_prediction("obs-1", "sha256:decision-1", 0.10)
            .unwrap();
        let first = process.observe("obs-1", "sha256:decision-1", 0.20).unwrap();
        let next_discrepancy = if first.log_e_value > 0.0 { 0.20 } else { -0.20 };
        process
            .commit_prediction("obs-2", "sha256:decision-2", next_discrepancy)
            .unwrap();
        let second = process.observe("obs-2", "sha256:decision-2", 0.25).unwrap();
        assert_eq!(second.observations, 2);
        assert!(second.log_e_value.is_finite());
    }

    #[test]
    fn nonfinite_values_fail_without_consuming_pending_prediction() {
        let mut process = PredictableGaussianEProcess::new(0.15, 0.05).unwrap();
        process
            .commit_prediction("obs-1", "sha256:decision-1", 0.24)
            .unwrap();
        assert_eq!(
            process.observe("obs-1", "sha256:decision-1", f64::NAN),
            Err(SequentialError::NonFiniteObservation)
        );
        assert!(process.has_pending_prediction());
        assert_eq!(process.observations, 0);
    }

    #[test]
    fn adaptive_heteroskedastic_null_stays_within_anytime_boundary_smoke_rate() {
        const PATHS: u64 = 20_000;
        const HORIZON: u64 = 100;
        let mut state = 0x5eed_cafe_d00d_f00d_u64;
        let mut false_alarms = 0_u64;

        for path in 0..PATHS {
            let mut process = PredictableGaussianEProcess::new(1.0, 0.05).unwrap();
            let mut previous_observation = 0.0;
            let mut crossed = false;

            for step in 0..HORIZON {
                // Both the discrepancy and measurement precision adapt only
                // to the previous observation, never the current draw.
                let (discrepancy, sigma) = if previous_observation >= 0.0 {
                    (0.25, 0.7)
                } else {
                    (-0.25, 1.3)
                };
                let observed = sigma * deterministic_standard_normal(&mut state);
                let observation_id = format!("path-{path}-step-{step}");
                let decision_digest = format!("decision-{path}-{step}");
                process
                    .commit_prediction_with_sigma(
                        &observation_id,
                        &decision_digest,
                        discrepancy,
                        sigma,
                    )
                    .unwrap();
                let update = process
                    .observe(&observation_id, &decision_digest, observed)
                    .unwrap();
                if update.crossed_alpha_boundary {
                    crossed = true;
                    break;
                }
                previous_observation = observed;
            }
            if crossed {
                false_alarms += 1;
            }
        }

        let empirical_rate = false_alarms as f64 / PATHS as f64;
        assert!(
            empirical_rate <= 0.06,
            "empirical anytime-null crossing rate {empirical_rate:.4} exceeded the 0.06 smoke bound"
        );
    }

    #[test]
    fn fixed_mixture_matches_weighted_component_e_values() {
        let mut mixture = PredictableGaussianMixture::new(
            0.50,
            0.05,
            vec![
                WeightedAlternative {
                    id: "positive".to_owned(),
                    weight: 0.5,
                },
                WeightedAlternative {
                    id: "negative".to_owned(),
                    weight: 0.5,
                },
            ],
        )
        .unwrap();
        mixture
            .commit_predictions("obs-1", "sha256:decision-1", &[0.20, -0.20])
            .unwrap();
        let result = mixture.observe("obs-1", "sha256:decision-1", 0.30).unwrap();
        let expected = (0.5_f64 * 0.16_f64.exp() + 0.5_f64 * (-0.32_f64).exp()).ln();
        assert!((result.log_e_value - expected).abs() < 1.0e-12);
        assert_eq!(result.alpha, 0.05);
        assert_eq!(result.sigma, 0.50);
        assert_eq!(result.components[0].expected_residual, 0.20);
        assert_eq!(result.components[1].expected_residual, -0.20);
        assert_eq!(result.components.len(), 2);
        assert_eq!(result.observations, 1);
    }

    #[test]
    fn fixed_mixture_rejects_bad_weights_and_duplicate_ids() {
        let zero_weight = PredictableGaussianMixture::new(
            0.5,
            0.05,
            vec![WeightedAlternative {
                id: "a".to_owned(),
                weight: 0.0,
            }],
        );
        assert_eq!(
            zero_weight.err(),
            Some(SequentialError::InvalidAlternativeWeight)
        );

        let duplicate = PredictableGaussianMixture::new(
            0.5,
            0.05,
            vec![
                WeightedAlternative {
                    id: "a".to_owned(),
                    weight: 0.5,
                },
                WeightedAlternative {
                    id: "a".to_owned(),
                    weight: 0.5,
                },
            ],
        );
        assert_eq!(
            duplicate.err(),
            Some(SequentialError::DuplicateAlternativeId("a".to_owned()))
        );
    }

    #[test]
    fn fixed_mixture_rejects_wrong_prediction_count_without_mutation() {
        let mut mixture = PredictableGaussianMixture::new(
            0.5,
            0.05,
            vec![
                WeightedAlternative {
                    id: "a".to_owned(),
                    weight: 0.5,
                },
                WeightedAlternative {
                    id: "b".to_owned(),
                    weight: 0.5,
                },
            ],
        )
        .unwrap();
        assert_eq!(
            mixture.commit_predictions("obs-1", "sha256:decision-1", &[0.2]),
            Err(SequentialError::AlternativeCountMismatch {
                expected: 2,
                actual: 1
            })
        );
        assert!(!mixture.has_pending_prediction());
    }

    #[test]
    fn fixed_mixture_requires_prediction_before_observation() {
        let mut mixture = PredictableGaussianMixture::new(
            0.5,
            0.05,
            vec![WeightedAlternative {
                id: "a".to_owned(),
                weight: 1.0,
            }],
        )
        .unwrap();
        assert_eq!(
            mixture.observe("obs-1", "sha256:decision-1", 0.3),
            Err(SequentialError::NoPredictionPending)
        );
    }
}
