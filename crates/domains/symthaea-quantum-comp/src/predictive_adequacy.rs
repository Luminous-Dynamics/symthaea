//! Held-out predictive adequacy checks before experiment selection.
//!
//! Compare a declared categorical predictive distribution to held-out outcome
//! counts. Report empirical discrepancy, Brier score, log loss, and a finite-
//! sample total-variation bound only under an explicit IID categorical model.
//! Passing this check does not prove a physical model is globally correct.

use std::error::Error;
use std::fmt;

const PROBABILITY_SUM_TOLERANCE: f64 = 1.0e-9;
const LN_2: f64 = std::f64::consts::LN_2;

/// Provenance class for the observations used to evaluate the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationSource {
    /// Outcomes measured on a physical system with raw evidence retained.
    MeasuredPhysical,
    /// Outcomes from an independently identified simulator.
    IndependentSimulator,
    /// An analytic or exact reference generator.
    AnalyticReference,
    /// Synthetic fixture used to exercise the checker.
    SyntheticFixture,
}

/// Whether the concentration bound's sampling assumption is defensible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingAssumption {
    /// Observations are independent draws from one categorical distribution.
    IidCategorical,
    /// Sampling is correlated, adaptive, shifted, or otherwise not justified as IID.
    UnknownOrNonIid,
}

/// Frozen prediction and held-out categorical outcome counts.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictiveObservationSet {
    model_id: String,
    model_version: String,
    workload_fingerprint: String,
    outcomes: Vec<String>,
    predicted_probabilities: Vec<f64>,
    observed_counts: Vec<u64>,
    total_observations: u64,
    source: ObservationSource,
    sampling_assumption: SamplingAssumption,
}

impl PredictiveObservationSet {
    /// Create a validated predictive-evaluation record.
    ///
    /// Outcome labels are ordered and must match the frozen evaluation
    /// protocol. Predicted probabilities are never silently renormalized.
    pub fn new(
        model_id: impl Into<String>,
        model_version: impl Into<String>,
        workload_fingerprint: impl Into<String>,
        outcomes: Vec<String>,
        predicted_probabilities: Vec<f64>,
        observed_counts: Vec<u64>,
        source: ObservationSource,
        sampling_assumption: SamplingAssumption,
    ) -> Result<Self, PredictiveAdequacyError> {
        let model_id = model_id.into();
        let model_version = model_version.into();
        let workload_fingerprint = workload_fingerprint.into();
        if model_id.trim().is_empty()
            || model_version.trim().is_empty()
            || workload_fingerprint.trim().is_empty()
        {
            return Err(PredictiveAdequacyError::EmptyMetadata);
        }
        validate_labels(&outcomes)?;
        if outcomes.len() != predicted_probabilities.len()
            || outcomes.len() != observed_counts.len()
        {
            return Err(PredictiveAdequacyError::DimensionMismatch);
        }
        validate_distribution(&predicted_probabilities)?;
        let mut total_observations = 0_u64;
        for count in &observed_counts {
            total_observations = total_observations
                .checked_add(*count)
                .ok_or(PredictiveAdequacyError::ObservationCountOverflow)?;
        }
        Ok(Self {
            model_id,
            model_version,
            workload_fingerprint,
            outcomes,
            predicted_probabilities,
            observed_counts,
            total_observations,
            source,
            sampling_assumption,
        })
    }

    /// Stable model identifier.
    pub fn model_id(&self) -> &str { &self.model_id }

    /// Model version or immutable artifact identifier.
    pub fn model_version(&self) -> &str { &self.model_version }

    /// Workload/scenario fingerprint.
    pub fn workload_fingerprint(&self) -> &str { &self.workload_fingerprint }

    /// Provenance class for holdout observations.
    pub fn source(&self) -> ObservationSource { self.source }

    /// Number of held-out outcomes represented by the counts.
    pub fn total_observations(&self) -> u64 { self.total_observations }

    /// Frozen outcome names in model probability order.
    pub fn outcomes(&self) -> &[String] { &self.outcomes }
}

/// Predeclared thresholds for the three-way adequacy gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictiveAdequacyPolicy {
    min_observations: u64,
    confidence_level: f64,
    max_total_variation_upper_bound: f64,
    max_brier_score: f64,
    max_mean_log_loss: Option<f64>,
}

impl PredictiveAdequacyPolicy {
    /// Construct a preregistered acceptance policy.
    ///
    /// The multiclass Brier score is unscaled and lies in [0, 2].
    /// The total-variation bound lies in [0, 1]. A missing log-loss threshold
    /// disables that threshold but never disables zero-probability rejection.
    pub fn new(
        min_observations: u64,
        confidence_level: f64,
        max_total_variation_upper_bound: f64,
        max_brier_score: f64,
        max_mean_log_loss: Option<f64>,
    ) -> Result<Self, PredictiveAdequacyError> {
        if min_observations == 0
            || !confidence_level.is_finite()
            || !(0.0..1.0).contains(&confidence_level)
            || !max_total_variation_upper_bound.is_finite()
            || !(0.0..=1.0).contains(&max_total_variation_upper_bound)
            || !max_brier_score.is_finite()
            || !(0.0..=2.0).contains(&max_brier_score)
            || max_mean_log_loss.is_some_and(|x| !x.is_finite() || x < 0.0)
        {
            return Err(PredictiveAdequacyError::InvalidPolicy);
        }
        Ok(Self {
            min_observations,
            confidence_level,
            max_total_variation_upper_bound,
            max_brier_score,
            max_mean_log_loss,
        })
    }

    /// Minimum held-out sample count.
    pub fn min_observations(&self) -> u64 { self.min_observations }

    /// Declared confidence level for the finite-sample radius.
    pub fn confidence_level(&self) -> f64 { self.confidence_level }
}

/// Log-loss result, with impossible observed events represented explicitly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PredictiveLogLoss {
    /// Finite average negative log likelihood in nats per observation.
    Finite(f64),
    /// An observed category had model probability zero.
    InfiniteDueToZeroProbability,
}

/// Three-way result of the preregistered predictive-adequacy policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictiveAdequacyStatus {
    /// All required evidence and predeclared metric thresholds were satisfied.
    Pass,
    /// Sufficient/valid evidence shows one or more thresholds failed.
    Reject,
    /// Evidence is too small or sampling assumptions do not justify inference.
    InsufficientEvidence,
}

/// Diagnostic report; input predictions and counts are retained.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictiveAdequacyReport {
    /// Model identifier.
    pub model_id: String,
    /// Model version.
    pub model_version: String,
    /// Workload/scenario fingerprint.
    pub workload_fingerprint: String,
    /// Holdout evidence provenance.
    pub source: ObservationSource,
    /// Frozen outcome order.
    pub outcomes: Vec<String>,
    /// Model-predicted probability for each outcome.
    pub predicted_probabilities: Vec<f64>,
    /// Observed count for each outcome.
    pub observed_counts: Vec<u64>,
    /// Total held-out count.
    pub total_observations: u64,
    /// Empirical outcome frequencies.
    pub empirical_frequencies: Vec<f64>,
    /// Total-variation distance between predicted and empirical distributions.
    pub observed_total_variation: Option<f64>,
    /// Finite-sample radius under IID categorical sampling; none otherwise.
    pub total_variation_sampling_radius: Option<f64>,
    /// Upper bound min(1, observed TV + sampling radius), when justified.
    pub total_variation_upper_bound: Option<f64>,
    /// Mean unscaled multiclass Brier score.
    pub brier_score: Option<f64>,
    /// Mean log loss, including explicit impossible-event diagnostic.
    pub mean_log_loss: Option<PredictiveLogLoss>,
    /// Whether any observed outcome had predicted probability zero.
    pub observed_zero_probability_event: bool,
    /// Three-way result from the predeclared policy.
    pub status: PredictiveAdequacyStatus,
    /// Deterministic diagnostic reasons.
    pub reasons: Vec<String>,
}

/// Validation and calculation errors in predictive adequacy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictiveAdequacyError {
    /// Required model, version, or workload metadata is blank.
    EmptyMetadata,
    /// Outcome labels must be unique and non-empty.
    InvalidOutcomeLabels,
    /// Outcome, probability, and count dimensions differ.
    DimensionMismatch,
    /// Predicted probabilities do not form a valid distribution.
    InvalidProbabilityDistribution,
    /// Total held-out count overflowed u64.
    ObservationCountOverflow,
    /// The preregistered policy is invalid.
    InvalidPolicy,
    /// A derived metric became non-finite.
    DerivedMetricOverflow,
}

impl fmt::Display for PredictiveAdequacyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::EmptyMetadata => "model/version/workload metadata must be non-empty",
            Self::InvalidOutcomeLabels => "outcome labels must be unique and non-empty",
            Self::DimensionMismatch => "outcome, probability and count dimensions must match",
            Self::InvalidProbabilityDistribution => "predicted probabilities are invalid",
            Self::ObservationCountOverflow => "held-out observation count overflowed",
            Self::InvalidPolicy => "predictive adequacy policy is invalid",
            Self::DerivedMetricOverflow => "a derived predictive metric became non-finite",
        };
        f.write_str(text)
    }
}

impl Error for PredictiveAdequacyError {}

/// Assess a frozen model against held-out observations under a declared policy.
///
/// Under IID categorical observations, the Weissman-style bound uses
/// P(TV(empirical, true) >= r) <= (2^K - 2) exp(-2 n r^2).
/// The model-vs-true TV upper bound is min(1, observed TV + sampling radius).
/// This result only applies to the specified holdout and assumptions.
pub fn assess_predictive_adequacy(
    data: &PredictiveObservationSet,
    policy: PredictiveAdequacyPolicy,
) -> Result<PredictiveAdequacyReport, PredictiveAdequacyError> {
    let n = data.total_observations;
    let k = data.outcomes.len();
    let predicted = &data.predicted_probabilities;
    let counts = &data.observed_counts;

    let empirical_frequencies = if n == 0 {
        vec![0.0; k]
    } else {
        counts.iter().map(|c| *c as f64 / n as f64).collect::<Vec<_>>()
    };

    let (observed_tv, brier, log_loss, zero_event) = if n == 0 {
        (None, None, None, false)
    } else {
        let tv = 0.5 * predicted.iter().zip(&empirical_frequencies)
            .map(|(p, q)| (p - q).abs()).sum::<f64>();
        let n_f64 = n as f64;
        let mut brier_sum = 0.0;
        let mut log_loss_sum = 0.0;
        let mut zero_probability_event = false;

        for (i, p) in predicted.iter().enumerate() {
            let count = counts[i] as f64;
            brier_sum += count * (1.0 - p).powi(2) + (n_f64 - count) * p.powi(2);
            if counts[i] > 0 {
                if *p == 0.0 {
                    zero_probability_event = true;
                } else {
                    log_loss_sum -= count * p.ln();
                }
            }
        }

        let brier = brier_sum / n_f64;
        let log_loss = if zero_probability_event {
            PredictiveLogLoss::InfiniteDueToZeroProbability
        } else {
            PredictiveLogLoss::Finite(log_loss_sum / n_f64)
        };
        if !tv.is_finite()
            || !brier.is_finite()
            || matches!(log_loss, PredictiveLogLoss::Finite(x) if !x.is_finite())
        {
            return Err(PredictiveAdequacyError::DerivedMetricOverflow);
        }
        (Some(tv), Some(brier), Some(log_loss), zero_probability_event)
    };

    let (sampling_radius, tv_upper_bound) =
        match (data.sampling_assumption, n, k) {
            (SamplingAssumption::IidCategorical, n, k) if n > 0 && k >= 2 => {
                let kf = k as f64;
                let log_cardinality = kf * LN_2 + (-2.0 * (-(kf * LN_2)).exp()).ln_1p();
                let log_inverse_alpha = -((1.0 - policy.confidence_level).ln());
                let radius = ((log_cardinality + log_inverse_alpha) / (2.0 * n as f64))
                    .sqrt().min(1.0);
                let upper = (observed_tv.unwrap_or(0.0) + radius).min(1.0);
                (Some(radius), Some(upper))
            }
            (SamplingAssumption::IidCategorical, n, 1) if n > 0 => (Some(0.0), Some(0.0)),
            _ => (None, None),
        };

    let mut reasons = Vec::new();
    let status = if n < policy.min_observations {
        reasons.push(format!("insufficient held-out observations: {} < {}", n, policy.min_observations));
        PredictiveAdequacyStatus::InsufficientEvidence
    } else if data.sampling_assumption != SamplingAssumption::IidCategorical {
        reasons.push("finite-sample bound unavailable: IID categorical assumption not established".into());
        PredictiveAdequacyStatus::InsufficientEvidence
    } else if n == 0 {
        reasons.push("no held-out outcomes were observed".into());
        PredictiveAdequacyStatus::InsufficientEvidence
    } else if zero_event {
        reasons.push("held-out data contains an outcome assigned probability zero by the model".into());
        PredictiveAdequacyStatus::Reject
    } else {
        let mut failed = false;
        if tv_upper_bound.unwrap_or(1.0) > policy.max_total_variation_upper_bound {
            reasons.push("total-variation upper bound exceeds predeclared threshold".into());
            failed = true;
        }
        if brier.unwrap_or(f64::INFINITY) > policy.max_brier_score {
            reasons.push("Brier score exceeds predeclared threshold".into());
            failed = true;
        }
        if let (Some(maximum), Some(PredictiveLogLoss::Finite(value))) =
            (policy.max_mean_log_loss, log_loss)
        {
            if value > maximum {
                reasons.push("mean log loss exceeds predeclared threshold".into());
                failed = true;
            }
        }
        if failed {
            PredictiveAdequacyStatus::Reject
        } else {
            reasons.push("all predeclared thresholds satisfied on this holdout".into());
            PredictiveAdequacyStatus::Pass
        }
    };

    Ok(PredictiveAdequacyReport {
        model_id: data.model_id.clone(),
        model_version: data.model_version.clone(),
        workload_fingerprint: data.workload_fingerprint.clone(),
        source: data.source,
        outcomes: data.outcomes.clone(),
        predicted_probabilities: predicted.clone(),
        observed_counts: counts.clone(),
        total_observations: n,
        empirical_frequencies,
        observed_total_variation: observed_tv,
        total_variation_sampling_radius: sampling_radius,
        total_variation_upper_bound: tv_upper_bound,
        brier_score: brier,
        mean_log_loss: log_loss,
        observed_zero_probability_event: zero_event,
        status,
        reasons,
    })
}

fn validate_labels(labels: &[String]) -> Result<(), PredictiveAdequacyError> {
    if labels.is_empty() {
        return Err(PredictiveAdequacyError::InvalidOutcomeLabels);
    }
    for (i, label) in labels.iter().enumerate() {
        if label.trim().is_empty() || labels[..i].iter().any(|x| x == label) {
            return Err(PredictiveAdequacyError::InvalidOutcomeLabels);
        }
    }
    Ok(())
}

fn validate_distribution(probabilities: &[f64]) -> Result<(), PredictiveAdequacyError> {
    if probabilities.is_empty()
        || probabilities.iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        || (probabilities.iter().sum::<f64>() - 1.0).abs() > PROBABILITY_SUM_TOLERANCE
    {
        return Err(PredictiveAdequacyError::InvalidProbabilityDistribution);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(
        probabilities: &[f64],
        counts: &[u64],
        sampling: SamplingAssumption,
        source: ObservationSource,
    ) -> PredictiveObservationSet {
        PredictiveObservationSet::new(
            "model-A", "fixture-sha", "workload-1",
            (0..probabilities.len()).map(|i| format!("o{i}")).collect(),
            probabilities.to_vec(), counts.to_vec(), source, sampling,
        ).unwrap()
    }

    fn policy(min: u64, tv: f64, brier: f64, log: Option<f64>) -> PredictiveAdequacyPolicy {
        PredictiveAdequacyPolicy::new(min, 0.95, tv, brier, log).unwrap()
    }

    #[test]
    fn matching_prediction_has_expected_metrics() {
        let input = data(&[0.5, 0.5], &[50, 50], SamplingAssumption::IidCategorical, ObservationSource::AnalyticReference);
        let report = assess_predictive_adequacy(&input, policy(100, 0.25, 0.6, Some(1.0))).unwrap();
        assert_eq!(report.observed_total_variation, Some(0.0));
        assert!((report.brier_score.unwrap() - 0.5).abs() < 1e-12);
        assert!(report.total_variation_sampling_radius.unwrap() > 0.0);
        assert_eq!(report.status, PredictiveAdequacyStatus::Pass);
    }

    #[test]
    fn systematic_prediction_bias_is_rejected() {
        let input = data(&[0.9, 0.1], &[10, 90], SamplingAssumption::IidCategorical, ObservationSource::MeasuredPhysical);
        let report = assess_predictive_adequacy(&input, policy(100, 0.25, 0.6, Some(1.0))).unwrap();
        assert!((report.observed_total_variation.unwrap() - 0.8).abs() < 1e-12);
        assert_eq!(report.status, PredictiveAdequacyStatus::Reject);
    }

    #[test]
    fn observed_zero_probability_event_is_explicit_and_rejected() {
        let input = data(&[1.0, 0.0], &[99, 1], SamplingAssumption::IidCategorical, ObservationSource::MeasuredPhysical);
        let report = assess_predictive_adequacy(&input, policy(100, 1.0, 2.0, None)).unwrap();
        assert_eq!(report.mean_log_loss, Some(PredictiveLogLoss::InfiniteDueToZeroProbability));
        assert!(report.observed_zero_probability_event);
        assert_eq!(report.status, PredictiveAdequacyStatus::Reject);
    }

    #[test]
    fn too_few_observations_mean_insufficient_not_pass() {
        let input = data(&[0.5, 0.5], &[4, 6], SamplingAssumption::IidCategorical, ObservationSource::SyntheticFixture);
        let report = assess_predictive_adequacy(&input, policy(100, 1.0, 2.0, None)).unwrap();
        assert_eq!(report.status, PredictiveAdequacyStatus::InsufficientEvidence);
    }

    #[test]
    fn non_iid_sampling_has_no_iid_confidence_radius() {
        let input = data(&[0.5, 0.5], &[50, 50], SamplingAssumption::UnknownOrNonIid, ObservationSource::MeasuredPhysical);
        let report = assess_predictive_adequacy(&input, policy(100, 1.0, 2.0, None)).unwrap();
        assert_eq!(report.total_variation_sampling_radius, None);
        assert_eq!(report.status, PredictiveAdequacyStatus::InsufficientEvidence);
    }

    #[test]
    fn invalid_probabilities_duplicate_labels_and_count_overflow_fail_closed() {
        assert_eq!(
            PredictiveObservationSet::new("m", "v", "w", vec!["x".into(), "y".into()],
                vec![0.4, 0.4], vec![1, 1], ObservationSource::SyntheticFixture,
                SamplingAssumption::IidCategorical),
            Err(PredictiveAdequacyError::InvalidProbabilityDistribution)
        );
        assert_eq!(
            PredictiveObservationSet::new("m", "v", "w", vec!["x".into(), "x".into()],
                vec![0.5, 0.5], vec![1, 1], ObservationSource::SyntheticFixture,
                SamplingAssumption::IidCategorical),
            Err(PredictiveAdequacyError::InvalidOutcomeLabels)
        );
        assert_eq!(
            PredictiveObservationSet::new("m", "v", "w", vec!["x".into(), "y".into()],
                vec![0.5, 0.5], vec![u64::MAX, 1], ObservationSource::SyntheticFixture,
                SamplingAssumption::IidCategorical),
            Err(PredictiveAdequacyError::ObservationCountOverflow)
        );
    }

    #[test]
    fn one_category_has_zero_tv_and_exact_model_agreement() {
        let input = data(&[1.0], &[100], SamplingAssumption::IidCategorical, ObservationSource::AnalyticReference);
        let report = assess_predictive_adequacy(&input, policy(100, 0.0, 0.0, Some(0.0))).unwrap();
        assert_eq!(report.observed_total_variation, Some(0.0));
        assert_eq!(report.total_variation_sampling_radius, Some(0.0));
        assert_eq!(report.status, PredictiveAdequacyStatus::Pass);
    }
}
