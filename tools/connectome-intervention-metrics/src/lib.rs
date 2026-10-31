//! Independent paired-intervention metrics for connectome benchmarks.
//!
//! Treatment and control observations remain paired. Point metrics report both
//! pooled timepoint results and equal-weight cluster-macro results. Bootstrap
//! inference resamples whole independent clusters (normally animal/session).
//! This crate scores predictions; it is not a causal estimator.

use core::fmt;
use std::collections::{BTreeMap, HashMap, HashSet};

const CI_LOW: f64 = 0.025;
const CI_HIGH: f64 = 0.975;

#[derive(Debug, Clone, PartialEq)]
pub struct PairedOutcome {
    pub sample_id: String,
    pub cluster_id: String,
    pub observed_treatment: Vec<f64>,
    pub observed_control: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelPrediction {
    pub sample_id: String,
    pub predicted_treatment: Vec<f64>,
    pub predicted_control: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EffectIntervalPrediction {
    pub sample_id: String,
    pub lower_effect: Vec<f64>,
    pub upper_effect: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InterventionMetricSummary {
    pub sample_count: usize,
    pub cluster_count: usize,
    pub timepoint_count: usize,
    pub pooled_effect_rmse: f64,
    pub pooled_effect_mae: f64,
    pub cluster_macro_effect_rmse: f64,
    pub cluster_macro_effect_mae: f64,
    pub pooled_response_sign_accuracy: f64,
    pub cluster_macro_response_sign_accuracy: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntervalMetricSummary {
    pub sample_count: usize,
    pub cluster_count: usize,
    pub timepoint_count: usize,
    pub pooled_coverage: f64,
    pub cluster_macro_coverage: f64,
    pub pooled_mean_width: f64,
    pub cluster_macro_mean_width: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusterBootstrapComparison {
    pub cluster_count: usize,
    pub bootstrap_replicates: usize,
    pub seed: u64,
    /// Positive means the candidate has lower per-cluster RMSE.
    pub mean_rmse_improvement: f64,
    pub percentile_95_ci_low: f64,
    pub percentile_95_ci_high: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetricError {
    EmptyOutcomeSet,
    EmptySampleId,
    EmptyClusterId { sample_id: String },
    DuplicateSampleId { sample_id: String },
    DuplicatePredictionId { sample_id: String },
    MissingPrediction { sample_id: String },
    UnexpectedPrediction { sample_id: String },
    EmptyTrajectory { sample_id: String },
    ShapeMismatch {
        sample_id: String,
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    NonFiniteValue { sample_id: String, field: &'static str, index: usize },
    InvalidSignTolerance { tolerance: f64 },
    InvalidInterval { sample_id: String, index: usize, lower: f64, upper: f64 },
    TooFewIndependentClusters { minimum: usize, actual: usize },
    InvalidBootstrapReplicates { replicates: usize },
    ZeroBootstrapSeed,
    NumericalOverflow { sample_id: String, field: &'static str },
}

impl fmt::Display for MetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyOutcomeSet => write!(f, "outcome set must not be empty"),
            Self::EmptySampleId => write!(f, "sample id must not be empty"),
            Self::EmptyClusterId { sample_id } => write!(f, "sample {sample_id:?} has empty cluster id"),
            Self::DuplicateSampleId { sample_id } => write!(f, "duplicate outcome sample id {sample_id:?}"),
            Self::DuplicatePredictionId { sample_id } => write!(f, "duplicate prediction sample id {sample_id:?}"),
            Self::MissingPrediction { sample_id } => write!(f, "missing prediction for sample {sample_id:?}"),
            Self::UnexpectedPrediction { sample_id } => write!(f, "prediction exists for unknown sample {sample_id:?}"),
            Self::EmptyTrajectory { sample_id } => write!(f, "sample {sample_id:?} contains an empty trajectory"),
            Self::ShapeMismatch { sample_id, field, expected, actual } => write!(
                f, "sample {sample_id:?} field {field} has length {actual}; expected {expected}"
            ),
            Self::NonFiniteValue { sample_id, field, index } => write!(
                f, "sample {sample_id:?} field {field} is non-finite at index {index}"
            ),
            Self::InvalidSignTolerance { tolerance } => write!(f, "invalid sign tolerance {tolerance}"),
            Self::InvalidInterval { sample_id, index, lower, upper } => write!(
                f, "sample {sample_id:?} interval {index} is invalid: [{lower}, {upper}]"
            ),
            Self::TooFewIndependentClusters { minimum, actual } => write!(
                f, "at least {minimum} independent clusters are required; received {actual}"
            ),
            Self::InvalidBootstrapReplicates { replicates } => write!(
                f, "at least 100 bootstrap replicates are required; received {replicates}"
            ),
            Self::ZeroBootstrapSeed => write!(f, "bootstrap seed must be non-zero"),
            Self::NumericalOverflow { sample_id, field } => write!(
                f, "numerical overflow for sample {sample_id:?}, field {field}"
            ),
        }
    }
}
impl std::error::Error for MetricError {}

#[derive(Debug, Clone, Copy, Default)]
struct ErrorAccumulator {
    squared_error_sum: f64,
    absolute_error_sum: f64,
    sign_correct: usize,
    count: usize,
}

impl ErrorAccumulator {
    fn add(
        &mut self,
        sample_id: &str,
        observed: f64,
        predicted: f64,
        tolerance: f64,
    ) -> Result<(), MetricError> {
        let error = predicted - observed;
        let square = error * error;
        if !error.is_finite() || !square.is_finite() {
            return Err(MetricError::NumericalOverflow {
                sample_id: sample_id.to_owned(),
                field: "effect_error",
            });
        }
        self.squared_error_sum += square;
        self.absolute_error_sum += error.abs();
        if !self.squared_error_sum.is_finite() || !self.absolute_error_sum.is_finite() {
            return Err(MetricError::NumericalOverflow {
                sample_id: sample_id.to_owned(),
                field: "error_accumulator",
            });
        }
        if sign_class(observed, tolerance) == sign_class(predicted, tolerance) {
            self.sign_correct += 1;
        }
        self.count += 1;
        Ok(())
    }

    fn rmse(self) -> f64 {
        (self.squared_error_sum / self.count as f64).sqrt()
    }
    fn mae(self) -> f64 {
        self.absolute_error_sum / self.count as f64
    }
    fn sign_accuracy(self) -> f64 {
        self.sign_correct as f64 / self.count as f64
    }
}

fn sign_class(value: f64, tolerance: f64) -> i8 {
    if value > tolerance { 1 } else if value < -tolerance { -1 } else { 0 }
}

fn validate_finite(
    sample_id: &str,
    field: &'static str,
    values: &[f64],
) -> Result<(), MetricError> {
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(MetricError::NonFiniteValue {
                sample_id: sample_id.to_owned(),
                field,
                index,
            });
        }
    }
    Ok(())
}

fn validate_outcomes(outcomes: &[PairedOutcome]) -> Result<(), MetricError> {
    if outcomes.is_empty() {
        return Err(MetricError::EmptyOutcomeSet);
    }
    let mut ids = HashSet::with_capacity(outcomes.len());
    for outcome in outcomes {
        if outcome.sample_id.trim().is_empty() {
            return Err(MetricError::EmptySampleId);
        }
        if !ids.insert(outcome.sample_id.as_str()) {
            return Err(MetricError::DuplicateSampleId {
                sample_id: outcome.sample_id.clone(),
            });
        }
        if outcome.cluster_id.trim().is_empty() {
            return Err(MetricError::EmptyClusterId {
                sample_id: outcome.sample_id.clone(),
            });
        }
        if outcome.observed_treatment.is_empty() {
            return Err(MetricError::EmptyTrajectory {
                sample_id: outcome.sample_id.clone(),
            });
        }
        let horizon = outcome.observed_treatment.len();
        if outcome.observed_control.len() != horizon {
            return Err(MetricError::ShapeMismatch {
                sample_id: outcome.sample_id.clone(),
                field: "observed_control",
                expected: horizon,
                actual: outcome.observed_control.len(),
            });
        }
        validate_finite(&outcome.sample_id, "observed_treatment", &outcome.observed_treatment)?;
        validate_finite(&outcome.sample_id, "observed_control", &outcome.observed_control)?;
    }
    Ok(())
}

fn align_predictions<'a>(
    outcomes: &[PairedOutcome],
    predictions: &'a [ModelPrediction],
) -> Result<Vec<&'a ModelPrediction>, MetricError> {
    let mut by_id = HashMap::with_capacity(predictions.len());
    for prediction in predictions {
        if prediction.sample_id.trim().is_empty() {
            return Err(MetricError::EmptySampleId);
        }
        if by_id.insert(prediction.sample_id.as_str(), prediction).is_some() {
            return Err(MetricError::DuplicatePredictionId {
                sample_id: prediction.sample_id.clone(),
            });
        }
    }

    if predictions.len() != outcomes.len() {
        for prediction in predictions {
            if !outcomes.iter().any(|outcome| outcome.sample_id == prediction.sample_id) {
                return Err(MetricError::UnexpectedPrediction {
                    sample_id: prediction.sample_id.clone(),
                });
            }
        }
        for outcome in outcomes {
            if !by_id.contains_key(outcome.sample_id.as_str()) {
                return Err(MetricError::MissingPrediction {
                    sample_id: outcome.sample_id.clone(),
                });
            }
        }
    }

    let mut aligned = Vec::with_capacity(outcomes.len());
    for outcome in outcomes {
        let prediction = by_id.get(outcome.sample_id.as_str()).copied()
            .ok_or_else(|| MetricError::MissingPrediction {
                sample_id: outcome.sample_id.clone(),
            })?;
        let horizon = outcome.observed_treatment.len();
        for (field, values) in [
            ("predicted_treatment", prediction.predicted_treatment.as_slice()),
            ("predicted_control", prediction.predicted_control.as_slice()),
        ] {
            if values.len() != horizon {
                return Err(MetricError::ShapeMismatch {
                    sample_id: outcome.sample_id.clone(),
                    field,
                    expected: horizon,
                    actual: values.len(),
                });
            }
            validate_finite(&outcome.sample_id, field, values)?;
        }
        aligned.push(prediction);
    }
    Ok(aligned)
}

fn effect_trace(
    sample_id: &str,
    treatment: &[f64],
    control: &[f64],
) -> Result<Vec<f64>, MetricError> {
    let mut effects = Vec::with_capacity(treatment.len());
    for (&treated, &controlled) in treatment.iter().zip(control) {
        let value = treated - controlled;
        if !value.is_finite() {
            return Err(MetricError::NumericalOverflow {
                sample_id: sample_id.to_owned(),
                field: "treatment_minus_control",
            });
        }
        effects.push(value);
    }
    Ok(effects)
}

/// Score complete paired treatment-minus-control trajectories.
/// Pooled metrics weight time points equally; cluster-macro metrics weight
/// each independent cluster equally. Tolerance defines a three-class sign
/// dead zone: positive, negative, or approximately null.
pub fn score_intervention_effects(
    outcomes: &[PairedOutcome],
    predictions: &[ModelPrediction],
    sign_tolerance: f64,
) -> Result<InterventionMetricSummary, MetricError> {
    if !sign_tolerance.is_finite() || sign_tolerance < 0.0 {
        return Err(MetricError::InvalidSignTolerance { tolerance: sign_tolerance });
    }
    validate_outcomes(outcomes)?;
    let aligned = align_predictions(outcomes, predictions)?;

    let mut pooled = ErrorAccumulator::default();
    let mut clusters: BTreeMap<&str, ErrorAccumulator> = BTreeMap::new();
    let mut timepoint_count = 0usize;

    for (outcome, prediction) in outcomes.iter().zip(aligned) {
        let observed = effect_trace(
            &outcome.sample_id,
            &outcome.observed_treatment,
            &outcome.observed_control,
        )?;
        let predicted = effect_trace(
            &outcome.sample_id,
            &prediction.predicted_treatment,
            &prediction.predicted_control,
        )?;
        let cluster = clusters.entry(outcome.cluster_id.as_str()).or_default();

        for (&actual, &forecast) in observed.iter().zip(&predicted) {
            pooled.add(&outcome.sample_id, actual, forecast, sign_tolerance)?;
            cluster.add(&outcome.sample_id, actual, forecast, sign_tolerance)?;
            timepoint_count += 1;
        }
    }

    let cluster_count = clusters.len();
    let macro_rmse = clusters.values().map(|item| item.rmse()).sum::<f64>()
        / cluster_count as f64;
    let macro_mae = clusters.values().map(|item| item.mae()).sum::<f64>()
        / cluster_count as f64;
    let macro_sign = clusters.values().map(|item| item.sign_accuracy()).sum::<f64>()
        / cluster_count as f64;

    Ok(InterventionMetricSummary {
        sample_count: outcomes.len(),
        cluster_count,
        timepoint_count,
        pooled_effect_rmse: pooled.rmse(),
        pooled_effect_mae: pooled.mae(),
        cluster_macro_effect_rmse: macro_rmse,
        cluster_macro_effect_mae: macro_mae,
        pooled_response_sign_accuracy: pooled.sign_accuracy(),
        cluster_macro_response_sign_accuracy: macro_sign,
    })
}

#[derive(Debug, Clone, Copy, Default)]
struct IntervalAccumulator {
    covered: usize,
    width_sum: f64,
    count: usize,
}

/// Score treatment-minus-control intervals. Coverage and width are both
/// returned; broad uninformative intervals cannot pass on coverage alone.
pub fn score_effect_intervals(
    outcomes: &[PairedOutcome],
    intervals: &[EffectIntervalPrediction],
) -> Result<IntervalMetricSummary, MetricError> {
    validate_outcomes(outcomes)?;
    let mut by_id = HashMap::with_capacity(intervals.len());
    for interval in intervals {
        if interval.sample_id.trim().is_empty() {
            return Err(MetricError::EmptySampleId);
        }
        if by_id.insert(interval.sample_id.as_str(), interval).is_some() {
            return Err(MetricError::DuplicatePredictionId {
                sample_id: interval.sample_id.clone(),
            });
        }
    }
    if intervals.len() != outcomes.len() {
        for interval in intervals {
            if !outcomes.iter().any(|outcome| outcome.sample_id == interval.sample_id) {
                return Err(MetricError::UnexpectedPrediction {
                    sample_id: interval.sample_id.clone(),
                });
            }
        }
        for outcome in outcomes {
            if !by_id.contains_key(outcome.sample_id.as_str()) {
                return Err(MetricError::MissingPrediction {
                    sample_id: outcome.sample_id.clone(),
                });
            }
        }
    }

    let mut pooled = IntervalAccumulator::default();
    let mut clusters: BTreeMap<&str, IntervalAccumulator> = BTreeMap::new();
    let mut timepoint_count = 0usize;

    for outcome in outcomes {
        let interval = by_id.get(outcome.sample_id.as_str()).copied()
            .ok_or_else(|| MetricError::MissingPrediction {
                sample_id: outcome.sample_id.clone(),
            })?;
        let horizon = outcome.observed_treatment.len();
        for (field, values) in [
            ("lower_effect", interval.lower_effect.as_slice()),
            ("upper_effect", interval.upper_effect.as_slice()),
        ] {
            if values.len() != horizon {
                return Err(MetricError::ShapeMismatch {
                    sample_id: outcome.sample_id.clone(),
                    field,
                    expected: horizon,
                    actual: values.len(),
                });
            }
            validate_finite(&outcome.sample_id, field, values)?;
        }

        let observed = effect_trace(
            &outcome.sample_id,
            &outcome.observed_treatment,
            &outcome.observed_control,
        )?;
        let cluster = clusters.entry(outcome.cluster_id.as_str()).or_default();
        for index in 0..horizon {
            let lower = interval.lower_effect[index];
            let upper = interval.upper_effect[index];
            if lower > upper {
                return Err(MetricError::InvalidInterval {
                    sample_id: outcome.sample_id.clone(),
                    index,
                    lower,
                    upper,
                });
            }
            let width = upper - lower;
            if !width.is_finite() {
                return Err(MetricError::NumericalOverflow {
                    sample_id: outcome.sample_id.clone(),
                    field: "interval_width",
                });
            }
            let covered = observed[index] >= lower && observed[index] <= upper;
            if covered {
                pooled.covered += 1;
                cluster.covered += 1;
            }
            pooled.width_sum += width;
            cluster.width_sum += width;
            pooled.count += 1;
            cluster.count += 1;
            timepoint_count += 1;
        }
    }

    let cluster_count = clusters.len();
    let macro_coverage = clusters.values()
        .map(|item| item.covered as f64 / item.count as f64).sum::<f64>()
        / cluster_count as f64;
    let macro_width = clusters.values()
        .map(|item| item.width_sum / item.count as f64).sum::<f64>()
        / cluster_count as f64;

    Ok(IntervalMetricSummary {
        sample_count: outcomes.len(),
        cluster_count,
        timepoint_count,
        pooled_coverage: pooled.covered as f64 / pooled.count as f64,
        cluster_macro_coverage: macro_coverage,
        pooled_mean_width: pooled.width_sum / pooled.count as f64,
        cluster_macro_mean_width: macro_width,
    })
}

#[derive(Debug, Clone, Copy, Default)]
struct ClusterSquaredError {
    squared_sum: f64,
    count: usize,
}

fn per_cluster_rmse(
    outcomes: &[PairedOutcome],
    predictions: &[ModelPrediction],
) -> Result<BTreeMap<String, f64>, MetricError> {
    validate_outcomes(outcomes)?;
    let aligned = align_predictions(outcomes, predictions)?;
    let mut accumulators: BTreeMap<String, ClusterSquaredError> = BTreeMap::new();

    for (outcome, prediction) in outcomes.iter().zip(aligned) {
        let actual = effect_trace(
            &outcome.sample_id,
            &outcome.observed_treatment,
            &outcome.observed_control,
        )?;
        let forecast = effect_trace(
            &outcome.sample_id,
            &prediction.predicted_treatment,
            &prediction.predicted_control,
        )?;
        let accumulator = accumulators.entry(outcome.cluster_id.clone()).or_default();

        for (&observed, &predicted) in actual.iter().zip(&forecast) {
            let error = predicted - observed;
            let square = error * error;
            if !square.is_finite() {
                return Err(MetricError::NumericalOverflow {
                    sample_id: outcome.sample_id.clone(),
                    field: "cluster_squared_error",
                });
            }
            accumulator.squared_sum += square;
            accumulator.count += 1;
            if !accumulator.squared_sum.is_finite() {
                return Err(MetricError::NumericalOverflow {
                    sample_id: outcome.sample_id.clone(),
                    field: "cluster_squared_sum",
                });
            }
        }
    }

    Ok(accumulators.into_iter().map(|(id, item)| {
        (id, (item.squared_sum / item.count as f64).sqrt())
    }).collect())
}

#[derive(Debug, Clone, Copy)]
struct XorShift64 {
    state: u64,
}
impl XorShift64 {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value
    }
    fn index(&mut self, length: usize) -> usize {
        ((self.next() as u128 * length as u128) >> 64) as usize
    }
}

fn percentile(sorted: &[f64], probability: f64) -> f64 {
    let position = probability * (sorted.len() - 1) as f64;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    if low == high {
        sorted[low]
    } else {
        let fraction = position - low as f64;
        sorted[low] * (1.0 - fraction) + sorted[high] * fraction
    }
}

/// Paired percentile bootstrap over independent clusters. Positive improvement
/// means candidate RMSE is lower than comparator RMSE. Requires at least two
/// clusters and 100 replicates. The interval is not a p-value.
pub fn paired_cluster_bootstrap_rmse(
    outcomes: &[PairedOutcome],
    candidate_predictions: &[ModelPrediction],
    comparator_predictions: &[ModelPrediction],
    bootstrap_replicates: usize,
    seed: u64,
) -> Result<ClusterBootstrapComparison, MetricError> {
    if bootstrap_replicates < 100 {
        return Err(MetricError::InvalidBootstrapReplicates {
            replicates: bootstrap_replicates,
        });
    }
    if seed == 0 {
        return Err(MetricError::ZeroBootstrapSeed);
    }

    let candidate = per_cluster_rmse(outcomes, candidate_predictions)?;
    let comparator = per_cluster_rmse(outcomes, comparator_predictions)?;
    if candidate.len() != comparator.len() || candidate.keys().ne(comparator.keys()) {
        return Err(MetricError::NumericalOverflow {
            sample_id: "cluster_alignment".to_owned(),
            field: "cluster_set_mismatch",
        });
    }

    let differences: Vec<f64> = candidate.iter()
        .map(|(cluster, rmse)| comparator[cluster] - rmse).collect();
    let cluster_count = differences.len();
    if cluster_count < 2 {
        return Err(MetricError::TooFewIndependentClusters {
            minimum: 2,
            actual: cluster_count,
        });
    }

    let point_estimate = differences.iter().sum::<f64>() / cluster_count as f64;
    let mut rng = XorShift64::new(seed);
    let mut bootstrap_means = Vec::with_capacity(bootstrap_replicates);
    for _ in 0..bootstrap_replicates {
        let mut sum = 0.0;
        for _ in 0..cluster_count {
            sum += differences[rng.index(cluster_count)];
        }
        bootstrap_means.push(sum / cluster_count as f64);
    }
    bootstrap_means.sort_by(f64::total_cmp);

    Ok(ClusterBootstrapComparison {
        cluster_count,
        bootstrap_replicates,
        seed,
        mean_rmse_improvement: point_estimate,
        percentile_95_ci_low: percentile(&bootstrap_means, CI_LOW),
        percentile_95_ci_high: percentile(&bootstrap_means, CI_HIGH),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(id: &str, cluster: &str, treated: &[f64], control: &[f64]) -> PairedOutcome {
        PairedOutcome {
            sample_id: id.to_owned(),
            cluster_id: cluster.to_owned(),
            observed_treatment: treated.to_vec(),
            observed_control: control.to_vec(),
        }
    }
    fn prediction(id: &str, treated: &[f64], control: &[f64]) -> ModelPrediction {
        ModelPrediction {
            sample_id: id.to_owned(),
            predicted_treatment: treated.to_vec(),
            predicted_control: control.to_vec(),
        }
    }
    fn close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1.0e-12, "{actual} != {expected}");
    }

    #[test]
    fn exact_paired_effect_forecast_has_zero_error() {
        let o = vec![outcome("s1", "a", &[2.0, 1.0], &[1.0, 1.0])];
        let p = vec![prediction("s1", &[5.0, 3.0], &[4.0, 2.0])];
        let s = score_intervention_effects(&o, &p, 0.0).unwrap();
        close(s.pooled_effect_rmse, 0.0);
        close(s.pooled_effect_mae, 0.0);
        close(s.cluster_macro_effect_rmse, 0.0);
        close(s.pooled_response_sign_accuracy, 1.0);
    }

    #[test]
    fn treatment_minus_control_is_scored_on_both_sides() {
        let o = vec![outcome("s1", "a", &[3.0, 1.0], &[1.0, 1.0])];
        let p = vec![prediction("s1", &[4.0, 2.0], &[1.0, 1.0])];
        let s = score_intervention_effects(&o, &p, 0.0).unwrap();
        close(s.pooled_effect_rmse, 1.0);
        close(s.pooled_effect_mae, 1.0);
    }

    #[test]
    fn pooled_and_cluster_macro_metrics_expose_unequal_trial_counts() {
        let o = vec![
            outcome("a1", "large", &[2.0; 8], &[0.0; 8]),
            outcome("b1", "small", &[2.0], &[0.0]),
        ];
        let p = vec![
            prediction("a1", &[1.0; 8], &[0.0; 8]),
            prediction("b1", &[0.0], &[0.0]),
        ];
        let s = score_intervention_effects(&o, &p, 0.0).unwrap();
        close(s.pooled_effect_rmse, (12.0f64 / 9.0).sqrt());
        close(s.cluster_macro_effect_rmse, 1.5);
        assert_eq!(s.cluster_count, 2);
    }

    #[test]
    fn sign_accuracy_obeys_explicit_dead_zone() {
        let o = vec![outcome("s1", "a", &[1.0, 0.0, -1.0], &[0.0; 3])];
        let p = vec![prediction("s1", &[0.8, 0.05, -0.8], &[0.0; 3])];
        let s = score_intervention_effects(&o, &p, 0.1).unwrap();
        close(s.pooled_response_sign_accuracy, 1.0);
    }

    #[test]
    fn interval_metrics_report_coverage_and_width() {
        let o = vec![outcome("s1", "a", &[2.0, 1.0], &[1.0, 1.0])];
        let intervals = vec![EffectIntervalPrediction {
            sample_id: "s1".to_owned(),
            lower_effect: vec![0.5, -0.5],
            upper_effect: vec![1.5, 0.5],
        }];
        let s = score_effect_intervals(&o, &intervals).unwrap();
        close(s.pooled_coverage, 1.0);
        close(s.cluster_macro_coverage, 1.0);
        close(s.pooled_mean_width, 1.0);
    }

    #[test]
    fn interval_width_prevents_trivial_coverage_claim() {
        let o = vec![outcome("s1", "a", &[2.0], &[1.0])];
        let narrow = vec![EffectIntervalPrediction {
            sample_id: "s1".to_owned(), lower_effect: vec![0.9], upper_effect: vec![1.1],
        }];
        let wide = vec![EffectIntervalPrediction {
            sample_id: "s1".to_owned(), lower_effect: vec![-100.0], upper_effect: vec![100.0],
        }];
        let n = score_effect_intervals(&o, &narrow).unwrap();
        let w = score_effect_intervals(&o, &wide).unwrap();
        assert!(w.pooled_coverage >= n.pooled_coverage);
        assert!(w.pooled_mean_width > n.pooled_mean_width);
    }

    #[test]
    fn bootstrap_positive_improvement_when_candidate_is_better() {
        let o = vec![
            outcome("s1", "a", &[1.0], &[0.0]),
            outcome("s2", "b", &[1.0], &[0.0]),
        ];
        let candidate = vec![prediction("s1", &[1.0], &[0.0]), prediction("s2", &[1.0], &[0.0])];
        let comparator = vec![prediction("s1", &[2.0], &[0.0]), prediction("s2", &[2.0], &[0.0])];
        let s = paired_cluster_bootstrap_rmse(&o, &candidate, &comparator, 500, 7).unwrap();
        close(s.mean_rmse_improvement, 1.0);
        close(s.percentile_95_ci_low, 1.0);
        close(s.percentile_95_ci_high, 1.0);
    }

    #[test]
    fn bootstrap_is_deterministic_for_fixed_seed() {
        let o = vec![
            outcome("s1", "a", &[1.0], &[0.0]),
            outcome("s2", "b", &[1.0], &[0.0]),
            outcome("s3", "c", &[1.0], &[0.0]),
        ];
        let c = vec![
            prediction("s1", &[0.5], &[0.0]),
            prediction("s2", &[0.8], &[0.0]),
            prediction("s3", &[1.0], &[0.0]),
        ];
        let b = vec![
            prediction("s1", &[2.0], &[0.0]),
            prediction("s2", &[1.5], &[0.0]),
            prediction("s3", &[1.2], &[0.0]),
        ];
        let x = paired_cluster_bootstrap_rmse(&o, &c, &b, 1000, 42).unwrap();
        let y = paired_cluster_bootstrap_rmse(&o, &c, &b, 1000, 42).unwrap();
        assert_eq!(x, y);
    }

    #[test]
    fn bootstrap_counts_animals_not_individual_trials() {
        let o = vec![
            outcome("a1", "animal-a", &[1.0], &[0.0]),
            outcome("a2", "animal-a", &[1.0], &[0.0]),
            outcome("b1", "animal-b", &[1.0], &[0.0]),
        ];
        let c = vec![
            prediction("a1", &[1.0], &[0.0]),
            prediction("a2", &[1.0], &[0.0]),
            prediction("b1", &[1.0], &[0.0]),
        ];
        let b = vec![
            prediction("a1", &[2.0], &[0.0]),
            prediction("a2", &[2.0], &[0.0]),
            prediction("b1", &[2.0], &[0.0]),
        ];
        let s = paired_cluster_bootstrap_rmse(&o, &c, &b, 100, 9).unwrap();
        assert_eq!(s.cluster_count, 2);
        close(s.mean_rmse_improvement, 1.0);
    }

    #[test]
    fn missing_prediction_fails_closed() {
        let o = vec![outcome("s1", "a", &[1.0], &[0.0])];
        assert_eq!(
            score_intervention_effects(&o, &[], 0.0),
            Err(MetricError::MissingPrediction { sample_id: "s1".to_owned() })
        );
    }

    #[test]
    fn duplicate_sample_and_prediction_ids_fail_closed() {
        let duplicate_o = vec![
            outcome("s1", "a", &[1.0], &[0.0]),
            outcome("s1", "b", &[1.0], &[0.0]),
        ];
        assert!(matches!(
            score_intervention_effects(&duplicate_o, &[prediction("s1", &[1.0], &[0.0])], 0.0),
            Err(MetricError::DuplicateSampleId { .. })
        ));

        let o = vec![outcome("s1", "a", &[1.0], &[0.0])];
        let duplicate_p = vec![
            prediction("s1", &[1.0], &[0.0]),
            prediction("s1", &[1.0], &[0.0]),
        ];
        assert!(matches!(
            score_intervention_effects(&o, &duplicate_p, 0.0),
            Err(MetricError::DuplicatePredictionId { .. })
        ));
    }

    #[test]
    fn mismatched_horizon_fails_closed() {
        let o = vec![outcome("s1", "a", &[1.0, 2.0], &[0.0, 0.0])];
        let p = vec![prediction("s1", &[1.0], &[0.0])];
        assert!(matches!(
            score_intervention_effects(&o, &p, 0.0),
            Err(MetricError::ShapeMismatch { field: "predicted_treatment", .. })
        ));
    }

    #[test]
    fn non_finite_observation_fails_closed() {
        let o = vec![outcome("s1", "a", &[f64::NAN], &[0.0])];
        let p = vec![prediction("s1", &[1.0], &[0.0])];
        assert!(matches!(
            score_intervention_effects(&o, &p, 0.0),
            Err(MetricError::NonFiniteValue { .. })
        ));
    }

    #[test]
    fn reversed_interval_fails_closed() {
        let o = vec![outcome("s1", "a", &[1.0], &[0.0])];
        let i = vec![EffectIntervalPrediction {
            sample_id: "s1".to_owned(), lower_effect: vec![1.1], upper_effect: vec![0.9],
        }];
        assert!(matches!(
            score_effect_intervals(&o, &i),
            Err(MetricError::InvalidInterval { .. })
        ));
    }

    #[test]
    fn invalid_bootstrap_settings_fail_closed() {
        let o = vec![outcome("s1", "a", &[1.0], &[0.0]), outcome("s2", "b", &[1.0], &[0.0])];
        let p = vec![prediction("s1", &[1.0], &[0.0]), prediction("s2", &[1.0], &[0.0])];
        assert!(matches!(
            paired_cluster_bootstrap_rmse(&o, &p, &p, 99, 1),
            Err(MetricError::InvalidBootstrapReplicates { .. })
        ));
        assert_eq!(
            paired_cluster_bootstrap_rmse(&o, &p, &p, 100, 0),
            Err(MetricError::ZeroBootstrapSeed)
        );
    }

    #[test]
    fn one_cluster_is_not_enough_for_bootstrap() {
        let o = vec![
            outcome("s1", "one-animal", &[1.0], &[0.0]),
            outcome("s2", "one-animal", &[1.0], &[0.0]),
        ];
        let p = vec![prediction("s1", &[1.0], &[0.0]), prediction("s2", &[1.0], &[0.0])];
        assert!(matches!(
            paired_cluster_bootstrap_rmse(&o, &p, &p, 100, 1),
            Err(MetricError::TooFewIndependentClusters { .. })
        ));
    }

    #[test]
    fn invalid_sign_tolerance_is_rejected() {
        let o = vec![outcome("s1", "a", &[1.0], &[0.0])];
        let p = vec![prediction("s1", &[1.0], &[0.0])];
        assert!(matches!(
            score_intervention_effects(&o, &p, f64::NAN),
            Err(MetricError::InvalidSignTolerance { .. })
        ));
    }

    #[test]
    fn prediction_order_does_not_change_alignment() {
        let o = vec![
            outcome("s1", "a", &[1.0], &[0.0]),
            outcome("s2", "b", &[2.0], &[0.0]),
        ];
        let p = vec![prediction("s2", &[2.0], &[0.0]), prediction("s1", &[1.0], &[0.0])];
        let s = score_intervention_effects(&o, &p, 0.0).unwrap();
        close(s.pooled_effect_rmse, 0.0);
    }

    #[test]
    fn extra_prediction_id_fails_closed() {
        let o = vec![outcome("s1", "a", &[1.0], &[0.0])];
        let p = vec![
            prediction("s1", &[1.0], &[0.0]),
            prediction("extra", &[1.0], &[0.0]),
        ];
        assert!(matches!(
            score_intervention_effects(&o, &p, 0.0),
            Err(MetricError::UnexpectedPrediction { .. })
        ));
    }
}
