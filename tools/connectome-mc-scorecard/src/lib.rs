#![forbid(unsafe_code)]

//! Method-neutral Monte Carlo evaluation for clustered-IV and intervention benchmarks.
//!
//! This crate evaluates normalized replicate-level outputs. It does not fit estimators,
//! establish instrument validity, or certify any estimator's inferential assumptions.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStatus {
    Completed,
    Failed,
    NotApplicable,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    /// None means negative infinity. JSON adapters encode this as null.
    pub lower: Option<f64>,
    /// None means positive infinity. JSON adapters encode this as null.
    pub upper: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConfidenceSet {
    /// Canonical ordered union. An empty vector is an explicitly empty set.
    pub components: Vec<Interval>,
}

impl ConfidenceSet {
    pub fn validate(&self) -> Result<(), EvalError> {
        for (index, interval) in self.components.iter().enumerate() {
            if interval.lower.is_some_and(|x| !x.is_finite())
                || interval.upper.is_some_and(|x| !x.is_finite())
            {
                return Err(EvalError::NonFiniteConfidenceEndpoint);
            }
            if let (Some(lower), Some(upper)) = (interval.lower, interval.upper) {
                if lower > upper {
                    return Err(EvalError::ReversedConfidenceInterval);
                }
            }
            if index > 0 && interval.lower.is_none() {
                return Err(EvalError::NonCanonicalConfidenceSet);
            }
            if index + 1 < self.components.len() && interval.upper.is_none() {
                return Err(EvalError::NonCanonicalConfidenceSet);
            }
            if index > 0 {
                let previous = self.components[index - 1];
                let previous_upper = previous.upper.ok_or(EvalError::NonCanonicalConfidenceSet)?;
                let current_lower = interval.lower.ok_or(EvalError::NonCanonicalConfidenceSet)?;
                // Touching intervals must be merged into one component.
                if previous_upper >= current_lower {
                    return Err(EvalError::NonCanonicalConfidenceSet);
                }
            }
        }
        Ok(())
    }

    pub fn contains(&self, value: f64) -> bool {
        self.components.iter().any(|interval| {
            interval.lower.is_none_or(|lower| value >= lower)
                && interval.upper.is_none_or(|upper| value <= upper)
        })
    }

    pub fn topology(&self) -> ConfidenceSetTopology {
        match self.components.as_slice() {
            [] => ConfidenceSetTopology::Empty,
            [only] => match (only.lower, only.upper) {
                (None, None) => ConfidenceSetTopology::WholeLine,
                (None, Some(_)) => ConfidenceSetTopology::LeftRay,
                (Some(_), None) => ConfidenceSetTopology::RightRay,
                (Some(_), Some(_)) => ConfidenceSetTopology::FiniteBounded,
            },
            [first, last] if first.lower.is_none() && last.upper.is_none() => {
                ConfidenceSetTopology::TwoRays
            }
            _ => ConfidenceSetTopology::DisconnectedMultiComponent,
        }
    }

    /// Total width for non-empty bounded sets. Empty and unbounded sets have no finite width.
    pub fn finite_total_width(&self) -> Result<Option<f64>, EvalError> {
        if self.components.is_empty()
            || self.components.iter().any(|part| part.lower.is_none() || part.upper.is_none())
        {
            return Ok(None);
        }
        let mut total = 0.0;
        for part in &self.components {
            let width = part.upper.unwrap() - part.lower.unwrap();
            if !width.is_finite() || width < 0.0 {
                return Err(EvalError::NumericOverflow);
            }
            total += width;
            if !total.is_finite() {
                return Err(EvalError::NumericOverflow);
            }
        }
        Ok(Some(total))
    }

    pub fn is_unbounded(&self) -> bool {
        self.components
            .iter()
            .any(|part| part.lower.is_none() || part.upper.is_none())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceSetTopology {
    Empty,
    FiniteBounded,
    LeftRay,
    RightRay,
    TwoRays,
    DisconnectedMultiComponent,
    WholeLine,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReplicateRecord {
    pub scenario_id: String,
    /// Must identify a precise method/test/calibration variant, not just a software package.
    pub method_id: String,
    pub replicate_id: String,
    pub seed: u64,
    pub structural_beta: f64,
    pub null_is_true: bool,
    pub status: ExecutionStatus,
    pub p_value: Option<f64>,
    pub point_estimate: Option<f64>,
    /// None means not reported. Some(empty set) means an explicitly reported empty set.
    pub confidence_set: Option<ConfidenceSet>,
    pub runtime_seconds: Option<f64>,
    pub peak_memory_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvaluatorConfig {
    /// Rejection threshold used to calculate empirical size and power.
    pub alpha: f64,
    /// Two-sided Wilson score critical value. Default is the 95% value.
    pub wilson_z_critical: f64,
}

impl Default for EvaluatorConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            wilson_z_critical: 1.959_963_984_540_054,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvalError {
    EmptyInput,
    InvalidConfig,
    EmptyIdentifier,
    NonFiniteStructuralTruth,
    InvalidPValue,
    NonFinitePointEstimate,
    InvalidRuntime,
    NonFiniteConfidenceEndpoint,
    ReversedConfidenceInterval,
    NonCanonicalConfidenceSet,
    MetricsOnNonCompletedRecord,
    DuplicateReplicate,
    DuplicateSeed,
    UnplannedReplicate,
    MissingComparisonMethod,
    SameMethodComparison,
    SeedAlignmentMismatch,
    InconsistentPairedTruth,
    InconsistentScenarioTruth,
    NonFiniteMetricCalibrationValue,
    InvalidMetricEffectMagnitude,
    DuplicateMetricCalibrationReplicate,
    NumericOverflow,
}

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyInput => "no replicate records supplied",
            Self::InvalidConfig => "invalid evaluator configuration",
            Self::EmptyIdentifier => "scenario, method, and replicate identifiers must be non-empty",
            Self::NonFiniteStructuralTruth => "structural beta must be finite",
            Self::InvalidPValue => "p-value must be finite and within [0, 1]",
            Self::NonFinitePointEstimate => "point estimate must be finite",
            Self::InvalidRuntime => "runtime must be finite and non-negative",
            Self::NonFiniteConfidenceEndpoint => "confidence-set endpoints must be finite or explicitly unbounded",
            Self::ReversedConfidenceInterval => "confidence-set interval has lower endpoint above upper endpoint",
            Self::NonCanonicalConfidenceSet => "confidence-set components must be sorted, disjoint, and canonically unbounded",
            Self::MetricsOnNonCompletedRecord => "failed/not-applicable records cannot carry inferential outputs",
            Self::DuplicateReplicate => "duplicate scenario/method/replicate key",
            Self::DuplicateSeed => "duplicate random seed within a scenario/method group",
            Self::UnplannedReplicate => "method output contains a replicate absent from the frozen plan",
            Self::MissingComparisonMethod => "baseline or candidate method has no records in the requested scenario",
            Self::SameMethodComparison => "baseline and candidate method identifiers must differ",
            Self::SeedAlignmentMismatch => "paired replicate identifiers must carry identical random seeds",
            Self::InconsistentPairedTruth => "paired methods disagree on structural truth or null status",
            Self::InconsistentScenarioTruth => "structural beta and null status must be constant within a scenario/method group",
            Self::NonFiniteMetricCalibrationValue => "metric-control scores must be finite",
            Self::InvalidMetricEffectMagnitude => "known effect magnitudes must be finite and non-negative",
            Self::DuplicateMetricCalibrationReplicate => "duplicate metric-calibration replicate within an effect-magnitude group",
            Self::NumericOverflow => "derived metric overflowed finite floating-point range",
        };
        f.write_str(message)
    }
}

impl std::error::Error for EvalError {}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FiniteInterval {
    pub lower: f64,
    pub upper: f64,
}

/// Metric orientation for paired positive-control calibration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricDirection {
    HigherIsBetter,
    LowerIsBetter,
}

/// Paired scores for an uninformative baseline and a known-positive predictor
/// evaluated on the same replicate and effect magnitude.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricCalibrationObservation {
    pub effect_magnitude: f64,
    pub replicate_id: String,
    pub seed: u64,
    pub uninformative_metric_value: f64,
    pub positive_control_metric_value: f64,
    /// Metric value for a perfect predictor on the same known-answer case.
    pub perfect_metric_value: f64,
}

/// Calibration requirements are declared before looking at model-comparison
/// results. Bounds use the supplied one-sided Wilson critical value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricCalibrationConfig {
    pub minimum_effect_magnitude: f64,
    pub minimum_detection_probability: f64,
    pub maximum_null_false_positive_probability: f64,
    pub tie_tolerance: f64,
    pub monotonicity_tolerance: f64,
    pub wilson_z_critical: f64,
}

impl Default for MetricCalibrationConfig {
    fn default() -> Self {
        Self {
            minimum_effect_magnitude: 1.0,
            minimum_detection_probability: 0.80,
            maximum_null_false_positive_probability: 0.20,
            tie_tolerance: 1.0e-12,
            monotonicity_tolerance: 1.0e-9,
            wilson_z_critical: 1.644_853_626_951_472_2,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetricCalibrationBin {
    pub effect_magnitude: f64,
    pub replicate_count: usize,
    pub positive_control_wins: usize,
    pub ties: usize,
    pub positive_control_losses: usize,
    pub positive_control_win_rate: f64,
    pub one_sided_wilson_lower_bound: f64,
    pub one_sided_wilson_upper_bound: f64,
    /// Positive means the known-positive predictor scores better than the
    /// uninformative predictor, after applying metric direction.
    pub mean_oriented_improvement: f64,
    /// Mean fraction of the available uninformative-to-perfect metric range
    /// captured by the positive-control predictor.
    pub mean_dynamic_range_fraction: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetricCalibrationReport {
    pub schema_version: &'static str,
    pub metric_id: String,
    pub direction: MetricDirection,
    pub config: MetricCalibrationConfig,
    pub bins: Vec<MetricCalibrationBin>,
    pub zero_effect_false_positive_upper_bound: Option<f64>,
    pub minimum_positive_control_detection_lower_bound: Option<f64>,
    pub monotonicity_violations: usize,
    /// The metric passed only the declared control grid and thresholds. This
    /// is not a universal property of the metric outside that calibration.
    pub calibrated_for_declared_grid: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RateEstimate {
    pub successes: usize,
    pub trials: usize,
    pub estimate: f64,
    pub wilson_interval: FiniteInterval,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionCounts {
    /// Number of submitted rows. Includes COMPLETED, FAILED, and NOT_APPLICABLE.
    pub attempted: usize,
    pub completed: usize,
    pub failed: usize,
    pub not_applicable: usize,
    pub failure_rate: f64,
    pub not_applicable_rate: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PValueDiagnostics {
    pub count: usize,
    /// Ten fixed bins: [0.0,0.1), ..., [0.9,1.0], with p=1 in the final bin.
    pub histogram_deciles: [usize; 10],
    /// Deterministic type-7 linearly interpolated quantiles.
    pub p10: f64,
    pub p50: f64,
    pub p90: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TopologyCounts {
    pub empty: usize,
    pub finite_bounded: usize,
    pub left_ray: usize,
    pub right_ray: usize,
    pub two_rays: usize,
    pub disconnected_multi_component: usize,
    pub whole_line: usize,
}

impl TopologyCounts {
    fn add(&mut self, topology: ConfidenceSetTopology) {
        match topology {
            ConfidenceSetTopology::Empty => self.empty += 1,
            ConfidenceSetTopology::FiniteBounded => self.finite_bounded += 1,
            ConfidenceSetTopology::LeftRay => self.left_ray += 1,
            ConfidenceSetTopology::RightRay => self.right_ray += 1,
            ConfidenceSetTopology::TwoRays => self.two_rays += 1,
            ConfidenceSetTopology::DisconnectedMultiComponent => {
                self.disconnected_multi_component += 1
            }
            ConfidenceSetTopology::WholeLine => self.whole_line += 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConfidenceSetDiagnostics {
    pub available: usize,
    pub missing: usize,
    pub covered: usize,
    /// Coverage among records with a reported confidence set. Empty sets count as non-coverage.
    pub observed_coverage: Option<RateEstimate>,
    /// Missing, failed, and not-applicable sets are treated as non-covering.
    pub conservative_coverage_lower_bound: Option<f64>,
    /// Missing, failed, and not-applicable sets are treated as covering.
    pub conservative_coverage_upper_bound: Option<f64>,
    pub topology_counts: TopologyCounts,
    pub unbounded_count: usize,
    pub unbounded_rate: Option<f64>,
    /// Finite bounded sets that exclude the known structural beta.
    /// In a zero/weak-first-stage scenario these are the explicit false-narrow-set count.
    pub finite_noncovering_set_count: usize,
    /// Proportion of reported finite bounded sets that exclude the true beta.
    pub finite_noncovering_set_rate: Option<f64>,
    /// Width is defined only for non-empty bounded sets.
    pub finite_width_count: usize,
    pub finite_width_median: Option<f64>,
    pub finite_width_p90: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EstimateDiagnostics {
    pub count: usize,
    pub mean_signed_bias: Option<f64>,
    pub rmse: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResourceDiagnostics {
    /// Timings are summarized only when every COMPLETED/FAILED row has a runtime.
    pub runtime_complete: bool,
    pub runtime_count: usize,
    pub runtime_p50_seconds: Option<f64>,
    pub runtime_p95_seconds: Option<f64>,
    /// Memory is summarized only when every COMPLETED/FAILED row has a measurement.
    pub memory_complete: bool,
    pub memory_count: usize,
    pub peak_memory_p50_bytes: Option<f64>,
    pub peak_memory_p95_bytes: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MethodScorecard {
    pub scenario_id: String,
    pub method_id: String,
    pub structural_beta: f64,
    pub null_is_true: bool,
    pub alpha: f64,
    pub wilson_z_critical: f64,
    pub records: ExecutionCounts,
    /// Present only for true-null scenarios, using completed records with p-values.
    pub empirical_size: Option<RateEstimate>,
    /// Present only for alternative scenarios, using completed records with p-values.
    pub empirical_power: Option<RateEstimate>,
    /// Worst-case upper size bound: all missing/failed/not-applicable null p-values reject.
    pub conservative_size_upper_bound: Option<f64>,
    /// Worst-case lower power bound: all missing/failed/not-applicable alternative p-values do not reject.
    pub conservative_power_lower_bound: Option<f64>,
    /// Reported only for true-null scenarios.
    pub p_values_under_null: Option<PValueDiagnostics>,
    pub confidence_sets: ConfidenceSetDiagnostics,
    pub estimates: EstimateDiagnostics,
    pub resources: ResourceDiagnostics,
}

/// Calibrate whether a metric can detect a known positive intervention
/// relative to an uninformative/no-change predictor across a declared grid.
///
/// Every positive-control score is paired with its uninformative score on the
/// same replicate and seed. A zero-effect bin estimates spurious positive
/// control wins; bins at or above minimum_effect_magnitude must meet the
/// preregistered lower-bound detection requirement. Mean oriented improvement
/// must be nondecreasing across positive effect magnitudes within tolerance.
///
/// This calibrates the metric on the supplied control grid only. It does not
/// certify biological validity, causal identification, or model quality.
pub fn calibrate_predictive_metric(
    metric_id: &str,
    direction: MetricDirection,
    observations: &[MetricCalibrationObservation],
    config: MetricCalibrationConfig,
) -> Result<MetricCalibrationReport, EvalError> {
    if metric_id.trim().is_empty() || observations.is_empty() {
        return Err(EvalError::EmptyInput);
    }
    if !config.minimum_effect_magnitude.is_finite()
        || config.minimum_effect_magnitude <= 0.0
        || !config.minimum_detection_probability.is_finite()
        || config.minimum_detection_probability <= 0.0
        || config.minimum_detection_probability > 1.0
        || !config.maximum_null_false_positive_probability.is_finite()
        || config.maximum_null_false_positive_probability < 0.0
        || config.maximum_null_false_positive_probability >= 1.0
        || !config.tie_tolerance.is_finite()
        || config.tie_tolerance < 0.0
        || !config.monotonicity_tolerance.is_finite()
        || config.monotonicity_tolerance < 0.0
        || !config.wilson_z_critical.is_finite()
        || config.wilson_z_critical <= 0.0
    {
        return Err(EvalError::InvalidConfig);
    }

    let mut ordered: Vec<&MetricCalibrationObservation> = observations.iter().collect();
    for observation in &ordered {
        if !observation.effect_magnitude.is_finite() || observation.effect_magnitude < 0.0 {
            return Err(EvalError::InvalidMetricEffectMagnitude);
        }
        if observation.replicate_id.trim().is_empty() {
            return Err(EvalError::EmptyIdentifier);
        }
        if !observation.uninformative_metric_value.is_finite()
            || !observation.positive_control_metric_value.is_finite()
            || !observation.perfect_metric_value.is_finite()
        {
            return Err(EvalError::NonFiniteMetricCalibrationValue);
        }
    }
    ordered.sort_by(|left, right| {
        left.effect_magnitude
            .total_cmp(&right.effect_magnitude)
            .then_with(|| left.replicate_id.cmp(&right.replicate_id))
    });

    let mut groups: Vec<(f64, Vec<&MetricCalibrationObservation>)> = Vec::new();
    for observation in ordered {
        if let Some((effect, group)) = groups.last_mut() {
            if *effect == observation.effect_magnitude {
                group.push(observation);
                continue;
            }
        }
        groups.push((observation.effect_magnitude, vec![observation]));
    }

    let mut bins = Vec::with_capacity(groups.len());
    for (effect_magnitude, group) in groups {
        let mut replicate_ids = BTreeSet::new();
        let mut seeds = BTreeSet::new();
        let mut wins = 0usize;
        let mut ties = 0usize;
        let mut losses = 0usize;
        let mut improvement_sum = 0.0f64;
        let mut dynamic_range_fraction_sum = 0.0f64;

        for observation in &group {
            if !replicate_ids.insert(observation.replicate_id.as_str()) {
                return Err(EvalError::DuplicateMetricCalibrationReplicate);
            }
            if !seeds.insert(observation.seed) {
                return Err(EvalError::DuplicateSeed);
            }
            let improvement = match direction {
                MetricDirection::HigherIsBetter => {
                    observation.positive_control_metric_value
                        - observation.uninformative_metric_value
                }
                MetricDirection::LowerIsBetter => {
                    observation.uninformative_metric_value
                        - observation.positive_control_metric_value
                }
            };
            if !improvement.is_finite() {
                return Err(EvalError::NumericOverflow);
            }
            improvement_sum += improvement;
            if !improvement_sum.is_finite() {
                return Err(EvalError::NumericOverflow);
            }

            let perfect_improvement = match direction {
                MetricDirection::HigherIsBetter => {
                    observation.perfect_metric_value - observation.uninformative_metric_value
                }
                MetricDirection::LowerIsBetter => {
                    observation.uninformative_metric_value - observation.perfect_metric_value
                }
            };
            if !perfect_improvement.is_finite()
                || perfect_improvement <= config.tie_tolerance
            {
                return Err(EvalError::NonFiniteMetricCalibrationValue);
            }
            let dynamic_range_fraction = improvement / perfect_improvement;
            if !dynamic_range_fraction.is_finite() {
                return Err(EvalError::NumericOverflow);
            }
            dynamic_range_fraction_sum += dynamic_range_fraction;
            if !dynamic_range_fraction_sum.is_finite() {
                return Err(EvalError::NumericOverflow);
            }

            if improvement > config.tie_tolerance {
                wins += 1;
            } else if improvement < -config.tie_tolerance {
                losses += 1;
            } else {
                ties += 1;
            }
        }

        let count = group.len();
        let interval = wilson_interval(wins, count, config.wilson_z_critical)?
            .ok_or(EvalError::NumericOverflow)?;
        bins.push(MetricCalibrationBin {
            effect_magnitude,
            replicate_count: count,
            positive_control_wins: wins,
            ties,
            positive_control_losses: losses,
            positive_control_win_rate: wins as f64 / count as f64,
            one_sided_wilson_lower_bound: interval.lower,
            one_sided_wilson_upper_bound: interval.upper,
            mean_oriented_improvement: improvement_sum / count as f64,
            mean_dynamic_range_fraction: dynamic_range_fraction_sum / count as f64,
        });
    }

    let zero_effect_bin = bins.iter().find(|bin| bin.effect_magnitude == 0.0);
    let zero_effect_false_positive_upper_bound =
        zero_effect_bin.map(|bin| bin.one_sided_wilson_upper_bound);

    let qualifying_positive_bins: Vec<&MetricCalibrationBin> = bins
        .iter()
        .filter(|bin| bin.effect_magnitude >= config.minimum_effect_magnitude)
        .collect();
    let minimum_positive_control_detection_lower_bound = qualifying_positive_bins
        .iter()
        .map(|bin| bin.one_sided_wilson_lower_bound)
        .min_by(f64::total_cmp);

    let mut monotonicity_violations = 0usize;
    let positive_bins: Vec<&MetricCalibrationBin> = bins
        .iter()
        .filter(|bin| bin.effect_magnitude > 0.0)
        .collect();
    for pair in positive_bins.windows(2) {
        if pair[1].mean_oriented_improvement + config.monotonicity_tolerance
            < pair[0].mean_oriented_improvement
        {
            monotonicity_violations += 1;
        }
    }

    let calibrated_for_declared_grid = zero_effect_false_positive_upper_bound
        .is_some_and(|bound| bound <= config.maximum_null_false_positive_probability)
        && minimum_positive_control_detection_lower_bound
            .is_some_and(|bound| bound >= config.minimum_detection_probability)
        && !qualifying_positive_bins.is_empty()
        && monotonicity_violations == 0;

    Ok(MetricCalibrationReport {
        schema_version: "connectome-metric-calibration-v1",
        metric_id: metric_id.to_owned(),
        direction,
        config,
        bins,
        zero_effect_false_positive_upper_bound,
        minimum_positive_control_detection_lower_bound,
        monotonicity_violations,
        calibrated_for_declared_grid,
    })
}

pub fn wilson_interval(
    successes: usize,
    trials: usize,
    z_critical: f64,
) -> Result<Option<FiniteInterval>, EvalError> {
    if successes > trials || !z_critical.is_finite() || z_critical <= 0.0 {
        return Err(EvalError::InvalidConfig);
    }
    if trials == 0 {
        return Ok(None);
    }
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = z_critical * z_critical;
    if !z2.is_finite() {
        return Err(EvalError::NumericOverflow);
    }
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let radicand = p * (1.0 - p) / n + z2 / (4.0 * n * n);
    let half_width = z_critical * radicand.sqrt() / denominator;
    let lower = (center - half_width).max(0.0);
    let upper = (center + half_width).min(1.0);
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return Err(EvalError::NumericOverflow);
    }
    Ok(Some(FiniteInterval { lower, upper }))
}

fn rate_estimate(
    successes: usize,
    trials: usize,
    z_critical: f64,
) -> Result<Option<RateEstimate>, EvalError> {
    if trials == 0 {
        return Ok(None);
    }
    let wilson_interval = wilson_interval(successes, trials, z_critical)?
        .ok_or(EvalError::NumericOverflow)?;
    Ok(Some(RateEstimate {
        successes,
        trials,
        estimate: successes as f64 / trials as f64,
        wilson_interval,
    }))
}

fn quantile(values: &[f64], probability: f64) -> Option<f64> {
    if values.is_empty() || !(0.0..=1.0).contains(&probability) {
        return None;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(|a, b| a.total_cmp(b));
    let position = probability * (ordered.len() - 1) as f64;
    let lower_index = position.floor() as usize;
    let upper_index = position.ceil() as usize;
    let fraction = position - lower_index as f64;
    Some(ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction)
}

fn validate_config(config: EvaluatorConfig) -> Result<(), EvalError> {
    if !config.alpha.is_finite()
        || config.alpha <= 0.0
        || config.alpha >= 1.0
        || !config.wilson_z_critical.is_finite()
        || config.wilson_z_critical <= 0.0
    {
        return Err(EvalError::InvalidConfig);
    }
    Ok(())
}

fn validate_record(record: &ReplicateRecord) -> Result<(), EvalError> {
    if record.scenario_id.trim().is_empty()
        || record.method_id.trim().is_empty()
        || record.replicate_id.trim().is_empty()
    {
        return Err(EvalError::EmptyIdentifier);
    }
    if !record.structural_beta.is_finite() {
        return Err(EvalError::NonFiniteStructuralTruth);
    }
    if record.p_value.is_some_and(|p| !p.is_finite() || !(0.0..=1.0).contains(&p)) {
        return Err(EvalError::InvalidPValue);
    }
    if record.point_estimate.is_some_and(|x| !x.is_finite()) {
        return Err(EvalError::NonFinitePointEstimate);
    }
    if record.runtime_seconds.is_some_and(|x| !x.is_finite() || x < 0.0) {
        return Err(EvalError::InvalidRuntime);
    }
    if let Some(confidence_set) = &record.confidence_set {
        confidence_set.validate()?;
    }
    if record.status != ExecutionStatus::Completed
        && (record.p_value.is_some()
            || record.point_estimate.is_some()
            || record.confidence_set.is_some())
    {
        return Err(EvalError::MetricsOnNonCompletedRecord);
    }
    Ok(())
}

fn p_value_diagnostics(values: &[f64]) -> Option<PValueDiagnostics> {
    if values.is_empty() {
        return None;
    }
    let mut bins = [0usize; 10];
    for &p in values {
        let index = ((p * 10.0) as usize).min(9);
        bins[index] += 1;
    }
    Some(PValueDiagnostics {
        count: values.len(),
        histogram_deciles: bins,
        p10: quantile(values, 0.10).unwrap(),
        p50: quantile(values, 0.50).unwrap(),
        p90: quantile(values, 0.90).unwrap(),
    })
}

fn confidence_set_diagnostics(
    records: &[&ReplicateRecord],
    beta: f64,
    z_critical: f64,
) -> Result<ConfidenceSetDiagnostics, EvalError> {
    let sets: Vec<&ConfidenceSet> = records
        .iter()
        .filter(|row| row.status == ExecutionStatus::Completed)
        .filter_map(|row| row.confidence_set.as_ref())
        .collect();
    let available = sets.len();
    let missing = records.len() - available;
    let covered = sets.iter().filter(|set| set.contains(beta)).count();
    let unbounded_count = sets.iter().filter(|set| set.is_unbounded()).count();
    let mut topology_counts = TopologyCounts::default();
    let mut widths = Vec::new();
    let mut finite_noncovering_set_count = 0usize;

    for set in &sets {
        let topology = set.topology();
        topology_counts.add(topology);
        if topology == ConfidenceSetTopology::FiniteBounded && !set.contains(beta) {
            finite_noncovering_set_count += 1;
        }
        if let Some(width) = set.finite_total_width()? {
            widths.push(width);
        }
    }

    let conservative_coverage_lower_bound = if records.is_empty() {
        None
    } else {
        Some(covered as f64 / records.len() as f64)
    };
    let conservative_coverage_upper_bound = if records.is_empty() {
        None
    } else {
        Some((covered + missing) as f64 / records.len() as f64)
    };
    let finite_noncovering_set_rate = (topology_counts.finite_bounded > 0).then_some(
        finite_noncovering_set_count as f64 / topology_counts.finite_bounded as f64,
    );

    Ok(ConfidenceSetDiagnostics {
        available,
        missing,
        covered,
        observed_coverage: rate_estimate(covered, available, z_critical)?,
        conservative_coverage_lower_bound,
        conservative_coverage_upper_bound,
        topology_counts,
        unbounded_count,
        unbounded_rate: (available > 0).then_some(unbounded_count as f64 / available as f64),
        finite_noncovering_set_count,
        finite_noncovering_set_rate,
        finite_width_count: widths.len(),
        finite_width_median: quantile(&widths, 0.50),
        finite_width_p90: quantile(&widths, 0.90),
    })
}

fn estimate_diagnostics(
    records: &[&ReplicateRecord],
    beta: f64,
) -> Result<EstimateDiagnostics, EvalError> {
    let estimates: Vec<f64> = records
        .iter()
        .filter(|row| row.status == ExecutionStatus::Completed)
        .filter_map(|row| row.point_estimate)
        .collect();
    if estimates.is_empty() {
        return Ok(EstimateDiagnostics {
            count: 0,
            mean_signed_bias: None,
            rmse: None,
        });
    }

    let mut bias_sum = 0.0;
    let mut squared_error_sum = 0.0;
    for estimate in &estimates {
        let error = *estimate - beta;
        let squared = error * error;
        if !error.is_finite() || !squared.is_finite() {
            return Err(EvalError::NumericOverflow);
        }
        bias_sum += error;
        squared_error_sum += squared;
        if !bias_sum.is_finite() || !squared_error_sum.is_finite() {
            return Err(EvalError::NumericOverflow);
        }
    }

    let n = estimates.len() as f64;
    Ok(EstimateDiagnostics {
        count: estimates.len(),
        mean_signed_bias: Some(bias_sum / n),
        rmse: Some((squared_error_sum / n).sqrt()),
    })
}

fn resource_diagnostics(records: &[&ReplicateRecord]) -> ResourceDiagnostics {
    let comparable: Vec<&ReplicateRecord> = records
        .iter()
        .copied()
        .filter(|row| row.status != ExecutionStatus::NotApplicable)
        .collect();

    let runtime_values: Vec<f64> = comparable
        .iter()
        .filter_map(|row| row.runtime_seconds)
        .collect();
    let runtime_complete = !comparable.is_empty() && runtime_values.len() == comparable.len();

    let memory_values: Vec<f64> = comparable
        .iter()
        .filter_map(|row| row.peak_memory_bytes.map(|value| value as f64))
        .collect();
    let memory_complete = !comparable.is_empty() && memory_values.len() == comparable.len();

    ResourceDiagnostics {
        runtime_complete,
        runtime_count: runtime_values.len(),
        runtime_p50_seconds: runtime_complete.then(|| quantile(&runtime_values, 0.50).unwrap()),
        runtime_p95_seconds: runtime_complete.then(|| quantile(&runtime_values, 0.95).unwrap()),
        memory_complete,
        memory_count: memory_values.len(),
        peak_memory_p50_bytes: memory_complete.then(|| quantile(&memory_values, 0.50).unwrap()),
        peak_memory_p95_bytes: memory_complete.then(|| quantile(&memory_values, 0.95).unwrap()),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScorecardReport {
    pub schema_version: &'static str,
    pub alpha: f64,
    pub wilson_z_critical: f64,
    pub scorecards: Vec<MethodScorecard>,
}

pub fn evaluate_report(
    records: &[ReplicateRecord],
    config: EvaluatorConfig,
) -> Result<ScorecardReport, EvalError> {
    let scorecards = evaluate(records, config)?;
    Ok(ScorecardReport {
        schema_version: "connectome-mc-scorecard-v1",
        alpha: config.alpha,
        wilson_z_critical: config.wilson_z_critical,
        scorecards,
    })
}

pub fn evaluate(
    records: &[ReplicateRecord],
    config: EvaluatorConfig,
) -> Result<Vec<MethodScorecard>, EvalError> {
    if records.is_empty() {
        return Err(EvalError::EmptyInput);
    }
    validate_config(config)?;

    let mut unique_keys = BTreeSet::new();
    let mut groups: BTreeMap<(String, String), Vec<&ReplicateRecord>> = BTreeMap::new();
    for record in records {
        validate_record(record)?;
        let unique_key = (
            record.scenario_id.clone(),
            record.method_id.clone(),
            record.replicate_id.clone(),
        );
        if !unique_keys.insert(unique_key) {
            return Err(EvalError::DuplicateReplicate);
        }
        groups
            .entry((record.scenario_id.clone(), record.method_id.clone()))
            .or_default()
            .push(record);
    }

    let mut scorecards = Vec::with_capacity(groups.len());
    for ((scenario_id, method_id), mut group) in groups {
        // Canonical ordering makes floating-point aggregation independent of input row order.
        group.sort_by(|a, b| a.replicate_id.cmp(&b.replicate_id));
        let first = group[0];
        if group.iter().any(|row| {
            row.structural_beta != first.structural_beta || row.null_is_true != first.null_is_true
        }) {
            return Err(EvalError::InconsistentScenarioTruth);
        }
        let beta = first.structural_beta;
        let null_is_true = first.null_is_true;
        let mut unique_seeds = BTreeSet::new();
        for row in &group {
            if !unique_seeds.insert(row.seed) {
                return Err(EvalError::DuplicateSeed);
            }
        }

        let attempted = group.len();
        let completed = group
            .iter()
            .filter(|row| row.status == ExecutionStatus::Completed)
            .count();
        let failed = group
            .iter()
            .filter(|row| row.status == ExecutionStatus::Failed)
            .count();
        let not_applicable = group
            .iter()
            .filter(|row| row.status == ExecutionStatus::NotApplicable)
            .count();

        let p_values: Vec<f64> = group
            .iter()
            .filter(|row| row.status == ExecutionStatus::Completed)
            .filter_map(|row| row.p_value)
            .collect();
        let rejections = p_values.iter().filter(|&&p| p <= config.alpha).count();
        let rate = rate_estimate(rejections, p_values.len(), config.wilson_z_critical)?;

        let (empirical_size, empirical_power, conservative_size_upper_bound, conservative_power_lower_bound) =
            if null_is_true {
                let conservative = (rejections + attempted.saturating_sub(p_values.len())) as f64
                    / attempted as f64;
                (rate, None, Some(conservative), None)
            } else {
                let conservative = rejections as f64 / attempted as f64;
                (None, rate, None, Some(conservative))
            };

        let p_values_under_null = if null_is_true {
            p_value_diagnostics(&p_values)
        } else {
            None
        };
        let confidence_sets = confidence_set_diagnostics(&group, beta, config.wilson_z_critical)?;
        let estimates = estimate_diagnostics(&group, beta)?;
        let resources = resource_diagnostics(&group);

        scorecards.push(MethodScorecard {
            scenario_id,
            method_id,
            structural_beta: beta,
            null_is_true,
            alpha: config.alpha,
            wilson_z_critical: config.wilson_z_critical,
            records: ExecutionCounts {
                attempted,
                completed,
                failed,
                not_applicable,
                failure_rate: failed as f64 / attempted as f64,
                not_applicable_rate: not_applicable as f64 / attempted as f64,
            },
            empirical_size,
            empirical_power,
            conservative_size_upper_bound,
            conservative_power_lower_bound,
            p_values_under_null,
            confidence_sets,
            estimates,
            resources,
        });
    }

    Ok(scorecards)
}


/// A replicate and seed declared before any method result is observed.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedReplicate {
    pub replicate_id: String,
    pub seed: u64,
    pub structural_beta: f64,
    pub null_is_true: bool,
}

/// Configuration for a common-random-number, paired method comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairedComparisonConfig {
    /// Rejection threshold applied identically to both methods.
    pub rejection_alpha: f64,
    /// Tail probability for the distribution-free Hoeffding confidence interval.
    pub confidence_alpha: f64,
}

impl Default for PairedComparisonConfig {
    fn default() -> Self {
        Self {
            rejection_alpha: 0.05,
            confidence_alpha: 0.05,
        }
    }
}

/// Candidate-minus-baseline difference in rejection probability.
///
/// Positive values mean the candidate rejects more often. In a true-null
/// scenario this is a size increase; in an alternative scenario it is a power
/// increase. The conservative interval accounts for unknown/missing pairs.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedRejectionDifference {
    pub planned_pairs: usize,
    pub baseline_record_count: usize,
    pub candidate_record_count: usize,
    pub matched_record_pairs: usize,
    pub complete_rejection_pairs: usize,
    pub unknown_pairs: usize,
    /// Descriptive mean among pairs with both p-values observed; may be biased
    /// if missingness is outcome-dependent and must not be used alone.
    pub observed_complete_pair_mean: Option<f64>,
    /// Missing pair differences are bounded by [-1, 1] before adding the
    /// distribution-free Monte Carlo uncertainty radius.
    pub conservative_interval: FiniteInterval,
    pub confidence_level: f64,
    pub hoeffding_radius: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairedMethodComparison {
    pub scenario_id: String,
    pub baseline_method_id: String,
    pub candidate_method_id: String,
    pub structural_beta: f64,
    pub null_is_true: bool,
    pub rejection_alpha: f64,
    pub rejection_difference: PairedRejectionDifference,
}

/// Compare candidate and baseline rejection rates on common replicate IDs.
///
/// A replicate contributes a known paired difference only when both methods
/// completed and returned valid p-values. Missing, failed, not-applicable, or
/// p-value-omitting runs remain in the planned-pair denominator and are
/// conservatively assigned any difference in [-1, 1]. Matching IDs must also
/// have identical seeds and registered structural truth.
///
/// The Hoeffding radius is distribution-free for independent Monte Carlo
/// replicate seeds and bounded paired differences. It is intentionally
/// conservative; it is not a subject-level or animal-level confidence interval.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedComparisonReport {
    pub schema_version: &'static str,
    pub rejection_alpha: f64,
    pub confidence_alpha: f64,
    pub comparisons: Vec<PairedMethodComparison>,
}

pub fn evaluate_paired_report(
    records: &[ReplicateRecord],
    planned_replicates: &[PlannedReplicate],
    scenario_id: &str,
    baseline_method_id: &str,
    candidate_method_id: &str,
    config: PairedComparisonConfig,
) -> Result<PairedComparisonReport, EvalError> {
    let comparison = compare_paired_rejection_rate(
        records,
        planned_replicates,
        scenario_id,
        baseline_method_id,
        candidate_method_id,
        config,
    )?;
    Ok(PairedComparisonReport {
        schema_version: "connectome-mc-paired-comparison-v1",
        rejection_alpha: config.rejection_alpha,
        confidence_alpha: config.confidence_alpha,
        comparisons: vec![comparison],
    })
}

/// Compare candidate and baseline rejection rates against an independently frozen
/// replicate plan and common random-number seeds.
///
/// Every planned replicate is retained, including a replicate absent from both
/// method outputs. A result contributes a known paired difference only when both
/// methods completed and returned valid p-values. Missing, failed, not-applicable,
/// or p-value-omitting runs remain in the planned denominator and are conservatively
/// bounded by [-1, 1]. Method outputs not present in the plan fail closed.
///
/// The Hoeffding radius is distribution-free for independent Monte Carlo seed-pairs
/// and bounded paired differences. It is intentionally conservative; it is not a
/// subject-level or animal-level confidence interval.
pub fn compare_paired_rejection_rate(
    records: &[ReplicateRecord],
    planned_replicates: &[PlannedReplicate],
    scenario_id: &str,
    baseline_method_id: &str,
    candidate_method_id: &str,
    config: PairedComparisonConfig,
) -> Result<PairedMethodComparison, EvalError> {
    if scenario_id.trim().is_empty()
        || baseline_method_id.trim().is_empty()
        || candidate_method_id.trim().is_empty()
    {
        return Err(EvalError::EmptyIdentifier);
    }
    if baseline_method_id == candidate_method_id {
        return Err(EvalError::SameMethodComparison);
    }
    if !config.rejection_alpha.is_finite()
        || config.rejection_alpha <= 0.0
        || config.rejection_alpha >= 1.0
        || !config.confidence_alpha.is_finite()
        || config.confidence_alpha <= 0.0
        || config.confidence_alpha >= 1.0
    {
        return Err(EvalError::InvalidConfig);
    }
    if planned_replicates.is_empty() {
        return Err(EvalError::EmptyInput);
    }

    let mut planned_by_id = BTreeMap::<String, &PlannedReplicate>::new();
    let mut planned_seeds = BTreeSet::new();
    let mut planned_truth: Option<(f64, bool)> = None;
    for planned in planned_replicates {
        if planned.replicate_id.trim().is_empty() {
            return Err(EvalError::EmptyIdentifier);
        }
        if !planned.structural_beta.is_finite() {
            return Err(EvalError::NonFiniteStructuralTruth);
        }
        let truth = (planned.structural_beta, planned.null_is_true);
        if planned_truth.is_some_and(|previous| previous != truth) {
            return Err(EvalError::InconsistentPairedTruth);
        }
        planned_truth = Some(truth);
        if planned_by_id.insert(planned.replicate_id.clone(), planned).is_some() {
            return Err(EvalError::DuplicateReplicate);
        }
        if !planned_seeds.insert(planned.seed) {
            return Err(EvalError::DuplicateSeed);
        }
    }

    let planned_truth = planned_truth.ok_or(EvalError::EmptyInput)?;

    let mut baseline = BTreeMap::<String, &ReplicateRecord>::new();
    let mut candidate = BTreeMap::<String, &ReplicateRecord>::new();

    for record in records.iter().filter(|row| {
        row.scenario_id == scenario_id
            && (row.method_id == baseline_method_id || row.method_id == candidate_method_id)
    }) {
        validate_record(record)?;
        let planned = planned_by_id
            .get(&record.replicate_id)
            .ok_or(EvalError::UnplannedReplicate)?;
        if planned.seed != record.seed {
            return Err(EvalError::SeedAlignmentMismatch);
        }
        if planned.structural_beta != record.structural_beta
            || planned.null_is_true != record.null_is_true
        {
            return Err(EvalError::InconsistentPairedTruth);
        }
        let destination = if record.method_id == baseline_method_id {
            &mut baseline
        } else {
            &mut candidate
        };
        if destination.insert(record.replicate_id.clone(), record).is_some() {
            return Err(EvalError::DuplicateReplicate);
        }
    }

    // The independent plan supplies scenario truth even when one or both
    // methods return no rows. Such total-run failures remain unknown pairs.
    let planned_pairs = planned_by_id.len();
    let baseline_record_count = baseline.len();
    let candidate_record_count = candidate.len();
    let mut matched_record_pairs = 0usize;
    let mut complete_rejection_pairs = 0usize;
    let mut known_difference_sum = 0.0f64;

    for (replicate_id, planned) in &planned_by_id {
        match (baseline.get(replicate_id), candidate.get(replicate_id)) {
            (Some(base), Some(cand)) => {
                matched_record_pairs += 1;
                if base.seed != cand.seed || base.seed != planned.seed {
                    return Err(EvalError::SeedAlignmentMismatch);
                }
                if base.structural_beta != cand.structural_beta
                    || base.null_is_true != cand.null_is_true
                    || base.structural_beta != planned.structural_beta
                    || base.null_is_true != planned.null_is_true
                {
                    return Err(EvalError::InconsistentPairedTruth);
                }

                if base.status == ExecutionStatus::Completed
                    && cand.status == ExecutionStatus::Completed
                {
                    if let (Some(base_p), Some(candidate_p)) = (base.p_value, cand.p_value) {
                        let base_rejects = base_p <= config.rejection_alpha;
                        let candidate_rejects = candidate_p <= config.rejection_alpha;
                        known_difference_sum += (candidate_rejects as i8 - base_rejects as i8) as f64;
                        complete_rejection_pairs += 1;
                    }
                }
            }
            _ => {}
        }
    }

    let unknown_pairs = planned_pairs - complete_rejection_pairs;
    let planned = planned_pairs as f64;
    let missing = unknown_pairs as f64;
    let confidence_level = 1.0 - config.confidence_alpha;
    let hoeffding_radius =
        (2.0 * (2.0 / config.confidence_alpha).ln() / planned).sqrt();
    let lower_empirical = (known_difference_sum - missing) / planned;
    let upper_empirical = (known_difference_sum + missing) / planned;
    let conservative_interval = FiniteInterval {
        lower: (lower_empirical - hoeffding_radius).max(-1.0),
        upper: (upper_empirical + hoeffding_radius).min(1.0),
    };

    if !hoeffding_radius.is_finite()
        || !conservative_interval.lower.is_finite()
        || !conservative_interval.upper.is_finite()
        || conservative_interval.lower > conservative_interval.upper
    {
        return Err(EvalError::NumericOverflow);
    }

    Ok(PairedMethodComparison {
        scenario_id: scenario_id.to_string(),
        baseline_method_id: baseline_method_id.to_string(),
        candidate_method_id: candidate_method_id.to_string(),
        structural_beta: planned_truth.0,
        null_is_true: planned_truth.1,
        rejection_alpha: config.rejection_alpha,
        rejection_difference: PairedRejectionDifference {
            planned_pairs,
            baseline_record_count,
            candidate_record_count,
            matched_record_pairs,
            complete_rejection_pairs,
            unknown_pairs,
            observed_complete_pair_mean: (complete_rejection_pairs > 0).then_some(
                known_difference_sum / complete_rejection_pairs as f64,
            ),
            conservative_interval,
            confidence_level,
            hoeffding_radius,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(
        scenario: &str,
        method: &str,
        replicate: &str,
        beta: f64,
        null_is_true: bool,
        status: ExecutionStatus,
        p_value: Option<f64>,
    ) -> ReplicateRecord {
        ReplicateRecord {
            scenario_id: scenario.into(),
            method_id: method.into(),
            replicate_id: replicate.into(),
            seed: replicate.parse().unwrap_or(0),
            structural_beta: beta,
            null_is_true,
            status,
            p_value,
            point_estimate: None,
            confidence_set: None,
            runtime_seconds: None,
            peak_memory_bytes: None,
        }
    }

    fn bounded(lower: f64, upper: f64) -> ConfidenceSet {
        ConfidenceSet {
            components: vec![Interval {
                lower: Some(lower),
                upper: Some(upper),
            }],
        }
    }

    fn metric_calibration_observations(
        effect_magnitude: f64,
        improvements: &[f64],
    ) -> Vec<MetricCalibrationObservation> {
        improvements
            .iter()
            .enumerate()
            .map(|(index, improvement)| MetricCalibrationObservation {
                effect_magnitude,
                replicate_id: format!("r{index:03}"),
                seed: index as u64 + 1,
                uninformative_metric_value: 0.0,
                positive_control_metric_value: *improvement,
                perfect_metric_value: 1.0,
            })
            .collect()
    }

    #[test]
    fn metric_calibration_detects_positive_controls_and_null_specificity() {
        let mut observations = metric_calibration_observations(0.0, &[0.0; 20]);
        observations.extend(metric_calibration_observations(1.0, &[0.5; 20]));
        observations.extend(metric_calibration_observations(2.0, &[1.0; 20]));
        let report = calibrate_predictive_metric(
            "calibrated-score",
            MetricDirection::HigherIsBetter,
            &observations,
            MetricCalibrationConfig::default(),
        ).unwrap();
        assert_eq!(report.schema_version, "connectome-metric-calibration-v1");
        assert_eq!(report.bins.len(), 3);
        assert_eq!(report.bins[0].ties, 20);
        assert_eq!(report.bins[1].positive_control_wins, 20);
        assert!(report.zero_effect_false_positive_upper_bound.unwrap() <= 0.20);
        assert!(report.minimum_positive_control_detection_lower_bound.unwrap() >= 0.80);
        assert_eq!(report.monotonicity_violations, 0);
        assert!(report.calibrated_for_declared_grid);
    }

    #[test]
    fn metric_calibration_handles_lower_is_better_metrics() {
        let mut observations = metric_calibration_observations(0.0, &[0.0; 20]);
        let mut positive = metric_calibration_observations(1.0, &[0.0; 20]);
        for row in &mut positive {
            row.uninformative_metric_value = 1.0;
            row.perfect_metric_value = 0.0;
        }
        observations.extend(positive);
        let report = calibrate_predictive_metric(
            "rmse-like",
            MetricDirection::LowerIsBetter,
            &observations,
            MetricCalibrationConfig::default(),
        ).unwrap();
        assert!(report.calibrated_for_declared_grid);
        assert!(report.bins[1].mean_oriented_improvement > 0.0);
    }

    #[test]
    fn metric_calibration_rejects_metrics_that_miss_positive_controls() {
        let mut observations = metric_calibration_observations(0.0, &[0.0; 20]);
        observations.extend(metric_calibration_observations(1.0, &[0.0; 20]));
        let report = calibrate_predictive_metric(
            "insensitive-metric",
            MetricDirection::HigherIsBetter,
            &observations,
            MetricCalibrationConfig::default(),
        ).unwrap();
        assert!(!report.calibrated_for_declared_grid);
        assert_eq!(report.bins[1].positive_control_wins, 0);
        assert_eq!(report.minimum_positive_control_detection_lower_bound, Some(0.0));
    }

    #[test]
    fn metric_calibration_rejects_non_monotonic_effect_sensitivity() {
        let mut observations = metric_calibration_observations(0.0, &[0.0; 20]);
        observations.extend(metric_calibration_observations(1.0, &[1.0; 20]));
        observations.extend(metric_calibration_observations(2.0, &[0.5; 20]));
        let report = calibrate_predictive_metric(
            "non-monotone-metric",
            MetricDirection::HigherIsBetter,
            &observations,
            MetricCalibrationConfig::default(),
        ).unwrap();
        assert_eq!(report.monotonicity_violations, 1);
        assert!(!report.calibrated_for_declared_grid);
    }

    #[test]
    fn metric_calibration_reports_scale_normalized_dynamic_range_fraction() {
        let mut observations = metric_calibration_observations(0.0, &[0.0; 20]);
        observations.extend(metric_calibration_observations(1.0, &[0.5; 20]));
        observations.extend(metric_calibration_observations(2.0, &[1.0; 20]));
        let report = calibrate_predictive_metric(
            "drf-score",
            MetricDirection::HigherIsBetter,
            &observations,
            MetricCalibrationConfig::default(),
        ).unwrap();
        assert_eq!(report.bins[0].mean_dynamic_range_fraction, 0.0);
        assert_eq!(report.bins[1].mean_dynamic_range_fraction, 0.5);
        assert_eq!(report.bins[2].mean_dynamic_range_fraction, 1.0);
    }

    #[test]
    fn metric_calibration_rejects_duplicate_seeds_and_non_finite_scores() {
        let mut observations = metric_calibration_observations(0.0, &[0.0, 0.0]);
        observations[1].seed = observations[0].seed;
        assert_eq!(
            calibrate_predictive_metric(
                "duplicate-seed",
                MetricDirection::HigherIsBetter,
                &observations,
                MetricCalibrationConfig::default(),
            ),
            Err(EvalError::DuplicateSeed)
        );

        let mut invalid = metric_calibration_observations(0.0, &[0.0]);
        invalid[0].positive_control_metric_value = f64::NAN;
        assert_eq!(
            calibrate_predictive_metric(
                "invalid-score",
                MetricDirection::HigherIsBetter,
                &invalid,
                MetricCalibrationConfig::default(),
            ),
            Err(EvalError::NonFiniteMetricCalibrationValue)
        );
    }

    #[test]
    fn wilson_zero_successes_has_nonzero_upper_bound() {
        let interval = wilson_interval(0, 20, 1.959_963_984_540_054).unwrap().unwrap();
        assert_eq!(interval.lower, 0.0);
        assert!(interval.upper > 0.0 && interval.upper < 0.2);
    }

    #[test]
    fn wilson_all_successes_has_upper_bound_one() {
        let interval = wilson_interval(20, 20, 1.959_963_984_540_054).unwrap().unwrap();
        assert!(interval.lower < 1.0);
        assert_eq!(interval.upper, 1.0);
    }

    #[test]
    fn wilson_interior_known_answer_is_ordered() {
        let interval = wilson_interval(5, 10, 1.959_963_984_540_054).unwrap().unwrap();
        assert!((interval.lower - 0.2366).abs() < 0.001);
        assert!((interval.upper - 0.7634).abs() < 0.001);
    }

    #[test]
    fn wilson_ten_thousand_replicates_resolves_five_percent_rate() {
        let interval = wilson_interval(500, 10_000, 1.959_963_984_540_054)
            .unwrap()
            .unwrap();
        assert!((interval.lower - 0.04589848).abs() < 1e-8);
        assert!((interval.upper - 0.05444712).abs() < 1e-8);
    }

    #[test]
    fn wilson_zero_trials_is_missing_not_zero() {
        assert!(wilson_interval(0, 0, 1.959_963_984_540_054).unwrap().is_none());
    }

    #[test]
    fn wilson_rejects_successes_above_trials() {
        assert_eq!(wilson_interval(2, 1, 1.96), Err(EvalError::InvalidConfig));
    }

    #[test]
    fn confidence_set_contains_closed_endpoints() {
        let set = bounded(0.2, 0.8);
        assert!(set.contains(0.2));
        assert!(set.contains(0.8));
        assert!(!set.contains(0.81));
    }

    #[test]
    fn confidence_set_empty_is_not_missing_and_never_covers() {
        let set = ConfidenceSet { components: vec![] };
        assert_eq!(set.topology(), ConfidenceSetTopology::Empty);
        assert!(!set.contains(0.0));
        assert_eq!(set.finite_total_width().unwrap(), None);
    }

    #[test]
    fn confidence_set_identifies_whole_line() {
        let set = ConfidenceSet {
            components: vec![Interval { lower: None, upper: None }],
        };
        set.validate().unwrap();
        assert_eq!(set.topology(), ConfidenceSetTopology::WholeLine);
        assert!(set.contains(-1.0e100));
    }

    #[test]
    fn confidence_set_identifies_left_and_right_rays() {
        let left = ConfidenceSet {
            components: vec![Interval { lower: None, upper: Some(-1.0) }],
        };
        let right = ConfidenceSet {
            components: vec![Interval { lower: Some(1.0), upper: None }],
        };
        assert_eq!(left.topology(), ConfidenceSetTopology::LeftRay);
        assert_eq!(right.topology(), ConfidenceSetTopology::RightRay);
    }

    #[test]
    fn confidence_set_identifies_two_rays() {
        let set = ConfidenceSet {
            components: vec![
                Interval { lower: None, upper: Some(-1.0) },
                Interval { lower: Some(1.0), upper: None },
            ],
        };
        set.validate().unwrap();
        assert_eq!(set.topology(), ConfidenceSetTopology::TwoRays);
    }

    #[test]
    fn confidence_set_identifies_disconnected_finite_union() {
        let set = ConfidenceSet {
            components: vec![
                Interval { lower: Some(-2.0), upper: Some(-1.0) },
                Interval { lower: Some(1.0), upper: Some(2.0) },
            ],
        };
        set.validate().unwrap();
        assert_eq!(set.topology(), ConfidenceSetTopology::DisconnectedMultiComponent);
        assert_eq!(set.finite_total_width().unwrap(), Some(2.0));
    }

    #[test]
    fn confidence_set_rejects_unsorted_components() {
        let set = ConfidenceSet {
            components: vec![
                Interval { lower: Some(2.0), upper: Some(3.0) },
                Interval { lower: Some(-3.0), upper: Some(-2.0) },
            ],
        };
        assert_eq!(set.validate(), Err(EvalError::NonCanonicalConfidenceSet));
    }

    #[test]
    fn confidence_set_rejects_overlapping_or_touching_components() {
        let set = ConfidenceSet {
            components: vec![
                Interval { lower: Some(0.0), upper: Some(1.0) },
                Interval { lower: Some(1.0), upper: Some(2.0) },
            ],
        };
        assert_eq!(set.validate(), Err(EvalError::NonCanonicalConfidenceSet));
    }

    #[test]
    fn confidence_set_rejects_nonfinal_positive_infinity() {
        let set = ConfidenceSet {
            components: vec![
                Interval { lower: Some(0.0), upper: None },
                Interval { lower: Some(2.0), upper: Some(3.0) },
            ],
        };
        assert_eq!(set.validate(), Err(EvalError::NonCanonicalConfidenceSet));
    }

    #[test]
    fn confidence_set_rejects_reversed_interval() {
        let set = bounded(2.0, 1.0);
        assert_eq!(set.validate(), Err(EvalError::ReversedConfidenceInterval));
    }

    #[test]
    fn confidence_set_rejects_nan_endpoint() {
        let set = bounded(f64::NAN, 1.0);
        assert_eq!(set.validate(), Err(EvalError::NonFiniteConfidenceEndpoint));
    }

    #[test]
    fn finite_width_is_not_assigned_to_unbounded_set() {
        let set = ConfidenceSet {
            components: vec![Interval { lower: None, upper: Some(0.0) }],
        };
        assert_eq!(set.finite_total_width().unwrap(), None);
    }

    #[test]
    fn evaluator_rejects_empty_input() {
        assert_eq!(evaluate(&[], EvaluatorConfig::default()), Err(EvalError::EmptyInput));
    }

    #[test]
    fn evaluator_groups_and_sorts_deterministically() {
        let mut a = record("s", "m", "2", 0.0, true, ExecutionStatus::Completed, Some(0.2));
        let mut b = record("s", "m", "1", 0.0, true, ExecutionStatus::Completed, Some(0.8));
        a.point_estimate = Some(0.5);
        b.point_estimate = Some(0.1);
        let forward = evaluate(&[a.clone(), b.clone()], EvaluatorConfig::default()).unwrap();
        let reverse = evaluate(&[b, a], EvaluatorConfig::default()).unwrap();
        assert_eq!(forward, reverse);
    }

    #[test]
    fn duplicate_replicate_keys_fail_closed() {
        let a = record("s", "m", "1", 0.0, true, ExecutionStatus::Completed, Some(0.2));
        let b = a.clone();
        assert_eq!(
            evaluate(&[a, b], EvaluatorConfig::default()),
            Err(EvalError::DuplicateReplicate)
        );
    }

    #[test]
    fn truth_must_be_constant_within_scenario_method() {
        let a = record("s", "m", "1", 0.0, true, ExecutionStatus::Completed, Some(0.2));
        let b = record("s", "m", "2", 0.1, true, ExecutionStatus::Completed, Some(0.2));
        assert_eq!(
            evaluate(&[a, b], EvaluatorConfig::default()),
            Err(EvalError::InconsistentScenarioTruth)
        );
    }

    #[test]
    fn invalid_probabilities_and_config_fail_closed() {
        let mut row = record("s", "m", "1", 0.0, true, ExecutionStatus::Completed, Some(1.1));
        assert_eq!(evaluate(&[row.clone()], EvaluatorConfig::default()), Err(EvalError::InvalidPValue));
        row.p_value = Some(0.5);
        assert_eq!(
            evaluate(&[row], EvaluatorConfig { alpha: 1.0, wilson_z_critical: 1.96 }),
            Err(EvalError::InvalidConfig)
        );
    }

    #[test]
    fn failed_and_not_applicable_rows_stay_in_attempted_denominator() {
        let rows = vec![
            record("null", "m", "1", 0.0, true, ExecutionStatus::Completed, Some(0.01)),
            record("null", "m", "2", 0.0, true, ExecutionStatus::Failed, None),
            record("null", "m", "3", 0.0, true, ExecutionStatus::NotApplicable, None),
        ];
        let score = evaluate(&rows, EvaluatorConfig::default()).unwrap().remove(0);
        assert_eq!(score.records.attempted, 3);
        assert_eq!(score.records.completed, 1);
        assert_eq!(score.records.failed, 1);
        assert_eq!(score.records.not_applicable, 1);
        assert_eq!(score.empirical_size.unwrap().trials, 1);
        assert!((score.conservative_size_upper_bound.unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn completed_rows_without_p_values_are_unknown_for_sensitivity_bound() {
        let rows = vec![
            record("null", "m", "1", 0.0, true, ExecutionStatus::Completed, Some(0.5)),
            record("null", "m", "2", 0.0, true, ExecutionStatus::Completed, None),
        ];
        let score = evaluate(&rows, EvaluatorConfig::default()).unwrap().remove(0);
        assert_eq!(score.empirical_size.unwrap().successes, 0);
        assert_eq!(score.conservative_size_upper_bound, Some(0.5));
    }

    #[test]
    fn empirical_power_is_separate_from_size() {
        let rows = vec![
            record("alternative", "m", "1", 1.0, false, ExecutionStatus::Completed, Some(0.01)),
            record("alternative", "m", "2", 1.0, false, ExecutionStatus::Completed, Some(0.2)),
        ];
        let score = evaluate(&rows, EvaluatorConfig::default()).unwrap().remove(0);
        assert!(score.empirical_size.is_none());
        assert_eq!(score.empirical_power.unwrap().successes, 1);
        assert_eq!(score.conservative_power_lower_bound, Some(0.5));
    }

    #[test]
    fn null_p_value_histogram_places_one_in_last_bin() {
        let rows = vec![
            record("null", "m", "1", 0.0, true, ExecutionStatus::Completed, Some(0.0)),
            record("null", "m", "2", 0.0, true, ExecutionStatus::Completed, Some(1.0)),
        ];
        let score = evaluate(&rows, EvaluatorConfig::default()).unwrap().remove(0);
        let histogram = score.p_values_under_null.unwrap().histogram_deciles;
        assert_eq!(histogram[0], 1);
        assert_eq!(histogram[9], 1);
        assert_eq!(histogram.iter().sum::<usize>(), 2);
    }

    #[test]
    fn p_value_diagnostics_are_omitted_for_alternatives() {
        let rows = vec![record(
            "alternative",
            "m",
            "1",
            1.0,
            false,
            ExecutionStatus::Completed,
            Some(0.01),
        )];
        let score = evaluate(&rows, EvaluatorConfig::default()).unwrap().remove(0);
        assert!(score.p_values_under_null.is_none());
    }

    #[test]
    fn confidence_coverage_uses_union_and_preserves_missingness() {
        let mut covered = record("null", "m", "1", 0.5, true, ExecutionStatus::Completed, Some(0.5));
        covered.confidence_set = Some(ConfidenceSet {
            components: vec![
                Interval { lower: Some(0.0), upper: Some(0.2) },
                Interval { lower: Some(0.4), upper: Some(0.6) },
            ],
        });
        let mut empty = record("null", "m", "2", 0.5, true, ExecutionStatus::Completed, Some(0.5));
        empty.confidence_set = Some(ConfidenceSet { components: vec![] });
        let missing = record("null", "m", "3", 0.5, true, ExecutionStatus::Completed, Some(0.5));
        let score = evaluate(&[covered, empty, missing], EvaluatorConfig::default()).unwrap().remove(0);
        assert_eq!(score.confidence_sets.available, 2);
        assert_eq!(score.confidence_sets.missing, 1);
        assert_eq!(score.confidence_sets.covered, 1);
        assert_eq!(score.confidence_sets.observed_coverage.unwrap().estimate, 0.5);
        assert!((score.confidence_sets.conservative_coverage_lower_bound.unwrap() - 1.0 / 3.0).abs() < 1e-12);
        assert!((score.confidence_sets.conservative_coverage_upper_bound.unwrap() - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(score.confidence_sets.topology_counts.empty, 1);
        assert_eq!(score.confidence_sets.topology_counts.disconnected_multi_component, 1);
    }

    #[test]
    fn finite_bounded_sets_excluding_truth_are_reported_as_false_narrow() {
        let mut narrow_miss = record("zero-relevance", "m", "1", 0.0, true, ExecutionStatus::Completed, Some(0.5));
        narrow_miss.confidence_set = Some(ConfidenceSet {
            components: vec![Interval { lower: Some(1.0), upper: Some(1.1) }],
        });
        let mut valid_bounded = record("zero-relevance", "m", "2", 0.0, true, ExecutionStatus::Completed, Some(0.5));
        valid_bounded.confidence_set = Some(ConfidenceSet {
            components: vec![Interval { lower: Some(-1.0), upper: Some(1.0) }],
        });
        let mut whole_line = record("zero-relevance", "m", "3", 0.0, true, ExecutionStatus::Completed, Some(0.5));
        whole_line.confidence_set = Some(ConfidenceSet {
            components: vec![Interval { lower: None, upper: None }],
        });
        let score = evaluate(
            &[narrow_miss, valid_bounded, whole_line],
            EvaluatorConfig::default(),
        ).unwrap().remove(0);
        assert_eq!(score.confidence_sets.finite_noncovering_set_count, 1);
        assert_eq!(score.confidence_sets.finite_noncovering_set_rate, Some(0.5));
        assert_eq!(score.confidence_sets.unbounded_count, 1);
    }

    #[test]
    fn estimate_bias_and_rmse_are_separate_from_inference() {
        let mut a = record("s", "m", "1", 1.0, true, ExecutionStatus::Completed, Some(0.5));
        let mut b = record("s", "m", "2", 1.0, true, ExecutionStatus::Completed, Some(0.5));
        a.point_estimate = Some(2.0);
        b.point_estimate = Some(0.0);
        let score = evaluate(&[a, b], EvaluatorConfig::default()).unwrap().remove(0);
        assert_eq!(score.estimates.count, 2);
        assert_eq!(score.estimates.mean_signed_bias, Some(0.0));
        assert_eq!(score.estimates.rmse, Some(1.0));
    }

    #[test]
    fn runtime_summary_is_suppressed_when_a_comparable_run_is_missing() {
        let mut a = record("s", "m", "1", 0.0, true, ExecutionStatus::Completed, Some(0.5));
        let b = record("s", "m", "2", 0.0, true, ExecutionStatus::Failed, None);
        a.runtime_seconds = Some(1.0);
        let score = evaluate(&[a, b], EvaluatorConfig::default()).unwrap().remove(0);
        assert!(!score.resources.runtime_complete);
        assert!(score.resources.runtime_p50_seconds.is_none());
    }

    #[test]
    fn runtime_quantiles_require_complete_comparable_records() {
        let mut a = record("s", "m", "1", 0.0, true, ExecutionStatus::Completed, Some(0.5));
        let mut b = record("s", "m", "2", 0.0, true, ExecutionStatus::Completed, Some(0.5));
        a.runtime_seconds = Some(1.0);
        b.runtime_seconds = Some(3.0);
        let score = evaluate(&[a, b], EvaluatorConfig::default()).unwrap().remove(0);
        assert!(score.resources.runtime_complete);
        assert_eq!(score.resources.runtime_p50_seconds, Some(2.0));
        assert_eq!(score.resources.runtime_p95_seconds, Some(2.9));
    }

    #[test]
    fn non_completed_rows_cannot_smuggle_inference_results() {
        let mut row = record("s", "m", "1", 0.0, true, ExecutionStatus::Failed, None);
        row.p_value = Some(0.5);
        assert_eq!(
            evaluate(&[row], EvaluatorConfig::default()),
            Err(EvalError::MetricsOnNonCompletedRecord)
        );
    }

    #[test]
    fn configured_wilson_critical_value_applies_to_coverage() {
        let mut row = record("s", "m", "1", 0.5, true, ExecutionStatus::Completed, Some(0.5));
        row.confidence_set = Some(bounded(0.0, 1.0));
        let ordinary = evaluate(
            &[row.clone()],
            EvaluatorConfig { alpha: 0.05, wilson_z_critical: 1.959_963_984_540_054 },
        ).unwrap().remove(0);
        let wider_critical = evaluate(
            &[row],
            EvaluatorConfig { alpha: 0.05, wilson_z_critical: 3.0 },
        ).unwrap().remove(0);
        let ordinary_lower = ordinary.confidence_sets.observed_coverage.unwrap().wilson_interval.lower;
        let wider_lower = wider_critical.confidence_sets.observed_coverage.unwrap().wilson_interval.lower;
        assert!(wider_lower < ordinary_lower);
    }

    #[test]
    fn report_wrapper_matches_versioned_output_contract() {
        let row = record("s", "m", "1", 0.0, true, ExecutionStatus::Completed, Some(0.5));
        let report = evaluate_report(&[row], EvaluatorConfig::default()).unwrap();
        assert_eq!(report.schema_version, "connectome-mc-scorecard-v1");
        assert_eq!(report.alpha, 0.05);
        assert_eq!(report.scorecards.len(), 1);
    }


    #[test]
    fn duplicate_seeds_within_a_method_group_fail_closed() {
        let a = record("s", "m", "1", 0.0, true, ExecutionStatus::Completed, Some(0.5));
        let mut b = record("s", "m", "2", 0.0, true, ExecutionStatus::Completed, Some(0.5));
        b.seed = a.seed;
        assert_eq!(evaluate(&[a, b], EvaluatorConfig::default()), Err(EvalError::DuplicateSeed));
    }

    #[test]
    fn paired_comparison_reports_candidate_minus_baseline_power() {
        let rows = vec![
            record("alt", "baseline", "1", 1.0, false, ExecutionStatus::Completed, Some(0.2)),
            record("alt", "candidate", "1", 1.0, false, ExecutionStatus::Completed, Some(0.01)),
            record("alt", "baseline", "2", 1.0, false, ExecutionStatus::Completed, Some(0.8)),
            record("alt", "candidate", "2", 1.0, false, ExecutionStatus::Completed, Some(0.04)),
        ];
        let plan = vec![
            PlannedReplicate { replicate_id: "1".into(), seed: 1, structural_beta: 1.0, null_is_true: false },
            PlannedReplicate { replicate_id: "2".into(), seed: 2, structural_beta: 1.0, null_is_true: false },
        ];
        let result = compare_paired_rejection_rate(
            &rows, &plan, "alt", "baseline", "candidate", PairedComparisonConfig::default()
        ).unwrap();
        assert!(!result.null_is_true);
        assert_eq!(result.rejection_difference.planned_pairs, 2);
        assert_eq!(result.rejection_difference.baseline_record_count, 2);
        assert_eq!(result.rejection_difference.candidate_record_count, 2);
        assert_eq!(result.rejection_difference.matched_record_pairs, 2);
        assert_eq!(result.rejection_difference.complete_rejection_pairs, 2);
        assert_eq!(result.rejection_difference.observed_complete_pair_mean, Some(1.0));
        assert_eq!(result.rejection_difference.unknown_pairs, 0);
    }

    #[test]
    fn paired_comparison_reports_candidate_minus_baseline_size() {
        let rows = vec![
            record("null", "baseline", "1", 0.0, true, ExecutionStatus::Completed, Some(0.01)),
            record("null", "candidate", "1", 0.0, true, ExecutionStatus::Completed, Some(0.4)),
            record("null", "baseline", "2", 0.0, true, ExecutionStatus::Completed, Some(0.2)),
            record("null", "candidate", "2", 0.0, true, ExecutionStatus::Completed, Some(0.01)),
        ];
        let plan = vec![
            PlannedReplicate { replicate_id: "1".into(), seed: 1, structural_beta: 0.0, null_is_true: true },
            PlannedReplicate { replicate_id: "2".into(), seed: 2, structural_beta: 0.0, null_is_true: true },
        ];
        let result = compare_paired_rejection_rate(
            &rows, &plan, "null", "baseline", "candidate", PairedComparisonConfig::default()
        ).unwrap();
        assert!(result.null_is_true);
        assert_eq!(result.rejection_difference.observed_complete_pair_mean, Some(0.0));
    }

    #[test]
    fn paired_comparison_counts_replicates_missing_from_both_methods() {
        let rows = vec![
            record("alt", "baseline", "1", 1.0, false, ExecutionStatus::Completed, Some(0.5)),
            record("alt", "candidate", "1", 1.0, false, ExecutionStatus::Completed, Some(0.01)),
            record("alt", "baseline", "2", 1.0, false, ExecutionStatus::Completed, Some(0.5)),
        ];
        let plan = vec![
            PlannedReplicate { replicate_id: "1".into(), seed: 1, structural_beta: 1.0, null_is_true: false },
            PlannedReplicate { replicate_id: "2".into(), seed: 2, structural_beta: 1.0, null_is_true: false },
            PlannedReplicate { replicate_id: "3".into(), seed: 3, structural_beta: 1.0, null_is_true: false },
        ];
        let result = compare_paired_rejection_rate(
            &rows, &plan, "alt", "baseline", "candidate", PairedComparisonConfig::default()
        ).unwrap();
        assert_eq!(result.rejection_difference.planned_pairs, 3);
        assert_eq!(result.rejection_difference.baseline_record_count, 2);
        assert_eq!(result.rejection_difference.candidate_record_count, 1);
        assert_eq!(result.rejection_difference.matched_record_pairs, 1);
        assert_eq!(result.rejection_difference.complete_rejection_pairs, 1);
        assert_eq!(result.rejection_difference.unknown_pairs, 2);
        assert_eq!(result.rejection_difference.observed_complete_pair_mean, Some(1.0));
        assert_eq!(result.rejection_difference.conservative_interval.lower, -1.0);
        assert_eq!(result.rejection_difference.conservative_interval.upper, 1.0);
    }

    #[test]
    fn paired_comparison_rejects_mismatched_seed_alignment() {
        let baseline = record("s", "baseline", "1", 0.0, true, ExecutionStatus::Completed, Some(0.5));
        let mut candidate = record("s", "candidate", "1", 0.0, true, ExecutionStatus::Completed, Some(0.5));
        candidate.seed = 1;
        assert_eq!(
            compare_paired_rejection_rate(
                &[baseline, candidate],
                &[PlannedReplicate { replicate_id: "1".into(), seed: 0, structural_beta: 0.0, null_is_true: true }],
                "s", "baseline", "candidate", PairedComparisonConfig::default()
            ),
            Err(EvalError::SeedAlignmentMismatch)
        );
    }

    #[test]
    fn paired_comparison_preserves_total_method_outage_from_frozen_truth() {
        let plan = vec![
            PlannedReplicate { replicate_id: "1".into(), seed: 1, structural_beta: 0.0, null_is_true: true },
            PlannedReplicate { replicate_id: "2".into(), seed: 2, structural_beta: 0.0, null_is_true: true },
        ];
        let result = compare_paired_rejection_rate(
            &[],
            &plan,
            "null",
            "baseline",
            "candidate",
            PairedComparisonConfig::default(),
        ).unwrap();
        assert_eq!(result.structural_beta, 0.0);
        assert!(result.null_is_true);
        assert_eq!(result.rejection_difference.planned_pairs, 2);
        assert_eq!(result.rejection_difference.baseline_record_count, 0);
        assert_eq!(result.rejection_difference.candidate_record_count, 0);
        assert_eq!(result.rejection_difference.matched_record_pairs, 0);
        assert_eq!(result.rejection_difference.complete_rejection_pairs, 0);
        assert_eq!(result.rejection_difference.unknown_pairs, 2);
        assert_eq!(result.rejection_difference.conservative_interval.lower, -1.0);
        assert_eq!(result.rejection_difference.conservative_interval.upper, 1.0);
    }

    #[test]
    fn paired_comparison_rejects_plan_truth_mismatch() {
        let mut row = record("s", "baseline", "1", 0.0, true, ExecutionStatus::Completed, Some(0.5));
        row.seed = 1;
        let plan = vec![PlannedReplicate {
            replicate_id: "1".into(),
            seed: 1,
            structural_beta: 1.0,
            null_is_true: false,
        }];
        assert_eq!(
            compare_paired_rejection_rate(
                &[row],
                &plan,
                "s",
                "baseline",
                "candidate",
                PairedComparisonConfig::default(),
            ),
            Err(EvalError::InconsistentPairedTruth)
        );
    }

    #[test]
    fn paired_plan_rejects_inconsistent_truth_across_replicates() {
        let plan = vec![
            PlannedReplicate { replicate_id: "1".into(), seed: 1, structural_beta: 0.0, null_is_true: true },
            PlannedReplicate { replicate_id: "2".into(), seed: 2, structural_beta: 1.0, null_is_true: false },
        ];
        let rows = vec![
            record("s", "baseline", "1", 0.0, true, ExecutionStatus::Completed, Some(0.5)),
            record("s", "candidate", "1", 0.0, true, ExecutionStatus::Completed, Some(0.5)),
        ];
        assert_eq!(
            compare_paired_rejection_rate(
                &rows,
                &plan,
                "s",
                "baseline",
                "candidate",
                PairedComparisonConfig::default(),
            ),
            Err(EvalError::InconsistentPairedTruth)
        );
    }

    #[test]
    fn paired_comparison_rejects_unplanned_replicates() {
        let baseline = record("s", "baseline", "1", 0.0, true, ExecutionStatus::Completed, Some(0.5));
        let candidate = record("s", "candidate", "1", 0.0, true, ExecutionStatus::Completed, Some(0.5));
        assert_eq!(
            compare_paired_rejection_rate(
                &[baseline, candidate],
                &[PlannedReplicate { replicate_id: "2".into(), seed: 2, structural_beta: 0.0, null_is_true: true }],
                "s", "baseline", "candidate", PairedComparisonConfig::default()
            ),
            Err(EvalError::UnplannedReplicate)
        );
    }

    #[test]
    fn paired_comparison_rejects_inconsistent_structural_truth() {
        let baseline = record("s", "baseline", "1", 0.0, true, ExecutionStatus::Completed, Some(0.5));
        let candidate = record("s", "candidate", "1", 0.1, true, ExecutionStatus::Completed, Some(0.5));
        assert_eq!(
            compare_paired_rejection_rate(
                &[baseline, candidate],
                &[PlannedReplicate { replicate_id: "1".into(), seed: 0, structural_beta: 0.0, null_is_true: true }],
                "s", "baseline", "candidate", PairedComparisonConfig::default()
            ),
            Err(EvalError::InconsistentPairedTruth)
        );
    }

    #[test]
    fn paired_comparison_is_input_order_independent() {
        let rows = vec![
            record("s", "b", "2", 0.0, true, ExecutionStatus::Completed, Some(0.2)),
            record("s", "c", "2", 0.0, true, ExecutionStatus::Completed, Some(0.1)),
            record("s", "b", "1", 0.0, true, ExecutionStatus::Completed, Some(0.01)),
            record("s", "c", "1", 0.0, true, ExecutionStatus::Completed, Some(0.5)),
        ];
        let plan = vec![
            PlannedReplicate { replicate_id: "2".into(), seed: 2, structural_beta: 0.0, null_is_true: true },
            PlannedReplicate { replicate_id: "1".into(), seed: 1, structural_beta: 0.0, null_is_true: true },
        ];
        let first = compare_paired_rejection_rate(
            &rows, &plan, "s", "b", "c", PairedComparisonConfig::default()
        ).unwrap();
        let reversed: Vec<_> = rows.iter().cloned().rev().collect();
        let reversed_plan: Vec<_> = plan.iter().cloned().rev().collect();
        let second = compare_paired_rejection_rate(
            &reversed, &reversed_plan, "s", "b", "c", PairedComparisonConfig::default()
        ).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn paired_report_wrapper_is_versioned() {
        let rows = vec![
            record("s", "b", "1", 0.0, true, ExecutionStatus::Completed, Some(0.5)),
            record("s", "c", "1", 0.0, true, ExecutionStatus::Completed, Some(0.5)),
        ];
        let plan = vec![PlannedReplicate { replicate_id: "1".into(), seed: 1, structural_beta: 0.0, null_is_true: true }];
        let report = evaluate_paired_report(
            &rows, &plan, "s", "b", "c", PairedComparisonConfig::default()
        ).unwrap();
        assert_eq!(report.schema_version, "connectome-mc-paired-comparison-v1");
        assert_eq!(report.comparisons.len(), 1);
    }

    #[test]
    fn paired_comparison_rejects_identical_method_names() {
        let row = record("s", "m", "1", 0.0, true, ExecutionStatus::Completed, Some(0.5));
        assert_eq!(
            compare_paired_rejection_rate(
                &[row],
                &[PlannedReplicate { replicate_id: "1".into(), seed: 1, structural_beta: 0.0, null_is_true: true }],
                "s", "m", "m", PairedComparisonConfig::default()
            ),
            Err(EvalError::SameMethodComparison)
        );
    }

    #[test]
    fn output_order_is_lexical_by_scenario_then_method() {
        let rows = vec![
            record("z", "b", "1", 0.0, true, ExecutionStatus::Completed, Some(0.5)),
            record("a", "z", "1", 0.0, true, ExecutionStatus::Completed, Some(0.5)),
            record("a", "b", "1", 0.0, true, ExecutionStatus::Completed, Some(0.5)),
        ];
        let scores = evaluate(&rows, EvaluatorConfig::default()).unwrap();
        let keys: Vec<(&str, &str)> = scores
            .iter()
            .map(|score| (score.scenario_id.as_str(), score.method_id.as_str()))
            .collect();
        assert_eq!(keys, vec![("a", "b"), ("a", "z"), ("z", "b")]);
    }
}
